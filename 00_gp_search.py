# 00_tod_gp_search.py (=stage1_rankshed_botorch_4param.py)
#!/usr/bin/env python3
import os, re, sys, json, argparse, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--run_file", default="run_nerf_ranksched.py")
    p.add_argument("--config", required=True)
    p.add_argument("--basedir", default="logs/sched/rankwsd")
    p.add_argument("--exp_prefix", default="botorch")

    p.add_argument("--n_trials", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--n_init", type=int, default=4)
    p.add_argument("--n_iters", type=int, default=100000)
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--optimizer", default="aux-sign-auto-cos-inc")
    p.add_argument("--train_scheduler", default="rank_wsd")
    return p.parse_args()


def unit_to_params(x):
    x = x.detach().cpu().double()

    def log_interp(u, lo, hi):
        lo = torch.log10(torch.tensor(lo, dtype=torch.double))
        hi = torch.log10(torch.tensor(hi, dtype=torch.double))
        return float(10 ** (lo + u * (hi - lo)))

    return {
        "muon_lrate": log_interp(x[0], 1e-4, 5e-3),
        "lrate": log_interp(x[1], 1e-4, 1e-3),
        "lrate_decay": int(round(float(100 + x[2] * 400))),
        "muon_momentum": float(0.85 + x[3] * 0.13),
    }


def parse_metric(stdout_path):
    vals = []
    pat_val = re.compile(r"\[VAL\].*?PSNR:\s*([-+0-9.eE]+)")
    pat_test = re.compile(r"testset_mean_psnr:\s*([-+0-9.eE]+)")

    text = Path(stdout_path).read_text(errors="ignore")
    for m in pat_val.finditer(text):
        vals.append(float(m.group(1)))
    for m in pat_test.finditer(text):
        vals.append(float(m.group(1)))

    if not vals:
        raise RuntimeError(f"No PSNR metric found in {stdout_path}")
    return max(vals)


def run_one_trial(args, root, hpo_dir, scene, trial_id, x_unit, gpu_id):
    params = unit_to_params(x_unit)

    expname = (
        f"{args.exp_prefix}_{scene}_trial{trial_id:03d}"
        f"_mlr{params['muon_lrate']:.2e}"
        f"_alr{params['lrate']:.2e}"
        f"_decay{params['lrate_decay']}"
        f"_mom{params['muon_momentum']:.3f}"
    ).replace("+", "").replace(".", "p")

    trial_dir = hpo_dir / expname
    trial_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = trial_dir / "stdout.txt"
    stderr_path = trial_dir / "stderr.txt"

    cmd = [
        sys.executable,
        str((root / args.run_file).resolve()),

        "--basedir", str((root / args.basedir).resolve()),
        "--config", str((root / args.config).resolve()),
        "--expname", expname,

        "--optimizer", args.optimizer,
        "--train_scheduler", args.train_scheduler,

        "--muon_lrate", str(params["muon_lrate"]),
        "--lrate", str(params["lrate"]),
        "--lrate_decay", str(params["lrate_decay"]),
        "--muon_momentum", str(params["muon_momentum"]),

        "--lowrank_rank_start", "150",
        "--lowrank_rank_end", "250",
        "--lowrank_auto_init_rank_start",

        "--N_iters", str(args.n_iters),
        "--i_print", "5000",
        "--i_weights", "100000",
        "--i_testset", "100000",
        "--i_video", "100000",
        "--no_reload",
        "--seed", str(args.seed + trial_id),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"

    with open(stdout_path, "w") as fout, open(stderr_path, "w") as ferr:
        ret = subprocess.call(
            cmd,
            cwd=str(root),
            env=env,
            stdout=fout,
            stderr=ferr,
        )

    if ret != 0:
        raise RuntimeError(f"Trial {trial_id} failed on GPU {gpu_id}. See {stderr_path}")

    value = parse_metric(stdout_path)

    return {
        "trial_number": trial_id,
        "value": value,
        **params,
        "unit_x": [float(v) for v in x_unit.tolist()],
        "expname": expname,
        "gpu": str(gpu_id),
        "trial_dir": str(trial_dir),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }


def propose_batch(train_x, train_y, q, seed):
    train_x = train_x.double()
    train_y = train_y.double()

    y_mean = train_y.mean()
    y_std = train_y.std().clamp_min(1e-6)
    y_stdzd = (train_y - y_mean) / y_std

    model = SingleTaskGP(train_x, y_stdzd)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]), seed=seed)

    acq = qExpectedImprovement(
        model=model,
        best_f=y_stdzd.max(),
        sampler=sampler,
    )

    bounds = torch.stack([
        torch.zeros(4, dtype=torch.double),
        torch.ones(4, dtype=torch.double),
    ])

    candidates, _ = optimize_acqf(
        acq_function=acq,
        bounds=bounds,
        q=q,
        num_restarts=10,
        raw_samples=256,
        options={"batch_limit": 5, "maxiter": 200},
    )
    return candidates.detach()


def save_summary(path, scene, results):
    top = sorted(results, key=lambda r: r["value"], reverse=True)
    best = top[0] if top else None
    summary = {
        "scene": scene,
        "method": "BoTorch batch BO for run_nerf_ranksched.py",
        "n_complete_trials": len(results),
        "best_value": best["value"] if best else None,
        "best_params": {
            "muon_lrate": best["muon_lrate"],
            "lrate": best["lrate"],
            "lrate_decay": best["lrate_decay"],
            "muon_momentum": best["muon_momentum"],
        } if best else None,
        "best_trial_number": best["trial_number"] if best else None,
        "top_trials": top,
    }
    path.write_text(json.dumps(summary, indent=2))


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    scene = Path(args.config).stem

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    hpo_dir = root / "logs" / "botorch_ranksched_4param" / scene
    hpo_dir.mkdir(parents=True, exist_ok=True)
    summary_path = hpo_dir / f"{scene}_botorch_ranksched_4param_summary.json"

    torch.manual_seed(args.seed)

    results = []
    train_x_list, train_y_list = [], []

    sobol = torch.quasirandom.SobolEngine(dimension=4, scramble=True, seed=args.seed)
    init_x = sobol.draw(args.n_init).double()
    init_pool = [init_x[i] for i in range(init_x.shape[0])]

    trial_id = 0
    while trial_id < args.n_trials:
        remaining = args.n_trials - trial_id
        q = min(args.batch_size, len(gpus), remaining)

        if len(train_x_list) < args.n_init:
            batch = init_pool[:q]
            init_pool = init_pool[q:]
        else:
            train_x = torch.stack(train_x_list).double()
            train_y = torch.tensor(train_y_list, dtype=torch.double).unsqueeze(-1)
            cand = propose_batch(train_x, train_y, q=q, seed=args.seed + trial_id)
            batch = [cand[i] for i in range(cand.shape[0])]

        print(f"[BATCH] launching trials {trial_id} ~ {trial_id + len(batch) - 1}", flush=True)

        batch_results = []
        with ThreadPoolExecutor(max_workers=len(batch)) as ex:
            futures = []
            for j, x_unit in enumerate(batch):
                gpu_id = gpus[j % len(gpus)]
                futures.append(
                    ex.submit(
                        run_one_trial,
                        args,
                        root,
                        hpo_dir,
                        scene,
                        trial_id + j,
                        x_unit,
                        gpu_id,
                    )
                )

            for fut in as_completed(futures):
                r = fut.result()
                batch_results.append(r)
                print(
                    f"[DONE] trial={r['trial_number']} PSNR={r['value']:.4f} "
                    f"muon_lr={r['muon_lrate']:.2e} "
                    f"adam_lr={r['lrate']:.2e} "
                    f"decay={r['lrate_decay']} "
                    f"mom={r['muon_momentum']:.3f} "
                    f"gpu={r['gpu']}",
                    flush=True,
                )

        batch_results = sorted(batch_results, key=lambda r: r["trial_number"])

        for r in batch_results:
            results.append(r)
            train_x_list.append(torch.tensor(r["unit_x"], dtype=torch.double))
            train_y_list.append(float(r["value"]))

        trial_id += len(batch_results)
        save_summary(summary_path, scene, results)
        print(f"[SAVE] {summary_path}", flush=True)

    print(f"[DONE] Summary saved to {summary_path}", flush=True)


if __name__ == "__main__":
    main()

"""
cd /data2/dong_yoon/MURF
conda activate murf

mkdir -p logs/botorch_ranksched_4param/chair

python -u stage1_ranksched_botorch_4param.py \
  --root /data2/dong_yoon/MURF \
  --run_file run_nerf_ranksched.py \
  --config configs/chair.txt \
  --basedir logs/sched/rankwsd \
  --exp_prefix chair_auto_bo \
  --n_trials 20 \
  --batch_size 4 \
  --n_init 4 \
  --n_iters 100000 \
  --gpus 2,2,3,3 \
  --optimizer aux-sign-auto-cos-inc \
  --train_scheduler rank_wsd \
  > logs/botorch_ranksched_4param/chair/chair_botorch_main.log 2>&1 &
"""