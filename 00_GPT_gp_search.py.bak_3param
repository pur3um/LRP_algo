# 00_GPT_gp_search.py
# -*- coding: utf-8 -*-
"""
BoTorch 3-parameter HPO with fixed muon_momentum=0.95 for run_nerf_ranksched_final_testpsnr_ssim.py.

Key behavior:
  1) BO objective = final validation-set mean PSNR parsed from [HPO_VAL].
  2) Acquisition = qLogExpectedImprovement.
  3) GP = SingleTaskGP(input_transform=Normalize(d=3), outcome_transform=Standardize(m=1)).
  4) Default n_init remains 4.
  5) --train_scheduler / --train-scheduler supports rank_wsd, warmup_cosine, exp_decay.
  6) After HPO, the best validation trial is evaluated once on the test set and
     test PSNR / SSIM / LPIPS are written back into the summary JSON.

The child training script must support:
  --i_valset
  --eval_testset_only
  --test_out_json
  --test_out_dir
and must print a line like:
  [HPO_VAL] Iter: 100000 mean_psnr: 31.234567 ...
"""

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
# from botorch.acquisition.logei import qLogExpectedImprovement
try:
    from botorch.acquisition.logei import qLogExpectedImprovement
    ACQF_NAME = "qLogExpectedImprovement"
    ACQF_IMPORT_SOURCE = "botorch.acquisition.logei"
except Exception:
    try:
        from botorch.acquisition import qLogExpectedImprovement
        ACQF_NAME = "qLogExpectedImprovement"
        ACQF_IMPORT_SOURCE = "botorch.acquisition"
    except Exception:
        try:
            from botorch.acquisition.monte_carlo import qLogExpectedImprovement
            ACQF_NAME = "qLogExpectedImprovement"
            ACQF_IMPORT_SOURCE = "botorch.acquisition.monte_carlo"
        except Exception:
            qLogExpectedImprovement = None
            ACQF_NAME = "CustomQLogExpectedImprovementFallback"
            ACQF_IMPORT_SOURCE = "custom_fallback"
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood


HPO_VAL_RE = re.compile(r"\[HPO_VAL\].*?mean_psnr:\s*([-+0-9.eE]+)")
TEST_PSNR_RE = re.compile(r"mean_psnr\s*:?\s*([-+0-9.eE]+)")
TEST_SSIM_RE = re.compile(r"mean_ssim\s*:?\s*([-+0-9.eE]+)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--run_file", default="run_nerf_ranksched_final_testpsnr_ssim.py")
    p.add_argument("--config", required=True)
    p.add_argument("--basedir", default="logs/sched/rankwsd")
    p.add_argument("--exp_prefix", default="botorch")

    p.add_argument("--n_trials", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--n_init", type=int, default=4)
    p.add_argument("--n_iters", type=int, default=100000)
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true")

    p.add_argument("--optimizer", default="aux-sign-auto-cos-inc")
    p.add_argument(
        "--train_scheduler", "--train-scheduler",
        dest="train_scheduler",
        default="rank_wsd",
        choices=["rank_wsd", "warmup_cosine", "exp_decay"],
        help=(
            "Scheduler passed to the child training script. "
            "Use exp_decay for the original nerf-pytorch exponential LR scheduler."
        ),
    )

    # Logging / evaluation cadence for each child training process.
    p.add_argument("--i_print", type=int, default=5000)
    p.add_argument(
        "--i_weights",
        type=int,
        default=0,
        help="0 means save weights at --n_iters.",
    )
    p.add_argument(
        "--i_valset",
        type=int,
        default=0,
        help="0 means evaluate validation-set mean PSNR at --n_iters.",
    )
    p.add_argument(
        "--i_testset",
        type=int,
        default=0,
        help=(
            "0 disables per-trial test-set evaluation during HPO by setting "
            "the child interval to n_iters + 1. Final test evaluation of the "
            "best validation trial is controlled separately."
        ),
    )
    p.add_argument(
        "--i_video",
        type=int,
        default=0,
        help="0 disables video rendering during HPO by setting interval to n_iters + 1.",
    )

    # Final test evaluation of the best validation trial.
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--eval_test_after_hpo",
        dest="eval_test_after_hpo",
        action="store_true",
        default=True,
        help="Evaluate the best validation trial once on the test set after HPO. Default: enabled.",
    )
    g.add_argument(
        "--no_eval_test_after_hpo",
        dest="eval_test_after_hpo",
        action="store_false",
        help="Disable final test evaluation after HPO.",
    )
    p.add_argument(
        "--test_gpu",
        type=str,
        default=None,
        help="GPU id for final test evaluation. Default: reuse the best trial's GPU id.",
    )
    p.add_argument("--lpips_net", type=str, default="alex", choices=["alex", "vgg", "squeeze"])
    p.add_argument("--test_out_dir_name", type=str, default="testset_eval")
    p.add_argument("--test_metrics_json_name", type=str, default="test_metrics_eval.json")

    return p.parse_args()


def _resolve_under_root(root: Path, path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def unit_to_params(x):
    """Map a point in [0, 1]^3 to the actual hyperparameter values. muon_momentum is fixed to 0.95."""
    x = x.detach().cpu().double()

    def log_interp(u, lo, hi):
        lo_t = torch.log10(torch.tensor(lo, dtype=torch.double))
        hi_t = torch.log10(torch.tensor(hi, dtype=torch.double))
        return float(10 ** (lo_t + u * (hi_t - lo_t)))

    return {
        "muon_lrate": log_interp(x[0], 1e-4, 5e-3),
        "lrate": log_interp(x[1], 1e-4, 1e-3),
        "lrate_decay": int(round(float(100 + x[2] * 400))),
        "muon_momentum": 0.95,
    }


def parse_metric(stdout_path):
    """
    Parse only validation-set mean PSNR emitted by the child training script.

    Expected line:
        [HPO_VAL] Iter: 100000 mean_psnr: 31.234567 ...

    This intentionally does NOT parse [VAL] single-image PSNR logs and does NOT
    parse test-set metrics, so BO selects hyperparameters by validation-set mean.
    """
    stdout_path = Path(stdout_path)
    text = stdout_path.read_text(errors="ignore")
    vals = [float(m.group(1)) for m in HPO_VAL_RE.finditer(text)]

    if not vals:
        tail = "\n".join(text.splitlines()[-40:])
        raise RuntimeError(
            f"No [HPO_VAL] validation mean PSNR found in {stdout_path}.\n"
            f"Make sure the child script supports --i_valset.\n"
            f"Last stdout lines:\n{tail}"
        )

    return vals[-1]


def _base_child_cmd(args, root, expname, params):
    """Common command-line arguments needed to reconstruct the same model/run."""
    return [
        sys.executable,
        str(_resolve_under_root(root, args.run_file)),
        "--basedir", str(_resolve_under_root(root, args.basedir)),
        "--config", str(_resolve_under_root(root, args.config)),
        "--expname", expname,
        "--optimizer", args.optimizer,
        "--train-scheduler", args.train_scheduler,
        "--muon_lrate", str(params["muon_lrate"]),
        "--lrate", str(params["lrate"]),
        "--lrate_decay", str(params["lrate_decay"]),
        "--muon_momentum", str(params["muon_momentum"]),
        "--lowrank_rank_start", "150",
        "--lowrank_rank_end", "250",
        "--lowrank_auto_init_rank_start",
        "--N_iters", str(args.n_iters),
        "--seed", str(params.get("seed", args.seed)),
    ]


def run_one_trial(args, root, hpo_dir, scene, trial_id, x_unit, gpu_id):
    params = unit_to_params(x_unit)
    params["seed"] = int(args.seed + trial_id)

    expname = (
        f"{args.exp_prefix}_{scene}_trial{trial_id:03d}"
        f"_mlr{params['muon_lrate']:.2e}"
        f"_alr{params['lrate']:.2e}"
        f"_decay{params['lrate_decay']}"
        f"_mom{params['muon_momentum']:.3f}"
    ).replace("+", "").replace(".", "p")

    hpo_trial_dir = hpo_dir / expname
    hpo_trial_dir.mkdir(parents=True, exist_ok=True)

    train_basedir = _resolve_under_root(root, args.basedir)
    train_run_dir = train_basedir / expname

    stdout_path = hpo_trial_dir / "stdout.txt"
    stderr_path = hpo_trial_dir / "stderr.txt"

    i_weights = args.i_weights if args.i_weights > 0 else args.n_iters
    i_valset = args.i_valset if args.i_valset > 0 else args.n_iters
    i_testset = args.i_testset if args.i_testset > 0 else args.n_iters + 1
    i_video = args.i_video if args.i_video > 0 else args.n_iters + 1

    cmd = _base_child_cmd(args, root, expname, params) + [
        "--i_print", str(args.i_print),
        "--i_weights", str(i_weights),
        "--i_valset", str(i_valset),
        "--i_testset", str(i_testset),
        "--i_video", str(i_video),
        "--no_reload",
    ]
    if args.deterministic:
        cmd.append("--deterministic")

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

    val_mean_psnr = parse_metric(stdout_path)

    return {
        "trial_number": trial_id,
        "value": val_mean_psnr,
        "val_mean_psnr": val_mean_psnr,
        "train_scheduler": args.train_scheduler,
        "seed": params["seed"],
        "muon_lrate": params["muon_lrate"],
        "lrate": params["lrate"],
        "lrate_decay": params["lrate_decay"],
        "muon_momentum": params["muon_momentum"],
        "unit_x": [float(v) for v in x_unit.tolist()],
        "expname": expname,
        "gpu": str(gpu_id),
        "hpo_trial_dir": str(hpo_trial_dir),
        "train_basedir": str(train_basedir),
        "train_run_dir": str(train_run_dir),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }


def propose_batch(train_x, train_y, q, seed):
    """Fit GP and propose q candidates using qLogExpectedImprovement."""
    train_x = train_x.double()
    train_y = train_y.double()

    if train_x.ndim != 2 or train_x.shape[-1] != 3:
        raise ValueError(f"train_x must have shape n x 3, got {tuple(train_x.shape)}")
    if train_y.ndim != 2 or train_y.shape[-1] != 1:
        raise ValueError(f"train_y must have shape n x 1, got {tuple(train_y.shape)}")
    if train_x.shape[0] < 2:
        raise ValueError("At least two completed trials are required to fit a GP.")

    model = SingleTaskGP(
        train_x,
        train_y,
        input_transform=Normalize(d=3),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]), seed=seed)

    # train_y is on the original PSNR scale. With outcome_transform=Standardize,
    # BoTorch untransforms the posterior to the original scale for acquisition use.
    acq = qLogExpectedImprovement(
        model=model,
        best_f=train_y.max(),
        sampler=sampler,
    )

    bounds = torch.stack([
        torch.zeros(3, dtype=torch.double),
        torch.ones(3, dtype=torch.double),
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


def _load_test_metrics_from_json_or_stdout(out_json, stdout_path):
    out_json = Path(out_json)
    stdout_path = Path(stdout_path)

    if out_json.exists():
        with open(out_json, "r") as f:
            metrics = json.load(f)
        return metrics

    text = stdout_path.read_text(errors="ignore") if stdout_path.exists() else ""
    psnr_match = None
    ssim_match = None
    for m in TEST_PSNR_RE.finditer(text):
        psnr_match = m
    for m in TEST_SSIM_RE.finditer(text):
        ssim_match = m

    if psnr_match is None or ssim_match is None:
        raise RuntimeError(
            f"Final test metrics JSON was not found at {out_json}, and PSNR/SSIM "
            f"could not be parsed from {stdout_path}."
        )

    return {
        "mean_psnr": float(psnr_match.group(1)),
        "mean_ssim": float(ssim_match.group(1)),
        "mean_lpips": None,
        "source": "stdout_fallback",
    }


def run_final_test_eval(args, root, hpo_dir, scene, best):
    """Evaluate the best validation trial once on the test set."""
    params = {
        "muon_lrate": best["muon_lrate"],
        "lrate": best["lrate"],
        "lrate_decay": best["lrate_decay"],
        "muon_momentum": best["muon_momentum"],
        "seed": best.get("seed", args.seed + int(best["trial_number"])),
    }
    expname = best["expname"]
    train_run_dir = Path(best["train_run_dir"]).resolve()
    train_run_dir.mkdir(parents=True, exist_ok=True)

    out_json = train_run_dir / args.test_metrics_json_name
    out_dir = train_run_dir / args.test_out_dir_name

    test_stdout = hpo_dir / f"{scene}_best_trial{int(best['trial_number']):03d}_test_eval_stdout.txt"
    test_stderr = hpo_dir / f"{scene}_best_trial{int(best['trial_number']):03d}_test_eval_stderr.txt"

    cmd = _base_child_cmd(args, root, expname, params) + [
        "--eval_testset_only",
        "--test_out_json", str(out_json),
        "--test_out_dir", str(out_dir),
        "--lpips_net", args.lpips_net,
        # These intervals are irrelevant in eval-only mode, but setting them
        # avoids accidental long-running render-video paths if the script changes.
        "--i_print", str(args.i_print),
        "--i_weights", str(args.n_iters),
        "--i_valset", str(args.n_iters + 1),
        "--i_testset", str(args.n_iters + 1),
        "--i_video", str(args.n_iters + 1),
    ]
    if args.deterministic:
        cmd.append("--deterministic")

    eval_gpu = args.test_gpu if args.test_gpu is not None else best.get("gpu", "0")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(eval_gpu)
    env["PYTHONUNBUFFERED"] = "1"

    with open(test_stdout, "w") as fout, open(test_stderr, "w") as ferr:
        ret = subprocess.call(
            cmd,
            cwd=str(root),
            env=env,
            stdout=fout,
            stderr=ferr,
        )

    if ret != 0:
        return {
            "status": "failed",
            "best_trial_number": int(best["trial_number"]),
            "gpu": str(eval_gpu),
            "stdout_log": str(test_stdout),
            "stderr_log": str(test_stderr),
            "test_metrics_json": str(out_json),
            "test_out_dir": str(out_dir),
            "error": f"Final test evaluation returned non-zero exit code {ret}.",
        }

    metrics = _load_test_metrics_from_json_or_stdout(out_json, test_stdout)
    return {
        "status": "ok",
        "best_trial_number": int(best["trial_number"]),
        "expname": expname,
        "gpu": str(eval_gpu),
        "test_mean_psnr": metrics.get("mean_psnr"),
        "test_mean_ssim": metrics.get("mean_ssim"),
        "test_mean_lpips": metrics.get("mean_lpips"),
        "num_test_views": metrics.get("num_test_views"),
        "checkpoint_step": metrics.get("checkpoint_step"),
        "test_metrics_json": str(out_json),
        "test_out_dir": str(out_dir),
        "stdout_log": str(test_stdout),
        "stderr_log": str(test_stderr),
        "raw_metrics": metrics,
    }


def save_summary(path, scene, results, train_scheduler=None, final_test_metrics=None):
    top = sorted(results, key=lambda r: r["value"], reverse=True)
    best = top[0] if top else None
    summary = {
        "scene": scene,
        "method": "BoTorch batch BO, 3 searched params with fixed muon_momentum=0.95",
        "train_scheduler": train_scheduler,
        "selection_metric": "final_validation_set_mean_psnr",
        "objective": "final_validation_set_mean_psnr",
        "final_test_policy": "evaluate_best_validation_trial_once_after_hpo",
        "acquisition": "qLogExpectedImprovement",
        "gp_model": "SingleTaskGP(input_transform=Normalize(d=3), outcome_transform=Standardize(m=1))",
        "n_complete_trials": len(results),
        "best_value": best["value"] if best else None,
        "best_val_mean_psnr": best["val_mean_psnr"] if best else None,
        "best_test_mean_psnr": final_test_metrics.get("test_mean_psnr") if final_test_metrics else None,
        "best_test_mean_ssim": final_test_metrics.get("test_mean_ssim") if final_test_metrics else None,
        "best_test_mean_lpips": final_test_metrics.get("test_mean_lpips") if final_test_metrics else None,
        "best_params": {
            "muon_lrate": best["muon_lrate"],
            "lrate": best["lrate"],
            "lrate_decay": best["lrate_decay"],
            "muon_momentum": best["muon_momentum"],
        } if best else None,
        "best_unit_x": best["unit_x"] if best else None,
        "best_trial_number": best["trial_number"] if best else None,
        "final_test_evaluation": final_test_metrics,
        "top_trials": top,
    }
    path.write_text(json.dumps(summary, indent=2))


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    scene = Path(args.config).stem

    if args.n_init != 4:
        print(f"[WARN] --n_init is {args.n_init}, but requested default is 4.", flush=True)

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id, e.g. --gpus 0,1")

    hpo_dir = root / "logs" / "botorch_ranksched_3param_mom095" / scene
    hpo_dir.mkdir(parents=True, exist_ok=True)
    summary_path = hpo_dir / f"{scene}_botorch_ranksched_3param_mom095_summary.json"

    torch.manual_seed(args.seed)

    results = []
    train_x_list, train_y_list = [], []

    sobol = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=args.seed)
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
                    f"[DONE] trial={r['trial_number']} val_mean_PSNR={r['value']:.4f} "
                    f"muon_lr={r['muon_lrate']:.2e} "
                    f"adam_lr={r['lrate']:.2e} "
                    f"decay={r['lrate_decay']} "
                    f"mom={r['muon_momentum']:.3f} "
                    f"scheduler={r.get('train_scheduler', args.train_scheduler)} "
                    f"gpu={r['gpu']}",
                    flush=True,
                )

        batch_results = sorted(batch_results, key=lambda r: r["trial_number"])

        for r in batch_results:
            results.append(r)
            train_x_list.append(torch.tensor(r["unit_x"], dtype=torch.double))
            train_y_list.append(float(r["value"]))

        trial_id += len(batch_results)
        save_summary(summary_path, scene, results, train_scheduler=args.train_scheduler)
        print(f"[SAVE] {summary_path}", flush=True)

    final_test_metrics = None
    if args.eval_test_after_hpo and results:
        best = sorted(results, key=lambda r: r["value"], reverse=True)[0]
        print(
            f"[TEST_EVAL] evaluating best validation trial={best['trial_number']} "
            f"expname={best['expname']} on test set",
            flush=True,
        )
        final_test_metrics = run_final_test_eval(args, root, hpo_dir, scene, best)

        # Store test metrics directly inside the corresponding trial entry too.
        for r in results:
            if r["trial_number"] == best["trial_number"]:
                r["final_test_evaluation"] = final_test_metrics
                if final_test_metrics.get("status") == "ok":
                    r["test_mean_psnr"] = final_test_metrics.get("test_mean_psnr")
                    r["test_mean_ssim"] = final_test_metrics.get("test_mean_ssim")
                    r["test_mean_lpips"] = final_test_metrics.get("test_mean_lpips")
                break

        if final_test_metrics.get("status") == "ok":
            test_psnr = final_test_metrics.get('test_mean_psnr')
            test_ssim = final_test_metrics.get('test_mean_ssim')
            if test_psnr is not None and test_ssim is not None:
                print(
                    f"[TEST_DONE] trial={best['trial_number']} "
                    f"test_PSNR={test_psnr:.6f} test_SSIM={test_ssim:.6f}",
                    flush=True,
                )
            else:
                print(
                    f"[TEST_DONE] trial={best['trial_number']} "
                    f"test_PSNR={test_psnr} test_SSIM={test_ssim}",
                    flush=True,
                )
        else:
            print(
                f"[TEST_FAIL] trial={best['trial_number']} "
                f"error={final_test_metrics.get('error')}",
                flush=True,
            )

    save_summary(
        summary_path,
        scene,
        results,
        train_scheduler=args.train_scheduler,
        final_test_metrics=final_test_metrics,
    )
    print(f"[DONE] Summary saved to {summary_path}", flush=True)


if __name__ == "__main__":
    main()

"""
Example:

cd /data2/dong_yoon/MURF
conda activate murf

mkdir -p logs/GP_test_log/chair

python -u 00_GPT_gp_search.py \
  --root /data2/dong_yoon/MURF \
  --run_file 00_run_nerf_ranksched_final.py \
  --config configs/chair.txt \
  --basedir logs/GP_test \
  --exp_prefix chair_auto_bo \
  --n_trials 2 \
  --batch_size 2 \
  --n_init 2 \
  --n_iters 100000 \
  --gpus 2,3 \
  --optimizer aux-sign-auto-cos-inc \
  --train-scheduler rank_wsd \
  --test_gpu 2 \
  --deterministic \
  --no_eval_test_after_hpo \
  > logs/GP_test_log/chair/chair_botorch_main.log 2>&1 &

#@ 다른 format
python -u 00_GPT_gp_search.py   --root /home/greenx9/nerf-pytorch/LRP_algo   --run_file 00_run_nerf_ranksched_final.py   --config configs/chair.txt   --basedir logs/GP_test   --exp_prefix chair_auto_bo   --n_trials 2   --batch_size 2   --n_init 2   --n_iters 100000   --gpus 0   --optimizer aux-muon   --train-scheduler exp_decay   --test_gpu 2   --deterministic > logs/GP_test_log/chair/chair_botorch_main.log 2>&1 &
"""



"""
cd /home/greenx9/data2/LRP_algo
conda activate nerf
bash run_104_ours_scene_queue.sh
"""

#Ctrl + b 누르고, 그 다음 d

"""
tmux attach -t ours104
"""

#watch -n 1 nvidia-smi