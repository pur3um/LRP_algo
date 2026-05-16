# 00_GPT_gp_search.py
# -*- coding: utf-8 -*-
"""
BoTorch BO for (optimizer, scheduler) pair HPO with up to 20 hyperparameters.

Supported (optimizer, scheduler) pairs:
  - adam              + exp_decay   : 2 active HPs
  - aux-muon          + exp_decay   : 10 active HPs
  - aux-sign-auto-cos-inc + rank_wsd: 19 active HPs

The union of all active HPs across the three pairs is exactly 20.
Each pair runs its own GP / qLogEI BoTorch loop over its own active subspace.

Child training script must support:
  --i_valset, --eval_testset_only, --test_out_json, --test_out_dir
and must print a line:
  [HPO_VAL] Iter: <N> mean_psnr: <X> ...
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch

# Acquisition function resolution.
#
# qLogExpectedImprovement (qLogEI) was introduced in botorch 0.9.0. On older
# botorch versions (e.g. 0.8.x shipped with some conda envs), fall back to the
# numerically-stable-enough qExpectedImprovement (qEI). For the HPO scale used
# here (~20-32 trials) qEI is a perfectly serviceable replacement.
ACQF_CLS = None
ACQF_NAME = None
ACQF_IMPORT_SOURCE = None
for _path, _attr in (
    ("botorch.acquisition.logei",       "qLogExpectedImprovement"),
    ("botorch.acquisition",             "qLogExpectedImprovement"),
    ("botorch.acquisition.monte_carlo", "qLogExpectedImprovement"),
    ("botorch.acquisition.monte_carlo", "qExpectedImprovement"),
    ("botorch.acquisition",             "qExpectedImprovement"),
):
    try:
        _mod = __import__(_path, fromlist=[_attr])
        ACQF_CLS = getattr(_mod, _attr)
        ACQF_NAME = _attr
        ACQF_IMPORT_SOURCE = _path
        break
    except (ImportError, AttributeError):
        continue
if ACQF_CLS is None:
    raise ImportError(
        "Neither qLogExpectedImprovement nor qExpectedImprovement is available "
        "in the installed botorch. Please install botorch>=0.8."
    )

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


# =============================================================================
# HP Registry: 20 master hyperparameters
# =============================================================================
# kind:
#   "log"     : 10**( log10(low) + u*(log10(hi)-log10(low)) )
#   "uniform" : low + u*(hi-low)
#   "int"     : round(low + u*(hi-low))

HP_REGISTRY = {
    # ---- shared core LR ----
    "lrate":                       {"kind": "log",     "low": 1e-4,  "high": 2e-3},
    # ---- Muon optimizer ----
    "muon_lrate":                  {"kind": "log",     "low": 1e-4,  "high": 5e-3},
    "muon_momentum":               {"kind": "uniform", "low": 0.85,  "high": 0.99},
    "muon_decay":                  {"kind": "log",     "low": 1e-6,  "high": 1e-2},
    "lowrank_ns_steps":            {"kind": "int",     "low": 3,     "high": 10},
    # ---- Muon's auxiliary Adam ----
    "muon_aux_beta1":              {"kind": "uniform", "low": 0.85,  "high": 0.95},
    "muon_aux_beta2":              {"kind": "uniform", "low": 0.95,  "high": 0.9999},
    "muon_aux_eps":                {"kind": "log",     "low": 1e-6,  "high": 1e-3},
    "muon_aux_weight_decay":       {"kind": "log",     "low": 1e-6,  "high": 1e-2},
    # ---- exp_decay scheduler ----
    "lrate_decay":                 {"kind": "int",     "low": 100,   "high": 500},
    # ---- rank_wsd scheduler ----
    "sched_warmup_frac":           {"kind": "uniform", "low": 0.005, "high": 0.05},
    "sched_min_lr_ratio":          {"kind": "uniform", "low": 0.05,  "high": 0.5},
    "sched_decay_start_frac":      {"kind": "uniform", "low": 0.5,   "high": 0.9},
    # ---- low-rank / aux-sign-auto-cos-inc rank schedule ----
    "lowrank_rank_start":          {"kind": "int",     "low": 50,    "high": 200},
    "lowrank_rank_end":            {"kind": "int",     "low": 150,   "high": 350},
    "lowrank_oversample":          {"kind": "int",     "low": 2,     "high": 8},
    "lowrank_init_energy":         {"kind": "uniform", "low": 0.85,  "high": 0.9999},
    "lowrank_init_probe_steps":    {"kind": "int",     "low": 4,     "high": 32},
    "lowrank_init_round_multiple": {"kind": "int",     "low": 4,     "high": 16},
    "rank_schedule_default_frac":  {"kind": "uniform", "low": 0.5,   "high": 0.95},
}
assert len(HP_REGISTRY) == 20, f"HP_REGISTRY must have exactly 20 entries, got {len(HP_REGISTRY)}."


# =============================================================================
# Pair presets: per (optimizer, scheduler) active hyperparameters
# =============================================================================
PAIR_PRESETS = {
    "adam+exp_decay": {
        "optimizer": "adam",
        "train_scheduler": "exp_decay",
        "active": ["lrate", "lrate_decay"],
    },
    "muon+exp_decay": {
        "optimizer": "aux-muon",
        "train_scheduler": "exp_decay",
        "active": [
            "lrate", "muon_lrate",
            "muon_momentum", "muon_decay", "lowrank_ns_steps",
            "muon_aux_beta1", "muon_aux_beta2",
            "muon_aux_eps", "muon_aux_weight_decay",
            "lrate_decay",
        ],
    },
    "ours+rank_wsd": {
        "optimizer": "aux-sign-auto-cos-inc",
        "train_scheduler": "rank_wsd",
        "active": [
            "lrate", "muon_lrate",
            "muon_momentum", "muon_decay", "lowrank_ns_steps",
            "muon_aux_beta1", "muon_aux_beta2",
            "muon_aux_eps", "muon_aux_weight_decay",
            "sched_warmup_frac", "sched_min_lr_ratio", "sched_decay_start_frac",
            "lowrank_rank_start", "lowrank_rank_end", "lowrank_oversample",
            "lowrank_init_energy", "lowrank_init_probe_steps",
            "lowrank_init_round_multiple", "rank_schedule_default_frac",
        ],
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--run_file", default="00_run_nerf_ranksched_final.py")
    p.add_argument("--config", required=True)
    p.add_argument("--basedir", default="logs/sched/multi_pair")
    p.add_argument("--exp_prefix", default="botorch20")

    p.add_argument(
        "--pair",
        default="all",
        choices=list(PAIR_PRESETS.keys()) + ["all"],
        help="Which (optimizer, scheduler) pair to run. 'all' runs the three pairs sequentially.",
    )

    p.add_argument("--n_trials", type=int, default=20,
                   help="Number of BO trials per pair. Same budget across pairs (fair comparison).")
    p.add_argument("--batch_size", type=int, default=4,
                   help="Parallel q for BO. Effective parallelism = min(batch_size, len(gpus)).")
    p.add_argument("--n_init", type=int, default=4,
                   help="Number of initial Sobol points per pair before BO starts.")
    p.add_argument("--n_iters", type=int, default=100000,
                   help="Child training script N_iters.")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true")

    # Per-pair n_trials override (e.g. give "ours" more trials due to high dim).
    p.add_argument("--n_trials_adam", type=int, default=None)
    p.add_argument("--n_trials_muon", type=int, default=None)
    p.add_argument("--n_trials_ours", type=int, default=None)

    # Logging cadence inside each child training process.
    p.add_argument("--i_print", type=int, default=5000)
    p.add_argument("--i_weights", type=int, default=0,
                   help="0 means save weights at --n_iters.")
    p.add_argument("--i_valset", type=int, default=0,
                   help="0 means evaluate validation-set mean PSNR at --n_iters.")
    p.add_argument("--i_testset", type=int, default=0,
                   help="0 disables per-trial test-set evaluation during HPO.")
    p.add_argument("--i_video", type=int, default=0,
                   help="0 disables video rendering during HPO.")

    g = p.add_mutually_exclusive_group()
    g.add_argument("--eval_test_after_hpo", dest="eval_test_after_hpo",
                   action="store_true", default=True,
                   help="Evaluate the best validation trial once on the test set after HPO.")
    g.add_argument("--no_eval_test_after_hpo", dest="eval_test_after_hpo",
                   action="store_false",
                   help="Disable final test evaluation after HPO.")
    p.add_argument("--test_gpu", type=str, default=None,
                   help="GPU id for final test evaluation. Default: reuse best trial's GPU.")
    p.add_argument("--lpips_net", type=str, default="alex",
                   choices=["alex", "vgg", "squeeze"])
    p.add_argument("--test_out_dir_name", type=str, default="testset_eval")
    p.add_argument("--test_metrics_json_name", type=str, default="test_metrics_eval.json")

    return p.parse_args()


# =============================================================================
# Unit-cube -> hyperparameter mapping (with constraint enforcement)
# =============================================================================
def _map_one(u_val, spec):
    if spec["kind"] == "log":
        lo = math.log10(spec["low"])
        hi = math.log10(spec["high"])
        return float(10 ** (lo + u_val * (hi - lo)))
    if spec["kind"] == "uniform":
        return float(spec["low"] + u_val * (spec["high"] - spec["low"]))
    if spec["kind"] == "int":
        return int(round(spec["low"] + u_val * (spec["high"] - spec["low"])))
    raise ValueError(f"Unknown HP kind: {spec['kind']}")


def unit_to_params(x, active_hp_names):
    """Map a point in [0,1]^d to a dict of active HP values. d = len(active_hp_names)."""
    x = x.detach().cpu().double()
    if x.numel() != len(active_hp_names):
        raise ValueError(
            f"unit_to_params: x has {x.numel()} dims but active_hp_names has {len(active_hp_names)}."
        )

    params = {}
    for i, name in enumerate(active_hp_names):
        params[name] = _map_one(float(x[i]), HP_REGISTRY[name])

    enforce_constraints(params)
    return params


def enforce_constraints(params):
    """Enforce HP constraints in-place after sampling.

    Currently:
      - lowrank_rank_start <= lowrank_rank_end (swap if violated).
      - lowrank_init_probe_steps: keep within a small safety bound.
    """
    s_key, e_key = "lowrank_rank_start", "lowrank_rank_end"
    if s_key in params and e_key in params:
        s, e = int(params[s_key]), int(params[e_key])
        if s > e:
            params[s_key], params[e_key] = e, s

    # Make sure auto-init round-multiple does not push rank_start above rank_end
    # too aggressively. The child script rounds rank_start up to a multiple of
    # init_round_multiple; we clamp to keep start < end after rounding.
    if all(k in params for k in (s_key, e_key, "lowrank_init_round_multiple")):
        m = max(1, int(params["lowrank_init_round_multiple"]))
        rounded_start = ((int(params[s_key]) + m - 1) // m) * m
        if rounded_start >= int(params[e_key]):
            # Reduce rounded_start down to the largest m-multiple < rank_end.
            new_start = max(int(HP_REGISTRY[s_key]["low"]),
                            ((int(params[e_key]) - 1) // m) * m)
            params[s_key] = int(new_start)


# =============================================================================
# Child command construction
# =============================================================================
def _resolve_under_root(root: Path, path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


# Mapping of HP_REGISTRY keys to child CLI flag building rules.
# Each entry: (flag_name, formatter_fn) — formatter_fn takes the raw value and returns str.
# 'muon_aux_beta1' and 'muon_aux_beta2' are handled jointly because the child takes
# them as a single --muon_aux_betas "b1,b2" argument.
_SIMPLE_FLAGS = {
    "lrate":                        ("--lrate",                        lambda v: f"{float(v):.8g}"),
    "lrate_decay":                  ("--lrate_decay",                  lambda v: str(int(v))),
    "muon_lrate":                   ("--muon_lrate",                   lambda v: f"{float(v):.8g}"),
    "muon_momentum":                ("--muon_momentum",                lambda v: f"{float(v):.6f}"),
    "muon_decay":                   ("--muon_decay",                   lambda v: f"{float(v):.8g}"),
    "muon_aux_eps":                 ("--muon_aux_eps",                 lambda v: f"{float(v):.8g}"),
    "muon_aux_weight_decay":        ("--muon_aux_weight_decay",        lambda v: f"{float(v):.8g}"),
    "lowrank_ns_steps":             ("--lowrank_ns_steps",             lambda v: str(int(v))),
    "sched_warmup_frac":            ("--sched_warmup_frac",            lambda v: f"{float(v):.6f}"),
    "sched_min_lr_ratio":           ("--sched_min_lr_ratio",           lambda v: f"{float(v):.6f}"),
    "sched_decay_start_frac":       ("--sched_decay_start_frac",       lambda v: f"{float(v):.6f}"),
    "lowrank_rank_start":           ("--lowrank_rank_start",           lambda v: str(int(v))),
    "lowrank_rank_end":             ("--lowrank_rank_end",             lambda v: str(int(v))),
    "lowrank_oversample":           ("--lowrank_oversample",           lambda v: str(int(v))),
    "lowrank_init_energy":          ("--lowrank_init_energy",          lambda v: f"{float(v):.6g}"),
    "lowrank_init_probe_steps":     ("--lowrank_init_probe_steps",     lambda v: str(int(v))),
    "lowrank_init_round_multiple":  ("--lowrank_init_round_multiple",  lambda v: str(int(v))),
    "rank_schedule_default_frac":   ("--rank_schedule_default_frac",   lambda v: f"{float(v):.6f}"),
}


def _base_child_cmd(args, root, pair_preset, expname, params):
    cmd = [
        sys.executable,
        str(_resolve_under_root(root, args.run_file)),
        "--basedir", str(_resolve_under_root(root, args.basedir)),
        "--config", str(_resolve_under_root(root, args.config)),
        "--expname", expname,
        "--optimizer", pair_preset["optimizer"],
        "--train-scheduler", pair_preset["train_scheduler"],
        "--N_iters", str(args.n_iters),
        "--seed", str(int(params.get("__seed__", args.seed))),
    ]

    # Simple 1-to-1 flags.
    for key, (flag, fmt) in _SIMPLE_FLAGS.items():
        if key in params:
            cmd += [flag, fmt(params[key])]

    # Joint flag: --muon_aux_betas "b1,b2".
    if "muon_aux_beta1" in params or "muon_aux_beta2" in params:
        b1 = float(params.get("muon_aux_beta1", 0.9))
        b2 = float(params.get("muon_aux_beta2", 0.95))
        cmd += ["--muon_aux_betas", f"{b1:.6f},{b2:.6f}"]

    # Pair-specific extras (not BO-tuned, but required for the optimizer).
    if pair_preset["optimizer"] == "aux-sign-auto-cos-inc":
        cmd += ["--lowrank_auto_init_rank_start"]
        # Sensible defaults if not BO-tuned (these are tuned for the ours pair).
        if "lowrank_rank_start" not in params:
            cmd += ["--lowrank_rank_start", "150"]
        if "lowrank_rank_end" not in params:
            cmd += ["--lowrank_rank_end", "250"]

    return cmd


def parse_metric(stdout_path):
    """Parse only validation-set mean PSNR emitted by the child training script."""
    stdout_path = Path(stdout_path)
    text = stdout_path.read_text(errors="ignore")
    vals = [float(m.group(1)) for m in HPO_VAL_RE.finditer(text)]

    if not vals:
        tail = "\n".join(text.splitlines()[-40:])
        raise RuntimeError(
            f"No [HPO_VAL] validation mean PSNR found in {stdout_path}.\n"
            f"Make sure the child script supports --i_valset and emits [HPO_VAL].\n"
            f"Last stdout lines:\n{tail}"
        )

    return vals[-1]


# =============================================================================
# Per-trial execution
# =============================================================================
def _format_expname(args, scene, pair_key, trial_id, params):
    lr_tag = f"lr{params['lrate']:.2e}" if "lrate" in params else "lr-na"
    mlr_tag = f"mlr{params['muon_lrate']:.2e}" if "muon_lrate" in params else ""
    decay_tag = f"d{int(params['lrate_decay'])}" if "lrate_decay" in params else ""
    rwsd_tag = ""
    if "sched_min_lr_ratio" in params:
        rwsd_tag = f"minr{params['sched_min_lr_ratio']:.2f}"
    pair_short = {"adam+exp_decay": "adamexp",
                  "muon+exp_decay": "muonexp",
                  "ours+rank_wsd": "ourswsd"}[pair_key]

    raw = (
        f"{args.exp_prefix}_{pair_short}_{scene}_t{trial_id:03d}"
        f"_{lr_tag}_{mlr_tag}_{decay_tag}_{rwsd_tag}"
    )
    return raw.replace("+", "").replace(".", "p").replace("__", "_").rstrip("_")


def run_one_trial(args, root, pair_dir, scene, pair_key, pair_preset,
                  trial_id, x_unit, gpu_id):
    params = unit_to_params(x_unit, pair_preset["active"])
    params["__seed__"] = int(args.seed + trial_id)

    expname = _format_expname(args, scene, pair_key, trial_id, params)

    hpo_trial_dir = pair_dir / expname
    hpo_trial_dir.mkdir(parents=True, exist_ok=True)

    train_basedir = _resolve_under_root(root, args.basedir)
    train_run_dir = train_basedir / expname

    stdout_path = hpo_trial_dir / "stdout.txt"
    stderr_path = hpo_trial_dir / "stderr.txt"

    i_weights = args.i_weights if args.i_weights > 0 else args.n_iters
    i_valset = args.i_valset if args.i_valset > 0 else args.n_iters
    i_testset = args.i_testset if args.i_testset > 0 else args.n_iters + 1
    i_video = args.i_video if args.i_video > 0 else args.n_iters + 1

    cmd = _base_child_cmd(args, root, pair_preset, expname, params) + [
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
        raise RuntimeError(f"Trial {trial_id} ({pair_key}) failed on GPU {gpu_id}. See {stderr_path}")

    val_mean_psnr = parse_metric(stdout_path)

    record = {
        "trial_number": trial_id,
        "pair_key": pair_key,
        "optimizer": pair_preset["optimizer"],
        "train_scheduler": pair_preset["train_scheduler"],
        "value": val_mean_psnr,
        "val_mean_psnr": val_mean_psnr,
        "seed": int(params["__seed__"]),
        "params": {k: v for k, v in params.items() if k != "__seed__"},
        "unit_x": [float(v) for v in x_unit.tolist()],
        "expname": expname,
        "gpu": str(gpu_id),
        "hpo_trial_dir": str(hpo_trial_dir),
        "train_basedir": str(train_basedir),
        "train_run_dir": str(train_run_dir),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }
    return record


# =============================================================================
# BoTorch proposal
# =============================================================================
def propose_batch(train_x, train_y, q, seed, d):
    """Fit GP and propose q candidates using the resolved acquisition (qLogEI or qEI)."""
    train_x = train_x.double()
    train_y = train_y.double()

    if train_x.ndim != 2 or train_x.shape[-1] != d:
        raise ValueError(f"train_x must have shape n x {d}, got {tuple(train_x.shape)}")
    if train_y.ndim != 2 or train_y.shape[-1] != 1:
        raise ValueError(f"train_y must have shape n x 1, got {tuple(train_y.shape)}")
    if train_x.shape[0] < 2:
        raise ValueError("At least two completed trials are required to fit a GP.")

    model = SingleTaskGP(
        train_x,
        train_y,
        input_transform=Normalize(d=d),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]), seed=seed)

    acq = ACQF_CLS(
        model=model,
        best_f=train_y.max(),
        sampler=sampler,
    )

    bounds = torch.stack([
        torch.zeros(d, dtype=torch.double),
        torch.ones(d, dtype=torch.double),
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


# =============================================================================
# Final test-set evaluation of the best validation trial
# =============================================================================
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


def run_final_test_eval(args, root, pair_dir, scene, pair_key, pair_preset, best):
    """Evaluate the best validation trial once on the test set."""
    params = dict(best["params"])
    params["__seed__"] = int(best.get("seed", args.seed + int(best["trial_number"])))
    expname = best["expname"]
    train_run_dir = Path(best["train_run_dir"]).resolve()
    train_run_dir.mkdir(parents=True, exist_ok=True)

    out_json = train_run_dir / args.test_metrics_json_name
    out_dir = train_run_dir / args.test_out_dir_name

    test_stdout = pair_dir / f"{scene}_best_trial{int(best['trial_number']):03d}_test_eval_stdout.txt"
    test_stderr = pair_dir / f"{scene}_best_trial{int(best['trial_number']):03d}_test_eval_stderr.txt"

    cmd = _base_child_cmd(args, root, pair_preset, expname, params) + [
        "--eval_testset_only",
        "--test_out_json", str(out_json),
        "--test_out_dir", str(out_dir),
        "--lpips_net", args.lpips_net,
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


# =============================================================================
# Summary persistence
# =============================================================================
def save_summary(path, scene, pair_key, pair_preset, results,
                 final_test_metrics=None, n_trials=None):
    top = sorted(results, key=lambda r: r["value"], reverse=True)
    best = top[0] if top else None
    summary = {
        "scene": scene,
        "pair_key": pair_key,
        "optimizer": pair_preset["optimizer"],
        "train_scheduler": pair_preset["train_scheduler"],
        "active_hps": pair_preset["active"],
        "search_dim": len(pair_preset["active"]),
        "method": "BoTorch batch BO (qLogEI) per (optimizer, scheduler) pair",
        "selection_metric": "final_validation_set_mean_psnr",
        "objective": "final_validation_set_mean_psnr",
        "final_test_policy": "evaluate_best_validation_trial_once_after_hpo",
        "acquisition": ACQF_NAME,
        "acquisition_source": ACQF_IMPORT_SOURCE,
        "gp_model": "SingleTaskGP(input_transform=Normalize, outcome_transform=Standardize(m=1))",
        "n_complete_trials": len(results),
        "n_trials_target": n_trials,
        "best_value": best["value"] if best else None,
        "best_val_mean_psnr": best["val_mean_psnr"] if best else None,
        "best_test_mean_psnr": final_test_metrics.get("test_mean_psnr") if final_test_metrics else None,
        "best_test_mean_ssim": final_test_metrics.get("test_mean_ssim") if final_test_metrics else None,
        "best_test_mean_lpips": final_test_metrics.get("test_mean_lpips") if final_test_metrics else None,
        "best_params": best["params"] if best else None,
        "best_unit_x": best["unit_x"] if best else None,
        "best_trial_number": best["trial_number"] if best else None,
        "final_test_evaluation": final_test_metrics,
        "top_trials": top,
    }
    path.write_text(json.dumps(summary, indent=2))


# =============================================================================
# Run BO for one pair
# =============================================================================
def _pair_n_trials(args, pair_key):
    override = {
        "adam+exp_decay": args.n_trials_adam,
        "muon+exp_decay": args.n_trials_muon,
        "ours+rank_wsd": args.n_trials_ours,
    }[pair_key]
    return int(override) if override is not None else int(args.n_trials)


def run_pair(args, root, scene, pair_key, pair_preset, gpus):
    n_trials = _pair_n_trials(args, pair_key)
    d = len(pair_preset["active"])

    pair_dir = root / "logs" / "botorch_ranksched_20hp" / pair_key / scene
    pair_dir.mkdir(parents=True, exist_ok=True)
    summary_path = pair_dir / f"{scene}_{pair_key}_summary.json"

    print(
        f"\n[PAIR] === {pair_key} | scene={scene} | dim={d} | n_trials={n_trials} | "
        f"gpus={gpus} ===",
        flush=True,
    )
    print(f"[PAIR] active HPs: {pair_preset['active']}", flush=True)

    n_init = min(int(args.n_init), n_trials)
    if n_init < 2:
        n_init = 2

    results = []
    train_x_list, train_y_list = [], []

    sobol = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=args.seed)
    init_x = sobol.draw(n_init).double()
    init_pool = [init_x[i] for i in range(init_x.shape[0])]

    trial_id = 0
    while trial_id < n_trials:
        remaining = n_trials - trial_id
        q = min(int(args.batch_size), len(gpus), remaining)

        if len(train_x_list) < n_init:
            take = min(q, len(init_pool))
            batch = init_pool[:take]
            init_pool = init_pool[take:]
        else:
            train_x = torch.stack(train_x_list).double()
            train_y = torch.tensor(train_y_list, dtype=torch.double).unsqueeze(-1)
            cand = propose_batch(train_x, train_y, q=q, seed=args.seed + trial_id, d=d)
            batch = [cand[i] for i in range(cand.shape[0])]

        print(
            f"[BATCH] {pair_key} launching trials {trial_id} ~ {trial_id + len(batch) - 1}",
            flush=True,
        )

        batch_results = []
        with ThreadPoolExecutor(max_workers=len(batch)) as ex:
            futures = []
            for j, x_unit in enumerate(batch):
                gpu_id = gpus[j % len(gpus)]
                futures.append(
                    ex.submit(
                        run_one_trial,
                        args, root, pair_dir, scene, pair_key, pair_preset,
                        trial_id + j, x_unit, gpu_id,
                    )
                )

            for fut in as_completed(futures):
                r = fut.result()
                batch_results.append(r)
                pretty = ", ".join(
                    f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in r["params"].items()
                )
                print(
                    f"[DONE] {pair_key} trial={r['trial_number']:03d} "
                    f"val_mean_PSNR={r['value']:.4f} gpu={r['gpu']} | {pretty}",
                    flush=True,
                )

        batch_results = sorted(batch_results, key=lambda r: r["trial_number"])
        for r in batch_results:
            results.append(r)
            train_x_list.append(torch.tensor(r["unit_x"], dtype=torch.double))
            train_y_list.append(float(r["value"]))

        trial_id += len(batch_results)
        save_summary(summary_path, scene, pair_key, pair_preset, results, n_trials=n_trials)
        print(f"[SAVE] {summary_path}", flush=True)

    # Final test evaluation of the best validation trial.
    final_test_metrics = None
    if args.eval_test_after_hpo and results:
        best = sorted(results, key=lambda r: r["value"], reverse=True)[0]
        print(
            f"[TEST_EVAL] {pair_key} best trial={best['trial_number']} "
            f"expname={best['expname']}",
            flush=True,
        )
        final_test_metrics = run_final_test_eval(
            args, root, pair_dir, scene, pair_key, pair_preset, best
        )

        for r in results:
            if r["trial_number"] == best["trial_number"]:
                r["final_test_evaluation"] = final_test_metrics
                if final_test_metrics.get("status") == "ok":
                    r["test_mean_psnr"] = final_test_metrics.get("test_mean_psnr")
                    r["test_mean_ssim"] = final_test_metrics.get("test_mean_ssim")
                    r["test_mean_lpips"] = final_test_metrics.get("test_mean_lpips")
                break

        if final_test_metrics.get("status") == "ok":
            print(
                f"[TEST_DONE] {pair_key} trial={best['trial_number']} "
                f"PSNR={final_test_metrics.get('test_mean_psnr')} "
                f"SSIM={final_test_metrics.get('test_mean_ssim')}",
                flush=True,
            )
        else:
            print(
                f"[TEST_FAIL] {pair_key} trial={best['trial_number']} "
                f"error={final_test_metrics.get('error')}",
                flush=True,
            )

    save_summary(summary_path, scene, pair_key, pair_preset, results,
                 final_test_metrics=final_test_metrics, n_trials=n_trials)
    print(f"[PAIR_DONE] {pair_key} summary saved to {summary_path}", flush=True)
    return summary_path


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    root = Path(args.root).resolve()
    scene = Path(args.config).stem

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id, e.g. --gpus 0,1")

    torch.manual_seed(args.seed)

    if args.pair == "all":
        pair_keys = list(PAIR_PRESETS.keys())
    else:
        pair_keys = [args.pair]

    overall_summary = {
        "scene": scene,
        "pairs": pair_keys,
        "hp_registry": HP_REGISTRY,
        "results": {},
    }
    overall_summary_path = root / "logs" / "botorch_ranksched_20hp" / f"{scene}_overall_summary.json"
    overall_summary_path.parent.mkdir(parents=True, exist_ok=True)

    for pair_key in pair_keys:
        preset = PAIR_PRESETS[pair_key]
        summary_path = run_pair(args, root, scene, pair_key, preset, gpus)
        overall_summary["results"][pair_key] = str(summary_path)
        overall_summary_path.write_text(json.dumps(overall_summary, indent=2))
        print(f"[OVERALL] {overall_summary_path}", flush=True)

    print("[ALL_DONE]", flush=True)


if __name__ == "__main__":
    main()


"""
Example usage:

  cd /home/greenx9/cnfr/ProPolar_NeurIPS26/nerf-pytorch/LRP_algo
  conda activate nerf

  mkdir -p logs/GP_test_log/chair

  # Run all three (opt, scheduler) pairs sequentially, each using all GPUs.
  python -u 00_GPT_gp_search.py \
    --root /home/greenx9/cnfr/ProPolar_NeurIPS26/nerf-pytorch/LRP_algo \
    --run_file 00_run_nerf_ranksched_final.py \
    --config configs/chair.txt \
    --basedir logs/GP_20hp \
    --exp_prefix chair_20hp \
    --pair all \
    --n_trials 20 \
    --batch_size 4 \
    --n_init 4 \
    --n_iters 100000 \
    --gpus 0,1,2,3 \
    --deterministic \
    > logs/GP_test_log/chair/chair_botorch_20hp_main.log 2>&1 &

  # Single pair only:
  python -u 00_GPT_gp_search.py \
    --root /home/greenx9/cnfr/ProPolar_NeurIPS26/nerf-pytorch/LRP_algo \
    --config configs/chair.txt \
    --pair ours+rank_wsd \
    --n_trials 30 \
    --gpus 0,1,2,3

  # Per-pair budgets (give 'ours' more trials due to higher search dim):
  python -u 00_GPT_gp_search.py \
    --root /home/greenx9/cnfr/ProPolar_NeurIPS26/nerf-pytorch/LRP_algo \
    --config configs/chair.txt \
    --pair all \
    --n_trials 20 \
    --n_trials_ours 32 \
    --gpus 0,1,2,3
"""
