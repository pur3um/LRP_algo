# stage1_optims_lr_decay_mlr_search.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Version (1): Optuna searches ONLY --muon_lrate.
- --lrate is fixed.
- --lrate_decay is fixed.
- --muon_decay is fixed weight decay, not searched.

Use this with optimizers that actually consume args.muon_lrate, e.g. aux-muon,
aux-sign, aux-sign10-rsclF, aux-sign-auto-cos-inc. If optimizer is ori-adam,
--muon_lrate is parsed by the training script but does not affect Adam.
"""

import os
import re
import sys
import json
import signal
import argparse
import subprocess
from pathlib import Path

import optuna


ACTIVE_PROC = None


def lr_to_tag(lr: float) -> str:
    s = f"{lr:.10f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def budget_to_tag(n_iters: int) -> str:
    if n_iters % 1000 == 0:
        return f"{n_iters // 1000}k"
    return str(n_iters)


def safe_tag(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")
    return text.replace(".", "p") or "default"


def terminate_active_child():
    global ACTIVE_PROC
    if ACTIVE_PROC is None:
        return
    try:
        if ACTIVE_PROC.poll() is None:
            os.killpg(os.getpgid(ACTIVE_PROC.pid), signal.SIGTERM)
    except Exception:
        pass


def signal_handler(signum, frame):
    terminate_active_child()
    raise KeyboardInterrupt


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--root", type=str, required=True)
    p.add_argument("--run_file", type=str, default="run_inc_optims_optuna_ready.py")
    p.add_argument("--config", type=str, required=True)

    p.add_argument("--n_trials", type=int, default=20)
    p.add_argument("--n_iters", type=int, default=100000)
    p.add_argument("--eval_every", type=int, default=5000)
    p.add_argument("--max_eval_views", type=int, default=2)

    p.add_argument("--gpu", type=int, default=0)

    # Fixed Adam/scheduler settings. These are NOT searched in this script.
    p.add_argument("--min_lr", type=float, default=3e-4)
    p.add_argument("--max_lr", type=float, default=3e-3)
    p.add_argument("--min_decay", type=int, default=100)
    p.add_argument("--max_decay", type=int, default=500)

    # Searched Muon learning-rate range.
    p.add_argument("--min_muon_lr", type=float, default=3e-4)
    p.add_argument("--max_muon_lr", type=float, default=3e-3)

    # Fixed Muon optimizer settings. These are NOT searched here.
    p.add_argument("--muon_decay", type=float, default=0.0, help="Fixed Muon weight decay, not Optuna-searched.")
    p.add_argument("--muon_momentum", type=float, default=0.90, help="Fixed Muon momentum, not Optuna-searched.")

    p.add_argument("--optimizer", type=str, default="aux-muon")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--n_startup_trials", type=int, default=5)
    p.add_argument("--n_ei_candidates", type=int, default=24)
    p.add_argument("--deterministic", action="store_true")

    return p.parse_args()


def build_paths(args):
    root = Path(args.root).resolve()
    scene = Path(args.config).stem
    budget_tag = budget_to_tag(args.n_iters)
    optimizer_tag = safe_tag(args.optimizer)
    fixed_tag = f"lr{lr_to_tag(args.min_lr)}_decay{args.min_decay}_mlr{args.min_muon_lr}_args"

    logs_root = root / "logs" / f"{budget_tag}_muon_lr"
    scene_dir = logs_root / scene / optimizer_tag / fixed_tag
    scene_dir.mkdir(parents=True, exist_ok=True)

    study_name = f"{scene}_{optimizer_tag}_stage1only_{budget_tag}_{fixed_tag}_muon_lr_only"
    storage = f"sqlite:///{(scene_dir / f'{scene}_{optimizer_tag}_{budget_tag}_{fixed_tag}_muon_lr_only_study.db').as_posix()}"

    summary_path = scene_dir / f"{scene}_{optimizer_tag}_{budget_tag}_{fixed_tag}_muon_lr_only_summary.json"
    main_info_path = scene_dir / f"{scene}_{optimizer_tag}_{budget_tag}_{fixed_tag}_muon_lr_only_study_info.json"

    return root, scene, budget_tag, optimizer_tag, fixed_tag, scene_dir, study_name, storage, summary_path, main_info_path


def save_summary(study, scene, budget_tag, optimizer_tag, fixed_tag, scene_dir, summary_path, args):
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    trials_sorted = sorted(trials, key=lambda t: t.value, reverse=True)

    summary = {
        "scene": scene,
        "budget": budget_tag,
        "optimizer": args.optimizer,
        "search_space": "muon_lrate_only",
        "fixed_muon_decay": float(args.muon_decay),
        "fixed_muon_momentum": float(args.muon_momentum),
        "min_lr": float(args.min_lr),
        "max_lr": float(args.max_lr),
        "min_decay": float(args.min_decay),
        "max_decay": float(args.max_decay),
        "min_muon_lr": float(args.min_muon_lr),
        "max_muon_lr": float(args.max_muon_lr),
        "study_name": study.study_name,
        "best_value": study.best_value if trials else None,
        "best_params": study.best_params if trials else None,
        "best_trial_number": study.best_trial.number if trials else None,
        "n_complete_trials": len(trials),
        "top_trials": [],
    }

    for t in trials_sorted:
        summary["top_trials"].append({
            "trial_number": t.number,
            "value": float(t.value),
            "muon_lrate": float(t.params["muon_lrate"]),
            "min_lr": float(args.min_lr),
            "max_lr": float(args.max_lr),
            "min_decay": float(args.min_decay),
            "max_decay": float(args.max_decay),
            "fixed_muon_decay": float(args.muon_decay),
            "fixed_muon_momentum": float(args.muon_momentum),
            "expname": t.user_attrs.get("expname"),
            "trial_dir": t.user_attrs.get("trial_dir"),
            "best_iter": t.user_attrs.get("best_iter"),
            "metric_out": t.user_attrs.get("metric_out"),
            "stdout_log": t.user_attrs.get("stdout_log"),
            "stderr_log": t.user_attrs.get("stderr_log"),
            "cmd": t.user_attrs.get("cmd"),
        })

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


def make_callback(scene, budget_tag, optimizer_tag, fixed_tag, scene_dir, summary_path, args):
    def _callback(study, trial):
        save_summary(study, scene, budget_tag, optimizer_tag, fixed_tag, scene_dir, summary_path, args)
    return _callback


def warn_if_muon_lr_unused(args):
    adam_only_names = {"ori-adam", "adam", "original", "ori"}
    if args.optimizer in adam_only_names:
        print(
            "[WARN] optimizer is Adam-only. --muon_lrate will be searched by Optuna "
            "but will not affect training unless the run_file uses it for this optimizer.",
            flush=True,
        )


def make_objective(args, root, scene, budget_tag, optimizer_tag, fixed_tag, scene_dir):
    def objective(trial: optuna.trial.Trial):
        global ACTIVE_PROC

        lr = trial.suggest_float("lrate", args.min_lr, args.max_lr) #! log=True
        lrate_decay = trial.suggest_int("lrate_decay", args.min_decay, args.max_decay)
        muon_lr = trial.suggest_float("muon_lrate", args.min_muon_lr, args.max_muon_lr, log=True)

        muon_lr_tag = lr_to_tag(muon_lr)
        adam_lr_tag = lr_to_tag(lr)
        expname = (
            f"{scene}_{optimizer_tag}_{budget_tag}"
            f"_muonlr{muon_lr_tag}_adam{adam_lr_tag}_decay{lrate_decay}"
            f"_trial{trial.number:03d}"
        )

        trial_dir = scene_dir / expname
        trial_dir.mkdir(parents=True, exist_ok=True)

        metric_out = trial_dir / "metric.json"
        stdout_path = trial_dir / "stdout.txt"
        stderr_path = trial_dir / "stderr.txt"

        cmd = [
            sys.executable,
            str((root / args.run_file).resolve()),
            "--config", str((root / args.config).resolve()),
            "--basedir", str(scene_dir.resolve()),
            "--expname", expname,
            "--optimizer", args.optimizer,
            "--lrate", str(lr),
            "--lrate_decay", str(lrate_decay),
            "--muon_lrate", str(muon_lr),
            "--muon_decay", str(args.muon_decay),
            "--muon_momentum", str(args.muon_momentum),
            "--N_iters", str(args.n_iters),
            "--eval_every", str(args.eval_every),
            "--max_eval_views", str(args.max_eval_views),
            "--metric_out", str(metric_out.resolve()),
            "--no_reload",
            "--seed", str(args.seed),
        ]
        if args.deterministic:
            cmd.append("--deterministic")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        env["PYTHONUNBUFFERED"] = "1"

        trial.set_user_attr("expname", expname)
        trial.set_user_attr("trial_dir", str(trial_dir))
        trial.set_user_attr("metric_out", str(metric_out))
        trial.set_user_attr("stdout_log", str(stdout_path))
        trial.set_user_attr("stderr_log", str(stderr_path))
        trial.set_user_attr("cmd", " ".join(cmd))

        with open(stdout_path, "w") as fout, open(stderr_path, "w") as ferr:
            ACTIVE_PROC = subprocess.Popen(
                cmd,
                cwd=str(root),
                env=env,
                stdout=fout,
                stderr=ferr,
                preexec_fn=os.setsid,
            )
            ret = ACTIVE_PROC.wait()
            ACTIVE_PROC = None

        if ret != 0:
            raise RuntimeError(f"Child process failed with exit code {ret}: {cmd}")

        if not metric_out.exists():
            raise RuntimeError(f"Metric file not found: {metric_out}")

        with open(metric_out, "r") as f:
            metric = json.load(f)

        best_val_psnr = float(metric["best_val_psnr"])
        trial.set_user_attr("best_iter", int(metric["best_iter"]))

        return best_val_psnr

    return objective


def main():
    args = parse_args()
    root, scene, budget_tag, optimizer_tag, fixed_tag, scene_dir, study_name, storage, summary_path, main_info_path = build_paths(args)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    warn_if_muon_lr_unused(args)

    sampler = optuna.samplers.TPESampler(
        n_startup_trials=args.n_startup_trials,
        n_ei_candidates=args.n_ei_candidates,
        seed=args.seed,
    )
    pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    remaining = max(0, args.n_trials - len(complete_trials))

    with open(main_info_path, "w") as f:
        json.dump({
            "scene": scene,
            "budget": budget_tag,
            "scene_dir": str(scene_dir),
            "study_name": study_name,
            "storage": storage,
            "target_total_trials": args.n_trials,
            "completed_trials_before_run": len(complete_trials),
            "remaining_trials_to_run": remaining,
            "n_startup_trials": args.n_startup_trials,
            "n_ei_candidates": args.n_ei_candidates,
            "pruning": "OFF (NopPruner)",
            "optimizer": args.optimizer,
            "search_space": "muon_lrate_only",
            "min_lr": args.min_lr,
            "max_lr": args.max_lr,
            "min_decay": args.min_decay,
            "max_decay": args.max_decay,
            "min_muon_lr": args.min_muon_lr,
            "max_muon_lr": args.max_muon_lr,
            "fixed_muon_decay": args.muon_decay,
            "fixed_muon_momentum": args.muon_momentum,
            "n_iters": args.n_iters,
            "eval_every": args.eval_every,
            "max_eval_views": args.max_eval_views,
            "run_file": args.run_file,
            "config": args.config,
        }, f, indent=2)

    if remaining == 0:
        save_summary(study, scene, budget_tag, optimizer_tag, fixed_tag, scene_dir, summary_path, args)
        print(f"[SKIP] Already reached target total trials: {args.n_trials}")
        print(f"[DONE] Summary saved to {summary_path}")
        return

    objective = make_objective(args, root, scene, budget_tag, optimizer_tag, fixed_tag, scene_dir)
    callback = make_callback(scene, budget_tag, optimizer_tag, fixed_tag, scene_dir, summary_path, args)

    try:
        study.optimize(objective, n_trials=remaining, callbacks=[callback])
    except KeyboardInterrupt:
        print("\n[STOPPED] Interrupted by user. Saving partial summary...")
    finally:
        terminate_active_child()
        study = optuna.load_study(study_name=study_name, storage=storage)
        save_summary(study, scene, budget_tag, optimizer_tag, fixed_tag, scene_dir, summary_path, args)
        print(f"[DONE] Summary saved to {summary_path}")


if __name__ == "__main__":
    main()


"""
mkdir -p logs/sign_100k_lr_decay_mlr
Example: chair, drums, ficus, hotdog, lego, materials, mic, ship
python stage1_optims_lr_decay_mlr_search.py \
  --root /workspace/nerf-pytorch \
  --run_file run_inc_optims_optuna_ready.py \
  --config configs/ship.txt \
  --n_trials 20 \
  --n_iters 100000 \
  --eval_every 5000 \
  --max_eval_views 2 \
  --gpu 0 \
  --min_lr 3e-4 \
  --max_lr 3e-3 \
  --min_decay 100 \
  --max_decay 500 \
  --min_muon_lr 3e-4 \
  --max_muon_lr 3e-3 \
  --optimizer aux-sign \
  --seed 0 \
  --n_startup_trials 5 \
  > logs/sign_100k_lr_decay_mlr/ship_100k.log 2>&1
  
python stage1_optims_lr_decay_mlr_search.py \
  --root /workspace/nerf-pytorch \
  --run_file run_inc_optims_optuna_ready.py \
  --config configs/materials.txt \
  --n_trials 20 \
  --n_iters 100000 \
  --eval_every 5000 \
  --max_eval_views 2 \
  --gpu 1 \
  --min_lr 3e-4 \
  --max_lr 3e-3 \
  --min_decay 100 \
  --max_decay 500 \
  --min_muon_lr 3e-4 \
  --max_muon_lr 3e-3 \
  --optimizer aux-sign \
  --seed 0 \
  --n_startup_trials 5 \
  > logs/sign_100k_lr_decay_mlr/materials_100k.log 2>&1
"""
