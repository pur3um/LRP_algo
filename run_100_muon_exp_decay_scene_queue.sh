#!/usr/bin/env bash
set -euo pipefail

cd /home/greenx9/data/LRP_algo

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    source "$HOME/.bashrc"
fi

conda activate lrp39

ROOT=/home/greenx9/data/LRP_algo
RUN_FILE=00_run_nerf_ranksched_final.py

BASEDIR=logs/100_muon_exp_decay_lrp39
LOGDIR=logs/100_muon_exp_decay_lrp39_main_logs

mkdir -p "$BASEDIR"
mkdir -p "$LOGDIR"

N_TRIALS=20
BATCH_SIZE=4
N_INIT=4
N_ITERS=100000

OPTIMIZER=aux-muon
SCHEDULER=exp_decay

run_scene () {
    local SCENE=$1
    local GPUS=$2
    local TEST_GPU=$3

    mkdir -p "${LOGDIR}/${SCENE}"

    echo "=================================================="
    echo "[START] scene=${SCENE}, gpus=${GPUS}, test_gpu=${TEST_GPU}"
    echo "optimizer=${OPTIMIZER}, scheduler=${SCHEDULER}, env=lrp39"
    echo "=================================================="

    python -u 00_GPT_gp_search.py \
        --root "$ROOT" \
        --run_file "$RUN_FILE" \
        --config "configs/${SCENE}.txt" \
        --basedir "$BASEDIR" \
        --exp_prefix "100_muon_${SCENE}" \
        --n_trials "$N_TRIALS" \
        --batch_size "$BATCH_SIZE" \
        --n_init "$N_INIT" \
        --n_iters "$N_ITERS" \
        --gpus "$GPUS" \
        --optimizer "$OPTIMIZER" \
        --train_scheduler "$SCHEDULER" \
        --test_gpu "$TEST_GPU" \
        --deterministic \
        > "${LOGDIR}/${SCENE}/${SCENE}_main.log" 2>&1

    echo "=================================================="
    echo "[DONE] scene=${SCENE}"
    echo "=================================================="
}

# Round 1: 3 scenes simultaneously, GPU 0/1 unused
run_scene chair "6,6,7,7" 6 &
PID1=$!

run_scene drums "2,2,3,3" 2 &
PID2=$!

run_scene ficus "4,4,5,5" 4 &
PID3=$!

wait $PID1
wait $PID2
wait $PID3

echo "[ROUND 1 DONE] chair, drums, ficus"

# Round 2
run_scene hotdog "6,6,7,7" 6 &
PID1=$!

run_scene lego "2,2,3,3" 2 &
PID2=$!

run_scene materials "4,4,5,5" 4 &
PID3=$!

wait $PID1
wait $PID2
wait $PID3

echo "[ROUND 2 DONE] hotdog, lego, materials"

# Round 3
run_scene mic "6,6,7,7" 6 &
PID1=$!

run_scene ship "2,2,3,3" 2 &
PID2=$!

wait $PID1
wait $PID2

echo "[ALL DONE] 100 Muon exp_decay all scenes finished"
