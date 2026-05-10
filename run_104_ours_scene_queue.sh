#!/usr/bin/env bash
set -euo pipefail

cd /home/greenx9/data2/LRP_algo

# conda activate가 non-interactive shell에서도 되도록 처리
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    source "$HOME/.bashrc"
fi

conda activate lrp39

ROOT=/home/greenx9/data2/LRP_algo
RUN_FILE=00_run_nerf_ranksched_final.py

# 104번 서버 Ours 결과 저장 위치
BASEDIR=logs/104_ours_rankwsd
LOGDIR=logs/104_ours_rankwsd_main_logs

mkdir -p "$BASEDIR"
mkdir -p "$LOGDIR"

# HPO 설정
N_TRIALS=20
BATCH_SIZE=4
N_INIT=4
N_ITERS=100000

# Ours 설정
OPTIMIZER=aux-sign-auto-cos-inc
SCHEDULER=rank_wsd

run_scene () {
    local SCENE=$1
    local GPUS=$2
    local TEST_GPU=$3

    mkdir -p "${LOGDIR}/${SCENE}"

    echo "=================================================="
    echo "[START] scene=${SCENE}, gpus=${GPUS}, test_gpu=${TEST_GPU}"
    echo "=================================================="

    python -u 00_GPT_gp_search.py \
        --root "$ROOT" \
        --run_file "$RUN_FILE" \
        --config "configs/${SCENE}.txt" \
        --basedir "$BASEDIR" \
        --exp_prefix "104_ours_${SCENE}" \
        --n_trials "$N_TRIALS" \
        --batch_size "$BATCH_SIZE" \
        --n_init "$N_INIT" \
        --n_iters "$N_ITERS" \
        --gpus "$GPUS" \
        --optimizer "$OPTIMIZER" \
        --train-scheduler "$SCHEDULER" \
        --test_gpu "$TEST_GPU" \
        --deterministic \
        > "${LOGDIR}/${SCENE}/${SCENE}_main.log" 2>&1

    echo "=================================================="
    echo "[DONE] scene=${SCENE}"
    echo "=================================================="
}

# Round 1
run_scene chair "0,1,2,3" 0 &
PID1=$!

run_scene drums "4,5,6,7" 4 &
PID2=$!

wait $PID1
wait $PID2

echo "[ROUND 1 DONE] chair, drums"

# Round 2
run_scene ficus "0,1,2,3" 0 &
PID1=$!

run_scene hotdog "4,5,6,7" 4 &
PID2=$!

wait $PID1
wait $PID2

echo "[ROUND 2 DONE] ficus, hotdog"

# Round 3
run_scene lego "0,1,2,3" 0 &
PID1=$!

run_scene materials "4,5,6,7" 4 &
PID2=$!

wait $PID1
wait $PID2

echo "[ROUND 3 DONE] lego, materials"

# Round 4
run_scene mic "0,1,2,3" 0 &
PID1=$!

run_scene ship "4,5,6,7" 4 &
PID2=$!

wait $PID1
wait $PID2

echo "[ALL DONE] 104 Ours all scenes finished"
