#? adam + exp_decay (2-D BO, n=20)
python -u 00_GPT_gp_search.py \
  --root /home/greenx9/cnfr/ProPolar_NeurIPS26/nerf-pytorch/LRP_algo \
  --run_file 00_run_nerf_ranksched_final.py \
  --config configs/chair.txt \
  --basedir logs/GP_20hp \
  --exp_prefix chair_20hp \
  --pair adam+exp_decay \
  --n_trials 20 --batch_size 2 --n_init 4 \
  --n_iters 100000 \
  --gpus 0,1 \
  --test_gpu 0 \
  --deterministic \
  > logs/GP_hyp20_log/chair/chair_adam_exp_2gpu.log 2>&1 &

python -u 00_GPT_gp_search.py \
  --root /home/greenx9/cnfr/ProPolar_NeurIPS26/nerf-pytorch/LRP_algo \
  --run_file 00_run_nerf_ranksched_final.py \
  --config configs/chair.txt \
  --basedir logs/GP_20hp \
  --exp_prefix chair_20hp \
  --pair adam+exp_decay \
  --n_trials 20 --batch_size 4 --n_init 4 \
  --n_iters 100000 \
  --gpus 0,1,2,3 \
  --test_gpu 0 \
  --deterministic \
  > logs/GP_hyp20_log/chair/chair_adam_exp_4gpu.log 2>&1 &
#================================================
#? aux-muon + exp_decay (10-D BO, n=20)
python -u 00_GPT_gp_search.py \
  --root /home/greenx9/cnfr/ProPolar_NeurIPS26/nerf-pytorch/LRP_algo \
  --run_file 00_run_nerf_ranksched_final.py \
  --config configs/chair.txt \
  --basedir logs/GP_20hp \
  --exp_prefix chair_20hp \
  --pair muon+exp_decay \
  --n_trials 20 --batch_size 2 --n_init 4 \
  --n_iters 100000 \
  --gpus 0,1 \
  --test_gpu 0 \
  --deterministic \
  > logs/GP_hyp20_log/chair/chair_muon_exp_2gpu.log 2>&1 &

python -u 00_GPT_gp_search.py \
  --root /home/greenx9/cnfr/ProPolar_NeurIPS26/nerf-pytorch/LRP_algo \
  --run_file 00_run_nerf_ranksched_final.py \
  --config configs/chair.txt \
  --basedir logs/GP_20hp \
  --exp_prefix chair_20hp \
  --pair muon+exp_decay \
  --n_trials 20 --batch_size 4 --n_init 4 \
  --n_iters 100000 \
  --gpus 0,1,2,3 \
  --test_gpu 0 \
  --deterministic \
  > logs/GP_hyp20_log/chair/chair_muon_exp_4gpu.log 2>&1 &
#================================================
#? aux-sign-auto-cos-inc + rank_wsd (19-D BO, n=20 기본)
# 차원이 19로 매우 큼 → 가능하면 --n_trials_ours 30 ~ 40 권장. 아래는 두 버전을 모두 보임.
python -u 00_GPT_gp_search.py \
  --root /home/greenx9/cnfr/ProPolar_NeurIPS26/nerf-pytorch/LRP_algo \
  --run_file 00_run_nerf_ranksched_final.py \
  --config configs/chair.txt \
  --basedir logs/GP_20hp \
  --exp_prefix chair_20hp \
  --pair ours+rank_wsd \
  --n_trials 20 --batch_size 2 --n_init 4 \
  --n_iters 100000 \
  --gpus 6,7 \
  --test_gpu 0 \
  --deterministic \
  > logs/GP_hyp20_log/chair/chair_ours_wsd_2gpu.log 2>&1 &

python -u 00_GPT_gp_search.py \
  --root /home/greenx9/cnfr/ProPolar_NeurIPS26/nerf-pytorch/LRP_algo \
  --run_file 00_run_nerf_ranksched_final.py \
  --config configs/chair.txt \
  --basedir logs/GP_20hp \
  --exp_prefix chair_20hp \
  --pair ours+rank_wsd \
  --n_trials 20 --batch_size 4 --n_init 4 \
  --n_iters 100000 \
  --gpus 0,1,2,3 \
  --test_gpu 0 \
  --deterministic \
  > logs/GP_hyp20_log/chair/chair_ours_wsd_4gpu.log 2>&1 &

# 차원에 맞춰 예산 증가 (추천)
python -u 00_GPT_gp_search.py \
  --root /home/greenx9/cnfr/ProPolar_NeurIPS26/nerf-pytorch/LRP_algo \
  --run_file 00_run_nerf_ranksched_final.py \
  --config configs/chair.txt \
  --basedir logs/GP_20hp \
  --exp_prefix chair_20hp \
  --pair ours+rank_wsd \
  --n_trials_ours 32 --batch_size 2 --n_init 8 \
  --n_iters 100000 \
  --gpus 6,7 \
  --test_gpu 0 \
  --deterministic \
  > logs/GP_hyp20_log/chair/chair_ours_wsd_2gpu_big.log 2>&1 &

python -u 00_GPT_gp_search.py \
  --root /home/greenx9/cnfr/ProPolar_NeurIPS26/nerf-pytorch/LRP_algo \
  --run_file 00_run_nerf_ranksched_final.py \
  --config configs/chair.txt \
  --basedir logs/GP_20hp \
  --exp_prefix chair_20hp \
  --pair ours+rank_wsd \
  --n_trials_ours 32 --batch_size 4 --n_init 8 \
  --n_iters 100000 \
  --gpus 0,1,2,3 \
  --test_gpu 0 \
  --deterministic \
  > logs/GP_hyp20_log/chair/chair_ours_wsd_4gpu_big.log 2>&1 &
#================================================
#? 세 페어를 순차로 한 번에 (--pair all)
python -u 00_GPT_gp_search.py \
  --root /home/greenx9/cnfr/ProPolar_NeurIPS26/nerf-pytorch/LRP_algo \
  --run_file 00_run_nerf_ranksched_final.py \
  --config configs/chair.txt \
  --basedir logs/GP_20hp \
  --exp_prefix chair_20hp \
  --pair all \
  --n_trials 20 --n_trials_ours 32 \
  --batch_size 2 --n_init 4 \
  --n_iters 100000 \
  --gpus 0,1 \
  --test_gpu 0 \
  --deterministic \
  > logs/GP_hyp20_log/chair/chair_all_2gpu.log 2>&1 &

python -u 00_GPT_gp_search.py \
  --root /home/greenx9/cnfr/ProPolar_NeurIPS26/nerf-pytorch/LRP_algo \
  --run_file 00_run_nerf_ranksched_final.py \
  --config configs/chair.txt \
  --basedir logs/GP_20hp \
  --exp_prefix chair_20hp \
  --pair all \
  --n_trials 20 --n_trials_ours 32 \
  --batch_size 4 --n_init 4 \
  --n_iters 100000 \
  --gpus 0,1,2,3 \
  --test_gpu 0 \
  --deterministic \
  > logs/GP_hyp20_log/chair/chair_all_4gpu.log 2>&1 &
#================================================
#// 결과 위치
# 페어별 요약 JSON: logs/botorch_ranksched_20hp/<pair_key>/<scene>/<scene>_<pair_key>_summary.json
# 전체 요약 JSON: logs/botorch_ranksched_20hp/<scene>_overall_summary.json
# 학습 산출물: logs/GP_20hp/<expname>/
#// 보조 팁
# tmux new -s gp; ... ; Ctrl-b d로 detach.
# 진행 확인: tail -f logs/GP_hyp20_log/chair/chair_all_4gpu.log.
# GPU 사용량: watch -n 1 nvidia-smi.
# --no_eval_test_after_hpo를 붙이면 HPO 끝나도 best trial의 test 평가를 건너뛸 수 있음(빠른 디버깅용).
# --n_iters를 작게(예: 20000) 두면 짧은 sanity-check 실행 가능. 본 실험은 100000 유지 권장.
#================================================