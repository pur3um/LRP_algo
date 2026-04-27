# LRP_algo
1. Conda 환경 설치
```
conda create -n lrp python=3.8.20
pip install -r requirements.txt

pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```

2. Dataset
```
cd data
bash download.sh
```

3. 파일 설명
    - `run_nerf.py`, `run_nerf_helper.py` : original NeRF run files.
    - `stage1_optims_lr_decay_mlr_search.py`, `run_nerf_helpers_optuna.py`, `run_inc_optims_optuna_ready.py` : Optuna run files.
    - `run_nerf_ranksched.py`, `optims/*.py` : LRP algorithm run files.

4. (지금까지) 최종 실행 코드
```
python run_nerf_ranksched.py \
  --basedir logs \
  --config configs/lego.txt \
  --expname lego_200k \
  --optimizer aux-sign-auto-cos-inc \
  --train_scheduler rank_wsd \
  --muon_lrate 3e-3 \
  --lowrank_rank_start 150 \
  --lowrank_rank_end 250 \
  --lowrank_auto_init_rank_start \
  --N_iters 200000
```