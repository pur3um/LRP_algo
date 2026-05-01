# LRP_algo
1. Conda 환경 설치
```
conda create -n lrp3.10 python=3.10
pip install -r requirements_murf2.txt

# X
# pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
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
