====================================================================================================
saved_time: 2026-05-03 01:39:15
script_path: /data1/greenx9/LRP_algo/00_run_nerf_ranksched_final.py
executed_command: /data1/greenx9/LRP_algo/00_run_nerf_ranksched_final.py --basedir /data1/greenx9/LRP_algo/logs/100_muon_exp_decay_lrp39 --config /data1/greenx9/LRP_algo/configs/chair.txt --expname 100_muon_chair_chair_trial002_mlr3p70e-03_alr5p70e-04_decay146_mom0p950 --optimizer aux-muon --train-scheduler exp_decay --muon_lrate 0.003704381968500205 --lrate 0.0005697747652055574 --lrate_decay 146 --muon_momentum 0.95 --lowrank_rank_start 150 --lowrank_rank_end 250 --lowrank_auto_init_rank_start --N_iters 100000 --seed 2 --eval_testset_only --test_out_json /data1/greenx9/LRP_algo/logs/100_muon_exp_decay_lrp39/100_muon_chair_chair_trial002_mlr3p70e-03_alr5p70e-04_decay146_mom0p950/test_metrics_eval.json --test_out_dir /data1/greenx9/LRP_algo/logs/100_muon_exp_decay_lrp39/100_muon_chair_chair_trial002_mlr3p70e-03_alr5p70e-04_decay146_mom0p950/testset_eval --lpips_net alex --i_print 5000 --i_weights 100000 --i_valset 100001 --i_testset 100001 --i_video 100001 --deterministic
expname: 100_muon_chair_chair_trial002_mlr3p70e-03_alr5p70e-04_decay146_mom0p950
iter: 99999
global_step: 99999
elapsed_time_from_train_start: 0 hour 2 min
current_train_loss: 0.0000000000
current_train_psnr: 0.000000
testset_mean_loss: 0.0004000021
testset_mean_psnr: 34.195485
testset_mean_ssim: 0.977709
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0001395304, psnr=38.553309, ssim=0.993305, lpips=unavailable
  image_001: loss=0.0002000315, psnr=36.989014, ssim=0.988510, lpips=unavailable
  image_002: loss=0.0003991862, psnr=33.988244, ssim=0.979118, lpips=unavailable
  image_003: loss=0.0004020131, psnr=33.957597, ssim=0.977871, lpips=unavailable
  image_004: loss=0.0004585212, psnr=33.386405, ssim=0.975200, lpips=unavailable
  image_005: loss=0.0004993966, psnr=33.015543, ssim=0.973604, lpips=unavailable
  image_006: loss=0.0005054563, psnr=32.963163, ssim=0.971537, lpips=unavailable
  image_007: loss=0.0005498551, psnr=32.597517, ssim=0.968524, lpips=unavailable
  image_008: loss=0.0004181818, psnr=33.786347, ssim=0.973889, lpips=unavailable
  image_009: loss=0.0004566533, psnr=33.404133, ssim=0.974105, lpips=unavailable
  image_010: loss=0.0005823566, psnr=32.348109, ssim=0.972484, lpips=unavailable
  image_011: loss=0.0003935266, psnr=34.050258, ssim=0.979214, lpips=unavailable
  image_012: loss=0.0002975811, psnr=35.263945, ssim=0.983198, lpips=unavailable
  image_013: loss=0.0003431072, psnr=34.645700, ssim=0.980713, lpips=unavailable
  image_014: loss=0.0002770527, psnr=35.574374, ssim=0.983288, lpips=unavailable
  image_015: loss=0.0004325896, psnr=33.639238, ssim=0.977197, lpips=unavailable
  image_016: loss=0.0003329730, psnr=34.775909, ssim=0.981234, lpips=unavailable
  image_017: loss=0.0004756623, psnr=33.227012, ssim=0.975426, lpips=unavailable
  image_018: loss=0.0005233345, psnr=32.812206, ssim=0.972879, lpips=unavailable
  image_019: loss=0.0005243497, psnr=32.803789, ssim=0.969821, lpips=unavailable
  image_020: loss=0.0004553871, psnr=33.416192, ssim=0.971434, lpips=unavailable
  image_021: loss=0.0003839878, psnr=34.156824, ssim=0.975006, lpips=unavailable
  image_022: loss=0.0004034812, psnr=33.941766, ssim=0.975420, lpips=unavailable
  image_023: loss=0.0003446878, psnr=34.625739, ssim=0.980774, lpips=unavailable
  image_024: loss=0.0002011505, psnr=36.964787, ssim=0.988967, lpips=unavailable
