====================================================================================================
saved_time: 2026-05-03 22:38:06
script_path: /data2/greenx9/LRP_algo/00_run_nerf_ranksched_final.py
executed_command: /data2/greenx9/LRP_algo/00_run_nerf_ranksched_final.py --basedir /data2/greenx9/LRP_algo/logs/104_ours_rankwsd --config /data2/greenx9/LRP_algo/configs/materials.txt --expname 104_ours_materials_materials_trial009_mlr1p05e-03_alr6p51e-04_decay100_mom0p950 --optimizer aux-sign-auto-cos-inc --train-scheduler rank_wsd --muon_lrate 0.0010463254855772274 --lrate 0.0006514806229083146 --lrate_decay 100 --muon_momentum 0.95 --lowrank_rank_start 150 --lowrank_rank_end 250 --lowrank_auto_init_rank_start --N_iters 100000 --seed 9 --eval_testset_only --test_out_json /data2/greenx9/LRP_algo/logs/104_ours_rankwsd/104_ours_materials_materials_trial009_mlr1p05e-03_alr6p51e-04_decay100_mom0p950/test_metrics_eval.json --test_out_dir /data2/greenx9/LRP_algo/logs/104_ours_rankwsd/104_ours_materials_materials_trial009_mlr1p05e-03_alr6p51e-04_decay100_mom0p950/testset_eval --lpips_net alex --i_print 5000 --i_weights 100000 --i_valset 100001 --i_testset 100001 --i_video 100001 --deterministic
expname: 104_ours_materials_materials_trial009_mlr1p05e-03_alr6p51e-04_decay100_mom0p950
iter: 99999
global_step: 99999
elapsed_time_from_train_start: 0 hour 2 min
current_train_loss: 0.0000000000
current_train_psnr: 0.000000
testset_mean_loss: 0.0010768343
testset_mean_psnr: 29.868447
testset_mean_ssim: 0.960707
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0017073763, psnr=27.676707, ssim=0.949584, lpips=unavailable
  image_001: loss=0.0021308893, psnr=26.714391, ssim=0.942287, lpips=unavailable
  image_002: loss=0.0016084394, psnr=27.935953, ssim=0.951318, lpips=unavailable
  image_003: loss=0.0011576110, psnr=29.364373, ssim=0.959985, lpips=unavailable
  image_004: loss=0.0010638435, psnr=29.731222, ssim=0.963147, lpips=unavailable
  image_005: loss=0.0010014781, psnr=29.993585, ssim=0.959047, lpips=unavailable
  image_006: loss=0.0006189001, psnr=32.083794, ssim=0.968066, lpips=unavailable
  image_007: loss=0.0013543237, psnr=28.682775, ssim=0.941585, lpips=unavailable
  image_008: loss=0.0011194313, psnr=29.510025, ssim=0.949943, lpips=unavailable
  image_009: loss=0.0009085367, psnr=30.416575, ssim=0.964046, lpips=unavailable
  image_010: loss=0.0008248613, psnr=30.836190, ssim=0.964537, lpips=unavailable
  image_011: loss=0.0009132072, psnr=30.394306, ssim=0.968153, lpips=unavailable
  image_012: loss=0.0012555799, psnr=29.011556, ssim=0.963897, lpips=unavailable
  image_013: loss=0.0009547491, psnr=30.201107, ssim=0.970106, lpips=unavailable
  image_014: loss=0.0008502245, psnr=30.704664, ssim=0.971694, lpips=unavailable
  image_015: loss=0.0012364165, psnr=29.078352, ssim=0.957466, lpips=unavailable
  image_016: loss=0.0009582634, psnr=30.185151, ssim=0.969392, lpips=unavailable
  image_017: loss=0.0012204493, psnr=29.134802, ssim=0.956088, lpips=unavailable
  image_018: loss=0.0009376990, psnr=30.279365, ssim=0.957939, lpips=unavailable
  image_019: loss=0.0008358193, psnr=30.778875, ssim=0.958465, lpips=unavailable
  image_020: loss=0.0012183208, psnr=29.142383, ssim=0.950100, lpips=unavailable
  image_021: loss=0.0007438182, psnr=31.285331, ssim=0.969154, lpips=unavailable
  image_022: loss=0.0006313900, psnr=31.997023, ssim=0.972645, lpips=unavailable
  image_023: loss=0.0008541803, psnr=30.684504, ssim=0.968032, lpips=unavailable
  image_024: loss=0.0008150489, psnr=30.888163, ssim=0.971004, lpips=unavailable
