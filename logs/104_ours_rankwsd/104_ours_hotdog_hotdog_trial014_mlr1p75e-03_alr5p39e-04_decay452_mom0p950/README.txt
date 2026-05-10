====================================================================================================
saved_time: 2026-05-03 01:17:16
script_path: /data2/greenx9/LRP_algo/00_run_nerf_ranksched_final.py
executed_command: /data2/greenx9/LRP_algo/00_run_nerf_ranksched_final.py --basedir /data2/greenx9/LRP_algo/logs/104_ours_rankwsd --config /data2/greenx9/LRP_algo/configs/hotdog.txt --expname 104_ours_hotdog_hotdog_trial014_mlr1p75e-03_alr5p39e-04_decay452_mom0p950 --optimizer aux-sign-auto-cos-inc --train-scheduler rank_wsd --muon_lrate 0.0017493541604244305 --lrate 0.0005386584733294548 --lrate_decay 452 --muon_momentum 0.95 --lowrank_rank_start 150 --lowrank_rank_end 250 --lowrank_auto_init_rank_start --N_iters 100000 --seed 14 --eval_testset_only --test_out_json /data2/greenx9/LRP_algo/logs/104_ours_rankwsd/104_ours_hotdog_hotdog_trial014_mlr1p75e-03_alr5p39e-04_decay452_mom0p950/test_metrics_eval.json --test_out_dir /data2/greenx9/LRP_algo/logs/104_ours_rankwsd/104_ours_hotdog_hotdog_trial014_mlr1p75e-03_alr5p39e-04_decay452_mom0p950/testset_eval --lpips_net alex --i_print 5000 --i_weights 100000 --i_valset 100001 --i_testset 100001 --i_video 100001 --deterministic
expname: 104_ours_hotdog_hotdog_trial014_mlr1p75e-03_alr5p39e-04_decay452_mom0p950
iter: 99999
global_step: 99999
elapsed_time_from_train_start: 0 hour 2 min
current_train_loss: 0.0000000000
current_train_psnr: 0.000000
testset_mean_loss: 0.0002146961
testset_mean_psnr: 37.452442
testset_mean_ssim: 0.982071
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0001861744, psnr=37.300797, ssim=0.982259, lpips=unavailable
  image_001: loss=0.0001719051, psnr=37.647110, ssim=0.983622, lpips=unavailable
  image_002: loss=0.0002087482, psnr=36.803770, ssim=0.980600, lpips=unavailable
  image_003: loss=0.0001449926, psnr=38.386538, ssim=0.983362, lpips=unavailable
  image_004: loss=0.0001075273, psnr=39.684809, ssim=0.985403, lpips=unavailable
  image_005: loss=0.0001568958, psnr=38.043884, ssim=0.983771, lpips=unavailable
  image_006: loss=0.0001927730, psnr=37.149536, ssim=0.983347, lpips=unavailable
  image_007: loss=0.0001257475, psnr=39.005003, ssim=0.986060, lpips=unavailable
  image_008: loss=0.0001364255, psnr=38.651041, ssim=0.982407, lpips=unavailable
  image_009: loss=0.0001548196, psnr=38.101738, ssim=0.977847, lpips=unavailable
  image_010: loss=0.0001418706, psnr=38.481073, ssim=0.978595, lpips=unavailable
  image_011: loss=0.0002206261, psnr=36.563429, ssim=0.977537, lpips=unavailable
  image_012: loss=0.0002292223, psnr=36.397429, ssim=0.984524, lpips=unavailable
  image_013: loss=0.0001338790, psnr=38.732873, ssim=0.989941, lpips=unavailable
  image_014: loss=0.0002648659, psnr=35.769738, ssim=0.984147, lpips=unavailable
  image_015: loss=0.0011223850, psnr=29.498581, ssim=0.968449, lpips=unavailable
  image_016: loss=0.0004673222, psnr=33.303835, ssim=0.973241, lpips=unavailable
  image_017: loss=0.0001455013, psnr=38.371329, ssim=0.983992, lpips=unavailable
  image_018: loss=0.0002148331, psnr=36.678987, ssim=0.981498, lpips=unavailable
  image_019: loss=0.0001740259, psnr=37.593858, ssim=0.984304, lpips=unavailable
  image_020: loss=0.0001134516, psnr=39.451892, ssim=0.986403, lpips=unavailable
  image_021: loss=0.0001121434, psnr=39.502259, ssim=0.983964, lpips=unavailable
  image_022: loss=0.0001219442, psnr=39.138386, ssim=0.982436, lpips=unavailable
  image_023: loss=0.0001336116, psnr=38.741556, ssim=0.982279, lpips=unavailable
  image_024: loss=0.0001857122, psnr=37.311593, ssim=0.981772, lpips=unavailable
