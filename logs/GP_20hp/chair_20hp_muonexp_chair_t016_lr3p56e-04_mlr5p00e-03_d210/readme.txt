====================================================================================================
saved_time: 2026-05-19 21:58:28
script_path: /data2/greenx9/LRP_algo/00_run_nerf_ranksched_final.py
executed_command: /data2/greenx9/LRP_algo/00_run_nerf_ranksched_final.py --basedir /data2/greenx9/LRP_algo/logs/GP_20hp --config /data2/greenx9/LRP_algo/configs/chair.txt --expname chair_20hp_muonexp_chair_t016_lr3p56e-04_mlr5p00e-03_d210 --optimizer aux-muon --train-scheduler exp_decay --N_iters 100000 --seed 16 --lrate 0.00035572339 --lrate_decay 210 --muon_lrate 0.005 --muon_momentum 0.850000 --muon_decay 2.1511176e-06 --muon_aux_eps 0.00015312015 --muon_aux_weight_decay 1.504084e-06 --lowrank_ns_steps 5 --muon_aux_betas 0.947257,0.968744 --eval_testset_only --test_out_json /data2/greenx9/LRP_algo/logs/GP_20hp/chair_20hp_muonexp_chair_t016_lr3p56e-04_mlr5p00e-03_d210/test_metrics_eval.json --test_out_dir /data2/greenx9/LRP_algo/logs/GP_20hp/chair_20hp_muonexp_chair_t016_lr3p56e-04_mlr5p00e-03_d210/testset_eval --lpips_net alex --i_print 5000 --i_weights 100000 --i_valset 100001 --i_testset 100001 --i_video 100001 --deterministic
expname: chair_20hp_muonexp_chair_t016_lr3p56e-04_mlr5p00e-03_d210
iter: 99999
global_step: 99999
elapsed_time_from_train_start: 0 hour 2 min
current_train_loss: 0.0000000000
current_train_psnr: 0.000000
testset_mean_loss: 0.0004316065
testset_mean_psnr: 33.851159
testset_mean_ssim: 0.976565
testset_mean_lpips: 0.016044
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0001607501, psnr=37.938484, ssim=0.992835, lpips=0.006375
  image_001: loss=0.0002263406, psnr=36.452373, ssim=0.987932, lpips=0.007319
  image_002: loss=0.0004364098, psnr=33.601054, ssim=0.977618, lpips=0.017036
  image_003: loss=0.0004450596, psnr=33.515818, ssim=0.976197, lpips=0.017035
  image_004: loss=0.0005030635, psnr=32.983771, ssim=0.973138, lpips=0.016912
  image_005: loss=0.0005399393, psnr=32.676550, ssim=0.971865, lpips=0.014162
  image_006: loss=0.0005428792, psnr=32.652967, ssim=0.969933, lpips=0.016576
  image_007: loss=0.0005856869, psnr=32.323344, ssim=0.966744, lpips=0.021877
  image_008: loss=0.0004397284, psnr=33.568154, ssim=0.973286, lpips=0.016016
  image_009: loss=0.0004797344, psnr=33.189990, ssim=0.973628, lpips=0.018438
  image_010: loss=0.0006448231, psnr=31.905593, ssim=0.971217, lpips=0.024768
  image_011: loss=0.0004263638, psnr=33.702196, ssim=0.978226, lpips=0.018582
  image_012: loss=0.0003070348, psnr=35.128123, ssim=0.982878, lpips=0.014841
  image_013: loss=0.0003626508, psnr=34.405113, ssim=0.980579, lpips=0.018095
  image_014: loss=0.0003116788, psnr=35.062926, ssim=0.982754, lpips=0.014712
  image_015: loss=0.0004613565, psnr=33.359633, ssim=0.976748, lpips=0.020315
  image_016: loss=0.0003744238, psnr=34.266364, ssim=0.979002, lpips=0.013666
  image_017: loss=0.0005180586, psnr=32.856210, ssim=0.973519, lpips=0.016189
  image_018: loss=0.0005630021, psnr=32.494899, ssim=0.970923, lpips=0.018646
  image_019: loss=0.0005526880, psnr=32.575199, ssim=0.968350, lpips=0.019870
  image_020: loss=0.0004812657, psnr=33.176150, ssim=0.970328, lpips=0.016082
  image_021: loss=0.0004085472, psnr=33.887577, ssim=0.974130, lpips=0.015578
  image_022: loss=0.0004252163, psnr=33.713900, ssim=0.974340, lpips=0.017800
  image_023: loss=0.0003721255, psnr=34.293104, ssim=0.979664, lpips=0.012696
  image_024: loss=0.0002213353, psnr=36.549491, ssim=0.988288, lpips=0.007509
