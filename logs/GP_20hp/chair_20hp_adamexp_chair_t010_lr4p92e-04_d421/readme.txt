====================================================================================================
saved_time: 2026-05-20 06:52:26
script_path: /data2/greenx9/LRP_algo/00_run_nerf_ranksched_final.py
executed_command: /data2/greenx9/LRP_algo/00_run_nerf_ranksched_final.py --basedir /data2/greenx9/LRP_algo/logs/GP_20hp --config /data2/greenx9/LRP_algo/configs/chair.txt --expname chair_20hp_adamexp_chair_t010_lr4p92e-04_d421 --optimizer adam --train-scheduler exp_decay --N_iters 100000 --seed 10 --lrate 0.00049203879 --lrate_decay 421 --eval_testset_only --test_out_json /data2/greenx9/LRP_algo/logs/GP_20hp/chair_20hp_adamexp_chair_t010_lr4p92e-04_d421/test_metrics_eval.json --test_out_dir /data2/greenx9/LRP_algo/logs/GP_20hp/chair_20hp_adamexp_chair_t010_lr4p92e-04_d421/testset_eval --lpips_net alex --i_print 5000 --i_weights 100000 --i_valset 100001 --i_testset 100001 --i_video 100001 --deterministic
expname: chair_20hp_adamexp_chair_t010_lr4p92e-04_d421
iter: 99999
global_step: 99999
elapsed_time_from_train_start: 0 hour 5 min
current_train_loss: 0.0000000000
current_train_psnr: 0.000000
testset_mean_loss: 0.0005868827
testset_mean_psnr: 32.533300
testset_mean_ssim: 0.968274
testset_mean_lpips: 0.025240
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0002158087, psnr=36.659309, ssim=0.989968, lpips=0.009740
  image_001: loss=0.0003016852, psnr=35.204458, ssim=0.983266, lpips=0.009675
  image_002: loss=0.0005436886, psnr=32.646497, ssim=0.972040, lpips=0.020636
  image_003: loss=0.0006108348, psnr=32.140762, ssim=0.966944, lpips=0.024900
  image_004: loss=0.0007226014, psnr=31.411011, ssim=0.961397, lpips=0.030285
  image_005: loss=0.0007989788, psnr=30.974647, ssim=0.957911, lpips=0.028003
  image_006: loss=0.0007740561, psnr=31.112275, ssim=0.956676, lpips=0.029952
  image_007: loss=0.0007858381, psnr=31.046669, ssim=0.955256, lpips=0.035418
  image_008: loss=0.0006342267, psnr=31.977554, ssim=0.963332, lpips=0.028813
  image_009: loss=0.0006377305, psnr=31.953627, ssim=0.966188, lpips=0.024953
  image_010: loss=0.0007550598, psnr=31.220186, ssim=0.965095, lpips=0.030998
  image_011: loss=0.0005495970, psnr=32.599555, ssim=0.972747, lpips=0.024827
  image_012: loss=0.0004227832, psnr=33.738821, ssim=0.977899, lpips=0.022990
  image_013: loss=0.0004448150, psnr=33.518205, ssim=0.975804, lpips=0.025185
  image_014: loss=0.0003811809, psnr=34.188688, ssim=0.977423, lpips=0.019550
  image_015: loss=0.0005693782, psnr=32.445991, ssim=0.970781, lpips=0.026175
  image_016: loss=0.0005415555, psnr=32.663570, ssim=0.970419, lpips=0.019777
  image_017: loss=0.0007613241, psnr=31.184303, ssim=0.961858, lpips=0.025873
  image_018: loss=0.0008053874, psnr=30.939951, ssim=0.957854, lpips=0.029192
  image_019: loss=0.0007633578, psnr=31.172718, ssim=0.955930, lpips=0.029142
  image_020: loss=0.0007053126, psnr=31.516183, ssim=0.957858, lpips=0.031749
  image_021: loss=0.0005805643, psnr=32.361496, ssim=0.964564, lpips=0.029061
  image_022: loss=0.0005564029, psnr=32.546105, ssim=0.968081, lpips=0.027546
  image_023: loss=0.0005021934, psnr=32.991289, ssim=0.973215, lpips=0.034643
  image_024: loss=0.0003077076, psnr=35.118617, ssim=0.984343, lpips=0.011918
