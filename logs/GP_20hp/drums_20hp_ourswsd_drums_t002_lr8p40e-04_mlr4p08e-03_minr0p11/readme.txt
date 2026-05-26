====================================================================================================
saved_time: 2026-05-22 11:31:53
script_path: /data2/greenx9/LRP_algo/00_run_nerf_ranksched_final.py
executed_command: /data2/greenx9/LRP_algo/00_run_nerf_ranksched_final.py --basedir /data2/greenx9/LRP_algo/logs/GP_20hp --config /data2/greenx9/LRP_algo/configs/drums.txt --expname drums_20hp_ourswsd_drums_t002_lr8p40e-04_mlr4p08e-03_minr0p11 --optimizer aux-sign-auto-cos-inc --train-scheduler rank_wsd --N_iters 100000 --seed 2 --lrate 0.00084023224 --muon_lrate 0.004079664 --muon_momentum 0.864092 --muon_decay 2.1538902e-06 --muon_aux_eps 0.00046833518 --muon_aux_weight_decay 1.0341532e-06 --lowrank_ns_steps 6 --sched_warmup_frac 0.026605 --sched_min_lr_ratio 0.113014 --sched_decay_start_frac 0.584771 --lowrank_rank_start 149 --lowrank_rank_end 290 --lowrank_oversample 8 --lowrank_init_energy 0.927152 --lowrank_init_probe_steps 31 --lowrank_init_round_multiple 16 --rank_schedule_default_frac 0.552776 --muon_aux_betas 0.900755,0.970082 --lowrank_auto_init_rank_start --eval_testset_only --test_out_json /data2/greenx9/LRP_algo/logs/GP_20hp/drums_20hp_ourswsd_drums_t002_lr8p40e-04_mlr4p08e-03_minr0p11/test_metrics_eval.json --test_out_dir /data2/greenx9/LRP_algo/logs/GP_20hp/drums_20hp_ourswsd_drums_t002_lr8p40e-04_mlr4p08e-03_minr0p11/testset_eval --lpips_net alex --i_print 5000 --i_weights 100000 --i_valset 100001 --i_testset 100001 --i_video 100001 --deterministic
expname: drums_20hp_ourswsd_drums_t002_lr8p40e-04_mlr4p08e-03_minr0p11
iter: 99999
global_step: 99999
elapsed_time_from_train_start: 0 hour 2 min
current_train_loss: 0.0000000000
current_train_psnr: 0.000000
testset_mean_loss: 0.0027791966
testset_mean_psnr: 25.817654
testset_mean_ssim: 0.933524
testset_mean_lpips: 0.053139
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0016727485, psnr=27.765693, ssim=0.934552, lpips=0.050277
  image_001: loss=0.0019472919, psnr=27.105689, ssim=0.932033, lpips=0.052279
  image_002: loss=0.0013093126, psnr=28.829566, ssim=0.951721, lpips=0.038610
  image_003: loss=0.0022788499, psnr=26.422843, ssim=0.939540, lpips=0.039438
  image_004: loss=0.0018515232, psnr=27.324708, ssim=0.937436, lpips=0.041266
  image_005: loss=0.0017719761, psnr=27.515421, ssim=0.943892, lpips=0.045863
  image_006: loss=0.0041490672, psnr=23.820495, ssim=0.908581, lpips=0.077099
  image_007: loss=0.0034181059, psnr=24.662145, ssim=0.924842, lpips=0.068989
  image_008: loss=0.0026389034, psnr=25.785765, ssim=0.938187, lpips=0.044118
  image_009: loss=0.0035531421, psnr=24.493874, ssim=0.933384, lpips=0.052918
  image_010: loss=0.0032107346, psnr=24.933956, ssim=0.939938, lpips=0.047888
  image_011: loss=0.0038313009, psnr=24.166537, ssim=0.923833, lpips=0.064228
  image_012: loss=0.0036525030, psnr=24.374094, ssim=0.931785, lpips=0.058759
  image_013: loss=0.0025584963, psnr=25.920152, ssim=0.940292, lpips=0.050519
  image_014: loss=0.0023807096, psnr=26.232936, ssim=0.943612, lpips=0.051106
  image_015: loss=0.0040348014, psnr=23.941778, ssim=0.934387, lpips=0.043857
  image_016: loss=0.0019583346, psnr=27.081131, ssim=0.952946, lpips=0.035704
  image_017: loss=0.0026816237, psnr=25.716022, ssim=0.935738, lpips=0.058071
  image_018: loss=0.0029665409, psnr=25.277497, ssim=0.934204, lpips=0.055893
  image_019: loss=0.0048590712, psnr=23.134467, ssim=0.898469, lpips=0.080630
  image_020: loss=0.0031163788, psnr=25.063497, ssim=0.922808, lpips=0.061297
  image_021: loss=0.0037955109, psnr=24.207298, ssim=0.916339, lpips=0.055963
  image_022: loss=0.0025180741, psnr=25.989315, ssim=0.934088, lpips=0.058877
  image_023: loss=0.0019024732, psnr=27.206814, ssim=0.938813, lpips=0.050131
  image_024: loss=0.0014224416, psnr=28.469655, ssim=0.946674, lpips=0.044696
