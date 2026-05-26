====================================================================================================
saved_time: 2026-05-25 00:37:45
script_path: /data2/greenx9/LRP_algo/00_run_nerf_ranksched_final.py
executed_command: /data2/greenx9/LRP_algo/00_run_nerf_ranksched_final.py --basedir /data2/greenx9/LRP_algo/logs/GP_20hp --config /data2/greenx9/LRP_algo/configs/drums.txt --expname drums_20hp_muonexp_drums_t015_lr7p54e-04_mlr5p00e-03_d189 --optimizer aux-muon --train-scheduler exp_decay --N_iters 100000 --seed 15 --lrate 0.0007535158 --lrate_decay 189 --muon_lrate 0.005 --muon_momentum 0.850000 --muon_decay 6.145799e-06 --muon_aux_eps 2.8225408e-05 --muon_aux_weight_decay 2.9346606e-05 --lowrank_ns_steps 4 --muon_aux_betas 0.933302,0.961155 --eval_testset_only --test_out_json /data2/greenx9/LRP_algo/logs/GP_20hp/drums_20hp_muonexp_drums_t015_lr7p54e-04_mlr5p00e-03_d189/test_metrics_eval.json --test_out_dir /data2/greenx9/LRP_algo/logs/GP_20hp/drums_20hp_muonexp_drums_t015_lr7p54e-04_mlr5p00e-03_d189/testset_eval --lpips_net alex --i_print 5000 --i_weights 100000 --i_valset 100001 --i_testset 100001 --i_video 100001 --deterministic
expname: drums_20hp_muonexp_drums_t015_lr7p54e-04_mlr5p00e-03_d189
iter: 99999
global_step: 99999
elapsed_time_from_train_start: 0 hour 2 min
current_train_loss: 0.0000000000
current_train_psnr: 0.000000
testset_mean_loss: 0.0030060282
testset_mean_psnr: 25.457693
testset_mean_ssim: 0.930343
testset_mean_lpips: 0.055991
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0017879723, psnr=27.476392, ssim=0.932856, lpips=0.052669
  image_001: loss=0.0023098008, psnr=26.364254, ssim=0.927126, lpips=0.056919
  image_002: loss=0.0014637890, psnr=28.345215, ssim=0.947554, lpips=0.043152
  image_003: loss=0.0023644564, psnr=26.262687, ssim=0.937498, lpips=0.042552
  image_004: loss=0.0019887052, psnr=27.014296, ssim=0.935757, lpips=0.045424
  image_005: loss=0.0019840347, psnr=27.024507, ssim=0.939166, lpips=0.053174
  image_006: loss=0.0046522263, psnr=23.323392, ssim=0.906707, lpips=0.079402
  image_007: loss=0.0036362938, psnr=24.393410, ssim=0.921176, lpips=0.068967
  image_008: loss=0.0029058175, psnr=25.367317, ssim=0.934680, lpips=0.047921
  image_009: loss=0.0037155177, psnr=24.299807, ssim=0.931864, lpips=0.053201
  image_010: loss=0.0032926758, psnr=24.824510, ssim=0.939061, lpips=0.046922
  image_011: loss=0.0042942003, psnr=23.671177, ssim=0.917081, lpips=0.068903
  image_012: loss=0.0041438676, psnr=23.825941, ssim=0.926768, lpips=0.065530
  image_013: loss=0.0027354814, psnr=25.629662, ssim=0.937546, lpips=0.055909
  image_014: loss=0.0025766196, psnr=25.889497, ssim=0.939511, lpips=0.052256
  image_015: loss=0.0040490525, psnr=23.926466, ssim=0.933472, lpips=0.046416
  image_016: loss=0.0021260700, psnr=26.724224, ssim=0.951210, lpips=0.038828
  image_017: loss=0.0032322621, psnr=24.904934, ssim=0.932332, lpips=0.061203
  image_018: loss=0.0030387482, psnr=25.173053, ssim=0.931284, lpips=0.059470
  image_019: loss=0.0050325943, psnr=22.982081, ssim=0.897040, lpips=0.084779
  image_020: loss=0.0035427322, psnr=24.506617, ssim=0.917051, lpips=0.060354
  image_021: loss=0.0038931710, psnr=24.096965, ssim=0.914081, lpips=0.062873
  image_022: loss=0.0026226672, psnr=25.812568, ssim=0.931144, lpips=0.054716
  image_023: loss=0.0021518585, psnr=26.671863, ssim=0.934539, lpips=0.051278
  image_024: loss=0.0016100914, psnr=27.931494, ssim=0.942081, lpips=0.046957
