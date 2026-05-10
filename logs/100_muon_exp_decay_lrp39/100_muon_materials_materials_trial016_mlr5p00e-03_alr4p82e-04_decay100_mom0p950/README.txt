====================================================================================================
saved_time: 2026-05-04 10:57:19
script_path: /data1/greenx9/LRP_algo/00_run_nerf_ranksched_final.py
executed_command: /data1/greenx9/LRP_algo/00_run_nerf_ranksched_final.py --basedir /data1/greenx9/LRP_algo/logs/100_muon_exp_decay_lrp39 --config /data1/greenx9/LRP_algo/configs/materials.txt --expname 100_muon_materials_materials_trial016_mlr5p00e-03_alr4p82e-04_decay100_mom0p950 --optimizer aux-muon --train-scheduler exp_decay --muon_lrate 0.004999999999999999 --lrate 0.0004819393571767165 --lrate_decay 100 --muon_momentum 0.95 --lowrank_rank_start 150 --lowrank_rank_end 250 --lowrank_auto_init_rank_start --N_iters 100000 --seed 16 --eval_testset_only --test_out_json /data1/greenx9/LRP_algo/logs/100_muon_exp_decay_lrp39/100_muon_materials_materials_trial016_mlr5p00e-03_alr4p82e-04_decay100_mom0p950/test_metrics_eval.json --test_out_dir /data1/greenx9/LRP_algo/logs/100_muon_exp_decay_lrp39/100_muon_materials_materials_trial016_mlr5p00e-03_alr4p82e-04_decay100_mom0p950/testset_eval --lpips_net alex --i_print 5000 --i_weights 100000 --i_valset 100001 --i_testset 100001 --i_video 100001 --deterministic
expname: 100_muon_materials_materials_trial016_mlr5p00e-03_alr4p82e-04_decay100_mom0p950
iter: 99999
global_step: 99999
elapsed_time_from_train_start: 0 hour 2 min
current_train_loss: 0.0000000000
current_train_psnr: 0.000000
testset_mean_loss: 0.0010602479
testset_mean_psnr: 29.953124
testset_mean_ssim: 0.962116
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0017295907, psnr=27.620566, ssim=0.950162, lpips=unavailable
  image_001: loss=0.0021499358, psnr=26.675745, ssim=0.944207, lpips=unavailable
  image_002: loss=0.0015536193, psnr=28.086554, ssim=0.953841, lpips=unavailable
  image_003: loss=0.0010983618, psnr=29.592545, ssim=0.962583, lpips=unavailable
  image_004: loss=0.0010009761, psnr=29.995763, ssim=0.965555, lpips=unavailable
  image_005: loss=0.0009814503, psnr=30.081316, ssim=0.961245, lpips=unavailable
  image_006: loss=0.0005737936, psnr=32.412442, ssim=0.971057, lpips=unavailable
  image_007: loss=0.0013680041, psnr=28.639126, ssim=0.942077, lpips=unavailable
  image_008: loss=0.0011268603, psnr=29.481299, ssim=0.950672, lpips=unavailable
  image_009: loss=0.0008677124, psnr=30.616241, ssim=0.966330, lpips=unavailable
  image_010: loss=0.0008212661, psnr=30.855160, ssim=0.965797, lpips=unavailable
  image_011: loss=0.0008284830, psnr=30.817163, ssim=0.970466, lpips=unavailable
  image_012: loss=0.0012267950, psnr=29.112280, ssim=0.965017, lpips=unavailable
  image_013: loss=0.0009729313, psnr=30.119178, ssim=0.970903, lpips=unavailable
  image_014: loss=0.0008444334, psnr=30.734345, ssim=0.972395, lpips=unavailable
  image_015: loss=0.0012572398, psnr=29.005818, ssim=0.957797, lpips=unavailable
  image_016: loss=0.0009087818, psnr=30.415403, ssim=0.971450, lpips=unavailable
  image_017: loss=0.0011749354, psnr=29.299860, ssim=0.958549, lpips=unavailable
  image_018: loss=0.0009320651, psnr=30.305537, ssim=0.959852, lpips=unavailable
  image_019: loss=0.0008603763, psnr=30.653115, ssim=0.959187, lpips=unavailable
  image_020: loss=0.0012547426, psnr=29.014453, ssim=0.948685, lpips=unavailable
  image_021: loss=0.0007412164, psnr=31.300549, ssim=0.970056, lpips=unavailable
  image_022: loss=0.0005939925, psnr=32.262190, ssim=0.973936, lpips=unavailable
  image_023: loss=0.0008279441, psnr=30.819989, ssim=0.969163, lpips=unavailable
  image_024: loss=0.0008106894, psnr=30.911454, ssim=0.971924, lpips=unavailable
