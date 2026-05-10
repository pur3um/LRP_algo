====================================================================================================
saved_time: 2026-05-03 22:37:18
script_path: /data2/greenx9/LRP_algo/00_run_nerf_ranksched_final.py
executed_command: /data2/greenx9/LRP_algo/00_run_nerf_ranksched_final.py --basedir /data2/greenx9/LRP_algo/logs/104_ours_rankwsd --config /data2/greenx9/LRP_algo/configs/lego.txt --expname 104_ours_lego_lego_trial014_mlr1p86e-03_alr6p28e-04_decay126_mom0p950 --optimizer aux-sign-auto-cos-inc --train-scheduler rank_wsd --muon_lrate 0.0018569052180253274 --lrate 0.0006282050023681401 --lrate_decay 126 --muon_momentum 0.95 --lowrank_rank_start 150 --lowrank_rank_end 250 --lowrank_auto_init_rank_start --N_iters 100000 --seed 14 --eval_testset_only --test_out_json /data2/greenx9/LRP_algo/logs/104_ours_rankwsd/104_ours_lego_lego_trial014_mlr1p86e-03_alr6p28e-04_decay126_mom0p950/test_metrics_eval.json --test_out_dir /data2/greenx9/LRP_algo/logs/104_ours_rankwsd/104_ours_lego_lego_trial014_mlr1p86e-03_alr6p28e-04_decay126_mom0p950/testset_eval --lpips_net alex --i_print 5000 --i_weights 100000 --i_valset 100001 --i_testset 100001 --i_video 100001 --deterministic
expname: 104_ours_lego_lego_trial014_mlr1p86e-03_alr6p28e-04_decay126_mom0p950
iter: 99999
global_step: 99999
elapsed_time_from_train_start: 0 hour 2 min
current_train_loss: 0.0000000000
current_train_psnr: 0.000000
testset_mean_loss: 0.0006745706
testset_mean_psnr: 31.958992
testset_mean_ssim: 0.967017
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0004794069, psnr=33.192957, ssim=0.971828, lpips=unavailable
  image_001: loss=0.0012292500, psnr=29.103598, ssim=0.957837, lpips=unavailable
  image_002: loss=0.0004228163, psnr=33.738482, ssim=0.973179, lpips=unavailable
  image_003: loss=0.0004821504, psnr=33.168174, ssim=0.971976, lpips=unavailable
  image_004: loss=0.0006942301, psnr=31.584965, ssim=0.965787, lpips=unavailable
  image_005: loss=0.0004616549, psnr=33.356825, ssim=0.969798, lpips=unavailable
  image_006: loss=0.0006943131, psnr=31.584446, ssim=0.967856, lpips=unavailable
  image_007: loss=0.0004278782, psnr=33.686798, ssim=0.968530, lpips=unavailable
  image_008: loss=0.0005043962, psnr=32.972281, ssim=0.972875, lpips=unavailable
  image_009: loss=0.0010332116, psnr=29.858107, ssim=0.967950, lpips=unavailable
  image_010: loss=0.0014292528, psnr=28.448909, ssim=0.954644, lpips=unavailable
  image_011: loss=0.0010149283, psnr=29.935646, ssim=0.957596, lpips=unavailable
  image_012: loss=0.0006659774, psnr=31.765405, ssim=0.968016, lpips=unavailable
  image_013: loss=0.0006593568, psnr=31.808795, ssim=0.967230, lpips=unavailable
  image_014: loss=0.0005283898, psnr=32.770455, ssim=0.966474, lpips=unavailable
  image_015: loss=0.0006500288, psnr=31.870674, ssim=0.968539, lpips=unavailable
  image_016: loss=0.0006413354, psnr=31.929147, ssim=0.976267, lpips=unavailable
  image_017: loss=0.0006479736, psnr=31.884426, ssim=0.968395, lpips=unavailable
  image_018: loss=0.0005853664, psnr=32.325721, ssim=0.967128, lpips=unavailable
  image_019: loss=0.0005248100, psnr=32.799978, ssim=0.964491, lpips=unavailable
  image_020: loss=0.0003707182, psnr=34.309560, ssim=0.970411, lpips=unavailable
  image_021: loss=0.0005487868, psnr=32.605963, ssim=0.968468, lpips=unavailable
  image_022: loss=0.0007502892, psnr=31.247713, ssim=0.964244, lpips=unavailable
  image_023: loss=0.0007742852, psnr=31.110990, ssim=0.962094, lpips=unavailable
  image_024: loss=0.0006434589, psnr=31.914791, ssim=0.963811, lpips=unavailable
