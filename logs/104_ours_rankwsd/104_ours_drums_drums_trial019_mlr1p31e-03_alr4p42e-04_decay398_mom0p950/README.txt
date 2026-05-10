====================================================================================================
saved_time: 2026-05-02 03:58:11
script_path: /data2/greenx9/LRP_algo/00_run_nerf_ranksched_final.py
executed_command: /data2/greenx9/LRP_algo/00_run_nerf_ranksched_final.py --basedir /data2/greenx9/LRP_algo/logs/104_ours_rankwsd --config /data2/greenx9/LRP_algo/configs/drums.txt --expname 104_ours_drums_drums_trial019_mlr1p31e-03_alr4p42e-04_decay398_mom0p950 --optimizer aux-sign-auto-cos-inc --train-scheduler rank_wsd --muon_lrate 0.0013068724959241743 --lrate 0.00044227240762720636 --lrate_decay 398 --muon_momentum 0.95 --lowrank_rank_start 150 --lowrank_rank_end 250 --lowrank_auto_init_rank_start --N_iters 100000 --seed 19 --eval_testset_only --test_out_json /data2/greenx9/LRP_algo/logs/104_ours_rankwsd/104_ours_drums_drums_trial019_mlr1p31e-03_alr4p42e-04_decay398_mom0p950/test_metrics_eval.json --test_out_dir /data2/greenx9/LRP_algo/logs/104_ours_rankwsd/104_ours_drums_drums_trial019_mlr1p31e-03_alr4p42e-04_decay398_mom0p950/testset_eval --lpips_net alex --i_print 5000 --i_weights 100000 --i_valset 100001 --i_testset 100001 --i_video 100001 --deterministic
expname: 104_ours_drums_drums_trial019_mlr1p31e-03_alr4p42e-04_decay398_mom0p950
iter: 99999
global_step: 99999
elapsed_time_from_train_start: 0 hour 2 min
current_train_loss: 0.0000000000
current_train_psnr: 0.000000
testset_mean_loss: 0.0028467321
testset_mean_psnr: 25.715303
testset_mean_ssim: 0.931685
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0017558035, psnr=27.555241, ssim=0.930571, lpips=unavailable
  image_001: loss=0.0021380815, psnr=26.699757, ssim=0.928315, lpips=unavailable
  image_002: loss=0.0014218342, psnr=28.471510, ssim=0.947377, lpips=unavailable
  image_003: loss=0.0022530232, psnr=26.472343, ssim=0.939318, lpips=unavailable
  image_004: loss=0.0018032938, psnr=27.439335, ssim=0.938494, lpips=unavailable
  image_005: loss=0.0017694349, psnr=27.521654, ssim=0.941880, lpips=unavailable
  image_006: loss=0.0043722321, psnr=23.592968, ssim=0.908639, lpips=unavailable
  image_007: loss=0.0035602730, psnr=24.485167, ssim=0.920845, lpips=unavailable
  image_008: loss=0.0029179945, psnr=25.349155, ssim=0.934009, lpips=unavailable
  image_009: loss=0.0036669276, psnr=24.356977, ssim=0.932122, lpips=unavailable
  image_010: loss=0.0031679147, psnr=24.992265, ssim=0.939575, lpips=unavailable
  image_011: loss=0.0041392348, psnr=23.830799, ssim=0.920427, lpips=unavailable
  image_012: loss=0.0036814502, psnr=24.339811, ssim=0.929708, lpips=unavailable
  image_013: loss=0.0026294871, psnr=25.801289, ssim=0.937041, lpips=unavailable
  image_014: loss=0.0024080526, psnr=26.183340, ssim=0.941816, lpips=unavailable
  image_015: loss=0.0040049814, psnr=23.973995, ssim=0.933964, lpips=unavailable
  image_016: loss=0.0019517487, psnr=27.095761, ssim=0.953150, lpips=unavailable
  image_017: loss=0.0026606678, psnr=25.750093, ssim=0.934195, lpips=unavailable
  image_018: loss=0.0028833637, psnr=25.401006, ssim=0.932814, lpips=unavailable
  image_019: loss=0.0050401324, psnr=22.975580, ssim=0.896474, lpips=unavailable
  image_020: loss=0.0032232890, psnr=24.917007, ssim=0.920087, lpips=unavailable
  image_021: loss=0.0037486081, psnr=24.261300, ssim=0.915726, lpips=unavailable
  image_022: loss=0.0026139973, psnr=25.826948, ssim=0.932113, lpips=unavailable
  image_023: loss=0.0019137182, psnr=27.181220, ssim=0.938950, lpips=unavailable
  image_024: loss=0.0014427592, psnr=28.408061, ssim=0.944517, lpips=unavailable
