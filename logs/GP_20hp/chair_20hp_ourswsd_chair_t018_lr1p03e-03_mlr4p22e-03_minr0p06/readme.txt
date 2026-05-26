====================================================================================================
saved_time: 2026-05-17 08:32:08
script_path: /data2/greenx9/LRP_algo/00_run_nerf_ranksched_final.py
executed_command: /data2/greenx9/LRP_algo/00_run_nerf_ranksched_final.py --basedir /data2/greenx9/LRP_algo/logs/GP_20hp --config /data2/greenx9/LRP_algo/configs/chair.txt --expname chair_20hp_ourswsd_chair_t018_lr1p03e-03_mlr4p22e-03_minr0p06 --optimizer aux-sign-auto-cos-inc --train-scheduler rank_wsd --N_iters 100000 --seed 18 --lrate 0.0010302807 --muon_lrate 0.0042184136 --muon_momentum 0.863322 --muon_decay 1.3032826e-06 --muon_aux_eps 0.00046471454 --muon_aux_weight_decay 1.001032e-06 --lowrank_ns_steps 7 --sched_warmup_frac 0.028365 --sched_min_lr_ratio 0.061564 --sched_decay_start_frac 0.602732 --lowrank_rank_start 162 --lowrank_rank_end 285 --lowrank_oversample 8 --lowrank_init_energy 0.938667 --lowrank_init_probe_steps 32 --lowrank_init_round_multiple 16 --rank_schedule_default_frac 0.561458 --muon_aux_betas 0.906008,0.970996 --lowrank_auto_init_rank_start --eval_testset_only --test_out_json /data2/greenx9/LRP_algo/logs/GP_20hp/chair_20hp_ourswsd_chair_t018_lr1p03e-03_mlr4p22e-03_minr0p06/test_metrics_eval.json --test_out_dir /data2/greenx9/LRP_algo/logs/GP_20hp/chair_20hp_ourswsd_chair_t018_lr1p03e-03_mlr4p22e-03_minr0p06/testset_eval --lpips_net alex --i_print 5000 --i_weights 100000 --i_valset 100001 --i_testset 100001 --i_video 100001 --deterministic
expname: chair_20hp_ourswsd_chair_t018_lr1p03e-03_mlr4p22e-03_minr0p06
iter: 99999
global_step: 99999
elapsed_time_from_train_start: 0 hour 2 min
current_train_loss: 0.0000000000
current_train_psnr: 0.000000
testset_mean_loss: 0.0003743686
testset_mean_psnr: 34.491843
testset_mean_ssim: 0.978814
testset_mean_lpips: 0.013942
testset_lpips_net: alex
testset_lpips_status: ok
testset_metrics_per_image:
  image_000: loss=0.0001278914, psnr=38.931582, ssim=0.993959, lpips=0.005378
  image_001: loss=0.0001844219, psnr=37.341872, ssim=0.989260, lpips=0.005793
  image_002: loss=0.0003706050, psnr=34.310886, ssim=0.980424, lpips=0.014685
  image_003: loss=0.0003859108, psnr=34.135130, ssim=0.978555, lpips=0.013453
  image_004: loss=0.0004375615, psnr=33.589608, ssim=0.975889, lpips=0.014789
  image_005: loss=0.0004587477, psnr=33.384260, ssim=0.975205, lpips=0.012497
  image_006: loss=0.0004655657, psnr=33.320189, ssim=0.973215, lpips=0.013743
  image_007: loss=0.0005086217, psnr=32.936050, ssim=0.969930, lpips=0.020549
  image_008: loss=0.0003718974, psnr=34.295767, ssim=0.975681, lpips=0.013299
  image_009: loss=0.0004429165, psnr=33.536780, ssim=0.975094, lpips=0.017399
  image_010: loss=0.0005832826, psnr=32.341209, ssim=0.972247, lpips=0.025271
  image_011: loss=0.0003818798, psnr=34.180732, ssim=0.979051, lpips=0.016656
  image_012: loss=0.0002705036, psnr=35.678268, ssim=0.984773, lpips=0.013192
  image_013: loss=0.0003409436, psnr=34.673173, ssim=0.981713, lpips=0.016466
  image_014: loss=0.0002642912, psnr=35.779171, ssim=0.983826, lpips=0.012854
  image_015: loss=0.0004242221, psnr=33.724066, ssim=0.977700, lpips=0.018406
  image_016: loss=0.0003148630, psnr=35.018782, ssim=0.981780, lpips=0.011174
  image_017: loss=0.0004531129, psnr=33.437935, ssim=0.976463, lpips=0.014461
  image_018: loss=0.0004781069, psnr=33.204749, ssim=0.974768, lpips=0.013545
  image_019: loss=0.0004807820, psnr=33.180517, ssim=0.971539, lpips=0.015739
  image_020: loss=0.0004029787, psnr=33.947179, ssim=0.973641, lpips=0.014326
  image_021: loss=0.0003353023, psnr=34.745633, ssim=0.977087, lpips=0.011949
  image_022: loss=0.0003672045, psnr=34.350919, ssim=0.977037, lpips=0.015795
  image_023: loss=0.0003235953, psnr=34.899977, ssim=0.981458, lpips=0.011359
  image_024: loss=0.0001840072, psnr=37.351650, ssim=0.990061, lpips=0.005773
