====================================================================================================
saved_time: 2026-05-03 01:21:37
script_path: /data2/greenx9/LRP_algo/00_run_nerf_ranksched_final.py
executed_command: /data2/greenx9/LRP_algo/00_run_nerf_ranksched_final.py --basedir /data2/greenx9/LRP_algo/logs/104_ours_rankwsd --config /data2/greenx9/LRP_algo/configs/ficus.txt --expname 104_ours_ficus_ficus_trial012_mlr1p29e-03_alr8p15e-04_decay176_mom0p950 --optimizer aux-sign-auto-cos-inc --train-scheduler rank_wsd --muon_lrate 0.0012931002195711574 --lrate 0.0008152056580021849 --lrate_decay 176 --muon_momentum 0.95 --lowrank_rank_start 150 --lowrank_rank_end 250 --lowrank_auto_init_rank_start --N_iters 100000 --seed 12 --eval_testset_only --test_out_json /data2/greenx9/LRP_algo/logs/104_ours_rankwsd/104_ours_ficus_ficus_trial012_mlr1p29e-03_alr8p15e-04_decay176_mom0p950/test_metrics_eval.json --test_out_dir /data2/greenx9/LRP_algo/logs/104_ours_rankwsd/104_ours_ficus_ficus_trial012_mlr1p29e-03_alr8p15e-04_decay176_mom0p950/testset_eval --lpips_net alex --i_print 5000 --i_weights 100000 --i_valset 100001 --i_testset 100001 --i_video 100001 --deterministic
expname: 104_ours_ficus_ficus_trial012_mlr1p29e-03_alr8p15e-04_decay176_mom0p950
iter: 99999
global_step: 99999
elapsed_time_from_train_start: 0 hour 2 min
current_train_loss: 0.0000000000
current_train_psnr: 0.000000
testset_mean_loss: 0.0011994974
testset_mean_psnr: 29.355827
testset_mean_ssim: 0.965642
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0008629177, psnr=30.640305, ssim=0.969547, lpips=unavailable
  image_001: loss=0.0013056907, psnr=28.841596, ssim=0.964710, lpips=unavailable
  image_002: loss=0.0014979980, psnr=28.244887, ssim=0.961286, lpips=unavailable
  image_003: loss=0.0017294341, psnr=27.620960, ssim=0.955586, lpips=unavailable
  image_004: loss=0.0017879888, psnr=27.476352, ssim=0.956818, lpips=unavailable
  image_005: loss=0.0011856760, psnr=29.260339, ssim=0.965568, lpips=unavailable
  image_006: loss=0.0013584109, psnr=28.669688, ssim=0.963442, lpips=unavailable
  image_007: loss=0.0008062572, psnr=30.935263, ssim=0.973688, lpips=unavailable
  image_008: loss=0.0012587284, psnr=29.000679, ssim=0.960502, lpips=unavailable
  image_009: loss=0.0014798348, psnr=28.297867, ssim=0.956816, lpips=unavailable
  image_010: loss=0.0009089205, psnr=30.414740, ssim=0.972477, lpips=unavailable
  image_011: loss=0.0008285408, psnr=30.816860, ssim=0.976648, lpips=unavailable
  image_012: loss=0.0008198873, psnr=30.862458, ssim=0.976205, lpips=unavailable
  image_013: loss=0.0007511213, psnr=31.242899, ssim=0.976977, lpips=unavailable
  image_014: loss=0.0011237623, psnr=29.493255, ssim=0.968567, lpips=unavailable
  image_015: loss=0.0010798780, psnr=29.666253, ssim=0.964722, lpips=unavailable
  image_016: loss=0.0011086954, psnr=29.551877, ssim=0.972929, lpips=unavailable
  image_017: loss=0.0010346234, psnr=29.852177, ssim=0.971333, lpips=unavailable
  image_018: loss=0.0011465705, psnr=29.405992, ssim=0.967203, lpips=unavailable
  image_019: loss=0.0012612984, psnr=28.991821, ssim=0.965329, lpips=unavailable
  image_020: loss=0.0013709102, psnr=28.629910, ssim=0.960467, lpips=unavailable
  image_021: loss=0.0007724897, psnr=31.121072, ssim=0.969640, lpips=unavailable
  image_022: loss=0.0013166014, psnr=28.805457, ssim=0.958829, lpips=unavailable
  image_023: loss=0.0018495658, psnr=27.329302, ssim=0.953887, lpips=unavailable
  image_024: loss=0.0013416334, psnr=28.723661, ssim=0.957870, lpips=unavailable
