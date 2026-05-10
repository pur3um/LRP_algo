====================================================================================================
saved_time: 2026-05-03 02:05:34
script_path: /data1/greenx9/LRP_algo/00_run_nerf_ranksched_final.py
executed_command: /data1/greenx9/LRP_algo/00_run_nerf_ranksched_final.py --basedir /data1/greenx9/LRP_algo/logs/100_muon_exp_decay_lrp39 --config /data1/greenx9/LRP_algo/configs/drums.txt --expname 100_muon_drums_drums_trial013_mlr3p16e-03_alr5p92e-04_decay100_mom0p950 --optimizer aux-muon --train-scheduler exp_decay --muon_lrate 0.003156695027066108 --lrate 0.0005921110455091653 --lrate_decay 100 --muon_momentum 0.95 --lowrank_rank_start 150 --lowrank_rank_end 250 --lowrank_auto_init_rank_start --N_iters 100000 --seed 13 --eval_testset_only --test_out_json /data1/greenx9/LRP_algo/logs/100_muon_exp_decay_lrp39/100_muon_drums_drums_trial013_mlr3p16e-03_alr5p92e-04_decay100_mom0p950/test_metrics_eval.json --test_out_dir /data1/greenx9/LRP_algo/logs/100_muon_exp_decay_lrp39/100_muon_drums_drums_trial013_mlr3p16e-03_alr5p92e-04_decay100_mom0p950/testset_eval --lpips_net alex --i_print 5000 --i_weights 100000 --i_valset 100001 --i_testset 100001 --i_video 100001 --deterministic
expname: 100_muon_drums_drums_trial013_mlr3p16e-03_alr5p92e-04_decay100_mom0p950
iter: 99999
global_step: 99999
elapsed_time_from_train_start: 0 hour 2 min
current_train_loss: 0.0000000000
current_train_psnr: 0.000000
testset_mean_loss: 0.0027303148
testset_mean_psnr: 25.895077
testset_mean_ssim: 0.934269
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0016389489, psnr=27.854346, ssim=0.934923, lpips=unavailable
  image_001: loss=0.0020226035, psnr=26.940892, ssim=0.931447, lpips=unavailable
  image_002: loss=0.0012452584, psnr=29.047405, ssim=0.951773, lpips=unavailable
  image_003: loss=0.0021281333, psnr=26.720011, ssim=0.940069, lpips=unavailable
  image_004: loss=0.0018067900, psnr=27.430923, ssim=0.939159, lpips=unavailable
  image_005: loss=0.0016854207, psnr=27.732916, ssim=0.945035, lpips=unavailable
  image_006: loss=0.0039839288, psnr=23.996884, ssim=0.911178, lpips=unavailable
  image_007: loss=0.0034161825, psnr=24.664589, ssim=0.924806, lpips=unavailable
  image_008: loss=0.0026237150, psnr=25.810833, ssim=0.939659, lpips=unavailable
  image_009: loss=0.0033675083, psnr=24.726913, ssim=0.935769, lpips=unavailable
  image_010: loss=0.0031273887, psnr=25.048181, ssim=0.941783, lpips=unavailable
  image_011: loss=0.0036393718, psnr=24.389736, ssim=0.924331, lpips=unavailable
  image_012: loss=0.0035033610, psnr=24.555151, ssim=0.933081, lpips=unavailable
  image_013: loss=0.0025330263, psnr=25.963603, ssim=0.940651, lpips=unavailable
  image_014: loss=0.0023950839, psnr=26.206793, ssim=0.944141, lpips=unavailable
  image_015: loss=0.0037537287, psnr=24.255371, ssim=0.936714, lpips=unavailable
  image_016: loss=0.0019451597, psnr=27.110447, ssim=0.954178, lpips=unavailable
  image_017: loss=0.0028771025, psnr=25.410447, ssim=0.934958, lpips=unavailable
  image_018: loss=0.0027318858, psnr=25.635374, ssim=0.936522, lpips=unavailable
  image_019: loss=0.0050901305, psnr=22.932711, ssim=0.895791, lpips=unavailable
  image_020: loss=0.0032511000, psnr=24.879697, ssim=0.921861, lpips=unavailable
  image_021: loss=0.0036965408, psnr=24.322045, ssim=0.917683, lpips=unavailable
  image_022: loss=0.0024281102, psnr=26.147316, ssim=0.935172, lpips=unavailable
  image_023: loss=0.0019611914, psnr=27.074800, ssim=0.939307, lpips=unavailable
  image_024: loss=0.0014061999, psnr=28.519529, ssim=0.946727, lpips=unavailable
