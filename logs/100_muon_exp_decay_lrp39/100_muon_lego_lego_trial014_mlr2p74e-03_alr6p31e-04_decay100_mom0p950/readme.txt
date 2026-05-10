====================================================================================================
saved_time: 2026-05-04 11:40:59
script_path: /data1/greenx9/LRP_algo/00_run_nerf_ranksched_final.py
executed_command: /data1/greenx9/LRP_algo/00_run_nerf_ranksched_final.py --basedir /data1/greenx9/LRP_algo/logs/100_muon_exp_decay_lrp39 --config /data1/greenx9/LRP_algo/configs/lego.txt --expname 100_muon_lego_lego_trial014_mlr2p74e-03_alr6p31e-04_decay100_mom0p950 --optimizer aux-muon --train-scheduler exp_decay --muon_lrate 0.0027442163470687785 --lrate 0.0006305249748576165 --lrate_decay 100 --muon_momentum 0.95 --lowrank_rank_start 150 --lowrank_rank_end 250 --lowrank_auto_init_rank_start --N_iters 100000 --seed 14 --eval_testset_only --test_out_json /data1/greenx9/LRP_algo/logs/100_muon_exp_decay_lrp39/100_muon_lego_lego_trial014_mlr2p74e-03_alr6p31e-04_decay100_mom0p950/test_metrics_eval.json --test_out_dir /data1/greenx9/LRP_algo/logs/100_muon_exp_decay_lrp39/100_muon_lego_lego_trial014_mlr2p74e-03_alr6p31e-04_decay100_mom0p950/testset_eval --lpips_net alex --i_print 5000 --i_weights 100000 --i_valset 100001 --i_testset 100001 --i_video 100001 --deterministic
expname: 100_muon_lego_lego_trial014_mlr2p74e-03_alr6p31e-04_decay100_mom0p950
iter: 99999
global_step: 99999
elapsed_time_from_train_start: 0 hour 2 min
current_train_loss: 0.0000000000
current_train_psnr: 0.000000
testset_mean_loss: 0.0006818842
testset_mean_psnr: 31.961947
testset_mean_ssim: 0.966903
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0004848659, psnr=33.143783, ssim=0.972350, lpips=unavailable
  image_001: loss=0.0015711351, psnr=28.037864, ssim=0.956184, lpips=unavailable
  image_002: loss=0.0004071522, psnr=33.902431, ssim=0.974070, lpips=unavailable
  image_003: loss=0.0004829880, psnr=33.160636, ssim=0.972279, lpips=unavailable
  image_004: loss=0.0006627442, psnr=31.786540, ssim=0.966413, lpips=unavailable
  image_005: loss=0.0004508484, psnr=33.459694, ssim=0.969763, lpips=unavailable
  image_006: loss=0.0006905926, psnr=31.607780, ssim=0.967351, lpips=unavailable
  image_007: loss=0.0004191544, psnr=33.776259, ssim=0.967340, lpips=unavailable
  image_008: loss=0.0004786990, psnr=33.199374, ssim=0.973690, lpips=unavailable
  image_009: loss=0.0009864963, psnr=30.059045, ssim=0.969183, lpips=unavailable
  image_010: loss=0.0014141529, psnr=28.495036, ssim=0.953512, lpips=unavailable
  image_011: loss=0.0010313246, psnr=29.866046, ssim=0.956092, lpips=unavailable
  image_012: loss=0.0006868804, psnr=31.631188, ssim=0.967848, lpips=unavailable
  image_013: loss=0.0006791714, psnr=31.680205, ssim=0.966920, lpips=unavailable
  image_014: loss=0.0005319293, psnr=32.741460, ssim=0.966729, lpips=unavailable
  image_015: loss=0.0006456327, psnr=31.900144, ssim=0.968541, lpips=unavailable
  image_016: loss=0.0006033879, psnr=32.194033, ssim=0.977149, lpips=unavailable
  image_017: loss=0.0006327216, psnr=31.987873, ssim=0.968655, lpips=unavailable
  image_018: loss=0.0006076936, psnr=32.163153, ssim=0.966657, lpips=unavailable
  image_019: loss=0.0005413437, psnr=32.665268, ssim=0.963225, lpips=unavailable
  image_020: loss=0.0003568359, psnr=34.475314, ssim=0.970166, lpips=unavailable
  image_021: loss=0.0005402181, psnr=32.674308, ssim=0.968669, lpips=unavailable
  image_022: loss=0.0007458530, psnr=31.273467, ssim=0.964114, lpips=unavailable
  image_023: loss=0.0007647897, psnr=31.164579, ssim=0.961503, lpips=unavailable
  image_024: loss=0.0006304943, psnr=32.003188, ssim=0.964163, lpips=unavailable
