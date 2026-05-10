====================================================================================================
saved_time: 2026-05-04 11:04:59
script_path: /data1/greenx9/LRP_algo/00_run_nerf_ranksched_final.py
executed_command: /data1/greenx9/LRP_algo/00_run_nerf_ranksched_final.py --basedir /data1/greenx9/LRP_algo/logs/100_muon_exp_decay_lrp39 --config /data1/greenx9/LRP_algo/configs/hotdog.txt --expname 100_muon_hotdog_hotdog_trial013_mlr2p27e-03_alr6p64e-04_decay100_mom0p950 --optimizer aux-muon --train-scheduler exp_decay --muon_lrate 0.002268049840710853 --lrate 0.0006635226608720027 --lrate_decay 100 --muon_momentum 0.95 --lowrank_rank_start 150 --lowrank_rank_end 250 --lowrank_auto_init_rank_start --N_iters 100000 --seed 13 --eval_testset_only --test_out_json /data1/greenx9/LRP_algo/logs/100_muon_exp_decay_lrp39/100_muon_hotdog_hotdog_trial013_mlr2p27e-03_alr6p64e-04_decay100_mom0p950/test_metrics_eval.json --test_out_dir /data1/greenx9/LRP_algo/logs/100_muon_exp_decay_lrp39/100_muon_hotdog_hotdog_trial013_mlr2p27e-03_alr6p64e-04_decay100_mom0p950/testset_eval --lpips_net alex --i_print 5000 --i_weights 100000 --i_valset 100001 --i_testset 100001 --i_video 100001 --deterministic
expname: 100_muon_hotdog_hotdog_trial013_mlr2p27e-03_alr6p64e-04_decay100_mom0p950
iter: 99999
global_step: 99999
elapsed_time_from_train_start: 0 hour 2 min
current_train_loss: 0.0000000000
current_train_psnr: 0.000000
testset_mean_loss: 0.0002170103
testset_mean_psnr: 37.417751
testset_mean_ssim: 0.981930
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0001957309, psnr=37.083405, ssim=0.981453, lpips=unavailable
  image_001: loss=0.0001694377, psnr=37.709898, ssim=0.983574, lpips=unavailable
  image_002: loss=0.0002088566, psnr=36.801515, ssim=0.980487, lpips=unavailable
  image_003: loss=0.0001421273, psnr=38.473221, ssim=0.983125, lpips=unavailable
  image_004: loss=0.0001107726, psnr=39.555674, ssim=0.985215, lpips=unavailable
  image_005: loss=0.0001585136, psnr=37.999331, ssim=0.983742, lpips=unavailable
  image_006: loss=0.0001979160, psnr=37.035188, ssim=0.983267, lpips=unavailable
  image_007: loss=0.0001300167, psnr=38.860005, ssim=0.985938, lpips=unavailable
  image_008: loss=0.0001328464, psnr=38.766500, ssim=0.982810, lpips=unavailable
  image_009: loss=0.0001517323, psnr=38.189216, ssim=0.978143, lpips=unavailable
  image_010: loss=0.0001372960, psnr=38.623418, ssim=0.979180, lpips=unavailable
  image_011: loss=0.0002199398, psnr=36.576959, ssim=0.977366, lpips=unavailable
  image_012: loss=0.0002354225, psnr=36.281518, ssim=0.983983, lpips=unavailable
  image_013: loss=0.0001392719, psnr=38.561361, ssim=0.989351, lpips=unavailable
  image_014: loss=0.0002644954, psnr=35.775817, ssim=0.983959, lpips=unavailable
  image_015: loss=0.0011653570, psnr=29.335410, ssim=0.966593, lpips=unavailable
  image_016: loss=0.0004511844, psnr=33.456458, ssim=0.973725, lpips=unavailable
  image_017: loss=0.0001368404, psnr=38.637854, ssim=0.984247, lpips=unavailable
  image_018: loss=0.0002146230, psnr=36.683235, ssim=0.981527, lpips=unavailable
  image_019: loss=0.0001783905, psnr=37.486281, ssim=0.983834, lpips=unavailable
  image_020: loss=0.0001170383, psnr=39.316716, ssim=0.986589, lpips=unavailable
  image_021: loss=0.0001148774, psnr=39.397652, ssim=0.984144, lpips=unavailable
  image_022: loss=0.0001280162, psnr=38.927346, ssim=0.982564, lpips=unavailable
  image_023: loss=0.0001360513, psnr=38.662970, ssim=0.982369, lpips=unavailable
  image_024: loss=0.0001885023, psnr=37.246832, ssim=0.981076, lpips=unavailable
