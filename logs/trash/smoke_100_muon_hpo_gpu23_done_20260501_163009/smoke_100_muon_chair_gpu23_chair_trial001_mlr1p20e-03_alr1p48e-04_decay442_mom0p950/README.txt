====================================================================================================
saved_time: 2026-05-01 16:23:48
script_path: /data1/greenx9/LRP_algo/00_run_nerf_ranksched_final.py
executed_command: /data1/greenx9/LRP_algo/00_run_nerf_ranksched_final.py --basedir /data1/greenx9/LRP_algo/logs/smoke_100_muon_hpo_gpu23 --config /data1/greenx9/LRP_algo/configs/chair.txt --expname smoke_100_muon_chair_gpu23_chair_trial001_mlr1p20e-03_alr1p48e-04_decay442_mom0p950 --optimizer aux-muon --train-scheduler exp_decay --muon_lrate 0.001201558947613998 --lrate 0.00014847413781803658 --lrate_decay 442 --muon_momentum 0.95 --lowrank_rank_start 150 --lowrank_rank_end 250 --lowrank_auto_init_rank_start --N_iters 20 --seed 1 --eval_testset_only --test_out_json /data1/greenx9/LRP_algo/logs/smoke_100_muon_hpo_gpu23/smoke_100_muon_chair_gpu23_chair_trial001_mlr1p20e-03_alr1p48e-04_decay442_mom0p950/test_metrics_eval.json --test_out_dir /data1/greenx9/LRP_algo/logs/smoke_100_muon_hpo_gpu23/smoke_100_muon_chair_gpu23_chair_trial001_mlr1p20e-03_alr1p48e-04_decay442_mom0p950/testset_eval --lpips_net alex --i_print 5000 --i_weights 20 --i_valset 21 --i_testset 21 --i_video 21 --deterministic
expname: smoke_100_muon_chair_gpu23_chair_trial001_mlr1p20e-03_alr1p48e-04_decay442_mom0p950
iter: 19
global_step: 19
elapsed_time_from_train_start: 0 hour 2 min
current_train_loss: 0.0000000000
current_train_psnr: 0.000000
testset_mean_loss: 0.0410867507
testset_mean_psnr: 14.104062
testset_mean_ssim: 0.823627
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0201348849, psnr=16.960508, ssim=0.909591, lpips=unavailable
  image_001: loss=0.0286691654, psnr=15.425849, ssim=0.879732, lpips=unavailable
  image_002: loss=0.0393990166, psnr=14.045146, ssim=0.840244, lpips=unavailable
  image_003: loss=0.0456968844, psnr=13.401134, ssim=0.817271, lpips=unavailable
  image_004: loss=0.0563747250, psnr=12.489156, ssim=0.773611, lpips=unavailable
  image_005: loss=0.0637303740, psnr=11.956535, ssim=0.749612, lpips=unavailable
  image_006: loss=0.0630010217, psnr=12.006524, ssim=0.754397, lpips=unavailable
  image_007: loss=0.0542971790, psnr=12.652227, ssim=0.777391, lpips=unavailable
  image_008: loss=0.0394548327, psnr=14.038998, ssim=0.812906, lpips=unavailable
  image_009: loss=0.0262933541, psnr=15.801540, ssim=0.853763, lpips=unavailable
  image_010: loss=0.0202537794, psnr=16.934939, ssim=0.864219, lpips=unavailable
  image_011: loss=0.0274488125, psnr=15.614764, ssim=0.853106, lpips=unavailable
  image_012: loss=0.0337924436, psnr=14.711804, ssim=0.853606, lpips=unavailable
  image_013: loss=0.0368661173, psnr=14.333726, ssim=0.852733, lpips=unavailable
  image_014: loss=0.0367253572, psnr=14.350340, ssim=0.843413, lpips=unavailable
  image_015: loss=0.0342165083, psnr=14.657643, ssim=0.849310, lpips=unavailable
  image_016: loss=0.0405976139, psnr=13.914995, ssim=0.822090, lpips=unavailable
  image_017: loss=0.0520416498, psnr=12.836489, ssim=0.778111, lpips=unavailable
  image_018: loss=0.0592216402, psnr=12.275196, ssim=0.762833, lpips=unavailable
  image_019: loss=0.0605667010, psnr=12.177661, ssim=0.762163, lpips=unavailable
  image_020: loss=0.0543787032, psnr=12.645712, ssim=0.776951, lpips=unavailable
  image_021: loss=0.0445965566, psnr=13.506987, ssim=0.814809, lpips=unavailable
  image_022: loss=0.0340853631, psnr=14.674321, ssim=0.851830, lpips=unavailable
  image_023: loss=0.0298549738, psnr=15.249833, ssim=0.855460, lpips=unavailable
  image_024: loss=0.0254711099, psnr=15.939521, ssim=0.881525, lpips=unavailable
