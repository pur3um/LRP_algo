====================================================================================================
saved_time: 2026-05-03 01:34:45
script_path: /data1/greenx9/LRP_algo/00_run_nerf_ranksched_final.py
executed_command: /data1/greenx9/LRP_algo/00_run_nerf_ranksched_final.py --basedir /data1/greenx9/LRP_algo/logs/100_muon_exp_decay_lrp39 --config /data1/greenx9/LRP_algo/configs/ficus.txt --expname 100_muon_ficus_ficus_trial005_mlr3p60e-03_alr3p78e-04_decay100_mom0p950 --optimizer aux-muon --train-scheduler exp_decay --muon_lrate 0.003603201820095812 --lrate 0.0003780565779036198 --lrate_decay 100 --muon_momentum 0.95 --lowrank_rank_start 150 --lowrank_rank_end 250 --lowrank_auto_init_rank_start --N_iters 100000 --seed 5 --eval_testset_only --test_out_json /data1/greenx9/LRP_algo/logs/100_muon_exp_decay_lrp39/100_muon_ficus_ficus_trial005_mlr3p60e-03_alr3p78e-04_decay100_mom0p950/test_metrics_eval.json --test_out_dir /data1/greenx9/LRP_algo/logs/100_muon_exp_decay_lrp39/100_muon_ficus_ficus_trial005_mlr3p60e-03_alr3p78e-04_decay100_mom0p950/testset_eval --lpips_net alex --i_print 5000 --i_weights 100000 --i_valset 100001 --i_testset 100001 --i_video 100001 --deterministic
expname: 100_muon_ficus_ficus_trial005_mlr3p60e-03_alr3p78e-04_decay100_mom0p950
iter: 99999
global_step: 99999
elapsed_time_from_train_start: 0 hour 2 min
current_train_loss: 0.0000000000
current_train_psnr: 0.000000
testset_mean_loss: 0.0011884623
testset_mean_psnr: 29.408617
testset_mean_ssim: 0.966699
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0008403139, psnr=30.755584, ssim=0.970637, lpips=unavailable
  image_001: loss=0.0012686738, psnr=28.966500, ssim=0.965256, lpips=unavailable
  image_002: loss=0.0014455554, psnr=28.399652, ssim=0.962531, lpips=unavailable
  image_003: loss=0.0016062619, psnr=27.941836, ssim=0.958672, lpips=unavailable
  image_004: loss=0.0016896834, psnr=27.721946, ssim=0.958919, lpips=unavailable
  image_005: loss=0.0011924361, psnr=29.235649, ssim=0.967152, lpips=unavailable
  image_006: loss=0.0013527941, psnr=28.687683, ssim=0.964937, lpips=unavailable
  image_007: loss=0.0008071720, psnr=30.930338, ssim=0.974761, lpips=unavailable
  image_008: loss=0.0012236941, psnr=29.123271, ssim=0.963176, lpips=unavailable
  image_009: loss=0.0014380146, psnr=28.422367, ssim=0.959120, lpips=unavailable
  image_010: loss=0.0009105545, psnr=30.406940, ssim=0.972633, lpips=unavailable
  image_011: loss=0.0008038129, psnr=30.948450, ssim=0.977495, lpips=unavailable
  image_012: loss=0.0007979298, psnr=30.980353, ssim=0.977206, lpips=unavailable
  image_013: loss=0.0006993011, psnr=31.553357, ssim=0.978410, lpips=unavailable
  image_014: loss=0.0011293999, psnr=29.471522, ssim=0.968986, lpips=unavailable
  image_015: loss=0.0010938065, psnr=29.610595, ssim=0.965830, lpips=unavailable
  image_016: loss=0.0011255149, psnr=29.486487, ssim=0.973107, lpips=unavailable
  image_017: loss=0.0010414432, psnr=29.823644, ssim=0.972218, lpips=unavailable
  image_018: loss=0.0011411632, psnr=29.426522, ssim=0.968466, lpips=unavailable
  image_019: loss=0.0013368968, psnr=28.739021, ssim=0.965168, lpips=unavailable
  image_020: loss=0.0013637989, psnr=28.652496, ssim=0.961982, lpips=unavailable
  image_021: loss=0.0007351965, psnr=31.335965, ssim=0.971011, lpips=unavailable
  image_022: loss=0.0013047921, psnr=28.844586, ssim=0.959579, lpips=unavailable
  image_023: loss=0.0020911936, psnr=26.796057, ssim=0.951233, lpips=unavailable
  image_024: loss=0.0012721545, psnr=28.954601, ssim=0.958995, lpips=unavailable
