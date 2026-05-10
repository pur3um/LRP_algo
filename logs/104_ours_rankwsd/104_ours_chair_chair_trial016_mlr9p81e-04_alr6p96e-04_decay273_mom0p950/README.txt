====================================================================================================
saved_time: 2026-05-02 04:03:30
script_path: /data2/greenx9/LRP_algo/00_run_nerf_ranksched_final.py
executed_command: /data2/greenx9/LRP_algo/00_run_nerf_ranksched_final.py --basedir /data2/greenx9/LRP_algo/logs/104_ours_rankwsd --config /data2/greenx9/LRP_algo/configs/chair.txt --expname 104_ours_chair_chair_trial016_mlr9p81e-04_alr6p96e-04_decay273_mom0p950 --optimizer aux-sign-auto-cos-inc --train-scheduler rank_wsd --muon_lrate 0.000981042338089727 --lrate 0.0006961338990544127 --lrate_decay 273 --muon_momentum 0.95 --lowrank_rank_start 150 --lowrank_rank_end 250 --lowrank_auto_init_rank_start --N_iters 100000 --seed 16 --eval_testset_only --test_out_json /data2/greenx9/LRP_algo/logs/104_ours_rankwsd/104_ours_chair_chair_trial016_mlr9p81e-04_alr6p96e-04_decay273_mom0p950/test_metrics_eval.json --test_out_dir /data2/greenx9/LRP_algo/logs/104_ours_rankwsd/104_ours_chair_chair_trial016_mlr9p81e-04_alr6p96e-04_decay273_mom0p950/testset_eval --lpips_net alex --i_print 5000 --i_weights 100000 --i_valset 100001 --i_testset 100001 --i_video 100001 --deterministic
expname: 104_ours_chair_chair_trial016_mlr9p81e-04_alr6p96e-04_decay273_mom0p950
iter: 99999
global_step: 99999
elapsed_time_from_train_start: 0 hour 2 min
current_train_loss: 0.0000000000
current_train_psnr: 0.000000
testset_mean_loss: 0.0004069152
testset_mean_psnr: 34.118847
testset_mean_ssim: 0.976180
testset_mean_lpips: unavailable
testset_lpips_net: alex
testset_lpips_status: unavailable
testset_lpips_error: LPIPS metric requires the `lpips` package. Install it with `pip install lpips`. If torchvision pretrained weights are not cached, the first run may also download the backbone weights.
testset_metrics_per_image:
  image_000: loss=0.0001468517, psnr=38.331206, ssim=0.992858, lpips=unavailable
  image_001: loss=0.0002039457, psnr=36.904852, ssim=0.987608, lpips=unavailable
  image_002: loss=0.0004171735, psnr=33.796832, ssim=0.977486, lpips=unavailable
  image_003: loss=0.0004280585, psnr=33.684967, ssim=0.975746, lpips=unavailable
  image_004: loss=0.0004741067, psnr=33.241238, ssim=0.973232, lpips=unavailable
  image_005: loss=0.0005096500, psnr=32.927279, ssim=0.971845, lpips=unavailable
  image_006: loss=0.0005169116, psnr=32.865836, ssim=0.969460, lpips=unavailable
  image_007: loss=0.0005535044, psnr=32.568789, ssim=0.965937, lpips=unavailable
  image_008: loss=0.0004149338, psnr=33.820210, ssim=0.972886, lpips=unavailable
  image_009: loss=0.0004510705, psnr=33.457555, ssim=0.973032, lpips=unavailable
  image_010: loss=0.0006120628, psnr=32.132040, ssim=0.970221, lpips=unavailable
  image_011: loss=0.0004320661, psnr=33.644497, ssim=0.976631, lpips=unavailable
  image_012: loss=0.0002940621, psnr=35.315608, ssim=0.982443, lpips=unavailable
  image_013: loss=0.0003454083, psnr=34.616671, ssim=0.979346, lpips=unavailable
  image_014: loss=0.0002917145, psnr=35.350418, ssim=0.981854, lpips=unavailable
  image_015: loss=0.0004301550, psnr=33.663749, ssim=0.976253, lpips=unavailable
  image_016: loss=0.0003447516, psnr=34.624936, ssim=0.979751, lpips=unavailable
  image_017: loss=0.0004872755, psnr=33.122253, ssim=0.973668, lpips=unavailable
  image_018: loss=0.0005241482, psnr=32.805458, ssim=0.971110, lpips=unavailable
  image_019: loss=0.0005252629, psnr=32.796232, ssim=0.967715, lpips=unavailable
  image_020: loss=0.0004459695, psnr=33.506948, ssim=0.969844, lpips=unavailable
  image_021: loss=0.0003747572, psnr=34.262498, ssim=0.973622, lpips=unavailable
  image_022: loss=0.0003975548, psnr=34.006029, ssim=0.974382, lpips=unavailable
  image_023: loss=0.0003508878, psnr=34.548316, ssim=0.979142, lpips=unavailable
  image_024: loss=0.0002005968, psnr=36.976758, ssim=0.988432, lpips=unavailable
