import os
import re
import sys
import json
import time
import random
from typing import Optional

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange

from run_nerf_helpers_optuna import *
from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data

from optims.muon import SingleDeviceMuon, SingleDeviceMuonWithAuxAdam
from optims.lr_sign import SingleDeviceSignWithAuxAdam
from optims.lr_sign10_rsclF import SingleDeviceSign10RsclFWithAuxAdam
from optims.auto_cos_inc_rank import SingleDeviceAutoCosIncWithAuxAdam
from optims.run_utils import parse_pair


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = False


def seed_everything(seed: int = 0, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def init_results_log_optim(results_path, args, optimizer, start):
    optim_tensor_count = sum(len(group['params']) for group in optimizer.param_groups)
    optim_total_params = sum(p.numel() for group in optimizer.param_groups for p in group['params'])
    with open(results_path, 'a') as f:
        f.write('=' * 100 + '\n')
        f.write(f"run_start_time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("script: run_nerf_muon.py\n")
        f.write(f"expname: {args.expname}\n")
        f.write(f"dataset_type: {args.dataset_type}\n")
        f.write(f"datadir: {args.datadir}\n")
        f.write(f"resume_start_step: {start}\n")
        f.write("optimizer_setup:\n")
        f.write("  optimizer_main: Muon\n")
        f.write(f"  muon_lr_init: {args.muon_lrate}\n")
        f.write(f"  muon_weight_decay: {args.muon_decay}\n")
        f.write(f"  muon_momentum: {args.muon_momentum}\n")
        f.write("  optimizer_aux: Adam\n")
        f.write(f"  aux_adam_lr_init: {args.lrate}\n")
        f.write("  aux_adam_betas: (0.9, 0.999)\n")
        f.write(f"  lrate_decay_ksteps: {args.lrate_decay}\n")
        f.write(f"  muon_tensor_count: {optim_tensor_count}\n")
        f.write(f"  muon_total_params: {optim_total_params}\n")
        if getattr(args, 'optimizer', '') == 'aux-lowrank-svd':
            f.write("  lowrank_schedule:\n")
            f.write(f"    rank_start: {args.lowrank_rank_start}\n")
            f.write(f"    rank_end: {args.lowrank_rank_end}\n")
            f.write(f"    schedule: {args.lowrank_schedule}\n")
            f.write(f"    schedule_steps: {args.lowrank_schedule_steps if args.lowrank_schedule_steps > 0 else args.N_iters}\n")
            f.write(f"    oversample: {args.lowrank_oversample}\n")
            f.write(f"    subspace_iters: {args.lowrank_subspace_iters}\n")
            f.write(f"    ns_steps: {args.lowrank_ns_steps}\n")
            f.write(f"    min_dim: {args.lowrank_min_dim}\n")
            f.write(f"    max_rank_ratio: {args.lowrank_max_rank_ratio}\n")
            f.write(f"    scale_mode: {args.lowrank_scale_mode}\n")
        f.write("train_psnr_log:\n")
        if getattr(args, 'optimizer', '') == 'aux-sign-auto-cos-inc':
            f.write("  lowrank_schedule:\n")
            f.write(f"    rank_start: {args.lowrank_rank_start}\n")
            f.write(f"    rank_end: {args.lowrank_rank_end}\n")
            f.write("    schedule: cosine_increase_closed_form\n")
            f.write(f"    schedule_steps: {args.lowrank_schedule_steps if args.lowrank_schedule_steps > 0 else args.N_iters}\n")
            f.write(f"    oversample: {args.lowrank_oversample}\n")
            f.write(f"    subspace_iters: {args.lowrank_subspace_iters}\n")
            f.write(f"    ns_steps: {args.lowrank_ns_steps}\n")
            f.write(f"    scale_mode: {args.lowrank_scale_mode}\n")
            if args.lowrank_auto_init_rank_start:
                f.write("  lowrank_auto_init_start:\n")
                f.write(f"    enabled: {args.lowrank_auto_init_rank_start}\n")
                f.write(f"    probe_steps: {args.lowrank_init_probe_steps}\n")
                f.write(f"    energy_tau: {args.lowrank_init_energy}\n")
                f.write(f"    round_multiple: {args.lowrank_init_round_multiple}\n")


def batchify(fn, chunk):
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret

def split_nerf_params(net):
    muon_params = []
    adam_params = []

    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue

        # bias / norm-like / scalar-vector params
        if p.ndim < 2:
            adam_params.append(p)
            continue

        # final output heads는 일단 Adam으로 두기
        if (
            "output_linear" in name
            or "rgb_linear" in name
            or "alpha_linear" in name
        ):
            adam_params.append(p)
            continue

        # 나머지 2D weight는 Muon
        muon_params.append(p)

    return muon_params, adam_params

def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64):
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024 * 32, **kwargs):
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i + chunk], **kwargs)
        for k in ret:
            all_ret.setdefault(k, []).append(ret[k])
    return {k: torch.cat(v, 0) for k, v in all_ret.items()}


def render(H, W, K, chunk=1024 * 32, rays=None, c2w=None, ndc=True,
           near=0.0, far=1.0, use_viewdirs=False, c2w_staticcam=None, **kwargs):
    if c2w is not None:
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        rays_o, rays_d = rays

    if use_viewdirs:
        viewdirs = rays_d
        if c2w_staticcam is not None:
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
    else:
        viewdirs = None

    sh = rays_d.shape
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1.0, rays_o, rays_d)

    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()
    near = near * torch.ones_like(rays_d[..., :1])
    far = far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ["rgb_map", "disp_map", "acc_map"]
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]



def evaluate_val_psnr(i_val, poses, images, hwf, K, args, render_kwargs_test, max_views=None):
    H, W, _ = hwf
    val_ids = list(i_val)
    if max_views is not None and max_views > 0:
        val_ids = val_ids[:max_views]
    if len(val_ids) == 0:
        return float("-inf")

    psnrs = []
    with torch.no_grad():
        for vid in val_ids:
            target = images[vid]
            if isinstance(target, np.ndarray):
                target = torch.tensor(target, dtype=torch.float32, device=device)
            else:
                target = target.to(device)
            pose = poses[vid, :3, :4]
            rgb, _, _, _ = render(H, W, K, chunk=args.chunk, c2w=pose, **render_kwargs_test)
            loss = img2mse(rgb, target)
            psnrs.append(mse2psnr(loss).item())
    return float(np.mean(psnrs))


def _is_numbered_ckpt(filename: str) -> bool:
    return re.fullmatch(r"\d{6}\.tar", filename) is not None


def save_checkpoint(path, global_step, render_kwargs_train, optimizer, best_iter=None, best_val_psnr=None):
    ckpt = {
        "global_step": global_step,
        "network_fn_state_dict": render_kwargs_train["network_fn"].state_dict(),
        "network_fine_state_dict": (
            render_kwargs_train["network_fine"].state_dict()
            if render_kwargs_train["network_fine"] is not None else None
        ),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if best_iter is not None:
        ckpt["best_iter"] = int(best_iter)
    if best_val_psnr is not None:
        ckpt["best_val_psnr"] = float(best_val_psnr)
    torch.save(ckpt, path)


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args):

    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)

    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(
        D=args.netdepth, W=args.netwidth, input_ch=input_ch, output_ch=output_ch,
        skips=skips, input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs,
    ).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(
            D=args.netdepth_fine, W=args.netwidth_fine, input_ch=input_ch, output_ch=output_ch,
            skips=skips, input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs,
        ).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn: run_network(
        inputs, viewdirs, network_fn,
        embed_fn=embed_fn, embeddirs_fn=embeddirs_fn, netchunk=args.netchunk,
    )

    #!!!! OPTIMIZER
    #@ Muon INR: split_muon_like_named_params
    muon_params, adam_params = split_nerf_params(model)
    if model_fine is not None:
        muon_params_fine, adam_params_fine = split_nerf_params(model_fine)
        muon_params += muon_params_fine
        adam_params += adam_params_fine
    #//======== Create optimizers ========//#
    aux_param_groups = [
            dict(params=muon_params,
                use_muon=True,
                lr=args.muon_lrate,
                weight_decay=args.muon_decay,
                momentum=args.muon_momentum
            ),
            dict(
                params=adam_params,
                use_muon=False,
                lr=args.lrate,
                betas=parse_pair(args.muon_aux_betas),
                eps=args.muon_aux_eps,
                weight_decay=args.muon_aux_weight_decay,
            ),
        ]
    
    if args.optimizer in ('ori-adam', 'adam', 'original', 'ori'):
        optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
    elif args.optimizer == 'aux-muon':
        optimizer = SingleDeviceMuonWithAuxAdam(aux_param_groups)
        print(
            f'INFO: Aux-Muon optimizer configured. Hidden params: {len(muon_params)}, '
            f'Aux params: {len(adam_params)}.'
        )
    elif args.optimizer == 'aux-sign' or args.optimizer == 'lr-sign':
        optimizer = SingleDeviceSignWithAuxAdam(aux_param_groups)
        print(
            f'INFO: Aux-sign optimizer configured. Hidden params: {len(muon_params)}, '
            f'Aux params: {len(adam_params)}. '
        )
    elif args.optimizer == 'lr-sign10-rsclF' or args.optimizer == 'aux-sign10-rsclF':
        aux_param_groups[0].update(
            dict(
                iters=args.N_iters,
                auto_init_rank_start=args.lowrank_auto_init_rank_start,
                init_probe_steps=args.lowrank_init_probe_steps,
                init_energy=args.lowrank_init_energy,
                init_round_multiple=args.lowrank_init_round_multiple,
            )
        )
        optimizer = SingleDeviceSign10RsclFWithAuxAdam(aux_param_groups)
        print(
            f'INFO: Sign10 Rescale False optimizer configured. Hidden params: {len(muon_params)}, '
            f'Aux params: {len(adam_params)}. '
        )
    elif args.optimizer == 'aux-sign-auto-cos-inc':
        aux_param_groups[0].update(
            dict(
                iters=args.N_iters,
                auto_init_rank_start=args.lowrank_auto_init_rank_start,
                init_probe_steps=args.lowrank_init_probe_steps,
                init_energy=args.lowrank_init_energy,
                init_round_multiple=args.lowrank_init_round_multiple,
            )
        )
        optimizer = SingleDeviceAutoCosIncWithAuxAdam(aux_param_groups)
        print(
            f'INFO: Sign Auto Cos increase optimizer configured. Hidden params: {len(muon_params)}, '
            f'Aux params: {len(adam_params)}. '
        )
    else:
        raise ValueError(f"Unknown optimizer={args.optimizer}")

    start = 0
    expdir = os.path.join(args.basedir, args.expname)
    if args.ft_path is not None and args.ft_path != "None":
        ckpts = [args.ft_path]
    elif os.path.exists(expdir):
        ckpts = [os.path.join(expdir, f) for f in sorted(os.listdir(expdir)) if _is_numbered_ckpt(f)]
    else:
        ckpts = []

    print("Found ckpts", ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print("Reloading from", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device)
        start = ckpt["global_step"]
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        model.load_state_dict(ckpt["network_fn_state_dict"])
        if model_fine is not None and ckpt.get("network_fine_state_dict") is not None:
            model_fine.load_state_dict(ckpt["network_fine_state_dict"])

    render_kwargs_train = {
        "network_query_fn": network_query_fn,
        "perturb": args.perturb,
        "N_importance": args.N_importance,
        "network_fine": model_fine,
        "N_samples": args.N_samples,
        "network_fn": model,
        "use_viewdirs": args.use_viewdirs,
        "white_bkgd": args.white_bkgd,
        "raw_noise_std": args.raw_noise_std,
    }
    if args.dataset_type != "llff" or args.no_ndc:
        print("Not ndc!")
        render_kwargs_train["ndc"] = False
        render_kwargs_train["lindisp"] = args.lindisp

    render_kwargs_test = dict(render_kwargs_train)
    render_kwargs_test["perturb"] = False
    render_kwargs_test["raw_noise_std"] = 0.0
    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.0 - torch.exp(-act_fn(raw) * dists)
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], -1)
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])
    noise = 0.0
    if raw_noise_std > 0.0:
        noise = torch.randn(raw[..., 3].shape, device=raw.device, dtype=raw.dtype) * raw_noise_std
        if pytest:
            np.random.seed(0)
            noise = torch.tensor(np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std, device=raw.device, dtype=raw.dtype)

    alpha = raw2alpha(raw[..., 3] + noise, dists)
    weights = alpha * torch.cumprod(
        torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device, dtype=alpha.dtype), 1.0 - alpha + 1e-10], -1), -1
    )[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)
    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)
    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])
    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch, network_fn, network_query_fn, N_samples, retraw=False,
                lindisp=False, perturb=0.0, N_importance=0, network_fine=None,
                white_bkgd=False, raw_noise_std=0.0, verbose=False, pytest=False):
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]

    t_vals = torch.linspace(0.0, 1.0, steps=N_samples, device=near.device, dtype=near.dtype)
    if not lindisp:
        z_vals = near * (1.0 - t_vals) + far * t_vals
    else:
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.0:
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        t_rand = torch.rand(z_vals.shape, device=z_vals.device, dtype=z_vals.dtype)
        if pytest:
            np.random.seed(0)
            t_rand = torch.tensor(np.random.rand(*list(z_vals.shape)), device=z_vals.device, dtype=z_vals.dtype)
        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map
        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.0), pytest=pytest)
        z_samples = z_samples.detach()
        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {"rgb_map": rgb_map, "disp_map": disp_map, "acc_map": acc_map}
    if retraw:
        ret["raw"] = raw
    if N_importance > 0:
        ret["rgb0"] = rgb_map_0
        ret["disp0"] = disp_map_0
        ret["acc0"] = acc_map_0
        ret["z_std"] = torch.std(z_samples, dim=-1, unbiased=False)
    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")
    return ret


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument("--expname", type=str, help="experiment name")
    parser.add_argument("--basedir", type=str, default="./logs/", help="where to store ckpts and logs")
    parser.add_argument("--datadir", type=str, default="./data/llff/fern", help="input data directory")

    parser.add_argument("--netdepth", type=int, default=8)
    parser.add_argument("--netwidth", type=int, default=256)
    parser.add_argument("--netdepth_fine", type=int, default=8)
    parser.add_argument("--netwidth_fine", type=int, default=256)
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4)
    parser.add_argument("--lrate", type=float, default=5e-4)
    parser.add_argument("--lrate_decay", type=int, default=250)
    parser.add_argument("--chunk", type=int, default=1024 * 32)
    parser.add_argument("--netchunk", type=int, default=1024 * 64)
    parser.add_argument("--no_batching", action="store_true")
    parser.add_argument("--no_reload", action="store_true")
    parser.add_argument("--ft_path", type=str, default=None)

    parser.add_argument("--N_samples", type=int, default=64)
    parser.add_argument("--N_importance", type=int, default=0)
    parser.add_argument("--perturb", type=float, default=1.0)
    parser.add_argument("--use_viewdirs", action="store_true")
    parser.add_argument("--i_embed", type=int, default=0)
    parser.add_argument("--multires", type=int, default=10)
    parser.add_argument("--multires_views", type=int, default=4)
    parser.add_argument("--raw_noise_std", type=float, default=0.0)

    parser.add_argument("--render_only", action="store_true")
    parser.add_argument("--render_test", action="store_true")
    parser.add_argument("--render_factor", type=int, default=0)
    parser.add_argument("--lpips_net", type=str, default="alex", choices=["alex", "vgg", "squeeze"])

    parser.add_argument("--precrop_iters", type=int, default=0)
    parser.add_argument("--precrop_frac", type=float, default=0.5)

    parser.add_argument("--dataset_type", type=str, default="llff")
    parser.add_argument("--testskip", type=int, default=8)
    parser.add_argument("--shape", type=str, default="greek")
    parser.add_argument("--white_bkgd", action="store_true")
    parser.add_argument("--half_res", action="store_true")
    parser.add_argument("--factor", type=int, default=8)
    parser.add_argument("--no_ndc", action="store_true")
    parser.add_argument("--lindisp", action="store_true")
    parser.add_argument("--spherify", action="store_true")
    parser.add_argument("--llffhold", type=int, default=8)

    parser.add_argument("--i_print", type=int, default=2000)
    parser.add_argument("--i_img", type=int, default=100000)
    parser.add_argument("--i_weights", type=int, default=100000)
    parser.add_argument("--i_testset", type=int, default=100000)
    parser.add_argument("--i_video", type=int, default=100000)
    parser.add_argument("--N_iters", type=int, default=100000)

    # Muon / Sign optimizer options
    parser.add_argument("--muon_lrate", type=float, default=5e-4)
    parser.add_argument("--muon_decay", type=float, default=0.0)  # muon_weight_decay
    parser.add_argument("--muon_momentum", type=float, default=0.90)
    parser.add_argument("--muon_aux_eps", type=float, default=5e-4, help="Epsilon for Muon auxiliary Adam branch.")
    parser.add_argument("--muon_aux_weight_decay", type=float, default=0.0, help="Weight decay for auxiliary Adam in Muon.")
    parser.add_argument("--muon_aux_betas", type=str, default="0.9,0.95", help="Betas for Muon auxiliary Adam branch.")
    parser.add_argument("--lowrank_rank_start", type=int, default=150,
                        help="Starting rank for low-rank Muon schedule.")
    parser.add_argument("--lowrank_rank_end", type=int, default=250,
                        help="Ending rank for low-rank Muon schedule.")
    parser.add_argument("--lowrank_schedule", type=str, default="constant",
                        choices=["constant", "linear", "cosine"],
                        help="Rank schedule for low-rank Muon branch.")
    parser.add_argument("--lowrank_schedule_steps", type=int, default=0,
                        help="Number of optimizer steps over which to apply the rank schedule. 0 means use N_iters.")
    parser.add_argument("--lowrank_oversample", type=int, default=4,
                        help="Oversampling for randomized low-rank subspace estimation.")
    parser.add_argument("--lowrank_subspace_iters", type=int, default=1,
                        help="Number of subspace power iterations for randomized low-rank Muon.")
    parser.add_argument("--lowrank_ns_steps", type=int, default=5,
                        help="Number of Newton-Schulz steps when falling back to full-rank orthogonalization.")
    parser.add_argument("--lowrank_min_dim", type=int, default=256,
                        help="Use Newton-Schulz instead of low-rank approximation when min(weight.shape) is smaller than this value.")
    parser.add_argument("--lowrank_max_rank_ratio", type=float, default=1.0,
                        help="Optional cap: effective rank <= lowrank_max_rank_ratio * min(weight.shape).")
    parser.add_argument("--lowrank_scale_mode", type=str, default="sqrt",
                        choices=["sqrt", "none"],
                        help="How to rescale truncated low-rank updates. sqrt preserves the original behavior; none makes smaller ranks truly weaker.")
    parser.add_argument(
        "--lowrank_auto_init_rank_start",
        action="store_true",
        help="If set, estimate rank_start from the first few Muon search matrices using a Frobenius-energy probe, then continue with the original cosine increase schedule.",
    )
    parser.add_argument(
        "--lowrank_init_probe_steps",
        type=int,
        default=8,
        help="Number of early steps used to estimate rank_start when --lowrank_auto_init_rank_start is enabled.",
    )
    parser.add_argument(
        "--lowrank_init_energy",
        type=float,
        default=0.999,
        help="Energy threshold tau for choosing rank_start from the sketched spectrum.",
    )
    parser.add_argument(
        "--lowrank_init_round_multiple",
        type=int,
        default=8,
        help="Round the auto-selected rank_start up to this multiple.",
    )
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")

    parser.add_argument("--eval_every", type=int, default=5000)
    parser.add_argument("--max_eval_views", type=int, default=2)
    parser.add_argument("--metric_out", type=str, default="")
    return parser


def train():
    parser = config_parser()
    args = parser.parse_args()
    seed_everything(args.seed, deterministic=args.deterministic)

    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor, recenter=True, bd_factor=.75, spherify=args.spherify)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]
        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]
        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if (i not in i_test and i not in i_val)])
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
        else:
            near = 0.
            far = 1.
    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split
        near, far = 2., 6.
        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]
    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        i_train, i_val, i_test = i_split
        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]
    elif args.dataset_type == 'deepvoxels':
        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape, basedir=args.datadir, testskip=args.testskip)
        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split
        hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
        near = hemi_R - 1.
        far = hemi_R + 1.
    else:
        raise ValueError(f'Unknown dataset type: {args.dataset_type}')

    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    if K is None:
        K = np.array([[focal, 0, 0.5 * W], [0, focal, 0.5 * H], [0, 0, 1]])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    expdir = os.path.join(args.basedir, args.expname)

    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start
    render_kwargs_train.update({'near': near, 'far': far})
    render_kwargs_test.update({'near': near, 'far': far})
    render_poses = torch.tensor(render_poses, dtype=torch.float32, device=device)

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())
    results_path = os.path.join(basedir, expname, 'results.txt')
    init_results_log_optim(results_path, args, optimizer, start)

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            images_render = images[i_test] if args.render_test else None
            testsavedir = os.path.join(expdir, f"renderonly_{'test' if args.render_test else 'path'}_{start:06d}")
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)
            rgbs, _ = render_path(
                render_poses, hwf, K, args.chunk, render_kwargs_test,
                gt_imgs=images_render, savedir=testsavedir, render_factor=args.render_factor,
            )
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)
        return

    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:, :3, :4]], 0)
        rays_rgb = np.concatenate([rays, images[:, None]], 1)
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0)
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3]).astype(np.float32)
        np.random.shuffle(rays_rgb)
        i_batch = 0

    if use_batching:
        images = torch.tensor(images, dtype=torch.float32, device=device)
    poses = torch.tensor(poses, dtype=torch.float32, device=device)
    if use_batching:
        rays_rgb = torch.tensor(rays_rgb, dtype=torch.float32, device=device)

    best_val_psnr = -1e9
    best_iter = -1
    best_ckpt_path = os.path.join(expdir, 'best.tar')
    start = start + 1
    train_start_time = time.time()

    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    for i in trange(start, args.N_iters + 1):
        time0 = time.time()
        if use_batching:
            batch = rays_rgb[i_batch:i_batch + N_rand]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]
            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print('Shuffle data after an epoch!')
                rays_rgb = rays_rgb[torch.randperm(rays_rgb.shape[0], device=rays_rgb.device)]
                i_batch = 0
        else:
            img_i = np.random.choice(i_train)
            target = torch.tensor(images[img_i], dtype=torch.float32, device=device)
            pose = poses[img_i, :3, :4]
            rays_o, rays_d = get_rays(H, W, K, pose)
            if i < args.precrop_iters:
                dH = int(H // 2 * args.precrop_frac)
                dW = int(W // 2 * args.precrop_frac)
                coords = torch.stack(torch.meshgrid(
                    torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH, device=pose.device),
                    torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW, device=pose.device), indexing='ij'
                ), -1)
            else:
                coords = torch.stack(torch.meshgrid(
                    torch.linspace(0, H - 1, H, device=pose.device),
                    torch.linspace(0, W - 1, W, device=pose.device), indexing='ij'
                ), -1)
            coords = torch.reshape(coords, [-1, 2])
            select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)
            select_inds = torch.as_tensor(select_inds, device=coords.device, dtype=torch.long)
            select_coords = coords[select_inds].long()
            rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]
            rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]
            batch_rays = torch.stack([rays_o, rays_d], 0)
            target_s = target[select_coords[:, 0], select_coords[:, 1]]

        rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays, verbose=i < 10, retraw=True, **render_kwargs_train)
        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        loss = img_loss
        psnr = mse2psnr(img_loss)
        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
        loss.backward()
        optimizer.step()

        #! (방법1) optimizer schedule도 실험
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_muon_lrate = args.muon_lrate * (decay_rate ** (global_step / decay_steps))
        new_adam_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            if param_group.get('use_muon', False):
                param_group['lr'] = new_muon_lrate
            else:
                param_group['lr'] = new_adam_lrate

        # decay_rate = 0.1
        # decay_steps = args.lrate_decay * 1000
        # new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        # for group in optimizer.param_groups:
        #     group['lr'] = new_lrate

        if i % args.i_weights == 0:
            path = os.path.join(expdir, f'{i:06d}.tar')
            save_checkpoint(path, global_step, render_kwargs_train, optimizer, best_iter, best_val_psnr if best_iter >= 0 else None)
            print('Saved checkpoints at', path)

        if args.eval_every > 0 and i % args.eval_every == 0 and i > 0:
            val_psnr = evaluate_val_psnr(i_val, poses, images, hwf, K, args, render_kwargs_test, max_views=args.max_eval_views)
            if val_psnr > best_val_psnr:
                best_val_psnr = val_psnr
                best_iter = i
                save_checkpoint(best_ckpt_path, global_step, render_kwargs_train, optimizer, best_iter, best_val_psnr)
                tqdm.write(f"[BEST] Saved best checkpoint to {best_ckpt_path} (PSNR={best_val_psnr:.4f}, iter={best_iter})")
            tqdm.write(f"[VAL] Iter: {i} mean PSNR: {val_psnr:.4f} | best: {best_val_psnr:.4f} @ {best_iter}")

        if i % args.i_video == 0 and i > 0:
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            moviebase = os.path.join(expdir, f'{args.expname}_spiral_{i:06d}_')
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(expdir, f'testset_{i:06d}')
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(
                    poses[i_test], hwf, K, args.chunk, render_kwargs_test,
                    gt_imgs=images[i_test], savedir=testsavedir,
                )
            print('Saved test set')

        if i % args.i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item():.10f} PSNR: {psnr.item():.6f}")

        global_step += 1

    if args.metric_out:
        os.makedirs(os.path.dirname(args.metric_out) or '.', exist_ok=True)
        with open(args.metric_out, 'w') as f:
            json.dump({
                'best_val_psnr': float(best_val_psnr),
                'best_iter': int(best_iter),
                'best_ckpt_path': best_ckpt_path if best_iter >= 0 else '',
                'lrate': float(args.lrate),
                'muon_lrate': float(args.muon_lrate),
                'expname': args.expname,
                'N_iters': int(args.N_iters),
                'eval_every': int(args.eval_every),
                'max_eval_views': int(args.max_eval_views),
                'elapsed_sec': float(time.time() - train_start_time),
                'optimizer': args.optimizer,
                'seed': int(args.seed),
            }, f, indent=2)
        print(f"Saved metric json to {args.metric_out}")


if __name__ == '__main__':
    train()

# CUDA_VISIBLE_DEVICES=0 python run_inc_optims_optuna_ready.py --config configs/lego.txt --basedir ./logs/time_test --optimizer aux-sign10-rsclF