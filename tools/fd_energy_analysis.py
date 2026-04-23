#!/usr/bin/env python3
"""Frequency-band energy analysis and band-removal experiment for FBM.

Produces:
- CSV with per-band energy (raw high residual and FBM-weighted contributions)
- Bar plot of band energies
- Reconstructions with specific bands removed and corresponding MIP images
- Simple edge-energy (mean gradient magnitude) metric for each reconstruction

Example:
python3 tools/fd_energy_analysis.py \
  --data_root /home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_preprocessed/Task530_EsoTJ_30pct/nnUNetData_plans_v2.1_trgSp_1x1x1_stage0 \
  --case_id ESO_TJ_60011222468 --channel 0 --save_prefix fd_energy --learnable_bands --soft_band_beta 30
"""
import argparse
import os
import csv
import json
from typing import List

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nnunet_mednext.network_architecture.le_networks.FDConv_3d import FrequencyBandModulation3D
# reuse loader from visualize script if available
try:
    from visualize_fdconv_highfreq import _load_case_volume_from_npy
except Exception:
    # fallback: local loader
    def _load_case_volume_from_npy(data_root: str, case_id: str) -> np.ndarray:
        if not os.path.isdir(data_root):
            raise FileNotFoundError(f"data_root {data_root} does not exist")
        exact_npy = os.path.join(data_root, f"{case_id}.npy")
        if os.path.isfile(exact_npy):
            npy_path = exact_npy
        else:
            candidates = [os.path.join(data_root, f) for f in os.listdir(data_root) if f.startswith(case_id) and f.endswith('.npy')]
            if not candidates:
                raise FileNotFoundError(f"No .npy for {case_id} in {data_root}")
            npy_path = candidates[0]
        arr = np.load(npy_path)
        if arr.ndim == 3:
            arr = arr[None]
        return arr.astype(np.float32)


def compute_energy(arr: np.ndarray) -> float:
    # energy as sum of squared values
    return float(np.sum(np.square(arr)))


def mean_gradient_magnitude(vol: np.ndarray) -> float:
    # vol: [D,H,W]
    gx = np.gradient(vol.astype(np.float32), axis=2)
    gy = np.gradient(vol.astype(np.float32), axis=1)
    gz = np.gradient(vol.astype(np.float32), axis=0)
    mag = np.sqrt(np.square(gx) + np.square(gy) + np.square(gz))
    return float(np.mean(np.abs(mag)))


def save_mip_png(vol: np.ndarray, out_path: str, title: str = None):
    # vol: [D,H,W]
    axial = vol.max(axis=0)
    coronal = vol.max(axis=1)
    sagittal = vol.max(axis=2)
    def norm(a):
        a = a.astype(np.float32)
        vmin, vmax = float(a.min()), float(a.max())
        if vmax > vmin:
            return (a - vmin) / (vmax - vmin)
        return np.zeros_like(a)
    axial, coronal, sagittal = map(norm, (axial, coronal, sagittal))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(axial, cmap='gray'); axes[0].set_title('axial'); axes[0].axis('off')
    axes[1].imshow(coronal, cmap='gray'); axes[1].set_title('coronal'); axes[1].axis('off')
    axes[2].imshow(sagittal, cmap='gray'); axes[2].set_title('sagittal'); axes[2].axis('off')
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--case_id', type=str, required=True)
    parser.add_argument('--channel', type=int, default=0)
    parser.add_argument('--k_list', type=int, nargs='+', default=[2,4,8])
    parser.add_argument('--learnable_bands', action='store_true')
    parser.add_argument('--soft_band_beta', type=float, default=30.0)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--save_prefix', type=str, default='fd_energy')
    parser.add_argument('--output_dir', type=str, default='.')
    parser.add_argument('--remove_bands', type=int, nargs='*', default=None, help='list of 1-based band indices to remove (e.g. 3 removes band3)')
    args = parser.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)

    print('fd_energy_analysis started with args:', args)

    # load volume
    print('Loading volume from', args.data_root, 'case', args.case_id)
    vol = _load_case_volume_from_npy(args.data_root, args.case_id)  # [C,D,H,W]
    print('Loaded volume shape:', getattr(vol, 'shape', None))

    if args.channel < 0 or args.channel >= vol.shape[0]:
        raise ValueError('channel out of range')
    vol_c = vol[args.channel:args.channel+1]
    B = 1
    x = torch.from_numpy(vol_c[None]).float()  # [1,1,D,H,W]
    device = 'cuda' if (args.device is None and torch.cuda.is_available()) or args.device == 'cuda' else 'cpu'
    x = x.to(device)

    fbm = FrequencyBandModulation3D(in_channels=vol_c.shape[0], k_list=args.k_list, learnable_bands=args.learnable_bands, soft_band_beta=args.soft_band_beta).to(device)
    print('FBM constructed. Device:', device)
    with torch.no_grad():
        out, high_acc = fbm(x, att_feat=None, return_high=True)
    print('FBM forward done; high_acc shape:', getattr(high_acc, 'shape', None))

    low_final = (x - high_acc).detach().cpu().numpy()  # [B,C,D,H,W]
    b, c, d, h, w = x.shape

    # compute per-band raw high and weighted contributions (same as visualization script)
    x_fft = torch.fft.rfftn(x, s=(d,h,w), dim=(-3,-2,-1), norm='ortho')
    band_masks = fbm._get_band_masks(d, h, w // 2 + 1)

    per_band_raw = []
    per_band_weighted = []
    pre_x = x.clone()
    att_feat = x
    for idx in range(len(band_masks)):
        mask = band_masks[idx].to(device)
        low_part = torch.fft.irfftn(x_fft * mask, s=(d,h,w), dim=(-3,-2,-1), norm='ortho')
        high_part = pre_x - low_part
        pre_x = low_part
        per_band_raw.append(high_part.detach().cpu().numpy())
        if idx < len(fbm.freq_weight_conv_list):
            freq_weight = fbm.freq_weight_conv_list[idx](att_feat)
            freq_weight = fbm._activate(freq_weight)
            group = fbm.spatial_group
            tmp = freq_weight.reshape(b, group, -1, d, h, w) * high_part.reshape(b, group, -1, d, h, w)
            band_weighted = tmp.reshape(b, -1, d, h, w)
            per_band_weighted.append(band_weighted.detach().cpu().numpy())
        else:
            per_band_weighted.append(high_part.detach().cpu().numpy())

    # compute energies
    band_indices = list(range(1, len(per_band_raw)+1))
    rows = []
    energies_raw = []
    energies_weighted = []
    for i, (r, wv) in enumerate(zip(per_band_raw, per_band_weighted), start=1):
        # r/wv shape [B, Cb, D, H, W]
        r_mean = r.mean(axis=1)[0]  # [D,H,W]
        w_mean = wv.mean(axis=1)[0]
        e_r = compute_energy(r_mean)
        e_w = compute_energy(w_mean)
        energies_raw.append(e_r)
        energies_weighted.append(e_w)
        rows.append({'band': i, 'energy_raw': e_r, 'energy_weighted': e_w})

    low_energy = compute_energy(low_final.mean(axis=1)[0])

    # save CSV
    csv_path = os.path.join(args.output_dir, f"{args.save_prefix}_{args.case_id}_band_energy.csv")
    with open(csv_path, 'w', newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=['band','energy_raw','energy_weighted'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print('Saved band energy CSV to', csv_path)
    print('Energies raw:', energies_raw)
    print('Energies weighted:', energies_weighted)

    # plot energies
    plt.figure(figsize=(8,4))
    xlocs = np.arange(len(band_indices))
    plt.bar(xlocs - 0.15, energies_raw, width=0.3, label='raw_high')
    plt.bar(xlocs + 0.15, energies_weighted, width=0.3, label='weighted_high')
    plt.xticks(xlocs, [str(i) for i in band_indices])
    plt.xlabel('band (1=lowest high band)')
    plt.ylabel('energy (sum squares)')
    plt.legend()
    plt.title(f'Per-band energy for {args.case_id} (low_energy={low_energy:.2e})')
    plot_path = os.path.join(args.output_dir, f"{args.save_prefix}_{args.case_id}_band_energy.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print('Saved energy plot to', plot_path)
    print('Saved individual band MIPs:')

    # save individual band MIPs for inspection
    for i, (r, wv) in enumerate(zip(per_band_raw, per_band_weighted), start=1):
        r_mean = r.mean(axis=1)[0]
        w_mean = wv.mean(axis=1)[0]
        save_mip_png(r_mean, os.path.join(args.output_dir, f"{args.save_prefix}_{args.case_id}_band{i}_raw_mip.png"), title=f'band{i}_raw')
        save_mip_png(w_mean, os.path.join(args.output_dir, f"{args.save_prefix}_{args.case_id}_band{i}_weighted_mip.png"), title=f'band{i}_weighted')
        print(' - band', i)

    # Band-removal reconstructions
    recon_metrics = []
    total_weighted = sum([wv for wv in per_band_weighted])  # list of arrays
    total_weighted = np.sum(np.stack(per_band_weighted, axis=0), axis=0)  # [num_bands, B, Cb, D, H, W] -> sum -> [B, C, D, H, W]
    # note: per_band_weighted elements have shape [B, Cb, D, H, W]
    total_weighted_sum = np.sum(np.stack(per_band_weighted, axis=0), axis=0)[0]  # [C,D,H,W]

    low_arr = low_final[0].mean(axis=0) if low_final.shape[0] == 1 else low_final.mean(axis=0)[0]

    def reconstruct_excluding(exclude: List[int]):
        # exclude: 1-based indices
        included = [i for i in range(1, len(per_band_weighted)+1) if i not in exclude]
        parts = [per_band_weighted[i-1] for i in included]
        if len(parts) == 0:
            sum_parts = np.zeros_like(per_band_weighted[0])
        else:
            sum_parts = np.sum(np.stack(parts, axis=0), axis=0)
        recon = low_final + sum_parts  # [B,C,D,H,W]
        recon_mean = recon.mean(axis=1)[0]
        return recon_mean

    # default: baseline (no removal)
    baseline = (low_final + total_weighted_sum[None]).mean(axis=1)[0]
    baseline_edge = mean_gradient_magnitude(baseline)
    recon_metrics.append({'removed': [], 'edge_mean': baseline_edge})

    # if user provided remove_bands, test them individually and all-high removal
    remove_list = args.remove_bands if args.remove_bands is not None else []
    # also test removing each band individually
    for i in range(1, len(per_band_weighted)+1):
        recon = reconstruct_excluding([i])
        e = mean_gradient_magnitude(recon)
        out_png = os.path.join(args.output_dir, f"{args.save_prefix}_{args.case_id}_recon_exclude_band{i}.png")
        save_mip_png(recon, out_png, title=f'exclude band {i}')
        recon_metrics.append({'removed':[i], 'edge_mean': e, 'out_png': out_png})

    # test removing user-specified list if provided
    if len(remove_list) > 0:
        recon = reconstruct_excluding(remove_list)
        e = mean_gradient_magnitude(recon)
        out_png = os.path.join(args.output_dir, f"{args.save_prefix}_{args.case_id}_recon_exclude_{'_'.join(map(str,remove_list))}.png")
        save_mip_png(recon, out_png, title=f'exclude {remove_list}')
        recon_metrics.append({'removed': remove_list, 'edge_mean': e, 'out_png': out_png})

    # save metrics JSON
    json_path = os.path.join(args.output_dir, f"{args.save_prefix}_{args.case_id}_recon_metrics.json")
    with open(json_path, 'w') as jf:
        json.dump({'band_energy_rows': rows, 'low_energy': low_energy, 'recon_metrics': recon_metrics}, jf, indent=2)
    print('Saved recon metrics to', json_path)
    print('fd_energy_analysis completed successfully')


if __name__ == '__main__':
    main()
