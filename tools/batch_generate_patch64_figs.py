#!/usr/bin/env python3
"""Batch generate patch-64 ratio maps and visualizations for cases.

For each case this does:
- Ensure per-patch CSV exists (runs tools/simulate_patch64_energy.py if needed)
- Read per-patch CSV and reconstruct a voxel-wise average ratio map for each (layer,band)
- Save mid-slice and 3-view MIP ratio images to output directory

Usage:
  python3 tools/batch_generate_patch64_figs.py --data_root <dir> --ckpt <ckpt> --max_cases 10
"""
import os, sys, argparse, csv, json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def list_cases(data_root):
    files = [f for f in os.listdir(data_root) if f.endswith('.npy')]
    files.sort()
    return [os.path.splitext(f)[0] for f in files]


def ensure_patch_csv(case, data_root, ckpt, patch=64, stride=32):
    out_dir = Path('tools/patch64_sim')
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f'patch64_sim_{case}.json'
    csv_path = out_dir / f'patch64_sim_{case}.csv'
    if json_path.exists() and csv_path.exists():
        return str(csv_path)
    # run simulate script
    cmd = f'python3 tools/simulate_patch64_energy.py --case {case} --data_root "{data_root}" --ckpt {ckpt} --patch {patch} --stride {stride} --device cpu'
    print('Running:', cmd)
    rc = os.system(cmd)
    if rc != 0:
        raise RuntimeError('simulate_patch64_energy failed for '+case)
    if csv_path.exists():
        return str(csv_path)
    else:
        raise FileNotFoundError('Expected csv not found for '+case)


def reconstruct_ratio_maps(csv_path, vol_shape, patch=64):
    # vol_shape: (C,D,H,W) or (D,H,W) -> handle
    # returns dict keyed by (layer,band) -> ratio_map [D,H,W]
    D = vol_shape[1] if len(vol_shape)==4 else vol_shape[0]
    H = vol_shape[2] if len(vol_shape)==4 else vol_shape[1]
    W = vol_shape[3] if len(vol_shape)==4 else vol_shape[2]
    sums = {}  # (layer,band) -> accum array
    counts = {}
    with open(csv_path, newline='') as cf:
        reader = csv.DictReader(cf)
        for r in reader:
            sd = int(r['sd']); sh = int(r['sh']); sw = int(r['sw'])
            layer = r['layer']; band = int(r['band'])
            ratio = float(r['ratio'])
            key = (layer, band)
            if key not in sums:
                sums[key] = np.zeros((D,H,W), dtype=np.float32)
                counts[key] = np.zeros((D,H,W), dtype=np.uint16)
            sums[key][sd:sd+patch, sh:sh+patch, sw:sw+patch] += ratio
            counts[key][sd:sd+patch, sh:sh+patch, sw:sw+patch] += 1
    maps = {}
    for k in sums:
        c = counts[k]
        with np.errstate(divide='ignore', invalid='ignore'):
            avg = np.zeros_like(sums[k])
            mask = c>0
            avg[mask] = sums[k][mask] / c[mask]
        maps[k] = avg
    return maps


def save_case_maps(case, maps, vol, out_dir):
    # maps: dict (layer,band)->[D,H,W]
    os.makedirs(out_dir, exist_ok=True)
    D = vol.shape[1]
    mid = D//2
    for (layer,band), arr in maps.items():
        # mid slice and MIP
        slice_mid = arr[mid]
        # mips
        axial = arr.max(axis=0)
        coronal = arr.max(axis=1)
        sagittal = arr.max(axis=2)
        # normalize for display
        def norm(a):
            mn, mx = float(a.min()), float(a.max())
            if mx>mn:
                return (a-mn)/(mx-mn)
            return np.zeros_like(a)
        slice_n = norm(slice_mid)
        axial_n, coronal_n, sagittal_n = map(norm, (axial, coronal, sagittal))
        fig, axes = plt.subplots(1,4, figsize=(16,4))
        axes[0].imshow(slice_n, cmap='inferno'); axes[0].set_title(f'{case} {layer} band{band} mid-slice'); axes[0].axis('off')
        axes[1].imshow(axial_n, cmap='inferno'); axes[1].set_title('MIP axial'); axes[1].axis('off')
        axes[2].imshow(coronal_n, cmap='inferno'); axes[2].set_title('MIP coronal'); axes[2].axis('off')
        axes[3].imshow(sagittal_n, cmap='inferno'); axes[3].set_title('MIP sagittal'); axes[3].axis('off')
        plt.tight_layout()
        outp = os.path.join(out_dir, f'patch64_{case}_{layer}_band{band}_map.png')
        fig.savefig(outp, dpi=200)
        plt.close(fig)
    # also save a combined summary per case (all bands in a grid)
    print('Saved maps to', out_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--out_dir', default='paper_figs_patch64')
    parser.add_argument('--max_cases', type=int, default=3)
    parser.add_argument('--patch', type=int, default=64)
    parser.add_argument('--stride', type=int, default=32)
    args = parser.parse_args()

    cases = list_cases(args.data_root)
    if args.max_cases:
        cases = cases[:args.max_cases]
    print('Will process', len(cases), 'cases')

    for case in cases:
        print('Processing', case)
        csv_path = ensure_patch_csv(case, args.data_root, args.ckpt, patch=args.patch, stride=args.stride)
        # load vol to know shape
        p = os.path.join(args.data_root, case + '.npy')
        if not os.path.isfile(p):
            for f in os.listdir(args.data_root):
                if f.startswith(case) and f.endswith('.npy'):
                    p = os.path.join(args.data_root, f); break
        vol = np.load(p)
        if vol.ndim==3:
            vol = vol[None]
        maps = reconstruct_ratio_maps(csv_path, vol.shape, patch=args.patch)
        out_case_dir = os.path.join(args.out_dir, case)
        save_case_maps(case, maps, vol, out_case_dir)

if __name__=='__main__':
    main()
