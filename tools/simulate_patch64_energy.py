#!/usr/bin/env python3
"""Simulate patch-based (64^3) FBM energy by running model encoder on overlapping patches.

Outputs per-patch per-band energies and aggregated stats (mean/std/total) saved to JSON/CSV.
"""
import os, sys, argparse, json, csv
import numpy as np
import torch

# monkeypatch torch.utils.cpp_extension.load to avoid CUDA compile during imports
_original_cpp_load = None
try:
    import torch.utils.cpp_extension as _cpp_ext
    _original_cpp_load = getattr(_cpp_ext, 'load', None)
    def _fake_load(*args, **kwargs):
        class _Dummy: pass
        return _Dummy()
    _cpp_ext.load = _fake_load
except Exception:
    _cpp_ext = None

from nnunet_mednext.network_architecture.le_networks.Double_RWKV_MedNeXt import Double_RWKV_MedNeXt

# restore
try:
    if _cpp_ext is not None and _original_cpp_load is not None:
        _cpp_ext.load = _original_cpp_load
except Exception:
    pass


def load_case(data_root, case_id):
    p = os.path.join(data_root, case_id + '.npy')
    if not os.path.isfile(p):
        for f in os.listdir(data_root):
            if f.startswith(case_id) and f.endswith('.npy'):
                p = os.path.join(data_root, f)
                break
    arr = np.load(p)
    if arr.ndim == 3:
        arr = arr[None]
    return arr.astype(np.float32)


def get_starts(L, patch, stride):
    starts = list(range(0, L - patch + 1, stride))
    if len(starts) == 0:
        return [0]
    if starts[-1] != L - patch:
        starts.append(L - patch)
    return starts


def compute_per_band_from_feat(feat, fbm):
    # feat: torch [B,C,D,H,W]
    b,c,d,h,w = feat.shape
    x_fft = torch.fft.rfftn(feat, s=(d,h,w), dim=(-3,-2,-1), norm='ortho')
    band_masks = fbm._get_band_masks(d, h, w//2 + 1)
    pre_x = feat.clone()
    att_feat = feat
    per_raw = []
    per_weighted = []
    for idx, mask in enumerate(band_masks):
        mask = mask.to(feat.device)
        low_part = torch.fft.irfftn(x_fft * mask, s=(d,h,w), dim=(-3,-2,-1), norm='ortho')
        high_part = pre_x - low_part
        pre_x = low_part
        raw_np = high_part.detach().cpu().numpy()
        if idx < len(fbm.freq_weight_conv_list):
            fw = fbm.freq_weight_conv_list[idx](att_feat)
            fw = fbm._activate(fw)
            grp = fbm.spatial_group
            tmp = fw.reshape(b, grp, -1, d, h, w) * high_part.reshape(b, grp, -1, d, h, w)
            weighted_np = tmp.detach().cpu().numpy()
        else:
            weighted_np = high_part.detach().cpu().numpy()
        # reduce mean over channel groups
        raw_mean = raw_np.mean(axis=1)[0]
        weighted_mean = weighted_np.mean(axis=1)[0]
        per_raw.append(raw_mean)
        per_weighted.append(weighted_mean)
    return per_raw, per_weighted


def energy(arr):
    return float((arr**2).sum())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_preprocessed/Task530_EsoTJ_30pct/nnUNetData_plans_v2.1_trgSp_1x1x1_stage0')
    parser.add_argument('--case', default='ESO_TJ_1010018171')
    parser.add_argument('--ckpt', default='tools/fd_model_ckpt_learnable.pth')
    parser.add_argument('--patch', type=int, default=64)
    parser.add_argument('--stride', type=int, default=32)
    parser.add_argument('--k_list', nargs='+', type=int, default=[2,4,8])
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--out_dir', default='tools/patch64_sim')
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)

    vol = load_case(args.data_root, args.case)
    print('Loaded', args.case, 'shape', vol.shape)
    C,D,H,W = vol.shape
    starts_d = get_starts(D, args.patch, args.stride)
    starts_h = get_starts(H, args.patch, args.stride)
    starts_w = get_starts(W, args.patch, args.stride)
    coords = [(sd, sh, sw) for sd in starts_d for sh in starts_h for sw in starts_w]
    print(f'Generating {len(coords)} patches (patch={args.patch}, stride={args.stride})')

    # load model
    data = torch.load(args.ckpt, map_location='cpu') if os.path.isfile(args.ckpt) else None
    st = data.get('model_state', data) if data is not None else None
    model = Double_RWKV_MedNeXt(in_channels=C, n_channels=16, n_classes=2, exp_r=2, kernel_size=3, fbm_k_list=args.k_list, fbm_learnable=True)
    if st is not None:
        model.load_state_dict(st, strict=False)
    model.to(args.device)
    model.eval()

    enc = model.encoder
    # ensure fbm0/fbm1 exist
    if not hasattr(enc, 'fbm0'):
        print('Encoder has no fbm0; abort'); return

    per_patch_records = []
    for i,(sd,sh,sw) in enumerate(coords):
        patch = vol[:, sd:sd+args.patch, sh:sh+args.patch, sw:sw+args.patch]
        x = torch.from_numpy(patch[None]).float().to(args.device)
        with torch.no_grad():
            try:
                feats = model.encoder.mednext_enc.forward_encoder(x)
            except Exception:
                feats = model.encoder(x)
        for li, layer in enumerate(['f0','f1']):
            feat = feats[li]
            fbm = getattr(enc, 'fbm0' if layer=='f0' else 'fbm1')
            per_raw, per_w = compute_per_band_from_feat(feat, fbm)
            # compute energies per band
            for bi,(r,wv) in enumerate(zip(per_raw, per_w), start=1):
                e_raw = energy(r)
                e_w = energy(wv)
                per_patch_records.append({'patch_idx': i, 'sd': sd, 'sh': sh, 'sw': sw, 'layer': layer, 'band': bi, 'raw_energy': e_raw, 'weighted_energy': e_w, 'ratio': e_w/(e_raw+1e-12)})
        if (i+1) % 10 == 0:
            print('Processed', i+1, 'patches')

    # aggregate per (layer,band)
    agg = {}
    for rec in per_patch_records:
        key = (rec['layer'], rec['band'])
        agg.setdefault(key, []).append(rec)
    stats = {}
    for k, items in agg.items():
        ratios = np.array([it['ratio'] for it in items])
        e_raws = np.array([it['raw_energy'] for it in items])
        e_ws = np.array([it['weighted_energy'] for it in items])
        stats[f'{k[0]}_band{k[1]}'] = {'n_patches': int(len(items)), 'mean_ratio': float(ratios.mean()), 'std_ratio': float(ratios.std(ddof=1) if len(ratios)>1 else 0.0), 'sum_weighted_energy': float(e_ws.sum()), 'sum_raw_energy': float(e_raws.sum())}

    out_json = os.path.join(args.out_dir, f'patch64_sim_{args.case}.json')
    with open(out_json, 'w') as jf:
        json.dump({'case': args.case, 'patch': args.patch, 'stride': args.stride, 'coords_len': len(coords), 'per_patch': per_patch_records, 'stats': stats}, jf, indent=2)

    out_csv = os.path.join(args.out_dir, f'patch64_sim_{args.case}.csv')
    with open(out_csv, 'w', newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=['patch_idx','sd','sh','sw','layer','band','raw_energy','weighted_energy','ratio'])
        writer.writeheader()
        for r in per_patch_records:
            writer.writerow(r)

    print('Saved', out_json, out_csv)
    # print summary
    for k,v in stats.items():
        print(k, v)

if __name__=='__main__':
    main()
