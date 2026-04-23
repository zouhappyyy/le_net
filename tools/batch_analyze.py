#!/usr/bin/env python3
"""Batch analyze FBM per-band energies across many cases using a trained checkpoint.

Saves:
 - tools/batch_band_energies.csv  (per-case rows)
 - tools/batch_band_stats.json    (aggregated mean/std and optional p-values)

Usage:
  python3 tools/batch_analyze.py --data_root <path> --ckpt <checkpoint> [--max_cases N]

"""
import os, argparse, json, csv
import numpy as np
import torch
from tqdm import tqdm

# Monkeypatch torch.utils.cpp_extension.load to a stub during import to avoid compiling CUDA extensions
_original_cpp_load = None
try:
    import torch.utils.cpp_extension as _cpp_ext
    _original_cpp_load = getattr(_cpp_ext, 'load', None)
    def _fake_load(*args, **kwargs):
        class _Dummy:
            pass
        return _Dummy()
    _cpp_ext.load = _fake_load
except Exception:
    _cpp_ext = None

from nnunet_mednext.network_architecture.le_networks.Double_RWKV_MedNeXt import Double_RWKV_MedNeXt

# restore original load if we replaced it
try:
    if _cpp_ext is not None and _original_cpp_load is not None:
        _cpp_ext.load = _original_cpp_load
except Exception:
    pass

OUT_CSV = 'tools/batch_band_energies.csv'
OUT_JSON = 'tools/batch_band_stats.json'


def list_cases(data_root):
    files = [f for f in os.listdir(data_root) if f.endswith('.npy')]
    files.sort()
    cases = [os.path.splitext(f)[0] for f in files]
    return cases


def load_case(data_root, case_id):
    p = os.path.join(data_root, case_id + '.npy')
    if not os.path.isfile(p):
        # fallback: pick first starting with case_id
        for f in os.listdir(data_root):
            if f.startswith(case_id) and f.endswith('.npy'):
                p = os.path.join(data_root, f)
                break
    if not os.path.isfile(p):
        raise FileNotFoundError(p)
    arr = np.load(p)
    if arr.ndim == 3:
        arr = arr[None]
    return arr.astype(np.float32)


def compute_energy(arr):
    return float((arr**2).sum())


def analyze_case(model, vol, device, fbm_k_list):
    # vol: np [C,D,H,W]
    in_ch = vol.shape[0]
    x = torch.from_numpy(vol[None]).float().to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        # Use mednext_enc.forward_encoder to avoid running RWKV (and its CUDA ops)
        try:
            feats = model.encoder.mednext_enc.forward_encoder(x)
        except Exception:
            # fallback to full encoder if mednext_enc not available
            feats = model.encoder(x)
    results = []
    enc = model.encoder
    for name_idx, name in enumerate(['f0','f1']):
        feat = feats[name_idx]
        if name == 'f0' and hasattr(enc, 'fbm0'):
            fbm = enc.fbm0
        elif name == 'f1' and hasattr(enc, 'fbm1'):
            fbm = enc.fbm1
        else:
            # no fbm
            continue
        t = feat
        b,c,d,h,w = t.shape
        x_fft = torch.fft.rfftn(t, s=(d,h,w), dim=(-3,-2,-1), norm='ortho')
        band_masks = fbm._get_band_masks(d, h, w//2 + 1)
        pre_x = t.clone()
        att_feat = t
        for idx in range(len(band_masks)):
            mask = band_masks[idx].to(device)
            low_part = torch.fft.irfftn(x_fft * mask, s=(d,h,w), dim=(-3,-2,-1), norm='ortho')
            high_part = pre_x - low_part
            pre_x = low_part
            raw_np = high_part.detach().cpu().numpy()
            # weighted
            if idx < len(fbm.freq_weight_conv_list):
                fw = fbm.freq_weight_conv_list[idx](att_feat)
                fw = fbm._activate(fw)
                grp = fbm.spatial_group
                tmp = fw.reshape(b, grp, -1, d, h, w) * high_part.reshape(b, grp, -1, d, h, w)
                weighted_np = tmp.detach().cpu().numpy()
            else:
                weighted_np = high_part.detach().cpu().numpy()
            # reduce to mean over group channels then compute energy
            raw_mean = raw_np.mean(axis=1)[0]
            weighted_mean = weighted_np.mean(axis=1)[0]
            e_raw = compute_energy(raw_mean)
            e_w = compute_energy(weighted_mean)
            results.append({'layer': name, 'band': idx+1, 'raw_energy': e_raw, 'weighted_energy': e_w})
    return results


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--max_cases', type=int, default=None)
    parser.add_argument('--fbm_k_list', nargs='+', type=int, default=[2,4,8])
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args(argv)

    device = 'cuda' if (args.device is None and torch.cuda.is_available()) or args.device == 'cuda' else 'cpu'

    cases = list_cases(args.data_root)
    if args.max_cases:
        cases = cases[:args.max_cases]
    print(f'Found {len(cases)} cases, analyzing {len(cases)} on {device}')

    all_rows = []
    per_band_values = {}  # key: (layer,band) -> list of ratios

    # load checkpoint once
    data = torch.load(args.ckpt, map_location='cpu')
    st = data.get('model_state', data)

    for case in tqdm(cases, desc='cases'):
        try:
            vol = load_case(args.data_root, case)
        except Exception as e:
            print('skip', case, 'error', e)
            continue
        # build a model with appropriate input channels for this case to avoid size mismatch
        in_ch = vol.shape[0]
        model = Double_RWKV_MedNeXt(in_channels=in_ch, n_channels=16, n_classes=2, exp_r=2, kernel_size=3, fbm_k_list=args.fbm_k_list, fbm_learnable=True)
        # load checkpoint permissively (strict=False) so missing/unexpected keys are tolerated
        try:
            missing = model.load_state_dict(st, strict=False)
        except Exception as e:
            print(f'Failed to load checkpoint into model for case {case}:', e)
            continue
        res = analyze_case(model, vol, device, args.fbm_k_list)
        for r in res:
            ratio = r['weighted_energy'] / (r['raw_energy'] + 1e-12)
            all_rows.append({'case': case, **r, 'ratio': ratio})
            key = (r['layer'], r['band'])
            per_band_values.setdefault(key, []).append(ratio)

    # save per-case CSV
    with open(OUT_CSV, 'w', newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=['case','layer','band','raw_energy','weighted_energy','ratio'])
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    # aggregate stats
    stats = {}
    for k, vals in per_band_values.items():
        arr = np.array(vals)
        stats_key = f'{k[0]}_band{k[1]}'
        stats[stats_key] = {'n': int(arr.size), 'mean_ratio': float(arr.mean()), 'std_ratio': float(arr.std(ddof=1) if arr.size>1 else 0.0)}
    # try t-test paired? not applicable here since baseline absent; we just provide mean/std
    with open(OUT_JSON, 'w') as jf:
        json.dump({'stats': stats, 'cases_analyzed': len(cases)}, jf, indent=2)

    print('Saved', OUT_CSV, OUT_JSON)

if __name__ == '__main__':
    main()
