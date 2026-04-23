#!/usr/bin/env python3
"""Reproduce large-ratio results and generate annotated comparison figures.

- Reads tools/batch_band_energies.csv to pick cases with large f1 band ratios.
- For each selected case, loads full volume, runs model.encoder (mednext_enc) to get feats,
  applies FBM on f0/f1, computes per-band raw/weighted energies, and generates
  annotated figures with numeric ratios (saved to paper_figs_reproduced/).

Usage:
  python3 tools/reproduce_and_plot.py --ckpt tools/fd_model_ckpt_learnable.pth --topk 3
"""
import os, sys, argparse, csv, json
import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# avoid CUDA extension compile at import time
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

CSV = 'tools/batch_band_energies.csv'
DATA_ROOT_DEFAULT = '/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_preprocessed/Task530_EsoTJ_30pct/nnUNetData_plans_v2.1_trgSp_1x1x1_stage0'
OUT_DIR_DEFAULT = 'paper_figs_reproduced'


def read_batch_csv():
    rows = []
    with open(CSV, newline='') as cf:
        r = csv.DictReader(cf)
        for row in r:
            row['raw_energy'] = float(row['raw_energy'])
            row['weighted_energy'] = float(row['weighted_energy'])
            row['ratio'] = float(row['ratio'])
            rows.append(row)
    return rows


def pick_top_cases(rows, topk=3, layer='f1', band=3):
    # select cases ordered by ratio for given layer/band
    filtered = [r for r in rows if r['layer']==layer and int(r['band'])==band]
    filtered.sort(key=lambda x: x['ratio'], reverse=True)
    return [r['case'] for r in filtered[:topk]]


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


def compute_per_band_from_feat(feat, fbm):
    b,c,d,h,w = feat.shape
    x_fft = torch.fft.rfftn(feat, s=(d,h,w), dim=(-3,-2,-1), norm='ortho')
    band_masks = fbm._get_band_masks(d, h, w//2 + 1)
    pre_x = feat.clone()
    att_feat = feat
    per_raw = []
    per_w = []
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
        raw_mean = raw_np.mean(axis=1)[0]
        w_mean = weighted_np.mean(axis=1)[0]
        per_raw.append(raw_mean)
        per_w.append(w_mean)
    return per_raw, per_w


def make_annotated_figure(case, layer, band_idx, raw_mean, w_mean, input_vol, stats, out_path):
    # Coerce inputs to numpy arrays and collapse extra channel dimensions by mean
    raw_mean = np.asarray(raw_mean)
    w_mean = np.asarray(w_mean)
    # If arrays have more than 3 dims, try to collapse leading/channel dims by mean
    if raw_mean.ndim > 3:
        # collapse all leading dims until 3D
        while raw_mean.ndim > 3:
            raw_mean = raw_mean.mean(axis=0)
    if w_mean.ndim > 3:
        while w_mean.ndim > 3:
            w_mean = w_mean.mean(axis=0)

    rd, rh, rw = raw_mean.shape
    wd, wh, ww = w_mean.shape
    # log shapes
    print(f"make_annotated_figure: case={case} layer={layer} band={band_idx} raw_shape={raw_mean.shape} w_shape={w_mean.shape}")

    mid = rw // 2
    raw_slice = raw_mean[:, :, mid]
    w_slice = w_mean[:, :, mid]
    diff = w_slice - raw_slice
    # normalize slices for display
    def norm(a):
        a = a.astype(np.float32)
        mn, mx = a.min(), a.max()
        return (a-mn)/(mx-mn) if mx>mn else np.zeros_like(a)
    rs = norm(raw_slice); ws = norm(w_slice)
    # ratio map
    ratio_map = w_mean/(raw_mean + 1e-12)
    RMAX = max(2.0, np.percentile(ratio_map, 99))
    ratio_disp = np.clip(ratio_map, 0, RMAX)/RMAX

    fig, axes = plt.subplots(3,4, figsize=(16,12))
    axes[0,0].imshow(rs, cmap='gray'); axes[0,0].set_title('Raw slice'); axes[0,0].axis('off')
    axes[0,1].imshow(ws, cmap='gray'); axes[0,1].set_title('Weighted slice'); axes[0,1].axis('off')
    axes[0,2].imshow(diff, cmap='RdBu', vmin=-np.percentile(np.abs(diff),99), vmax=np.percentile(np.abs(diff),99)); axes[0,2].set_title('Diff'); axes[0,2].axis('off')
    axes[0,3].axis('off')

    # MIPs
    axial_raw = raw_mean.max(axis=0); cor_raw = raw_mean.max(axis=1); sag_raw = raw_mean.max(axis=2)
    axial_w = w_mean.max(axis=0); cor_w = w_mean.max(axis=1); sag_w = w_mean.max(axis=2)
    axes[1,0].imshow(norm(axial_raw), cmap='gray'); axes[1,0].set_title('Raw axial'); axes[1,0].axis('off')
    axes[1,1].imshow(norm(axial_w), cmap='gray'); axes[1,1].set_title('Weighted axial'); axes[1,1].axis('off')
    im = axes[1,2].imshow((axial_w-axial_raw)/ (np.percentile(np.abs(axial_w-axial_raw),99)+1e-12), cmap='RdBu'); axes[1,2].set_title('Axial diff'); axes[1,2].axis('off')
    axes[1,3].imshow(np.clip(axial_w/(axial_raw+1e-12),0, RMAX)/RMAX, cmap='inferno'); axes[1,3].set_title('Axial ratio'); axes[1,3].axis('off')

    # hist & scatter
    flat_raw = raw_slice.flatten(); flat_w = w_slice.flatten()
    axes[2,0].hist(flat_raw**2+1e-12, bins=200, alpha=0.6, label='raw'); axes[2,0].hist(flat_w**2+1e-12, bins=200, alpha=0.6, label='weighted'); axes[2,0].set_yscale('log'); axes[2,0].legend(); axes[2,0].set_title('Per-voxel energy (log)')
    idxs = np.random.choice(flat_raw.size, min(5000, flat_raw.size), replace=False)
    axes[2,1].scatter(flat_raw[idxs], flat_w[idxs], s=1, alpha=0.3); axes[2,1].set_title('raw vs weighted')
    # annotation
    txt = f'Case: {case}\nLayer: {layer} Band: {band_idx}\nRawE={stats["raw_energy"]:.2f} WgtE={stats["weighted_energy"]:.2f} Ratio={stats["ratio"]:.2f}'
    axes[2,2].axis('off'); axes[2,2].text(0,0.5,txt, fontsize=12)
    axes[2,3].axis('off')
    plt.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--data_root', default=DATA_ROOT_DEFAULT)
    parser.add_argument('--out_dir', default=OUT_DIR_DEFAULT)
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = read_batch_csv()
    cases = pick_top_cases(rows, topk=args.topk, layer='f1', band=3)
    print('Selected cases:', cases)

    # load checkpoint
    data = torch.load(args.ckpt, map_location='cpu') if os.path.isfile(args.ckpt) else None
    st = data.get('model_state', data) if data is not None else None

    summary = []
    for case in cases:
        print('Processing', case)
        vol = load_case(args.data_root, case)
        in_ch = vol.shape[0]
        model = Double_RWKV_MedNeXt(in_channels=in_ch, n_channels=16, n_classes=2, exp_r=2, kernel_size=3, fbm_k_list=[2,4,8], fbm_learnable=True)
        if st is not None:
            model.load_state_dict(st, strict=False)
        model.to(args.device); model.eval()
        x = torch.from_numpy(vol[None]).float().to(args.device)
        with torch.no_grad():
            try:
                feats_med = model.encoder.mednext_enc.forward_encoder(x)
            except Exception:
                feats_med = model.encoder(x)
        enc = model.encoder
        for li, layer in enumerate(['f0','f1']):
            feat = feats_med[li]
            fbm = getattr(enc, 'fbm0' if layer=='f0' else 'fbm1', None)
            if fbm is None:
                continue
            per_raw, per_w = compute_per_band_from_feat(feat, fbm)
            for bi,(r,wv) in enumerate(zip(per_raw, per_w), start=1):
                e_raw = float((r**2).sum()); e_w = float((wv**2).sum()); ratio = e_w/(e_raw+1e-12)
                # lookup batch csv stat for this case/layer/band
                stat_row = next((rr for rr in rows if rr['case']==case and rr['layer']==layer and int(rr['band'])==bi), None)
                stats = stat_row if stat_row is not None else {'raw_energy': e_raw, 'weighted_energy': e_w, 'ratio': ratio}
                outp = os.path.join(args.out_dir, f'repro_{case}_{layer}_band{bi}.png')
                make_annotated_figure(case, layer, bi, r, wv, vol, stats, outp)
                print('Saved', outp, f'computed_ratio={ratio:.2f}', f'csv_ratio={stats["ratio"]:.2f}')
                summary.append({'case':case, 'layer':layer, 'band':bi, 'computed_ratio':ratio, 'csv_ratio':stats['ratio']})
    # save summary
    with open(os.path.join(args.out_dir, 'repro_summary.json'), 'w') as jf:
        json.dump(summary, jf, indent=2)
    print('Done. Summary saved to', os.path.join(args.out_dir, 'repro_summary.json'))

if __name__=='__main__':
    main()
