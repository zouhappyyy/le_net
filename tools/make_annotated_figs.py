#!/usr/bin/env python3
"""Create annotated publication-ready figures that show raw vs weighted per-band comparisons
using the trained model and batch CSV numbers.

For each selected case and layer/band, the figure contains:
 - Input mid-slice, low/high mid-slice
 - Band raw slice, weighted slice, diff (RdBu), ratio map (clipped)
 - Histogram of per-voxel energy (squared) for raw and weighted (log scale)
 - Scatter sample raw vs weighted
 - Text annotation with energy numbers and ratio (from CSV)

Usage:
  python3 tools/make_annotated_figs.py --cases ESO_TJ_1010018171 ESO_TJ_1010214968 --ckpt tools/fd_model_ckpt_learnable.pth --out_dir paper_figs_final

If --cases omitted, uses first 3 cases from the CSV.
"""
import os, sys, argparse, csv
import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# monkeypatch to avoid CUDA extension build during imports
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

# restore
try:
    if _cpp_ext is not None and _original_cpp_load is not None:
        _cpp_ext.load = _original_cpp_load
except Exception:
    pass

CSV = 'tools/batch_band_energies.csv'
DATA_ROOT_DEFAULT = '/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_preprocessed/Task530_EsoTJ_30pct/nnUNetData_plans_v2.1_trgSp_1x1x1_stage0'


def read_csv():
    rows = []
    with open(CSV, newline='') as cf:
        r = csv.DictReader(cf)
        for row in r:
            row['raw_energy'] = float(row['raw_energy'])
            row['weighted_energy'] = float(row['weighted_energy'])
            row['ratio'] = float(row['ratio'])
            rows.append(row)
    return rows


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


def make_figure(case_id, layer, band_idx, raw_mean, w_mean, input_vol, out_path, stats):
    # raw_mean, w_mean: [D,H,W]
    mid = raw_mean.shape[0]//2
    raw_slice = raw_mean[mid]
    w_slice = w_mean[mid]
    diff_slice = w_slice - raw_slice
    # ratio map
    ratio_map = w_mean/(raw_mean + 1e-12)
    RMAX = max(2.0, np.percentile(ratio_map, 99))
    ratio_disp = np.clip(ratio_map, 0, RMAX)/RMAX

    # energy hist (squared values) log
    flat_raw_energy = (raw_mean**2).flatten()
    flat_w_energy = (w_mean**2).flatten()

    # sample scatter
    N = flat_raw_energy.size
    idxs = np.random.choice(N, min(5000, N), replace=False)
    sraw = raw_mean.flatten()[idxs]
    sw = w_mean.flatten()[idxs]

    fig = plt.figure(figsize=(14,10))
    gs = fig.add_gridspec(3,4)
    ax_in = fig.add_subplot(gs[0,0])
    ax_low = fig.add_subplot(gs[0,1])
    ax_high = fig.add_subplot(gs[0,2])
    ax_text = fig.add_subplot(gs[0,3])

    # input / low / high mid slices from input_vol (assume input_vol [C,D,H,W])
    inp_slice = input_vol[0, mid]
    ax_in.imshow((inp_slice - inp_slice.min())/(inp_slice.max()-inp_slice.min()+1e-12), cmap='gray'); ax_in.set_title('Input mid-slice'); ax_in.axis('off')
    # low/high approximated as raw_mean aggregated low/high (not available here), so reuse raw/w
    ax_low.imshow((raw_slice - raw_slice.min())/(raw_slice.max()-raw_slice.min()+1e-12), cmap='gray'); ax_low.set_title('Band raw (mid)'); ax_low.axis('off')
    ax_high.imshow((w_slice - w_slice.min())/(w_slice.max()-w_slice.min()+1e-12), cmap='gray'); ax_high.set_title('Band weighted (mid)'); ax_high.axis('off')

    # annotation text
    ax_text.axis('off')
    txt = f'Case: {case_id}\nLayer: {layer}  Band: {band_idx}\nRaw E: {stats["raw_energy"]:.1f}\nWeighted E: {stats["weighted_energy"]:.1f}\nRatio: {stats["ratio"]:.2f}\n\nMean raw: {raw_mean.mean():.4f}\nMean weighted: {w_mean.mean():.4f}'
    ax_text.text(0,0.5,txt, fontsize=12, va='center')

    # large panels: diff and ratio
    ax_diff = fig.add_subplot(gs[1:,0:2])
    ax_ratio = fig.add_subplot(gs[1,2])
    ax_hist = fig.add_subplot(gs[1,3])
    ax_scatter = fig.add_subplot(gs[2,2:])

    im = ax_diff.imshow(diff_slice, cmap='RdBu', vmin=-np.percentile(np.abs(diff_slice),99), vmax=np.percentile(np.abs(diff_slice),99))
    ax_diff.set_title('Weighted - Raw (mid slice)'); ax_diff.axis('off'); fig.colorbar(im, ax=ax_diff, fraction=0.046)

    ax_ratio.imshow(ratio_disp[mid], cmap='inferno'); ax_ratio.set_title('Ratio map (clipped)'); ax_ratio.axis('off')

    ax_hist.hist(flat_raw_energy+1e-12, bins=200, alpha=0.6, label='raw', color='C0')
    ax_hist.hist(flat_w_energy+1e-12, bins=200, alpha=0.6, label='weighted', color='C1')
    ax_hist.set_yscale('log'); ax_hist.set_title('Per-voxel energy hist (log)'); ax_hist.legend()

    ax_scatter.scatter(sraw, sw, s=1, alpha=0.3)
    ax_scatter.set_xlabel('raw value'); ax_scatter.set_ylabel('weighted value'); ax_scatter.set_title('raw vs weighted (sample)')

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', nargs='*')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--out_dir', default='paper_figs_final')
    parser.add_argument('--data_root', default=DATA_ROOT_DEFAULT)
    parser.add_argument('--max_cases', type=int, default=3)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = read_csv()
    # group rows by case
    grouped = {}
    for r in rows:
        grouped.setdefault(r['case'], []).append(r)
    cases = args.cases if args.cases else list(grouped.keys())[:args.max_cases]

    data = torch.load(args.ckpt, map_location='cpu')
    st = data.get('model_state', data)

    for case in cases:
        if case not in grouped:
            print('case not in CSV:', case); continue
        vol = load_case(args.data_root, case)
        in_ch = vol.shape[0]
        model = Double_RWKV_MedNeXt(in_channels=in_ch, n_channels=16, n_classes=2, exp_r=2, kernel_size=3, fbm_k_list=[2,4,8], fbm_learnable=True)
        model.load_state_dict(st, strict=False)
        model.to(args.device)
        model.eval()
        x = torch.from_numpy(vol[None]).float().to(args.device)
        with torch.no_grad():
            try:
                feats_med = model.encoder.mednext_enc.forward_encoder(x)
            except Exception:
                feats_med = model.encoder(x)
        enc = model.encoder
        # pick the row with largest ratio for layer f1 first for visibility
        # but iterate all rows for this case
        for r in grouped[case]:
            layer = r['layer']
            band = int(r['band'])
            # get feat
            layer_idx = 0 if layer=='f0' else 1
            feat = feats_med[layer_idx].to(args.device)
            fbm = getattr(enc, 'fbm0' if layer=='f0' else 'fbm1', None)
            if fbm is None:
                print('no fbm', layer); continue
            # compute per-band
            b,c,d,h,w = feat.shape
            x_fft = torch.fft.rfftn(feat, s=(d,h,w), dim=(-3,-2,-1), norm='ortho')
            band_masks = fbm._get_band_masks(d,h,w//2+1)
            pre_x = feat.clone()
            att_feat = feat
            per_raw = []
            per_w = []
            for idx, mask in enumerate(band_masks, start=1):
                mask = mask.to(args.device)
                low_part = torch.fft.irfftn(x_fft * mask, s=(d,h,w), dim=(-3,-2,-1), norm='ortho')
                high_part = pre_x - low_part
                pre_x = low_part
                raw_np = high_part.detach().cpu().numpy()[0]
                if idx-1 < len(fbm.freq_weight_conv_list):
                    fw = fbm.freq_weight_conv_list[idx-1](att_feat)
                    fw = fbm._activate(fw)
                    grp = fbm.spatial_group
                    tmp = fw.reshape(b, grp, -1, d, h, w) * high_part.reshape(b, grp, -1, d, h, w)
                    weighted_np = tmp.reshape(b, -1, d, h, w).detach().cpu().numpy()[0]
                else:
                    weighted_np = high_part.detach().cpu().numpy()[0]
                per_raw.append(raw_np.mean(axis=0))
                per_w.append(weighted_np.mean(axis=0))
            # select band
            raw_mean = per_raw[band-1]
            w_mean = per_w[band-1]
            out_path = os.path.join(args.out_dir, f'final_{case}_{layer}_band{band}_fig.png')
            stats = {'raw_energy': r['raw_energy'], 'weighted_energy': r['weighted_energy'], 'ratio': r['ratio']}
            make_figure(case, layer, band, raw_mean, w_mean, vol, out_path, stats)
            print('Saved figure', out_path)

if __name__=='__main__':
    main()
