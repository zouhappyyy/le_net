#!/usr/bin/env python3
"""Generate per-band comparison visuals using FBM modules from a trained model.
This loads the trained checkpoint into Double_RWKV_MedNeXt, extracts encoder features
via mednext_enc.forward_encoder, applies encoder.fbm0/fbm1 to the features, and
saves comparison images (slice + 3D MIP + diff/ratio/hist) per band.

Usage:
python3 tools/generate_paper_figs_trained.py --data_root <dir> --ckpt <path> --max_cases 3 --output_dir paper_figs_trained
"""
import os, argparse, sys
import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from nnunet_mednext.network_architecture.le_networks.Double_RWKV_MedNeXt import Double_RWKV_MedNeXt


def list_case_ids(data_root):
    files = [f for f in os.listdir(data_root) if f.endswith('.npy')]
    files.sort()
    return [os.path.splitext(f)[0] for f in files]


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


def _norm_for_display(a):
    a = a.astype(np.float32)
    vmin, vmax = float(a.min()), float(a.max())
    if vmax > vmin:
        return (a - vmin) / (vmax - vmin)
    return np.zeros_like(a)


def save_band_comparisons(raw_mean, w_mean, prefix, band_idx):
    mid = raw_mean.shape[0]//2
    raw_slice = raw_mean[mid]
    w_slice = w_mean[mid]
    raw_slice_n = _norm_for_display(raw_slice)
    w_slice_n = _norm_for_display(w_slice)
    diff_slice = w_slice - raw_slice
    max_abs = max(abs(diff_slice).max(), 1e-8)
    diff_slice_n = diff_slice / max_abs
    ratio_slice = w_slice / (raw_slice + 1e-8)
    RMAX = 5.0
    ratio_slice_clipped = np.clip(ratio_slice, 0.0, RMAX) / RMAX
    raw_vals = raw_slice.flatten()
    w_vals = w_slice.flatten()
    figb, axb = plt.subplots(2, 2, figsize=(10,8))
    axb[0,0].imshow(raw_slice_n, cmap='gray'); axb[0,0].set_title(f'Band {band_idx} raw slice'); axb[0,0].axis('off')
    axb[0,1].imshow(w_slice_n, cmap='gray'); axb[0,1].set_title(f'Band {band_idx} weighted slice'); axb[0,1].axis('off')
    im = axb[1,0].imshow(diff_slice_n, cmap='RdBu', vmin=-1, vmax=1); axb[1,0].set_title('Weighted - Raw (normalized)'); axb[1,0].axis('off')
    axb[1,1].hist([raw_vals, w_vals], bins=50, label=['raw','weighted'], color=['0.3','0.7']); axb[1,1].set_title('Intensity hist'); axb[1,1].legend()
    figb.colorbar(im, ax=axb[1,0], fraction=0.046, pad=0.04)
    png1 = f"{prefix}_band{band_idx}_comparison_slice.png"
    figb.savefig(png1, dpi=200); plt.close(figb)

    # 3D comparison
    def mip_views(vol):
        axial = vol.max(axis=0)
        coronal = vol.max(axis=1)
        sagittal = vol.max(axis=2)
        return axial, coronal, sagittal
    raw_ax, raw_cor, raw_sag = mip_views(raw_mean)
    w_ax, w_cor, w_sag = mip_views(w_mean)
    raw_ax_n, raw_cor_n, raw_sag_n = map(_norm_for_display, (raw_ax, raw_cor, raw_sag))
    w_ax_n, w_cor_n, w_sag_n = map(_norm_for_display, (w_ax, w_cor, w_sag))
    diff_ax, diff_cor, diff_sag = mip_views(w_mean - raw_mean)
    maxabs = max(abs(diff_ax).max(), abs(diff_cor).max(), abs(diff_sag).max(), 1e-8)
    diff_ax_n = diff_ax/maxabs; diff_cor_n = diff_cor/maxabs; diff_sag_n = diff_sag/maxabs
    def ratio_map(a,b,rmax=5.0):
        r = b/(a+1e-8)
        return np.clip(r,0.0,rmax)/rmax
    ratio_ax = ratio_map(raw_ax_n, w_ax_n); ratio_cor = ratio_map(raw_cor_n, w_cor_n); ratio_sag = ratio_map(raw_sag_n, w_sag_n)
    fig3, axes3 = plt.subplots(3,4, figsize=(16,12))
    axes3[0,0].imshow(raw_ax_n, cmap='gray'); axes3[0,0].set_title('raw - axial'); axes3[0,0].axis('off')
    axes3[0,1].imshow(raw_cor_n, cmap='gray'); axes3[0,1].set_title('raw - coronal'); axes3[0,1].axis('off')
    axes3[0,2].imshow(raw_sag_n, cmap='gray'); axes3[0,2].set_title('raw - sagittal'); axes3[0,2].axis('off')
    axes3[0,3].axis('off')
    axes3[1,0].imshow(w_ax_n, cmap='gray'); axes3[1,0].set_title('weighted - axial'); axes3[1,0].axis('off')
    axes3[1,1].imshow(w_cor_n, cmap='gray'); axes3[1,1].set_title('weighted - coronal'); axes3[1,1].axis('off')
    axes3[1,2].imshow(w_sag_n, cmap='gray'); axes3[1,2].set_title('weighted - sagittal'); axes3[1,2].axis('off')
    axes3[1,3].axis('off')
    im0 = axes3[2,0].imshow(diff_ax_n, cmap='RdBu', vmin=-1, vmax=1); axes3[2,0].set_title('diff - axial'); axes3[2,0].axis('off')
    axes3[2,1].imshow(diff_cor_n, cmap='RdBu', vmin=-1, vmax=1); axes3[2,1].set_title('diff - coronal'); axes3[2,1].axis('off')
    axes3[2,2].imshow(diff_sag_n, cmap='RdBu', vmin=-1, vmax=1); axes3[2,2].set_title('diff - sagittal'); axes3[2,2].axis('off')
    axes3[0,3].imshow(ratio_ax, cmap='inferno'); axes3[0,3].set_title('ratio axial'); axes3[0,3].axis('off')
    axes3[1,3].imshow(ratio_cor, cmap='inferno'); axes3[1,3].set_title('ratio coronal'); axes3[1,3].axis('off')
    axes3[2,3].imshow(ratio_sag, cmap='inferno'); axes3[2,3].set_title('ratio sagittal'); axes3[2,3].axis('off')
    fig3.colorbar(im0, ax=axes3[2,0], fraction=0.046, pad=0.04)
    png2 = f"{prefix}_band{band_idx}_comparison_3d.png"
    fig3.savefig(png2, dpi=200); plt.close(fig3)
    return png1, png2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--max_cases', type=int, default=3)
    parser.add_argument('--output_dir', type=str, default='paper_figs_trained')
    parser.add_argument('--channel', type=int, default=0)
    parser.add_argument('--k_list', nargs='+', type=int, default=[2,4,8])
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cases = list_case_ids(args.data_root)[:args.max_cases]
    print('Will process', len(cases), 'cases')

    # load checkpoint
    data = torch.load(args.ckpt, map_location='cpu')
    st = data.get('model_state', data)

    for case in cases:
        vol = load_case(args.data_root, case)
        in_ch = vol.shape[0]
        model = Double_RWKV_MedNeXt(in_channels=in_ch, n_channels=16, n_classes=2, exp_r=2, kernel_size=3, fbm_k_list=args.k_list, fbm_learnable=True)
        model.load_state_dict(st, strict=False)
        model.to(args.device)
        model.eval()
        x = torch.from_numpy(vol[None]).float().to(args.device)
        # use mednext_enc to get features without invoking RWKV
        with torch.no_grad():
            try:
                feats_med = model.encoder.mednext_enc.forward_encoder(x)
            except Exception:
                feats_med = model.encoder(x)
        f0 = feats_med[0].to(args.device)
        f1 = feats_med[1].to(args.device)
        # apply fbm0 and fbm1
        enc = model.encoder
        outputs = []
        for layer_name, feat, fbm in [('f0', f0, getattr(enc, 'fbm0', None)), ('f1', f1, getattr(enc, 'fbm1', None))]:
            if fbm is None:
                print('No fbm for', layer_name, 'skipping')
                continue
            # compute fbm forward to get high_acc
            out, high_acc = fbm(feat, att_feat=None, return_high=True)
            low = feat - high_acc
            # build per-band raw and weighted as in previous code
            b,c,d,h,w = feat.shape
            x_fft = torch.fft.rfftn(feat, s=(d,h,w), dim=(-3,-2,-1), norm='ortho')
            band_masks = fbm._get_band_masks(d, h, w//2+1)
            pre_x = feat.clone()
            per_raw = []
            per_weighted = []
            att_feat = feat
            for idx, mask in enumerate(band_masks):
                mask = mask.to(args.device)
                low_part = torch.fft.irfftn(x_fft * mask, s=(d,h,w), dim=(-3,-2,-1), norm='ortho')
                high_part = pre_x - low_part
                pre_x = low_part
                per_raw.append(high_part.detach().cpu().numpy())
                if idx < len(fbm.freq_weight_conv_list):
                    fw = fbm.freq_weight_conv_list[idx](att_feat)
                    fw = fbm._activate(fw)
                    grp = fbm.spatial_group
                    tmp = fw.reshape(b, grp, -1, d, h, w) * high_part.reshape(b, grp, -1, d, h, w)
                    band_weighted = tmp.reshape(b, -1, d, h, w).detach().cpu().numpy()
                    per_weighted.append(band_weighted)
                else:
                    per_weighted.append(high_part.detach().cpu().numpy())
            # for each band, reduce mean over channels and save comparisons
            for i,(r,wv) in enumerate(zip(per_raw, per_weighted), start=1):
                raw_mean = r.mean(axis=1)[0]
                w_mean = wv.mean(axis=1)[0]
                prefix = os.path.join(args.output_dir, f'fdconv_trained_{case}_{layer_name}')
                p1,p2 = save_band_comparisons(raw_mean, w_mean, prefix, i)
                print('Saved', p1, p2)

if __name__ == '__main__':
    main()
