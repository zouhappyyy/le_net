#!/usr/bin/env python3
"""Generate FBM per-band comparison visuals for center 64^3 crops from 128^3 volumes.

For each case: load .npy volume, center-crop to 64^3, feed into trained model encoder,
apply encoder.fbm0/fbm1, and save per-band comparison images (slice/diff/ratio/hist).

Usage:
  python3 tools/generate_center_crop_figs.py --data_root <dir> --ckpt <checkpoint> --max_cases 3 --out_dir paper_figs_center64
"""
import os, sys, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# prevent CUDA extension build during import
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


def list_cases(data_root):
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


def center_crop(vol, crop_size=64):
    # vol: [C,D,H,W]
    C,D,H,W = vol.shape
    sd = (D - crop_size)//2
    sh = (H - crop_size)//2
    sw = (W - crop_size)//2
    return vol[:, sd:sd+crop_size, sh:sh+crop_size, sw:sw+crop_size]


def _norm_for_display(a):
    a = a.astype(np.float32)
    vmin, vmax = float(a.min()), float(a.max())
    if vmax > vmin:
        return (a - vmin) / (vmax - vmin)
    return np.zeros_like(a)


def save_band_comparison_images(raw_mean, w_mean, prefix, case, layer, band):
    mid = raw_mean.shape[0]//2
    raw_slice = raw_mean[mid]
    w_slice = w_mean[mid]

    # normalize
    raw_n = _norm_for_display(raw_slice)
    w_n = _norm_for_display(w_slice)

    diff = w_slice - raw_slice
    max_abs = max(abs(diff).max(), 1e-8)
    diff_n = diff / max_abs
    ratio_map = w_mean / (raw_mean + 1e-12)
    RMAX = max(2.0, np.percentile(ratio_map, 99))
    ratio_disp = np.clip(ratio_map, 0.0, RMAX)/RMAX

    # hist
    flat_raw = raw_slice.flatten()
    flat_w = w_slice.flatten()

    fig, axs = plt.subplots(2,2, figsize=(10,8))
    axs[0,0].imshow(raw_n, cmap='gray'); axs[0,0].set_title('Raw slice'); axs[0,0].axis('off')
    axs[0,1].imshow(w_n, cmap='gray'); axs[0,1].set_title('Weighted slice'); axs[0,1].axis('off')
    im = axs[1,0].imshow(diff_n, cmap='RdBu', vmin=-1, vmax=1); axs[1,0].set_title('Weighted - Raw (normalized)'); axs[1,0].axis('off')
    axs[1,1].hist([flat_raw, flat_w], bins=80, label=['raw','weighted'], color=['0.3','0.7']); axs[1,1].legend(); axs[1,1].set_title('Intensity hist')
    fig.colorbar(im, ax=axs[1,0], fraction=0.046, pad=0.04)
    plt.suptitle(f'{case} {layer} band{band}')
    out1 = f"{prefix}_{case}_{layer}_band{band}_center64_comp_slice.png"
    fig.savefig(out1, dpi=200); plt.close(fig)

    # 3D MIP comparison
    def mip(v):
        return v.max(axis=0), v.max(axis=1), v.max(axis=2)
    raw_ax, raw_cor, raw_sag = mip(raw_mean)
    w_ax, w_cor, w_sag = mip(w_mean)
    raw_ax_n, raw_cor_n, raw_sag_n = map(_norm_for_display, (raw_ax, raw_cor, raw_sag))
    w_ax_n, w_cor_n, w_sag_n = map(_norm_for_display, (w_ax, w_cor, w_sag))
    diff_ax, diff_cor, diff_sag = mip(w_mean - raw_mean)
    maxabs = max(abs(diff_ax).max(), abs(diff_cor).max(), abs(diff_sag).max(), 1e-8)
    diff_ax_n, diff_cor_n, diff_sag_n = diff_ax/maxabs, diff_cor/maxabs, diff_sag/maxabs
    ratio_ax = np.clip(w_ax_n/(raw_ax_n+1e-8), 0, RMAX)/RMAX
    ratio_cor = np.clip(w_cor_n/(raw_cor_n+1e-8), 0, RMAX)/RMAX
    ratio_sag = np.clip(w_sag_n/(raw_sag_n+1e-8), 0, RMAX)/RMAX

    fig2, axes = plt.subplots(3,4, figsize=(16,12))
    axes[0,0].imshow(raw_ax_n, cmap='gray'); axes[0,0].set_title('raw axial'); axes[0,0].axis('off')
    axes[0,1].imshow(raw_cor_n, cmap='gray'); axes[0,1].set_title('raw coronal'); axes[0,1].axis('off')
    axes[0,2].imshow(raw_sag_n, cmap='gray'); axes[0,2].set_title('raw sagittal'); axes[0,2].axis('off')
    axes[0,3].axis('off')
    axes[1,0].imshow(w_ax_n, cmap='gray'); axes[1,0].set_title('weighted axial'); axes[1,0].axis('off')
    axes[1,1].imshow(w_cor_n, cmap='gray'); axes[1,1].set_title('weighted coronal'); axes[1,1].axis('off')
    axes[1,2].imshow(w_sag_n, cmap='gray'); axes[1,2].set_title('weighted sagittal'); axes[1,2].axis('off')
    axes[1,3].axis('off')
    im0 = axes[2,0].imshow(diff_ax_n, cmap='RdBu', vmin=-1, vmax=1); axes[2,0].set_title('diff axial'); axes[2,0].axis('off')
    axes[2,1].imshow(diff_cor_n, cmap='RdBu', vmin=-1, vmax=1); axes[2,1].set_title('diff coronal'); axes[2,1].axis('off')
    axes[2,2].imshow(diff_sag_n, cmap='RdBu', vmin=-1, vmax=1); axes[2,2].set_title('diff sagittal'); axes[2,2].axis('off')
    axes[0,3].imshow(ratio_ax, cmap='inferno'); axes[0,3].set_title('ratio axial'); axes[0,3].axis('off')
    axes[1,3].imshow(ratio_cor, cmap='inferno'); axes[1,3].set_title('ratio coronal'); axes[1,3].axis('off')
    axes[2,3].imshow(ratio_sag, cmap='inferno'); axes[2,3].set_title('ratio sagittal'); axes[2,3].axis('off')
    fig2.colorbar(im0, ax=axes[2,0], fraction=0.046, pad=0.04)
    out2 = f"{prefix}_{case}_{layer}_band{band}_center64_comp_3d.png"
    plt.tight_layout(); fig2.savefig(out2, dpi=200); plt.close(fig2)
    return out1, out2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--out_dir', default='paper_figs_center64')
    parser.add_argument('--max_cases', type=int, default=3)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cases = list_cases(args.data_root)[:args.max_cases]
    print('Processing', len(cases), 'cases')

    # load checkpoint
    data = torch.load(args.ckpt, map_location='cpu') if os.path.isfile(args.ckpt) else None
    st = data.get('model_state', data) if data is not None else None

    for case in cases:
        print('Case', case)
        vol = load_case(args.data_root, case)
        vol_crop = center_crop(vol, crop_size=64)
        in_ch = vol.shape[0]
        model = Double_RWKV_MedNeXt(in_channels=in_ch, n_channels=16, n_classes=2, exp_r=2, kernel_size=3, fbm_k_list=[2,4,8], fbm_learnable=True)
        if st is not None:
            model.load_state_dict(st, strict=False)
        model.to(args.device)
        model.eval()
        x = torch.from_numpy(vol_crop[None]).float().to(args.device)
        with torch.no_grad():
            try:
                feats_med = model.encoder.mednext_enc.forward_encoder(x)
            except Exception:
                feats_med = model.encoder(x)
        enc = model.encoder
        prefix = os.path.join(args.out_dir, 'center64')
        for li, layer in enumerate(['f0','f1']):
            feat = feats_med[li]
            fbm = getattr(enc, 'fbm0' if layer=='f0' else 'fbm1', None)
            if fbm is None:
                print('no fbm for', layer); continue
            # compute per-band
            b,c,d,h,w = feat.shape
            x_fft = torch.fft.rfftn(feat, s=(d,h,w), dim=(-3,-2,-1), norm='ortho')
            band_masks = fbm._get_band_masks(d, h, w//2 + 1)
            pre_x = feat.clone()
            per_raw = []
            per_w = []
            att_feat = feat
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
                raw_mean = raw_np.mean(axis=0)
                w_mean = weighted_np.mean(axis=0)
                out1, out2 = save_band_comparison_images(raw_mean, w_mean, prefix, case, layer, idx)
                print('Saved', out1, out2)

if __name__=='__main__':
    main()
