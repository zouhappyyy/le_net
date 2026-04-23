#!/usr/bin/env python3
"""Produce alignment plots to reconcile histograms and energy ratios.
Loads trained checkpoint, extracts encoder features (mednext_enc.forward_encoder),
applies fbm on f0/f1, reconstructs per-band raw and weighted arrays, and
saves for each band:
 - linear histogram of intensities (same bins) for raw and weighted
 - log histogram (log10 of absolute intensity + eps), showing tails
 - histogram of squared values (energy per voxel)
 - scatter sample of raw vs weighted voxels
 - a 2D map highlighting voxels where weighted/raw > threshold
"""
import os, sys, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# avoid compiling CUDA extensions during import
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


def _norm(a):
    a = a.astype(np.float32)
    mn, mx = a.min(), a.max()
    if mx>mn:
        return (a-mn)/(mx-mn)
    return a*0


def analyze_case(data_root, case_id, ckpt, out_dir, channel=0, device='cpu'):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(data_root, case_id + '.npy')
    if not os.path.isfile(p):
        for f in os.listdir(data_root):
            if f.startswith(case_id) and f.endswith('.npy'):
                p = os.path.join(data_root, f)
                break
    arr = np.load(p)
    if arr.ndim==3:
        arr = arr[None]
    vol = arr.astype(np.float32)
    vol_c = vol[channel:channel+1]
    x = torch.from_numpy(vol_c[None]).float().to(device)  # [1,1,D,H,W]

    # load checkpoint
    data = torch.load(ckpt, map_location='cpu')
    st = data.get('model_state', data)

    in_ch = vol.shape[0]
    model = Double_RWKV_MedNeXt(in_channels=in_ch, n_channels=16, n_classes=2, exp_r=2, kernel_size=3, fbm_k_list=[2,4,8], fbm_learnable=True)
    model.load_state_dict(st, strict=False)
    model.to(device)
    model.eval()

    with torch.no_grad():
        try:
            feats_med = model.encoder.mednext_enc.forward_encoder(x.to(device))
        except Exception:
            feats_med = model.encoder(x.to(device))
    enc = model.encoder
    results = []
    for layer_idx, layer_name in enumerate(['f0','f1']):
        feat = feats_med[layer_idx].to(device)
        fbm = getattr(enc, 'fbm0' if layer_name=='f0' else 'fbm1', None)
        if fbm is None:
            print('no fbm for', layer_name); continue
        with torch.no_grad():
            out, high_acc = fbm(feat, att_feat=None, return_high=True)
        # reconstruct per-band similar to earlier
        b,c,d,h,w = feat.shape
        x_fft = torch.fft.rfftn(feat, s=(d,h,w), dim=(-3,-2,-1), norm='ortho')
        band_masks = fbm._get_band_masks(d,h,w//2+1)
        pre_x = feat.clone()
        att_feat = feat
        for idx, mask in enumerate(band_masks, start=1):
            mask = mask.to(device)
            low_part = torch.fft.irfftn(x_fft * mask, s=(d,h,w), dim=(-3,-2,-1), norm='ortho')
            high_part = pre_x - low_part
            pre_x = low_part
            # weighted
            if idx-1 < len(fbm.freq_weight_conv_list):
                fw = fbm.freq_weight_conv_list[idx-1](att_feat)
                fw = fbm._activate(fw)
                grp = fbm.spatial_group
                tmp = fw.reshape(b, grp, -1, d, h, w) * high_part.reshape(b, grp, -1, d, h, w)
                weighted = tmp.reshape(b, -1, d, h, w).detach().cpu().numpy()[0]
            else:
                weighted = high_part.detach().cpu().numpy()[0]
            raw = high_part.detach().cpu().numpy()[0]
            # raw/weighted mean over channels
            raw_mean = raw.mean(axis=0)  # [D,H,W]
            w_mean = weighted.mean(axis=0)
            # energies
            e_raw = float((raw_mean**2).sum())
            e_w = float((w_mean**2).sum())
            ratio = e_w/(e_raw+1e-12)
            print(f'{case_id} {layer_name} band{idx} energy raw {e_raw:.4f} weighted {e_w:.4f} ratio {ratio:.4f}')

            # histograms: use same bins across raw & weighted for intensity
            flat_raw = raw_mean.flatten()
            flat_w = w_mean.flatten()
            # linear hist
            vmin = min(flat_raw.min(), flat_w.min())
            vmax = max(flat_raw.max(), flat_w.max())
            bins = np.linspace(vmin, vmax, 200)
            fig, ax = plt.subplots(1,3, figsize=(15,4))
            ax[0].hist(flat_raw, bins=bins, alpha=0.6, label='raw', color='C0')
            ax[0].hist(flat_w, bins=bins, alpha=0.6, label='weighted', color='C1')
            ax[0].legend(); ax[0].set_title('Linear hist of intensities')
            # log hist of abs
            logbins = np.logspace(np.log10(max(abs(vmin),1e-6)), np.log10(max(abs(vmax),1e-6)+1e-6), 200)
            ax[1].hist(np.abs(flat_raw)+1e-12, bins=logbins, alpha=0.6, label='raw', color='C0')
            ax[1].hist(np.abs(flat_w)+1e-12, bins=logbins, alpha=0.6, label='weighted', color='C1')
            ax[1].set_xscale('log'); ax[1].legend(); ax[1].set_title('Log hist of abs intensity')
            # energy per voxel hist (squared)
            ax[2].hist(flat_raw**2, bins=200, alpha=0.6, label='raw', color='C0')
            ax[2].hist(flat_w**2, bins=200, alpha=0.6, label='weighted', color='C1')
            ax[2].legend(); ax[2].set_title('Hist of per-voxel energy (squared)')
            plt.suptitle(f'{case_id} {layer_name} band{idx} energies: raw {e_raw:.2f}, weighted {e_w:.2f}, ratio {ratio:.2f}')
            fn = os.path.join(out_dir, f'align_{case_id}_{layer_name}_band{idx}_hist.png')
            fig.savefig(fn, dpi=200); plt.close(fig)

            # scatter sample (sample up to 10000 voxels)
            N = flat_raw.size
            idxs = np.random.choice(N, min(10000, N), replace=False)
            sraw = flat_raw[idxs]; sw = flat_w[idxs]
            fig2, ax2 = plt.subplots(1,2, figsize=(10,4))
            ax2[0].scatter(sraw, sw, s=1, alpha=0.3)
            ax2[0].set_xlabel('raw'); ax2[0].set_ylabel('weighted'); ax2[0].set_title('raw vs weighted (sample)')
            # show ratio histogram
            ax2[1].hist(sw/(sraw+1e-12), bins=np.linspace(0, np.percentile(sw/(sraw+1e-12),99),200))
            ax2[1].set_title('voxel-wise ratio (clipped)')
            fn2 = os.path.join(out_dir, f'align_{case_id}_{layer_name}_band{idx}_scatter.png')
            fig2.suptitle(f'ratio summary: mean {ratio:.2f}'); fig2.savefig(fn2, dpi=200); plt.close(fig2)

            # ratio map of middle slice
            mid = raw_mean.shape[0]//2
            ratio_map = w_mean/(raw_mean+1e-12)
            # clip for display
            RMAX = np.percentile(ratio_map, 99)
            RMAX = max(RMAX, 2.0)
            ratio_disp = np.clip(ratio_map, 0, RMAX)/RMAX
            fig3, ax3 = plt.subplots(1,3, figsize=(12,4))
            ax3[0].imshow(_norm(raw_mean[mid]), cmap='gray'); ax3[0].set_title('raw slice')
            ax3[1].imshow(_norm(w_mean[mid]), cmap='gray'); ax3[1].set_title('weighted slice')
            im = ax3[2].imshow(ratio_disp[mid], cmap='inferno'); ax3[2].set_title(f'ratio map clipped @ {RMAX:.2f}'); fig3.colorbar(im, ax=ax3[2], fraction=0.046)
            for a in ax3: a.axis('off')
            fn3 = os.path.join(out_dir, f'align_{case_id}_{layer_name}_band{idx}_ratiomap.png')
            fig3.suptitle(f'per-voxel ratio (middle slice) mean_ratio {ratio:.2f}')
            fig3.savefig(fn3, dpi=200); plt.close(fig3)

            results.append({'case':case_id, 'layer':layer_name, 'band':idx, 'e_raw':e_raw, 'e_weighted':e_w, 'ratio':ratio, 'hist':fn, 'scatter':fn2, 'ratiomap':fn3})
    return results

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--out_dir', default='paper_figs_align')
    parser.add_argument('--channel', type=int, default=0)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    res = analyze_case(args.data_root, args.case, args.ckpt, args.out_dir, channel=args.channel, device=args.device)
    print('Done. Results files:')
    for r in res:
        print(r)
