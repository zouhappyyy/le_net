#!/usr/bin/env python3
"""Compare per-band raw vs weighted energies between full volume and center crop.
Saves results to tools/center_crop_compare_<case>.json and prints a summary.
"""
import os, sys, json, argparse
import numpy as np
import torch

# prevent CUDA extension compilation during import
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
from nnunet_mednext.network_architecture.le_networks.FDConv_3d import FrequencyBandModulation3D

# restore original loader
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


def center_crop(vol, crop_size):
    # vol: [C,D,H,W]
    C,D,H,W = vol.shape
    sd = (D - crop_size) // 2
    sh = (H - crop_size) // 2
    sw = (W - crop_size) // 2
    return vol[:, sd:sd+crop_size, sh:sh+crop_size, sw:sw+crop_size]


def compute_per_band(feat, fbm):
    # feat: torch tensor [B,C,D,H,W]
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
        raw_np = high_part.detach().cpu().numpy()  # [B, Cb, D, H, W]
        if idx < len(fbm.freq_weight_conv_list):
            fw = fbm.freq_weight_conv_list[idx](att_feat)
            fw = fbm._activate(fw)
            grp = fbm.spatial_group
            tmp = fw.reshape(b, grp, -1, d, h, w) * high_part.reshape(b, grp, -1, d, h, w)
            weighted_np = tmp.detach().cpu().numpy()
        else:
            weighted_np = high_part.detach().cpu().numpy()
        # reduce by mean over group channels (consistent with earlier analysis)
        raw_mean = raw_np.mean(axis=1)[0]
        weighted_mean = weighted_np.mean(axis=1)[0]
        per_raw.append(raw_mean)
        per_weighted.append(weighted_mean)
    return per_raw, per_weighted


def energy(arr):
    return float((arr**2).sum())


def analyze_case(data_root, case_id, ckpt, crop_size=64, k_list=[2,4,8], device='cpu'):
    vol = load_case(data_root, case_id)
    print('Loaded', case_id, 'shape', vol.shape)
    vol_full = vol
    vol_crop = center_crop(vol, crop_size)
    print('Center crop shape', vol_crop.shape)

    in_ch = vol.shape[0]
    model = Double_RWKV_MedNeXt(in_channels=in_ch, n_channels=16, n_classes=2, exp_r=2, kernel_size=3, fbm_k_list=k_list, fbm_learnable=True)
    if ckpt and os.path.isfile(ckpt):
        data = torch.load(ckpt, map_location='cpu')
        st = data.get('model_state', data)
        model.load_state_dict(st, strict=False)
    model.to(device)
    model.eval()

    results = {'case': case_id, 'crop_size': crop_size, 'layers': {}}

    # helper to process a volume
    def process_vol(vol_np, tag):
        x = torch.from_numpy(vol_np[None]).float().to(device)
        with torch.no_grad():
            try:
                feats = model.encoder.mednext_enc.forward_encoder(x)
            except Exception:
                feats = model.encoder(x)
        enc = model.encoder
        res = {}
        for li, layer in enumerate(['f0','f1']):
            feat = feats[li]
            fbm = getattr(enc, 'fbm0' if layer=='f0' else 'fbm1', None)
            if fbm is None:
                continue
            # compute per-band
            per_raw, per_w = compute_per_band(feat, fbm)
            bands = []
            for bi,(r,wv) in enumerate(zip(per_raw, per_w), start=1):
                e_r = energy(r)
                e_w = energy(wv)
                bands.append({'band': bi, 'raw_energy': e_r, 'weighted_energy': e_w, 'ratio': e_w/(e_r+1e-12)})
            res[layer] = bands
        return res

    res_full = process_vol(vol_full, 'full')
    res_crop = process_vol(vol_crop, 'crop')

    results['full'] = res_full
    results['crop'] = res_crop

    out_path = f'tools/center_crop_compare_{case_id}.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print('Saved results to', out_path)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=False, default='/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_preprocessed/Task530_EsoTJ_30pct/nnUNetData_plans_v2.1_trgSp_1x1x1_stage0')
    parser.add_argument('--case', required=False, default='ESO_TJ_1010018171')
    parser.add_argument('--ckpt', required=False, default='tools/fd_model_ckpt_learnable.pth')
    parser.add_argument('--crop', type=int, default=64)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    res = analyze_case(args.data_root, args.case, args.ckpt, crop_size=args.crop, device=args.device)
    # print concise summary
    for layer in ['f0','f1']:
        print('\nLayer', layer)
        full_bands = res['full'].get(layer, [])
        crop_bands = res['crop'].get(layer, [])
        for fb, cb in zip(full_bands, crop_bands):
            print(f"Band {fb['band']}: full ratio={fb['ratio']:.3f}, crop ratio={cb['ratio']:.3f}, full E_w={fb['weighted_energy']:.1f}, crop E_w={cb['weighted_energy']:.1f}")
