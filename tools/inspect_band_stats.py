#!/usr/bin/env python3
"""Inspect per-band raw vs weighted statistics for a case to diagnose histogram vs energy ratio mismatch.

Prints per-band:
 - raw energy (sum squares), weighted energy (sum squares), ratio
 - raw slice stats (min,max,mean,std,percentiles), weighted slice stats
 - prints a small sample of values
"""
import os
import numpy as np
import torch
from nnunet_mednext.network_architecture.le_networks.FDConv_3d import FrequencyBandModulation3D

DATA_ROOT = '/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_preprocessed/Task530_EsoTJ_30pct/nnUNetData_plans_v2.1_trgSp_1x1x1_stage0'
CASE = 'ESO_TJ_1010018171'
CHANNEL = 0
K_LIST = [2,4,8]
LEARNABLE = True
SOFT_BETA = 30.0
DEVICE = 'cpu'


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


def energy(arr):
    return float((arr**2).sum())


def stats(a):
    a = a.astype(np.float32)
    return {'min':float(a.min()), 'max':float(a.max()), 'mean':float(a.mean()), 'std':float(a.std()), 'p50':float(np.percentile(a,50)),'p95':float(np.percentile(a,95))}


def main():
    vol = load_case(DATA_ROOT, CASE)
    print('Loaded', CASE, 'shape', vol.shape)
    vol_c = vol[CHANNEL:CHANNEL+1]
    x = torch.from_numpy(vol_c[None]).float().to(DEVICE)

    fbm = FrequencyBandModulation3D(in_channels=1, k_list=K_LIST, learnable_bands=LEARNABLE, soft_band_beta=SOFT_BETA).to(DEVICE)
    with torch.no_grad():
        out, high_acc = fbm(x, att_feat=None, return_high=True)
    low = x - high_acc

    b,c,d,h,w = x.shape
    x_fft = torch.fft.rfftn(x, s=(d,h,w), dim=(-3,-2,-1), norm='ortho')
    band_masks = fbm._get_band_masks(d,h,w//2+1)

    pre_x = x.clone()
    att_feat = x
    for idx, mask in enumerate(band_masks, start=1):
        mask = mask.to(DEVICE)
        low_part = torch.fft.irfftn(x_fft * mask, s=(d,h,w), dim=(-3,-2,-1), norm='ortho')
        high_part = pre_x - low_part
        pre_x = low_part
        raw_np = high_part.detach().cpu().numpy()
        if idx-1 < len(fbm.freq_weight_conv_list):
            fw = fbm.freq_weight_conv_list[idx-1](att_feat)
            fw = fbm._activate(fw)
            grp = fbm.spatial_group
            tmp = fw.reshape(b, grp, -1, d, h, w) * high_part.reshape(b, grp, -1, d, h, w)
            weighted_np = tmp.reshape(b, -1, d, h, w).detach().cpu().numpy()
        else:
            weighted_np = high_part.detach().cpu().numpy()

        # reduce to mean over channels like analysis code
        raw_mean = raw_np.mean(axis=1)[0]
        w_mean = weighted_np.mean(axis=1)[0]

        print('\n=== Band', idx, '===')
        print('energy raw:', energy(raw_mean), 'energy weighted:', energy(w_mean), 'ratio:', energy(w_mean)/(energy(raw_mean)+1e-12))

        # slice stats (middle)
        mid = raw_mean.shape[0]//2
        raw_slice = raw_mean[mid]
        w_slice = w_mean[mid]
        print('raw slice stats:', stats(raw_slice))
        print('weighted slice stats:', stats(w_slice))
        # sample values
        raw_flat = raw_slice.flatten()
        w_flat = w_slice.flatten()
        print('raw slice sample:', raw_flat[:10])
        print('weighted slice sample:', w_flat[:10])

if __name__ == '__main__':
    main()
