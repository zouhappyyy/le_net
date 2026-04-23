#!/usr/bin/env python3
import os, sys
import numpy as np
import torch
from nnunet_mednext.network_architecture.le_networks.FDConv_3d import FrequencyBandModulation3D
import json, traceback

DATA_ROOT = '/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_preprocessed/Task530_EsoTJ_30pct/nnUNetData_plans_v2.1_trgSp_1x1x1_stage0'
CASE_ID = 'ESO_TJ_60011222468'
CHANNEL = 0
K_LIST = [2,4,8]
LEARNABLE = True
SOFT_BETA = 30.0
# If set, force the learnable band's underlying radii to correspond to these k values (r = 0.5 / k)
# and freeze them (requires_grad=False). Set to None to keep normal learnable behavior.
FIX_LEARNABLE_TO_KLIST = [2,4,8]

np.set_printoptions(precision=6, suppress=True)

RESULT_JSON = 'fd_quick_results.json'
LOG_FILE = 'fd_quick_run.log'


def log(msg):
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(str(msg) + '\n')
    except Exception:
        pass
    print(msg, flush=True)

def load_case(data_root, case_id):
    p = os.path.join(data_root, f"{case_id}.npy")
    if not os.path.isfile(p):
        # pick first starting with case_id
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


def compute():
    log('START run_fd_energy_quick')
    log(f'DATA_ROOT={DATA_ROOT} CASE_ID={CASE_ID}')
    vol = load_case(DATA_ROOT, CASE_ID)
    log('Loaded volume shape: ' + str(vol.shape))
    if CHANNEL < 0 or CHANNEL >= vol.shape[0]:
        raise ValueError('channel OOB')
    vol_c = vol[CHANNEL:CHANNEL+1]
    x = torch.from_numpy(vol_c[None]).float()  # [1,1,D,H,W]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = x.to(device)
    log('Tensor on device: ' + str(device))
    fbm = FrequencyBandModulation3D(in_channels=1, k_list=K_LIST, learnable_bands=LEARNABLE, soft_band_beta=SOFT_BETA, max_size=(x.shape[2], x.shape[3], x.shape[4])).to(device)

    # Optionally force the learnable radii to correspond to a fixed k-list and freeze them
    if LEARNABLE and FIX_LEARNABLE_TO_KLIST is not None:
        try:
            base_radii = torch.tensor([0.5 / float(k) for k in FIX_LEARNABLE_TO_KLIST], dtype=torch.float32, device=device)
            logits = fbm._init_band_radius_logits(base_radii.cpu()).to(device)
            # overwrite parameter and freeze
            fbm.band_radius_logits = torch.nn.Parameter(logits, requires_grad=False)
            log('Force-set learnable radii to correspond to k_list ' + str(FIX_LEARNABLE_TO_KLIST))
        except Exception as e:
            log('Failed to force-set learnable radii: ' + repr(e))
            log(traceback.format_exc())

    log('FBM built; running forward')
    with torch.no_grad():
        out, high_acc = fbm(x, att_feat=None, return_high=True)
    log(f'x mean {float(x.mean())} min {float(x.min())} max {float(x.max())}')
    log('high_acc shape: ' + str(getattr(high_acc, 'shape', None)))
    low_final = (x - high_acc).cpu().numpy()
    b,c,d,h,w = x.shape
    x_fft = torch.fft.rfftn(x, s=(d,h,w), dim=(-3,-2,-1), norm='ortho')
    band_masks = fbm._get_band_masks(d, h, w//2 + 1)
    per_raw = []
    per_weight = []
    pre_x = x.clone()
    att_feat = x
    for idx in range(len(band_masks)):
        mask = band_masks[idx].to(device)
        low_part = torch.fft.irfftn(x_fft * mask, s=(d,h,w), dim=(-3,-2,-1), norm='ortho')
        high_part = pre_x - low_part
        pre_x = low_part
        per_raw.append(high_part.detach().cpu().numpy())
        if idx < len(fbm.freq_weight_conv_list):
            fw = fbm.freq_weight_conv_list[idx](att_feat)
            fw = fbm._activate(fw)
            grp = fbm.spatial_group
            tmp = fw.reshape(b, grp, -1, d, h, w) * high_part.reshape(b, grp, -1, d, h, w)
            per_weight.append(tmp.detach().cpu().numpy())
        else:
            per_weight.append(high_part.detach().cpu().numpy())
    # energies
    def energy(arr):
        return float((arr**2).sum())
    log('\nPer-band energies:')
    results = {'case': CASE_ID, 'bands': []}
    for i,(r,wv) in enumerate(zip(per_raw, per_weight), start=1):
        r_mean = r.mean(axis=1)[0]
        w_mean = wv.mean(axis=1)[0]
        e_r = energy(r_mean)
        e_w = energy(w_mean)
        ratio = e_w / (e_r + 1e-12)
        log(f'Band {i}: raw_energy={e_r:.6e}, weighted_energy={e_w:.6e}, ratio={ratio:.4f}')
        results['bands'].append({'band': i, 'raw_energy': e_r, 'weighted_energy': e_w, 'ratio': ratio})
    low_e = energy(low_final.mean(axis=1)[0])
    log('Low energy: ' + f'{low_e:.6e}')
    results['low_energy'] = low_e
    # print some statistics of masks if learnable
    try:
        if LEARNABLE:
            radii = fbm._get_learnable_radii()
            radii_np = radii.cpu().numpy().tolist()
            log('Learnable radii: ' + str(radii_np))
            results['radii'] = radii_np
    except Exception as e:
        log('Could not get learnable radii: ' + repr(e))
        log(traceback.format_exc())

    # save results
    try:
        with open(RESULT_JSON, 'w') as jf:
            json.dump(results, jf, indent=2)
        log('Saved results to ' + RESULT_JSON)
    except Exception as e:
        log('Failed to save results: ' + repr(e))
        log(traceback.format_exc())


if __name__ == '__main__':
    try:
        compute()
    except Exception as e:
        log('Unhandled exception: ' + repr(e))
        log(traceback.format_exc())
        sys.exit(1)
