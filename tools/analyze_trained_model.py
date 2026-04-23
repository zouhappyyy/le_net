#!/usr/bin/env python3
"""Analyze trained model's FBM energy on encoder first two layers.

Usage:
  python3 tools/analyze_trained_model.py --ckpt tools/fd_model_ckpt.pth

If no checkpoint provided, the model will use its random initialization.
Outputs:
  - tools/analyze_model_results.json
  - tools/analyze_model_band_energy.csv
"""
import argparse, os, json, csv
import numpy as np
import torch

from nnunet_mednext.network_architecture.le_networks.Double_RWKV_MedNeXt import Double_RWKV_MedNeXt

RESULT_JSON = 'tools/analyze_model_results.json'
CSV_PATH = 'tools/analyze_model_band_energy.csv'


def load_case(data_root, case_id):
    p = os.path.join(data_root, f"{case_id}.npy")
    if not os.path.isfile(p):
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


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=False, default='/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_preprocessed/Task530_EsoTJ_30pct/nnUNetData_plans_v2.1_trgSp_1x1x1_stage0')
    parser.add_argument('--case_id', type=str, required=False, default='ESO_TJ_60011222468')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--fbm_k_list', nargs='+', type=int, default=[2,4,8])
    parser.add_argument('--fbm_learnable', action='store_true')
    args = parser.parse_args(argv)

    device = 'cuda' if (args.device is None and torch.cuda.is_available()) or args.device == 'cuda' else 'cpu'

    vol = load_case(args.data_root, args.case_id)
    print('Loaded vol', vol.shape)

    in_ch = vol.shape[0]
    model = Double_RWKV_MedNeXt(in_channels=in_ch, n_channels=16, n_classes=2, exp_r=2, kernel_size=3, fbm_k_list=args.fbm_k_list, fbm_learnable=args.fbm_learnable)
    if args.ckpt is not None and os.path.isfile(args.ckpt):
        print('Loading checkpoint', args.ckpt)
        data = torch.load(args.ckpt, map_location='cpu')
        st = data.get('model_state', data)
        # load with strict=False to allow presence/absence of learnable band logits
        missing = model.load_state_dict(st, strict=False)
        print('State dict load completed (missing/unexpected):', missing)
    model.to(device)
    model.eval()

    # run encoder to get f0,f1
    x = torch.from_numpy(vol[None]).float().to(device)  # [1,C,D,H,W]
    with torch.no_grad():
        # Use mednext_enc.forward_encoder to avoid invoking RWKV (and its CUDA extension build)
        try:
            feats_med = model.encoder.mednext_enc.forward_encoder(x)
        except Exception:
            # fallback to full encoder forward if mednext_enc not available or fails
            feats_med = model.encoder(x)

        # feats_med: [f0,f1,f2,f3,bottleneck]
        f0_tensor = feats_med[0]
        f1_tensor = feats_med[1]

        # If FBM modules exist inside encoder, apply them (mednext forward normally does this)
        enc = model.encoder
        if hasattr(enc, 'fbm0'):
            try:
                f0_out = enc.fbm0(f0_tensor)[1] if False else enc.fbm0(f0_tensor)
            except Exception:
                f0_out = enc.fbm0(f0_tensor.to(next(enc.fbm0.parameters()).device))
        else:
            f0_out = f0_tensor
        if hasattr(enc, 'fbm1'):
            try:
                f1_out = enc.fbm1(f1_tensor)
            except Exception:
                f1_out = enc.fbm1(f1_tensor.to(next(enc.fbm1.parameters()).device))
        else:
            f1_out = f1_tensor

    f0 = f0_out.detach().cpu().numpy()
    f1 = f1_out.detach().cpu().numpy()

    results = {'case': args.case_id, 'layers': []}
    rows = []
    # analyze f0 with fbm0 if exists
    enc = model.encoder
    for name, feat_np in [('f0', f0), ('f1', f1)]:
        layer_info = {'layer': name}
        if name == 'f0' and hasattr(enc, 'fbm0'):
            fbm = enc.fbm0
            print('Analyzing fbm0 on f0')
        elif name == 'f1' and hasattr(enc, 'fbm1'):
            fbm = enc.fbm1
            print('Analyzing fbm1 on f1')
        else:
            print('No FBM for', name)
            layer_info['note'] = 'no_fbm'
            results['layers'].append(layer_info)
            continue

        # call fbm on torch tensor to get out and high_acc
        t = torch.from_numpy(feat_np).to(device)
        with torch.no_grad():
            out, high_acc = fbm(t, att_feat=None, return_high=True)
        high = high_acc.detach().cpu().numpy()  # [B,C,D,H,W]
        low = (t - high_acc).detach().cpu().numpy()

        # compute per-band via internal masks
        b, c, d, h, w = t.shape
        x_fft = torch.fft.rfftn(t, s=(d,h,w), dim=(-3,-2,-1), norm='ortho')
        band_masks = fbm._get_band_masks(d, h, w//2 + 1)
        per_raw = []
        per_weighted = []
        pre_x = t.clone()
        att_feat = t
        for idx in range(len(band_masks)):
            mask = band_masks[idx].to(device)
            low_part = torch.fft.irfftn(x_fft * mask, s=(d,h,w), dim=(-3,-2,-1), norm='ortho')
            high_part = pre_x - low_part
            pre_x = low_part
            raw_np = high_part.detach().cpu().numpy()
            per_raw.append(raw_np)
            # weighted
            if idx < len(fbm.freq_weight_conv_list):
                fw = fbm.freq_weight_conv_list[idx](att_feat)
                fw = fbm._activate(fw)
                grp = fbm.spatial_group
                tmp = fw.reshape(b, grp, -1, d, h, w) * high_part.reshape(b, grp, -1, d, h, w)
                per_weighted.append(tmp.detach().cpu().numpy())
            else:
                per_weighted.append(high_part.detach().cpu().numpy())

        # energies
        band_rows = []
        for i,(r,wv) in enumerate(zip(per_raw, per_weighted), start=1):
            r_mean = r.mean(axis=1)[0]
            w_mean = wv.mean(axis=1)[0]
            e_r = compute_energy(r_mean)
            e_w = compute_energy(w_mean)
            band_rows.append({'layer': name, 'band': i, 'raw_energy': e_r, 'weighted_energy': e_w, 'ratio': e_w/(e_r+1e-12)})
            rows.append({'layer': name, 'band': i, 'raw_energy': e_r, 'weighted_energy': e_w})
        layer_info['bands'] = band_rows
        # low energy
        low_energy = compute_energy(low.mean(axis=1)[0])
        layer_info['low_energy'] = low_energy
        results['layers'].append(layer_info)

    # save JSON and CSV
    with open(RESULT_JSON, 'w') as jf:
        json.dump(results, jf, indent=2)
    with open(CSV_PATH, 'w', newline='') as cf:
        writer = csv.DictWriter(cf, fieldnames=['layer','band','raw_energy','weighted_energy'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print('Saved results to', RESULT_JSON, CSV_PATH)

if __name__ == '__main__':
    main()
