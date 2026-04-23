#!/usr/bin/env python3
"""Generate annotated final figures for center-cropped 64^3 inputs (paper-style).

This version emphasizes the reviewer-requested story: compare the original input
against the feature response after the progressive spectrum residual decomposer.
At each step, the module first extracts a smaller low-frequency component and
then preserves the remaining high-frequency residual for enhancement.
"""
import os, sys, argparse, csv
import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# avoid JIT compile during import
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

DATA_ROOT_DEFAULT = '/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_preprocessed/Task530_EsoTJ_30pct/nnUNetData_plans_v2.1_trgSp_1x1x1_stage0'


def load_case(data_root, case_id):
    p = os.path.join(data_root, case_id + '.npy')
    if not os.path.isfile(p):
        matches = [
            os.path.join(data_root, f)
            for f in os.listdir(data_root)
            if f.startswith(case_id) and f.endswith('.npy')
        ]
        if not matches:
            raise FileNotFoundError(f'No .npy file found for case {case_id} under {data_root}')
        p = sorted(matches)[0]
    arr = np.load(p)
    if arr.ndim == 3:
        arr = arr[None]
    if arr.ndim != 4:
        raise RuntimeError(f'Unexpected array shape {arr.shape} in {p}, expected (C,D,H,W) or (D,H,W)')
    return arr.astype(np.float32)


def center_crop(vol, crop_size=64):
    C, D, H, W = vol.shape
    if min(D, H, W) < crop_size:
        raise ValueError(f'Cannot center-crop shape {vol.shape} to {crop_size}^3')
    sd = (D - crop_size) // 2
    sh = (H - crop_size) // 2
    sw = (W - crop_size) // 2
    return vol[:, sd:sd + crop_size, sh:sh + crop_size, sw:sw + crop_size]


def _norm01(a):
    a = a.astype(np.float32)
    vmin, vmax = float(a.min()), float(a.max())
    if vmax > vmin:
        return (a - vmin) / (vmax - vmin)
    return np.zeros_like(a)


def _collapse_channels(arr):
    if arr.ndim == 4:
        return arr.mean(axis=0)
    if arr.ndim == 3:
        return arr
    raise ValueError(f'Unsupported array ndim {arr.ndim}, expected 3 or 4')


def _band_energy(vol3d):
    return float(np.sum(vol3d ** 2))


def _spectrum_features(vol3d):
    """Compute compact whole-image spectrum statistics from a 3D volume."""
    fft = np.fft.rfftn(vol3d.astype(np.float32), norm='ortho')
    mag = np.abs(fft)
    d, h, w = vol3d.shape
    fd, fh, fw = np.meshgrid(np.fft.fftfreq(d), np.fft.fftfreq(h), np.fft.rfftfreq(w), indexing='ij')
    freq_radius = np.sqrt(fd ** 2 + fh ** 2 + fw ** 2)
    high_mask = freq_radius >= 0.30
    total_energy = float(np.sum(mag ** 2) + 1e-12)
    high_energy = float(np.sum(mag[high_mask] ** 2))
    return {
        'total_energy': total_energy,
        'high_energy': high_energy,
        'high_ratio': high_energy / total_energy,
        'mag': mag,
    }


def _boundary_mask_3d(shape, width=8):
    d, h, w = shape
    mask = np.zeros((d, h, w), dtype=np.float32)
    mask[:width, :, :] = 1
    mask[-width:, :, :] = 1
    mask[:, :width, :] = 1
    mask[:, -width:, :] = 1
    mask[:, :, :width] = 1
    mask[:, :, -width:] = 1
    return mask


def _boundary_focus_ratio(vol3d, high_cut=0.30, boundary_width=8):
    """High-frequency ratio measured only on the boundary shell."""
    fft = np.fft.rfftn(vol3d.astype(np.float32), norm='ortho')
    mag2 = np.abs(fft) ** 2
    d, h, w = vol3d.shape
    fd, fh, fw = np.meshgrid(np.fft.fftfreq(d), np.fft.fftfreq(h), np.fft.rfftfreq(w), indexing='ij')
    freq_radius = np.sqrt(fd ** 2 + fh ** 2 + fw ** 2)
    high_mask = freq_radius >= float(high_cut)
    total_energy = float(np.sum(mag2) + 1e-12)
    high_energy = float(np.sum(mag2[high_mask]))
    boundary_mask = _boundary_mask_3d((d, h, w), width=boundary_width)
    boundary_energy = float(np.sum(vol3d.astype(np.float32) ** 2 * boundary_mask) + 1e-12)
    return {
        'high_ratio': high_energy / total_energy,
        'boundary_energy_ratio': boundary_energy / float(np.sum(vol3d.astype(np.float32) ** 2) + 1e-12),
        'boundary_mask': boundary_mask,
    }


def make_figure(case, layer, band, input_vol, enhanced_vol, out_path, stats, spectrum_path=None):
    """Build a reviewer-facing comparison plot for progressive spectrum residual decomposition."""
    mid = input_vol.shape[1] // 2
    inp = _collapse_channels(input_vol)
    enh = _collapse_channels(enhanced_vol)
    gain = enh - inp

    inp_slice = inp[mid]
    enh_slice = enh[mid]
    gain_slice = gain[mid]

    flat_inp = inp.flatten()
    flat_enh = enh.flatten()
    N = flat_inp.size
    sample_n = min(5000, N)
    if sample_n > 0:
        idxs = np.random.choice(N, sample_n, replace=False)
        s_inp = flat_inp[idxs]
        s_enh = flat_enh[idxs]
    else:
        s_inp = flat_inp
        s_enh = flat_enh

    gain_abs = np.abs(gain_slice)
    gain_lim = max(1e-6, float(np.percentile(gain_abs, 99)))
    ratio_map = np.abs(enh) / (np.abs(inp) + 1e-6)
    ratio_mid = ratio_map[mid]
    ratio_lim = max(2.0, float(np.percentile(ratio_map, 99)))

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 4)

    ax_in = fig.add_subplot(gs[0, 0])
    ax_enh = fig.add_subplot(gs[0, 1])
    ax_gain = fig.add_subplot(gs[0, 2])
    ax_text = fig.add_subplot(gs[0, 3])

    ax_in.imshow(_norm01(inp_slice), cmap='gray', interpolation='bicubic')
    ax_in.set_title('Original input')
    ax_in.axis('off')

    ax_enh.imshow(_norm01(enh_slice), cmap='gray', interpolation='bicubic')
    ax_enh.set_title('Enhanced feature after residual decomposition')
    ax_enh.axis('off')

    im_gain = ax_gain.imshow(gain_slice, cmap='RdBu', vmin=-gain_lim, vmax=gain_lim, interpolation='bicubic')
    ax_gain.set_title('High-frequency residual gain: output - input')
    ax_gain.axis('off')
    fig.colorbar(im_gain, ax=ax_gain, fraction=0.046)

    ax_text.axis('off')
    txt = (
        f'Case: {case}\n'
        f'Layer: {layer}  Band: {band}\n'
        f'Input energy: {stats["input_energy"]:.1f}\n'
        f'Output energy: {stats["output_energy"]:.1f}\n'
        f'Gain energy: {stats["gain_energy"]:.1f}\n'
        f'Band gain ratio: {stats["gain_ratio"]:.2f}\n\n'
        f'Mean input: {inp.mean():.4f}\n'
        f'Mean output: {enh.mean():.4f}'
    )
    ax_text.text(0, 0.5, txt, fontsize=12, va='center')

    ax_diff = fig.add_subplot(gs[1:, 0:2])
    ax_ratio = fig.add_subplot(gs[1, 2])
    ax_hist = fig.add_subplot(gs[1, 3])
    ax_scatter = fig.add_subplot(gs[2, 2:])

    im = ax_diff.imshow(gain_slice, cmap='RdBu', vmin=-gain_lim, vmax=gain_lim, interpolation='bicubic')
    ax_diff.set_title('High-frequency enhancement map')
    ax_diff.axis('off')
    fig.colorbar(im, ax=ax_diff, fraction=0.046)

    ax_ratio.imshow(np.clip(ratio_mid, 0, ratio_lim) / ratio_lim, cmap='inferno', interpolation='bicubic')
    ax_ratio.set_title('Relative residual enhancement (clipped)')
    ax_ratio.axis('off')

    ax_hist.hist(flat_inp ** 2 + 1e-12, bins=200, alpha=0.6, label='input', color='C0')
    ax_hist.hist(flat_enh ** 2 + 1e-12, bins=200, alpha=0.6, label='output', color='C1')
    ax_hist.set_yscale('log')
    ax_hist.set_title('Voxel energy histogram (log)')
    ax_hist.legend()

    ax_scatter.scatter(s_inp, s_enh, s=1, alpha=0.3)
    ax_scatter.set_xlabel('input value')
    ax_scatter.set_ylabel('output value')
    ax_scatter.set_title('Input vs enhanced residual feature (sample)')

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    if spectrum_path is not None:
        fig2 = plt.figure(figsize=(16, 6))
        gs2 = fig2.add_gridspec(1, 4)
        ax_spec1 = fig2.add_subplot(gs2[0, 0])
        ax_spec2 = fig2.add_subplot(gs2[0, 1])
        ax_bar = fig2.add_subplot(gs2[0, 2])
        ax_boundary = fig2.add_subplot(gs2[0, 3])

        inp_spec = _spectrum_features(inp)
        enh_spec = _spectrum_features(enh)

        inp_mag = np.log1p(inp_spec['mag'])
        enh_mag = np.log1p(enh_spec['mag'])

        mid_d = inp_mag.shape[0] // 2
        ax_spec1.imshow(inp_mag[mid_d], cmap='magma', interpolation='bicubic')
        ax_spec1.set_title('Input spectrum |FFT| (log1p)')
        ax_spec1.axis('off')

        ax_spec2.imshow(enh_mag[mid_d], cmap='magma', interpolation='bicubic')
        ax_spec2.set_title('Output spectrum |FFT| (log1p)')
        ax_spec2.axis('off')

        bars = ['high']
        inp_vals = [inp_spec['high_ratio']]
        enh_vals = [enh_spec['high_ratio']]
        x = np.arange(len(bars))
        width = 0.35
        ax_bar.bar(x - width/2, inp_vals, width, label='input')
        ax_bar.bar(x + width/2, enh_vals, width, label='output')
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(bars)
        ax_bar.set_ylabel('High-frequency ratio')
        ax_bar.set_title('Whole-image high-frequency share')
        ax_bar.legend()
        ax_bar.grid(axis='y', alpha=0.3)

        # boundary-focused enhancement map: where the high-frequency gain concentrates
        boundary_metrics_in = _boundary_focus_ratio(inp)
        boundary_metrics_out = _boundary_focus_ratio(enh)
        boundary_gain = boundary_metrics_out['boundary_energy_ratio'] - boundary_metrics_in['boundary_energy_ratio']
        ax_boundary.axis('off')
        ax_boundary.text(
            0.0, 0.95,
            f'Boundary shell width: 8\n'
            f'Input boundary energy ratio: {boundary_metrics_in["boundary_energy_ratio"]:.3f}\n'
            f'Output boundary energy ratio: {boundary_metrics_out["boundary_energy_ratio"]:.3f}\n'
            f'Boundary delta: {boundary_gain:+.3f}\n\n'
            f'Input HF ratio: {inp_spec["high_ratio"]:.3f}\n'
            f'Output HF ratio: {enh_spec["high_ratio"]:.3f}\n'
            f'HF delta: {enh_spec["high_ratio"] - inp_spec["high_ratio"]:+.3f}',
            va='top', fontsize=10
        )
        ax_boundary.set_title('Boundary-focused enhancement')

        plt.tight_layout()
        fig2.savefig(spectrum_path, dpi=200)
        plt.close(fig2)

    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', nargs='*')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--out_dir', default='paper_figs_final_center64')
    parser.add_argument('--data_root', default=DATA_ROOT_DEFAULT)
    parser.add_argument('--max_cases', type=int, default=0, help='0 means export all cases')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--gain', type=float, default=1.0, help='Scalar gain applied to the enhanced response before plotting')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    data = torch.load(args.ckpt, map_location='cpu') if os.path.isfile(args.ckpt) else None
    st = data.get('model_state', data) if data is not None else None

    if args.cases:
        cases = args.cases
    else:
        files = [f for f in os.listdir(args.data_root) if f.endswith('.npy')]
        files.sort()
        cases = [os.path.splitext(f)[0] for f in files]
        if args.max_cases and args.max_cases > 0:
            cases = cases[:args.max_cases]

    if not cases:
        raise RuntimeError(f'No cases found under {args.data_root}')

    band_rows = []
    spectrum_rows = []
    case_rows = []
    for case in cases:
        vol = load_case(args.data_root, case)
        vol_crop = center_crop(vol, crop_size=64)
        in_ch = vol.shape[0]
        model = Double_RWKV_MedNeXt(in_channels=in_ch, n_channels=16, n_classes=2, exp_r=2, kernel_size=3, fbm_k_list=[2, 4, 8], fbm_learnable=True)
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

        case_high_in = []
        case_high_out = []
        case_bnd_in = []
        case_bnd_out = []

        for li, layer in enumerate(['f0', 'f1']):
            feat = feats_med[li]
            fbm = getattr(enc, 'fbm0' if layer == 'f0' else 'fbm1', None)
            if fbm is None:
                continue

            b, c, d, h, w = feat.shape
            x_fft = torch.fft.rfftn(feat, s=(d, h, w), dim=(-3, -2, -1), norm='ortho')
            band_masks = fbm._get_band_masks(d, h, w // 2 + 1)

            pre_x = feat.clone()
            att_feat = feat
            for idx, mask in enumerate(band_masks, start=1):
                mask = mask.to(args.device)
                low_part = torch.fft.irfftn(x_fft * mask, s=(d, h, w), dim=(-3, -2, -1), norm='ortho')
                high_part = pre_x - low_part
                pre_x = low_part

                if idx - 1 < len(fbm.freq_weight_conv_list):
                    fw = fbm.freq_weight_conv_list[idx - 1](att_feat)
                    fw = fbm._activate(fw)
                    grp = fbm.spatial_group
                    tmp = fw.reshape(b, grp, -1, d, h, w) * high_part.reshape(b, grp, -1, d, h, w)
                    enhanced = tmp.reshape(b, -1, d, h, w)
                else:
                    enhanced = high_part

                input_np = high_part.detach().cpu().numpy()[0]
                enhanced_np = (enhanced * args.gain).detach().cpu().numpy()[0]

                input_mean = input_np.mean(axis=0)
                output_mean = enhanced_np.mean(axis=0)
                gain_mean = output_mean - input_mean
                e_in = _band_energy(input_mean)
                e_out = _band_energy(output_mean)
                e_gain = _band_energy(gain_mean)
                gain_ratio = e_gain / (e_in + 1e-12)

                stats = {
                    'input_energy': e_in,
                    'output_energy': e_out,
                    'gain_energy': e_gain,
                    'gain_ratio': gain_ratio,
                }
                band_rows.append({
                    'case': case,
                    'layer': layer,
                    'band': idx,
                    'input_energy': e_in,
                    'output_energy': e_out,
                    'gain_energy': e_gain,
                    'gain_ratio': gain_ratio,
                })

                out_path = os.path.join(args.out_dir, f'final_center64_{case}_{layer}_band{idx}.png')
                spectrum_path = os.path.join(args.out_dir, f'final_center64_{case}_{layer}_band{idx}_spectrum.png')
                make_figure(case, layer, idx, input_np, enhanced_np, out_path, stats, spectrum_path=spectrum_path)
                print('Saved', out_path, f'gain_ratio={gain_ratio:.3f} (output/input energy ratio={(e_out / (e_in + 1e-12)):.3f})')
                spec_in = _spectrum_features(input_mean)
                spec_out = _spectrum_features(output_mean)
                boundary_in = _boundary_focus_ratio(input_mean)
                boundary_out = _boundary_focus_ratio(output_mean)
                spectrum_rows.append({
                    'case': case,
                    'layer': layer,
                    'band': idx,
                    'input_high_ratio': spec_in['high_ratio'],
                    'output_high_ratio': spec_out['high_ratio'],
                    'high_ratio_delta': spec_out['high_ratio'] - spec_in['high_ratio'],
                    'input_boundary_energy_ratio': boundary_in['boundary_energy_ratio'],
                    'output_boundary_energy_ratio': boundary_out['boundary_energy_ratio'],
                    'boundary_energy_delta': boundary_out['boundary_energy_ratio'] - boundary_in['boundary_energy_ratio'],
                })

                case_high_in.append(spec_in['high_ratio'])
                case_high_out.append(spec_out['high_ratio'])
                case_bnd_in.append(boundary_in['boundary_energy_ratio'])
                case_bnd_out.append(boundary_out['boundary_energy_ratio'])

        if case_high_in:
            case_rows.append({
                'case': case,
                'mean_input_high_ratio': float(np.mean(case_high_in)),
                'mean_output_high_ratio': float(np.mean(case_high_out)),
                'mean_high_ratio_delta': float(np.mean(np.array(case_high_out) - np.array(case_high_in))),
                'mean_input_boundary_energy_ratio': float(np.mean(case_bnd_in)),
                'mean_output_boundary_energy_ratio': float(np.mean(case_bnd_out)),
                'mean_boundary_energy_delta': float(np.mean(np.array(case_bnd_out) - np.array(case_bnd_in))),
            })

    if band_rows:
        csv_path = os.path.join(args.out_dir, 'band_gain_summary.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['case', 'layer', 'band', 'input_energy', 'output_energy', 'gain_energy', 'gain_ratio'])
            writer.writeheader()
            writer.writerows(band_rows)
        print('Saved summary CSV:', csv_path)

        spec_csv = os.path.join(args.out_dir, 'band_spectrum_summary.csv')
        with open(spec_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['case', 'layer', 'band', 'input_high_ratio', 'output_high_ratio', 'high_ratio_delta', 'input_boundary_energy_ratio', 'output_boundary_energy_ratio', 'boundary_energy_delta'])
            writer.writeheader()
            writer.writerows(spectrum_rows)
        print('Saved spectrum CSV:', spec_csv)

        # whole-image aggregation across all band samples
        all_in = np.array([r['input_high_ratio'] for r in spectrum_rows], dtype=np.float32)
        all_out = np.array([r['output_high_ratio'] for r in spectrum_rows], dtype=np.float32)
        all_b_in = np.array([r['input_boundary_energy_ratio'] for r in spectrum_rows], dtype=np.float32)
        all_b_out = np.array([r['output_boundary_energy_ratio'] for r in spectrum_rows], dtype=np.float32)
        whole_csv = os.path.join(args.out_dir, 'whole_image_high_summary.csv')
        with open(whole_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['metric', 'value'])
            writer.writeheader()
            writer.writerow({'metric': 'mean_input_high_ratio', 'value': float(all_in.mean())})
            writer.writerow({'metric': 'mean_output_high_ratio', 'value': float(all_out.mean())})
            writer.writerow({'metric': 'mean_high_ratio_delta', 'value': float((all_out - all_in).mean())})
            writer.writerow({'metric': 'frac_output_gt_input', 'value': float((all_out > all_in).mean())})
            writer.writerow({'metric': 'mean_input_boundary_energy_ratio', 'value': float(all_b_in.mean())})
            writer.writerow({'metric': 'mean_output_boundary_energy_ratio', 'value': float(all_b_out.mean())})
            writer.writerow({'metric': 'mean_boundary_energy_delta', 'value': float((all_b_out - all_b_in).mean())})
        print('Saved whole-image summary CSV:', whole_csv)

        case_csv = os.path.join(args.out_dir, 'case_overview_summary.csv')
        with open(case_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['case', 'mean_input_high_ratio', 'mean_output_high_ratio', 'mean_high_ratio_delta', 'mean_input_boundary_energy_ratio', 'mean_output_boundary_energy_ratio', 'mean_boundary_energy_delta'])
            writer.writeheader()
            writer.writerows(case_rows)
        print('Saved case overview CSV:', case_csv)

        grouped = {}
        for row in band_rows:
            grouped.setdefault((row['layer'], row['band']), []).append(row['gain_ratio'])
        for (layer, band), vals in sorted(grouped.items()):
            print(f'[SUMMARY] {layer} band{band}: mean_gain_ratio={np.mean(vals):.3f}, min={np.min(vals):.3f}, max={np.max(vals):.3f}, n={len(vals)}')


if __name__ == '__main__':
    main()
