#!/usr/bin/env python3
"""Plot 3D frequency-domain energy spectrum: frequency radius vs normalized energy.

This script now supports two modes:
1) raw volume compare (legacy)
2) FBM feature compare from a trained Double_RWKV_MedNeXt checkpoint

For the FBM mode, we extract the true pre/post FBM features from the encoder
forward pass and compare their radial spectra on the same figure.
"""
import argparse
import os
from typing import Tuple, Optional
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch

from nnunet_mednext.network_architecture.le_networks.Double_RWKV_MedNeXt import Double_RWKV_MedNeXt


def _load_volume(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 4:
        arr = arr.mean(axis=0)
    elif arr.ndim != 3:
        raise RuntimeError(f"Expected 3D or 4D array, got shape {arr.shape} from {path}")
    return arr.astype(np.float32)


def _radial_energy_spectrum(vol3d: np.ndarray, num_bins: int = 100, highpass_remove_mean: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bin_centers, normalized_energy, raw_avg_energy)."""
    if vol3d.ndim != 3:
        raise ValueError(f"vol3d must be 3D, got {vol3d.shape}")

    x = vol3d.astype(np.float32)
    if highpass_remove_mean:
        x = x - x.mean()

    fft = np.fft.fftn(x, norm='ortho')
    power = np.abs(fft) ** 2

    d, h, w = x.shape
    fd, fh, fw = np.meshgrid(np.fft.fftfreq(d), np.fft.fftfreq(h), np.fft.fftfreq(w), indexing='ij')
    radius = np.sqrt(fd ** 2 + fh ** 2 + fw ** 2).reshape(-1)
    power = power.reshape(-1)

    max_r = float(radius.max())
    edges = np.linspace(0.0, max_r + 1e-12, num_bins + 1)
    bin_idx = np.digitize(radius, edges) - 1
    bin_idx = np.clip(bin_idx, 0, num_bins - 1)

    energy = np.zeros(num_bins, dtype=np.float64)
    counts = np.zeros(num_bins, dtype=np.float64)
    for i, p in zip(bin_idx, power):
        energy[i] += float(p)
        counts[i] += 1.0

    avg_energy = energy / np.maximum(counts, 1.0)
    normalized_energy = avg_energy / (avg_energy.sum() + 1e-12)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, normalized_energy, avg_energy


def _find_case_path(data_root: str, case_id: str) -> str:
    exact = os.path.join(data_root, f"{case_id}.npy")
    if os.path.isfile(exact):
        return exact
    matches = [os.path.join(data_root, f) for f in os.listdir(data_root) if f.startswith(case_id) and f.endswith('.npy')]
    if not matches:
        raise FileNotFoundError(f"No .npy file found for case {case_id} under {data_root}")
    return sorted(matches)[0]


def _load_input_and_output(input_path: str, output_path: Optional[str], data_root: Optional[str]) -> Tuple[np.ndarray, np.ndarray, str, str]:
    if data_root is not None and not input_path.endswith('.npy'):
        input_path = _find_case_path(data_root, input_path)
    if output_path is None:
        output_path = input_path
    elif data_root is not None and not output_path.endswith('.npy'):
        output_path = _find_case_path(data_root, output_path)

    if not os.path.isfile(input_path):
        raise FileNotFoundError(input_path)
    if not os.path.isfile(output_path):
        raise FileNotFoundError(output_path)
    return _load_volume(input_path), _load_volume(output_path), input_path, output_path


def _high_ratio(vol3d: np.ndarray, high_cut: float = 0.30) -> float:
    x = vol3d.astype(np.float32) - vol3d.astype(np.float32).mean()
    fft = np.fft.fftn(x, norm='ortho')
    power = np.abs(fft) ** 2
    d, h, w = x.shape
    fd, fh, fw = np.meshgrid(np.fft.fftfreq(d), np.fft.fftfreq(h), np.fft.fftfreq(w), indexing='ij')
    radius = np.sqrt(fd ** 2 + fh ** 2 + fw ** 2)
    high = float(np.sum(power[radius >= float(high_cut)]))
    total = float(np.sum(power) + 1e-12)
    return high / total


def _load_model(ckpt: str, in_channels: int = 1, n_channels: int = 16) -> Double_RWKV_MedNeXt:
    st = torch.load(ckpt, map_location='cpu') if os.path.isfile(ckpt) else None
    sd = st.get('model_state', st) if st is not None else None

    model = Double_RWKV_MedNeXt(
        in_channels=in_channels,
        n_channels=n_channels,
        n_classes=2,
        exp_r=2,
        kernel_size=3,
        fbm_k_list=[2, 4, 8],
        fbm_learnable=True,
    )
    if sd is not None:
        model.load_state_dict(sd, strict=False)
    model.eval()
    print(f'Loaded checkpoint with in_channels={in_channels}')
    return model


def _extract_fbm_pair(model: Double_RWKV_MedNeXt, vol: np.ndarray, layer: str = 'f1', device: str = 'cpu'):
    """Return (pre_fbm_feat, post_fbm_feat, feature_name)."""
    if vol.ndim == 3:
        vol = np.stack([vol, vol], axis=0)
    x = torch.from_numpy(vol[None]).float().to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        try:
            feats_med = model.encoder.mednext_enc.forward_encoder(x)
        except Exception:
            feats_med = model.encoder(x)
    enc = model.encoder

    if layer not in ('f0', 'f1'):
        raise ValueError("layer must be 'f0' or 'f1'")
    feat = feats_med[0] if layer == 'f0' else feats_med[1]
    fbm = getattr(enc, 'fbm0' if layer == 'f0' else 'fbm1', None)
    if fbm is None:
        raise RuntimeError(f"FBM module not found for {layer}")
    with torch.no_grad():
        post = fbm(feat, att_feat=feat)
    return feat.detach().cpu().numpy()[0], post.detach().cpu().numpy()[0]


def _save_feature_arrays(out_dir: str, case_tag: str, layer: str, pre_feat: np.ndarray, post_feat: np.ndarray) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    pre_path = os.path.join(out_dir, f'{case_tag}_{layer}_pre.npy')
    post_path = os.path.join(out_dir, f'{case_tag}_{layer}_post.npy')
    np.save(pre_path, pre_feat)
    np.save(post_path, post_feat)
    return pre_path, post_path


def _collapse_feature_for_display(feat: np.ndarray) -> np.ndarray:
    """Collapse a CxDxHxW feature tensor to a single 3D volume for visualization."""
    if feat.ndim == 4:
        return feat.mean(axis=0)
    if feat.ndim == 3:
        return feat
    raise ValueError(f'Unsupported feature ndim {feat.ndim}')


def _save_feature_triplet_figure(input_feat: np.ndarray, output_feat: np.ndarray, out_png: str, title: Optional[str], case_tag: str, layer: str):
    inp = _collapse_feature_for_display(input_feat)
    out = _collapse_feature_for_display(output_feat)
    diff = out - inp

    mid = inp.shape[0] // 2
    inp_slice = inp[mid]
    out_slice = out[mid]
    diff_slice = diff[mid]

    lim = max(1e-6, float(np.percentile(np.abs(diff_slice), 99)))

    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    ax1.imshow(inp_slice, cmap='gray', interpolation='bicubic')
    ax1.set_title('Input feature')
    ax1.axis('off')

    ax2.imshow(out_slice, cmap='gray', interpolation='bicubic')
    ax2.set_title('Output feature')
    ax2.axis('off')

    im = ax3.imshow(diff_slice, cmap='RdBu', vmin=-lim, vmax=lim, interpolation='bicubic')
    ax3.set_title('Difference (output - input)')
    ax3.axis('off')
    fig.colorbar(im, ax=ax3, fraction=0.046)

    fig.suptitle(title or f'Feature enhancement comparison ({case_tag}, {layer})')
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def _spectrum_features(vol: np.ndarray) -> dict:
    """Compute and return the magnitude and phase spectra of a 3D volume."""
    fft = np.fft.fftn(vol, norm='ortho')
    mag = np.abs(fft)
    phase = np.angle(fft)
    return {'mag': mag, 'phase': phase}


def _save_spectrum_triplet_figure(input_feat: np.ndarray, output_feat: np.ndarray, out_png: str, title: Optional[str], hf_in: float, hf_out: float, case_tag: str, layer: str):
    inp = _collapse_feature_for_display(input_feat)
    out = _collapse_feature_for_display(output_feat)
    diff = out - inp

    inp_spec = _spectrum_features(inp)
    out_spec = _spectrum_features(out)
    diff_spec = _spectrum_features(diff)

    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    mid_in = inp_spec['mag'].shape[0] // 2
    mid_out = out_spec['mag'].shape[0] // 2
    mid_diff = diff_spec['mag'].shape[0] // 2

    ax1.imshow(np.log1p(inp_spec['mag'])[mid_in], cmap='magma', interpolation='bicubic')
    ax1.set_title('Input spectrum |FFT| (log1p)')
    ax1.axis('off')

    ax2.imshow(np.log1p(out_spec['mag'])[mid_out], cmap='magma', interpolation='bicubic')
    ax2.set_title('Output spectrum |FFT| (log1p)')
    ax2.axis('off')

    ax3.imshow(np.log1p(diff_spec['mag'])[mid_diff], cmap='magma', interpolation='bicubic')
    ax3.set_title('Difference spectrum |FFT| (log1p)')
    ax3.axis('off')

    fig.suptitle(title or f'Frequency-domain enhancement ({case_tag}, {layer})')
    fig.text(0.5, 0.02, f'HF ratio in={hf_in:.3f}   out={hf_out:.3f}   Δ={hf_out-hf_in:+.3f}', ha='center')
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def _save_compare_plot(radii_in, norm_in, radii_out, norm_out, out_png: str, title: Optional[str], hf_in: float, hf_out: float, input_label='input', output_label='output'):
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.plot(radii_in, norm_in, lw=2.2, color='C0', label=input_label)
    ax.plot(radii_out, norm_out, lw=2.2, color='C1', label=output_label)
    ax.set_xlabel('Frequency radius')
    ax.set_ylabel('Normalized energy')
    ax.set_title(title or 'Frequency-domain energy spectrum')
    ax.grid(True, alpha=0.25)
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.text(
        0.98, 0.95,
        f'HF ratio in:  {hf_in:.3f}\nHF ratio out: {hf_out:.3f}\nΔ: {hf_out - hf_in:+.3f}',
        transform=ax.transAxes,
        ha='right', va='top', fontsize=10,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=4)
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to a .npy file or a case id if --data_root is provided')
    parser.add_argument('--output', default=None, help='Optional path/case id for the output volume to compare against input')
    parser.add_argument('--data_root', default=None, help='Preprocessed data root used to resolve case id')
    parser.add_argument('--ckpt', default=None, help='Trained checkpoint for FBM feature comparison mode')
    parser.add_argument('--layer', default='f1', choices=['f0', 'f1'], help='Which FBM layer to compare in feature mode')
    parser.add_argument('--out_png', default='fd_energy_spectrum_compare.png')
    parser.add_argument('--out_csv', default='fd_energy_spectrum_compare.csv')
    parser.add_argument('--num_bins', type=int, default=100)
    parser.add_argument('--no_remove_mean', action='store_true', help='Keep the mean component instead of subtracting it')
    parser.add_argument('--title', default=None)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--feature_mode', action='store_true', help='Compare FBM pre/post features instead of raw volumes')
    parser.add_argument('--feature_triplet', action='store_true', help='Save input/output/difference feature maps instead of spectral curves')
    parser.add_argument('--batch_all', action='store_true', help='Export feature-mode spectra for all cases under --data_root')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_png)) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or '.', exist_ok=True)

    if args.batch_all:
        if not args.feature_mode:
            raise ValueError('--batch_all requires --feature_mode')
        if args.ckpt is None or args.data_root is None:
            raise ValueError('--batch_all requires --ckpt and --data_root')
        case_ids = sorted({f.split('.npy')[0] for f in os.listdir(args.data_root) if f.endswith('.npy') and f.startswith('ESO_TJ_')})
        if not case_ids:
            raise RuntimeError(f'No ESO_TJ_ cases found under {args.data_root}')
        for case_id in case_ids:
            for layer in ('f0', 'f1'):
                out_png = os.path.join(os.path.dirname(os.path.abspath(args.out_png)) or '.', f'{case_id}_{layer}_fbm_feature_spectrum_compare.png')
                out_csv = os.path.join(os.path.dirname(os.path.abspath(args.out_csv)) or '.', f'{case_id}_{layer}_fbm_feature_spectrum_compare.csv')
                cmd_args = [
                    '--feature_mode', '--input', case_id, '--data_root', args.data_root, '--ckpt', args.ckpt,
                    '--layer', layer, '--out_png', out_png, '--out_csv', out_csv, '--num_bins', str(args.num_bins),
                    '--device', args.device
                ]
                if args.no_remove_mean:
                    cmd_args.append('--no_remove_mean')
                if args.title is not None:
                    cmd_args.extend(['--title', args.title])
                # Re-enter main() logic by executing the same script in a subprocess for each case/layer.
                os.system(' '.join([sys.executable, os.path.abspath(__file__)] + [f'"{a}"' if ' ' in a else a for a in cmd_args]))
        return

    if args.feature_triplet:
        if args.ckpt is None:
            raise ValueError('--feature_triplet requires --ckpt')
        if args.data_root is None and args.input.endswith('.npy') is False:
            raise ValueError('--feature_triplet with case id requires --data_root')
        path = _find_case_path(args.data_root, args.input) if args.data_root is not None and not args.input.endswith('.npy') else args.input
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        vol = _load_volume(path)
        vol = np.stack([vol, vol], axis=0)
        model = _load_model(args.ckpt, in_channels=vol.shape[0])
        inp_feat, out_feat = _extract_fbm_pair(model, vol, layer=args.layer, device=args.device)
        case_tag = os.path.splitext(os.path.basename(path))[0]
        title = args.title or f'Input vs Output Feature Enhancement ({args.layer})'
        _save_feature_triplet_figure(inp_feat, out_feat, args.out_png, title, case_tag, args.layer)
        _save_spectrum_triplet_figure(inp_feat, out_feat, args.out_png.replace('.png', '_spectrum.png'), title, _high_ratio(_collapse_feature_for_display(inp_feat)), _high_ratio(_collapse_feature_for_display(out_feat)), case_tag, args.layer)
        print(f'Saved feature triplet figure: {args.out_png}')
        spectrum_out = args.out_png.replace('.png', '_spectrum.png')
        print(f'Saved spectrum triplet figure: {spectrum_out}')
        return

    if args.feature_mode:
        if args.ckpt is None:
            raise ValueError('--feature_mode requires --ckpt')
        if args.data_root is None and args.input.endswith('.npy') is False:
            raise ValueError('--feature_mode with case id requires --data_root')
        path = _find_case_path(args.data_root, args.input) if args.data_root is not None and not args.input.endswith('.npy') else args.input
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        vol = _load_volume(path)
        # The checkpoint expects a 2-channel stem. Duplicate the scalar volume into 2 channels
        # so we can run the FBM branch and inspect its true pre/post feature spectra.
        vol = np.stack([vol, vol], axis=0)
        model = _load_model(args.ckpt, in_channels=vol.shape[0])
        inp_feat, out_feat = _extract_fbm_pair(model, vol, layer=args.layer, device=args.device)
        print(f'Feature mode shapes: pre={inp_feat.shape}, post={out_feat.shape}')
        pre_vol = inp_feat.mean(axis=0)
        post_vol = out_feat.mean(axis=0)
        radii_in, norm_in, raw_in = _radial_energy_spectrum(pre_vol, num_bins=args.num_bins, highpass_remove_mean=not args.no_remove_mean)
        radii_out, norm_out, raw_out = _radial_energy_spectrum(post_vol, num_bins=args.num_bins, highpass_remove_mean=not args.no_remove_mean)
        hf_in = _high_ratio(pre_vol)
        hf_out = _high_ratio(post_vol)
        label_in = f'{args.layer} pre-FBM'
        label_out = f'{args.layer} post-FBM'
        title = args.title or f'FBM feature spectrum ({args.layer})'
        out_note = f'feature_mode_{args.layer}'
        case_tag = os.path.splitext(os.path.basename(path))[0]
        feature_dir = os.path.join(os.path.dirname(os.path.abspath(args.out_png)) or '.', 'fbm_feature_cache')
        pre_path, post_path = _save_feature_arrays(feature_dir, case_tag, args.layer, inp_feat, out_feat)
        print(f'Saved feature arrays: {pre_path}, {post_path}')
    else:
        inp, out, input_path, output_path = _load_input_and_output(args.input, args.output, args.data_root)
        radii_in, norm_in, raw_in = _radial_energy_spectrum(inp, num_bins=args.num_bins, highpass_remove_mean=not args.no_remove_mean)
        radii_out, norm_out, raw_out = _radial_energy_spectrum(out, num_bins=args.num_bins, highpass_remove_mean=not args.no_remove_mean)
        hf_in = _high_ratio(inp)
        hf_out = _high_ratio(out)
        label_in = 'input'
        label_out = 'output'
        title = args.title or 'Frequency-domain energy spectrum'
        out_note = 'raw_volume_compare'
        if not np.allclose(radii_in, radii_out):
            raise RuntimeError('Input and output radius bins do not match')

    np.savetxt(
        args.out_csv,
        np.c_[radii_in, norm_in, norm_out, raw_in, raw_out],
        delimiter=',',
        header='radius,input_normalized_energy,output_normalized_energy,input_raw_avg_energy,output_raw_avg_energy',
        comments=''
    )

    print(f'HF ratio in:  {hf_in:.6f}')
    print(f'HF ratio out: {hf_out:.6f}')
    print(f'HF delta:     {hf_out - hf_in:+.6f}')

    _save_compare_plot(radii_in, norm_in, radii_out, norm_out, args.out_png, title, hf_in, hf_out, input_label=label_in, output_label=label_out)

    print(f'Saved spectrum plot: {args.out_png}')
    print(f'Saved spectrum CSV:  {args.out_csv}')
    print(f'Mode: {out_note}')
    if not args.feature_mode:
        print(f'Input:  {input_path}')
        print(f'Output: {output_path}')


if __name__ == '__main__':
    main()
