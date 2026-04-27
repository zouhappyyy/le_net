import argparse
import csv
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from nnunet_mednext.training.model_restore import restore_model


SUPPORTED_SUFFIXES = (".npy", ".npz", ".nii", ".nii.gz")


def _is_supported(path: str) -> bool:
    lower = path.lower()
    return any(lower.endswith(s) for s in SUPPORTED_SUFFIXES)


def _basename(path: str) -> str:
    name = os.path.basename(path)
    if name.endswith(".nii.gz"):
        return name[:-7]
    return os.path.splitext(name)[0]


def _collect_files(image_path: Optional[str], image_dir: Optional[str]) -> List[str]:
    if bool(image_path) == bool(image_dir):
        raise ValueError("Specify exactly one of --image or --image_dir")
    if image_path is not None:
        if not os.path.isfile(image_path):
            raise FileNotFoundError(image_path)
        if not _is_supported(image_path):
            raise ValueError(f"Unsupported input file: {image_path}")
        return [image_path]

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(image_dir)
    files = []
    for name in sorted(os.listdir(image_dir)):
        full = os.path.join(image_dir, name)
        if os.path.isfile(full) and _is_supported(full):
            files.append(full)
    if not files:
        raise FileNotFoundError(f"No supported files found under {image_dir}")
    return files


def _load_volume(path: str) -> np.ndarray:
    lower = path.lower()
    if lower.endswith(".npy"):
        arr = np.load(path)
    elif lower.endswith(".npz"):
        npz = np.load(path)
        arr = npz["data"] if "data" in npz else npz[npz.files[0]]
    else:
        raise ValueError(f"This script currently supports .npy/.npz inputs best, got {path}")

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        return arr[None]
    if arr.ndim == 4:
        return arr
    if arr.ndim == 5 and arr.shape[0] == 1:
        return arr[0]
    raise RuntimeError(f"Unexpected input shape {arr.shape} in {path}")


def _resolve_checkpoint_paths(path: str, checkpoint_name: str) -> Tuple[str, str]:
    if os.path.isfile(path):
        if path.endswith(".model.pkl"):
            model_path = path[:-4]
            pkl_path = path
        elif path.endswith(".model"):
            model_path = path
            pkl_path = path + ".pkl"
        else:
            raise ValueError("Checkpoint file must end with .model or .model.pkl")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)
        if not os.path.isfile(pkl_path):
            raise FileNotFoundError(pkl_path)
        return model_path, pkl_path

    if not os.path.isdir(path):
        raise FileNotFoundError(path)

    names = [checkpoint_name]
    if not checkpoint_name.endswith(".model"):
        names.append(checkpoint_name + ".model")
    for name in names:
        model_path = os.path.join(path, name)
        pkl_path = model_path + ".pkl"
        if os.path.isfile(model_path) and os.path.isfile(pkl_path):
            return model_path, pkl_path
    raise FileNotFoundError(f"Could not find {checkpoint_name}(.pkl) under {path}")


def _capture_encoder_outputs(stage_store: "OrderedDict[str, torch.Tensor]"):
    def _hook(_module, _inputs, output):
        if not isinstance(output, (list, tuple)) or len(output) != 5:
            return
        for name, tensor in zip(["enc0", "enc1", "enc2", "enc3", "bottleneck"], output):
            if torch.is_tensor(tensor):
                stage_store[name] = tensor.detach().cpu()

    return _hook


def _load_network(checkpoint_path: str, checkpoint_name: str, device: str):
    model_path, pkl_path = _resolve_checkpoint_paths(checkpoint_path, checkpoint_name)
    trainer = restore_model(pkl_path, checkpoint=model_path, train=False)
    network = trainer.network.to(device).eval()
    return network, model_path


def _radial_spectrum_from_feature(feat: np.ndarray, num_bins: int = 80, remove_mean: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    if feat.ndim == 3:
        feat = feat[None]
    if feat.ndim != 4:
        raise ValueError(f"Feature must be [C,D,H,W] or [D,H,W], got {feat.shape}")

    feat = feat.astype(np.float32)
    if remove_mean:
        feat = feat - feat.mean(axis=(1, 2, 3), keepdims=True)

    c, d, h, w = feat.shape
    fft = np.fft.fftn(feat, axes=(-3, -2, -1), norm="ortho")
    power = np.abs(fft) ** 2
    power = power.mean(axis=0)  # average across channels -> [D,H,W]

    fd, fh, fw = np.meshgrid(
        np.fft.fftfreq(d),
        np.fft.fftfreq(h),
        np.fft.fftfreq(w),
        indexing="ij",
    )
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
    norm_energy = avg_energy / (avg_energy.sum() + 1e-12)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, norm_energy


def _high_freq_ratio(radii: np.ndarray, norm_energy: np.ndarray, high_cut: float = 0.30) -> float:
    mask = radii >= float(high_cut)
    return float(norm_energy[mask].sum())


def _prepare_input_tensor(vol: np.ndarray, input_channel: int, device: str) -> Tuple[torch.Tensor, np.ndarray]:
    if input_channel >= vol.shape[0]:
        raise ValueError(f"input_channel={input_channel} out of range for volume with shape {vol.shape}")
    image = vol[input_channel:input_channel + 1]
    x = torch.from_numpy(image[None].astype(np.float32)).to(device)
    return x, image[0]


def _save_case_plot(
    out_png: str,
    case_id: str,
    radii: np.ndarray,
    input_spec: np.ndarray,
    enc0_spec: np.ndarray,
    enc1_spec: np.ndarray,
    hf_input: float,
    hf_enc0: float,
    hf_enc1: float,
):
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    ax.plot(radii, input_spec, lw=2.2, label="input", color="C0")
    ax.plot(radii, enc0_spec, lw=2.2, label="enc0", color="C1")
    ax.plot(radii, enc1_spec, lw=2.2, label="enc1", color="C2")
    ax.set_xlabel("Frequency radius")
    ax.set_ylabel("Normalized energy")
    ax.set_title(f"Encoder spectrum shift: {case_id}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax.text(
        0.98,
        0.97,
        f"HF input: {hf_input:.4f}\nHF enc0: {hf_enc0:.4f}\nHF enc1: {hf_enc1:.4f}\nΔ(enc0-input): {hf_enc0-hf_input:+.4f}\nΔ(enc1-input): {hf_enc1-hf_input:+.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=4),
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def analyze_case(
    network: torch.nn.Module,
    image_path: str,
    out_dir: str,
    input_channel: int,
    num_bins: int,
    high_cut: float,
    device: str,
) -> Dict[str, float]:
    vol = _load_volume(image_path)
    x, input_vol = _prepare_input_tensor(vol, input_channel=input_channel, device=device)

    stage_store: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    inner = network.net if hasattr(network, "net") else network
    if not hasattr(inner, "encoder"):
        raise RuntimeError("Loaded model does not expose an encoder module for this analysis.")
    handle = inner.encoder.register_forward_hook(_capture_encoder_outputs(stage_store))
    try:
        with torch.no_grad():
            _ = network(x)
    finally:
        handle.remove()

    if "enc0" not in stage_store or "enc1" not in stage_store:
        raise RuntimeError(f"Failed to capture enc0/enc1 for {image_path}")

    enc0 = stage_store["enc0"].numpy()[0]
    enc1 = stage_store["enc1"].numpy()[0]

    radii, input_spec = _radial_spectrum_from_feature(input_vol, num_bins=num_bins)
    _, enc0_spec = _radial_spectrum_from_feature(enc0, num_bins=num_bins)
    _, enc1_spec = _radial_spectrum_from_feature(enc1, num_bins=num_bins)

    hf_input = _high_freq_ratio(radii, input_spec, high_cut=high_cut)
    hf_enc0 = _high_freq_ratio(radii, enc0_spec, high_cut=high_cut)
    hf_enc1 = _high_freq_ratio(radii, enc1_spec, high_cut=high_cut)

    case_id = _basename(image_path)
    case_dir = os.path.join(out_dir, case_id)
    os.makedirs(case_dir, exist_ok=True)
    np.save(os.path.join(case_dir, "enc0_feature.npy"), enc0)
    np.save(os.path.join(case_dir, "enc1_feature.npy"), enc1)
    np.save(os.path.join(case_dir, "input_channel.npy"), input_vol)

    csv_path = os.path.join(case_dir, f"{case_id}_encoder_spectrum.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["radius", "input_energy", "enc0_energy", "enc1_energy"])
        for r, a, b, c in zip(radii, input_spec, enc0_spec, enc1_spec):
            writer.writerow([f"{r:.8f}", f"{a:.10f}", f"{b:.10f}", f"{c:.10f}"])

    _save_case_plot(
        out_png=os.path.join(case_dir, f"{case_id}_encoder_spectrum_compare.png"),
        case_id=case_id,
        radii=radii,
        input_spec=input_spec,
        enc0_spec=enc0_spec,
        enc1_spec=enc1_spec,
        hf_input=hf_input,
        hf_enc0=hf_enc0,
        hf_enc1=hf_enc1,
    )

    return {
        "case_id": case_id,
        "hf_input": hf_input,
        "hf_enc0": hf_enc0,
        "hf_enc1": hf_enc1,
        "delta_enc0_minus_input": hf_enc0 - hf_input,
        "delta_enc1_minus_input": hf_enc1 - hf_input,
        "delta_enc1_minus_enc0": hf_enc1 - hf_enc0,
    }


def _write_summary(summary_path: str, rows: Sequence[Dict[str, float]]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    mean_row = {"case_id": "MEAN"}
    for key in fieldnames[1:]:
        mean_row[key] = float(np.mean([float(r[key]) for r in rows]))

    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(mean_row)


def _build_argparser():
    parser = argparse.ArgumentParser(
        description="Analyze radial frequency energy spectrum shifts after the first two encoder stages."
    )
    parser.add_argument("--checkpoint_path", required=True, type=str, help="Path to a fold directory, .model file, or .model.pkl file.")
    parser.add_argument("--checkpoint_name", default="model_best", type=str, help="Checkpoint basename when checkpoint_path is a directory.")
    parser.add_argument("--image", default=None, type=str, help="Single .npy/.npz case file.")
    parser.add_argument("--image_dir", default=None, type=str, help="Directory containing .npy/.npz case files.")
    parser.add_argument("--out_dir", default="./encoder_frequency_spectrum", type=str, help="Output directory.")
    parser.add_argument("--input_channel", default=0, type=int, help="Input channel index for multi-channel arrays.")
    parser.add_argument("--num_bins", default=80, type=int, help="Number of radial spectrum bins.")
    parser.add_argument("--high_cut", default=0.30, type=float, help="Frequency radius threshold for high-frequency ratio.")
    parser.add_argument("--device", default=None, type=str, choices=["cpu", "cuda", None], help="Execution device.")
    return parser


def main():
    args = _build_argparser().parse_args()
    device = args.device if args.device not in (None, "None") else ("cuda" if torch.cuda.is_available() else "cpu")
    files = _collect_files(image_path=args.image, image_dir=args.image_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    network, model_path = _load_network(args.checkpoint_path, args.checkpoint_name, device=device)
    print(f"Checkpoint: {model_path}")
    print(f"Found {len(files)} case(s) to process")

    rows: List[Dict[str, float]] = []
    for idx, image_path in enumerate(files, start=1):
        print(f"[{idx}/{len(files)}] Processing {image_path}")
        try:
            row = analyze_case(
                network=network,
                image_path=image_path,
                out_dir=args.out_dir,
                input_channel=args.input_channel,
                num_bins=args.num_bins,
                high_cut=args.high_cut,
                device=device,
            )
            rows.append(row)
        except Exception as exc:
            print(f"[WARN] Failed on {image_path}: {exc}")

    summary_path = os.path.join(args.out_dir, "encoder_frequency_spectrum_summary.csv")
    _write_summary(summary_path, rows)
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
