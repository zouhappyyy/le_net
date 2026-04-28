import argparse
import importlib.util
import os
from typing import List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def _load_fbm_class():
    module_path = os.path.join(
        os.path.dirname(__file__),
        "nnunet_mednext",
        "network_architecture",
        "le_networks",
        "FDConv_3d.py",
    )
    spec = importlib.util.spec_from_file_location("fdconv3d_local", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load FDConv_3d.py from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FrequencyBandModulation3D


FrequencyBandModulation3D = _load_fbm_class()


def _load_feature(path: str) -> np.ndarray:
    lower = path.lower()
    if lower.endswith(".npy"):
        arr = np.load(path)
    elif lower.endswith(".npz"):
        data = np.load(path)
        arr = data["data"] if "data" in data else data[data.files[0]]
    else:
        raise ValueError(f"Unsupported file type: {path}")

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        return arr[None]
    if arr.ndim == 4:
        return arr
    if arr.ndim == 5 and arr.shape[0] == 1:
        return arr[0]
    raise RuntimeError(f"Expected [D,H,W], [C,D,H,W] or [1,C,D,H,W], got {arr.shape}")


def _is_feature_file(path: str) -> bool:
    lower = path.lower()
    return lower.endswith(".npy") or lower.endswith(".npz")


def _collect_feature_files(
    feature_path: Optional[str],
    feature_dir: Optional[str],
    max_cases: Optional[int],
) -> List[str]:
    if bool(feature_path) == bool(feature_dir):
        raise ValueError("Specify exactly one of --feature or --feature_dir")

    if feature_path is not None:
        if not os.path.isfile(feature_path):
            raise FileNotFoundError(feature_path)
        return [feature_path]

    if not os.path.isdir(feature_dir):
        raise FileNotFoundError(feature_dir)

    files = []
    for name in sorted(os.listdir(feature_dir)):
        full = os.path.join(feature_dir, name)
        if os.path.isfile(full) and _is_feature_file(full):
            files.append(full)

    if not files:
        raise FileNotFoundError(f"No .npy or .npz files found under {feature_dir}")

    if max_cases is not None:
        files = files[:max_cases]
    return files


def _normalize_01(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax > vmin:
        return (arr - vmin) / (vmax - vmin)
    return np.zeros_like(arr, dtype=np.float32)


def _normalize_signed(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    max_abs = float(np.max(np.abs(arr)))
    if max_abs > 0:
        return arr / max_abs
    return np.zeros_like(arr, dtype=np.float32)


def _numpy_to_torch(arr: np.ndarray, device: str) -> torch.Tensor:
    # Avoid torch<2 + numpy>=2 bridge issues by converting through Python lists.
    return torch.tensor(arr.tolist(), dtype=torch.float32, device=device)


def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return np.asarray(tensor.detach().cpu().to(torch.float32).tolist(), dtype=np.float32)


def _reduce_channels(arr_4d: np.ndarray, mode: str) -> np.ndarray:
    if mode == "mean":
        return arr_4d.mean(axis=0)
    if mode == "max":
        return arr_4d.max(axis=0)
    if mode == "absmean":
        return np.abs(arr_4d).mean(axis=0)
    raise ValueError(f"Unsupported reduce mode: {mode}")


def _take_slice(vol_3d: np.ndarray, axis: str, index: int = None) -> np.ndarray:
    axis_to_dim = {"z": 0, "y": 1, "x": 2}
    dim = axis_to_dim[axis]
    if index is None:
        index = vol_3d.shape[dim] // 2

    if dim == 0:
        return vol_3d[index, :, :]
    if dim == 1:
        return vol_3d[:, index, :]
    return vol_3d[:, :, index]


def _mip_views(vol_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    axial = vol_3d.max(axis=0)
    coronal = vol_3d.max(axis=1)
    sagittal = vol_3d.max(axis=2)
    return axial, coronal, sagittal


def _prepare_display(vol_3d: np.ndarray, view_mode: str, axis: str, slice_index: int = None) -> np.ndarray:
    if view_mode == "slice":
        return _take_slice(vol_3d, axis=axis, index=slice_index)
    if view_mode == "mip":
        mip_idx = {"z": 0, "y": 1, "x": 2}[axis]
        return _mip_views(vol_3d)[mip_idx]
    raise ValueError(f"Unsupported view_mode: {view_mode}")


def _fbm_decompose(
    x: torch.Tensor,
    fbm: FrequencyBandModulation3D,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    with torch.no_grad():
        out, high_acc = fbm(x, att_feat=None, return_high=True)

        b, _, d, h, w = x.shape
        x_fft = torch.fft.rfftn(x, s=(d, h, w), dim=(-3, -2, -1), norm="ortho")
        masks = F.interpolate(
            fbm.cached_masks.float(),
            size=(d, h, w // 2 + 1),
            mode="nearest",
        )

        pre_x = x.clone()
        raw_high_parts: List[torch.Tensor] = []
        weighted_high_parts: List[torch.Tensor] = []
        for idx, _k in enumerate(fbm.k_list):
            mask = masks[idx]
            low_part = torch.fft.irfftn(x_fft * mask, s=(d, h, w), dim=(-3, -2, -1), norm="ortho")
            high_part = pre_x - low_part
            pre_x = low_part
            raw_high_parts.append(high_part)

            freq_weight = fbm._activate(fbm.freq_weight_conv_list[idx](x))
            weighted = (
                freq_weight.reshape(b, fbm.spatial_group, -1, d, h, w)
                * high_part.reshape(b, fbm.spatial_group, -1, d, h, w)
            ).reshape(b, -1, d, h, w)
            weighted_high_parts.append(weighted)

        low_final = pre_x
    return out, high_acc, low_final, raw_high_parts, weighted_high_parts


def _plot_overview(
    save_path: str,
    title: str,
    original_3d: np.ndarray,
    out_3d: np.ndarray,
    low_3d: np.ndarray,
    high_3d: np.ndarray,
    view_mode: str,
    axis: str,
    slice_index: int = None,
) -> None:
    panels = [
        ("Original", original_3d, "gray", False),
        ("FBM Output", out_3d, "gray", False),
        ("Low Frequency", low_3d, "gray", False),
        ("High Frequency", high_3d, "gray", False),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
    for ax, (name, vol, cmap, signed) in zip(axes, panels):
        img = _prepare_display(vol, view_mode=view_mode, axis=axis, slice_index=slice_index)
        if name == "High Frequency":
            disp = _normalize_01(np.abs(img))
        else:
            disp = _normalize_signed(img) if signed else _normalize_01(img)
        ax.imshow(disp, cmap=cmap)
        ax.set_title(name)
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_band_grid(
    save_path: str,
    title: str,
    raw_bands_3d: Sequence[np.ndarray],
    weighted_bands_3d: Sequence[np.ndarray],
    view_mode: str,
    axis: str,
    slice_index: int = None,
) -> None:
    num_bands = len(raw_bands_3d)
    fig, axes = plt.subplots(num_bands, 2, figsize=(9, 3.6 * num_bands))
    if num_bands == 1:
        axes = np.asarray([axes])

    for i in range(num_bands):
        raw_img = _prepare_display(raw_bands_3d[i], view_mode=view_mode, axis=axis, slice_index=slice_index)
        weighted_img = _prepare_display(weighted_bands_3d[i], view_mode=view_mode, axis=axis, slice_index=slice_index)

        axes[i, 0].imshow(_normalize_signed(raw_img), cmap="RdBu_r", vmin=-1, vmax=1)
        axes[i, 0].set_title(f"Band {i + 1} Raw High")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(_normalize_signed(weighted_img), cmap="RdBu_r", vmin=-1, vmax=1)
        axes[i, 1].set_title(f"Band {i + 1} Weighted High")
        axes[i, 1].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_publication_figure(
    save_path: str,
    title: str,
    original_3d: np.ndarray,
    low_3d: np.ndarray,
    high_3d: np.ndarray,
    raw_bands_3d: Sequence[np.ndarray],
    view_mode: str,
    axis: str,
    slice_index: int = None,
    overlay_alpha: float = 0.45,
) -> None:
    orig = _prepare_display(original_3d, view_mode=view_mode, axis=axis, slice_index=slice_index)
    low = _prepare_display(low_3d, view_mode=view_mode, axis=axis, slice_index=slice_index)
    high = _prepare_display(high_3d, view_mode=view_mode, axis=axis, slice_index=slice_index)

    orig_n = _normalize_01(orig)
    low_n = _normalize_01(low)
    high_abs = _normalize_01(np.abs(high))

    band_imgs = []
    for band in raw_bands_3d:
        band_img = _prepare_display(band, view_mode=view_mode, axis=axis, slice_index=slice_index)
        band_imgs.append(_normalize_01(np.abs(band_img)))

    ncols = max(3, len(band_imgs))
    fig, axes = plt.subplots(2, ncols, figsize=(3.2 * ncols, 6.0))
    if ncols == 1:
        axes = np.asarray([axes])

    for ax in axes.ravel():
        ax.axis("off")

    axes[0, 0].imshow(orig_n, cmap="gray")
    axes[0, 0].set_title("Original", fontsize=12)

    axes[0, 1].imshow(low_n, cmap="gray")
    axes[0, 1].set_title("Low-frequency", fontsize=12)

    axes[0, 2].imshow(high_abs, cmap="gray")
    axes[0, 2].set_title("High-frequency", fontsize=12)

    for idx, band_img in enumerate(band_imgs):
        axes[1, idx].imshow(band_img, cmap="gray")
        axes[1, idx].set_title(f"High band {idx + 1}", fontsize=12)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def visualize_fbm_features(
    feature_path: str,
    out_dir: str,
    k_list: Sequence[int],
    lowfreq_att: bool,
    spatial_group: int,
    spatial_kernel: int,
    reduce_mode: str,
    view_mode: str,
    axis: str,
    slice_index: int,
    device: str,
) -> None:
    feat = _load_feature(feature_path)
    case_name = os.path.splitext(os.path.basename(feature_path))[0]
    os.makedirs(out_dir, exist_ok=True)

    x = _numpy_to_torch(feat[None], device=device)
    fbm = FrequencyBandModulation3D(
        in_channels=feat.shape[0],
        k_list=list(k_list),
        lowfreq_att=lowfreq_att,
        spatial_group=spatial_group,
        spatial_kernel=spatial_kernel,
    ).to(device)
    fbm.eval()

    out, high_acc, low_final, raw_high_parts, weighted_high_parts = _fbm_decompose(x, fbm)

    x_np = _tensor_to_numpy(x)[0]
    out_np = _tensor_to_numpy(out)[0]
    low_np = _tensor_to_numpy(low_final)[0]
    high_np = _tensor_to_numpy(high_acc)[0]
    raw_band_np = [_tensor_to_numpy(t)[0] for t in raw_high_parts]
    weighted_band_np = [_tensor_to_numpy(t)[0] for t in weighted_high_parts]

    x_vis = _reduce_channels(x_np, reduce_mode)
    out_vis = _reduce_channels(out_np, reduce_mode)
    low_vis = _reduce_channels(low_np, reduce_mode)
    high_vis = _reduce_channels(high_np, "absmean" if reduce_mode == "absmean" else reduce_mode)
    raw_band_vis = [_reduce_channels(b, "absmean" if reduce_mode == "absmean" else reduce_mode) for b in raw_band_np]
    weighted_band_vis = [_reduce_channels(b, "absmean" if reduce_mode == "absmean" else reduce_mode) for b in weighted_band_np]

    overview_path = os.path.join(out_dir, f"{case_name}_fbm_overview.png")
    _plot_overview(
        save_path=overview_path,
        title=f"FBM Overview: {case_name}",
        original_3d=x_vis,
        out_3d=out_vis,
        low_3d=low_vis,
        high_3d=high_vis,
        view_mode=view_mode,
        axis=axis,
        slice_index=slice_index,
    )

    bands_path = os.path.join(out_dir, f"{case_name}_fbm_bands.png")
    _plot_band_grid(
        save_path=bands_path,
        title=f"FBM Band Decomposition: {case_name}",
        raw_bands_3d=raw_band_vis,
        weighted_bands_3d=weighted_band_vis,
        view_mode=view_mode,
        axis=axis,
        slice_index=slice_index,
    )

    paper_path = os.path.join(out_dir, f"{case_name}_fbm_paper.png")
    _plot_publication_figure(
        save_path=paper_path,
        title=f"Frequency-band decomposition ({case_name})",
        original_3d=x_vis,
        low_3d=low_vis,
        high_3d=high_vis,
        raw_bands_3d=raw_band_vis,
        view_mode=view_mode,
        axis=axis,
        slice_index=slice_index,
    )

    np.save(os.path.join(out_dir, f"{case_name}_fbm_output.npy"), out_np)
    np.save(os.path.join(out_dir, f"{case_name}_fbm_low.npy"), low_np)
    np.save(os.path.join(out_dir, f"{case_name}_fbm_high.npy"), high_np)
    for idx, (raw_band, weighted_band) in enumerate(zip(raw_band_np, weighted_band_np), start=1):
        np.save(os.path.join(out_dir, f"{case_name}_band{idx}_raw_high.npy"), raw_band)
        np.save(os.path.join(out_dir, f"{case_name}_band{idx}_weighted_high.npy"), weighted_band)

    print(f"Saved overview figure to: {overview_path}")
    print(f"Saved band figure to: {bands_path}")
    print(f"Saved publication figure to: {paper_path}")
    print(f"Saved decomposition arrays under: {out_dir}")


def visualize_fbm_features_batch(
    feature_files: Sequence[str],
    out_dir: str,
    k_list: Sequence[int],
    lowfreq_att: bool,
    spatial_group: int,
    spatial_kernel: int,
    reduce_mode: str,
    view_mode: str,
    axis: str,
    slice_index: int,
    device: str,
) -> None:
    print(f"Found {len(feature_files)} case(s) to process")
    for idx, feature_path in enumerate(feature_files, start=1):
        print(f"[{idx}/{len(feature_files)}] Processing {feature_path}")
        visualize_fbm_features(
            feature_path=feature_path,
            out_dir=out_dir,
            k_list=k_list,
            lowfreq_att=lowfreq_att,
            spatial_group=spatial_group,
            spatial_kernel=spatial_kernel,
            reduce_mode=reduce_mode,
            view_mode=view_mode,
            axis=axis,
            slice_index=slice_index,
            device=device,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize how a feature volume changes after FrequencyBandModulation3D."
    )
    parser.add_argument("--feature", type=str, default=None, help="Input feature file (.npy or .npz)")
    parser.add_argument("--feature_dir", type=str, default=None, help="Directory of feature files (.npy or .npz)")
    parser.add_argument("--out_dir", type=str, default="fbm_feature_vis", help="Directory to save results")
    parser.add_argument("--max_cases", type=int, default=None, help="Only process the first N cases when using --feature_dir")
    parser.add_argument("--k_list", type=int, nargs="+", default=[2, 4, 8], help="Frequency band split factors")
    parser.add_argument("--lowfreq_att", action="store_true", help="Enable low-frequency attention in FBM")
    parser.add_argument("--spatial_group", type=int, default=1, help="Spatial group used by FBM")
    parser.add_argument("--spatial_kernel", type=int, default=3, help="Spatial kernel used by FBM")
    parser.add_argument(
        "--reduce",
        type=str,
        default="mean",
        choices=["mean", "max", "absmean"],
        help="How to reduce channels to a single 3D volume for visualization",
    )
    parser.add_argument(
        "--view_mode",
        type=str,
        default="slice",
        choices=["slice", "mip"],
        help="Visualize a middle slice or a maximum-intensity projection",
    )
    parser.add_argument("--axis", type=str, default="z", choices=["x", "y", "z"], help="Display axis")
    parser.add_argument("--slice_index", type=int, default=None, help="Slice index for view_mode=slice")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", None], help="Run device")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    feature_files = _collect_feature_files(
        feature_path=args.feature,
        feature_dir=args.feature_dir,
        max_cases=args.max_cases,
    )
    visualize_fbm_features_batch(
        feature_files=feature_files,
        out_dir=args.out_dir,
        k_list=args.k_list,
        lowfreq_att=args.lowfreq_att,
        spatial_group=args.spatial_group,
        spatial_kernel=args.spatial_kernel,
        reduce_mode=args.reduce,
        view_mode=args.view_mode,
        axis=args.axis,
        slice_index=args.slice_index,
        device=device,
    )
