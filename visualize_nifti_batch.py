import os
import copy
import argparse
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

from nnunet_mednext.utilities.overlay_plots import generate_overlay




DEFAULT_PARAMS = {
    "images_dir": "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_raw_data/Task602_ls/imagesTr",
    "gt_dir": "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_raw_data/Task602_ls/labelsTr",
    "models": {
        "BGHNetV4": "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_predictions/Task602_ls/BGHNetV4Trainer/preds",
        "nnFormer": "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_predictions/Task602_ls/nnFormer",
        "MedNeXt": "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_predictions/Task602_ls/MedNeXt",
        "nnUNet": "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_predictions/Task602_ls/nnU-Net",
        "SwinUNETR": "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_predictions/Task602_ls/SwinUNETR",
        "UMamba": "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_predictions/Task602_ls/UMamba",
        "VoComni": "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_predictions/Task602_ls/VoComni_nnunet",
    },
    "output_dir": "./Task602_ls_vis",
    "axes": ["z", "y", "x"],
    "alpha": 0.6,
    "image_suffix": "_0000.nii.gz",
    "pred_suffix": ".nii.gz",
    "gt_suffix": ".nii.gz",
    "cases": None,
    "mm_per_inch": 50.0,
}


def _strip_suffix(filename: str, suffix: Optional[str]) -> str:
    """Remove the provided suffix or default NIfTI extension to derive a case id."""
    if suffix and filename.endswith(suffix):
        return filename[: -len(suffix)]
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]
    return os.path.splitext(filename)[0]


def _list_cases(images_dir: str, suffix: Optional[str]) -> Dict[str, str]:
    """Return mapping case_id -> image_path for files in images_dir respecting suffix filter."""
    mapping: Dict[str, str] = {}
    for fname in sorted(os.listdir(images_dir)):
        if not (fname.endswith(".nii") or fname.endswith(".nii.gz")):
            continue
        if suffix and not fname.endswith(suffix):
            continue
        case_id = _strip_suffix(fname, suffix)
        mapping[case_id] = os.path.join(images_dir, fname)
    if not mapping:
        raise RuntimeError(f"No NIfTI files found in {images_dir} with suffix {suffix or '[.nii/.nii.gz]'}")
    return mapping


def _load_image(path: str) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Load original image NIfTI and return array [1, D, H, W] plus spacing (sz, sy, sx)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    img = nib.load(path)
    data = img.get_fdata()
    zooms = img.header.get_zooms()
    if len(zooms) < 3:
        raise RuntimeError(f"Invalid spacing metadata in {path}: {zooms}")
    spacing_xyz = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
    if data.ndim == 3:
        data = np.transpose(data, (2, 1, 0))
        data = data[None, ...]
    elif data.ndim == 4:
        data = data[..., 0]
        data = np.transpose(data, (2, 1, 0))
        data = data[None, ...]
    else:
        raise RuntimeError(f"Unsupported image ndim {data.ndim} for {path}")
    spacing = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    return data.astype(np.float32), spacing


def _load_seg(path: str) -> np.ndarray:
    """Load segmentation/label NIfTI and return array shaped [D, H, W]."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Segmentation file not found: {path}")
    img = nib.load(path)
    data = img.get_fdata()
    if data.ndim == 3:
        data = np.transpose(data, (2, 1, 0))
    elif data.ndim == 4:
        data = np.argmax(data, axis=-1)
        data = np.transpose(data, (2, 1, 0))
    else:
        raise RuntimeError(f"Unsupported segmentation ndim {data.ndim} for {path}")
    return data.astype(np.int16)


def _match_shape(image: np.ndarray, volume: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Crop both tensors to identical spatial shape."""
    _, d_i, h_i, w_i = image.shape
    d_s, h_s, w_s = volume.shape
    d = min(d_i, d_s)
    h = min(h_i, h_s)
    w = min(w_i, w_s)
    return image[:, :d, :h, :w], volume[:d, :h, :w]


def _determine_slices(length: int, requested: Optional[List[int]]) -> List[int]:
    if not requested:
        return [length // 2]
    valid = sorted({idx for idx in requested if 0 <= idx < length})
    if not valid:
        raise ValueError(f"Requested slices {requested} are outside volume length {length}")
    return valid


def _center_slices_for_axis(
    gt_vol: np.ndarray,
    axis: str,
    fractions: Optional[List[float]] = None,
) -> List[int]:
    """Compute representative slice indices along a given axis based on GT foreground.

    If fractions is None, a single middle slice of the foreground range is returned.
    """
    if gt_vol is None:
        return []

    if fractions is None or len(fractions) == 0:
        fractions = [0.5]

    # Boolean mask: for each slice index along `axis`, whether there is any foreground.
    if axis == "z":
        mask_1d = (gt_vol != 0).any(axis=(1, 2))
    elif axis == "y":
        mask_1d = (gt_vol != 0).any(axis=(0, 2))
    elif axis == "x":
        mask_1d = (gt_vol != 0).any(axis=(0, 1))
    else:
        raise ValueError("axis must be one of 'z', 'y', 'x'")

    if not mask_1d.any():
        return []

    fg_indices = np.flatnonzero(mask_1d)
    start = int(fg_indices[0])
    end = int(fg_indices[-1])
    length = end - start + 1

    if length <= 1:
        return [start]

    indices: List[int] = []
    for f in fractions:
        # Clamp fraction to [0, 1]
        if f < 0.0:
            f = 0.0
        elif f > 1.0:
            f = 1.0
        pos = start + int(round((length - 1) * f))
        indices.append(pos)

    # Deduplicate and sort
    return sorted(set(indices))


def _auto_slices_from_gt(
    gt_vol: Optional[np.ndarray],
    axis: str,
    fallback_length: int,
    fractions: Optional[List[float]] = None,
) -> List[int]:
    """Determine default slices for an axis using GT when available.

    If GT is missing or empty along this axis, fall back to the geometric middle slice.
    """
    if gt_vol is None:
        return _determine_slices(fallback_length, None)

    indices = _center_slices_for_axis(gt_vol, axis, fractions)
    if not indices:
        return _determine_slices(fallback_length, None)

    valid = sorted({idx for idx in indices if 0 <= idx < fallback_length})
    if not valid:
        return _determine_slices(fallback_length, None)
    return valid


def _get_axis_slices_for_case(
    length: int,
    axis: str,
    slices: Optional[List[int]],
    gt_vol: Optional[np.ndarray],
) -> List[int]:
    """Resolve slice indices for a given axis, optionally using GT when slices are not provided."""
    if slices is not None:
        return _determine_slices(length, slices)
    return _auto_slices_from_gt(gt_vol, axis, length)


def _extract_slice(image: np.ndarray, volume: np.ndarray, axis: str, index: int) -> Tuple[np.ndarray, np.ndarray]:
    if axis == "z":
        return image[0, index], volume[index]
    if axis == "y":
        return image[0, :, index, :], volume[:, index, :]
    if axis == "x":
        return image[0, :, :, index], volume[:, :, index]
    raise ValueError("axis must be one of 'z', 'y', 'x'")


def _plane_energy(image: np.ndarray, axis: str) -> np.ndarray:
    """Compute per-slice energy (sum of absolute intensities) along the requested axis."""
    vol = np.abs(image[0])
    if axis == "z":
        return vol.sum(axis=(1, 2))
    if axis == "y":
        return vol.sum(axis=(0, 2))
    if axis == "x":
        return vol.sum(axis=(0, 1))
    raise ValueError("axis must be one of 'z', 'y', 'x'")


def _pick_valid_slice(energy: np.ndarray, requested_idx: int) -> Tuple[int, bool]:
    """Return an index with non-zero energy, falling back to the nearest slice if needed."""
    n = len(energy)
    if 0 <= requested_idx < n and energy[requested_idx] > 0:
        return requested_idx, False
    for offset in range(1, n):
        for candidate in (requested_idx - offset, requested_idx + offset):
            if 0 <= candidate < n and energy[candidate] > 0:
                return candidate, True
    # All slices empty; return the clamped original index so downstream code can decide how to handle it
    return max(0, min(requested_idx, n - 1)), True


def _axis_slice_spacing(axis: str, spacing: Tuple[float, float, float]) -> Tuple[float, float]:
    sz, sy, sx = spacing
    if axis == "z":
        return sy, sx
    if axis == "y":
        return sz, sx
    if axis == "x":
        return sz, sy
    raise ValueError("axis must be one of 'z', 'y', 'x'")


def _slice_extent(
    axis: str,
    spacing: Tuple[float, float, float],
    shape: Tuple[int, int],
    mm_per_inch: float,
) -> Tuple[Tuple[float, float, float, float], float, float]:
    row_spacing, col_spacing = _axis_slice_spacing(axis, spacing)
    height_mm = max(shape[0] * row_spacing, 1e-3)
    width_mm = max(shape[1] * col_spacing, 1e-3)
    extent = (0.0, width_mm, 0.0, height_mm)
    scale = max(mm_per_inch, 1e-3)
    fig_w = max(width_mm / scale, 1.0)
    fig_h = max(height_mm / scale, 1.0)
    return extent, fig_w, fig_h


def _save_overlay_image(
    overlay: np.ndarray,
    axis: str,
    spacing: Tuple[float, float, float],
    out_path: str,
    title: Optional[str] = None,
    show_labels: bool = False,
    mm_per_inch: float = 50.0,
) -> None:
    extent, fig_w, fig_h = _slice_extent(axis, spacing, overlay.shape[:2], mm_per_inch)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(overlay, extent=extent, origin="lower")
    if show_labels:
        ax.set_xlabel("mm")
        ax.set_ylabel("mm")
    if title:
        ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.axis("off")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _parse_model_args(items: List[str]) -> Dict[str, str]:
    models: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Model argument must be name=dir, got: {item}")
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Invalid model entry: {item}")
        if not os.path.isdir(path):
            raise NotADirectoryError(f"Model directory does not exist: {path}")
        models[name] = path
    if not models:
        raise ValueError("At least one --model name=dir pair is required")
    return models


def _build_pred_path(model_dir: str, case_id: str, suffix: Optional[str]) -> Optional[str]:
    candidates: List[str] = []
    if suffix:
        candidates.append(os.path.join(model_dir, f"{case_id}{suffix}"))
    candidates.append(os.path.join(model_dir, f"{case_id}.nii.gz"))
    candidates.append(os.path.join(model_dir, f"{case_id}.nii"))
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def _build_gt_path(gt_dir: str, case_id: str, suffix: Optional[str]) -> Optional[str]:
    candidates: List[str] = []
    if suffix:
        candidates.append(os.path.join(gt_dir, f"{case_id}{suffix}"))
    candidates.append(os.path.join(gt_dir, f"{case_id}.nii.gz"))
    candidates.append(os.path.join(gt_dir, f"{case_id}.nii"))
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def process_case(
    case_id: str,
    image_path: str,
    models: Dict[str, str],
    gt_dir: Optional[str],
    output_dir: str,
    axes: List[str],
    slices: Optional[List[int]],
    alpha: float,
    image_suffix: Optional[str],
    pred_suffix: Optional[str],
    gt_suffix: Optional[str],
    mm_per_inch: float,
) -> None:
    overlays_dir = os.path.join(output_dir, "overlays", case_id)
    panels_dir = os.path.join(output_dir, "panels")
    os.makedirs(overlays_dir, exist_ok=True)
    os.makedirs(panels_dir, exist_ok=True)

    image, spacing = _load_image(image_path)
    gt = None
    if gt_dir:
        gt_path = _build_gt_path(gt_dir, case_id, gt_suffix)
        if gt_path is None:
            raise FileNotFoundError(f"GT not found for {case_id} under {gt_dir}")
        gt = _load_seg(gt_path)
        image, gt = _match_shape(image, gt)

    pred_volumes: Dict[str, np.ndarray] = {}
    missing_models: List[str] = []
    for model_name, model_dir in models.items():
        pred_path = _build_pred_path(model_dir, case_id, pred_suffix)
        if pred_path is None:
            missing_models.append(model_name)
            continue
        pred = _load_seg(pred_path)
        image, pred = _match_shape(image, pred)
        pred_volumes[model_name] = pred
    if missing_models:
        raise FileNotFoundError(
            f"Missing predictions for case {case_id} in models: {', '.join(missing_models)}"
        )

    _, depth, height, width = image.shape
    axis_to_slices: Dict[str, List[int]] = {}
    for axis in axes:
        if axis == "z":
            axis_to_slices[axis] = _get_axis_slices_for_case(depth, axis, slices, gt)
        elif axis == "y":
            axis_to_slices[axis] = _get_axis_slices_for_case(height, axis, slices, gt)
        elif axis == "x":
            axis_to_slices[axis] = _get_axis_slices_for_case(width, axis, slices, gt)
        else:
            raise ValueError("Axes must be any of ['z','y','x']")

    axis_energy = {axis: _plane_energy(image, axis) for axis in axis_to_slices.keys()}

    for axis, slice_list in axis_to_slices.items():
        energy = axis_energy[axis]
        for s_idx in slice_list:
            actual_idx, adjusted = _pick_valid_slice(energy, s_idx)
            if adjusted:
                print(
                    f"[WARN] Case {case_id}: axis {axis} slice {s_idx} has no signal, using slice {actual_idx} instead."
                )

            panel_images: List[np.ndarray] = []
            panel_titles: List[str] = []

            for model_name, pred_vol in pred_volumes.items():
                img_slice, pred_slice = _extract_slice(image, pred_vol, axis, actual_idx)
                if not np.isfinite(img_slice).any() or np.nanmax(np.abs(img_slice)) == 0:
                    print(
                        f"[WARN] Case {case_id}: axis {axis} slice {actual_idx} still empty for model {model_name}, skipping."
                    )
                    continue
                overlay = generate_overlay(img_slice, pred_slice, overlay_intensity=alpha)
                out_name = f"{case_id}_{model_name}_axis-{axis}_slice-{actual_idx}.png"
                out_path = os.path.join(overlays_dir, out_name)
                _save_overlay_image(overlay, axis, spacing, out_path, show_labels=True, mm_per_inch=mm_per_inch)
                panel_images.append(overlay)
                panel_titles.append(model_name)

            if gt is not None:
                img_slice, gt_slice = _extract_slice(image, gt, axis, actual_idx)
                if np.isfinite(img_slice).any() and np.nanmax(np.abs(img_slice)) > 0:
                    mapping = {0: 0, 1: 4}
                    overlay_gt = generate_overlay(img_slice, gt_slice, mapping=mapping, overlay_intensity=alpha)
                    out_name = f"{case_id}_GT_axis-{axis}_slice-{actual_idx}.png"
                    out_path = os.path.join(overlays_dir, out_name)
                    _save_overlay_image(
                        overlay_gt,
                        axis,
                        spacing,
                        out_path,
                        title="GT",
                        show_labels=True,
                        mm_per_inch=mm_per_inch,
                    )
                    panel_images.append(overlay_gt)
                    panel_titles.append("GT")
                else:
                    print(
                        f"[WARN] Case {case_id}: axis {axis} slice {actual_idx} empty for GT overlay, skipping."
                    )

            if panel_images:
                cols = len(panel_images)
                extent, fig_w, fig_h = _slice_extent(
                    axis,
                    spacing,
                    panel_images[0].shape[:2],
                    mm_per_inch,
                )
                fig, axes_fig = plt.subplots(1, cols, figsize=(fig_w * cols, fig_h))
                if cols == 1:
                    axes_fig = [axes_fig]
                for ax_obj, img_arr, title in zip(axes_fig, panel_images, panel_titles):
                    ax_obj.imshow(img_arr, extent=extent, origin="lower")
                    ax_obj.set_title(title)
                    ax_obj.set_xlim(extent[0], extent[1])
                    ax_obj.set_ylim(extent[2], extent[3])
                    ax_obj.set_aspect("equal")
                    ax_obj.axis("off")
                plt.tight_layout()
                panel_name = f"{case_id}_axis-{axis}_slice-{actual_idx}_panel.png"
                panel_path = os.path.join(panels_dir, panel_name)
                fig.savefig(panel_path, dpi=150)
                plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch visualize .nii.gz images, GT masks, and multiple model predictions.",
    )
    parser.add_argument("--images_dir", type=str, default=None, help="Directory with original images (.nii.gz)")
    parser.add_argument("--gt_dir", type=str, default=None, help="Directory with GT masks (.nii.gz)")
    parser.add_argument(
        "--model",
        type=str,
        action="append",
        default=None,
        help="Model specification as name=/path/to/predictions. Repeat for multiple models.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to store overlays and panels")
    parser.add_argument(
        "--axes",
        type=str,
        nargs="*",
        default=None,
        help="Axes to visualize (subset of z y x).",
    )
    parser.add_argument(
        "--slices",
        type=int,
        nargs="*",
        default=None,
        help="Slice indices to visualize per axis. Defaults to middle slice if omitted.",
    )
    parser.add_argument("--alpha", type=float, default=None, help="Overlay transparency (0-1)")
    parser.add_argument("--image_suffix", type=str, default=None, help="Suffix filter for images, e.g. '_0000.nii.gz'")
    parser.add_argument("--pred_suffix", type=str, default=None, help="Suffix to append when looking for preds")
    parser.add_argument("--gt_suffix", type=str, default=None, help="Suffix to append when looking for GT masks")
    parser.add_argument(
        "--cases",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit list of case_ids to visualize. Defaults to all in images_dir.",
    )
    parser.add_argument(
        "--use_default",
        action="store_true",
        help="Use DEFAULT_PARAMS so most arguments can be omitted.",
    )
    parser.add_argument(
        "--mm_per_inch",
        type=float,
        default=None,
        help="Millimeters per inch for output image scaling.",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    config = copy.deepcopy(DEFAULT_PARAMS) if args.use_default else {}

    images_dir = args.images_dir or config.get("images_dir")
    if not images_dir:
        raise ValueError("images_dir must be provided either via CLI or DEFAULT_PARAMS")

    gt_dir = args.gt_dir if args.gt_dir is not None else config.get("gt_dir")
    output_dir = args.output_dir or config.get("output_dir")
    if not output_dir:
        raise ValueError("output_dir must be provided either via CLI or DEFAULT_PARAMS")

    model_specs = args.model if args.model is not None else [
        f"{name}={path}" for name, path in config.get("models", {}).items()
    ]
    if not model_specs:
        raise ValueError("At least one --model name=dir pair or DEFAULT_PARAMS['models'] must be provided")
    models = _parse_model_args(model_specs)

    axes = args.axes or config.get("axes") or ["z"]
    slices = args.slices if args.slices is not None else config.get("slices")
    alpha = args.alpha if args.alpha is not None else config.get("alpha", 0.6)
    image_suffix = args.image_suffix if args.image_suffix is not None else config.get("image_suffix")
    pred_suffix = args.pred_suffix if args.pred_suffix is not None else config.get("pred_suffix")
    gt_suffix = args.gt_suffix if args.gt_suffix is not None else config.get("gt_suffix")
    case_filter = args.cases if args.cases is not None else config.get("cases")
    mm_per_inch = args.mm_per_inch if args.mm_per_inch is not None else config.get("mm_per_inch", 50.0)

    if not os.path.isdir(images_dir):
        raise NotADirectoryError(f"images_dir does not exist: {images_dir}")
    if gt_dir and not os.path.isdir(gt_dir):
        raise NotADirectoryError(f"gt_dir does not exist: {gt_dir}")

    os.makedirs(output_dir, exist_ok=True)

    case_map = _list_cases(images_dir, image_suffix)
    if case_filter:
        case_ids = [cid for cid in case_filter if cid in case_map]
        missing = sorted(set(case_filter) - set(case_ids))
        if missing:
            raise FileNotFoundError(f"Requested cases not found in images_dir: {missing}")
    else:
        case_ids = list(case_map.keys())

    for case_id in case_ids:
        image_path = case_map[case_id]
        process_case(
            case_id=case_id,
            image_path=image_path,
            models=models,
            gt_dir=gt_dir,
            output_dir=output_dir,
            axes=axes,
            slices=slices,
            alpha=alpha,
            image_suffix=image_suffix,
            pred_suffix=pred_suffix,
            gt_suffix=gt_suffix,
            mm_per_inch=mm_per_inch,
        )


if __name__ == "__main__":
    main()

