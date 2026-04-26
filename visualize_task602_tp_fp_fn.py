import argparse
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from nnunet_mednext.utilities.overlay_plots import generate_overlay
from visualize_fd_edge_and_ds import _extract_case_data


# Built-in model mapping for Task602_ls.
MODELS_TASK602_LS = {
    "nnFormer": "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_predictions/Task602_ls/nnFormer",
    "BGHNetV4": "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_predictions/Task602_ls/BGHNetV4Trainer",
    "MedNeXt": "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_predictions/Task602_ls/MedNeXt",
    "nnUNet": "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_predictions/Task602_ls/nnU-Net",
    "SwinUNETR": "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_predictions/Task602_ls/SwinUNETR",
    "UMamba": "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_predictions/Task602_ls/UMamba",
    "VoComni": "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_predictions/Task602_ls/VoComni_nnunet",
}

TASK602_DEFAULT_CONFIG = {
    "data_root": "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_preprocessed/Task602_ls/nnUNetData_plans_v2.1_stage0",
    "dataset_directory": "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_preprocessed/Task602_ls",
    "fold": 0,
    "output_dir": "./Task602_tp_fp_fn_vis",
    "slices": None,
    "save_gt": True,
    "save_raw": True,
    "save_gt_overlay": True,
}


# Black background + distinct error-type colors.
COLOR_BACKGROUND = np.array([0, 0, 0], dtype=np.uint8)
COLOR_TP = np.array([34, 197, 94], dtype=np.uint8)     # green
COLOR_FP = np.array([239, 68, 68], dtype=np.uint8)     # red
COLOR_FN = np.array([59, 130, 246], dtype=np.uint8)    # blue
COLOR_GT = np.array([255, 255, 255], dtype=np.uint8)   # white


def _load_label_nifti(label_path: str) -> np.ndarray:
    """Load label nifti and return array in [Z, Y, X] order."""
    if not os.path.isfile(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")

    arr = nib.load(label_path).get_fdata()
    if arr.ndim != 3:
        raise RuntimeError(f"Expected 3D label map, got shape {arr.shape} from {label_path}")

    # NIfTI commonly comes as [X, Y, Z]; unify to [Z, Y, X].
    return np.transpose(arr, (2, 1, 0)).astype(np.int16)


def _find_gt_path(dataset_directory: str, case_id: str) -> str:
    """Find GT path under a Task directory."""
    gt_candidates = [
        os.path.join(dataset_directory, "gt_segmentations"),
        os.path.join(dataset_directory, "labelsTr"),
        dataset_directory,
    ]
    for gt_dir in gt_candidates:
        if not os.path.isdir(gt_dir):
            continue
        for ext in (".nii.gz", ".nii"):
            cand = os.path.join(gt_dir, f"{case_id}{ext}")
            if os.path.isfile(cand):
                return cand
    raise FileNotFoundError(f"Could not find GT for case {case_id} under {dataset_directory}")


def _load_preprocessed_image_and_gt(data_root: str, dataset_directory: str, case_id: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load Task602 exactly the same way as visualize_val_overlay.py.

    Returns
    -------
    image : np.ndarray
        Image volume in [C, D, H, W].
    gt : np.ndarray
        Label map in [D, H, W].
    """
    image, gt = _extract_case_data(data_root, case_id, dataset_directory)
    if image.ndim != 4:
        raise RuntimeError(f"Expected image from _extract_case_data to be [C,D,H,W], got {image.shape}")
    if gt.ndim != 4:
        raise RuntimeError(f"Expected gt from _extract_case_data to be [1,D,H,W], got {gt.shape}")
    return image.astype(np.float32), gt[0].astype(np.int16)


def _find_pred_path(pred_folder: str, case_id: str) -> str:
    for ext in (".nii.gz", ".nii"):
        cand = os.path.join(pred_folder, f"{case_id}{ext}")
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(f"Could not find prediction for case {case_id} in {pred_folder}")


def _match_shapes_like_overlay(image: np.ndarray, seg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Match shapes exactly like visualize_val_overlay._match_shapes.

    image: [C, D, H, W]
    seg: [D, H, W] or [C2, D, H, W]
    """
    d_i, h_i, w_i = image.shape[1:]

    if seg.ndim == 3:
        d_s, h_s, w_s = seg.shape
        d = min(d_i, d_s)
        h = min(h_i, h_s)
        w = min(w_i, w_s)
        image_c = image[:, :d, :h, :w]
        seg_c = seg[:d, :h, :w]
    elif seg.ndim == 4:
        _, d_s, h_s, w_s = seg.shape
        d = min(d_i, d_s)
        h = min(h_i, h_s)
        w = min(w_i, w_s)
        image_c = image[:, :d, :h, :w]
        seg_c = seg[:, :d, :h, :w]
    else:
        raise RuntimeError(f"Unexpected seg ndim={seg.ndim} in _match_shapes_like_overlay")

    return image_c, seg_c


def _collect_case_ids_from_gt(dataset_directory: str) -> List[str]:
    ids: List[str] = []
    gt_dirs = [
        os.path.join(dataset_directory, "gt_segmentations"),
        os.path.join(dataset_directory, "labelsTr"),
        dataset_directory,
    ]
    gt_dir = None
    for cand in gt_dirs:
        if os.path.isdir(cand):
            gt_dir = cand
            break
    if gt_dir is None:
        raise FileNotFoundError(f"No GT directory found under {dataset_directory}")

    for fn in os.listdir(gt_dir):
        if fn.endswith(".nii.gz"):
            case_id = fn[:-7]
        elif fn.endswith(".nii"):
            case_id = fn[:-4]
        else:
            continue
        if case_id not in ids:
            ids.append(case_id)
    ids.sort()
    return ids


def _collect_case_ids_from_pred_folder(pred_folder: str) -> List[str]:
    if not os.path.isdir(pred_folder):
        raise FileNotFoundError(f"Prediction folder not found: {pred_folder}")

    ids: List[str] = []
    for fn in os.listdir(pred_folder):
        if fn.endswith(".nii.gz"):
            case_id = fn[:-7]
        elif fn.endswith(".nii"):
            case_id = fn[:-4]
        else:
            continue
        if case_id not in ids:
            ids.append(case_id)
    ids.sort()
    return ids


def _make_tp_fp_fn_map(pred_slice: np.ndarray, gt_slice: np.ndarray) -> np.ndarray:
    """Encode one slice into 0=bg, 1=TP, 2=FP, 3=FN."""
    pred_fg = pred_slice > 0
    gt_fg = gt_slice > 0

    out = np.zeros(pred_slice.shape, dtype=np.uint8)
    out[np.logical_and(pred_fg, gt_fg)] = 1
    out[np.logical_and(pred_fg, ~gt_fg)] = 2
    out[np.logical_and(~pred_fg, gt_fg)] = 3
    return out


def _render_tp_fp_fn_rgb(label_map: np.ndarray) -> np.ndarray:
    rgb = np.zeros(label_map.shape + (3,), dtype=np.uint8)
    rgb[label_map == 0] = COLOR_BACKGROUND
    rgb[label_map == 1] = COLOR_TP
    rgb[label_map == 2] = COLOR_FP
    rgb[label_map == 3] = COLOR_FN
    return rgb


def _render_binary_mask(mask_2d: np.ndarray, fg_color: np.ndarray) -> np.ndarray:
    rgb = np.zeros(mask_2d.shape + (3,), dtype=np.uint8)
    rgb[mask_2d > 0] = fg_color
    return rgb


def _render_raw_slice(image_slice: np.ndarray) -> np.ndarray:
    image = image_slice.astype(np.float32, copy=True)
    image -= image.min()
    max_val = image.max()
    if max_val > 0:
        image = image / max_val
    image = (image * 255).clip(0, 255).astype(np.uint8)
    return np.stack([image, image, image], axis=-1)


def _extract_slice(volume: np.ndarray, axis: str, index: int) -> np.ndarray:
    if axis == "z":
        return volume[index]
    if axis == "y":
        return volume[:, index, :]
    if axis == "x":
        return volume[:, :, index]
    raise ValueError("axis must be one of 'z', 'y', 'x'")


def _default_slices_for_axis(shape_zyx: Tuple[int, int, int], axis: str) -> List[int]:
    d, h, w = shape_zyx
    if axis == "z":
        return [d // 2]
    if axis == "y":
        return [h // 2]
    if axis == "x":
        return [w // 2]
    raise ValueError("axis must be one of 'z', 'y', 'x'")


def _normalize_slices(shape_zyx: Tuple[int, int, int], axis: str, slices: Optional[List[int]]) -> List[int]:
    if not slices:
        return _default_slices_for_axis(shape_zyx, axis)

    axis_len = {"z": shape_zyx[0], "y": shape_zyx[1], "x": shape_zyx[2]}[axis]
    return sorted({s for s in slices if 0 <= s < axis_len})


def visualize_case_tp_fp_fn(
    data_root: str,
    dataset_directory: str,
    pred_folder: str,
    case_id: str,
    output_dir: str,
    mode: str = "slice",
    axis: str = "z",
    slices: Optional[List[int]] = None,
    name_prefix: str = "",
    save_gt: bool = False,
    save_raw: bool = False,
    save_gt_overlay: bool = False,
) -> List[Dict[str, object]]:
    """Visualize a single case without the original image.

    Outputs color-coded mask images:
      - TP: green
      - FP: red
      - FN: blue
      - background: black
    """
    os.makedirs(output_dir, exist_ok=True)
    results: List[Dict[str, object]] = []

    image, gt = _load_preprocessed_image_and_gt(data_root, dataset_directory, case_id)
    pred = _load_label_nifti(_find_pred_path(pred_folder, case_id))
    image, pred = _match_shapes_like_overlay(image, pred)
    _, gt = _match_shapes_like_overlay(image, gt)

    if mode not in ("slice", "mip"):
        raise ValueError("mode must be 'slice' or 'mip'")

    if mode == "slice":
        _, d, h, w = image.shape
        slice_indices = _normalize_slices((d, h, w), axis, slices)
        for s in slice_indices:
            if axis == "z":
                image_slice = image[0, s]
                pred_slice = pred[s]
                gt_slice = gt[s]
            elif axis == "y":
                image_slice = image[0, :, s, :]
                pred_slice = pred[:, s, :]
                gt_slice = gt[:, s, :]
            else:
                image_slice = image[0, :, :, s]
                pred_slice = pred[:, :, s]
                gt_slice = gt[:, :, s]
            diff_map = _make_tp_fp_fn_map(pred_slice, gt_slice)
            rgb = _render_tp_fp_fn_rgb(diff_map)

            if save_raw:
                raw_rgb = _render_raw_slice(image_slice)
                raw_base = f"{case_id}_axis-{axis}_slice-{s}_raw.png"
                raw_name = f"{name_prefix}_{raw_base}" if name_prefix else raw_base
                raw_path = os.path.join(output_dir, raw_name)
                plt.imsave(raw_path, raw_rgb)
                print(f"Saved raw slice to: {raw_path}")
                results.append(
                    {
                        "case_id": case_id,
                        "axis": axis,
                        "slice": s,
                        "path": raw_path,
                        "pred_folder": pred_folder,
                        "type": "raw",
                    }
                )

            if save_gt_overlay:
                gt_overlay = generate_overlay(image_slice, gt_slice, overlay_intensity=0.8)
                gt_overlay_base = f"{case_id}_axis-{axis}_slice-{s}_gt_overlay.png"
                gt_overlay_name = f"{name_prefix}_{gt_overlay_base}" if name_prefix else gt_overlay_base
                gt_overlay_path = os.path.join(output_dir, gt_overlay_name)
                plt.imsave(gt_overlay_path, gt_overlay)
                print(f"Saved GT overlay to: {gt_overlay_path}")
                results.append(
                    {
                        "case_id": case_id,
                        "axis": axis,
                        "slice": s,
                        "path": gt_overlay_path,
                        "pred_folder": pred_folder,
                        "type": "gt_overlay",
                    }
                )

            base_name = f"{case_id}_axis-{axis}_slice-{s}_tp_fp_fn.png"
            out_name = f"{name_prefix}_{base_name}" if name_prefix else base_name
            out_path = os.path.join(output_dir, out_name)
            plt.imsave(out_path, rgb)
            print(f"Saved TP/FP/FN map to: {out_path}")
            results.append(
                {
                    "case_id": case_id,
                    "axis": axis,
                    "slice": s,
                    "path": out_path,
                    "pred_folder": pred_folder,
                    "type": "tp_fp_fn",
                }
            )

            if save_gt:
                gt_rgb = _render_binary_mask(gt_slice, COLOR_GT)
                gt_base = f"{case_id}_axis-{axis}_slice-{s}_gt.png"
                gt_name = f"{name_prefix}_{gt_base}" if name_prefix else gt_base
                gt_path = os.path.join(output_dir, gt_name)
                plt.imsave(gt_path, gt_rgb)
                print(f"Saved GT mask to: {gt_path}")
                results.append(
                    {
                        "case_id": case_id,
                        "axis": axis,
                        "slice": s,
                        "path": gt_path,
                        "pred_folder": pred_folder,
                        "type": "gt",
                    }
                )

    else:
        vol = image[0]
        pred_fg = pred > 0
        gt_fg = gt > 0
        tp = np.logical_and(pred_fg, gt_fg)
        fp = np.logical_and(pred_fg, ~gt_fg)
        fn = np.logical_and(~pred_fg, gt_fg)

        tp_fp_fn_views = {
            "axial": np.zeros(tp.max(axis=0).shape, dtype=np.uint8),
            "coronal": np.zeros(tp.max(axis=1).shape, dtype=np.uint8),
            "sagittal": np.zeros(tp.max(axis=2).shape, dtype=np.uint8),
        }
        tp_fp_fn_views["axial"][tp.max(axis=0)] = 1
        tp_fp_fn_views["axial"][fp.max(axis=0)] = 2
        tp_fp_fn_views["axial"][fn.max(axis=0)] = 3
        tp_fp_fn_views["coronal"][tp.max(axis=1)] = 1
        tp_fp_fn_views["coronal"][fp.max(axis=1)] = 2
        tp_fp_fn_views["coronal"][fn.max(axis=1)] = 3
        tp_fp_fn_views["sagittal"][tp.max(axis=2)] = 1
        tp_fp_fn_views["sagittal"][fp.max(axis=2)] = 2
        tp_fp_fn_views["sagittal"][fn.max(axis=2)] = 3

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, (title, label_map) in zip(axes, tp_fp_fn_views.items()):
            ax.imshow(_render_tp_fp_fn_rgb(label_map))
            ax.set_title(title)
            ax.axis("off")

        plt.tight_layout()
        base_name = f"{case_id}_mip_tp_fp_fn.png"
        out_name = f"{name_prefix}_{base_name}" if name_prefix else base_name
        out_path = os.path.join(output_dir, out_name)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved TP/FP/FN MIP panel to: {out_path}")
        results.append(
            {
                "case_id": case_id,
                "axis": "mip",
                "slice": None,
                "path": out_path,
                "pred_folder": pred_folder,
                "type": "tp_fp_fn",
            }
        )

        if save_raw:
            raw_views = [
                _render_raw_slice(vol.max(axis=0)),
                _render_raw_slice(vol.max(axis=1)),
                _render_raw_slice(vol.max(axis=2)),
            ]
            fig_raw, axes_raw = plt.subplots(1, 3, figsize=(12, 4))
            for ax, title, raw_view in zip(axes_raw, ("axial", "coronal", "sagittal"), raw_views):
                ax.imshow(raw_view)
                ax.set_title(title)
                ax.axis("off")
            plt.tight_layout()
            raw_base_name = f"{case_id}_mip_raw.png"
            raw_out_name = f"{name_prefix}_{raw_base_name}" if name_prefix else raw_base_name
            raw_out_path = os.path.join(output_dir, raw_out_name)
            fig_raw.savefig(raw_out_path, dpi=200)
            plt.close(fig_raw)
            print(f"Saved raw MIP panel to: {raw_out_path}")
            results.append(
                {
                    "case_id": case_id,
                    "axis": "mip",
                    "slice": None,
                    "path": raw_out_path,
                    "pred_folder": pred_folder,
                    "type": "raw",
                }
            )

        if save_gt_overlay:
            gt_overlay_views = [
                generate_overlay(vol.max(axis=0), gt_fg.max(axis=0).astype(np.int16), overlay_intensity=0.8),
                generate_overlay(vol.max(axis=1), gt_fg.max(axis=1).astype(np.int16), overlay_intensity=0.8),
                generate_overlay(vol.max(axis=2), gt_fg.max(axis=2).astype(np.int16), overlay_intensity=0.8),
            ]
            fig_gt_overlay, axes_gt_overlay = plt.subplots(1, 3, figsize=(12, 4))
            for ax, title, overlay_view in zip(axes_gt_overlay, ("axial", "coronal", "sagittal"), gt_overlay_views):
                ax.imshow(overlay_view)
                ax.set_title(title)
                ax.axis("off")
            plt.tight_layout()
            gt_overlay_base_name = f"{case_id}_mip_gt_overlay.png"
            gt_overlay_out_name = f"{name_prefix}_{gt_overlay_base_name}" if name_prefix else gt_overlay_base_name
            gt_overlay_out_path = os.path.join(output_dir, gt_overlay_out_name)
            fig_gt_overlay.savefig(gt_overlay_out_path, dpi=200)
            plt.close(fig_gt_overlay)
            print(f"Saved GT overlay MIP panel to: {gt_overlay_out_path}")
            results.append(
                {
                    "case_id": case_id,
                    "axis": "mip",
                    "slice": None,
                    "path": gt_overlay_out_path,
                    "pred_folder": pred_folder,
                    "type": "gt_overlay",
                }
            )

    return results


def _infer_model_name_from_pred_folder(pred_folder: str) -> str:
    parts = os.path.normpath(pred_folder).split(os.sep)
    if len(parts) < 2:
        return os.path.basename(pred_folder)
    return parts[-1] or parts[-2]


def _compose_panels_for_folder(output_dir: str, model_names: List[str], fold: int) -> None:
    """Compose multi-model comparison panels from per-model PNGs."""
    single_dir = os.path.join(output_dir, "single")
    panels_dir = os.path.join(output_dir, "panels")
    os.makedirs(panels_dir, exist_ok=True)

    if not os.path.isdir(single_dir):
        print(f"[PANEL] single dir does not exist, skip panels: {single_dir}")
        return

    files = [f for f in os.listdir(single_dir) if f.endswith(".png")]
    if not files:
        print(f"[PANEL] no PNG files found in {single_dir}, nothing to do")
        return

    panel_map = defaultdict(lambda: {"preds": {}, "gt": None, "raw": None, "gt_overlay": None})

    for fn in files:
        name, _ = os.path.splitext(fn)
        parts = name.split("-")
        if len(parts) < 6:
            continue

        fold_str, case_id, axis, slice_str, model_name = parts[:5]
        img_type = "-".join(parts[5:])
        try:
            if int(fold_str) != fold:
                continue
            slice_idx = int(slice_str)
        except ValueError:
            continue

        key = (case_id, axis, slice_idx)
        full_path = os.path.join(single_dir, fn)
        if img_type == "tp_fp_fn":
            panel_map[key]["preds"][model_name] = full_path
        elif img_type == "gt":
            panel_map[key]["gt"] = full_path
        elif img_type == "raw":
            panel_map[key]["raw"] = full_path
        elif img_type == "gt_overlay":
            panel_map[key]["gt_overlay"] = full_path

    for (case_id, axis, slice_idx), item in panel_map.items():
        preds = item["preds"]
        gt_path = item["gt"]
        raw_path = item["raw"]
        gt_overlay_path = item["gt_overlay"]
        if any(model_name not in preds for model_name in model_names):
            continue

        imgs = []
        titles = []
        if raw_path is not None:
            imgs.append(mpimg.imread(raw_path))
            titles.append("Raw")
        if gt_overlay_path is not None:
            imgs.append(mpimg.imread(gt_overlay_path))
            titles.append("GT Overlay")
        imgs.extend(mpimg.imread(preds[m]) for m in model_names)
        titles.extend(model_names)
        if gt_path is not None:
            imgs.append(mpimg.imread(gt_path))
            titles.append("GT")

        cols = len(imgs)
        fig, axes = plt.subplots(1, cols, figsize=(3 * cols, 3.4))
        if cols == 1:
            axes = [axes]

        for ax, img, title in zip(axes, imgs, titles):
            ax.imshow(img)
            ax.set_title(title)
            ax.axis("off")

        legend_lines = [
            "TP = green",
            "FP = red",
            "FN = blue",
        ]
        if gt_path is not None:
            legend_lines.append("GT = white")
        fig.text(0.5, 0.02, " | ".join(legend_lines), ha="center", va="bottom", fontsize=10)
        plt.tight_layout(rect=(0, 0.05, 1, 1))

        panel_name = f"{fold}-{case_id}-{axis}-{slice_idx}-panel.png"
        panel_path = os.path.join(panels_dir, panel_name)
        fig.savefig(panel_path, dpi=160)
        plt.close(fig)
        print(f"[PANEL] Saved panel to: {panel_path}")


def _run_batch(
    data_root: str,
    dataset_directory: str,
    pred_folders: Optional[List[str]],
    model_names: Optional[List[str]],
    fold: int,
    output_dir: str,
    slices: Optional[List[int]],
    save_gt: bool,
    save_raw: bool,
    save_gt_overlay: bool,
    case_ids: Optional[List[str]] = None,
) -> None:
    if not pred_folders:
        pred_folders = list(MODELS_TASK602_LS.values())
        if not model_names:
            model_names = list(MODELS_TASK602_LS.keys())

    if model_names is not None and len(model_names) != len(pred_folders):
        raise ValueError("If model_names is provided, its length must match pred_folders")

    if model_names is None:
        model_names = [_infer_model_name_from_pred_folder(p) for p in pred_folders]

    gt_case_ids = set(_collect_case_ids_from_gt(dataset_directory))
    selected_case_ids = sorted(gt_case_ids if not case_ids else {cid for cid in case_ids if cid in gt_case_ids})
    if case_ids:
        missing_gt = [cid for cid in case_ids if cid not in gt_case_ids]
        if missing_gt:
            print(f"[WARN] {len(missing_gt)} requested cases do not have GT, examples: {missing_gt[:5]}")

    single_dir = os.path.join(output_dir, "single")
    os.makedirs(single_dir, exist_ok=True)

    shared_written = set()
    for pred_folder, model_name in zip(pred_folders, model_names):
        pred_case_ids = set(_collect_case_ids_from_pred_folder(pred_folder))
        case_ids_eff = [cid for cid in selected_case_ids if cid in pred_case_ids]
        missing_pred = [cid for cid in selected_case_ids if cid not in pred_case_ids]
        if missing_pred:
            print(
                f"[WARN] Model {model_name}: {len(missing_pred)} selected cases do not have predictions, "
                f"examples: {missing_pred[:5]}"
            )
        if not case_ids_eff:
            print(f"[WARN] Model {model_name}: no overlapping cases, skip")
            continue

        print(f"[INFO] Processing model {model_name} (fold {fold}) with {len(case_ids_eff)} cases from {pred_folder}")
        for case_id in case_ids_eff:
            for axis in ("z", "y", "x"):
                try:
                    results = visualize_case_tp_fp_fn(
                        data_root=data_root,
                        dataset_directory=dataset_directory,
                        pred_folder=pred_folder,
                        case_id=case_id,
                        output_dir=single_dir,
                        mode="slice",
                        axis=axis,
                        slices=slices,
                        save_gt=save_gt and (case_id, axis) not in shared_written,
                        save_raw=save_raw and (case_id, axis) not in shared_written,
                        save_gt_overlay=save_gt_overlay and (case_id, axis) not in shared_written,
                    )
                    for item in results:
                        slice_idx = item["slice"]
                        img_type = item["type"]
                        old_path = item["path"]
                        ext = os.path.splitext(old_path)[1]
                        if img_type == "tp_fp_fn":
                            out_model_name = model_name
                        elif img_type == "gt":
                            out_model_name = "GT"
                        elif img_type == "raw":
                            out_model_name = "RAW"
                        else:
                            out_model_name = "GTOVERLAY"
                        new_name = f"{fold}-{case_id}-{axis}-{slice_idx}-{out_model_name}-{img_type}{ext}"
                        new_path = os.path.join(single_dir, new_name)
                        os.replace(old_path, new_path)
                        if img_type in {"gt", "raw", "gt_overlay"}:
                            shared_written.add((case_id, axis))
                except Exception as e:
                    print(f"[ERROR] Failed on case {case_id}, axis {axis}, model {model_name}: {e}")

    _compose_panels_for_folder(output_dir=output_dir, model_names=model_names, fold=fold)


def run_task602_default() -> None:
    cfg = TASK602_DEFAULT_CONFIG
    _run_batch(
        data_root=cfg["data_root"],
        dataset_directory=cfg["dataset_directory"],
        pred_folders=None,
        model_names=None,
        fold=cfg["fold"],
        output_dir=cfg["output_dir"],
        slices=cfg["slices"],
        save_gt=cfg["save_gt"],
        save_raw=cfg["save_raw"],
        save_gt_overlay=cfg["save_gt_overlay"],
    )


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize Task602_ls predictions as TP/FP/FN mask maps without overlaying the original image.",
    )
    subparsers = parser.add_subparsers(dest="command")

    p_single = subparsers.add_parser("single", help="Visualize one case")
    p_single.add_argument("--data_root", type=str, required=True, help="Preprocessed stage0 .npy folder")
    p_single.add_argument("--dataset_directory", type=str, required=True, help="Task directory with GT masks")
    p_single.add_argument("--pred_folder", type=str, required=True, help="Folder with prediction NIfTI files")
    p_single.add_argument("--case_id", type=str, required=True, help="Case id to visualize")
    p_single.add_argument("--output_dir", type=str, required=True, help="Where to save output PNGs")
    p_single.add_argument("--mode", type=str, choices=["slice", "mip"], default="slice")
    p_single.add_argument("--axis", type=str, choices=["z", "y", "x"], default="z")
    p_single.add_argument("--slices", type=int, nargs="*", default=None, help="Slice indices for slice mode")
    p_single.add_argument("--save_gt", action="store_true", help="Also export GT masks")
    p_single.add_argument("--save_raw", action="store_true", help="Also export raw image slices")
    p_single.add_argument("--save_gt_overlay", action="store_true", help="Also export GT-overlaid image slices")

    p_batch = subparsers.add_parser("batch", help="Batch visualize multiple models")
    p_batch.add_argument("--data_root", type=str, required=True, help="Preprocessed stage0 .npy folder")
    p_batch.add_argument("--dataset_directory", type=str, required=True, help="Task directory with GT masks")
    p_batch.add_argument(
        "--pred_folders",
        type=str,
        nargs="+",
        required=False,
        help="Prediction folders for multiple models; defaults to built-in Task602 mapping",
    )
    p_batch.add_argument(
        "--model_names",
        type=str,
        nargs="*",
        default=None,
        help="Optional model names aligned with pred_folders",
    )
    p_batch.add_argument("--case_ids", type=str, nargs="*", default=None, help="Optional subset of case ids")
    p_batch.add_argument("--fold", type=int, required=True, help="Fold index used in output filename prefix")
    p_batch.add_argument("--output_dir", type=str, required=True, help="Where to save output PNGs and panels")
    p_batch.add_argument("--slices", type=int, nargs="*", default=None, help="Slice indices for all axes")
    p_batch.add_argument("--save_gt", action="store_true", help="Also export GT masks and append GT to panels")
    p_batch.add_argument("--save_raw", action="store_true", help="Also export raw image slices and append them to panels")
    p_batch.add_argument(
        "--save_gt_overlay",
        action="store_true",
        help="Also export GT-overlaid image slices and append them to panels",
    )

    subparsers.add_parser("task602_default", help="Use built-in Task602_ls config and model list")
    return parser


if __name__ == "__main__":
    parser = _build_argparser()
    args = parser.parse_args()

    if args.command == "single":
        visualize_case_tp_fp_fn(
            data_root=args.data_root,
            dataset_directory=args.dataset_directory,
            pred_folder=args.pred_folder,
            case_id=args.case_id,
            output_dir=args.output_dir,
            mode=args.mode,
            axis=args.axis,
            slices=args.slices,
            save_gt=args.save_gt,
            save_raw=args.save_raw,
            save_gt_overlay=args.save_gt_overlay,
        )
    elif args.command == "batch":
        _run_batch(
            data_root=args.data_root,
            dataset_directory=args.dataset_directory,
            pred_folders=args.pred_folders,
            model_names=args.model_names,
            fold=args.fold,
            output_dir=args.output_dir,
            slices=args.slices,
            save_gt=args.save_gt,
            save_raw=args.save_raw,
            save_gt_overlay=args.save_gt_overlay,
            case_ids=args.case_ids,
        )
    elif args.command == "task602_default":
        run_task602_default()
    else:
        parser.print_help()
