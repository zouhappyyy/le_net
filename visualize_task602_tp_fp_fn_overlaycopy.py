import os
import argparse
from typing import List, Optional, Tuple, Dict
from collections import defaultdict

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from nnunet_mednext.utilities.overlay_plots import generate_overlay
from visualize_fd_edge_and_ds import _extract_case_data


MODELS_TASK602_EsoTJ83 = {
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
    "output_dir": "./Task602_tp_fp_fn_vis_overlaycopy",
    "alpha": 0.99,
    "slices": None,
    "save_gt": True,
    "save_raw": True,
    "save_gt_overlay": True,
}


COLOR_BACKGROUND = np.array([0, 0, 0], dtype=np.uint8)
COLOR_TP = np.array([34, 197, 94], dtype=np.uint8)
COLOR_FP = np.array([239, 68, 68], dtype=np.uint8)
COLOR_FN = np.array([59, 130, 246], dtype=np.uint8)
COLOR_GT = np.array([255, 255, 255], dtype=np.uint8)


def _load_pred_nifti(pred_path: str) -> np.ndarray:
    """Load prediction nifti and return array in [Z, Y, X] order."""
    if not os.path.isfile(pred_path):
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    img = nib.load(pred_path)
    arr = img.get_fdata()

    if arr.ndim == 3:
        arr = np.transpose(arr, (2, 1, 0))
    elif arr.ndim == 4:
        arr = np.argmax(arr, axis=-1)
        arr = np.transpose(arr, (2, 1, 0))
    else:
        raise RuntimeError(f"Expected 3D or 4D prediction, got shape {arr.shape} from {pred_path}")

    return arr.astype(np.int16)


def _match_shapes(image: np.ndarray, seg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Match shapes exactly like visualize_val_overlay.py."""
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
        raise RuntimeError(f"Unexpected seg ndim={seg.ndim} in _match_shapes")

    return image_c, seg_c


def _render_raw_slice(img_slice: np.ndarray) -> np.ndarray:
    image = img_slice.astype(np.float32, copy=True)
    image -= image.min()
    max_val = image.max()
    if max_val > 0:
        image = image / max_val
    image = (image * 255).clip(0, 255).astype(np.uint8)
    return np.stack([image, image, image], axis=-1)


def _make_tp_fp_fn_map(pred_slice: np.ndarray, gt_slice: np.ndarray) -> np.ndarray:
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


def visualize_case_tp_fp_fn(
    data_root: str,
    dataset_directory: str,
    pred_folder: str,
    case_id: str,
    output_dir: str,
    alpha: float = 0.6,
    mode: str = "slice",
    axis: str = "z",
    slices: Optional[List[int]] = None,
    name_prefix: str = "",
    save_gt: bool = False,
    save_raw: bool = False,
    save_gt_overlay: bool = False,
) -> List[Dict[str, object]]:
    """Copied from visualize_val_overlay.py, but pred rendering becomes TP/FP/FN."""
    os.makedirs(output_dir, exist_ok=True)
    results: List[Dict[str, object]] = []

    image, gt = _extract_case_data(data_root, case_id, dataset_directory)
    gt = gt[0]

    pred_path = None
    for ext in (".nii.gz", ".nii"):
        cand = os.path.join(pred_folder, f"{case_id}{ext}")
        if os.path.isfile(cand):
            pred_path = cand
            break
    if pred_path is None:
        raise FileNotFoundError(f"Could not find prediction for case {case_id} in {pred_folder}")

    pred = _load_pred_nifti(pred_path)

    image, pred = _match_shapes(image, pred)
    _, gt = _match_shapes(image, gt)

    _, d, h, w = image.shape

    if mode == "slice":
        if axis not in ("z", "y", "x"):
            raise ValueError("axis must be one of 'z', 'y', 'x'")
        if slices is None or len(slices) == 0:
            if axis == "z":
                slices = [d // 2]
            elif axis == "y":
                slices = [h // 2]
            else:
                slices = [w // 2]

        for s in slices:
            if axis == "z":
                if not (0 <= s < d):
                    continue
                img_slice = image[0, s]
                pred_slice = pred[s]
                gt_slice = gt[s]
            elif axis == "y":
                if not (0 <= s < h):
                    continue
                img_slice = image[0, :, s, :]
                pred_slice = pred[:, s, :]
                gt_slice = gt[:, s, :]
            else:
                if not (0 <= s < w):
                    continue
                img_slice = image[0, :, :, s]
                pred_slice = pred[:, :, s]
                gt_slice = gt[:, :, s]

            if save_raw:
                raw_img = _render_raw_slice(img_slice)
                base_name_raw = f"{case_id}_axis-{axis}_slice-{s}_raw.png"
                out_name_raw = f"{name_prefix}_{base_name_raw}" if name_prefix else base_name_raw
                out_path_raw = os.path.join(output_dir, out_name_raw)
                plt.imsave(out_path_raw, raw_img)
                print(f"Saved raw slice to: {out_path_raw}")
                results.append(
                    {
                        "case_id": case_id,
                        "axis": axis,
                        "slice": s,
                        "path": out_path_raw,
                        "pred_folder": pred_folder,
                        "type": "raw",
                    }
                )

            if save_gt_overlay:
                overlay_gt = generate_overlay(img_slice, gt_slice, overlay_intensity=alpha)
                base_name_gt_overlay = f"{case_id}_axis-{axis}_slice-{s}_gt_overlay.png"
                out_name_gt_overlay = (
                    f"{name_prefix}_{base_name_gt_overlay}" if name_prefix else base_name_gt_overlay
                )
                out_path_gt_overlay = os.path.join(output_dir, out_name_gt_overlay)
                plt.imsave(out_path_gt_overlay, overlay_gt)
                print(f"Saved GT overlay to: {out_path_gt_overlay}")
                results.append(
                    {
                        "case_id": case_id,
                        "axis": axis,
                        "slice": s,
                        "path": out_path_gt_overlay,
                        "pred_folder": pred_folder,
                        "type": "gt_overlay",
                    }
                )

            tp_fp_fn = _make_tp_fp_fn_map(pred_slice, gt_slice)
            tp_fp_fn_rgb = _render_tp_fp_fn_rgb(tp_fp_fn)
            base_name_pred = f"{case_id}_axis-{axis}_slice-{s}_tp_fp_fn.png"
            out_name_pred = f"{name_prefix}_{base_name_pred}" if name_prefix else base_name_pred
            out_path_pred = os.path.join(output_dir, out_name_pred)
            plt.imsave(out_path_pred, tp_fp_fn_rgb)
            print(f"Saved TP/FP/FN map to: {out_path_pred}")
            results.append(
                {
                    "case_id": case_id,
                    "axis": axis,
                    "slice": s,
                    "path": out_path_pred,
                    "pred_folder": pred_folder,
                    "type": "tp_fp_fn",
                }
            )

            if save_gt:
                gt_rgb = _render_binary_mask(gt_slice, COLOR_GT)
                base_name_gt = f"{case_id}_axis-{axis}_slice-{s}_gt.png"
                out_name_gt = f"{name_prefix}_{base_name_gt}" if name_prefix else base_name_gt
                out_path_gt = os.path.join(output_dir, out_name_gt)
                plt.imsave(out_path_gt, gt_rgb)
                print(f"Saved GT mask to: {out_path_gt}")
                results.append(
                    {
                        "case_id": case_id,
                        "axis": axis,
                        "slice": s,
                        "path": out_path_gt,
                        "pred_folder": pred_folder,
                        "type": "gt",
                    }
                )

    elif mode == "mip":
        vol = image[0]
        pred_fg = pred > 0
        gt_fg = gt > 0
        tp = np.logical_and(pred_fg, gt_fg)
        fp = np.logical_and(pred_fg, ~gt_fg)
        fn = np.logical_and(~pred_fg, gt_fg)

        axial = np.zeros(tp.max(axis=0).shape, dtype=np.uint8)
        cor = np.zeros(tp.max(axis=1).shape, dtype=np.uint8)
        sag = np.zeros(tp.max(axis=2).shape, dtype=np.uint8)

        axial[tp.max(axis=0)] = 1
        axial[fp.max(axis=0)] = 2
        axial[fn.max(axis=0)] = 3
        cor[tp.max(axis=1)] = 1
        cor[fp.max(axis=1)] = 2
        cor[fn.max(axis=1)] = 3
        sag[tp.max(axis=2)] = 1
        sag[fp.max(axis=2)] = 2
        sag[fn.max(axis=2)] = 3

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, title, label_map in zip(axes, ("Axial", "Coronal", "Sagittal"), (axial, cor, sag)):
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

    else:
        raise ValueError("mode must be 'slice' or 'mip'")

    return results


def _collect_case_ids_from_pred_folder(pred_folder: str) -> List[str]:
    if not os.path.isdir(pred_folder):
        raise FileNotFoundError(f"Prediction folder not found: {pred_folder}")
    ids = []
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


def _collect_case_ids_from_preprocessed(data_root: str) -> List[str]:
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"Preprocessed data_root not found: {data_root}")
    ids: List[str] = []
    for fn in os.listdir(data_root):
        if fn.endswith(".npy") or fn.endswith(".npz"):
            case_id = fn.split(".")[0]
            if case_id not in ids:
                ids.append(case_id)
    ids.sort()
    return ids


def _run_batch(
    data_root: str,
    dataset_directory: str,
    pred_folders: List[str],
    model_names: Optional[List[str]],
    fold: int,
    output_dir: str,
    alpha: float,
    slices: Optional[List[int]],
    save_gt: bool,
    save_raw: bool,
    save_gt_overlay: bool,
) -> None:
    if not pred_folders:
        pred_folders = list(MODELS_TASK602_EsoTJ83.values())
        if model_names is None or len(model_names) == 0:
            model_names = list(MODELS_TASK602_EsoTJ83.keys())

    if model_names is not None and len(model_names) != len(pred_folders):
        raise ValueError("If model_names is provided, its length must match pred_folders")

    if model_names is None:
        model_names = [os.path.basename(os.path.normpath(p)) for p in pred_folders]

    preproc_case_ids = set(_collect_case_ids_from_preprocessed(data_root))
    single_dir = os.path.join(output_dir, "single")
    os.makedirs(single_dir, exist_ok=True)

    shared_written = set()
    for pred_folder, model_name in zip(pred_folders, model_names):
        pred_case_ids = _collect_case_ids_from_pred_folder(pred_folder)
        if not pred_case_ids:
            print(f"[WARN] No prediction files found in {pred_folder}, skip")
            continue

        case_ids = [cid for cid in pred_case_ids if cid in preproc_case_ids]
        if not case_ids:
            print(f"[WARN] No overlapping cases between preds and preprocessed for model {model_name}, skip")
            continue

        print(
            f"[INFO] Processing model {model_name} (fold {fold}) with {len(case_ids)} cases "
            f"(out of {len(pred_case_ids)} preds) from {pred_folder}"
        )

        for case_id in case_ids:
            for axis in ("z", "y", "x"):
                try:
                    results = visualize_case_tp_fp_fn(
                        data_root=data_root,
                        dataset_directory=dataset_directory,
                        pred_folder=pred_folder,
                        case_id=case_id,
                        output_dir=single_dir,
                        alpha=alpha,
                        mode="slice",
                        axis=axis,
                        slices=slices,
                        save_gt=save_gt,
                        save_raw=False,
                        save_gt_overlay=False,
                    )

                    if save_raw or save_gt_overlay:
                        key_slices = []
                        if slices is None or len(slices) == 0:
                            image_tmp, _ = _extract_case_data(data_root, case_id, dataset_directory)
                            _, d, h, w = image_tmp.shape
                            if axis == "z":
                                key_slices = [d // 2]
                            elif axis == "y":
                                key_slices = [h // 2]
                            else:
                                key_slices = [w // 2]
                        else:
                            key_slices = list(slices)

                        for s in key_slices:
                            if (case_id, axis, s) in shared_written:
                                continue
                            aux_results = visualize_case_tp_fp_fn(
                                data_root=data_root,
                                dataset_directory=dataset_directory,
                                pred_folder=pred_folder,
                                case_id=case_id,
                                output_dir=single_dir,
                                alpha=alpha,
                                mode="slice",
                                axis=axis,
                                slices=[s],
                                save_gt=False,
                                save_raw=save_raw,
                                save_gt_overlay=save_gt_overlay,
                            )
                            for item in aux_results:
                                slice_idx = item["slice"]
                                img_type = item["type"]
                                old_path = item["path"]
                                ext = os.path.splitext(old_path)[1]
                                if img_type == "raw":
                                    out_model_name = "RAW"
                                elif img_type == "gt_overlay":
                                    out_model_name = "GTOVERLAY"
                                else:
                                    continue
                                new_name = f"{fold}-{case_id}-{axis}-{slice_idx}-{out_model_name}-{img_type}{ext}"
                                new_path = os.path.join(single_dir, new_name)
                                os.replace(old_path, new_path)
                                shared_written.add((case_id, axis, slice_idx))

                    for item in results:
                        slice_idx = item["slice"]
                        img_type = item["type"]
                        old_path = item["path"]
                        ext = os.path.splitext(old_path)[1]
                        out_model_name = model_name if img_type == "tp_fp_fn" else "GT"
                        new_name = f"{fold}-{case_id}-{axis}-{slice_idx}-{out_model_name}-{img_type}{ext}"
                        new_path = os.path.join(single_dir, new_name)
                        os.replace(old_path, new_path)
                except Exception as e:
                    print(f"[ERROR] Failed on case {case_id}, axis {axis}, model {model_name}: {e}")


def _compose_panels_for_folder(output_dir: str, model_names: List[str], fold: int) -> None:
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

    order = list(model_names)
    for (case_id, axis, slice_idx), item in panel_map.items():
        preds = item["preds"]
        gt_path = item["gt"]
        raw_path = item["raw"]
        gt_overlay_path = item["gt_overlay"]
        if any(m not in preds for m in order):
            continue

        imgs = []
        titles = []
        if raw_path is not None:
            imgs.append(mpimg.imread(raw_path))
            titles.append("Raw")
        if gt_overlay_path is not None:
            imgs.append(mpimg.imread(gt_overlay_path))
            titles.append("GT Overlay")
        for m in order:
            imgs.append(mpimg.imread(preds[m]))
            titles.append(m)
        if gt_path is not None:
            imgs.append(mpimg.imread(gt_path))
            titles.append("GT")

        cols = len(imgs)
        fig, axes = plt.subplots(1, cols, figsize=(3 * cols, 3.2))
        if cols == 1:
            axes = [axes]
        for ax, img, title in zip(axes, imgs, titles):
            ax.imshow(img)
            ax.set_title(title)
            ax.axis("off")

        fig.text(0.5, 0.02, "TP = green | FP = red | FN = blue | GT = white", ha="center", va="bottom", fontsize=10)
        plt.tight_layout(rect=(0, 0.05, 1, 1))
        panel_name = f"{fold}-{case_id}-{axis}-{slice_idx}-panel.png"
        panel_path = os.path.join(panels_dir, panel_name)
        fig.savefig(panel_path, dpi=150)
        plt.close(fig)
        print(f"[PANEL] Saved panel to: {panel_path}")


def run_task602_batch_default() -> None:
    cfg = TASK602_DEFAULT_CONFIG
    out_dir = cfg["output_dir"]
    _run_batch(
        data_root=cfg["data_root"],
        dataset_directory=cfg["dataset_directory"],
        pred_folders=None,
        model_names=None,
        fold=cfg["fold"],
        output_dir=out_dir,
        alpha=cfg["alpha"],
        slices=cfg["slices"],
        save_gt=cfg["save_gt"],
        save_raw=cfg["save_raw"],
        save_gt_overlay=cfg["save_gt_overlay"],
    )
    _compose_panels_for_folder(
        output_dir=out_dir,
        model_names=list(MODELS_TASK602_EsoTJ83.keys()),
        fold=cfg["fold"],
    )


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Task602 TP/FP/FN visualization copied from visualize_val_overlay.py logic.",
    )
    subparsers = parser.add_subparsers(dest="command")

    p_single = subparsers.add_parser("single", help="Visualize one case")
    p_single.add_argument("--data_root", type=str, required=True)
    p_single.add_argument("--dataset_directory", type=str, required=True)
    p_single.add_argument("--pred_folder", type=str, required=True)
    p_single.add_argument("--case_id", type=str, required=True)
    p_single.add_argument("--output_dir", type=str, required=True)
    p_single.add_argument("--alpha", type=float, default=0.6)
    p_single.add_argument("--mode", type=str, choices=["slice", "mip"], default="slice")
    p_single.add_argument("--axis", type=str, choices=["z", "y", "x"], default="z")
    p_single.add_argument("--slices", type=int, nargs="*", default=None)
    p_single.add_argument("--save_gt", action="store_true")
    p_single.add_argument("--save_raw", action="store_true")
    p_single.add_argument("--save_gt_overlay", action="store_true")

    p_batch = subparsers.add_parser("batch", help="Batch visualize all cases for multiple models")
    p_batch.add_argument("--data_root", type=str, required=True)
    p_batch.add_argument("--dataset_directory", type=str, required=True)
    p_batch.add_argument("--pred_folders", type=str, nargs="+", required=False)
    p_batch.add_argument("--model_names", type=str, nargs="*", default=None)
    p_batch.add_argument("--fold", type=int, required=True)
    p_batch.add_argument("--output_dir", type=str, required=True)
    p_batch.add_argument("--alpha", type=float, default=0.6)
    p_batch.add_argument("--slices", type=int, nargs="*", default=None)
    p_batch.add_argument("--save_gt", action="store_true")
    p_batch.add_argument("--save_raw", action="store_true")
    p_batch.add_argument("--save_gt_overlay", action="store_true")

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
            alpha=args.alpha,
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
            alpha=args.alpha,
            slices=args.slices,
            save_gt=args.save_gt,
            save_raw=args.save_raw,
            save_gt_overlay=args.save_gt_overlay,
        )
        used_model_names = args.model_names if args.model_names else list(MODELS_TASK602_EsoTJ83.keys())
        _compose_panels_for_folder(args.output_dir, used_model_names, args.fold)
    elif args.command == "task602_default":
        run_task602_batch_default()
    else:
        parser.print_help()
