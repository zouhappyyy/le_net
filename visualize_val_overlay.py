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

# python visualize_val_overlay.py task570_default

# Explicit mapping for Task570_EsoTJ83 models and their prediction folders on the Linux host
MODELS_TASK570_EsoTJ83 = {
    # 模型名: preds 目录
    "BANet": "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_predictions/Task570_EsoTJ83/BANetTrainerV2/preds",
    "BGHNetV4": "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_predictions/Task570_EsoTJ83/BGHNetV4Trainer/preds",
    "MedNeXt": "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_predictions/Task570_EsoTJ83/MedNeXt_S_kernel3/preds",
    "nnUNet": "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_predictions/Task570_EsoTJ83/nnUNetTrainerV2/preds",
    "RWKV_fd": "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_predictions/Task570_EsoTJ83/Double_CCA_UPSam_fd_RWKV/preds",
    "RWKV_fd_loss": "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_predictions/Task570_EsoTJ83/Double_CCA_UPSam_fd_loss_RWKV/preds",
    "UNet3D": "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_predictions/Task570_EsoTJ83/UNet3DTrainer/preds",
}

# 默认的 Task570 可视化配置（使用 nnUNet 预处理后的 .npy 数据）
TASK570_DEFAULT_CONFIG = {
    # 请根据实际生成的 plans 名称调整下行路径中的 plans 标识
    "data_root": "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_preprocessed/Task570_EsoTJ83/nnUNetData_plans_v2.1_trgSp_1x1x1_stage0",
    "dataset_directory": "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_preprocessed/Task570_EsoTJ83",
    "fold": 1,
    "output_dir": "./Task570_val_vis_all",
    "alpha": 0.6,
    "slices": None,  # None: 每个方向取中间层；也可以改成 [80, 100, ...]
    "save_gt": True,
}


def _load_pred_nifti(pred_path: str) -> np.ndarray:
    """Load prediction nifti and return array in [Z, Y, X] order.

    nnUNet usually stores NIfTI as [X, Y, Z] or [X, Y, Z, C]. We always
    return a 3D label map [Z, Y, X].
    """
    if not os.path.isfile(pred_path):
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    img = nib.load(pred_path)
    arr = img.get_fdata()

    if arr.ndim == 3:
        # [X, Y, Z] -> [Z, Y, X]
        arr = np.transpose(arr, (2, 1, 0))
    elif arr.ndim == 4:
        # [X, Y, Z, C] -> argmax over channels -> [X, Y, Z] -> [Z, Y, X]
        arr = np.argmax(arr, axis=-1)
        arr = np.transpose(arr, (2, 1, 0))
    else:
        raise RuntimeError(f"Expected 3D or 4D prediction, got shape {arr.shape} from {pred_path}")

    return arr.astype(np.int16)


def _match_shapes(image: np.ndarray, seg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Center-crop seg to match image spatial size if needed.

    image: [C, D, H, W]
    seg: [D, H, W] or [C2, D, H, W] (e.g. GT as [1, D, H, W])
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
        # seg: [C2, D, H, W]
        _, d_s, h_s, w_s = seg.shape
        d = min(d_i, d_s)
        h = min(h_i, h_s)
        w = min(w_i, w_s)
        image_c = image[:, :d, :h, :w]
        seg_c = seg[:, :d, :h, :w]
    else:
        raise RuntimeError(f"Unexpected seg ndim={seg.ndim} in _match_shapes")

    return image_c, seg_c


def visualize_case_overlay(
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
) -> List[Dict[str, object]]:
    """Visualize overlay of original image, prediction, and optional GT for a single case.

    data_root: nnUNet preprocessed stage0 .npy root (nnUNetData_plans..._stage0)
    dataset_directory: nnUNet preprocessed Task directory (contains gt_segmentations, etc.)
    pred_folder: folder containing prediction nifti files (validation_raw or validation_raw_postprocessed)
    case_id: case identifier (e.g. ESO_TJ_60011222468)
    output_dir: where to save pngs
    mode: "slice" (single or multiple 2D slices) or "mip" (3D MIP views)
    axis: for slice mode, one of {"z", "y", "x"}
    slices: optional list of slice indices; if None, use middle slice
    name_prefix: optional string prefix to prepend in filename (e.g. fold-modelname)
    save_gt: if True, also export GT overlay images alongside predictions.

    Returns a list of dicts with information about generated images.
    """
    os.makedirs(output_dir, exist_ok=True)
    results: List[Dict[str, object]] = []

    # 使用 nnUNet 提供的 _extract_case_data 从 .npy 读取图像和标签
    image, gt = _extract_case_data(data_root, case_id, dataset_directory)
    gt = gt[0]  # [1, D, H, W] -> [D, H, W]

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

    _, D, H, W = image.shape

    if mode == "slice":
        if axis not in ("z", "y", "x"):
            raise ValueError("axis must be one of 'z', 'y', 'x'")
        if slices is None or len(slices) == 0:
            if axis == "z":
                slices = [D // 2]
            elif axis == "y":
                slices = [H // 2]
            else:
                slices = [W // 2]

        for s in slices:
            # --- build slices for pred & gt ---
            if axis == "z":
                if not (0 <= s < D):
                    continue
                img_slice = image[0, s]
                pred_slice = pred[s]
                gt_slice = gt[s]
            elif axis == "y":
                if not (0 <= s < H):
                    continue
                img_slice = image[0, :, s, :]
                pred_slice = pred[:, s, :]
                gt_slice = gt[:, s, :]
            else:  # x
                if not (0 <= s < W):
                    continue
                img_slice = image[0, :, :, s]
                pred_slice = pred[:, :, s]
                gt_slice = gt[:, :, s]

            # --- prediction overlay ---
            overlay_pred = generate_overlay(img_slice, pred_slice, overlay_intensity=alpha)

            base_name_pred = f"{case_id}_axis-{axis}_slice-{s}_pred.png"
            if name_prefix:
                out_name_pred = f"{name_prefix}_{base_name_pred}"
            else:
                out_name_pred = base_name_pred

            out_path_pred = os.path.join(output_dir, out_name_pred)
            plt.imsave(out_path_pred, overlay_pred)
            print(f"Saved pred overlay to: {out_path_pred}")
            results.append(
                {
                    "case_id": case_id,
                    "axis": axis,
                    "slice": s,
                    "path": out_path_pred,
                    "pred_folder": pred_folder,
                    "type": "pred",
                }
            )

            # --- GT overlay ---
            if save_gt:
                overlay_gt = generate_overlay(img_slice, gt_slice, overlay_intensity=alpha)

                base_name_gt = f"{case_id}_axis-{axis}_slice-{s}_gt.png"
                if name_prefix:
                    out_name_gt = f"{name_prefix}_{base_name_gt}"
                else:
                    out_name_gt = base_name_gt

                out_path_gt = os.path.join(output_dir, out_name_gt)
                plt.imsave(out_path_gt, overlay_gt)
                print(f"Saved GT overlay to: {out_path_gt}")
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
        # Simple 3-view MIP overlay
        vol = image[0]
        # Axial (Z): max over depth
        axial_img = vol.max(axis=0)
        axial_pred = (pred > 0).max(axis=0).astype(np.int16)
        axial_overlay = generate_overlay(axial_img, axial_pred, overlay_intensity=alpha)

        # Coronal (Y): max over height
        cor_img = vol.max(axis=1)
        cor_pred = (pred > 0).max(axis=1).astype(np.int16)
        cor_overlay = generate_overlay(cor_img, cor_pred, overlay_intensity=alpha)

        # Sagittal (X): max over width
        sag_img = vol.max(axis=2)
        sag_pred = (pred > 0).max(axis=2).astype(np.int16)
        sag_overlay = generate_overlay(sag_img, sag_pred, overlay_intensity=alpha)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(axial_overlay)
        axes[0].set_title("Axial MIP")
        axes[0].axis("off")

        axes[1].imshow(cor_overlay)
        axes[1].set_title("Coronal MIP")
        axes[1].axis("off")

        axes[2].imshow(sag_overlay)
        axes[2].set_title("Sagittal MIP")
        axes[2].axis("off")

        plt.tight_layout()
        base_name = f"{case_id}_mip.png"
        if name_prefix:
            out_name = f"{name_prefix}_{base_name}"
        else:
            out_name = base_name
        out_path = os.path.join(output_dir, out_name)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved MIP overlays to: {out_path}")
        results.append(
            {
                "case_id": case_id,
                "axis": "mip",
                "slice": None,
                "path": out_path,
                "pred_folder": pred_folder,
            }
        )

    else:
        raise ValueError("mode must be 'slice' or 'mip'")

    return results


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize overlay of validation predictions and original images (single case or batch over models)",
    )
    # Make subcommands mandatory so we don't have ambiguous legacy mode when a subcommand is passed
    subparsers = parser.add_subparsers(dest="command")

    # single-case mode (backwards compatible)
    p_single = subparsers.add_parser("single", help="Visualize one case")
    p_single.add_argument("--data_root", type=str, required=True, help="Folder with preprocessed stage0 .npy volumes")
    p_single.add_argument(
        "--dataset_directory",
        type=str,
        required=True,
        help="nnUNet dataset directory (contains gt_segmentations)",
    )
    p_single.add_argument(
        "--pred_folder",
        type=str,
        required=True,
        help="Folder with prediction NIfTI files (e.g. validation_raw_postprocessed)",
    )
    p_single.add_argument("--case_id", type=str, required=True, help="Case id to visualize (e.g. ESO_TJ_60011222468)")
    p_single.add_argument("--output_dir", type=str, required=True, help="Where to save output PNGs")
    p_single.add_argument("--alpha", type=float, default=0.6, help="Overlay intensity (alpha) for segmentation")
    p_single.add_argument(
        "--mode",
        type=str,
        choices=["slice", "mip"],
        default="slice",
        help="Visualization mode: slice or mip",
    )
    p_single.add_argument(
        "--axis",
        type=str,
        choices=["z", "y", "x"],
        default="z",
        help="Slice axis for slice mode",
    )
    p_single.add_argument(
        "--slices",
        type=int,
        nargs="*",
        default=None,
        help="Slice indices to visualize (if empty, use middle slice)",
    )

    # batch mode: multiple models, all cases, all three axes
    p_batch = subparsers.add_parser("batch", help="Batch visualize all cases for multiple models")
    p_batch.add_argument("--data_root", type=str, required=True, help="Folder with preprocessed stage0 .npy volumes")
    p_batch.add_argument(
        "--dataset_directory",
        type=str,
        required=True,
        help="nnUNet dataset directory (contains gt_segmentations)",
    )
    p_batch.add_argument(
        "--pred_folders",
        type=str,
        nargs="+",
        required=False,
        help=(
            "List of prediction folders (e.g. different models' validation_raw_postprocessed). "
            "If omitted, the built-in MODELS_TASK570_EsoTJ83 mapping will be used. "
            "Model name will be inferred from the folder path unless --model_names is given."
        ),
    )
    p_batch.add_argument(
        "--model_names",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Optional explicit model names aligned with pred_folders. "
            "If omitted and pred_folders is also omitted, we will use the keys of MODELS_TASK570_EsoTJ83."
        ),
    )
    p_batch.add_argument(
        "--fold", type=int, required=True, help="Fold index used in filename prefix (e.g. 0/1/2/3/4)",
    )
    p_batch.add_argument(
        "--output_dir", type=str, required=True, help="Base folder where visualization PNGs will be stored",
    )
    p_batch.add_argument(
        "--alpha", type=float, default=0.6, help="Overlay intensity (alpha) for segmentation",
    )
    p_batch.add_argument(
        "--slices",
        type=int,
        nargs="*",
        default=None,
        help="Slice indices to visualize. If omitted, middle slice along each axis is used.",
    )
    p_batch.add_argument("--save_gt", action="store_true", help="Also export GT overlays for each slice")

    # task570 default mode: use built-in config and model list
    subparsers.add_parser(
        "task570_default",
        help="Use built-in Task570_EsoTJ83 config and model list to batch visualize all cases",
    )

    # Remove legacy top-level arguments to avoid conflicts
    return parser


def _infer_model_name_from_pred_folder(pred_folder: str) -> str:
    """Infer a model name from the prediction folder path.

    For typical nnUNet layout, pred_folder is something like:
      .../TaskXXX/TrainerName__Plans/fold_1/validation_raw_postprocessed
    We take the TrainerName__Plans part as model name by default.
    """
    parts = os.path.normpath(pred_folder).split(os.sep)
    if len(parts) < 3:
        return os.path.basename(pred_folder)
    # ... fold_x / validation_raw_postprocessed / Trainer__Plans / Task...
    # assume trainer dir is two levels above pred_folder
    trainer_dir = parts[-3]
    return trainer_dir


def _collect_case_ids_from_pred_folder(pred_folder: str) -> List[str]:
    """List unique case_ids from NIfTI files in prediction folder."""
    if not os.path.isdir(pred_folder):
        raise FileNotFoundError(f"Prediction folder not found: {pred_folder}")
    ids = []
    for fn in os.listdir(pred_folder):
        if fn.endswith(".nii") or fn.endswith(".nii.gz"):
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
) -> None:
    # If no pred_folders provided, fall back to the explicit mapping for Task570_EsoTJ83
    if not pred_folders:
        pred_folders = list(MODELS_TASK570_EsoTJ83.values())
        # if model_names not given, use the keys of the mapping
        if model_names is None or len(model_names) == 0:
            model_names = list(MODELS_TASK570_EsoTJ83.keys())

    # normalize model names
    if model_names is not None and len(model_names) != len(pred_folders):
        raise ValueError("If model_names is provided, its length must match pred_folders")

    if model_names is None:
        model_names = [_infer_model_name_from_pred_folder(p) for p in pred_folders]

    # create subfolder for per-model overlays
    single_dir = os.path.join(output_dir, "single")
    os.makedirs(single_dir, exist_ok=True)

    for pred_folder, model_name in zip(pred_folders, model_names):
        case_ids = _collect_case_ids_from_pred_folder(pred_folder)
        if not case_ids:
            print(f"[WARN] No prediction files found in {pred_folder}, skip")
            continue

        print(f"[INFO] Processing model {model_name} (fold {fold}) with {len(case_ids)} cases from {pred_folder}")

        for case_id in case_ids:
            for axis in ("z", "y", "x"):
                try:
                    results = visualize_case_overlay(
                        data_root=data_root,
                        dataset_directory=dataset_directory,
                        pred_folder=pred_folder,
                        case_id=case_id,
                        output_dir=single_dir,
                        alpha=alpha,
                        mode="slice",
                        axis=axis,
                        slices=slices,
                        name_prefix=None,
                        save_gt=save_gt,
                    )
                    # rename files to pattern in the single/ subfolder: fold-id-axis-sliceindex-modelname[-type]
                    for item in results:
                        slice_idx = item["slice"]
                        img_type = item.get("type", "pred")
                        old_path = item["path"]
                        ext = os.path.splitext(old_path)[1]
                        new_name = f"{fold}-{case_id}-{axis}-{slice_idx}-{model_name}-{img_type}{ext}"
                        new_path = os.path.join(single_dir, new_name)
                        os.replace(old_path, new_path)
                except Exception as e:
                    print(f"[ERROR] Failed on case {case_id}, axis {axis}, model {model_name}: {e}")


def _compose_panels_for_folder(
    output_dir: str,
    model_names: List[str],
    fold: int,
) -> None:
    """Compose panel images (7 models + GT) from per-model overlays.

    Reads from `<output_dir>/single` and writes panels into `<output_dir>/panels`.
    Each panel is named: fold-case-axis-slice-panel.png
    """
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

    panel_map = defaultdict(lambda: {"preds": {}, "gt": None})

    for fn in files:
        name, _ = os.path.splitext(fn)
        parts = name.split("-")
        # expect: fold-case-axis-slice-model-type
        if len(parts) < 6:
            continue
        fold_str, case_id, axis, slice_str, model_name, img_type = parts
        try:
            if int(fold_str) != fold:
                continue
            slice_idx = int(slice_str)
        except ValueError:
            continue

        key = (case_id, axis, slice_idx)
        full_path = os.path.join(single_dir, fn)
        if img_type == "pred":
            panel_map[key]["preds"][model_name] = full_path
        elif img_type == "gt":
            # any GT for this key is fine
            panel_map[key]["gt"] = full_path

    order = list(model_names)

    for (case_id, axis, slice_idx), item in panel_map.items():
        preds = item["preds"]
        gt_path = item["gt"]
        # require all models and a GT to be present
        if gt_path is None or any(m not in preds for m in order):
            continue

        imgs = []
        titles = []
        for m in order:
            img = mpimg.imread(preds[m])
            imgs.append(img)
            titles.append(m)
        # last column is GT
        imgs.append(mpimg.imread(gt_path))
        titles.append("GT")

        cols = len(imgs)
        fig, axes = plt.subplots(1, cols, figsize=(3 * cols, 3))
        if cols == 1:
            axes = [axes]
        for ax, img, title in zip(axes, imgs, titles):
            ax.imshow(img)
            ax.set_title(title)
            ax.axis("off")

        plt.tight_layout()
        panel_name = f"{fold}-{case_id}-{axis}-{slice_idx}-panel.png"
        panel_path = os.path.join(panels_dir, panel_name)
        fig.savefig(panel_path, dpi=150)
        plt.close(fig)
        print(f"[PANEL] Saved panel to: {panel_path}")


def run_task570_batch_default() -> None:
    """One-click batch visualization for Task570_EsoTJ83 using built-in paths and params.

    1) 使用 MODELS_TASK570_EsoTJ83 对 7 个模型分别输出 overlay 图（含 GT），存放在 output_dir/single。
    2) 对每个 case+axis+slice 生成 1 张大图（7 个模型 + GT），存放在 output_dir/panels。
    """
    cfg = TASK570_DEFAULT_CONFIG
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
    )

    model_names = list(MODELS_TASK570_EsoTJ83.keys())
    _compose_panels_for_folder(
        output_dir=out_dir,
        model_names=model_names,
        fold=cfg["fold"],
    )


if __name__ == "__main__":
    parser = _build_argparser()
    args = parser.parse_args()

    cmd = getattr(args, "command", None)
    if cmd == "single":
        visualize_case_overlay(
            data_root=args.data_root,
            dataset_directory=args.dataset_directory,
            pred_folder=args.pred_folder,
            case_id=args.case_id,
            output_dir=args.output_dir,
            alpha=args.alpha,
            mode=args.mode,
            axis=args.axis,
            slices=args.slices,
        )
    elif cmd == "batch":
        _run_batch(
            data_root=args.data_root,
            dataset_directory=args.dataset_directory,
            pred_folders=args.pred_folders,
            model_names=args.model_names,
            fold=args.fold,
            output_dir=args.output_dir,
            alpha=args.alpha,
            slices=args.slices,
            save_gt=getattr(args, "save_gt", False),
        )
    elif cmd == "task570_default":
        run_task570_batch_default()
    else:
        parser.print_help()
