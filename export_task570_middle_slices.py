import os
import argparse
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from nnunet_mednext.utilities.overlay_plots import generate_overlay
from visualize_fd_edge_and_ds import _extract_case_data
from visualize_val_overlay import _match_shapes


def _collect_case_ids_from_preprocessed(data_root: str) -> List[str]:
    """Collect available case_ids from nnUNet preprocessed stage0 directory.

    Supports both .npy and .npz files.
    """
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


def _prepare_middle_indices(shape_3d: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Given (D, H, W) return middle indices (z_mid, y_mid, x_mid)."""
    d, h, w = shape_3d
    return d // 2, h // 2, w // 2


def _ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _export_case_middle_slices(
    data_root: str,
    dataset_directory: str,
    case_id: str,
    output_dir: str,
    alpha: float = 0.99,
) -> None:
    """Export middle-slice CT and CT+GT-overlay PNGs along z, y, x for one case.

    - Loads image & GT with _extract_case_data (nnUNet preprocessed .npy/.npz).
    - Matches shapes using the same logic as visualize_val_overlay._match_shapes.
    - Computes middle slices for each axis and saves:
        * {case_id}_axis-{axis}_mid-{idx}_ct.png
        * {case_id}_axis-{axis}_mid-{idx}_ct_gt.png
    """
    # image: [C, D, H, W], gt_raw: [1, D, H, W]
    image, gt_raw = _extract_case_data(data_root, case_id, dataset_directory)
    gt = gt_raw[0]  # [D, H, W]

    # ensure shapes match (handles potential cropping)
    image_m, gt_m = _match_shapes(image, gt)

    # image channel 0 is used for visualization
    _, D, H, W = image_m.shape
    z_mid, y_mid, x_mid = _prepare_middle_indices((D, H, W))

    # Each axis: extract CT slice and GT slice in the same orientation
    axes = {
        "z": z_mid,
        "y": y_mid,
        "x": x_mid,
    }

    for axis, idx in axes.items():
        if axis == "z":
            img_slice = image_m[0, idx]        # [H, W]
            gt_slice = gt_m[idx]               # [H, W]
        elif axis == "y":
            img_slice = image_m[0, :, idx, :]  # [D, W] (but D==H in naming above)
            gt_slice = gt_m[:, idx, :]
        else:  # "x"
            img_slice = image_m[0, :, :, idx]  # [D, H]
            gt_slice = gt_m[:, :, idx]

        # Save raw CT slice
        ct_name = f"{case_id}_axis-{axis}_mid-{idx}_ct.png"
        ct_path = os.path.join(output_dir, ct_name)
        plt.imsave(ct_path, img_slice, cmap="gray")

        # Build mapping so all foreground labels are the same red color
        uniques = np.unique(gt_slice)
        uniques = uniques.tolist()
        if 0 not in uniques:
            uniques = [0] + uniques
        # 4 corresponds to a red color in generate_overlay's default color cycle
        mapping = {l: (0 if l == 0 else 4) for l in uniques}

        overlay = generate_overlay(img_slice, gt_slice, mapping=mapping, overlay_intensity=alpha)
        overlay_name = f"{case_id}_axis-{axis}_mid-{idx}_ct_gt.png"
        overlay_path = os.path.join(output_dir, overlay_name)
        plt.imsave(overlay_path, overlay)

        print(f"Saved {axis}-axis middle slices for {case_id}:\n  CT -> {ct_path}\n  CT+GT -> {overlay_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export middle-slice CT images and CT+GT overlays (red mask, alpha=0.99) "
            "along x/y/z for nnUNet preprocessed datasets (e.g. Task570_EsoTJ83)."
        )
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help=(
            "Path to nnUNet preprocessed stage0 data (e.g. "
            ".../Task570_EsoTJ83/nnUNetData_plans_v2.1_trgSp_1x1x1_stage0)"
        ),
    )
    parser.add_argument(
        "--dataset_directory",
        type=str,
        required=True,
        help=(
            "nnUNet dataset directory that contains gt_segmentations (e.g. "
            ".../Task570_EsoTJ83)"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to save the exported PNGs.",
    )
    parser.add_argument(
        "--case_ids",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Optional list of case IDs to process. If omitted, all cases under "
            "data_root (.npy/.npz) will be used."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.99,
        help="Overlay transparency for GT mask (default: 0.99)",
    )

    args = parser.parse_args()

    data_root = args.data_root
    dataset_directory = args.dataset_directory
    output_dir = args.output_dir
    alpha = args.alpha

    _ensure_output_dir(output_dir)

    if args.case_ids:
        available = set(_collect_case_ids_from_preprocessed(data_root))
        case_ids = []
        missing = []
        for cid in args.case_ids:
            if cid in available:
                case_ids.append(cid)
            else:
                missing.append(cid)
        if missing:
            print(
                f"[WARN] {len(missing)} requested case_ids do not exist in data_root; "
                f"examples: {missing[:5]}"
            )
        if not case_ids:
            print("[ERROR] No valid case_ids to process after filtering; exit.")
            return
    else:
        case_ids = _collect_case_ids_from_preprocessed(data_root)

    print(f"[INFO] Found {len(case_ids)} cases to process.")

    for cid in case_ids:
        try:
            _export_case_middle_slices(
                data_root=data_root,
                dataset_directory=dataset_directory,
                case_id=cid,
                output_dir=output_dir,
                alpha=alpha,
            )
        except Exception as e:
            print(f"[ERROR] Failed to export middle slices for case {cid}: {e}")


if __name__ == "__main__":
    main()

