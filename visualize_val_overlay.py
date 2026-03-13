import os
import argparse
from typing import List, Optional, Tuple

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from nnunet_mednext.utilities.overlay_plots import generate_overlay
from visualize_fd_edge_and_ds import _extract_case_data


def _load_pred_nifti(pred_path: str) -> np.ndarray:
    """Load prediction nifti and return array in [Z, Y, X] order.

    nnUNet usually stores NIfTI as [X, Y, Z]. We transpose to [Z, Y, X]
    to match the convention used in `_extract_case_data` for GT.
    """
    if not os.path.isfile(pred_path):
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    img = nib.load(pred_path)
    arr = img.get_fdata()
    if arr.ndim != 3:
        raise RuntimeError(f"Expected 3D prediction, got shape {arr.shape} from {pred_path}")
    # [X, Y, Z] -> [Z, Y, X]
    arr = np.transpose(arr, (2, 1, 0))
    return arr.astype(np.int16)


def _match_shapes(image: np.ndarray, seg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Center-crop seg to match image spatial size if needed.

    image: [C, D, H, W]
    seg: [D, H, W]
    """
    d_i, h_i, w_i = image.shape[1:]
    d_s, h_s, w_s = seg.shape
    d = min(d_i, d_s)
    h = min(h_i, h_s)
    w = min(w_i, w_s)
    image_c = image[:, :d, :h, :w]
    seg_c = seg[:d, :h, :w]
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
) -> None:
    """Visualize overlay of original image and prediction for a single case.

    data_root: directory with preprocessed stage0 .npy volumes
    dataset_directory: nnUNet dataset directory (contains gt_segmentations)
    pred_folder: folder containing prediction nifti files (validation_raw or validation_raw_postprocessed)
    case_id: case identifier (e.g. ESO_TJ_60011222468)
    output_dir: where to save pngs
    mode: "slice" (single or multiple 2D slices) or "mip" (3D MIP views)
    axis: for slice mode, one of {"z", "y", "x"}
    slices: optional list of slice indices; if None, use middle slice
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load preprocessed image and GT (for potential future use / sanity check)
    image, gt = _extract_case_data(data_root, case_id, dataset_directory)
    # image: [C, D, H, W], gt: [1, D, H, W]
    gt = gt[0]

    # Find prediction nifti
    pred_path = None
    for ext in (".nii.gz", ".nii"):
        cand = os.path.join(pred_folder, f"{case_id}{ext}")
        if os.path.isfile(cand):
            pred_path = cand
            break
    if pred_path is None:
        raise FileNotFoundError(f"Could not find prediction for case {case_id} in {pred_folder}")

    pred = _load_pred_nifti(pred_path)  # [D, H, W] (Z, Y, X)

    # Align shapes
    image, pred = _match_shapes(image, pred)
    # Also align GT in case sizes differ
    _, gt = _match_shapes(image, gt)

    # Select slices
    C, D, H, W = image.shape

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
            if axis == "z":
                if not (0 <= s < D):
                    continue
                img_slice = image[0, s]
                pred_slice = pred[s]
            elif axis == "y":
                if not (0 <= s < H):
                    continue
                img_slice = image[0, :, s, :]
                pred_slice = pred[:, s, :]
            else:  # x
                if not (0 <= s < W):
                    continue
                img_slice = image[0, :, :, s]
                pred_slice = pred[:, :, s]

            overlay = generate_overlay(img_slice, pred_slice, overlay_intensity=alpha)
            out_name = f"{case_id}_axis-{axis}_slice-{s}.png"
            out_path = os.path.join(output_dir, out_name)
            plt.imsave(out_path, overlay)
            print(f"Saved overlay to: {out_path}")

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
        out_path = os.path.join(output_dir, f"{case_id}_mip.png")
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved MIP overlays to: {out_path}")

    else:
        raise ValueError("mode must be 'slice' or 'mip'")


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize overlay of validation predictions and original images")
    parser.add_argument("--data_root", type=str, required=True, help="Folder with preprocessed stage0 .npy volumes")
    parser.add_argument("--dataset_directory", type=str, required=True, help="nnUNet dataset directory (contains gt_segmentations)")
    parser.add_argument("--pred_folder", type=str, required=True, help="Folder with prediction NIfTI files (e.g. validation_raw_postprocessed)")
    parser.add_argument("--case_id", type=str, required=True, help="Case id to visualize (e.g. ESO_TJ_60011222468)")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save output PNGs")
    parser.add_argument("--alpha", type=float, default=0.6, help="Overlay intensity (alpha) for segmentation")
    parser.add_argument("--mode", type=str, choices=["slice", "mip"], default="slice", help="Visualization mode: slice or mip")
    parser.add_argument("--axis", type=str, choices=["z", "y", "x"], default="z", help="Slice axis for slice mode")
    parser.add_argument("--slices", type=int, nargs="*", default=None, help="Slice indices to visualize (if empty, use middle slice)")
    return parser


if __name__ == "__main__":
    parser = _build_argparser()
    args = parser.parse_args()

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

