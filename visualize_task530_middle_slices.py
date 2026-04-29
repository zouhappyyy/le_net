import argparse
import os
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion


DEFAULT_DATA_ROOT = (
    "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_preprocessed/"
    "Task530_EsoTJ_30pct/nnUNetData_plans_v2.1_trgSp_1x1x1_stage0"
)
DEFAULT_DATASET_DIRECTORY = (
    "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_preprocessed/Task530_EsoTJ_30pct"
)


def _find_case_file(data_root: str, case_id: str) -> str:
    for ext in (".npy", ".npz"):
        path = os.path.join(data_root, f"{case_id}{ext}")
        if os.path.isfile(path):
            return path
    matches = [
        os.path.join(data_root, name)
        for name in sorted(os.listdir(data_root))
        if name.startswith(case_id) and (name.endswith(".npy") or name.endswith(".npz"))
    ]
    if not matches:
        raise FileNotFoundError(
            f"Could not find preprocessed file for case_id={case_id} under {data_root}"
        )
    return matches[0]


def _load_preprocessed_image(data_root: str, case_id: str) -> np.ndarray:
    path = _find_case_file(data_root, case_id)
    if path.endswith(".npy"):
        arr = np.load(path)
    else:
        npz = np.load(path)
        keys = list(npz.keys())
        if "data" in keys:
            arr = npz["data"]
        elif len(keys) == 1:
            arr = npz[keys[0]]
        else:
            raise RuntimeError(
                f"Unexpected npz keys in {path}: {keys}. "
                "Please specify a file that stores image data clearly."
            )

    if arr.ndim == 4:
        image = arr[0]
    elif arr.ndim == 3:
        image = arr
    else:
        raise RuntimeError(
            f"Unsupported preprocessed image shape {arr.shape} in {path}, "
            "expected (C, D, H, W) or (D, H, W)."
        )

    return image.astype(np.float32)


def _load_mask(dataset_directory: str, case_id: str) -> np.ndarray:
    gt_dir = os.path.join(dataset_directory, "gt_segmentations")
    for ext in (".nii.gz", ".nii"):
        path = os.path.join(gt_dir, f"{case_id}{ext}")
        if os.path.isfile(path):
            mask = nib.load(path).get_fdata()
            if mask.ndim != 3:
                raise RuntimeError(
                    f"Unsupported mask shape {mask.shape} in {path}, expected 3D NIfTI."
                )
            return np.transpose(mask, (2, 1, 0)).astype(np.uint8)
    raise FileNotFoundError(
        f"Could not find mask for case_id={case_id} in {gt_dir}"
    )


def _crop_to_match(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    d = min(image.shape[0], mask.shape[0])
    h = min(image.shape[1], mask.shape[1])
    w = min(image.shape[2], mask.shape[2])
    return image[:d, :h, :w], mask[:d, :h, :w]


def _mask_boundary(mask_slice: np.ndarray) -> np.ndarray:
    fg = mask_slice > 0
    if not np.any(fg):
        return np.zeros_like(mask_slice, dtype=bool)
    eroded = binary_erosion(fg)
    return np.logical_and(fg, np.logical_not(eroded))


def _extract_slice(volume: np.ndarray, axis: str, index: int) -> np.ndarray:
    if axis == "z":
        return volume[index, :, :]
    if axis == "y":
        return volume[:, index, :]
    if axis == "x":
        return volume[:, :, index]
    raise ValueError("axis must be one of: z, y, x")


def _normalize_slice(image_slice: np.ndarray) -> np.ndarray:
    lo = float(np.percentile(image_slice, 1))
    hi = float(np.percentile(image_slice, 99))
    if hi <= lo:
        lo = float(np.min(image_slice))
        hi = float(np.max(image_slice))
    if hi <= lo:
        return np.zeros_like(image_slice, dtype=np.float32)
    out = np.clip((image_slice - lo) / (hi - lo), 0.0, 1.0)
    return out.astype(np.float32)


def _resolve_case_id(data_root: str, case_id: Optional[str]) -> str:
    if case_id:
        return case_id
    candidates = sorted(
        {
            os.path.splitext(name)[0]
            for name in os.listdir(data_root)
            if name.endswith(".npy") or name.endswith(".npz")
        }
    )
    if not candidates:
        raise RuntimeError(f"No .npy or .npz files found in {data_root}")
    return candidates[0]


def visualize_case(
    data_root: str,
    dataset_directory: str,
    case_id: str,
    output_dir: str,
    dpi: int = 200,
) -> None:
    image = _load_preprocessed_image(data_root, case_id)
    mask = _load_mask(dataset_directory, case_id)
    image, mask = _crop_to_match(image, mask)

    d, h, w = image.shape
    mid_indices = {"z": d // 2, "y": h // 2, "x": w // 2}

    os.makedirs(output_dir, exist_ok=True)

    for axis_name in ("z", "y", "x"):
        idx = mid_indices[axis_name]
        img_slice = _extract_slice(image, axis_name, idx)
        mask_slice = _extract_slice(mask, axis_name, idx)
        boundary = _mask_boundary(mask_slice)
        img_show = _normalize_slice(img_slice)
        mask_show = (mask_slice > 0).astype(np.uint8) * 255
        boundary_show = boundary.astype(np.uint8) * 255

        base_name = f"{case_id}_axis-{axis_name}_mid-{idx}"
        image_path = os.path.join(output_dir, f"{base_name}_image.png")
        mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
        boundary_path = os.path.join(output_dir, f"{base_name}_boundary.png")

        plt.imsave(image_path, img_show, cmap="gray")
        plt.imsave(mask_path, mask_show, cmap="gray", vmin=0, vmax=255)
        plt.imsave(boundary_path, boundary_show, cmap="gray", vmin=0, vmax=255)

        print(f"Saved {axis_name}-axis image to: {image_path}")
        print(f"Saved {axis_name}-axis mask to: {mask_path}")
        print(f"Saved {axis_name}-axis boundary to: {boundary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize original image, mask overlay, and mask boundary overlay "
            "using the middle slice along z/y/x."
        )
    )
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--dataset_directory",
        type=str,
        default=DEFAULT_DATASET_DIRECTORY,
        help="Task directory containing gt_segmentations.",
    )
    parser.add_argument(
        "--case_id",
        type=str,
        default=None,
        help="Case id to visualize. If omitted, the first case in data_root is used.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./task530_middle_slices",
        help="Directory to save exported PNG files.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    case_id = _resolve_case_id(args.data_root, args.case_id)
    visualize_case(
        data_root=args.data_root,
        dataset_directory=args.dataset_directory,
        case_id=case_id,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
