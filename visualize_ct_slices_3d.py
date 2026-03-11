import os
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    import nibabel as nib
    _HAS_NIB = True
except ImportError:  # 允许没有 nibabel 时仅支持 .npy
    nib = None
    _HAS_NIB = False


SUPPORTED_EXTS = [".nii.gz", ".nii", ".npy"]


def _list_files_with_ext(root: Path, exts):
    files = []
    for p in root.iterdir():
        if not p.is_file():
            continue
        for ext in exts:
            if str(p.name).endswith(ext):
                files.append(p)
                break
    return sorted(files)


def _strip_ext(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    base, _ = os.path.splitext(name)
    return base


def _collect_cases(images_dir: str, masks_dir: str, case_id: str | None = None):
    img_root = Path(images_dir)
    msk_root = Path(masks_dir)
    if not img_root.is_dir():
        raise FileNotFoundError(f"images_dir not found or not a directory: {images_dir}")
    if not msk_root.is_dir():
        raise FileNotFoundError(f"masks_dir not found or not a directory: {masks_dir}")

    pairs = []

    if case_id is not None:
        candidates = []
        for ext in SUPPORTED_EXTS:
            p = img_root / f"{case_id}{ext}"
            if p.is_file():
                candidates.append(p)
        if not candidates:
            raise FileNotFoundError(
                f"No image file found for case_id={case_id} under {images_dir} with extensions {SUPPORTED_EXTS}"
            )
        img_path = candidates[0]
        base_id = _strip_ext(img_path.name)

        msk_candidates = []
        for ext in SUPPORTED_EXTS:
            q = msk_root / f"{base_id}{ext}"
            if q.is_file():
                msk_candidates.append(q)
        if not msk_candidates:
            raise FileNotFoundError(
                f"No mask file found for case_id={base_id} under {masks_dir} with extensions {SUPPORTED_EXTS}"
            )
        msk_path = msk_candidates[0]
        pairs.append((base_id, img_path, msk_path))
        return pairs

    img_files = _list_files_with_ext(img_root, SUPPORTED_EXTS)
    if not img_files:
        raise FileNotFoundError(f"No image files with extensions {SUPPORTED_EXTS} found in {images_dir}")

    for img_path in img_files:
        base_id = _strip_ext(img_path.name)
        msk_candidates = []
        for ext in SUPPORTED_EXTS:
            q = msk_root / f"{base_id}{ext}"
            if q.is_file():
                msk_candidates.append(q)
        if not msk_candidates:
            print(f"[WARN] mask for case {base_id} not found in {masks_dir}, skip this case")
            continue
        msk_path = msk_candidates[0]
        pairs.append((base_id, img_path, msk_path))

    if not pairs:
        raise RuntimeError("No image-mask pairs found. Check directory contents and naming.")

    return pairs


def _load_image_volume(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    name = str(path.name).lower()
    if name.endswith(".nii.gz") or suffix == ".nii":
        if not _HAS_NIB:
            raise ImportError("nibabel is required to read NIfTI files (.nii/.nii.gz)")
        img = nib.load(str(path))
        data = img.get_fdata(dtype=np.float32)
    elif suffix == ".npy":
        data = np.load(str(path))
    else:
        raise RuntimeError(f"Unsupported image file extension for {path}")

    if data.ndim == 3:
        vol = data.astype(np.float32)
    elif data.ndim == 4:
        if data.shape[-1] in (1, 3):
            vol = np.mean(data, axis=-1).astype(np.float32)
        else:
            vol = np.mean(data, axis=0).astype(np.float32)
    else:
        raise RuntimeError(f"Unexpected image volume shape {data.shape} for {path}")

    return vol


def _load_mask_volume(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    name = str(path.name).lower()
    if name.endswith(".nii.gz") or suffix == ".nii":
        if not _HAS_NIB:
            raise ImportError("nibabel is required to read NIfTI files (.nii/.nii.gz)")
        img = nib.load(str(path))
        data = img.get_fdata(dtype=np.float32)
    elif suffix == ".npy":
        data = np.load(str(path))
    else:
        raise RuntimeError(f"Unsupported mask file extension for {path}")

    if data.ndim == 3:
        vol = data
    elif data.ndim == 4:
        vol = data[..., 0]
    else:
        raise RuntimeError(f"Unexpected mask volume shape {data.shape} for {path}")

    vol = np.asarray(vol)
    if np.issubdtype(vol.dtype, np.floating):
        vol = (vol > 0.5).astype(np.int16)
    else:
        vol = vol.astype(np.int16)
    return vol


def _apply_window_or_norm(vol: np.ndarray, window_center: float | None, window_width: float | None, norm_mode: str) -> np.ndarray:
    vol = vol.astype(np.float32)
    if window_center is not None and window_width is not None:
        w_min = window_center - window_width / 2.0
        w_max = window_center + window_width / 2.0
        vol = np.clip(vol, w_min, w_max)
        if w_max > w_min:
            vol = (vol - w_min) / (w_max - w_min)
        else:
            vol = np.zeros_like(vol, dtype=np.float32)
        return vol

    if norm_mode == "none":
        return vol

    if norm_mode == "minmax":
        v_min = float(np.min(vol))
        v_max = float(np.max(vol))
        if v_max > v_min:
            vol = (vol - v_min) / (v_max - v_min)
        else:
            vol = np.zeros_like(vol, dtype=np.float32)
        return vol

    if norm_mode == "zscore":
        mean = float(np.mean(vol))
        std = float(np.std(vol))
        if std > 0:
            vol = (vol - mean) / std
            vol = np.clip(vol, -3.0, 3.0)
            vol = (vol + 3.0) / 6.0
        else:
            vol = np.zeros_like(vol, dtype=np.float32)
        return vol

    raise ValueError(f"Unknown norm_mode: {norm_mode}")


def _sample_indices(length: int, num: int, mode: str) -> list[int]:
    if length <= 0:
        raise ValueError("Volume length must be > 0")
    if num <= 0:
        raise ValueError("num slices must be >= 1")

    if num == 1:
        return [length // 2]

    if num >= length:
        return list(range(length))

    if mode == "middle":
        return list(sorted(set(np.linspace(0, length - 1, num, dtype=int).tolist())))
    elif mode == "uniform":
        return list(sorted(set(np.linspace(0, length - 1, num, dtype=int).tolist())))
    else:
        raise ValueError(f"Unknown sampling mode: {mode}")


def _compute_mask_center(mask_vol: np.ndarray) -> tuple[int, int, int]:
    """Compute the center (z, y, x) of labeled voxels in the mask.

    If the mask is empty, fall back to geometrical center.
    """
    if mask_vol.ndim != 3:
        raise ValueError(f"mask_vol must be 3D, got shape {mask_vol.shape}")

    coords = np.argwhere(mask_vol > 0)
    if coords.size == 0:
        # no labeled voxels, use geometric center
        D, H, W = mask_vol.shape
        return D // 2, H // 2, W // 2

    cz = int(np.round(coords[:, 0].mean()))
    cy = int(np.round(coords[:, 1].mean()))
    cx = int(np.round(coords[:, 2].mean()))

    D, H, W = mask_vol.shape
    cz = int(np.clip(cz, 0, D - 1))
    cy = int(np.clip(cy, 0, H - 1))
    cx = int(np.clip(cx, 0, W - 1))
    return cz, cy, cx


def _sample_indices_around_center(length: int, num: int, center: int) -> list[int]:
    """Sample indices around a given center index within [0, length-1]."""
    if length <= 0:
        raise ValueError("Volume length must be > 0")
    if num <= 0:
        raise ValueError("num slices must be >= 1")

    center = int(np.clip(center, 0, length - 1))
    if num == 1:
        return [center]

    # Try to make a symmetric window around center
    half = num // 2
    start = center - half
    end = start + num - 1

    # shift window if it goes out of bounds
    if start < 0:
        shift = -start
        start += shift
        end += shift
    if end >= length:
        shift = end - length + 1
        start -= shift
        end -= shift

    start = max(start, 0)
    end = min(end, length - 1)

    idx = list(range(start, end + 1))
    # If due to boundary clipping we got fewer than num indices, pad by repeating closest indices
    while len(idx) < num:
        if idx[0] > 0:
            idx.insert(0, idx[0] - 1)
        elif idx[-1] < length - 1:
            idx.append(idx[-1] + 1)
        else:
            break
    return sorted(set(idx))[:num]


def _overlay_slice(img_slice: np.ndarray, mask_slice: np.ndarray, alpha: float = 0.4, cmap_mask: str = "jet"):
    img = img_slice.astype(np.float32)
    v_min = float(np.min(img))
    v_max = float(np.max(img))
    if v_max > v_min:
        img = (img - v_min) / (v_max - v_min)
    else:
        img = np.zeros_like(img, dtype=np.float32)

    plt.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)

    mask = mask_slice.astype(np.float32)
    if np.any(mask > 0):
        mask_vis = mask.copy()
        mask_vis[mask_vis == 0] = np.nan
        plt.imshow(mask_vis, cmap=cmap_mask, alpha=alpha)


def _show_image_slice(img_slice: np.ndarray):
    """Show image slice only, normalized to [0, 1]."""
    img = img_slice.astype(np.float32)
    v_min = float(np.min(img))
    v_max = float(np.max(img))
    if v_max > v_min:
        img = (img - v_min) / (v_max - v_min)
    else:
        img = np.zeros_like(img, dtype=np.float32)
    plt.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)


def _show_mask_slice(mask_slice: np.ndarray, cmap_mask: str = "jet"):
    """Show mask slice only (non-zero as colored, background transparent)."""
    mask = mask_slice.astype(np.float32)
    if np.any(mask > 0):
        mask_vis = mask.copy()
        mask_vis[mask_vis == 0] = np.nan
    else:
        mask_vis = np.full_like(mask, np.nan, dtype=np.float32)
    plt.imshow(mask_vis, cmap=cmap_mask)


def _plot_and_save_slices(case_id: str, img_vol: np.ndarray, mask_vol: np.ndarray, out_dir: Path,
                          slices_per_direction: int, sampling_mode: str, directions: list[str],
                          mask_alpha: float, dpi: int, figsize: tuple[float, float],
                          export_mode: str = "overlay"):
    if img_vol.shape != mask_vol.shape:
        raise RuntimeError(f"Image and mask shapes differ for case {case_id}: {img_vol.shape} vs {mask_vol.shape}")

    D, H, W = img_vol.shape
    out_dir.mkdir(parents=True, exist_ok=True)

    # center of labeled voxels in mask, used when sampling_mode == 'center_mask'
    cz, cy, cx = _compute_mask_center(mask_vol)

    for direction in directions:
        if direction == "axial":
            if sampling_mode == "center_mask":
                idx_list = _sample_indices_around_center(D, slices_per_direction, cz)
            else:
                idx_list = _sample_indices(D, slices_per_direction, sampling_mode)
            for idx in idx_list:
                if export_mode == "overlay":
                    fig = plt.figure(figsize=figsize)
                    _overlay_slice(img_vol[idx, :, :], mask_vol[idx, :, :], alpha=mask_alpha)
                    plt.axis("off")
                    fname = f"{case_id}_axial_{idx:03d}.png"
                    fig.savefig(out_dir / fname, dpi=dpi, bbox_inches="tight", pad_inches=0)
                    plt.close(fig)
                elif export_mode == "separate":
                    # image-only
                    fig = plt.figure(figsize=figsize)
                    _show_image_slice(img_vol[idx, :, :])
                    plt.axis("off")
                    fname_img = f"{case_id}_axial_{idx:03d}_img.png"
                    fig.savefig(out_dir / fname_img, dpi=dpi, bbox_inches="tight", pad_inches=0)
                    plt.close(fig)
                    # mask-only
                    fig = plt.figure(figsize=figsize)
                    _show_mask_slice(mask_vol[idx, :, :])
                    plt.axis("off")
                    fname_msk = f"{case_id}_axial_{idx:03d}_mask.png"
                    fig.savefig(out_dir / fname_msk, dpi=dpi, bbox_inches="tight", pad_inches=0)
                    plt.close(fig)
                else:
                    raise ValueError(f"Unknown export_mode: {export_mode}")
        elif direction == "coronal":
            if sampling_mode == "center_mask":
                idx_list = _sample_indices_around_center(H, slices_per_direction, cy)
            else:
                idx_list = _sample_indices(H, slices_per_direction, sampling_mode)
            for idx in idx_list:
                if export_mode == "overlay":
                    fig = plt.figure(figsize=figsize)
                    _overlay_slice(img_vol[:, idx, :], mask_vol[:, idx, :], alpha=mask_alpha)
                    plt.axis("off")
                    fname = f"{case_id}_coronal_{idx:03d}.png"
                    fig.savefig(out_dir / fname, dpi=dpi, bbox_inches="tight", pad_inches=0)
                    plt.close(fig)
                elif export_mode == "separate":
                    fig = plt.figure(figsize=figsize)
                    _show_image_slice(img_vol[:, idx, :])
                    plt.axis("off")
                    fname_img = f"{case_id}_coronal_{idx:03d}_img.png"
                    fig.savefig(out_dir / fname_img, dpi=dpi, bbox_inches="tight", pad_inches=0)
                    plt.close(fig)

                    fig = plt.figure(figsize=figsize)
                    _show_mask_slice(mask_vol[:, idx, :])
                    plt.axis("off")
                    fname_msk = f"{case_id}_coronal_{idx:03d}_mask.png"
                    fig.savefig(out_dir / fname_msk, dpi=dpi, bbox_inches="tight", pad_inches=0)
                    plt.close(fig)
                else:
                    raise ValueError(f"Unknown export_mode: {export_mode}")
        elif direction == "sagittal":
            if sampling_mode == "center_mask":
                idx_list = _sample_indices_around_center(W, slices_per_direction, cx)
            else:
                idx_list = _sample_indices(W, slices_per_direction, sampling_mode)
            for idx in idx_list:
                if export_mode == "overlay":
                    fig = plt.figure(figsize=figsize)
                    _overlay_slice(img_vol[:, :, idx], mask_vol[:, :, idx], alpha=mask_alpha)
                    plt.axis("off")
                    fname = f"{case_id}_sagittal_{idx:03d}.png"
                    fig.savefig(out_dir / fname, dpi=dpi, bbox_inches="tight", pad_inches=0)
                    plt.close(fig)
                elif export_mode == "separate":
                    fig = plt.figure(figsize=figsize)
                    _show_image_slice(img_vol[:, :, idx])
                    plt.axis("off")
                    fname_img = f"{case_id}_sagittal_{idx:03d}_img.png"
                    fig.savefig(out_dir / fname_img, dpi=dpi, bbox_inches="tight", pad_inches=0)
                    plt.close(fig)

                    fig = plt.figure(figsize=figsize)
                    _show_mask_slice(mask_vol[:, :, idx])
                    plt.axis("off")
                    fname_msk = f"{case_id}_sagittal_{idx:03d}_mask.png"
                    fig.savefig(out_dir / fname_msk, dpi=dpi, bbox_inches="tight", pad_inches=0)
                    plt.close(fig)
                else:
                    raise ValueError(f"Unknown export_mode: {export_mode}")
        else:
            raise ValueError(f"Unknown direction: {direction}")


def visualize_ct_slices(
    images_dir: str,
    masks_dir: str,
    output_dir: str,
    case_id: str | None = None,
    slices_per_direction: int = 3,
    sampling_mode: str = "uniform",
    directions: str = "all",
    window_center: float | None = None,
    window_width: float | None = None,
    norm_mode: str = "minmax",
    mask_alpha: float = 0.4,
    dpi: int = 150,
    figsize: tuple[float, float] = (4, 4),
    export_mode: str = "overlay",
):
    """Visualize CT and mask slices.

    export_mode:
        - "overlay": (default) image + mask overlay in a single PNG per slice.
        - "separate": export image-only and mask-only PNGs per slice, with filename suffixes
          "_img" and "_mask" respectively.
    Note: mask_alpha only affects "overlay" mode and is ignored in "separate" mode.
    """
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if directions == "all":
        dir_list = ["axial", "coronal", "sagittal"]
    else:
        dir_list = [directions]

    pairs = _collect_cases(images_dir, masks_dir, case_id=case_id)

    print(f"Found {len(pairs)} image-mask pairs to process.")

    for cid, img_path, msk_path in pairs:
        print(f"Processing case {cid}:\n  image: {img_path}\n  mask:  {msk_path}")
        img_vol = _load_image_volume(img_path)
        msk_vol = _load_mask_volume(msk_path)

        if img_vol.shape != msk_vol.shape:
            min_shape = tuple(min(i, m) for i, m in zip(img_vol.shape, msk_vol.shape))
            img_vol = img_vol[: min_shape[0], : min_shape[1], : min_shape[2]]
            msk_vol = msk_vol[: min_shape[0], : min_shape[1], : min_shape[2]]
            print(f"[WARN] Shape mismatch, cropped to {min_shape} for case {cid}")

        img_vol_norm = _apply_window_or_norm(img_vol, window_center, window_width, norm_mode)

        _plot_and_save_slices(
            cid,
            img_vol_norm,
            msk_vol,
            out_root,
            slices_per_direction=slices_per_direction,
            sampling_mode=sampling_mode,
            directions=dir_list,
            mask_alpha=mask_alpha,
            dpi=dpi,
            figsize=figsize,
            export_mode=export_mode,
        )


def _build_argparser():
    parser = argparse.ArgumentParser(
        description="Visualize CT volumes and segmentation masks as multi-direction slices (axial/coronal/sagittal)."
    )
    parser.add_argument("--images_dir", type=str, required=True, help="Directory of 3D CT volumes, e.g. E:/ESO_SEG_TJ/train/images")
    parser.add_argument("--masks_dir", type=str, required=True, help="Directory of corresponding segmentation masks, e.g. E:/ESO_SEG_TJ/train/masks")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output slice images")
    parser.add_argument("--case_id", type=str, default=None, help="If set, only visualize this specific case id (without extension)")
    parser.add_argument("--slices_per_direction", type=int, default=3, help="Number of slices to export per direction")
    parser.add_argument("--sampling_mode", type=str, default="uniform", choices=["uniform", "middle", "center_mask"], help="How to sample slice indices along each axis (including mask-center-based)")
    parser.add_argument("--directions", type=str, default="all", choices=["all", "axial", "coronal", "sagittal"], help="Which directions to export")
    parser.add_argument("--window_center", type=float, default=None, help="CT window center (HU). If set with window_width, overrides norm_mode")
    parser.add_argument("--window_width", type=float, default=None, help="CT window width (HU). If set with window_center, overrides norm_mode")
    parser.add_argument("--norm_mode", type=str, default="minmax", choices=["none", "minmax", "zscore"], help="Normalization mode when window is not provided")
    parser.add_argument("--mask_alpha", type=float, default=0.4, help="Alpha (transparency) for mask overlay (overlay mode only)")
    parser.add_argument("--dpi", type=int, default=150, help="DPI of saved PNGs")
    parser.add_argument("--figsize", type=float, nargs=2, default=[4.0, 4.0], metavar=("W", "H"), help="Figure size in inches (width height)")
    parser.add_argument("--export_mode", type=str, default="overlay", choices=["overlay", "separate"], help="How to export slices: 'overlay' (image+mask) or 'separate' (image-only and mask-only PNGs)")
    return parser


if __name__ == "__main__":
    parser = _build_argparser()
    args = parser.parse_args()

    if (args.window_center is None) != (args.window_width is None):
        parser.error("Both --window_center and --window_width must be provided together, or both omitted.")

    visualize_ct_slices(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        output_dir=args.output_dir,
        case_id=args.case_id,
        slices_per_direction=args.slices_per_direction,
        sampling_mode=args.sampling_mode,
        directions=args.directions,
        window_center=args.window_center,
        window_width=args.window_width,
        norm_mode=args.norm_mode,
        mask_alpha=args.mask_alpha,
        dpi=args.dpi,
        figsize=(args.figsize[0], args.figsize[1]),
        export_mode=args.export_mode,
    )

    # Example (PowerShell):
    # python visualize_ct_slices_3d.py \
    #   --images_dir "E:\\ESO_SEG_TJ\\train\\images" \
    #   --masks_dir  "E:\\ESO_SEG_TJ\\train\\masks" \
    #   --output_dir "E:\\ESO_SEG_TJ\\vis_slices" \
    #   --slices_per_direction 3 \
    #   --sampling_mode center_mask \
    #   --window_center 40 --window_width 400 \
    #   --export_mode separate
