import argparse
import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import nibabel as nib
except Exception:  # pragma: no cover - optional dependency
    nib = None


def _normalize01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    v_min = float(arr.min())
    v_max = float(arr.max())
    if v_max > v_min:
        arr = (arr - v_min) / (v_max - v_min)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    return arr


def _load_volume(path: str) -> np.ndarray:
    """Load a 3D volume and return it as [D, H, W] float32."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image file not found: {path}")

    lower = path.lower()
    if lower.endswith((".nii", ".nii.gz")):
        if nib is None:
            raise RuntimeError("nibabel is required to read NIfTI files, but it is not available.")
        nii = nib.load(path)
        vol = np.asarray(nii.dataobj, dtype=np.float32)  # type: ignore[attr-defined]
    elif lower.endswith(".npy"):
        vol = np.asarray(np.load(path), dtype=np.float32)
    elif lower.endswith(".npz"):
        npz = np.load(path)
        if "data" in npz:
            vol = np.asarray(npz["data"], dtype=np.float32)
        else:
            first_key = npz.files[0]
            vol = np.asarray(npz[first_key], dtype=np.float32)
    else:
        raise ValueError(
            f"Unsupported 3D input format: {path}. Please use .nii/.nii.gz/.npy/.npz for 3D data."
        )

    if vol.ndim == 4:
        # [D, H, W, C] or [C, D, H, W] -> take first channel by default
        if vol.shape[-1] <= 4:
            vol = vol[..., 0]
        else:
            vol = vol[0]

    if vol.ndim != 3:
        raise RuntimeError(f"Expected a 3D volume, got shape {vol.shape} from {path}")

    return _normalize01(vol)


def _extract_2d_slice(vol: np.ndarray, slice_axis: str = "z", slice_index: Optional[int] = None) -> np.ndarray:
    axis_map = {"z": 0, "y": 1, "x": 2}
    if slice_axis not in axis_map:
        raise ValueError("slice_axis must be one of 'x', 'y', 'z'")
    axis = axis_map[slice_axis]
    if slice_index is None:
        slice_index = vol.shape[axis] // 2
    if not (0 <= slice_index < vol.shape[axis]):
        raise ValueError(f"slice_index {slice_index} out of range for shape {vol.shape}")
    if axis == 0:
        return vol[slice_index]
    if axis == 1:
        return vol[:, slice_index, :]
    return vol[:, :, slice_index]


class FixedFourKernelEncoder3D(nn.Module):
    """A tiny 2-layer 3D encoder with 4 fixed kernels per layer.

    The first layer maps 1 channel -> 4 channels with four basic 3D filters.
    The second layer keeps 4 channels -> 4 channels so each output channel still
    corresponds to one response branch that can be visualized directly.
    """

    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv3d(1, 4, kernel_size=3, padding=1, bias=False)
        self.enc2 = nn.Conv3d(4, 4, kernel_size=3, padding=1, bias=False)
        self._init_fixed_kernels()
        for p in self.parameters():
            p.requires_grad_(False)

    def _init_fixed_kernels(self):
        # 4 basic 3D kernels:
        # 0) local mean
        # 1) z-axis edge
        # 2) y-axis edge
        # 3) x-axis edge / center-surround
        kernels = torch.zeros((4, 3, 3, 3), dtype=torch.float32)
        kernels[0] = 1.0 / 27.0
        kernels[1, 0, :, :] = -1.0 / 9.0
        kernels[1, 2, :, :] = 1.0 / 9.0
        kernels[2, :, 0, :] = -1.0 / 9.0
        kernels[2, :, 2, :] = 1.0 / 9.0
        kernels[3, :, :, 0] = -1.0 / 9.0
        kernels[3, :, :, 2] = 1.0 / 9.0

        with torch.no_grad():
            self.enc1.weight.zero_()
            self.enc2.weight.zero_()

            # Layer 1: same four kernels on the single input channel.
            self.enc1.weight[:, 0, :, :, :] = kernels

            # Layer 2: keep 4 response branches, average across the 4 input channels.
            for out_ch in range(4):
                for in_ch in range(4):
                    self.enc2.weight[out_ch, in_ch, :, :, :] = kernels[out_ch] / 4.0

    def forward(self, x: torch.Tensor):
        x1 = F.relu(self.enc1(x))
        x2 = F.relu(self.enc2(x1))
        return x1, x2


def _prepare_tensor(vol3d: np.ndarray, device: str) -> torch.Tensor:
    tensor = torch.from_numpy(vol3d[None, None].astype(np.float32))
    return tensor.to(device)


def _render_response_grid(
    responses: Tuple[torch.Tensor, torch.Tensor],
    out_path: str,
    title: str = "Encoder kernel responses",
    slice_axis: str = "z",
):
    layer1, layer2 = responses
    layer1 = layer1.detach().cpu().numpy()[0]  # [4, D, H, W]
    layer2 = layer2.detach().cpu().numpy()[0]

    layer1_slices = [_extract_2d_slice(layer1[c], slice_axis=slice_axis) for c in range(4)]
    layer2_slices = [_extract_2d_slice(layer2[c], slice_axis=slice_axis) for c in range(4)]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for row, layer in enumerate((layer1_slices, layer2_slices)):
        for col in range(4):
            ax = axes[row, col]
            resp = _normalize01(layer[col])
            ax.imshow(resp, cmap="gray")
            ax.set_title(f"Layer {row + 1} - Kernel {col + 1}")
            ax.axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def visualize_encoder_first_two_layers(
    image_path: str,
    out_dir: str,
    slice_axis: str = "z",
    slice_index: Optional[int] = None,
    device: Optional[str] = None,
):
    os.makedirs(out_dir, exist_ok=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    vol3d = _load_volume(image_path)
    img2d = _extract_2d_slice(vol3d, slice_axis=slice_axis, slice_index=slice_index)
    x = _prepare_tensor(vol3d, device)

    model = FixedFourKernelEncoder3D().to(device).eval()
    with torch.no_grad():
        layer1, layer2 = model(x)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    if base_name.endswith(".nii"):
        base_name = os.path.splitext(base_name)[0]
    out_path = os.path.join(out_dir, f"{base_name}_encoder_kernel_responses.png")
    _render_response_grid((layer1, layer2), out_path, slice_axis=slice_axis)

    # Save the chosen input slice for reference.
    input_path = os.path.join(out_dir, f"{base_name}_input_slice_{slice_axis}.png")
    plt.imsave(input_path, img2d, cmap="gray")

    # Also save individual maps for convenience.
    for layer_idx, layer in enumerate((layer1, layer2), start=1):
        arr = layer.detach().cpu().numpy()[0]
        for kernel_idx in range(4):
            kernel_path = os.path.join(out_dir, f"{base_name}_layer{layer_idx}_kernel{kernel_idx + 1}.png")
            plt.imsave(kernel_path, _normalize01(_extract_2d_slice(arr[kernel_idx], slice_axis=slice_axis)), cmap="gray")

    print(f"Saved response grid to: {out_path}")
    return out_path


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize the response maps of four base kernels in the first two encoder layers."
    )
    parser.add_argument("--image", type=str, required=True, help="Path to a 2D image or a NIfTI volume.")
    parser.add_argument("--out_dir", type=str, default="./encoder_kernel_vis", help="Directory for outputs.")
    parser.add_argument(
        "--slice_axis",
        type=str,
        default="z",
        choices=["x", "y", "z"],
        help="For 3D volumes, which axis to use for slice visualization.",
    )
    parser.add_argument(
        "--slice_index",
        type=int,
        default=None,
        help="For 3D volumes, the slice index along the chosen axis. Defaults to middle slice.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", None],
        help="Device to run on. Defaults to CUDA if available.",
    )
    return parser


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    visualize_encoder_first_two_layers(
        image_path=args.image,
        out_dir=args.out_dir,
        slice_axis=args.slice_axis,
        slice_index=args.slice_index,
        device=None if args.device in (None, "None") else args.device,
    )

