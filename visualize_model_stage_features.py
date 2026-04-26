import argparse
import os
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import nibabel as nib
except Exception:  # pragma: no cover - optional dependency
    nib = None

from nnunet_mednext.training.model_restore import restore_model


def _normalize01(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, copy=False)
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax > vmin:
        return (arr - vmin) / (vmax - vmin)
    return np.zeros_like(arr, dtype=np.float32)


def _load_volume(path: str, input_channel: int = 0) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input volume not found: {path}")

    lower = path.lower()
    if lower.endswith((".nii", ".nii.gz")):
        if nib is None:
            raise RuntimeError("nibabel is required to read NIfTI files, but it is not installed.")
        vol = np.asarray(nib.load(path).dataobj, dtype=np.float32)  # type: ignore[attr-defined]
    elif lower.endswith(".npy"):
        vol = np.asarray(np.load(path), dtype=np.float32)
    elif lower.endswith(".npz"):
        npz = np.load(path)
        if "data" in npz:
            vol = np.asarray(npz["data"], dtype=np.float32)
        else:
            vol = np.asarray(npz[npz.files[0]], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported input format: {path}")

    if vol.ndim == 3:
        return vol[None]
    if vol.ndim == 4:
        # Prefer [C, D, H, W]. If this looks like [D, H, W, C], move C first.
        if vol.shape[0] <= 4:
            return vol
        if vol.shape[-1] <= 4:
            return np.moveaxis(vol, -1, 0)
    if vol.ndim == 5 and vol.shape[0] == 1:
        return vol[0]
    raise RuntimeError(f"Expected [D,H,W] or [C,D,H,W] input, got shape {vol.shape}")


def _to_input_tensor(chw_vol: np.ndarray, device: str) -> torch.Tensor:
    chw_vol = np.asarray(chw_vol, dtype=np.float32)
    return torch.from_numpy(chw_vol[None]).to(device)


def _axis_to_dim(axis: str) -> int:
    return {"z": 0, "y": 1, "x": 2}[axis]


def _extract_slice(vol3d: np.ndarray, axis: str, index: int) -> np.ndarray:
    dim = _axis_to_dim(axis)
    if dim == 0:
        return vol3d[index, :, :]
    if dim == 1:
        return vol3d[:, index, :]
    return vol3d[:, :, index]


def _scaled_slice_index(input_index: int, input_shape: Tuple[int, int, int], feat_shape: Tuple[int, int, int], axis: str) -> int:
    dim = _axis_to_dim(axis)
    src = max(1, input_shape[dim])
    dst = max(1, feat_shape[dim])
    idx = int(round(input_index * dst / src))
    return max(0, min(dst - 1, idx))


def _basename_without_double_ext(path: str) -> str:
    name = os.path.basename(path)
    if name.endswith(".nii.gz"):
        return name[:-7]
    return os.path.splitext(name)[0]


def _resolve_checkpoint_paths(path: str, checkpoint_name: str) -> Tuple[str, str]:
    if os.path.isfile(path):
        if path.endswith(".model.pkl"):
            model_path = path[:-4]
            pkl_path = path
        elif path.endswith(".model"):
            model_path = path
            pkl_path = path + ".pkl"
        else:
            raise ValueError("Checkpoint file must end with .model or .model.pkl")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Missing model file: {model_path}")
        if not os.path.isfile(pkl_path):
            raise FileNotFoundError(f"Missing checkpoint pkl file: {pkl_path}")
        return model_path, pkl_path

    if not os.path.isdir(path):
        raise FileNotFoundError(f"Checkpoint path not found: {path}")

    search_names = [checkpoint_name]
    if not checkpoint_name.endswith(".model"):
        search_names.append(checkpoint_name + ".model")

    for name in search_names:
        model_path = os.path.join(path, name)
        pkl_path = model_path + ".pkl"
        if os.path.isfile(model_path) and os.path.isfile(pkl_path):
            return model_path, pkl_path

    raise FileNotFoundError(
        f"Could not find checkpoint '{checkpoint_name}' under {path}. "
        f"Expected files like model_best.model and model_best.model.pkl."
    )


def _get_inner_network(network: torch.nn.Module) -> torch.nn.Module:
    return network.net if hasattr(network, "net") else network


def _capture_encoder_outputs(stage_store: "OrderedDict[str, torch.Tensor]"):
    def _hook(_module, _inputs, output):
        if not isinstance(output, (list, tuple)) or len(output) != 5:
            return
        names = ["enc0", "enc1", "enc2", "enc3", "bottleneck"]
        for name, tensor in zip(names, output):
            if torch.is_tensor(tensor):
                stage_store[name] = tensor.detach().cpu()

    return _hook


def _capture_tensor_output(stage_store: "OrderedDict[str, torch.Tensor]", stage_name: str):
    def _hook(_module, _inputs, output):
        if torch.is_tensor(output):
            stage_store[stage_name] = output.detach().cpu()

    return _hook


def _register_stage_hooks(network: torch.nn.Module, stage_store: "OrderedDict[str, torch.Tensor]"):
    inner = _get_inner_network(network)
    handles = []

    if hasattr(inner, "encoder"):
        handles.append(inner.encoder.register_forward_hook(_capture_encoder_outputs(stage_store)))

    for module_name, stage_name in [
        ("dec_block_3", "dec3"),
        ("dec_block_2", "dec2"),
        ("dec_block_1", "dec1"),
        ("dec_block_0", "dec0"),
        ("out_0", "logits"),
        ("edge_head_f0", "edge_f0"),
        ("edge_head_f1", "edge_f1"),
    ]:
        if hasattr(inner, module_name):
            handles.append(getattr(inner, module_name).register_forward_hook(_capture_tensor_output(stage_store, stage_name)))

    return handles


def _select_topk_channels(feat: np.ndarray, topk: int) -> List[int]:
    if feat.shape[0] <= topk:
        return list(range(feat.shape[0]))
    scores = np.mean(np.abs(feat), axis=(1, 2, 3))
    return list(np.argsort(scores)[::-1][:topk])


def _save_stage_visualization(
    stage_name: str,
    tensor: torch.Tensor,
    input_slice: np.ndarray,
    input_shape: Tuple[int, int, int],
    input_index: int,
    axis: str,
    out_dir: str,
    topk: int,
    save_raw: bool,
):
    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 5:
        arr = arr[0]
    elif arr.ndim == 4:
        pass
    else:
        raise RuntimeError(f"Stage {stage_name} expected [B,C,D,H,W] or [C,D,H,W], got {arr.shape}")

    if save_raw:
        np.save(os.path.join(out_dir, f"{stage_name}.npy"), arr)

    feat_shape = tuple(int(v) for v in arr.shape[1:])
    feat_index = _scaled_slice_index(input_index, input_shape, feat_shape, axis)
    feat2d = np.stack([_extract_slice(arr[c], axis, feat_index) for c in range(arr.shape[0])], axis=0)

    mean_map = _normalize01(np.mean(np.abs(feat2d), axis=0))
    max_map = _normalize01(np.max(np.abs(feat2d), axis=0))
    top_channels = _select_topk_channels(arr, topk=topk)

    cols = 2 + len(top_channels)
    fig, axes = plt.subplots(1, cols, figsize=(3.2 * cols, 3.6))
    axes[0].imshow(_normalize01(input_slice), cmap="gray")
    axes[0].set_title("input")
    axes[0].axis("off")

    axes[1].imshow(mean_map, cmap="magma")
    axes[1].set_title(f"{stage_name}\nmean|feat|")
    axes[1].axis("off")

    for plot_idx, ch_idx in enumerate(top_channels, start=2):
        ch_map = _normalize01(np.abs(feat2d[ch_idx]))
        axes[plot_idx].imshow(ch_map, cmap="magma")
        axes[plot_idx].set_title(f"ch {ch_idx}")
        axes[plot_idx].axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{stage_name}.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)

    plt.figure(figsize=(4, 4))
    plt.imshow(max_map, cmap="magma")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{stage_name}_max.png"), dpi=180, bbox_inches="tight")
    plt.close()


def visualize_model_stage_features(
    checkpoint_path: str,
    image_path: str,
    out_dir: str,
    checkpoint_name: str = "model_best",
    axis: str = "z",
    slice_index: Optional[int] = None,
    input_channel: int = 0,
    topk: int = 6,
    save_raw: bool = True,
    device: Optional[str] = None,
):
    os.makedirs(out_dir, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path, pkl_path = _resolve_checkpoint_paths(checkpoint_path, checkpoint_name)
    trainer = restore_model(pkl_path, checkpoint=model_path, train=False)
    network = trainer.network.to(device).eval()

    vol = _load_volume(image_path, input_channel=input_channel)
    if input_channel >= vol.shape[0]:
        raise ValueError(f"input_channel={input_channel} out of range for input with {vol.shape[0]} channels")
    x = _to_input_tensor(vol, device)

    input_vol = vol[input_channel]
    input_shape = tuple(int(v) for v in input_vol.shape)
    if slice_index is None:
        slice_index = input_shape[_axis_to_dim(axis)] // 2
    if not (0 <= slice_index < input_shape[_axis_to_dim(axis)]):
        raise ValueError(f"slice_index {slice_index} is out of range for shape {input_shape} along axis {axis}")
    input_slice = _extract_slice(input_vol, axis, slice_index)

    stage_store: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    handles = _register_stage_hooks(network, stage_store)
    try:
        with torch.no_grad():
            _ = network(x)
    finally:
        for handle in handles:
            handle.remove()

    if not stage_store:
        raise RuntimeError("No stage features were captured. Check that the loaded model matches the expected architecture.")

    case_out_dir = os.path.join(out_dir, _basename_without_double_ext(image_path))
    os.makedirs(case_out_dir, exist_ok=True)
    plt.imsave(os.path.join(case_out_dir, f"input_{axis}_{slice_index}.png"), _normalize01(input_slice), cmap="gray")

    for stage_name, tensor in stage_store.items():
        _save_stage_visualization(
            stage_name=stage_name,
            tensor=tensor,
            input_slice=input_slice,
            input_shape=input_shape,
            input_index=slice_index,
            axis=axis,
            out_dir=case_out_dir,
            topk=topk,
            save_raw=save_raw,
        )

    print(f"Checkpoint: {model_path}")
    print(f"Input: {image_path}")
    print(f"Saved visualizations to: {case_out_dir}")


def _build_argparser():
    parser = argparse.ArgumentParser(
        description="Load an nnUNet/MedNeXt checkpoint and visualize stage-wise feature maps for one 3D volume."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to a fold directory, .model file, or .model.pkl file.",
    )
    parser.add_argument("--image", type=str, required=True, help="Path to a 3D .nii/.nii.gz/.npy/.npz volume.")
    parser.add_argument("--out_dir", type=str, default="./stage_feature_vis", help="Output directory.")
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="model_best",
        help="Checkpoint basename when --checkpoint_path points to a directory.",
    )
    parser.add_argument("--axis", type=str, default="z", choices=["x", "y", "z"], help="Slice axis.")
    parser.add_argument("--slice_index", type=int, default=None, help="Slice index on the chosen axis.")
    parser.add_argument("--input_channel", type=int, default=0, help="Channel index for multi-channel input volumes.")
    parser.add_argument("--topk", type=int, default=6, help="How many high-response channels to render per stage.")
    parser.add_argument("--no_save_raw", action="store_true", help="Do not save stage tensors as .npy files.")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", None], help="Execution device.")
    return parser


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    visualize_model_stage_features(
        checkpoint_path=args.checkpoint_path,
        image_path=args.image,
        out_dir=args.out_dir,
        checkpoint_name=args.checkpoint_name,
        axis=args.axis,
        slice_index=args.slice_index,
        input_channel=args.input_channel,
        topk=args.topk,
        save_raw=not args.no_save_raw,
        device=None if args.device in (None, "None") else args.device,
    )

# python visualize_model_stage_features.py \
#   --checkpoint_path ./ckpt/nnUNet/3d_fullres/Task530_EsoTJ_30pct/nnUNetTrainerV2_Double_CCA_UPSam_fd_loss_RWKV_MedNeXt__nnUNetPlansv2.1_trgSp_1x1x1_rwkv/fold_1 \
#   --checkpoint_name model_best \
#   --image ./ckpt/nnUNet/3d_fullres/Task530_EsoTJ_30pct/nnUNetTrainerV2_Double_CCA_UPSam_fd_loss_RWKV_MedNeXt__nnUNetPlansv2.1_trgSp_1x1x1_rwkv/fold_1/validation_raw/ESO_TJ_60011222468.nii.gz \
#   --out_dir ./stage_feature_vis \
#   --axis z \
#   --topk 6