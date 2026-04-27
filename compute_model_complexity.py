import argparse
import json
import os
from typing import Any

import torch
import torch.nn as nn

from nnunet_mednext.training.model_restore import restore_model
from nnunet_mednext.training.network_training.nnUNetTrainerV2_Double_CCA_UPSam_fd_loss_RWKV_MedNeXt import (
    nnUNetTrainerV2_Double_CCA_UPSam_fd_loss_RWKV_MedNeXt,
)


def _count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


class _TensorOutputWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def _first_tensor(self, x: Any):
        if torch.is_tensor(x):
            return x
        if isinstance(x, (list, tuple)):
            for item in x:
                out = self._first_tensor(item)
                if out is not None:
                    return out
        if isinstance(x, dict):
            for item in x.values():
                out = self._first_tensor(item)
                if out is not None:
                    return out
        return None

    def forward(self, x):
        out = self.model(x)
        tensor = self._first_tensor(out)
        if tensor is None:
            raise RuntimeError("Model forward did not return a tensor output.")
        return tensor


def build_task530_rwkv_mednext_model(in_channels: int = 1, num_classes: int = 2):
    trainer = nnUNetTrainerV2_Double_CCA_UPSam_fd_loss_RWKV_MedNeXt(
        plans_file="dummy_plans.pkl",
        fold=0,
        output_folder=".",
        dataset_directory=".",
        batch_dice=False,
        stage=0,
        unpack_data=False,
        deterministic=True,
        fp16=False,
    )
    trainer.num_input_channels = in_channels
    trainer.num_classes = num_classes
    trainer.initialize_network()
    model = trainer.network
    model.eval()
    return model


def _resolve_checkpoint_paths(path: str, checkpoint_name: str):
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


def load_model_from_checkpoint_path(checkpoint_path: str, checkpoint_name: str = "model_best"):
    model_path, pkl_path = _resolve_checkpoint_paths(checkpoint_path, checkpoint_name)
    trainer = restore_model(pkl_path, checkpoint=model_path, train=False)
    model = trainer.network
    model.eval()
    return model, model_path


def compute_complexity(
    checkpoint_path: str = None,
    checkpoint_name: str = "model_best",
    in_channels: int = 1,
    num_classes: int = 2,
    spatial_size=(64, 64, 64),
    device: str = "cpu",
):
    loaded_checkpoint = None
    if checkpoint_path is not None:
        model, loaded_checkpoint = load_model_from_checkpoint_path(checkpoint_path, checkpoint_name=checkpoint_name)
    else:
        model = build_task530_rwkv_mednext_model(in_channels=in_channels, num_classes=num_classes)
    model = model.to(device)
    total_params, trainable_params = _count_parameters(model)

    x = torch.zeros((1, in_channels, *spatial_size), device=device, dtype=torch.float32)
    wrapped = _TensorOutputWrapper(model).to(device).eval()

    flops_total = None
    unsupported_ops = None
    uncalled_modules = None
    try:
        from fvcore.nn import FlopCountAnalysis

        flops = FlopCountAnalysis(wrapped, x)
        flops_total = int(flops.total())
        unsupported_ops = flops.unsupported_ops()
        uncalled_modules = flops.uncalled_modules()
    except ImportError:
        pass

    result = {
        "model": "nnUNetTrainerV2_Double_CCA_UPSam_fd_loss_RWKV_MedNeXt",
        "loaded_checkpoint": loaded_checkpoint,
        "input_shape": [1, in_channels, *spatial_size],
        "device": device,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_params_million": round(total_params / 1e6, 4),
        "trainable_params_million": round(trainable_params / 1e6, 4),
        "flops": flops_total,
        "flops_giga": None if flops_total is None else round(flops_total / 1e9, 4),
        "unsupported_ops": None if unsupported_ops is None else {str(k): int(v) for k, v in unsupported_ops.items()},
        "uncalled_modules": None if uncalled_modules is None else sorted(list(uncalled_modules)),
        "notes": [
            "FLOPs are input-size dependent.",
            "If checkpoint_path is omitted, the script instantiates the architecture directly from the trainer config and does not require checkpoint weights.",
            "If fvcore is unavailable, only parameter counts are reported.",
        ],
    }
    return result


def _build_argparser():
    parser = argparse.ArgumentParser(description="Compute parameter count and FLOPs for the Task530 RWKV-MedNeXt model.")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to a fold directory, .model file, or .model.pkl file.")
    parser.add_argument("--checkpoint_name", type=str, default="model_best", help="Checkpoint basename when checkpoint_path is a directory.")
    parser.add_argument("--in_channels", type=int, default=1, help="Number of input channels.")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of output classes.")
    parser.add_argument("--depth", type=int, default=64, help="Input depth.")
    parser.add_argument("--height", type=int, default=64, help="Input height.")
    parser.add_argument("--width", type=int, default=64, help="Input width.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device for analysis.")
    parser.add_argument("--json", action="store_true", help="Print results as JSON.")
    return parser


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    result = compute_complexity(
        checkpoint_path=args.checkpoint_path,
        checkpoint_name=args.checkpoint_name,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        spatial_size=(args.depth, args.height, args.width),
        device=args.device,
    )

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"Model: {result['model']}")
        if result["loaded_checkpoint"] is not None:
            print(f"Loaded checkpoint: {result['loaded_checkpoint']}")
        print(f"Input shape: {result['input_shape']}")
        print(f"Total params: {result['total_params']} ({result['total_params_million']} M)")
        print(f"Trainable params: {result['trainable_params']} ({result['trainable_params_million']} M)")
        if result["flops"] is None:
            print("FLOPs: unavailable (install fvcore to enable)")
        else:
            print(f"FLOPs: {result['flops']} ({result['flops_giga']} GFLOPs)")
            print(f"Unsupported ops: {result['unsupported_ops']}")
            print(f"Uncalled modules: {result['uncalled_modules']}")
