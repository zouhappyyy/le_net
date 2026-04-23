#!/usr/bin/env python3
"""Compute parameter count and FLOPs for Double_UpSam_RWKV_MedNeXt.

Usage examples:
    python tools/compute_model_stats.py --in-channels 1 --n-classes 2 --n-channels 16 --input-shape 32 32 32 --batch 1 --device cpu

The script will try thop first, then fvcore as a fallback. If neither is
available it will print parameter count and advise installing one of the
profilers.
"""
import argparse
import sys
import torch
import math


def human(n):
    if n >= 1e9:
        return f"{n/1e9:.3f} G"
    if n >= 1e6:
        return f"{n/1e6:.3f} M"
    if n >= 1e3:
        return f"{n/1e3:.3f} K"
    return str(n)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def try_thop(model, x):
    try:
        from thop import profile
    except Exception as e:
        raise
    # thop may raise for unknown ops; leave exception to caller
    flops, _ = profile(model, inputs=(x,), verbose=False)
    return float(flops)


def try_fvcore(model, x):
    from fvcore.nn import FlopCountAnalysis

    fca = FlopCountAnalysis(model, x)
    # returns a torch-compatible number
    total = fca.total()
    try:
        # total may be a dict for ops; convert to int
        total = int(total)
    except Exception:
        total = float(total)
    return float(total)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--n-classes", type=int, default=2)
    parser.add_argument("--n-channels", type=int, default=16)
    parser.add_argument("--exp-r", type=int, default=2)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--block-counts", nargs="*", type=int, default=None)
    parser.add_argument("--deep-supervision", action="store_true")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--input-shape", nargs=3, type=int, metavar=("D", "H", "W"), default=[32, 32, 32])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--no-thop", action="store_true", help="Don't try thop")
    parser.add_argument("--no-fvcore", action="store_true", help="Don't try fvcore")
    args = parser.parse_args(argv)

    # lazy import model (project relative import)
    try:
        from nnunet_mednext.network_architecture.le_networks.Double_UPSam_RWKV_MedNeXt import (
            Double_UpSam_RWKV_MedNeXt,
        )
    except Exception as e:
        print("Failed to import Double_UpSam_RWKV_MedNeXt:", e)
        sys.exit(1)

    kw = dict(
        in_channels=args.in_channels,
        n_channels=args.n_channels,
        n_classes=args.n_classes,
        exp_r=args.exp_r,
        kernel_size=args.kernel_size,
        deep_supervision=args.deep_supervision,
        block_counts=args.block_counts,
        dim="3d",
    )

    model = Double_UpSam_RWKV_MedNeXt(**kw)
    # For profiling the usual single-output inference path we prefer no deep
    # supervision. If user passed --deep-supervision keep it.
    if not args.deep_supervision:
        model.do_ds = False

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    model.to(device)
    model.eval()

    batch = args.batch
    D, H, W = args.input_shape
    x = torch.randn(batch, args.in_channels, D, H, W, device=device)

    params = count_params(model)
    print("Parameters:", params, "->", human(params))

    flops = None
    tried = []
    # Try thop first
    if not args.no_thop:
        try:
            tried.append("thop")
            flops = try_thop(model, x)
            print("FLOPs (thop):", int(flops), "->", human(flops))
        except Exception as e:
            print("thop failed:", e)
            flops = None

    # Try fvcore
    if flops is None and not args.no_fvcore:
        try:
            tried.append("fvcore")
            flops = try_fvcore(model, x)
            print("FLOPs (fvcore):", int(flops), "->", human(flops))
        except Exception as e:
            print("fvcore failed:", e)
            flops = None

    if flops is None:
        print("FLOPs: not available. Install thop (`pip install thop`) or fvcore (`pip install fvcore`) and retry.")
    else:
        # report GFLOPs for convenience
        print("GFLOPs:", flops / 1e9)


if __name__ == "__main__":
    main(sys.argv[1:])
