import os
import re
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


FILENAME_PATTERN = re.compile(
    r"^(?P<fold>\d+)-(?P<case>[^-]+)-(?P<axis>[xyz])-(?P<slice>\d+)-(?P<model>.+?)-(?P<type>pred|gt)\.png$"
)


def parse_filename(name: str):
    """Parse a filename like `1-CASE-x-64-ModelName-pred.png`.

    Returns (fold, case_id, axis, slice_idx, model, img_type) or None if not match.
    """
    m = FILENAME_PATTERN.match(name)
    if not m:
        return None
    d = m.groupdict()
    return (
        int(d["fold"]),
        d["case"],
        d["axis"],
        int(d["slice"]),
        d["model"],
        d["type"],
    )


def collect_images(root: str):
    """Collect all images in root, grouped by (fold, case_id, axis, slice)."""
    groups: Dict[Tuple[int, str, str, int], Dict[str, Dict[str, str]]] = defaultdict(lambda: defaultdict(dict))
    # groups[(fold, case_id, axis, slice)][model][type] = path

    for fname in os.listdir(root):
        if not fname.endswith(".png"):
            continue
        parsed = parse_filename(fname)
        if parsed is None:
            continue
        fold, case_id, axis, slice_idx, model, img_type = parsed
        key = (fold, case_id, axis, slice_idx)
        groups[key][model][img_type] = os.path.join(root, fname)

    return groups


def plot_group(
    fold: int,
    case_id: str,
    axis: str,
    slice_idx: int,
    model_to_paths: Dict[str, Dict[str, str]],
    output_dir: str,
):
    """For one (fold, case_id, axis, slice), plot all models' pred+gt into one figure.

    Layout: one row per model, two columns (pred, gt).
    """
    models = sorted(model_to_paths.keys())
    if not models:
        return

    n_models = len(models)
    ncols = 2
    nrows = n_models
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))

    # When only one row/col, axes may not be 2D array
    if nrows == 1:
        axes = [axes]
    else:
        axes = axes  # type: ignore

    for row, model in enumerate(models):
        paths = model_to_paths[model]
        # pred image (required)
        pred_path = paths.get("pred")
        gt_path = paths.get("gt")

        # column 0: pred
        ax_pred = axes[row][0] if nrows > 1 else axes[0]
        if pred_path and os.path.isfile(pred_path):
            img_pred = mpimg.imread(pred_path)
            ax_pred.imshow(img_pred)
        ax_pred.set_title(f"{model} - pred")
        ax_pred.axis("off")

        # column 1: gt
        ax_gt = axes[row][1] if nrows > 1 else axes[1]
        if gt_path and os.path.isfile(gt_path):
            img_gt = mpimg.imread(gt_path)
            ax_gt.imshow(img_gt)
        ax_gt.set_title(f"{model} - gt")
        ax_gt.axis("off")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_name = f"{fold}-{case_id}-{axis}-{slice_idx}-ALL_MODELS.png"
    out_path = os.path.join(output_dir, out_name)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved combined figure to: {out_path}")


def combine_val_vis(root: str, output_dir: str):
    groups = collect_images(root)
    if not groups:
        print(f"No valid PNGs found in {root}")
        return

    for (fold, case_id, axis, slice_idx), model_to_paths in sorted(groups.items()):
        plot_group(fold, case_id, axis, slice_idx, model_to_paths, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Combine per-model pred+gt overlays from val_vis_all into per-case, per-axis figures "
            "(one figure per (fold, case_id, axis, slice))."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./val_vis_all",
        help="Folder containing individual model PNGs (default: ./val_vis_all)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./val_vis_combined",
        help="Folder to save combined figures (default: ./val_vis_combined)",
    )

    args = parser.parse_args()
    combine_val_vis(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()


