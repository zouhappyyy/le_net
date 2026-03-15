import os
from typing import List, Tuple

from visualize_val_overlay import (
    visualize_case_overlay,
    TASK570_DEFAULT_CONFIG,
    MODELS_TASK570_EsoTJ83,
)


CASES_X64: List[Tuple[str, str, int]] = [
    ("ESO_TJ_1006597148", "x", 64),
    ("ESO_TJ_1009540774", "x", 64),
    ("ESO_TJ_2002525279", "x", 64),
]


def run_export_three_cases_x64(
    output_dir: str,
    alpha: float,
    save_gt: bool = True,
    save_raw: bool = True,
) -> None:
    """Export overlays for three fixed cases at axis x, slice 64 for all Task570 models.

    Uses TASK570_DEFAULT_CONFIG for data_root/dataset_directory and
    MODELS_TASK570_EsoTJ83 for model prediction folders.

    Additionally, can export the original (no-overlay) slice image once per
    case/axis/slice (controlled by save_raw).
    """
    cfg = TASK570_DEFAULT_CONFIG
    data_root = cfg["data_root"]
    dataset_directory = cfg["dataset_directory"]

    if output_dir is None:
        # default output folder next to the main default folder
        base_default = cfg.get("output_dir", "./Task570_val_vis_all")
        output_dir = os.path.join(os.path.dirname(base_default), "Task570_three_cases_x64")

    if alpha is None:
        alpha = cfg.get("alpha", 0.99)

    os.makedirs(output_dir, exist_ok=True)

    # 为了避免重复导出同一张原图，这里用一个集合做去重
    exported_raw_keys = set()

    for case_id, axis, s in CASES_X64:
        # 先根据需要导出一次原图（不含叠加蒙版）
        if save_raw:
            raw_key = (case_id, axis, s)
            if raw_key not in exported_raw_keys:
                raw_name_prefix = f"raw_{case_id}"
                print(f"[INFO] Exporting RAW image for {case_id} axis-{axis} slice-{s}")
                try:
                    visualize_case_overlay(
                        data_root=data_root,
                        dataset_directory=dataset_directory,
                        pred_folder=None,
                        case_id=case_id,
                        output_dir=output_dir,
                        alpha=alpha,
                        mode="slice",
                        axis=axis,
                        slices=[s],
                        name_prefix=raw_name_prefix,
                        save_gt=False,
                        save_overlay=False,
                        save_raw_only=True,
                    )
                    exported_raw_keys.add(raw_key)
                except TypeError:
                    # 如果旧版本的 visualize_case_overlay 不支持这些参数，给出提示
                    print("[WARN] visualize_case_overlay does not support raw-only saving; "
                          "please update it to accept save_overlay/save_raw_only flags.")
                except Exception as e:
                    print(
                        f"[ERROR] Failed to export RAW image for {case_id} axis-{axis} slice-{s}: {e}",
                    )

        # 再为每个模型导出 overlay（以及可选的 GT）
        for model_name, pred_folder in MODELS_TASK570_EsoTJ83.items():
            print(f"[INFO] Exporting {case_id} axis-{axis} slice-{s} for model {model_name}")
            try:
                visualize_case_overlay(
                    data_root=data_root,
                    dataset_directory=dataset_directory,
                    pred_folder=pred_folder,
                    case_id=case_id,
                    output_dir=output_dir,
                    alpha=alpha,
                    mode="slice",
                    axis=axis,
                    slices=[s],
                    name_prefix=model_name,
                    save_gt=save_gt,
                )
            except Exception as e:
                print(
                    f"[ERROR] Failed to export {case_id} axis-{axis} slice-{s} for model {model_name}: {e}",
                )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export overlays for three fixed ESO Task570 cases at x-axis slice 64.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for exported PNGs (default: Task570_three_cases_x64 next to default config output)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Overlay intensity alpha (default: use TASK570_DEFAULT_CONFIG alpha)",
    )
    parser.add_argument(
        "--no_gt",
        action="store_true",
        help="Do not save GT overlays (only prediction overlays)",
    )
    parser.add_argument(
        "--no_raw",
        action="store_true",
        help="Do not export the original (no-overlay) slice image.",
    )

    args = parser.parse_args()

    run_export_three_cases_x64(
        output_dir=args.output_dir,
        alpha=args.alpha,
        save_gt=not args.no_gt,
        save_raw=not args.no_raw,
    )
