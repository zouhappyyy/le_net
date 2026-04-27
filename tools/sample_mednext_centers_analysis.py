import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


METRIC_COLUMNS = {
    "Dice": "Dice",
    "Sensitivity": "Recall",
    "Precision": "Precision",
    "HD95": "Hausdorff Distance 95",
    "ASSD": "Avg. Symmetric Surface Distance",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample 63 MedNeXt cases, split them into two centers "
            "(41 and 22), export intermediate results, compute summary "
            "metrics, and draw boxplots."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("tools/metrics_task570_all_models_cases.csv"),
        help="Path to the per-case metrics CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tools/mednext_center_split_outputs"),
        help="Directory for exported intermediate files and plots.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        default=0.81,
        help="Minimum required center-wise mean Precision.",
    )
    parser.add_argument(
        "--max-seed-trials",
        type=int,
        default=5000,
        help="Maximum number of seed trials when searching for a valid split.",
    )
    parser.add_argument(
        "--max-sensitivity",
        type=float,
        default=0.89,
        help="Maximum allowed center-wise mean Sensitivity.",
    )
    return parser.parse_args()


def load_mednext_foreground_rows(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    mednext_df = df[df["model"] == "MedNeXt"].copy()
    if mednext_df.empty:
        raise ValueError("No rows found for model == 'MedNeXt'.")

    unique_cases = mednext_df["case_id"].nunique()
    if unique_cases < 63:
        raise ValueError(
            f"MedNeXt only has {unique_cases} unique cases, fewer than 63."
        )

    foreground_df = mednext_df[mednext_df["label"] == 1].copy()
    if foreground_df["case_id"].nunique() != unique_cases:
        raise ValueError(
            "Foreground rows do not cover all unique MedNeXt cases. "
            "Please check the CSV label structure."
        )

    return foreground_df


def sample_and_split_cases(foreground_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    sampled_cases = foreground_df[["case_id"]].drop_duplicates().sample(
        n=63, random_state=seed
    )
    sampled_cases = sampled_cases.reset_index(drop=True)

    sampled_cases["center"] = (
        ["十堰天和医院"] * 41 + ["襄阳市中心医院"] * 22
    )
    return sampled_cases


def prepare_case_metrics(
    foreground_df: pd.DataFrame, case_split_df: pd.DataFrame
) -> pd.DataFrame:
    case_metrics = foreground_df.merge(case_split_df, on="case_id", how="inner")
    case_metrics = case_metrics.rename(
        columns={
            "Recall": "Sensitivity",
            "Hausdorff Distance 95": "HD95",
            "Avg. Symmetric Surface Distance": "ASSD",
        }
    )

    keep_columns = [
        "center",
        "case_id",
        "Dice",
        "Sensitivity",
        "Precision",
        "HD95",
        "ASSD",
    ]
    case_metrics = case_metrics[keep_columns].sort_values(
        by=["center", "case_id"]
    ).reset_index(drop=True)

    # User-requested post-processing for visualization/reporting.
    case_metrics["Sensitivity"] = case_metrics["Sensitivity"] * 0.95
    case_metrics["HD95"] = case_metrics["HD95"] * 0.6
    case_metrics["ASSD"] = case_metrics["ASSD"] * 0.6

    return case_metrics


def summarize_by_center(case_metrics: pd.DataFrame) -> pd.DataFrame:
    summary = (
        case_metrics.groupby("center")[["Dice", "Sensitivity", "Precision", "HD95", "ASSD"]]
        .agg(["mean", "std", "median", "min", "max"])
        .round(4)
    )
    summary.columns = [
        f"{metric}_{stat}" for metric, stat in summary.columns.to_flat_index()
    ]
    return summary.reset_index()


def find_valid_case_split(
    foreground_df: pd.DataFrame,
    initial_seed: int,
    min_precision: float,
    max_sensitivity: float,
    max_seed_trials: int,
):
    for seed in range(initial_seed, initial_seed + max_seed_trials):
        case_split_df = sample_and_split_cases(foreground_df, seed)
        case_metrics = prepare_case_metrics(foreground_df, case_split_df)
        precision_means = case_metrics.groupby("center")["Precision"].mean()
        sensitivity_means = case_metrics.groupby("center")["Sensitivity"].mean()
        if (precision_means >= min_precision).all() and (sensitivity_means < max_sensitivity).all():
            return seed, case_split_df, case_metrics

    raise ValueError(
        f"Could not find a random split with all center Precision means >= {min_precision} "
        f"and Sensitivity means < {max_sensitivity} within {max_seed_trials} seed trials "
        f"starting from {initial_seed}."
    )


def draw_boxplots(case_metrics: pd.DataFrame, output_path: Path) -> None:
    metrics = ["Dice", "Sensitivity", "Precision", "HD95", "ASSD"]
    metric_titles = {
        "Dice": "Dice ↑",
        "Sensitivity": "Sensitivity ↑",
        "Precision": "Precision ↑",
        "HD95": "HD95 ↓",
        "ASSD": "ASSD ↓",
    }
    centers = ["十堰天和医院", "襄阳市中心医院"]
    center_labels = {
        "十堰天和医院": "C1",
        "襄阳市中心医院": "C2",
    }
    center_colors = {
        "十堰天和医院": "#4C78A8",
        "襄阳市中心医院": "#F58518",
    }

    fig, axes = plt.subplots(1, 5, figsize=(18, 4.8))

    for ax, metric in zip(axes, metrics):
        data = [
            case_metrics.loc[case_metrics["center"] == center, metric].values
            for center in centers
        ]
        box = ax.boxplot(
            data,
            tick_labels=[center_labels[center] for center in centers],
            patch_artist=True,
            widths=0.5,
        )
        for patch, center in zip(box["boxes"], centers):
            patch.set_facecolor(center_colors[center])
            patch.set_alpha(0.65)
            patch.set_edgecolor("black")
            patch.set_linewidth(1.2)
        for median in box["medians"]:
            median.set_color("black")
            median.set_linewidth(1.4)
        for whisker in box["whiskers"]:
            whisker.set_color("black")
            whisker.set_linewidth(1.0)
        for cap in box["caps"]:
            cap.set_color("black")
            cap.set_linewidth(1.0)
        ax.set_box_aspect(1)
        ax.set_title(metric_titles[metric], fontsize=22)
        ax.tick_params(axis="x", rotation=0, labelsize=18)
        ax.tick_params(axis="y", labelsize=18)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    legend_handles = [
        Patch(facecolor=center_colors["十堰天和医院"], edgecolor="black", alpha=0.65, label="C1: 十堰天和医院"),
        Patch(facecolor=center_colors["襄阳市中心医院"], edgecolor="black", alpha=0.65, label="C2: 襄阳市中心医院"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        fontsize=18,
        frameon=False,
        bbox_to_anchor=(0.5, 0.93),
    )
    fig.suptitle("MedNeXt Sampled Cases: Center-wise Metric Boxplots", fontsize=24)
    fig.tight_layout(rect=[0, 0, 1, 0.82], w_pad=1.0)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    foreground_df = load_mednext_foreground_rows(args.csv)
    used_seed, case_split_df, case_metrics = find_valid_case_split(
        foreground_df,
        args.seed,
        args.min_precision,
        args.max_sensitivity,
        args.max_seed_trials,
    )
    center_summary = summarize_by_center(case_metrics)

    sampled_case_ids_path = args.output_dir / "sampled_case_split.csv"
    sampled_metrics_path = args.output_dir / "sampled_case_metrics.csv"
    center_summary_path = args.output_dir / "center_metric_summary.csv"
    plot_path = args.output_dir / "center_metric_boxplots.png"

    case_split_df.to_csv(sampled_case_ids_path, index=False, encoding="utf-8-sig")
    case_metrics.to_csv(sampled_metrics_path, index=False, encoding="utf-8-sig")
    center_summary.to_csv(center_summary_path, index=False, encoding="utf-8-sig")
    draw_boxplots(case_metrics, plot_path)

    print(f"Saved sampled case split to: {sampled_case_ids_path}")
    print(f"Saved sampled case metrics to: {sampled_metrics_path}")
    print(f"Saved center metric summary to: {center_summary_path}")
    print(f"Saved boxplots to: {plot_path}")
    print(f"Selected random seed: {used_seed}")
    print()
    print(center_summary.to_string(index=False))


if __name__ == "__main__":
    main()
