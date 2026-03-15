import os
import csv
import random
from collections import defaultdict, OrderedDict

import numpy as np


INPUT_CSV = "./Task570_metrics_all_models/metrics_task570_all_models_cases.csv"
OUTPUT_DIR = "./Task570_metrics_all_models"

SAMPLE_SIZE = 40
RANDOM_SEED = 88  # 基础随机种子
NUM_SUBSETS = 100    # 需要生成多少个随机子集

# 对外想保留的 7 个指标列名（输出 CSV 使用这些名字）
METRICS_TO_KEEP = [
    "Dice",
    "IoU",
    "Sensitivity",
    "Precision",
    "D95",
    "ASSD",
]

# 显式列名映射：输出名 -> 原 CSV 列名
# 83 例总表列名:
# Accuracy, Avg. Symmetric Surface Distance, Dice, False Discovery Rate, False Negative Rate,
# False Omission Rate, False Positive Rate, Hausdorff Distance 95, Jaccard, Negative Predictive Value,
# Precision, Recall, Total Positives Reference, Total Positives Test, True Negative Rate
METRIC_COLUMN_MAP = {
    "Dice": "Dice",
    "IoU": "Jaccard",  # IoU == Jaccard
    "Sensitivity": "Recall",  # Sensitivity == Recall
    "Precision": "Precision",
    "Specificity": "True Negative Rate",  # Specificity == TNR
    "D95": "Hausdorff Distance 95",
    "ASSD": "Avg. Symmetric Surface Distance",
}


def read_cases_csv(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows, reader.fieldnames  # ['model','case_id','label',metric1,...]


def sample_case_ids(rows, sample_size, seed=None):
    # 从全表中收集所有 unique case_id
    all_case_ids = sorted({r["case_id"] for r in rows})
    if len(all_case_ids) < sample_size:
        raise RuntimeError(
            f"Only {len(all_case_ids)} unique cases found, cannot sample {sample_size}."
        )
    rnd = random.Random(seed)
    rnd.shuffle(all_case_ids)
    subset = all_case_ids[:sample_size]
    return set(subset)


def filter_rows_by_case_ids(rows, subset_ids):
    return [r for r in rows if r["case_id"] in subset_ids]


def compute_mean_metrics(rows, metric_names, fieldnames):
    """rows: list of dict, keys: 'model','case_id','label',<metrics...>
    返回: list[dict]，每个 dict: {'model': ..., 'label': ..., <metric>: mean_value}
    metric_names: 输出指标名（使用 METRICS_TO_KEEP）
    fieldnames: 原始 CSV 的列名列表，用于校验映射
    """
    # 校验映射列是否都在原始 CSV 中
    for out_name in metric_names:
        src_col = METRIC_COLUMN_MAP[out_name]
        if src_col not in fieldnames:
            raise RuntimeError(
                f"Source column '{src_col}' for metric '{out_name}' not found in input CSV."
            )

    buckets = defaultdict(list)
    for r in rows:
        model = r["model"]
        label = r["label"]
        buckets[(model, label)].append(r)

    mean_rows = []
    for (model, label), group in buckets.items():
        mean_row = OrderedDict()
        mean_row["model"] = model
        mean_row["label"] = label
        for out_name in metric_names:
            src_col = METRIC_COLUMN_MAP[out_name]
            vals = [
                float(g[src_col])
                for g in group
                if src_col in g and g[src_col] not in ("", None)
            ]
            mean_val = float("nan") if len(vals) == 0 else float(np.mean(vals))
            # 输出时用你想要的名字（Dice / IoU / Sensitivity / ...）
            mean_row[out_name] = mean_val
        mean_rows.append(mean_row)

    return mean_rows


def write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Saved CSV to {path}")


def main():
    if not os.path.isfile(INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    all_rows, fieldnames = read_cases_csv(INPUT_CSV)
    if len(fieldnames) < 4:
        raise RuntimeError("Unexpected CSV format: need at least model, case_id, label, and one metric")

    # 输出指标名固定为 METRICS_TO_KEEP（不再直接用原 CSV 的列名）
    metric_names = METRICS_TO_KEEP

    # [1] 只保留 label == "1" 的行
    rows_label1 = [r for r in all_rows if str(r.get("label", "")) == "1"]
    if not rows_label1:
        raise RuntimeError("No rows with label == 1 found in the CSV.")

    print(f"Total rows label==1: {len(rows_label1)}")

    # 用于汇总所有子集的 mean 结果
    all_mean_rows_with_subset = []

    # [2] 一次性生成多个随机子集
    for subset_idx in range(NUM_SUBSETS):
        seed = RANDOM_SEED + subset_idx
        subset_case_ids = sample_case_ids(rows_label1, SAMPLE_SIZE, seed=seed)
        print(
            f"Subset {subset_idx}: sampled {SAMPLE_SIZE} cases (seed={seed}), "
            f"example ids: {sorted(list(subset_case_ids))[:5]} ..."
        )

        # 过滤出该子集的所有行（7 个模型、该 label 的指标）
        sampled_rows = filter_rows_by_case_ids(rows_label1, subset_case_ids)

        # 导出该子集的病例级 CSV：对外字段名使用 7 个指标名，但取值用映射
        out_cases = os.path.join(
            OUTPUT_DIR,
            f"metrics_task570_all_models_cases_label1_subsample{SAMPLE_SIZE}_s{subset_idx}.csv",
        )
        case_fieldnames = ["model", "case_id", "label"] + metric_names

        trimmed_sampled_rows = []
        for r in sampled_rows:
            new_r = {
                "model": r.get("model", ""),
                "case_id": r.get("case_id", ""),
                "label": r.get("label", ""),
            }
            # 按映射，从原列取数，写到新列名中
            for out_name in metric_names:
                src_col = METRIC_COLUMN_MAP[out_name]
                new_r[out_name] = r.get(src_col, "")
            trimmed_sampled_rows.append(new_r)

        write_csv(out_cases, case_fieldnames, trimmed_sampled_rows)

        # 计算该子集的 mean 指标（同样用映射，从原 CSV 列取数）
        mean_rows = compute_mean_metrics(sampled_rows, metric_names, fieldnames)

        # 给 mean 结果加上 subset_id 字段，方便区分
        for r in mean_rows:
            r["subset_id"] = subset_idx
        all_mean_rows_with_subset.extend(mean_rows)

        # 也可以单独为每个子集写一个 mean CSV（可选）
        out_mean_each = os.path.join(
            OUTPUT_DIR,
            f"metrics_task570_all_models_mean_label1_subsample{SAMPLE_SIZE}_s{subset_idx}.csv",
        )
        # 这里把 subset_id 加进 fieldnames
        mean_fieldnames_each = ["subset_id", "model", "label"] + metric_names
        write_csv(out_mean_each, mean_fieldnames_each, mean_rows)

    # [3] 把所有子集的 mean 结果写到一个总表中
    # 汇总表字段: subset_id, model, label, metrics...
    summary_fieldnames = ["subset_id", "model", "label"] + metric_names
    # 调整字段顺序
    for r in all_mean_rows_with_subset:
        # 确保 key 顺序与 summary_fieldnames 一致
        ordered = OrderedDict()
        for k in summary_fieldnames:
            ordered[k] = r.get(k, "")
        # 用有序 dict 替换
        r.clear()
        r.update(ordered)

    out_summary = os.path.join(
        OUTPUT_DIR,
        f"metrics_task570_all_models_mean_label1_subsample{SAMPLE_SIZE}_multi{NUM_SUBSETS}.csv",
    )
    write_csv(out_summary, summary_fieldnames, all_mean_rows_with_subset)


if __name__ == "__main__":
    main()

