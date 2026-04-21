import os
import csv
from typing import List, Dict, Tuple


# 现在不再依赖 TP 列，而是从 FN 和 GT 推出 TP = GT - FN
# 如果你仍然有 TP 列，也会被忽略，以 FN 和 GT 为准。

def compute_metrics_from_counts(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Given TP, FP, FN, compute Dice, IoU, Sensitivity, Precision.

    Returns a dict with keys: dice, iou, sensitivity, precision.
    """
    tp_f = float(tp)
    fp_f = float(fp)
    fn_f = float(fn)

    # Dice = 2TP / (2TP + FP + FN)
    denom_dice = 2.0 * tp_f + fp_f + fn_f
    dice = 0.0 if denom_dice == 0.0 else 2.0 * tp_f / denom_dice

    # IoU = TP / (TP + FP + FN)
    denom_iou = tp_f + fp_f + fn_f
    iou = 0.0 if denom_iou == 0.0 else tp_f / denom_iou

    # Sensitivity (Recall) = TP / (TP + FN)
    denom_sens = tp_f + fn_f
    sensitivity = 0.0 if denom_sens == 0.0 else tp_f / denom_sens

    # Precision = TP / (TP + FP)
    denom_prec = tp_f + fp_f
    precision = 0.0 if denom_prec == 0.0 else tp_f / denom_prec

    return {
        "Dice": dice,
        "IoU": iou,
        "Sensitivity": sensitivity,
        "Precision": precision,
    }


def process_tp_fp_fn_csv(input_csv: str, output_csv: str) -> None:
    """Read FN/FP/GT CSV and write a new CSV with Dice/IoU/Sensitivity/Precision columns.

    The input CSV is expected to have at least the following columns:
      - FN
      - FP
      - GT  (ground-truth voxel count for this case/label)
    TP will be derived as: TP = GT - FN.

    It may also contain other columns (e.g. model, case_id, label, n_pred, n_gt),
    which will be copied to the output unchanged.

    The output CSV will contain all input columns plus:
      - Dice
      - IoU
      - Sensitivity
      - Precision
    """
    if not os.path.isfile(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    rows: List[Dict[str, str]] = []
    with open(input_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames_in = reader.fieldnames or []
        required = {"FN", "FP", "GT"}
        missing = required - set(fieldnames_in)
        if missing:
            raise ValueError(
                f"Input CSV {input_csv} is missing required columns (need FN, FP, GT): {sorted(missing)}"
            )
        for r in reader:
            rows.append(r)

    if not rows:
        print(f"No data rows found in {input_csv}, nothing to do.")
        return

    # Prepare fieldnames for output (keep original order, then add new metrics if not already present)
    new_cols = ["Dice", "IoU", "Sensitivity", "Precision"]
    fieldnames_out: List[str] = list(rows[0].keys())
    for c in new_cols:
        if c not in fieldnames_out:
            fieldnames_out.append(c)

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_out)
        writer.writeheader()
        for r in rows:
            try:
                fn = int(r.get("FN", 0))
                fp = int(r.get("FP", 0))
                gt = int(r.get("GT", 0))
            except ValueError:
                # 如果有异常字符串，直接跳过这一行
                print(f"[WARN] Invalid FN/FP/GT values in row, skip: {r}")
                continue

            # 由 GT 和 FN 推出 TP
            tp = gt - fn
            if tp < 0:
                print(f"[WARN] Derived TP < 0 for row (gt={gt}, fn={fn}), clamp to 0: {r}")
                tp = 0

            metrics = compute_metrics_from_counts(tp, fp, fn)
            # 写回新列
            for k, v in metrics.items():
                r[k] = v
            writer.writerow(r)

    print(f"Wrote metrics CSV with Dice/IoU/Sensitivity/Precision to: {output_csv}")


def aggregate_mean_metrics_by_model(
    input_csv: str,
    group_by_label: bool = False,
) -> Tuple[List[Dict[str, object]], List[str]]:
    """Aggregate per-case metrics into dataset-level mean metrics.

    Assumes that `input_csv` already has per-case metrics columns:
      - Dice
      - IoU
      - Sensitivity
      - Precision

    If `group_by_label` is False (default):
      - Each row in the result corresponds to one model (averaged over all cases and labels).
      - Output columns: model, [label if present and not grouping by label? omitted], Dice, IoU, Sensitivity, Precision.

    If `group_by_label` is True:
      - Each row corresponds to (model, label) pair.
      - Output columns: model, label, Dice, IoU, Sensitivity, Precision.

    Returns
    -------
    rows_out : list of dict
        Aggregated rows suitable for writing to CSV.
    metric_names : list of str
        Names of metric columns that were aggregated.
    """
    if not os.path.isfile(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    with open(input_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for col in ("model",):
            if col not in fieldnames:
                raise ValueError(f"Column '{col}' not found in {input_csv}, cannot aggregate by model.")
        if group_by_label and "label" not in fieldnames:
            raise ValueError(f"Column 'label' not found in {input_csv}, cannot aggregate by label.")

        metric_names = [c for c in ("Dice", "IoU", "Sensitivity", "Precision") if c in fieldnames]
        if not metric_names:
            raise ValueError(
                f"No metric columns (Dice/IoU/Sensitivity/Precision) found in {input_csv}. "
                f"Did you run process_tp_fp_fn_csv first?"
            )

        # group key: model or (model, label)
        groups: Dict[Tuple[object, ...], List[Dict[str, float]]] = {}
        for row in reader:
            model = row.get("model")
            if model is None:
                # skip rows without model
                continue
            if group_by_label:
                label = row.get("label")
                key = (model, label)
            else:
                key = (model,)

            metrics_row: Dict[str, float] = {}
            skip_row = False
            for m in metric_names:
                val = row.get(m)
                if val is None or val == "":
                    skip_row = True
                    break
                try:
                    metrics_row[m] = float(val)
                except ValueError:
                    skip_row = True
                    break
            if skip_row:
                continue

            groups.setdefault(key, []).append(metrics_row)

    rows_out: List[Dict[str, object]] = []
    for key, m_list in groups.items():
        if not m_list:
            continue
        agg: Dict[str, object] = {}
        agg["model"] = key[0]
        if group_by_label:
            agg["label"] = key[1]
        for m in metric_names:
            vals = [d[m] for d in m_list]
            if not vals:
                continue
            agg[m] = sum(vals) / float(len(vals))
        rows_out.append(agg)

    return rows_out, metric_names


def write_aggregate_csv(
    rows: List[Dict[str, object]],
    metric_names: List[str],
    output_csv: str,
    group_by_label: bool = False,
) -> None:
    """Write aggregated mean-metric rows to CSV.

    This implementation appends rows to the existing file if it already exists.
    The header is written only when the file is created for the first time.
    """
    if not rows:
        print(f"No aggregated rows to write for {output_csv}, skip.")
        return

    if group_by_label:
        fieldnames = ["model", "label"] + metric_names
    else:
        fieldnames = ["model"] + metric_names

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    file_exists = os.path.isfile(output_csv)
    mode = "a" if file_exists else "w"
    with open(output_csv, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Appended {len(rows)} aggregated rows to: {output_csv}")


if __name__ == "__main__":  # 使用时更推荐通过命令行调用，见下方示例
    # 示例1：从子集 FN/FP/GT CSV 计算每 case 的高级指标
    in_csv = "./Task602_metrics_all_models/metrics_task602_11.csv"
    out_csv = "./Task602_metrics_all_models/metrics_cases.csv"
    process_tp_fp_fn_csv(in_csv, out_csv)

    # 示例2：在上一步结果基础上，按模型计算整个子集的数据集平均指标
    # 注意：这里假设 out_csv 里已经有 "model" 列和四个 metric 列
    mean_rows, metric_cols = aggregate_mean_metrics_by_model(out_csv, group_by_label=False)
    mean_csv = "./Task602_metrics_all_models/metrics_602.csv"
    write_aggregate_csv(mean_rows, metric_cols, mean_csv, group_by_label=False)
