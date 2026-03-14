import os
import json
import csv
from datetime import datetime
from collections import OrderedDict

import nibabel as nib
import numpy as np

from nnunet_mednext.evaluation.evaluator import Evaluator


GT_DIR = "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_raw_data/Task570_EsoTJ83/labelsTr"

# 统一汇总输出目录，存放所有模型的总表 CSV
AGGREGATE_OUT_DIR = "./Task570_metrics_all_models"

# 七个模型在 Task570 上的预测目录，根据你实际保存的位置稍作调整
MODELS = {
    # 模型名: preds 目录
    "BANet": "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_predictions/Task570_EsoTJ83/BANetTrainerV2/preds",
    "BGHNetV4": "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_predictions/Task570_EsoTJ83/BGHNetV4Trainer/preds",
    "MedNeXt": "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_predictions/Task570_EsoTJ83/MedNeXt_S_kernel3/preds",
    "nnUNet": "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_predictions/Task570_EsoTJ83/nnUNetTrainerV2/preds",
    "RWKV_fd": "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_predictions/Task570_EsoTJ83/Double_CCA_UPSam_fd_RWKV/preds",
    "RWKV_fd_loss": "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_predictions/Task570_EsoTJ83/Double_CCA_UPSam_fd_loss_RWKV/preds",
    "UNet3D": "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_predictions/Task570_EsoTJ83/UNet3DTrainer/preds",
}


def list_case_ids(gt_dir):
    ids = []
    for f in os.listdir(gt_dir):
        if f.endswith(".nii.gz") or f.endswith(".nii"):
            cid = f.split(".")[0]
            ids.append(cid)
    ids.sort()
    return ids


def load_nii_int(path):
    img = nib.load(path)
    arr = img.get_fdata()
    return arr.astype(np.int16)


def compute_case_metrics(pred, gt):
    """Use Evaluator to compute all default metrics for one case.

    Returns a dict mapping label str -> metric dict, e.g. {"1": {"Dice": ..., ...}}.
    """
    evaluator = Evaluator(test=pred, reference=gt)
    result = evaluator.evaluate()
    out = {}
    for lbl, metrics in result.items():
        out[str(lbl)] = OrderedDict(metrics)
    return out


def evaluate_one_model(model_name: str, pred_dir: str):
    case_ids = list_case_ids(GT_DIR)
    print(f"\n[MODEL] {model_name}: found {len(case_ids)} GT cases")

    per_case_results = []
    per_label_accumulator = {}
    rows_cases = []  # for CSV

    for cid in case_ids:
        gt_path = os.path.join(GT_DIR, f"{cid}.nii.gz")
        if not os.path.isfile(gt_path):
            gt_path = os.path.join(GT_DIR, f"{cid}.nii")
            if not os.path.isfile(gt_path):
                print(f"  [WARN] GT not found for {cid}, skip")
                continue

        pred_path = os.path.join(pred_dir, f"{cid}.nii.gz")
        if not os.path.isfile(pred_path):
            pred_path = os.path.join(pred_dir, f"{cid}.nii")
        if not os.path.isfile(pred_path):
            print(f"  [WARN] prediction not found for {cid}, skip")
            continue

        gt = load_nii_int(gt_path)
        pred = load_nii_int(pred_path)

        if pred.shape != gt.shape:
            print(f"  [WARN] shape mismatch for {cid}: pred {pred.shape}, gt {gt.shape} (no auto-fix)")

        case_metrics = compute_case_metrics(pred, gt)  # {label: {metric: value}}

        # assemble case entry similar to nnUNet summary.json
        case_entry = {lbl: metrics for lbl, metrics in case_metrics.items()}
        case_entry["reference"] = gt_path
        case_entry["test"] = pred_path
        per_case_results.append(case_entry)

        # accumulate for mean & CSV
        for lbl, metrics in case_metrics.items():
            if lbl not in per_label_accumulator:
                per_label_accumulator[lbl] = []
            per_label_accumulator[lbl].append(metrics)

            row = {"case_id": cid, "label": lbl}
            for m_name, m_val in metrics.items():
                row[m_name] = float(m_val)
            rows_cases.append(row)

    if not per_case_results:
        print(f"[MODEL] {model_name}: no valid cases, skip summary/CSV")
        return

    # compute mean over cases per label
    mean_results = {}
    rows_mean = []
    # metric_names 从第一个 label 的第一条样本里取
    some_lbl = next(iter(per_label_accumulator.keys()))
    metric_names = list(per_label_accumulator[some_lbl][0].keys())

    for lbl, metrics_list in per_label_accumulator.items():
        mean_metrics = {}
        for m in metric_names:
            values = [float(d[m]) for d in metrics_list]
            mean_metrics[m] = float(np.mean(values))
        mean_results[str(lbl)] = mean_metrics

        row_m = {"label": lbl}
        row_m.update(mean_metrics)
        rows_mean.append(row_m)

    # summary.json
    summary = OrderedDict()
    summary["author"] = "Task570_eval"
    summary["description"] = (
        f"Evaluation summary for Task570_EsoTJ83 using predictions from {pred_dir} (model={model_name})"
    )
    summary["id"] = datetime.now().strftime("%Y%m%d%H%M%S")
    summary["name"] = "Task570_EsoTJ83"
    summary["results"] = {
        "all": per_case_results,
        "mean": mean_results,
    }
    summary["task"] = "Task570_EsoTJ83"
    summary["timestamp"] = str(datetime.now())

    os.makedirs(pred_dir, exist_ok=True)
    summary_path = os.path.join(pred_dir, f"summary_task570_{model_name}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"[MODEL] {model_name}: saved summary to {summary_path}")

    # CSV: per-case metrics
    cases_csv = os.path.join(pred_dir, f"metrics_task570_{model_name}_cases.csv")
    case_fieldnames = ["case_id", "label"] + metric_names
    with open(cases_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=case_fieldnames)
        writer.writeheader()
        for r in rows_cases:
            writer.writerow(r)
    print(f"[MODEL] {model_name}: saved per-case metrics CSV to {cases_csv}")

    # CSV: mean metrics per label
    mean_csv = os.path.join(pred_dir, f"metrics_task570_{model_name}_mean.csv")
    mean_fieldnames = ["label"] + metric_names
    with open(mean_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=mean_fieldnames)
        writer.writeheader()
        for r in rows_mean:
            writer.writerow(r)
    print(f"[MODEL] {model_name}: saved mean metrics CSV to {mean_csv}")

    return rows_cases, rows_mean, metric_names


def main():
    os.makedirs(AGGREGATE_OUT_DIR, exist_ok=True)

    # 全模型汇总：per-case 和 mean
    all_rows_cases = []
    all_rows_mean = []
    global_metric_names = None

    for model_name, pred_dir in MODELS.items():
        rows_cases, rows_mean, metric_names = evaluate_one_model(model_name, pred_dir)
        if not rows_cases:
            continue
        # 记录本模型名
        for r in rows_cases:
            r_with_model = {"model": model_name}
            r_with_model.update(r)
            all_rows_cases.append(r_with_model)
        for r in rows_mean:
            r_with_model = {"model": model_name}
            r_with_model.update(r)
            all_rows_mean.append(r_with_model)
        # 统一 metric_names（假设各模型一致）
        if global_metric_names is None:
            global_metric_names = metric_names

    if all_rows_cases and global_metric_names is not None:
        # 汇总 per-case CSV：model, case_id, label, metrics...
        agg_cases_csv = os.path.join(AGGREGATE_OUT_DIR, "metrics_task570_all_models_cases.csv")
        case_fieldnames = ["model", "case_id", "label"] + global_metric_names
        with open(agg_cases_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=case_fieldnames)
            writer.writeheader()
            for r in all_rows_cases:
                writer.writerow(r)
        print(f"[AGG] Saved aggregated per-case metrics to {agg_cases_csv}")

    if all_rows_mean and global_metric_names is not None:
        # 汇总 mean CSV：model, label, metrics...
        agg_mean_csv = os.path.join(AGGREGATE_OUT_DIR, "metrics_task570_all_models_mean.csv")
        mean_fieldnames = ["model", "label"] + global_metric_names
        with open(agg_mean_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=mean_fieldnames)
            writer.writeheader()
            for r in all_rows_mean:
                writer.writerow(r)
        print(f"[AGG] Saved aggregated mean metrics to {agg_mean_csv}")


if __name__ == "__main__":
    main()

