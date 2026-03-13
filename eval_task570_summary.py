import os
import json
from datetime import datetime
from collections import OrderedDict

import nibabel as nib
import numpy as np

from nnunet_mednext.evaluation.evaluator import Evaluator


GT_DIR = "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_raw_data/Task570_EsoTJ83/labelsTr"
# 这里选择一个模型的预测结果来构建 summary；可以按需改为三个模型分别生成
PRED_DIR = "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_predictions/Task570_EsoTJ83/Double_CCA_UPSam_fd_loss_RWKV/preds"

OUTPUT_SUMMARY = os.path.join(PRED_DIR, "summary_task570.json")


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
    # result keys are labels as strings, e.g. "0", "1", ...
    out = {}
    for lbl, metrics in result.items():
        out[str(lbl)] = OrderedDict(metrics)
    return out


def main():
    case_ids = list_case_ids(GT_DIR)
    print(f"Found {len(case_ids)} GT cases in {GT_DIR}")

    per_case_results = []  # will become results["all"]
    per_label_accumulator = {}  # label -> list of metric dicts

    for cid in case_ids:
        gt_path = os.path.join(GT_DIR, f"{cid}.nii.gz")
        if not os.path.isfile(gt_path):
            gt_path = os.path.join(GT_DIR, f"{cid}.nii")
            if not os.path.isfile(gt_path):
                print(f"[WARN] GT not found for {cid}, skip")
                continue

        pred_path = os.path.join(PRED_DIR, f"{cid}.nii.gz")
        if not os.path.isfile(pred_path):
            pred_path = os.path.join(PRED_DIR, f"{cid}.nii")
        if not os.path.isfile(pred_path):
            print(f"[WARN] prediction not found for {cid}, skip")
            continue

        gt = load_nii_int(gt_path)
        pred = load_nii_int(pred_path)

        if pred.shape != gt.shape:
            print(f"[WARN] shape mismatch for {cid}: pred {pred.shape}, gt {gt.shape} (no auto-fix)")

        case_metrics = compute_case_metrics(pred, gt)  # {label: {metric: value}}

        # assemble case entry similar to nnUNet summary.json
        case_entry = {lbl: metrics for lbl, metrics in case_metrics.items()}
        case_entry["reference"] = gt_path
        case_entry["test"] = pred_path
        per_case_results.append(case_entry)

        # accumulate for mean
        for lbl, metrics in case_metrics.items():
            if lbl not in per_label_accumulator:
                per_label_accumulator[lbl] = []
            per_label_accumulator[lbl].append(metrics)

    # compute mean over cases per label
    mean_results = {}
    for lbl, metrics_list in per_label_accumulator.items():
        # collect all metric names
        metric_names = metrics_list[0].keys()
        mean_metrics = {}
        for m in metric_names:
            values = [float(d[m]) for d in metrics_list]
            mean_metrics[m] = float(np.mean(values))
        mean_results[str(lbl)] = mean_metrics

    summary = OrderedDict()
    summary["author"] = "Task570_eval"
    summary["description"] = "Evaluation summary for Task570_EsoTJ83 using predictions from " + PRED_DIR
    summary["id"] = datetime.now().strftime("%Y%m%d%H%M%S")
    summary["name"] = "Task570_EsoTJ83"
    summary["results"] = {
        "all": per_case_results,
        "mean": mean_results,
    }
    summary["task"] = "Task570_EsoTJ83"
    summary["timestamp"] = str(datetime.now())

    with open(OUTPUT_SUMMARY, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"Saved Task570 summary to {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()

