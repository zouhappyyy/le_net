import os
import csv
from typing import Set, List, Dict


def load_case_ids(csv_path: str, case_id_column: str = "case_id") -> Set[str]:
    """Load unique case_ids from a CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the CSV that contains case_id column.
    case_id_column : str, default "case_id"
        Name of the column that stores case IDs.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    case_ids: Set[str] = set()
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if case_id_column not in (reader.fieldnames or []):
            raise ValueError(
                f"Column '{case_id_column}' not found in {csv_path}. "
                f"Available columns: {reader.fieldnames}"
            )
        for row in reader:
            cid = row.get(case_id_column)
            if cid:
                case_ids.add(str(cid))
    return case_ids


def filter_csv_by_case_ids(
    src_csv: str,
    dst_csv: str,
    case_ids: Set[str],
    case_id_column: str = "case_id",
) -> None:
    """Filter rows from src_csv whose case_id is in case_ids and write to dst_csv.

    Keeps all columns and the original header.
    """
    if not os.path.isfile(src_csv):
        raise FileNotFoundError(f"Source CSV not found: {src_csv}")

    with open(src_csv, "r", newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames: List[str] = reader.fieldnames or []
        if case_id_column not in fieldnames:
            raise ValueError(
                f"Column '{case_id_column}' not found in {src_csv}. "
                f"Available columns: {fieldnames}"
            )

        rows_out: List[Dict[str, str]] = []
        for row in reader:
            cid = str(row.get(case_id_column, ""))
            if cid in case_ids:
                rows_out.append(row)

    if not rows_out:
        print(
            f"No rows in {src_csv} matched the given case_ids (size={len(case_ids)}). "
            f"Output CSV will not be created."
        )
        return

    os.makedirs(os.path.dirname(dst_csv) or ".", exist_ok=True)
    with open(dst_csv, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(
        f"Filtered {len(rows_out)} rows from {src_csv} into {dst_csv} "
        f"(from {len(case_ids)} case_ids)."
    )


def main() -> None:
    """Convenience entry point for your Task570 CSV filtering.

    1. 从 metrics_task570_all_models_cases_label1_subsample40_s12.csv 读取 case_id 列；
    2. 在 metrics_task570_all_models_cases_tp_fp_fn.csv 中筛选这些 case_id 的行；
    3. 把结果写到 metrics_task570_all_models_cases_tp_fp_fn_subsample40_s12.csv。
    """
    base_dir = os.path.join(".", "Task570_metrics_all_models")

    src_id_csv = os.path.join(base_dir, "metrics_task570_all_models_cases_label1_subsample40_s12.csv")
    src_tp_fp_fn_csv = os.path.join(base_dir, "metrics_task570_all_models_cases_tp_fp_fn.csv")
    dst_csv = os.path.join(base_dir, "metrics_task570_all_models_cases_tp_fp_fn_subsample40_s12.csv")

    case_ids = load_case_ids(src_id_csv, case_id_column="case_id")
    print(f"Loaded {len(case_ids)} unique case_ids from {src_id_csv}")

    filter_csv_by_case_ids(
        src_csv=src_tp_fp_fn_csv,
        dst_csv=dst_csv,
        case_ids=case_ids,
        case_id_column="case_id",
    )


if __name__ == "__main__":
    main()

