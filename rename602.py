import os
from pathlib import Path

def rename_nii_gz_in_dir(root_dir: Path, old_prefix: str = "esophagus_", new_prefix: str = "Task602_ls_"):
    """
    递归遍历 root_dir 下的所有文件，将 .nii.gz 文件名中的 old_prefix 替换为 new_prefix。
    只改文件名，不改路径。
    """
    root_dir = Path(root_dir)
    if not root_dir.is_dir():
        print(f"[ERROR] {root_dir} 不是有效文件夹")
        return

    count_total = 0
    count_renamed = 0

    for path in root_dir.rglob("*.nii.gz"):
        count_total += 1
        old_name = path.name

        if old_prefix not in old_name:
            # 不包含指定前缀的，跳过
            continue

        new_name = old_name.replace(old_prefix, new_prefix)
        if new_name == old_name:
            # 替换后相同，跳过
            continue

        new_path = path.with_name(new_name)

        # 如果目标文件已存在，避免覆盖
        if new_path.exists():
            print(f"[WARN] 目标已存在，跳过：{new_path}")
            continue

        print(f"Rename: {path}  ->  {new_path}")
        path.rename(new_path)
        count_renamed += 1

    print(f"\n扫描到 .nii.gz 文件总数: {count_total}")
    print(f"成功重命名文件数: {count_renamed}")


if __name__ == "__main__":

    target_dirs = [
        "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_predictions/Task602_ls/nnFormer",
        "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_predictions/Task602_ls/MedNeXt",
"/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_predictions/Task602_ls/nnU-Net",
     "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_predictions/Task602_ls/SwinUNETR",
   "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_predictions/Task602_ls/UMamba",
   "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_predictions/Task602_ls/VoComni_nnunet",

    ]

    for d in target_dirs:
        print(f"\n=== 处理目录: {d} ===")
        rename_nii_gz_in_dir(Path(d))