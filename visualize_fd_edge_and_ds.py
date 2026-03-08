import os
import argparse
from typing import List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from nnunet_mednext.training.network_training.nnUNetTrainerV2_Double_CCA_UPSam_fd_loss_RWKV_MedNeXt import (
    nnUNetTrainerV2_Double_CCA_UPSam_fd_loss_RWKV_MedNeXt,
)
from nnunet_mednext.utilities.to_torch import maybe_to_torch
from nnunet_mednext.utilities.nd_softmax import softmax_helper


def get_trainer(plans_file: str, fold: int, output_folder: str) -> nnUNetTrainerV2_Double_CCA_UPSam_fd_loss_RWKV_MedNeXt:
    """Initialize trainer from an explicit plans_file, fold, and trained output_folder.

    output_folder must be the same as used during training, e.g.
    /home/fangzheng/zoule/mednext/ckpt/nnUNet/3d_fullres/Task530_EsoTJ_30pct/
    nnUNetTrainerV2_Double_CCA_UPSam_fd_loss_RWKV_MedNeXt__nnUNetPlansv2.1_trgSp_1x1x1_rwkv
    so that fold-specific checkpoints (fold_X/model_*.model) can be found.
    """
    plans_dir = os.path.dirname(plans_file)
    # For Task530_EsoTJ_30pct the dataset_directory is the Task folder itself
    # (it contains dataset.json and nnUNetData_plans_v2.1_trgSp_1x1x1_stage0)
    dataset_directory = plans_dir  # /.../nnUNet_preprocessed/Task530_EsoTJ_30pct

    trainer = nnUNetTrainerV2_Double_CCA_UPSam_fd_loss_RWKV_MedNeXt(
        plans_file,
        fold,
        output_folder=output_folder,
        dataset_directory=dataset_directory,
        batch_dice=True,
        stage=0,
        unpack_data=False,
        deterministic=False,
        fp16=False,
    )
    trainer.initialize(training=False)
    try:
        trainer.load_best_checkpoint(train=False)
    except Exception:
        trainer.load_final_checkpoint(train=False)
    trainer.network.eval()
    return trainer


def _extract_case_data(data_root: str, case_id: str, dataset_directory: str) -> Tuple[np.ndarray, np.ndarray]:
    """从预处理数据目录和 gt_segmentations 中读取某病例的 data 和 seg.

    对于 Task530_EsoTJ_30pct，stage0 预处理数据每个病例对应单个 .npy 文件，
    其 shape 为 (C, D, H, W)。因此这里：
      - 直接加载 <case_id>.npy（或以 case_id 开头的 .npy），并规范为 data[C, D, H, W]
      - 从 dataset_directory/gt_segmentations 读取对应的 NIfTI 标签作为 seg[D, H, W]
    """
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"data_root {data_root} does not exist")

    # 1) 找到单个 .npy 文件（例如 ESO_TJ_60011222468.npy）
    exact_npy = os.path.join(data_root, f"{case_id}.npy")
    if os.path.isfile(exact_npy):
        npy_candidates = [exact_npy]
    else:
        npy_files = [
            os.path.join(data_root, f) for f in os.listdir(data_root)
            if f.startswith(case_id) and f.endswith(".npy")
        ]
        if not npy_files:
            raise FileNotFoundError(
                f"No preprocessed .npy file found for case {case_id} under {data_root}. "
                f"Expected a file like {case_id}.npy."
            )
        if len(npy_files) > 1:
            print(f"[WARN] Multiple .npy files match case_id {case_id}: {[os.path.basename(f) for f in npy_files]}. Using {os.path.basename(npy_files[0])}")
        npy_candidates = [npy_files[0]]

    arr = np.load(npy_candidates[0])
    # 规范化为 [C, D, H, W]
    if arr.ndim == 4:
        # 常见情况: (C, D, H, W)
        data = arr.astype(np.float32)
    elif arr.ndim == 3:
        # 单通道 (D, H, W)
        data = arr[None].astype(np.float32)
    else:
        raise RuntimeError(f"Unexpected array shape {arr.shape} in {npy_candidates[0]}, expected (C,D,H,W) or (D,H,W)")

    # 当前 Task530 模型以单通道训练，这里仅保留第一个通道，避免 conv3d 输入通道数不匹配
    data = data[:1]

    # 2) 从 gt_segmentations 中读取 NIfTI 标签
    import nibabel as nib

    gt_dir = os.path.join(dataset_directory, "gt_segmentations")
    if not os.path.isdir(gt_dir):
        raise FileNotFoundError(f"gt_segmentations folder not found at {gt_dir}")

    gt_path = None
    for ext in (".nii.gz", ".nii"):
        p = os.path.join(gt_dir, f"{case_id}{ext}")
        if os.path.isfile(p):
            gt_path = p
            break
    if gt_path is None:
        raise FileNotFoundError(f"Could not find GT NIfTI for case {case_id} in {gt_dir}")

    gt_img = nib.load(gt_path)
    gt_arr = gt_img.get_fdata()
    gt = gt_arr.astype(np.int16)
    if gt.ndim == 4 and gt.shape[-1] == 1:
        gt = gt[..., 0]

    # 对齐 data 与 gt 的空间维度（简单裁剪到最小形状）
    if gt.shape != data.shape[1:]:
        min_shape = tuple(min(g, d) for g, d in zip(gt.shape, data.shape[1:]))
        gt = gt[:min_shape[0], :min_shape[1], :min_shape[2]]
        data = data[:, :min_shape[0], :min_shape[1], :min_shape[2]]

    seg = gt[None].astype(np.uint8)
    return data, seg


def _morphological_edge(label_3d: np.ndarray) -> np.ndarray:
    """根据 3D 标签生成简单的边界图 (二值)。"""
    from scipy.ndimage import binary_dilation, binary_erosion

    if label_3d.ndim == 4 and label_3d.shape[0] == 1:
        label_3d = label_3d[0]
    # 转为二值前景
    fg = label_3d > 0
    dil = binary_dilation(fg)
    ero = binary_erosion(fg)
    edge = np.logical_xor(dil, ero)
    return edge.astype(np.uint8)


def run_inference_on_case(
    trainer: nnUNetTrainerV2_Double_CCA_UPSam_fd_loss_RWKV_MedNeXt,
    case_id: str,
    data_root: str,
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """对单个病例前向推理，返回：
    - 主分割预测 (seg_main_pred)
    - 深监督各尺度预测列表 (ds_preds)
    - 边界预测 edge_f0
    - GT 标签 (gt)
    - 原始预处理图像 (image)
    """
    # 直接从用户指定的预处理目录中读取数据 + 从 gt_segmentations 读取 GT
    data_np, seg_np = _extract_case_data(data_root, case_id, trainer.dataset_directory)

    # 转为 torch tensor 并增加 batch 维度: [1, C, D, H, W]
    data_t = maybe_to_torch(data_np[None])
    if torch.cuda.is_available():
        data_t = data_t.cuda()

    net = trainer.network
    net.eval()

    # 使用底层 net.net 获取原始结构输出 (seg_outputs, edge_f0, edge_f1)
    with torch.no_grad():
        out = net.net(data_t)

    seg_outputs, edge_logit_f0, edge_logit_f1 = out

    # 处理深监督 seg 输出
    if isinstance(seg_outputs, (list, tuple)):
        seg_logits_list = [s for s in seg_outputs]
    else:
        seg_logits_list = [seg_outputs]

    # 主输出
    main_logits = seg_logits_list[0]  # [1, C, D, H, W]
    main_probs = softmax_helper(main_logits)
    main_pred = main_probs.argmax(1)[0].cpu().numpy().astype(np.uint8)  # [D, H, W]

    # 深监督其他尺度预测（若存在）
    ds_preds: List[np.ndarray] = []
    for lvl_logits in seg_logits_list[1:]:
        probs = softmax_helper(lvl_logits)
        pred = probs.argmax(1)[0].cpu().numpy().astype(np.uint8)
        ds_preds.append(pred)

    # 边界预测 f0
    edge_prob_f0 = torch.sigmoid(edge_logit_f0)[0, 0].cpu().numpy()  # [D, H, W]

    # GT 标签
    gt = seg_np[0].astype(np.uint8)  # [D, H, W]

    return main_pred, ds_preds, edge_prob_f0, gt, data_np


def save_visualizations(
    output_dir: str,
    case_id: str,
    image: np.ndarray,
    seg_pred: np.ndarray,
    ds_preds: List[np.ndarray],
    edge_pred: np.ndarray,
    gt: np.ndarray,
):
    """保存原图 / 分割预测 / GT / 边界预测 / GT 边界 / 深监督预测切片图。"""
    os.makedirs(output_dir, exist_ok=True)

    # 选中间一张切片进行可视化
    z = image.shape[1] // 2  # image: [C, D, H, W]
    img_slice = image[0, z]
    seg_slice = seg_pred[z]
    edge_slice = edge_pred[z]
    gt_slice = gt[z]
    gt_edge_slice = _morphological_edge(gt)[z]

    # 1) 原图 + 主分割 + GT + GT 边界 + 预测边界
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    axes[0, 0].imshow(img_slice, cmap="gray")
    axes[0, 0].set_title("Input")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img_slice, cmap="gray")
    axes[0, 1].imshow(seg_slice, alpha=0.5)
    axes[0, 1].set_title("Segmentation Pred")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(img_slice, cmap="gray")
    axes[0, 2].imshow(gt_slice, alpha=0.5)
    axes[0, 2].set_title("GT Segmentation")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(img_slice, cmap="gray")
    im0 = axes[1, 0].imshow(edge_slice, cmap="jet", alpha=0.5)
    axes[1, 0].set_title("Pred Edge (f0)")
    axes[1, 0].axis("off")
    fig.colorbar(im0, ax=axes[1, 0], fraction=0.046, pad=0.04)

    axes[1, 1].imshow(img_slice, cmap="gray")
    axes[1, 1].imshow(gt_edge_slice, cmap="jet", alpha=0.5)
    axes[1, 1].set_title("GT Edge")
    axes[1, 1].axis("off")

    axes[1, 2].axis("off")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{case_id}_main_and_edge.png"), dpi=300)
    plt.close(fig)

    # 2) 深监督各尺度预测
    if ds_preds:
        cols = min(3, len(ds_preds))
        rows = int(np.ceil(len(ds_preds) / cols))
        fig_ds, axes_ds = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        if not isinstance(axes_ds, np.ndarray):
            axes_ds = np.array([[axes_ds]])
        axes_ds = axes_ds.reshape(rows, cols)

        for i, ds in enumerate(ds_preds):
            r = i // cols
            c = i % cols
            axes_ds[r, c].imshow(img_slice, cmap="gray")
            axes_ds[r, c].imshow(ds[z], alpha=0.5)
            axes_ds[r, c].set_title(f"DS level {i+1}")
            axes_ds[r, c].axis("off")

        # 关闭未使用的子图
        for j in range(len(ds_preds), rows * cols):
            r = j // cols
            c = j % cols
            axes_ds[r, c].axis("off")

        plt.tight_layout()
        fig_ds.savefig(os.path.join(output_dir, f"{case_id}_ds_preds.png"), dpi=300)
        plt.close(fig_ds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans_file", type=str, required=True, help="Path to your custom plans.pkl (e.g. nnUNetPlansv2.1_trgSp_1x1x1_rwkv_plans_3D.pkl)")
    parser.add_argument("--fold", type=int, default=0, help="Fold index")
    parser.add_argument("--output_folder", type=str, required=True, help="Trained output folder used during training (without /fold_X)")
    parser.add_argument("--case_id", type=str, required=True, help="Case id (e.g. 'ESO_TJ_60011222468')")
    parser.add_argument("--output_dir", type=str, default="fd_edge_vis", help="Directory to save visualizations")
    parser.add_argument("--data_root", type=str, required=True, help="Root folder of preprocessed data for this Task/stage (e.g. /home/.../Task530_.../nnUNetData_plans_v2.1_trgSp_1x1x1_stage0)")
    args = parser.parse_args()

    trainer = get_trainer(args.plans_file, args.fold, args.output_folder)

    seg_pred, ds_preds, edge_pred, gt, image = run_inference_on_case(trainer, args.case_id, args.data_root)

    save_visualizations(
        args.output_dir,
        args.case_id,
        image,
        seg_pred,
        ds_preds,
        edge_pred,
        gt,
    )


if __name__ == "__main__":
    main()

