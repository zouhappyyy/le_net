import os
import argparse
from typing import List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from nnunet_mednext.run.default_configuration import get_default_configuration
from nnunet_mednext.training.network_training.nnUNetTrainerV2_Double_CCA_UPSam_fd_loss_RWKV_MedNeXt import (
    nnUNetTrainerV2_Double_CCA_UPSam_fd_loss_RWKV_MedNeXt,
)
from nnunet_mednext.utilities.to_torch import maybe_to_torch
from nnunet_mednext.utilities.nd_softmax import softmax_helper


def get_trainer(network: str, network_trainer: str, task: str, fold: int, plans_identifier: str) -> nnUNetTrainerV2_Double_CCA_UPSam_fd_loss_RWKV_MedNeXt:
    """Initialize trainer using nnUNet's default configuration and load checkpoint.

    This mirrors nnunet_mednext.run.run_training: we first resolve plans_file,
    output_folder, dataset_directory, batch_dice and stage via
    get_default_configuration, then instantiate our custom trainer.
    """
    # resolve TaskXXX name if task is an int
    from nnunet_mednext.utilities.task_name_id_conversion import convert_id_to_task_name
    from nnunet_mednext.paths import default_plans_identifier

    if not task.startswith("Task"):
        task_id = int(task)
        task = convert_id_to_task_name(task_id)

    if not plans_identifier:
        plans_identifier = default_plans_identifier

    plans_file, output_folder_name, dataset_directory, batch_dice, stage, _ = get_default_configuration(
        network, task, network_trainer, plans_identifier
    )

    trainer = nnUNetTrainerV2_Double_CCA_UPSam_fd_loss_RWKV_MedNeXt(
        plans_file,
        fold,
        output_folder=output_folder_name,
        dataset_directory=dataset_directory,
        batch_dice=batch_dice,
        stage=stage,
        unpack_data=False,
        deterministic=False,
        fp16=False,
    )
    trainer.initialize(training=False)
    # 尝试优先加载 best checkpoint
    try:
        trainer.load_best_checkpoint(train=False)
    except Exception:
        trainer.load_final_checkpoint(train=False)
    trainer.network.eval()
    return trainer


def _extract_case_data(trainer, case_id: str) -> Tuple[np.ndarray, np.ndarray]:
    """从 trainer 的数据集字典中获取某个样本的图像和标签 (预处理后的 numpy)。

    Assumes entries follow nnUNet's standard dataset dict structure with
    'data_file' and 'seg_file' keys pointing to .npz files containing 'data'.
    """
    dataset = trainer.dataset
    if case_id not in dataset:
        raise KeyError(f"Case id {case_id} not found in trainer.dataset. Available keys: {list(dataset.keys())[:5]} ...")
    entry = dataset[case_id]
    data_npz = np.load(entry['data_file'])
    seg_npz = np.load(entry['seg_file'])
    data = data_npz['data']  # [C, D, H, W]
    seg = seg_npz['data']    # [1, D, H, W]
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
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, np.ndarray]:
    """对单个病例前向推理，返回：
    - 主分割预测 (seg_main_pred)
    - 深监督各尺度预测列表 (ds_preds)
    - 边界预测 edge_f0
    - GT 标签 (gt)
    """
    # 直接从 dataset 中取出预处理数据
    data_np, seg_np = _extract_case_data(trainer, case_id)

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
    parser.add_argument("--network", type=str, required=True, help="nnUNet network (e.g. 3d_fullres)")
    parser.add_argument("--network_trainer", type=str, required=True, help="Trainer class name (e.g. nnUNetTrainerV2_Double_CCA_UPSam_fd_loss_RWKV_MedNeXt)")
    parser.add_argument("--task", type=str, required=True, help="Task name or id (e.g. 505 or Task505)")
    parser.add_argument("--fold", type=int, default=0, help="Fold index")
    parser.add_argument("--plans_identifier", type=str, default="nnUNetPlansv2.1_trgSp_1x1x1_rwkv", help="Plans identifier")
    parser.add_argument("--case_id", type=str, required=True, help="Case id key in trainer.dataset (e.g. 'ESO_TJ_...')")
    parser.add_argument("--output_dir", type=str, default="fd_edge_vis", help="Directory to save visualizations")
    args = parser.parse_args()

    trainer = get_trainer(args.network, args.network_trainer, args.task, args.fold, args.plans_identifier)

    seg_pred, ds_preds, edge_pred, gt, image = run_inference_on_case(trainer, args.case_id)

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

