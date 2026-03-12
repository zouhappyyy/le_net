import os
import argparse
from typing import List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

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


def _center_crop_3d_slices(shape_dhw: Tuple[int, int, int], patch_size: Tuple[int, int, int]) -> Tuple[slice, slice, slice]:
    """Compute center crop slices for a 3D volume.

    shape_dhw: (D, H, W) of the source volume.
    patch_size: desired (pd, ph, pw). If any source dim < patch dim, we just
        use the full source dim for that axis (i.e. no padding, only crop when larger).
    Returns three slice objects (sd, sh, sw) that can be applied to [D,H,W].
    """
    d, h, w = shape_dhw
    pd, ph, pw = patch_size

    def _one_dim(center_len: int, target_len: int) -> slice:
        if center_len <= target_len:
            # smaller than or equal to patch: just take full range
            return slice(0, center_len)
        start = (center_len - target_len) // 2
        end = start + target_len
        return slice(start, end)

    sd = _one_dim(d, pd)
    sh = _one_dim(h, ph)
    sw = _one_dim(w, pw)
    return sd, sh, sw


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

    # === 关键：将 NIfTI 标签从 [X, Y, Z] 调整到与预处理 .npy 一致的 [Z, Y, X] 坐标系，并可选做左右翻转 ===
    if gt_arr.ndim != 3:
        raise RuntimeError(f"Unexpected GT ndim {gt_arr.ndim} for {gt_path}, expected 3D array")

    # 步骤 1：轴重排，假设 NIfTI 为 (X, Y, Z)，转为 (Z, Y, X)
    gt_arr = np.transpose(gt_arr, (2, 1, 0))

    # 步骤 2（可选）：如果发现仍然左右颠倒，可以取消下一行注释做左右翻转（axis=2 对应 X 方向）
    # gt_arr = np.flip(gt_arr, axis=2)

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
    patch_size: Optional[Tuple[int, int, int]] = (64, 64, 64),
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run inference on a single case.

    Returns (all with the *same* spatial size [D,H,W]):
      - main segmentation prediction (seg_main_pred)
      - list of deep supervision predictions (ds_preds)
      - edge prediction f0 (edge_f0)
      - edge prediction f1 (edge_f1)
      - GT label (gt)
      - image volume (image)
    """
    data_np, seg_np = _extract_case_data(data_root, case_id, trainer.dataset_directory)
    gt_full = seg_np[0].astype(np.uint8)

    # Optionally center-crop to a 3D patch (default 64^3)
    if patch_size is not None:
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        assert len(patch_size) == 3, "patch_size must be a tuple of three ints or an int"
        d, h, w = data_np.shape[1:]
        sd, sh, sw = _center_crop_3d_slices((d, h, w), patch_size)
        data_np = data_np[:, sd, sh, sw]
        gt = gt_full[sd, sh, sw]
    else:
        gt = gt_full

    # Convert to torch tensor and add batch dimension: [1, C, D, H, W]
    data_t = maybe_to_torch(data_np[None])
    if torch.cuda.is_available():
        data_t = data_t.cuda()

    net = trainer.network
    net.eval()
    with torch.no_grad():
        seg_outputs, edge_logit_f0, edge_logit_f1 = net.net(data_t)

    if isinstance(seg_outputs, (list, tuple)):
        seg_logits_list = [s for s in seg_outputs]
    else:
        seg_logits_list = [seg_outputs]

    main_logits = seg_logits_list[0]
    main_probs = softmax_helper(main_logits)
    main_pred = main_probs.argmax(1)[0].cpu().numpy().astype(np.uint8)

    ds_preds: List[np.ndarray] = []
    for lvl_logits in seg_logits_list[1:]:
        probs = softmax_helper(lvl_logits)
        pred = probs.argmax(1)[0].cpu().numpy().astype(np.uint8)
        ds_preds.append(pred)

    edge_prob_f0 = torch.sigmoid(edge_logit_f0)[0, 0].cpu().numpy().astype(np.float32)
    edge_prob_f1 = torch.sigmoid(edge_logit_f1)[0, 0].cpu().numpy().astype(np.float32)

    return main_pred, ds_preds, edge_prob_f0, edge_prob_f1, gt, data_np


def save_visualizations(
    output_dir: str,
    case_id: str,
    image: np.ndarray,
    seg_pred: np.ndarray,
    ds_preds: List[np.ndarray],
    edge_pred_f0: np.ndarray,
    edge_pred_f1: np.ndarray,
    gt: np.ndarray,
):
    """Save 2D slice visualizations for main seg, GT, and both edge supervision maps.

    Note: edge_pred_f0 and edge_pred_f1 may have different depth (D) from the
    main output due to downsampling. Here we choose a safe slice index for each.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 主输出使用的 z 索引（image/seg/gt 共用）
    z_main = image.shape[1] // 2  # image: [C, D, H, W]
    img_slice_main = image[0, z_main]
    seg_slice = seg_pred[z_main]
    gt_slice = gt[z_main]
    gt_edge_slice = _morphological_edge(gt)[z_main]

    # 为 f0/f1 选择各自合法的切片索引，避免由于下采样造成越界
    D_f0 = edge_pred_f0.shape[0]
    z_f0 = min(z_main, D_f0 - 1)
    edge_slice_f0 = edge_pred_f0[z_f0]

    D_f1, H_f1, W_f1 = edge_pred_f1.shape
    z_f1 = min(z_main, D_f1 - 1)
    edge_slice_f1 = edge_pred_f1[z_f1]

    # 为第二层边界监督构造与其分辨率匹配的下采样原图切片
    # image: [C, D, H, W] -> 下采样到 [C, D_f1, H_f1, W_f1]
    C, D, H, W = image.shape
    image_t = torch.from_numpy(image[None])  # [1, C, D, H, W]
    with torch.no_grad():
        image_down_t = F.interpolate(
            image_t,
            size=(D_f1, H_f1, W_f1),
            mode="trilinear",
            align_corners=False,
        )
    image_down = image_down_t[0].cpu().numpy()  # [C, D_f1, H_f1, W_f1]
    img_slice_f1 = image_down[0, z_f1]

    # 1) 原图 + 主分割 + GT + GT 边界 + 预测边界
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    axes[0, 0].imshow(img_slice_main, cmap="gray")
    axes[0, 0].set_title("Input")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(img_slice_main, cmap="gray")
    axes[0, 1].imshow(seg_slice, alpha=0.5)
    axes[0, 1].set_title("Segmentation Pred")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(img_slice_main, cmap="gray")
    axes[0, 2].imshow(gt_slice, alpha=0.5)
    axes[0, 2].set_title("GT Segmentation")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(img_slice_main, cmap="gray")
    im0 = axes[1, 0].imshow(edge_slice_f0, cmap="jet", alpha=0.5)
    axes[1, 0].set_title("Pred Edge f0")
    axes[1, 0].axis("off")
    fig.colorbar(im0, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # 第二层边界监督：在与其分辨率匹配的下采样原图上叠加
    axes[1, 1].imshow(img_slice_f1, cmap="gray")
    im1 = axes[1, 1].imshow(edge_slice_f1, cmap="jet", alpha=0.5)
    axes[1, 1].set_title("Pred Edge f1 (downsampled scale)")
    axes[1, 1].axis("off")
    fig.colorbar(im1, ax=axes[1, 1], fraction=0.046, pad=0.04)

    axes[1, 2].imshow(img_slice_main, cmap="gray")
    axes[1, 2].imshow(gt_edge_slice, cmap="jet", alpha=0.5)
    axes[1, 2].set_title("GT Edge")
    axes[1, 2].axis("off")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{case_id}_main_and_edges.png"), dpi=300)
    plt.close(fig)

    # 2) 深监督各尺度预测
    if ds_preds:
        cols = min(3, len(ds_preds))
        rows = int(np.ceil(len(ds_preds) / cols))
        fig_ds, axes_ds = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        if not isinstance(axes_ds, np.ndarray):
            axes_ds = np.array([[axes_ds]])
        axes_ds = axes_ds.reshape(rows, cols)

        # 使用主输出对应的原图切片作为背景
        img_slice_for_ds = img_slice_main

        for i, ds in enumerate(ds_preds):
            r = i // cols
            c = i % cols
            # 对每个 ds 使用它自身深度维度的中间切片，避免越界
            z_ds = ds.shape[0] // 2
            axes_ds[r, c].imshow(img_slice_for_ds, cmap="gray")
            axes_ds[r, c].imshow(ds[z_ds], alpha=0.5)
            axes_ds[r, c].set_title(f"DS level {i+1} (z={z_ds})")
            axes_ds[r, c].axis("off")

        # 关闭未使用的子图
        for j in range(len(ds_preds), rows * cols):
            r = j // cols
            c = j % cols
            axes_ds[r, c].axis("off")

        plt.tight_layout()
        fig_ds.savefig(os.path.join(output_dir, f"{case_id}_ds_preds.png"), dpi=300)
        plt.close(fig_ds)


def _compute_2d_fft_spectrum(feat_2d: np.ndarray) -> np.ndarray:
    """对 2D 特征做 FFT 并返回对数幅度谱 (归一化到 [0, 1])。

    feat_2d: [H, W]，可以是原图切片或高频特征切片。
    """
    # 防 NaN：转 float32，并去掉非常大的值
    x = feat_2d.astype(np.float32)
    # 减去均值可以让 DC 分量更有对比度
    x = x - x.mean()
    F = np.fft.fftshift(np.fft.fft2(x))
    mag = np.abs(F)
    log_mag = np.log1p(mag)
    # 归一化到 [0, 1]
    vmin, vmax = np.percentile(log_mag, [1, 99])
    log_mag = np.clip(log_mag, vmin, vmax)
    log_mag = (log_mag - vmin) / (vmax - vmin + 1e-8)
    return log_mag


def _compute_band_energy_3d(feat_3d: np.ndarray, num_bins: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """对 3D 特征做 3D FFT，并按频率半径划分若干频带统计能量。

    返回 (band_centers, energy_norm)，其中 energy_norm 已按总能量归一化。
    """
    x = feat_3d.astype(np.float32)
    x = x - x.mean()
    F = np.fft.fftn(x)
    mag2 = np.abs(F) ** 2  # 能量

    D, H, W = x.shape
    kz = np.fft.fftfreq(D)
    ky = np.fft.fftfreq(H)
    kx = np.fft.fftfreq(W)
    gz, gy, gx = np.meshgrid(kz, ky, kx, indexing="ij")
    radius = np.sqrt(gz ** 2 + gy ** 2 + gx ** 2)

    r_flat = radius.reshape(-1)
    e_flat = mag2.reshape(-1)

    r_max = r_flat.max() + 1e-8
    bins = np.linspace(0.0, r_max, num_bins + 1)
    energy = np.zeros(num_bins, dtype=np.float64)
    for i in range(num_bins):
        mask = (r_flat >= bins[i]) & (r_flat < bins[i + 1])
        if np.any(mask):
            energy[i] = e_flat[mask].sum()

    total = energy.sum() + 1e-8
    energy_norm = energy / total
    band_centers = 0.5 * (bins[:-1] + bins[1:])
    return band_centers, energy_norm


def _compute_band_energy_3d_with_volumes(
    feat_3d: np.ndarray,
    num_bins: int = 3,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Compute band energy and also return per-band spatial volumes.

    Returns:
        band_centers: [num_bins]
        energy_norm: [num_bins]
        band_volumes: list of [D,H,W] arrays, inverse-FFT of each band.
    """
    x = feat_3d.astype(np.float32)
    x = x - x.mean()
    F = np.fft.fftn(x)

    D, H, W = x.shape
    kz = np.fft.fftfreq(D)
    ky = np.fft.fftfreq(H)
    kx = np.fft.fftfreq(W)
    gz, gy, gx = np.meshgrid(kz, ky, kx, indexing="ij")
    radius = np.sqrt(gz ** 2 + gy ** 2 + gx ** 2)

    r_flat = radius.reshape(-1)
    r_max = r_flat.max() + 1e-8
    bins = np.linspace(0.0, r_max, num_bins + 1)

    energy = np.zeros(num_bins, dtype=np.float64)
    band_volumes: List[np.ndarray] = []

    for i in range(num_bins):
        mask = (radius >= bins[i]) & (radius < bins[i + 1])
        F_band = np.zeros_like(F, dtype=np.complex64)
        F_band[mask] = F[mask]
        x_band = np.fft.ifftn(F_band).real.astype(np.float32)
        band_volumes.append(x_band)
        energy[i] = (np.abs(F_band) ** 2).sum()

    total = energy.sum() + 1e-8
    energy_norm = energy / total
    band_centers = 0.5 * (bins[:-1] + bins[1:])
    return band_centers, energy_norm, band_volumes


def _fbm_like_decompose_3d(vol: np.ndarray, k_list: Tuple[int, ...] = (2, 4, 8)) -> Tuple[np.ndarray, np.ndarray]:
    """Replicate the core FBM frequency decomposition logic on a single-channel 3D volume.

    This mirrors FrequencyBandModulation3D.forward (rfftn + cached_masks + low/high split):
      - x_fft: rFFTN of the volume
      - for each k in k_list:
          mask: freq_dist < 0.5/k (low-frequency mask in rFFT grid)
          low_part  = irFFTN(x_fft * mask)
          high_part = pre_x - low_part
          pre_x     = low_part
          high_acc += high_part

    Args:
        vol: np.ndarray of shape [D, H, W]
        k_list: tuple of integers, same semantics as FBM.k_list

    Returns:
        low_cum: final low-frequency accumulation (pre_x), shape [D,H,W]
        high_acc: accumulated high-frequency residual, shape [D,H,W]
    """
    assert vol.ndim == 3, f"Expected 3D volume, got shape {vol.shape}"
    D, H, W = vol.shape
    x = vol.astype(np.float32)
    x = x - x.mean()

    # rFFTN over (D,H,W), last dim uses rfftfreq like FBM (use_rfft=True)
    x_fft = np.fft.rfftn(x, s=(D, H, W), axes=(0, 1, 2), norm='ortho')

    # Build frequency grid consistent with get_fft3freq(..., use_rfft=True)
    kz = np.fft.fftfreq(D)
    ky = np.fft.fftfreq(H)
    kx = np.fft.rfftfreq(W)
    gz, gy, gx = np.meshgrid(kz, ky, kx, indexing='ij')
    freq_dist = np.sqrt(gz ** 2 + gy ** 2 + gx ** 2)

    pre_x = x.copy()
    high_acc = np.zeros_like(x, dtype=np.float32)

    for k in k_list:
        # Same mask rule as FBM._precompute_masks: freq_dist < 0.5 / k
        mask = (freq_dist < (0.5 / k + 1e-8)).astype(np.float32)
        low_part = np.fft.irfftn(x_fft * mask, s=(D, H, W), axes=(0, 1, 2), norm='ortho').astype(np.float32)
        high_part = pre_x - low_part
        pre_x = low_part
        high_acc += high_part

    return pre_x, high_acc


def visualize_frequency_bands(
    output_dir: str,
    case_id: str,
    image: np.ndarray,
    num_bins: int = 3,
) -> None:
    """Explicitly visualize several low/mid/high frequency bands and cumulative high-frequency component.

    This uses the radius-based 3D FFT band decomposition implemented in
    `_compute_band_energy_3d_with_volumes`. For `num_bins=3`, we treat:

      - band 1: lowest frequency band
      - band 2: mid frequency band
      - band 3: highest frequency band

    We export:
      1) A figure with input slice + each individual band slice.
      2) A figure with input slice + cumulative high-frequency component
         (sum of mid and high bands), including an overlay mask.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Normalize to [D,H,W] single-channel volume
    if image.ndim == 4:
        # assume [C,D,H,W], take first channel
        img_vol = image[0].astype(np.float32)
    elif image.ndim == 3:
        img_vol = image.astype(np.float32)
    else:
        raise ValueError(f"image ndim must be 3 or 4, got {image.shape}")

    # Compute band volumes via 3D FFT radius bins
    bands, energy, band_volumes = _compute_band_energy_3d_with_volumes(img_vol, num_bins=num_bins)

    # Pick a representative slice in depth
    z = img_vol.shape[0] // 2

    def _norm_slice(sl: np.ndarray) -> np.ndarray:
        vmin, vmax = np.percentile(sl, [1, 99])
        sl = np.clip(sl, vmin, vmax)
        if vmax > vmin:
            sl = (sl - vmin) / (vmax - vmin)
        return sl

    img_slice_n = _norm_slice(img_vol[z])

    # 1) Individual band slices (low/mid/high)
    # figure: Input + num_bins bands
    fig, axes = plt.subplots(1, num_bins + 1, figsize=(4 * (num_bins + 1), 4))

    axes[0].imshow(img_slice_n, cmap="gray")
    axes[0].set_title(f"Input (z={z})")
    axes[0].axis("off")

    for i in range(num_bins):
        sl = band_volumes[i][z]
        sl_n = _norm_slice(sl)
        axes[i + 1].imshow(sl_n, cmap="gray")
        axes[i + 1].set_title(f"Band {i+1}")
        axes[i + 1].axis("off")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{case_id}_fd_bands_individual.png"), dpi=300)
    plt.close(fig)

    # 2) Cumulative high-frequency component and mask overlay
    # Treat highest bands as "high"; for num_bins=3, we use band2+band3.
    if num_bins >= 2:
        high_cum_vol = np.zeros_like(img_vol, dtype=np.float32)
        # accumulate from band index 1 to end (mid+high)
        for i in range(1, num_bins):
            high_cum_vol += band_volumes[i].astype(np.float32)

        high_slice = high_cum_vol[z]
        high_slice_n = _norm_slice(high_slice)

        # simple mask from high-frequency magnitude
        high_abs = np.abs(high_slice_n)
        thr = high_abs.mean() + high_abs.std()
        mask = (high_abs > thr).astype(np.float32)

        # figure: input / high_cum / input+mask
        fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4))

        axes2[0].imshow(img_slice_n, cmap="gray")
        axes2[0].set_title(f"Input (z={z})")
        axes2[0].axis("off")

        axes2[1].imshow(high_slice_n, cmap="gray")
        axes2[1].set_title("High-cum (bands 2..N)")
        axes2[1].axis("off")

        axes2[2].imshow(img_slice_n, cmap="gray")
        axes2[2].imshow(mask, cmap="jet", alpha=0.5)
        axes2[2].set_title("Input with high-cum mask")
        axes2[2].axis("off")

        plt.tight_layout()
        fig2.savefig(os.path.join(output_dir, f"{case_id}_fd_bands_high_cum.png"), dpi=300)
        plt.close(fig2)


def _load_npy_or_nii(path: str) -> np.ndarray:
    """根据扩展名加载 .npy 或 .nii(.gz) 文件，返回 numpy 数组。

    - .npy: 直接 np.load
    - .nii/.nii.gz: 使用 nibabel 加载并返回 get_fdata()
    """
    import nibabel as nib

    if path is None:
        raise ValueError("Path is None")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path)
    if ext in (".nii", ".gz") or path.endswith(".nii.gz"):
        img = nib.load(path)
        return img.get_fdata()
    raise ValueError(f"Unsupported file extension for {path}")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=False)

    # ---- 模式 1：跑模型 + 可视化（现有功能，保留） ----
    p_run = subparsers.add_parser("run_model", help="Run network inference and visualize seg/edge (original mode)")
    p_run.add_argument("--plans_file", type=str, required=True, help="Path to your custom plans.pkl (e.g. nnUNetPlansv2.1_trgSp_1x1x1_rwkv_plans_3D.pkl)")
    p_run.add_argument("--fold", type=int, default=0, help="Fold index")
    p_run.add_argument("--output_folder", type=str, required=True, help="Trained output folder used during training (without /fold_X)")
    p_run.add_argument("--case_id", type=str, required=True, help="Case id (e.g. 'ESO_TJ_60011222468')")
    p_run.add_argument("--output_dir", type=str, default="fd_edge_vis", help="Directory to save visualizations")
    p_run.add_argument("--data_root", type=str, required=True, help="Root folder of preprocessed data for this Task/stage (e.g. /home/.../Task530_.../nnUNetData_plans_v2.1_trgSp_1x1x1_stage0)")
    p_run.add_argument("--patch_size", type=int, default=64, help="Center-crop cubic patch size (D=H=W=patch_size). Set <=0 to disable patch cropping.")
    p_run.add_argument("--do_fd_vis", action="store_true", help="Also perform frequency-domain visualization on the same patch.")

    # ---- 模式 2：纯频域可视化，不跑模型 ----
    p_fd = subparsers.add_parser("fd_only", help="Visualize frequency decomposition from existing arrays, without running the model")
    p_fd.add_argument("--image", type=str, required=True, help="Path to image volume (.npy or .nii.gz), shape [C,D,H,W] or [D,H,W]")
    p_fd.add_argument("--seg", type=str, default=None, help="Optional segmentation label (.npy or .nii.gz), shape [D,H,W]")
    p_fd.add_argument("--edge", type=str, default=None, help="Optional edge/probability volume (.npy or .nii.gz), shape [D,H,W]")
    p_fd.add_argument("--case_id", type=str, default="case", help="Case id used in output file names")
    p_fd.add_argument("--output_dir", type=str, default="fd_vis_only", help="Directory to save frequency visualizations")

    # ---- 模式 3：仅基于 3D FFT 频带分解，导出各个频带及累加高频的可视化 ----
    p_fd_bands = subparsers.add_parser(
        "fd_bands",
        help="Visualize individual 3D FFT frequency bands and cumulative high-frequency component",
    )
    p_fd_bands.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to image volume (.npy or .nii.gz), shape [C,D,H,W] or [D,H,W]",
    )
    p_fd_bands.add_argument(
        "--case_id",
        type=str,
        default="case",
        help="Case id used in output file names",
    )
    p_fd_bands.add_argument(
        "--output_dir",
        type=str,
        default="fd_bands_vis",
        help="Directory to save band visualizations",
    )
    p_fd_bands.add_argument(
        "--num_bins",
        type=int,
        default=3,
        help="Number of radial frequency bands (default: 3)",
    )

    args = parser.parse_args()

    if args.mode == "fd_only":
        # 纯后处理频域可视化
        img_arr = _load_npy_or_nii(args.image)
        seg_arr = _load_npy_or_nii(args.seg) if args.seg is not None else None
        edge_arr = _load_npy_or_nii(args.edge) if args.edge is not None else None

        # 简单规范到 [C,D,H,W] 或 [D,H,W]
        if img_arr.ndim == 3:
            image = img_arr.astype(np.float32)
        elif img_arr.ndim == 4:
            image = img_arr.astype(np.float32)
        else:
            raise RuntimeError(f"Unsupported image shape {img_arr.shape}, expected 3D or 4D volume")

        if seg_arr is not None and seg_arr.ndim > 3:
            # 通常 NIfTI seg 为 [X,Y,Z]，这里不做轴重排，由用户保证一致性；仅去掉多余通道
            seg_arr = np.squeeze(seg_arr)
        if edge_arr is not None and edge_arr.ndim > 3:
            edge_arr = np.squeeze(edge_arr)

        visualize_frequency_from_arrays(args.output_dir, args.case_id, image, seg_arr, edge_arr)
        return

    if args.mode == "fd_bands":
        img_arr = _load_npy_or_nii(args.image)
        if img_arr.ndim == 3:
            image = img_arr.astype(np.float32)
        elif img_arr.ndim == 4:
            image = img_arr.astype(np.float32)
        else:
            raise RuntimeError(f"Unsupported image shape {img_arr.shape}, expected 3D or 4D volume")

        visualize_frequency_bands(args.output_dir, args.case_id, image, num_bins=args.num_bins)
        return

    # 默认或显式 run_model 模式：保持原逻辑
    if args.mode is None or args.mode == "run_model":
        trainer = get_trainer(args.plans_file, args.fold, args.output_folder)

        # Determine patch_size tuple or disable cropping
        if getattr(args, "patch_size", 0) is not None and args.patch_size > 0:
            patch_sz: Optional[Tuple[int, int, int]] = (args.patch_size, args.patch_size, args.patch_size)
        else:
            patch_sz = None

        seg_pred, ds_preds, edge_pred_f0, edge_pred_f1, gt, image = run_inference_on_case(
            trainer,
            args.case_id,
            args.data_root,
            patch_size=patch_sz,
        )

        # Spatial-domain visualizations (seg/edge/deep supervision)
        save_visualizations(
            args.output_dir,
            args.case_id,
            image,
            seg_pred,
            ds_preds,
            edge_pred_f0,
            edge_pred_f1,
            gt,
        )

        # Optional frequency-domain visualizations on the same (possibly cropped) volume
        if getattr(args, "do_fd_vis", False):
            # Use predicted segmentation as overlay; also pass edge prediction as optional edge map
            visualize_frequency_from_arrays(
                args.output_dir,
                args.case_id + "_patch" if patch_sz is not None else args.case_id,
                image,
                seg=seg_pred,
                edge_f0=edge_pred_f0,
                edge_f1=edge_pred_f1,
            )


if __name__ == "__main__":
    main()

