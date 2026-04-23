import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from nnunet_mednext.network_architecture.le_networks.FDConv_3d import FrequencyBandModulation3D


def _load_case_volume_from_npy(data_root: str, case_id: str) -> np.ndarray:
    """从预处理目录中加载指定 case 的 3D 体数据，返回 [C, D, H, W] 的 numpy 数组。

    优先查找 <case_id>.npy，如果不存在则查找以 case_id 开头的任意 .npy。
    """
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"data_root {data_root} does not exist")

    exact_npy = os.path.join(data_root, f"{case_id}.npy")
    if os.path.isfile(exact_npy):
        npy_path = exact_npy
    else:
        candidates = [
            os.path.join(data_root, f) for f in os.listdir(data_root)
            if f.startswith(case_id) and f.endswith(".npy")
        ]
        if not candidates:
            raise FileNotFoundError(
                f"No .npy file found for case {case_id} under {data_root}. "
                f"Expected {case_id}.npy or similar."
            )
        if len(candidates) > 1:
            print(f"[WARN] Multiple .npy files match case_id {case_id}: {[os.path.basename(f) for f in candidates]}. Using {os.path.basename(candidates[0])}")
        npy_path = candidates[0]

    arr = np.load(npy_path)
    if arr.ndim == 4:
        data = arr.astype(np.float32)
    elif arr.ndim == 3:
        data = arr[None].astype(np.float32)
    else:
        raise RuntimeError(f"Unexpected array shape {arr.shape} in {npy_path}, expected (C,D,H,W) or (D,H,W)")

    return data  # [C, D, H, W]


def visualize_fdconv_highfreq_3d(
    in_channels: int = 32,
    shape=(1, 32, 32, 128, 128),
    k_list=(2, 4, 8),
    lowfreq_att: bool = False,
    learnable_bands: bool = False,
    soft_band_beta: float = 30.0,
    device: str = None,
    save_path_prefix: str = "fdconv_highfreq_3d",
    # 新增：支持从真实数据加载
    data_root: str = None,
    case_id: str = None,
    channel: int = 0,
):
    """对 3D FBM 的高频分解进行可视化。

    默认使用随机输入；如果提供了 data_root 和 case_id，则从数据集中加载对应病例体数据。

    - 构造 FrequencyBandModulation3D
    - 用输入特征做一次前向，获取 high_acc（高频累积）
    - 从 high_acc 和 x 反推低频部分：low_final = x - high_acc
    - 2D: 按深度中间切片，将原始特征 / 低频 / 高频图保存出来
    - 3D: 对 Input/Low/High 体分别做三个方向的最大强度投影（axial/coronal/sagittal），
           得到 3x3 子图的“伪 3D”可视化
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if data_root is not None and case_id is not None:
        # 从预处理目录中加载真实病例数据
        vol = _load_case_volume_from_npy(data_root, case_id)  # [C, D, H, W]
        if channel < 0 or channel >= vol.shape[0]:
            raise ValueError(f"channel index {channel} out of range for volume with shape {vol.shape}")
        # 只取一个通道作为 FBM 输入
        vol_c = vol[channel:channel + 1]  # [1, D, H, W]
        C, D, H, W = vol_c.shape
        x = torch.from_numpy(vol_c[None]).to(device)  # [1, 1, D, H, W]
        # 覆盖 in_channels 与 shape
        in_channels = C
        shape = (1, C, D, H, W)
        print(f"Loaded case {case_id} from {data_root}, using volume shape {vol_c.shape} as FBM input")
    else:
        # 使用随机输入
        B, C, D, H, W = shape
        x = torch.randn(shape, dtype=torch.float32, device=device)

    fbm = FrequencyBandModulation3D(
        in_channels=in_channels,
        k_list=list(k_list),
        lowfreq_att=lowfreq_att,
        learnable_bands=learnable_bands,
        soft_band_beta=soft_band_beta,
        spatial_group=1,
        spatial_kernel=3,
    ).to(device)

    with torch.no_grad():
        # 直接利用已有接口：return_high=True 会返回 (out, high_acc)
        out, high_acc = fbm(x, att_feat=None, return_high=True)

    # 从 high_acc 反推“低频累积”：pre_x_final = x - high_acc
    low_final = x - high_acc

    # 聚合通道：这里简单取通道均值，得到 [B, D, H, W]
    x_mean = x.mean(dim=1)            # 原始
    low_mean = low_final.mean(dim=1)  # 低频
    high_mean = high_acc.mean(dim=1)  # 高频

    # -------- 2D: 中间切片可视化（保留原逻辑） --------
    B, _, D, H, W = x.shape
    mid = D // 2
    x_slice = x_mean[0, mid].detach().cpu()
    low_slice = low_mean[0, mid].detach().cpu()
    high_slice = high_mean[0, mid].detach().cpu()

    def _norm_img(t: torch.Tensor) -> torch.Tensor:
        t_min, t_max = float(t.min()), float(t.max())
        if t_max > t_min:
            t = (t - t_min) / (t_max - t_min)
        else:
            t = t * 0.0
        return t

    x_slice = _norm_img(x_slice)
    low_slice = _norm_img(low_slice)
    high_slice = _norm_img(high_slice)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(x_slice.numpy(), cmap="gray")
    axes[0].set_title("Input (mean over C)")
    axes[0].axis("off")

    axes[1].imshow(low_slice.numpy(), cmap="gray")
    axes[1].set_title("Low-frequency (x - high_acc)")
    axes[1].axis("off")

    axes[2].imshow(high_slice.numpy(), cmap="gray")
    axes[2].set_title("High-frequency accum (high_acc)")
    axes[2].axis("off")

    plt.tight_layout()
    if data_root is not None and case_id is not None:
        png_path = f"{save_path_prefix}_{case_id}_slice.png"
    else:
        png_path = f"{save_path_prefix}_slice.png"
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"Saved FDConv high-frequency 2D slice visualization to: {png_path}")

    # -------- 3D: 多视角最大强度投影 (MIP) 可视化 --------
    # 使用 B=1 的体：vol_in/vol_low/vol_high 形状为 [D, H, W]
    vol_in = x_mean[0].detach().cpu().numpy()
    vol_low = low_mean[0].detach().cpu().numpy()
    vol_high = high_mean[0].detach().cpu().numpy()

    def _mip_views(vol: np.ndarray):
        """计算 3 个方向的 MIP：axial(Z), coronal(Y), sagittal(X)。"""
        # vol: [D, H, W]
        axial = vol.max(axis=0)    # [H, W]
        coronal = vol.max(axis=1)  # [D, W]
        sagittal = vol.max(axis=2) # [D, H]
        return axial, coronal, sagittal

    def _norm_np(arr: np.ndarray) -> np.ndarray:
        arr = arr.astype(np.float32)
        v_min, v_max = float(arr.min()), float(arr.max())
        if v_max > v_min:
            arr = (arr - v_min) / (v_max - v_min)
        else:
            arr = np.zeros_like(arr)
        return arr

    in_ax, in_cor, in_sag = _mip_views(vol_in)
    low_ax, low_cor, low_sag = _mip_views(vol_low)
    high_ax, high_cor, high_sag = _mip_views(vol_high)

    in_ax, in_cor, in_sag = map(_norm_np, (in_ax, in_cor, in_sag))
    low_ax, low_cor, low_sag = map(_norm_np, (low_ax, low_cor, low_sag))
    high_ax, high_cor, high_sag = map(_norm_np, (high_ax, high_cor, high_sag))

    fig3d, axes3d = plt.subplots(3, 3, figsize=(12, 12))
    # 第 1 行：Input
    axes3d[0, 0].imshow(in_ax, cmap="gray");     axes3d[0, 0].set_title("Input - axial")
    axes3d[0, 1].imshow(in_cor, cmap="gray");    axes3d[0, 1].set_title("Input - coronal")
    axes3d[0, 2].imshow(in_sag, cmap="gray");    axes3d[0, 2].set_title("Input - sagittal")
    # 第 2 行：Low
    axes3d[1, 0].imshow(low_ax, cmap="gray");    axes3d[1, 0].set_title("Low - axial")
    axes3d[1, 1].imshow(low_cor, cmap="gray");   axes3d[1, 1].set_title("Low - coronal")
    axes3d[1, 2].imshow(low_sag, cmap="gray");   axes3d[1, 2].set_title("Low - sagittal")
    # 第 3 行：High
    axes3d[2, 0].imshow(high_ax, cmap="gray");   axes3d[2, 0].set_title("High - axial")
    axes3d[2, 1].imshow(high_cor, cmap="gray");  axes3d[2, 1].set_title("High - coronal")
    axes3d[2, 2].imshow(high_sag, cmap="gray");  axes3d[2, 2].set_title("High - sagittal")

    for ax in axes3d.ravel():
        ax.axis("off")

    plt.tight_layout()
    if data_root is not None and case_id is not None:
        png_path_3d = f"{save_path_prefix}_{case_id}_3d_mip.png"
    else:
        png_path_3d = f"{save_path_prefix}_3d_mip.png"
    fig3d.savefig(png_path_3d, dpi=200)
    plt.close(fig3d)
    print(f"Saved FDConv high-frequency 3D MIP visualization to: {png_path_3d}")

    # -------- Per-band 可视化（仅在从真实数据加载时生成） --------
    if data_root is not None and case_id is not None:
        try:
            # 使用 FBM 内部方法计算每个频带的掩码，并重构每个频带的高频残差
            with torch.no_grad():
                b, c, d, h, w = x.shape
                x_fft = torch.fft.rfftn(x, s=(d, h, w), dim=(-3, -2, -1), norm='ortho')
                # 获取 band masks，注意 FBM 接口需要 freq_w = w//2+1
                band_masks = fbm._get_band_masks(d, h, w // 2 + 1)

                pre_x = x.clone()
                per_band_raw = []
                per_band_weighted = []
                att_feat = x  # 与 FBM.forward 一致的默认 att_feat
                for idx in range(len(band_masks)):
                    mask = band_masks[idx].to(x.device)
                    low_part = torch.fft.irfftn(x_fft * mask, s=(d, h, w), dim=(-3, -2, -1), norm='ortho')
                    high_part = pre_x - low_part
                    pre_x = low_part
                    per_band_raw.append(high_part.detach().cpu().numpy())

                    # 计算 FBM 中用于加权的 freq_weight -> activated
                    if idx < len(fbm.freq_weight_conv_list):
                        freq_weight = fbm.freq_weight_conv_list[idx](att_feat)
                        freq_weight = fbm._activate(freq_weight)
                        # 重建与 forward 中相同的按组加权
                        group = fbm.spatial_group
                        tmp = freq_weight.reshape(b, group, -1, d, h, w) * high_part.reshape(b, group, -1, d, h, w)
                        band_weighted = tmp.reshape(b, -1, d, h, w)
                        per_band_weighted.append(band_weighted.detach().cpu().numpy())
                    else:
                        per_band_weighted.append(high_part.detach().cpu().numpy())

                # 最后 pre_x 为低频累积
                low_final_full = pre_x.detach().cpu().numpy()

            # 保存每个频带的 slice 与 3D MIP
            for i, (raw_np, w_np) in enumerate(zip(per_band_raw, per_band_weighted), start=1):
                # raw_np shape: [B, Cband, D, H, W]
                raw_mean = raw_np.mean(axis=1)[0]  # [D,H,W]
                w_mean = w_np.mean(axis=1)[0]

                # 中间切片
                mid = raw_mean.shape[0] // 2
                raw_slice = raw_mean[mid]
                w_slice = w_mean[mid]
                # normalize for display but keep comparable ranges
                def _norm_for_display(arr):
                    a = arr.astype('float32')
                    vmin, vmax = float(a.min()), float(a.max())
                    if vmax > vmin:
                        return (a - vmin) / (vmax - vmin)
                    return a * 0.0
                raw_slice_n = _norm_for_display(raw_slice)
                w_slice_n = _norm_for_display(w_slice)

                # difference and ratio maps
                eps = 1e-8
                diff_slice = w_slice - raw_slice
                # symmetric normalization for diff
                max_abs = max(abs(diff_slice).max(), 1e-8)
                diff_slice_n = diff_slice / max_abs
                ratio_slice = w_slice / (raw_slice + eps)
                # clip ratio for display to [0, RMAX]
                RMAX = 5.0
                ratio_slice_clipped = np.clip(ratio_slice, 0.0, RMAX) / RMAX

                # histogram data
                raw_vals = raw_slice.flatten()
                w_vals = w_slice.flatten()

                figb, axb = plt.subplots(2, 2, figsize=(10, 8))
                axb[0,0].imshow(raw_slice_n, cmap='gray'); axb[0,0].set_title(f'Band {i} raw slice'); axb[0,0].axis('off')
                axb[0,1].imshow(w_slice_n, cmap='gray'); axb[0,1].set_title(f'Band {i} weighted slice'); axb[0,1].axis('off')
                im = axb[1,0].imshow(diff_slice_n, cmap='RdBu', vmin=-1, vmax=1); axb[1,0].set_title('Weighted - Raw (normalized)'); axb[1,0].axis('off')
                axb[1,1].hist([raw_vals, w_vals], bins=50, label=['raw','weighted'], color=['0.3','0.7']); axb[1,1].set_title('Intensity hist'); axb[1,1].legend()
                figb.colorbar(im, ax=axb[1,0], fraction=0.046, pad=0.04)
                png_band_slice = f"{save_path_prefix}_{case_id}_band{i}_comparison_slice.png"
                figb.savefig(png_band_slice, dpi=200)
                plt.close(figb)

                # 3D MIP 三视图
                def _mip_views_np(vol):
                    axial = vol.max(axis=0)
                    coronal = vol.max(axis=1)
                    sagittal = vol.max(axis=2)
                    return axial, coronal, sagittal

                raw_ax, raw_cor, raw_sag = _mip_views_np(raw_mean)
                w_ax, w_cor, w_sag = _mip_views_np(w_mean)

                # 归一化
                def _norm_arr(a):
                    a = a.astype(np.float32)
                    vmin, vmax = float(a.min()), float(a.max())
                    if vmax > vmin:
                        return (a - vmin) / (vmax - vmin)
                    return np.zeros_like(a)

                raw_ax, raw_cor, raw_sag = map(_norm_arr, (raw_ax, raw_cor, raw_sag))
                w_ax, w_cor, w_sag = map(_norm_arr, (w_ax, w_cor, w_sag))

                # compute diff and ratio MIPs for 3D
                diff_ax, diff_cor, diff_sag = _mip_views_np(w_mean - raw_mean)
                # normalize diffs symmetrically
                maxabs = max(abs(diff_ax).max(), abs(diff_cor).max(), abs(diff_sag).max(), 1e-8)
                diff_ax_n = diff_ax / maxabs
                diff_cor_n = diff_cor / maxabs
                diff_sag_n = diff_sag / maxabs

                # ratio maps (clipped)
                def _ratio_map(a, b, rmax=5.0):
                    r = b / (a + 1e-8)
                    return np.clip(r, 0.0, rmax) / rmax

                raw_ax_n, raw_cor_n, raw_sag_n = raw_ax, raw_cor, raw_sag
                w_ax_n, w_cor_n, w_sag_n = w_ax, w_cor, w_sag
                ratio_ax = _ratio_map(raw_ax_n, w_ax_n)
                ratio_cor = _ratio_map(raw_cor_n, w_cor_n)
                ratio_sag = _ratio_map(raw_sag_n, w_sag_n)

                fig3, axes3 = plt.subplots(3, 4, figsize=(16, 12))
                # row0 raw, row1 weighted, row2 diff, extra col ratios
                axes3[0, 0].imshow(raw_ax_n, cmap='gray'); axes3[0, 0].set_title(f'Band {i} raw - axial'); axes3[0, 0].axis('off')
                axes3[0, 1].imshow(raw_cor_n, cmap='gray'); axes3[0, 1].set_title('raw - coronal'); axes3[0, 1].axis('off')
                axes3[0, 2].imshow(raw_sag_n, cmap='gray'); axes3[0, 2].set_title('raw - sagittal'); axes3[0, 2].axis('off')
                axes3[0, 3].axis('off')

                axes3[1, 0].imshow(w_ax_n, cmap='gray'); axes3[1, 0].set_title('weighted - axial'); axes3[1, 0].axis('off')
                axes3[1, 1].imshow(w_cor_n, cmap='gray'); axes3[1, 1].set_title('weighted - coronal'); axes3[1, 1].axis('off')
                axes3[1, 2].imshow(w_sag_n, cmap='gray'); axes3[1, 2].set_title('weighted - sagittal'); axes3[1, 2].axis('off')
                axes3[1, 3].axis('off')

                im0 = axes3[2, 0].imshow(diff_ax_n, cmap='RdBu', vmin=-1, vmax=1); axes3[2, 0].set_title('diff - axial'); axes3[2, 0].axis('off')
                axes3[2, 1].imshow(diff_cor_n, cmap='RdBu', vmin=-1, vmax=1); axes3[2, 1].set_title('diff - coronal'); axes3[2, 1].axis('off')
                axes3[2, 2].imshow(diff_sag_n, cmap='RdBu', vmin=-1, vmax=1); axes3[2, 2].set_title('diff - sagittal'); axes3[2, 2].axis('off')
                # ratio maps in last col
                axes3[0, 3].imshow(ratio_ax, cmap='inferno'); axes3[0, 3].set_title('ratio axial (clipped)'); axes3[0, 3].axis('off')
                axes3[1, 3].imshow(ratio_cor, cmap='inferno'); axes3[1, 3].set_title('ratio coronal (clipped)'); axes3[1, 3].axis('off')
                axes3[2, 3].imshow(ratio_sag, cmap='inferno'); axes3[2, 3].set_title('ratio sagittal (clipped)'); axes3[2, 3].axis('off')

                fig3.colorbar(im0, ax=axes3[2,0], fraction=0.046, pad=0.04)
                plt.tight_layout()
                png_band_3d = f"{save_path_prefix}_{case_id}_band{i}_comparison_3d.png"
                fig3.savefig(png_band_3d, dpi=200)
                plt.close(fig3)
                print(f"Saved per-band comparison visuals for band {i}: {png_band_slice}, {png_band_3d}")

            # 保存低频体的 3D MIP（已通过 low_final 变量保存了中间切片，但这里保存完整体的 MIP）
            # low_final_full shape: [B, C, D, H, W] -> 聚合通道与 batch 后得到 [D, H, W]
            low_final_mean = low_final_full.mean(axis=1)[0]
            low_ax, low_cor, low_sag = (_norm_np(low_final_mean.max(axis=0)), _norm_np(low_final_mean.max(axis=1)), _norm_np(low_final_mean.max(axis=2)))
            fig_low, axes_low = plt.subplots(1, 3, figsize=(12, 4))
            axes_low[0].imshow(low_ax, cmap='gray'); axes_low[0].set_title('Low final - axial'); axes_low[0].axis('off')
            axes_low[1].imshow(low_cor, cmap='gray'); axes_low[1].set_title('Low final - coronal'); axes_low[1].axis('off')
            axes_low[2].imshow(low_sag, cmap='gray'); axes_low[2].set_title('Low final - sagittal'); axes_low[2].axis('off')
            png_low = f"{save_path_prefix}_{case_id}_low_final_3d_mip.png"
            fig_low.savefig(png_low, dpi=200)
            plt.close(fig_low)
            print(f"Saved low-frequency 3D MIP to: {png_low}")

        except Exception as e:
            print(f"Per-band visualization failed: {e}")


def _build_argparser():
    parser = argparse.ArgumentParser(description="Visualize 3D FDConv high-frequency decomposition")
    parser.add_argument("--in_channels", type=int, default=32, help="Number of input channels for FBM when using random input")
    parser.add_argument("--shape", type=int, nargs=5, metavar=("B", "C", "D", "H", "W"), default=[1, 32, 32, 128, 128], help="Input shape for random mode: B C D H W")
    parser.add_argument("--k_list", type=int, nargs="+", default=[2, 4, 8], help="k_list for FrequencyBandModulation3D")
    parser.add_argument("--lowfreq_att", action="store_true", help="Enable low frequency attention in FBM")
    parser.add_argument("--learnable_bands", action="store_true", help="Use learnable soft frequency band masks in FBM")
    parser.add_argument("--soft_band_beta", type=float, default=30.0, help="Soft-mask slope beta when --learnable_bands is enabled")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", None], help="Device to run on (cpu/cuda), default auto")
    parser.add_argument("--save_prefix", type=str, default="fdconv_highfreq_3d", help="Prefix of output png file")
    # 新增：从数据集加载时需要的参数
    parser.add_argument("--data_root", type=str, default=None, help="Root folder of preprocessed .npy volumes (e.g. nnUNetData_plans_v2.1_trgSp_1x1x1_stage0)")
    parser.add_argument("--case_id", type=str, default=None, help="Case id to visualize (e.g. ESO_TJ_60011222468)")
    parser.add_argument("--channel", type=int, default=0, help="Which channel from loaded volume to feed into FBM")
    return parser


if __name__ == "__main__":
    # 支持命令行参数：如果提供了 data_root 和 case_id，则使用真实数据；否则退回随机模式
    parser = _build_argparser()
    args = parser.parse_args()

    shape = tuple(args.shape)

    visualize_fdconv_highfreq_3d(
        in_channels=args.in_channels,
        shape=shape,
        k_list=tuple(args.k_list),
        lowfreq_att=args.lowfreq_att,
        learnable_bands=args.learnable_bands,
        soft_band_beta=args.soft_band_beta,
        device=None if args.device in (None, "None") else args.device,
        save_path_prefix=args.save_prefix,
        data_root=args.data_root,
        case_id=args.case_id,
        channel=args.channel,
    )

#
# python visualize_fdconv_highfreq.py \
#   --data_root /home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_preprocessed/Task530_EsoTJ_30pct/nnUNetData_plans_v2.1_trgSp_1x1x1_stage0 \
#   --case_id ESO_TJ_60011222468 \
#   --channel 0 \
#   --save_prefix fdconv_highfreq_real


# python visualize_fdconv_highfreq.py --k_list 2 4 8 --save_prefix fdconv_fixed
# python visualize_fdconv_highfreq.py --k_list 2 4 8 --learnable_bands --soft_band_beta 30 --save_prefix fdconv_learnable
