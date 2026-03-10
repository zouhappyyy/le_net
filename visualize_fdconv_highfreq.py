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
    - 按深度中间切片，将原始特征 / 低频 / 高频图保存出来
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
        spatial_group=1,
        spatial_kernel=3,
    ).to(device)

    with torch.no_grad():
        # 直接利用已有接口：return_high=True 会返回 (out, high_acc)
        out, high_acc = fbm(x, att_feat=None, return_high=True)

    # 从 high_acc 反推“低频累积”：pre_x_final = x - high_acc
    low_final = x - high_acc

    # 聚合通道：这里简单取通道均值，得到 [B, D, H, W]
    x_mean = x.mean(dim=1)         # 原始
    low_mean = low_final.mean(dim=1)  # 低频
    high_mean = high_acc.mean(dim=1)  # 高频

    # 选取中间深度切片
    B, _, D, H, W = x.shape
    mid = D // 2
    x_slice = x_mean[0, mid].detach().cpu()
    low_slice = low_mean[0, mid].detach().cpu()
    high_slice = high_mean[0, mid].detach().cpu()

    def _norm_img(t):
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
    # 如果有 case_id，则在文件名中带上，便于区分
    if data_root is not None and case_id is not None:
        png_path = f"{save_path_prefix}_{case_id}_slice.png"
    else:
        png_path = f"{save_path_prefix}_slice.png"
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"Saved FDConv high-frequency 3D visualization to: {png_path}")


def _build_argparser():
    parser = argparse.ArgumentParser(description="Visualize 3D FDConv high-frequency decomposition")
    parser.add_argument("--in_channels", type=int, default=32, help="Number of input channels for FBM when using random input")
    parser.add_argument("--shape", type=int, nargs=5, metavar=("B", "C", "D", "H", "W"), default=[1, 32, 32, 128, 128], help="Input shape for random mode: B C D H W")
    parser.add_argument("--k_list", type=int, nargs="+", default=[2, 4, 8], help="k_list for FrequencyBandModulation3D")
    parser.add_argument("--lowfreq_att", action="store_true", help="Enable low frequency attention in FBM")
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
        device=None if args.device in (None, "None") else args.device,
        save_path_prefix=args.save_prefix,
        data_root=args.data_root,
        case_id=args.case_id,
        channel=args.channel,
    )

