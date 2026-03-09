import torch
import matplotlib.pyplot as plt

from nnunet_mednext.network_architecture.le_networks.FDConv_3d import FrequencyBandModulation3D


def visualize_fdconv_highfreq_3d(
    in_channels: int = 32,
    shape=(1, 32, 32, 128, 128),
    k_list=(2, 4, 8),
    lowfreq_att: bool = False,
    device: str = None,
    save_path_prefix: str = "fdconv_highfreq_3d"
):
    """最小示例：对 3D FBM 的高频分解进行可视化（基于随机输入）。

    - 构造 FrequencyBandModulation3D
    - 用随机特征做一次前向，获取 high_acc（高频累积）和 pre_x/low 部分（从 high_acc 和 x 反推）
    - 按深度中间切片，将原始特征 / 低频 / 高频图保存出来
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

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
    png_path = f"{save_path_prefix}_slice.png"
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"Saved FDConv high-frequency 3D visualization to: {png_path}")


if __name__ == "__main__":
    # 默认跑一个 3D FBM 的随机可视化例子
    visualize_fdconv_highfreq_3d()

