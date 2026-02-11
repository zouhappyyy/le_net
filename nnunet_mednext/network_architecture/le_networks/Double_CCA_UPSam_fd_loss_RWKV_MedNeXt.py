import torch
import torch.nn as nn
import torch.nn.functional as F

from .Double_CCA_UPSam_fd_RWKV_MedNeXt import Double_CCA_UPSam_fd_RWKV_MedNeXt
from .Double_RWKV_MedNeXt import Double_RWKV_MedNeXt_Encoder
from nnunet_mednext.network_architecture.le_networks.FDConv_3d import FDConv


class EdgeHead3D(nn.Module):
    """基于高频特征的 3D 边界预测头。

    输入: [B,C,D,H,W] 高频特征
    输出: [B,1,D,H,W] 边界 logit
    """

    def __init__(self, in_channels: int, mid_channels: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, 1, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def extract_edge_gt(mask: torch.Tensor) -> torch.Tensor:
    """从 3D mask 中提取边界 GT，返回 [B,1,D,H,W] 的 0/1 边界标签。"""
    if mask.size(1) > 1:
        mask = mask.argmax(dim=1, keepdim=True).float()
    else:
        mask = mask.float()

    dx = mask[:, :, 1:, :, :] - mask[:, :, :-1, :, :]
    dy = mask[:, :, :, 1:, :] - mask[:, :, :, :-1, :]
    dz = mask[:, :, :, :, 1:] - mask[:, :, :, :, :-1]

    pad_dx = F.pad(dx.abs(), (0, 0, 0, 0, 1, 0))
    pad_dy = F.pad(dy.abs(), (0, 0, 1, 0, 0, 0))
    pad_dz = F.pad(dz.abs(), (1, 0, 0, 0, 0, 0))

    grad_mag = pad_dx + pad_dy + pad_dz
    edge = (grad_mag > 0).float()
    return edge


class Double_CCA_UPSam_fd_loss_RWKV_MedNeXt(Double_CCA_UPSam_fd_RWKV_MedNeXt):
    """fd 版 Double_CCA_UPSam_RWKV_MedNeXt + 多层边界监督。

    - f0 尺度：使用 enc_block_0 中 FDConv.FBM 的高频残差 high_acc 做边界监督；
    - f1 尺度：再增加一层边界监督，使用 enc_block_1 中 FDConv.FBM 的高频残差；
    - forward 返回 (seg_outputs, edge_logit_f0, edge_logit_f1)。
    """

    def __init__(self, *args, edge_feat_stage: int = 0, **kwargs):
        super().__init__(*args, **kwargs)

        assert isinstance(self.encoder, Double_RWKV_MedNeXt_Encoder), "Encoder must be Double_RWKV_MedNeXt_Encoder"

        n_channels = kwargs.get("n_channels", None)

        # ---- f0 尺度边界 head ----
        sample_block0 = self.encoder.mednext_enc.enc_block_0[0]
        in_ch_edge0 = sample_block0.out_channels if hasattr(sample_block0, "out_channels") else None
        if in_ch_edge0 is None:
            in_ch_edge0 = n_channels if n_channels is not None else 16
        self.edge_head_f0 = EdgeHead3D(in_channels=in_ch_edge0)

        # ---- f1 尺度边界 head ----
        sample_block1 = self.encoder.mednext_enc.enc_block_1[0]
        in_ch_edge1 = sample_block1.out_channels if hasattr(sample_block1, "out_channels") else None
        if in_ch_edge1 is None:
            in_ch_edge1 = (n_channels * 2) if n_channels is not None else 32
        self.edge_head_f1 = EdgeHead3D(in_channels=in_ch_edge1)

        self.edge_feat_stage = edge_feat_stage

    def _get_high_freq_feat_from_block(self, block: nn.Module, fallback: torch.Tensor) -> torch.Tensor:
        """在给定 block 中查找带 last_high_feat 的 FDConv，返回其高频残差，否则使用 fallback。"""
        last_high = None
        for m in block.modules():
            if isinstance(m, FDConv) and getattr(m, "last_high_feat", None) is not None:
                last_high = m.last_high_feat
        if last_high is None:
            return fallback
        return last_high

    def _get_high_freq_feat_f0(self, f0: torch.Tensor) -> torch.Tensor:
        enc_block_0 = getattr(self.encoder.mednext_enc, "enc_block_0", None)
        if enc_block_0 is None:
            return f0
        return self._get_high_freq_feat_from_block(enc_block_0, f0)

    def _get_high_freq_feat_f1(self, f1: torch.Tensor) -> torch.Tensor:
        enc_block_1 = getattr(self.encoder.mednext_enc, "enc_block_1", None)
        if enc_block_1 is None:
            return f1
        return self._get_high_freq_feat_from_block(enc_block_1, f1)

    def forward(self, x: torch.Tensor, return_edge: bool = True):
        # encoder 输出 [f0, f1, f2, f3, f4]
        f0, f1, f2, f3, f4 = self.encoder(x)

        # f0 尺度高频残差
        edge_feat_f0 = self._get_high_freq_feat_f0(f0)
        edge_logit_f0 = self.edge_head_f0(edge_feat_f0)

        # f1 尺度高频残差
        edge_feat_f1 = self._get_high_freq_feat_f1(f1)
        edge_logit_f1 = self.edge_head_f1(edge_feat_f1)

        # decoder + segmentation head 与父类一致
        x_dec, ds_feats = self.decoder(f0, f1, f2, f3, f4)
        x_main = self.out_0(x_dec)

        if not self.do_ds:
            if return_edge:
                return x_main, edge_logit_f0, edge_logit_f1
            return x_main

        x_ds_0 = x_main
        x_ds_1 = self.out_1(ds_feats[1]) if self.out_1 is not None else None
        x_ds_2 = self.out_2(ds_feats[2]) if self.out_2 is not None else None
        x_ds_3 = self.out_3(ds_feats[3]) if self.out_3 is not None else None
        x_ds_4 = self.out_4(f4) if self.out_4 is not None else None

        seg_outputs = [x_ds_0, x_ds_1, x_ds_2, x_ds_3, x_ds_4]
        if return_edge:
            return seg_outputs, edge_logit_f0, edge_logit_f1
        return seg_outputs
