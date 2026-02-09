import torch
import torch.nn as nn

from nnunet_mednext.network_architecture.le_networks.Double_RWKV_MedNeXt import (
    Double_RWKV_MedNeXt,
    Double_RWKV_MedNeXt_Encoder,
)


def _build_model(deep_supervision: bool = False):
    model = Double_RWKV_MedNeXt(
        in_channels=1,
        n_channels=16,
        n_classes=3,
        exp_r=2,
        kernel_size=3,
        deep_supervision=deep_supervision,
        do_res=True,
        do_res_up_down=True,
        block_counts=[1, 1, 1, 1, 1, 1, 1, 1, 1],
        dim="3d",
        grn=True,
        fusion_mode="concat",
    )
    return model


class TestDoubleRWKVMedNeXt:
    def test_forward_shape(self):
        model = _build_model(deep_supervision=False)
        x = torch.randn(1, 1, 32, 32, 32)
        with torch.no_grad():
            y = model(x)
        assert isinstance(y, torch.Tensor)
        assert y.shape[0] == 1
        assert y.shape[1] == 3

    def test_encoder_feature_shapes(self):
        enc = Double_RWKV_MedNeXt_Encoder(
            in_channels=1,
            n_channels=16,
            exp_r=2,
            kernel_size=3,
            deep_supervision=False,
            do_res=True,
            do_res_up_down=True,
            block_counts=[1, 1, 1, 1, 1, 1, 1, 1, 1],
            dim="3d",
            grn=True,
        )
        x = torch.randn(1, 1, 32, 32, 32)
        with torch.no_grad():
            feats = enc(x)
        assert isinstance(feats, list)
        assert len(feats) == 5
        # 通道是否按 [C, 2C, 4C, 8C, 16C]
        C = 16
        chs = [f.shape[1] for f in feats]
        assert chs[0] == C
        assert chs[1] == 2 * C
        assert chs[2] == 4 * C
        assert chs[3] == 8 * C
        assert chs[4] == 16 * C

    def test_no_crash_backward(self):
        model = _build_model(deep_supervision=False)
        x = torch.randn(1, 1, 16, 16, 16, requires_grad=True)
        y = model(x)
        loss = y.mean()
        loss.backward()
        # 确保梯度存在于至少一层参数
        grad_exists = any(p.grad is not None for p in model.parameters())
        assert grad_exists
