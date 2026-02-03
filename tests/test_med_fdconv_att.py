import unittest
import torch
import torch.nn as nn

from nnunet_mednext.network_architecture.le_networks.Med_FDConv_Att import MedNeXt_FDConv_Att
from nnunet_mednext.network_architecture.le_networks.FDConv_3d import FDConv
from nnunet_mednext.network_architecture.mednextv1.blocks import MedNeXtBlock


class TestMedNeXt_FDConv_Att(unittest.TestCase):
    def _build_model(self):
        n_classes = 3
        model = MedNeXt_FDConv_Att(
            in_channels=1,
            n_channels=32,
            n_classes=n_classes,
            exp_r=[2, 2, 2, 2, 2, 2, 2, 2, 2],
            kernel_size=3,
            deep_supervision=False,
            do_res=True,
            do_res_up_down=True,
            block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2],
            dim='3d',
            grn=True
        )
        return model

    def test_fdconv_count(self):
        model = self._build_model()

        fdconv_count = 0
        conv3d_count = 0
        for m in model.modules():
            if isinstance(m, FDConv):
                fdconv_count += 1
            elif isinstance(m, nn.Conv3d):
                conv3d_count += 1

        print("\nFDConv 个数:", fdconv_count)
        print("Conv3d 个数:", conv3d_count)
        self.assertGreater(fdconv_count, 0)

    def test_second_mednextblock_conv1_replaced(self):
        model = self._build_model()

        total_second_blocks = 0
        replaced_cnt = 0

        for container in model.modules():
            if not isinstance(container, (nn.Sequential, nn.ModuleList, nn.Module)):
                continue

            blocks = []
            for name, child in container.named_children():
                if isinstance(child, MedNeXtBlock):
                    blocks.append((name, child))

            if len(blocks) < 2:
                continue

            total_second_blocks += 1
            _, second_block = blocks[1]
            conv1 = getattr(second_block, "conv1", None)

            if isinstance(conv1, FDConv):
                replaced_cnt += 1

        print("\n总共找到的第二个 MedNeXtBlock 数量:", total_second_blocks)
        print("其中被 FDConv 替换的数量:", replaced_cnt)
        self.assertEqual(replaced_cnt, total_second_blocks)

    def test_forward_pass(self):
        model = self._build_model()
        model.eval()

        x = torch.randn(1, 1, 32, 64, 64)
        with torch.no_grad():
            y = model(x)

        if isinstance(y, (list, tuple)):
            y0 = y[0]
        else:
            y0 = y

        print("\n输入 shape:", x.shape)
        print("输出 shape:", y0.shape)

        self.assertEqual(y0.shape[0], 1)
        self.assertEqual(y0.shape[1], 3)


if __name__ == "__main__":
    unittest.main()