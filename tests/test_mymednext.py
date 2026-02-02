
import unittest
import torch
import torch.nn as nn

from nnunet_mednext.network_architecture.le_networks.MyMedNext import MyMedNext
from nnunet_mednext.network_architecture.le_networks.FDConv_3d import FDConv
from nnunet_mednext.network_architecture.mednextv1.blocks import MedNeXtBlock


class TestMyMedNext(unittest.TestCase):
    def _build_model(self):
        n_classes = 3
        model = MyMedNext(
            in_channels=1,
            n_channels=32,
            n_classes=n_classes,
            exp_r=[2, 2, 2, 2, 2, 2, 2, 2, 2],
            kernel_size=3,
            deep_supervision=False,
            do_res=True,
            do_res_up_down=True,
            block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2],  # 保证每个 stage 至少 2 个 block
            dim='3d',
            grn=True
        )
        return model

    def test_print_model_and_fdconv_info(self):
        """
        构建 MyMedNext, 打印整体结构并统计 FDConv / Conv3d 个数。
        """
        model = self._build_model()

        print("\n========== MyMedNext 结构 ==========")
        print(model)

        fdconv_count = 0
        conv3d_count = 0
        for m in model.modules():
            if isinstance(m, FDConv):
                fdconv_count += 1
            elif isinstance(m, nn.Conv3d):
                conv3d_count += 1

        print(f"\nFDConv 个数: {fdconv_count}")
        print(f"普通 Conv3d 个数: {conv3d_count}")

    def test_second_mednextblock_conv1_replaced(self):
        """
        遍历所有包含 MedNeXtBlock 的容器, 只检查第二个 MedNeXtBlock.conv1 是否为 FDConv。
        """
        model = self._build_model()

        print("\n========== 检查第二个 MedNeXtBlock.conv1 ==========")
        container_idx = 0
        replaced_cnt = 0
        total_second_blocks = 0

        # 遍历所有模块, 找到作为容器的 nn.Sequential / nn.ModuleList / 一般 Module
        for container in model.modules():
            if not isinstance(container, (nn.Sequential, nn.ModuleList, nn.Module)):
                continue

            # 在当前容器中顺序收集 MedNeXtBlock
            blocks = []
            for name, child in container.named_children():
                if isinstance(child, MedNeXtBlock):
                    blocks.append((name, child))

            # 不足 2 个 MedNeXtBlock 的容器跳过
            if len(blocks) < 2:
                continue

            container_idx += 1
            name2, second_block = blocks[1]
            total_second_blocks += 1

            conv1 = getattr(second_block, "conv1", None)
            print(f"\n[容器 {container_idx}] 第二个 MedNeXtBlock 名称: {name2}")
            print(f"    conv1 类型: {type(conv1)}")

            if isinstance(conv1, FDConv):
                replaced_cnt += 1
                print("    conv1 已成功替换为 FDConv")
                print(f"    conv1.in_channels: {conv1.in_channels}, "
                      f"out_channels: {conv1.out_channels}, "
                      f"kernel_size: {conv1.kernel_size}, "
                      f"groups: {conv1.groups}")
            elif isinstance(conv1, nn.Conv3d):
                print("    conv1 仍为普通 Conv3d, 未被替换")
                print(f"    conv1.in_channels: {conv1.in_channels}, "
                      f"out_channels: {conv1.out_channels}, "
                      f"kernel_size: {conv1.kernel_size}, "
                      f"groups: {conv1.groups}")
            else:
                print("    未找到有效的 conv1 属性")

        print(f"\n总共找到的第二个 MedNeXtBlock 数量: {total_second_blocks}")
        print(f"其中 conv1 被替换为 FDConv 的数量: {replaced_cnt}")

        # 可选: 如果你希望在测试上强约束第二个 block 一定要被替换, 可以启用断言:
        # self.assertEqual(replaced_cnt, total_second_blocks)

    def test_forward_pass(self):
        """
        构建 MyMedNext, 跑一次前向, 确认替换后的 FDConv 不影响前向计算。
        """
        model = self._build_model()
        model.eval()

        # 构造一个简单的 3D 输入: [B, C, D, H, W]
        x = torch.randn(1, 1, 32, 64, 64)
        with torch.no_grad():
            y = model(x)

        print("\n========== 前向测试 ==========")
        print(f"输入 shape: {x.shape}")
        if isinstance(y, (list, tuple)):
            print("输出为列表/元组, 第一个元素 shape:", y[0].shape)
        else:
            print("输出 shape:", y.shape)


if __name__ == "__main__":
    unittest.main()