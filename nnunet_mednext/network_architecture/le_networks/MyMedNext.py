# python
from nnunet_mednext.network_architecture.mednextv1.MedNextV1 import MedNeXt
import torch.nn as nn

class MyMedNext(MedNeXt):
    """
    基于原始 MedNextV1 的子类，便于按需修改网络结构。
    示例修改：替换 final conv 为新的输出通道数（使用传入的 n_classes）。
    """
    def __init__(self, in_channels=1, n_classes=2, base_num_features=None, **kwargs):
        # 如果原始类构造参数不完全相同，请根据原始定义调整参数传递
        super().__init__(in_channels=in_channels, n_classes=n_classes, **kwargs)
        # 示例：确保 final conv 输出通道等于 n_classes（如果原类字段名不同请改）
        if hasattr(self, "final_conv"):
            out_ch = n_classes
            in_ch = self.final_conv.in_channels
            self.final_conv = nn.Conv3d(in_ch, out_ch, kernel_size=1)
        # 你可以在这里插入/替换任意子模块以做修改