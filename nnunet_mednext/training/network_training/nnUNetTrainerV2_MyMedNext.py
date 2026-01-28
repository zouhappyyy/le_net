# python
from nnunet_mednext.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet_mednext.network_architecture.le_networks.MyMedNext import MyMedNext
import torch

class nnUNetTrainerV2_MyMedNext(nnUNetTrainerV2):
    """
    将自定义网络接入 nnUNet 的 trainer。把网络赋值给 self.network。
    """
    def initialize_network(self):
        # 实例化自定义网络，使用 trainer 的通道/类别信息
        net = MyMedNext(
            in_channels=self.num_input_channels,
            n_classes=self.num_classes,
            # 如需传 base features 或其它参数，可从 trainer 配置传入
        )

        # 移动到 trainer 使用的设备（优先 GPU）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net.to(device)

        # 可选：如果想使用简单的 DataParallel（单机多卡），否则删除
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            net = torch.nn.DataParallel(net)

        self.network = net