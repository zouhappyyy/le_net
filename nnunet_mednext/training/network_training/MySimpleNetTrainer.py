# python
from nnunet_mednext.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet_mednext.network_architecture.le_networks.MyModel import MySimpleNet
import torch

class nnUNetTrainerV2_MySimpleNet(nnUNetTrainerV2):
    """
    最小 trainer：实现 initialize_network 来实例化上面的 MySimpleNet
    """
    def initialize_network(self):
        self.network = MySimpleNet(
            in_channels=self.num_input_channels,
            n_classes=self.num_classes,
            base_num_features=32
        )
        if torch.cuda.is_available():
            self.network.cuda()
