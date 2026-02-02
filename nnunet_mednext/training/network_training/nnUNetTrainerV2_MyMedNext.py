
from nnunet_mednext.training.network_training.MedNeXt.nnUNetTrainerV2_MedNeXt import nnUNetTrainerV2_Optim_and_LR
from nnunet_mednext.network_architecture.le_networks.MyMedNext import MyMedNext
import torch


class nnUNetTrainerV2_MyMedNext(nnUNetTrainerV2_Optim_and_LR):
    """
    使用自定义 MyMedNext 的 trainer，参数配置基本沿用 MedNeXt_S_kernel3。
    """
    def initialize_network(self):
        self.network = MyMedNext(
            in_channels=self.num_input_channels,
            n_channels=32,
            n_classes=self.num_classes,
            exp_r=2,                 # Expansion ratio
            kernel_size=3,           # 与 MedNeXt_S_kernel3 一致
            deep_supervision=True,
            do_res=True,
            do_res_up_down=True,
            block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2],
        )

        if torch.cuda.is_available():
            self.network.cuda()