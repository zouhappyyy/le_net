
from nnunet_mednext.training.network_training.MedNeXt.nnUNetTrainerV2_MedNeXt import nnUNetTrainerV2_Optim_and_LR
from nnunet_mednext.network_architecture.le_networks.MyMedNext import MyMedNext as MyMedNext_Orig
import torch
import torch.nn as nn
from nnunet_mednext.network_architecture.neural_network import SegmentationNetwork
from nnunet_mednext.utilities.nd_softmax import softmax_helper


class MyMedNext(MyMedNext_Orig, SegmentationNetwork):
    def __init__(self, *args, **kwargs):
        """
        参数签名保持和 MyMedNext_Orig 一致:
        in_channels, n_channels, n_classes, exp_r, kernel_size, deep_supervision, ...
        """
        super().__init__(*args, **kwargs)
        # nnUNet 评估/推理所需的接口
        self.conv_op = nn.Conv3d
        self.inference_apply_nonlin = softmax_helper
        self.input_shape_must_be_divisible_by = 2**5
        # 这里和 MedNeXt 包装类一致，从 kwargs 拿 n_classes
        self.num_classes = kwargs["n_classes"]



class nnUNetTrainerV2_MyMedNext(nnUNetTrainerV2_Optim_and_LR):
    """
    使用自定义 MyMedNext 的 trainer，参数配置基本沿用 MedNeXt_S_kernel3。
    """
    def initialize_network(self):
        self.network = MyMedNext(
            in_channels=self.num_input_channels,
            n_channels=16,
            n_classes=self.num_classes,
            exp_r=2,                 # Expansion ratio
            kernel_size=3,           # 与 MedNeXt_S_kernel3 一致
            deep_supervision=True,
            do_res=True,
            do_res_up_down=True,
            block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2],
        )
        self.batch_size = 1
        self.max_epochs = 300

        if torch.cuda.is_available():
            self.network.cuda()