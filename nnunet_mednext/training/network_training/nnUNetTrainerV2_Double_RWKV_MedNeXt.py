import torch
import torch.nn as nn

from nnunet_mednext.training.network_training.MedNeXt.nnUNetTrainerV2_MedNeXt import (
    nnUNetTrainerV2_Optim_and_LR,
)
from nnunet_mednext.network_architecture.le_networks.Double_RWKV_MedNeXt import (
    Double_RWKV_MedNeXt as Double_RWKV_MedNeXt_Orig,
)
from nnunet_mednext.network_architecture.neural_network import SegmentationNetwork
from nnunet_mednext.utilities.nd_softmax import softmax_helper


class Double_RWKV_MedNeXt(Double_RWKV_MedNeXt_Orig, SegmentationNetwork):
    """Wrap Double_RWKV_MedNeXt to be compatible with nnUNet SegmentationNetwork API."""

    def __init__(self, *args, **kwargs):
        """Signature follows MedNeXt / MyMedNext: in_channels, n_channels, n_classes, ..."""
        super().__init__(*args, **kwargs)
        # nnUNet evaluation / inference interface
        self.conv_op = nn.Conv3d
        self.inference_apply_nonlin = softmax_helper
        self.input_shape_must_be_divisible_by = 2 ** 5
        self.num_classes = kwargs["n_classes"]


class nnUNetTrainerV2_Double_RWKV_MedNeXt(nnUNetTrainerV2_Optim_and_LR):
    """nnUNet Trainer using Double_RWKV_MedNeXt as the backbone.

    Training loop and loss definition are inherited from nnUNetTrainerV2_Optim_and_LR.
    Here we only override network construction and enforce fp32 (no fp16).
    """

    def __init__(
        self,
        plans_file,
        fold,
        output_folder=None,
        dataset_directory=None,
        batch_dice=True,
        stage=None,
        unpack_data=True,
        deterministic=True,
        fp16: bool = False,
        sample_by_frequency: bool = False,
    ):
        """Explicit signature so fp16 is only passed once and defaults to False.

        This mirrors the pattern used in nnUNetTrainerV2_Double_CCA_UPSam_fd_loss_RWKV_MedNeXt:
        we keep fp16 in the signature (for checkpoint compatibility) but default it to
        False so this trainer runs in fp32. We do not inject fp16 via **kwargs, which
        avoids '__init__() got multiple values for argument fp16'.
        """
        super().__init__(
            plans_file,
            fold,
            output_folder=output_folder,
            dataset_directory=dataset_directory,
            batch_dice=batch_dice,
            stage=stage,
            unpack_data=unpack_data,
            deterministic=deterministic,
            fp16=fp16,
            sample_by_frequency=sample_by_frequency,
        )

        # unify max epochs as in your other custom trainers
        self.max_epochs = 300
        if hasattr(self, "max_num_epochs"):
            self.max_num_epochs = 300

    def initialize_network(self):
        # Some parent trainers may define use_amp; be defensive
        use_amp = getattr(self, "use_amp", False)

        self.network = Double_RWKV_MedNeXt(
            in_channels=self.num_input_channels,
            n_channels=16,
            n_classes=self.num_classes,
            exp_r=2,
            kernel_size=3,
            deep_supervision=True,
            do_res=True,
            do_res_up_down=True,
            checkpoint_style="outside_block" if use_amp else None,
            block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2],
            norm_type="group",
            dim="3d",
            grn=True,
            fusion_mode="concat",
        )

        # You can adjust batch size depending on memory constraints
        self.batch_size = 4

        if torch.cuda.is_available():
            self.network.cuda()
