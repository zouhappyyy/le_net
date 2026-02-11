#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from torch import nn
import torch.nn.functional as F
from nnunet_mednext.network_architecture.le_networks.Double_CCA_UPSam_fd_loss_RWKV_MedNeXt import extract_edge_gt


class MultipleOutputLoss2(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(MultipleOutputLoss2, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, x, y):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        l = weights[0] * self.loss(x[0], y[0])
        for i in range(1, len(x)):
            if weights[i] != 0:
                l += weights[i] * self.loss(x[i], y[i])
        return l


class MultipleOutputWithEdgeLoss(nn.Module):
    """包装 (seg_outputs, edge_logit_f0, edge_logit_f1) 的多输出 loss。

    - seg_loss: 使用内部的 multiple-output loss 对 seg_outputs 与 y 做监督；
    - edge_loss_f0/f1: 使用 BCEWithLogits 对边界 logit 与从 y 提取的 edge_gt 做监督；
    - total_loss = seg_loss + w0 * edge_loss_f0 + w1 * edge_loss_f1。
    """

    def __init__(self, seg_loss: nn.Module, edge_weight_f0: float = 0.4, edge_weight_f1: float = 0.2):
        super().__init__()
        self.seg_loss = seg_loss
        self.edge_weight_f0 = edge_weight_f0
        self.edge_weight_f1 = edge_weight_f1

    def _compute_edge_loss_single(self, edge_logit, target):
        # target: [B,D,H,W] or [B,1,D,H,W] or [B,C,D,H,W]
        if target.dim() == 4:
            target = target.unsqueeze(1)
        elif target.dim() == 5 and target.size(1) != 1:
            # 多通道 / one-hot，交给 extract_edge_gt 内部处理
            pass
        edge_gt = extract_edge_gt(target)
        return F.binary_cross_entropy_with_logits(edge_logit, edge_gt)

    def forward(self, x, y):
        """x: (seg_outputs, edge_logit_f0, edge_logit_f1), y: list/tuple of targets"""
        assert isinstance(x, (tuple, list)), "x must be tuple/list of (seg_outputs, edge_f0, edge_f1)"
        assert len(x) == 3, "expected (seg_outputs, edge_logit_f0, edge_logit_f1)"

        seg_outputs, edge_logit_f0, edge_logit_f1 = x

        # seg_outputs 与 y 一起交给原来的 multiple-output loss
        seg_loss = self.seg_loss(seg_outputs, y)

        # y[0] 是最高分辨率 target，与 edge 分支共享
        target_main = y[0]
        edge_loss0 = self._compute_edge_loss_single(edge_logit_f0, target_main)
        edge_loss1 = self._compute_edge_loss_single(edge_logit_f1, target_main)

        return seg_loss + self.edge_weight_f0 * edge_loss0 + self.edge_weight_f1 * edge_loss1

