import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank
from torch.utils.checkpoint import checkpoint
from mmcv.cnn import CONV_LAYERS
from torch import Tensor
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_
import time


class StarReLU(nn.Module):
    """StarReLU激活函数：增强对正特征的响应，提升特征表达能力
    公式：out = scale * (ReLU(x))² + bias
    核心逻辑：
        - ReLU保留有效正特征，平方项放大强响应特征
        - 可学习的scale和bias动态调整激活强度，适配不同任务
    参数：
        scale_value: 初始缩放因子（默认1.0）
        bias_value: 初始偏置（默认0.0）
        scale_learnable: 缩放因子是否可学习（默认True）
        bias_learnable: 偏置是否可学习（默认True）
        inplace: 是否原地操作（默认False，避免覆盖原始数据）
    输入：x (Tensor) - 任意形状特征张量
    输出：激活后特征张量（与输入形状一致）
    """

    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        # 定义可学习参数（或固定值）
        self.scale = nn.Parameter(scale_value * torch.ones(1), requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1), requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


class KernelSpatialModulation_Global(nn.Module):
    """全局核空间调制模块（KSM-Global）
    功能：基于全局特征统计，生成四类注意力权重，动态调整卷积核的通道、滤波器、空间位置及核选择
    核心逻辑：
        1. 全局平均池化提取特征统计，压缩空间冗余
        2. 1×1卷积+StarReLU构建轻量注意力生成器
        3. 针对卷积核不同维度（通道/滤波器/空间/核）生成适配权重
    参数：
        in_planes: 输入特征通道数
        out_planes: 输出特征通道数
        kernel_size: 卷积核尺寸（如3）
        groups: 分组卷积数（默认1，=in_planes时为深度可分离卷积）
        reduction: 通道压缩系数（默认0.0625=1/16，降低注意力计算量）
        kernel_num: 候选卷积核数量（默认4，通过注意力选择最优核）
        temp: 注意力温度系数（控制权重分布尖锐程度，默认1.0）
        ksm_only_kernel_att: 是否仅启用核注意力（默认False，全维度调制）
        act_type: 注意力激活函数（默认'sigmoid'，可选'tanh'/'softmax'）
    输入：x (Tensor) - [B, in_planes, H, W]（通常为全局池化后的特征）
    输出：(channel_att, filter_att, spatial_att, kernel_att) - 四类注意力权重
    """

    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16,
                 temp=1.0, kernel_temp=None, kernel_att_init='dyconv_as_extra', att_multi=2.0,
                 ksm_only_kernel_att=False, att_grid=1, stride=1, spatial_freq_decompose=False,
                 act_type='sigmoid'):
        super().__init__()
        self.act_type = act_type
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = temp
        self.kernel_temp = kernel_temp if kernel_temp is not None else temp
        self.ksm_only_kernel_att = ksm_only_kernel_att
        self.att_multi = att_multi  # 注意力权重放大系数
        # 注意力通道数：取“输入通道×压缩系数”与“最小通道数”的最大值
        attention_channel = max(int(in_planes * reduction), min_channel)
        # 全局特征提取：平均池化+1×1卷积+批归一化+StarReLU
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = StarReLU()
        # 1. 通道注意力：控制输入通道的重要性
        if ksm_only_kernel_att:
            self.func_channel = self.skip  # 仅核注意力时，通道注意力失效（返回1.0）
        else:
            out_c = in_planes * 2 if (spatial_freq_decompose and kernel_size > 1) else in_planes
            self.channel_fc = nn.Conv2d(attention_channel, out_c, 1, bias=True)
            self.func_channel = self.get_channel_attention
        # 2. 滤波器注意力：控制输出通道（滤波器）的重要性
        if (in_planes == groups and in_planes == out_planes) or ksm_only_kernel_att:
            self.func_filter = self.skip  # 深度可分离卷积时，滤波器注意力失效
        else:
            out_f = out_planes * 2 if spatial_freq_decompose else out_planes
            self.filter_fc = nn.Conv2d(attention_channel, out_f, 1, stride=stride, bias=True)
            self.func_filter = self.get_filter_attention
        # 3. 空间注意力：控制卷积核每个空间位置的重要性
        if kernel_size == 1 or ksm_only_kernel_att:
            self.func_spatial = self.skip  # 1×1卷积无需空间调制
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention
        # 4. 核选择注意力：从多个候选核中选择最优核
        if kernel_num == 1:
            self.func_kernel = self.skip  # 单候选核无需选择
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention
        self._initialize_weights()  # 权重初始化

    @staticmethod
    def skip(_):
        """跳过注意力（返回1.0，无调制作用）"""
        return 1.0

    def get_channel_attention(self, x):
        """生成通道注意力：[B, 1, 1, C, H, W]，适配后续广播相乘"""
        x = self.channel_fc(x).view(x.size(0), 1, 1, -1, x.size(-2), x.size(-1))
        return self._activate(x, self.temperature)

    def get_filter_attention(self, x):
        """生成滤波器注意力：[B, 1, Cout, 1, H, W]"""
        x = self.filter_fc(x).view(x.size(0), 1, -1, 1, x.size(-2), x.size(-1))
        return self._activate(x, self.temperature)

    def get_spatial_attention(self, x):
        """生成空间注意力：[B, 1, 1, 1, k, k]（k为卷积核尺寸）"""
        x = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        return self._activate(x, self.temperature)

    def get_kernel_attention(self, x):
        """生成核选择注意力：[B, K, 1, 1, 1, 1]（K为候选核数量）"""
        x = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        if self.act_type == 'softmax':
            return F.softmax(x / self.kernel_temp, dim=1)  # 概率分布选择
        return self._activate(x, self.kernel_temp) / self.kernel_num  # 归一化权重

    def _activate(self, x, temp):
        """注意力激活函数封装"""
        if self.act_type == 'sigmoid':
            return torch.sigmoid(x / temp) * self.att_multi
        elif self.act_type == 'tanh':
            return 1 + torch.tanh(x / temp)
        raise NotImplementedError(f"不支持的激活类型: {self.act_type}")

    def _initialize_weights(self):
        """权重初始化：卷积核用Kaiming正态分布，注意力层用小方差初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if 'spatial_fc' in str(m) or 'filter_fc' in str(m) or 'kernel_fc' in str(m) or 'channel_fc' in str(m):
                    nn.init.normal_(m.weight, std=1e-6)  # 注意力层小方差初始化
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, use_checkpoint=False):
        """前向传播：支持checkpoint节省显存"""
        if use_checkpoint:
            return checkpoint(self._forward, x)
        return self._forward(x)

    def _forward(self, x):
        """核心逻辑：提取全局特征→生成四类注意力"""
        avg_x = self.relu(self.bn(self.fc(x)))
        return (self.func_channel(avg_x), self.func_filter(avg_x),
                self.func_spatial(avg_x), self.func_kernel(avg_x))


class KernelSpatialModulation_Local(nn.Module):
    """局部核空间调制模块（KSM-Local）
    功能：基于通道局部统计，生成卷积核的精细注意力，增强对局部特征的适配性
    核心逻辑：
        1. 对通道维度做1D卷积，捕捉通道间局部依赖
        2. 可选频域增强（FFT/IFFT），强化全局通道关联
        3. 输出适配卷积核尺寸的注意力权重
    参数：
        channel: 输入特征通道数（默认None，自动适配）
        kernel_num: 候选核数量（默认1）
        out_n: 输出注意力维度（默认1，通常设为 Cout×k×k）
        k_size: 1D卷积核尺寸（默认3，自适应通道数调整）
        use_global: 是否启用频域全局增强（默认False）
    输入：x (Tensor) - [B, 1, C, 1]（通道维度的局部统计特征）
    输出：att_logit (Tensor) - [B, kernel_num, C, out_n]（局部注意力权重）
    """

    def __init__(self, channel=None, kernel_num=1, out_n=1, k_size=3, use_global=False):
        super().__init__()
        self.kn = kernel_num
        self.out_n = out_n
        self.channel = channel
        self.use_global = use_global
        # 自适应调整1D卷积核尺寸（基于通道数的对数，确保为奇数）
        if channel is not None:
            k_size = round((math.log2(channel) / 2) + 0.5) // 2 * 2 + 1
        # 1D卷积：捕捉通道间局部依赖
        self.conv = nn.Conv1d(1, kernel_num * out_n, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)
        nn.init.constant_(self.conv.weight, 1e-6)  # 初始权重极小，避免干扰
        # 频域增强：复杂权重用于FFT后实部/虚部调制
        if self.use_global:
            self.complex_weight = nn.Parameter(torch.randn(1, self.channel // 2 + 1, 2) * 1e-6)
        # 层归一化：稳定通道维度特征分布
        self.norm = nn.LayerNorm(self.channel)

    def forward(self, x, x_std=None):
        # 维度调整：[B, 1, C, 1] → [B, 1, C]（适配1D卷积）
        x = x.squeeze(-1).transpose(-1, -2)
        b, _, c = x.shape
        # 频域增强（可选）：FFT→权重调制→IFFT
        if self.use_global:
            x_rfft = torch.fft.rfft(x.float(), dim=-1)  # 通道维度FFT
            x_real = x_rfft.real * self.complex_weight[..., 0][None]  # 实部调制
            x_imag = x_rfft.imag * self.complex_weight[..., 1][None]  # 虚部调制
            # 逆FFT还原并叠加原始特征，增强全局通道关联
            x = x + torch.fft.irfft(torch.view_as_complex(torch.stack([x_real, x_imag], dim=-1)), dim=-1)
        # 层归一化→1D卷积生成注意力→维度重塑
        x = self.norm(x)
        att_logit = self.conv(x)  # [B, kn×out_n, C]
        att_logit = att_logit.reshape(b, self.kn, self.out_n, c).permute(0, 1, 3, 2)  # [B, kn, C, out_n]
        return att_logit


import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyBandModulation(nn.Module):
    """频率带调制模块（FBM）
    功能：将特征按频率划分为多个频段，分别用注意力增强，强化全局/局部特征表达
    核心逻辑：
        1. 预计算不同频率的掩码（mask），避免重复计算
        2. FFT将特征转换至频域，按掩码分割为低频/高频段
        3. 对各频段用分组卷积生成注意力，加权后融合
    参数：
        in_channels: 输入特征通道数
        k_list: 频率划分系数（默认[2]，系数越小，低频段范围越大）
        lowfreq_att: 是否对低频段单独增强（默认False，仅增强高频）
        spatial_group: 分组卷积组数（默认1，最大不超过64）
        spatial_kernel: 空间卷积核尺寸（默认3）
        max_size: 预计算掩码的最大尺寸（默认(64,64)，适配多数特征）
    输入：x (Tensor) - [B, in_channels, H, W]；att_feat (Tensor) - 注意力计算用特征（默认=x）
    输出：sum(x_list) (Tensor) - 各频段增强后的融合特征
    """

    def __init__(self,
                 in_channels,
                 k_list=[2],
                 lowfreq_att=False,
                 fs_feat='feat',
                 act='sigmoid',
                 spatial='conv',
                 spatial_group=1,
                 spatial_kernel=3,
                 init='zero',
                 max_size=(64, 64),
                 **kwargs,
                 ):
        super().__init__()
        self.k_list = k_list
        self.lowfreq_att = lowfreq_att
        self.in_channels = in_channels
        self.act = act
        self.spatial_group = spatial_group if spatial_group <= 64 else in_channels  # 限制分组数
        # 初始化分组卷积层：每个频段对应一个卷积层（生成频段注意力）
        self.freq_weight_conv_list = nn.ModuleList()
        num_layers = len(k_list) + (1 if lowfreq_att else 0)  # 频段数+低频层（可选）
        for _ in range(num_layers):
            freq_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.spatial_group,
                kernel_size=spatial_kernel,
                groups=self.spatial_group,  # 分组卷积降低计算量
                padding=spatial_kernel // 2,
                bias=True
            )
            # 注意力层小方差初始化，避免初始干扰
            if init == 'zero':
                nn.init.normal_(freq_conv.weight, std=1e-6)
                if freq_conv.bias is not None:
                    freq_conv.bias.data.zero_()
            self.freq_weight_conv_list.append(freq_conv)
        # 预计算并缓存不同频率的掩码（避免每次前向重复计算）
        self.register_buffer('cached_masks', self._precompute_masks(max_size, k_list), persistent=False)

    def _precompute_masks(self, max_size, k_list):
        """预计算频率掩码：按k_list划分低频/高频区域，存储为布尔张量"""
        max_h, max_w = max_size
        _, freq_indices = get_fft2freq(d1=max_h, d2=max_w, use_rfft=True)  # 获取频域坐标
        freq_dist = freq_indices.abs().max(dim=-1, keepdims=False)[0]  # 频域距离（距原点）
        masks = []
        for k in k_list:
            # 掩码规则：频率距离 < 0.5/k 为低频，否则为高频
            mask = freq_dist < (0.5 / k + 1e-8)
            masks.append(mask)
        # 堆叠为 [num_masks, 1, max_h, max_w//2+1]（适配频域尺寸）
        return torch.stack(masks, dim=0).unsqueeze(1)

    def _activate(self, x):
        """注意力激活函数"""
        if self.act == 'sigmoid':
            return torch.sigmoid(x) * 2
        elif self.act == 'tanh':
            return 1 + torch.tanh(x)
        raise NotImplementedError(f"不支持的激活类型: {self.act}")

    def forward(self, x, att_feat=None):
        att_feat = att_feat if att_feat is not None else x
        x_list = []
        x = x.to(torch.float32)
        pre_x = x.clone()  # 备份原始特征，用于频段分割
        b, _, h, w = x.shape
        freq_h, freq_w = h, w // 2 + 1  # 频域尺寸（rfft2输出格式）
        # 1. 特征转换至频域
        x_fft = torch.fft.rfft2(x, norm='ortho')
        # 2. 加载预计算掩码并调整尺寸（适配当前特征大小）
        current_masks = F.interpolate(self.cached_masks.float(), size=(freq_h, freq_w), mode='nearest')
        # 3. 按频段分割并增强（高频段）
        for idx, k in enumerate(self.k_list):
            mask = current_masks[idx]  # 当前频段掩码
            # 频域分割：低频部分（掩码内）和高频部分（掩码外）
            low_part = torch.fft.irfft2(x_fft * mask, s=(h, w), norm='ortho')
            high_part = pre_x - low_part
            pre_x = low_part  # 更新低频累积值，用于下一次分割
            # 生成频段注意力并加权
            freq_weight = self.freq_weight_conv_list[idx](att_feat)
            freq_weight = self._activate(freq_weight)
            # 分组加权：适配分组卷积的通道划分
            tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * \
                  high_part.reshape(b, self.spatial_group, -1, h, w)
            x_list.append(tmp.reshape(b, -1, h, w))
        # 4. 增强低频段（可选）
        if self.lowfreq_att:
            freq_weight = self.freq_weight_conv_list[len(self.k_list)](att_feat)
            freq_weight = self._activate(freq_weight)
            tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * \
                  pre_x.reshape(b, self.spatial_group, -1, h, w)
            x_list.append(tmp.reshape(b, -1, h, w))
        else:
            x_list.append(pre_x)  # 直接加入低频段，不增强
        # 5. 融合所有频段特征
        return sum(x_list)


def get_fft2freq(d1, d2, use_rfft=False):
    # Frequency components for rows and columns
    freq_h = torch.fft.fftfreq(d1)  # Frequency for the rows (d1)
    if use_rfft:
        freq_w = torch.fft.rfftfreq(d2)  # Frequency for the columns (d2)
    else:
        freq_w = torch.fft.fftfreq(d2)

    # Meshgrid to create a 2D grid of frequency coordinates
    freq_hw = torch.stack(torch.meshgrid(freq_h, freq_w), dim=-1)
    # print(freq_hw)
    # print(freq_hw.shape)
    # Calculate the distance from the origin (0, 0) in the frequency space
    dist = torch.norm(freq_hw, dim=-1)
    # print(dist.shape)
    # Sort the distances and get the indices
    sorted_dist, indices = torch.sort(dist.view(-1))  # Flatten the distance tensor for sorting
    # print(sorted_dist.shape)

    # Get the corresponding coordinates for the sorted distances
    if use_rfft:
        d2 = d2 // 2 + 1
        # print(d2)
    sorted_coords = torch.stack([indices // d2, indices % d2], dim=-1)  # Convert flat indices to 2D coords
    # print(sorted_coords.shape)
    # # Print sorted distances and corresponding coordinates
    # for i in range(sorted_dist.shape[0]):
    #     print(f"Distance: {sorted_dist[i]:.4f}, Coordinates: ({sorted_coords[i, 0]}, {sorted_coords[i, 1]})")

    if False:
        # Plot the distance matrix as a grayscale image
        plt.imshow(dist.cpu().numpy(), cmap='gray', origin='lower')
        plt.colorbar()
        plt.title('Frequency Domain Distance')
        plt.show()
    return sorted_coords.permute(1, 0), freq_hw


@CONV_LAYERS.register_module()  # 为 mmdet、mmseg 注册这个模块，表示该类是一个卷积层模块
class FDConv(nn.Conv2d):  # 定义 FDConv 类，继承自 nn.Conv2d
    def __init__(self,
                 *args,  # 允许传入任意数量的参数
                 reduction=0.0625,  # 参数：减少因子，用于空间-频域调制
                 kernel_num=4,  # 核心数目
                 use_fdconv_if_c_gt=16,  # 如果输入通道数大于或等于16，则使用频域卷积
                 use_fdconv_if_k_in=[1, 3],  # 如果卷积核大小在指定列表中，则使用频域卷积
                 use_fdconv_if_stride_in=[1],  # 如果步幅在指定列表中，则使用频域卷积
                 use_fbm_if_k_in=[3],  # 如果卷积核大小在指定列表中，则使用频带调制
                 use_fbm_for_stride=False,  # 是否在步幅大于1时使用频带调制
                 kernel_temp=1.0,  # 核心温度参数
                 temp=None,  # 温度参数
                 att_multi=2.0,  # 注意力缩放因子
                 param_ratio=1,  # 参数比率
                 param_reduction=1.0,  # 参数减少因子
                 ksm_only_kernel_att=False,  # 是否仅使用内核注意力
                 att_grid=1,  # 注意力网格
                 use_ksm_local=True,  # 是否使用局部内核空间调制
                 ksm_local_act='sigmoid',  # 局部空间调制激活函数
                 ksm_global_act='sigmoid',  # 全局空间调制激活函数
                 spatial_freq_decompose=False,  # 是否进行空间频率分解
                 convert_param=True,  # 是否转换参数
                 linear_mode=False,  # 是否启用线性模式
                 fbm_cfg={  # 频带调制配置字典
                     'k_list': [2, 4, 8],
                     'lowfreq_att': False,
                     'fs_feat': 'feat',
                     'act': 'sigmoid',
                     'spatial': 'conv',
                     'spatial_group': 1,
                     'spatial_kernel': 3,
                     'init': 'zero',
                     'global_selection': False,
                 },
                 **kwargs,  # 其他参数
                 ):
        super().__init__(*args, **kwargs)  # 调用父类构造函数
        self.use_fdconv_if_c_gt = use_fdconv_if_c_gt  # 设置使用频域卷积的输入通道数阈值
        self.use_fdconv_if_k_in = use_fdconv_if_k_in  # 设置使用频域卷积的核大小列表
        self.use_fdconv_if_stride_in = use_fdconv_if_stride_in  # 设置使用频域卷积的步幅列表
        self.kernel_num = kernel_num  # 设置核心数目
        self.param_ratio = param_ratio  # 设置参数比率
        self.param_reduction = param_reduction  # 设置参数减少因子
        self.use_ksm_local = use_ksm_local  # 设置是否使用局部内核空间调制
        self.att_multi = att_multi  # 设置注意力缩放因子
        self.spatial_freq_decompose = spatial_freq_decompose  # 设置是否使用空间频率分解
        self.use_fbm_if_k_in = use_fbm_if_k_in  # 设置使用频带调制的卷积核大小列表
        # 检查局部和全局空间调制的激活函数是否合法
        self.ksm_local_act = ksm_local_act
        self.ksm_global_act = ksm_global_act
        assert self.ksm_local_act in ['sigmoid', 'tanh']
        assert self.ksm_global_act in ['softmax', 'sigmoid', 'tanh']
        # 核心数量与温度的设置
        if self.kernel_num is None:
            self.kernel_num = self.out_channels // 2  # 默认情况下将核心数设置为输出通道数的一半
            kernel_temp = math.sqrt(self.kernel_num * self.param_ratio)  # 根据核心数和参数比率计算温度
        if temp is None:
            temp = kernel_temp  # 如果没有传递温度参数，则使用计算得到的温度
        # 判断是否满足使用频域卷积的条件
        if min(self.in_channels, self.out_channels) <= self.use_fdconv_if_c_gt \
                or self.kernel_size[0] not in self.use_fdconv_if_k_in:
            return
        print('*** kernel_num:', self.kernel_num)  # 打印核心数
        self.alpha = min(self.out_channels,
                         self.in_channels) // 2 * self.kernel_num * self.param_ratio / param_reduction  # 设置 alpha 参数
        # 创建全局空间调制层
        self.KSM_Global = KernelSpatialModulation_Global(self.in_channels, self.out_channels, self.kernel_size[0],
                                                         groups=self.groups,
                                                         temp=temp,
                                                         kernel_temp=kernel_temp,
                                                         reduction=reduction,
                                                         kernel_num=self.kernel_num * self.param_ratio,
                                                         kernel_att_init=None, att_multi=att_multi,
                                                         ksm_only_kernel_att=ksm_only_kernel_att,
                                                         act_type=self.ksm_global_act,
                                                         att_grid=att_grid, stride=self.stride,
                                                         spatial_freq_decompose=spatial_freq_decompose)
        # 如果需要，创建频带调制层（FBM）
        if self.kernel_size[0] in use_fbm_if_k_in or (use_fbm_for_stride and self.stride[0] > 1):
            self.FBM = FrequencyBandModulation(self.in_channels, **fbm_cfg)
        # 如果使用局部空间调制，创建对应的层
        if self.use_ksm_local:
            self.KSM_Local = KernelSpatialModulation_Local(channel=self.in_channels, kernel_num=1, out_n=int(
                self.out_channels * self.kernel_size[0] * self.kernel_size[1]))
        # 设置线性模式和参数转换
        self.linear_mode = linear_mode
        self.convert2dftweight(convert_param)  # 转换为频域卷积权重

    def convert2dftweight(self, convert_param):
        # 将卷积权重转换为频域表示
        d1, d2, k1, k2 = self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]
        freq_indices, _ = get_fft2freq(d1 * k1, d2 * k2, use_rfft=True)  # 获取FFT频率索引
        weight = self.weight.permute(0, 2, 1, 3).reshape(d1 * k1, d2 * k2)  # 重塑卷积权重
        weight_rfft = torch.fft.rfft2(weight, dim=(0, 1))  # 计算权重的FFT
        if self.param_reduction < 1:
            num_to_keep = int(freq_indices.size(1) * self.param_reduction)  # 保留前 k 个最低频的索引
            freq_indices = freq_indices[:, :num_to_keep]  # 截取频率索引
            weight_rfft = weight_rfft[freq_indices[0, :], freq_indices[1, :]]  # 选择对应频率的FFT结果
            weight_rfft = weight_rfft.reshape(-1, 2)[None,].repeat(self.param_ratio, 1, 1) / (
                        min(self.out_channels, self.in_channels) // 2)
        else:
            weight_rfft = torch.stack([weight_rfft.real, weight_rfft.imag], dim=-1)[None,].repeat(self.param_ratio, 1,
                                                                                                  1, 1) / (
                                      min(self.out_channels, self.in_channels) // 2)
        # 如果需要，将频域权重转换为可训练的参数
        if convert_param:
            self.dft_weight = nn.Parameter(weight_rfft, requires_grad=True)
            del self.weight  # 删除原始权重
        else:
            if self.linear_mode:
                assert self.kernel_size[0] == 1 and self.kernel_size[1] == 1
                self.weight = torch.nn.Parameter(self.weight.squeeze(), requires_grad=True)
        indices = []
        for i in range(self.param_ratio):
            indices.append(freq_indices.reshape(2, self.kernel_num, -1))  # 将频率索引重塑为需要的形状
        self.register_buffer('indices', torch.stack(indices, dim=0), persistent=False)  # 注册索引

    def get_FDW(self, ):
        # 获取频域权重
        d1, d2, k1, k2 = self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]
        weight = self.weight.reshape(d1, d2, k1, k2).permute(0, 2, 1, 3).reshape(d1 * k1, d2 * k2)
        weight_rfft = torch.fft.rfft2(weight, dim=(0, 1)).contiguous()  # 计算权重的FFT
        weight_rfft = torch.stack([weight_rfft.real, weight_rfft.imag], dim=-1)[None,].repeat(self.param_ratio, 1, 1,
                                                                                              1) / (
                                  min(self.out_channels, self.in_channels) // 2)
        return weight_rfft

    def forward(self, x):
        # 正向传播方法
        if min(self.in_channels, self.out_channels) <= self.use_fdconv_if_c_gt or self.kernel_size[
            0] not in self.use_fdconv_if_k_in:
            return super().forward(x)  # 如果不满足使用频域卷积的条件，使用标准卷积
        global_x = F.adaptive_avg_pool2d(x, 1)  # 对输入进行全局平均池化
        # 获取全局、滤波器、空间和内核注意力
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.KSM_Global(global_x)
        if self.use_ksm_local:
            # 使用局部空间调制获取局部注意力
            hr_att_logit = self.KSM_Local(global_x)
            hr_att_logit = hr_att_logit.reshape(x.size(0), 1, self.in_channels, self.out_channels, self.kernel_size[0],
                                                self.kernel_size[1])
            hr_att_logit = hr_att_logit.permute(0, 1, 3, 2, 4, 5)
            if self.ksm_local_act == 'sigmoid':
                hr_att = hr_att_logit.sigmoid() * self.att_multi
            elif self.ksm_local_act == 'tanh':
                hr_att = 1 + hr_att_logit.tanh()
            else:
                raise NotImplementedError
        else:
            hr_att = 1
        # 执行频域卷积和注意力加权
        b = x.size(0)
        batch_size, in_planes, height, width = x.size()
        DFT_map = torch.zeros(
            (b, self.out_channels * self.kernel_size[0], self.in_channels * self.kernel_size[1] // 2 + 1, 2),
            device=x.device)
        kernel_attention = kernel_attention.reshape(b, self.param_ratio, self.kernel_num, -1)
        if hasattr(self, 'dft_weight'):
            dft_weight = self.dft_weight  # 获取频域卷积权重
        else:
            dft_weight = self.get_FDW()  # 计算频域卷积权重
        # 执行频域卷积
        for i in range(self.param_ratio):
            indices = self.indices[i]
            if self.param_reduction < 1:
                w = dft_weight[i].reshape(self.kernel_num, -1, 2)[None]
                DFT_map[:, indices[0, :, :], indices[1, :, :]] += torch.stack(
                    [w[..., 0] * kernel_attention[:, i], w[..., 1] * kernel_attention[:, i]], dim=-1)
            else:
                w = dft_weight[i][indices[0, :, :], indices[1, :, :]][None] * self.alpha
                DFT_map[:, indices[0, :, :], indices[1, :, :]] += torch.stack(
                    [w[..., 0] * kernel_attention[:, i], w[..., 1] * kernel_attention[:, i]], dim=-1)
        # 逆FFT变换获得最终权重
        adaptive_weights = torch.fft.irfft2(torch.view_as_complex(DFT_map), dim=(1, 2)).reshape(batch_size, 1,
                                                                                                self.out_channels,
                                                                                                self.kernel_size[0],
                                                                                                self.in_channels,
                                                                                                self.kernel_size[1])
        adaptive_weights = adaptive_weights.permute(0, 1, 2, 4, 3, 5)
        # 如果启用了频带调制，对输入进行处理
        if hasattr(self, 'FBM'):
            x = self.FBM(x)
        # 根据注意力加权进行卷积操作
        if self.out_channels * self.in_channels * self.kernel_size[0] * self.kernel_size[1] < (
                in_planes + self.out_channels) * height * width:
            aggregate_weight = spatial_attention * channel_attention * filter_attention * adaptive_weights * hr_att
            aggregate_weight = torch.sum(aggregate_weight, dim=1)
            aggregate_weight = aggregate_weight.view(
                [-1, self.in_channels // self.groups, self.kernel_size[0], self.kernel_size[1]])
            x = x.reshape(1, -1, height, width)
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)
            if isinstance(filter_attention, float):
                output = output.view(batch_size, self.out_channels, output.size(-2), output.size(-1))
            else:
                output = output.view(batch_size, self.out_channels, output.size(-2), output.size(-1))
        else:
            aggregate_weight = spatial_attention * adaptive_weights * hr_att
            aggregate_weight = torch.sum(aggregate_weight, dim=1)
            if not isinstance(channel_attention, float):
                x = x * channel_attention.view(b, -1, 1, 1)
            aggregate_weight = aggregate_weight.view(
                [-1, self.in_channels // self.groups, self.kernel_size[0], self.kernel_size[1]])
            x = x.reshape(1, -1, height, width)
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)
            if isinstance(filter_attention, float):
                output = output.view(batch_size, self.out_channels, output.size(-2), output.size(-1))
            else:
                output = output.view(batch_size, self.out_channels, output.size(-2),
                                     output.size(-1)) * filter_attention.view(b, -1, 1, 1)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)  # 如果有偏置，加上偏置
        return output  # 返回卷积结果

    def profile_module(self, input: Tensor, *args, **kwargs):
        # 用于模块的性能分析
        b_sz, c, h, w = input.shape
        seq_len = h * w
        # 计算FFT和IFFT的计算量
        p_ff, m_ff = 0, 5 * b_sz * seq_len * int(math.log(seq_len)) * c
        params = macs = self.hidden_size * self.hidden_size_factor * self.hidden_size * 2 * 2 // self.num_blocks
        macs = macs * b_sz * seq_len
        # 返回输入，参数和计算量
        return input, params, macs + m_ff


if __name__ == '__main__':
    x = torch.rand(4, 128, 64, 64) * 1
    m = FDConv(in_channels=128, out_channels=64, kernel_num=8, kernel_size=3, padding=1, bias=True)
    y = m(x)
    print("微信公众号：十小大的底层视觉工坊")
    print("知乎、CSDN：十小大")
    print("输入特征维度：", x.shape)
    print("输出特征维度：", y.shape)