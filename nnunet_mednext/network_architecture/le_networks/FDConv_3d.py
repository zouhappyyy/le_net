import math
import torch.autograd
# python
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint


def get_fft3freq(d1, d2, d3, use_rfft=False):
    """
    3D 频率格点与按距离排序的索引。
    3D FFT 频率空间的“坐标生成 + 按频率半径排序”工具函数
    给定 3D 尺寸 (d1, d2, d3)，构造 3D FFT 频率坐标网格 (fx, fy, fz)，
    计算每个频点到原点的频率距离，并按距离从低频到高频排序，返回排序后的 3D 频率索引坐标 和 频率网格本身。
    返回 (sorted_coords.permute(1,0), freq_grid) 与原 2D 版本接口类似。
    """
    freq_d1 = torch.fft.fftfreq(d1)
    freq_d2 = torch.fft.fftfreq(d2)

    # 实数 FFT（rFFT）在最后一个维度：
    # 只保留 非负频率
    # 利用共轭对称性节省一半存储
    if use_rfft:
        freq_d3 = torch.fft.rfftfreq(d3)
        d3_out = d3 // 2 + 1
    else:
        freq_d3 = torch.fft.fftfreq(d3)
        d3_out = d3

    # meshgrid (d1, d2, d3_out, 3)
    g1, g2, g3 = torch.meshgrid(freq_d1, freq_d2, freq_d3, indexing='ij')
    freq_grid = torch.stack([g1, g2, g3], dim=-1)
    dist = torch.norm(freq_grid, dim=-1)  # 计算频率“半径”（频率模长）
    sorted_dist, indices = torch.sort(dist.reshape(-1))  # 按频率距离排序
    # convert flattened index -> (i,j,k)
    coords = torch.stack([
        indices // (d2 * d3_out),
        (indices % (d2 * d3_out)) // d3_out,
        indices % d3_out
    ], dim=-1)
    return coords.permute(1, 0), freq_grid


class KernelSpatialModulation_Global3D(nn.Module):
    """
    3D 版本的全局核空间调制（KSM-Global）。
    输入应为经过 AdaptiveAvgPool3d(1) 后的张量 [B, C, 1, 1, 1] 或常规 5D 特征。
    输出 (channel_att, filter_att, spatial_att, kernel_att)：
      - channel_att: [B,1,1,C,1,1,1]  哪些输入通道重要  给每个输入通道一个权重
      - filter_att:  [B,1,Cout,1,1,1,1] 哪些卷积核（输出通道）重要  给每个输出通道（卷积核）一个权重
      - spatial_att: [B,1,1,1,k,k,k] 核里哪些空间位置重要  作用在卷积核空间结构上的注意力
      - kernel_att:  [B,K,1,1,1,1,1]  哪些候选核重要  在多个基础卷积核（experts）之间做加权组合
    """

    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4,
                 min_channel=16, temp=1.0, kernel_temp=None, ksm_only_kernel_att=False,
                 att_multi=2.0, act_type='sigmoid', att_grid=1, stride=1, spatial_freq_decompose=False):
        super().__init__()
        self.act_type = act_type
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = temp
        self.kernel_temp = kernel_temp if kernel_temp is not None else temp
        self.ksm_only_kernel_att = ksm_only_kernel_att
        self.att_multi = att_multi

        attention_channel = max(int(in_planes * reduction), min_channel)
        # 3D avgpool + 1x1x1 conv + BN + StarReLU（这里用 ReLU^2 替代 StarReLU 简洁化）
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Conv3d(in_planes, attention_channel, kernel_size=1, bias=False)
        self.bn = nn.GroupNorm(num_groups=1, num_channels=attention_channel)
        self.relu = nn.ReLU(inplace=True)

        # channel 通道注意力：控制输入通道的重要性
        if ksm_only_kernel_att:
            self.func_channel = lambda _: 1.0
        else:
            out_c = in_planes * 2 if (spatial_freq_decompose and kernel_size > 1) else in_planes
            self.channel_fc = nn.Conv3d(attention_channel, out_c, kernel_size=1, bias=True)
            self.func_channel = self.get_channel_attention

        # filter 滤波器注意力：控制输出通道（滤波器）的重要性
        if (in_planes == groups and in_planes == out_planes) or ksm_only_kernel_att:
            self.func_filter = lambda _: 1.0
        else:
            out_f = out_planes * 2 if spatial_freq_decompose else out_planes
            self.filter_fc = nn.Conv3d(attention_channel, out_f, kernel_size=1, stride=stride, bias=True)
            self.func_filter = self.get_filter_attention

        # spatial (k^3) 空间注意力：控制卷积核每个空间位置的重要性
        if kernel_size == 1 or ksm_only_kernel_att:
            self.func_spatial = lambda _: 1.0
        else:
            self.spatial_fc = nn.Conv3d(attention_channel, kernel_size ** 3, kernel_size=1, bias=True)
            self.func_spatial = self.get_spatial_attention

        # kernel selection 核选择注意力：从多个候选核中选择最优核
        if kernel_num == 1:
            self.func_kernel = lambda _: 1.0
        else:
            self.kernel_fc = nn.Conv3d(attention_channel, kernel_num, kernel_size=1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def get_channel_attention(self, x):
        # x: [B, Catt, 1,1,1] -> reshape -> [B,1,1,C,1,1,1]
        x = self.channel_fc(x).view(x.size(0), 1, 1, -1, 1, 1, 1)
        return self._activate(x, self.temperature)

    def get_filter_attention(self, x):
        x = self.filter_fc(x).view(x.size(0), 1, -1, 1, 1, 1, 1)
        return self._activate(x, self.temperature)

    def get_spatial_attention(self, x):
        x = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size, self.kernel_size)
        return self._activate(x, self.temperature)

    def get_kernel_attention(self, x):
        x = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1, 1)
        if self.act_type == 'softmax':
            return F.softmax(x / self.kernel_temp, dim=1)
        return self._activate(x, self.kernel_temp) / self.kernel_num

    def _activate(self, x, temp):
        if self.act_type == 'sigmoid':
            return torch.sigmoid(x / temp) * self.att_multi
        elif self.act_type == 'tanh':
            return 1 + torch.tanh(x / temp)
        raise NotImplementedError

    def _initialize_weights(self):
        """权重初始化：卷积核用Kaiming正态分布，注意力层用小方差初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                if any(n in str(m) for n in ['spatial_fc', 'filter_fc', 'kernel_fc', 'channel_fc']):
                    nn.init.normal_(m.weight, std=1e-6)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm3d):
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


class KernelSpatialModulation_Local_3d(nn.Module):
    """
    3D 版本的局部核空间调制（KSM-Local）。
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
    输入：x (Tensor) - [B, C, 1, 1, 1]（或任意被压缩到单一空间位置的 5D 张量）
    输出：att_logit (Tensor) - [B, kn, C, out_n]
    参数与原接口一致：channel, kernel_num, out_n, k_size, use_global
    """

    def __init__(self, channel=None, kernel_num=1, out_n=1, k_size=3, use_global=False):
        super().__init__()
        self.kn = kernel_num
        self.out_n = out_n
        self.channel = channel
        self.use_global = use_global

        # 自适应 k_size：基于通道数做简单缩放并保证为奇数
        if channel is not None:
            ks = max(3, int(round(math.log2(max(2, channel)) / 2)))
            if ks % 2 == 0:
                ks += 1
            k_size = ks
        self.k_size = k_size

        # 1D 卷积：输入通道为 1（沿通道维的序列卷积）
        self.conv = nn.Conv1d(in_channels=1, out_channels=self.kn * self.out_n,
                              kernel_size=self.k_size, padding=(self.k_size - 1) // 2, bias=False)
        nn.init.constant_(self.conv.weight, 1e-6)  # 初始权重极小，避免干扰

        # 可选的频域全局增强：对通道序列做 rfft 调制
        # 频域增强：复杂权重用于FFT后实部/虚部调制
        if self.use_global:
            assert channel is not None, "use_global=True 时必须提供 channel"
            self.complex_weight = nn.Parameter(torch.randn(1, channel // 2 + 1, 2) * 1e-6)

        # 层归一化针对通道维度（最后一维）
        if channel is not None:
            self.norm = nn.LayerNorm(channel)
        else:
            self.norm = None

    def forward(self, x, x_std=None):
        """
        x: [B, C, 1, 1, 1] 或 [B, C, S1, S2, S3]（但通常已通过 AdaptiveAvgPool3d(1) 变为 [B,C,1,1,1]）
        返回: [B, kn, C, out_n]
        """
        b = x.size(0)
        c = x.size(1)
        # 合并/压缩空间维到通道序列：得到 [B, C, 1] -> 再 squeeze -> [B, C]
        x = x.view(b, c, -1)  # [B, C, S]
        if x.size(-1) != 1:
            x = x.mean(dim=-1)  # 若空间不是严格1，取平均作为全局统计
        else:
            x = x.squeeze(-1)  # [B, C]

        # 频域增强（可选）
        if self.use_global:
            # rfft 在通道维上：输入需为 [B, C]
            x_rfft = torch.fft.rfft(x.float(), dim=-1)  # [B, C//2+1] complex
            # 调制实部/虚部并叠回时域（保持形状 [B, C]）
            real = x_rfft.real * self.complex_weight[..., 0]  # 实部调制
            imag = x_rfft.imag * self.complex_weight[..., 1]  # 虚部调制
            # 逆FFT还原并叠加原始特征，增强全局通道关联
            x = x + torch.fft.irfft(torch.view_as_complex(torch.stack([real, imag], dim=-1)), n=c, dim=-1)

        # 归一化 -> 准备 1D conv 输入 [B, 1, C]
        if self.norm is not None:
            x = self.norm(x)
        x = x.unsqueeze(1)  # [B, 1, C]

        att = self.conv(x)  # [B, kn*out_n, C]
        # reshape -> [B, kn, out_n, C] -> permute -> [B, kn, C, out_n]
        att = att.view(b, self.kn, self.out_n, c).permute(0, 1, 3, 2).contiguous()
        return att


class FrequencyBandModulation3D(nn.Module):
    """
    3D 频带调制（FBM）：
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
    - 对输入执行 3D rfftn/irfftn，按预计算的频域 mask 分割频带
    - 每个频带用分组 3D 卷积生成注意力并加权
    """

    def __init__(self, in_channels, k_list=[2], lowfreq_att=False, fs_feat='feat',
                 act='sigmoid', spatial='conv', spatial_group=1, spatial_kernel=3,
                 init='zero', max_size=(64, 64, 64), **kwargs):
        super().__init__()
        self.k_list = k_list
        self.lowfreq_att = lowfreq_att
        self.in_channels = in_channels
        self.act = act
        self.spatial_group = spatial_group if spatial_group <= 64 else in_channels
        self.spatial_kernel = spatial_kernel
        # 初始化分组卷积层：每个频段对应一个卷积层（生成频段注意力）
        self.freq_weight_conv_list = nn.ModuleList()
        num_layers = len(k_list) + (1 if lowfreq_att else 0)
        for _ in range(num_layers):
            conv3d = nn.Conv3d(in_channels=in_channels, out_channels=self.spatial_group,
                               kernel_size=spatial_kernel, groups=self.spatial_group,
                               padding=spatial_kernel // 2, bias=True)
            if init == 'zero':
                nn.init.normal_(conv3d.weight, std=1e-6)
                if conv3d.bias is not None:
                    conv3d.bias.data.zero_()
            self.freq_weight_conv_list.append(conv3d)

        max_d, max_h, max_w = max_size
        # cached_masks: [num_masks, 1, max_d, max_h, max_w_out]
        # 预计算并缓存不同频率的掩码（避免每次前向重复计算）
        self.register_buffer('cached_masks', self._precompute_masks(max_size, k_list), persistent=False)

    def _precompute_masks(self, max_size, k_list):
        max_d, max_h, max_w = max_size
        _, freq_grid = get_fft3freq(max_d, max_h, max_w, use_rfft=True)
        # 频率距离
        freq_dist = torch.norm(freq_grid, dim=-1)
        masks = []
        for k in k_list:
            # 掩码规则：频率距离 < 0.5/k 为低频，否则为高频
            mask = freq_dist < (0.5 / k + 1e-8)
            masks.append(mask)
        return torch.stack(masks, dim=0).unsqueeze(1)  # [num_masks,1,D,H,W_out]

    def _activate(self, x):
        if self.act == 'sigmoid':
            return torch.sigmoid(x) * 2
        elif self.act == 'tanh':
            return 1 + torch.tanh(x)
        raise NotImplementedError

    def forward(self, x, att_feat=None):
        att_feat = att_feat if att_feat is not None else x
        x_list = []
        x = x.to(torch.float32)
        pre_x = x.clone()  # 备份原始特征，用于频段分割
        b, _, d, h, w = x.shape
        freq_d, freq_h, freq_w = d, h, w // 2 + 1

        # 1. 特征转换至频域 3D rfftn -> 频域
        x_fft = torch.fft.rfftn(x, s=(d, h, w), dim=(-3, -2, -1), norm='ortho')
        # 2. 加载预计算掩码并调整尺寸（适配当前特征大小） 调整 cached masks 到当前频域尺寸
        current_masks = F.interpolate(self.cached_masks.float(), size=(freq_d, freq_h, freq_w), mode='nearest')
        # 3. 按频段分割并增强（高频段）
        for idx, k in enumerate(self.k_list):
            mask = current_masks[idx]  # [1, D, H, W_out]
            # 频域分割并回到时域
            # 频域分割：低频部分（掩码内）和高频部分（掩码外）
            low_part = torch.fft.irfftn(x_fft * mask, s=(d, h, w), dim=(-3, -2, -1), norm='ortho')
            high_part = pre_x - low_part
            pre_x = low_part  # 更新低频累积值，用于下一次分割
            # 生成频段注意力并加权
            freq_weight = self.freq_weight_conv_list[idx](att_feat)
            freq_weight = self._activate(freq_weight)
            # 分组加权：适配分组卷积的通道划分
            tmp = freq_weight.reshape(b, self.spatial_group, -1, d, h, w) * \
                  high_part.reshape(b, self.spatial_group, -1, d, h, w)
            x_list.append(tmp.reshape(b, -1, d, h, w))
        # 4. 增强低频段（可选）
        if self.lowfreq_att:
            freq_weight = self.freq_weight_conv_list[len(self.k_list)](att_feat)
            freq_weight = self._activate(freq_weight)
            tmp = freq_weight.reshape(b, self.spatial_group, -1, d, h, w) * \
                  pre_x.reshape(b, self.spatial_group, -1, d, h, w)
            x_list.append(tmp.reshape(b, -1, d, h, w))
        else:
            x_list.append(pre_x)  # 直接加入低频段，不增强
        # 5. 融合所有频段特征
        return sum(x_list)



class FDConv(nn.Conv3d):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 reduction=0.0625,
                 kernel_num=4,
                 use_fdconv_if_c_gt=16,
                 use_fdconv_if_k_in=[1, 3],
                 use_fdconv_if_stride_in=[1],
                 use_fbm_if_k_in=[3],
                 use_fbm_for_stride=False,
                 kernel_temp=1.0,
                 temp=None,
                 att_multi=2.0,
                 param_ratio=1,
                 param_reduction=1.0,
                 ksm_only_kernel_att=False,
                 att_grid=1,
                 use_ksm_local=True,
                 ksm_local_act='sigmoid',
                 ksm_global_act='sigmoid',
                 spatial_freq_decompose=False,
                 convert_param=True,
                 linear_mode=False,
                 fbm_cfg=None,
                 **kwargs):
        # 不再强制 depthwise，直接使用传入的参数
        if fbm_cfg is None:
            fbm_cfg = {
                'k_list': [2, 4, 8],
                'lowfreq_att': False,
                'fs_feat': 'feat',
                'act': 'sigmoid',
                'spatial': 'conv',
                'spatial_group': 1,
                'spatial_kernel': 3,
                'init': 'zero',
                'global_selection': False,
            }

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            **kwargs
        )

        self.use_fdconv_if_c_gt = use_fdconv_if_c_gt
        self.use_fdconv_if_k_in = use_fdconv_if_k_in
        self.use_fdconv_if_stride_in = use_fdconv_if_stride_in
        self.kernel_num = kernel_num
        self.param_ratio = param_ratio
        self.param_reduction = param_reduction
        self.use_ksm_local = use_ksm_local
        self.att_multi = att_multi
        self.spatial_freq_decompose = spatial_freq_decompose
        self.use_fbm_if_k_in = use_fbm_if_k_in
        self.ksm_local_act = ksm_local_act
        self.ksm_global_act = ksm_global_act
        assert self.ksm_local_act in ['sigmoid', 'tanh']
        assert self.ksm_global_act in ['softmax', 'sigmoid', 'tanh']

        if self.kernel_num is None:
            self.kernel_num = self.out_channels // 2
            kernel_temp = math.sqrt(self.kernel_num * self.param_ratio)
        else:
            kernel_temp = kernel_temp if 'kernel_temp' in locals() else kernel_temp
        if temp is None:
            temp = kernel_temp

        # 不满足条件：直接退化为普通 Conv3d（不启用频域分支）
        if min(self.in_channels, self.out_channels) <= self.use_fdconv_if_c_gt \
                or self.kernel_size[0] not in self.use_fdconv_if_k_in:
            self.linear_mode = linear_mode
            return

        self.alpha = min(self.out_channels,
                         self.in_channels) // 2 * self.kernel_num * self.param_ratio / param_reduction

        # 全局 KSM：如果 in==groups==out，会自动关闭 filter_fc，兼容 depthwise 和普通 conv
        self.KSM_Global = KernelSpatialModulation_Global3D(
            self.in_channels,
            self.out_channels,
            self.kernel_size[0],
            groups=self.groups,
            temp=temp,
            kernel_temp=kernel_temp,
            reduction=reduction,
            kernel_num=self.kernel_num * self.param_ratio,
            att_multi=att_multi,
            ksm_only_kernel_att=ksm_only_kernel_att,
            act_type=self.ksm_global_act,
            att_grid=att_grid,
            stride=self.stride,
            spatial_freq_decompose=spatial_freq_decompose
        )

        # 频带调制（可选）
        if self.kernel_size[0] in use_fbm_if_k_in or (use_fbm_for_stride and self.stride[0] > 1):
            self.FBM = FrequencyBandModulation3D(self.in_channels, **fbm_cfg)

        # 局部 KSM（可选）
        if self.use_ksm_local:
            kT, kH, kW = self.kernel_size
            out_n = int(self.out_channels * kT * kH * kW)
            self.KSM_Local = KernelSpatialModulation_Local_3d(
                channel=self.in_channels,
                kernel_num=1,
                out_n=out_n
            )

        self.linear_mode = linear_mode
        self.convert2dftweight(convert_param)

    def convert2dftweight(self, convert_param: bool):
        # 根据是否为 depthwise 决定 d2
        d1 = self.out_channels
        if self.groups == self.in_channels == self.out_channels:
            d2 = 1  # depthwise 情况
        else:
            d2 = self.in_channels

        kT, kH, kW = self.kernel_size
        freq_indices, _ = get_fft3freq(d1, d2, kT * kH * kW, use_rfft=True)

        weight = self.weight.permute(0, 2, 1, 3, 4).reshape(d1 * kT, d2 * kH * kW)
        weight_rfft = torch.fft.rfft2(weight, dim=(0, 1))

        if self.param_reduction < 1:
            num_to_keep = int(freq_indices.size(1) * self.param_reduction)
            freq_indices = freq_indices[:, :num_to_keep]
            weight_rfft = weight_rfft[freq_indices[0, :], freq_indices[1, :]]
            weight_rfft = weight_rfft.reshape(-1, 2)[None,].repeat(self.param_ratio, 1, 1) / (
                min(self.out_channels, self.in_channels) // 2
            )
        else:
            weight_rfft = torch.stack(
                [weight_rfft.real, weight_rfft.imag],
                dim=-1
            )[None,].repeat(self.param_ratio, 1, 1, 1) / (
                min(self.out_channels, self.in_channels) // 2
            )

        if convert_param:
            self.dft_weight = nn.Parameter(weight_rfft, requires_grad=True)
            del self.weight
        else:
            if self.linear_mode:
                assert kT == 1 and kH == 1 and kW == 1
                self.weight = torch.nn.Parameter(self.weight.squeeze(), requires_grad=True)

        indices = []
        for i in range(self.param_ratio):
            indices.append(freq_indices.reshape(2, self.kernel_num, -1))
        self.register_buffer('indices', torch.stack(indices, dim=0), persistent=False)

    def get_FDW(self):
        d1 = self.out_channels
        d2 = self.in_channels
        kT, kH, kW = self.kernel_size
        weight = self.weight.reshape(d1, d2, kT, kH, kW).permute(0, 2, 1, 3, 4).reshape(d1 * kT, d2 * kH * kW)
        weight_rfft = torch.fft.rfft2(weight, dim=(0, 1)).contiguous()
        weight_rfft = torch.stack(
            [weight_rfft.real, weight_rfft.imag], dim=-1
        )[None,].repeat(self.param_ratio, 1, 1, 1) / (
            min(self.out_channels, self.in_channels) // 2
        )
        return weight_rfft

    def forward(self, x: Tensor) -> Tensor:
        # 不启用 FD 分支时，退化成普通 Conv3d
        if not hasattr(self, 'KSM_Global'):
            return super().forward(x)

        global_x = F.adaptive_avg_pool3d(x, 1)
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.KSM_Global(global_x)

        if self.use_ksm_local:
            hr_att_logit = self.KSM_Local(global_x)
            b = x.size(0)
            kT, kH, kW = self.kernel_size
            hr_att_logit = hr_att_logit.reshape(
                b, 1, self.in_channels, self.out_channels, kT, kH, kW
            ).permute(0, 1, 3, 2, 4, 5, 6)
            if self.ksm_local_act == 'sigmoid':
                hr_att = hr_att_logit.sigmoid() * self.att_multi
            elif self.ksm_local_act == 'tanh':
                hr_att = 1 + hr_att_logit.tanh()
            else:
                raise NotImplementedError
        else:
            hr_att = 1

        b = x.size(0)
        batch_size, in_planes, depth, height, width = x.size()
        kT, kH, kW = self.kernel_size
        DFT_map = torch.zeros(
            (b, self.out_channels * kT, self.in_channels * kH * kW // 2 + 1, 2),
            device=x.device
        )

        kernel_attention = kernel_attention.reshape(b, self.param_ratio, self.kernel_num, -1)
        if hasattr(self, 'dft_weight'):
            dft_weight = self.dft_weight
        else:
            dft_weight = self.get_FDW()

        for i in range(self.param_ratio):
            indices = self.indices[i]
            if self.param_reduction < 1:
                w = dft_weight[i].reshape(self.kernel_num, -1, 2)[None]
                DFT_map[:, indices[0, :, :], indices[1, :, :]] += torch.stack(
                    [w[..., 0] * kernel_attention[:, i], w[..., 1] * kernel_attention[:, i]], dim=-1
                )
            else:
                w = dft_weight[i][indices[0, :, :], indices[1, :, :]][None] * self.alpha
                DFT_map[:, indices[0, :, :], indices[1, :, :]] += torch.stack(
                    [w[..., 0] * kernel_attention[:, i], w[..., 1] * kernel_attention[:, i]], dim=-1
                )

        adaptive_weights = torch.fft.irfft2(
            torch.view_as_complex(DFT_map), dim=(1, 2)
        ).reshape(batch_size, 1, self.out_channels, kT, self.in_channels, kH, kW)
        adaptive_weights = adaptive_weights.permute(0, 1, 2, 4, 3, 5, 6)

        if hasattr(self, 'FBM'):
            x = self.FBM(x)

        if self.out_channels * self.in_channels * kT * kH * kW < (
            in_planes + self.out_channels
        ) * depth * height * width:
            aggregate_weight = spatial_attention * channel_attention * filter_attention * adaptive_weights * hr_att
            aggregate_weight = torch.sum(aggregate_weight, dim=1)
            aggregate_weight = aggregate_weight.view(
                [-1, self.in_channels // self.groups, kT, kH, kW]
            )
            x = x.reshape(1, -1, depth, height, width)
            output = F.conv3d(
                x,
                weight=aggregate_weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups * batch_size
            )
            output = output.view(batch_size, self.out_channels, output.size(-3), output.size(-2), output.size(-1))
        else:
            aggregate_weight = spatial_attention * adaptive_weights * hr_att
            aggregate_weight = torch.sum(aggregate_weight, dim=1)
            if not isinstance(channel_attention, float):
                x = x * channel_attention.view(b, -1, 1, 1, 1)
            aggregate_weight = aggregate_weight.view(
                [-1, self.in_channels // self.groups, kT, kH, kW]
            )
            x = x.reshape(1, -1, depth, height, width)
            output = F.conv3d(
                x,
                weight=aggregate_weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups * batch_size
            )
            output = output.view(batch_size, self.out_channels, output.size(-3), output.size(-2), output.size(-1))
            if not isinstance(filter_attention, float):
                output = output * filter_attention.view(b, -1, 1, 1, 1)

        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1, 1)
        return output

if __name__ == '__main__':
    # x = torch.rand(4, 128, 64, 64) * 1
    x = torch.zeros((1, 32, 128, 128, 128)).cuda()
    # python
    m = FDConv(in_channels=32, out_channels=64, kernel_num=8, kernel_size=3, padding=1, bias=True).cuda()

    y = m(x)
    print("输入特征维度：", x.shape)
    print("输出特征维度：", y.shape)
