import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np


def draw_u_shape_model():
    fig, ax = plt.subplots(figsize=(22, 16))
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 24)
    ax.axis('off')

    # 颜色配置
    colors = {
        'enc_mednext': '#AEE2FF',  # 浅蓝 - MedNeXt 编码器
        'enc_rwkv': '#FFB4B4',  # 浅红 - RWKV 并行编码器
        'skip_fuse': '#E6E6FA',  # 浅紫 - Fusion 融合层
        'skip_cca': '#C1E1C1',  # 浅绿 - CCA 注意力层
        'dec_lrdu': '#FFFACD',  # 浅黄 - 解码器
        'output': '#FFDAB9'  # 蜜桃色 - 输出层
    }

    # 各层的高度 Y 坐标 (Level 0 ~ Level 4)
    y_levels = {
        'L0': 20,  # 16 channels (Highest Res)
        'L1': 16,  # 32 channels
        'L2': 12,  # 64 channels
        'L3': 8,  # 128 channels
        'L4': 3  # 256 channels (Bottleneck)
    }

    # 各列的 X 坐标
    x_enc = 3
    x_rwkv = 6.5
    x_fuse = 10
    x_cca = 13.5
    x_dec = 17
    x_out = 21

    def add_block(x, y, text, color, width=2.8, height=1.6):
        box = FancyBboxPatch((x - width / 2, y - height / 2), width, height,
                             boxstyle="round,pad=0.1", facecolor=color,
                             edgecolor='#333333', linewidth=1.5, zorder=3)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=10,
                fontweight='bold', zorder=4)

    def add_arrow(x1, y1, x2, y2, color='#555555', style='-|>'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=2, shrinkA=0, shrinkB=0), zorder=2)

    # ================= 1. 左侧 Encoder (Downward) =================
    # Input
    add_block(x_enc, 23, "Input Image\n(1×H×W×D)", '#FFFFFF')
    add_arrow(x_enc, 22.2, x_enc, y_levels['L0'] + 0.8)

    # MedNeXt Encoder Blocks
    add_block(x_enc, y_levels['L0'], "MedNeXt Stem\n+ Block 0 (16c)", colors['enc_mednext'])
    add_block(x_enc, y_levels['L1'], "MedNeXt Down 0\n+ Block 1 (32c)", colors['enc_mednext'])
    add_block(x_enc, y_levels['L2'], "MedNeXt Down 1\n+ Block 2 (64c)", colors['enc_mednext'])
    add_block(x_enc, y_levels['L3'], "MedNeXt Down 2\n+ Block 3 (128c)", colors['enc_mednext'])
    add_block(x_enc, y_levels['L4'], "Bottleneck\nDown 3 + Block (256c)", colors['enc_mednext'])

    # Encoder Down Arrows
    add_arrow(x_enc, y_levels['L0'] - 0.8, x_enc, y_levels['L1'] + 0.8)
    add_arrow(x_enc, y_levels['L1'] - 0.8, x_enc, y_levels['L2'] + 0.8)
    add_arrow(x_enc, y_levels['L2'] - 0.8, x_enc, y_levels['L3'] + 0.8)
    add_arrow(x_enc, y_levels['L3'] - 0.8, x_enc, y_levels['L4'] + 0.8)

    # RWKV Parallel Encoder (Starts at L2)
    add_block(x_rwkv, y_levels['L2'], "RWKV Stage 2\nGLSP3D (64c)", colors['enc_rwkv'])
    add_block(x_rwkv, y_levels['L3'], "RWKV Stage 3\nGLSP3D (128c)", colors['enc_rwkv'])

    # Connect Enc to RWKV
    add_arrow(x_enc + 1.4, y_levels['L2'], x_rwkv - 1.4, y_levels['L2'])  # MedNext -> RWKV
    add_arrow(x_rwkv, y_levels['L2'] - 0.8, x_rwkv, y_levels['L3'] + 0.8)  # RWKV Down

    # ================= 2. 中间 Skip Connections =================
    # Fusion Blocks (Fuse 2, 3, 4)
    add_block(x_fuse, y_levels['L2'], "Fusion Block 2\n(128c -> 64c)", colors['skip_fuse'])
    add_block(x_fuse, y_levels['L3'], "Fusion Block 3\n(256c -> 128c)", colors['skip_fuse'])
    add_block(x_fuse, y_levels['L4'], "Fusion Block 4\n(512c -> 256c)", colors['skip_fuse'])

    # Connect to Fusion
    add_arrow(x_rwkv + 1.4, y_levels['L2'], x_fuse - 1.4, y_levels['L2'])
    add_arrow(x_rwkv + 1.4, y_levels['L3'], x_fuse - 1.4, y_levels['L3'])
    add_arrow(x_enc + 1.4, y_levels['L4'], x_fuse - 1.4, y_levels['L4'])

    # CCA Blocks (Channel Cross Attention 0,1,2,3)
    add_block(x_cca, y_levels['L0'], "CCA 0\nAttention", colors['skip_cca'])
    add_block(x_cca, y_levels['L1'], "CCA 1\nAttention", colors['skip_cca'])
    add_block(x_cca, y_levels['L2'], "CCA 2\nAttention", colors['skip_cca'])
    add_block(x_cca, y_levels['L3'], "CCA 3\nAttention", colors['skip_cca'])

    # Connections to CCA
    add_arrow(x_enc + 1.4, y_levels['L0'], x_cca - 1.4, y_levels['L0'], style='--|>')  # Direct skip L0
    add_arrow(x_enc + 1.4, y_levels['L1'], x_cca - 1.4, y_levels['L1'], style='--|>')  # Direct skip L1
    add_arrow(x_fuse + 1.4, y_levels['L2'], x_cca - 1.4, y_levels['L2'])  # Fusion to CCA
    add_arrow(x_fuse + 1.4, y_levels['L3'], x_cca - 1.4, y_levels['L3'])

    # CCA to Decoder
    add_arrow(x_cca + 1.4, y_levels['L0'], x_dec - 1.4, y_levels['L0'])
    add_arrow(x_cca + 1.4, y_levels['L1'], x_dec - 1.4, y_levels['L1'])
    add_arrow(x_cca + 1.4, y_levels['L2'], x_dec - 1.4, y_levels['L2'])
    add_arrow(x_cca + 1.4, y_levels['L3'], x_dec - 1.4, y_levels['L3'])

    # ================= 3. 右侧 Decoder (Upward) =================
    add_block(x_dec, y_levels['L4'], "Bottleneck Base\n(256c)", colors['dec_lrdu'])
    add_block(x_dec, y_levels['L3'], "LRDU Up 3 +\nDec Block 3 (128c)", colors['dec_lrdu'])
    add_block(x_dec, y_levels['L2'], "LRDU Up 2 +\nDec Block 2 (64c)", colors['dec_lrdu'])
    add_block(x_dec, y_levels['L1'], "LRDU Up 1 +\nDec Block 1 (32c)", colors['dec_lrdu'])
    add_block(x_dec, y_levels['L0'], "LRDU Up 0 +\nDec Block 0 (16c)", colors['dec_lrdu'])

    # Decoder Up Arrows
    add_arrow(x_fuse + 1.4, y_levels['L4'], x_dec - 1.4, y_levels['L4'])  # Route Bottom to Dec
    add_arrow(x_dec, y_levels['L4'] + 0.8, x_dec, y_levels['L3'] - 0.8)
    add_arrow(x_dec, y_levels['L3'] + 0.8, x_dec, y_levels['L2'] - 0.8)
    add_arrow(x_dec, y_levels['L2'] + 0.8, x_dec, y_levels['L1'] - 0.8)
    add_arrow(x_dec, y_levels['L1'] + 0.8, x_dec, y_levels['L0'] - 0.8)

    # ================= 4. 输出层 Multi-scale Outputs =================
    add_block(x_out, y_levels['L3'], "Out 4\n(2×H/8)", colors['output'], width=1.8, height=1.2)
    add_block(x_out, y_levels['L2'], "Out 3\n(2×H/4)", colors['output'], width=1.8, height=1.2)
    add_block(x_out, y_levels['L1'], "Out 2\n(2×H/2)", colors['output'], width=1.8, height=1.2)
    add_block(x_out, y_levels['L0'] - 1, "Out 1\n(2×H)", colors['output'], width=1.8, height=1.2)
    add_block(x_out, y_levels['L0'] + 1.5, "Out 0\n(Final 2×H)", '#FF8C00', width=1.8, height=1.2)  # Final Out

    add_arrow(x_dec + 1.4, y_levels['L3'], x_out - 0.9, y_levels['L3'])
    add_arrow(x_dec + 1.4, y_levels['L2'], x_out - 0.9, y_levels['L2'])
    add_arrow(x_dec + 1.4, y_levels['L1'], x_out - 0.9, y_levels['L1'])
    add_arrow(x_dec + 1.4, y_levels['L0'], x_out - 0.9, y_levels['L0'] - 1)
    add_arrow(x_dec + 1.4, y_levels['L0'], x_out - 0.9, y_levels['L0'] + 1.5)

    # ================= 图例标题区 =================
    plt.title("U-Shape Architecture of Double_CCA_UPSam_RWKV_MedNeXt", fontsize=22, fontweight='bold', pad=20)

    legend_elements = [
        patches.Patch(facecolor=colors['enc_mednext'], edgecolor='#333', label='Encoder (MedNeXt)'),
        patches.Patch(facecolor=colors['enc_rwkv'], edgecolor='#333', label='Parallel Encoder (RWKV)'),
        patches.Patch(facecolor=colors['skip_fuse'], edgecolor='#333', label='Fusion Layers'),
        patches.Patch(facecolor=colors['skip_cca'], edgecolor='#333', label='Channel Cross Attention (CCA)'),
        patches.Patch(facecolor=colors['dec_lrdu'], edgecolor='#333', label='Decoder (LRDU3D & MedNeXt)'),
        patches.Patch(facecolor=colors['output'], edgecolor='#333', label='Deep Supervision Outputs')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=1).set_zorder(5)

    plt.tight_layout()
    plt.savefig('u_shape_model_structure.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    draw_u_shape_model()