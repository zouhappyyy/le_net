import matplotlib.pyplot as plt
import numpy as np

models = [
    "nnUNet",
    "MedNeXt",
    "SwinUNETRv2",
    "UMamba",
    "RWKV-UNet",
    "RSM-NeXt",
    "FD-RSM-NeXt",
]

params = np.array([31.19, 5.54, 69.95, 24.51, 20.67, 13.43, 17.32])
flops = np.array([534.03, 138.07, 779.83, 1009.24, 218.56, 141.09, 190.75])
dice = np.array([80.57, 82.72, 79.17, 81.10, 82.90, 84.15, 85.27])

fig, ax = plt.subplots(figsize=(5.6, 4.1), dpi=300)

# Bubble size is proportional to parameter count.
sizes = params * 14

ax.scatter(flops, dice, s=sizes, alpha=0.75, edgecolors="none", linewidths=0)

# Keep labels centered below each bubble while staggering crowded regions.
label_offsets = {
    "nnUNet": (0, -0.24),
    "MedNeXt": (0, -0.40),
    "SwinUNETRv2": (0, -0.24),
    "UMamba": (0, -0.24),
    "RWKV-UNet": (0, -0.28),
    "RSM-NeXt": (0, -0.20),
    "FD-RSM-NeXt": (0, -0.48),
}

for i, model in enumerate(models):
    dx, dy = label_offsets.get(model, (0, -0.24))
    ax.text(
        flops[i] + dx,
        dice[i] + dy,
        model,
        fontsize=10,
        ha="center",
        va="center",
    )

ax.set_xlabel("FLOPs (G)", fontsize=10)
ax.set_ylabel("Dice (%)", fontsize=10)
ax.set_title("模型精度与计算复杂度对比", fontsize=11)
ax.grid(True, linestyle="--", alpha=0.4)
ax.set_xlim(0, 1120)
ax.set_ylim(78.45, 85.75)

# Highlight the proposed model.
idx = models.index("FD-RSM-NeXt")
ax.scatter(
    flops[idx],
    dice[idx],
    s=sizes[idx] * 1.3,
    edgecolors="red",
    linewidths=1.8,
    facecolors="none",
)

plt.tight_layout()
plt.savefig("./tools/model_complexity_dice_comparison.png", dpi=300, bbox_inches="tight")
plt.savefig("./tools/model_complexity_dice_comparison.pdf", bbox_inches="tight")
plt.show()
