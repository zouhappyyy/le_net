import numpy as np
import matplotlib.pyplot as plt

# Input data
centers = ["Center 1", "Center 2"]

data = np.array([
    [85.44, 86.86, 85.28, 10.147, 1.979],
    [83.60, 88.21, 81.87, 9.083, 1.762],
])

metric_names = ["Dice", "Sensitivity", "Precision", "HD95", "ASSD"]


def normalize_for_radar(values: np.ndarray) -> np.ndarray:
    """
    Map all metrics into a shared 0-100 radar scale.

    - Dice / Sensitivity / Precision are already percentages.
    - HD95 / ASSD are lower-is-better, so use min / x to turn them into
      higher-is-better scores without producing zeros for the best center.
    """
    values = values.astype(float)
    normalized = np.zeros_like(values)

    normalized[:, :3] = values[:, :3]

    for col in [3, 4]:
        min_val = values[:, col].min()
        normalized[:, col] = (min_val / values[:, col]) * 100.0

    return normalized


data_plot = normalize_for_radar(data)

angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
angles += angles[:1]

plt.figure(figsize=(6, 6))
ax = plt.subplot(111, polar=True)

for i, center in enumerate(centers):
    values = data_plot[i].tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, label=center)
    ax.fill(angles, values, alpha=0.15)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metric_names, fontsize=11)
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=9)
ax.grid(True)

plt.title("Cross-center Performance Radar", fontsize=14, pad=20)
plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

plt.tight_layout()
plt.savefig("radar_reverse_axis.png", dpi=300, bbox_inches="tight")
plt.show()
