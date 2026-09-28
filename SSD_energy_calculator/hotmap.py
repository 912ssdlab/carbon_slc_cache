#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Draw a 12 SSD × 5 factor heatmap.

Cell value:
    Delta Gamma(device, factor)
        = normalized parameter difference
        * fitted factor effect

Put this file in the same directory as gamma.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

import gamma


# ============================================================
# 0. Global style settings
# ============================================================

FONT_FAMILY = "Arial"
MAIN_FONT_SIZE = 24     # PPT-like 25 pt for title/axes/ticks/colorbar
CELL_FONT_SIZE = 20       # keep smaller, otherwise the heatmap becomes crowded

BLUE_COLOR = "#1E4782"
MID_COLOR = "#FFFFFF"     # color for 0.0
RED_COLOR = "#800000"

plt.rcParams["font.family"] = FONT_FAMILY
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


# ============================================================
# 1. Run the Gamma fitting from gamma.py
# ============================================================

gamma.validate_tables()

target_gamma, _ = gamma.infer_all_target_gammas()

fit_result = gamma.fit_iterative_effects(
    target_gamma,
    max_rounds=gamma.MAX_ITERATIVE_ROUNDS,
    convergence_tol=gamma.CONVERGENCE_TOLERANCE,
)

best_effects = fit_result["effects"]


# ============================================================
# 2. Select the 12 experimental SSDs
# NV2000ME5 is the baseline, so it is excluded.
# ============================================================

devices = [
    device
    for device in gamma.ARCH_TABLE
    if device != gamma.BASELINE_DEVICE
]


# ============================================================
# 3. Five factors
# ============================================================

factor_keys = gamma.FACTOR_ORDER

factor_labels = [
    "SLC",
    "UWR",
    "TWRGap",
    "OP",
    "Max.Parall",
]


# ============================================================
# 4. Calculate factor contribution for every SSD
# Delta Gamma = normalized_change × fitted_effect
# ============================================================

matrix = []

for device in devices:
    normalized = gamma.normalized_changes(device)

    row = []
    for factor in factor_keys:
        contribution = normalized[factor] * best_effects[factor]
        row.append(contribution)

    matrix.append(row)

matrix = np.array(matrix)

raw_matrix = matrix.copy()

baseline_values = np.zeros(matrix.shape[1])
baseline_indices = np.zeros(matrix.shape[1], dtype=int)


# First three columns: minimum value as baseline
baseline_values[:3] = raw_matrix[:, :3].min(axis=0)
baseline_indices[:3] = raw_matrix[:, :3].argmin(axis=0)


# Last two columns: maximum value as baseline
baseline_values[3:] = raw_matrix[:, 3:].max(axis=0)
baseline_indices[3:] = raw_matrix[:, 3:].argmax(axis=0)


#Subtract each column's own plotting baseline
matrix = raw_matrix - baseline_values


# Remove tiny floating-point errors such as -0.000000
matrix[np.abs(matrix) < 1e-12] = 0.0


# Print which SSD is used as the plotting baseline
baseline_devices = [
    devices[i]
    for i in baseline_indices
]

print("\n==============================")
print("Plot baseline for each factor")
print("==============================")

for factor, device, value in zip(
    factor_labels,
    baseline_devices,
    baseline_values,
):
    print(
        f"{factor:<12} "
        f"baseline = {device:<20} "
        f"value = {value:+.6f}"
    )
# ============================================================
# 5. Convert to DataFrame
# ============================================================

df = pd.DataFrame(
    matrix,
    index=devices,
    columns=factor_labels,
)

print("\n==============================")
print("12 × 5 Gamma contribution")
print("==============================")
print(df.round(4))

df.to_csv(
    "gamma_heatmap_values.csv",
    encoding="utf-8-sig",
    float_format="%.6f",
)


# ============================================================
# 6. Print fitted factor effects
# ============================================================

print("\n==============================")
print("Best fitted Gamma effects")
print("==============================")

for factor in factor_keys:
    factor_name = gamma.FACTOR_INFO[factor]["name"]
    change = gamma.FACTOR_INFO[factor]["change_text"]
    effect = best_effects[factor]
    print(f"{factor_name:30s} {change:12s} {effect:+.4f}")




custom_cmap = LinearSegmentedColormap.from_list(
    "blue_gray_red_custom",
    [BLUE_COLOR, MID_COLOR, RED_COLOR]
)


# ============================================================
# 8. Draw heatmap
# ============================================================

fig, ax = plt.subplots(figsize=(16, 10))

norm = TwoSlopeNorm(
    vmin=-0.6,
    vcenter=0.0,
    vmax=0.6,
)

heatmap = ax.imshow(
    matrix,
    aspect=0.3,
    cmap=custom_cmap,
    norm=norm,
)
ax.set_xticks(
    np.arange(-0.5, matrix.shape[1], 1),
    minor=True,
)

ax.set_yticks(
    np.arange(-0.5, matrix.shape[0], 1),
    minor=True,
)

# ax.grid(
#     which="minor",
#     color="white",
#     linewidth=2.0,
# )

ax.tick_params(
    which="minor",
    bottom=False,
    left=False,
)

# ============================================================
# 9. Axis labels
# ============================================================

ax.set_xticks(np.arange(len(factor_labels)))
ax.set_xticklabels(
    factor_labels,
    rotation=0,
    ha="center",
    fontsize=MAIN_FONT_SIZE,
    fontweight="bold",
)

ax.set_yticks(np.arange(len(devices)))
ax.set_yticklabels(
    devices,
    fontsize=MAIN_FONT_SIZE,
    fontweight="bold",
)
ax.tick_params(
    axis="x",
    which="both",
    length=0,
    pad=10,
)

ax.tick_params(
    axis="y",
    which="both",
    length=0,
    pad=10,
)
ax.set_xlabel(
    "Incremental Impact on BVF",
    fontsize=MAIN_FONT_SIZE,
   labelpad=15,
    fontweight="bold",
)




# ============================================================
# 10. Put numerical values in cells
# ============================================================

for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        value = matrix[i, j]

        # darker cells use white text, lighter cells use black text
        text_color = "white" if abs(value) > 0.22 else "black"

        ax.text(
            j,
            i,
            f"{value:+.3f}",
            ha="center",
            va="center",
            fontsize=CELL_FONT_SIZE,
            color=text_color,
        )


# ============================================================
# 11. Color bar
# ============================================================
cax = ax.inset_axes([
    0,      # left
    1.02,   # bottom，控制离主图多远
    1.0,    # width，1.0 = 和主图一样长
    0.025,  # height，只控制热度条厚度
])
cbar = fig.colorbar(
    heatmap,
    cax=cax,
    location="top",
    orientation="horizontal",
)
ticks = np.arange(
    np.ceil(-0.6 / 0.2) * 0.2,
    np.floor(0.6 / 0.2) * 0.2 + 0.001,
    0.2,
)

cbar.set_ticks(ticks)
cbar.ax.invert_xaxis()
cbar.set_label(
    r"Contribution to BVF ($\Delta BVF$)",
    fontsize=MAIN_FONT_SIZE,
    labelpad=15,
    #fontweight="bold",
)
cbar.ax.tick_params(length=0, labelsize=MAIN_FONT_SIZE)


# ============================================================
# 12. Layout and output
# ============================================================

plt.tight_layout()

# PDF file name as requested
plt.savefig(
    "write_bvf_heatmap.pdf",
    bbox_inches="tight",
)

# optional PNG
plt.savefig(
    "write_bvf_heatmap.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
