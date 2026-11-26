"""Run this file from the updated four-box simulation folder.

The simulator uses the actual fixed obstacles for contact checks. Both
planners receive the same initial grid of cell occupancy probabilities.
"""
from itertools import combinations
from math import hypot
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np

from navsim import run_comparison
from plot_settings import apply_plot_style
from world import GOAL, H, START, W, box_mask, offsets_disk

ERROR_CM = 5
# Each box is (row, column, height, width) in 1 cm grid units.
NOMINAL = [
    (20, 5, 5, 10),    # B1
    (11, 26, 12, 11),  # B2
    (23, 37, 6, 6),    # B3
    (13, 46, 11, 8),   # B4
]
SHIFTS = [(-1, 3), (-1, -3), (-4, 0), (0, 0)]
COLORS = ["#007bac", "#bd5a26", "#0d7963", "#5656a0"]


def shifted(box, shift):
    row, col, height, width = box
    dr, dc = shift
    return row + dr, col + dc, height, width


def box_gap(a, b):
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    dx = max(0, bc - ac - aw, ac - bc - bw)
    dy = max(0, br - ar - ah, ar - br - bh)
    return hypot(dx, dy)


def draw_box(ax, box, color, label=None, nominal=False):
    row, col, height, width = box
    ax.add_patch(Rectangle(
        (col, row), width, height,
        facecolor="none" if nominal else color, edgecolor=color,
        linestyle=(0, (4, 2)) if nominal else "solid", linewidth=1.6, zorder=4,
    ))
    if label:
        on_fill = not nominal
        ax.text(col + width / 2, row + height / 2, label,
                ha="center", va="center", fontsize=16,
                color="white" if on_fill else "#142438", zorder=5)


@apply_plot_style("map")
def draw_map(truth, footprint_count, output):
    fig, axes = plt.subplots(1, 2)
    # fig.subplots_adjust(left=.067, right=.981, bottom=.22, top=.79, wspace=.12)
    # fig.text(.067, .94, "Four-box touch navigation map",
    #          fontsize=24, fontweight="bold", color="#142438")
    fig.suptitle("Four-box touch navigation map")
    # fig.text(.067, .878, "60 × 40 cm workspace   •   1 cm grid   •   5 cm placement-error bound",
    #          fontsize=12)
    axes[0].set_title("Possible footprints from the rough map", loc="left")
    axes[1].set_title("Actual fixed positions", loc="left")
    axes[0].imshow(np.minimum(footprint_count, 2), origin="lower",
                   extent=(0, W, 0, H), interpolation="nearest", vmin=0, vmax=2,
                   cmap=ListedColormap(["#ffffff", "#d4e6f3", "#c4a1dc"]))
    for ax in axes:
        ax.set(xlim=(0, W), ylim=(0, H), aspect="equal", xlabel="x (cm)")
        ax.set_xticks(range(0, W + 1, 10))
        ax.set_yticks(range(0, H + 1, 10))
        for cell, label, marker, color in [
            (START, "Start", "o", "green"),
            (GOAL, "Goal", "*", "red"),
        ]:
            row, col = cell
            ax.plot(col + .5, row + .5, marker, markersize=9, color=color, zorder=7)
            ax.annotate(label, (col + .5, row + .5), xytext=(0, -14),
                        textcoords="offset points", ha="center", color=color,
                        fontsize=9, zorder=7)
    axes[0].set_ylabel("y (cm)")
    axes[1].plot([START[1] + .5, GOAL[1] + .5], [START[0] + .5, GOAL[0] + .5],
                 color="#6a7786", ls=":", lw=1.5, zorder=3)
    for number, (nominal, actual, color) in enumerate(zip(NOMINAL, truth, COLORS), 1):
        label = f"B{number}"
        draw_box(axes[0], nominal, color, label, nominal=True)
        draw_box(axes[1], actual, color, label)
    fig.legend(handles=[
        Line2D([0], [0], ls="--", label="Nominal box"),
        Patch(facecolor="#d4e6f3", label="Cell could contain one box"),
        Patch(facecolor="#c4a1dc", label="Possible footprints overlap"),
        Patch(facecolor="#007bac", label="Actual box"),
        Line2D([0], [0], ls=":", label="Direct start–goal line"),
    ], bbox_to_anchor=(.5, 0), ncol=5)
    fig.savefig(output)
    plt.close(fig)


def overlap_count(first_masks, second_masks):
    return sum(bool(np.any(a & b)) for a in first_masks for b in second_masks)


def rough_probability_map():
    offsets = offsets_disk(ERROR_CM)
    footprint_probabilities = np.array([
        np.mean([box_mask(box, *offset) for offset in offsets], axis=0) for box in NOMINAL
    ])
    return 1.0 - np.prod(1.0 - footprint_probabilities, axis=0)


def nominal_occupancy(offsets):
    box_footprints = np.array([
        [box_mask(box, *offset) for offset in offsets] for box in NOMINAL
    ])
    footprint_count = box_footprints.any(axis=1).sum(axis=0)
    box_occupancy = box_footprints.mean(axis=1)
    prior = 1.0 - np.prod(1.0 - box_occupancy, axis=0)
    return box_footprints, footprint_count, prior


def rough_probability_map():
    return nominal_occupancy(offsets_disk(ERROR_CM))[2]


def fixed_layout():
    offsets = offsets_disk(ERROR_CM)
    truth = [shifted(box, shift) for box, shift in zip(NOMINAL, SHIFTS)]
    assert all(shift in offsets for shift in SHIFTS)
    for box in truth:
        row, col, height, width = box
        assert 0 <= row < row + height <= H and 0 <= col < col + width <= W
    gaps = [box_gap(a, b) for a, b in combinations(truth, 2)]
    assert min(gaps) >= 3
    box_footprints, footprint_count, prior = nominal_occupancy(offsets)
    assert np.any(footprint_count >= 2)
    total = len(offsets)**2
    overlap_left = overlap_count(*box_footprints[:2])
    overlap_right = overlap_count(*box_footprints[2:])
    assert overlap_left == 0
    assert 0 < overlap_right / total <= 0.10
    true_mask = np.logical_or.reduce([box_mask(box) for box in truth])
    assert not true_mask[START] and not true_mask[GOAL]
    assert START[0] == GOAL[0]
    direct_cols = slice(START[1] + 1, GOAL[1])
    direct_obstacles = [index for index, box in enumerate(truth, 1)
                        if box_mask(box)[START[0], direct_cols].any()]
    assert len(direct_obstacles) == len(truth)
    return truth, true_mask, prior, footprint_count, min(gaps), overlap_left, overlap_right, total


def main():
    truth, true_mask, prior, footprint_count, gap, overlap_left, overlap_right, total = fixed_layout()
    figures_dir = Path.cwd() / "figures"
    figures_dir.mkdir(exist_ok=True)
    output = figures_dir / "four_box_touch_navigation_map.png"
    draw_map(truth, footprint_count, output)
    print(f"Map: {output}")
    print(f"Minimum actual edge-to-edge gap: {gap:g} cm")
    print(f"Prior B1/B2 overlap: {overlap_left / total:.2%} ({overlap_left}/{total})")
    print(f"Prior B3/B4 overlap: {overlap_right / total:.2%} ({overlap_right}/{total})")
    for planner, result in run_comparison(prior, true_mask).items():
        print(f"{planner:16} reached={result['success']}  contacts={result['contacts']}  "
              f"path={result['length']:.2f} cm  time={result['time_total']:.2f} s")


if __name__ == "__main__":
    main()
