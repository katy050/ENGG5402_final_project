"""Draw the paths, contact approaches, and fixed obstacles for both scenarios."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np

from world import GOAL, H, START, W


def xy(points):
    return np.asarray(points, dtype=float)[:, ::-1] + 0.5


def save_path_plot(truth, traces, output):
    trial_count = len(next(iter(traces.values())))
    colors = ("#066f89", "#2c82a5", "#3386a8", "#447e9a", "#163f58")
    row_specs = [("rough_map", None, "Rough map")]
    row_specs.extend(("unknown_boxes", limit, f"Unknown boxes · {limit} cm edge steps")
                     for limit in (1, 3, 5))
    fig, axes = plt.subplots(4, 2, figsize=(12.5, 14.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=.09, right=.98, top=.895, bottom=.085, hspace=.31, wspace=.12)
    fig.text(.09, .972, f"Touch-navigation paths across {trial_count} trials",
             fontsize=22, fontweight="bold", color="#12293e")
    fig.text(.09, .939,
             "Same fixed boxes in both simulations. Identical trial paths are overlaid.",
             fontsize=11, color="#46596b")
    for i, (simulation, limit, label) in enumerate(row_specs):
        for j, planner in enumerate(("A*", "MDP")):
            ax = axes[i, j]
            ax.set_title(f"{label} — {planner}", loc="left", fontweight="bold", fontsize=11)
            ax.set(xlim=(0, W), ylim=(0, H), aspect="equal")
            ax.set_xticks(range(0, W + 1, 10))
            ax.set_yticks(range(0, H + 1, 10))
            ax.grid(color="#dce5ec", linewidth=.6)
            ax.tick_params(colors="#46596b", labelsize=8, length=3)
            for spine in ax.spines.values():
                spine.set_color("#a8b7c5")
            for number, (row, col, height, width) in enumerate(truth, 1):
                ax.add_patch(Rectangle((col, row), width, height, facecolor="#dfe7ed",
                                       edgecolor="#97a9b9", lw=.8, zorder=2))
                ax.text(col + width / 2, row + height / 2, f"B{number}",
                        ha="center", va="center", fontsize=8, color="#435466", zorder=3)
            for number, result in enumerate(traces[(simulation, planner, limit)]):
                coordinates = xy(result["path"])
                ax.plot(*coordinates.T, lw=1.7, alpha=.58,
                        color=colors[number % len(colors)], zorder=4)
                for segment in result.get("edge_moves", []):
                    endpoints = xy(segment)
                    ax.plot(*endpoints.T, color="#5941a9", lw=1.9,
                            marker=".", markersize=2.6, zorder=5)
                for _, checked_cell, blocked in result.get("edge_checks", []):
                    if not blocked:
                        ax.plot(*xy([checked_cell])[0], marker="s", markersize=3,
                                markerfacecolor="white", markeredgecolor="#198269",
                                linestyle="none", zorder=6)
                for segment in result["touches"]:
                    approach = xy(segment)
                    ax.plot(*approach.T, color="#c44636", ls=":", lw=1, alpha=.5, zorder=5)
                    ax.plot(*approach[-1], "x", color="#a72f22", ms=4, mew=1, zorder=6)
            ax.plot(*xy([START])[0], "o", color="#198269", ms=5, zorder=7)
            ax.plot(*xy([GOAL])[0], "*", color="#82542b", ms=8, zorder=7)
            if j == 0:
                ax.set_ylabel("y (cm)")
            if i == 3:
                ax.set_xlabel("x (cm)")
    handles = [
        Line2D([], [], color="#066f89", lw=2, label="Trial paths"),
        Line2D([], [], color="#5941a9", lw=2, marker=".", label="Box-edge moves"),
        Line2D([], [], color="#a72f22", ls=":", marker="x", label="Contact approach and return"),
        Line2D([], [], color="#198269", marker="s", markerfacecolor="white",
               ls="", label="Clear check"),
        Patch(facecolor="#dfe7ed", edgecolor="#97a9b9", label="Actual box"),
        Line2D([], [], color="#198269", marker="o", ls="", label="Start"),
        Line2D([], [], color="#82542b", marker="*", ls="", label="Goal"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .023),
               ncol=7, frameon=False, fontsize=8)
    fig.savefig(output, dpi=170, facecolor="white")
    plt.close(fig)
