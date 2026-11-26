"""Plot the starting occupancy maps and mean time by strategy."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from unknown_boxes import UNKNOWN_OCCUPANCY
from world import GOAL, H, START, W


def save_probability_maps(prior, output):
    unknown = np.full(prior.shape, UNKNOWN_OCCUPANCY)
    unknown[START] = 0.0
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), layout="constrained", sharey=True)
    fig.suptitle("Initial cell occupancy probabilities", fontsize=16, fontweight="bold")
    for ax, probabilities, label in zip(axes, [prior, unknown], ["Rough map", "Unknown boxes"]):
        im = ax.imshow(probabilities, origin="lower", extent=(0, W, 0, H),
                       vmin=0, vmax=1, cmap="viridis", interpolation="nearest")
        ax.set(title=label, xlabel="x (cm)", xlim=(0, W), ylim=(0, H))
        for cell, marker, label in [(START, "o", "Start"), (GOAL, "*", "Goal")]:
            xy = (cell[1] + .5, cell[0] + .5)
            ax.plot(*xy, marker=marker, color="white", markeredgecolor="#222222",
                    markersize=8, linestyle="none")
            ax.annotate(label, xy, xytext=(0, -15), textcoords="offset points",
                        ha="center", color="white", fontsize=8)
    axes[0].set_ylabel("y (cm)")
    fig.colorbar(im, ax=axes, shrink=.8, label="Probability of occupancy")
    fig.savefig(output, dpi=170, facecolor="white")
    plt.close(fig)


def save_strategy_times(averages, output):
    fig, ax = plt.subplots(figsize=(9.2, 5.7), layout="constrained")
    labels = []
    values = []
    colors = []
    for row in averages:
        rough = row["simulation"] == "rough_map"
        context = "rough map" if rough else f"unknown, {row['step_limit_cm']} cm"
        labels.append(f"{row['planner']}  ·  {context}")
        values.append(row["time_s"])
        colors.append("#087888" if row["planner"] == "A*" else "#a44b22")
    bars = ax.barh(labels, values, color=colors, height=.62)
    ax.bar_label(bars, labels=[f"{value:.2f} s" for value in values], padding=5, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.16)
    ax.set_xlabel("Mean simulated time to goal (s)")
    ax.set_title(f"Time to goal by navigation strategy  ·  {averages[0]['trials']} trials each",
                 loc="left", fontweight="bold", fontsize=14, pad=11)
    ax.grid(axis="x", color="#d7e0e7", alpha=.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.savefig(output, dpi=170, facecolor="white")
    plt.close(fig)
