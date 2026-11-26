"""Plot the starting occupancy maps and mean performance by strategy."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_settings import apply_plot_style
from unknown_boxes import UNKNOWN_OCCUPANCY
from world import GOAL, H, START, W


@apply_plot_style("occupancy")
def save_probability_maps(prior, output):
    unknown = np.full(prior.shape, UNKNOWN_OCCUPANCY)
    unknown[START] = 0.0
    fig, axes = plt.subplots(1, 2, sharey=True)
    fig.suptitle("Initial cell occupancy probabilities")
    for ax, probabilities, label in zip(axes, [prior, unknown], ["Rough map", "Unknown boxes"]):
        im = ax.imshow(probabilities, origin="lower", extent=(0, W, 0, H),
                       vmin=0, vmax=1, interpolation="nearest")
        ax.set(title=label, xlabel="x (cm)", xlim=(0, W), ylim=(0, H))
        for cell, marker, label in [(START, "o", "Start"), (GOAL, "*", "Goal")]:
            xy = (cell[1] + .5, cell[0] + .5)
            ax.plot(*xy, marker=marker, color="white",
                    markersize=8, linestyle="none")
            ax.annotate(label, xy, xytext=(0, -15), textcoords="offset points",
                        ha="center", color="white", fontsize=8)
    axes[0].set_ylabel("y (cm)")
    fig.colorbar(im, ax=axes, shrink=.8, label="Probability of occupancy")
    fig.savefig(output)
    plt.close(fig)


@apply_plot_style("times")
def save_strategy_times(averages, output):
    fig, ax = plt.subplots()
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
    trial_count = averages[0]["trials"]
    time_label = "Simulated time to goal" if trial_count == 1 else "Mean simulated time to goal"
    trials_label = "trial" if trial_count == 1 else "trials"
    ax.set_xlabel(f"{time_label} (s)")
    ax.set_title(f"Time to goal by navigation strategy  ·  {trial_count} {trials_label} each",
                 loc="left")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.savefig(output)
    plt.close(fig)


@apply_plot_style("strategy_metrics")
def save_strategy_metrics(averages, output):
    fig, axes = plt.subplots(1, 3, sharey=True, width_ratios=(1.2, 1.2, 1))
    scenarios = ["Rough map" if row["simulation"] == "rough_map"
                 else f"Unknown, {row['step_limit_cm']} cm" for row in averages]
    labels = [f"{scenario}  |  {row['planner']}" for scenario, row in zip(scenarios, averages)]
    colors = ["#087888" if row["planner"] == "A*" else "#a44b22" for row in averages]
    prefix = "" if averages[0]["trials"] == 1 else "Mean "
    axes[0].invert_yaxis()
    metrics = (
        ("travel_cm", "total distance (cm)", ".2f"),
        ("time_s", "simulated time (s)", ".2f"),
        ("contacts", "obstacle contacts", "g"),
    )
    for ax, (field, title, number_format) in zip(axes, metrics):
        values = [row[field] for row in averages]
        bars = ax.barh(labels, values, color=colors, height=.66)
        ax.bar_label(bars, fmt="%" + number_format, padding=3, fontsize=9)
        ax.set_xlim(0, max(1, max(values)) * 1.30)
        ax.set_xlabel((prefix + title).capitalize())
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.savefig(output)
    plt.close(fig)
