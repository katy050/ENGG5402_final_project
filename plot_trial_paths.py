"""Draw the paths, contact approaches, and fixed obstacles for both scenarios."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np

from navsim import SPEED_CM_S
from unknown_boxes import STEP_LIMITS
from world import GOAL, H, START, W

PATH_ROWS = [("rough_map", None, "Rough map")]
PATH_ROWS.extend(("unknown_boxes", limit, f"Unknown boxes · {limit} cm edge steps")
                 for limit in STEP_LIMITS)


def xy(points):
    return np.asarray(points, dtype=float).reshape(-1, 2)[:, ::-1] + 0.5


def draw_workspace(ax, truth):
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
    ax.plot(*xy([START])[0], "o", color="#198269", ms=5, zorder=7, label="Start")
    ax.plot(*xy([GOAL])[0], "*", color="#82542b", ms=8, zorder=7, label="Goal")


def save_path_plot(truth, traces, output):
    trial_count = len(next(iter(traces.values())))
    colors = ("#066f89", "#2c82a5", "#3386a8", "#447e9a", "#163f58")
    fig, axes = plt.subplots(len(PATH_ROWS), 2, figsize=(12.5, 14.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=.09, right=.98, top=.895, bottom=.085, hspace=.31, wspace=.12)
    fig.text(.09, .972, f"Touch-navigation paths across {trial_count} trials",
             fontsize=22, fontweight="bold", color="#12293e")
    fig.text(.09, .939,
             "Same fixed boxes in both simulations. Identical trial paths are overlaid.",
             fontsize=11, color="#46596b")
    for i, (simulation, limit, label) in enumerate(PATH_ROWS):
        for j, planner in enumerate(("A*", "MDP")):
            ax = axes[i, j]
            ax.set_title(f"{label} — {planner}", loc="left", fontweight="bold", fontsize=11)
            draw_workspace(ax, truth)
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
            if j == 0:
                ax.set_ylabel("y (cm)")
            if i == len(PATH_ROWS) - 1:
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


def animate_trial(ax, result):
    attempts = result["attempts"]
    starts = xy([attempt.start for attempt in attempts])
    ends = xy([attempt.end for attempt in attempts])
    segments = np.stack([starts, ends], axis=1)
    times = np.r_[0, np.cumsum([attempt.seconds for attempt in attempts])]
    outbound = np.linalg.norm(ends - starts, axis=1) / SPEED_CM_S
    contacts = np.array([attempt.contact for attempt in attempts], dtype=bool)
    phases = np.array([attempt.phase for attempt in attempts], dtype=str)
    edge_moves = ~contacts & (phases == "Edge follow")
    clear_checks = ~contacts & (phases == "Edge check")

    route_times = times[1:][~contacts]
    route_points = xy(result["path"])
    edge_times = times[1:][edge_moves]
    edge_segments = segments[edge_moves]
    check_times = times[1:][clear_checks]
    check_points = ends[clear_checks]
    impact_times = (times[:-1] + outbound)[contacts]
    contact_segments = segments[contacts]
    impact_points = ends[contacts]

    route, = ax.plot([], [], color="#066f89", lw=1.3, zorder=4, label="Route")
    edge_path = LineCollection([], colors="#5941a9", linewidths=1.6, zorder=5,
                               label="Edge follow")
    contact_path = LineCollection([], colors="#c44636", linewidths=1, linestyles=":",
                                  zorder=5)
    ax.add_collection(edge_path)
    ax.add_collection(contact_path)
    checks, = ax.plot([], [], "s", color="#198269", markerfacecolor="white",
                      ms=3, zorder=6, label="Clear check")
    hits, = ax.plot([], [], "x", color="#a72f22", ms=4, zorder=6, label="Contact")
    probe, = ax.plot([], [], "o", color="#12293e", markeredgecolor="white",
                     ms=7, zorder=8, label="Probe")
    status = ax.text(.02, .96, "", transform=ax.transAxes, ha="left", va="top",
                     color="#24384b", fontsize=8, zorder=9,
                     bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=2))

    def update(seconds):
        move = np.searchsorted(times[1:], seconds, side="right")
        impacts = np.searchsorted(impact_times, seconds, side="right")
        route_count = np.searchsorted(route_times, seconds, side="right")
        check_count = np.searchsorted(check_times, seconds, side="right")
        edge_count = np.searchsorted(edge_times, seconds, side="right")
        route.set_data(*route_points[:route_count + 1].T)
        checks.set_data(*check_points[:check_count].T)
        hits.set_data(*impact_points[:impacts].T)
        edge_path.set_segments(edge_segments[:edge_count])
        contact_path.set_segments(contact_segments[:impacts])

        if move == len(attempts):
            position = route_points[-1]
            phase = "Goal reached" if result["success"] else "Stopped"
        else:
            elapsed = seconds - times[move]
            leg_time = outbound[move]
            fraction = min(elapsed / leg_time, 1.0)
            phase = attempts[move].phase
            if contacts[move] and elapsed >= leg_time:
                fraction = min((times[move + 1] - seconds) / leg_time, 1.0)
                phase = "Contact" if fraction == 1.0 else "Retreat"
            position = starts[move] + fraction * (ends[move] - starts[move])
        probe.set_data([position[0]], [position[1]])
        probe.set_markerfacecolor("#a72f22" if phase in ("Contact", "Retreat") else "#12293e")
        status.set_text(f"{phase} · contacts {impacts}")

    return update, times[-1]


def save_path_animation(truth, traces, output):
    fps = 10
    playback_speed = 12
    fig, axes = plt.subplots(len(PATH_ROWS), 2, figsize=(8.4, 10.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=.09, right=.98, top=.876, bottom=.10, hspace=.33, wspace=.08)
    fig.suptitle("Touch-navigation simulations", fontsize=19, fontweight="bold",
                 color="#12293e", y=.97)
    timer = fig.text(.5, .925, "", color="#46596b", fontsize=11, ha="center")
    updates = []
    durations = []
    for i, (simulation, limit, label) in enumerate(PATH_ROWS):
        for j, planner in enumerate(("A*", "MDP")):
            ax = axes[i, j]
            ax.set_title(f"{label} — {planner}", loc="left", fontweight="bold", fontsize=9)
            draw_workspace(ax, truth)
            if j == 0:
                ax.set_ylabel("y (cm)", fontsize=9)
            if i == len(PATH_ROWS) - 1:
                ax.set_xlabel("x (cm)", fontsize=9)
            update, duration = animate_trial(ax, traces[(simulation, planner, limit)][0])
            updates.append(update)
            durations.append(duration)

    handles, _ = axes[0, 0].get_legend_handles_labels()
    handles.append(Patch(facecolor="#dfe7ed", edgecolor="#97a9b9", label="Actual box"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .023),
               ncol=8, frameon=False, fontsize=8, columnspacing=1.3)
    duration = max(durations)
    frames = [*np.arange(0, duration, playback_speed / fps), *([duration] * fps)]

    try:
        with PillowWriter(fps=fps).saving(fig, output, dpi=80) as writer:
            for seconds in frames:
                timer.set_text(
                    f"Trial 1 · Simulated time {seconds:5.1f} s · {playback_speed}× playback"
                )
                for update in updates:
                    update(seconds)
                writer.grab_frame()
    finally:
        plt.close(fig)
