"""Draw the paths, contact approaches, and fixed obstacles for both scenarios."""
from math import ceil, floor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
from PIL import Image

from navsim import SPEED_CM_S
from plot_settings import apply_plot_style
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
    for number, (row, col, height, width) in enumerate(truth, 1):
        ax.add_patch(Rectangle((col, row), width, height, facecolor="#dfe7ed",
                               edgecolor="#97a9b9", lw=.8, zorder=2))
        ax.text(col + width / 2, row + height / 2, f"B{number}",
                ha="center", va="center", fontsize=8, color="#435466", zorder=3)
    ax.plot(*xy([START])[0], "o", color="#198269", ms=5, zorder=7, label="Start")
    ax.plot(*xy([GOAL])[0], "*", color="#82542b", ms=8, zorder=7, label="Goal")


def draw_path(ax, truth, result):
    draw_workspace(ax, truth)
    ax.plot(*xy(result["path"]).T, lw=1.7, color="#066f89", zorder=4)
    for segment in result.get("edge_moves", ()):
        ax.plot(*xy(segment).T, color="#5941a9", lw=1.9,
                marker=".", markersize=2.6, zorder=5)

    clear_checks = [cell for _, cell, blocked in result.get("edge_checks", ())
                    if not blocked]
    if clear_checks:
        ax.plot(*xy(clear_checks).T, marker="s", markersize=3,
                markerfacecolor="white", markeredgecolor="#198269",
                linestyle="none", zorder=6)

    for segment in result["touches"]:
        approach = xy(segment)
        ax.plot(*approach.T, color="#c44636", ls=":", lw=1, zorder=5)
        ax.plot(*approach[-1], "x", color="#a72f22", ms=4, mew=1, zorder=6)


@apply_plot_style("paths")
def save_path_plot(truth, result, title, output):
    fig, ax = plt.subplots()
    try:
        draw_path(ax, truth, result)
        ax.set(title=title, xlabel="x (cm)", ylabel="y (cm)")
        fig.savefig(output)
    finally:
        plt.close(fig)


def path_legend_handles():
    return [
        Line2D([], [], color="#066f89", lw=1.7, label="Path"),
        Line2D([], [], color="#5941a9", lw=1.9, marker=".", markersize=2.6,
               label="Edge follow"),
        Line2D([], [], marker="s", markersize=3, markerfacecolor="white",
               markeredgecolor="#198269", ls="none", label="Clear check"),
        Line2D([], [], color="#c44636", ls=":", lw=1, marker="x", markersize=4,
               markeredgecolor="#a72f22", label="Contact and return"),
        Patch(facecolor="#dfe7ed", edgecolor="#97a9b9", lw=.8, label="Actual box"),
        Line2D([], [], marker="o", color="#198269", markersize=5, ls="none", label="Start"),
        Line2D([], [], marker="*", color="#82542b", markersize=8, ls="none", label="Goal"),
    ]


@apply_plot_style("paths")
def save_path_legend(output):
    fig = plt.figure()
    try:
        fig.legend(handles=path_legend_handles(), ncols=3)
        fig.savefig(output)
    finally:
        plt.close(fig)


def path_stem(simulation, planner, limit, trial):
    scenario = simulation if limit is None else f"{simulation}_{limit}cm"
    planner_name = planner.lower().replace("*", "star")
    return f"{scenario}_{planner_name}_trial_{trial:02d}"


def save_path_plots(truth, traces, output_dir):
    """Save one plot per distinct trace within each navigation strategy."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for simulation, limit, label in PATH_ROWS:
        for planner in ("A*", "MDP"):
            seen = set()
            for trial, result in enumerate(traces[(simulation, planner, limit)], 1):
                trace = tuple(tuple(result.get(field, ()))
                              for field in ("path", "touches", "edge_moves", "edge_checks"))
                if trace in seen:
                    continue
                seen.add(trace)
                output = output_dir / f"{path_stem(simulation, planner, limit, trial)}.png"
                save_path_plot(truth, result, f"{label} — {planner}", output)
    save_path_legend(output_dir / "path_legend.png")
    save_path_grid(truth, traces, output_dir / "simulation_paths.png")


@apply_plot_style("path_grid")
def save_path_grid(truth, traces, output):
    """Compare the first trace for each planner and scenario."""
    fig, axes = plt.subplots(len(PATH_ROWS), 2, sharex=True, sharey=True)
    try:
        for row, (simulation, limit, title) in zip(axes, PATH_ROWS):
            for ax, planner in zip(row, ("A*", "MDP")):
                draw_path(ax, truth, traces[(simulation, planner, limit)][0])
                ax.set_title(f"{title} | {planner}", loc="left")
        for ax in axes[:, 0]:
            ax.set_ylabel("y (cm)")
        for ax in axes[-1]:
            ax.set_xlabel("x (cm)")
        fig.legend(handles=path_legend_handles(), ncols=4, bbox_to_anchor=(.5, .005))
        fig.savefig(output)
    finally:
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


@apply_plot_style("animation")
def save_path_animation(truth, traces, output):
    fig, axes = plt.subplots(len(PATH_ROWS), 2, sharex=True, sharey=True)
    fig.suptitle("Touch-navigation simulations", color="#12293e", y=.97)
    timer = fig.text(.5, .925, "", color="#46596b", fontsize=11, ha="center")
    updates = []
    durations = []
    for i, (simulation, limit, label) in enumerate(PATH_ROWS):
        for j, planner in enumerate(("A*", "MDP")):
            ax = axes[i, j]
            ax.set_title(f"{label} — {planner}", loc="left")
            draw_workspace(ax, truth)
            if j == 0:
                ax.set_ylabel("y (cm)")
            if i == len(PATH_ROWS) - 1:
                ax.set_xlabel("x (cm)")
            update, duration = animate_trial(ax, traces[(simulation, planner, limit)][0])
            updates.append(update)
            durations.append(duration)

    handles, _ = axes[0, 0].get_legend_handles_labels()
    handles.append(Patch(facecolor="#dfe7ed", edgecolor="#97a9b9", label="Actual box"))
    fig.legend(handles=handles, bbox_to_anchor=(.5, .023), ncol=8)
    try:
        write_path_gif(fig, timer, updates, max(durations), output)
    finally:
        plt.close(fig)


@apply_plot_style("path_animation")
def save_individual_path_animations(truth, traces, output_dir):
    """Export the first trial of each strategy as a standalone replay."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for simulation, limit, label in PATH_ROWS:
        for planner in ("A*", "MDP"):
            result = traces[(simulation, planner, limit)][0]
            fig, ax = plt.subplots()
            try:
                fig.suptitle(f"{label} — {planner}", y=.97)
                timer = fig.text(.5, .88, "", va="center", ha="center")
                draw_workspace(ax, truth)
                ax.set(xlabel="x (cm)", ylabel="y (cm)")
                update, duration = animate_trial(ax, result)
                output = output_dir / f"{path_stem(simulation, planner, limit, 1)}.gif"
                write_path_gif(fig, timer, (update,), duration, output)
            finally:
                plt.close(fig)


def write_path_gif(fig, timer, updates, duration, output):
    fps = 10
    playback_speed = 12
    frame_times = [*np.arange(0, duration, playback_speed / fps), duration]
    frame_durations = [1000 // fps] * (len(frame_times) - 1) + [1000]

    # Measure once with the widest time label so the crop stays fixed across frames.
    def set_time(seconds):
        timer.set_text(f"Trial 1 · Simulated time {seconds:5.1f} s · {playback_speed}× playback")

    set_time(duration)
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    padding = matplotlib.rcParams["savefig.pad_inches"]
    left, bottom, right, top = fig.get_tightbbox().padded(padding).transformed(
        fig.dpi_scale_trans
    ).extents
    crop = (max(0, floor(left)), max(0, height - ceil(top)),
            min(width, ceil(right)), min(height, height - floor(bottom)))

    def render_frames():
        for seconds in frame_times:
            set_time(seconds)
            for update in updates:
                update(seconds)
            fig.canvas.draw()
            frame = Image.frombuffer("RGBA", (width, height), fig.canvas.buffer_rgba())
            yield frame.crop(crop).convert("RGB")

    frames = render_frames()
    next(frames).save(output, save_all=True, append_images=frames,
                      duration=frame_durations, loop=0)
