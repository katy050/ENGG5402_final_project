"""Run each planner and step-limit configuration and save its results."""
import argparse
import csv
from collections import defaultdict
from itertools import product
from pathlib import Path
from statistics import mean

from four_box_trial import draw_map, fixed_layout
from navsim import PLANNERS, run_comparison
from plot_trial_paths import save_individual_path_animations, save_path_animation, save_path_plots
from plot_experiment_metrics import (save_probability_maps, save_strategy_metrics,
                                     save_strategy_times)
from unknown_boxes import MoveGraph, STEP_LIMITS, run_unknown_trial

TRIALS = 1
CONDITIONS = tuple(product((name for name, _ in PLANNERS), STEP_LIMITS))
METRICS = ("success", "contacts", "route_cm", "travel_cm", "time_s")
TRIAL_COLUMNS = ("simulation", "planner", "step_limit_cm", "trial", *METRICS, "largest_move_cm")


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=TRIALS)
    parser.add_argument("--animate", action="store_true",
                        help="Save individual and combined GIF replays of the first trial.")
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    truth, occupied, prior, footprint_count, *_ = fixed_layout()
    graphs = {limit: MoveGraph.build(limit) for limit in STEP_LIMITS}
    rows = []
    traces = defaultdict(list)
    for trial in range(1, args.trials + 1):
        for name, data in run_comparison(prior, occupied).items():
            traces[("rough_map", name, None)].append(data)
            rows.append(dict(simulation="rough_map", planner=name, step_limit_cm=None, trial=trial,
                             success=data["success"], contacts=data["contacts"],
                             route_cm=data["length"], travel_cm=data["travel"],
                             time_s=data["motion_time"], largest_move_cm=None))
        for name, limit in CONDITIONS:
            data = run_unknown_trial(occupied.ravel(), graphs, name, limit)
            traces[("unknown_boxes", name, limit)].append(data)
            measurements = {field: data[field] for field in (*METRICS, "largest_move_cm")}
            rows.append(dict(simulation="unknown_boxes", planner=name, step_limit_cm=limit,
                             trial=trial, **measurements))

    groups = {}
    keys = ("simulation", "planner", "step_limit_cm")
    for row in rows:
        key = tuple(row[field] for field in keys)
        groups.setdefault(key, []).append(row)

    averages = []
    for key, group in groups.items():
        assert len(group) == args.trials
        averages.append({
            **dict(zip(keys, key)),
            "trials": len(group),
            "goals_reached": sum(row["success"] for row in group),
            **{field: round(mean(row[field] for row in group), 4) for field in METRICS[1:]},
        })

    csv_dir = Path("csv")
    csv_dir.mkdir(exist_ok=True)
    write_csv(csv_dir / "trial_results.csv", TRIAL_COLUMNS, rows)
    write_csv(csv_dir / "averages.csv",
              (*keys, "trials", "goals_reached", *METRICS[1:]), averages)
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)
    draw_map(truth, footprint_count, figures_dir / "four_box_touch_navigation_map.png")
    save_probability_maps(prior, figures_dir / "occupancy_probability_maps.png")
    save_path_plots(truth, traces, figures_dir)
    save_strategy_times(averages, figures_dir / "time_by_strategy.png")
    save_strategy_metrics(averages, figures_dir / "strategy_metrics.pdf")
    if args.animate:
        animation_dir = Path("animation")
        save_individual_path_animations(truth, traces, animation_dir)
        animation_path = animation_dir / "simulation_animation.gif"
        save_path_animation(truth, traces, animation_path)
        print(f"Animations: {animation_dir.resolve()}")
    for row in averages:
        step = row["step_limit_cm"]
        print(f"{row['simulation']:14} {row['planner']:3} {str(step or '-'):>2} cm | "
              f"goals {row['goals_reached']}/{row['trials']} | "
              f"contacts {row['contacts']:4.1f} | route {row['route_cm']:6.2f} cm | "
              f"total travel {row['travel_cm']:6.2f} cm | time {row['time_s']:6.2f} s")


if __name__ == "__main__":
    main()
