"""Run each planner and step-limit configuration and save its results."""
import argparse
import csv
from collections import defaultdict
from itertools import product
from pathlib import Path
from statistics import mean

from four_box_trial import draw_map, fixed_layout
from navsim import PLANNERS, run_comparison
from plot_trial_paths import save_path_plot
from plot_experiment_metrics import save_probability_maps, save_strategy_times
from unknown_boxes import MoveGraph, STEP_LIMITS, run_unknown_trial

TRIALS = 5
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

    write_csv(Path("trial_results.csv"), TRIAL_COLUMNS, rows)
    write_csv(Path("averages.csv"),
              (*keys, "trials", "goals_reached", *METRICS[1:]), averages)
    draw_map(truth, footprint_count, Path("four_box_touch_navigation_map.png"))
    save_probability_maps(prior, Path("occupancy_probability_maps.png"))
    save_path_plot(truth, traces, Path("simulation_paths.png"))
    save_strategy_times(averages, Path("time_by_strategy.png"))
    for row in averages:
        step = row["step_limit_cm"]
        print(f"{row['simulation']:14} {row['planner']:3} {str(step or '-'):>2} cm | "
              f"goals {row['goals_reached']}/{row['trials']} | "
              f"contacts {row['contacts']:4.1f} | route {row['route_cm']:6.2f} cm | "
              f"total travel {row['travel_cm']:6.2f} cm | time {row['time_s']:6.2f} s")


if __name__ == "__main__":
    main()
