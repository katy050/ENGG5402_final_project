"""Run the touch-navigation comparison against one fixed obstacle mask."""
from time import perf_counter

from planners import AStarPlanner, MDPPlanner, OccupancyGrid
from world import GOAL, START, STEP_LEN, move_cells

SPEED_CM_S = 1.0
SETTLE_S = 0.2
CONTACT_HANDLING_S = 1.5  # Stops, retraction overhead, and restart.
MAX_STEPS = 600
PLANNERS = (
    ("A*", AStarPlanner),
    ("MDP", MDPPlanner),
)


def run_comparison(prior, occupied):
    results = {}
    for name, planner_type in PLANNERS:
        belief = OccupancyGrid(prior)
        planner = planner_type(belief)
        cell = START
        contacts = 0
        distance = 0.0
        motion_time = 0.0
        travel = 0.0
        planning_time = 0.0
        path = [START]
        touches = []
        for _ in range(MAX_STEPS):
            if cell == GOAL:
                break
            started = perf_counter()
            action = planner.act(cell)
            if action is None:
                break
            target, sides = move_cells(cell, action)
            swept = sides + [target]
            hit = next((candidate for candidate in swept if occupied[candidate]), None)
            if hit is None:
                belief.observe_free(swept)
                cell = target
                path.append(cell)
                distance += STEP_LEN[action]
                travel += STEP_LEN[action]
                motion_time += STEP_LEN[action] / SPEED_CM_S + SETTLE_S
            else:
                belief.observe_contact(hit)
                impact = tuple((start + end) / 2 for start, end in zip(cell, target))
                touches.append((cell, impact))
                contacts += 1
                travel += STEP_LEN[action]
                motion_time += STEP_LEN[action] / SPEED_CM_S + CONTACT_HANDLING_S
            planning_time += perf_counter() - started
        results[name] = {
            "success": cell == GOAL,
            "contacts": contacts,
            "length": distance,
            "time_total": motion_time + planning_time,
            "motion_time": motion_time,
            "travel": travel,
            "path": path,
            "touches": touches,
        }
    return results
