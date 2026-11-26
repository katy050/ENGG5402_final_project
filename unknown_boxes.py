"""A* and MDP navigation with contact-triggered box-face following."""
from dataclasses import dataclass
from heapq import heappop, heappush
from math import hypot

import numpy as np

from box_edge import FaceFollow
from navsim import CONTACT_HANDLING_S, SETTLE_S, SPEED_CM_S, MoveAttempt
from planners import GAMMA, R_CONTACT, R_GOAL, OccupancyGrid
from world import ACTIONS, GOAL, H, START, W

STEP_LIMITS = (1, 3, 5)
UNKNOWN_OCCUPANCY = 0.05
MAX_ATTEMPTS = 600
GOAL_INDEX = GOAL[0] * W + GOAL[1]
START_INDEX = START[0] * W + START[1]


@dataclass
class MoveGraph:
    """Shared movement geometry, indexed by move and flattened grid cell."""

    lengths: np.ndarray
    targets: np.ndarray
    valid: np.ndarray
    swept: list

    @classmethod
    def build(cls, limit):
        lengths = []
        targets = []
        valid = []
        swept = []
        rows, cols = np.indices((H, W)).reshape(2, -1)

        def indices(dr, dc):
            r, c = rows + dr, cols + dc
            inside = (r >= 0) & (r < H) & (c >= 0) & (c < W)
            return np.where(inside, r * W + c, 0), inside

        for dr, dc in ACTIONS:
            unit_length = hypot(dr, dc)
            for stride in range(int(limit / unit_length), 0, -1):
                target, inside = indices(dr * stride, dc * stride)
                groups = []
                for i in range(1, stride + 1):
                    group = []
                    if dr and dc:
                        group.extend([indices(dr * i, dc * (i - 1))[0],
                                      indices(dr * (i - 1), dc * i)[0]])
                    group.append(indices(dr * i, dc * i)[0])
                    groups.append(np.stack(group))
                lengths.append(stride * unit_length)
                targets.append(target)
                valid.append(inside)
                swept.append(groups)
        return cls(np.array(lengths), np.stack(targets), np.stack(valid), swept)

    def success_probabilities(self, probability):
        success = self.valid.astype(float)
        for action, groups in enumerate(self.swept):
            for group in groups:
                success[action] *= np.prod(1.0 - probability[group], axis=0)
        return success


def astar_action(graph, probability, start):
    """Find a shortest route on the thresholded grid, preferring fewer moves on ties."""
    permitted = graph.success_probabilities((probability > 0.5).astype(float)) > 0
    frontier = [(0.0, 0, 0.0, start)]
    best = {start: (0.0, 0)}
    first_move = {}
    while frontier:
        _, moves, cost, cell = heappop(frontier)
        if (cost, moves) != best[cell]:
            continue
        if cell == GOAL_INDEX:
            return first_move[cell]
        for action in np.flatnonzero(permitted[:, cell]):
            target = int(graph.targets[action, cell])
            candidate = cost + graph.lengths[action]
            old_cost, old_moves = best.get(target, (np.inf, 0))
            if candidate > old_cost + 1e-9:
                continue
            if abs(candidate - old_cost) <= 1e-9 and moves + 1 >= old_moves:
                continue
            best[target] = (candidate, moves + 1)
            first_move[target] = int(action) if cell == start else first_move[cell]
            row, col = divmod(target, W)
            lower_bound = hypot(GOAL[0] - row, GOAL[1] - col)
            heappush(frontier, (candidate + lower_bound, moves + 1, candidate, target))
    return None


class MDP:
    """Warm-started value iteration using the probability of clearing a full sweep."""

    def __init__(self):
        self.value = np.full(H * W, -50.0)
        self.value[GOAL_INDEX] = R_GOAL

    def action(self, graph, probability, cell):
        q = graph.success_probabilities(probability)
        if not np.any(q[:, cell]):
            return None
        costs = graph.lengths[:, None]
        denominator = 1.0 - GAMMA * (1.0 - q)
        valid = q > 0.0
        for _ in range(2000):
            values = (q * (-costs + GAMMA * self.value[graph.targets])
                      - (1.0 - q) * R_CONTACT) / denominator
            values[~valid] = -1e6
            updated = values.max(axis=0)
            updated[GOAL_INDEX] = R_GOAL
            error = np.max(np.abs(updated - self.value))
            self.value = updated
            if error < 1e-4:
                break
        else:
            raise RuntimeError("MDP value iteration did not converge.")
        return int(values[:, cell].argmax())


def execute_move(graph, action, cell, occupied, occupancy_grid):
    """Sweep in order and return the end cell, impact, hit cell, distance, and time."""
    distance = graph.lengths[action]
    groups = graph.swept[action]
    for i, group in enumerate(groups):
        candidates = group[:, cell]
        hits = candidates[occupied[candidates]]
        if len(hits):
            hit = divmod(int(hits[0]), W)
            occupancy_grid.observe_contact(hit)
            # Entry at the boundary halfway to the next cell center.
            fraction = (i + 0.5) / len(groups)
            travel = 2.0 * fraction * distance
            start_rc = divmod(cell, W)
            end_rc = divmod(int(graph.targets[action, cell]), W)
            impact = tuple(start + fraction * (end - start) for start, end in zip(start_rc, end_rc))
            return cell, impact, hit, travel, travel / SPEED_CM_S + CONTACT_HANDLING_S
        occupancy_grid.observe_free(divmod(int(candidate), W) for candidate in candidates)
    return int(graph.targets[action, cell]), None, None, distance, distance / SPEED_CM_S + SETTLE_S


def run_unknown_trial(occupied, graphs, planner_name, step_limit):
    grid = OccupancyGrid(np.full((H, W), UNKNOWN_OCCUPANCY))
    grid.observe_free([START])
    probability = grid.p.ravel()
    choose_action = astar_action if planner_name == "A*" else MDP().action
    cell = START_INDEX
    contacts = 0
    face_follow = None
    edge_moves = []
    edge_checks = []
    distance = 0.0
    route = 0.0
    seconds = 0.0
    largest_move = 0.0
    path = [START]
    touches = []
    attempts = []
    for _ in range(MAX_ATTEMPTS):
        if cell == GOAL_INDEX:
            break
        source = divmod(cell, W)
        target = face_follow.next_target(source, step_limit) if face_follow else None
        is_check = target is not None and face_follow.check_pending
        if target is None:
            face_follow = None
            phase = "Navigate"
            graph = graphs[1]
            action = choose_action(graph, probability, cell)
            if action is None:
                break
        else:
            phase = "Edge check" if is_check else "Edge follow"
            graph = graphs[1 if is_check else step_limit]
            target_index = target[0] * W + target[1]
            matching = np.flatnonzero(graph.valid[:, cell] & (graph.targets[:, cell] == target_index))
            action = int(matching[0])

        largest_move = max(largest_move, graph.lengths[action])
        cell, impact, hit, travel, elapsed = execute_move(graph, action, cell, occupied, grid)
        endpoint = divmod(cell, W)
        attempts.append(MoveAttempt(source, impact if hit is not None else endpoint,
                                    elapsed, hit is not None, phase))
        if hit is None:
            path.append(endpoint)
            route += travel
        else:
            contacts += 1
            touches.append((source, impact))

        if face_follow is not None:
            if is_check:
                edge_checks.append((source, target, hit is not None))
            elif hit is None:
                edge_moves.append((source, endpoint))
            if not face_follow.after_attempt(hit is not None):
                face_follow = None
        elif hit is not None:
            face_follow = FaceFollow.from_contact(source, hit)
        distance += travel
        seconds += elapsed
    return {
        "success": cell == GOAL_INDEX, "contacts": contacts,
        "route_cm": route, "travel_cm": distance, "time_s": seconds,
        "largest_move_cm": largest_move, "path": path, "touches": touches,
        "edge_moves": edge_moves, "edge_checks": edge_checks,
        "attempts": attempts,
    }
