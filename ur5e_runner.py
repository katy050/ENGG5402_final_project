"""Run one touch-navigation trial on a commissioned UR5e.

The default mode prints a plan without importing ur_rtde or connecting.
Use --teach to capture the grid with read-only RTDE and --execute for motion.
"""
import argparse
from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
from time import monotonic, sleep

import numpy as np

from box_edge import FaceFollow
from four_box_trial import rough_probability_map
from planners import AStarPlanner, MDPPlanner, OccupancyGrid
from unknown_boxes import MDP, MoveGraph, STEP_LIMITS, UNKNOWN_OCCUPANCY, astar_action
from world import GOAL, H, START, W, move_cells

CELL_M = 0.01
POSE_TOL_M = 0.001
STOPPED_M_S = 0.001
POLL_S = 0.008
STALE_S = 0.12
MAX_MOVES = 600


def norm(values):
    return float(np.linalg.norm(values))


@dataclass
class Frame:
    origin: np.ndarray
    col: np.ndarray
    row: np.ndarray
    rotation: np.ndarray

    @classmethod
    def from_poses(cls, poses):
        p0, pc, pr = np.asarray(poses, dtype=float)
        if any(p.shape != (6,) or not np.all(np.isfinite(p)) for p in (p0, pc, pr)):
            raise ValueError("Three finite RTDE TCP poses are required.")
        dc, dr = pc[:3] - p0[:3], pr[:3] - p0[:3]
        if abs(norm(dc) - (W - 1) * CELL_M) > .004 or abs(norm(dr) - (H - 1) * CELL_M) > .004:
            raise ValueError("Taught points do not match the 59 cm and 39 cm cell-center spans.")
        col, row = dc / norm(dc), dr / norm(dr)
        if abs(float(col @ row)) > .01 or max(abs(col[2]), abs(row[2])) > .01:
            raise ValueError("Taught grid axes must be nearly orthogonal and level.")
        if max(norm(pc[3:] - p0[3:]), norm(pr[3:] - p0[3:])) > .05:
            raise ValueError("Use the same tool orientation at each taught point.")
        return cls(p0[:3], col, row, p0[3:])

    def pose(self, cell):
        r, c = cell
        return np.r_[self.origin + CELL_M * (r * self.row + c * self.col), self.rotation]


def capture_frame(ip, output):
    from rtde_receive import RTDEReceiveInterface

    receiver = RTDEReceiveInterface(ip)
    try:
        poses = []
        for cell in ((0, 0), (0, W - 1), (H - 1, 0)):
            input(f"Teach the probe at cell {cell} at probing height, then press Enter. ")
            if not receiver.isConnected() or norm(receiver.getActualTCPSpeed()[:3]) > STOPPED_M_S:
                raise RuntimeError("The TCP must be stationary with RTDE connected.")
            poses.append(receiver.getActualTCPPose())
        Frame.from_poses(poses)
        with output.open("x", encoding="utf-8") as stream:
            json.dump({"poses": poses}, stream, indent=2)
    finally:
        receiver.disconnect()


def read_settings(path):
    settings = json.loads(path.read_text(encoding="utf-8"))
    fields = (
        "speed_m_s", "retreat_speed_m_s", "acceleration_m_s2", "stop_deceleration_m_s2",
        "contact_force_n", "abort_force_n", "filter_tau_s", "max_segment_seconds",
    )
    for key in fields:
        value = settings.get(key)
        if type(value) not in (int, float) or not isfinite(value) or value <= 0:
            raise ValueError(f"Enter a positive commissioning value for {key} in {path}.")
    if settings["contact_force_n"] >= settings["abort_force_n"]:
        raise ValueError("The software contact threshold must be below the software abort threshold.")
    if settings["retreat_speed_m_s"] > settings["speed_m_s"]:
        raise ValueError("The retreat speed must be at most the probing speed.")
    return settings


class LiveProbe:
    def __init__(self, control, receive, settings):
        self.c = control
        self.r = receive
        self.settings = settings
        self.timestamp = None
        self.last_update = monotonic()

    def read(self):
        if not self.c.isConnected() or not self.r.isConnected():
            raise RuntimeError("RTDE connection lost.")
        status = self.r.getSafetyStatusBits()
        if not (status & 0b11) or (status & ~0b11) or self.r.getRobotMode() != 7:
            raise RuntimeError("The UR controller is outside its normal/reduced running state.")
        timestamp = self.r.getTimestamp()
        if not isfinite(timestamp) or (self.timestamp is not None and timestamp < self.timestamp):
            raise RuntimeError("Invalid RTDE timestamp.")
        if timestamp != self.timestamp:
            self.timestamp = timestamp
            self.last_update = monotonic()
        elif monotonic() - self.last_update > STALE_S:
            raise RuntimeError("Stale RTDE stream.")
        pose = np.asarray(self.r.getActualTCPPose(), dtype=float)
        force = np.asarray(self.r.getActualTCPForce(), dtype=float)[:3]
        velocity = np.asarray(self.r.getActualTCPSpeed(), dtype=float)[:3]
        if pose.shape != (6,) or force.shape != (3,) or velocity.shape != (3,):
            raise RuntimeError("Incomplete RTDE sample.")
        if not all(np.all(np.isfinite(a)) for a in (pose, force, velocity)):
            raise RuntimeError("Nonfinite RTDE data.")
        return pose, force, norm(velocity)

    def require_at(self, expected):
        pose, _, speed = self.read()
        if (norm(pose[:3] - expected[:3]) > POSE_TOL_M
                or norm(pose[3:] - expected[3:]) > .05 or speed > STOPPED_M_S):
            raise RuntimeError("Place the stationary probe at the expected grid pose on the pendant.")
        return pose

    def guarded(self, target, on_sample, retreat=False):
        s = self.settings
        start, initial_force, current_speed = self.read()
        if current_speed > STOPPED_M_S:
            raise RuntimeError("The TCP was moving before the commanded segment.")
        distance = norm(target[:3] - start[:3])
        if not self.c.isPoseWithinSafetyLimits(target.tolist()):
            raise RuntimeError("The controller rejected the target pose.")
        if norm(initial_force) >= s["abort_force_n"]:
            raise RuntimeError("Software force abort before motion.")
        released = norm(initial_force) < s["contact_force_n"]
        if not retreat and not released:
            raise RuntimeError("Probe is loaded before navigation move.")
        if distance <= POSE_TOL_M:
            if not released:
                raise RuntimeError("Probe remains loaded after retreat.")
            return None, distance, norm(initial_force)
        speed = s["retreat_speed_m_s" if retreat else "speed_m_s"]
        peak_force = 0.0
        filtered = norm(initial_force)
        last_sample = started = monotonic()
        active = True
        impact = None
        try:
            if not self.c.moveL(target.tolist(), speed, s["acceleration_m_s2"], True):
                raise RuntimeError("UR controller rejected moveL.")
            while monotonic() - started <= s["max_segment_seconds"]:
                pose, force, tcp_speed = self.read()
                now = monotonic()
                magnitude = norm(force)
                peak_force = max(peak_force, magnitude)
                alpha = 1.0 - np.exp(-(now - last_sample) / s["filter_tau_s"])
                filtered += alpha * (magnitude - filtered)
                last_sample = now
                on_sample(now - started, pose, force, filtered, retreat)
                if magnitude >= s["abort_force_n"]:
                    raise RuntimeError("Software force abort. Inspect before restarting.")
                if retreat:
                    if released and filtered >= s["contact_force_n"]:
                        raise RuntimeError("Unexpected contact during retreat.")
                    if not released and filtered < s["contact_force_n"]:
                        released = True
                    if not released and norm(pose[:3] - start[:3]) > .004:
                        raise RuntimeError("Contact force did not release during retreat.")
                elif filtered >= s["contact_force_n"]:
                    impact = (pose.copy(), force.copy())
                    break
                if (norm(pose[:3] - target[:3]) <= POSE_TOL_M and tcp_speed <= STOPPED_M_S
                        and self.c.getAsyncOperationProgress() < 0):
                    if retreat and not released:
                        raise RuntimeError("Probe remains loaded after retreat.")
                    active = False
                    return None, norm(pose[:3] - start[:3]), peak_force
                sleep(POLL_S)
            if impact is None:
                raise RuntimeError("Guarded motion timed out before verified completion.")
        finally:
            if active and not self.c.stopL(s["stop_deceleration_m_s2"]):
                raise RuntimeError("The controller did not acknowledge stopL.")
        pose, force, tcp_speed = self.read()
        peak_force = max(peak_force, norm(force))
        if peak_force >= s["abort_force_n"] or tcp_speed > STOPPED_M_S:
            raise RuntimeError("Contact stop was not verified for automatic retreat.")
        return impact, norm(pose[:3] - start[:3]), peak_force


class TrialPlanner:
    def __init__(self, scenario, name, edge_step):
        self.scenario = scenario
        self.name = name
        self.edge_step = edge_step
        self.face_follow = None
        prior = rough_probability_map() if scenario == "rough" else np.full((H, W), UNKNOWN_OCCUPANCY)
        self.grid = OccupancyGrid(prior)
        self.grid.observe_free([START])
        if scenario == "rough":
            self.policy = (AStarPlanner if name == "A*" else MDPPlanner)(self.grid)
        else:
            self.graph = MoveGraph.build(1)
            self.choose_unknown = astar_action if name == "A*" else MDP().action

    def move(self, cell):
        if self.scenario == "rough":
            action = self.policy.act(cell)
            if action is None:
                return None
            target, sides = move_cells(cell, action)
            return target, [sides + [target]], "navigate"
        if self.face_follow is not None:
            target = self.face_follow.next_target(cell, self.edge_step)
            if target is not None:
                phase = "edge_check" if self.face_follow.check_pending else "edge_follow"
                axis = 0 if target[0] != cell[0] else 1
                direction = 1 if target[axis] > cell[axis] else -1
                groups = []
                for coordinate in range(cell[axis] + direction, target[axis] + direction, direction):
                    point = list(cell)
                    point[axis] = coordinate
                    groups.append([tuple(point)])
                return target, groups, phase
            self.face_follow = None
        graph = self.graph
        index = cell[0] * W + cell[1]
        action = self.choose_unknown(graph, self.grid.p.ravel(), index)
        if action is None:
            return None
        target = divmod(int(graph.targets[action, index]), W)
        groups = [[divmod(int(value), W) for value in group[:, index]]
                  for group in graph.swept[action]]
        return target, groups, "navigate"

    def observe_move(self, source, groups, hit):
        if hit is None:
            self.grid.observe_free(group[-1] for group in groups)
        if self.scenario != "unknown":
            return
        if self.face_follow is not None:
            if not self.face_follow.after_attempt(hit is not None):
                self.face_follow = None
        elif hit is not None:
            self.face_follow = FaceFollow.from_contact(source, hit)

    def contact_cell(self, frame, start, target, groups, impact):
        pose, force = impact
        segment = target[:3] - start[:3]
        fraction = float((pose[:3] - start[:3]) @ segment / (segment @ segment))
        index = int(np.clip(round(fraction * len(groups) - 0.5), 0, len(groups) - 1))
        group = groups[index]
        candidates = group[-1:]  # center-line cell by default
        if len(group) == 3:
            row, col = abs(float(force @ frame.row)), abs(float(force @ frame.col))
            if row > 2 * col:
                candidates = [group[0], *candidates]
            elif col > 2 * row:
                candidates = [group[1], *candidates]
            candidates.extend(cell for cell in group if cell not in candidates)
        hit = next((cell for cell in candidates if self.grid.known[cell] != 0), None)
        if hit is None:
            raise RuntimeError("Contact could not be assigned to a previously uncleared cell.")
        self.grid.observe_contact(hit)
        return hit


def execute_trial(ip, frame, settings, options, planner):
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface

    with options.log.open("x", encoding="utf-8") as log:
        def event(kind, **values):
            log.write(json.dumps({"event": kind, **values}) + "\n")
            if kind != "force":
                log.flush()

        receiver = RTDEReceiveInterface(ip)
        controller = None
        try:
            controller = RTDEControlInterface(ip)
            probe = LiveProbe(controller, receiver, settings)
            probe.require_at(frame.pose(START))
            if not controller.zeroFtSensor():
                raise RuntimeError("Force zeroing failed.")
            sleep(0.2)
            cell = START
            total_contacts = 0
            travelled = 0.0
            route = 0.0
            started = monotonic()
            event("start", scenario=options.scenario, planner=options.planner,
                  edge_step_cm=options.step if options.scenario == "unknown" else None,
                  cell=cell, settings=settings)
            for step in range(MAX_MOVES):
                if cell == GOAL:
                    break
                source = cell
                decision = planner.move(source)
                if decision is None:
                    raise RuntimeError("The planner found no route on the observed map.")
                target, groups, phase = decision
                source_pose = probe.require_at(frame.pose(cell))
                target_pose = frame.pose(target)

                def sample(t, pose, force, filtered, retreat):
                    event("force", step=step, t_s=t, pose=pose.tolist(), force_n=force.tolist(),
                          filtered_n=filtered, retreat=retreat)

                impact, distance, peak = probe.guarded(target_pose, sample)
                travelled += distance
                hit = None
                if impact is None:
                    route += distance
                    cell = target
                else:
                    total_contacts += 1
                    # Return only along the segment already traversed.
                    event("contact", step=step, cell=cell, target=target,
                          pose=impact[0].tolist(), peak_force_n=peak)
                    _, returned, _ = probe.guarded(source_pose, sample, retreat=True)
                    travelled += returned
                    probe.require_at(frame.pose(cell))
                    hit = planner.contact_cell(frame, source_pose, target_pose, groups, impact)
                planner.observe_move(source, groups, hit)
                event("move", step=step, phase=phase, source=source, cell=cell,
                      target=target, obstacle_cell=hit, probe_travel_m=distance,
                      peak_force_n=peak)
            event("finish", success=cell == GOAL, contacts=total_contacts,
                  route_cm=100 * route, travel_cm=100 * travelled, time_s=monotonic() - started)
            if cell != GOAL:
                raise RuntimeError("The probe did not reach the goal within the move limit.")
        except BaseException as error:
            event("abort", message=str(error))
            raise
        finally:
            try:
                if controller is not None:
                    try:
                        if controller.isConnected():
                            controller.stopScript()
                    finally:
                        controller.disconnect()
            finally:
                receiver.disconnect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--teach", action="store_true", help="read-only RTDE capture of the grid")
    mode.add_argument("--execute", action="store_true", help="enable one physical trial")
    parser.add_argument("--scenario", choices=("rough", "unknown"), default="rough")
    parser.add_argument("--planner", choices=("A*", "MDP"), default="MDP")
    parser.add_argument("--step", type=int, choices=STEP_LIMITS, default=1)
    parser.add_argument("--ip")
    parser.add_argument("--frame", type=Path, default=Path("table_frame.json"))
    parser.add_argument("--settings", type=Path, default=Path("robot_settings.json"))
    parser.add_argument("--log", type=Path)
    options = parser.parse_args()
    if options.teach or options.execute:
        if not options.ip:
            parser.error("--teach and --execute require --ip")
    if options.teach:
        capture_frame(options.ip, options.frame)
        print(f"Saved {options.frame}")
        return
    if options.execute and not options.log:
        parser.error("--execute requires a new --log path")
    if options.scenario == "rough" and options.step != 1:
        parser.error("--step is used only with --scenario unknown")
    planner = TrialPlanner(options.scenario, options.planner, options.step)
    if options.execute:
        frame = Frame.from_poses(json.loads(options.frame.read_text(encoding="utf-8"))["poses"])
        execute_trial(options.ip, frame, read_settings(options.settings), options, planner)
    else:
        decision = planner.move(START)
        print(f"Dry run: {options.scenario}, {options.planner}, edge step {options.step} cm.")
        print(f"Start {START}; goal {GOAL}; first target {decision[0] if decision else None}.")
        print("RTDE was not loaded. No robot connection was made.")


if __name__ == "__main__":
    main()
