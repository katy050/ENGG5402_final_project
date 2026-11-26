"""Grid geometry and shortest paths for the 60 × 40 cm workspace."""
from __future__ import annotations

import heapq
import math

import numpy as np

H, W = 40, 60
START = (20, 3)
GOAL = (20, 56)

# 8-connected moves: (dr, dc, step length)
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
STEP_LEN = np.array([1.0 if (dr == 0 or dc == 0) else math.sqrt(2) for dr, dc in ACTIONS])


def offsets_disk(e: float):
    """Integer offsets (dr, dc) with dr^2 + dc^2 <= e^2 (always includes (0,0))."""
    k = int(math.floor(e))
    out = [(dr, dc) for dr in range(-k, k + 1) for dc in range(-k, k + 1)
           if dr * dr + dc * dc <= e * e + 1e-9]
    return out


def box_mask(box, dr=0, dc=0):
    r0, c0, h, w = box
    m = np.zeros((H, W), dtype=bool)
    ra, rb = max(r0 + dr, 0), min(r0 + dr + h, H)
    ca, cb = max(c0 + dc, 0), min(c0 + dc + w, W)
    if ra < rb and ca < cb:
        m[ra:rb, ca:cb] = True
    return m


def in_grid(r, c):
    return 0 <= r < H and 0 <= c < W


def move_cells(s, a):
    """Cells a probe sweeps for action a from s: (target, [side cells])."""
    dr, dc = ACTIONS[a]
    t = (s[0] + dr, s[1] + dc)
    sides = [] if (dr == 0 or dc == 0) else [(s[0] + dr, s[1]), (s[0], s[1] + dc)]
    return t, sides


def astar(blocked: np.ndarray, start, goal):
    """8-connected A* without corner cutting.

    blocked  : bool array, cells that may not be entered or swept.
    Returns list of cells from start to goal or None.
    """
    def h(p):
        dr, dc = abs(p[0] - goal[0]), abs(p[1] - goal[1])
        return (math.sqrt(2) - 1) * min(dr, dc) + max(dr, dc)
    g = {start: 0.0}
    parent = {start: None}
    openq = [(h(start), 0.0, start)]
    closed = set()
    while openq:
        f, gs, s = heapq.heappop(openq)
        if s in closed:
            continue
        if s == goal:
            path = [s]
            while parent[path[-1]] is not None:
                path.append(parent[path[-1]])
            return path[::-1]
        closed.add(s)
        for a in range(8):
            t, sides = move_cells(s, a)
            if not in_grid(*t) or blocked[t]:
                continue
            if any(blocked[x] for x in sides):
                continue
            ng = gs + STEP_LEN[a]
            if ng < g.get(t, 1e18) - 1e-12:
                g[t] = ng
                parent[t] = s
                heapq.heappush(openq, (ng + h(t), ng, t))
    return None


def path_length(path):
    return sum(math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(path[:-1], path[1:]))
