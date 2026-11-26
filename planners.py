"""A* and MDP policies using the same cell occupancy model."""
from __future__ import annotations

import numpy as np

from world import ACTIONS, GOAL, H, STEP_LEN, W, astar, move_cells

R_CONTACT = 20.0
R_GOAL = 100.0
GAMMA = 0.99


class OccupancyGrid:
    """A copy of the given probability map, with cell-wise observations."""

    def __init__(self, prior):
        self.p = prior.copy()
        self.known = np.full(self.p.shape, -1, dtype=np.int8)
        self.version = 0

    def observe_free(self, cells):
        changed = False
        for cell in cells:
            changed |= self.p[cell] != 0.0
            self.known[cell] = 0
            self.p[cell] = 0.0
        if changed:
            self.version += 1

    def observe_contact(self, cell):
        self.known[cell] = 1
        self.p[cell] = 1.0
        self.version += 1


def success_probs(p):
    """q[a] (H,W): probability that action a from each cell is contact-free.
    valid[a] (H,W): action stays inside the table area.
    """
    P = np.pad(p, 1, constant_values=1.0)
    valid_full = np.pad(np.ones((H, W), bool), 1, constant_values=False)
    q = np.empty((8, H, W))
    valid = np.empty((8, H, W), bool)
    for a, (dr, dc) in enumerate(ACTIONS):
        pt = P[1 + dr:1 + dr + H, 1 + dc:1 + dc + W]
        qa = 1.0 - pt
        if dr != 0 and dc != 0:
            qa = qa * (1.0 - P[1 + dr:1 + dr + H, 1:1 + W]) * (1.0 - P[1:1 + H, 1 + dc:1 + dc + W])
        q[a] = qa
        valid[a] = valid_full[1 + dr:1 + dr + H, 1 + dc:1 + dc + W]
    return q, valid


# ----------------------------------------------------------------------------
# Planners
# ----------------------------------------------------------------------------
class MDPPlanner:
    """Value iteration on the touch MDP.

    Bellman equation from the proposal (with q = success probability):
        V(s) = max_a q (-L_a + g V(s')) + (1-q)(-C + g V(s))
    Because the failure branch returns to s, each action's fixed point has a
    closed form, which we use as the backup (same fixed point, faster):
        Q_a(s) = [q(-L_a + g V(s')) - (1-q) C] / (1 - g (1-q))
    """

    def __init__(self, belief, gamma=GAMMA, tol=1e-4, max_iter=5000, C=R_CONTACT):
        self.b = belief
        self.C = C
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
        self.V = np.full((H, W), -50.0)
        self.V[GOAL] = R_GOAL
        self._ver = None
        self.sweeps = 0
        self.solves = 0

    def _solve(self):
        g = self.gamma
        q, valid = success_probs(self.b.p)
        den = 1.0 - g * (1.0 - q)
        L = STEP_LEN[:, None, None]
        V = self.V
        for it in range(self.max_iter):
            Vp = np.pad(V, 1, constant_values=-1e6)
            Q = np.empty((8, H, W))
            for a, (dr, dc) in enumerate(ACTIONS):
                Vt = Vp[1 + dr:1 + dr + H, 1 + dc:1 + dc + W]
                Q[a] = (q[a] * (-L[a] + g * Vt) - (1.0 - q[a]) * self.C) / den[a]
            Q[~valid] = -1e9
            Vn = Q.max(axis=0)
            Vn[GOAL] = R_GOAL
            d = np.abs(Vn - V).max()
            V = Vn
            if d < self.tol:
                break
        self.sweeps += it + 1
        self.solves += 1
        self.V = V
        self.Q = Q
        self._ver = self.b.version

    def act(self, s):
        if self._ver != self.b.version:
            self._solve()
        return int(np.argmax(self.Q[:, s[0], s[1]]))


class AStarPlanner:
    """Baseline: A* on cells with p > 0.5, replanned after every contact
    (the D* Lite behaviour, re-computed from scratch here)."""

    def __init__(self, belief, thresh=0.5):
        self.b = belief
        self.thresh = thresh
        self.path = None
        self._ver = None
        self.solves = 0

    def _plan(self, s):
        blocked = self.b.p > self.thresh
        blocked[s] = False
        path = astar(blocked, s, GOAL)
        if path is None:   # map too pessimistic: only trust confirmed contacts
            blocked = self.b.known == 1
            blocked[s] = False
            path = astar(blocked, s, GOAL)
        self.path = path
        self.solves += 1

    def _needs_replan(self, s):
        if self.path is None or s not in self.path:
            return True
        i = self.path.index(s)
        nxt = self.path[i + 1]
        a = ACTIONS.index((nxt[0] - s[0], nxt[1] - s[1]))
        t, sides = move_cells(s, a)
        return any(self.b.p[x] > self.thresh for x in [t] + sides)

    def act(self, s):
        if self._needs_replan(s):
            self._plan(s)
        if self.path is None:
            return None
        i = self.path.index(s)
        nxt = self.path[i + 1]
        return ACTIONS.index((nxt[0] - s[0], nxt[1] - s[1]))
