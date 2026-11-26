"""Contact-triggered box following with discrete clearance checks."""
from dataclasses import dataclass

from world import GOAL, H, W

GRID_SHAPE = (H, W)


@dataclass
class FaceFollow:
    inward: tuple[int, int]
    tangent_sign: int
    check_pending: bool = False
    reversed_once: bool = False

    @classmethod
    def from_contact(cls, source, contact):
        inward = tuple(target - start for start, target in zip(source, contact))
        if sum(map(abs, inward)) != 1:
            raise ValueError("A box face must be entered from an adjacent cardinal cell.")
        tangent_axis = 0 if inward[1] else 1
        toward_goal = GOAL[tangent_axis] - source[tangent_axis]
        return cls(inward, 1 if toward_goal > 0 else -1)

    @property
    def tangent_axis(self):
        return 0 if self.inward[1] else 1

    def reverse(self):
        if self.reversed_once:
            return False
        self.tangent_sign *= -1
        self.reversed_once = True
        return True

    def next_target(self, cell, step_cm):
        """Return a lateral step or a one-cell check into the contacted face."""
        if self.check_pending:
            return tuple(position + direction for position, direction in zip(cell, self.inward))

        axis = self.tangent_axis
        target = list(cell)
        target[axis] = max(0, min(GRID_SHAPE[axis] - 1,
                                  cell[axis] + self.tangent_sign * step_cm))
        if target[axis] == cell[axis]:
            return self.next_target(cell, step_cm) if self.reverse() else None
        return tuple(target)

    def after_attempt(self, hit: bool) -> bool:
        """Update the phase and report whether face following continues."""
        if self.check_pending:
            self.check_pending = False
            return hit
        if hit:
            return self.reverse()
        self.check_pending = True
        return True
