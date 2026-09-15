"""Smooth joint-space motion segments for the scripted teacher."""
from dataclasses import dataclass, field
import math


@dataclass
class Move:
    """Go from the current targets to `goal` (a subset of joints) in `steps` control steps."""
    goal: dict
    steps: int
    label: str = ""
    _start: dict = field(default=None, repr=False)
    _k: int = field(default=0, repr=False)


class MotionFollower:
    """Turns Move segments into per-step joint targets within a step-size limit."""

    def __init__(self, targets: dict, max_delta: float):
        if max_delta <= 0:
            raise ValueError("max_delta must be positive")
        self.targets = dict(targets)
        self.max_delta = max_delta

    def begin(self, move: Move):
        unknown = set(move.goal) - set(self.targets)
        if unknown:
            raise KeyError(f"Unknown joints in move: {sorted(unknown)}")
        move._start = {n: self.targets[n] for n in move.goal}
        move._k = 0
        # smoothstep peaks at 1.5x the mean speed: stretch so the peak fits the limit
        span = max((abs(move.goal[n] - move._start[n]) for n in move.goal), default=0.0)
        move.steps = max(move.steps, math.ceil(1.5 * span / (self.max_delta * 0.95)), 1)

    def advance(self, move: Move) -> bool:
        """Advance one control step. Returns True when the move is finished."""
        move._k += 1
        a = min(move._k / move.steps, 1.0)
        s = a * a * (3 - 2 * a)  # smoothstep: gentle start and stop
        for n, goal in move.goal.items():
            start = move._start[n]
            wanted = start + (goal - start) * s
            step = max(-self.max_delta * 0.95, min(self.max_delta * 0.95, wanted - self.targets[n]))
            self.targets[n] += step
        reached = all(abs(self.targets[n] - g) < 1e-9 for n, g in move.goal.items())
        return move._k >= move.steps and reached
