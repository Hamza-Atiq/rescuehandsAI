"""Seeded failure injection for robustness and recovery evaluation.

Tried first: sideways pushes up to 15 N and a slippery utensil. The SO-101 jaws
squeeze hard enough that neither dislodged a held utensil, so they would not
test recovery. A gripper glitch reliably produces a real physical drop.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class GripperGlitch:
    """The hand holding the task utensil briefly opens by itself (motor fault).

    Fires once, a seeded number of steps after the utensil is first lifted in a
    hand, for `duration_steps`. The fingers open through physics; the utensil
    falls under gravity. Nothing is moved by hand.
    """
    lift_height: float = 0.035
    open_value: float = 0.9
    duration_steps: int = 10
    fired: bool = False
    arm: str | None = None

    def reset(self, seed: int):
        rng = np.random.default_rng([seed, 4242])
        self.delay_steps = int(rng.integers(3, 12))
        self.fired, self.arm, self._armed_at, self._left = False, None, None, 0

    def before_step(self, sim, facts, task, step: int) -> bool:
        """Call before each sim.step. Returns True on the step the fault starts."""
        item = task.utensil
        if self._left > 0:
            self._left -= 1
            if self._left == 0:
                sim.set_actuator_fault(f"{self.arm}/gripper", None)
            return False
        if self.fired or facts is None:
            return False
        holders = sorted(facts.held_by[item])
        if self._armed_at is None and holders and facts.height[item] > self.lift_height:
            self._armed_at = step
        if self._armed_at is not None and step - self._armed_at >= self.delay_steps and holders:
            self.arm = holders[0]
            sim.set_actuator_fault(f"{self.arm}/gripper", self.open_value)
            self.fired, self._left = True, self.duration_steps
            return True
        return False
