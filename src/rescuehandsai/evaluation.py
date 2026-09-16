"""Continuous physical placement checks, independent of policy claims."""
import math

from .scene import UTENSILS

SETTLED_SPEED = 0.02   # m/s
SETTLED_SPIN = 0.3     # rad/s; a rolling cup or spinning utensil is not settled
SPARE_MAX_SHIFT = 0.03  # m the spare utensil may move and still count as untouched
UPRIGHT_TOLERANCE = 0.008  # m of cup centre height error before it counts as tipped
UPRIGHT_MIN_UP_Z = math.cos(math.radians(15))  # cup axis within 15 degrees of vertical


class HandoffTracker:
    """Ordered in-air hand-off of the task utensil, from auditor facts.

    Stages: the right hand holds it alone, then both hands hold it while it is off
    the table, then the left hand holds it alone while still in the air. A drop to
    the table before completion starts over, so "right drops it, left picks it up"
    is not a hand-off.
    """
    def __init__(self, item: str):
        self.item = item
        self.reset()

    def reset(self):
        self.stage = "none"

    @property
    def done(self) -> bool:
        return self.stage == "done"

    def update(self, facts) -> bool:
        if self.done:
            return True
        held, airborne = facts.held_by[self.item], not facts.supported[self.item]
        if self.stage == "shared" and not airborne:
            self.stage = "none"  # it is resting again: the in-air transfer was not completed
        if held == {"right_arm"}:
            self.stage = "right"
        elif held == {"left_arm", "right_arm"} and airborne and self.stage in ("right", "shared"):
            self.stage = "shared"
        elif held == {"left_arm"} and airborne and self.stage == "shared":
            self.stage = "done"
        elif not held and not airborne:
            self.stage = "none"
        return self.done


def task_outcome(facts, task, handoff_done: bool, scene_params) -> dict:
    """Physical success for the dinner task from final auditor facts.

    `handoff_done` comes from a HandoffTracker that watched the whole episode.
    `scene_params` gives the cup size and the spare utensil's starting pose.
    """
    if not isinstance(handoff_done, bool):
        raise TypeError("handoff_done must be a bool from HandoffTracker")
    if facts.up_z is None or facts.angular_speed is None:
        raise ValueError("facts need up_z and angular_speed for the success check")
    other = next(u for u in UTENSILS if u != task.utensil)
    sx, sy, _ = scene_params.poses[other]
    ox, oy, _ = facts.positions[other]
    items = ("cup", task.utensil)
    checks = {
        "cup_in_zone": facts.in_zone["cup"] == "cup_zone",
        "utensil_in_zone": facts.in_zone[task.utensil] == "utensil_zone",
        "handoff": handoff_done,
        "released": not facts.touching["cup"] and not facts.touching[task.utensil],
        "supported": facts.supported["cup"] and facts.supported[task.utensil],
        "cup_upright": (facts.up_z["cup"] >= UPRIGHT_MIN_UP_Z
                        and abs(facts.height["cup"] - scene_params.cup_half_height) < UPRIGHT_TOLERANCE),
        "settled": all(facts.speed[i] < SETTLED_SPEED and facts.angular_speed[i] < SETTLED_SPIN for i in items),
        "spare_utensil_in_place": (other not in facts.out_of_bounds and facts.supported[other]
                                   and math.hypot(ox - sx, oy - sy) < SPARE_MAX_SHIFT),
    }
    return {"success": all(checks.values()), **checks}

class PlacementTracker:
    """Consumes facts from a physics auditor; does not infer them itself."""
    def __init__(self, required_seconds: float, max_speed: float):
        if not math.isfinite(required_seconds) or required_seconds <= 0:
            raise ValueError("Required stability duration must be positive")
        if not math.isfinite(max_speed) or max_speed < 0:
            raise ValueError("Maximum speed must be nonnegative")
        self.required_seconds = required_seconds
        self.max_speed = max_speed
        self._since = None
        self._last = None

    def update(self, *, time: float, inside: bool, supported: bool,
               released: bool, both_arms_used: bool, speed: float) -> bool:
        if not math.isfinite(time) or time < 0 or (self._last is not None and time < self._last):
            raise ValueError("Simulation time must be finite and nondecreasing")
        self._last = time
        valid = (inside and supported and released and both_arms_used
                 and math.isfinite(speed) and 0 <= speed <= self.max_speed)
        if not valid:
            self._since = None
            return False
        if self._since is None:
            self._since = time
        return time - self._since >= self.required_seconds - 1e-9
