"""Continuous physical placement checks, independent of policy claims."""
import math

from .scene import UTENSILS

SETTLED_SPEED = 0.02   # m/s
SPARE_MAX_SHIFT = 0.03  # m the spare utensil may move and still count as untouched
UPRIGHT_TOLERANCE = 0.008  # m of cup centre height error before it counts as tipped


def task_outcome(facts, task, utensil_holders: set, scene_params) -> dict:
    """Physical success for the dinner task from final auditor facts.

    `utensil_holders` is every arm that held the utensil with both jaws during the
    episode; a hand-off means both arms appear in it. `scene_params` gives the cup
    size and the spare utensil's starting pose.
    """
    other = next(u for u in UTENSILS if u != task.utensil)
    sx, sy, _ = scene_params.poses[other]
    ox, oy, _ = facts.positions[other]
    checks = {
        "cup_in_zone": facts.in_zone["cup"] == "cup_zone",
        "utensil_in_zone": facts.in_zone[task.utensil] == "utensil_zone",
        "handoff": utensil_holders == {"left_arm", "right_arm"},
        "released": not facts.touching["cup"] and not facts.touching[task.utensil],
        "supported": facts.supported["cup"] and facts.supported[task.utensil],
        "cup_upright": abs(facts.height["cup"] - scene_params.cup_half_height) < UPRIGHT_TOLERANCE,
        "settled": facts.speed["cup"] < SETTLED_SPEED and facts.speed[task.utensil] < SETTLED_SPEED,
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
