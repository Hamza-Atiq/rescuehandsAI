"""Continuous physical placement checks, independent of policy claims."""
import math

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
