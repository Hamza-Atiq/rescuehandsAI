"""Validate full named commands before touching simulator state."""
import math
from .contracts import BimanualAction

def validate_action(action: BimanualAction, names, limits, previous, *,
                    now: float, max_age: float, max_delta: float) -> tuple[float, ...]:
    if not all(math.isfinite(x) for x in (now, action.timestamp, max_age, max_delta)):
        raise ValueError("Time and limits must be finite")
    if max_age < 0 or max_delta <= 0:
        raise ValueError("Invalid control limits")
    if not names or len(set(names)) != len(names):
        raise ValueError("Expected joint names must be unique")
    if set(action.targets) != set(names):
        raise ValueError("Command must contain exactly the expected joint names")
    if action.timestamp > now + 1e-9 or now - action.timestamp > max_age + 1e-9:
        raise ValueError("Stale or future action timestamp")
    values = []
    for name in names:
        value = float(action.targets[name])
        low, high = limits[name]
        old = previous[name]
        if not all(math.isfinite(x) for x in (value, low, high, old)) or low > high:
            raise ValueError(f"Invalid numerical value: {name}")
        if not low <= value <= high:
            raise ValueError(f"Joint limit exceeded: {name}")
        if abs(value - old) > max_delta + 1e-9:
            raise ValueError(f"Command step too large: {name}")
        values.append(value)
    return tuple(values)
