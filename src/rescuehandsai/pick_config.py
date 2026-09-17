"""Pick-milestone rules and contact settings, with step counts derived from the time steps (spec §5)."""
from dataclasses import dataclass
import json
import math
from pathlib import Path

from .scene import ROOT

RULES_PATH = ROOT / "configs" / "pick_rules.json"
CONTACTS_PATH = ROOT / "configs" / "pick_contacts.json"


def load_rules(path=None) -> dict:
    return json.loads(Path(path or RULES_PATH).read_text())


def load_contacts(path=None) -> dict:
    return json.loads(Path(path or CONTACTS_PATH).read_text())


def whole_steps(seconds: float, dt: float, what: str) -> int:
    ratio = seconds / dt
    if not math.isfinite(ratio) or ratio < 1 or not math.isclose(ratio, round(ratio), abs_tol=1e-9):
        raise ValueError(f"{what}: {seconds} s is not a whole number (>= 1) of {dt} s steps")
    return round(ratio)


@dataclass(frozen=True)
class StepCounts:
    substeps: int          # physics steps per control step
    hold: int              # physics steps in the hold window
    max_gap: int           # longest run of physics steps allowed without both jaws touching
    final_speed: int       # physics steps at the end of the window that must be slow
    deadline_control: int  # control steps before the deadline


def derive_steps(rules: dict, physics_dt: float, control_dt: float) -> StepCounts:
    return StepCounts(
        substeps=whole_steps(control_dt, physics_dt, "control_dt"),
        hold=whole_steps(rules["hold_s"], physics_dt, "hold_s"),
        max_gap=whole_steps(rules["max_single_jaw_gap_s"], physics_dt, "max_single_jaw_gap_s"),
        final_speed=whole_steps(rules["final_speed_window_s"], physics_dt, "final_speed_window_s"),
        deadline_control=int(rules["deadline_control_steps"]),
    )
