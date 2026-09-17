"""Pick-milestone success rules (spec §5): a pure judge over per-physics-step facts.

Failures latch; immediate violations are recorded when they happen, unmet goals at the
deadline. The judge never sees invalid-run labels (spec §6)."""
from collections import deque
from dataclasses import dataclass
import math
from types import SimpleNamespace

import numpy as np

from .pick_config import StepCounts
from .pick_contacts import JAW_UTENSIL

LABEL_RULE = {
    "WRONG_ITEM_TOUCHED": "R1", "FORBIDDEN_CONTACT": "R1", "UNKNOWN_CONTACT_PAIR": "R1",
    "NO_LIFT": "R2", "IMPROPER_HOLD": "R2",
    "SPARE_DISTURBED": "R3",
    "CUP_DISTURBED": "R4", "ITEM_FELL_OR_OUT": "R4", "ARM_ARM_CONTACT": "R4", "SELF_COLLISION": "R4",
    "EXCESS_FORCE": "R4",
    "TIMEOUT": "R5",
    "INVALID_ACTION": "POLICY", "POLICY_ERROR": "POLICY",
}
SEVERE = frozenset({"ITEM_FELL_OR_OUT", "ARM_ARM_CONTACT", "EXCESS_FORCE", "INVALID_ACTION", "POLICY_ERROR"})
RULES = ("R1", "R2", "R3", "R4", "R5")
STOPS = ("success", "deadline", "early_stop", "crash")


@dataclass(frozen=True)
class SubstepFacts:
    physics_step: int
    control_step: int
    positions: dict        # item -> (3,) world position of the body origin
    rotations: dict        # item -> (3, 3) world rotation
    linear_speed: dict     # item -> m/s
    angular_speed: dict    # item -> rad/s
    gripper_pos: np.ndarray
    gripper_rot: np.ndarray
    verdicts: tuple        # ContactVerdict for every touching pair at this physics step


@dataclass
class PickOutcome:
    success: bool
    stop: str
    failures: list
    first_failure: str | None
    failed_rules: list
    unevaluable_rules: list
    picked: str
    lifted_both: bool
    # A valid hold window can complete in a failed episode (e.g. the spare was touched
    # first); it is reported apart from success so that evidence is not lost.
    hold_completed: bool
    hold_completed_physics_step: int | None
    hold_completed_control_step: int | None
    success_physics_step: int | None
    success_control_step: int | None
    max_lift_m: dict
    spare_max_shift_m: float
    spare_max_yaw_deg: float
    cup_max_shift_m: float
    cup_max_tilt_deg: float
    longest_eligible_streak: int


def _yaw(rotation) -> float:
    return math.atan2(rotation[1, 0], rotation[0, 0])


def _yaw_change_deg(a: float, b: float) -> float:
    return abs(math.degrees((a - b + math.pi) % (2 * math.pi) - math.pi))


def _rotation_deg(r0, r1) -> float:
    c = (float(np.trace(r0.T @ r1)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _tilt_deg(rotation) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(rotation[2, 2])))))


class PickJudge:
    def __init__(self, named: str, spare: str, rules: dict, steps: StepCounts, table_center, table_half):
        self.named, self.spare, self.rules, self.steps = named, spare, rules, steps
        self.table_center, self.table_half = table_center, table_half
        self.start = None
        self.failures = {}
        self.success_at = None
        self.hold_ok_at = None
        self._window = deque(maxlen=steps.hold)
        self._streak = 0
        self.longest_streak = 0
        self.lift_step = {named: None, spare: None}
        self.max_lift = {named: 0.0, spare: 0.0}
        self.spare_shift = self.spare_yaw = self.cup_shift = self.cup_tilt = 0.0

    @property
    def succeeded(self) -> bool:
        return self.success_at is not None

    def start_from(self, facts: SubstepFacts):
        self.start = {item: (np.array(facts.positions[item], float), np.array(facts.rotations[item], float))
                      for item in facts.positions}

    def _fail(self, label, at, detail=None) -> bool:
        if label in self.failures:
            return False
        self.failures[label] = {"label": label, "rule": LABEL_RULE[label], "physics_step": at.physics_step,
                                "control_step": at.control_step, "detail": detail}
        return True

    def record_error(self, label: str, physics_step: int, control_step: int, detail: str):
        if label not in ("INVALID_ACTION", "POLICY_ERROR"):
            raise ValueError(f"record_error takes policy failures only, got {label}")
        self._fail(label, SimpleNamespace(physics_step=physics_step, control_step=control_step), detail)

    def update(self, f: SubstepFacts) -> list:
        if self.start is None:
            raise RuntimeError("start_from() must be called before update()")
        if self.succeeded:
            # The episode ends at the physics step that completed the hold; a caller that
            # keeps stepping would hide later contacts, so this is an error, not a no-op.
            raise RuntimeError("update() after success: the runner must stop at the success step")
        r, new = self.rules, []
        fixed = moving = named_other = False
        for verdict in f.verdicts:
            if verdict.kind == JAW_UTENSIL:
                fixed |= verdict.jaw == "fixed"
                moving |= verdict.jaw == "moving"
            elif any(name.startswith(self.named + "_") for name in verdict.geoms):
                named_other = True
            for label in verdict.labels:
                if self._fail(label, f, list(verdict.geoms)):
                    new.append(label)
        p0, r0 = self.start[self.spare]
        self.spare_shift = max(self.spare_shift, float(np.linalg.norm(f.positions[self.spare] - p0)))
        self.spare_yaw = max(self.spare_yaw, _yaw_change_deg(_yaw(f.rotations[self.spare]), _yaw(r0)))
        if self.spare_shift >= r["spare_max_shift_m"] or self.spare_yaw >= r["spare_max_yaw_deg"]:
            if self._fail("SPARE_DISTURBED", f, {"shift_m": self.spare_shift, "yaw_deg": self.spare_yaw}):
                new.append("SPARE_DISTURBED")
        c0, _ = self.start["cup"]
        self.cup_shift = max(self.cup_shift, float(np.linalg.norm(f.positions["cup"] - c0)))
        self.cup_tilt = max(self.cup_tilt, _tilt_deg(f.rotations["cup"]))
        if self.cup_shift >= r["cup_max_shift_m"] or self.cup_tilt >= r["cup_max_tilt_deg"]:
            if self._fail("CUP_DISTURBED", f, {"shift_m": self.cup_shift, "tilt_deg": self.cup_tilt}):
                new.append("CUP_DISTURBED")
        (cx, cy), (hx, hy) = self.table_center, self.table_half
        for item, p in f.positions.items():
            if abs(p[0] - cx) > hx or abs(p[1] - cy) > hy or p[2] < r["workspace_min_z_m"]:
                if self._fail("ITEM_FELL_OR_OUT", f, item):
                    new.append("ITEM_FELL_OR_OUT")
        for item in (self.named, self.spare):
            lift = float(f.positions[item][2] - self.start[item][0][2])
            self.max_lift[item] = max(self.max_lift[item], lift)
            if lift >= r["lift_height_m"] and self.lift_step[item] is None:
                self.lift_step[item] = f.control_step
        named_lift = float(f.positions[self.named][2] - self.start[self.named][0][2])
        if named_lift >= r["lift_height_m"] and not named_other and (fixed or moving):
            rel_p = f.gripper_rot.T @ (f.positions[self.named] - f.gripper_pos)
            rel_r = f.gripper_rot.T @ f.rotations[self.named]
            self._window.append((fixed and moving, rel_p, rel_r,
                                 f.linear_speed[self.named], f.angular_speed[self.named]))
            self._streak += 1
        else:
            self._window.clear()
            self._streak = 0
        self.longest_streak = max(self.longest_streak, self._streak)
        if len(self._window) == self.steps.hold and self._window_ok():
            if self.hold_ok_at is None:
                self.hold_ok_at = (f.physics_step, f.control_step)
            if not self.failures:
                self.success_at = (f.physics_step, f.control_step)
        return [label for label in new if label in SEVERE]

    def _window_ok(self) -> bool:
        r, w = self.rules, list(self._window)
        both = [entry[0] for entry in w]
        if sum(both) / len(w) < r["both_jaw_fraction"]:
            return False
        gap = longest = 0
        for b in both:
            gap = 0 if b else gap + 1
            longest = max(longest, gap)
        if longest > self.steps.max_gap:
            return False
        p0, r0 = w[0][1], w[0][2]
        for _, p, rot, _, _ in w:
            if np.linalg.norm(p - p0) >= r["hold_max_shift_m"] or _rotation_deg(r0, rot) >= r["hold_max_turn_deg"]:
                return False
        return all(v < r["final_max_speed_mps"] and spin < r["final_max_angular_speed_rps"]
                   for *_, v, spin in w[-self.steps.final_speed:])

    def finish(self, stop: str, *, physics_step: int, control_step: int) -> PickOutcome:
        if stop not in STOPS:
            raise ValueError(f"unknown stop: {stop}")
        if stop == "success" and not self.succeeded:
            raise ValueError("finish('success') without a completed hold")
        if self.succeeded and stop != "success":
            raise ValueError("the episode must end at success")
        if stop == "deadline" and self.hold_ok_at is None:
            at = SimpleNamespace(physics_step=physics_step, control_step=control_step)
            if self.lift_step[self.named] is None:
                self._fail("NO_LIFT", at)
            elif 0 < self._streak < self.steps.hold:
                self._fail("TIMEOUT", at)
            else:
                self._fail("IMPROPER_HOLD", at)
        ordered = sorted(self.failures.values(), key=lambda item: item["physics_step"])
        failed_rules = sorted({item["rule"] for item in ordered if item["rule"] in RULES})
        if stop == "early_stop":
            unevaluable = [rule for rule in ("R2", "R5") if rule not in failed_rules]
        elif stop == "crash":
            unevaluable = [rule for rule in RULES if rule not in failed_rules]
        else:
            unevaluable = []
        n, s = self.lift_step[self.named], self.lift_step[self.spare]
        if n is None and s is None:
            picked = "neither"
        elif s is None or (n is not None and n < s):
            picked = "named_first"
        elif n is None or s < n:
            picked = "spare_first"
        else:
            picked = "simultaneous"
        return PickOutcome(
            success=stop == "success", stop=stop, failures=ordered,
            first_failure=ordered[0]["label"] if ordered else None, failed_rules=failed_rules,
            unevaluable_rules=unevaluable, picked=picked, lifted_both=n is not None and s is not None,
            hold_completed=self.hold_ok_at is not None,
            hold_completed_physics_step=self.hold_ok_at[0] if self.hold_ok_at else None,
            hold_completed_control_step=self.hold_ok_at[1] if self.hold_ok_at else None,
            success_physics_step=self.success_at[0] if self.success_at else None,
            success_control_step=self.success_at[1] if self.success_at else None,
            max_lift_m=dict(self.max_lift), spare_max_shift_m=self.spare_shift, spare_max_yaw_deg=self.spare_yaw,
            cup_max_shift_m=self.cup_shift, cup_max_tilt_deg=self.cup_tilt,
            longest_eligible_streak=self.longest_streak)
