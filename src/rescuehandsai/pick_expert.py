# src/rescuehandsai/pick_expert.py
"""Pick-only expert: ScriptedExpert with two changes for the pick-and-lift milestone.

1. The grasp height is chosen by checking every collidable shape of the right hand
   along the whole commanded path (approach, reach, jaw closing, lift), and must keep
   1 mm above the table. The full-task expert aims the fingertip point 1.5 mm up and
   so drives the fixed-jaw mesh about 4.5 mm into the table
   (docs/research/2026-09-22-collision-inspection/FINDINGS.md).
2. Left-arm staging is blocked with a named error: the milestone forbids left-arm
   object contact (spec section 5).

ScriptedExpert (expert.py) is unchanged; the full-task teacher keeps both behaviours.
IK lives here only; the deployed policy never uses it.
"""
import numpy as np

from .auditor import compute_facts
from .expert import (CLOSE, GRASP_DEPTH, LIFTED_HEIGHT, OPEN, RIGHT_SIGN, TABLE_CLEARANCE, UTENSIL_MARGIN,
                     LostItemError, PlanningError, ScriptedExpert)
from .motion import Move
from .pick_clearance import ClearanceChecker

MIN_TABLE_CLEARANCE_M = 0.001
GRASP_HEIGHT_STEP_M = 0.0005
# Above ~9 mm the sweep kept under ~70 % of the handle between the pads (FINDINGS §4).
MAX_GRASP_CENTER_Z_M = 0.010
LIFT_REQUEST_M = 0.05
GRASP_PITCHES = (1.3, 1.4, 1.2, 1.0)


class StagingBlockedError(PlanningError):
    """The pick-only teacher never hands the utensil to the left arm."""


class NoClearGraspError(PlanningError):
    """A grasp is reachable, but no tried height keeps the hand clear of the table."""


class PickExpert(ScriptedExpert):
    def __init__(self, sim, task, subtasks=("pick_utensil",)):
        super().__init__(sim, task, subtasks)
        self.clearance = ClearanceChecker(sim.model, "right_arm")
        self.grasp_plans = []

    def _stage_utensil(self):
        raise StagingBlockedError(
            f"left-arm staging is disabled in the pick-only teacher: the right arm cannot grasp the "
            f"{self.task.utensil}")
        yield  # unreachable; keeps this a generator like ScriptedExpert._stage_utensil

    def _plan_grasp(self, item, g):
        """Lowest grasp height whose whole commanded path keeps the hand 1 mm above the table.

        Returns (q, q_lift, q_pre, plan) or None when no height is reachable at all
        (the staging case). Raises NoClearGraspError when heights are reachable but none clears.
        """
        arm, grip = "right_arm", "right_arm/gripper"
        pos, axis = self._utensil_frame(item)
        base = pos + g["end"] * axis
        qpos = self.sim.data.qpos.copy()
        now = dict(self.follower.targets)
        tried, reachable = [], False
        start_z = max(TABLE_CLEARANCE, pos[2] - GRASP_DEPTH)
        for z in np.arange(start_z, MAX_GRASP_CENTER_Z_M + 1e-9, GRASP_HEIGHT_STEP_M):
            center = base.copy()
            center[2] = z
            try:
                q, q_lift, site, finger, pitch = self._grasp_with_clearance(
                    arm, center, g["half_width"], LIFT_REQUEST_M, approach_xy=axis[:2], pitches=GRASP_PITCHES,
                    margin=UTENSIL_MARGIN, closing_sign=RIGHT_SIGN)
                q_pre = self._approach_pose(arm, site, finger, pitch, axis[:2], q, RIGHT_SIGN)
            except PlanningError:
                tried.append([round(float(z), 5), "unreachable"])
                continue
            reachable = True
            pre = now | q_pre | {grip: OPEN}
            at = pre | q
            closed = at | {grip: CLOSE}
            lifted = closed | q_lift
            legs = {name: self.clearance.along(qpos, a, b)
                    for name, (a, b) in {"approach": (now, pre), "reach": (pre, at),
                                         "close": (at, closed), "lift": (closed, lifted)}.items()}
            low = min(c.z for c in legs.values())
            tried.append([round(float(z), 5), round(low, 5)])
            if low >= MIN_TABLE_CLEARANCE_M:
                plan = {"center_z_m": float(z), "pitch": float(pitch),
                        "reach_site_z_m": self.clearance.site_z(qpos, at),
                        "lift_request_z_m": float(site[2] + LIFT_REQUEST_M),
                        "lift_site_z_m": self.clearance.site_z(qpos, lifted),
                        "min_clearance_m": low,
                        "legs": {k: {"z_m": c.z, "shape": c.shape, "fraction": c.fraction, "samples": c.samples,
                                     "max_point_step_m": c.max_point_step_m} for k, c in legs.items()},
                        "tried": tried}
                self.grasp_plans.append(plan)
                return q, q_lift, q_pre, plan
        if reachable:
            raise NoClearGraspError(
                f"TARGET_MISSED: no grasp height up to {MAX_GRASP_CENTER_Z_M} m keeps the right hand "
                f"{MIN_TABLE_CLEARANCE_M} m above the table; tried {tried}")
        return None

    def _pick_utensil(self, attempts: int = 3):
        """Same sequence and retries as ScriptedExpert._pick_utensil, with a clearance-checked grasp."""
        grip, item = "right_arm/gripper", self.task.utensil
        g = self._handle_geometry(item)
        if attempts < 1:
            raise ValueError(f"attempts must be at least 1, got {attempts}")
        missed = 0
        while True:
            planned = self._plan_grasp(item, g)
            if planned is None:
                yield from self._stage_utensil()   # raises StagingBlockedError
            q, q_lift, q_pre, plan = planned
            self._right_pitch = plan["pitch"]
            yield Move(q_pre | {grip: OPEN}, 25, "utensil_approach")
            yield Move(q, 18, "utensil_reach")
            yield Move({grip: CLOSE}, 12, "utensil_close")
            yield Move({}, 4, "utensil_squeeze")
            yield Move(q_lift, 18, "utensil_lift")
            if compute_facts(self.sim).height[item] > LIFTED_HEIGHT:
                break
            missed += 1
            if missed >= attempts:
                raise LostItemError(f"FAILED_GRASP: the right hand could not grasp the {item}")
            yield Move({grip: OPEN}, 8, "regrasp_open")
            yield Move(q_pre, 15, "regrasp_back_off")
            yield Move({}, 5, "regrasp_look")
        self._q_right = q_lift
