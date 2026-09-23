"""Measurement helpers for the pick teacher. Diagnostics only: no rule or threshold lives here."""
import numpy as np

from .pick_expert import LIFT_REQUEST_M


class TableContactTally:
    """Per robot shape: how often it touched the table, peak normal force, deepest overlap, first move."""

    def __init__(self):
        self._rows = {}

    def add(self, shape: str, force_n: float, dist_m: float, label):
        row = self._rows.setdefault(shape, {"shape": shape, "samples": 0, "peak_force_n": 0.0,
                                            "deepest_m": 0.0, "first_label": label})
        row["samples"] += 1
        row["peak_force_n"] = max(row["peak_force_n"], float(force_n))
        row["deepest_m"] = min(row["deepest_m"], float(dist_m))

    def rows(self) -> list:
        return [dict(self._rows[k]) for k in sorted(self._rows)]


class UtensilContactTally:
    """Per robot shape touching the named utensil: samples, peak normal force, first move."""

    def __init__(self):
        self._rows = {}

    def add(self, shape: str, force_n: float, label):
        row = self._rows.setdefault(shape, {"shape": shape, "samples": 0, "peak_force_n": 0.0,
                                            "first_label": label})
        row["samples"] += 1
        row["peak_force_n"] = max(row["peak_force_n"], float(force_n))

    def rows(self) -> list:
        return [dict(self._rows[k]) for k in sorted(self._rows)]


def lift_chain(plan: dict, samples: list, utensil_start_z: float) -> dict:
    """Split a short lift into its causes.

    requested_rise_m: what the teacher asks for (`LIFT_REQUEST_M`, 0.06 m).
    solved_rise_m: `plan["lift_site_z_m"] - plan["reach_site_z_m"]`, what IK actually
        solved -- shows whether IK error explains the shortfall.
    reach_gap_m: hand height at the last `utensil_squeeze` sample minus
        `plan["reach_site_z_m"]`. A positive gap means the hand stopped above its
        planned pose.
    hand_rise_m: highest hand height in `utensil_lift`/`hold` minus the hand height at
        the first `utensil_lift` sample.
    utensil_rise_in_lift_m: utensil height at the last `utensil_lift` sample minus the
        first.
    utensil_final_rise_m: last utensil height minus `utensil_start_z`.
    max_slip_m: largest distance of `in_hand` from its value at the first
        `utensil_lift` sample, over `utensil_lift`/`hold`.

    A key whose samples are missing (for example, no `utensil_lift` sample because the
    episode ended early) is None, never 0.
    """
    squeeze = [x for x in samples if x["label"] == "utensil_squeeze"]
    lift = [x for x in samples if x["label"] == "utensil_lift"]
    after = [x for x in samples if x["label"] in ("utensil_lift", "hold")]
    out = {"requested_rise_m": LIFT_REQUEST_M,
           "solved_rise_m": plan["lift_site_z_m"] - plan["reach_site_z_m"],
           "reach_gap_m": squeeze[-1]["site_z"] - plan["reach_site_z_m"] if squeeze else None,
           "hand_rise_m": None, "utensil_rise_in_lift_m": None, "max_slip_m": None,
           "utensil_final_rise_m": samples[-1]["utensil_z"] - utensil_start_z if samples else None}
    if lift:
        out["hand_rise_m"] = max(x["site_z"] for x in after) - lift[0]["site_z"]
        out["utensil_rise_in_lift_m"] = lift[-1]["utensil_z"] - lift[0]["utensil_z"]
        ref = np.asarray(lift[0]["in_hand"], dtype=float)
        out["max_slip_m"] = max(float(np.linalg.norm(np.asarray(x["in_hand"], dtype=float) - ref)) for x in after)
    return out
