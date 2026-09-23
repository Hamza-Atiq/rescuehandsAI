"""Robot-to-table forces from a real, speed-measured cartesian descent (Plan 2 Task 4, spec §5).

Run from the repo root (PowerShell):
    $env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe scripts/measure_forces.py

The right gripper (open, pitch 1.4, as in the teacher's reach) is lowered onto bare table at a
fixed cartesian speed: the gripper SITE target drops `speed * control_dt` metres per control
step and the IK solver the expert uses turns each target into joint targets. The achieved speed
is measured from the site height at every physics step and reported next to the requested one.
Forces are sampled at every physics step through `on_substep`. Nothing is frozen here.

Every robot-table contact is forbidden (owner decision 23 Sep 2026), so these are controlled
BAD contacts: evidence for the severe-force early stop, never an "acceptable force" baseline.
Units match the checker: the severe-force rule compares each single contact's normal force, so
the headline number is the largest single contact. The per-step sum over contacts is kept as
context only. "First touch" means ANY right-arm shape touching the table; the touch-and-hold
condition stops descending there instead of waiting for a pad (the meshes always touch first).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rescuehandsai.contracts import BimanualAction  # noqa: E402
from rescuehandsai.expert import OPEN  # noqa: E402
from rescuehandsai.kinematics import IKSolver  # noqa: E402
from rescuehandsai.pick_config import PICK_PHYSICS_VERSION, load_contacts  # noqa: E402
from rescuehandsai.pick_contacts import ContactClassifier  # noqa: E402
from rescuehandsai.sim import MujocoSimulation  # noqa: E402

SEED = 3100000                      # development seed; the spot below is bare table in every scene
SPOT_XY = (0.15, 0.22)              # bare table: ~0.1 m from the utensil slots, ~0.08 m from the plate edge
PITCH, APPROACH_XY = 1.4, None     # reachable here only with a free approach direction
START_HEIGHT_M = 0.04               # site height above its first-contact height, roughly
SETTLE_STEPS = 40
TOUCH_SPEED = 0.01
PRESS_SPEEDS = (0.02, 0.05, 0.10)
PRESS_DEPTHS = (0.005, 0.010)
HOLD_S = 1.0
GRIP = "right_arm/gripper"


def _targets(sim, q):
    targets = dict(sim.home_targets)
    targets.update(q)
    targets[GRIP] = OPEN
    return targets


def descend(sim, classifier_factory, ik, *, target_speed_m_per_s, press_depth_m, stop_at_contact=False,
            hold_s=0.0, max_drop_m=0.08):
    """Lower the open right gripper onto the table; record forces at every physics step.

    classifier_factory(model) builds the ContactClassifier after the reset (geom ids are per model)."""
    sim.reset(SEED)
    classifier = classifier_factory(sim.model)
    site = sim.model.site("right_arm/gripperframe").id
    physics_dt, control_dt = sim.config["physics_dt"], sim.config["control_dt"]
    z0 = START_HEIGHT_M
    q = ik.solve("right_arm", (*SPOT_XY, z0), pitch=PITCH, approach_xy=APPROACH_XY, tol=0.0005)
    if q is None:
        raise RuntimeError(f"IK cannot reach the start pose {SPOT_XY, z0}")
    # Ramp from home to the start pose within the simulator's per-step command limit, then settle.
    start, goal = dict(sim.home_targets), _targets(sim, q)
    ramp = max(1, int(np.ceil(max(abs(goal[k] - start[k]) for k in goal) / (0.9 * sim.config["max_command_delta"]))))
    for i in range(1, ramp + SETTLE_STEPS + 1):
        a = min(1.0, i / ramp)
        targets = {k: start[k] + a * (goal[k] - start[k]) for k in goal}
        sim.step(BimanualAction(sim.observe(images=False).timestamp, targets), stop_on_cross_arm=False)
    state = {"step": 0, "contact_step": None, "contact_z": None, "first_shapes": None, "phase": "descend"}
    samples, contact_log = [], []

    def on_substep():
        state["step"] += 1
        z = float(sim.data.site_xpos[site][2])
        touching = [{"geoms": list(v.geoms), "force_n": v.force, "labels": list(v.labels)}
                    for v in classifier.contacts(sim.data)
                    if "table" in v.geoms and any(name.startswith("right_arm/") for name in v.geoms)]
        if touching and state["contact_step"] is None:
            # First touch by ANY right-arm shape; the approach speed is measured just before it.
            state["contact_step"], state["contact_z"] = state["step"], z
            state["first_shapes"] = sorted({n for c in touching for n in c["geoms"] if n != "table"})
        samples.append({"step": state["step"], "z_m": z, "phase": state["phase"],
                        "max_single_n": max((c["force_n"] for c in touching), default=0.0),
                        "sum_n": sum(c["force_n"] for c in touching), "contacts": len(touching)})
        if touching:
            contact_log.append({"step": state["step"], "contacts": touching})
        return False

    z_target, dz = z0, target_speed_m_per_s * control_dt
    floor = None
    for _ in range(int(max_drop_m / dz) + 200):
        if state["contact_step"] is not None and floor is None:
            floor = state["contact_z"] if stop_at_contact else state["contact_z"] - press_depth_m
        if floor is not None and z_target <= floor:
            break
        z_target = z_target - dz if floor is None else max(floor, z_target - dz)
        if z0 - z_target > max_drop_m:
            break
        q = ik.solve("right_arm", (*SPOT_XY, z_target), pitch=PITCH, approach_xy=APPROACH_XY, q_init=q, tol=0.0005)
        if q is None:
            raise RuntimeError(f"IK failed at z={z_target:.4f}")
        sim.step(BimanualAction(sim.observe(images=False).timestamp, _targets(sim, q)),
                 on_substep=on_substep, stop_on_cross_arm=False)
    if stop_at_contact and state["contact_step"] is not None:
        # Touch and hold: command the height of the first touch, not the lagging target below it.
        q = ik.solve("right_arm", (*SPOT_XY, state["contact_z"]), pitch=PITCH, approach_xy=APPROACH_XY, q_init=q,
                     tol=0.0005) or q
    state["phase"] = "hold"
    for _ in range(round(hold_s / control_dt)):
        sim.step(BimanualAction(sim.observe(images=False).timestamp, _targets(sim, q)),
                 on_substep=on_substep, stop_on_cross_arm=False)
    # Measured vertical speed at impact: median site speed over the 0.05 s before the first touch
    # by any right-arm shape. After the touch the hand is stopped by the table, so speeds measured
    # there (the first version of this script) are ~0 and say nothing about the descent.
    speed = None
    if state["contact_step"] is not None:
        hi = state["contact_step"]
        window = [x for x in samples if hi - round(0.05 / physics_dt) <= x["step"] <= hi]
        if len(window) > 1:
            dz_steps = [-(b["z_m"] - a["z_m"]) / physics_dt for a, b in zip(window, window[1:])]
            speed = float(np.median(dz_steps))
    return {"requested_speed_m_per_s": target_speed_m_per_s, "press_depth_m": None if stop_at_contact
            else press_depth_m, "measured_speed_m_per_s": speed, "first_touch_step": state["contact_step"],
            "first_touch_shapes": state["first_shapes"], "forces": samples, "table_contacts": contact_log}


def summarise(result):
    """Headline = largest SINGLE contact, the quantity the severe-force rule compares."""
    single = np.array([x["max_single_n"] for x in result["forces"] if x["contacts"]])
    summed = np.array([x["sum_n"] for x in result["forces"] if x["contacts"]])
    per_shape = {}
    for entry in result["table_contacts"]:
        for c in entry["contacts"]:
            shape = next(n for n in c["geoms"] if n != "table")
            per_shape[shape] = max(per_shape.get(shape, 0.0), c["force_n"])
    req, got = result["requested_speed_m_per_s"], result["measured_speed_m_per_s"]
    return {"samples": int(single.size),
            "max_single_n": float(single.max()) if single.size else 0.0,
            "p95_single_n": float(np.percentile(single, 95)) if single.size else 0.0,
            "max_step_sum_n": float(summed.max()) if summed.size else 0.0,
            "max_single_n_by_shape": dict(sorted(per_shape.items())),
            "requested_speed_m_per_s": req, "measured_speed_m_per_s": got,
            "speed_off_by_more_than_20pct": None if got is None else abs(got - req) > 0.2 * req,
            "press_depth_m": result["press_depth_m"], "first_touch_shapes": result["first_touch_shapes"]}


def measure_forces(sim, speeds=PRESS_SPEEDS, depths=PRESS_DEPTHS):
    contacts = load_contacts()
    ik = IKSolver(sim.model)
    factory = lambda model: ContactClassifier(model, "fork", contacts)  # noqa: E731
    t0 = time.perf_counter()
    conditions = {"touch_and_hold": descend(sim, factory, ik, target_speed_m_per_s=TOUCH_SPEED, press_depth_m=0.0,
                                    stop_at_contact=True, hold_s=HOLD_S)}
    for speed in speeds:
        for depth in depths:
            conditions[f"pressed_{speed:g}mps_{depth * 1000:g}mm"] = descend(
                sim, factory, ik, target_speed_m_per_s=speed, press_depth_m=depth, hold_s=HOLD_S)
    return {"conditions": conditions, "summary": {k: summarise(v) for k, v in conditions.items()},
            "runtime_s": time.perf_counter() - t0}


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    sim = MujocoSimulation(physics_version=PICK_PHYSICS_VERSION)
    try:
        report = measure_forces(sim)
    finally:
        sim.close()
    report.update(physics_version=PICK_PHYSICS_VERSION, seed=SEED, spot_xy=SPOT_XY, pitch=PITCH,
                  gripper="open", hold_s=HOLD_S,
                  source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    out = ROOT / "results/measurements/forces_single_contact.json"  # forces.json = earlier pad-sum version
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    for name, s in report["summary"].items():
        print(f"{name:24s} req {s['requested_speed_m_per_s']:.3f} got {s['measured_speed_m_per_s']} "
              f"depth {s['press_depth_m']} n={s['samples']} single max {s['max_single_n']:.2f} "
              f"p95 {s['p95_single_n']:.2f} | step-sum max {s['max_step_sum_n']:.2f} | first {s['first_touch_shapes']}")
    print("Saved", out)


if __name__ == "__main__":
    main()
