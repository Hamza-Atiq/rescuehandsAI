"""Jaw-to-table forces from a real, speed-measured cartesian descent (Plan 2 Task 4, spec §5).

Run from the repo root (PowerShell):
    $env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe scripts/measure_forces.py

The right gripper (open, pitch 1.4, as in the teacher's reach) is lowered onto bare table at a
fixed cartesian speed: the gripper SITE target drops `speed * control_dt` metres per control
step and the IK solver the expert uses turns each target into joint targets. The achieved speed
is measured from the site height at every physics step and reported next to the requested one.
Forces are sampled at every physics step through `on_substep`. Nothing is frozen here.
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
from rescuehandsai.pick_contacts import JAW_TABLE, ContactClassifier  # noqa: E402
from rescuehandsai.sim import MujocoSimulation  # noqa: E402

SEED = 3100000                      # development seed; the spot below is bare table in every scene
SPOT_XY = (0.15, 0.22)              # bare table: ~0.1 m from the utensil slots, ~0.08 m from the plate edge
PITCH, APPROACH_XY = 1.4, None     # reachable here only with a free approach direction
START_HEIGHT_M = 0.04               # site height above its first-contact height, roughly
SETTLE_STEPS = 40
GENTLE_SPEED = 0.01
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
    state = {"step": 0, "contact_step": None, "any_contact_step": None, "first_shapes": None, "phase": "descend"}
    samples, other_robot_table = [], []

    def on_substep():
        state["step"] += 1
        z = float(sim.data.site_xpos[site][2])
        jaw, other = 0.0, []
        for v in classifier.contacts(sim.data):
            if "table" not in v.geoms:
                continue
            if v.kind == JAW_TABLE and v.jaw is not None:
                jaw += v.force
            elif any(name.startswith("right_arm/") for name in v.geoms):
                other.append({"geoms": list(v.geoms), "force_n": v.force, "labels": list(v.labels)})
        if jaw > 0 and state["contact_step"] is None:
            state["contact_step"] = state["step"]
            state["contact_z"] = z
        if (jaw > 0 or other) and state["any_contact_step"] is None:
            # First touch by ANY right-arm shape; the approach speed is measured just before it.
            state["any_contact_step"] = state["step"]
            state["first_shapes"] = sorted({n for c in other for n in c["geoms"] if n != "table"}
                                           | ({"jaw pad"} if jaw > 0 else set()))
        samples.append({"step": state["step"], "z_m": z, "force_n": jaw,
                        "other_force_n": sum(c["force_n"] for c in other), "phase": state["phase"]})
        if other:
            other_robot_table.append({"step": state["step"], "contacts": other})
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
        # Gentle rest: command the height where the pads first touched, not the lagging target below it.
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
    if state["any_contact_step"] is not None:
        hi = state["any_contact_step"]
        window = [s for s in samples if hi - round(0.05 / physics_dt) <= s["step"] <= hi]
        if len(window) > 1:
            dz_steps = [-(b["z_m"] - a["z_m"]) / physics_dt for a, b in zip(window, window[1:])]
            speed = float(np.median(dz_steps))
    touching = [s["force_n"] for s in samples if s["force_n"] > 0]
    return {"requested_speed_m_per_s": target_speed_m_per_s, "press_depth_m": None if stop_at_contact
            else press_depth_m, "measured_speed_m_per_s": speed, "contact_started_step": state["contact_step"],
            "first_touch_step": state["any_contact_step"], "first_touch_shapes": state["first_shapes"],
            "max_force_n": max(touching) if touching else 0.0, "forces": samples,
            "other_robot_table_contacts": other_robot_table}


def summarise(result):
    touching = np.array([s["force_n"] for s in result["forces"] if s["force_n"] > 0])
    req, got = result["requested_speed_m_per_s"], result["measured_speed_m_per_s"]
    return {"samples": int(touching.size),
            "max_n": float(touching.max()) if touching.size else 0.0,
            "p95_n": float(np.percentile(touching, 95)) if touching.size else 0.0,
            "mean_n": float(touching.mean()) if touching.size else 0.0,
            "requested_speed_m_per_s": req, "measured_speed_m_per_s": got,
            "speed_off_by_more_than_20pct": None if got is None else abs(got - req) > 0.2 * req,
            "press_depth_m": result["press_depth_m"],
            "first_touch_shapes": result["first_touch_shapes"],
            "other_robot_table_contact_steps": len(result["other_robot_table_contacts"]),
            "other_robot_table_shapes": sorted({c["geoms"][0] if c["geoms"][0] != "table" else c["geoms"][1]
                                                for e in result["other_robot_table_contacts"]
                                                for c in e["contacts"]}),
            "other_robot_table_max_n": max((c["force_n"] for e in result["other_robot_table_contacts"]
                                            for c in e["contacts"]), default=0.0)}


def measure_forces(sim, speeds=PRESS_SPEEDS, depths=PRESS_DEPTHS):
    contacts = load_contacts()
    ik = IKSolver(sim.model)
    factory = lambda model: ContactClassifier(model, "fork", contacts)  # noqa: E731
    t0 = time.perf_counter()
    conditions = {"gentle": descend(sim, factory, ik, target_speed_m_per_s=GENTLE_SPEED, press_depth_m=0.0,
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
    out = ROOT / "results/measurements/forces.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    for name, s in report["summary"].items():
        print(f"{name:24s} req {s['requested_speed_m_per_s']:.3f} got {s['measured_speed_m_per_s']} "
              f"depth {s['press_depth_m']} n={s['samples']} mean {s['mean_n']:.2f} p95 {s['p95_n']:.2f} "
              f"max {s['max_n']:.2f} | other robot-table: {s['other_robot_table_shapes']} "
              f"max {s['other_robot_table_max_n']:.2f}")
    print("Saved", out)


if __name__ == "__main__":
    main()
