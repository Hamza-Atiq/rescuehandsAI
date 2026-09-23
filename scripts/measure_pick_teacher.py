"""Run the pick teacher on four development scenes in four configurations and report everything.

Run from the repo root (PowerShell):
    $env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe scripts/measure_pick_teacher.py
    ... scripts/measure_pick_teacher.py --physics-version 3   (experimental NoSlip world)
Development seeds 3100000-3100003 only. Changes nothing; failed and invalid runs stay in the report.
These are 4 scenes x 4 configurations, not 16 independent scenes.
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rescuehandsai.pick_cells import CELLS, make_pick_task  # noqa: E402
from rescuehandsai.pick_config import PICK_PHYSICS_VERSION, load_rules  # noqa: E402
from rescuehandsai.pick_records import InvalidRun  # noqa: E402
from rescuehandsai.pick_runner import PickEpisodeRunner  # noqa: E402
from rescuehandsai.pick_teacher import PickTeacher  # noqa: E402
from rescuehandsai.pick_teacher_diagnostics import TableContactTally, UtensilContactTally, lift_chain  # noqa: E402
from rescuehandsai.sim import MujocoSimulation  # noqa: E402

SEEDS = range(3100000, 3100004)
SAMPLED = ("utensil_squeeze", "utensil_lift", "hold")


def run_one(sim, seed, cell, physics_version=PICK_PHYSICS_VERSION):
    task = make_pick_task(seed, cell, "T1", physics_version=physics_version)
    teacher = PickTeacher()
    tally, samples, state = TableContactTally(), [], {}
    utensil = UtensilContactTally()
    original = sim.step
    # Only the version field differs from configs/pick_rules.json; every rule value is the file's.
    rules = dict(load_rules(), physics_version=physics_version)
    real_step, clock = mujoco.mj_step, {"s": 0.0, "n": 0}
    # The runner times policy calls only when the policy asks for images; the teacher never does,
    # so time teacher.act here (it includes the grasp planning).
    real_act, teacher_clock = teacher.act, {"s": 0.0}

    def timed_act(obs):
        t = time.perf_counter()
        try:
            return real_act(obs)
        finally:
            teacher_clock["s"] += time.perf_counter() - t
    teacher.act = timed_act

    def timed_step(m, d):
        t = time.perf_counter()
        real_step(m, d)
        clock["s"] += time.perf_counter() - t
        clock["n"] += 1

    def watched(action, *, on_substep=None, **kw):
        if "start_z" not in state:
            state["start_z"] = float(sim.data.body(task.utensil).xpos[2])
            state["table"] = sim.model.geom("table").id
            m = sim.model
            state["utensil_geoms"] = {g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name == task.utensil}

        def cb():
            m, d = sim.model, sim.data
            move = teacher.expert._move if teacher.expert is not None else None
            label = "hold" if teacher.phase == "hold" else (move.label if move else None)
            force = np.zeros(6)
            for i, c in enumerate(d.contact):
                g1, g2 = int(c.geom1), int(c.geom2)
                ug = state["utensil_geoms"]
                if (g1 in ug) != (g2 in ug):
                    other = g2 if g1 in ug else g1
                    if sim.geom_arm[other] is not None:
                        mujoco.mj_contactForce(m, d, i, force)
                        utensil.add(m.geom(other).name or f"geom_{other}", force[0], label)
                if state["table"] not in (g1, g2):
                    continue
                g = g2 if g1 == state["table"] else g1
                if sim.geom_arm[g] is None:
                    continue
                mujoco.mj_contactForce(m, d, i, force)
                tally.add(m.geom(g).name or f"geom_{g}", force[0], c.dist, label)
            if label in SAMPLED:
                site = d.site("right_arm/gripperframe")
                body = d.body(task.utensil)
                R = site.xmat.reshape(3, 3)
                samples.append({"label": label, "site_z": float(site.xpos[2]), "utensil_z": float(body.xpos[2]),
                                "in_hand": (R.T @ (body.xpos - site.xpos)).tolist()})
            return on_substep() if on_substep else False
        return original(action, on_substep=cb, **kw)

    sim.step = watched
    mujoco.mj_step = timed_step
    t_wall = time.perf_counter()
    record = None
    try:
        record = PickEpisodeRunner(sim, teacher, rules=rules).run(task)
        o = record["outcome"]
        ep = {"valid": True, "success": o["success"], "first_failure": o["first_failure"],
              "failures": [{k: f[k] for k in ("label", "physics_step", "detail")} for f in o["failures"]],
              "picked": o["picked"], "hold_completed": o["hold_completed"],
              "max_lift_m": o["max_lift_m"], "spare_max_shift_m": o["spare_max_shift_m"],
              "cup_max_shift_m": o["cup_max_shift_m"]}
    except InvalidRun as exc:
        ep = {"valid": False, "error": str(exc)}
    finally:
        sim.step = original
        mujoco.mj_step = real_step
    plans = teacher.expert.grasp_plans if teacher.expert is not None else []
    ep.update(seed=seed, cell=cell, utensil=task.utensil, grasp_plans=plans, table_contacts=tally.rows(),
              lift_chain=lift_chain(plans[-1], samples, state["start_z"]) if plans and "start_z" in state else None)
    ep.update(physics_version=physics_version, utensil_contacts=utensil.rows(),
              other_warnings=(record or {}).get("other_warnings", {}),
              timing={"physics_step_s": clock["s"], "physics_steps": clock["n"],
                      "teacher_s": teacher_clock["s"],
                      "wall_s": time.perf_counter() - t_wall})
    return ep


def summary(episodes):
    valid = [e for e in episodes if e["valid"]]
    labels = lambda e: {f["label"] for f in e["failures"]}  # noqa: E731
    return {"runs": len(episodes), "valid": len(valid),
            "success": sum(e["success"] for e in valid),
            "hold_completed": sum(e["hold_completed"] for e in valid),
            "any_table_contact": sum(bool(e["table_contacts"]) for e in valid),
            "forbidden_contact": sum("FORBIDDEN_CONTACT" in labels(e) for e in valid),
            "wrong_item_touched": sum("WRONG_ITEM_TOUCHED" in labels(e) for e in valid),
            "spare_disturbed": sum("SPARE_DISTURBED" in labels(e) for e in valid),
            "cup_disturbed": sum("CUP_DISTURBED" in labels(e) for e in valid),
            "no_lift": sum("NO_LIFT" in labels(e) for e in valid),
            "improper_hold": sum("IMPROPER_HOLD" in labels(e) for e in valid),
            "policy_error": sum("POLICY_ERROR" in labels(e) for e in valid),
            "named_rise_at_least_5cm": sum(e["max_lift_m"].get(e["utensil"], 0.0) >= 0.05 for e in valid),
            "any_warning": sum(bool(e.get("other_warnings")) for e in episodes),
            "mean_physics_ms_per_step": 1000 * sum(e["timing"]["physics_step_s"] for e in episodes)
            / max(1, sum(e["timing"]["physics_steps"] for e in episodes))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--physics-version", type=int, choices=(2, 3), default=PICK_PHYSICS_VERSION)
    args = parser.parse_args()
    sim = MujocoSimulation(physics_version=args.physics_version)
    try:
        episodes = []
        for seed in SEEDS:
            for cell in CELLS:
                ep = run_one(sim, seed, cell, args.physics_version)
                episodes.append(ep)
                lc = ep["lift_chain"] or {}
                print(seed, cell, "success" if ep.get("success") else [f["label"] for f in ep.get("failures", [])],
                      "| table shapes:", [r["shape"] for r in ep["table_contacts"]],
                      "| hand rise:", lc.get("hand_rise_m"), "| slip:", lc.get("max_slip_m"), flush=True)
    finally:
        sim.close()
    report = {"scope": "4 development scenes x 4 configurations (not 16 independent scenes); diagnostic",
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "working_tree_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT,
                                                             text=True).splitlines(),
              "summary": summary(episodes), "episodes": episodes}
    stem = "pick_teacher_revised" if args.physics_version == 2 else f"pick_teacher_v{args.physics_version}"
    out = ROOT / f"results/measurements/{stem}_{date.today():%Y%m%d}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report["summary"], indent=1))
    print("Saved", out)


if __name__ == "__main__":
    main()
