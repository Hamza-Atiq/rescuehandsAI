"""Run the pick teacher on four development scenes in four configurations and report everything.

Run from the repo root (PowerShell):
    $env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe scripts/measure_pick_teacher.py
Development seeds 3100000-3100003 only. Changes nothing; failed and invalid runs stay in the report.
These are 4 scenes x 4 configurations, not 16 independent scenes.
"""
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rescuehandsai.pick_cells import CELLS, make_pick_task  # noqa: E402
from rescuehandsai.pick_records import InvalidRun  # noqa: E402
from rescuehandsai.pick_runner import PickEpisodeRunner  # noqa: E402
from rescuehandsai.pick_teacher import PickTeacher  # noqa: E402
from rescuehandsai.pick_teacher_diagnostics import TableContactTally, lift_chain  # noqa: E402
from rescuehandsai.sim import MujocoSimulation  # noqa: E402

SEEDS = range(3100000, 3100004)
SAMPLED = ("utensil_squeeze", "utensil_lift", "hold")


def run_one(sim, seed, cell):
    task = make_pick_task(seed, cell, "T1")
    teacher = PickTeacher()
    tally, samples, state = TableContactTally(), [], {}
    original = sim.step

    def watched(action, *, on_substep=None, **kw):
        if "start_z" not in state:
            state["start_z"] = float(sim.data.body(task.utensil).xpos[2])
            state["table"] = sim.model.geom("table").id

        def cb():
            m, d = sim.model, sim.data
            move = teacher.expert._move if teacher.expert is not None else None
            label = "hold" if teacher.phase == "hold" else (move.label if move else None)
            force = np.zeros(6)
            for i, c in enumerate(d.contact):
                g1, g2 = int(c.geom1), int(c.geom2)
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
    try:
        record = PickEpisodeRunner(sim, teacher).run(task)
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
    plans = teacher.expert.grasp_plans if teacher.expert is not None else []
    ep.update(seed=seed, cell=cell, utensil=task.utensil, grasp_plans=plans, table_contacts=tally.rows(),
              lift_chain=lift_chain(plans[-1], samples, state["start_z"]) if plans and "start_z" in state else None)
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
            "named_rise_at_least_5cm": sum(e["max_lift_m"].get(e["utensil"], 0.0) >= 0.05 for e in valid)}


def main():
    sim = MujocoSimulation(physics_version=2)
    try:
        episodes = []
        for seed in SEEDS:
            for cell in CELLS:
                ep = run_one(sim, seed, cell)
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
    out = ROOT / f"results/measurements/pick_teacher_revised_{date.today():%Y%m%d}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report["summary"], indent=1))
    print("Saved", out)


if __name__ == "__main__":
    main()
