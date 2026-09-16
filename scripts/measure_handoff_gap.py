"""Measure how long a real teacher hand-off goes with nobody holding the utensil.

The hand-off check must tolerate brief contact flicker but not an unlimited gap,
so its bound comes from this measurement, not a guess.

  PYTHONPATH=src .venv-sim/Scripts/python.exe scripts/measure_handoff_gap.py --seeds 0:10
"""
import argparse
import json

from rescuehandsai.auditor import compute_facts
from rescuehandsai.perturb import GripperGlitch
from rescuehandsai.policies.scripted import ScriptedPolicy
from rescuehandsai.runner import EpisodeRunner
from rescuehandsai.sim import MujocoSimulation
from rescuehandsai.task import make_task


class GapProbe:
    """Longest run of control steps where the utensil is airborne, unheld, after both hands held it."""

    def __init__(self, item):
        self.item, self.shared_seen, self.run, self.longest = item, False, 0, 0

    def __call__(self, sim, state):
        facts = compute_facts(sim)
        held, airborne = facts.held_by[self.item], not facts.supported[self.item]
        if held == {"left_arm", "right_arm"} and airborne:
            self.shared_seen = True
        if self.shared_seen and airborne and not held:
            self.run += 1
            self.longest = max(self.longest, self.run)
        else:
            self.run = 0
        if not airborne:
            self.shared_seen = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0:10")
    parser.add_argument("--fault", choices=["none", "glitch"], default="none")
    args = parser.parse_args()
    start, stop = (int(v) for v in args.seeds.split(":"))
    sim = MujocoSimulation()
    rows = []
    for seed in range(start, stop):
        task = make_task(seed)
        probe = GapProbe(task.utensil)
        log = EpisodeRunner(sim, ScriptedPolicy(), supervisor=True,
                            fault=GripperGlitch() if args.fault == "glitch" else None, on_frame=probe).run(task)
        rows.append({"seed": seed, "state": log.state, "handoff": log.outcome.get("handoff"),
                     "longest_unheld_airborne_steps": probe.longest})
        print(json.dumps(rows[-1]), flush=True)
    print(json.dumps({"max_over_successful_handoffs": max(
        (r["longest_unheld_airborne_steps"] for r in rows if r["handoff"]), default=None)}))
    sim.close()


if __name__ == "__main__":
    main()
