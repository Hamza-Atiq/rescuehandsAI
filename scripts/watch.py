"""Watch an episode live in MuJoCo's 3D viewer (drag to rotate, scroll to zoom).

  PYTHONPATH=src .venv-sim/Scripts/python.exe scripts/watch.py --seed 0 [--fault glitch] [--no-supervisor]

Runs the scripted teacher through the same runner used for evaluation, at about
real-time speed. The window title bar shows nothing; the terminal prints each
failure event and the final result.
"""
import argparse
import time

import mujoco.viewer

from rescuehandsai.perturb import GripperGlitch
from rescuehandsai.policies.scripted import ScriptedPolicy
from rescuehandsai.runner import EpisodeRunner
from rescuehandsai.sim import MujocoSimulation
from rescuehandsai.task import make_task


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fault", choices=["none", "glitch"], default="none")
    parser.add_argument("--no-supervisor", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0, help="1.0 = real time")
    args = parser.parse_args()

    sim = MujocoSimulation()
    task = make_task(args.seed)
    print("Instruction:", task.instruction)
    state = {"viewer": None, "model": None, "last": time.perf_counter()}

    def on_frame(sim, status):
        if state["model"] is not sim.model:  # a reset builds a new randomized model
            if state["viewer"] is not None:
                state["viewer"].close()
            state["viewer"] = mujoco.viewer.launch_passive(sim.model, sim.data)
            state["model"] = sim.model
        viewer = state["viewer"]
        if not viewer.is_running():
            raise KeyboardInterrupt
        viewer.sync()
        wait = sim.config["control_dt"] / args.speed - (time.perf_counter() - state["last"])
        if wait > 0:
            time.sleep(wait)
        state["last"] = time.perf_counter()

    runner = EpisodeRunner(sim, ScriptedPolicy(), supervisor=not args.no_supervisor,
                           fault=GripperGlitch() if args.fault == "glitch" else None,
                           max_steps=1500, on_frame=on_frame)
    try:
        log = runner.run(task)
        for event in log.events:
            print("event:", event)
        print(f"Result: {log.state}  failure={log.failure}  recoveries={log.recoveries}")
        print("Checks:", log.outcome)
        while state["viewer"] is not None and state["viewer"].is_running():
            state["viewer"].sync()
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Viewer closed.")
    finally:
        if state["viewer"] is not None:
            state["viewer"].close()
        sim.close()


if __name__ == "__main__":
    main()
