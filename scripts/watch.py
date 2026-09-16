"""Watch an episode live in MuJoCo's 3D viewer (drag to rotate, scroll to zoom).

Scripted teacher (fast, no model needed), in the simulation environment:

  PYTHONPATH=src .venv-sim/Scripts/python.exe scripts/watch.py --seed 0 [--fault glitch] [--no-supervisor]

Learned SmolVLA policy on Intel (OpenVINO), in the Physical AI Studio environment:

  PYTHONPATH=src .venv-pai/Scripts/python.exe scripts/watch.py --policy smolvla \
      --export models/openvino/fp32 --device GPU --seed 0

Both run through the same runner used for evaluation. The terminal prints the
instruction, every failure event, the recoveries and the final physical checks.
The learned policy pauses the simulation while it thinks (about 5 s per chunk on
the HD 520 iGPU), so the motion is not real time; --speed only paces the replay.
"""
import argparse
import time

import mujoco.viewer

from pathlib import Path

from rescuehandsai.perturb import GripperGlitch
from rescuehandsai.policies.scripted import ScriptedPolicy
from rescuehandsai.randomize import perturb_start
from rescuehandsai.runner import EpisodeRunner
from rescuehandsai.sim import MujocoSimulation
from rescuehandsai.task import make_task


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fault", choices=["none", "glitch"], default="none")
    parser.add_argument("--no-supervisor", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0, help="1.0 = real time")
    parser.add_argument("--policy", choices=["scripted", "smolvla"], default="scripted")
    parser.add_argument("--export", type=Path, help="OpenVINO export directory for --policy smolvla")
    parser.add_argument("--device", default="GPU", help="OpenVINO device: GPU (iGPU) or CPU")
    parser.add_argument("--n-action-steps", type=int, default=25)
    parser.add_argument("--utensil", choices=["fork", "spoon"], help="force which utensil the words ask for")
    parser.add_argument("--perturb", action="store_true", help="start off-nominal: arms nudged, items moved")
    args = parser.parse_args()
    if args.policy == "smolvla" and not (args.export and (args.export / "manifest.json").is_file()):
        parser.error("--export must point to an exported policy directory with manifest.json")

    sim = MujocoSimulation()
    task = make_task(args.seed, utensil=args.utensil)
    if args.policy == "scripted":
        policy = ScriptedPolicy()
    else:
        from rescuehandsai.policies.smolvla_exported import ExportedSmolVLAPolicy
        print(f"Loading the exported policy on {args.device} (this takes a few minutes on the iGPU)...")
        policy = ExportedSmolVLAPolicy(args.export, sim.names, device=args.device,
                                       n_action_steps=args.n_action_steps)
    print("Policy:", policy.metadata()["name"])
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

    if args.perturb:  # the runner resets the scene itself, so perturb from the frame hook once
        state["perturb"] = True

        def on_first_frame(sim_, status, _hook=on_frame):
            if state.pop("perturb", False):
                perturb_start(sim_, args.seed)
            _hook(sim_, status)

        on_frame = on_first_frame

    runner = EpisodeRunner(sim, policy, supervisor=not args.no_supervisor,
                           fault=GripperGlitch() if args.fault == "glitch" else None,
                           on_frame=on_frame)
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
