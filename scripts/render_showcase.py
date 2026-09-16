"""Presentation stills and video of one episode, from nicer camera angles.

  PYTHONPATH=src .venv-sim/Scripts/python.exe scripts/render_showcase.py --seed 1 --out docs/media
  PYTHONPATH=src .venv-sim/Scripts/python.exe scripts/render_showcase.py --seed 1 --fault glitch --video artifacts/showcase

Runs the scripted teacher (clearly labelled: this is the demonstration teacher, not the
learned policy) with the supervisor, records the physical state every control step, then
draws key moments in the showcase twin: start, right-hand grasp, in-air hand-off,
finished table. The policy's own cameras and the physics are unchanged.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from rescuehandsai.auditor import compute_facts
from rescuehandsai.perturb import GripperGlitch
from rescuehandsai.policies.scripted import ScriptedPolicy
from rescuehandsai.runner import EpisodeRunner
from rescuehandsai.showcase import ShowcaseRenderer
from rescuehandsai.sim import MujocoSimulation
from rescuehandsai.task import make_task


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fault", choices=["none", "glitch"], default="none")
    parser.add_argument("--out", type=Path, default=Path("docs/media"))
    parser.add_argument("--video", type=Path, help="also write MP4s (one per camera) to this folder")
    parser.add_argument("--cameras", default="hero,handoff")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    sim = MujocoSimulation()
    task = make_task(args.seed)
    states, moments = [], {}

    def on_frame(sim_, state):
        states.append(sim_.data.qpos.copy())
        facts = compute_facts(sim_)
        i = len(states) - 1
        held = facts.held_by[task.utensil]
        if "grasp" not in moments and held == {"right_arm"} and not facts.supported[task.utensil]:
            moments["grasp"] = i
        if held == {"left_arm", "right_arm"} and not facts.supported[task.utensil]:
            moments.setdefault("handoff", i + 3)  # a few steps in, both jaws closed
        if state == "RECOVERING":
            moments.setdefault("recovery", i)

    sim.reset(args.seed, instruction=task.instruction)
    start = sim.data.qpos.copy()
    log = EpisodeRunner(sim, ScriptedPolicy(), supervisor=True,
                        fault=GripperGlitch() if args.fault == "glitch" else None, on_frame=on_frame).run(task)
    moments = {"start": None, **moments, "done": len(states) - 1}
    print(json.dumps({"seed": args.seed, "instruction": task.instruction, "state": log.state,
                      "steps": log.steps, "recoveries": log.recoveries, "moments": moments}))

    import imageio.v2 as imageio
    args.out.mkdir(parents=True, exist_ok=True)
    renderer = ShowcaseRenderer(args.width, args.height)
    cameras = args.cameras.split(",")
    for label, index in moments.items():
        qpos = start if index is None else states[min(index, len(states) - 1)]
        for camera in cameras:
            path = args.out / f"seed{args.seed}_{args.fault}_{label}_{camera}.jpg"
            imageio.imwrite(path, renderer.draw(sim, qpos, camera), quality=88)
            print("wrote", path)
    if args.video:
        args.video.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.video / f"seed{args.seed}_{args.fault}_states.npz", qpos=np.array(states))
        for camera in cameras:
            path = args.video / f"seed{args.seed}_{args.fault}_{camera}.mp4"
            with imageio.get_writer(path, fps=20, codec="libx264", quality=8, macro_block_size=1) as writer:
                for qpos in states:  # one frame per 0.05 s control step: real-time playback
                    writer.append_data(renderer.draw(sim, qpos, camera))
            print("wrote", path)
    renderer.close()
    sim.close()


if __name__ == "__main__":
    main()
