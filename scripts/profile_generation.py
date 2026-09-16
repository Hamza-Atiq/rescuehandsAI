"""Time each part of one teacher demonstration step: render, teacher, physics, facts, recorder.

  PYTHONPATH=src python scripts/profile_generation.py --steps 120 [--record]
"""
import argparse
import os
import shutil
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import mujoco

from rescuehandsai.auditor import compute_facts
from rescuehandsai.expert import ScriptedExpert
from rescuehandsai.sim import POLICY_CAMERAS, MujocoSimulation
from rescuehandsai.task import make_task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=5001)
    parser.add_argument("--record", action="store_true", help="also time LeRobot add_frame")
    args = parser.parse_args()

    print("MUJOCO_GL =", os.environ.get("MUJOCO_GL"), "| mujoco", mujoco.__version__, flush=True)
    sim = MujocoSimulation()
    task = make_task(args.seed)
    sim.reset(args.seed, instruction=task.instruction)
    print("GL renderer:", sim.gl_renderer(), flush=True)
    sim.reset(args.seed, instruction=task.instruction)
    expert = ScriptedExpert(sim, task)

    recorder, tmp = None, None
    if args.record:
        from rescuehandsai.recorder import EpisodeRecorder
        tmp = Path(tempfile.mkdtemp())
        recorder = EpisodeRecorder(tmp / "ds", "local/profile", sim.names, sim.config["height"], sim.config["width"])

    cost = defaultdict(float)
    for _ in range(args.steps):
        t = time.perf_counter()
        frames = sim.render(POLICY_CAMERAS)
        cost["render 3 cameras"] += time.perf_counter() - t

        t = time.perf_counter()
        obs = sim.observe(images=False)
        obs = type(obs)(obs.timestamp, obs.instruction, obs.positions, obs.velocities, frames)
        action = expert.act(obs)
        cost["teacher act"] += time.perf_counter() - t

        if recorder:
            t = time.perf_counter()
            recorder.add(obs, action)
            cost["recorder add_frame"] += time.perf_counter() - t

        t = time.perf_counter()
        sim.step(action)
        cost["physics step"] += time.perf_counter() - t

        t = time.perf_counter()
        compute_facts(sim)
        cost["compute_facts"] += time.perf_counter() - t
        if expert.done:
            break

    total = sum(cost.values())
    print(f"\nper step over {args.steps} steps (ms):")
    for name, secs in sorted(cost.items(), key=lambda kv: -kv[1]):
        print(f"  {name:20s} {1000 * secs / args.steps:8.1f}   ({100 * secs / total:4.1f}%)")
    print(f"  {'TOTAL':20s} {1000 * total / args.steps:8.1f}  -> ~{total / args.steps * 566:.0f} s per 566-step episode")
    if recorder:
        recorder.discard()
    sim.close()
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
