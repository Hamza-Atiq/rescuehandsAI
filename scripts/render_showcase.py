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


def label_frame(frame, lines):
    """Burn a small caption into a frame so a viewer knows who is driving the robot."""
    from PIL import Image, ImageDraw, ImageFont
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    try:
        font = ImageFont.truetype("segoeui.ttf", max(14, frame.shape[0] // 34))
    except OSError:
        font = ImageFont.load_default()
    height = (font.size + 8) * len(lines) + 12
    draw.rectangle([0, 0, frame.shape[1], height], fill=(10, 14, 22, 185))
    for i, (text, colour) in enumerate(lines):
        draw.text((16, 8 + i * (font.size + 8)), text, font=font, fill=colour)
    return np.asarray(image)


def replay(args):
    """Draw a saved evaluation (evaluate.py --save-states) from a presentation camera."""
    import imageio.v2 as imageio
    data = np.load(args.states)
    seed, policy = int(data["seed"]), str(data["policy"])
    sim = MujocoSimulation()
    sim.reset(seed, instruction=str(data["instruction"]))
    renderer = ShowcaseRenderer(args.width, args.height)
    who = "learned policy (SmolVLA)" if "smolvla" in policy else "scripted teacher"
    args.video.mkdir(parents=True, exist_ok=True)
    for camera in args.cameras.split(","):
        path = args.video / f"{args.states.stem.replace('_states', '')}_{camera}.mp4"
        with imageio.get_writer(path, fps=20 // args.every, codec="libx264", quality=8, macro_block_size=1) as writer:
            for i in range(0, len(data["qpos"]), args.every):
                state = str(data["runner_state"][i])
                colour = (255, 170, 60, 255) if state == "RECOVERING" else (120, 220, 160, 255)
                frame = renderer.draw(sim, data["qpos"][i], camera)
                writer.append_data(label_frame(frame, [
                    (f"{who} · seed {seed} · t = {i * 0.05:4.1f} s sim · {state}", colour),
                    (f"“{data['instruction']}”", (225, 232, 240, 255))]))
        print("wrote", path, flush=True)
    renderer.close()
    sim.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fault", choices=["none", "glitch"], default="none")
    parser.add_argument("--out", type=Path, default=Path("docs/media"))
    parser.add_argument("--video", type=Path, help="also write MP4s (one per camera) to this folder")
    parser.add_argument("--cameras", default="hero,handoff")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--states", type=Path, help="replay a saved evaluation episode instead of running the teacher")
    parser.add_argument("--every", type=int, default=2, help="with --states: draw every Nth control step")
    args = parser.parse_args()
    if args.states:
        if not args.video:
            parser.error("--states needs --video <folder>")
        return replay(args)

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
