"""Run a recorded motor/camera check. This is not a learned policy or task demo."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess

import imageio.v2 as imageio
import mujoco

from .contracts import BimanualAction, EpisodeResult
from .sim import MujocoSimulation, ROOT, VIDEO_CAMERAS

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps < 50:
        parser.error("--steps must be at least 50 to measure motor response")
    output = args.output or ROOT / "artifacts" / datetime.now(timezone.utc).strftime("control-%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True, exist_ok=False)
    sim = None
    report = {"kind": "simulation_control_check", "seed": args.seed,
              "learned_policy": False, "manipulation_success": None}
    frames = []
    try:
        sim = MujocoSimulation()
        sim.reset(args.seed)
        initial = sim.observe(images=True)
        for name, image in (initial.images | sim.render(VIDEO_CAMERAS)).items():
            imageio.imwrite(output / f"{name}-initial.png", image)
        ranges = {n: [v, v] for n, v in initial.positions.items()}
        with (output / "steps.jsonl").open("w", encoding="utf-8") as log:
            for step in range(args.steps):
                phase = math.sin(2 * math.pi * step / args.steps)
                targets = dict(sim.home_targets)
                for arm, direction in (("left_arm", 1), ("right_arm", -1)):
                    targets[arm + "/shoulder_pan"] += direction * 0.12 * phase
                    targets[arm + "/gripper"] += direction * 0.15 * phase
                before = sim.observe()
                action = BimanualAction(before.timestamp, targets)
                sim.step(action)
                obs = sim.observe()
                for name, value in obs.positions.items():
                    ranges[name][0] = min(ranges[name][0], value)
                    ranges[name][1] = max(ranges[name][1], value)
                if step % 5 == 0:
                    frames.append(sim.render(VIDEO_CAMERAS)["front"])
                log.write(json.dumps({"step": step, "action": asdict(action),
                                      "positions": obs.positions,
                                      "state": asdict(sim.privileged())}, allow_nan=False) + "\n")
        travel = {name: hi - lo for name, (lo, hi) in ranges.items()}
        tested = [f"{arm}/{joint}" for arm in ("left_arm", "right_arm")
                  for joint in ("shoulder_pan", "gripper")]
        passed = all(travel[n] > 0.06 for n in tested)
        report.update(asdict(EpisodeResult(args.seed, "scripted_motor_check", passed)))
        report["joint_travel_radians"] = travel
        report["simulated_seconds"] = sim.observe().timestamp
        report["contact_samples"] = sim.contact_samples
        report["configuration"] = sim.config
        report["mujoco_version"] = mujoco.__version__
        report["python_version"] = platform.python_version()
        report["platform"] = platform.platform()
        report["source_sha256"] = {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "src/rescuehandsai").glob("*.py"))}
        report["robot_xml_sha256"] = hashlib.sha256(sim.asset_path.read_bytes()).hexdigest()
        revision = subprocess.run(["git", "-C", str(sim.asset_path.parents[1]), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, timeout=10)
        report["robot_revision_actual"] = revision.stdout.strip() if revision.returncode == 0 else None
        for name, image in (sim.observe(images=True).images | sim.render(VIDEO_CAMERAS)).items():
            imageio.imwrite(output / f"{name}-final.png", image)
        if frames:
            imageio.mimsave(output / "control-preview.gif", frames,
                           duration=sim.config["control_dt"] * 5 * 1000, loop=0)
        report["limitations"] = [
            "Motor and camera check only; no grasp, hand-off or table-setting policy.",
            "Scene randomization is applied, but no task is attempted.",
            "PlacementTracker has unit tests but is not a complete physics auditor.",
            "Cross-arm contacts stop stepping; full self/table collision handling remains.",
            "No VLA training, recovery or OpenVINO benchmark yet.",
            "No Intel hardware compliance or acceleration claim."
        ]
    except Exception as error:
        report["control_check_passed"] = False
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        if sim is not None:
            sim.close()
        (output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                          encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), **report}, indent=2))
    return 0 if report.get("control_check_passed") else 1
