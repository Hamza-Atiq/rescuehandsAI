"""Seeded evaluation: every number in the README comes from these result files.

Examples (project root):
  # scripted baseline, with and without the supervisor, with the gripper fault
  PYTHONPATH=src .venv-sim/Scripts/python.exe scripts/evaluate.py --policy scripted --seeds 0:10 --fault glitch --supervisor on
  # exported SmolVLA on the Intel iGPU (Physical AI Studio environment)
  PYTHONPATH=src .venv-pai/Scripts/python.exe scripts/evaluate.py --policy smolvla --export models/openvino/fp32 --device GPU --seeds 0:10 --supervisor on --video

Writes results/<run>/summary.json, one episode_<seed>.json per seed, and
optional front-camera MP4s. Seeds 0-9 are the evaluation seeds (never trained on).
"""
import argparse
from dataclasses import asdict
import hashlib
from datetime import datetime, timezone
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

from rescuehandsai.perturb import GripperGlitch
from rescuehandsai.randomize import perturb_start
from rescuehandsai.runner import EpisodeRunner
from rescuehandsai.scene import ROOT
from rescuehandsai.sim import VIDEO_CAMERAS, MujocoSimulation
from rescuehandsai.task import make_task


def seed_range(text):
    a, b = (int(v) for v in text.split(":"))
    return range(a, b)


def git_revision():
    out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip() or None


def source_state():
    """Revision plus uncommitted changes, taken before the first episode.

    A HEAD string alone cannot tell two runs apart when files were edited but not committed."""
    def git(*cmd):
        return subprocess.run(["git", "-C", str(ROOT), *cmd], capture_output=True, text=True).stdout
    changed = [line[3:] for line in git("status", "--porcelain", "--untracked-files=no").splitlines()]
    diff = git("diff", "HEAD")
    return {"git_revision": git_revision(), "dirty": bool(changed), "changed_files": changed,
            "diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if changed else None}


def package_versions():
    from importlib.metadata import PackageNotFoundError, version
    found = {}
    for name in ("mujoco", "numpy", "openvino", "physicalai-train", "lerobot", "torch"):
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            pass
    return found


def make_policy(args, sim):
    if args.policy == "scripted":
        from rescuehandsai.policies.scripted import ScriptedPolicy
        return ScriptedPolicy()
    from rescuehandsai.policies.smolvla_exported import ExportedSmolVLAPolicy
    return ExportedSmolVLAPolicy(args.export, sim.names, device=args.device, n_action_steps=args.n_action_steps)


class VideoWriter:
    """Front-camera video with a small status bar; written only when --video is set."""

    def __init__(self, path: Path, every: int, fps: int, size=(480, 360)):
        self.every, self.k, self.size = every, 0, size
        try:  # imageio-ffmpeg (H.264) where installed, e.g. the Physical AI Studio env
            import imageio.v2 as imageio
            self.writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=1)
            self._cv = None
        except (ImportError, ValueError):
            import cv2  # the simulation env ships OpenCV
            self._cv = cv2
            self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def __call__(self, sim, state):
        self.k += 1
        if self.k % self.every:
            return
        frame = sim.render(VIDEO_CAMERAS, *self.size)["front"].copy()
        colour = {"EXECUTING": (40, 160, 60), "RECOVERING": (220, 140, 20)}.get(state, (200, 40, 40))
        frame[:10, :] = colour
        if self._cv is None:
            self.writer.append_data(frame)
        else:
            self.writer.write(self._cv.cvtColor(frame, self._cv.COLOR_RGB2BGR))

    def close(self):
        if self._cv is None:
            self.writer.close()
        else:
            self.writer.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", choices=["scripted", "smolvla"], required=True)
    parser.add_argument("--export", type=Path, help="exported SmolVLA directory (smolvla only)")
    parser.add_argument("--device", default="GPU", help="OpenVINO device for smolvla: CPU or GPU")
    parser.add_argument("--n-action-steps", type=int, default=25)
    parser.add_argument("--seeds", type=seed_range, default=range(0, 10))
    parser.add_argument("--supervisor", choices=["on", "off"], required=True)
    parser.add_argument("--fault", choices=["none", "glitch"], default="none")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="override the task timeout (default: task.timeout_s / control_dt)")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--perturb", action="store_true",
                        help="start off-nominal: arms nudged, items shifted (robustness check)")
    parser.add_argument("--perturb-scale", type=float, default=1.0, help="largest perturbation, 0-1")
    parser.add_argument("--name", help="run folder name under results/")
    args = parser.parse_args()
    if args.policy == "smolvla" and not (args.export and (args.export / "manifest.json").is_file()):
        parser.error("--export must point to an exported policy directory with manifest.json")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    name = args.name or f"{args.policy}_{args.device.lower() if args.policy != 'scripted' else 'cpu'}_sup-{args.supervisor}_fault-{args.fault}_{stamp}"
    out = ROOT / "results" / name
    out.mkdir(parents=True, exist_ok=False)
    sim = MujocoSimulation()
    policy = make_policy(args, sim)
    source = source_state()
    manifest = {"run": name, "created_utc": stamp, "source": source,
                "args": {k: (str(v) if isinstance(v, (Path, range)) else v) for k, v in vars(args).items()},
                "seeds": list(args.seeds), "policy": policy.metadata(), "sim_config": sim.config,
                "packages": package_versions(), "python": platform.python_version(),
                "processor": platform.processor(), "platform": platform.platform()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    episodes = []
    for seed in args.seeds:
        task = make_task(seed)
        fault = GripperGlitch() if args.fault == "glitch" else None
        video = VideoWriter(out / f"episode_{seed}.mp4", every=2, fps=10) if args.video else None
        runner = EpisodeRunner(sim, policy, supervisor=args.supervisor == "on", fault=fault,
                               max_steps=args.max_steps, on_frame=video,
                               on_reset=(lambda sim_, task_: perturb_start(sim_, task_.seed, max_scale=args.perturb_scale))
                               if args.perturb else None)
        try:
            log = runner.run(task)
        finally:
            if video:
                video.close()
        record = asdict(log)
        (out / f"episode_{seed}.json").write_text(json.dumps(record, indent=2, default=str))
        episodes.append(record)
        print(json.dumps({"seed": seed, "state": log.state, "failure": log.failure,
                          "recoveries": log.recoveries, "steps": log.steps, "wall_s": log.wall_seconds}), flush=True)
        # rewritten after every episode, so an interrupted run still has a truthful partial summary
        (out / "summary.json").write_text(json.dumps(
            summarize(episodes, args, name, stamp, source, complete=False), indent=2))
    summary = summarize(episodes, args, name, stamp, source, complete=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    sim.close()


def summarize(episodes, args, name, stamp, source, *, complete: bool) -> dict:
    n = len(episodes)
    successes = [e for e in episodes if e["state"] == "SUCCEEDED"]
    latencies = [t for e in episodes for t in e["inference_seconds"]]
    failures = {}
    for e in episodes:
        if e["failure"]:
            failures[e["failure"]] = failures.get(e["failure"], 0) + 1
    faulted = [e for e in episodes if e["fault_step"] is not None]
    summary = {
        "run": name, "created_utc": stamp, "complete": complete, "planned_seeds": len(args.seeds),
        "git_revision": source["git_revision"], "source_dirty": source["dirty"],
        "policy": episodes[0]["policy"] if episodes else None, "supervisor": args.supervisor == "on",
        "fault": args.fault, "perturbed_start": args.perturb,
        "perturb_scale": args.perturb_scale if args.perturb else None,
        "seeds": list(args.seeds), "episodes": n, "max_steps_override": args.max_steps,
        "success_rate": len(successes) / n if n else None, "successes": len(successes),
        "failures": failures,
        "collision_episodes": sum(any(ev["label"] == "COLLISION" for ev in e["events"]) for e in episodes),
        "faults_injected": len(faulted),
        "recovered_after_fault": sum(e["state"] == "SUCCEEDED" and e["recoveries"] > 0 for e in faulted),
        "mean_recoveries": float(np.mean([e["recoveries"] for e in episodes])) if n else None,
        "mean_sim_seconds_success": float(np.mean([e["sim_seconds"] for e in successes])) if successes else None,
        "inference_calls": len(latencies),
        "inference_mean_s": float(np.mean(latencies)) if latencies else None,
        "inference_p95_s": float(np.percentile(latencies, 95)) if latencies else None,
        "platform": {"processor": platform.processor(), "system": platform.platform(),
                     "python": platform.python_version()},
        "note": "Simulation time pauses during policy inference; latency is wall-clock and reported separately.",
    }
    return summary


if __name__ == "__main__":
    main()
