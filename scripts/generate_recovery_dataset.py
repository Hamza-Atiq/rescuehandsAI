"""Record demonstrations that contain a real failure and the teacher's recovery.

  PYTHONPATH=src python scripts/generate_recovery_dataset.py \
      --root data/recovery0 --repo-id <hf_user>/rescuehands_table_recovery --seeds 3000:3040

Why: a policy trained only on flawless teacher runs never sees an off-track state,
so small errors compound until the task is lost. These episodes inject the gripper
fault, let the supervisor stop and re-approach, and record **every executed command,
including the recovery motion**, so the learned policy can imitate getting back on
track. Only episodes that both recovered and finished the task are kept.

Camera, joint order, action units and rate are identical to the clean dataset, so
the two can be aggregated for fine-tuning.
"""
import argparse
import json
import time
from pathlib import Path

from rescuehandsai.perturb import GripperGlitch
from rescuehandsai.policies.scripted import ScriptedPolicy
from rescuehandsai.recorder import EpisodeRecorder, fps_for
from rescuehandsai.runner import EpisodeRunner
from rescuehandsai.sim import MujocoSimulation, is_software_renderer
from rescuehandsai.task import make_task


def seed_range(text: str):
    start, stop = (int(part) for part in text.split(":"))
    return range(start, stop)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--seeds", type=seed_range, required=True)
    parser.add_argument("--vcodec", default="libsvtav1")
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--allow-cpu-render", action="store_true", help="run even if images are drawn on the CPU")
    args = parser.parse_args()
    if args.root.exists():
        parser.error(f"{args.root} already exists; choose a new shard directory")
    if min(args.seeds) < 10:
        parser.error("seeds 0-9 are reserved for evaluation")

    sim = MujocoSimulation()
    renderer = sim.gl_renderer()
    print(json.dumps({"gl_renderer": renderer}), flush=True)
    if is_software_renderer(renderer) and not args.allow_cpu_render:
        parser.error(f"camera images would be drawn on the CPU ({renderer}), ~4 min per episode; "
                     "on Kaggle run training/kaggle_gpu_render.sh first, or pass --allow-cpu-render")
    recorder = EpisodeRecorder(args.root, args.repo_id, sim.names, sim.config["height"],
                               sim.config["width"], fps=fps_for(sim.config["control_dt"]), vcodec=args.vcodec)
    log_path = args.root.parent / f"{args.root.name}_attempts.jsonl"
    saved = 0
    with log_path.open("w", encoding="utf-8") as log_file:
        for seed in args.seeds:
            task = make_task(seed)
            started = time.time()

            def record(sim_, action):
                recorder.add(sim_.observe(images=True), action)

            runner = EpisodeRunner(sim, ScriptedPolicy(), supervisor=True, fault=GripperGlitch(),
                                   max_steps=args.max_steps, on_command=record)
            try:
                log = runner.run(task)
                error = None
            except Exception as exc:  # planning or safety stop: not a demonstration
                log, error = None, f"{type(exc).__name__}: {exc}"
            keep = bool(log and log.state == "SUCCEEDED" and log.recoveries > 0)
            if keep:
                recorder.save()
                saved += 1
            else:
                recorder.discard()
            record_row = {"seed": seed, "utensil": task.utensil, "saved": keep, "error": error,
                          "state": log.state if log else None, "recoveries": log.recoveries if log else None,
                          "fault_step": log.fault_step if log else None,
                          "steps": log.steps if log else None, "wall_s": round(time.time() - started, 1)}
            log_file.write(json.dumps(record_row) + "\n")
            log_file.flush()
            print(json.dumps(record_row), flush=True)
    recorder.finalize()
    sim.close()
    print(json.dumps({"root": str(args.root), "saved": saved, "attempted": len(args.seeds)}))


if __name__ == "__main__":
    main()
