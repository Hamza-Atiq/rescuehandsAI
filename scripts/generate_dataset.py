"""Record scripted-teacher demonstrations in LeRobot format.

Run with the Physical AI Studio environment (LeRobot 0.5.1), from the project root:
  PYTHONPATH=src .venv-pai/Scripts/python.exe scripts/generate_dataset.py \
      --root data/shard0 --repo-id <hf_user>/rescuehands_table --seeds 1000:1060

Seeds 0-9 are reserved for evaluation and are refused. Only physically
successful episodes are saved; every attempt is logged to attempts.jsonl.
"""
import argparse
import json
import time
from pathlib import Path

from rescuehandsai.auditor import compute_facts
from rescuehandsai.evaluation import HandoffTracker, task_outcome
from rescuehandsai.randomize import perturb_start, start_problems


class SkipEpisode(Exception):
    """The starting state itself is not a valid table, so the attempt is not a demonstration."""
from rescuehandsai.expert import ScriptedExpert
from rescuehandsai.recorder import EpisodeRecorder
from rescuehandsai.sim import MujocoSimulation
from rescuehandsai.task import make_task

EVAL_SEEDS = range(0, 10)
MAX_STEPS = 900


def seed_range(text: str):
    start, stop = (int(v) for v in text.split(":"))
    return range(start, stop)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--seeds", type=seed_range, required=True, help="start:stop")
    parser.add_argument("--vcodec", default="libsvtav1")
    parser.add_argument("--perturb", action="store_true",
                        help="start from an off-nominal state (arms nudged, items shifted and spun)")
    parser.add_argument("--perturb-scale", type=float, default=0.6,
                        help="largest perturbation, 0-1; measured teacher success: 1.0 -> 35%%, 0.6 -> see README")
    args = parser.parse_args()
    if set(args.seeds) & set(EVAL_SEEDS):
        parser.error("seeds 0-9 are reserved for evaluation")
    if args.root.exists():
        parser.error(f"{args.root} already exists; choose a new shard directory")

    sim = MujocoSimulation()
    recorder = EpisodeRecorder(args.root, args.repo_id, sim.names, sim.config["height"],
                               sim.config["width"], vcodec=args.vcodec)
    log_path = args.root.parent / f"{args.root.name}_attempts.jsonl"
    saved = 0
    with log_path.open("w", encoding="utf-8") as log:
        for seed in args.seeds:
            task = make_task(seed)
            sim.reset(seed, instruction=task.instruction)
            perturbation = perturb_start(sim, seed, max_scale=args.perturb_scale) if args.perturb else None
            start = compute_facts(sim)
            invalid = start_problems(start, sim.scene_params)
            expert = ScriptedExpert(sim, task)
            started = time.time()
            handoff, steps, error = HandoffTracker(task.utensil), 0, invalid
            try:
                if invalid:
                    raise SkipEpisode(invalid)
                while not expert.done and steps < MAX_STEPS:
                    obs = sim.observe(images=True)
                    action = expert.act(obs)
                    recorder.add(obs, action)
                    sim.step(action)
                    handoff.update(compute_facts(sim))
                    steps += 1
            except Exception as exc:  # planning or safety stop: not a demonstration
                error = f"{type(exc).__name__}: {exc}"
            outcome = (task_outcome(compute_facts(sim), task, handoff.done, sim.scene_params, start.positions)
                       if error is None else {"success": False})
            keep = error is None and expert.done and outcome["success"]
            if keep:
                recorder.save()
                saved += 1
            else:
                recorder.discard()
            record = {"seed": seed, "utensil": task.utensil, "instruction": task.instruction, "steps": steps,
                      "saved": keep, "error": error, "outcome": outcome, "perturbed": bool(perturbation),
                      "wall_s": round(time.time() - started, 1)}
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
    recorder.finalize()
    sim.close()
    print(json.dumps({"root": str(args.root), "saved": saved, "attempted": len(args.seeds)}))


if __name__ == "__main__":
    main()
