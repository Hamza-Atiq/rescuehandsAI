# RescueHands AI — Dinner Table Edition: design

Approved by the project owner on 2026-09-15. AGENTS.md challenge corrections
still apply. Where this file is more specific, it records the decisions made.

## Facts that shape the design

- Submission also needs: public GitHub repo (MIT), video, slides, cover image.
- Development laptop: Intel Core i5-6300U, 4 threads, no discrete GPU.
- No Core Ultra machine is available. An organizer Discord reply (relayed by
  the owner) says development/training may happen elsewhere, but the VLA/VLM
  policy **must execute on Intel XPUs**, and a non-Core-Ultra Intel CPU + iGPU
  was accepted as an alternative configuration. Plan: deploy and benchmark on
  this i5-6300U CPU and its Intel HD Graphics 520 iGPU (if the OpenVINO GPU
  plugin works). Benchmarks record the real device names. No Core Ultra claim.
- Organizer: inverse kinematics is not preferable if it dominates robot control.
  IK is used only by the scripted teacher to generate demonstrations; at
  deployment SmolVLA outputs joint targets directly. The README must say so.
- Organizer: combining a simpler VLA with a more complex VLM is welcome.
  Optional layer after the core pipeline works: an OpenVINO-run VLM reads the
  overhead image and instruction and selects the next subtask text for SmolVLA.
- Training runs on a free Kaggle GPU.
- Model decision (owner): **SmolVLA only.** No hidden switch to another model.

## Scene

- Both SO-101 arms mounted side by side on one table edge, facing the place
  setting, so `left_arm` and `right_arm` match a person's hands.
- Items: plate (static target reference), cup, tray holding a fork and a spoon.
- Target zones: cup right of plate; utensil left of plate.
- All positions live in configuration and are checked for reachability.
- Grasp stability: small box finger pads if mesh contacts prove unstable;
  gripper asymmetry handled with a measured grasp offset.

## Task

Instruction example: "Set the table: put the cup by the plate and pass the fork
to the left hand." Paraphrases are generated from templates.

1. Right arm places the cup right of the plate.
2. Right arm picks the named utensil (fork or spoon) and lifts it to a hand-off pose.
3. Left arm grasps, right arm releases, left arm places it left of the plate.

Language matters: the tray holds both utensils, so the instruction and camera
decide which one to move.

## Components (new or changed)

| Module | Responsibility |
| --- | --- |
| `scene.py` | Build the MJCF world from config: table, items, zones, lights, cameras, randomization |
| `sim.py` | Existing adapter, extended for multiple items, wrist cameras, named sites |
| `kinematics.py` | Damped least-squares IK for the `gripperframe` site, top-down orientation |
| `task.py` | `TaskSpec`, instruction templates, success conditions |
| `expert.py` | Scripted phase state machine (approach, descend, close, lift, move, place, open, hand-off) |
| `auditor.py` | Physics facts: held (both jaws touch), dropped, placed-and-stable, out of bounds, cross-arm collision, stall |
| `recovery.py` | Bounded recovery decisions from `FailureEvent`s |
| `recorder.py` | Episode recording to LeRobot dataset format (ML env) |
| `policies/` | `ScriptedPolicy`, `SmolVLAPolicy` (PyTorch), `OpenVINOSmolVLAPolicy` behind one interface |
| `evaluate.py` | Seeded evaluation, perturbations, metrics, videos |
| `benchmark.py` | PyTorch vs OpenVINO FP32 vs INT8 latency, CPU name, task success |

## Supervisor and recovery

- Pre-action: existing `validate_action` (names, finite values, limits, step size, age).
- Post-action auditor facts produce structured failures: `FAILED_GRASP`,
  `OBJECT_DROPPED`, `COLLISION`, `OBJECT_OUT_OF_BOUNDS`, `TARGET_MISSED`,
  `TIMEOUT`, `INVALID_ACTION`, `POLICY_ERROR`, `SIMULATION_ERROR`, `RECOVERY_EXHAUSTED`.
- Recovery budget: 2 attempts per episode. Failed grasp: open, retreat, retry.
  Drop: return to the reacquire subgoal. Collision: stop, safe pose, abort.
- Expert demonstrations include injected perturbations with recoveries, so the
  learned policy sees re-grasp behavior.
- Headline comparison on the same 10 seeds with the same mid-task utensil
  knock: SmolVLA alone vs SmolVLA + supervisor. Scripted expert is the baseline row.

## Data and training

- Observation: overhead camera, left and right wrist cameras; 12 joint
  positions (left six then right six, by name); instruction text.
- Action: 12 absolute joint targets in radians, same order.
- About 120 episodes, half fork, half spoon, LeRobot format, uploaded to the
  owner's Hugging Face account.
- Fine-tune `lerobot/smolvla_base` on Kaggle with frequent checkpoints.
- Privileged simulator state is used only by the expert, auditor and evaluation.

## Intel / OpenVINO

- Early spike: convert pretrained SmolVLA components (vision encoder, language
  model prefix, action expert step) with OpenVINO; verify numerical agreement.
- Benchmark on measured hardware: PyTorch CPU, OpenVINO FP32, OpenVINO INT8
  weights; mean and p95 latency, memory, and task success after conversion.

## Evaluation

10 seeds varying item placement, mass, friction, item size, lighting and table
colour. Report success rate, failure labels, recoveries, completion time and
inference latency. Save per-seed video and a combined video. Every number comes
from stored run artifacts.

## Honest failure handling

If SmolVLA does not learn the task well in time, report the measured result,
show the supervisor catching failures and the scripted baseline alongside it.
Never present a failed or partial run as success.

## Testing

Unit tests for IK convergence, auditor facts, recovery limits, instruction
parsing, action ordering. Simulation tests for scene load, reachability of every
zone, a real expert cup placement, and a real hand-off. Policy tests for input
schema and action shape. OpenVINO test for output agreement with PyTorch.

## Schedule

| Hours | Work |
| --- | --- |
| 0–2 | Spec, ML environment, OpenVINO spike started |
| 2–10 | Scene, IK, expert cup place and hand-off, auditor |
| 10–13 | Randomization, recorder, dataset, upload |
| 13–22 | Kaggle training; supervisor, evaluation, OpenVINO pipeline in parallel |
| 22–30 | SmolVLA in simulation, OpenVINO swap, 10-seed evaluation |
| 30–35 | Video, README, slides, cover, MIT license |
| 35–37 | Buffer and submission |
