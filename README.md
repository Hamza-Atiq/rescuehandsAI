# RescueHands AI — a dinner table that survives a dropped fork

Two simulated SO-101 arms in MuJoCo set a dinner place from a spoken-style
instruction: the right arm picks the **named** utensil (fork or spoon), hands
it to the left arm, the left arm places it beside the plate, and the right arm
places the cup. A deterministic, physics-aware **supervisor** watches every
control step. When an item really drops, it stops, moves both arms to safety
and lets the policy try again, instead of blindly continuing.

| In-air hand-off | Both grippers on the fork |
| --- | --- |
| ![Right arm hands the fork to the left arm above the table](docs/media/teacher_handoff_wide.jpg) | ![Close view of both grippers holding the fork during the hand-off](docs/media/teacher_handoff_close.jpg) |
| **Gripper fault: the fork drops, the arms back off** | **Table set: fork left of the plate, cup right** |
| ![The fork has fallen onto the mat and the right arm retreats before retrying](docs/media/teacher_drop_recovery.jpg) | ![Finished place setting with the fork and cup in their zones](docs/media/teacher_table_set.jpg) |

<sub>Scripted demonstration teacher, seed 1, drawn with `scripts/render_showcase.py`
(presentation cameras only; the policy's own cameras are unchanged). Learned-policy
frames will be added from the final evaluation.</sub>

> **Thesis:** a VLA should not be trusted just because it produced an action.
> A useful bimanual system checks what physically happened and recovers safely.

Built for the Intel **Bimanual VLA Manipulation** online track (AI Infra Summit
Hackathon 2026). Status: **work in progress** — see [Results](#results) for what
is measured and what is still pending.

## What makes it different

| | |
| --- | --- |
| Real two-arm teamwork | An in-air hand-off; both arms must hold the utensil for success. |
| Language matters | Fork *and* spoon lie on the mat in random order; only the instruction says which one. |
| Honest physics | Contact grasps only. No welding, no teleporting. Success is measured from simulator state, never from the policy's claim. |
| Failure awareness | Auditor labels `OBJECT_DROPPED`, `FAILED_GRASP`, `COLLISION` (between the two arms), `TIMEOUT`… plus a progress watchdog for grasps that never start. |
| Bounded recovery | Open, retreat to a safe pose, retry — at most twice. |
| Intel deployment | SmolVLA exported with Intel Physical AI Studio to OpenVINO and run on an Intel CPU and iGPU. |

## Architecture

```mermaid
flowchart LR
  I["Instruction"] --> P
  C["3 cameras<br/>overhead, left wrist, right wrist"] --> P
  S["12 joint positions"] --> P
  P["SmolVLA policy<br/>(OpenVINO on Intel CPU/iGPU)"] --> G["Action guard<br/>limits + step size"]
  G --> M["MuJoCo physics<br/>2x SO-101 + dinner table"]
  M --> A["Physics auditor<br/>held? dropped? placed? collided?"]
  A -->|ok| P
  A -->|drop / stalled grasp| R["Supervisor recovery<br/>open, safe pose, retry (max 2)"]
  R --> P
  A --> E["Evaluation<br/>stable, supported, in zone, hand-off"]
```

Policy-visible data: camera images, joint positions, instruction. Simulator
ground truth (object poses, contacts) is used only by the auditor, the
evaluation and the scripted teacher — never as policy input.

### Where inverse kinematics is used

Only inside the **scripted teacher** (`expert.py`), which generates the training
demonstrations and serves as the baseline row. At deployment, SmolVLA outputs
the 12 joint targets directly; there is no IK in the learned control loop.

## Results

All numbers come from files under `results/` produced by `scripts/evaluate.py`
and `scripts/benchmark_intel.py`. Each run folder has a `manifest.json` (code
revision, uncommitted-change flag and diff hash, arguments, package versions)
written before the first episode, and a `summary.json` updated after every episode
(runs made before September 16 afternoon lack the manifest; their folders say so).
The scripted rows below are committed in `results/audit_final_*`. Evaluation seeds 0–9 are never
used for training data (training seeds start at 1000), but they did guide debugging
of the scripted teacher, so they are not a pristine unseen set.

### Task success on 10 randomized seeds

Each seed randomizes item positions, cup size, mass, friction, utensil size and
order, lighting and table colour. **Limit, measured:** the gripper pads have a
higher MuJoCo contact priority with a fixed friction of 1.0, so the randomized item
friction changes table and item-to-item contacts but **not** the jaw grasp itself
(seed 0: sampled cup friction 0.717, every jaw contact used 1.0). These results
therefore do not show robustness to slippery grasps. "Gripper glitch" forces the holding hand open
for 0.5 s of simulated time after the utensil is lifted (a real physical drop); the
fault clock runs the same way with and without recovery.

Success is physical and checked over 10 consecutive control steps: cup and
utensil inside their zones, released, resting on the table or plate, cup axis
within 15° of vertical, low linear and angular speed, the spare utensil released, still and near its start **at the end** (earlier contact is not tracked), and
an **ordered in-air hand-off** (right hand alone → both hands while airborne →
left hand alone). Dropping the utensil and picking it up with the other hand does
not count.

| Policy | Supervisor | Fault | Success | Notes |
| --- | --- | --- | ---: | --- |
| Scripted teacher (baseline) | on | none | 8/10 | seed 6 planning error, seed 9 recovery budget exhausted |
| Scripted teacher (baseline) | off | gripper glitch | 2/10 | the teacher's own re-grasp saves 2 drops (seeds 0, 4); 8 end in `FAILED_GRASP` |
| Scripted teacher (baseline) | on | gripper glitch | **7/10** | 2 recovery budgets exhausted (seeds 6, 8), 1 planning error (seed 9) |
| SmolVLA v1 (OpenVINO, iGPU) | off | none | 1 of 9 completed | seeds 0–8 of an **interrupted** run (seed 9 has no result): seed 4 succeeded; 5 × `OBJECT_OUT_OF_BOUNDS`, 3 × `TIMEOUT`. Not a 10-seed result. Made before run manifests existed: see `results/smolvla_ov-gpu_sup-off_fault-none/PROVENANCE.md`. |
| SmolVLA (OpenVINO, iGPU) | on | none / gripper glitch | pending | |

"Collision" in these results means contact **between the two arms**; arm–table
and arm–plate contacts are not yet detected or counted.

History: the no-supervisor fault row was 0/10 until the teacher learned to
re-grasp a dropped utensil (commit `7767ee5`); a bisect shows the same 2/10 at that
commit, before the September 16 audit fixes. An independent review
([report](docs/research/2026-09-15-independent-code-review.md)) and a later quality
audit ([report](docs/research/2026-09-16-quality-audit.md)) tightened the success
checks: a hand-off may not have more than 5 control steps with no hand on the
utensil (measured teacher maximum: 1), and at the end the spare utensil must have no
gripper contact and be still. Every scripted episode on seeds 0–9 kept the same result and step count
after those changes.

### Intel optimization

Pretrained SmolVLA spike (FP32, random inputs, before fine-tuning):

| Device | Mean per 50-step chunk |
| --- | ---: |
| Intel HD Graphics 520 iGPU (OpenVINO) | 4.68 s |
| Intel Core i5-6300U CPU (OpenVINO) | 30.8 s (measured under CPU contention; re-measurement pending) |

The final benchmark (`scripts/benchmark_intel.py`) compares PyTorch CPU,
OpenVINO FP32 on CPU, FP32/FP16 execution on the iGPU, NNCF INT8 weights and
model caching, each with its action difference from the FP32 reference.
**Pending** on the fine-tuned model. Simulation time pauses during inference;
latency is reported separately as wall-clock time.

### Hardware statement

Organizers allowed other Intel platforms when optimizations are documented. The
demo and benchmarks run on an **Intel Core i5-6300U with Intel HD Graphics 520**
(not Core Ultra). Training and bulk data generation used a cloud GPU, which the
challenge allows. No Core Ultra numbers are claimed.

## Reproduce

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). Three environments
are kept separate because their dependencies conflict:

| Env | Purpose | Key packages |
| --- | --- | --- |
| `.venv-sim` | simulation, teacher, tests, scripted evaluation | MuJoCo 3.13.0 |
| `.venv-pai` | data recording, export, SmolVLA inference, benchmark | Physical AI Studio 0.1.0, LeRobot 0.5.1, OpenVINO 2026.1, NNCF |
| Kaggle `.venv-train` | GPU data generation and SmolVLA fine-tuning | LeRobot 0.5.1 |

```bash
# 1. core environment (Windows: use your standalone Python 3.12 path)
UV_PROJECT_ENVIRONMENT=.venv-sim uv sync --frozen --python 3.12

# 2. SO-101 model (Apache-2.0, pinned revision)
git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git .cache/menagerie
git -C .cache/menagerie sparse-checkout set robotstudio_so101
git -C .cache/menagerie checkout 8161bba264d7fa7c99ca301e91e7fb44737676ad

# 3. tests (99 unit and physics tests)
.venv-sim/Scripts/python.exe -m unittest discover -s tests -v

# 4. watch the teacher live, with a dropped utensil and recovery
PYTHONPATH=src .venv-sim/Scripts/python.exe scripts/watch.py --seed 0 --fault glitch

# 5. scripted baseline on the 10 evaluation seeds
PYTHONPATH=src .venv-sim/Scripts/python.exe scripts/evaluate.py --policy scripted --seeds 0:10 --supervisor on --fault glitch

# 6. Physical AI Studio environment
uv venv .venv-pai --python 3.12
uv pip install --python .venv-pai/Scripts/python.exe --extra-index-url https://download.pytorch.org/whl/cpu \
  "physicalai-train[smolvla,cpu]==0.1.0" "lerobot[dataset]==0.5.1" "transformers==5.3.0" nncf mujoco==3.13.0 imageio psutil

# 7. data + training on a Kaggle GPU notebook (Internet on, HF_TOKEN secret)
python training/kaggle_pipeline.py --hf-user <you> --stage setup
bash training/kaggle_gpu_render.sh          # draw camera images on the GPU, not the CPU
python training/kaggle_pipeline.py --hf-user <you> --stage data --episodes 200 --perturbed-episodes 200 --recovery-episodes 100
python training/kaggle_pipeline.py --hf-user <you> --stage train --steps 12000   # verifies and uploads task_contract.json

# 8. download (contract hash check), export to OpenVINO, evaluate on the iGPU, benchmark
bash scripts/deploy_learned.sh <you>/smolvla_rescuehands_v2 v2 download export eval bench
# or step by step:
PYTHONPATH=src .venv-pai/Scripts/python.exe scripts/export_openvino.py --checkpoint models/smolvla_rescuehands --out models/openvino/fp32
PYTHONPATH=src .venv-pai/Scripts/python.exe scripts/evaluate.py --policy smolvla --export models/openvino/fp32 --device GPU --seeds 0:10 --supervisor on --fault glitch --video
PYTHONPATH=src .venv-pai/Scripts/python.exe scripts/benchmark_intel.py --export models/openvino/fp32
```

## Project layout

| Path | Responsibility |
| --- | --- |
| `src/rescuehandsai/scene.py` | Seeded dinner-table MJCF: table, plate, mats, cup, fork, spoon, lights, cameras |
| `src/rescuehandsai/sim.py` | The only module that owns MuJoCo state; named joints, cameras, fault injection |
| `src/rescuehandsai/control.py` | Rejects wrong names, non-finite values, stale commands, limit and step violations |
| `src/rescuehandsai/auditor.py` | Physics facts (held by both jaws, supported, zones, collisions) and debounced failure events |
| `src/rescuehandsai/runner.py` | Episode state machine, action guard, supervisor, recovery, progress watchdog |
| `src/rescuehandsai/evaluation.py` | Physical task success: in zone, supported, upright, released, settled, hand-off, spare utensil in place |
| `src/rescuehandsai/expert.py`, `kinematics.py`, `motion.py` | Scripted teacher (IK, contact grasps, hand-off) for data and baseline |
| `src/rescuehandsai/perturb.py` | Seeded gripper-glitch fault |
| `src/rescuehandsai/recorder.py`, `scripts/generate_dataset.py` | LeRobot dataset recording (successful episodes only) |
| `src/rescuehandsai/policies/` | Scripted and exported-SmolVLA policies behind one contract |
| `training/kaggle_pipeline.py` | GPU data generation and SmolVLA fine-tuning |
| `src/rescuehandsai/contract.py` | Model contract: joint order, cameras, units, rate and file hashes travel with the model |
| `src/rescuehandsai/showcase.py`, `scripts/render_showcase.py` | Presentation renders from nicer cameras (policy cameras untouched) |
| `scripts/evaluate.py`, `scripts/benchmark_intel.py` | Seeded evaluation (manifest + per-episode summary, optional video); Intel benchmark |
| `scripts/deploy_learned.sh`, `scripts/export_openvino.py` | Laptop deployment: download, contract check, OpenVINO export, evaluations |
| `scripts/review_*.py`, `scripts/spike_*.py` | Dated one-off probes from reviews and the OpenVINO spike (kept for traceability) |
| `results/audit_final_*`, `results/smolvla_*` | Committed evidence behind the README numbers |
| `docs/research/` | Independent reviews and audits (dated history) |
| `docs/superpowers/` | Design spec and implementation plans |
| `docs/media/` | README images |

## Design notes learned by measurement

- A straight-down SO-101 hand has only ~8 cm of vertical travel mid-table, so
  the teacher grasps with the hand tilted 1.2–1.4 rad.
- The fixed jaw sits 2 cm off the fingertip frame; grasps aim it 5–8 mm outside
  the object and 7 mm below the handle centre, otherwise fingertips pinch an edge and slip.
- The wrist camera mount sticks out 6 cm; jaw directions are chosen per arm so
  the cameras never meet during the hand-off.
- Sideways pushes up to 15 N and a low-friction "slip" did not dislodge a held
  utensil, so recovery is tested with a gripper glitch that really drops it.
- Shadows made offscreen rendering 6× slower on the iGPU; they are off for both
  training data and deployment so the policy sees identical images.

## Licenses

Project code: MIT (see `LICENSE`). SO-101 model: Apache-2.0, from
google-deepmind/mujoco_menagerie (derived from TheRobotStudio SO-ARM100),
downloaded separately. Table items are primitive shapes created for this project.
