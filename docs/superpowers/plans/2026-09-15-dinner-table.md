# Dinner Table Edition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans
> (inline; the owner has not asked for subagents). Steps use checkbox syntax.

**Goal:** Two SO-101 arms in MuJoCo set a dinner place (cup, named utensil
hand-off) under SmolVLA control, supervised and recovered, evaluated on 10
randomized seeds, deployed with OpenVINO on Intel CPU/iGPU.

**Architecture:** A config-built MuJoCo scene behind the existing simulator
adapter. A scripted IK teacher (data generation only) records LeRobot episodes.
SmolVLA is fine-tuned on Kaggle and deployed through a policy interface with
PyTorch and OpenVINO backends. A deterministic physics auditor and bounded
recovery wrap any policy. Evaluation and benchmark scripts write every number.

**Tech Stack:** Python 3.12, MuJoCo 3.13.0, NumPy, unittest; ML env: LeRobot
0.6.1 (SmolVLA), PyTorch CPU, OpenVINO 2026.3.1, NNCF; Kaggle GPU for training.

**Spec:** docs/superpowers/specs/2026-09-15-dinner-table-design.md

## Global Constraints

- Commit after every task.
- Two environments: `.venv-sim` (core, no torch) and `.venv-ml` (LeRobot,
  torch, OpenVINO, MuJoCo). `src/rescuehandsai` core modules never import torch.
- Units: metres, radians, simulated seconds. Joint order: left six, right six,
  names `JOINTS` in `sim.py`.
- No teleporting or welding objects. All grasps are contact and friction.
- IK only in `expert.py` (teacher). Deployment actions come from the policy.
- Privileged state only in teacher, auditor, evaluation. Never in policy input.
- Dataset/control rate 20 Hz (`control_dt` 0.05 = 10 physics steps of 0.005).
- Cameras for policy: `overhead`, `left_wrist`, `right_wrist`, 256×256 RGB.
- Record real device names in every benchmark; no Core Ultra claim.
- Geometry numbers below are starting values; tests decide the final values.
- Plain unittest: `.venv-sim/Scripts/python.exe -m unittest discover -s tests -v`.

## Implementation-detail note

Grasp geometry, IK gains and item sizes must be found by measurement in physics.
For those tasks this plan fixes the interface, the acceptance tests and the
method; the numeric values are tuned during execution and recorded in config.

---

## Phase A — De-risk early (runs alongside Phase B)

### Task A1: OpenVINO feasibility spike on pretrained SmolVLA (throwaway)

**Files:** `scripts/spike_openvino_smolvla.py` (kept only as evidence, labelled spike)

- [ ] Load `lerobot/smolvla_base` on CPU in `.venv-ml`; run one forward with
  dummy 3 images 256×256, 12-D state, instruction; time it.
- [ ] Identify the submodules: vision encoder + connector, VLM prefix (text +
  image tokens → KV cache), action expert denoising step.
- [ ] `openvino.convert_model` each submodule with example inputs; compile on
  `CPU`; compare outputs to PyTorch (max abs diff < 1e-3 FP32).
- [ ] List `openvino.Core().available_devices`; try `GPU` compile.
- [ ] Write findings to `docs/research/2026-09-15-openvino-spike.md`: what
  converted, diffs, latencies, device list, blockers. Commit.

---

## Phase B — Scene, teacher, auditor (core env)

### Task B1: Config-driven dinner scene

**Files:** Create `configs/scene.json`, `src/rescuehandsai/scene.py`,
`tests/test_scene.py`. Modify `src/rescuehandsai/sim.py`, `configs/simulation.json`.

**Interfaces:**
- Produces: `build_spec(config: dict, asset_path: Path, seed: int) -> mujoco.MjSpec`;
  `SceneParams` dataclass (sampled item poses, masses, frictions, sizes,
  light, table colour); `sample_params(config, seed) -> SceneParams`.
- Items (free bodies): `cup`, `fork`, `spoon`. Static: `plate`, `tray`.
  Zones (visual, no collision): `cup_zone`, `utensil_zone`.
- Arms: `left_arm` base (-0.20, 0, 0), `right_arm` base (0.20, 0, 0), both
  facing +y. Cameras: `overhead`, `front`, `left_arm/wrist_cam`, `right_arm/wrist_cam`.

- [ ] Tests first: scene compiles for seeds 0–9; all names exist; same seed →
  identical `SceneParams`; different seeds differ; every sampled value lies in
  its configured range; wrist cameras render non-empty 256×256 frames.
- [ ] Implement; run tests; commit.

### Task B2: Simulator adapter for many items

**Files:** Modify `contracts.py`, `sim.py`, `tests/test_simulation.py`.

**Interfaces:**
- `PrivilegedState.objects: Mapping[str, ObjectState]` with `position`,
  `quaternion`, `linear_velocity`; `contacts` unchanged.
- `MujocoSimulation.reset(seed)` rebuilds/reapplies `SceneParams`.
- `Observation.images` keys: `overhead`, `left_wrist`, `right_wrist` (+ `front`
  for video only when requested).
- `site_position(name) -> np.ndarray` for teacher/auditor (privileged).

- [ ] Update existing tests to the new names; add: each item settles on its
  support; `front` excluded from policy images. Implement; run; commit.

### Task B3: IK for the teacher

**Files:** Create `src/rescuehandsai/kinematics.py`, `tests/test_kinematics.py`.

**Interfaces:**
- `solve_ik(model, data, arm: str, target_pos, *, down: bool = True,
  q_init: Mapping[str,float]) -> dict[str, float] | None` — damped least squares
  on `{arm}/gripperframe` using `mj_jacSite` on a scratch `MjData`; five arm
  joints; optional downward-pointing orientation residual; respects limits;
  returns None if error > 5 mm after 200 iterations.

- [ ] Tests first: reaches `cup_zone`, `utensil_zone`, tray item spots and the
  hand-off point for the correct arm within 5 mm; unreachable point returns
  None; does not change the live `MjData`. Implement; commit.

### Task B4: Physics auditor

**Files:** Create `src/rescuehandsai/auditor.py`, `tests/test_auditor.py`.

**Interfaces:**
- `AuditFacts` dataclass: `held_by: dict[item, set[arm]]`, `on_table`,
  `in_zone: dict[item, zone|None]`, `speed: dict[item,float]`,
  `cross_arm_contact: bool`, `out_of_bounds: set[item]`.
- `PhysicsAuditor.update(state: PrivilegedState, sim) -> AuditFacts`;
  `held` = contacts with both the fixed-jaw and moving-jaw geoms of that arm.
- `FailureEvent(label: str, item: str|None, arm: str|None, time: float)`,
  labels from the spec.
- `detect_failures(prev: AuditFacts, now: AuditFacts, expectation) -> list[FailureEvent]`
  where expectation says which item should be held by which arm.

- [ ] Tests with pure data (no sim) for drop, failed grasp, out-of-bounds,
  collision; sim test: closing gripper around the cup gives `held_by`.
  Implement; commit.

### Task B5: Scripted teacher — cup placement

**Files:** Create `src/rescuehandsai/expert.py`, `tests/test_expert.py`.

**Interfaces:**
- `ScriptedExpert(sim, task: TaskSpec)`; `act(obs, state) -> BimanualAction`;
  `phase: str`; `done: bool`. Phases: `approach`, `descend`, `close`, `lift`,
  `carry`, `lower`, `open`, `retreat`. Joint targets are interpolated and
  clipped to `max_command_delta`.
- Grasp stability: add finger pad boxes in `scene.py` only if the cup slips in
  tests (measure first). Record the asymmetric grasp offset in `configs/scene.json`.

- [ ] Test first: seeds 0–4, right arm places cup; `PlacementTracker` success
  (inside `cup_zone`, supported, released, speed < 0.02 m/s for 0.5 s); no
  cross-arm contact. Render a debug image with the target site; inspect.
  Implement; commit.

### Task B6: Task spec, instructions, hand-off

**Files:** Create `src/rescuehandsai/task.py`, `tests/test_task.py`;
modify `expert.py`, `tests/test_expert.py`.

**Interfaces:**
- `TaskSpec(task_id, instruction, utensil: Literal["fork","spoon"], seed)`;
  `make_task(seed, utensil=None) -> TaskSpec` (instruction from paraphrase
  templates; utensil chosen by seed when None).
- `task_success(facts_history) -> bool`: cup in `cup_zone` stable; named
  utensil in `utensil_zone` stable, touched by both arms during the episode
  (hand-off); other utensil still in tray.
- Expert subtasks: `place_cup` → `pick_utensil` → `handoff` → `place_utensil`.

- [ ] Tests first: templates always name the utensil; full expert episode
  succeeds on seeds 0–9 for both utensils (target ≥ 9/10 each; record failures).
  Implement; commit.

### Task B7: Perturbation and bounded recovery

**Files:** Create `src/rescuehandsai/recovery.py`, `src/rescuehandsai/perturb.py`,
`tests/test_recovery.py`.

**Interfaces:**
- `Perturbation(kind="knock_utensil", time_s)` applied as an external force
  via `xfrc_applied` for 0.1 s (physics, not pose rewrite).
- `RecoveryPlanner(max_attempts=2).decide(event, subtask) -> RecoveryDecision`
  (`retry_grasp`, `reacquire`, `safe_abort`, `exhausted`).
- Expert honours decisions by rewinding to the subtask start phase.

- [ ] Tests first: planner limits and mapping (pure); sim: knock during
  `handoff` → `OBJECT_DROPPED` detected → expert reacquires → success on at
  least 7/10 seeds. Implement; commit.

---

## Phase C — Data (ML env)

### Task C1: LeRobot recorder and generator

**Files:** Create `scripts/generate_dataset.py`, `src/rescuehandsai/recorder.py`,
`tests/test_recorder.py` (skipped when lerobot is missing).

**Interfaces:**
- Features: `observation.images.overhead|left_wrist|right_wrist` (video,
  256×256×3), `observation.state` (12 float32, names = joint names),
  `action` (12 float32), task string = instruction. fps 20.
- `generate_dataset.py --episodes 120 --repo-id <hf_user>/rescuehands_table
  --root data/rescuehands_table --perturb-fraction 0.3`. Only successful
  episodes are saved; the log counts discarded ones.

- [ ] Test: 2-episode dataset loads with `LeRobotDataset`, shapes and names
  match. Generate full dataset; push to Hugging Face; commit scripts.

## Phase D — Training (Kaggle)

### Task D1: Kaggle training notebook

**Files:** Create `training/kaggle_train_smolvla.ipynb`, `training/README.md`.

- [ ] Install `lerobot[smolvla]==0.6.1`; `lerobot-train --policy.path=lerobot/smolvla_base
  --dataset.repo_id=<repo> --batch_size=16 --steps=<fit to time>
  --save_freq=1000 --policy.push_to_hub=true`; resume instructions.
- [ ] Record GPU type, steps, loss curve, wall time in `training/README.md`.

## Phase E — Deployment and Intel optimization

### Task E1: Policy interface and SmolVLA PyTorch adapter

**Files:** Create `src/rescuehandsai/policies/base.py`, `policies/scripted.py`,
`policies/smolvla_torch.py`, `tests/test_policies.py`.

**Interfaces:**
- `Policy.reset(task: TaskSpec)`; `Policy.act(obs: Observation) -> BimanualAction`;
  `Policy.metadata() -> dict` (name, backend, device, checkpoint).
- SmolVLA adapter maps named state ↔ 12-D vector by `sim.names` and asserts
  checkpoint feature names equal them.

- [ ] Tests: vector ordering round-trip; wrong feature names raise. Commit.

### Task E2: OpenVINO SmolVLA backend and benchmark

**Files:** Create `src/rescuehandsai/policies/smolvla_openvino.py`,
`scripts/export_openvino.py`, `scripts/benchmark_intel.py`, `tests/test_openvino.py`.

- [ ] Export converted submodules (from spike findings) to `models/openvino/{fp32,int8}`.
- [ ] Test: OpenVINO action chunk vs PyTorch within tolerance on 5 recorded frames.
- [ ] Benchmark: PyTorch CPU, OV FP32 CPU, OV INT8 CPU, OV GPU (if available):
  mean/p95 latency, peak memory, device names; write `results/benchmark.json`
  and `docs/intel-optimization.md` listing each optimization and measured effect.

## Phase F — Evaluation

### Task F1: Seeded evaluation runner

**Files:** Create `src/rescuehandsai/runner.py`, `scripts/evaluate.py`, `tests/test_runner.py`.

- [ ] `EpisodeRunner(sim, policy, supervisor: bool, perturbation)` state
  machine (IDLE … SUCCEEDED/FAILED/ABORTED) writes `result.json`,
  `steps.jsonl`, video.
- [ ] `evaluate.py --policy {scripted,smolvla_torch,smolvla_ov} --supervisor
  --seeds 0-9 --perturb knock_utensil` writes `results/eval_*.json` and a
  combined video. Test runner with scripted policy on 1 seed. Commit.
- [ ] Run matrix: scripted; SmolVLA; SmolVLA + supervisor; OpenVINO + supervisor.

### Task F2 (optional, after F1): VLM subtask selector

- [ ] OpenVINO GenAI VLM reads overhead image + instruction and outputs the next
  subtask string for SmolVLA; measured latency; ablation on seeds 0–9.

## Phase G — Submission

### Task G1: Package

- [ ] `LICENSE` (MIT; keep Apache-2.0 notice for SO-101 assets), README
  (setup, reproduce, results tables from `results/`, IK-only-in-teacher note,
  hardware statement), 10-seed video, slides, cover image. Final test run. Commit.
