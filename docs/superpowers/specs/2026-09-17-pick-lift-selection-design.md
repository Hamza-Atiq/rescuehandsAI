# RescueHands AI — Milestone 1: select and lift the named utensil (design)

Status: design approved section by section by the project owner on 2026-09-17.
This file is the written spec for owner review before implementation.
AGENTS.md and `2026-09-15-dinner-table-design.md` still apply; where this file
is more specific, it wins for this milestone.

## 1. Why this milestone

The post-submission audit (`docs/research/2026-09-17-*`) measured the real
weakness of the v2 SmolVLA policy on seeds 0–9:

- the requested utensil was lifted first in only 3/10 scenes in each valid run;
- in the open-loop language probe, swapping "fork" ↔ "spoon" changed the model's
  actions by 0.024–0.126 rad, against 0.387–0.629 rad for the teacher;
- the model's hand leaned to tray slot 1 in 17/20 probes.

So the robot mostly does not follow the utensil word. This milestone fixes that
first, on a short task, before the hand-off and table-setting work resumes.

**Goal:** the right arm picks up and holds the utensil named in the instruction,
never the spare, across both tray orders.

**The full dinner-table task with a hand-off stays the final goal.** Later
milestones add hand-off and placement demonstrations to the same policy and
dataset family. This milestone's four-run test stays in use after that as a
regression check, because adding hand-off data does not guarantee the policy
keeps its pickup skill.

## 2. Scope

### In scope

1. Run validity and separate invalid-run labels (§6).
2. Physics version 2: item friction governs grasps, proven by a slip test, with a
   scene/config hash (§4).
3. Contact facts checked at every physics step, with a named contact table (§5).
4. Scene cells (tray-order swap) and frozen start-validity checks (§3).
5. A pick task and `pick_outcome` with the success rules (§5).
6. A pick-only teacher that must pass a 100-episode gate (§7).
7. A four-cell pick data generator with provenance and a balance report (§8).
8. A pick evaluation with per-cell, per-scene and grouped statistics (§10, §11).
9. The language probe as a standard script that tests real object selection (§9).
10. Model-path checks: action replay, tiny overfit, native vs OpenVINO parity (§9).

### Out of scope

- Supervisor and recovery changes. **This milestone runs with recovery disabled.**
- The teacher's slot-0 placement weakness in the full task (the teacher must still
  pass the pick gate in both slots).
- DAgger or other on-policy data collection.
- A phase planner or VLM. Any change of wording or task phase at run time must be
  explicit in the manifest.

### Rule for existing code

All changes are additive. The new pick task, outcome and scripts sit next to the
existing full-task code. The existing full-task evaluation, teacher and tests
must keep working unchanged.

**Physics version 1 is preserved explicitly.** Every existing full-task command
(`evaluate.py`, `generate_dataset.py`, `generate_recovery_dataset.py`, the Kaggle
pipeline's existing modes, showcase rendering) keeps `physics_version = 1` by
default and builds a byte-identical world XML. Only the new pick-task commands use
`physics_version = 2`, and they pass it explicitly. A regression check (§12)
proves version 1 is unchanged.

## 3. Scenes and the four cells

### Base scene

One seed makes one base scene with the existing randomization in `scene.py`:
utensil slot jitter (position and yaw), utensil mass, friction and size, cup
position, size, mass and friction, lighting and table color. The arms start at
the configured home pose with no perturbation.

### The four cells

Each base scene is run four times:

| Cell | Word | Tray order | Named item sits in |
|---|---|---|---|
| F-A | fork | fork slot 0, spoon slot 1 | slot 0 |
| F-B | fork | spoon slot 0, fork slot 1 | slot 1 |
| S-A | spoon | fork slot 0, spoon slot 1 | slot 1 |
| S-B | spoon | spoon slot 0, fork slot 1 | slot 0 |

"Everything else identical" means:

- slot jitter (position and yaw noise) belongs to the **slot**, so both items land
  on exactly the same spots after a swap;
- mass, friction and size belong to the **item**;
- cup, lighting, table color and cameras are the same in all four cells;
- all four cells use the same wording template; only the utensil word changes.

A test checks that the four generated scenes differ only in word and tray order.

### Start validity (frozen rules)

A start is physically invalid if items overlap, any item moves at rest beyond a
small speed threshold during the zero-action check, or any item is off the table.
"Zero action" means **both arms hold their configured home joint targets** (not
zero joint targets). The check runs on a **separate copy** of the simulation; the episode
itself starts from the untouched initial state (stepping the episode's world
before it starts was measured to change outcomes).
Thresholds are written in config and frozen before the teacher gate. Only these
rules may reject a scene. A scene is rejected as a whole if any of its four
cells has an invalid start. Rejected seeds are logged with the reason.

**No final-test scene is ever removed because the teacher or a learned policy
fails it.**

## 4. Physics version 2

Problem: item friction must actually govern grasp contact and slip. Measured
cause: the SO-101 asset's `collision_gripper` and `collision_gripper_mesh` classes
set `priority="1"` and `friction="1 5e-3 5e-4"`, so in MuJoCo the gripper's
friction wins over the utensil's sampled friction in every jaw contact.

- Change the version-2 contact setup so the utensil's sampled friction determines
  the jaw–utensil contact friction (for example by matching priority or setting the
  contact pair explicitly). The version-1 world is not changed.
- **Slip test:** with an identical scripted grasp and lift, compare low and high
  friction values. The test must show both (a) the measured jaw–utensil contact
  friction changes, and (b) slip (utensil movement in the gripper frame during
  the lift) is larger at low friction.
- Every dataset, checkpoint record and evaluation manifest stores
  `physics_version = 2` **and** two hashes: a **config hash** (SHA-256 over
  `configs/scene.json`, `configs/simulation.json`, `configs/pick_contacts.json`,
  the robot asset and the scene-building source file) and a per-scene **scene
  hash** (SHA-256 of that scene's generated world XML).
- Loading data or running evaluation with a mismatched version or hash is
  rejected by default. A future migration needs an explicit flag and a recorded
  reason.
- The 569 old v2-dataset demonstrations are not reused as training data. This
  check applies to **datasets and evaluations**, not to initial model weights: an
  old checkpoint may be used as a training starting point (§9).

## 5. Contacts and success rules

### Time steps

`substeps = control_dt / physics_dt`, checked to be a whole number (currently
0.05 / 0.005 = 10). The hold window is set in seconds (`hold_s = 1.0`); its
control-step and physics-step counts are derived from the configured time steps,
never hard-coded.

### Contact table

Contacts are read at **every physics step**, including between control updates.
Every contact is classified by its pair of **collision geoms (shapes)**, not
whole bodies, using a config file (`configs/pick_contacts.json`). Unnamed asset
geoms are identified by body, class, type and mesh name or index, recorded in
that file.

#### Physical vs decorative shapes (version-2 world)

Only shapes that can collide produce contacts. The contact monitor cannot see
anything else, so no rule may depend on a decorative shape.

| Physical (can collide) | Decorative (`contype=0`, no contacts) |
|---|---|
| `table`, `plate`, `cup_body`, all fork geoms (`fork_handle`, `fork_neck`, `fork_prong*`) and spoon geoms (`spoon_handle`, `spoon_bowl`) | `mat` (the red "tray" area the utensils visually sit on), `cup_zone`, `utensil_zone`, `plate_rim`, `cup_top` |
| robot `collision` boxes (group 3), `collision_gripper` shapes (group 3), `collision_gripper_mesh` meshes (group 4) | all robot `visual` meshes, all `sts3215` motor shapes, showcase floor |

The utensils physically rest on the **table**; "tray" and "mat" in wordings and
in this spec name the decorative area only. `{arm}/base` has no collision shapes,
so no base contact can occur. A test places shapes in contact for **every
forbidden contact class claimed below** and asserts the monitor detects it; any
class that cannot be detected is removed from the rules, not silently kept.

#### Right-jaw grasp shapes (exact list)

Only these right-arm shapes count as jaw contact for a grasp:

- fixed jaw (in `right_arm/gripper`): `fixed_jaw_box3` … `fixed_jaw_box7`,
  `fixed_jaw_sph_tip1` … `fixed_jaw_sph_tip3`;
- moving jaw (in `right_arm/moving_jaw_so101_v1`): `moving_jaw_box2`,
  `moving_jaw_box3`, `moving_jaw_sph_tip1` … `moving_jaw_sph_tip3`.

**Not** jaw contact: the unnamed gripper housing `collision` box,
`fixed_jaw_box1`, `fixed_jaw_box2`, `moving_jaw_box1`, `camera_box1`,
`camera_box2`, and the unnamed `collision_gripper_mesh` meshes of both jaws.
The first implementation step records which shapes actually touch the utensil
during teacher grasps. If a shape outside the list carries real grasp contact
(for example a jaw mesh), the list changes only by an owner-approved spec edit
naming that shape and the evidence.

#### Contact classes

1. **Allowed structural robot contacts.** MuJoCo's built-in filter already skips
   parent–child and welded-body pairs (for example `gripper`–`moving_jaw_so101_v1`,
   `gripper`–`camera_mount`). **No additional robot–robot pair is allowed.** Any
   other contact between two shapes of the same arm is a **self-collision** and
   fails. The home pose and teacher runs are checked for such contacts before
   freezing; an exception needs an owner-approved spec edit naming the exact
   shape pair.
2. **Normal scene contacts** (no robot shape involved): items resting on the
   table, items resting on the plate, and items touching each other at rest.
   Allowed at the start and during the episode; judged only by the movement rules
   and by the stricter support rule during the hold.
3. **Allowed task contacts:** a right-jaw grasp shape with the named utensil; a
   right-jaw grasp shape with the table below the force limit.
4. **Forbidden contacts:** every other contact that involves a robot shape,
   including:
   - any robot shape touching the spare utensil;
   - any left-arm shape touching anything;
   - any robot shape touching the plate or `cup_body`;
   - any right-arm shape that is not a jaw grasp shape touching the table or the
     named utensil (for example the housing box or a jaw mesh — recorded as
     `FORBIDDEN_CONTACT` until the list is changed as described above);
   - a jaw grasp shape with the table above the force limit;
   - arm–arm contact;
   - any pair not listed in the file (also flagged `UNKNOWN_CONTACT_PAIR` for
     review before freezing).

### Force limit

The jaw-to-table force limit is **not** "teacher maximum plus a margin". It is
set from controlled measurements: the gripper resting gently on the table, and
the gripper pressed down at set speeds. The limit is the value that clearly
separates gentle contact from hard hits, and the spec is updated with the
measurements and reason. A separate, higher severe-force limit triggers an early
stop. The teacher must satisfy the limit; if it cannot, the teacher is fixed, not
the limit. Both limits are frozen before the final test.

### The five success rules

All five must pass. **Failures latch:** once any rule fails, the episode can never
become successful.

1. **Right item only.** No forbidden contact with the spare utensil by any robot
   part, no left-arm contact with any item, and no other forbidden contact from
   the contact table, at any physics step.
2. **A real, stable hold.** A hold window of `hold_s` seconds in which, at every
   physics step:
   - the named utensil is at least 5 cm above its start height;
   - its only contacts are right-jaw grasp shapes (no table, plate, cup, spare,
     gripper housing, jaw mesh, other robot shape or left-arm support);
   - at least one jaw (fixed or moving) touches it through a grasp shape, and
     both jaws together touch it on at
     least 80% of the window's physics steps, with no single-jaw gap longer than
     a gap limit.

   Over the whole window, the utensil's pose **in the right gripper's frame**
   changes by less than 1 cm and less than 10°. Over the last 0.25 s of the
   window, its linear speed is below 2 cm/s and its angular speed below
   0.5 rad/s.

   The 80% coverage, gap limit and the pose/speed tolerances are starting values.
   They are calibrated on teacher runs and frozen before the final test. A test
   must show that an item swinging around the hand at constant distance fails.
3. **Spare stays put.** Maximum spare displacement over the episode < 1 cm and
   maximum yaw change < 5°.
4. **Tidy, safe scene.** Over the whole episode: maximum cup displacement < 1 cm,
   maximum cup tilt < 15°; no item falls or leaves the allowed workspace (the
   named utensil leaving the table surface is expected); no arm–arm contact; no
   self-collision; no forbidden table or plate contact.
5. **In time.** The hold window must complete by the deadline: 300 control steps
   (15 s) as a starting value. The teacher gate records the teacher's p95 and
   maximum hold-completion time; if either leaves too little margin under the
   deadline (p95 above 2/3 of the deadline, or maximum above 90% of it), the
   deadline is revised before any data is generated.

### Episode end and failure timing

- The episode ends at success, at the deadline, or at an early stop.
- **Immediate violations** are recorded at the physics step they happen:
  `WRONG_ITEM_TOUCHED`, `FORBIDDEN_CONTACT`, `SELF_COLLISION`,
  `ARM_ARM_CONTACT`, `SPARE_DISTURBED`, `CUP_DISTURBED`, `ITEM_FELL_OR_OUT`,
  `EXCESS_FORCE`, `UNKNOWN_CONTACT_PAIR`, `INVALID_ACTION`, `POLICY_ERROR`.
- **Unmet goals** are decided only at the deadline: `NO_LIFT` (never 5 cm up),
  `IMPROPER_HOLD` (lifted but no valid hold window), `TIMEOUT` (other).
- After a normal failure the episode keeps running to the deadline for diagnosis.
  **Early stop** (still a **valid failure**) on severe violations: an item falls or
  leaves the workspace, arm–arm contact, contact force above the severe limit
  (`EXCESS_FORCE` — a recorded physical outcome, not a simulator failure), or a
  NaN/non-finite action from the policy (`INVALID_ACTION`).
- A **simulator failure** ends the episode as **invalid** `SIM_ERROR` (§6). It is
  defined narrowly, and only these count:
  - MuJoCo raises an exception while stepping;
  - `qpos`, `qvel` or `qacc` becomes non-finite;
  - MuJoCo raises its bad-acceleration warning (`mjWARN_BADQACC`, where it
    automatically resets the state);
  - a contact or constraint buffer overflows (`mjWARN_CONTACTFULL`,
    `mjWARN_CNSTRFULL`), because the contact rules can no longer be trusted.

  All other MuJoCo warnings are recorded in the episode file and do **not** end
  the episode or change validity. Every `SIM_ERROR` counts toward the retry and
  blocking rule in §6, so repeated physics blow-ups cannot hide.
- The report lists **every** failed rule with its first step, plus the first
  failure in time. When an episode stops early or crashes, it lists the rules
  that could not be evaluated.

### "What did it pick up?"

Every episode records one of: **named first**, **spare first**, **neither**, or
**ambiguous/simultaneous** (both utensils first lifted within the same control
step). It also records whether both utensils were lifted at any time. "Lifted"
means 5 cm above start height.

### Record-only extras

First touch (step and object), maximum lift of each utensil, jaw coverage, peak
contact forces per pair, and time to hold.

## 6. Run validity

| Situation | Label | Result |
|---|---|---|
| Simulator or renderer crash | `SIM_ERROR` | invalid |
| Model or checkpoint fails to load | `MODEL_LOAD_ERROR` | invalid |
| Physics version, scene/config hash, camera names or shapes, state order, checkpoint hash or noise file mismatch | `CONTRACT_MISMATCH` | invalid |
| Exception or NaN while the policy runs | `POLICY_ERROR` / `INVALID_ACTION` | **valid failure** |
| Timeout or any rule failure | rule labels | valid failure |

- **Errors are labelled by their source, not by when they happen.** The runner
  wraps each stage separately: physics stepping and rendering/observation building
  → `SIM_ERROR`; model preprocessing, forward pass and action postprocessing
  inside the policy adapter → `POLICY_ERROR`; model loading → `MODEL_LOAD_ERROR`.
  A renderer failure while building an observation for a policy call is still
  `SIM_ERROR`.
- Contract checks run before the first episode; a mismatch stops the whole run.
- Policy exceptions are never reclassified as infrastructure errors.
- **Retry rule:** an invalid attempt gets **one** retry, only after the problem is
  diagnosed and the diagnosis is written in the attempt log. A second invalid
  attempt for the same episode **blocks the evaluation** until understood.
- **Valid task failures never get replacement attempts.**
- No silent fallbacks: for example OpenVINO must not quietly move from GPU to CPU.
  The device that actually ran is recorded.

## 7. Pick-only teacher and gate

- The teacher is the existing IK-based expert restricted to: approach the named
  utensil, grasp, lift, hold. IK is used only inside the teacher.
- It must pass both slots and satisfy the force limit and contact table.
- **Gate:** 25 dev scenes × 4 cells = 100 episodes. Pass if ≥ 99/100 overall and
  each cell ≥ 24/25. Also reported: scenes passing all four cells, failed teacher
  attempts, rejected starts, p95 and maximum time to hold.
- The gate uses the same `pick_outcome` as model evaluation.

## 8. Data

### Wordings

No template mentions left/right position or slot.

Training templates (8):

1. "pick up the {u} with the right hand" (the text the later planner already uses)
2. "grab the {u} with your right hand"
3. "lift the {u} off the tray using the right arm"
4. "take the {u} and hold it up"
5. "use the right gripper to pick up the {u}"
6. "get the {u} from the tray and hold it"
7. "please raise the {u} with your right hand"
8. "right hand: lift the {u}"

Held-back templates (4), never used in training or dev:

1. "I need the {u}, lift it with your right hand"
2. "hold up the {u} for me"
3. "could you pick the {u} up?"
4. "the {u}, please, in the right hand"

Templates are assigned by a **fixed, balanced rotation** over the ordered list of
accepted scenes, not randomly. All four cells of a scene share its template.
Where counts do not divide evenly (e.g. 100 ÷ 8), the templates that receive the
extra scenes are fixed in the frozen list before any result exists.

### Scene identity and seeds

New seed blocks:

| Use | Seed block |
|---|---|
| Training scenes | 3,000,000 – 3,000,999 |
| Dev: teacher gate (25 scenes) | from 3,100,000 |
| Dev: quick set (10 scenes = 40 episodes) | next accepted after gate |
| Dev: selection set (50 scenes = 200 episodes) | next accepted after quick set |
| Final main test (100 scenes = 400 episodes) | from 3,200,000 |
| Final wording test (24 scenes = 96 episodes) | from 3,300,000 |

Seeds are only one identity check:

- the generated scene settings of every scene are saved with a settings hash,
  `physics_version` and the scene/config hash;
- a duplicate-scene check runs across training, dev, main test and wording test;
- the check script compares new seed blocks against all ranges in existing
  provenance logs. Some older laptop shards have incomplete provenance; the report
  states that limitation instead of claiming overlap is impossible;
- dev lists, the main test list and the wording test list are saved as separate
  frozen files with hashes.

**Final-test lists are never run during development.** `evaluate.py` refuses
test-list scenes unless given `--final` and a frozen manifest containing the spec
hash and model hash.

### Balanced four-cell generation

- The generator runs the teacher on all four cells of a scene. If any cell fails,
  the whole scene is dropped from **training data**, keeping word × slot balance.
- Every attempt (kept or dropped, with reason) goes to provenance.
- **Acceptance-bias report:** teacher acceptance rate split by utensil size,
  mass, friction, slot jitter position and yaw, lighting, table color and cup
  position, at every data stage.

### Data stages — pilot is a firm gate

1. **Pilot: 24 training scenes × 4 = 96 episodes** (3 per template). Gate checks,
   all required before full generation or any long training:
   - sample-image sheet from every camera looks correct;
   - labels (instruction, word, cell, slot, template) match the scene;
   - replaying the saved actions in simulation reproduces each episode's outcome;
   - tiny overfit on 1 scene × 4 cells, judged two separate ways:
     - **open-loop (action comparison only):** on the recorded observations of
       all four cells, the model's predicted action chunks match the teacher's
       recorded actions within a mean-absolute-error tolerance written in config,
       and swapping the word changes the model's actions by at least half as much
       as it changes the teacher's. Open-loop results never claim a pickup;
     - **closed-loop (execution):** the model drives the simulation with its normal
       chunked control, no teacher, no replanning and no recovery, and
       `pick_outcome` succeeds in all four cells;
   - balance and acceptance-bias reports are complete.
2. **Full: grow to 300 training scenes × 4 = 1,200 episodes** (the planned
   maximum). Growth stops only if data checks fail. Attempt budgets come from
   the measured pilot yield and speed.

## 9. Models and model-path checks

### Starting models

Both `lerobot/smolvla_base` and `ABDHAM/smolvla_rescuehands_v2` are trained with
the **same data, the same batch size and the same number of steps**, so both see
the same number of training examples. The step count and batch size are declared
in the training manifest before training starts. Checkpoints every 2,000 steps
are uploaded to Hugging Face.

**Pinned training stack:** `lerobot[smolvla]==0.5.1` and `transformers==5.3.0`
(the versions already pinned in `training/kaggle_pipeline.py`, used to train v2).
The laptop's `.venv-ml` currently has lerobot 0.6.1 and transformers 5.5.4; any
native model check (tiny overfit, probe, parity) must run on the pinned versions,
and the implementation plan resolves this before those checks.

**Approach A — exact trainable components** (SmolVLA settings
`freeze_vision_encoder=True`, `train_expert_only=True`, `train_state_proj=True`):

- **frozen:** the whole vision-language model (SigLIP vision encoder, connector
  and the SmolLM2 text model);
- **trained:** the action expert transformer, the state projection, the action
  input and output projections, and the action-time MLP.

At training start the launcher lists every trainable parameter name and the
trainable/total counts into the training manifest. If the list differs from the
above, training stops. Approach B (unfreeze more) is considered only if A misses
the 95% bar on the dev selection set and the language probe still shows weak
word use.

**Old checkpoint as a starting point.** `ABDHAM/smolvla_rescuehands_v2` was trained
on physics-version-1 data. It is allowed **only as initial weights** for training;
its old demonstrations remain excluded. The physics-version and hash checks apply
to the new dataset, to the trained checkpoints (which record the version-2
dataset they were trained on) and to evaluations — not to the initial weights.
The training manifest records the init checkpoint ID and its Hub revision hash.

### Checkpoint selection (rule frozen before any result)

1. Screen **every** checkpoint of both runs on the 40-episode quick set (native
   PyTorch on Kaggle).
2. The **top 3 overall** (across both starting models) go to the 200-episode
   selection set.
3. Rank by: scenes passing all four cells, then episode successes, then fewer
   spare touches, then the earlier checkpoint.

### Checks

- **Action replay:** saved dataset actions replayed in simulation give the
  recorded outcome.
- **Tiny overfit:** see pilot gate.
- **Language probe (standard script):** tests actual object selection — for each
  dev quick-set scene, closed-loop runs of all four cells report the "what did it pick
  up?" result, plus open-loop action differences between word swaps.
- **Native vs OpenVINO parity:**
  - the same **saved noise tensors** (from file) are fed to both runtimes, with
    identical inputs and preprocessing;
  - observations come from several task stages (start, approach, grasp, lift);
  - per-stage maximum and mean action differences are reported against a tolerance
    written in config;
  - behavior check: the chosen model runs natively and on OpenVINO on the **same
    40 scene–instruction pairs**, compared episode by episode. Both sides run on
    the laptop, so the simulator build and machine are the same and only the
    inference runtime differs: **native PyTorch on the laptop CPU** versus
    **OpenVINO on the laptop Intel iGPU**, with the same saved noise.

## 10. Where each step runs

1. **Laptop:** physics v2 and slip test, force measurements, contact-table check,
   teacher gate, and a timed run of 4 complete episodes (including model load and
   rendering) to replace the 7-hour estimate for the final test. The timing run
   also measures native CPU episode time for the paired comparison.
2. **Kaggle (native PyTorch on the Kaggle T4 GPU):** pilot and full data
   generation (GPU/EGL rendering, provenance), tiny overfit, both training runs,
   checkpoint screening on the quick set and selection on the selection set.
3. **Laptop:** stage-wise parity checks (native PyTorch on CPU vs OpenVINO on
   iGPU, saved noise); the paired 40-episode comparison (native PyTorch on the
   laptop CPU vs OpenVINO on the laptop Intel iGPU); and — once, after the design
   and model are frozen — the 400-episode main test and the 96-episode wording
   test on **OpenVINO on the Intel iGPU**.

## 11. Evaluation reporting

### Pass bars (final main test, 100 scenes × 4 = 400 episodes)

Explicit quality targets, not promised training results:

| Measure | Target |
|---|---|
| Episode successes | ≥ 380 / 400 |
| Each cell | ≥ 92 / 100 |
| Episodes with any spare touch | ≤ 4 / 400 |
| Scenes passing all four cells | ≥ 92 / 100 |

The wording test (24 scenes × 4, 6 scenes per held-back template) has no pass
bar. It measures generalization to new wordings; it does not prove arbitrary
instructions work. Each template is reported separately.

### Statistics

- The scene is the unit of grouping; the four cells of a scene are not
  independent.
- Episode success rate and scene all-four rate each get a 95% confidence interval
  from a **scene-level bootstrap** (resample whole scenes).
- Denominators are always stated.
- **If the evaluation is incomplete or `evaluation_valid` is false, the summary
  shows "incomplete — not eligible to pass"** and does not apply the pass bars to
  the surviving episodes.

### Run directory and files

A run directory is tied to one experiment. Any change to model, code, scene list,
physics, rules or configuration needs a **new run directory**.

- `manifest.json`: git commit, **snapshot record** (below), spec hash, physics version,
  scene/config hash, rules/thresholds hash, model ID and checkpoint hash, backend,
  device actually used, noise mode and noise-file hash, scene-list hash, flags,
  start and end time, `finished`, `valid_count`, invalid counts per label,
  `evaluation_valid`.
- `attempts.jsonl`: one line per attempt (valid or invalid) with attempt ID,
  scene, cell, label, diagnosis note for invalid attempts, and file name.
- `episode_<scene>_<cell>_a<attempt>.json`: one file per attempt, **including
  invalid attempts with all measurements collected before the failure** — scene
  settings, cell, template and instruction, success, every failed rule with its
  first step, first failure, unevaluable rules, forbidden contacts (physics step,
  pair, force), hold measurements, "what did it pick up?", timing (load, inference
  per chunk, rendering).
- `summary.json` and `summary.md`: for each scored episode, **which attempt ID
  supplies the result**; episode and scene rates with intervals; results per cell,
  slot, word and template; spare touches; failure-label counts; the "what did it
  pick up?" table; pass or fail against each bar, or the incomplete notice.
- `evaluation_valid` is true only if every planned episode has exactly one scored
  valid attempt and no episode is blocked.

### Snapshot record

A hash of tracked files alone can miss code that actually runs, so the snapshot
record contains:

- **source hash:** SHA-256 over every file under `src/`, `scripts/`, `training/`
  and `configs/` as it is on disk, tracked or not;
- **untracked-file rule:** for data generation, training and the final test, the
  run **refuses to start** if any untracked or modified file exists in those
  folders. Development runs may start, but they list every untracked or modified
  file there with its own hash;
- **asset hash:** the robot XML and every mesh file it references, plus any other
  referenced asset;
- **runtime versions:** Python, MuJoCo, NumPy, PyTorch, lerobot, transformers,
  OpenVINO, NNCF, the OS, and device names and driver versions for CPU/iGPU/GPU;
- **model files:** checkpoint or export file hashes and the noise-file hash.

### Resume

A run resumes only when model, snapshot record, scene list, physics version,
rules/thresholds and configuration all match the manifest. Otherwise it refuses
and asks for a new run directory. Resume skips episodes that already have a
scored valid attempt and never overwrites attempt files.

## 12. Tests

- **Cells:** jitter follows slot, properties follow item, cells differ only in word
  and order; template rotation is balanced and fixed.
- **Rules (synthetic fact sequences):** each rule passes and fails; failures latch;
  `NO_LIFT` decided only at the deadline; all failed rules reported; early stop on
  severe violations; unevaluable rules listed on crash.
- **Hold:** item swinging at constant distance fails; slow stable hold passes;
  support by table or other robot part fails.
- **Contacts:** classification is by shape, not body — a utensil touching the
  gripper housing box, `fixed_jaw_box1` or a jaw mesh is not jaw contact; only the
  listed grasp shapes count; MuJoCo-filtered parent/weld pairs produce no contact;
  any other same-arm contact is a self-collision; normal item–table and
  item–plate contact at start does not fail; unknown pair forbidden and flagged;
  a brief contact between control steps is caught (simulation test); substeps
  derived from config and a non-integer ratio rejected.
- **Detectability:** for every forbidden contact class in §5, a simulation test
  forces that contact and asserts the monitor reports it; decorative shapes
  (`mat`, zones, `plate_rim`, `cup_top`) are asserted to have `contype=0` and to be
  absent from every rule.
- **Start check:** the zero-action check holds home joint targets on a copy of the
  simulation and leaves the episode's own state untouched (state compared before
  and after).
- **Physics errors:** exception, non-finite state, `mjWARN_BADQACC` and contact /
  constraint buffer overflow each give invalid `SIM_ERROR`; another MuJoCo warning
  is recorded without ending the episode; a severe force gives a valid
  `EXCESS_FORCE` failure.
- **Validity and retry:** each error maps to its label **by source** — a renderer
  failure while building an observation for a policy call is `SIM_ERROR`, an
  exception in the policy forward pass is `POLICY_ERROR`; policy exceptions stay
  valid failures; one diagnosed retry allowed; second invalid attempt blocks;
  valid failures never retried.
- **Snapshot and resume:** an untracked or modified file under `src/`, `scripts/`,
  `training/` or `configs/` changes the source hash and blocks data generation,
  training and the final test; a changed mesh file changes the asset hash; resume
  refuses on any mismatch; attempts are never overwritten.
- **Model settings:** the trainable-parameter list check stops training when it
  differs from approach A; a v1 init checkpoint is accepted as initial weights
  while a v1 dataset is rejected.
- **Physics version 1 regression:** before any code change, record (a) the
  version-1 world XML hash for seeds 0–9 and (b) the scripted full-task evaluation
  outcomes for seeds 0–9 (supervisor on, no fault) as a reference. After the
  changes, the default full-task commands must reproduce both exactly, and
  `physics_version` defaults to 1 for them.
- **Seeds and scenes:** overlapping seed ranges and duplicate scene settings are
  detected; test lists refused without `--final` and a frozen manifest.
- **Statistics:** known hand-worked examples give exact rates; grouping by scene is
  used (a case where episode-level and scene-level intervals differ); denominators
  are correct; invalid attempts never enter denominators; an incomplete run reports
  "incomplete — not eligible to pass".
- **"What did it pick up?":** named first, spare first, neither, simultaneous, and
  lifted-both flag.
- **Physics v2:** slip test (friction changes contact friction and slip); hash
  mismatch rejected.
- **End to end:** teacher on 2 dev scenes × 4 cells writes valid files; parity script
  with saved noise on a tiny export.
- **No regression:** all existing tests pass; the version-1 regression check above
  passes.

## 13. Order of work

0. Record the physics-version-1 reference (world XML hashes and scripted outcomes,
   seeds 0–9) before touching code.
1. Validity labels, contact table (shape-level, detectability tests),
   `pick_outcome`, cells, scene identity, snapshot record (with tests).
2. Physics v2, slip test, force measurements; record which shapes touch the
   utensil in teacher grasps; freeze start rules and force limits.
3. Pick teacher; calibrate hold tolerances on teacher; teacher gate (100 episodes);
   timeout check with p95 and maximum.
4. Seed/scene lists frozen (dev, main test, wording test).
5. Timed 4-episode laptop run.
6. Pilot data (96 episodes) and its full gate.
7. Full data (up to 1,200 episodes).
8. Two training runs, checkpoint screening and selection.
9. Export, parity, paired native vs OpenVINO check.
10. Freeze design and model; final main test and wording test, once.
