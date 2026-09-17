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
The zero-action check runs on a **separate copy** of the simulation; the episode
itself starts from the untouched initial state (stepping the episode's world
before it starts was measured to change outcomes).
Thresholds are written in config and frozen before the teacher gate. Only these
rules may reject a scene. A scene is rejected as a whole if any of its four
cells has an invalid start. Rejected seeds are logged with the reason.

**No final-test scene is ever removed because the teacher or a learned policy
fails it.**

## 4. Physics version 2

Problem: item friction must actually govern grasp contact and slip.

- Change the contact setup so the utensil's sampled friction determines the
  jaw–utensil contact friction.
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
- The 569 old v2-dataset demonstrations are not reused.

## 5. Contacts and success rules

### Time steps

`substeps = control_dt / physics_dt`, checked to be a whole number (currently
0.05 / 0.005 = 10). The hold window is set in seconds (`hold_s = 1.0`); its
control-step and physics-step counts are derived from the configured time steps,
never hard-coded.

### Contact table

Contacts are read at **every physics step**, including between control updates.
Every contact is classified by the pair of bodies or geoms in a config file
(`configs/pick_contacts.json`). Robot body names are `{arm}/base`,
`{arm}/shoulder`, `{arm}/upper_arm`, `{arm}/lower_arm`, `{arm}/wrist`,
`{arm}/gripper` (holds the fixed jaw), `{arm}/camera_mount`,
`{arm}/moving_jaw_so101_v1`.

1. **Allowed structural robot contacts** — only explicitly named pairs:
   - each arm's `base` with the table (mounting);
   - `gripper` with `moving_jaw_so101_v1` of the same arm.

   No other robot–robot pair is allowed. Any other contact between two parts of
   the same arm is a **self-collision** and fails the episode. Before freezing,
   the list is checked against actual contacts at the home pose and during
   teacher runs; adding a pair requires a written reason in the spec, never an
   automatic addition.
2. **Normal scene contacts** (not robot): items resting on the table, mat and
   tray surfaces, and items touching each other at rest. These are allowed at
   the start and during the episode, and are checked only by the movement rules
   and the stricter support rule during the hold.
3. **Allowed task contacts:** right `gripper` or right `moving_jaw_so101_v1` (the
   "right jaws" in this spec; the `gripper` body carries the fixed jaw) with
   the named utensil; right jaws with the table or mat below the force limit.
4. **Forbidden contacts:** every other robot contact, including:
   - any part of either arm touching the spare utensil;
   - the left arm touching any item, the plate, the mat or the table (except its
     base mount);
   - any robot part touching the plate or cup;
   - any right-arm part other than the jaws touching the table, mat or tray;
   - right jaws with table or mat above the force limit;
   - arm–arm contact;
   - any pair not listed in the file (also flagged as `UNKNOWN_CONTACT_PAIR` for
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
   - its only contacts are the right gripper's jaws (no table, mat, tray, plate,
     cup, spare, other robot part or left arm support);
   - at least one right jaw touches it, and both jaws together touch it on at
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
   self-collision; no forbidden table, mat or plate contact.
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
- After a normal failure the episode keeps running to the deadline for diagnosis
  while the simulation stays stable. **Early stop** on severe violations: an item
  falls or leaves the workspace, arm–arm contact, force above the severe limit,
  NaN action, or unstable physics (non-finite state or simulator warning).
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
   - tiny overfit: the model fits 1 scene × 4 cells and picks the named item in
     all four open-loop and closed-loop;
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
are uploaded to Hugging Face. Default freezing (approach A). Unfreezing more
(approach B) is considered only if A misses the 95% bar on the dev selection set
and the language probe still shows weak word use.

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
    40 scene–instruction pairs**, compared episode by episode.

## 10. Where each step runs

1. **Laptop:** physics v2 and slip test, force measurements, contact-table check,
   teacher gate, and a timed run of 4 complete episodes (including model load and
   rendering) to replace the 7-hour estimate for the final test.
2. **Kaggle:** pilot and full data generation (GPU/EGL rendering, provenance),
   tiny overfit, both training runs, checkpoint screening and selection (native).
3. **Laptop Intel iGPU (OpenVINO):** parity checks, paired 40-episode native vs
   OpenVINO comparison, and — once, after the design and model are frozen — the
   400-episode main test and the 96-episode wording test.

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

- `manifest.json`: git commit, **source snapshot hash** (SHA-256 over all tracked
  source, config and script files as they are on disk), spec hash, physics version,
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

### Resume

A run resumes only when model, source snapshot hash, scene list, physics version,
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
- **Contacts:** named structural pairs allowed; unnamed same-arm contact is a
  self-collision; normal item–table contact at start does not fail; unknown pair
  forbidden and flagged; a brief contact between control steps is caught
  (simulation test); substeps derived from config and non-integer ratio rejected.
- **Validity and retry:** each error maps to its label; policy exceptions stay valid
  failures; one diagnosed retry allowed; second invalid attempt blocks; valid
  failures never retried.
- **Resume:** refuses on any mismatched hash; never overwrites attempts.
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
- **No regression:** all existing tests pass; the existing full-task evaluation is
  unchanged.

## 13. Order of work

1. Validity labels, contact table, `pick_outcome`, cells, scene identity (with tests).
2. Physics v2, slip test, force measurements; freeze start rules and force limits.
3. Pick teacher; calibrate hold tolerances on teacher; teacher gate (100 episodes);
   timeout check with p95 and maximum.
4. Seed/scene lists frozen (dev, main test, wording test).
5. Timed 4-episode laptop run.
6. Pilot data (96 episodes) and its full gate.
7. Full data (up to 1,200 episodes).
8. Two training runs, checkpoint screening and selection.
9. Export, parity, paired native vs OpenVINO check.
10. Freeze design and model; final main test and wording test, once.
