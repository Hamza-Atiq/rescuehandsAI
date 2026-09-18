# Pick-Lift Milestone — Plan 2: Measurements, Pick Teacher, Gate, Frozen Lists and the Evaluation CLI

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Plan 1 foundations into a measured evaluation system: a pick-only teacher, hold measurements the judge does not yet report, controlled force measurements, thresholds chosen from evidence that separates good holds from bad ones, a 100-episode teacher gate, frozen scene lists, an `evaluate_pick.py` CLI with resume and a hash-checked `--final` guard, and an honest timing measurement.

**Architecture:** Plan 1 built the judge, the fact reader, the contact classifier, the attempts ledger, the snapshot record, the statistics and `PickEpisodeRunner`, proven with a trivial policy. Plan 2 adds the first real policy (the teacher), the measurements the thresholds must come from, and the command-line entry point. New modules stay flat next to the Plan 1 `pick_*` modules. `expert.py` is **not** modified: the pick teacher wraps `ScriptedExpert` with `subtasks=("pick_utensil",)` and stops feeding it once it reaches its `home` phase, so the lift is held instead of being put back.

**Tech Stack:** Python 3.12, MuJoCo 3.13.0, NumPy, `unittest` (the repo has no pytest), Windows PowerShell, `.venv-sim`.

**Spec:** `docs/superpowers/specs/2026-09-17-pick-lift-selection-design.md` (commits `ab58560`, `2fb1a0b`). Read it before starting. Section numbers below (§) refer to it.

**Previous plan:** `docs/superpowers/plans/2026-09-17-pick-lift-plan1-foundations.md` — complete, 216 tests passing, HEAD `7fa26be`.

## Revision note (2026-09-18)

The first version of this plan was reviewed before execution and six measurement faults were found and fixed here. They are recorded because each one would have produced a confident but wrong number:

1. **The force script confused metres with radians** — it turned a speed in metres per second into a joint-angle step, so the "set speed" meant nothing. Task 4 now drives a real gripper trajectory and *measures* the vertical speed it achieved.
2. **Interfaces did not match the code.** `ContactVerdict` has `geoms`, not `names`. `StartCheck` has `valid` and `reasons`, not `ok` and `reason`. `PickOutcome` has **no** `hold` dictionary at all. Missing measurements are now added explicitly in Task 3, never faked with an empty dictionary.
3. **Hold calibration was circular** — it loosened the rules, called whatever passed a "good hold", and set thresholds from those episodes. Task 5 now records fixed-length traces, labels stable holds independently of the rules, deliberately collects slipping and swinging examples, and chooses thresholds that separate the two.
4. **The deadline used the wrong event** — when the hold *started*, although the rule requires the one-second hold to *finish* before the deadline. Task 6 uses `hold_completed_control_step`.
5. **Teacher timing cannot stand in for learned-policy timing.** The teacher requests no images and runs no neural network. Task 9 measures simulator and teacher cost only and explicitly does **not** replace the seven-hour estimate; that needs the real model and is Plan 4 work.
6. **The final-test guard only required hashes to be present**, so any string passed. Task 8 compares them with the real model, spec and frozen list, and tests the mismatches.

## Plan series

| Plan | Covers spec §13 steps | Output |
|---|---|---|
| 1 (done) | 0, 1, code half of 2 | Foundations and physics v2, proven with a trivial policy |
| **2 (this file)** | rest of 2, 3, 4, and the simulator half of 5 | Teacher, hold measurements, force evidence, calibrated thresholds, gate, frozen lists, `evaluate_pick.py`, simulator timing |
| 3 | 6, 7 | Four-cell data generator, provenance, balance and acceptance-bias reports, pilot gate, full data |
| 4 | 8, 9 | Two training runs, checkpoint screening and selection, export, saved-noise parity, paired native vs OpenVINO, **learned-policy timing** |
| 5 | 10 | Freeze, final main test and wording test, reports |

## Owner checkpoint — stop before freezing anything

**After Task 4 (force evidence), stop and report.** Tasks 1–4 only *measure*. Tasks 5 and 6 freeze thresholds that every later result depends on, and the owner reviews the evidence before that happens. Do not start Task 5 without that review.

## Global Constraints

- All changes are additive. `src/rescuehandsai/expert.py`, `runner.py`, `control.py` and `scripts/evaluate.py` keep today's behaviour; the full-task path stays on `physics_version = 1` (§2).
- **Commit after each task. Never push to GitHub** (owner instruction).
- **Never commit the owner's documentation edits.** `README.md`, `docs/ROADMAP.md`, `docs/submission/lablab-submission.md` and everything under `docs/research/` stay uncommitted until the owner explicitly says to commit them. Stage files by explicit path; never `git add -A` or `git add .`.
- Run tests from the project root in PowerShell with `$env:PYTHONPATH = "src"` and `.venv-sim/Scripts/python.exe -m unittest ...`.
- **No final-test episode is ever executed in this plan.** Seed blocks from 3,200,000 upward may be generated and frozen in Task 7, and nothing more (§8).
- The force limit is **not** "teacher maximum plus a margin" (§5). If the teacher cannot satisfy the measured limit, **the teacher is fixed, not the limit**.
- A threshold may never be chosen so that the teacher passes. Every frozen value needs evidence that it separates good behaviour from bad behaviour, and that evidence goes into the spec with its date.
- When a measurement is missing, **add it** (additively, with tests). Never substitute an empty dictionary, a default or a guess.
- Text files are hashed with CRLF normalised to LF (`core.autocrlf` is on in this checkout, off on Kaggle).

## Facts verified in the code on 2026-09-18 (do not re-derive; do not assume anything beyond these)

- Policy protocol required by `PickEpisodeRunner` (`pick_runner.py:32-130`): attribute `uses_privileged_state`, methods `reset(sim, task)`, `wants_images() -> bool`, `act(obs) -> BimanualAction`, `metadata() -> dict`.
- `ContactVerdict` (`pick_contacts.py:15-22`) fields: `kind` (`SCENE_NORMAL`, `JAW_UTENSIL`, `JAW_TABLE`, `VIOLATION`), `labels`, **`geoms`** (a 2-tuple of names), `force`, `jaw` (`"fixed"`, `"moving"` or `None`).
- `StartCheck` (`pick_cells.py:96-100`) fields: `cell`, **`valid`**, **`reasons`** (a tuple).
- `PickOutcome` (`pick_outcome.py:43-66`) has **no** `hold` dictionary. It does have `hold_completed`, `hold_completed_physics_step`, **`hold_completed_control_step`**, `success_physics_step`, `success_control_step`, `max_lift_m`, `spare_max_shift_m`, `spare_max_yaw_deg`, `cup_max_shift_m`, `cup_max_tilt_deg`, `longest_eligible_streak`.
- `PickJudge._window` (`pick_outcome.py:93`) is a `deque` of 5-tuples `(both_jaws, position, rotation, speed, spin)` with `maxlen=steps.hold`. `_window_ok()` (`pick_outcome.py:177-194`) compares these against the thresholds but **never reports the values**. Task 3 adds the reporting.
- `PickEpisodeRunner` builds the `ContactClassifier` **after** `sim.reset(...)` (`pick_runner.py:63-68`). Any script that resets the simulation must rebuild the classifier afterwards; geom ids are only valid for the model that produced them.
- `sim.step(action, on_substep=..., stop_on_cross_arm=...)` calls `on_substep()` after **every physics step**, and a truthy return stops that control step (`sim.py:198-226`). This is the only way to sample a quantity at physics rate.
- `ScriptedExpert(sim, task: TaskSpec, subtasks=SUBTASKS)`; `SUBTASKS = ("pick_utensil", "handoff", "place_utensil", "place_cup")` (`task.py:22`). `_script()` always ends with a home move and sets `self.subtask = "home"` (`expert.py:148-163`). `act()` returns `BimanualAction(obs.timestamp, dict(self.follower.targets))`; `self.done` marks the exhausted script (`expert.py:59-71`). It raises `PlanningError` and `LostItemError` (`expert.py:34-40`).
- `configs/pick_rules.json` has `"frozen": false`; `configs/pick_contacts.json` has both force limits `null` and `"force_limits_frozen": false`.
- `ContactClassifier.classify()` applies a force limit only when it is not `None` (`pick_contacts.py:93-99`).
- Seed blocks (`configs/pick_seed_blocks.json`): train `3000000–3000999`, dev `3100000–3199999`, test_main `3200000–3299999`, test_wording `3300000–3399999`.
- `deadline_control_steps` is currently `300` (15 s at 20 Hz control).

## File map

| File | Status | Responsibility |
|---|---|---|
| `src/rescuehandsai/pick_teacher.py` | create | `PickTeacher` — pick-only teacher matching the runner's policy protocol |
| `src/rescuehandsai/pick_outcome.py` | modify | **Add** hold-trace measurements and report them on `PickOutcome` (Task 3) |
| `src/rescuehandsai/pick_run.py` | create | Run-directory plumbing: plan, final guard, episode loop, summary |
| `scripts/measure_grasp_shapes.py` | create | Which shapes really touch the utensil in teacher grasps (§13 step 2) |
| `scripts/measure_forces.py` | create | Gentle-rest and pressed-down forces from a real, speed-measured trajectory (§5) |
| `scripts/calibrate_hold.py` | create | Fixed-length hold traces, independent labels, threshold separation (§5) |
| `scripts/teacher_gate.py` | create | 25 dev scenes × 4 cells gate; deadline from hold completion (§7) |
| `scripts/freeze_scene_lists.py` | create | Generate, duplicate-check and freeze the scene lists (§8) |
| `scripts/evaluate_pick.py` | create | Evaluation CLI: manifest, resume, hash-checked `--final` guard (§11) |
| `scripts/time_simulation.py` | create | Simulator and teacher cost only — **not** a learned-policy projection (§10) |
| `tests/test_pick_teacher.py`, `tests/test_hold_trace.py`, `tests/test_grasp_shape_evidence.py`, `tests/test_force_limits.py`, `tests/test_hold_calibration.py`, `tests/test_teacher_gate.py`, `tests/test_scene_lists.py`, `tests/test_pick_run.py`, `tests/test_timing_report.py` | create | Tests |
| `configs/pick_contacts.json` | modify | Measured force limits, `force_limits_frozen: true` (Task 6 only) |
| `configs/pick_rules.json` | modify | Calibrated hold tolerances, measured deadline, `frozen: true` (Task 6 only) |
| `configs/pick_scene_lists/*.json` | create (generated) | Frozen lists with hashes and rejection records |
| `docs/superpowers/specs/2026-09-17-pick-lift-selection-design.md` | modify | Every measurement and the reason for every frozen value |
| `results/measurements/`, `results/teacher_gate/` | create (generated) | Evidence |

---

### Task 1: The pick-only teacher

**Files:**
- Create: `src/rescuehandsai/pick_teacher.py`
- Test: `tests/test_pick_teacher.py`

**Interfaces:**
- Consumes: `ScriptedExpert`, `TaskSpec` (`task.py:26`), `PickTask` (`pick_cells.py:37`), `BimanualAction`.
- Produces: `PickTeacher()` with `uses_privileged_state = True`, `reset(sim, task)`, `wants_images() -> bool` (always `False`), `act(obs) -> BimanualAction`, `metadata() -> dict`, and read-only attributes `phase: str` (`"start"`, `"pick"`, `"hold"`) and `finished_pick_at: int | None` (the control step at which the expert reached its home phase).

- [ ] **Step 1: Write the failing test**

Create `tests/test_pick_teacher.py`:

```python
import unittest

from rescuehandsai.pick_cells import make_pick_task
from rescuehandsai.pick_config import load_rules
from rescuehandsai.pick_runner import PickEpisodeRunner
from rescuehandsai.pick_teacher import PickTeacher
from rescuehandsai.sim import MujocoSimulation


class TeacherProtocolTests(unittest.TestCase):
    """The runner accepts only a policy with this exact shape."""

    def test_metadata_names_the_teacher_and_claims_no_checkpoint(self):
        meta = PickTeacher().metadata()
        self.assertEqual(meta["name"], "pick_teacher")
        self.assertEqual(meta["backend"], "python")
        self.assertEqual(meta["device"], "cpu")
        self.assertIsNone(meta["checkpoint"])

    def test_teacher_declares_privileged_state(self):
        # It reads exact object poses and uses IK; the learned policy never does.
        self.assertTrue(PickTeacher.uses_privileged_state)

    def test_teacher_needs_no_images(self):
        self.assertFalse(PickTeacher().wants_images())

    def test_act_before_reset_is_a_clear_error(self):
        with self.assertRaises(RuntimeError):
            PickTeacher().act(object())


class TeacherPickTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = MujocoSimulation(physics_version=2)
        cls.rules = load_rules()

    @classmethod
    def tearDownClass(cls):
        cls.sim.close()

    def test_teacher_lifts_the_named_utensil_and_keeps_holding_it(self):
        task = make_pick_task(3100000, "F-A", "T1")
        teacher = PickTeacher()
        record = PickEpisodeRunner(self.sim, teacher, rules=self.rules).run(task)
        self.assertTrue(record["success"], record["outcome"]["failures"])
        self.assertEqual(record["outcome"]["picked"], task.utensil)
        self.assertIsNotNone(teacher.finished_pick_at)

    def test_teacher_holds_position_after_the_pick_instead_of_going_home(self):
        """The expert's script ends with a home move that would put the utensil back."""
        task = make_pick_task(3100000, "F-A", "T1")
        teacher = PickTeacher()
        teacher.reset(self.sim, task)
        seen = []
        for _ in range(400):
            action = teacher.act(self.sim.observe(images=False))
            seen.append(dict(action.targets))
            if teacher.finished_pick_at is not None and len(seen) > teacher.finished_pick_at + 5:
                break
            self.sim.step(action, stop_on_cross_arm=False)
        self.assertIsNotNone(teacher.finished_pick_at, "the teacher never finished the pick")
        after = seen[teacher.finished_pick_at:]
        for targets in after[1:]:
            self.assertEqual(targets, after[0], "targets moved after the pick was finished")
        self.assertEqual(teacher.phase, "hold")

    def test_a_planning_failure_is_reported_as_a_policy_error_not_a_crash(self):
        from rescuehandsai.expert import PlanningError

        class Unreachable(PickTeacher):
            def act(self, obs):
                raise PlanningError("no reachable pose")

        record = PickEpisodeRunner(self.sim, Unreachable(),
                                   rules=self.rules).run(make_pick_task(3100000, "F-A", "T1"))
        self.assertEqual(record["outcome"]["first_failure"], "POLICY_ERROR")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_pick_teacher.py"`
Expected: `ModuleNotFoundError: No module named 'rescuehandsai.pick_teacher'`.

- [ ] **Step 3: Write the teacher**

Create `src/rescuehandsai/pick_teacher.py`:

```python
"""Pick-only teacher: approach the named utensil, grasp, lift, then hold (spec section 7).

It wraps the existing ScriptedExpert with only the `pick_utensil` subtask. The expert's
script always ends with a home move that would lower the utensil again, so the teacher
stops feeding the expert as soon as it reaches that phase and repeats the last command
instead. IK lives inside the teacher only; the deployed policy never uses it.
"""
from .contracts import BimanualAction
from .expert import ScriptedExpert
from .pick_cells import PickTask
from .task import TaskSpec


class PickTeacher:
    uses_privileged_state = True

    def __init__(self):
        self.expert = None
        self.phase = "start"
        self.finished_pick_at = None
        self._targets = None
        self._steps = 0

    def reset(self, sim, task: PickTask):
        spec = TaskSpec(task_id=f"pick_{task.seed}_{task.cell}", instruction=task.instruction,
                        utensil=task.utensil, seed=task.seed)
        self.expert = ScriptedExpert(sim, spec, subtasks=("pick_utensil",))
        self.phase = "pick"
        self.finished_pick_at = None
        self._targets = dict(sim.previous)
        self._steps = 0

    def wants_images(self) -> bool:
        return False

    def metadata(self) -> dict:
        return {"name": "pick_teacher", "backend": "python", "device": "cpu", "checkpoint": None,
                "subtasks": ["pick_utensil"], "uses_ik": True}

    def act(self, obs) -> BimanualAction:
        if self.expert is None:
            raise RuntimeError("PickTeacher.act called before reset")
        if self.phase == "pick":
            action = self.expert.act(obs)
            # `home` is the expert's own wind-down move and `done` means the script ran out.
            # Either way the pick is over: freeze the command so the lift is held.
            if self.expert.subtask == "home" or self.expert.done:
                self.phase = "hold"
                self.finished_pick_at = self._steps
            else:
                self._targets = dict(action.targets)
        self._steps += 1
        return BimanualAction(obs.timestamp, dict(self._targets))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_pick_teacher.py"`
Expected: 7 tests `ok`.

If the lift test fails, **do not relax any success rule**. Print `record["outcome"]["failures"]` and report which rule failed and at which step. Likely causes in order: the expert reached `home` before the lift height was met; the frozen grasp-shape list misses a shape that really holds the utensil (that is Task 2 — record the failure and continue); the hold window is longer than the frozen command holds still.

- [ ] **Step 5: Commit**

```powershell
git add src/rescuehandsai/pick_teacher.py tests/test_pick_teacher.py
git commit -m "Pick plan 2 task 1: pick-only teacher that holds the lift instead of going home"
```

---

### Task 2: Evidence for the frozen grasp-shape list

**Files:**
- Create: `scripts/measure_grasp_shapes.py`
- Create: `tests/test_grasp_shape_evidence.py`
- Generated: `results/measurements/grasp_shapes.json`

**Interfaces:**
- Consumes: `PickTeacher`, `ContactClassifier` (`pick_contacts.py:34`), `collides` (`pick_contacts.py:30`), `load_contacts`, `PickEpisodeRunner`, `make_pick_task`.
- Produces: `measure(sim, seeds, cells) -> dict` with `touch_counts` (robot shape name without the arm prefix → number of episodes in which it touched the named utensil), `jaw_shapes_seen`, `unlisted_shapes`, `non_colliding_listed`, `episodes`.

Why this task exists: the jaw grasp list in `configs/pick_contacts.json` was written by reading the robot XML, not by watching a grasp. If a shape that really holds the utensil is missing from the list, rule 2 ("a real, stable hold") rejects good grasps for ever and no amount of training fixes it (§5).

**Interface note:** `ContactVerdict` exposes **`geoms`**, a 2-tuple of names — not `names`. Robot shapes carry an arm prefix (`right_arm/fixed_jaw_box5`); scene shapes do not (`fork_handle`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_grasp_shape_evidence.py`:

```python
import json
import unittest
from pathlib import Path

from rescuehandsai.pick_config import load_contacts

REPORT = Path(__file__).resolve().parents[1] / "results" / "measurements" / "grasp_shapes.json"


class GraspShapeEvidenceTests(unittest.TestCase):
    """The frozen jaw list must match what actually touches the utensil in real grasps."""

    def setUp(self):
        if not REPORT.is_file():
            self.skipTest("run scripts/measure_grasp_shapes.py first (Task 2 step 3)")
        self.report = json.loads(REPORT.read_text(encoding="utf-8"))
        contacts = load_contacts()
        self.fixed = set(contacts["jaw_grasp_geoms"]["fixed"])
        self.moving = set(contacts["jaw_grasp_geoms"]["moving"])

    def test_no_unlisted_shape_holds_the_utensil(self):
        self.assertEqual(self.report["unlisted_shapes"], [],
                         "these shapes touched the utensil but are not in jaw_grasp_geoms")

    def test_both_jaws_are_represented_in_real_grasps(self):
        seen = set(self.report["jaw_shapes_seen"])
        self.assertTrue(seen & self.fixed, "no fixed-jaw shape ever touched the utensil")
        self.assertTrue(seen & self.moving, "no moving-jaw shape ever touched the utensil")

    def test_every_listed_shape_can_actually_collide(self):
        # A listed shape that cannot collide would be dead weight in the rules.
        self.assertEqual(self.report["non_colliding_listed"], [])

    def test_the_evidence_covers_successful_grasps(self):
        successes = [e for e in self.report["episodes"] if e["success"]]
        self.assertGreaterEqual(len(successes), 8, "too few successful grasps to conclude anything")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it skips**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_grasp_shape_evidence.py"`
Expected: 4 tests `skipped`.

- [ ] **Step 3: Write the measurement script**

Create `scripts/measure_grasp_shapes.py`:

```python
"""Record which collision shapes really touch the named utensil during teacher grasps.

The frozen jaw list in configs/pick_contacts.json was read off the robot XML. This checks
it against real grasps: every shape that touches the named utensil at any physics step is
counted, and any shape outside the list is reported. Spec section 13, step 2.

Usage (project root):
  $env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe scripts/measure_grasp_shapes.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rescuehandsai.pick_cells import make_pick_task                             # noqa: E402
from rescuehandsai.pick_config import load_contacts, load_rules                 # noqa: E402
from rescuehandsai.pick_contacts import ContactClassifier, collides             # noqa: E402
from rescuehandsai.pick_runner import PickEpisodeRunner                         # noqa: E402
from rescuehandsai.pick_teacher import PickTeacher                              # noqa: E402
from rescuehandsai.sim import MujocoSimulation                                  # noqa: E402

SEEDS = (3100000, 3100001, 3100002, 3100003)
CELLS = ("F-A", "F-B", "S-A", "S-B")


class Watcher(PickTeacher):
    """A teacher that also notes every robot shape touching the named utensil.

    The classifier is built in reset(), after the runner has reset the simulation, because
    geom ids belong to the model that produced them.
    """

    def __init__(self, sim, contacts):
        super().__init__()
        self._sim, self._contacts = sim, contacts
        self.classifier = None
        self.named = None
        self.touched = Counter()

    def reset(self, sim, task):
        super().reset(sim, task)
        self.named = task.utensil
        self.classifier = ContactClassifier(sim.model, task.utensil, self._contacts)

    def act(self, obs):
        for verdict in self.classifier.contacts(self._sim.data):
            if self.named in verdict.geoms or any(g.startswith(f"{self.named}_") for g in verdict.geoms):
                for name in verdict.geoms:
                    if "/" in name:  # robot shapes carry an arm prefix
                        self.touched[name.split("/", 1)[1]] += 1
        return super().act(obs)


def measure(sim, seeds=SEEDS, cells=CELLS) -> dict:
    contacts, rules = load_contacts(), load_rules()
    listed = set(contacts["jaw_grasp_geoms"]["fixed"]) | set(contacts["jaw_grasp_geoms"]["moving"])
    touch_episodes, episodes = Counter(), []
    for seed in seeds:
        for cell in cells:
            task = make_pick_task(seed, cell, "T1")
            watcher = Watcher(sim, contacts)
            record = PickEpisodeRunner(sim, watcher, rules=rules).run(task)
            for name in watcher.touched:
                touch_episodes[name] += 1
            episodes.append({"seed": seed, "cell": cell, "success": record["success"],
                             "picked": record["outcome"]["picked"],
                             "shapes": sorted(watcher.touched)})
    non_colliding = sorted(n for n in listed
                           if not collides(sim.model, int(sim.model.geom(f"right_arm/{n}").id)))
    seen = set(touch_episodes)
    return {"touch_counts": dict(sorted(touch_episodes.items())),
            "jaw_shapes_seen": sorted(seen & listed),
            "unlisted_shapes": sorted(seen - listed),
            "non_colliding_listed": non_colliding,
            "episodes": episodes}


def main():
    sim = MujocoSimulation(physics_version=2)
    try:
        report = measure(sim)
    finally:
        sim.close()
    out = ROOT / "results" / "measurements" / "grasp_shapes.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print("jaw shapes seen :", ", ".join(report["jaw_shapes_seen"]) or "(none)")
    print("unlisted shapes :", ", ".join(report["unlisted_shapes"]) or "(none)")
    total = len(report["episodes"])
    for name, count in report["touch_counts"].items():
        print(f"  {name}: touched the utensil in {count} of {total} episodes")


if __name__ == "__main__":
    main()
```

**Check before running:** open `pick_contacts.py` and confirm how the utensil's own shapes are named in `geoms` (the scene list gives `fork_handle`, `fork_neck`, `fork_prong0..2`, `spoon_handle`, `spoon_bowl`). If the prefix test above does not match those names, use the scene-shape list from `load_contacts()["scene_geoms"][task.utensil]` instead of a string prefix, and say so in the task report.

- [ ] **Step 4: Run the measurement**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe scripts/measure_grasp_shapes.py`
Expected: a printed table and `results/measurements/grasp_shapes.json`.

**Stop and report** if `unlisted shapes` is not empty. Do not add shapes to the config on your own. Report which shape held the utensil, in how many episodes, and which part of the gripper it belongs to. The owner decides whether the frozen list changes, and the spec records the change with this measurement as its reason.

- [ ] **Step 5: Run the evidence tests**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_grasp_shape_evidence.py"`
Expected: 4 tests `ok`.

- [ ] **Step 6: Record the measurement in the spec and commit**

Add a dated line under "#### Right-jaw grasp shapes (exact list)": which shapes were observed holding the utensil, over how many episodes, and that the list was confirmed — or exactly what changed and why.

```powershell
git add scripts/measure_grasp_shapes.py tests/test_grasp_shape_evidence.py docs/superpowers/specs/2026-09-17-pick-lift-selection-design.md
git add -f results/measurements/grasp_shapes.json
git commit -m "Pick plan 2 task 2: grasp-shape evidence from real teacher grasps"
```

---

### Task 3: Make the judge report its hold measurements

**Files:**
- Modify: `src/rescuehandsai/pick_outcome.py` (additive only)
- Test: `tests/test_hold_trace.py`

**Interfaces:**
- Consumes: `SubstepFacts` (`pick_outcome.py:30`), `StepCounts` (`pick_config.py:29`).
- Produces:
  - `PickJudge(named, spare, rules, steps, table_center, table_half, *, keep_trace: bool = False)` — the new keyword defaults to today's behaviour.
  - `PickJudge.window_measurements() -> dict | None` — measurements of the window as it stands, or `None` when the window is not full. Keys: `both_jaw_fraction`, `longest_single_jaw_gap_steps`, `max_shift_m`, `max_turn_deg`, `max_speed_mps`, `max_angular_speed_rps`.
  - `PickJudge.trace: list` — one entry per physics step when `keep_trace=True`, each with `physics_step`, `control_step`, `eligible` (bool: lifted, gripped, nothing else touching the utensil), `both_jaws`, `rel_pos` (3 floats, gripper frame), `speed_mps`, `angular_speed_rps`, `lift_m`.
  - `PickOutcome.hold_measurements: dict | None` — the measurements of the window that completed the hold, or of the longest full window seen if the hold never completed.
  - `PickOutcome.best_window_measurements: dict | None` — the longest full window's measurements, always, so a failed hold still carries evidence.
  - `PickOutcome.hold_trace: list` — the trace when `keep_trace=True`, otherwise empty.

Why this task exists: `PickOutcome` has **no** `hold` dictionary today. `PickJudge._window` holds 5-tuples `(both_jaws, position, rotation, speed, spin)` and `_window_ok()` compares them with the thresholds but throws the values away (`pick_outcome.py:177-194`). Thresholds cannot be calibrated against numbers nobody records. **The rule logic in `_window_ok()` does not change** — this task only adds reporting, so the v1-style behaviour of every existing test is untouched.

- [ ] **Step 1: Write the failing test**

Create `tests/test_hold_trace.py`. Build synthetic `SubstepFacts` the way `tests/test_pick_outcome.py` already does — read that file first and copy its fact-building helper rather than inventing a second one:

```python
import unittest

import numpy as np

from rescuehandsai.pick_config import StepCounts, load_rules
from rescuehandsai.pick_outcome import PickJudge


def steps() -> StepCounts:
    # Short windows keep the synthetic sequences readable.
    return StepCounts(hold=4, max_gap=1, final_speed=2, deadline_control=50, deadline_physics=200)


class WindowMeasurementTests(unittest.TestCase):
    """The judge must report the numbers it already compares against thresholds."""

    def setUp(self):
        self.rules = load_rules()
        self.judge = PickJudge("fork", "spoon", self.rules, steps(), (0.0, 0.0), (1.0, 1.0),
                               keep_trace=True)

    def test_an_empty_window_reports_nothing_rather_than_zeros(self):
        # Zeros would look like a perfectly steady hold that never happened.
        self.assertIsNone(self.judge.window_measurements())

    def test_a_full_steady_window_reports_small_movement_and_full_jaw_coverage(self):
        self._feed(count=4, drift=0.0, both=True, speed=0.0)
        m = self.judge.window_measurements()
        self.assertEqual(m["both_jaw_fraction"], 1.0)
        self.assertEqual(m["longest_single_jaw_gap_steps"], 0)
        self.assertAlmostEqual(m["max_shift_m"], 0.0, places=9)
        self.assertAlmostEqual(m["max_speed_mps"], 0.0, places=9)

    def test_a_drifting_window_reports_the_drift_it_measured(self):
        self._feed(count=4, drift=0.002, both=True, speed=0.01)
        m = self.judge.window_measurements()
        self.assertAlmostEqual(m["max_shift_m"], 0.006, places=6)  # three steps of 2 mm
        self.assertAlmostEqual(m["max_speed_mps"], 0.01, places=6)

    def test_single_jaw_steps_are_counted_as_a_gap(self):
        self._feed(count=2, drift=0.0, both=True, speed=0.0)
        self._feed(count=1, drift=0.0, both=False, speed=0.0)
        self._feed(count=1, drift=0.0, both=True, speed=0.0)
        m = self.judge.window_measurements()
        self.assertEqual(m["longest_single_jaw_gap_steps"], 1)
        self.assertAlmostEqual(m["both_jaw_fraction"], 0.75, places=6)

    def test_the_trace_keeps_ineligible_steps_so_slips_are_visible(self):
        self._feed(count=2, drift=0.0, both=True, speed=0.0)
        self._feed(count=2, drift=0.0, both=True, speed=0.0, lifted=False)
        self.assertEqual(len(self.judge.trace), 4)
        self.assertEqual([e["eligible"] for e in self.judge.trace], [True, True, False, False])

    def test_the_trace_is_off_by_default(self):
        quiet = PickJudge("fork", "spoon", self.rules, steps(), (0.0, 0.0), (1.0, 1.0))
        self.assertEqual(quiet.trace, [])

    def test_a_failed_hold_still_reports_its_best_window(self):
        self._feed(count=4, drift=0.05, both=True, speed=0.5)  # far too much movement
        outcome = self.judge.finish("deadline", physics_step=100, control_step=25)
        self.assertFalse(outcome.success)
        self.assertIsNotNone(outcome.best_window_measurements)
        self.assertGreater(outcome.best_window_measurements["max_shift_m"], 0.0)

    def _feed(self, *, count, drift, both, speed, lifted=True):
        """Append `count` physics steps; see tests/test_pick_outcome.py for the fact helper."""
        raise NotImplementedError("copy the SubstepFacts helper from tests/test_pick_outcome.py")


if __name__ == "__main__":
    unittest.main()
```

Replace `_feed` with a real helper built on the existing one in `tests/test_pick_outcome.py`: it must set `positions`, `rotations`, `linear_speed`, `angular_speed`, `gripper_pos`, `gripper_rot` and `verdicts` so that the named utensil is lifted above `lift_height_m` (unless `lifted=False`), is touched by one or both jaws, and drifts by `drift` metres per step.

- [ ] **Step 2: Run the test to verify it fails**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_hold_trace.py"`
Expected: FAIL — `PickJudge() got an unexpected keyword argument 'keep_trace'`.

- [ ] **Step 3: Add the measurements to `pick_outcome.py`**

Additive changes only:

```python
# in PickJudge.__init__, after the existing assignments
        self.keep_trace = keep_trace
        self.trace = []
        self.best_window = None          # measurements of the longest full window seen
        self.hold_window = None          # measurements of the window that completed the hold
```

```python
    def window_measurements(self) -> dict | None:
        """The numbers _window_ok() compares against the thresholds, reported rather than discarded."""
        w = list(self._window)
        if len(w) < self.steps.hold:
            return None
        both = [entry[0] for entry in w]
        gap = longest = 0
        for b in both:
            gap = 0 if b else gap + 1
            longest = max(longest, gap)
        p0, r0 = w[0][1], w[0][2]
        final = w[-self.steps.final_speed:]
        return {"both_jaw_fraction": sum(both) / len(w),
                "longest_single_jaw_gap_steps": longest,
                "max_shift_m": max(float(np.linalg.norm(p - p0)) for _, p, _, _, _ in w),
                "max_turn_deg": max(_rotation_deg(r0, rot) for _, _, rot, _, _ in w),
                "max_speed_mps": max(float(v) for *_, v, _ in final),
                "max_angular_speed_rps": max(float(spin) for *_, spin in final)}
```

In `update()`, immediately after the existing `self.longest_streak = max(...)` line, record the measurements and the trace. The eligibility condition is the one the existing code already computes — reuse it, do not restate it:

```python
        measured = self.window_measurements()
        if measured is not None and (self.best_window is None
                                     or measured["max_shift_m"] > self.best_window["max_shift_m"]):
            self.best_window = measured
        if self.keep_trace:
            rel = self._window[-1][1] if self._window else None
            self.trace.append({"physics_step": f.physics_step, "control_step": f.control_step,
                               "eligible": bool(self._window),
                               "both_jaws": bool(self._window[-1][0]) if self._window else False,
                               "rel_pos": [float(x) for x in rel] if rel is not None else None,
                               "speed_mps": float(f.linear_speed[self.named]),
                               "angular_speed_rps": float(f.angular_speed[self.named]),
                               "lift_m": float(f.positions[self.named][2] - self.start[self.named][0][2])})
```

and where the hold completes (`if self.hold_ok_at is None: self.hold_ok_at = ...`), also store `self.hold_window = measured`.

Add three fields to `PickOutcome` and fill them in `finish()`. `pick_outcome.py` currently imports only `dataclass` (`from dataclasses import dataclass`), so widen that import to `from dataclasses import dataclass, field` — the fields need a default so the existing constructor calls in Plan 1's tests keep working:

```python
    hold_measurements: dict | None = None
    best_window_measurements: dict | None = None
    hold_trace: list = field(default_factory=list)
```

Check while editing: `PickOutcome` is built with positional arguments in `finish()`. New fields with defaults must come **after** every field that has no default, or Python raises `TypeError: non-default argument follows default argument` at import time. If the dataclass already ends with defaulted fields, keep the new ones last.

`hold_measurements` is `self.hold_window` when the hold completed, otherwise `self.best_window`. `best_window_measurements` is always `self.best_window`. `hold_trace` is `self.trace`.

**Note on `best_window`:** the rule above keeps the *most-moved* full window, which is the one that shows how bad a near-miss was. If Task 5 finds that a different window is more useful for calibration (for example the longest-lasting one), change it there with a test, and say so in the report.

- [ ] **Step 4: Run the new tests and the whole suite**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_hold_trace.py"`
Expected: 7 tests `ok`.

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests`
Expected: `OK`, 216 + new tests. **Every Plan 1 test must still pass unchanged** — this task adds reporting and must not alter a single rule decision. If a Plan 1 judge test fails, the change was not additive: revert and redo it.

- [ ] **Step 5: Commit**

```powershell
git add src/rescuehandsai/pick_outcome.py tests/test_hold_trace.py
git commit -m "Pick plan 2 task 3: the judge reports its hold measurements and an optional trace"
```

---

### Task 4: Force measurements from a real, speed-measured descent

**Files:**
- Create: `scripts/measure_forces.py`
- Create: `tests/test_force_limits.py`
- Generated: `results/measurements/forces.json`

**Interfaces:**
- Consumes: `MujocoSimulation`, `ContactClassifier`, `load_contacts`, `IKSolver` (`kinematics.py:26`), `BimanualAction`.
- Produces: `descend(sim, classifier, *, target_speed_m_per_s, press_depth_m) -> dict` with `forces` (one entry per physics step: `{"step", "z_m", "speed_m_per_s", "force_n"}`), `measured_speed_m_per_s` (median vertical speed of the gripper site while descending in contact), `contact_started_step`, `max_force_n`; and `measure_forces(sim, speeds, depths) -> dict` with `conditions`, `summary` (per condition: `samples`, `max_n`, `p95_n`, `mean_n`, `measured_speed_m_per_s`) and `runtime`.

**What the first version of this plan got wrong, and must not be repeated:** it computed `speed_m_per_s * control_dt` and subtracted that number from *joint angles*, which are radians. The result was not a descent at a known speed — it was an arbitrary joint step. The speed must be **commanded through the gripper's cartesian height and then verified from what the gripper actually did**, and if the achieved speed is far from the requested one, the achieved speed is what gets reported.

**Read before writing:** `src/rescuehandsai/expert.py:73-105` (`_solve`, `_pose`, `_grasp_site`) and `src/rescuehandsai/kinematics.py:26-56`. Use the same solver call the expert uses to turn a cartesian gripper target into joint targets. Do not invent a new IK interface.

- [ ] **Step 1: Write the failing test**

Create `tests/test_force_limits.py`:

```python
import json
import unittest
from pathlib import Path

from rescuehandsai.pick_config import load_contacts

REPORT = Path(__file__).resolve().parents[1] / "results" / "measurements" / "forces.json"


class ForceReportTests(unittest.TestCase):
    """The evidence must be usable before any limit is chosen from it."""

    def setUp(self):
        if not REPORT.is_file():
            self.skipTest("run scripts/measure_forces.py first (Task 4 step 3)")
        self.report = json.loads(REPORT.read_text(encoding="utf-8"))

    def test_every_condition_actually_touched_the_table(self):
        for name, s in self.report["summary"].items():
            with self.subTest(condition=name):
                self.assertGreater(s["samples"], 0, "no contact was ever recorded")

    def test_every_condition_reports_the_speed_it_really_achieved(self):
        for name, s in self.report["summary"].items():
            with self.subTest(condition=name):
                self.assertIsNotNone(s["measured_speed_m_per_s"])

    def test_pressing_harder_produces_more_force_than_resting(self):
        gentle = self.report["summary"]["gentle"]["max_n"]
        pressed = [v["max_n"] for k, v in self.report["summary"].items() if k != "gentle"]
        self.assertTrue(pressed, "no pressed conditions were measured")
        self.assertGreater(min(pressed), gentle,
                           "pressing did not register more force than resting; the descent is wrong")


class ForceLimitTests(unittest.TestCase):
    """These stay failing until Task 6 freezes the limits after the owner's review."""

    def setUp(self):
        self.contacts = load_contacts()
        if not self.contacts["force_limits_frozen"]:
            self.skipTest("limits are frozen in Task 6, after the owner reviews the evidence")

    def test_limits_are_set(self):
        self.assertIsNotNone(self.contacts["jaw_table_force_limit_n"])
        self.assertIsNotNone(self.contacts["severe_force_limit_n"])

    def test_severe_limit_is_above_the_ordinary_limit(self):
        self.assertGreater(self.contacts["severe_force_limit_n"],
                           self.contacts["jaw_table_force_limit_n"])

    def test_the_limit_sits_between_gentle_contact_and_hard_presses(self):
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        limit = self.contacts["jaw_table_force_limit_n"]
        gentle_max = report["summary"]["gentle"]["max_n"]
        pressed_min = min(v["max_n"] for k, v in report["summary"].items() if k != "gentle")
        self.assertGreater(limit, gentle_max, "the limit would flag a gentle rest as excess force")
        self.assertLess(limit, pressed_min, "the limit would let every deliberate press through")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it skips**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_force_limits.py"`
Expected: all tests `skipped` (no report yet, limits not frozen).

- [ ] **Step 3: Write `scripts/measure_forces.py`**

Requirements, all of which the reviewer's findings depend on:

1. **Cartesian descent.** Choose a start pose above bare table with the gripper pointing down, then lower the gripper *site* by a fixed height increment per control step: `dz = target_speed_m_per_s * sim.config["control_dt"]`. Convert each cartesian target into joint targets with the expert's solver call. `dz` is metres here because it is applied to a position, never to a joint angle.
2. **Measure the achieved speed.** Sample the gripper site height (`sim.data.site_xpos[site_id][2]`) every physics step and report the median of `-(dz_actual / physics_dt)` over the steps between first contact and the end of the descent. Report this measured speed next to the requested one; if they differ by more than 20%, say so in the report and use the measured value in the spec.
3. **Sample forces at physics rate.** Pass an `on_substep` callback to `sim.step(...)` (`sim.py:198-226`) that appends every jaw-to-table contact force at that physics step. Sampling only once per control step would miss the peak, which is exactly the number the limit is about. The callback must return a falsy value so the control step is not cut short.
4. **Rebuild the classifier after every reset.** `ContactClassifier` holds geom ids from one model; construct it *after* `sim.reset(...)`, exactly as `PickEpisodeRunner` does (`pick_runner.py:63-68`).
5. **Conditions.** `gentle`: descend slowly and stop at first contact, then hold for at least one second of simulated time. `pressed_<speed>`: keep descending `press_depth_m` past first contact at each of three requested speeds. Record the depth as well as the speed — a slow deep press and a fast shallow one are different experiments.
6. **Identify jaw-to-table contacts through `ContactVerdict`**: `verdict.jaw is not None` and `"table"` in `verdict.geoms`. The field is `geoms`, not `names`.

- [ ] **Step 4: Run the measurement**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe scripts/measure_forces.py`
Expected: per condition, the requested speed, the measured speed, sample count, mean, p95 and maximum force.

**Stop and report** if any of these is true — each one means the measurement is wrong, and none of them is fixed by adjusting a limit:
- a condition records zero contact samples (the descent never reached the table);
- the measured speed differs from the requested speed by more than 20% (the descent is not doing what the name says);
- pressing does not produce more force than resting (the press is not pressing);
- gentle and pressed force ranges overlap (no honest limit exists between them).

- [ ] **Step 5: Run the report tests**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_force_limits.py"`
Expected: the three `ForceReportTests` pass; `ForceLimitTests` still skip (limits are frozen in Task 6).

- [ ] **Step 6: Commit the evidence**

```powershell
git add scripts/measure_forces.py tests/test_force_limits.py
git add -f results/measurements/forces.json
git commit -m "Pick plan 2 task 4: jaw-table forces from a speed-measured cartesian descent"
```

- [ ] **Step 7: OWNER CHECKPOINT — stop here**

Report to the owner, in simple English:
- the grasp-shape evidence from Task 2 (which shapes really hold the utensil; anything unlisted);
- the force table: requested speed, achieved speed, depth, sample count, mean, p95 and maximum force per condition;
- the gap between gentle and pressed, and the limits you would propose from it, with the reason;
- anything that triggered a "stop and report" and what it turned out to be.

**Do not start Task 5.** Tasks 5 and 6 freeze thresholds that every later number depends on, and the owner reviews this evidence first.

---

### Task 5: Hold thresholds chosen from labelled evidence, not from the rules themselves

**Files:**
- Create: `scripts/calibrate_hold.py`
- Create: `tests/test_hold_calibration.py`
- Generated: `results/measurements/hold_traces.json`, `results/measurements/hold_separation.md`

**Interfaces:**
- Consumes: `PickTeacher`, `PickEpisodeRunner`, `PickJudge(keep_trace=True)` (Task 3), `load_rules`, `make_pick_task`.
- Produces:
  - `collect_traces(sim, scenes, *, rules) -> list` — one entry per episode: `seed`, `cell`, `kind` (`"teacher"`, `"slip"`, `"swing"`), `trace`, `outcome`.
  - `label_window(window: list, *, dt: float) -> str` — `"stable"`, `"moving"` or `"lost"`, decided **from the trace alone**, with no reference to any threshold in `pick_rules.json`.
  - `separation(labelled: dict) -> dict` — per metric: `stable_max`, `bad_min`, `overlap` (bool), `suggested` (the midpoint of a clean gap, or `None` when the ranges overlap).

**Why the previous version was wrong, in plain terms:** it switched the rules off, ran the teacher, called every episode that passed a "good hold", and then set the thresholds from those episodes. That is circular — the thresholds were fitted to whatever the teacher happened to do, so they could never catch the teacher doing something bad. The rule `2 × p95` is dropped entirely.

**What replaces it:** record fixed-length windows from three kinds of episode, label each window by what physically happened (not by any rule), and choose each threshold so it separates `stable` from `moving`/`lost`. A threshold with no gap is reported as "no separation" and the owner decides — it is never split down the middle and presented as measured.

Independent labels, using only the trace:
- **`lost`** — the utensil's height above its start drops below half `lift_height_m` at any point in the window, or the jaws stop touching it entirely. It fell or was dropped.
- **`moving`** — not lost, but the utensil's position in the gripper frame moves more than 3 mm across the window, or it turns more than 10 degrees. It is sliding or swinging in the hand.
- **`stable`** — neither. It stays put in the hand.

These three numbers (half the lift height, 3 mm, 10 degrees) are **definitions of the physical situation**, fixed before any data is collected, and they are written into the spec as such. They are not the thresholds being calibrated; the thresholds are chosen afterwards so that they separate these labels.

Negative examples must be produced deliberately, because the teacher rarely fails on purpose:
- **`slip`** — run the teacher, then at a fixed control step after the lift, command the gripper open by a set amount so the utensil slides or drops;
- **`swing`** — run the teacher, then command a fast lateral wrist motion so the utensil swings while still gripped.

Both are produced by a small wrapper around `PickTeacher` that edits the command after `finished_pick_at`; neither touches `expert.py`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_hold_calibration.py`. The labelling and separation are pure functions and are tested on synthetic windows, with no simulation:

```python
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("calibrate_hold", ROOT / "scripts" / "calibrate_hold.py")
calibrate_hold = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(calibrate_hold)


def window(*, drift=0.0, turn=0.0, lift=0.06, both=True, steps=20):
    return [{"rel_pos": [drift * i, 0.0, 0.0], "rel_turn_deg": turn * i, "lift_m": lift,
             "both_jaws": both, "speed_mps": 0.0, "angular_speed_rps": 0.0, "eligible": True}
            for i in range(steps)]


class LabelTests(unittest.TestCase):
    """Labels describe what physically happened and never consult pick_rules.json."""

    def test_a_steady_window_is_stable(self):
        self.assertEqual(calibrate_hold.label_window(window(), dt=0.05), "stable")

    def test_a_dropped_utensil_is_lost(self):
        self.assertEqual(calibrate_hold.label_window(window(lift=0.01), dt=0.05), "lost")

    def test_an_ungripped_utensil_is_lost(self):
        w = window()
        for entry in w[5:]:
            entry["both_jaws"] = False
            entry["eligible"] = False
        self.assertEqual(calibrate_hold.label_window(w, dt=0.05), "lost")

    def test_a_sliding_utensil_is_moving(self):
        self.assertEqual(calibrate_hold.label_window(window(drift=0.001), dt=0.05), "moving")

    def test_a_turning_utensil_is_moving(self):
        self.assertEqual(calibrate_hold.label_window(window(turn=1.0), dt=0.05), "moving")


class SeparationTests(unittest.TestCase):
    def test_a_clean_gap_suggests_the_midpoint(self):
        result = calibrate_hold.separation({"max_shift_m": {"stable": [0.001, 0.002],
                                                            "moving": [0.010, 0.012],
                                                            "lost": [0.030]}})
        self.assertFalse(result["max_shift_m"]["overlap"])
        self.assertAlmostEqual(result["max_shift_m"]["suggested"], 0.006, places=6)

    def test_overlapping_ranges_suggest_nothing(self):
        result = calibrate_hold.separation({"max_shift_m": {"stable": [0.001, 0.020],
                                                            "moving": [0.010, 0.012]}})
        self.assertTrue(result["max_shift_m"]["overlap"])
        self.assertIsNone(result["max_shift_m"]["suggested"])

    def test_a_metric_with_no_bad_examples_suggests_nothing(self):
        # Without negative examples there is nothing to separate from.
        result = calibrate_hold.separation({"max_shift_m": {"stable": [0.001, 0.002]}})
        self.assertIsNone(result["max_shift_m"]["suggested"])
        self.assertIn("no bad examples", result["max_shift_m"]["note"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_hold_calibration.py"`
Expected: FAIL — `scripts/calibrate_hold.py` does not exist.

- [ ] **Step 3: Write `scripts/calibrate_hold.py`**

Requirements:

1. **Scenes:** use dev seeds **3,100,100 upward** — deliberately *not* the gate's 25 scenes. Thresholds calibrated on the same scenes the gate scores would flatter the gate. Record the exact seeds used.
2. **Three kinds of episode** per scene: plain `teacher`, `slip`, `swing`, produced by the wrappers described above.
3. **Traces:** run with `PickJudge(keep_trace=True)`, using rules that are loose enough not to end the episode early, and record the whole trace. State in the report that the loose rules affect only when the episode *stops*, never how a window is *labelled*.
4. **Windows:** cut each trace into consecutive windows of `steps.hold` physics steps, discard windows in which the utensil was never lifted, label each with `label_window`, and compute the same metrics `PickJudge.window_measurements()` reports, so the calibration numbers and the rule numbers are the same quantities.
5. **Separation:** for each metric, collect the values of `stable` windows and of `moving`/`lost` windows and report `stable_max`, `bad_min`, `overlap` and `suggested`.
6. **Output:** `hold_traces.json` with everything, and `hold_separation.md` — a short table a human can read: metric, stable range, bad range, gap, suggestion.

- [ ] **Step 4: Run the calibration**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe scripts/calibrate_hold.py`
Expected: the separation table, with counts of stable, moving and lost windows.

**Stop and report** if any of these is true:
- fewer than 20 `stable` windows, or fewer than 10 `moving`/`lost` windows — there is not enough evidence to separate anything, and the `slip`/`swing` wrappers need to work harder;
- any metric's ranges overlap — report the overlap; the owner decides whether to keep the current value, change the definition, or accept a threshold that mislabels some windows;
- the teacher's own plain episodes produce `lost` windows — that is a teacher problem and must be reported before any threshold is set.

- [ ] **Step 5: Propose the thresholds (do not freeze them yet)**

Write the proposal into the task report: for each of `hold_max_shift_m`, `hold_max_turn_deg`, `final_max_speed_mps`, `final_max_angular_speed_rps`, `both_jaw_fraction` and `max_single_jaw_gap_s` — the stable range, the bad range, and the value you propose with its reason. Sanity ceilings still apply: shift below `lift_height_m`, turn at most 45 degrees.

Freezing happens in Task 6, in one commit, together with the deadline.

- [ ] **Step 6: Commit the evidence**

```powershell
git add scripts/calibrate_hold.py tests/test_hold_calibration.py
git add -f results/measurements/hold_traces.json results/measurements/hold_separation.md
git commit -m "Pick plan 2 task 5: hold traces, independent labels and threshold separation"
```

---

### Task 6: Freeze the thresholds, then run the teacher gate

**Files:**
- Create: `scripts/teacher_gate.py`
- Create: `tests/test_teacher_gate.py`
- Modify: `configs/pick_rules.json` (hold thresholds, `deadline_control_steps`, `frozen: true`)
- Modify: `configs/pick_contacts.json` (force limits, `force_limits_frozen: true`)
- Modify: `docs/superpowers/specs/2026-09-17-pick-lift-selection-design.md` (§5, §7)
- Generated: `results/teacher_gate/`

**Interfaces:**
- Consumes: `PickTeacher`, `PickEpisodeRunner`, `AttemptLedger`, `write_episode`, `episode_key`, `episode_filename`, `InvalidRun`, `check_start` (`pick_cells.py:103`).
- Produces: `judge_gate(results) -> dict`, `suggested_deadline(verdict) -> int`, `run_gate(sim, seeds, cells, rules, run_dir) -> dict`.

Gate rule (§7), exactly: 25 dev scenes × 4 cells = 100 episodes. Pass if **≥ 99/100 overall and each cell ≥ 24/25**. Also reported: scenes passing all four cells, failed teacher attempts, rejected starts, and the p95 and maximum **time to complete the hold**.

**Two corrections carried from the review:**
- The deadline comes from `hold_completed_control_step` — the step at which the one-second hold *finished*. The earlier draft used the step at which the hold *started*, which would have set a deadline that the rule itself cannot meet.
- `StartCheck` exposes **`valid`** and **`reasons`** (a tuple), not `ok` and `reason`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_teacher_gate.py`. The gate rule is arithmetic and is tested without simulation:

```python
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("teacher_gate", ROOT / "scripts" / "teacher_gate.py")
teacher_gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(teacher_gate)


class GateArithmeticTests(unittest.TestCase):
    def _results(self, failures=(), count=None):
        cells = ["F-A", "F-B", "S-A", "S-B"]
        out = [{"seed": seed, "cell": cell, "success": (seed, cell) not in failures,
                "hold_completed_control_step": 100}
               for seed in range(3100000, 3100025) for cell in cells]
        return out if count is None else out[:count]

    def test_a_perfect_run_passes(self):
        verdict = teacher_gate.judge_gate(self._results())
        self.assertTrue(verdict["gate_passed"], verdict["reason"])
        self.assertEqual(verdict["episodes"], 100)
        self.assertEqual(verdict["scenes_all_four"], 25)

    def test_one_failure_still_passes_at_99_of_100(self):
        verdict = teacher_gate.judge_gate(self._results(failures={(3100000, "F-A")}))
        self.assertTrue(verdict["gate_passed"], verdict["reason"])
        self.assertEqual(verdict["scenes_all_four"], 24)

    def test_two_failures_fail_the_overall_bar(self):
        verdict = teacher_gate.judge_gate(self._results(failures={(3100000, "F-A"), (3100001, "S-B")}))
        self.assertFalse(verdict["gate_passed"])
        self.assertIn("overall", verdict["reason"])

    def test_two_failures_in_one_cell_fail_that_cell(self):
        verdict = teacher_gate.judge_gate(self._results(failures={(3100000, "F-A"), (3100001, "F-A")}))
        self.assertFalse(verdict["gate_passed"])
        self.assertEqual(verdict["per_cell"]["F-A"], 23)
        self.assertIn("cell F-A", verdict["reason"])

    def test_an_incomplete_run_never_passes(self):
        verdict = teacher_gate.judge_gate(self._results(count=99))
        self.assertFalse(verdict["gate_passed"])
        self.assertIn("incomplete", verdict["reason"])

    def test_the_deadline_uses_hold_completion_and_the_maximum(self):
        results = self._results()
        results[0]["hold_completed_control_step"] = 250
        verdict = teacher_gate.judge_gate(results)
        self.assertEqual(verdict["max_steps"], 250)
        self.assertGreaterEqual(teacher_gate.suggested_deadline(verdict), 375)

    def test_an_episode_without_a_completed_hold_contributes_no_deadline_evidence(self):
        results = self._results()
        results[0]["hold_completed_control_step"] = None
        verdict = teacher_gate.judge_gate(results)
        self.assertEqual(len(verdict["hold_completed_control_steps"]), 99)

    def test_a_deadline_cannot_be_derived_without_completed_holds(self):
        with self.assertRaises(ValueError):
            teacher_gate.suggested_deadline(teacher_gate.judge_gate([]))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_teacher_gate.py"`
Expected: FAIL — `scripts/teacher_gate.py` does not exist.

- [ ] **Step 3: Write `scripts/teacher_gate.py`**

`judge_gate(results)` applies the rule above and returns `episodes`, `successes`, `per_cell`, `scenes_all_four`, `hold_completed_control_steps` (only the values that are not `None`), `p95_steps`, `max_steps`, `gate_passed`, `reason`.

`suggested_deadline(verdict)` returns `ceil(max_steps * 1.5)` and raises `ValueError` when there are no completed holds. It uses the maximum, not the median: a slow success is still a success, and the deadline must not become a second bar on top of the five rules.

`run_gate(...)` walks the scenes and for each cell:
- calls `check_start(seed, cell, physics_version=..., rules=...)` and skips when **`not start.valid`**, recording `{"seed", "cell", "reasons": list(start.reasons)}`;
- takes the attempt number from `ledger.next_attempt(key)`;
- runs `PickEpisodeRunner(sim, PickTeacher(), rules=rules).run(task)`;
- on `InvalidRun`, writes `invalid.partial` with `write_episode` and records the ledger line with the invalid label;
- on success or valid failure, writes the record and records a valid ledger line, then collects `{"seed", "cell", "success", "hold_completed_control_step": record["outcome"]["hold_completed_control_step"], "picked", "failures"}`.

The episode file is always written **before** the ledger line, so an interrupted run never has a ledger entry without its evidence.

- [ ] **Step 4: Run the gate arithmetic tests**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_teacher_gate.py"`
Expected: 8 tests `ok`.

- [ ] **Step 5: Freeze the thresholds the owner approved**

Only now, and in one commit:
- `configs/pick_contacts.json`: the two force limits from Task 4's evidence; `force_limits_frozen: true`;
- `configs/pick_rules.json`: the hold thresholds from Task 5's separation table.

Leave `deadline_control_steps` alone for the moment and leave `"frozen": false` until step 7.

Write into the spec, with dates: the force numbers and the gap they sit in; the hold separation table; the physical definitions used for labelling (half the lift height, 3 mm, 10 degrees) and that they are definitions, not calibrated values.

- [ ] **Step 6: Run the gate with a provisional deadline**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe scripts/teacher_gate.py --run-dir results/teacher_gate_provisional`

The current `deadline_control_steps` of 300 stays in place for this run; its only job is to produce hold-completion times.

**If the gate fails, the teacher is fixed — never the bar, never the force limit, never a hold threshold** (§5, §7). Report which cells failed, the failure labels, and whether failures cluster on one slot. The Sep 17 audit found the old teacher was much weaker at slot 0 (clean yield ~0.57–0.64 there versus ~0.82–0.93 at slot 1, mostly "utensil not in place zone"). If that pattern reappears, name it; do not average it away.

- [ ] **Step 7: Set the deadline and freeze the rules**

Put `suggested_deadline(...)` into `deadline_control_steps`, then set `"frozen": true`. Record in §7: the p95 and maximum hold-completion step, the chosen deadline, and the reason.

- [ ] **Step 8: Re-run the gate under the final frozen settings**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe scripts/teacher_gate.py --run-dir results/teacher_gate`

**This run is the gate result.** The provisional run was only evidence for the deadline; the reported verdict must come from a run in which every frozen threshold was already in place. If the two verdicts differ, report both and explain which threshold changed the outcome.

- [ ] **Step 9: Run the whole suite and commit**

```powershell
git add scripts/teacher_gate.py tests/test_teacher_gate.py configs/pick_rules.json configs/pick_contacts.json docs/superpowers/specs/2026-09-17-pick-lift-selection-design.md
git add -f results/teacher_gate results/teacher_gate_provisional
git commit -m "Pick plan 2 task 6: thresholds frozen from evidence; teacher gate run under final settings"
```

---

### Task 7: Frozen scene lists

**Files:**
- Create: `scripts/freeze_scene_lists.py`
- Create: `tests/test_scene_lists.py`
- Generated: `configs/pick_scene_lists/{dev_gate,dev_calibration,dev_quick,dev_selection,test_main,test_wording,index}.json`

**Interfaces:**
- Consumes: `check_start` (`pick_cells.py:103`), `settings_hash`, `find_duplicates` (`pick_identity.py:53`), `load_seed_blocks`, `block_of` (`pick_identity.py:81`), `sample_params`.
- Produces: `list_digest(path) -> str`; `build_list(name, block, count, *, physics_version, rules, check=check_start, settings=None, start_at=None) -> dict` with `name`, `seeds`, `count`, `first_seed`, `next_free_seed`, `rejected` (each `{"seed", "reasons"}`), `settings_sha256`, `physics_version`, `cells`, `created_utc`; `freeze(lists, out_dir) -> dict` writing each list plus `index.json` holding every list's SHA-256.

Sizes (§8): dev gate 25, dev calibration 10, dev quick 10, dev selection 50, main test 100, wording test 24. The calibration list records the scenes Task 5 used, so it is visible that they are **not** the gate scenes.

**Interface note:** `StartCheck` has **`valid`** and **`reasons`** (a tuple). A seed is accepted only when all four cells are valid.

- [ ] **Step 1: Write the failing test**

Create `tests/test_scene_lists.py` with two groups.

`ListBuildingTests` (no physics, using a stub check):

```python
import importlib.util
import json
import unittest
from pathlib import Path

from rescuehandsai.pick_identity import block_of, load_seed_blocks

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("freeze_scene_lists",
                                               ROOT / "scripts" / "freeze_scene_lists.py")
freeze_scene_lists = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(freeze_scene_lists)

LIST_DIR = ROOT / "configs" / "pick_scene_lists"
EXPECTED = {"dev_gate": 25, "dev_calibration": 10, "dev_quick": 10, "dev_selection": 50,
            "test_main": 100, "test_wording": 24}


def stub(ok_when):
    def check(seed, cell, **kwargs):
        ok = ok_when(seed)
        return type("S", (), {"cell": cell, "valid": ok,
                              "reasons": () if ok else ("utensil_overlap",)})()
    return check


class ListBuildingTests(unittest.TestCase):
    def test_rejected_seeds_are_skipped_and_their_reasons_kept(self):
        built = freeze_scene_lists.build_list("dev_gate", (3100000, 3200000), 4,
                                              physics_version=2, rules={},
                                              check=stub(lambda s: s % 3 != 0),
                                              settings=lambda s: f"h{s}")
        self.assertEqual(len(built["seeds"]), 4)
        self.assertTrue(all(s % 3 != 0 for s in built["seeds"]))
        self.assertTrue(built["rejected"])
        self.assertEqual(built["rejected"][0]["reasons"], ["utensil_overlap"])

    def test_a_seed_is_rejected_when_any_one_cell_is_invalid(self):
        seen = {"cells": []}

        def check(seed, cell, **kwargs):
            seen["cells"].append(cell)
            ok = not (seed == 3100000 and cell == "S-B")
            return type("S", (), {"cell": cell, "valid": ok,
                                  "reasons": () if ok else ("moving",)})()

        built = freeze_scene_lists.build_list("dev_gate", (3100000, 3200000), 1,
                                              physics_version=2, rules={}, check=check,
                                              settings=lambda s: f"h{s}")
        self.assertNotIn(3100000, built["seeds"])

    def test_building_stops_rather_than_leaving_its_block(self):
        with self.assertRaises(RuntimeError):
            freeze_scene_lists.build_list("dev_gate", (3100000, 3100010), 5,
                                          physics_version=2, rules={},
                                          check=stub(lambda s: False), settings=lambda s: f"h{s}")
```

`FrozenListTests` (skipped until the lists exist) must check: every list has the size the spec asks for; no seed appears in two lists; no two scenes share a settings hash; every seed sits in the block its list belongs to (`block_of`); `index.json`'s hash matches each stored file; `rejected` is present in every list; and **`dev_calibration` and `dev_gate` share no seed**.

- [ ] **Step 2: Run the test to verify it fails**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_scene_lists.py"`
Expected: FAIL — `scripts/freeze_scene_lists.py` does not exist.

- [ ] **Step 3: Write `scripts/freeze_scene_lists.py`**

Requirements:

1. `build_list` takes the next seeds in the block whose start is valid **in all four cells**, records every rejection with its reasons, and raises `RuntimeError` rather than running past the end of the block.
2. Lists are built in this order from the dev block, each continuing from the previous list's `next_free_seed`: `dev_gate`, `dev_calibration`, `dev_quick`, `dev_selection`. `test_main` and `test_wording` come from their own blocks.
3. `list_digest` hashes with CRLF normalised to LF (`core.autocrlf` is on here, off on Kaggle).
4. `freeze` refuses to overwrite an existing `index.json`; refreezing is a deliberate act with an explicit deletion first.
5. The script prints, for every list, the count, first seed, number of rejections and the short hash, and ends with a plain reminder that the final-test lists are never run during development.

- [ ] **Step 4: Freeze the lists**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe scripts/freeze_scene_lists.py`

**Stop and report** if a list rejects more than 20% of the seeds it examined: the start rules and the scene sampler disagree, and every list would then be biased towards easy scenes. Report the rejection reasons grouped by reason.

- [ ] **Step 5: Reconcile the gate with the frozen list**

The gate in Task 6 ran from seed 3,100,000 upward, skipping invalid starts as it went. The frozen `dev_gate` list is built the same way, so the two should contain the same 25 seeds. **Check it, do not assume it:** compare the gate's scored seeds with `dev_gate.json`. If they differ, re-run the gate against the frozen list and use that as the gate result. Record both seed sets in the report.

- [ ] **Step 6: Duplicate check across everything**

Run the list tests, then check the new lists against the legacy ranges in `configs/pick_seed_blocks.json` with `find_duplicates`. Write the outcome as "no overlap with the ranges we have recorded" — some old laptop shards have incomplete provenance, so a stronger claim would be false (§8).

- [ ] **Step 7: Commit**

```powershell
git add scripts/freeze_scene_lists.py tests/test_scene_lists.py configs/pick_scene_lists
git commit -m "Pick plan 2 task 7: frozen scene lists with hashes, rejection records and a separate calibration list"
```

---

### Task 8: The evaluation CLI, with a guard that compares hashes

**Files:**
- Create: `src/rescuehandsai/pick_run.py`
- Create: `scripts/evaluate_pick.py`
- Create: `tests/test_pick_run.py`

**Interfaces:**
- Consumes: `AttemptLedger`, `check_resume`, `check_run_contract`, `ResumeRefused`, `InvalidRun`, `write_episode`, `episode_key`, `episode_filename`; `snapshot_record` (`snapshot.py:135`); `summarize`, `MAIN_TEST_BARS`, `INCOMPLETE` (`pick_stats.py`); `PickEpisodeRunner`; `PickTeacher`; `make_pick_task`.
- Produces:
  - `RunPlan(name, seeds, cells, templates, list_name, list_sha256, is_final)` — frozen dataclass.
  - `planned_keys(plan) -> list[str]`.
  - `guard_final(plan, *, final_flag, manifest, expected) -> None` — raises `FinalRunRefused`.
  - `build_manifest(plan, *, sim, rules, contacts, policy_meta, snapshot, args) -> dict`.
  - `run_episodes(sim, plan, policy_factory, *, rules, contacts, run_dir, ledger) -> dict`.
  - `write_summary(run_dir, plan, ledger, *, bars=None) -> dict`.
  - `FinalRunRefused(RuntimeError)`.

**The guard correction:** the earlier draft only required `spec_sha256` and `model_sha256` to be *present*, so `"x"` would pass. `guard_final` now takes an `expected` mapping — `{"spec_sha256": ..., "model_sha256": ..., "scene_list_sha256": ...}` computed from the real spec file, the real model file and `configs/pick_scene_lists/index.json` — and every value must **match**. A mismatch names the field that differs.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pick_run.py`. The guard tests, which are the point of this task:

```python
import unittest

from rescuehandsai.pick_run import FinalRunRefused, RunPlan, guard_final, planned_keys

SPEC = "a" * 64
MODEL = "b" * 64
LIST = "c" * 64
EXPECTED = {"spec_sha256": SPEC, "model_sha256": MODEL, "scene_list_sha256": LIST}
GOOD = dict(EXPECTED)


def plan(list_name="dev_gate", is_final=False):
    return RunPlan(name="run", seeds=(3100000, 3100001), cells=("F-A", "F-B", "S-A", "S-B"),
                   templates=("T1",), list_name=list_name, list_sha256=LIST, is_final=is_final)


class FinalGuardTests(unittest.TestCase):
    """The final lists are the one thing that must never run by accident or by mistake."""

    def test_a_final_list_without_the_flag_is_refused(self):
        with self.assertRaises(FinalRunRefused):
            guard_final(plan("test_main", True), final_flag=False, manifest=GOOD, expected=EXPECTED)

    def test_a_final_list_without_a_manifest_is_refused(self):
        with self.assertRaises(FinalRunRefused):
            guard_final(plan("test_main", True), final_flag=True, manifest=None, expected=EXPECTED)

    def test_a_missing_field_is_named(self):
        manifest = {k: v for k, v in GOOD.items() if k != "model_sha256"}
        with self.assertRaises(FinalRunRefused) as ctx:
            guard_final(plan("test_main", True), final_flag=True, manifest=manifest, expected=EXPECTED)
        self.assertIn("model_sha256", str(ctx.exception))

    def test_a_wrong_model_hash_is_refused(self):
        with self.assertRaises(FinalRunRefused) as ctx:
            guard_final(plan("test_main", True), final_flag=True,
                        manifest=dict(GOOD, model_sha256="d" * 64), expected=EXPECTED)
        self.assertIn("model_sha256", str(ctx.exception))

    def test_a_wrong_spec_hash_is_refused(self):
        with self.assertRaises(FinalRunRefused) as ctx:
            guard_final(plan("test_main", True), final_flag=True,
                        manifest=dict(GOOD, spec_sha256="d" * 64), expected=EXPECTED)
        self.assertIn("spec_sha256", str(ctx.exception))

    def test_a_scene_list_hash_that_does_not_match_the_plan_is_refused(self):
        p = RunPlan(name="run", seeds=(1,), cells=("F-A",), templates=("T1",),
                    list_name="test_main", list_sha256="e" * 64, is_final=True)
        with self.assertRaises(FinalRunRefused) as ctx:
            guard_final(p, final_flag=True, manifest=GOOD, expected=EXPECTED)
        self.assertIn("scene_list_sha256", str(ctx.exception))

    def test_a_matching_manifest_is_allowed(self):
        guard_final(plan("test_main", True), final_flag=True, manifest=GOOD, expected=EXPECTED)

    def test_the_flag_on_a_development_list_is_refused(self):
        # --final is not a way to make a dev run look official.
        with self.assertRaises(FinalRunRefused):
            guard_final(plan(), final_flag=True, manifest=GOOD, expected=EXPECTED)
```

Add summary tests covering: a complete run is `evaluation_valid` and every scored entry names its attempt and file; a missing episode makes the run incomplete, not eligible to pass, and shows the `INCOMPLETE` notice instead of a verdict; a blocked episode makes the run invalid and appears in `blocked_keys`; resume skips episodes that already have a scored valid attempt.

- [ ] **Step 2: Run the test to verify it fails**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_pick_run.py"`
Expected: `ModuleNotFoundError: No module named 'rescuehandsai.pick_run'`.

- [ ] **Step 3: Write `src/rescuehandsai/pick_run.py`**

Rules, all load-bearing (§6, §11):

- `guard_final` refuses when: the plan is final and the flag is absent; the plan is final and the manifest is `None`; any expected field is missing from the manifest, or differs from `expected` (the message names the field); `plan.list_sha256` differs from `expected["scene_list_sha256"]`; or the flag is set for a plan that is not final. It never repairs a manifest.
- `run_episodes` skips an episode when `ledger.scored(key)` is not `None` — that is resume. It takes the attempt number from `ledger.next_attempt(key)` and lets `AlreadyScored`, `RetryNeedsDiagnosis` and `EvaluationBlocked` propagate: the ledger is the authority, and Plan 1's Task 10 fixes exist precisely so this loop cannot score an episode twice.
- The episode file is written **before** the ledger line, always.
- `write_summary` sets `evaluation_valid` true only when every planned key has exactly one scored valid attempt and no key is blocked; otherwise it writes the `INCOMPLETE` notice and applies no pass bars. Every scored entry names `attempt` and `file`.

- [ ] **Step 4: Write `scripts/evaluate_pick.py`**

Flags: `--list` (frozen list name, required), `--policy` (`teacher`; learned policies arrive in Plan 4), `--run-dir`, `--resume`, `--final`, `--final-manifest`, `--video`.

Order of operations, not to be rearranged:

1. load the frozen list and compare its SHA-256 with `configs/pick_scene_lists/index.json`; a mismatch is refused before any episode runs;
2. compute `expected` from the real files (spec, model, list index) and call `guard_final`;
3. `snapshot_record(ROOT, sim.asset_path, strict=plan.is_final, ...)` — a final run refuses to start with any untracked or modified file under `src/`, `scripts/`, `training/`, `configs/` (§11);
4. `check_resume(run_dir, current)` when `--resume`;
5. run the episodes;
6. write the summary, applying `MAIN_TEST_BARS` only for the final main-test list.

- [ ] **Step 5: Prove resume and the refusals on a real run**

```powershell
$env:PYTHONPATH = "src"
.venv-sim/Scripts/python.exe scripts/evaluate_pick.py --list dev_quick --policy teacher --run-dir results/pick_resume_check
```
Interrupt with Ctrl+C after a few episodes, then re-run with `--resume`. Expected: the finished episodes are skipped, the rest are added, and the summary reports one scored attempt per planned episode with `evaluation_valid: true`.

Then prove three refusals and paste the exact messages into the report:
- `--list test_main` without `--final` → `FinalRunRefused`;
- `--list test_main --final` with a manifest holding a wrong model hash → `FinalRunRefused` naming `model_sha256`;
- resuming after temporarily changing a rule → `ResumeRefused`.

Undo the temporary rule change and re-run the suite afterwards. **No final-test episode is ever executed** — these checks stop at the refusal.

- [ ] **Step 6: Run the whole suite and commit**

```powershell
git add src/rescuehandsai/pick_run.py scripts/evaluate_pick.py tests/test_pick_run.py
git commit -m "Pick plan 2 task 8: evaluation CLI with resume and a hash-checked final-list guard"
```

---

### Task 9: Simulator and teacher timing — what it can and cannot tell us

**Files:**
- Create: `scripts/time_simulation.py`
- Create: `tests/test_timing_report.py`
- Generated: `results/measurements/timing_simulation.json`
- Modify: `docs/superpowers/specs/2026-09-17-pick-lift-selection-design.md` (§10)

**Interfaces:**
- Consumes: `PickTeacher`, `PickEpisodeRunner`, `load_rules`, `make_pick_task`, `runtime_versions` (`snapshot.py:124`).
- Produces: `time_episodes(sim, seeds, cells, *, with_video) -> dict` with `episodes` (`seed`, `cell`, `wall_s`, `control_steps`, `observe_s_total`, `success`), `summary` (`mean_wall_s`, `p95_wall_s`, `max_wall_s`), `runtime`; and `project(episodes, count) -> dict` with `hours_mean`, `hours_worst`.

**What this task must not claim.** The seven-hour estimate for the final test is about **400 evaluation episodes driven by SmolVLA through OpenVINO on the laptop's iGPU**. `PickTeacher` returns `False` from `wants_images()`, so no camera is rendered, and it runs no neural network at all. This measurement therefore covers **simulation, contact checking and the teacher's own IK cost** — the floor, not the total. Every printed line, the JSON, the spec entry and the owner report must say so. The learned-policy timing is measured in Plan 4 with the real exported model, and only then does the seven-hour figure get replaced.

- [ ] **Step 1: Write the failing test**

Create `tests/test_timing_report.py`:

```python
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("time_simulation",
                                               ROOT / "scripts" / "time_simulation.py")
time_simulation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(time_simulation)


class ProjectionTests(unittest.TestCase):
    def test_the_projection_uses_the_mean_and_the_worst_case(self):
        episodes = [{"wall_s": 10.0}, {"wall_s": 20.0}, {"wall_s": 30.0}, {"wall_s": 40.0}]
        p = time_simulation.project(episodes, 400)
        self.assertAlmostEqual(p["hours_mean"], 400 * 25.0 / 3600, places=6)
        self.assertAlmostEqual(p["hours_worst"], 400 * 40.0 / 3600, places=6)

    def test_an_empty_run_cannot_be_projected(self):
        with self.assertRaises(ValueError):
            time_simulation.project([], 400)

    def test_the_report_states_what_it_does_not_cover(self):
        note = time_simulation.SCOPE_NOTE.lower()
        self.assertIn("no camera", note)
        self.assertIn("no neural network", note)
        self.assertIn("not a replacement", note)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_timing_report.py"`
Expected: FAIL — `scripts/time_simulation.py` does not exist.

- [ ] **Step 3: Write `scripts/time_simulation.py`**

It must: define `SCOPE_NOTE` saying in one sentence that this run renders no camera, runs no neural network, and is **not a replacement** for the seven-hour OpenVINO estimate; run four complete episodes (one per cell) including reset and the judge's per-step work; measure wall-clock seconds per episode with `time.perf_counter()`; keep the `observe_s` series `PickEpisodeRunner` already records; expose `project(episodes, count)` using the mean and the maximum, raising `ValueError` on an empty list; record `runtime_versions()`; and print `SCOPE_NOTE` first, then the per-episode times and the projections for 400 and 496 episodes.

- [ ] **Step 4: Run the measurement**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe scripts/time_simulation.py`
Expected: the scope note, four episode times, and both projections.

- [ ] **Step 5: Update the spec honestly**

In §10, record the measured simulator-and-teacher cost per episode and the projections, naming the machine (Intel Core i5-6300U, HD Graphics 520) and the date. **Leave the seven-hour estimate in place, marked as unverified**, with a note that it covers rendering and SmolVLA inference which this measurement does not, and that Plan 4 measures it with the real model.

- [ ] **Step 6: Run the whole suite and commit**

```powershell
git add scripts/time_simulation.py tests/test_timing_report.py docs/superpowers/specs/2026-09-17-pick-lift-selection-design.md
git add -f results/measurements/timing_simulation.json
git commit -m "Pick plan 2 task 9: simulator and teacher timing, scoped honestly"
```

- [ ] **Step 7: Report to the owner**

In simple English:
- the gate result: successes out of 100, per cell, scenes passing all four, rejected starts, failed attempts, and whether failures cluster on one slot;
- the frozen thresholds, each with the evidence that separated good from bad;
- the measured deadline with the p95 and maximum hold-completion step;
- the frozen list sizes, the gate/frozen-list seed reconciliation, and the duplicate-check wording;
- the simulator timing **and** a plain statement that the seven-hour estimate is still unverified;
- every "stop and report" that triggered, and what it turned out to be;
- confirmation that no final-test episode ran and that the owner's documents are still uncommitted.

Then state that Plan 3 (data generation, provenance, balance and acceptance-bias reports, pilot gate, full data) is next and is written only after this report.

---

## Self-review against the spec

**Coverage of the spec sections Plan 2 owns:**

| Spec requirement | Task |
|---|---|
| §5 force limit from controlled measurements; severe limit higher; teacher must satisfy it | 4, 6 |
| §5 hold tolerances calibrated against evidence | 3, 5, 6 |
| §5 right-jaw grasp shapes confirmed against real grasps (§13 step 2) | 2 |
| §7 pick-only teacher; IK only inside the teacher | 1 |
| §7 gate of 25 × 4, ≥99/100 and ≥24/25 per cell, plus the extra reported numbers | 6 |
| §7 timeout checked against teacher p95 **and** maximum | 6 |
| §8 seed blocks, duplicate check, frozen lists with hashes, honest legacy note | 7 |
| §8 final lists never run in development; `--final` plus a matching manifest | 7, 8 |
| §11 run directory, manifest, attempts, per-episode files including invalid attempts | 8 |
| §11 summary names the attempt supplying each result; `evaluation_valid`; incomplete notice | 8 |
| §11 snapshot record, strict for final runs; resume only on a matching experiment | 8 |
| §10 laptop timing run | 9 (simulator floor only; the full figure is Plan 4) |

**Deliberately outside Plan 2:** data generation and provenance (Plan 3); training, export, parity and the paired native-versus-OpenVINO comparison, including learned-policy timing (Plan 4); the final test itself (Plan 5).

**Open items that must be settled in the open, never silently:**

1. Task 2's utensil-shape matching depends on how the utensil's own shapes are named in `ContactVerdict.geoms`. The task says to check `load_contacts()["scene_geoms"]` first and to report whichever form was used.
2. Task 3 keeps the *most-moved* full window as `best_window`. If Task 5 finds a different choice more useful, it changes it with a test and reports the change.
3. Task 4 depends on the expert's IK call, which the implementer must read rather than invent, and on the achieved descent speed, which is measured rather than assumed.
4. Task 5's three physical definitions (half the lift height, 3 mm, 10 degrees) are fixed before data collection and recorded in the spec as definitions, not as calibrated values.
5. Task 6 runs the gate twice — once for deadline evidence, once under the final frozen settings — and only the second run is the gate result.
6. Task 7 reconciles the gate's seeds with the frozen `dev_gate` list by comparison, not assumption.
