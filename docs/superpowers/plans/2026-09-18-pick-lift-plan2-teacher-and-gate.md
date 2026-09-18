# Pick-Lift Milestone — Plan 2: Force Limits, Pick Teacher, Gate, Frozen Lists and the Evaluation CLI

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Plan 1 foundations into a working, measured evaluation system: a pick-only teacher, measured force limits and hold tolerances, a 100-episode teacher gate, frozen scene lists, an `evaluate_pick.py` CLI with resume and a `--final` guard, and a timed run that replaces the 7-hour estimate with a measurement.

**Architecture:** Plan 1 built the judge, the fact reader, the contact classifier, the attempts ledger, the snapshot record, the statistics and `PickEpisodeRunner`, all proven with a trivial policy. Plan 2 adds the first real policy (the teacher), the measurements that let the frozen thresholds be set honestly, and the command-line entry point that ties a run directory together. New modules stay flat next to the Plan 1 `pick_*` modules. `expert.py` is **not** modified: the pick teacher wraps `ScriptedExpert` with `subtasks=("pick_utensil",)` and stops feeding it once it reaches its `home` phase, so the lift is held instead of being put back.

**Tech Stack:** Python 3.12, MuJoCo 3.13.0, NumPy, `unittest` (the repo has no pytest), Windows PowerShell, `.venv-sim`.

**Spec:** `docs/superpowers/specs/2026-09-17-pick-lift-selection-design.md` (commits `ab58560`, `2fb1a0b`). Read it before starting. Section numbers below (§) refer to it.

**Previous plan:** `docs/superpowers/plans/2026-09-17-pick-lift-plan1-foundations.md` — complete, 216 tests passing, HEAD `7fa26be`.

## Plan series

| Plan | Covers spec §13 steps | Output |
|---|---|---|
| 1 (done) | 0, 1, code half of 2 | Foundations and physics v2, proven with a trivial policy |
| **2 (this file)** | rest of 2, 3, 4, 5 | Force measurements, pick teacher, jaw-shape evidence, hold calibration, teacher gate, frozen scene lists, `evaluate_pick.py`, timed run |
| 3 | 6, 7 | Four-cell data generator, provenance, balance and acceptance-bias reports, pilot gate, full data |
| 4 | 8, 9 | Two training runs, checkpoint screening and selection, export, saved-noise parity, paired native vs OpenVINO |
| 5 | 10 | Freeze, final main test and wording test, reports |

## Global Constraints

- All changes are additive. Do not delete or rewrite existing behaviour. `src/rescuehandsai/expert.py`, `runner.py`, `control.py` and `scripts/evaluate.py` keep today's behaviour; the full-task path stays on `physics_version = 1` (§2).
- **Commit after each task. Never push to GitHub** (owner instruction).
- **Never commit the owner's documentation edits.** `README.md`, `docs/ROADMAP.md`, `docs/submission/lablab-submission.md` and everything under `docs/research/` stay uncommitted until the owner explicitly says to commit them. Stage files by explicit path; never `git add -A` or `git add .`.
- Run tests from the project root in PowerShell with `$env:PYTHONPATH = "src"` and `.venv-sim/Scripts/python.exe -m unittest ...`.
- **Never run final-test scene lists during development.** Seed blocks from 3,200,000 upward (`test_main`, `test_wording`) may be generated and frozen in Task 6, but no episode from them is ever executed in this plan (§8).
- Contacts are classified by collision shape (geom), never by whole body (§5).
- Invalid labels are exactly `SIM_ERROR`, `MODEL_LOAD_ERROR`, `CONTRACT_MISMATCH`. Policy exceptions are valid `POLICY_ERROR` failures (§6).
- One diagnosed retry per invalid attempt. A second invalid attempt blocks the episode. Valid failures are never retried (§6).
- The force limit is **not** "teacher maximum plus a margin". It comes from controlled measurements that separate gentle contact from hard hits. If the teacher cannot satisfy the limit, **the teacher is fixed, not the limit** (§5).
- Hold tolerances *are* calibrated on the teacher (§5, section-2 revision), and the reason for each chosen value is written into the spec.
- Text files are hashed with CRLF normalised to LF (`core.autocrlf` is on in this checkout, off on Kaggle).
- No measured number may be invented. Every threshold this plan freezes is written into `docs/superpowers/specs/2026-09-17-pick-lift-selection-design.md` together with the measurement that produced it and the date.

## Facts already measured (do not re-derive)

- Policy protocol required by `PickEpisodeRunner` (`src/rescuehandsai/pick_runner.py:32-130`): attribute `uses_privileged_state`, and methods `reset(sim, task)`, `wants_images() -> bool`, `act(obs) -> BimanualAction`, `metadata() -> dict`.
- `ScriptedExpert(sim, task: TaskSpec, subtasks=SUBTASKS)` accepts a subtask tuple; `SUBTASKS = ("pick_utensil", "handoff", "place_utensil", "place_cup")` (`src/rescuehandsai/task.py:22`).
- `ScriptedExpert._script()` always yields a final home move and sets `self.subtask = "home"` (`src/rescuehandsai/expert.py:148-163`). The pick teacher must stop before that move so the utensil stays lifted.
- `ScriptedExpert.act(obs)` returns `BimanualAction(obs.timestamp, dict(self.follower.targets))`; `self.done` becomes `True` when the script is exhausted (`expert.py:59-71`).
- `ScriptedExpert` raises `PlanningError` (unreachable pose) and `LostItemError` (item not where the plan assumed) (`expert.py:34-40`).
- `configs/pick_rules.json` currently has `"frozen": false`; `configs/pick_contacts.json` has `"jaw_table_force_limit_n": null`, `"severe_force_limit_n": null`, `"force_limits_frozen": false`. Plan 2 sets and freezes all three.
- `ContactClassifier.classify()` only applies the force limits when they are not `None` (`src/rescuehandsai/pick_contacts.py:93-99`), so the classifier already works before and after freezing.
- Seed blocks (`configs/pick_seed_blocks.json`): train `3000000–3000999`, dev `3100000–3199999`, test_main `3200000–3299999`, test_wording `3300000–3399999`.
- Current `deadline_control_steps` is `300` (15 s at 20 Hz control). Task 5 replaces it with a value derived from measured teacher time-to-hold.

## File map

| File | Status | Responsibility |
|---|---|---|
| `src/rescuehandsai/pick_teacher.py` | create | `PickTeacher` — pick-only teacher matching the runner's policy protocol |
| `tests/test_pick_teacher.py` | create | Teacher unit and simulation tests |
| `scripts/measure_grasp_shapes.py` | create | Record which shapes actually touch the utensil in teacher grasps (§13 step 2) |
| `scripts/measure_forces.py` | create | Controlled gentle-rest and pressed-down force measurements (§5) |
| `scripts/calibrate_hold.py` | create | Teacher hold-metric distributions used to set hold tolerances (§5) |
| `scripts/teacher_gate.py` | create | 25 dev scenes × 4 cells gate, p95/max time-to-hold, deadline check (§7) |
| `scripts/freeze_scene_lists.py` | create | Generate, duplicate-check and freeze dev/test scene lists (§8) |
| `scripts/evaluate_pick.py` | create | The evaluation CLI: run directory, manifest, resume, summary, `--final` guard (§11) |
| `scripts/time_episodes.py` | create | Timed 4-episode run including model load and rendering (§10) |
| `src/rescuehandsai/pick_run.py` | create | Run-directory plumbing shared by the gate and the CLI (manifest, ledger loop, summary writing) |
| `tests/test_pick_run.py` | create | Run-directory, resume and `--final` guard tests |
| `configs/pick_contacts.json` | modify | Measured force limits, `force_limits_frozen: true` |
| `configs/pick_rules.json` | modify | Calibrated hold tolerances, measured deadline, `frozen: true` |
| `configs/pick_scene_lists/*.json` | create (generated) | Frozen dev, main-test and wording-test scene lists with hashes |
| `docs/superpowers/specs/2026-09-17-pick-lift-selection-design.md` | modify | Record every measurement and the reason for each frozen value |
| `results/teacher_gate/` | create (generated) | Gate run directory and report |
| `results/measurements/` | create (generated) | Force, grasp-shape, hold and timing reports |

---

### Task 1: The pick-only teacher

**Files:**
- Create: `src/rescuehandsai/pick_teacher.py`
- Test: `tests/test_pick_teacher.py`

**Interfaces:**
- Consumes: `ScriptedExpert` (`src/rescuehandsai/expert.py:42`), `PlanningError`, `LostItemError`, `TaskSpec` (`src/rescuehandsai/task.py:26`), `PickTask` (`src/rescuehandsai/pick_cells.py:37`), `BimanualAction` (`src/rescuehandsai/contracts.py`).
- Produces: `PickTeacher(hold_extra_steps: int = 0)` with `uses_privileged_state = True`, `reset(sim, task)`, `wants_images() -> bool`, `act(obs) -> BimanualAction`, `metadata() -> dict`, and the read-only attributes `phase: str` and `finished_pick_at: int | None` (control step at which the expert reached its home phase, i.e. the lift was complete).

- [ ] **Step 1: Write the failing test**

Create `tests/test_pick_teacher.py`:

```python
import unittest

from rescuehandsai.contracts import BimanualAction
from rescuehandsai.pick_cells import make_pick_task
from rescuehandsai.pick_config import load_rules
from rescuehandsai.pick_runner import PickEpisodeRunner
from rescuehandsai.pick_teacher import PickTeacher
from rescuehandsai.sim import MujocoSimulation


class TeacherProtocolTests(unittest.TestCase):
    """The runner only accepts a policy with this exact shape."""

    def test_metadata_names_the_teacher_and_claims_no_checkpoint(self):
        teacher = PickTeacher()
        meta = teacher.metadata()
        self.assertEqual(meta["name"], "pick_teacher")
        self.assertEqual(meta["backend"], "python")
        self.assertEqual(meta["device"], "cpu")
        self.assertIsNone(meta["checkpoint"])

    def test_teacher_declares_privileged_state(self):
        # The teacher reads exact object poses and uses IK; the learned policy never does.
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
            obs = self.sim.observe(images=False)
            action = teacher.act(obs)
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
        task = make_pick_task(3100000, "F-A", "T1")

        class Unreachable(PickTeacher):
            def act(self, obs):
                from rescuehandsai.expert import PlanningError
                raise PlanningError("no reachable pose")

        record = PickEpisodeRunner(self.sim, Unreachable(), rules=self.rules).run(task)
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
"""Pick-only teacher: approach the named utensil, grasp, lift, then hold (spec §7).

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

    def __init__(self, hold_extra_steps: int = 0):
        self.hold_extra_steps = hold_extra_steps
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
            # `home` is the expert's own wind-down move, and `done` means the script ran out.
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

If `test_teacher_lifts_the_named_utensil_and_keeps_holding_it` fails, **do not relax the success rules**. Print the outcome's `failures` list and find out which rule failed. Likely causes, in order: (a) the hold window is longer than the frozen command — check `record["outcome"]["hold"]`; (b) the expert reached `home` before the lift height was met; (c) the frozen grasp-shape list misses a shape that actually holds the utensil — that is Task 2's measurement, so record the failure and continue to Task 2 before changing anything.

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
- Consumes: `PickTeacher` (Task 1), `ContactClassifier` (`src/rescuehandsai/pick_contacts.py:34`), `load_contacts` (`pick_config.py:17`), `make_pick_task`, `MujocoSimulation`.
- Produces: `measure_grasp_shapes(sim, seeds, cells) -> dict` with keys `touch_counts` (shape name → episodes in which it touched the named utensil), `jaw_shapes_seen` (sorted list), `unlisted_shapes` (sorted list of shapes that touched the utensil but are not in the frozen jaw list), `episodes`.

Why this task exists: the jaw grasp list in `configs/pick_contacts.json` was written from the robot XML by reading it, not by watching a grasp. If a shape that really holds the utensil is missing from the list, rule 2 ("a real, stable hold") would reject good grasps for ever, and no amount of training would fix it (§5).

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
        self.listed = set(load_contacts()["jaw_grasp_geoms"]["fixed"] + load_contacts()["jaw_grasp_geoms"]["moving"])

    def test_no_unlisted_shape_holds_the_utensil(self):
        self.assertEqual(self.report["unlisted_shapes"], [],
                         "these shapes held the utensil but are not in jaw_grasp_geoms")

    def test_both_jaws_are_represented_in_real_grasps(self):
        seen = set(self.report["jaw_shapes_seen"])
        fixed = set(load_contacts()["jaw_grasp_geoms"]["fixed"])
        moving = set(load_contacts()["jaw_grasp_geoms"]["moving"])
        self.assertTrue(seen & fixed, "no fixed-jaw shape ever touched the utensil")
        self.assertTrue(seen & moving, "no moving-jaw shape ever touched the utensil")

    def test_every_listed_shape_is_a_real_geom_that_can_collide(self):
        # A listed shape that cannot collide would be dead weight in the rules.
        self.assertEqual(self.report["non_colliding_listed"], [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it skips (no report yet)**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_grasp_shape_evidence.py"`
Expected: 3 tests, all `skipped` with "run scripts/measure_grasp_shapes.py first".

- [ ] **Step 3: Write the measurement script**

Create `scripts/measure_grasp_shapes.py`:

```python
"""Record which collision shapes actually touch the named utensil during teacher grasps.

The frozen jaw list in configs/pick_contacts.json was read off the robot XML. This script
checks it against real grasps: every shape that touches the named utensil at any physics
step is counted, and any shape outside the list is reported. Spec §13 step 2.

Usage (project root):
  $env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe scripts/measure_grasp_shapes.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rescuehandsai.pick_cells import make_pick_task                     # noqa: E402
from rescuehandsai.pick_config import load_contacts, load_rules         # noqa: E402
from rescuehandsai.pick_contacts import ContactClassifier, collides, geom_name  # noqa: E402
from rescuehandsai.pick_runner import PickEpisodeRunner                 # noqa: E402
from rescuehandsai.pick_teacher import PickTeacher                      # noqa: E402
from rescuehandsai.sim import MujocoSimulation                          # noqa: E402

SEEDS = (3100000, 3100001, 3100002, 3100003)
CELLS = ("F-A", "F-B", "S-A", "S-B")


class Watcher(PickTeacher):
    """A teacher that also notes every shape touching the named utensil."""

    def __init__(self, sim, named, contacts):
        super().__init__()
        self.classifier = ContactClassifier(sim.model, named, contacts)
        self.sim, self.named = sim, named
        self.touched = Counter()

    def act(self, obs):
        for verdict in self.classifier.contacts(self.sim.data):
            if self.named in verdict.names:
                for name in verdict.names:
                    if name != self.named and "/" in name:  # robot shapes carry an arm prefix
                        self.touched[name.split("/", 1)[1]] += 1
        return super().act(obs)


def measure(sim, seeds=SEEDS, cells=CELLS) -> dict:
    contacts, rules = load_contacts(), load_rules()
    listed = set(contacts["jaw_grasp_geoms"]["fixed"]) | set(contacts["jaw_grasp_geoms"]["moving"])
    touch_episodes, episodes = Counter(), []
    for seed in seeds:
        for cell in cells:
            task = make_pick_task(seed, cell, "T1")
            watcher = Watcher(sim, task.utensil, contacts)
            record = PickEpisodeRunner(sim, watcher, rules=rules).run(task)
            for name in watcher.touched:
                touch_episodes[name] += 1
            episodes.append({"seed": seed, "cell": cell, "success": record["success"],
                             "picked": record["outcome"]["picked"],
                             "shapes": sorted(watcher.touched)})
    seen = set(touch_episodes)
    non_colliding = sorted(n for n in listed
                           if not collides(sim.model, sim.model.geom(f"right_arm/{n}").id))
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
    for name, count in report["touch_counts"].items():
        print(f"  {name}: touched the utensil in {count} of {len(report['episodes'])} episodes")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the measurement**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe scripts/measure_grasp_shapes.py`
Expected: a printed table and `results/measurements/grasp_shapes.json`.

**Stop and report** if `unlisted shapes` is not empty. Do not silently add the shapes to the config. Report which shape held the utensil, in how many episodes, and what part of the gripper it belongs to; the owner decides whether the frozen list changes, and the spec records the change with this measurement as its reason.

- [ ] **Step 5: Run the evidence tests**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_grasp_shape_evidence.py"`
Expected: 3 tests `ok` (no longer skipped).

- [ ] **Step 6: Record the measurement in the spec**

In `docs/superpowers/specs/2026-09-17-pick-lift-selection-design.md`, under "#### Right-jaw grasp shapes (exact list)", add a dated line: which shapes were observed holding the utensil, over how many episodes, and that the list was confirmed (or what changed and why).

- [ ] **Step 7: Commit**

```powershell
git add scripts/measure_grasp_shapes.py tests/test_grasp_shape_evidence.py results/measurements/grasp_shapes.json docs/superpowers/specs/2026-09-17-pick-lift-selection-design.md
git commit -m "Pick plan 2 task 2: grasp-shape evidence from real teacher grasps"
```

Note: `results/` is in `.gitignore`, so the report needs `git add -f results/measurements/grasp_shapes.json`. Add only that file.

---

### Task 3: Force measurements and the frozen force limits

**Files:**
- Create: `scripts/measure_forces.py`
- Create: `tests/test_force_limits.py`
- Modify: `configs/pick_contacts.json` (limits and `force_limits_frozen`)
- Modify: `docs/superpowers/specs/2026-09-17-pick-lift-selection-design.md` (§5 "Force limit")
- Generated: `results/measurements/forces.json`

**Interfaces:**
- Consumes: `MujocoSimulation`, `ContactClassifier` (`pick_contacts.py:34`), `load_contacts` (`pick_config.py:17`).
- Produces: `measure_forces(sim, speeds) -> dict` with keys `conditions` (condition name → list of per-step normal forces in newtons), `summary` (condition → `samples`, `max_n`, `p95_n`, `mean_n`) and `press_speeds_m_per_s`.

Why this task exists: the spec forbids setting the limit from the teacher's own maximum (§5). A limit copied from the teacher would simply declare whatever the teacher does to be acceptable. Instead two controlled conditions are measured — a gentle rest and deliberate presses — and the limit is placed in the gap between them.

- [ ] **Step 1: Write the failing test**

Create `tests/test_force_limits.py`:

```python
import json
import unittest
from pathlib import Path

from rescuehandsai.pick_config import load_contacts

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results" / "measurements" / "forces.json"


class ForceLimitTests(unittest.TestCase):
    def setUp(self):
        self.contacts = load_contacts()

    def test_limits_are_set_and_frozen(self):
        self.assertIsNotNone(self.contacts["jaw_table_force_limit_n"])
        self.assertIsNotNone(self.contacts["severe_force_limit_n"])
        self.assertTrue(self.contacts["force_limits_frozen"])

    def test_severe_limit_is_above_the_ordinary_limit(self):
        self.assertGreater(self.contacts["severe_force_limit_n"],
                           self.contacts["jaw_table_force_limit_n"])

    def test_the_limit_sits_between_gentle_contact_and_hard_presses(self):
        """The whole point of the limit: a gentle rest passes, a deliberate press fails."""
        if not REPORT.is_file():
            self.skipTest("run scripts/measure_forces.py first (Task 3 step 3)")
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        limit = self.contacts["jaw_table_force_limit_n"]
        gentle_max = report["summary"]["gentle"]["max_n"]
        pressed_min = min(v["max_n"] for k, v in report["summary"].items() if k != "gentle")
        self.assertGreater(limit, gentle_max, "the limit would flag a gentle rest as excess force")
        self.assertLess(limit, pressed_min, "the limit would let every deliberate press through")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_force_limits.py"`
Expected: `test_limits_are_set_and_frozen` FAILS (the config still holds `null`); the third test skips.

- [ ] **Step 3: Write the measurement script**

Create `scripts/measure_forces.py`:

```python
"""Controlled jaw-to-table force measurements (spec section 5, "Force limit").

Two conditions, both with an empty gripper over bare table:
  gentle  - the arm is lowered until the jaws just touch, then held there; this is
            what an acceptable brush against the table looks like.
  pressed - the arm keeps descending after contact at a set command speed, so the
            jaws push into the table; this is what a hard hit looks like.
The limit is then placed in the gap between the two and written into the spec with
these numbers. The teacher's own maximum is deliberately NOT used (spec section 5).

Usage (project root):
  $env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe scripts/measure_forces.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rescuehandsai.contracts import BimanualAction                      # noqa: E402
from rescuehandsai.pick_config import load_contacts                     # noqa: E402
from rescuehandsai.pick_contacts import ContactClassifier               # noqa: E402
from rescuehandsai.sim import MujocoSimulation                          # noqa: E402

PRESS_SPEEDS_M_PER_S = (0.01, 0.02, 0.05)
DESCEND_STEPS = 120
SETTLE_STEPS = 60
DESCEND_JOINTS = ("elbow_flex", "wrist_flex")


def jaw_table_forces(sim, classifier) -> list:
    """Normal forces, in newtons, of every right-jaw-to-table contact at this instant."""
    return [v.force for v in classifier.contacts(sim.data) if v.jaw and "table" in v.names]


def lower_onto_table(sim, classifier, *, extra_drop_m: float, speed_m_per_s: float) -> list:
    """Lower the right jaws onto the table, then continue for extra_drop_m of command travel."""
    sim.reset(0)
    targets = dict(sim.previous)
    forces, dropped, touching = [], 0.0, False
    step = speed_m_per_s * sim.config["control_dt"]
    for _ in range(DESCEND_STEPS):
        contact = jaw_table_forces(sim, classifier)
        if contact:
            touching = True
            forces.extend(contact)
        if touching:
            dropped += step
            if dropped >= extra_drop_m:
                break
        for name, (low, high) in sim.limits.items():
            if name.startswith("right_arm") and name.endswith(DESCEND_JOINTS):
                targets[name] = float(np.clip(targets[name] - step, low, high))
        sim.step(BimanualAction(float(sim.data.time), dict(targets)), stop_on_cross_arm=False)
    for _ in range(SETTLE_STEPS):  # hold the pose and keep measuring
        sim.step(BimanualAction(float(sim.data.time), dict(targets)), stop_on_cross_arm=False)
        forces.extend(jaw_table_forces(sim, classifier))
    return forces


def measure_forces(sim, speeds=PRESS_SPEEDS_M_PER_S) -> dict:
    classifier = ContactClassifier(sim.model, "fork", load_contacts())
    conditions = {"gentle": lower_onto_table(sim, classifier, extra_drop_m=0.0, speed_m_per_s=0.01)}
    for speed in speeds:
        conditions[f"pressed_{speed}"] = lower_onto_table(sim, classifier, extra_drop_m=0.01,
                                                          speed_m_per_s=speed)
    summary = {}
    for name, forces in conditions.items():
        arr = np.array(forces, dtype=float)
        summary[name] = {"samples": int(arr.size),
                         "max_n": float(arr.max()) if arr.size else 0.0,
                         "p95_n": float(np.percentile(arr, 95)) if arr.size else 0.0,
                         "mean_n": float(arr.mean()) if arr.size else 0.0}
    return {"conditions": {k: [float(x) for x in v] for k, v in conditions.items()},
            "summary": summary, "press_speeds_m_per_s": list(speeds)}


def main():
    sim = MujocoSimulation(physics_version=2)
    try:
        report = measure_forces(sim)
    finally:
        sim.close()
    out = ROOT / "results" / "measurements" / "forces.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    for name, s in report["summary"].items():
        print(f"  {name:>14}: max {s['max_n']:.2f} N | p95 {s['p95_n']:.2f} N | "
              f"mean {s['mean_n']:.2f} N | {s['samples']} samples")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the measurement**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe scripts/measure_forces.py`
Expected: one line per condition, every condition with more than zero samples.

**Stop and report** if either is true:
- the gentle condition records zero samples — the descent never reached the table, so the script is wrong, not the physics. Check which joints `DESCEND_JOINTS` actually moves for this arm before changing anything else;
- the gentle maximum is not clearly below the smallest pressed maximum. With no gap there is no honest limit; report both numbers and stop.

- [ ] **Step 5: Choose and freeze the limits**

With a clear gap, set in `configs/pick_contacts.json`:
- `jaw_table_force_limit_n`: a round number inside the gap — at least 1.5× the gentle maximum and at most 0.7× the smallest pressed maximum;
- `severe_force_limit_n`: a round number at or above the largest pressed maximum, so only a genuinely hard hit stops the episode early;
- `force_limits_frozen`: `true`.

Write the numbers, the gap, the date and this rule into the spec's "### Force limit" section, replacing the wording that says the limits are not yet measured.

- [ ] **Step 6: Run the whole suite**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests`
Expected: `OK`, with the three force tests passing.

If a Plan 1 contact test now fails because the limits stopped being `None`, that is a real finding, not noise: the test relied on the limits being switched off. Fix the test so it states the limit it expects, and say so in the task report.

- [ ] **Step 7: Commit**

```powershell
git add scripts/measure_forces.py tests/test_force_limits.py configs/pick_contacts.json docs/superpowers/specs/2026-09-17-pick-lift-selection-design.md
git add -f results/measurements/forces.json
git commit -m "Pick plan 2 task 3: measured jaw-table forces; force limits set and frozen"
```

---

### Task 4: Hold tolerances calibrated on the teacher

**Files:**
- Create: `scripts/calibrate_hold.py`
- Create: `tests/test_hold_calibration.py`
- Modify: `configs/pick_rules.json` (hold tolerances; `frozen` stays `false` until Task 5)
- Modify: `docs/superpowers/specs/2026-09-17-pick-lift-selection-design.md` (§5)
- Generated: `results/measurements/hold.json`

**Interfaces:**
- Consumes: `PickTeacher` (Task 1), `PickEpisodeRunner`, `load_rules`, `make_pick_task`.
- Produces: `collect_hold_metrics(sim, seeds, cells, rules) -> dict` with `values` (metric → list of per-episode values), `summary` (metric → `count`, `min`, `p50`, `p95`, `max`), `episodes`, `successful_episodes`, `total`. Metrics: `shift_m`, `turn_deg`, `speed_mps`, `angular_speed_rps`, `both_jaw_fraction`, `single_jaw_gap_s`.

Why this task exists: the hold rule decides what counts as "really holding it". Tolerances that are too tight reject good teacher holds and make the gate unreachable; too loose and a wobbling, half-dropped utensil counts as held. The spec allows these to be calibrated on the teacher because the teacher is the reference for a *good* hold — unlike the force limit, which must not be.

**Read before starting:** open `src/rescuehandsai/pick_outcome.py` and confirm the exact key names the judge writes into `PickOutcome.hold`. The metric names above must match those keys exactly; if they differ, use the judge's names in the script, the test and the report, and say so in the task report.

- [ ] **Step 1: Write the failing test**

Create `tests/test_hold_calibration.py`:

```python
import json
import unittest
from pathlib import Path

from rescuehandsai.pick_config import load_rules

REPORT = Path(__file__).resolve().parents[1] / "results" / "measurements" / "hold.json"


class HoldCalibrationTests(unittest.TestCase):
    def setUp(self):
        if not REPORT.is_file():
            self.skipTest("run scripts/calibrate_hold.py first (Task 4 step 3)")
        self.report = json.loads(REPORT.read_text(encoding="utf-8"))
        self.rules = load_rules()

    def test_every_tolerance_clears_the_teacher_with_margin(self):
        """A good teacher hold must pass comfortably, or the gate can never be met."""
        pairs = [("hold_max_shift_m", "shift_m"), ("hold_max_turn_deg", "turn_deg"),
                 ("final_max_speed_mps", "speed_mps"),
                 ("final_max_angular_speed_rps", "angular_speed_rps")]
        for rule_key, metric in pairs:
            with self.subTest(rule=rule_key):
                self.assertGreater(self.rules[rule_key], self.report["summary"][metric]["p95"],
                                   f"{rule_key} is tighter than the teacher's 95th percentile")

    def test_jaw_coverage_requirement_is_met_by_the_teacher(self):
        self.assertLessEqual(self.rules["both_jaw_fraction"],
                             self.report["summary"]["both_jaw_fraction"]["p50"])

    def test_tolerances_are_not_so_loose_they_accept_a_swinging_item(self):
        # A hold that drifts further than the lift height is not a hold.
        self.assertLess(self.rules["hold_max_shift_m"], self.rules["lift_height_m"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it skips**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_hold_calibration.py"`
Expected: 3 tests `skipped`.

- [ ] **Step 3: Write the calibration script**

Create `scripts/calibrate_hold.py`:

```python
"""Measure what a good teacher hold looks like, so the hold tolerances can be set honestly.

Runs the pick teacher on dev scenes with deliberately loose tolerances, so the judge
measures the hold instead of cutting it short, and records the hold measurements of the
successful episodes. Tolerances are then set from that distribution (spec section 5).
Force limits are NOT set this way - see scripts/measure_forces.py.

Usage (project root):
  $env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe scripts/calibrate_hold.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rescuehandsai.pick_cells import make_pick_task                     # noqa: E402
from rescuehandsai.pick_config import load_rules                        # noqa: E402
from rescuehandsai.pick_runner import PickEpisodeRunner                 # noqa: E402
from rescuehandsai.pick_teacher import PickTeacher                      # noqa: E402
from rescuehandsai.sim import MujocoSimulation                          # noqa: E402

SEEDS = tuple(range(3100000, 3100010))
CELLS = ("F-A", "F-B", "S-A", "S-B")
METRICS = ("shift_m", "turn_deg", "speed_mps", "angular_speed_rps",
           "both_jaw_fraction", "single_jaw_gap_s")
# Loose on purpose: the judge should measure the hold, not end it.
LOOSE = {"hold_max_shift_m": 1.0, "hold_max_turn_deg": 180.0, "final_max_speed_mps": 10.0,
         "final_max_angular_speed_rps": 100.0, "both_jaw_fraction": 0.0,
         "max_single_jaw_gap_s": 10.0}


def collect_hold_metrics(sim, seeds=SEEDS, cells=CELLS, rules=None) -> dict:
    rules = dict(rules or load_rules(), **LOOSE)
    values = {m: [] for m in METRICS}
    episodes = []
    for seed in seeds:
        for cell in cells:
            task = make_pick_task(seed, cell, "T1")
            record = PickEpisodeRunner(sim, PickTeacher(), rules=rules).run(task)
            hold = record["outcome"].get("hold") or {}
            episodes.append({"seed": seed, "cell": cell, "success": record["success"],
                             "failures": record["outcome"]["failures"], "hold": hold})
            if not record["success"]:
                continue  # only a successful hold describes a good hold
            for metric in METRICS:
                if hold.get(metric) is not None:
                    values[metric].append(float(hold[metric]))
    summary = {}
    for metric, series in values.items():
        arr = np.array(series, dtype=float)
        summary[metric] = {"count": int(arr.size),
                           "min": float(arr.min()) if arr.size else None,
                           "p50": float(np.percentile(arr, 50)) if arr.size else None,
                           "p95": float(np.percentile(arr, 95)) if arr.size else None,
                           "max": float(arr.max()) if arr.size else None}
    return {"values": values, "summary": summary, "episodes": episodes,
            "successful_episodes": sum(1 for e in episodes if e["success"]),
            "total": len(episodes)}


def main():
    sim = MujocoSimulation(physics_version=2)
    try:
        report = collect_hold_metrics(sim)
    finally:
        sim.close()
    out = ROOT / "results" / "measurements" / "hold.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(f"successful teacher holds: {report['successful_episodes']} / {report['total']}")
    for metric, s in report["summary"].items():
        if s["count"]:
            print(f"  {metric:>20}: p50 {s['p50']:.4f} | p95 {s['p95']:.4f} | max {s['max']:.4f} "
                  f"({s['count']} episodes)")
        else:
            print(f"  {metric:>20}: no data")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the calibration**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe scripts/calibrate_hold.py`
Expected: a distribution line for every metric.

**Stop and report** if fewer than 30 of the 40 episodes produce a successful hold even with loose tolerances. That means the teacher itself is unreliable, and no choice of tolerance fixes it. Report the failure labels and which cells they cluster in before touching any threshold.

- [ ] **Step 5: Set the tolerances**

In `configs/pick_rules.json`, set each tolerance from the measured distribution using this rule, then write the rule and the numbers into the spec:
- `hold_max_shift_m`, `hold_max_turn_deg`, `final_max_speed_mps`, `final_max_angular_speed_rps`: **twice the teacher's 95th percentile**, rounded up to a readable number, but never above the sanity ceilings — shift below `lift_height_m`, turn at most 45 degrees;
- `both_jaw_fraction`: at or below the teacher's median, never above;
- `max_single_jaw_gap_s`: at least the teacher's 95th percentile single-jaw gap.

Leave `"frozen": false` for now; Task 5 sets the deadline and freezes the file as a whole.

- [ ] **Step 6: Run the whole suite**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests`
Expected: `OK`, with the hold-calibration tests now passing.

- [ ] **Step 7: Commit**

```powershell
git add scripts/calibrate_hold.py tests/test_hold_calibration.py configs/pick_rules.json docs/superpowers/specs/2026-09-17-pick-lift-selection-design.md
git add -f results/measurements/hold.json
git commit -m "Pick plan 2 task 4: hold tolerances calibrated on measured teacher holds"
```

---

### Task 5: The teacher gate (100 episodes) and the measured deadline

**Files:**
- Create: `scripts/teacher_gate.py`
- Create: `tests/test_teacher_gate.py`
- Modify: `configs/pick_rules.json` (`deadline_control_steps`, then `frozen: true`)
- Modify: `docs/superpowers/specs/2026-09-17-pick-lift-selection-design.md` (§7)
- Generated: `results/teacher_gate/`, `results/teacher_gate_frozen/`

**Interfaces:**
- Consumes: `PickTeacher`, `PickEpisodeRunner`, `AttemptLedger`, `write_episode`, `episode_key`, `episode_filename`, `InvalidRun` (`pick_records.py`), `check_start` (`pick_cells.py:103`).
- Produces: `judge_gate(results: list) -> dict`, `suggested_deadline(verdict: dict) -> int`, and `run_gate(sim, seeds, cells, rules, run_dir) -> dict`. The verdict holds `episodes`, `successes`, `per_cell`, `scenes_all_four`, `time_to_hold_control_steps`, `p95_steps`, `max_steps`, `gate_passed`, `reason`, and (from `run_gate`) `rejected_starts`, `failed_attempts`, `results`.

Gate rule (§7), copied exactly: 25 dev scenes × 4 cells = 100 episodes. Pass if **≥ 99/100 overall and each cell ≥ 24/25**. Also reported: scenes passing all four cells, failed teacher attempts, rejected starts, p95 and maximum time to hold.

- [ ] **Step 1: Write the failing test**

Create `tests/test_teacher_gate.py`. The gate rule is arithmetic, so it is tested on made-up results with no simulation:

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
        out = []
        for seed in range(3100000, 3100025):
            for cell in cells:
                out.append({"seed": seed, "cell": cell, "success": (seed, cell) not in failures,
                            "time_to_hold": 100})
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
        failures = {(3100000, "F-A"), (3100001, "F-A")}
        verdict = teacher_gate.judge_gate(self._results(failures=failures))
        self.assertFalse(verdict["gate_passed"])
        self.assertEqual(verdict["per_cell"]["F-A"], 23)
        self.assertIn("cell F-A", verdict["reason"])

    def test_an_incomplete_run_never_passes(self):
        verdict = teacher_gate.judge_gate(self._results(count=99))
        self.assertFalse(verdict["gate_passed"])
        self.assertIn("incomplete", verdict["reason"])

    def test_deadline_comes_from_the_maximum_not_the_median(self):
        results = self._results()
        results[0]["time_to_hold"] = 250
        verdict = teacher_gate.judge_gate(results)
        self.assertEqual(verdict["max_steps"], 250)
        self.assertGreaterEqual(teacher_gate.suggested_deadline(verdict), 375)

    def test_a_deadline_cannot_be_derived_without_successful_holds(self):
        verdict = teacher_gate.judge_gate([])
        with self.assertRaises(ValueError):
            teacher_gate.suggested_deadline(verdict)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_teacher_gate.py"`
Expected: FAIL — `scripts/teacher_gate.py` does not exist.

- [ ] **Step 3: Write the gate script**

Create `scripts/teacher_gate.py`:

```python
"""The teacher gate: 25 dev scenes x 4 cells = 100 episodes (spec section 7).

Pass = at least 99 of 100 successes AND at least 24 of 25 in every cell. Also reported:
scenes passing all four cells, rejected starts, failed teacher attempts, and the 95th
percentile and maximum time to hold, which is where the episode deadline comes from.

Usage (project root):
  $env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe scripts/teacher_gate.py --run-dir results/teacher_gate
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rescuehandsai.pick_cells import check_start, make_pick_task                      # noqa: E402
from rescuehandsai.pick_config import load_rules                                      # noqa: E402
from rescuehandsai.pick_records import (AttemptLedger, InvalidRun, episode_filename,  # noqa: E402
                                        episode_key, write_episode)
from rescuehandsai.pick_runner import PickEpisodeRunner                               # noqa: E402
from rescuehandsai.pick_teacher import PickTeacher                                    # noqa: E402
from rescuehandsai.sim import MujocoSimulation                                        # noqa: E402

CELLS = ("F-A", "F-B", "S-A", "S-B")
GATE_SCENES = 25
OVERALL_MIN = 99
PER_CELL_MIN = 24
DEADLINE_SAFETY = 1.5  # the deadline must not become a second bar on top of the success rules


def judge_gate(results: list) -> dict:
    """Apply the gate rule to {seed, cell, success, time_to_hold} dictionaries."""
    per_cell = {c: 0 for c in CELLS}
    scenes, times, successes = {}, [], 0
    for r in results:
        scenes.setdefault(r["seed"], []).append(bool(r["success"]))
        if r["success"]:
            successes += 1
            per_cell[r["cell"]] = per_cell.get(r["cell"], 0) + 1
            if r.get("time_to_hold") is not None:
                times.append(int(r["time_to_hold"]))
    arr = np.array(times, dtype=float)
    planned = GATE_SCENES * len(CELLS)
    reasons = []
    if len(results) != planned:
        reasons.append(f"incomplete: {len(results)} of {planned} episodes")
    if successes < OVERALL_MIN:
        reasons.append(f"overall {successes}/{len(results)} below {OVERALL_MIN}")
    for cell, count in sorted(per_cell.items()):
        if count < PER_CELL_MIN:
            reasons.append(f"cell {cell} {count} below {PER_CELL_MIN}")
    return {"episodes": len(results), "successes": successes, "per_cell": per_cell,
            "scenes_all_four": sum(1 for v in scenes.values() if len(v) == len(CELLS) and all(v)),
            "time_to_hold_control_steps": times,
            "p95_steps": float(np.percentile(arr, 95)) if arr.size else None,
            "max_steps": int(arr.max()) if arr.size else None,
            "gate_passed": not reasons, "reason": "; ".join(reasons) or "passed"}


def suggested_deadline(verdict: dict) -> int:
    """From the measured maximum, not the median: a slow success is still a success."""
    if verdict.get("max_steps") is None:
        raise ValueError("no successful holds, so no deadline can be derived")
    return int(math.ceil(verdict["max_steps"] * DEADLINE_SAFETY))


def run_gate(sim, seeds, cells, rules, run_dir: Path) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    ledger = AttemptLedger(run_dir)
    results, rejected, failed_attempts = [], [], []
    for seed in seeds:
        for cell in cells:
            start = check_start(seed, cell, physics_version=sim.physics_version, rules=rules)
            if not start.ok:
                rejected.append({"seed": seed, "cell": cell, "reason": start.reason})
                continue
            key = episode_key(seed, cell)
            attempt = ledger.next_attempt(key)
            task = make_pick_task(seed, cell, "T1")
            try:
                record = PickEpisodeRunner(sim, PickTeacher(), rules=rules).run(task)
            except InvalidRun as invalid:
                write_episode(run_dir, seed, cell, attempt,
                              invalid.partial or {"invalid_label": invalid.label,
                                                  "detail": invalid.detail})
                ledger.record(key, attempt, valid=False, label=invalid.label,
                              filename=episode_filename(seed, cell, attempt))
                failed_attempts.append({"seed": seed, "cell": cell, "label": invalid.label,
                                        "detail": invalid.detail})
                continue
            write_episode(run_dir, seed, cell, attempt, record)
            ledger.record(key, attempt, valid=True, label=None,
                          filename=episode_filename(seed, cell, attempt))
            hold = record["outcome"].get("hold") or {}
            results.append({"seed": seed, "cell": cell, "success": record["success"],
                            "time_to_hold": hold.get("started_control_step"),
                            "picked": record["outcome"]["picked"],
                            "failures": record["outcome"]["failures"]})
    verdict = judge_gate(results)
    verdict.update(rejected_starts=rejected, failed_attempts=failed_attempts, results=results)
    return verdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results" / "teacher_gate")
    parser.add_argument("--first-seed", type=int, default=3100000)
    args = parser.parse_args()
    rules = load_rules()
    sim = MujocoSimulation(physics_version=2)
    try:
        verdict = run_gate(sim, range(args.first_seed, args.first_seed + GATE_SCENES),
                           CELLS, rules, args.run_dir)
    finally:
        sim.close()
    (args.run_dir / "gate.json").write_text(json.dumps(verdict, indent=2, default=str),
                                            encoding="utf-8")
    print(f"gate: {verdict['successes']}/{verdict['episodes']} - {verdict['reason']}")
    print("per cell:", ", ".join(f"{c} {n}/{GATE_SCENES}" for c, n in sorted(verdict["per_cell"].items())))
    print(f"scenes passing all four: {verdict['scenes_all_four']}/{GATE_SCENES}")
    print(f"rejected starts: {len(verdict['rejected_starts'])} | "
          f"failed attempts: {len(verdict['failed_attempts'])}")
    if verdict["max_steps"] is not None:
        print(f"time to hold: p95 {verdict['p95_steps']:.0f} steps | max {verdict['max_steps']} steps"
              f" | suggested deadline {suggested_deadline(verdict)} steps")
    raise SystemExit(0 if verdict["gate_passed"] else 1)


if __name__ == "__main__":
    main()
```

**Note on `time_to_hold`:** the script reads `record["outcome"]["hold"]["started_control_step"]`. Before writing the script, open `src/rescuehandsai/pick_outcome.py` and use the judge's real key for the control step at which the hold window began. If the judge does not record it yet, add it there (additively) together with a unit test in `tests/test_pick_outcome.py` — a gate that cannot measure time-to-hold cannot set the deadline.

- [ ] **Step 4: Run the gate arithmetic tests**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_teacher_gate.py"`
Expected: 7 tests `ok`.

- [ ] **Step 5: Run the real gate**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe scripts/teacher_gate.py --run-dir results/teacher_gate`
Expected: a printed verdict and exit code 0.

**If the gate fails, the teacher is fixed — never the bar, and never the force limit** (§5, §7). Report which cells failed, the failure labels, and whether the failures cluster on one slot. The Sep 17 audit found the old teacher was weaker at slot 0 (clean yield ~0.57–0.64 there versus ~0.82–0.93 at slot 1, mostly "utensil not in place zone"), and this milestone exists partly to fix that. If the same pattern appears, name it in the report; do not average it away.

- [ ] **Step 6: Set and freeze the deadline**

Put the printed suggested deadline (measured maximum × 1.5, rounded up) into `deadline_control_steps` in `configs/pick_rules.json`, then set `"frozen": true`.

Write into the spec's §7: the p95, the maximum, the chosen deadline, and the reason — the timeout is checked against the teacher's p95 **and** maximum so that it never becomes a second hidden bar on top of the five success rules.

- [ ] **Step 7: Re-run the gate with the frozen rules**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe scripts/teacher_gate.py --run-dir results/teacher_gate_frozen`
Expected: the same verdict, exit code 0.

A different verdict means the deadline is doing work the success rules should do. Stop and report both verdicts side by side.

- [ ] **Step 8: Run the whole suite and commit**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests`
Expected: `OK`.

```powershell
git add scripts/teacher_gate.py tests/test_teacher_gate.py configs/pick_rules.json docs/superpowers/specs/2026-09-17-pick-lift-selection-design.md
git add -f results/teacher_gate results/teacher_gate_frozen
git commit -m "Pick plan 2 task 5: teacher gate over 100 dev episodes; deadline measured, rules frozen"
```

---

### Task 6: Frozen scene lists

**Files:**
- Create: `scripts/freeze_scene_lists.py`
- Create: `tests/test_scene_lists.py`
- Generated: `configs/pick_scene_lists/dev_gate.json`, `dev_quick.json`, `dev_selection.json`, `test_main.json`, `test_wording.json`, `index.json`

**Interfaces:**
- Consumes: `check_start` (`pick_cells.py:103`), `settings_hash`, `scene_hash`, `find_duplicates` (`pick_identity.py:53`), `load_seed_blocks`, `block_of` (`pick_identity.py:81`).
- Produces: `build_list(name, block, count, *, physics_version, rules) -> dict` with keys `name`, `seeds`, `count`, `first_seed`, `rejected` (list of `{seed, reason}`), `settings_sha256` (seed → hash), `physics_version`, `created_utc`; and `freeze(lists, out_dir) -> dict` writing one file per list plus `index.json` holding each list's own SHA-256.

List sizes (§8): dev gate 25, dev quick 10, dev selection 50, main test 100, wording test 24.

- [ ] **Step 1: Write the failing test**

Create `tests/test_scene_lists.py`:

```python
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from rescuehandsai.pick_identity import block_of, load_seed_blocks

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("freeze_scene_lists",
                                               ROOT / "scripts" / "freeze_scene_lists.py")
freeze_scene_lists = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(freeze_scene_lists)

LIST_DIR = ROOT / "configs" / "pick_scene_lists"
EXPECTED = {"dev_gate": 25, "dev_quick": 10, "dev_selection": 50,
            "test_main": 100, "test_wording": 24}


class FrozenListTests(unittest.TestCase):
    def setUp(self):
        if not (LIST_DIR / "index.json").is_file():
            self.skipTest("run scripts/freeze_scene_lists.py first (Task 6 step 3)")
        self.index = json.loads((LIST_DIR / "index.json").read_text(encoding="utf-8"))
        self.lists = {name: json.loads((LIST_DIR / f"{name}.json").read_text(encoding="utf-8"))
                      for name in EXPECTED}

    def test_every_list_has_the_size_the_spec_asks_for(self):
        for name, size in EXPECTED.items():
            with self.subTest(list=name):
                self.assertEqual(len(self.lists[name]["seeds"]), size)

    def test_no_seed_appears_in_two_lists(self):
        seen = {}
        for name, data in self.lists.items():
            for seed in data["seeds"]:
                self.assertNotIn(seed, seen, f"seed {seed} is in both {seen.get(seed)} and {name}")
                seen[seed] = name

    def test_no_two_scenes_are_physically_identical(self):
        digests = {}
        for name, data in self.lists.items():
            for seed, digest in data["settings_sha256"].items():
                self.assertNotIn(digest, digests,
                                 f"scene {seed} in {name} duplicates {digests.get(digest)}")
                digests[digest] = f"{name}:{seed}"

    def test_each_seed_sits_in_the_block_its_list_belongs_to(self):
        blocks = load_seed_blocks()
        wanted = {"dev_gate": "dev", "dev_quick": "dev", "dev_selection": "dev",
                  "test_main": "test_main", "test_wording": "test_wording"}
        for name, data in self.lists.items():
            for seed in data["seeds"]:
                self.assertEqual(block_of(seed, blocks), wanted[name], f"{name} seed {seed}")

    def test_the_index_hash_matches_every_stored_list(self):
        for name in EXPECTED:
            with self.subTest(list=name):
                digest = freeze_scene_lists.list_digest(LIST_DIR / f"{name}.json")
                self.assertEqual(self.index["lists"][name]["sha256"], digest)

    def test_rejected_starts_are_kept_not_hidden(self):
        for name, data in self.lists.items():
            with self.subTest(list=name):
                self.assertIn("rejected", data)


class ListBuildingTests(unittest.TestCase):
    """Building is tested without physics by stubbing the start check."""

    def test_rejected_seeds_are_skipped_and_recorded(self):
        calls = {"n": 0}

        def fake_check(seed, cell, **kwargs):
            calls["n"] += 1
            ok = seed % 3 != 0
            return type("S", (), {"ok": ok, "reason": None if ok else "overlap"})()

        built = freeze_scene_lists.build_list("dev_gate", (3100000, 3200000), 4,
                                              physics_version=2, rules={},
                                              check=fake_check, settings=lambda s: f"h{s}")
        self.assertEqual(len(built["seeds"]), 4)
        self.assertTrue(all(s % 3 != 0 for s in built["seeds"]))
        self.assertTrue(built["rejected"])
        self.assertTrue(all(r["reason"] == "overlap" for r in built["rejected"]))

    def test_building_stops_rather_than_leaving_the_block(self):
        def always_bad(seed, cell, **kwargs):
            return type("S", (), {"ok": False, "reason": "overlap"})()

        with self.assertRaises(RuntimeError):
            freeze_scene_lists.build_list("dev_gate", (3100000, 3100010), 5,
                                          physics_version=2, rules={},
                                          check=always_bad, settings=lambda s: f"h{s}")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_scene_lists.py"`
Expected: FAIL — `scripts/freeze_scene_lists.py` does not exist.

- [ ] **Step 3: Write the freezing script**

Create `scripts/freeze_scene_lists.py`:

```python
"""Generate, check and freeze the scene lists (spec section 8).

Each list takes the next seeds in its block whose start passes the frozen start check,
in all four cells. Rejected seeds are recorded with their reason, never silently skipped.
Every list is written with the settings hash of each scene, and index.json holds the
SHA-256 of each list file, so a changed list can never pass unnoticed.

The final-test lists are generated here but NEVER executed during development
(spec section 8); evaluate_pick.py refuses them without --final and a frozen manifest.

Usage (project root):
  $env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe scripts/freeze_scene_lists.py
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rescuehandsai.pick_cells import check_start                        # noqa: E402
from rescuehandsai.pick_config import load_rules                        # noqa: E402
from rescuehandsai.pick_identity import load_seed_blocks, settings_hash # noqa: E402
from rescuehandsai.scene import sample_params                           # noqa: E402
from rescuehandsai.sim import MujocoSimulation                          # noqa: E402

CELLS = ("F-A", "F-B", "S-A", "S-B")
SIZES = (("dev_gate", "dev", 25), ("dev_quick", "dev", 10), ("dev_selection", "dev", 50),
         ("test_main", "test_main", 100), ("test_wording", "test_wording", 24))
LIST_DIR = ROOT / "configs" / "pick_scene_lists"


def list_digest(path: Path) -> str:
    """Hash with CRLF normalised, because this checkout converts line endings."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def build_list(name: str, block, count: int, *, physics_version: int, rules: dict,
               check=check_start, settings=None, start_at: int | None = None) -> dict:
    """Take the next `count` seeds in `block` whose start is valid in all four cells."""
    low, high = block
    seed = start_at if start_at is not None else low
    seeds, rejected, hashes = [], [], {}
    while len(seeds) < count:
        if seed >= high:
            raise RuntimeError(f"{name}: ran out of seeds in block {block} after "
                               f"{len(seeds)} of {count} (rejected {len(rejected)})")
        verdicts = [check(seed, cell, physics_version=physics_version, rules=rules)
                    for cell in CELLS]
        bad = next((v for v in verdicts if not v.ok), None)
        if bad is None:
            seeds.append(seed)
            hashes[str(seed)] = settings(seed) if settings else None
        else:
            rejected.append({"seed": seed, "reason": bad.reason})
        seed += 1
    return {"name": name, "seeds": seeds, "count": len(seeds), "first_seed": seeds[0],
            "next_free_seed": seed, "rejected": rejected, "settings_sha256": hashes,
            "physics_version": physics_version, "cells": list(CELLS),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def freeze(lists: dict, out_dir: Path = LIST_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    index = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "lists": {}}
    for name, data in lists.items():
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        index["lists"][name] = {"sha256": list_digest(path), "count": data["count"],
                                "first_seed": data["first_seed"]}
    (out_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=LIST_DIR)
    args = parser.parse_args()
    if (args.out_dir / "index.json").is_file():
        raise SystemExit(f"{args.out_dir}/index.json already exists; frozen lists are never "
                         f"overwritten. Delete it deliberately if you really mean to refreeze.")
    blocks, rules = load_seed_blocks(), load_rules()
    sim = MujocoSimulation(physics_version=2)
    try:
        def settings_for(seed):
            return settings_hash(sample_params(sim.scene_config, seed))

        lists, cursor = {}, {}
        for name, block_name, count in SIZES:
            block = tuple(blocks["blocks"][block_name])
            lists[name] = build_list(name, block, count, physics_version=2, rules=rules,
                                     settings=settings_for,
                                     start_at=cursor.get(block_name, block[0]))
            cursor[block_name] = lists[name]["next_free_seed"]
    finally:
        sim.close()
    index = freeze(lists, args.out_dir)
    for name, info in index["lists"].items():
        data = lists[name]
        print(f"{name:>14}: {info['count']} scenes from {info['first_seed']} "
              f"({len(data['rejected'])} rejected) sha {info['sha256'][:12]}")
    print("\nFinal-test lists are frozen but must NEVER be run during development.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the unit tests (no physics)**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_scene_lists.py"`
Expected: the two `ListBuildingTests` pass; the `FrozenListTests` skip.

- [ ] **Step 5: Freeze the lists**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe scripts/freeze_scene_lists.py`
Expected: one line per list, each with the requested count.

**Stop and report** if any list rejects more than 20% of the seeds it examined. A high rejection rate means the start rules and the scene sampler disagree, and that would bias every list towards easy scenes. Report the rejection reasons grouped by reason before freezing anything.

- [ ] **Step 6: Run the duplicate check across everything**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_scene_lists.py"`
Expected: all tests `ok`.

Then check the new lists against the legacy ranges recorded in `configs/pick_seed_blocks.json` using `find_duplicates` (`src/rescuehandsai/pick_identity.py:53`), and write the result into the task report. Some old laptop shards have incomplete provenance: say so plainly — "no overlap with the ranges we have recorded" — rather than claiming overlap is impossible (§8).

- [ ] **Step 7: Commit**

```powershell
git add scripts/freeze_scene_lists.py tests/test_scene_lists.py configs/pick_scene_lists
git commit -m "Pick plan 2 task 6: frozen dev and final-test scene lists with hashes and rejection records"
```

---

### Task 7: The evaluation CLI — run directory, resume and the `--final` guard

**Files:**
- Create: `src/rescuehandsai/pick_run.py`
- Create: `scripts/evaluate_pick.py`
- Create: `tests/test_pick_run.py`

**Interfaces:**
- Consumes: `AttemptLedger`, `check_resume`, `check_run_contract`, `ResumeRefused`, `InvalidRun`, `write_episode`, `episode_key`, `episode_filename` (`pick_records.py`); `snapshot_record` (`snapshot.py:135`); `summarize` (`pick_stats.py:51`), `MAIN_TEST_BARS`, `scored_entry`, `INCOMPLETE`; `PickEpisodeRunner`; `make_pick_task`; `load_rules`, `load_contacts`.
- Produces, in `src/rescuehandsai/pick_run.py`:
  - `RunPlan(name, seeds, cells, templates, list_name, list_sha256, is_final)` — a frozen dataclass describing what a run must cover.
  - `planned_keys(plan) -> list[str]` — every `episode_key` the run must end up with.
  - `guard_final(plan, *, final_flag: bool, manifest: dict | None) -> None` — raises `FinalRunRefused` unless a final list is run with `final_flag` **and** a manifest holding `spec_sha256` and `model_sha256`.
  - `build_manifest(plan, *, sim, rules, contacts, policy_meta, snapshot, args) -> dict`.
  - `run_episodes(sim, plan, policy_factory, *, rules, contacts, run_dir, ledger) -> dict` — the episode loop with one diagnosed retry, honouring the ledger's rules.
  - `write_summary(run_dir, plan, ledger, *, bars=None) -> dict` — writes `summary.json` and `summary.md`, sets `evaluation_valid`.
  - `FinalRunRefused(RuntimeError)`.

Why `pick_run.py` exists as a module rather than living in the script: Task 5's gate and Task 8's timing run need the same run-directory behaviour, and the tests must exercise it without a command line.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pick_run.py`:

```python
import json
import tempfile
import unittest
from pathlib import Path

from rescuehandsai.pick_records import AttemptLedger, ResumeRefused, episode_key
from rescuehandsai.pick_run import (FinalRunRefused, RunPlan, guard_final, planned_keys,
                                    write_summary)


def plan(name="dev", seeds=(3100000, 3100001), is_final=False, list_name="dev_gate"):
    return RunPlan(name=name, seeds=tuple(seeds), cells=("F-A", "F-B", "S-A", "S-B"),
                   templates=("T1",), list_name=list_name, list_sha256="abc123",
                   is_final=is_final)


class FinalGuardTests(unittest.TestCase):
    """The final-test lists are the one thing that must never run by accident."""

    def test_a_final_list_without_the_flag_is_refused(self):
        with self.assertRaises(FinalRunRefused):
            guard_final(plan(list_name="test_main", is_final=True), final_flag=False, manifest=None)

    def test_a_final_list_with_the_flag_but_no_manifest_is_refused(self):
        with self.assertRaises(FinalRunRefused):
            guard_final(plan(list_name="test_main", is_final=True), final_flag=True, manifest=None)

    def test_a_final_manifest_missing_the_model_hash_is_refused(self):
        with self.assertRaises(FinalRunRefused) as ctx:
            guard_final(plan(list_name="test_main", is_final=True), final_flag=True,
                        manifest={"spec_sha256": "s" * 64})
        self.assertIn("model_sha256", str(ctx.exception))

    def test_a_complete_final_manifest_is_allowed(self):
        guard_final(plan(list_name="test_main", is_final=True), final_flag=True,
                    manifest={"spec_sha256": "s" * 64, "model_sha256": "m" * 64})

    def test_the_flag_on_a_development_list_is_refused(self):
        # --final is not a way to make a dev run look official.
        with self.assertRaises(FinalRunRefused):
            guard_final(plan(), final_flag=True,
                        manifest={"spec_sha256": "s" * 64, "model_sha256": "m" * 64})


class PlannedKeyTests(unittest.TestCase):
    def test_every_scene_and_cell_is_planned(self):
        keys = planned_keys(plan())
        self.assertEqual(len(keys), 8)
        self.assertIn(episode_key(3100000, "F-A"), keys)
        self.assertEqual(len(set(keys)), len(keys))


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _record(self, seed, cell, success):
        return {"seed": seed, "cell": cell, "success": success, "utensil": "fork",
                "template": "T1", "outcome": {"picked": "fork" if success else "neither",
                                              "failures": [] if success else ["NO_LIFT"],
                                              "first_failure": None if success else "NO_LIFT"}}

    def _fill(self, ledger, p, missing=0):
        from rescuehandsai.pick_records import episode_filename, write_episode
        keys = [(s, c) for s in p.seeds for c in p.cells]
        for seed, cell in keys[:len(keys) - missing]:
            write_episode(self.dir, seed, cell, 1, self._record(seed, cell, True))
            ledger.record(episode_key(seed, cell), 1, valid=True, label=None,
                          filename=episode_filename(seed, cell, 1))

    def test_a_complete_run_is_valid_and_scored(self):
        p = plan()
        ledger = AttemptLedger(self.dir)
        self._fill(ledger, p)
        summary = write_summary(self.dir, p, ledger)
        self.assertTrue(summary["evaluation_valid"])
        self.assertEqual(summary["scored_episodes"], 8)
        self.assertTrue((self.dir / "summary.md").is_file())

    def test_a_missing_episode_makes_the_run_incomplete_and_unpassable(self):
        p = plan()
        ledger = AttemptLedger(self.dir)
        self._fill(ledger, p, missing=1)
        summary = write_summary(self.dir, p, ledger)
        self.assertFalse(summary["evaluation_valid"])
        self.assertIn("incomplete", json.dumps(summary).lower())
        self.assertNotIn("passed", str(summary.get("bars", "")).lower())

    def test_a_blocked_episode_makes_the_run_invalid(self):
        p = plan()
        ledger = AttemptLedger(self.dir)
        key = episode_key(3100000, "F-A")
        ledger.record(key, 1, valid=False, label="SIM_ERROR", filename="a1.json")
        ledger.add_diagnosis(key, 1, "renderer died")
        ledger.record(key, 2, valid=False, label="SIM_ERROR", filename="a2.json")
        summary = write_summary(self.dir, p, ledger)
        self.assertFalse(summary["evaluation_valid"])
        self.assertIn(key, summary["blocked_keys"])

    def test_the_summary_names_which_attempt_supplied_each_result(self):
        p = plan(seeds=(3100000,))
        ledger = AttemptLedger(self.dir)
        self._fill(ledger, p)
        summary = write_summary(self.dir, p, ledger)
        for entry in summary["scored"]:
            self.assertEqual(entry["attempt"], 1)
            self.assertIn("file", entry)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_pick_run.py"`
Expected: `ModuleNotFoundError: No module named 'rescuehandsai.pick_run'`.

- [ ] **Step 3: Write `src/rescuehandsai/pick_run.py`**

Write the module so that the tests above pass, following these rules exactly (§6, §11):

- `RunPlan` is a frozen dataclass with the fields listed in **Interfaces**.
- `planned_keys(plan)` returns `episode_key(seed, cell)` for every seed × cell, in order.
- `guard_final(plan, *, final_flag, manifest)` raises `FinalRunRefused` when
  (a) `plan.is_final` and not `final_flag`; (b) `plan.is_final` and `manifest` is `None`;
  (c) `plan.is_final` and the manifest lacks `spec_sha256` or `model_sha256` (the message
  names the missing key); (d) `final_flag` is set for a plan that is not final. It returns
  `None` on success and never repairs a bad manifest.
- `run_episodes(...)` walks `plan` in order and for each episode:
  skips it when `ledger.scored(key)` is not `None` (that is how resume works);
  asks `ledger.next_attempt(key)` for the number, letting `AlreadyScored`,
  `RetryNeedsDiagnosis` and `EvaluationBlocked` propagate — the ledger is the authority;
  runs `PickEpisodeRunner(...).run(task)`; writes the episode file with `write_episode`
  **before** recording the ledger line, so an interrupted run never has a ledger line
  without its evidence; records valid results with `label=None` and `InvalidRun` failures
  with the invalid label and `invalid.partial` as the saved record.
- `write_summary(...)` collects `ledger.scored(key)` for every planned key, loads each
  scored episode file, calls `summarize(scored, planned_keys=..., blocked_keys=ledger.blocked_keys(), invalid_counts=ledger.invalid_counts(), bars=bars)`,
  sets `evaluation_valid` to true **only** when every planned key has exactly one scored
  valid attempt and no key is blocked, writes `summary.json` and a human-readable
  `summary.md`, and when the run is not valid writes the `INCOMPLETE` notice from
  `pick_stats` instead of a pass or fail verdict.
- Every summary entry names the attempt that supplied it (`attempt`, `file`).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_pick_run.py"`
Expected: 12 tests `ok`.

- [ ] **Step 5: Write `scripts/evaluate_pick.py`**

A thin command line over `pick_run.py`. Required flags:

| Flag | Meaning |
|---|---|
| `--list` | frozen list name from `configs/pick_scene_lists/` (required) |
| `--policy` | `teacher` now; learned policies arrive in Plan 4 |
| `--run-dir` | run directory; created fresh, or resumed |
| `--resume` | allow an existing run directory (refuses without it) |
| `--final` | required for a final list, together with `--final-manifest` |
| `--final-manifest` | path to the frozen manifest holding `spec_sha256` and `model_sha256` |
| `--video` | optional rendering of episodes |

Order of operations, which must not be rearranged:

1. load the frozen list and check its SHA-256 against `configs/pick_scene_lists/index.json`; a mismatch is a `CONTRACT_MISMATCH` refusal before any episode runs;
2. `guard_final(...)`;
3. `snapshot_record(ROOT, sim.asset_path, strict=plan.is_final, ...)` — strict for final runs, which refuse to start with any untracked or modified file under `src/`, `scripts/`, `training/`, `configs/` (§11);
4. build the manifest; on `--resume`, `check_resume(run_dir, current)` first and refuse a changed experiment;
5. run the episodes;
6. write the summary, applying `MAIN_TEST_BARS` only for the final main-test list.

- [ ] **Step 6: Prove resume works on a real 8-episode run**

```powershell
$env:PYTHONPATH = "src"
.venv-sim/Scripts/python.exe scripts/evaluate_pick.py --list dev_quick --policy teacher --run-dir results/pick_resume_check
```
Interrupt it with Ctrl+C after a few episodes, then:
```powershell
.venv-sim/Scripts/python.exe scripts/evaluate_pick.py --list dev_quick --policy teacher --run-dir results/pick_resume_check --resume
```
Expected: the second run skips the episodes that already have a scored valid attempt, adds the rest, and the summary reports one scored attempt per planned episode with `evaluation_valid: true`.

Then prove the refusals, and paste the exact messages into the task report:
- resuming with a changed rule (temporarily edit a tolerance) → `ResumeRefused`;
- `--list test_main` without `--final` → `FinalRunRefused`.
Undo the temporary edit afterwards and re-run the suite.

- [ ] **Step 7: Run the whole suite and commit**

```powershell
git add src/rescuehandsai/pick_run.py scripts/evaluate_pick.py tests/test_pick_run.py
git commit -m "Pick plan 2 task 7: evaluation CLI with run directory, resume and the final-list guard"
```

---

### Task 8: The timed run that replaces the 7-hour estimate

**Files:**
- Create: `scripts/time_episodes.py`
- Create: `tests/test_timing_report.py`
- Generated: `results/measurements/timing.json`
- Modify: `docs/superpowers/specs/2026-09-17-pick-lift-selection-design.md` (§10)

**Interfaces:**
- Consumes: `PickTeacher`, `PickEpisodeRunner`, `load_rules`, `make_pick_task`, `runtime_versions` (`snapshot.py:124`).
- Produces: `time_episodes(sim, seeds, cells, *, with_video: bool) -> dict` with `episodes` (per episode: `seed`, `cell`, `wall_s`, `control_steps`, `observe_s_total`, `inference_s_total`, `success`), `summary` (`mean_wall_s`, `p95_wall_s`, `max_wall_s`), `projection` (`episodes_400_hours`, `episodes_496_hours`), `runtime`.

Why this task exists: the 7-hour figure for the final test is an estimate nobody measured. The final run is 400 main-test episodes plus 96 wording-test episodes on the laptop's Intel iGPU, and the schedule depends on the real number (§10).

- [ ] **Step 1: Write the failing test**

Create `tests/test_timing_report.py`:

```python
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("time_episodes", ROOT / "scripts" / "time_episodes.py")
time_episodes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(time_episodes)


class ProjectionTests(unittest.TestCase):
    """The projection is arithmetic and must not quietly use the fastest episode."""

    def test_the_projection_uses_the_mean_and_states_both_test_sizes(self):
        episodes = [{"wall_s": 10.0}, {"wall_s": 20.0}, {"wall_s": 30.0}, {"wall_s": 40.0}]
        projection = time_episodes.project(episodes)
        self.assertAlmostEqual(projection["episodes_400_hours"], 400 * 25.0 / 3600, places=6)
        self.assertAlmostEqual(projection["episodes_496_hours"], 496 * 25.0 / 3600, places=6)

    def test_a_worst_case_projection_is_reported_too(self):
        episodes = [{"wall_s": 10.0}, {"wall_s": 100.0}]
        projection = time_episodes.project(episodes)
        self.assertGreater(projection["episodes_400_hours_worst"], projection["episodes_400_hours"])

    def test_an_empty_run_cannot_be_projected(self):
        with self.assertRaises(ValueError):
            time_episodes.project([])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_timing_report.py"`
Expected: FAIL — `scripts/time_episodes.py` does not exist.

- [ ] **Step 3: Write the timing script**

Create `scripts/time_episodes.py`. It must:

- run four complete episodes (one per cell) on dev scenes, including simulation reset, rendering and the policy's own time — the whole episode as the final test will run it, not just the physics;
- measure wall-clock seconds per episode with `time.perf_counter()`, and keep the `observe_s` and `inference_s` series that `PickEpisodeRunner` already records in `record["timing"]`;
- expose `project(episodes) -> dict` with `episodes_400_hours` and `episodes_496_hours` from the **mean**, and `episodes_400_hours_worst` / `episodes_496_hours_worst` from the **maximum**, raising `ValueError` on an empty list;
- record `runtime_versions()` so the timing is tied to the machine that produced it;
- write `results/measurements/timing.json` and print one line per episode plus both projections.

Run it once with `--video` and once without, because rendering video is a large part of the cost and the final test may or may not record it.

- [ ] **Step 4: Run the timing measurement**

```powershell
$env:PYTHONPATH = "src"
.venv-sim/Scripts/python.exe scripts/time_episodes.py
.venv-sim/Scripts/python.exe scripts/time_episodes.py --video
```
Expected: per-episode wall-clock times and both projections, with and without video.

- [ ] **Step 5: Replace the estimate in the spec**

In §10, replace the 7-hour estimate with the measured mean and worst-case projections for 400 and 496 episodes, with and without video, naming the machine (Intel Core i5-6300U, HD Graphics 520) and the date. If the measured projection is far from 7 hours, say so plainly in the task report — the schedule depends on it.

- [ ] **Step 6: Run the whole suite and commit**

```powershell
git add scripts/time_episodes.py tests/test_timing_report.py docs/superpowers/specs/2026-09-17-pick-lift-selection-design.md
git add -f results/measurements/timing.json
git commit -m "Pick plan 2 task 8: measured episode time and the projection for the final test"
```

- [ ] **Step 7: Report to the owner**

Report in simple English:

- the gate result: successes out of 100, per cell, scenes passing all four, rejected starts, failed attempts;
- the measured force numbers and the limits chosen from them;
- the hold tolerances and the teacher distribution they came from;
- the measured deadline, with the teacher's p95 and maximum;
- the frozen list sizes and the duplicate-check result, including the honest note about incomplete legacy provenance;
- the measured episode time and what the final test will really cost;
- every point where a step said "stop and report", and what was found;
- confirmation that no final-test episode was ever executed, and that the owner's documentation files are still uncommitted.

State that Plan 3 (four-cell data generation, provenance, balance and acceptance-bias reports, the pilot gate, then full data) is next and will be written only after this report.

---

## Self-review against the spec

Checked after writing, as the writing-plans skill requires.

**Spec coverage for the sections Plan 2 owns:**

| Spec requirement | Task |
|---|---|
| §5 force limit from controlled measurements, severe limit higher, teacher must satisfy it | 3 |
| §5 hold tolerances calibrated on the teacher | 4 |
| §5 right-jaw grasp shapes confirmed against real grasps (§13 step 2) | 2 |
| §7 pick-only teacher restricted to approach, grasp, lift, hold; IK only inside the teacher | 1 |
| §7 gate of 25 × 4 with ≥99/100 and ≥24/25 per cell, plus the extra reported numbers | 5 |
| §7 timeout checked against teacher p95 **and** maximum | 5 |
| §8 seed blocks, duplicate check, separate frozen lists with hashes, honest legacy note | 6 |
| §8 final-test lists never run during development; `--final` plus frozen manifest | 6, 7 |
| §11 run directory, manifest, attempts, per-episode files including invalid attempts | 7 |
| §11 summary names the attempt supplying each result; `evaluation_valid` rules; incomplete notice | 7 |
| §11 snapshot record, strict for final runs | 7 |
| §11 resume only on a matching experiment | 7 |
| §10 timed four-episode laptop run replacing the 7-hour estimate | 8 |

**Deliberately not in Plan 2** (they belong to later plans, per the plan series table): data generation and provenance (Plan 3), wordings beyond `T1` in the teacher path (Plan 3), training, export, parity and the paired native-versus-OpenVINO comparison (Plan 4), and the final test itself (Plan 5).

**Known gaps to settle during execution, not silently:**

1. Task 4 and Task 5 read `record["outcome"]["hold"]` keys (`shift_m`, `turn_deg`, `speed_mps`, `angular_speed_rps`, `both_jaw_fraction`, `single_jaw_gap_s`, `started_control_step`). Plan 1's `PickJudge` may name them differently, or may not record `started_control_step` at all. Both tasks say to check `pick_outcome.py` first and to add a missing measurement additively with its own test. Nothing here may be renamed to make a number look better.
2. Task 3's descent uses the right arm's `elbow_flex` and `wrist_flex` joints to lower the jaws. If those joints do not lower the jaws onto the table on this robot, the script is wrong and step 4 says to stop — the fix is the joint choice, never the force numbers.
3. The gate's `--first-seed` default (3,100,000) must match `dev_gate` in Task 6's frozen list. Task 6 runs after Task 5, so the gate is re-run against the frozen list at the start of Task 7 if the frozen `dev_gate` seeds differ from the ones the gate used. Record both seed sets in the report.
