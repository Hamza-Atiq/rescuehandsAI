# Pick-Lift Milestone — Plan 1: Evaluation Foundations and Physics v2

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the tested foundations for the pick-and-lift milestone: the frozen physics-version-1 reference, physics version 2 with the slip test, shape-level contact classification, the four scene cells, scene identity, the success-rule judge, run validity and attempts, the snapshot record, grouped statistics, and a pick episode runner. All of this is proven with a trivial policy.

**Architecture:** New flat modules `src/rescuehandsai/pick_*.py` sit next to the existing full-task code. The existing code changes only by adding keyword arguments whose defaults keep today's behaviour (`physics_version=1`, `params=None`, `on_substep=None`, `stop_on_cross_arm=True`). The judge is pure Python over per-physics-step facts, so every rule is unit-tested without MuJoCo. MuJoCo-facing pieces (contact classifier, fact reader, start check, runner) get simulation tests.

**Tech Stack:** Python 3.12, MuJoCo 3.13.0, NumPy, `unittest` (the repo has no pytest), Windows PowerShell, `.venv-sim`.

**Spec:** `docs/superpowers/specs/2026-09-17-pick-lift-selection-design.md` (commits `ab58560`, `2fb1a0b`). Read it before starting. Section numbers below (§) refer to it.

## Plan series

The spec's order of work (§13) is split into five plans. Each one produces working, tested software. Each later plan is written only after the previous plan works.

| Plan | Covers spec §13 steps | Output |
|---|---|---|
| **1 (this file)** | 0, 1, and the code half of 2 | Foundations and physics v2, proven with a trivial policy |
| 2 | rest of 2, 3, 4, 5 | Force measurements, pick-only teacher, jaw-shape evidence, hold calibration, teacher gate, frozen scene lists, `evaluate_pick.py` CLI with `--final` guard, timed 4-episode run |
| 3 | 6, 7 | Four-cell data generator, provenance, balance and acceptance-bias reports, pilot gate, full data |
| 4 | 8, 9 | Training manifest and trainable-parameter check, both training runs, checkpoint screening and selection, export, saved-noise parity, paired native vs OpenVINO |
| 5 | 10 | Freeze, final main test and wording test, reports |

## Global Constraints

- All changes are additive. Do not delete or rewrite existing behaviour. Existing full-task commands keep `physics_version = 1` by default and build a byte-identical world XML (§2).
- Only pick-task code passes `physics_version=2` explicitly (§2).
- **Commit after each task. Never push to GitHub** (owner instruction).
- Run tests from the project root in PowerShell with `$env:PYTHONPATH = "src"` and `.venv-sim/Scripts/python.exe -m unittest ...`.
- Never run final-test scene lists (seed blocks from 3,200,000 upward) during development (§8).
- Contacts are classified by collision shape (geom), never by whole body (§5).
- Right-jaw grasp shapes, exactly (§5): fixed `fixed_jaw_box3`…`fixed_jaw_box7`, `fixed_jaw_sph_tip1`…`fixed_jaw_sph_tip3`; moving `moving_jaw_box2`, `moving_jaw_box3`, `moving_jaw_sph_tip1`…`moving_jaw_sph_tip3`.
- Decorative shapes (`contype=0`): `mat`, `cup_zone`, `utensil_zone`, `plate_rim`, `cup_top` (§5).
- Invalid labels are exactly `SIM_ERROR`, `MODEL_LOAD_ERROR`, `CONTRACT_MISMATCH`. Policy exceptions are valid `POLICY_ERROR` failures (§6).
- One diagnosed retry per invalid attempt. A second invalid attempt blocks. Valid failures are never retried (§6).
- Text files are hashed with CRLF normalised to LF. This checkout converts line endings (`core.autocrlf`), and Kaggle checkouts do not.
- Starting values in `configs/pick_rules.json` and `configs/pick_contacts.json` are **not frozen** in this plan (`"frozen": false`, force limits `null`). Plan 2 calibrates and freezes them.

## Facts measured on this machine (2026-09-17)

- `model.geom("missing")` raises `KeyError`. `mujoco.FatalError` exists. `data.warning[i].number` is writable.
- `mujoco.mjtWarning` members include `mjWARN_INERTIA`, `mjWARN_CONTACTFULL`, `mjWARN_CNSTRFULL`, `mjWARN_BADQPOS`, `mjWARN_BADQVEL`, `mjWARN_BADQACC`, `mjWARN_BADCTRL`.
- Robot geom names carry the arm prefix, for example `right_arm/fixed_jaw_box5`. Scene geoms do not (`fork_handle`).
- In `right_arm/gripper` the collision shapes are: an unnamed box (group 3, priority 0, the housing), `fixed_jaw_box1`/`box2`/`box4`–`box7` (boxes), `fixed_jaw_box3` (capsule), `fixed_jaw_sph_tip1`–`3` (spheres), and one unnamed mesh (group 4, priority 1).
- Parent-body filtering is on (`mjDSBL_FILTERPARENT` not set).
- Version-1 contacts: `table`–`fork_handle` friction 1.065 (the larger of table 0.8 and the fork's sampled value). Gripper contacts use the gripper's friction 1.0 because of `priority="1"`.
- Test runner: `unittest`. `.venv-sim` has no pytest.

## File map

| File | Status | Responsibility |
|---|---|---|
| `scripts/record_v1_reference.py` | create | Save v1 world-XML hashes for seeds 0–9 |
| `scripts/check_v1_regression.py` | create | Compare two scripted full-task result folders, episode by episode |
| `tests/data/physics_v1_reference.json` | create (generated) | v1 world-XML hashes |
| `results/v1_reference_pre_pick/` | create (generated) | Scripted outcomes, seeds 0–9, taken before code changes |
| `src/rescuehandsai/scene.py` | modify | `physics_version` for `world_xml`, `build_model`, `_utensil_xml` |
| `src/rescuehandsai/sim.py` | modify | `physics_version`, `reset(params=)`, `step(on_substep=, stop_on_cross_arm=)` |
| `src/rescuehandsai/showcase.py` | modify | Pass the physics version through |
| `configs/pick_rules.json` | create | Success-rule starting values and start-check limits |
| `configs/pick_contacts.json` | create | Grasp shapes, scene shapes, decorative shapes, force limits |
| `configs/pick_seed_blocks.json` | create | New seed blocks and the legacy range |
| `src/rescuehandsai/pick_config.py` | create | Load configs; derive step counts from time steps |
| `src/rescuehandsai/pick_contacts.py` | create | `ContactClassifier`, `ContactVerdict` |
| `src/rescuehandsai/pick_cells.py` | create | Cells, templates, `PickTask`, start check |
| `src/rescuehandsai/pick_identity.py` | create | Config/scene/settings hashes, duplicates, seed blocks |
| `src/rescuehandsai/pick_outcome.py` | create | `SubstepFacts`, `PickJudge`, `PickOutcome` |
| `src/rescuehandsai/pick_facts.py` | create | `FactReader`, `SimulatorFailure` |
| `src/rescuehandsai/pick_records.py` | create | `InvalidRun`, `AttemptLedger`, contract and resume checks |
| `src/rescuehandsai/snapshot.py` | create | Snapshot record |
| `src/rescuehandsai/pick_stats.py` | create | Scene-grouped summary and pass bars |
| `src/rescuehandsai/pick_runner.py` | create | `PickEpisodeRunner` |
| `tests/test_physics_v1_reference.py`, `tests/test_physics_v2.py`, `tests/test_pick_config.py`, `tests/test_pick_contacts.py`, `tests/test_pick_cells.py`, `tests/test_pick_identity.py`, `tests/test_pick_outcome.py`, `tests/test_pick_facts.py`, `tests/test_pick_records.py`, `tests/test_snapshot.py`, `tests/test_pick_stats.py`, `tests/test_pick_runner.py` | create | Tests |

---

### Task 1: Record the physics-version-1 reference (before any code change)

**Files:**
- Create: `scripts/record_v1_reference.py`
- Create: `scripts/check_v1_regression.py`
- Create: `tests/test_physics_v1_reference.py`
- Generated: `tests/data/physics_v1_reference.json`, `results/v1_reference_pre_pick/`

**Interfaces:**
- Consumes: `rescuehandsai.scene.load_config`, `sample_params`, `world_xml` (current signatures).
- Produces: `check_v1_regression.compare(reference: Path, candidate: Path) -> list[str]`; reference file shape `{"git_revision": str, "world_xml_sha256": {"0": hex, ... "9": hex}, "identity": {relative path: hex}}`. `identity` covers `configs/scene.json`, `configs/simulation.json`, the robot XML and every file in its `assets/` folder.

**Wording rule:** a passing comparison means the new run **matched the recorded results** (every episode record field except wall-clock timings, plus the asset and configuration identities). It does not prove that all behaviour is identical, and reports must not claim that.

- [ ] **Step 1: Confirm the working tree has no source changes**

Run: `git status --porcelain -- src scripts training configs tests`
Expected: no output. If anything is listed, stop and ask the owner. The reference must be taken from committed code.

- [ ] **Step 2: Write the reference recorder**

`scripts/record_v1_reference.py`:

```python
"""Record the physics-version-1 reference before any pick-milestone code change (spec §12).

Writes tests/data/physics_v1_reference.json: the world XML SHA-256 for seeds 0-9, the
identity of the scene/simulation configs and the robot asset (XML and every mesh), and
the git revision it was taken at. Scripted outcomes come from
  scripts/evaluate.py --policy scripted --seeds 0:10 --supervisor on --name v1_reference_pre_pick
"""
import hashlib
import json
import subprocess

from rescuehandsai.scene import ROOT, load_config, sample_params, world_xml

TEXT_SUFFIXES = {".json", ".xml", ".py"}


def digest(path) -> str:
    data = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES:
        data = data.replace(b"\r\n", b"\n")  # this checkout converts line endings
    return hashlib.sha256(data).hexdigest()


def identity_files() -> list:
    sim_config = json.loads((ROOT / "configs/simulation.json").read_text())
    asset = ROOT / sim_config["asset_path"]
    meshes = sorted(p for p in (asset.parent / "assets").rglob("*") if p.is_file())
    return [ROOT / "configs/scene.json", ROOT / "configs/simulation.json", asset, *meshes]


def identity() -> dict:
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in identity_files()}


def main():
    config = load_config()
    hashes = {str(seed): hashlib.sha256(world_xml(sample_params(config, seed), config).encode()).hexdigest()
              for seed in range(10)}
    revision = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    out = ROOT / "tests" / "data" / "physics_v1_reference.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"git_revision": revision, "world_xml_sha256": hashes, "identity": identity()},
                              indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Record both references**

Run:
```powershell
$env:PYTHONPATH = "src"
.venv-sim/Scripts/python.exe scripts/record_v1_reference.py
.venv-sim/Scripts/python.exe scripts/evaluate.py --policy scripted --seeds 0:10 --supervisor on --name v1_reference_pre_pick
```
Expected: the JSON path is printed; ten `{"seed": ..., "state": ...}` lines; `results/v1_reference_pre_pick/episode_0.json` … `episode_9.json` and `summary.json` exist. This takes about 1–3 minutes.

- [ ] **Step 4: Write the regression comparer**

`scripts/check_v1_regression.py`:

```python
"""Compare a fresh scripted full-task evaluation with the pre-pick reference, episode by episode.

Usage (project root):
  scripts/check_v1_regression.py results/v1_reference_pre_pick results/<fresh run>
Exit code 0 only when the candidate matched the recorded results: every episode record field
(outcome, events, recoveries, steps, sim time, ...) except wall-clock timings, and the robot
asset and configuration identities in manifest.json. A match is evidence, not proof that all
behaviour is identical.
"""
import argparse
import json
from pathlib import Path

WALL_CLOCK_FIELDS = ("wall_seconds", "inference_seconds")  # machine speed, not behaviour
MANIFEST_IDENTITY = ("asset_sha256", "scene_config", "sim_config", "seeds", "args")


def _without_timing(record: dict) -> dict:
    return {k: v for k, v in record.items() if k not in WALL_CLOCK_FIELDS}


def compare(reference: Path, candidate: Path) -> list[str]:
    reference, candidate = Path(reference), Path(candidate)
    ref_files = sorted(reference.glob("episode_*.json"))
    if not ref_files:
        raise ValueError(f"{reference} has no episode files")
    problems = []
    ref_manifest, cand_manifest = reference / "manifest.json", candidate / "manifest.json"
    if not ref_manifest.is_file() or not cand_manifest.is_file():
        problems.append("manifest.json: missing from reference or candidate")
    else:
        ref_m, cand_m = json.loads(ref_manifest.read_text()), json.loads(cand_manifest.read_text())
        for key in MANIFEST_IDENTITY:
            ref_value, cand_value = ref_m.get(key), cand_m.get(key)
            if key == "args":  # the run name differs by design
                ref_value = {k: v for k, v in (ref_value or {}).items() if k != "name"}
                cand_value = {k: v for k, v in (cand_value or {}).items() if k != "name"}
            if ref_value != cand_value:
                problems.append(f"manifest.json: {key} differs")
    for ref_file in ref_files:
        cand_file = candidate / ref_file.name
        if not cand_file.is_file():
            problems.append(f"{ref_file.name}: missing from candidate")
            continue
        ref = _without_timing(json.loads(ref_file.read_text()))
        cand = _without_timing(json.loads(cand_file.read_text()))
        for field in sorted(set(ref) | set(cand)):
            if ref.get(field) != cand.get(field):
                problems.append(f"{ref_file.name}: {field} {ref.get(field)!r} != {cand.get(field)!r}")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    problems = compare(args.reference, args.candidate)
    for line in problems:
        print(line)
    print("v1 regression:", "FAILED" if problems else "matched recorded results")
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Write the tests**

`tests/test_physics_v1_reference.py`:

```python
"""Physics version 1 must stay byte-identical for every existing full-task command (spec §2, §12)."""
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from rescuehandsai.scene import ROOT, load_config, sample_params, world_xml

REFERENCE = ROOT / "tests" / "data" / "physics_v1_reference.json"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PhysicsV1ReferenceTests(unittest.TestCase):
    def test_default_world_xml_matches_pre_pick_reference(self):
        reference = json.loads(REFERENCE.read_text())
        config = load_config()
        self.assertEqual(len(reference["world_xml_sha256"]), 10)
        for seed, digest in reference["world_xml_sha256"].items():
            xml = world_xml(sample_params(config, int(seed)), config)
            self.assertEqual(hashlib.sha256(xml.encode()).hexdigest(), digest, f"seed {seed}")

    def test_asset_and_config_identity_is_preserved(self):
        reference = json.loads(REFERENCE.read_text())
        self.assertIn("configs/scene.json", reference["identity"])
        self.assertTrue(any(name.endswith(".stl") for name in reference["identity"]))
        self.assertEqual(load_script("record_v1_reference").identity(), reference["identity"])


class RegressionCompareTests(unittest.TestCase):
    MANIFEST = {"asset_sha256": "a", "scene_config": {"x": 1}, "sim_config": {"y": 2}, "seeds": [0, 1],
                "args": {"policy": "scripted", "name": "run"}}

    def folder(self, tmp: str, manifest=None) -> Path:
        path = Path(tmp)
        (path / "manifest.json").write_text(json.dumps(manifest or self.MANIFEST))
        return path

    def write(self, folder: Path, seed: int, **fields):
        record = {"state": "SUCCEEDED", "failure": None, "steps": 100, "recoveries": 0, "sim_seconds": 5.0,
                  "events": [{"label": "OBJECT_DROPPED", "time": 1.5}], "outcome": {"cup": True},
                  "wall_seconds": 1.0, "inference_seconds": [0.1]}
        record.update(fields)
        (folder / f"episode_{seed}.json").write_text(json.dumps(record))

    def test_matching_records_pass_and_wall_time_and_run_name_are_ignored(self):
        compare = load_script("check_v1_regression").compare
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            renamed = dict(self.MANIFEST, args={"policy": "scripted", "name": "other"})
            ref, cand = self.folder(a), self.folder(b, renamed)
            self.write(ref, 0)
            self.write(cand, 0, wall_seconds=9.0, inference_seconds=[0.5])
            self.assertEqual(compare(ref, cand), [])

    def test_any_outcome_or_event_difference_is_reported(self):
        compare = load_script("check_v1_regression").compare
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            ref, cand = self.folder(a), self.folder(b)
            self.write(ref, 0)
            self.write(ref, 1)
            self.write(ref, 2)
            self.write(cand, 0, events=[{"label": "OBJECT_DROPPED", "time": 1.55}])  # same counts, other physics
            self.write(cand, 1, outcome={"cup": False})
            problems = compare(ref, cand)
            self.assertTrue(any(p.startswith("episode_0.json: events") for p in problems), problems)
            self.assertTrue(any(p.startswith("episode_1.json: outcome") for p in problems), problems)
            self.assertIn("episode_2.json: missing from candidate", problems)

    def test_asset_or_config_identity_difference_is_reported(self):
        compare = load_script("check_v1_regression").compare
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            ref = self.folder(a)
            cand = self.folder(b, dict(self.MANIFEST, asset_sha256="changed"))
            self.write(ref, 0)
            self.write(cand, 0)
            self.assertEqual(compare(ref, cand), ["manifest.json: asset_sha256 differs"])

    def test_empty_reference_is_an_error(self):
        compare = load_script("check_v1_regression").compare
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            with self.assertRaises(ValueError):
                compare(Path(a), Path(b))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 6: Run the tests**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_physics_v1_reference.py" -v`
Expected: 6 tests, all `ok`.

Also run the comparer on the reference against itself as a sanity check:
`.venv-sim/Scripts/python.exe scripts/check_v1_regression.py results/v1_reference_pre_pick results/v1_reference_pre_pick`
Expected: `v1 regression: matched recorded results`.

- [ ] **Step 7: Commit (do not push)**

```powershell
git add scripts/record_v1_reference.py scripts/check_v1_regression.py tests/test_physics_v1_reference.py tests/data/physics_v1_reference.json results/v1_reference_pre_pick
git commit -m "Pick plan 1 task 1: record the physics-version-1 reference before any change"
```

---

### Task 2: Physics version plumbing and the version-2 friction fix

**Files:**
- Modify: `src/rescuehandsai/scene.py` (`_utensil_xml` line 85, `world_xml` line 151, `build_model` line 207)
- Modify: `src/rescuehandsai/sim.py` (`__init__`, `reset`)
- Modify: `src/rescuehandsai/showcase.py` (`scene_identity` line 20, `ShowcaseRenderer._sync` line 35)
- Test: `tests/test_physics_v2.py`

**Interfaces:**
- Consumes: Task 1 reference file.
- Produces:
  - `scene.PHYSICS_VERSIONS = (1, 2)`
  - `world_xml(params, config, showcase=False, physics_version=1) -> str`
  - `build_model(params, config=None, asset_path=None, showcase=False, physics_version=1)`
  - `MujocoSimulation(config_path=None, scene_config_path=None, seed=0, physics_version=1)` with attribute `sim.physics_version`
  - `MujocoSimulation.reset(seed, instruction=None, *, params: SceneParams | None = None)`. When `params` is given, `params.seed` must equal `seed`.

**Design of the fix (§4):** in version 2 each utensil geom gets `priority="2"`, `condim="6"` and `friction="<sampled> 5e-3 5e-4"`. The higher priority makes the utensil's sampled sliding friction govern jaw contacts. It also governs table contacts, which were `max(table, utensil)` before. `condim` and the torsional/rolling values copy what the gripper (`collision_gripper`: `condim="6" friction="1 5e-3 5e-4"`) already imposed on jaw contacts. So in jaw contacts, only sliding friction changes. The cup and the version-1 world are unchanged.

- [ ] **Step 1: Write the failing tests**

`tests/test_physics_v2.py`:

```python
"""Physics version 2: the utensil's sampled friction governs its contacts (spec §4)."""
from dataclasses import replace
import hashlib
import json
import unittest

import numpy as np

from rescuehandsai.contracts import BimanualAction
from rescuehandsai.scene import ROOT, build_model, load_config, sample_params, world_xml
from rescuehandsai.showcase import scene_identity
from rescuehandsai.sim import MujocoSimulation


class PhysicsVersionPlumbingTests(unittest.TestCase):
    def test_version_1_default_and_explicit_match_reference(self):
        reference = json.loads((ROOT / "tests/data/physics_v1_reference.json").read_text())
        config = load_config()
        for seed, digest in reference["world_xml_sha256"].items():
            params = sample_params(config, int(seed))
            self.assertEqual(hashlib.sha256(world_xml(params, config, physics_version=1).encode()).hexdigest(), digest)

    def test_version_2_changes_only_utensil_geoms(self):
        config = load_config()
        params = sample_params(config, 3)
        v1 = world_xml(params, config).splitlines()
        v2 = world_xml(params, config, physics_version=2).splitlines()
        self.assertEqual(len(v1), len(v2))
        changed = [(a, b) for a, b in zip(v1, v2) if a != b]
        self.assertTrue(changed)
        for _, line in changed:
            self.assertTrue(any(f'name="{n}' in line for n in ("fork_", "spoon_")), line)
            self.assertIn('priority="2"', line)
            self.assertIn('condim="6"', line)

    def test_unknown_version_is_rejected(self):
        config = load_config()
        with self.assertRaises(ValueError):
            world_xml(sample_params(config, 0), config, physics_version=3)
        with self.assertRaises(ValueError):
            MujocoSimulation(physics_version=0)

    def test_simulation_records_version_and_scene_identity_follows_it(self):
        v1, v2 = MujocoSimulation(), MujocoSimulation(physics_version=2)
        try:
            self.assertEqual((v1.physics_version, v2.physics_version), (1, 2))
            self.assertNotEqual(scene_identity(v1), scene_identity(v2))
        finally:
            v1.close()
            v2.close()

    def test_reset_accepts_explicit_params_with_matching_seed(self):
        sim = MujocoSimulation(physics_version=2)
        try:
            config = sim.scene_config
            params = sample_params(config, 7)
            params = replace(params, frictions={**params.frictions, "fork": 0.2})
            sim.reset(7, params=params)
            self.assertEqual(sim.scene_params.frictions["fork"], 0.2)
            with self.assertRaises(ValueError):
                sim.reset(8, params=params)
        finally:
            sim.close()


class ContactFrictionTests(unittest.TestCase):
    """The friction MuJoCo actually uses in a jaw-utensil contact."""

    def jaw_contact_friction(self, physics_version, fork_friction):
        sim = MujocoSimulation(physics_version=physics_version)
        try:
            params = sample_params(sim.scene_config, 11)
            params = replace(params, frictions={**params.frictions, "fork": fork_friction})
            sim.reset(11, params=params)
            model, data = sim.model, sim.data
            jaw = model.geom("right_arm/fixed_jaw_box5").id
            handle = model.geom("fork_handle").id
            body = int(model.geom_bodyid[jaw])
            target = data.geom_xpos[handle]
            model.geom_pos[jaw] = data.xmat[body].reshape(3, 3).T @ (target - data.xpos[body])
            import mujoco
            mujoco.mj_forward(model, data)
            for contact in data.contact:
                if {int(contact.geom1), int(contact.geom2)} == {jaw, handle}:
                    return float(contact.friction[0])
            self.fail("no jaw-handle contact was created")
        finally:
            sim.close()

    def test_version_1_ignores_utensil_friction(self):
        self.assertAlmostEqual(self.jaw_contact_friction(1, 0.2), 1.0, places=6)
        self.assertAlmostEqual(self.jaw_contact_friction(1, 1.2), 1.0, places=6)

    def test_version_2_uses_utensil_friction(self):
        self.assertAlmostEqual(self.jaw_contact_friction(2, 0.2), 0.2, places=6)
        self.assertAlmostEqual(self.jaw_contact_friction(2, 1.2), 1.2, places=6)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_physics_v2.py" -v`
Expected: errors such as `TypeError: world_xml() got an unexpected keyword argument 'physics_version'`.

- [ ] **Step 3: Implement in `scene.py`**

Add below `SCENE_ITEMS = ("cup",) + UTENSILS`:

```python
# Physics version 1 is the hackathon world and stays byte-identical. In version 1 the
# SO-101 gripper shapes (priority="1", friction 1) win every jaw contact, so the
# utensil's sampled friction never matters. Version 2 gives utensil shapes priority 2,
# so their sampled sliding friction governs jaw and table contacts; condim and the
# torsional/rolling values copy what the gripper already imposed on jaw contacts.
PHYSICS_VERSIONS = (1, 2)


def check_physics_version(physics_version: int) -> int:
    if physics_version not in PHYSICS_VERSIONS:
        raise ValueError(f"physics_version must be one of {PHYSICS_VERSIONS}, got {physics_version!r}")
    return physics_version
```

Change `_utensil_xml`'s signature and its `common` line:

```python
def _utensil_xml(item: str, params: SceneParams, config: dict, physics_version: int = 1) -> str:
    s = params.utensil_scale[item]
    hx, hy, hz = config["utensil"]["handle_half"]
    hx *= s
    x, y, yaw = params.poses[item]
    mass, fr = params.masses[item], params.frictions[item]
    if physics_version == 1:
        common = f'friction="{fr:.4g} 0.05 0.002" condim="4" solref="0.01 1"'
    else:
        common = f'friction="{fr:.4g} 5e-3 5e-4" condim="6" solref="0.01 1" priority="2"'
```
(the rest of `_utensil_xml` is unchanged)

Change `world_xml`'s signature, add the check as its first line, and pass the version to the utensil call:

```python
def world_xml(params: SceneParams, config: dict, showcase: bool = False, physics_version: int = 1) -> str:
    """The MJCF world. showcase=True adds presentation-only extras; the default output is unchanged."""
    check_physics_version(physics_version)
```
and replace
`{"".join(_utensil_xml(item, params, config) for item in UTENSILS)}`
with
`{"".join(_utensil_xml(item, params, config, physics_version) for item in UTENSILS)}`.

Change `build_model`:

```python
def build_model(params: SceneParams, config: dict | None = None, asset_path: Path | None = None,
                showcase: bool = False, physics_version: int = 1):
```
and its `from_string` line to
`spec = mujoco.MjSpec.from_string(world_xml(params, config, showcase=showcase, physics_version=physics_version))`.

- [ ] **Step 4: Implement in `sim.py`**

Import `check_physics_version` alongside the existing scene imports:
`from .scene import ROOT, SCENE_ITEMS, build_model, check_physics_version, load_config, sample_params`

Constructor signature and first line:

```python
    def __init__(self, config_path=None, scene_config_path=None, seed: int = 0, physics_version: int = 1):
        self.physics_version = check_physics_version(physics_version)
        self.config = json.loads(Path(config_path or ROOT / "configs/simulation.json").read_text())
```

`reset`:

```python
    def reset(self, seed: int, instruction: str | None = None, *, params=None):
        """params: explicit SceneParams (pick cells); its seed must equal `seed`."""
        if params is not None and params.seed != seed:
            raise ValueError(f"params were sampled for seed {params.seed}, not {seed}")
        self.seed = seed
        self.instruction = instruction or self.config["instruction"]
        self.scene_params = params if params is not None else sample_params(self.scene_config, seed)
        self.close()
        self.model = build_model(self.scene_params, self.scene_config, self.asset_path,
                                 physics_version=self.physics_version)
```
(the rest of `reset` is unchanged)

- [ ] **Step 5: Implement in `showcase.py`**

Line 20:
`digest = hashlib.sha256(world_xml(sim.scene_params, sim.scene_config, physics_version=getattr(sim, "physics_version", 1)).encode())`

Line 35:
`self.model = build_model(sim.scene_params, sim.scene_config, sim.asset_path, showcase=True, physics_version=getattr(sim, "physics_version", 1))`

- [ ] **Step 6: Run the new tests, then the whole suite**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_physics_v2.py" -v`
Expected: 7 tests `ok`.

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests`
Expected: `OK`. Every previously passing test still passes.

- [ ] **Step 7: Commit (do not push)**

```powershell
git add src/rescuehandsai/scene.py src/rescuehandsai/sim.py src/rescuehandsai/showcase.py tests/test_physics_v2.py
git commit -m "Pick plan 1 task 2: physics_version plumbing; v2 utensil friction governs contacts"
```

---

### Task 3: Slip test with a real grasp and lift

**Files:**
- Modify: `tests/test_physics_v2.py` (append a class)

**Interfaces:**
- Consumes: `MujocoSimulation(physics_version=2)`, `reset(params=)`, `ScriptedExpert(sim, task, subtasks=("pick_utensil",))`, `make_task(seed, utensil="fork", template=0)`, `expert.phase`.
- Produces: evidence for §4 part (b) only. No new API.

- [ ] **Step 1: Write the test**

Append to `tests/test_physics_v2.py`:

```python
from rescuehandsai.expert import ScriptedExpert
from rescuehandsai.task import make_task


def fork_in_gripper_frame(sim):
    site = sim.data.site("right_arm/gripperframe")
    rotation = site.xmat.reshape(3, 3)
    return rotation.T @ (sim.data.body("fork").xpos - site.xpos)


def lift_slip(physics_version, seed, friction, max_steps=160):
    """Teacher grasp on the fork; return (slip in gripper frame over the first lift, fork rise).

    Returns None when the teacher never reaches its first lift without staging."""
    sim = MujocoSimulation(physics_version=physics_version)
    try:
        params = sample_params(sim.scene_config, seed)
        params = replace(params, frictions={**params.frictions, "fork": friction})
        sim.reset(seed, params=params)
        expert = ScriptedExpert(sim, make_task(seed, utensil="fork", template=0), subtasks=("pick_utensil",))
        before = start_z = None
        for _ in range(max_steps):
            action = expert.act(sim.observe())
            if expert.phase.startswith("stage_"):  # the left arm is staging the utensil: not this test
                return None
            if expert.phase == "utensil_lift" and before is None:
                before, start_z = fork_in_gripper_frame(sim), float(sim.data.body("fork").xpos[2])
            if before is not None and expert.phase != "utensil_lift":
                break
            sim.step(action)
        if before is None:
            return None
        return float(np.linalg.norm(fork_in_gripper_frame(sim) - before)), float(sim.data.body("fork").xpos[2] - start_z)
    finally:
        sim.close()


class SlipTest(unittest.TestCase):
    """Same scene, same scripted grasp and lift; only the fork's friction differs (spec §4 b)."""

    def test_low_friction_slips_more_than_high_friction(self):
        for seed in range(20):
            high = lift_slip(2, seed, 1.2)
            low = lift_slip(2, seed, 0.05)
            if high is None or low is None:
                continue
            high_slip, high_rise = high
            low_slip, low_rise = low
            print(f"seed {seed}: slip high={high_slip:.4f} m rise={high_rise:.4f}; slip low={low_slip:.4f} m rise={low_rise:.4f}")
            self.assertGreater(high_rise, 0.02, "the high-friction grasp must lift the fork")
            self.assertGreater(low_slip, high_slip + 0.005)
            return
        self.fail("no seed in 0-19 reached a lift without staging; investigate the teacher before continuing")
```

- [ ] **Step 2: Run the test**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_physics_v2.py" -k SlipTest -v`
Expected: one printed `seed N: slip high=... low=...` line and `ok`.

If the assertion fails, **stop and report the printed numbers to the owner**. Do not loosen the test and do not change the friction design. A failure means version 2 does not make friction govern slip, and that is a spec-level finding.

- [ ] **Step 3: Commit (do not push)**

```powershell
git add tests/test_physics_v2.py
git commit -m "Pick plan 1 task 3: slip test - low utensil friction slips more under an identical grasp"
```

---

### Task 4: Pick configs and step counts derived from time steps

**Files:**
- Create: `configs/pick_rules.json`, `configs/pick_contacts.json`
- Create: `src/rescuehandsai/pick_config.py`
- Test: `tests/test_pick_config.py`

**Interfaces:**
- Produces: `load_rules(path=None) -> dict`, `load_contacts(path=None) -> dict`, `StepCounts(substeps, hold, max_gap, final_speed, deadline_control)`, `derive_steps(rules, physics_dt, control_dt) -> StepCounts`, `whole_steps(seconds, dt, what) -> int`.

- [ ] **Step 1: Create the config files**

`configs/pick_rules.json`:

```json
{
  "physics_version": 2,
  "frozen": false,
  "hold_s": 1.0,
  "lift_height_m": 0.05,
  "both_jaw_fraction": 0.8,
  "max_single_jaw_gap_s": 0.1,
  "hold_max_shift_m": 0.01,
  "hold_max_turn_deg": 10.0,
  "final_speed_window_s": 0.25,
  "final_max_speed_mps": 0.02,
  "final_max_angular_speed_rps": 0.5,
  "spare_max_shift_m": 0.01,
  "spare_max_yaw_deg": 5.0,
  "cup_max_shift_m": 0.01,
  "cup_max_tilt_deg": 15.0,
  "workspace_min_z_m": -0.02,
  "deadline_control_steps": 300,
  "start_check": {
    "hold_control_steps": 10,
    "overlap_depth_m": 0.001,
    "max_shift_m": 0.005,
    "max_speed_mps": 0.02
  }
}
```

`configs/pick_contacts.json`:

```json
{
  "grasp_arm": "right_arm",
  "jaw_grasp_geoms": {
    "fixed": ["fixed_jaw_box3", "fixed_jaw_box4", "fixed_jaw_box5", "fixed_jaw_box6", "fixed_jaw_box7",
              "fixed_jaw_sph_tip1", "fixed_jaw_sph_tip2", "fixed_jaw_sph_tip3"],
    "moving": ["moving_jaw_box2", "moving_jaw_box3",
               "moving_jaw_sph_tip1", "moving_jaw_sph_tip2", "moving_jaw_sph_tip3"]
  },
  "scene_geoms": {
    "table": ["table"],
    "plate": ["plate"],
    "cup": ["cup_body"],
    "fork": ["fork_handle", "fork_neck", "fork_prong0", "fork_prong1", "fork_prong2"],
    "spoon": ["spoon_handle", "spoon_bowl"]
  },
  "decorative_geoms": ["mat", "cup_zone", "utensil_zone", "plate_rim", "cup_top"],
  "jaw_table_force_limit_n": null,
  "severe_force_limit_n": null,
  "force_limits_frozen": false
}
```

- [ ] **Step 2: Write the failing test**

`tests/test_pick_config.py`:

```python
import unittest

from rescuehandsai.pick_config import StepCounts, derive_steps, load_contacts, load_rules, whole_steps


class PickConfigTests(unittest.TestCase):
    def test_steps_come_from_configured_time_steps(self):
        steps = derive_steps(load_rules(), physics_dt=0.005, control_dt=0.05)
        self.assertEqual(steps, StepCounts(substeps=10, hold=200, max_gap=20, final_speed=50, deadline_control=300))

    def test_other_time_steps_change_the_counts(self):
        steps = derive_steps(load_rules(), physics_dt=0.002, control_dt=0.05)
        self.assertEqual((steps.substeps, steps.hold, steps.max_gap, steps.final_speed), (25, 500, 50, 125))

    def test_non_whole_ratios_are_rejected(self):
        with self.assertRaises(ValueError):
            whole_steps(0.052, 0.005, "control_dt")
        rules = dict(load_rules(), hold_s=1.0025)
        with self.assertRaises(ValueError):
            derive_steps(rules, 0.005, 0.05)

    def test_configs_start_unfrozen(self):
        self.assertFalse(load_rules()["frozen"])
        contacts = load_contacts()
        self.assertFalse(contacts["force_limits_frozen"])
        self.assertIsNone(contacts["jaw_table_force_limit_n"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run it to verify it fails**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_pick_config.py" -v`
Expected: `ModuleNotFoundError: No module named 'rescuehandsai.pick_config'`.

- [ ] **Step 4: Implement**

`src/rescuehandsai/pick_config.py`:

```python
"""Pick-milestone rules and contact settings, with step counts derived from the time steps (spec §5)."""
from dataclasses import dataclass
import json
import math
from pathlib import Path

from .scene import ROOT

RULES_PATH = ROOT / "configs" / "pick_rules.json"
CONTACTS_PATH = ROOT / "configs" / "pick_contacts.json"


def load_rules(path=None) -> dict:
    return json.loads(Path(path or RULES_PATH).read_text())


def load_contacts(path=None) -> dict:
    return json.loads(Path(path or CONTACTS_PATH).read_text())


def whole_steps(seconds: float, dt: float, what: str) -> int:
    ratio = seconds / dt
    if not math.isfinite(ratio) or ratio < 1 or not math.isclose(ratio, round(ratio), abs_tol=1e-9):
        raise ValueError(f"{what}: {seconds} s is not a whole number (>= 1) of {dt} s steps")
    return round(ratio)


@dataclass(frozen=True)
class StepCounts:
    substeps: int          # physics steps per control step
    hold: int              # physics steps in the hold window
    max_gap: int           # longest run of physics steps allowed without both jaws touching
    final_speed: int       # physics steps at the end of the window that must be slow
    deadline_control: int  # control steps before the deadline


def derive_steps(rules: dict, physics_dt: float, control_dt: float) -> StepCounts:
    return StepCounts(
        substeps=whole_steps(control_dt, physics_dt, "control_dt"),
        hold=whole_steps(rules["hold_s"], physics_dt, "hold_s"),
        max_gap=whole_steps(rules["max_single_jaw_gap_s"], physics_dt, "max_single_jaw_gap_s"),
        final_speed=whole_steps(rules["final_speed_window_s"], physics_dt, "final_speed_window_s"),
        deadline_control=int(rules["deadline_control_steps"]),
    )
```

- [ ] **Step 5: Run the test**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_pick_config.py" -v`
Expected: 4 tests `ok`.

- [ ] **Step 6: Commit (do not push)**

```powershell
git add configs/pick_rules.json configs/pick_contacts.json src/rescuehandsai/pick_config.py tests/test_pick_config.py
git commit -m "Pick plan 1 task 4: pick rules/contacts configs; step counts derived from time steps"
```

---

### Task 5: Shape-level contact classifier with detectability tests

**Files:**
- Create: `src/rescuehandsai/pick_contacts.py`
- Test: `tests/test_pick_contacts.py`

**Interfaces:**
- Consumes: `load_contacts()` (Task 4); `MujocoSimulation(physics_version=2)` (Task 2).
- Produces:
  - constants `SCENE_NORMAL`, `JAW_UTENSIL`, `JAW_TABLE`, `VIOLATION`
  - `ContactVerdict(kind: str, labels: tuple[str, ...], geoms: tuple[str, str], force: float, jaw: str | None)`
  - `geom_name(model, g) -> str`, `collides(model, g) -> bool`
  - `ContactClassifier(model, named: str, config: dict)` with `.classify(g1, g2, force) -> ContactVerdict`, `.contacts(data) -> list[ContactVerdict]`, and `.spare`

- [ ] **Step 1: Write the failing tests**

`tests/test_pick_contacts.py`:

```python
"""Every forbidden contact class the rules claim must actually be detected (spec §5, §12)."""
import copy
import unittest

import mujoco
import numpy as np

from rescuehandsai.contracts import BimanualAction
from rescuehandsai.pick_config import load_contacts
from rescuehandsai.pick_contacts import (JAW_TABLE, JAW_UTENSIL, SCENE_NORMAL, VIOLATION, ContactClassifier,
                                         collides, geom_name)
from rescuehandsai.sim import MujocoSimulation


def body_collision_geoms(model, body):
    b = model.body(body).id
    return [g for g in range(model.ngeom) if model.geom_bodyid[g] == b and collides(model, g)]


def move_geom_to(sim, geom, world_point):
    """Test-only: put a geom's centre at a world point, then recompute contacts."""
    model, data = sim.model, sim.data
    body = int(model.geom_bodyid[geom])
    model.geom_pos[geom] = data.xmat[body].reshape(3, 3).T @ (np.asarray(world_point, float) - data.xpos[body])
    mujoco.mj_forward(model, data)


class ContactClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = MujocoSimulation(physics_version=2)
        cls.config = load_contacts()

    @classmethod
    def tearDownClass(cls):
        cls.sim.close()

    def setUp(self):
        self.sim.reset(21)  # rebuilds the model, undoing any moved geom
        self.named = "fork"
        self.clf = ContactClassifier(self.sim.model, self.named, self.config)

    def g(self, name):
        return self.sim.model.geom(name).id

    def verdict_for(self, a, b):
        names = {geom_name(self.sim.model, a), geom_name(self.sim.model, b)}
        found = [v for v in self.clf.contacts(self.sim.data) if set(v.geoms) == names]
        self.assertTrue(found, f"no contact detected between {sorted(names)}")
        return found[0]

    # -- configuration checks ---------------------------------------------------
    def test_config_matches_the_model(self):
        self.assertEqual(self.clf.spare, "spoon")
        for name in self.config["decorative_geoms"]:
            self.assertFalse(collides(self.sim.model, self.g(name)), name)

    def test_collidable_shape_listed_as_decorative_is_rejected(self):
        bad = copy.deepcopy(self.config)
        bad["decorative_geoms"].append("table")
        with self.assertRaises(ValueError):
            ContactClassifier(self.sim.model, "fork", bad)

    def test_missing_geom_is_rejected(self):
        bad = copy.deepcopy(self.config)
        bad["jaw_grasp_geoms"]["fixed"].append("no_such_pad")
        with self.assertRaises(ValueError):
            ContactClassifier(self.sim.model, "fork", bad)

    # -- allowed contacts ----------------------------------------------------------
    def test_items_resting_on_the_table_are_normal(self):
        for _ in range(5):
            self.sim.step(BimanualAction(self.sim.observe().timestamp, self.sim.home_targets))
        verdicts = self.clf.contacts(self.sim.data)
        self.assertTrue(verdicts)
        self.assertTrue(all(v.kind == SCENE_NORMAL for v in verdicts), verdicts)

    def test_fixed_and_moving_grasp_shapes_on_the_named_utensil(self):
        handle = self.g("fork_handle")
        fixed = self.g("right_arm/fixed_jaw_box5")
        move_geom_to(self.sim, fixed, self.sim.data.geom_xpos[handle])
        verdict = self.verdict_for(fixed, handle)
        self.assertEqual((verdict.kind, verdict.jaw, verdict.labels), (JAW_UTENSIL, "fixed", ()))
        self.setUp()
        handle, moving = self.g("fork_handle"), self.g("right_arm/moving_jaw_box2")
        move_geom_to(self.sim, moving, self.sim.data.geom_xpos[handle])
        verdict = self.verdict_for(moving, handle)
        self.assertEqual((verdict.kind, verdict.jaw), (JAW_UTENSIL, "moving"))

    def test_grasp_shape_on_table_is_allowed_below_limit_and_judged_above(self):
        pad = self.g("right_arm/fixed_jaw_box5")
        x, y, _ = self.sim.data.geom_xpos[pad]
        move_geom_to(self.sim, pad, (x, y, -0.002))
        self.assertEqual(self.verdict_for(pad, self.g("table")).kind, JAW_TABLE)
        strict = dict(self.config, jaw_table_force_limit_n=-1.0)
        self.clf = ContactClassifier(self.sim.model, "fork", strict)
        self.assertEqual(self.verdict_for(pad, self.g("table")).labels, ("FORBIDDEN_CONTACT",))
        severe = dict(self.config, severe_force_limit_n=-1.0)
        self.clf = ContactClassifier(self.sim.model, "fork", severe)
        self.assertIn("EXCESS_FORCE", self.verdict_for(pad, self.g("table")).labels)

    # -- forbidden contacts: each one must be detectable ----------------------------
    def test_any_robot_shape_on_the_spare(self):
        spoon = self.g("spoon_handle")
        pad = self.g("right_arm/fixed_jaw_box5")
        move_geom_to(self.sim, pad, self.sim.data.geom_xpos[spoon])
        self.assertEqual(self.verdict_for(pad, spoon).labels, ("WRONG_ITEM_TOUCHED",))

    def test_housing_box_and_jaw_mesh_are_not_grasp_contacts(self):
        model = self.sim.model
        housing = next(g for g in body_collision_geoms(model, "right_arm/gripper")
                       if not model.geom(g).name and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX)
        handle = self.g("fork_handle")
        move_geom_to(self.sim, housing, self.sim.data.geom_xpos[handle])
        self.assertEqual(self.verdict_for(housing, handle).labels, ("FORBIDDEN_CONTACT",))
        self.setUp()
        model = self.sim.model
        mesh = next(g for g in body_collision_geoms(model, "right_arm/moving_jaw_so101_v1")
                    if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH)
        handle = self.g("fork_handle")
        move_geom_to(self.sim, mesh, self.sim.data.geom_xpos[handle])
        self.assertEqual(self.verdict_for(mesh, handle).labels, ("FORBIDDEN_CONTACT",))
        box1 = self.g("right_arm/fixed_jaw_box1")
        self.setUp()
        handle = self.g("fork_handle")
        move_geom_to(self.sim, box1, self.sim.data.geom_xpos[handle])
        self.assertEqual(self.verdict_for(box1, handle).labels, ("FORBIDDEN_CONTACT",))

    def test_left_arm_touching_anything(self):
        left_pad = self.g("left_arm/fixed_jaw_box5")
        handle = self.g("fork_handle")
        move_geom_to(self.sim, left_pad, self.sim.data.geom_xpos[handle])
        self.assertEqual(self.verdict_for(left_pad, handle).labels, ("FORBIDDEN_CONTACT",))

    def test_robot_on_plate_and_cup(self):
        pad = self.g("right_arm/fixed_jaw_box5")
        plate = self.g("plate")
        move_geom_to(self.sim, plate, self.sim.data.geom_xpos[pad])  # the plate is a world geom
        self.assertEqual(self.verdict_for(pad, plate).labels, ("FORBIDDEN_CONTACT",))
        self.setUp()
        pad, cup = self.g("right_arm/fixed_jaw_box5"), self.g("cup_body")
        move_geom_to(self.sim, pad, self.sim.data.geom_xpos[cup])
        self.assertEqual(self.verdict_for(pad, cup).labels, ("FORBIDDEN_CONTACT",))

    def test_non_jaw_right_arm_shape_on_table(self):
        forearm = body_collision_geoms(self.sim.model, "right_arm/lower_arm")[0]
        x, y, _ = self.sim.data.geom_xpos[forearm]
        move_geom_to(self.sim, forearm, (x, y, -0.005))
        self.assertEqual(self.verdict_for(forearm, self.g("table")).labels, ("FORBIDDEN_CONTACT",))

    def test_self_collision_and_arm_arm_contact(self):
        model = self.sim.model
        upper = body_collision_geoms(model, "right_arm/upper_arm")[0]
        wrist = body_collision_geoms(model, "right_arm/wrist")[0]
        move_geom_to(self.sim, upper, self.sim.data.geom_xpos[wrist])
        self.assertEqual(self.verdict_for(upper, wrist).labels, ("SELF_COLLISION",))
        self.setUp()
        model = self.sim.model
        left = body_collision_geoms(model, "left_arm/lower_arm")[0]
        wrist = body_collision_geoms(model, "right_arm/wrist")[0]
        move_geom_to(self.sim, left, self.sim.data.geom_xpos[wrist])
        self.assertEqual(self.verdict_for(left, wrist).labels, ("ARM_ARM_CONTACT",))

    def test_unlisted_scene_shape_is_unknown(self):
        partial = copy.deepcopy(self.config)
        del partial["scene_geoms"]["plate"]
        clf = ContactClassifier(self.sim.model, "fork", partial)
        verdict = clf.classify(self.g("plate"), self.g("cup_body"), 0.0)
        self.assertEqual((verdict.kind, verdict.labels), (VIOLATION, ("UNKNOWN_CONTACT_PAIR",)))
        verdict = clf.classify(self.g("right_arm/fixed_jaw_box5"), self.g("plate"), 0.0)
        self.assertEqual(verdict.labels, ("UNKNOWN_CONTACT_PAIR",))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_pick_contacts.py" -v`
Expected: `ModuleNotFoundError: No module named 'rescuehandsai.pick_contacts'`.

- [ ] **Step 3: Implement**

`src/rescuehandsai/pick_contacts.py`:

```python
"""Shape-level contact classification for the pick milestone (spec §5).

Contacts are classified by collision shape (geom), never by whole body. Decorative
shapes cannot collide, so no rule may depend on them; the constructor enforces that.
"""
from dataclasses import dataclass

import mujoco
import numpy as np

from .scene import ARMS, UTENSILS

SCENE_NORMAL, JAW_UTENSIL, JAW_TABLE, VIOLATION = "scene_normal", "jaw_utensil", "jaw_table", "violation"


@dataclass(frozen=True)
class ContactVerdict:
    kind: str               # SCENE_NORMAL, JAW_UTENSIL, JAW_TABLE or VIOLATION
    labels: tuple           # failure labels; non-empty only for VIOLATION
    geoms: tuple            # (name1, name2)
    force: float            # contact normal force, N
    jaw: str | None = None  # "fixed" or "moving" for JAW_UTENSIL / JAW_TABLE


def geom_name(model, g: int) -> str:
    name = model.geom(g).name
    return name or f"{model.body(int(model.geom_bodyid[g])).name}/geom_{g}"


def collides(model, g: int) -> bool:
    return bool(model.geom_contype[g] or model.geom_conaffinity[g])


class ContactClassifier:
    def __init__(self, model, named: str, config: dict):
        if named not in UTENSILS:
            raise ValueError(f"unknown utensil: {named}")
        self.model, self.named = model, named
        self.spare = next(u for u in UTENSILS if u != named)
        self.grasp_arm = config["grasp_arm"]
        self.jaw_limit = config["jaw_table_force_limit_n"]
        self.severe_limit = config["severe_force_limit_n"]
        self.arm_of = []
        for g in range(model.ngeom):
            root = model.body(int(model.body_rootid[model.geom_bodyid[g]])).name
            self.arm_of.append(next((arm for arm in ARMS if root.startswith(arm + "/")), None))
        self.jaw = {}
        for side, names in config["jaw_grasp_geoms"].items():
            for name in names:
                self.jaw[self._geom(f"{self.grasp_arm}/{name}")] = side
        self.role = {}
        for role, names in config["scene_geoms"].items():
            for name in names:
                g = self._geom(name)
                if not collides(model, g):
                    raise ValueError(f"{name} is listed as a physical scene shape but cannot collide")
                self.role[g] = role
        for name in config["decorative_geoms"]:
            if collides(model, self._geom(name)):
                raise ValueError(f"{name} is listed as decorative but can collide")
        self._force = np.zeros(6)

    def _geom(self, name: str) -> int:
        try:
            return self.model.geom(name).id
        except KeyError as exc:
            raise ValueError(f"contact config names a geom the model does not have: {name}") from exc

    def classify(self, g1: int, g2: int, force: float) -> ContactVerdict:
        names = (geom_name(self.model, g1), geom_name(self.model, g2))
        a1, a2 = self.arm_of[g1], self.arm_of[g2]
        labels, kind, jaw = [], VIOLATION, None
        if a1 is not None and a2 is not None:
            labels.append("ARM_ARM_CONTACT" if a1 != a2 else "SELF_COLLISION")
        elif a1 is None and a2 is None:
            if g1 in self.role and g2 in self.role:
                kind = SCENE_NORMAL
            else:
                labels.append("UNKNOWN_CONTACT_PAIR")
        else:
            robot, other, arm = (g1, g2, a1) if a1 is not None else (g2, g1, a2)
            role, side = self.role.get(other), self.jaw.get(robot)
            if role is None:
                labels.append("UNKNOWN_CONTACT_PAIR")
            elif role == self.spare:
                labels.append("WRONG_ITEM_TOUCHED")
            elif arm != self.grasp_arm or role in ("plate", "cup"):
                labels.append("FORBIDDEN_CONTACT")
            elif side is not None and role == self.named:
                kind, jaw = JAW_UTENSIL, side
            elif side is not None and role == "table":
                kind, jaw = JAW_TABLE, side
                if self.jaw_limit is not None and force > self.jaw_limit:
                    kind, jaw = VIOLATION, None
                    labels.append("FORBIDDEN_CONTACT")
            else:
                labels.append("FORBIDDEN_CONTACT")
        if (a1 is not None or a2 is not None) and self.severe_limit is not None and force > self.severe_limit:
            kind, jaw = VIOLATION, None
            labels.append("EXCESS_FORCE")
        return ContactVerdict(kind, tuple(labels), names, float(force), jaw)

    def contacts(self, data) -> list:
        verdicts = []
        for i, contact in enumerate(data.contact):
            if contact.dist > 0:
                continue
            mujoco.mj_contactForce(self.model, data, i, self._force)
            verdicts.append(self.classify(int(contact.geom1), int(contact.geom2), abs(float(self._force[0]))))
        return verdicts
```

- [ ] **Step 4: Run the tests**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_pick_contacts.py" -v`
Expected: 13 tests `ok`.

If a "no contact detected" assertion fails, print `data.geom_xpos`, the geom sizes and `data.ncon`. Move the geom deeper into the target instead of weakening the assertion. Every forbidden class must be shown to be detectable (§5).

If `test_items_resting_on_the_table_are_normal` fails because a **robot** shape touches the table or an item at the home pose, **stop and report it to the owner**. §5 says the home pose is checked for such contacts, and any exception needs an owner-approved spec edit. Do not add an exception in code.

- [ ] **Step 5: Commit (do not push)**

```powershell
git add src/rescuehandsai/pick_contacts.py tests/test_pick_contacts.py
git commit -m "Pick plan 1 task 5: shape-level contact classifier with detectability tests"
```

---

### Task 6: Scene cells, templates, pick task and start check

**Files:**
- Create: `src/rescuehandsai/pick_cells.py`
- Test: `tests/test_pick_cells.py`

**Interfaces:**
- Consumes: `sample_params`, `SceneParams`, `SCENE_ITEMS`, `UTENSILS`; `MujocoSimulation(physics_version=, ...)` and `reset(params=)` (Task 2); `load_rules()` (Task 4).
- Produces:
  - `CELLS: dict[str, tuple[str, tuple[str, str]]]` (cell → (named, (slot-0 item, slot-1 item)))
  - `TRAIN_TEMPLATES`, `HELD_BACK_TEMPLATES` (id → text with `{u}`)
  - `PickTask(seed, cell, utensil, spare, template, instruction, physics_version=2)`
  - `named_slot(cell) -> int`, `slot_jitter(base, config) -> dict[int, tuple]`, `cell_params(base, cell, config) -> SceneParams`
  - `template_text(template_id) -> str`, `make_pick_task(seed, cell, template, physics_version=2) -> PickTask`, `assign_templates(count, template_ids) -> list[str]`
  - `StartCheck(cell, valid, reasons)`, `check_start(seed, cell, *, physics_version=2, rules=None, edit=None) -> StartCheck`, `check_scene(seed, **kw) -> dict[str, StartCheck]`

- [ ] **Step 1: Write the failing tests**

`tests/test_pick_cells.py`:

```python
"""The four runs of one scene differ only in the word and the tray order (spec §3, §8)."""
from collections import Counter
from dataclasses import replace
import unittest

import numpy as np

from rescuehandsai.pick_cells import (CELLS, HELD_BACK_TEMPLATES, TRAIN_TEMPLATES, assign_templates, cell_params,
                                      check_scene, check_start, make_pick_task, named_slot, slot_jitter)
from rescuehandsai.scene import load_config, sample_params
from rescuehandsai.sim import MujocoSimulation


class CellTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.base = sample_params(self.config, 5)
        self.cells = {cell: cell_params(self.base, cell, self.config) for cell in CELLS}

    def test_properties_follow_the_item_and_everything_else_is_shared(self):
        for params in self.cells.values():
            for field in ("masses", "frictions", "utensil_scale", "cup_radius", "cup_half_height",
                          "table_rgb", "light_diffuse", "light_offset", "seed"):
                self.assertEqual(getattr(params, field), getattr(self.base, field), field)
            self.assertEqual(params.poses["cup"], self.base.poses["cup"])

    def test_jitter_follows_the_slot(self):
        a, b = self.cells["F-A"], self.cells["F-B"]
        np.testing.assert_allclose(b.poses["fork"], a.poses["spoon"], atol=1e-12)
        np.testing.assert_allclose(b.poses["spoon"], a.poses["fork"], atol=1e-12)
        self.assertEqual(self.cells["F-A"].poses, self.cells["S-A"].poses)
        self.assertEqual(self.cells["F-B"].poses, self.cells["S-B"].poses)
        for params in self.cells.values():
            for slot, jitter in slot_jitter(params, self.config).items():
                np.testing.assert_allclose(jitter, slot_jitter(self.base, self.config)[slot], atol=1e-12)

    def test_named_slots(self):
        self.assertEqual({c: named_slot(c) for c in CELLS}, {"F-A": 0, "F-B": 1, "S-A": 1, "S-B": 0})
        self.assertEqual(self.cells["F-A"].slots, {"fork": 0, "spoon": 1})
        self.assertEqual(self.cells["S-B"].slots, {"spoon": 0, "fork": 1})

    def test_tasks_differ_only_in_the_word(self):
        tasks = {cell: make_pick_task(5, cell, "T3") for cell in CELLS}
        texts = {task.instruction.replace(task.utensil, "{u}") for task in tasks.values()}
        self.assertEqual(texts, {TRAIN_TEMPLATES["T3"]})
        self.assertEqual((tasks["F-A"].utensil, tasks["F-A"].spare), ("fork", "spoon"))
        self.assertEqual((tasks["S-B"].utensil, tasks["S-B"].spare), ("spoon", "fork"))
        self.assertEqual(make_pick_task(5, "S-A", "H2").instruction, HELD_BACK_TEMPLATES["H2"].format(u="spoon"))
        with self.assertRaises(KeyError):
            make_pick_task(5, "X-Y", "T1")

    def test_no_template_names_a_side_or_slot(self):
        for text in list(TRAIN_TEMPLATES.values()) + list(HELD_BACK_TEMPLATES.values()):
            for word in ("left", "slot", "first", "second", "near", "far"):
                self.assertNotIn(word, text.lower())

    def test_template_rotation_is_fixed_and_balanced(self):
        main = assign_templates(100, sorted(TRAIN_TEMPLATES))
        self.assertEqual(main, assign_templates(100, sorted(TRAIN_TEMPLATES)))
        self.assertEqual(sorted(Counter(main).values()), [12, 12, 12, 12, 13, 13, 13, 13])
        self.assertEqual(set(Counter(assign_templates(24, sorted(HELD_BACK_TEMPLATES))).values()), {6})
        self.assertEqual(set(Counter(assign_templates(24, sorted(TRAIN_TEMPLATES))).values()), {3})


class StartCheckTests(unittest.TestCase):
    def test_normal_scene_is_valid_in_all_cells(self):
        results = check_scene(5)
        self.assertEqual(set(results), set(CELLS))
        for result in results.values():
            self.assertTrue(result.valid, result.reasons)

    def test_overlap_and_off_table_are_rejected(self):
        def overlap(params):
            return replace(params, poses={**params.poses, "spoon": params.poses["fork"]})

        result = check_start(5, "F-A", edit=overlap)
        self.assertFalse(result.valid)
        self.assertTrue(any(r.startswith("overlap:") for r in result.reasons), result.reasons)

        def off_table(params):
            x, y, yaw = params.poses["fork"]
            return replace(params, poses={**params.poses, "fork": (x + 2.0, y, yaw)})

        result = check_start(5, "F-A", edit=off_table)
        self.assertIn("off_table:fork", result.reasons)

    def test_check_runs_on_its_own_copy(self):
        sim = MujocoSimulation(physics_version=2)
        try:
            sim.reset(5)
            qpos, time = sim.data.qpos.copy(), float(sim.data.time)
            check_start(5, "F-A")
            np.testing.assert_array_equal(sim.data.qpos, qpos)
            self.assertEqual(float(sim.data.time), time)
        finally:
            sim.close()


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_pick_cells.py" -v`
Expected: `ModuleNotFoundError: No module named 'rescuehandsai.pick_cells'`.

- [ ] **Step 3: Implement**

`src/rescuehandsai/pick_cells.py`:

```python
"""The four runs of one pick scene (spec §3), the pick wordings (spec §8) and the start check."""
from dataclasses import dataclass, replace
import math

import numpy as np

from .contracts import BimanualAction
from .pick_config import load_rules
from .scene import SCENE_ITEMS, SceneParams, sample_params

# cell -> (named utensil, (utensil in slot 0, utensil in slot 1))
CELLS = {
    "F-A": ("fork", ("fork", "spoon")),
    "F-B": ("fork", ("spoon", "fork")),
    "S-A": ("spoon", ("fork", "spoon")),
    "S-B": ("spoon", ("spoon", "fork")),
}

# No wording names a side or a slot: the word is the only clue to which utensil.
TRAIN_TEMPLATES = {
    "T1": "pick up the {u} with the right hand",
    "T2": "grab the {u} with your right hand",
    "T3": "lift the {u} off the tray using the right arm",
    "T4": "take the {u} and hold it up",
    "T5": "use the right gripper to pick up the {u}",
    "T6": "get the {u} from the tray and hold it",
    "T7": "please raise the {u} with your right hand",
    "T8": "right hand: lift the {u}",
}
HELD_BACK_TEMPLATES = {
    "H1": "I need the {u}, lift it with your right hand",
    "H2": "hold up the {u} for me",
    "H3": "could you pick the {u} up?",
    "H4": "the {u}, please, in the right hand",
}


@dataclass(frozen=True)
class PickTask:
    seed: int
    cell: str
    utensil: str      # the named utensil
    spare: str
    template: str
    instruction: str
    physics_version: int = 2


def named_slot(cell: str) -> int:
    named, order = CELLS[cell]
    return order.index(named)


def slot_jitter(base: SceneParams, config: dict) -> dict:
    """slot index -> (dx, dy, dyaw) that the scene drew for whichever utensil sits there."""
    slots = config["utensil"]["slots"]
    jitter = {}
    for item, slot in base.slots.items():
        x, y, yaw = base.poses[item]
        sx, sy = slots[slot]
        jitter[slot] = (x - sx, y - sy, yaw - math.pi / 2)
    return jitter


def cell_params(base: SceneParams, cell: str, config: dict) -> SceneParams:
    """Jitter stays with the slot; mass, friction and size stay with the item."""
    _, order = CELLS[cell]
    jitter = slot_jitter(base, config)
    poses, slots = dict(base.poses), {}
    for slot, item in enumerate(order):
        sx, sy = config["utensil"]["slots"][slot]
        dx, dy, dyaw = jitter[slot]
        poses[item] = (sx + dx, sy + dy, math.pi / 2 + dyaw)
        slots[item] = slot
    return replace(base, poses=poses, slots=slots)


def template_text(template_id: str) -> str:
    if template_id in TRAIN_TEMPLATES:
        return TRAIN_TEMPLATES[template_id]
    return HELD_BACK_TEMPLATES[template_id]


def make_pick_task(seed: int, cell: str, template: str, physics_version: int = 2) -> PickTask:
    named, _ = CELLS[cell]
    spare = next(item for item in CELLS[cell][1] if item != named)
    return PickTask(seed, cell, named, spare, template, template_text(template).format(u=named), physics_version)


def assign_templates(count: int, template_ids) -> list:
    """Fixed rotation over the ordered scene list; the first templates take any extra scenes."""
    ids = list(template_ids)
    return [ids[i % len(ids)] for i in range(count)]


@dataclass(frozen=True)
class StartCheck:
    cell: str
    valid: bool
    reasons: tuple


def check_start(seed: int, cell: str, *, physics_version: int = 2, rules: dict | None = None, edit=None) -> StartCheck:
    """Frozen start-validity rules on a separate simulation; the arms hold their home targets.

    `edit` (tests only) changes the cell's SceneParams before the scene is built."""
    from .sim import MujocoSimulation  # sim imports scene; keep this module importable without a model

    limits = (rules or load_rules())["start_check"]
    sim = MujocoSimulation(physics_version=physics_version)
    try:
        params = cell_params(sample_params(sim.scene_config, seed), cell, sim.scene_config)
        if edit is not None:
            params = edit(params)
        sim.reset(seed, params=params)
        model, data = sim.model, sim.data
        item_bodies = {model.body(item).id for item in SCENE_ITEMS}
        reasons = []
        for contact in data.contact:
            b1, b2 = int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])
            if contact.dist < -limits["overlap_depth_m"] and (b1 in item_bodies or b2 in item_bodies):
                reasons.append(f"overlap:{sim._geom_name(int(contact.geom1))}-{sim._geom_name(int(contact.geom2))}")
        start = {item: data.body(item).xpos.copy() for item in SCENE_ITEMS}
        for _ in range(limits["hold_control_steps"]):
            sim.step(BimanualAction(sim.observe().timestamp, sim.home_targets))
        table = sim.scene_config["table"]
        state = sim.privileged().objects
        for item in SCENE_ITEMS:
            position = data.body(item).xpos
            shift = float(np.linalg.norm(position - start[item]))
            speed = float(np.linalg.norm(state[item].linear_velocity))
            if shift > limits["max_shift_m"] or speed > limits["max_speed_mps"]:
                reasons.append(f"moving:{item}")
            if (abs(position[0] - table["center"][0]) > table["half_size"][0]
                    or abs(position[1] - table["center"][1]) > table["half_size"][1] or position[2] < 0):
                reasons.append(f"off_table:{item}")
        reasons = tuple(dict.fromkeys(reasons))
        return StartCheck(cell, not reasons, reasons)
    finally:
        sim.close()


def check_scene(seed: int, **kwargs) -> dict:
    """All four cells; a scene is usable only if every cell has a valid start."""
    return {cell: check_start(seed, cell, **kwargs) for cell in CELLS}
```

- [ ] **Step 4: Run the tests**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_pick_cells.py" -v`
Expected: 9 tests `ok`.

If `test_normal_scene_is_valid_in_all_cells` fails for seed 5, print the reasons and report them. Do not change the limits to make the test pass. A normal scene failing the check is a finding about the scene or the limits.

- [ ] **Step 5: Commit (do not push)**

```powershell
git add src/rescuehandsai/pick_cells.py tests/test_pick_cells.py
git commit -m "Pick plan 1 task 6: four cells, balanced wordings, pick task, start check on a separate copy"
```

---

### Task 7: Scene identity, duplicate check and seed blocks

**Files:**
- Create: `configs/pick_seed_blocks.json`
- Create: `src/rescuehandsai/pick_identity.py`
- Test: `tests/test_pick_identity.py`

**Interfaces:**
- Consumes: `world_xml(..., physics_version=)` (Task 2).
- Produces: `text_digest(path) -> str`, `config_hash(physics_version, asset_path, root=ROOT) -> str`, `scene_hash(params, config, physics_version) -> str`, `settings_record(params) -> dict`, `settings_hash(params, digits=5) -> str`, `find_duplicates(lists: dict[str, list[SceneParams]]) -> list[dict]`, `load_seed_blocks(path=None) -> dict`, `check_seed_blocks(config) -> dict`, `block_of(seed, config) -> str | None`.

- [ ] **Step 1: Create the seed block config**

`configs/pick_seed_blocks.json`:

```json
{
  "blocks": {
    "train": [3000000, 3001000],
    "dev": [3100000, 3200000],
    "test_main": [3200000, 3300000],
    "test_wording": [3300000, 3400000]
  },
  "legacy_ranges": [
    {
      "start": 0,
      "stop": 200000,
      "complete": false,
      "source": "seeds used before this milestone: evaluation 0-9, teacher dev 4000-4015, Kaggle clean data from 1000, perturbed from 51000, recovery from 101000, perturbed measurement 55000-55099, laptop shards local_w0/w1 (seed provenance incomplete)"
    }
  ]
}
```

- [ ] **Step 2: Write the failing tests**

`tests/test_pick_identity.py`:

```python
import copy
from dataclasses import replace
import tempfile
import unittest
from pathlib import Path

from rescuehandsai.pick_identity import (block_of, check_seed_blocks, config_hash, find_duplicates,
                                         load_seed_blocks, scene_hash, settings_hash, settings_record, text_digest)
from rescuehandsai.scene import ROOT, load_config, sample_params


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.asset = ROOT / ".cache/menagerie/robotstudio_so101/so101.xml"

    def test_line_endings_do_not_change_text_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            lf, crlf = Path(tmp) / "a.json", Path(tmp) / "b.json"
            lf.write_bytes(b'{"a": 1}\n{"b": 2}\n')
            crlf.write_bytes(b'{"a": 1}\r\n{"b": 2}\r\n')
            self.assertEqual(text_digest(lf), text_digest(crlf))

    def test_config_hash_depends_on_physics_version(self):
        self.assertEqual(config_hash(2, self.asset), config_hash(2, self.asset))
        self.assertNotEqual(config_hash(1, self.asset), config_hash(2, self.asset))

    def test_scene_hash_depends_on_version_and_scene(self):
        a, b = sample_params(self.config, 1), sample_params(self.config, 2)
        self.assertNotEqual(scene_hash(a, self.config, 1), scene_hash(a, self.config, 2))
        self.assertNotEqual(scene_hash(a, self.config, 2), scene_hash(b, self.config, 2))

    def test_settings_hash_ignores_seed_and_float_noise(self):
        params = sample_params(self.config, 3)
        record = settings_record(params)
        self.assertIsInstance(record["poses"]["fork"], list)
        self.assertEqual(settings_hash(params), settings_hash(replace(params, seed=999)))
        nudged = replace(params, light_diffuse=params.light_diffuse + 1e-9)
        self.assertEqual(settings_hash(params), settings_hash(nudged))
        self.assertNotEqual(settings_hash(params), settings_hash(sample_params(self.config, 4)))

    def test_duplicates_across_lists_are_found(self):
        a, b = sample_params(self.config, 1), sample_params(self.config, 2)
        copy_of_a = replace(a, seed=3200000)
        found = find_duplicates({"train": [a, b], "test_main": [copy_of_a]})
        self.assertEqual(len(found), 1)
        self.assertEqual(sorted(found[0]["members"]), [["test_main", 3200000], ["train", 1]])
        self.assertEqual(find_duplicates({"train": [a], "dev": [b]}), [])

    def test_seed_blocks_do_not_overlap_and_legacy_is_reported_incomplete(self):
        report = check_seed_blocks(load_seed_blocks())
        self.assertEqual(report["overlaps"], [])
        self.assertFalse(report["legacy_provenance_complete"])
        self.assertEqual(block_of(3200050, load_seed_blocks()), "test_main")
        self.assertIsNone(block_of(42, load_seed_blocks()))

    def test_overlapping_blocks_are_reported(self):
        config = copy.deepcopy(load_seed_blocks())
        config["blocks"]["train"] = [150000, 160000]
        config["blocks"]["dev"] = [3099999, 3200001]
        overlaps = {tuple(sorted(o)) for o in check_seed_blocks(config)["overlaps"]}
        self.assertIn(("legacy_0", "train"), overlaps)
        self.assertIn(("dev", "test_main"), overlaps)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run it to verify it fails**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_pick_identity.py" -v`
Expected: `ModuleNotFoundError: No module named 'rescuehandsai.pick_identity'`.

- [ ] **Step 4: Implement**

`src/rescuehandsai/pick_identity.py`:

```python
"""Scene identity for the pick milestone (spec §4, §8): hashes, duplicate scenes, seed blocks."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .scene import ROOT, SceneParams, world_xml

SEED_BLOCKS_PATH = ROOT / "configs" / "pick_seed_blocks.json"
CONFIG_HASH_FILES = ("configs/scene.json", "configs/simulation.json", "configs/pick_contacts.json",
                     "src/rescuehandsai/scene.py")


def text_digest(path) -> str:
    """SHA-256 of a text file with CRLF normalised, so Windows and Linux checkouts agree."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def config_hash(physics_version: int, asset_path, root: Path = ROOT) -> str:
    digest = hashlib.sha256(f"physics_version={physics_version}\n".encode())
    for rel in CONFIG_HASH_FILES:
        digest.update(f"{rel} {text_digest(Path(root) / rel)}\n".encode())
    digest.update(f"asset {text_digest(asset_path)}\n".encode())
    return digest.hexdigest()


def scene_hash(params: SceneParams, config: dict, physics_version: int) -> str:
    return hashlib.sha256(world_xml(params, config, physics_version=physics_version).encode()).hexdigest()


def settings_record(params: SceneParams) -> dict:
    """JSON-ready generated scene settings (tuples become lists)."""
    return json.loads(json.dumps(asdict(params)))


def _rounded(value, digits):
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, list):
        return [_rounded(v, digits) for v in value]
    if isinstance(value, dict):
        return {k: _rounded(v, digits) for k, v in value.items()}
    return value


def settings_hash(params: SceneParams, digits: int = 5) -> str:
    record = settings_record(params)
    record.pop("seed")
    text = json.dumps(_rounded(record, digits), sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def find_duplicates(lists: dict) -> list:
    seen = {}
    for name, params_list in lists.items():
        for params in params_list:
            seen.setdefault(settings_hash(params), []).append([name, params.seed])
    return [{"settings_sha256": h, "members": members} for h, members in seen.items() if len(members) > 1]


def load_seed_blocks(path=None) -> dict:
    return json.loads(Path(path or SEED_BLOCKS_PATH).read_text())


def check_seed_blocks(config: dict) -> dict:
    intervals = [(name, start, stop) for name, (start, stop) in config["blocks"].items()]
    legacy = [(f"legacy_{i}", r["start"], r["stop"]) for i, r in enumerate(config["legacy_ranges"])]
    overlaps = []
    everything = intervals + legacy
    for i, (a, a0, a1) in enumerate(everything):
        for b, b0, b1 in everything[i + 1:]:
            if a.startswith("legacy_") and b.startswith("legacy_"):
                continue
            if a0 < b1 and b0 < a1:
                overlaps.append([a, b])
    return {"overlaps": overlaps,
            "legacy_provenance_complete": all(r["complete"] for r in config["legacy_ranges"]),
            "legacy_sources": [r["source"] for r in config["legacy_ranges"]]}


def block_of(seed: int, config: dict):
    return next((name for name, (start, stop) in config["blocks"].items() if start <= seed < stop), None)
```

- [ ] **Step 5: Run the tests**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_pick_identity.py" -v`
Expected: 7 tests `ok`.

- [ ] **Step 6: Commit (do not push)**

```powershell
git add configs/pick_seed_blocks.json src/rescuehandsai/pick_identity.py tests/test_pick_identity.py
git commit -m "Pick plan 1 task 7: scene identity hashes, duplicate check, seed blocks with honest legacy note"
```

---

### Task 8: The success-rule judge (pure Python)

**Files:**
- Create: `src/rescuehandsai/pick_outcome.py`
- Test: `tests/test_pick_outcome.py`

**Interfaces:**
- Consumes: `StepCounts`, `derive_steps`, `load_rules` (Task 4); `ContactVerdict`, `JAW_UTENSIL`, `SCENE_NORMAL`, `VIOLATION` (Task 5).
- Produces:
  - `SubstepFacts(physics_step, control_step, positions, rotations, linear_speed, angular_speed, gripper_pos, gripper_rot, verdicts)`
  - `LABEL_RULE: dict[str, str]`, `SEVERE: frozenset[str]`, `RULES = ("R1", "R2", "R3", "R4", "R5")`
  - `PickOutcome` dataclass (fields below)
  - `PickJudge(named, spare, rules, steps, table_center, table_half)` with `.start_from(facts)`, `.update(facts) -> list[str]` (new severe labels; raises `RuntimeError` if called after success), `.record_error(label, physics_step, control_step, detail)`, `.succeeded`, `.finish(stop, *, physics_step, control_step) -> PickOutcome`; `stop` is one of `"success"`, `"deadline"`, `"early_stop"`, `"crash"`
  - `PickOutcome` reports `hold_completed` / `hold_completed_*_step` (a valid hold window happened, even in a failed episode) separately from `success_*_step`

- [ ] **Step 1: Write the failing tests**

`tests/test_pick_outcome.py`:

```python
"""Success rules on synthetic per-physics-step facts (spec §5, §12)."""
import math
import unittest

import numpy as np

from rescuehandsai.pick_config import derive_steps, load_rules
from rescuehandsai.pick_contacts import JAW_UTENSIL, SCENE_NORMAL, VIOLATION, ContactVerdict
from rescuehandsai.pick_outcome import PickJudge, SubstepFacts

RULES = load_rules()
STEPS = derive_steps(RULES, 0.005, 0.05)  # hold 200, max_gap 20, final_speed 50, deadline 300
START = {"fork": np.array([0.25, 0.24, 0.006]), "spoon": np.array([0.33, 0.24, 0.006]),
         "cup": np.array([0.40, 0.12, 0.03])}
FIXED = ContactVerdict(JAW_UTENSIL, (), ("right_arm/fixed_jaw_box5", "fork_handle"), 1.0, "fixed")
MOVING = ContactVerdict(JAW_UTENSIL, (), ("right_arm/moving_jaw_box2", "fork_handle"), 1.0, "moving")


def rot_z(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])


def rot_x(deg):
    a = math.radians(deg)
    return np.array([[1.0, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


def facts(step, *, lift=0.0, verdicts=(FIXED, MOVING), spare=None, cup=None, cup_rot=None, spare_rot=None,
          offset=(0.0, 0.0, -0.01), speed=0.0, spin=0.0, fork_rot=None):
    fork = START["fork"] + np.array([0, 0, lift])
    positions = {"fork": fork, "spoon": START["spoon"] if spare is None else np.asarray(spare, float),
                 "cup": START["cup"] if cup is None else np.asarray(cup, float)}
    rotations = {"fork": np.eye(3) if fork_rot is None else fork_rot,
                 "spoon": np.eye(3) if spare_rot is None else spare_rot,
                 "cup": np.eye(3) if cup_rot is None else cup_rot}
    return SubstepFacts(step, (step - 1) // STEPS.substeps + 1, positions, rotations,
                        {"fork": speed, "spoon": 0.0, "cup": 0.0}, {"fork": spin, "spoon": 0.0, "cup": 0.0},
                        fork - np.asarray(offset), np.eye(3), tuple(verdicts))


def judge():
    j = PickJudge("fork", "spoon", RULES, STEPS, (0.0, 0.15), (0.6, 0.45))
    j.start_from(facts(0, verdicts=()))
    return j


def run(j, sequence):
    severe = []
    for f in sequence:
        severe += j.update(f)
    return severe


class HoldTests(unittest.TestCase):
    def test_stable_hold_succeeds_after_exactly_the_window(self):
        j = judge()
        run(j, [facts(s, lift=0.06) for s in range(1, 200)])
        self.assertFalse(j.succeeded)
        j.update(facts(200, lift=0.06))
        self.assertTrue(j.succeeded)
        with self.assertRaises(RuntimeError):  # the runner must stop at the success step
            j.update(facts(201, lift=0.06))
        out = j.finish("success", physics_step=200, control_step=20)
        self.assertTrue(out.success)
        self.assertEqual((out.failures, out.picked), ([], "named_first"))
        self.assertEqual((out.hold_completed, out.hold_completed_physics_step, out.success_physics_step),
                         (True, 200, 200))

    def test_swinging_at_constant_distance_fails(self):
        j = judge()
        seq = []
        for s in range(1, 301):
            angle = math.radians(90.0 * s / 300)
            offset = (0.02 * math.cos(angle), 0.02 * math.sin(angle), 0.0)
            seq.append(facts(s, lift=0.06, offset=offset))
        run(j, seq)
        self.assertFalse(j.succeeded)
        out = j.finish("deadline", physics_step=300, control_step=30)
        self.assertEqual([f["label"] for f in out.failures], ["IMPROPER_HOLD"])

    def test_rotation_in_gripper_frame_fails(self):
        j = judge()
        run(j, [facts(s, lift=0.06, fork_rot=rot_x(15.0 * s / 250)) for s in range(1, 251)])
        self.assertFalse(j.succeeded)

    def test_support_by_table_or_other_shape_is_not_a_hold(self):
        table = ContactVerdict(SCENE_NORMAL, (), ("fork_handle", "table"), 0.1)
        j = judge()
        run(j, [facts(s, lift=0.06, verdicts=(FIXED, MOVING, table)) for s in range(1, 301)])
        self.assertFalse(j.succeeded)
        housing = ContactVerdict(VIOLATION, ("FORBIDDEN_CONTACT",), ("right_arm/gripper/geom_87", "fork_handle"), 0.1)
        j = judge()
        run(j, [facts(s, lift=0.06, verdicts=(FIXED, housing)) for s in range(1, 301)])
        self.assertFalse(j.succeeded)
        self.assertIn("FORBIDDEN_CONTACT", j.failures)

    def test_both_jaw_fraction_and_gap(self):
        j = judge()  # every 4th step only one jaw: 75% < 80%
        run(j, [facts(s, lift=0.06, verdicts=(FIXED,) if s % 4 == 0 else (FIXED, MOVING)) for s in range(1, 301)])
        self.assertFalse(j.succeeded)
        j = judge()  # exactly one full window; its 25-step single-jaw gap (87.5% both) exceeds the 20-step limit
        run(j, [facts(s, lift=0.06, verdicts=(FIXED,) if 50 <= s < 75 else (FIXED, MOVING)) for s in range(1, 201)])
        self.assertFalse(j.succeeded)
        j = judge()  # a 20-step gap is allowed
        run(j, [facts(s, lift=0.06, verdicts=(FIXED,) if 50 <= s < 70 else (FIXED, MOVING)) for s in range(1, 201)])
        self.assertTrue(j.succeeded)

    def test_no_jaw_at_all_breaks_the_window(self):
        j = judge()
        run(j, [facts(s, lift=0.06, verdicts=() if s == 100 else (FIXED, MOVING)) for s in range(1, 201)])
        self.assertFalse(j.succeeded)

    def test_final_window_must_be_slow(self):
        j = judge()
        run(j, [facts(s, lift=0.06, speed=0.05 if s > 180 else 0.0) for s in range(1, 301)])
        self.assertFalse(j.succeeded)
        j = judge()
        run(j, [facts(s, lift=0.06, spin=1.0 if s > 180 else 0.0) for s in range(1, 301)])
        self.assertFalse(j.succeeded)


class ViolationTests(unittest.TestCase):
    def test_failures_latch(self):
        wrong = ContactVerdict(VIOLATION, ("WRONG_ITEM_TOUCHED",), ("right_arm/fixed_jaw_box5", "spoon_handle"), 0.2)
        j = judge()
        j.update(facts(1, verdicts=(wrong,)))
        run(j, [facts(s, lift=0.06) for s in range(2, 400)])
        self.assertFalse(j.succeeded)
        out = j.finish("deadline", physics_step=3000, control_step=300)
        self.assertEqual([f["label"] for f in out.failures], ["WRONG_ITEM_TOUCHED"])  # hold itself was fine
        self.assertEqual(out.failed_rules, ["R1"])
        # the later correct hold is still reported, apart from success
        self.assertFalse(out.success)
        self.assertEqual((out.hold_completed, out.hold_completed_physics_step), (True, 201))
        self.assertIsNone(out.success_physics_step)

    def test_no_lift_is_decided_only_at_the_deadline(self):
        j = judge()
        run(j, [facts(s, verdicts=()) for s in range(1, 100)])
        self.assertEqual(j.failures, {})
        out = j.finish("deadline", physics_step=3000, control_step=300)
        self.assertEqual(out.first_failure, "NO_LIFT")
        self.assertEqual(out.picked, "neither")

    def test_timeout_when_a_hold_is_still_forming(self):
        j = judge()
        run(j, [facts(s, lift=0.06 if s > 50 else 0.0, verdicts=(FIXED, MOVING) if s > 50 else ()) for s in range(1, 150)])
        out = j.finish("deadline", physics_step=149, control_step=15)
        self.assertEqual(out.first_failure, "TIMEOUT")
        self.assertEqual(out.failed_rules, ["R5"])

    def test_spare_and_cup_limits_over_the_whole_episode(self):
        j = judge()
        j.update(facts(1, verdicts=(), spare=START["spoon"] + [0.011, 0, 0]))
        j.update(facts(2, verdicts=()))  # moved back: the maximum still counts
        self.assertIn("SPARE_DISTURBED", j.failures)
        j = judge()
        j.update(facts(1, verdicts=(), spare_rot=rot_z(6.0)))
        self.assertIn("SPARE_DISTURBED", j.failures)
        j = judge()
        j.update(facts(1, verdicts=(), cup_rot=rot_x(16.0)))
        self.assertIn("CUP_DISTURBED", j.failures)
        j = judge()
        j.update(facts(1, verdicts=(), cup=START["cup"] + [0, 0.011, 0]))
        self.assertIn("CUP_DISTURBED", j.failures)

    def test_severe_violations_are_returned_and_early_stop_lists_unevaluable_rules(self):
        arms = ContactVerdict(VIOLATION, ("ARM_ARM_CONTACT",), ("left_arm/x", "right_arm/y"), 0.3)
        j = judge()
        self.assertEqual(j.update(facts(1, verdicts=(arms,))), ["ARM_ARM_CONTACT"])
        self.assertEqual(j.update(facts(2, verdicts=(arms,))), [])  # reported once
        out = j.finish("early_stop", physics_step=2, control_step=1)
        self.assertEqual(out.unevaluable_rules, ["R2", "R5"])
        j = judge()
        self.assertEqual(j.update(facts(1, verdicts=(), spare=[0.33, 0.24, -0.05])), ["ITEM_FELL_OR_OUT"])

    def test_crash_lists_every_unconfirmed_rule(self):
        j = judge()
        j.update(facts(1, verdicts=(), cup_rot=rot_x(20.0)))
        out = j.finish("crash", physics_step=1, control_step=1)
        self.assertEqual(out.unevaluable_rules, ["R1", "R2", "R3", "R5"])

    def test_every_failed_rule_is_reported_in_time_order(self):
        wrong = ContactVerdict(VIOLATION, ("WRONG_ITEM_TOUCHED",), ("right_arm/a", "spoon_handle"), 0.1)
        j = judge()
        j.update(facts(1, verdicts=(), cup_rot=rot_x(20.0)))
        j.update(facts(2, verdicts=(wrong,)))
        j.record_error("POLICY_ERROR", 3, 1, "boom")
        out = j.finish("early_stop", physics_step=3, control_step=1)
        self.assertEqual([f["label"] for f in out.failures], ["CUP_DISTURBED", "WRONG_ITEM_TOUCHED", "POLICY_ERROR"])
        self.assertEqual(out.first_failure, "CUP_DISTURBED")
        self.assertEqual(out.failed_rules, ["R1", "R4"])

    def test_misuse_is_rejected(self):
        j = PickJudge("fork", "spoon", RULES, STEPS, (0.0, 0.15), (0.6, 0.45))
        with self.assertRaises(RuntimeError):
            j.update(facts(1))
        j = judge()
        with self.assertRaises(ValueError):
            j.finish("success", physics_step=1, control_step=1)
        with self.assertRaises(ValueError):
            j.record_error("SIM_ERROR", 1, 1, "invalid runs are not judged")


class PickedTests(unittest.TestCase):
    def test_spare_first_simultaneous_and_both(self):
        j = judge()
        j.update(facts(5, verdicts=(), spare=START["spoon"] + [0, 0, 0.06]))
        j.update(facts(40, lift=0.06, verdicts=(), spare=START["spoon"] + [0, 0, 0.06]))
        out = j.finish("early_stop", physics_step=40, control_step=4)
        self.assertEqual((out.picked, out.lifted_both), ("spare_first", True))
        j = judge()
        j.update(facts(5, lift=0.06, verdicts=(), spare=START["spoon"] + [0, 0, 0.06]))
        out = j.finish("early_stop", physics_step=5, control_step=1)
        self.assertEqual(out.picked, "simultaneous")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_pick_outcome.py" -v`
Expected: `ModuleNotFoundError: No module named 'rescuehandsai.pick_outcome'`.

- [ ] **Step 3: Implement**

`src/rescuehandsai/pick_outcome.py`:

```python
"""Pick-milestone success rules (spec §5): a pure judge over per-physics-step facts.

Failures latch; immediate violations are recorded when they happen, unmet goals at the
deadline. The judge never sees invalid-run labels (spec §6)."""
from collections import deque
from dataclasses import dataclass
import math
from types import SimpleNamespace

import numpy as np

from .pick_config import StepCounts
from .pick_contacts import JAW_UTENSIL

LABEL_RULE = {
    "WRONG_ITEM_TOUCHED": "R1", "FORBIDDEN_CONTACT": "R1", "UNKNOWN_CONTACT_PAIR": "R1",
    "NO_LIFT": "R2", "IMPROPER_HOLD": "R2",
    "SPARE_DISTURBED": "R3",
    "CUP_DISTURBED": "R4", "ITEM_FELL_OR_OUT": "R4", "ARM_ARM_CONTACT": "R4", "SELF_COLLISION": "R4",
    "EXCESS_FORCE": "R4",
    "TIMEOUT": "R5",
    "INVALID_ACTION": "POLICY", "POLICY_ERROR": "POLICY",
}
SEVERE = frozenset({"ITEM_FELL_OR_OUT", "ARM_ARM_CONTACT", "EXCESS_FORCE", "INVALID_ACTION", "POLICY_ERROR"})
RULES = ("R1", "R2", "R3", "R4", "R5")
STOPS = ("success", "deadline", "early_stop", "crash")


@dataclass(frozen=True)
class SubstepFacts:
    physics_step: int
    control_step: int
    positions: dict        # item -> (3,) world position of the body origin
    rotations: dict        # item -> (3, 3) world rotation
    linear_speed: dict     # item -> m/s
    angular_speed: dict    # item -> rad/s
    gripper_pos: np.ndarray
    gripper_rot: np.ndarray
    verdicts: tuple        # ContactVerdict for every touching pair at this physics step


@dataclass
class PickOutcome:
    success: bool
    stop: str
    failures: list
    first_failure: str | None
    failed_rules: list
    unevaluable_rules: list
    picked: str
    lifted_both: bool
    # A valid hold window can complete in a failed episode (e.g. the spare was touched
    # first); it is reported apart from success so that evidence is not lost.
    hold_completed: bool
    hold_completed_physics_step: int | None
    hold_completed_control_step: int | None
    success_physics_step: int | None
    success_control_step: int | None
    max_lift_m: dict
    spare_max_shift_m: float
    spare_max_yaw_deg: float
    cup_max_shift_m: float
    cup_max_tilt_deg: float
    longest_eligible_streak: int


def _yaw(rotation) -> float:
    return math.atan2(rotation[1, 0], rotation[0, 0])


def _yaw_change_deg(a: float, b: float) -> float:
    return abs(math.degrees((a - b + math.pi) % (2 * math.pi) - math.pi))


def _rotation_deg(r0, r1) -> float:
    c = (float(np.trace(r0.T @ r1)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _tilt_deg(rotation) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(rotation[2, 2])))))


class PickJudge:
    def __init__(self, named: str, spare: str, rules: dict, steps: StepCounts, table_center, table_half):
        self.named, self.spare, self.rules, self.steps = named, spare, rules, steps
        self.table_center, self.table_half = table_center, table_half
        self.start = None
        self.failures = {}
        self.success_at = None
        self.hold_ok_at = None
        self._window = deque(maxlen=steps.hold)
        self._streak = 0
        self.longest_streak = 0
        self.lift_step = {named: None, spare: None}
        self.max_lift = {named: 0.0, spare: 0.0}
        self.spare_shift = self.spare_yaw = self.cup_shift = self.cup_tilt = 0.0

    @property
    def succeeded(self) -> bool:
        return self.success_at is not None

    def start_from(self, facts: SubstepFacts):
        self.start = {item: (np.array(facts.positions[item], float), np.array(facts.rotations[item], float))
                      for item in facts.positions}

    def _fail(self, label, at, detail=None) -> bool:
        if label in self.failures:
            return False
        self.failures[label] = {"label": label, "rule": LABEL_RULE[label], "physics_step": at.physics_step,
                                "control_step": at.control_step, "detail": detail}
        return True

    def record_error(self, label: str, physics_step: int, control_step: int, detail: str):
        if label not in ("INVALID_ACTION", "POLICY_ERROR"):
            raise ValueError(f"record_error takes policy failures only, got {label}")
        self._fail(label, SimpleNamespace(physics_step=physics_step, control_step=control_step), detail)

    def update(self, f: SubstepFacts) -> list:
        if self.start is None:
            raise RuntimeError("start_from() must be called before update()")
        if self.succeeded:
            # The episode ends at the physics step that completed the hold; a caller that
            # keeps stepping would hide later contacts, so this is an error, not a no-op.
            raise RuntimeError("update() after success: the runner must stop at the success step")
        r, new = self.rules, []
        fixed = moving = named_other = False
        for verdict in f.verdicts:
            if verdict.kind == JAW_UTENSIL:
                fixed |= verdict.jaw == "fixed"
                moving |= verdict.jaw == "moving"
            elif any(name.startswith(self.named + "_") for name in verdict.geoms):
                named_other = True
            for label in verdict.labels:
                if self._fail(label, f, list(verdict.geoms)):
                    new.append(label)
        p0, r0 = self.start[self.spare]
        self.spare_shift = max(self.spare_shift, float(np.linalg.norm(f.positions[self.spare] - p0)))
        self.spare_yaw = max(self.spare_yaw, _yaw_change_deg(_yaw(f.rotations[self.spare]), _yaw(r0)))
        if self.spare_shift >= r["spare_max_shift_m"] or self.spare_yaw >= r["spare_max_yaw_deg"]:
            if self._fail("SPARE_DISTURBED", f, {"shift_m": self.spare_shift, "yaw_deg": self.spare_yaw}):
                new.append("SPARE_DISTURBED")
        c0, _ = self.start["cup"]
        self.cup_shift = max(self.cup_shift, float(np.linalg.norm(f.positions["cup"] - c0)))
        self.cup_tilt = max(self.cup_tilt, _tilt_deg(f.rotations["cup"]))
        if self.cup_shift >= r["cup_max_shift_m"] or self.cup_tilt >= r["cup_max_tilt_deg"]:
            if self._fail("CUP_DISTURBED", f, {"shift_m": self.cup_shift, "tilt_deg": self.cup_tilt}):
                new.append("CUP_DISTURBED")
        (cx, cy), (hx, hy) = self.table_center, self.table_half
        for item, p in f.positions.items():
            if abs(p[0] - cx) > hx or abs(p[1] - cy) > hy or p[2] < r["workspace_min_z_m"]:
                if self._fail("ITEM_FELL_OR_OUT", f, item):
                    new.append("ITEM_FELL_OR_OUT")
        for item in (self.named, self.spare):
            lift = float(f.positions[item][2] - self.start[item][0][2])
            self.max_lift[item] = max(self.max_lift[item], lift)
            if lift >= r["lift_height_m"] and self.lift_step[item] is None:
                self.lift_step[item] = f.control_step
        named_lift = float(f.positions[self.named][2] - self.start[self.named][0][2])
        if named_lift >= r["lift_height_m"] and not named_other and (fixed or moving):
            rel_p = f.gripper_rot.T @ (f.positions[self.named] - f.gripper_pos)
            rel_r = f.gripper_rot.T @ f.rotations[self.named]
            self._window.append((fixed and moving, rel_p, rel_r,
                                 f.linear_speed[self.named], f.angular_speed[self.named]))
            self._streak += 1
        else:
            self._window.clear()
            self._streak = 0
        self.longest_streak = max(self.longest_streak, self._streak)
        if len(self._window) == self.steps.hold and self._window_ok():
            if self.hold_ok_at is None:
                self.hold_ok_at = (f.physics_step, f.control_step)
            if not self.failures:
                self.success_at = (f.physics_step, f.control_step)
        return [label for label in new if label in SEVERE]

    def _window_ok(self) -> bool:
        r, w = self.rules, list(self._window)
        both = [entry[0] for entry in w]
        if sum(both) / len(w) < r["both_jaw_fraction"]:
            return False
        gap = longest = 0
        for b in both:
            gap = 0 if b else gap + 1
            longest = max(longest, gap)
        if longest > self.steps.max_gap:
            return False
        p0, r0 = w[0][1], w[0][2]
        for _, p, rot, _, _ in w:
            if np.linalg.norm(p - p0) >= r["hold_max_shift_m"] or _rotation_deg(r0, rot) >= r["hold_max_turn_deg"]:
                return False
        return all(v < r["final_max_speed_mps"] and spin < r["final_max_angular_speed_rps"]
                   for *_, v, spin in w[-self.steps.final_speed:])

    def finish(self, stop: str, *, physics_step: int, control_step: int) -> PickOutcome:
        if stop not in STOPS:
            raise ValueError(f"unknown stop: {stop}")
        if stop == "success" and not self.succeeded:
            raise ValueError("finish('success') without a completed hold")
        if self.succeeded and stop != "success":
            raise ValueError("the episode must end at success")
        if stop == "deadline" and self.hold_ok_at is None:
            at = SimpleNamespace(physics_step=physics_step, control_step=control_step)
            if self.lift_step[self.named] is None:
                self._fail("NO_LIFT", at)
            elif 0 < self._streak < self.steps.hold:
                self._fail("TIMEOUT", at)
            else:
                self._fail("IMPROPER_HOLD", at)
        ordered = sorted(self.failures.values(), key=lambda item: item["physics_step"])
        failed_rules = sorted({item["rule"] for item in ordered if item["rule"] in RULES})
        if stop == "early_stop":
            unevaluable = [rule for rule in ("R2", "R5") if rule not in failed_rules]
        elif stop == "crash":
            unevaluable = [rule for rule in RULES if rule not in failed_rules]
        else:
            unevaluable = []
        n, s = self.lift_step[self.named], self.lift_step[self.spare]
        if n is None and s is None:
            picked = "neither"
        elif s is None or (n is not None and n < s):
            picked = "named_first"
        elif n is None or s < n:
            picked = "spare_first"
        else:
            picked = "simultaneous"
        return PickOutcome(
            success=stop == "success", stop=stop, failures=ordered,
            first_failure=ordered[0]["label"] if ordered else None, failed_rules=failed_rules,
            unevaluable_rules=unevaluable, picked=picked, lifted_both=n is not None and s is not None,
            hold_completed=self.hold_ok_at is not None,
            hold_completed_physics_step=self.hold_ok_at[0] if self.hold_ok_at else None,
            hold_completed_control_step=self.hold_ok_at[1] if self.hold_ok_at else None,
            success_physics_step=self.success_at[0] if self.success_at else None,
            success_control_step=self.success_at[1] if self.success_at else None,
            max_lift_m=dict(self.max_lift), spare_max_shift_m=self.spare_shift, spare_max_yaw_deg=self.spare_yaw,
            cup_max_shift_m=self.cup_shift, cup_max_tilt_deg=self.cup_tilt,
            longest_eligible_streak=self.longest_streak)
```

- [ ] **Step 4: Run the tests**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_pick_outcome.py" -v`
Expected: 16 tests `ok`.

- [ ] **Step 5: Commit (do not push)**

```powershell
git add src/rescuehandsai/pick_outcome.py tests/test_pick_outcome.py
git commit -m "Pick plan 1 task 8: success-rule judge - latched failures, gripper-frame hold, deadline goals"
```

---

### Task 9: Per-physics-step facts and the simulation hook

**Files:**
- Modify: `src/rescuehandsai/sim.py` (`step`)
- Create: `src/rescuehandsai/pick_facts.py`
- Test: `tests/test_pick_facts.py`

**Interfaces:**
- Consumes: `ContactClassifier` (Task 5), `SubstepFacts` (Task 8).
- Produces:
  - `MujocoSimulation.step(action, *, on_substep=None, stop_on_cross_arm=True) -> int`. `on_substep()` is called after every physics step; **if it returns a truthy value, no further physics step is taken in this control step**. It returns the number of physics steps taken. The defaults keep today's behaviour, and existing callers ignore the return value.
  - `SIM_ERROR_WARNINGS = ("mjWARN_BADQACC", "mjWARN_CONTACTFULL", "mjWARN_CNSTRFULL")`
  - `SimulatorFailure(RuntimeError)`
  - `FactReader(sim, classifier, gripper_site="right_arm/gripperframe")` with `.read(physics_step, control_step) -> SubstepFacts` and `.other_warnings: dict[str, int]`

- [ ] **Step 1: Write the failing tests**

`tests/test_pick_facts.py`:

```python
import unittest

import mujoco
import numpy as np

from rescuehandsai.contracts import BimanualAction
from rescuehandsai.pick_config import load_contacts
from rescuehandsai.pick_contacts import ContactClassifier
from rescuehandsai.pick_facts import FactReader, SimulatorFailure
from rescuehandsai.sim import MujocoSimulation


class FactReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = MujocoSimulation(physics_version=2)

    @classmethod
    def tearDownClass(cls):
        cls.sim.close()

    def setUp(self):
        self.sim.reset(21)
        self.reader = FactReader(self.sim, ContactClassifier(self.sim.model, "fork", load_contacts()))

    def hold(self, **kwargs):
        self.sim.step(BimanualAction(self.sim.observe().timestamp, self.sim.home_targets), **kwargs)

    def test_read_returns_poses_speeds_and_gripper_frame(self):
        f = self.reader.read(0, 0)
        self.assertEqual(set(f.positions), {"cup", "fork", "spoon"})
        self.assertEqual(f.rotations["fork"].shape, (3, 3))
        np.testing.assert_allclose(f.gripper_pos, self.sim.data.site("right_arm/gripperframe").xpos)
        self.assertEqual(f.linear_speed["fork"], 0.0)

    def test_hook_runs_after_every_physics_step(self):
        times = []
        taken = self.sim.step(BimanualAction(self.sim.observe().timestamp, self.sim.home_targets),
                              on_substep=lambda: times.append(float(self.sim.data.time)))
        self.assertEqual((len(times), taken), (self.sim.substeps, self.sim.substeps))
        self.assertTrue(all(b > a for a, b in zip(times, times[1:])))

    def test_hook_can_stop_at_the_triggering_physics_step(self):
        calls = []

        def stop_at_third():
            calls.append(float(self.sim.data.time))
            return len(calls) == 3

        start = float(self.sim.data.time)
        taken = self.sim.step(BimanualAction(self.sim.observe().timestamp, self.sim.home_targets),
                              on_substep=stop_at_third)
        self.assertEqual((taken, len(calls)), (3, 3))
        self.assertAlmostEqual(float(self.sim.data.time) - start, 3 * self.sim.config["physics_dt"])

    def test_brief_contact_between_control_updates_is_caught(self):
        model, data = self.sim.model, self.sim.data
        plate, pad = model.geom("plate").id, model.geom("right_arm/fixed_jaw_box5").id
        home = model.geom_pos[plate].copy()
        seen, count = [], [0]

        def hook():
            # After physics step 3 only, the plate touches the jaw; it is moved back before the
            # next physics step, so no step ever integrates that contact and the end of the
            # control step shows nothing. Only a per-physics-step read can see it.
            count[0] += 1
            if count[0] == 3:
                model.geom_pos[plate] = data.geom_xpos[pad].copy()
                mujoco.mj_forward(model, data)
            f = self.reader.read(count[0], 1)
            seen.append((count[0], [v.labels for v in f.verdicts if "plate" in v.geoms]))
            if count[0] == 3:
                model.geom_pos[plate] = home
                mujoco.mj_forward(model, data)

        self.hold(on_substep=hook)
        flagged = [step for step, labels in seen if ("FORBIDDEN_CONTACT",) in labels]
        self.assertEqual(flagged, [3])
        after = self.reader.read(99, 1)
        self.assertFalse([v for v in after.verdicts if "plate" in v.geoms])

    def test_sim_error_warnings_raise_and_other_warnings_are_recorded(self):
        self.sim.data.warning[int(mujoco.mjtWarning.mjWARN_INERTIA)].number += 2
        self.reader.read(1, 1)
        self.assertEqual(self.reader.other_warnings, {"mjWARN_INERTIA": 2})
        self.sim.data.warning[int(mujoco.mjtWarning.mjWARN_BADQACC)].number += 1
        with self.assertRaises(SimulatorFailure):
            self.reader.read(2, 1)

    def test_default_step_behaviour_is_unchanged(self):
        before = self.sim.contact_samples
        self.hold()
        self.assertGreaterEqual(self.sim.contact_samples, before)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_pick_facts.py" -v`
Expected: `ModuleNotFoundError: No module named 'rescuehandsai.pick_facts'`.

- [ ] **Step 3: Modify `sim.step`**

Replace the signature and the substep loop of `MujocoSimulation.step`. Everything before `for _ in range(self.substeps):` stays as it is:

```python
    def step(self, action: BimanualAction, *, on_substep=None, stop_on_cross_arm: bool = True) -> int:
        """on_substep() runs after every physics step (pick evaluation reads contacts there);
        a truthy return stops this control step at that physics step. stop_on_cross_arm=False
        lets the pick judge score arm-arm contact instead of raising. Returns physics steps taken."""
```
(unchanged validation and fault lines)
```python
        taken = 0
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
            taken += 1
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                raise RuntimeError("SIMULATION_ERROR: nonfinite physical state")
            if on_substep is not None and on_substep():
                return taken
            for c in self.data.contact:
                if c.dist > 0:
                    continue
                self.contact_samples += 1
                a, b = self.geom_arm[c.geom1], self.geom_arm[c.geom2]
                if stop_on_cross_arm and a is not None and b is not None and a != b:
                    raise RuntimeError(f"COLLISION: {self._geom_name(int(c.geom1))} with "
                                       f"{self._geom_name(int(c.geom2))}; simulation stopped")
        return taken
```

- [ ] **Step 4: Implement `pick_facts.py`**

```python
"""Per-physics-step facts for the pick judge, read from MuJoCo after every substep (spec §5)."""
import mujoco
import numpy as np

from .pick_outcome import SubstepFacts
from .scene import SCENE_ITEMS

# Only these warnings make an episode invalid (spec §5); every other warning is recorded.
SIM_ERROR_WARNINGS = ("mjWARN_BADQACC", "mjWARN_CONTACTFULL", "mjWARN_CNSTRFULL")
WARNING_NAMES = {int(value): name for name, value in mujoco.mjtWarning.__members__.items() if name != "mjNWARNING"}


class SimulatorFailure(RuntimeError):
    """A simulator failure: the episode is invalid (SIM_ERROR), not a robot failure."""


class FactReader:
    def __init__(self, sim, classifier, gripper_site: str = "right_arm/gripperframe"):
        self.sim, self.classifier, self.site = sim, classifier, gripper_site
        model = sim.model
        self._dof = {item: int(model.jnt_dofadr[model.joint(item + "_free").id]) for item in SCENE_ITEMS}
        self._fatal = {int(getattr(mujoco.mjtWarning, name)) for name in SIM_ERROR_WARNINGS}
        self._baseline = self._counts()
        self.other_warnings = {}

    def _counts(self) -> list:
        return [int(w.number) for w in self.sim.data.warning]

    def read(self, physics_step: int, control_step: int) -> SubstepFacts:
        data = self.sim.data
        counts = self._counts()
        for index, (now, before) in enumerate(zip(counts, self._baseline)):
            if now <= before:
                continue
            name = WARNING_NAMES.get(index, f"warning_{index}")
            if index in self._fatal:
                raise SimulatorFailure(f"{name} at physics step {physics_step}")
            self.other_warnings[name] = now - before
        positions, rotations, linear, angular = {}, {}, {}, {}
        for item in SCENE_ITEMS:
            body = data.body(item)
            positions[item] = body.xpos.copy()
            rotations[item] = body.xmat.reshape(3, 3).copy()
            v = data.qvel[self._dof[item]:self._dof[item] + 6]
            linear[item] = float(np.linalg.norm(v[:3]))
            angular[item] = float(np.linalg.norm(v[3:]))
        site = data.site(self.site)
        return SubstepFacts(physics_step, control_step, positions, rotations, linear, angular,
                            site.xpos.copy(), site.xmat.reshape(3, 3).copy(), tuple(self.classifier.contacts(data)))
```

- [ ] **Step 5: Run the new tests and the whole suite**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_pick_facts.py" -v`
Expected: 6 tests `ok`.

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests`
Expected: `OK`.

- [ ] **Step 6: Commit (do not push)**

```powershell
git add src/rescuehandsai/sim.py src/rescuehandsai/pick_facts.py tests/test_pick_facts.py
git commit -m "Pick plan 1 task 9: per-physics-step fact reader; sim.step hook keeps default behaviour"
```

---

### Task 10: Validity labels, attempts ledger, contract and resume checks

**Files:**
- Create: `src/rescuehandsai/pick_records.py`
- Test: `tests/test_pick_records.py`

**Interfaces:**
- Produces:
  - `INVALID_LABELS = ("SIM_ERROR", "MODEL_LOAD_ERROR", "CONTRACT_MISMATCH")`
  - `InvalidRun(label, detail, partial=None)` (Exception with `.label`, `.detail`, `.partial`)
  - `EvaluationBlocked`, `RetryNeedsDiagnosis`, `AlreadyScored`, `ResumeRefused` (all `RuntimeError`)
  - `load_model_or_invalid(factory)`, `check_contract(expected: dict, actual: dict) -> None`
  - `episode_key(seed, cell) -> str`, `episode_filename(seed, cell, attempt) -> str`, `write_episode(run_dir, seed, cell, attempt, record) -> Path`
  - `AttemptLedger(run_dir)`: `.attempts(key)`, `.next_attempt(key) -> int`, `.record(key, attempt, *, valid, label, filename)`, `.add_diagnosis(key, attempt, text)`, `.scored(key)`, `.blocked_keys()`, `.invalid_counts() -> dict`
  - `RESUME_KEYS`, `check_resume(run_dir, current: dict) -> None`

- [ ] **Step 1: Write the failing tests**

`tests/test_pick_records.py`:

```python
import json
import tempfile
import unittest
from pathlib import Path

from rescuehandsai.pick_records import (RESUME_KEYS, AlreadyScored, AttemptLedger, EvaluationBlocked, InvalidRun,
                                        ResumeRefused, RetryNeedsDiagnosis, check_contract, check_resume,
                                        episode_filename, episode_key, load_model_or_invalid, write_episode)


class LabelTests(unittest.TestCase):
    def test_only_the_three_invalid_labels_exist(self):
        for label in ("SIM_ERROR", "MODEL_LOAD_ERROR", "CONTRACT_MISMATCH"):
            self.assertEqual(InvalidRun(label, "x").label, label)
        for label in ("POLICY_ERROR", "SIMULATION_ERROR", "TIMEOUT"):
            with self.assertRaises(ValueError):
                InvalidRun(label, "x")

    def test_model_load_failure_is_invalid(self):
        def broken():
            raise OSError("checkpoint missing")
        with self.assertRaises(InvalidRun) as ctx:
            load_model_or_invalid(broken)
        self.assertEqual(ctx.exception.label, "MODEL_LOAD_ERROR")
        self.assertEqual(load_model_or_invalid(lambda: "model"), "model")

    def test_contract_mismatch_names_every_difference(self):
        check_contract({"physics_version": 2, "cameras": ["a"]}, {"physics_version": 2, "cameras": ["a"], "x": 1})
        with self.assertRaises(InvalidRun) as ctx:
            check_contract({"physics_version": 2, "state_order": ["a", "b"]}, {"physics_version": 1, "state_order": ["b", "a"]})
        self.assertEqual(ctx.exception.label, "CONTRACT_MISMATCH")
        self.assertIn("physics_version", ctx.exception.detail)
        self.assertIn("state_order", ctx.exception.detail)


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmp.name)
        self.key = episode_key(3100007, "F-B")

    def tearDown(self):
        self.tmp.cleanup()

    def test_valid_failure_is_never_replaced(self):
        ledger = AttemptLedger(self.run_dir)
        self.assertEqual(ledger.next_attempt(self.key), 1)
        ledger.record(self.key, 1, valid=True, label="NO_LIFT", filename=episode_filename(3100007, "F-B", 1))
        with self.assertRaises(AlreadyScored):
            ledger.next_attempt(self.key)
        self.assertEqual(ledger.scored(self.key)["attempt"], 1)

    def test_one_diagnosed_retry_then_block(self):
        ledger = AttemptLedger(self.run_dir)
        ledger.record(self.key, 1, valid=False, label="SIM_ERROR", filename="episode_3100007_F-B_a1.json")
        with self.assertRaises(RetryNeedsDiagnosis):
            ledger.next_attempt(self.key)
        with self.assertRaises(ValueError):
            ledger.add_diagnosis(self.key, 1, "   ")
        ledger.add_diagnosis(self.key, 1, "EGL context lost after driver sleep; restarted renderer")
        self.assertEqual(ledger.next_attempt(self.key), 2)
        ledger.record(self.key, 2, valid=False, label="SIM_ERROR", filename="episode_3100007_F-B_a2.json")
        with self.assertRaises(EvaluationBlocked):
            ledger.next_attempt(self.key)
        self.assertEqual(ledger.blocked_keys(), [self.key])
        self.assertEqual(ledger.invalid_counts(), {"SIM_ERROR": 2})

    def test_ledger_survives_reload_and_rejects_out_of_order_attempts(self):
        ledger = AttemptLedger(self.run_dir)
        with self.assertRaises(ValueError):
            ledger.record(self.key, 2, valid=True, label=None, filename="x")
        ledger.record(self.key, 1, valid=False, label="MODEL_LOAD_ERROR", filename="x")
        again = AttemptLedger(self.run_dir)
        self.assertEqual(len(again.attempts(self.key)), 1)

    def test_episode_files_are_never_overwritten(self):
        path = write_episode(self.run_dir, 3100007, "F-B", 1, {"success": False})
        self.assertEqual(path.name, "episode_3100007_F-B_a1.json")
        with self.assertRaises(FileExistsError):
            write_episode(self.run_dir, 3100007, "F-B", 1, {"success": True})


class ResumeTests(unittest.TestCase):
    def test_resume_requires_every_key_to_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            current = {key: {"value": key} for key in RESUME_KEYS}
            with self.assertRaises(ResumeRefused):
                check_resume(tmp, current)
            (Path(tmp) / "manifest.json").write_text(json.dumps(current))
            check_resume(tmp, current)
            changed = dict(current, snapshot={"source_sha256": "other"})
            with self.assertRaises(ResumeRefused) as ctx:
                check_resume(tmp, changed)
            self.assertIn("snapshot", str(ctx.exception))
            with self.assertRaises(ValueError):
                check_resume(tmp, {"model": 1})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_pick_records.py" -v`
Expected: `ModuleNotFoundError: No module named 'rescuehandsai.pick_records'`.

- [ ] **Step 3: Implement**

`src/rescuehandsai/pick_records.py`:

```python
"""Run validity, attempts and resume rules for pick evaluations (spec §6, §11)."""
import json
from pathlib import Path

INVALID_LABELS = ("SIM_ERROR", "MODEL_LOAD_ERROR", "CONTRACT_MISMATCH")
RESUME_KEYS = ("model", "snapshot", "scene_list_sha256", "physics_version", "config_hash", "rules_sha256",
               "run_config")


class InvalidRun(Exception):
    """Broken test equipment: never scored. `partial` keeps the measurements taken before the failure."""

    def __init__(self, label: str, detail: str, partial: dict | None = None):
        if label not in INVALID_LABELS:
            raise ValueError(f"not an invalid-run label: {label}")
        super().__init__(f"{label}: {detail}")
        self.label, self.detail, self.partial = label, detail, partial


class EvaluationBlocked(RuntimeError):
    """A second invalid attempt for the same episode: stop until it is understood."""


class RetryNeedsDiagnosis(RuntimeError):
    """An invalid attempt may be retried once, only after its cause is written down."""


class AlreadyScored(RuntimeError):
    """This episode already has a valid result; valid failures are never replaced."""


class ResumeRefused(RuntimeError):
    """The experiment changed; use a new run directory."""


def load_model_or_invalid(factory):
    try:
        return factory()
    except Exception as exc:
        raise InvalidRun("MODEL_LOAD_ERROR", f"{type(exc).__name__}: {exc}") from exc


def check_contract(expected: dict, actual: dict) -> None:
    diff = {key: {"expected": value, "actual": actual.get(key)}
            for key, value in expected.items() if actual.get(key) != value}
    if diff:
        raise InvalidRun("CONTRACT_MISMATCH", json.dumps(diff, sort_keys=True, default=str))


def episode_key(seed: int, cell: str) -> str:
    return f"{seed}_{cell}"


def episode_filename(seed: int, cell: str, attempt: int) -> str:
    return f"episode_{seed}_{cell}_a{attempt}.json"


def write_episode(run_dir, seed: int, cell: str, attempt: int, record: dict) -> Path:
    path = Path(run_dir) / episode_filename(seed, cell, attempt)
    with path.open("x", encoding="utf-8") as handle:  # "x": an attempt file is never overwritten
        json.dump(record, handle, indent=2, default=str)
    return path


class AttemptLedger:
    def __init__(self, run_dir):
        self.path = Path(run_dir) / "attempts.jsonl"
        self.lines = ([json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
                      if self.path.is_file() else [])

    def _append(self, line: dict):
        self.lines.append(line)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, sort_keys=True) + "\n")

    def attempts(self, key: str) -> list:
        return [line for line in self.lines if line["type"] == "attempt" and line["key"] == key]

    def _diagnosed(self, key: str) -> set:
        return {line["attempt"] for line in self.lines if line["type"] == "diagnosis" and line["key"] == key}

    def next_attempt(self, key: str) -> int:
        attempts = self.attempts(key)
        if any(a["valid"] for a in attempts):
            raise AlreadyScored(f"{key} already has a valid result")
        invalid = [a for a in attempts if not a["valid"]]
        if len(invalid) >= 2:
            labels = ", ".join(a["label"] for a in invalid)
            raise EvaluationBlocked(f"{key}: two invalid attempts ({labels}); understand the cause before continuing")
        if invalid and invalid[-1]["attempt"] not in self._diagnosed(key):
            raise RetryNeedsDiagnosis(f"{key}: attempt {invalid[-1]['attempt']} was invalid "
                                      f"({invalid[-1]['label']}); write its diagnosis first")
        return len(attempts) + 1

    def add_diagnosis(self, key: str, attempt: int, text: str):
        if not text.strip():
            raise ValueError("a diagnosis must say what went wrong")
        if not any(a["attempt"] == attempt and not a["valid"] for a in self.attempts(key)):
            raise ValueError("a diagnosis must name an invalid attempt of this episode")
        self._append({"type": "diagnosis", "key": key, "attempt": attempt, "text": text})

    def record(self, key: str, attempt: int, *, valid: bool, label: str | None, filename: str):
        expected = len(self.attempts(key)) + 1
        if attempt != expected:
            raise ValueError(f"{key}: expected attempt {expected}, got {attempt}")
        self._append({"type": "attempt", "key": key, "attempt": attempt, "valid": valid, "label": label,
                      "file": filename})

    def scored(self, key: str):
        return next((a for a in self.attempts(key) if a["valid"]), None)

    def blocked_keys(self) -> list:
        keys = sorted({line["key"] for line in self.lines if line["type"] == "attempt"})
        return [k for k in keys if sum(not a["valid"] for a in self.attempts(k)) >= 2]

    def invalid_counts(self) -> dict:
        counts = {}
        for line in self.lines:
            if line["type"] == "attempt" and not line["valid"]:
                counts[line["label"]] = counts.get(line["label"], 0) + 1
        return counts


def check_resume(run_dir, current: dict) -> None:
    missing = [key for key in RESUME_KEYS if key not in current]
    if missing:
        raise ValueError(f"the current run description lacks {missing}")
    manifest = Path(run_dir) / "manifest.json"
    if not manifest.is_file():
        raise ResumeRefused(f"no manifest.json in {run_dir}; nothing to resume")
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    now = json.loads(json.dumps({key: current[key] for key in RESUME_KEYS}, default=str))
    changed = [key for key in RESUME_KEYS if saved.get(key) != now[key]]
    if changed:
        raise ResumeRefused(f"experiment changed ({', '.join(changed)}); use a new run directory")
```

- [ ] **Step 4: Run the tests**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_pick_records.py" -v`
Expected: 8 tests `ok`.

- [ ] **Step 5: Commit (do not push)**

```powershell
git add src/rescuehandsai/pick_records.py tests/test_pick_records.py
git commit -m "Pick plan 1 task 10: invalid labels, attempts ledger with one diagnosed retry, resume checks"
```

---

### Task 11: Snapshot record

**Files:**
- Create: `src/rescuehandsai/snapshot.py`
- Test: `tests/test_snapshot.py`

**Interfaces:**
- Consumes: `text_digest` idea from Task 7 (re-implemented here per suffix, because snapshot files include binaries).
- Produces: `SNAPSHOT_DIRS`, `SnapshotRefused(RuntimeError)`, `file_digest(path) -> str`, `source_hash(root) -> tuple[str, dict]`, `uncommitted_files(root) -> list[str]`, `asset_files(asset_xml) -> list[Path]`, `asset_hash(asset_xml) -> tuple[str, dict]`, `runtime_versions() -> dict`, `snapshot_record(root, asset_xml, *, strict, model_files=None, noise_file=None) -> dict`.

- [ ] **Step 1: Write the failing tests**

`tests/test_snapshot.py`:

```python
import subprocess
import tempfile
import unittest
from pathlib import Path

from rescuehandsai.snapshot import (SnapshotRefused, asset_files, asset_hash, file_digest, runtime_versions,
                                    snapshot_record, source_hash, uncommitted_files)


def git(root, *args):
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                   check=True, capture_output=True)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "src").mkdir()
        (self.root / "configs").mkdir()
        (self.root / "src" / "a.py").write_text("x = 1\n")
        (self.root / "configs" / "c.json").write_text("{}\n")
        assets = self.root / "asset" / "meshes"
        assets.mkdir(parents=True)
        (assets / "part.stl").write_bytes(b"\x00\x01binary")
        self.xml = self.root / "asset" / "robot.xml"
        self.xml.write_text('<mujoco><compiler meshdir="meshes"/><asset><mesh file="part.stl"/></asset></mujoco>\n')
        git(self.root, "init", "-q")
        git(self.root, "add", ".")
        git(self.root, "commit", "-q", "-m", "init")

    def tearDown(self):
        self.tmp.cleanup()

    def test_text_hash_ignores_crlf_but_binary_hash_does_not(self):
        a, b = self.root / "x.py", self.root / "y.py"
        a.write_bytes(b"a = 1\n")
        b.write_bytes(b"a = 1\r\n")
        self.assertEqual(file_digest(a), file_digest(b))
        c, d = self.root / "x.stl", self.root / "y.stl"
        c.write_bytes(b"\n")
        d.write_bytes(b"\r\n")
        self.assertNotEqual(file_digest(c), file_digest(d))

    def test_untracked_source_changes_hash_and_blocks_strict_runs(self):
        clean_digest, _ = source_hash(self.root)
        self.assertEqual(snapshot_record(self.root, self.xml, strict=True)["source_sha256"], clean_digest)
        (self.root / "src" / "extra.py").write_text("y = 2\n")
        self.assertEqual(uncommitted_files(self.root), ["src/extra.py"])
        self.assertNotEqual(source_hash(self.root)[0], clean_digest)
        with self.assertRaises(SnapshotRefused):
            snapshot_record(self.root, self.xml, strict=True)
        record = snapshot_record(self.root, self.xml, strict=False)
        self.assertIn("src/extra.py", record["uncommitted"])
        self.assertIsNotNone(record["uncommitted"]["src/extra.py"])

    def test_pycache_is_ignored(self):
        before = source_hash(self.root)[0]
        cache = self.root / "src" / "__pycache__"
        cache.mkdir()
        (cache / "a.cpython-312.pyc").write_bytes(b"junk")
        self.assertEqual(source_hash(self.root)[0], before)

    def test_asset_hash_covers_referenced_meshes(self):
        self.assertEqual([p.name for p in asset_files(self.xml)], ["robot.xml", "part.stl"])
        before = asset_hash(self.xml)[0]
        (self.root / "asset" / "meshes" / "part.stl").write_bytes(b"changed")
        self.assertNotEqual(asset_hash(self.xml)[0], before)
        (self.root / "asset" / "meshes" / "part.stl").unlink()
        with self.assertRaises(FileNotFoundError):
            asset_files(self.xml)

    def test_model_and_noise_files_and_runtime_are_recorded(self):
        model = self.root / "model.bin"
        model.write_bytes(b"weights")
        noise = self.root / "noise.npy"
        noise.write_bytes(b"noise")
        record = snapshot_record(self.root, self.xml, strict=True, model_files={"export": model}, noise_file=noise)
        self.assertEqual(record["model_files"]["export"], file_digest(model))
        self.assertEqual(record["noise_sha256"], file_digest(noise))
        runtime = runtime_versions()
        self.assertIn("python", runtime)
        self.assertIn("mujoco", runtime["packages"])
        self.assertIn("devices", runtime)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_snapshot.py" -v`
Expected: `ModuleNotFoundError: No module named 'rescuehandsai.snapshot'`.

- [ ] **Step 3: Implement**

`src/rescuehandsai/snapshot.py`:

```python
"""What actually ran: source, assets, runtime and model files (spec §11 snapshot record).

A hash of tracked files alone can miss code that runs, so every file under the snapshot
folders is hashed, tracked or not, and strict runs refuse uncommitted changes there."""
import hashlib
from importlib.metadata import PackageNotFoundError, version
import platform
import re
import subprocess
from pathlib import Path

SNAPSHOT_DIRS = ("src", "scripts", "training", "configs")
TEXT_SUFFIXES = {".py", ".json", ".jsonl", ".xml", ".sh", ".md", ".txt", ".toml", ".yaml", ".yml", ".cfg", ".ipynb"}
PACKAGES = ("mujoco", "numpy", "torch", "lerobot", "transformers", "openvino", "nncf")


class SnapshotRefused(RuntimeError):
    """Data generation, training and the final test need committed source."""


def file_digest(path) -> str:
    path = Path(path)
    data = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES:
        data = data.replace(b"\r\n", b"\n")  # Windows checkouts convert line endings; hash the content
    return hashlib.sha256(data).hexdigest()


def _source_files(root: Path) -> list:
    files = []
    for folder in SNAPSHOT_DIRS:
        base = root / folder
        if base.is_dir():
            files += [p for p in base.rglob("*")
                      if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    return sorted(files)


def source_hash(root) -> tuple:
    root = Path(root)
    files = {p.relative_to(root).as_posix(): file_digest(p) for p in _source_files(root)}
    digest = hashlib.sha256("\n".join(f"{name} {h}" for name, h in sorted(files.items())).encode()).hexdigest()
    return digest, files


def _git(root, *args) -> str:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, encoding="utf-8",
                          errors="replace", check=True).stdout


def uncommitted_files(root) -> list:
    out = _git(root, "status", "--porcelain", "--untracked-files=all", "--", *SNAPSHOT_DIRS)
    return sorted(line[3:].strip().strip('"') for line in out.splitlines()
                  if line.strip() and "__pycache__" not in line)


def asset_files(asset_xml) -> list:
    asset_xml = Path(asset_xml)
    text = asset_xml.read_text(encoding="utf-8")
    meshdir = re.search(r'<compiler[^>]*\bmeshdir="([^"]+)"', text)
    base = asset_xml.parent / (meshdir.group(1) if meshdir else "")
    files = [asset_xml] + [base / name for name in re.findall(r'<mesh[^>]*\bfile="([^"]+)"', text)]
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"asset files missing: {missing}")
    return files


def asset_hash(asset_xml) -> tuple:
    parent = Path(asset_xml).parent
    files = {p.relative_to(parent).as_posix(): file_digest(p) for p in asset_files(asset_xml)}
    digest = hashlib.sha256("\n".join(f"{name} {h}" for name, h in sorted(files.items())).encode()).hexdigest()
    return digest, files


def _devices() -> dict:
    devices = {}
    try:
        import openvino as ov
        core = ov.Core()
        for name in core.available_devices:
            info = {"name": str(core.get_property(name, "FULL_DEVICE_NAME"))}
            try:
                info["driver_version"] = str(core.get_property(name, "GPU_DRIVER_VERSION"))
            except Exception:
                info["driver_version"] = "unavailable"
            devices[f"openvino:{name}"] = info
    except ImportError:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            devices["cuda:0"] = {"name": torch.cuda.get_device_name(0), "cuda": str(torch.version.cuda)}
    except ImportError:
        pass
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
        if out.returncode == 0 and out.stdout.strip():
            devices["nvidia-smi"] = out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return devices


def runtime_versions() -> dict:
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "os": platform.platform(), "processor": platform.processor(),
            "packages": packages, "devices": _devices()}


def snapshot_record(root, asset_xml, *, strict: bool, model_files: dict | None = None, noise_file=None) -> dict:
    root = Path(root)
    dirty = uncommitted_files(root)
    if strict and dirty:
        raise SnapshotRefused(f"uncommitted or untracked files in {SNAPSHOT_DIRS}: {dirty}")
    source_digest, files = source_hash(root)
    asset_digest, assets = asset_hash(asset_xml)
    return {
        "git_revision": _git(root, "rev-parse", "HEAD").strip(),
        "source_sha256": source_digest,
        "uncommitted": {name: files.get(name) for name in dirty},  # None: deleted
        "asset_sha256": asset_digest,
        "asset_files": assets,
        "runtime": runtime_versions(),
        "model_files": {name: file_digest(path) for name, path in (model_files or {}).items()},
        "noise_sha256": file_digest(noise_file) if noise_file is not None else None,
    }
```

- [ ] **Step 4: Run the tests**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_snapshot.py" -v`
Expected: 5 tests `ok`.

- [ ] **Step 5: Commit (do not push)**

```powershell
git add src/rescuehandsai/snapshot.py tests/test_snapshot.py
git commit -m "Pick plan 1 task 11: snapshot record - all source files, referenced meshes, runtime, strict refusal"
```

---

### Task 12: Scene-grouped statistics and pass bars

**Files:**
- Create: `src/rescuehandsai/pick_stats.py`
- Test: `tests/test_pick_stats.py`

**Interfaces:**
- Consumes: `CELLS`, `named_slot` (Task 6).
- Produces: `MAIN_TEST_BARS`, `INCOMPLETE`, `scored_entry(record: dict, attempt: int) -> dict`, `summarize(scored, *, planned_keys, blocked_keys=(), invalid_counts=None, bars=None, n_boot=2000, boot_seed=0) -> dict`.

A scored entry has the shape `{"scene": int, "cell": str, "template": str, "valid": True, "attempt": int, "success": bool, "picked": str, "failure_labels": list[str]}`.

- [ ] **Step 1: Write the failing tests**

`tests/test_pick_stats.py`:

```python
import unittest

from rescuehandsai.pick_cells import CELLS
from rescuehandsai.pick_stats import INCOMPLETE, MAIN_TEST_BARS, scored_entry, summarize


def entry(scene, cell, success=True, template="T1", labels=(), picked="named_first"):
    return {"scene": scene, "cell": cell, "template": template, "valid": True, "attempt": 1, "success": success,
            "picked": picked, "failure_labels": list(labels)}


def keys(scenes):
    return [f"{s}_{c}" for s in scenes for c in CELLS]


class SummaryTests(unittest.TestCase):
    def test_hand_worked_example(self):
        scored = [entry(1, c) for c in CELLS] + [entry(2, "F-A"), entry(2, "F-B"),
                                                 entry(2, "S-A", False, labels=["WRONG_ITEM_TOUCHED"], picked="spare_first"),
                                                 entry(2, "S-B", False, labels=["NO_LIFT"], picked="neither")]
        s = summarize(scored, planned_keys=keys([1, 2]))
        self.assertTrue(s["evaluation_valid"])
        self.assertEqual((s["episodes"]["successes"], s["episodes"]["episodes"]), (6, 8))
        self.assertEqual((s["scenes_all_four"]["passing"], s["scenes_all_four"]["complete_scenes"]), (1, 2))
        self.assertEqual(s["per_cell"]["S-A"], {"successes": 1, "episodes": 2})
        self.assertEqual(s["per_word"]["spoon"], {"successes": 2, "episodes": 4})
        self.assertEqual(s["per_slot"]["1"], {"successes": 3, "episodes": 4})  # F-B and S-A
        self.assertEqual(s["spare_touch_episodes"], 1)
        self.assertEqual(s["picked"], {"named_first": 6, "spare_first": 1, "neither": 1})
        self.assertEqual(s["failure_labels"], {"WRONG_ITEM_TOUCHED": 1, "NO_LIFT": 1})
        self.assertEqual(s["verdict"], "no pass bars for this run")

    def test_scene_grouping_widens_the_interval(self):
        scored = []
        for scene in range(10):
            scored += [entry(scene, c, success=scene < 5) for c in CELLS]
        s = summarize(scored, planned_keys=keys(range(10)), n_boot=4000)
        low, high = s["episodes"]["ci95"]
        self.assertAlmostEqual(s["episodes"]["rate"], 0.5)
        self.assertGreater(high - low, 0.45)  # 40 independent episodes would give about 0.31
        self.assertEqual(s, summarize(scored, planned_keys=keys(range(10)), n_boot=4000))

    def test_invalid_and_duplicate_entries_are_rejected(self):
        bad = dict(entry(1, "F-A"), valid=False)
        with self.assertRaises(ValueError):
            summarize([bad], planned_keys=keys([1]))
        with self.assertRaises(ValueError):
            summarize([entry(1, "F-A"), entry(1, "F-A")], planned_keys=keys([1]))
        with self.assertRaises(ValueError):
            summarize([entry(9, "F-A")], planned_keys=keys([1]))

    def test_incomplete_or_blocked_runs_are_not_eligible(self):
        scored = [entry(1, c) for c in CELLS]
        s = summarize(scored, planned_keys=keys([1, 2]), bars=MAIN_TEST_BARS)
        self.assertFalse(s["evaluation_valid"])
        self.assertEqual((s["verdict"], s["bars"]), (INCOMPLETE, None))
        self.assertEqual(len(s["missing_episodes"]), 4)
        s = summarize(scored, planned_keys=keys([1]), blocked_keys=["1_F-A"], bars=MAIN_TEST_BARS)
        self.assertEqual(s["verdict"], INCOMPLETE)

    def test_main_test_bars(self):
        scored = [entry(scene, c) for scene in range(100) for c in CELLS]
        s = summarize(scored, planned_keys=keys(range(100)), bars=MAIN_TEST_BARS, n_boot=200)
        self.assertEqual(s["verdict"], "pass")
        failing = [dict(e, success=False) if e["cell"] == "F-A" and e["scene"] < 9 else e for e in scored]
        s = summarize(failing, planned_keys=keys(range(100)), bars=MAIN_TEST_BARS, n_boot=200)
        self.assertEqual(s["bars"]["each_cell"]["pass"], False)  # F-A 91 < 92
        self.assertEqual(s["bars"]["episode_successes"]["pass"], True)  # 391 >= 380
        self.assertEqual(s["verdict"], "fail")

    def test_scored_entry_from_a_runner_record(self):
        record = {"seed": 3100001, "cell": "S-B", "template": "T2", "success": False,
                  "outcome": {"picked": "spare_first", "failures": [{"label": "WRONG_ITEM_TOUCHED"}]}}
        self.assertEqual(scored_entry(record, 2), {"scene": 3100001, "cell": "S-B", "template": "T2", "valid": True,
                                                   "attempt": 2, "success": False, "picked": "spare_first",
                                                   "failure_labels": ["WRONG_ITEM_TOUCHED"]})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_pick_stats.py" -v`
Expected: `ModuleNotFoundError: No module named 'rescuehandsai.pick_stats'`.

- [ ] **Step 3: Implement**

`src/rescuehandsai/pick_stats.py`:

```python
"""Pick evaluation summaries (spec §11): the scene is the unit, and incomplete runs never pass."""
import numpy as np

from .pick_cells import CELLS, named_slot

MAIN_TEST_BARS = {"episode_successes_min": 380, "cell_successes_min": 92, "spare_touch_episodes_max": 4,
                  "scenes_all_four_min": 92}
INCOMPLETE = "incomplete — not eligible to pass"


def scored_entry(record: dict, attempt: int) -> dict:
    """The fields a summary needs from one valid runner record."""
    return {"scene": record["seed"], "cell": record["cell"], "template": record["template"], "valid": True,
            "attempt": attempt, "success": bool(record["success"]), "picked": record["outcome"]["picked"],
            "failure_labels": [f["label"] for f in record["outcome"]["failures"]]}


def _groups(scored, key) -> dict:
    out = {}
    for e in scored:
        group = out.setdefault(key(e), {"successes": 0, "episodes": 0})
        group["successes"] += int(e["success"])
        group["episodes"] += 1
    return out


def _count(values) -> dict:
    out = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return out


def _scene_bootstrap(scenes: dict, n_boot: int, seed: int):
    ids = sorted(scenes)
    if not ids:
        return None, None
    successes = np.array([sum(e["success"] for e in scenes[s]) for s in ids], float)
    episodes = np.array([len(scenes[s]) for s in ids], float)
    complete = np.array([len(scenes[s]) == len(CELLS) for s in ids], float)
    all_four = np.array([len(scenes[s]) == len(CELLS) and all(e["success"] for e in scenes[s]) for s in ids], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ids), size=(n_boot, len(ids)))  # resample whole scenes
    episode_rates = successes[idx].sum(1) / episodes[idx].sum(1)
    complete_counts = complete[idx].sum(1)
    scene_rates = np.where(complete_counts > 0, all_four[idx].sum(1) / np.maximum(complete_counts, 1), np.nan)
    ci = lambda x: [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]
    return ci(episode_rates), (ci(scene_rates) if complete.any() else None)


def summarize(scored, *, planned_keys, blocked_keys=(), invalid_counts=None, bars=None, n_boot: int = 2000,
              boot_seed: int = 0) -> dict:
    for e in scored:
        if not e.get("valid", False):
            raise ValueError("summaries take scored valid attempts only; invalid attempts never enter denominators")
    keys = [f"{e['scene']}_{e['cell']}" for e in scored]
    if len(set(keys)) != len(keys):
        raise ValueError("more than one scored attempt for an episode")
    planned = set(planned_keys)
    extra = sorted(set(keys) - planned)
    if extra:
        raise ValueError(f"scored episodes outside the plan: {extra}")
    missing = sorted(planned - set(keys))
    eligible = not missing and not blocked_keys
    scenes = {}
    for e in scored:
        scenes.setdefault(e["scene"], []).append(e)
    complete = {s: es for s, es in scenes.items() if len(es) == len(CELLS)}
    passing = sum(all(e["success"] for e in es) for es in complete.values())
    n, successes = len(scored), sum(e["success"] for e in scored)
    episode_ci, scene_ci = _scene_bootstrap(scenes, n_boot, boot_seed)
    per_cell = _groups(scored, lambda e: e["cell"])
    spare_touches = sum("WRONG_ITEM_TOUCHED" in e["failure_labels"] for e in scored)
    summary = {
        "evaluation_valid": eligible,
        "planned_episodes": len(planned), "scored_episodes": n,
        "missing_episodes": missing, "blocked_episodes": sorted(blocked_keys),
        "invalid_attempts": dict(invalid_counts or {}),
        "episodes": {"successes": successes, "episodes": n, "rate": successes / n if n else None, "ci95": episode_ci},
        "scenes_all_four": {"passing": passing, "complete_scenes": len(complete),
                            "rate": passing / len(complete) if complete else None, "ci95": scene_ci},
        "per_cell": per_cell,
        "per_slot": _groups(scored, lambda e: str(named_slot(e["cell"]))),
        "per_word": _groups(scored, lambda e: CELLS[e["cell"]][0]),
        "per_template": _groups(scored, lambda e: e["template"]),
        "spare_touch_episodes": spare_touches,
        "picked": _count(e["picked"] for e in scored),
        "failure_labels": _count(label for e in scored for label in e["failure_labels"]),
        "attempt_used": {f"{e['scene']}_{e['cell']}": e["attempt"] for e in scored},
    }
    if not eligible:
        summary["bars"], summary["verdict"] = None, INCOMPLETE
    elif bars is None:
        summary["bars"], summary["verdict"] = None, "no pass bars for this run"
    else:
        checks = {
            "episode_successes": {"value": successes, "min": bars["episode_successes_min"],
                                  "pass": successes >= bars["episode_successes_min"]},
            "each_cell": {"values": {c: per_cell.get(c, {"successes": 0})["successes"] for c in CELLS},
                          "min": bars["cell_successes_min"],
                          "pass": all(per_cell.get(c, {"successes": 0})["successes"] >= bars["cell_successes_min"]
                                      for c in CELLS)},
            "spare_touch_episodes": {"value": spare_touches, "max": bars["spare_touch_episodes_max"],
                                     "pass": spare_touches <= bars["spare_touch_episodes_max"]},
            "scenes_all_four": {"value": passing, "min": bars["scenes_all_four_min"],
                                "pass": passing >= bars["scenes_all_four_min"]},
        }
        summary["bars"] = checks
        summary["verdict"] = "pass" if all(c["pass"] for c in checks.values()) else "fail"
    return summary
```

- [ ] **Step 4: Run the tests**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_pick_stats.py" -v`
Expected: 6 tests `ok`.

- [ ] **Step 5: Commit (do not push)**

```powershell
git add src/rescuehandsai/pick_stats.py tests/test_pick_stats.py
git commit -m "Pick plan 1 task 12: scene-grouped summary, bootstrap by scene, incomplete runs never pass"
```

---

### Task 13: Pick episode runner

**Files:**
- Create: `src/rescuehandsai/pick_runner.py`
- Test: `tests/test_pick_runner.py`

**Interfaces:**
- Consumes: `PickTask`, `cell_params` (Task 6); `derive_steps`, `load_rules`, `load_contacts` (Task 4); `ContactClassifier` (Task 5); `config_hash`, `scene_hash`, `settings_record`, `settings_hash` (Task 7); `PickJudge` (Task 8); `FactReader`, `SimulatorFailure` (Task 9); `InvalidRun` (Task 10); `rescuehandsai.runner.clamp_action` (existing); `sim.step(on_substep=, stop_on_cross_arm=)` (Task 9); `sim.reset(params=)` (Task 2).
- Produces: `PickEpisodeRunner(sim, policy, *, rules=None, contacts=None)` with `.run(task: PickTask) -> dict`. The record keys are `seed, cell, utensil, spare, template, instruction, physics_version, policy, rules, scene, settings_sha256, scene_sha256, config_sha256, success, outcome, control_steps, physics_steps, clamped_joint_steps, other_warnings, timing`. On an invalid run it raises `InvalidRun` with `.partial` holding the same keys that were measured.

**Error sources (§6):**
- reset or observation/rendering → `SIM_ERROR`;
- physics step (`SimulatorFailure`, `RuntimeError("SIMULATION_ERROR...")`, `mujoco.FatalError`) → `SIM_ERROR`;
- `policy.reset`, `policy.wants_images`, `policy.act` → valid `POLICY_ERROR`;
- a non-finite or malformed action → valid `INVALID_ACTION`;
- physics-version mismatch → `CONTRACT_MISMATCH`.

Any other exception (a harness bug) is not caught, so the run stops loudly.

- [ ] **Step 1: Write the failing tests**

`tests/test_pick_runner.py`:

```python
import unittest
from unittest.mock import patch

import mujoco

from rescuehandsai.contracts import BimanualAction
from rescuehandsai.pick_cells import make_pick_task
from rescuehandsai.pick_config import load_rules
from rescuehandsai.pick_outcome import PickJudge
from rescuehandsai.pick_records import InvalidRun
from rescuehandsai.pick_runner import PickEpisodeRunner
from rescuehandsai.sim import MujocoSimulation


class HoldHome:
    uses_privileged_state = False

    def __init__(self, images=False):
        self.images, self.sim = images, None

    def reset(self, sim, task):
        self.sim = sim

    def wants_images(self):
        return self.images

    def act(self, obs):
        return BimanualAction(obs.timestamp, dict(self.sim.home_targets))

    def metadata(self):
        return {"name": "hold_home", "backend": "python", "device": "cpu", "checkpoint": None}


class Raising(HoldHome):
    def act(self, obs):
        raise RuntimeError("tensor shape mismatch")


class NaNAction(HoldHome):
    def act(self, obs):
        targets = dict(self.sim.home_targets)
        targets["right_arm/gripper"] = float("nan")
        return BimanualAction(obs.timestamp, targets)


class BadQacc(HoldHome):
    def act(self, obs):
        self.sim.data.warning[int(mujoco.mjtWarning.mjWARN_BADQACC)].number += 1
        return super().act(obs)


class SuccessThenCollision(PickJudge):
    """Hold completes at physics step 3; a collision would follow at step 5 of the same control step."""

    def update(self, f):
        if self.succeeded:
            return super().update(f)  # raises: the runner must have stopped
        if f.physics_step == 3:
            self.success_at = self.hold_ok_at = (f.physics_step, f.control_step)
            return []
        if f.physics_step == 5:
            self._fail("ARM_ARM_CONTACT", f, "late collision")
            return ["ARM_ARM_CONTACT"]
        return []


class SevereAtFour(PickJudge):
    def update(self, f):
        if f.physics_step == 4:
            self._fail("ARM_ARM_CONTACT", f, "collision")
            return ["ARM_ARM_CONTACT"]
        if f.physics_step > 4:
            raise AssertionError("physics continued after a severe violation")
        return []


class PickRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = MujocoSimulation(physics_version=2)
        cls.rules = dict(load_rules(), deadline_control_steps=5)
        cls.task = make_pick_task(21, "F-B", "T1")  # a test seed outside every data/test block

    @classmethod
    def tearDownClass(cls):
        cls.sim.close()

    def run_policy(self, policy):
        return PickEpisodeRunner(self.sim, policy, rules=self.rules).run(self.task)

    def test_holding_home_is_a_valid_no_lift_failure_at_the_deadline(self):
        record = self.run_policy(HoldHome())
        self.assertFalse(record["success"])
        self.assertEqual(record["outcome"]["stop"], "deadline")
        self.assertEqual(record["outcome"]["first_failure"], "NO_LIFT")
        self.assertEqual(record["outcome"]["picked"], "neither")
        self.assertEqual((record["control_steps"], record["physics_steps"]), (5, 5 * self.sim.substeps))
        self.assertEqual(record["scene"]["slots"], {"spoon": 0, "fork": 1})
        self.assertEqual(len(record["scene_sha256"]), 64)

    def test_policy_exception_is_a_valid_policy_error(self):
        record = self.run_policy(Raising())
        self.assertEqual(record["outcome"]["first_failure"], "POLICY_ERROR")
        self.assertEqual(record["outcome"]["stop"], "early_stop")
        self.assertEqual(record["outcome"]["unevaluable_rules"], ["R2", "R5"])

    def test_non_finite_action_is_invalid_action(self):
        record = self.run_policy(NaNAction())
        self.assertEqual(record["outcome"]["first_failure"], "INVALID_ACTION")

    def test_renderer_failure_is_sim_error_even_during_a_policy_call(self):
        def broken(*args, **kwargs):
            raise RuntimeError("GL context lost")

        self.sim.render = broken  # instance attribute shadows the method for this test only
        try:
            with self.assertRaises(InvalidRun) as ctx:
                self.run_policy(HoldHome(images=True))
        finally:
            del self.sim.render
        self.assertEqual(ctx.exception.label, "SIM_ERROR")
        self.assertEqual(ctx.exception.partial["outcome"]["unevaluable_rules"], ["R1", "R2", "R3", "R4", "R5"])

    def test_bad_acceleration_warning_is_sim_error(self):
        with self.assertRaises(InvalidRun) as ctx:
            self.run_policy(BadQacc())
        self.assertEqual(ctx.exception.label, "SIM_ERROR")
        self.assertIn("mjWARN_BADQACC", ctx.exception.detail)

    def test_success_stops_at_its_physics_step_before_a_later_collision(self):
        with patch("rescuehandsai.pick_runner.PickJudge", SuccessThenCollision):
            record = self.run_policy(HoldHome())
        self.assertTrue(record["success"])
        self.assertEqual((record["physics_steps"], record["control_steps"]), (3, 1))
        self.assertEqual(record["outcome"]["failures"], [])
        self.assertEqual(record["outcome"]["success_physics_step"], 3)

    def test_severe_violation_stops_at_its_physics_step(self):
        with patch("rescuehandsai.pick_runner.PickJudge", SevereAtFour):
            record = self.run_policy(HoldHome())
        self.assertFalse(record["success"])
        self.assertEqual((record["physics_steps"], record["outcome"]["stop"]), (4, "early_stop"))
        self.assertEqual(record["outcome"]["first_failure"], "ARM_ARM_CONTACT")

    def test_physics_version_mismatch_is_a_contract_mismatch(self):
        v1 = MujocoSimulation()
        try:
            with self.assertRaises(InvalidRun) as ctx:
                PickEpisodeRunner(v1, HoldHome(), rules=self.rules)
            self.assertEqual(ctx.exception.label, "CONTRACT_MISMATCH")
        finally:
            v1.close()


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_pick_runner.py" -v`
Expected: `ModuleNotFoundError: No module named 'rescuehandsai.pick_runner'`.

- [ ] **Step 3: Implement**

`src/rescuehandsai/pick_runner.py`:

```python
"""One pick episode: simulation, policy, per-physics-step facts and the judge (spec §5, §6).

Errors are labelled by where they come from: reset, observation and physics are the
simulator (invalid SIM_ERROR); policy calls and actions are the robot (valid failures)."""
from dataclasses import asdict
import time

import mujoco

from .pick_cells import PickTask, cell_params
from .pick_config import derive_steps, load_contacts, load_rules
from .pick_contacts import ContactClassifier
from .pick_facts import FactReader, SimulatorFailure
from .pick_identity import config_hash, scene_hash, settings_hash, settings_record
from .pick_outcome import PickJudge
from .pick_records import InvalidRun
from .runner import clamp_action
from .scene import sample_params


class PickEpisodeRunner:
    def __init__(self, sim, policy, *, rules: dict | None = None, contacts: dict | None = None):
        self.sim, self.policy = sim, policy
        self.rules = rules if rules is not None else load_rules()
        self.contacts = contacts if contacts is not None else load_contacts()
        if sim.physics_version != self.rules["physics_version"]:
            raise InvalidRun("CONTRACT_MISMATCH", f"simulation physics_version {sim.physics_version} "
                                                  f"!= rules physics_version {self.rules['physics_version']}")
        self.steps = derive_steps(self.rules, sim.config["physics_dt"], sim.config["control_dt"])

    def run(self, task: PickTask) -> dict:
        sim, policy, steps = self.sim, self.policy, self.steps
        if task.physics_version != sim.physics_version:
            raise InvalidRun("CONTRACT_MISMATCH", f"task physics_version {task.physics_version} "
                                                  f"!= simulation {sim.physics_version}")
        record = {"seed": task.seed, "cell": task.cell, "utensil": task.utensil, "spare": task.spare,
                  "template": task.template, "instruction": task.instruction,
                  "physics_version": sim.physics_version, "policy": policy.metadata(), "rules": self.rules,
                  "clamped_joint_steps": 0, "timing": {"observe_s": [], "inference_s": []}}
        counters = {"physics": 0, "control": 0}
        judge = None

        def partial():
            out = dict(record, physics_steps=counters["physics"], control_steps=counters["control"])
            if judge is not None and judge.start is not None:
                out["outcome"] = asdict(judge.finish("crash", physics_step=counters["physics"],
                                                     control_step=counters["control"]))
            return out

        def simulator(what, fn):
            try:
                return fn()
            except InvalidRun:
                raise
            except Exception as exc:
                raise InvalidRun("SIM_ERROR", f"{what}: {type(exc).__name__}: {exc}", partial()) from exc

        def reset():
            params = cell_params(sample_params(sim.scene_config, task.seed), task.cell, sim.scene_config)
            sim.reset(task.seed, instruction=task.instruction, params=params)
            return params

        params = simulator("reset", reset)
        record.update(scene=settings_record(params), settings_sha256=settings_hash(params),
                      scene_sha256=scene_hash(params, sim.scene_config, sim.physics_version),
                      config_sha256=config_hash(sim.physics_version, sim.asset_path))
        reader = FactReader(sim, ContactClassifier(sim.model, task.utensil, self.contacts))
        table = sim.scene_config["table"]
        judge = PickJudge(task.utensil, task.spare, self.rules, steps, table["center"], table["half_size"])
        judge.start_from(simulator("read start", lambda: reader.read(0, 0)))
        stop = None
        try:
            policy.reset(sim, task)
        except Exception as exc:
            judge.record_error("POLICY_ERROR", 0, 0, f"reset: {type(exc).__name__}: {exc}")
            stop = "early_stop"
        severe = []

        def on_substep():
            # Stop at the physics step that decides the episode: success or a severe violation.
            counters["physics"] += 1
            severe.extend(judge.update(reader.read(counters["physics"], counters["control"])))
            return judge.succeeded or bool(severe)

        while stop is None and counters["control"] < steps.deadline_control:
            try:
                images = policy.wants_images()
            except Exception as exc:
                judge.record_error("POLICY_ERROR", counters["physics"], counters["control"],
                                   f"wants_images: {type(exc).__name__}: {exc}")
                stop = "early_stop"
                break
            t0 = time.perf_counter()
            obs = simulator("observe", lambda: sim.observe(images=images))
            record["timing"]["observe_s"].append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            try:
                raw = policy.act(obs)
            except Exception as exc:
                judge.record_error("POLICY_ERROR", counters["physics"], counters["control"],
                                   f"act: {type(exc).__name__}: {exc}")
                stop = "early_stop"
                break
            if images:
                record["timing"]["inference_s"].append(time.perf_counter() - t0)
            try:
                action, clamped = clamp_action(raw, sim.previous, sim.limits, sim.config["max_command_delta"])
            except (ValueError, AttributeError, TypeError) as exc:
                judge.record_error("INVALID_ACTION", counters["physics"], counters["control"], str(exc))
                stop = "early_stop"
                break
            record["clamped_joint_steps"] += clamped
            counters["control"] += 1
            try:
                sim.step(action, on_substep=on_substep, stop_on_cross_arm=False)
            except SimulatorFailure as exc:
                raise InvalidRun("SIM_ERROR", str(exc), partial()) from exc
            except mujoco.FatalError as exc:
                raise InvalidRun("SIM_ERROR", f"MuJoCo fatal error: {exc}", partial()) from exc
            except RuntimeError as exc:
                if str(exc).startswith("SIMULATION_ERROR"):
                    raise InvalidRun("SIM_ERROR", str(exc), partial()) from exc
                raise
            if judge.succeeded:
                stop = "success"
            elif severe:
                stop = "early_stop"
        stop = stop or "deadline"
        outcome = judge.finish(stop, physics_step=counters["physics"], control_step=counters["control"])
        record.update(success=outcome.success, outcome=asdict(outcome), control_steps=counters["control"],
                      physics_steps=counters["physics"], other_warnings=dict(reader.other_warnings))
        return record
```

- [ ] **Step 4: Run the tests**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_pick_runner.py" -v`
Expected: 8 tests `ok`.

- [ ] **Step 5: Commit (do not push)**

```powershell
git add src/rescuehandsai/pick_runner.py tests/test_pick_runner.py
git commit -m "Pick plan 1 task 13: pick episode runner with source-based error labels"
```

---

### Task 14: Full verification and the version-1 regression run

**Files:**
- Generated: `results/v1_regression_after_plan1/`

- [ ] **Step 1: Run the whole test suite**

Run: `$env:PYTHONPATH = "src"; .venv-sim/Scripts/python.exe -m unittest discover -s tests`
Expected: `OK`. Record the test count. It is the earlier count plus the new pick tests.

- [ ] **Step 2: Rerun the scripted full-task evaluation with default (version 1) physics**

Run:
```powershell
$env:PYTHONPATH = "src"
.venv-sim/Scripts/python.exe scripts/evaluate.py --policy scripted --seeds 0:10 --supervisor on --name v1_regression_after_plan1
.venv-sim/Scripts/python.exe scripts/check_v1_regression.py results/v1_reference_pre_pick results/v1_regression_after_plan1
```
Expected: the last line is `v1 regression: matched recorded results` with exit code 0. Also run `.venv-sim/Scripts/python.exe -m unittest discover -s tests -p "test_physics_v1_reference.py"`: the world XML and asset/config identities must still match the reference.

If anything differs, **stop**. List the differing episodes and find which change caused them (for example by testing each task's commit). Version 1 must match its recorded results (§2). In the report, say "matched recorded results", not "identical behaviour".

- [ ] **Step 3: Check that nothing was pushed and the tree is clean**

Run: `git status --porcelain -- src scripts training configs tests`
Expected: no output.
Run: `git log --oneline origin/dinner-table..HEAD`
Expected: the Plan 1 commits are listed, which means they exist locally and were not pushed.

- [ ] **Step 4: Commit the regression evidence (do not push)**

```powershell
git add results/v1_regression_after_plan1
git commit -m "Pick plan 1 task 14: v1 regression run matched the recorded pre-pick results"
```

- [ ] **Step 5: Report to the owner**

Report in simple English:
- test count before and after;
- the slip-test numbers printed in Task 3;
- the v1 regression result;
- any finding that needed a stop (start check, slip test, detectability).

State that Plan 2 (force measurements, pick teacher, gate, frozen lists, CLI, timing run) is next and will be written only after this report.
