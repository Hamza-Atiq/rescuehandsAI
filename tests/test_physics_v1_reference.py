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
