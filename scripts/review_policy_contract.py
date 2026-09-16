"""Independent learned-adapter contract tests, with inference replaced by a spy.

No claim about model quality or exported numeric agreement is made here.
"""
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rescuehandsai.contract import build_contract, file_sha256, write_contract
from rescuehandsai.contracts import Observation
from rescuehandsai.policies.smolvla_exported import ExportedSmolVLAPolicy, to_chw_float


class SpyModel:
    backend = "spy"
    def __init__(self, *args, **kwargs):
        self.input_features = [SimpleNamespace(name="images.images.camera"+str(i)) for i in (1, 2, 3)]
        self.calls, self.resets = [], 0
    def reset(self): self.resets += 1
    def predict_action_chunk(self, inputs):
        self.calls.append(inputs)
        return np.tile(np.arange(12, dtype=np.float32), (50, 1))


class ContractTests(unittest.TestCase):
    def test_all_joints_cameras_queue_and_recovery(self):
        fake = ModuleType("physicalai.inference.model")
        fake.InferenceModel = SpyModel
        names = tuple(f"joint_{i}" for i in range(12))
        # the maintained version of this check is tests/test_smolvla_adapter.py (normal test discovery)
        with tempfile.TemporaryDirectory() as td, patch.dict(sys.modules, {"physicalai.inference.model": fake}):
            Path(td, "manifest.json").write_text("{}")
            write_contract(Path(td), build_contract(names, dataset="spy", control_hz=20.0, source="spy",
                                                    files={"manifest.json": file_sha256(Path(td, "manifest.json"))}))
            policy = ExportedSmolVLAPolicy(Path(td), names, n_action_steps=2)
            policy.reset(SimpleNamespace(names=names, config={"control_dt": 0.05}), None)
            images = {name: np.full((256, 256, 3), value, np.uint8)
                      for name, value in (("overhead", 0), ("left_wrist", 127), ("right_wrist", 255))}
            obs = Observation(0, "pass the spoon", dict(zip(names, range(12))), {}, images)
            action = policy.act(obs)
            self.assertEqual(tuple(action.targets), names)
            self.assertEqual(list(action.targets.values()), list(range(12)))
            sent = policy.model.calls[0]
            np.testing.assert_array_equal(sent["state"], np.arange(12)[None])
            self.assertEqual(sent["task"], [obs.instruction])
            for i, expected in ((1, 0), (2, 127/255), (3, 1)):
                self.assertAlmostEqual(float(sent[f"images.images.camera{i}"].mean()), expected, places=6)
            self.assertFalse(policy.wants_images())
            policy.act(obs)
            self.assertTrue(policy.wants_images())
            self.assertEqual(len(policy.model.calls), 1)
            policy.after_recovery(None, None, {})
            self.assertTrue(policy.wants_images())
            self.assertEqual(policy.model.resets, 2)

    def test_float_images_rejected(self):
        with self.assertRaises(ValueError):
            to_chw_float(np.zeros((256, 256, 3), np.float32))

    def test_unknown_camera_schema_rejected(self):
        with self.assertRaises(ValueError):
            ExportedSmolVLAPolicy._resolve_image_keys({"wrong_camera"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
