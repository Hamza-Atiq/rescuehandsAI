"""The exported-SmolVLA adapter's plumbing, with OpenVINO inference replaced by a spy.

These prove the adapter sends the right joints, cameras and instruction, queues and
resets chunks, and refuses bad inputs or changed model files. They say nothing about
model quality or numeric agreement with PyTorch.
"""
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np

from rescuehandsai.contract import build_contract, file_sha256, write_contract
from rescuehandsai.contracts import Observation
from rescuehandsai.policies.smolvla_exported import ExportedSmolVLAPolicy, to_chw_float

NAMES = tuple(f"{arm}/{joint}" for arm in ("left_arm", "right_arm")
              for joint in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"))


class SpyModel:
    backend = "spy"
    chunk = np.tile(np.arange(12, dtype=np.float32), (50, 1))

    def __init__(self, *args, **kwargs):
        self.input_features = [SimpleNamespace(name=f"images.images.camera{i}") for i in (1, 2, 3)]
        self.calls, self.resets = [], 0

    def reset(self):
        self.resets += 1

    def predict_action_chunk(self, inputs):
        self.calls.append(inputs)
        return type(self).chunk


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.dir = Path(self.folder.name)
        for name, body in (("manifest.json", "{}"), ("smolvla.xml", "<xml/>"), ("smolvla.bin", "weights")):
            (self.dir / name).write_text(body)
        write_contract(self.dir, build_contract(
            NAMES, dataset="user/data", control_hz=20.0, source="test",
            files={n: file_sha256(self.dir / n) for n in ("manifest.json", "smolvla.xml", "smolvla.bin")}))
        fake = ModuleType("physicalai.inference.model")
        fake.InferenceModel = SpyModel
        self.modules = patch.dict(sys.modules, {"physicalai.inference.model": fake})
        self.modules.start()
        self.sim = SimpleNamespace(names=NAMES, config={"control_dt": 0.05})
        images = {name: np.full((256, 256, 3), value, np.uint8)
                  for name, value in (("overhead", 0), ("left_wrist", 127), ("right_wrist", 255))}
        self.obs = Observation(0.0, "pass the spoon", dict(zip(NAMES, range(12))), {}, images)

    def tearDown(self):
        self.modules.stop()
        SpyModel.chunk = np.tile(np.arange(12, dtype=np.float32), (50, 1))
        self.folder.cleanup()

    def test_joints_cameras_queue_and_recovery(self):
        policy = ExportedSmolVLAPolicy(self.dir, NAMES, n_action_steps=2)
        policy.reset(self.sim, None)
        action = policy.act(self.obs)
        self.assertEqual(tuple(action.targets), NAMES)
        self.assertEqual(list(action.targets.values()), list(range(12)))
        sent = policy.model.calls[0]
        np.testing.assert_array_equal(sent["state"], np.arange(12)[None])
        self.assertEqual(sent["task"], ["pass the spoon"])
        for i, expected in ((1, 0.0), (2, 127 / 255), (3, 1.0)):  # camera1 overhead, 2 left, 3 right
            self.assertAlmostEqual(float(sent[f"images.images.camera{i}"].mean()), expected, places=6)
        self.assertFalse(policy.wants_images())
        policy.act(self.obs)
        self.assertTrue(policy.wants_images())  # two actions executed, queue empty
        self.assertEqual(len(policy.model.calls), 1)
        policy.after_recovery(None, None, {})
        self.assertEqual(policy.model.resets, 2)

    def test_invalid_horizon_is_refused(self):
        for steps in (0, -5, 2.5, True):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                ExportedSmolVLAPolicy(self.dir, NAMES, n_action_steps=steps)

    def test_empty_chunk_is_a_policy_error(self):
        SpyModel.chunk = np.zeros((0, 12), np.float32)
        policy = ExportedSmolVLAPolicy(self.dir, NAMES)
        policy.reset(self.sim, None)
        with self.assertRaisesRegex(RuntimeError, "POLICY_ERROR"):
            policy.act(self.obs)

    def test_changed_model_file_is_refused(self):
        (self.dir / "smolvla.bin").write_text("different weights")
        with self.assertRaisesRegex(ValueError, "changed"):
            ExportedSmolVLAPolicy(self.dir, NAMES)

    def test_swapped_arms_are_refused(self):
        with self.assertRaisesRegex(ValueError, "joint order"):
            ExportedSmolVLAPolicy(self.dir, NAMES[6:] + NAMES[:6])

    def test_missing_images_or_float_images_are_refused(self):
        policy = ExportedSmolVLAPolicy(self.dir, NAMES)
        policy.reset(self.sim, None)
        with self.assertRaisesRegex(RuntimeError, "needs camera images"):
            policy.act(Observation(0.0, "x", self.obs.positions, {}, {}))
        with self.assertRaises(ValueError):
            to_chw_float(np.zeros((256, 256, 3), np.float32))
        with self.assertRaises(ValueError):
            ExportedSmolVLAPolicy._resolve_image_keys({"wrong_camera"})


if __name__ == "__main__":
    unittest.main()
