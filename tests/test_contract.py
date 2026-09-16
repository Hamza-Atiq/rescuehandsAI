"""The model contract: a shape check cannot catch a swapped arm, so names travel with the model."""
import json
import tempfile
import unittest
from pathlib import Path

from rescuehandsai.contract import (ACTION_UNITS, CAMERA_SLOTS, build_contract, check_contract,
                                    file_sha256, load_contract, verify_files, write_contract)

JOINTS = [f"{arm}/{joint}" for arm in ("left_arm", "right_arm")
          for joint in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")]


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.contract = build_contract(JOINTS, dataset="user/data", dataset_revision="v3.0",
                                       control_hz=20.0, source="checkpoint")

    def test_matching_robot_passes(self):
        check_contract(self.contract, JOINTS, cameras=CAMERA_SLOTS, control_hz=20.0)

    def test_swapped_arms_are_rejected(self):
        swapped = JOINTS[6:] + JOINTS[:6]
        with self.assertRaisesRegex(ValueError, "joint order"):
            check_contract(self.contract, swapped)

    def test_swapped_two_joints_are_rejected(self):
        names = list(JOINTS)
        names[0], names[1] = names[1], names[0]
        with self.assertRaisesRegex(ValueError, "joint order"):
            check_contract(self.contract, names)

    def test_wrong_cameras_rate_or_units_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "camera map"):
            check_contract(self.contract, JOINTS, cameras={"camera1": "left_wrist"})
        with self.assertRaisesRegex(ValueError, "control rate"):
            check_contract(self.contract, JOINTS, control_hz=30.0)
        wrong_units = dict(self.contract, action_units="joint velocities")
        with self.assertRaisesRegex(ValueError, "action units"):
            check_contract(wrong_units, JOINTS)

    def test_duplicate_or_empty_joint_order_is_refused(self):
        for names in ([], JOINTS[:1] * 2):
            with self.subTest(names=names), self.assertRaises(ValueError):
                build_contract(names, dataset="d", control_hz=20.0, source="s")

    def test_round_trip_and_missing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            with self.assertRaisesRegex(FileNotFoundError, "missing"):
                load_contract(directory)
            write_contract(directory, self.contract)
            loaded = load_contract(directory)
            self.assertEqual(loaded["joint_order"], JOINTS)
            self.assertEqual(loaded["action_units"], ACTION_UNITS)
            self.assertEqual(json.loads((directory / "task_contract.json").read_text()), loaded)

    def test_file_hash_changes_with_content(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "weights.bin"
            path.write_bytes(b"one")
            first = file_sha256(path)
            path.write_bytes(b"two")
            self.assertNotEqual(first, file_sha256(path))

    def test_nan_or_nonpositive_control_rate_is_refused(self):
        for rate in (float("nan"), float("inf"), 0.0, -20.0, None):
            with self.subTest(rate=rate):
                with self.assertRaises(ValueError):
                    build_contract(JOINTS, dataset="d", control_hz=rate, source="s")
                with self.assertRaisesRegex(ValueError, "control rate"):
                    check_contract(dict(self.contract, control_hz=rate), JOINTS, control_hz=20.0)

    def test_hashes_are_checked_when_loading_for_deployment(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            weights = directory / "weights.bin"
            weights.write_bytes(b"trained")
            write_contract(directory, dict(self.contract, files={"weights.bin": file_sha256(weights)}))
            load_contract(directory, verify=True)
            weights.write_bytes(b"other weights")
            with self.assertRaisesRegex(ValueError, "changed"):
                load_contract(directory, verify=True)
            weights.unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                load_contract(directory, verify=True)
            with self.assertRaisesRegex(ValueError, "no file hashes"):
                verify_files(directory, self.contract)


if __name__ == "__main__":
    unittest.main()
