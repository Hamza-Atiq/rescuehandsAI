import json
import tempfile
import unittest
from pathlib import Path

from rescuehandsai.pick_records import (RESUME_KEYS, AlreadyScored, AttemptLedger, EvaluationBlocked, InvalidRun,
                                        ResumeRefused, RetryNeedsDiagnosis, check_run_contract, check_resume,
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
        check_run_contract({"physics_version": 2, "cameras": ["a"]}, {"physics_version": 2, "cameras": ["a"], "x": 1})
        with self.assertRaises(InvalidRun) as ctx:
            check_run_contract({"physics_version": 2, "state_order": ["a", "b"]}, {"physics_version": 1, "state_order": ["b", "a"]})
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

    def test_invalid_label_is_rejected(self):
        ledger = AttemptLedger(self.run_dir)
        with self.assertRaises(ValueError) as ctx:
            ledger.record(self.key, 1, valid=False, label="POLICY_ERROR", filename="x")
        self.assertIn("POLICY_ERROR", str(ctx.exception))

    def test_sim_error_label_is_accepted(self):
        ledger = AttemptLedger(self.run_dir)
        ledger.record(self.key, 1, valid=False, label="SIM_ERROR", filename="x")
        self.assertEqual(ledger.attempts(self.key)[0]["label"], "SIM_ERROR")

    def test_reload_preserves_retry_needs_diagnosis(self):
        ledger = AttemptLedger(self.run_dir)
        ledger.record(self.key, 1, valid=False, label="SIM_ERROR", filename="x")
        again = AttemptLedger(self.run_dir)
        with self.assertRaises(RetryNeedsDiagnosis):
            again.next_attempt(self.key)

    def test_add_diagnosis_validation(self):
        ledger = AttemptLedger(self.run_dir)
        ledger.record(self.key, 1, valid=False, label="SIM_ERROR", filename="x")
        ledger.add_diagnosis(self.key, 1, "test diagnosis")
        with self.assertRaises(ValueError):
            ledger.add_diagnosis(self.key, 2, "attempt 2 does not exist")
        ledger.record(self.key, 2, valid=True, label="NO_LIFT", filename="x")
        with self.assertRaises(ValueError):
            ledger.add_diagnosis(self.key, 2, "attempt 2 is valid, not invalid")


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
