"""Force evidence must be usable before any limit is chosen from it (Plan 2 Task 4, spec §5).

Every robot-table contact is forbidden (owner decision 23 Sep 2026), so the only force limit
left is the severe-force early stop. It compares single-contact normal forces, and so does the
report. The report is a local measurement file: these checks skip on a fresh clone.
"""
import json
import unittest
from pathlib import Path

from rescuehandsai.pick_config import load_contacts

REPORT = Path(__file__).resolve().parents[1] / "results" / "measurements" / "forces_single_contact.json"


class ForceReportTests(unittest.TestCase):
    def setUp(self):
        if not REPORT.is_file():
            self.skipTest("local evidence file absent: run scripts/measure_forces.py to create it")
        self.report = json.loads(REPORT.read_text(encoding="utf-8"))

    def test_every_condition_actually_touched_the_table(self):
        for name, s in self.report["summary"].items():
            with self.subTest(condition=name):
                self.assertGreater(s["samples"], 0, "no contact was ever recorded")

    def test_every_condition_reports_the_speed_it_really_achieved(self):
        for name, s in self.report["summary"].items():
            with self.subTest(condition=name):
                self.assertIsNotNone(s["measured_speed_m_per_s"])

    def test_the_headline_is_a_single_contact_never_a_sum(self):
        for name, s in self.report["summary"].items():
            with self.subTest(condition=name):
                self.assertLessEqual(s["max_single_n"], s["max_step_sum_n"] + 1e-9)
                self.assertAlmostEqual(s["max_single_n"], max(s["max_single_n_by_shape"].values()), places=9)


class ForceLimitTests(unittest.TestCase):
    """Skipped until the severe limit is frozen from grasp and controlled bad-contact evidence."""

    def setUp(self):
        self.contacts = load_contacts()
        if not self.contacts["force_limits_frozen"]:
            self.skipTest("the severe limit stays unset until grasp and bad-contact evidence is reviewed")

    def test_the_severe_limit_is_set(self):
        self.assertIsNotNone(self.contacts["severe_force_limit_n"])

    def test_the_removed_jaw_table_limit_stays_removed(self):
        self.assertNotIn("jaw_table_force_limit_n", self.contacts)


if __name__ == "__main__":
    unittest.main()
