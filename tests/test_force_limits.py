"""Force evidence must be usable before any limit is chosen from it (Plan 2 Task 4, spec §5)."""
import json
import unittest
from pathlib import Path

from rescuehandsai.pick_config import load_contacts

REPORT = Path(__file__).resolve().parents[1] / "results" / "measurements" / "forces.json"


class ForceReportTests(unittest.TestCase):
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
    """Skipped until Task 6 freezes the limits after the owner's review."""

    def setUp(self):
        self.contacts = load_contacts()
        if not self.contacts["force_limits_frozen"]:
            self.skipTest("limits are frozen in Task 6, after the owner reviews the evidence")

    def test_limits_are_set(self):
        self.assertIsNotNone(self.contacts["jaw_table_force_limit_n"])
        self.assertIsNotNone(self.contacts["severe_force_limit_n"])

    def test_severe_limit_is_above_the_ordinary_limit(self):
        self.assertGreater(self.contacts["severe_force_limit_n"], self.contacts["jaw_table_force_limit_n"])

    def test_the_limit_sits_between_gentle_contact_and_hard_presses(self):
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        limit = self.contacts["jaw_table_force_limit_n"]
        gentle_max = report["summary"]["gentle"]["max_n"]
        pressed_min = min(v["max_n"] for k, v in report["summary"].items() if k != "gentle")
        self.assertGreater(limit, gentle_max, "the limit would flag a gentle rest as excess force")
        self.assertLess(limit, pressed_min, "the limit would let every deliberate press through")


if __name__ == "__main__":
    unittest.main()
