"""The judge reports the hold numbers it compares against thresholds (Plan 2 Task 3)."""
import unittest

import numpy as np

from rescuehandsai.pick_config import StepCounts, load_rules
from rescuehandsai.pick_contacts import JAW_UTENSIL, ContactVerdict
from rescuehandsai.pick_outcome import PickJudge, SubstepFacts

FIXED = ContactVerdict(JAW_UTENSIL, (), ("right_arm/fixed_jaw_box5", "fork_handle"), 1.0, "fixed")
MOVING = ContactVerdict(JAW_UTENSIL, (), ("right_arm/moving_jaw_box2", "fork_handle"), 1.0, "moving")
START = {"fork": np.array([0.25, 0.24, 0.006]), "spoon": np.array([0.33, 0.24, 0.006]),
         "cup": np.array([0.40, 0.12, 0.03])}


def steps() -> StepCounts:
    # Short windows keep the synthetic sequences readable.
    return StepCounts(substeps=1, hold=4, max_gap=1, final_speed=2, deadline_control=50)


class WindowMeasurementTests(unittest.TestCase):
    """The judge must report the numbers it already compares against thresholds."""

    def setUp(self):
        self.rules = load_rules()
        self.judge = PickJudge("fork", "spoon", self.rules, steps(), (0.0, 0.15), (0.6, 0.45), keep_trace=True)
        self.step, self.offset = 0, np.array([0.0, 0.0, 0.01])
        self.judge.start_from(self._facts(lift=0.0, verdicts=(), speed=0.0))

    def test_an_empty_window_reports_nothing_rather_than_zeros(self):
        # Zeros would look like a perfectly steady hold that never happened.
        self.assertIsNone(self.judge.window_measurements())

    def test_a_full_steady_window_reports_small_movement_and_full_jaw_coverage(self):
        self._feed(count=4, drift=0.0, both=True, speed=0.0)
        m = self.judge.window_measurements()
        self.assertEqual(m["both_jaw_fraction"], 1.0)
        self.assertEqual(m["longest_single_jaw_gap_steps"], 0)
        self.assertAlmostEqual(m["max_shift_m"], 0.0, places=9)
        self.assertAlmostEqual(m["max_turn_deg"], 0.0, places=6)
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
        self.assertEqual(len(self.judge.trace[0]["rel_rot"]), 9)

    def test_ineligible_steps_still_record_pose_and_actual_jaw_contacts(self):
        # A failed hold must stay explainable: pose and jaw contacts are kept on every step.
        self._feed(count=1, drift=0.0, both=True, speed=0.0, lifted=False)
        self._feed(count=1, drift=0.0, both=False, speed=0.0, lifted=False)
        first, second = self.judge.trace
        self.assertFalse(first["eligible"])
        self.assertEqual(len(first["rel_pos"]), 3)
        self.assertEqual(len(first["rel_rot"]), 9)
        self.assertTrue(first["both_jaws"])
        self.assertEqual([c["jaw"] for c in first["jaw_contacts"]], ["fixed", "moving"])
        self.assertTrue(second["fixed_jaw"])
        self.assertFalse(second["moving_jaw"])
        self.assertFalse(second["both_jaws"])
        self.assertEqual(second["jaw_contacts"][0]["geoms"], ["right_arm/fixed_jaw_box5", "fork_handle"])

    def test_the_trace_is_off_by_default(self):
        quiet = PickJudge("fork", "spoon", self.rules, steps(), (0.0, 0.15), (0.6, 0.45))
        self.assertEqual(quiet.trace, [])

    def test_a_failed_hold_still_reports_its_best_window(self):
        self._feed(count=4, drift=0.05, both=True, speed=0.5)  # far too much movement
        outcome = self.judge.finish("deadline", physics_step=4, control_step=4)
        self.assertFalse(outcome.success)
        self.assertIsNotNone(outcome.best_window_measurements)
        self.assertGreater(outcome.best_window_measurements["max_shift_m"], 0.0)
        self.assertEqual(outcome.hold_measurements, outcome.best_window_measurements)
        self.assertEqual(len(outcome.hold_trace), 4)

    def test_a_completed_hold_reports_the_window_that_completed_it(self):
        self._feed(count=4, drift=0.0, both=True, speed=0.0)
        outcome = self.judge.finish("success", physics_step=4, control_step=4)
        self.assertTrue(outcome.success)
        self.assertEqual(outcome.hold_measurements["both_jaw_fraction"], 1.0)

    def test_rule_decisions_are_unchanged_by_the_trace(self):
        quiet = PickJudge("fork", "spoon", self.rules, steps(), (0.0, 0.15), (0.6, 0.45))
        quiet.start_from(self._facts(lift=0.0, verdicts=(), speed=0.0))
        self.step = 0
        for _ in range(4):
            self.step += 1
            f = self._facts(lift=0.06, verdicts=(FIXED, MOVING), speed=0.0)
            quiet.update(f)
            self.judge.update(f)
        self.assertEqual(quiet.succeeded, self.judge.succeeded)

    def _facts(self, *, lift, verdicts, speed):
        fork = START["fork"] + np.array([0.0, 0.0, lift])
        positions = {"fork": fork, "spoon": START["spoon"], "cup": START["cup"]}
        rotations = {k: np.eye(3) for k in positions}
        return SubstepFacts(self.step, self.step, positions, rotations,
                            {"fork": speed, "spoon": 0.0, "cup": 0.0}, {"fork": 0.0, "spoon": 0.0, "cup": 0.0},
                            fork - self.offset, np.eye(3), tuple(verdicts))

    def _feed(self, *, count, drift, both, speed, lifted=True):
        """Append `count` physics steps; the utensil drifts `drift` m per step in the gripper frame."""
        for _ in range(count):
            self.step += 1
            if self.judge._window:
                self.offset = self.offset + np.array([drift, 0.0, 0.0])
            self.judge.update(self._facts(lift=0.06 if lifted else 0.0,
                                          verdicts=(FIXED, MOVING) if both else (FIXED,), speed=speed))


if __name__ == "__main__":
    unittest.main()
