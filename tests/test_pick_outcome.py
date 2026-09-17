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
