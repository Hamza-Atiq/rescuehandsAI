"""Task success must be physical: supported, upright, spare utensil still in place."""
import unittest

from rescuehandsai.auditor import AuditFacts
from rescuehandsai.evaluation import MAX_UNHELD_STEPS, HandoffTracker, task_outcome
from rescuehandsai.scene import load_config, sample_params
from rescuehandsai.task import make_task

ITEMS = ("cup", "fork", "spoon")


def good_facts(params, task, **changes):
    other = "spoon" if task.utensil == "fork" else "fork"
    sx, sy, _ = params.poses[other]
    positions = {"cup": (0.12, 0.26, params.cup_half_height), task.utensil: (-0.12, 0.26, 0.006),
                 other: (sx, sy, 0.006)}
    base = dict(time=10.0, held_by={i: set() for i in ITEMS}, touching={i: set() for i in ITEMS},
                supported={i: True for i in ITEMS},
                in_zone={"cup": "cup_zone", task.utensil: "utensil_zone", other: None},
                height={i: positions[i][2] for i in ITEMS}, speed={i: 0.0 for i in ITEMS},
                out_of_bounds=set(), cross_arm_contact=False, positions=positions,
                up_z={i: 1.0 for i in ITEMS}, angular_speed={i: 0.0 for i in ITEMS})
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = base[key] | value
        else:
            base[key] = value
    return AuditFacts(**base)


class OutcomeTests(unittest.TestCase):
    def setUp(self):
        self.task = make_task(1)
        self.params = sample_params(load_config(), 1)
        self.other = "spoon" if self.task.utensil == "fork" else "fork"
        self.handoff = True

    def outcome(self, **changes):
        return task_outcome(good_facts(self.params, self.task, **changes), self.task, self.handoff, self.params)

    def test_good_final_state_succeeds(self):
        self.assertTrue(self.outcome()["success"])

    def test_unsupported_items_fail(self):
        for item in ("cup", self.task.utensil):
            with self.subTest(item=item):
                self.assertFalse(self.outcome(supported={item: False})["success"])

    def test_spare_utensil_fallen_off_or_moved_fails(self):
        sx, sy, _ = self.params.poses[self.other]
        self.assertFalse(self.outcome(out_of_bounds={self.other})["success"])
        self.assertFalse(self.outcome(supported={self.other: False})["success"])
        self.assertFalse(self.outcome(positions={self.other: (sx + 0.08, sy, 0.006)})["success"])

    def test_spare_utensil_held_or_moving_at_the_end_fails(self):
        """In place is not untouched: a hand on it, or it still sliding, is not a finished table."""
        self.assertFalse(self.outcome(touching={self.other: {"right_arm"}})["spare_utensil_in_place"])
        self.assertFalse(self.outcome(held_by={self.other: {"right_arm"}},
                                      touching={self.other: {"right_arm"}})["success"])
        self.assertFalse(self.outcome(speed={self.other: 3.0})["spare_utensil_in_place"])
        self.assertFalse(self.outcome(angular_speed={self.other: 2.0})["spare_utensil_in_place"])

    def test_tipped_cup_fails(self):
        self.assertFalse(self.outcome(height={"cup": self.params.cup_radius})["success"])

    def test_upside_down_cup_at_the_right_height_fails(self):
        result = self.outcome(up_z={"cup": -1.0})
        self.assertFalse(result["cup_upright"])
        self.assertFalse(result["success"])

    def test_spinning_item_is_not_settled(self):
        for item in ("cup", self.task.utensil):
            with self.subTest(item=item):
                self.assertFalse(self.outcome(angular_speed={item: 2.0})["settled"])

    def test_no_handoff_fails(self):
        result = task_outcome(good_facts(self.params, self.task), self.task, False, self.params)
        self.assertFalse(result["success"])
        with self.assertRaises(TypeError):  # the old "set of holders" call must not pass silently
            task_outcome(good_facts(self.params, self.task), self.task, {"left_arm", "right_arm"}, self.params)


class HandoffTrackerTests(unittest.TestCase):
    def step(self, tracker, held, supported):
        facts = good_facts(sample_params(load_config(), 1), make_task(1, utensil="fork"),
                           held_by={"fork": set(held)}, supported={"fork": supported})
        return tracker.update(facts)

    def test_in_air_transfer_counts(self):
        t = HandoffTracker("fork")
        self.step(t, {"right_arm"}, True)       # picked from the tray
        self.step(t, {"right_arm"}, False)      # lifted
        self.step(t, {"left_arm", "right_arm"}, False)
        self.step(t, set(), False)              # contact flicker in the air
        self.assertTrue(self.step(t, {"left_arm"}, False))
        self.assertTrue(self.step(t, set(), True))  # placed afterwards: stays done

    def test_drop_then_other_hand_pickup_is_not_a_handoff(self):
        t = HandoffTracker("fork")
        self.step(t, {"right_arm"}, False)
        self.step(t, set(), True)               # dropped on the table
        self.step(t, {"left_arm"}, True)
        self.assertFalse(self.step(t, {"left_arm"}, False))

    def test_table_assisted_transfer_is_not_a_handoff(self):
        """Shared in the air, then it rests on the table while the left hand holds it."""
        t = HandoffTracker("fork")
        self.step(t, {"right_arm"}, False)
        self.step(t, {"left_arm", "right_arm"}, False)
        self.step(t, {"left_arm"}, True)        # supported again: transfer not completed
        self.assertFalse(self.step(t, {"left_arm"}, False))

    def test_long_unheld_gap_in_the_air_is_not_a_handoff(self):
        """Brief flicker is fine (the teacher's longest is 1 step); a long gap is not a transfer."""
        t = HandoffTracker("fork")
        self.step(t, {"right_arm"}, False)
        self.step(t, {"left_arm", "right_arm"}, False)
        for _ in range(100):
            self.step(t, set(), False)
        self.assertFalse(self.step(t, {"left_arm"}, False))

    def test_gap_up_to_the_bound_still_counts(self):
        t = HandoffTracker("fork")
        self.step(t, {"right_arm"}, False)
        self.step(t, {"left_arm", "right_arm"}, False)
        for _ in range(MAX_UNHELD_STEPS):
            self.step(t, set(), False)
        self.assertTrue(self.step(t, {"left_arm"}, False))

    def test_both_touching_on_the_table_is_not_shared_in_air(self):
        t = HandoffTracker("fork")
        self.step(t, {"right_arm"}, True)
        self.step(t, {"left_arm", "right_arm"}, True)
        self.assertFalse(self.step(t, {"left_arm"}, False))


if __name__ == "__main__":
    unittest.main()
