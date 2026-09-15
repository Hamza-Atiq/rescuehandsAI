"""Task success must be physical: supported, upright, spare utensil still in place."""
import unittest

from rescuehandsai.auditor import AuditFacts
from rescuehandsai.evaluation import task_outcome
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
                out_of_bounds=set(), cross_arm_contact=False, positions=positions)
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
        self.holders = {"left_arm", "right_arm"}

    def outcome(self, **changes):
        return task_outcome(good_facts(self.params, self.task, **changes), self.task, self.holders, self.params)

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

    def test_tipped_cup_fails(self):
        self.assertFalse(self.outcome(height={"cup": self.params.cup_radius})["success"])

    def test_no_handoff_fails(self):
        result = task_outcome(good_facts(self.params, self.task), self.task, {"right_arm"}, self.params)
        self.assertFalse(result["success"])


if __name__ == "__main__":
    unittest.main()
