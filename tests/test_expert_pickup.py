"""Pickup retry bookkeeping: staging the utensil must never end the pickup without a grasp."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from rescuehandsai.expert import LostItemError, PlanningError, ScriptedExpert
from rescuehandsai.motion import Move
from rescuehandsai.sim import MujocoSimulation
from rescuehandsai.task import make_task


class PickupRetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = MujocoSimulation()

    @classmethod
    def tearDownClass(cls):
        cls.sim.close()

    def drive(self, plans, heights, attempts=3):
        """Run _pick_utensil with scripted planning results and lift heights."""
        self.sim.reset(0)
        expert = ScriptedExpert(self.sim, make_task(0))
        item = expert.task.utensil
        plans, heights = list(plans), list(heights)
        labels = []

        def plan(*args, **kwargs):
            result = plans.pop(0)
            if result == "fail":
                raise PlanningError("out of reach")
            return {"q": 1}, {"lift": result}, np.zeros(3), np.array([1.0, 0, 0]), 1.3

        def staging():
            yield Move({}, 1, "stage_utensil")

        facts = lambda sim: SimpleNamespace(height={item: heights.pop(0)})
        with patch.object(expert, "_grasp_with_clearance", side_effect=plan), \
                patch.object(expert, "_approach_pose", return_value={"pre": 1}), \
                patch.object(expert, "_utensil_frame", return_value=(np.zeros(3), np.array([1.0, 0, 0]))), \
                patch.object(expert, "_handle_geometry", return_value={"end": 0.0, "half_width": 0.005}), \
                patch.object(expert, "_stage_utensil", side_effect=staging), \
                patch("rescuehandsai.expert.compute_facts", side_effect=facts):
            for move in expert._pick_utensil(attempts=attempts):
                labels.append(move.label)
        return expert, labels

    def test_staging_on_the_last_attempt_still_ends_with_a_real_grasp(self):
        # two missed lifts, then the utensil is out of reach, staged, and grasped
        expert, labels = self.drive(plans=[1, 2, "fail", 3], heights=[0.0, 0.0, 0.05])
        self.assertIn("stage_utensil", labels)
        self.assertEqual(labels[-1], "utensil_lift")
        self.assertEqual(expert._q_right, {"lift": 3})  # the lift that worked, not a stale one

    def test_staging_on_the_last_attempt_then_a_miss_raises(self):
        with self.assertRaises(LostItemError):
            self.drive(plans=[1, 2, "fail", 3], heights=[0.0, 0.0, 0.0])

    def test_three_missed_lifts_without_staging_raise_as_before(self):
        with self.assertRaises(LostItemError):
            self.drive(plans=[1, 2, 3], heights=[0.0, 0.0, 0.0])

    def test_second_planning_failure_after_staging_raises(self):
        with self.assertRaises(PlanningError):
            self.drive(plans=["fail", "fail"], heights=[])


if __name__ == "__main__":
    unittest.main()
