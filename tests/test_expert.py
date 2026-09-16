"""The scripted teacher really completes the task in physics (no welding, no teleport)."""
import unittest

from rescuehandsai.auditor import compute_facts
from rescuehandsai.evaluation import PlacementTracker
from rescuehandsai.expert import ScriptedExpert
from rescuehandsai.sim import MujocoSimulation
from rescuehandsai.task import make_task


def run(sim, expert, max_steps=900):
    steps = 0
    while not expert.done and steps < max_steps:
        sim.step(expert.act(sim.observe()))
        steps += 1
    return steps


class ExpertCupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = MujocoSimulation()

    @classmethod
    def tearDownClass(cls):
        cls.sim.close()

    def test_right_arm_places_cup_stably_in_zone(self):
        for seed in range(5):
            with self.subTest(seed=seed):
                self.sim.reset(seed)
                expert = ScriptedExpert(self.sim, make_task(seed), subtasks=("place_cup",))
                touched = False
                tracker = PlacementTracker(required_seconds=0.5, max_speed=0.02)
                steps = 0
                while not expert.done and steps < 400:
                    self.sim.step(expert.act(self.sim.observe()))
                    touched |= "right_arm" in compute_facts(self.sim).held_by["cup"]
                    steps += 1
                self.assertTrue(expert.done)
                self.assertTrue(touched, "cup was never held by both jaws")
                stable = False
                for _ in range(15):
                    self.sim.step(expert.act(self.sim.observe()))
                    f = compute_facts(self.sim)
                    stable = tracker.update(time=f.time, inside=f.in_zone["cup"] == "cup_zone",
                                            supported=f.supported["cup"], released=not f.touching["cup"],
                                            both_arms_used=True, speed=f.speed["cup"])
                self.assertTrue(stable)

    def test_unknown_subtask_rejected(self):
        with self.assertRaises(ValueError):
            ScriptedExpert(self.sim, make_task(0), subtasks=("juggle",))


class ExpertFullTaskTests(unittest.TestCase):
    """Hand-off task end to end. Seeds 0-2 cover both fork and spoon."""

    @classmethod
    def setUpClass(cls):
        cls.sim = MujocoSimulation()

    @classmethod
    def tearDownClass(cls):
        cls.sim.close()

    def test_handoff_and_place_both_items(self):
        for seed in (0, 1, 2):
            with self.subTest(seed=seed):
                self.sim.reset(seed)
                task = make_task(seed)
                expert = ScriptedExpert(self.sim, task)
                holders = set()
                steps = 0
                while not expert.done and steps < 900:
                    self.sim.step(expert.act(self.sim.observe()))
                    holders |= compute_facts(self.sim).held_by[task.utensil]
                    steps += 1
                for _ in range(10):
                    self.sim.step(expert.act(self.sim.observe()))
                f = compute_facts(self.sim)
                self.assertTrue(expert.done)
                self.assertEqual(holders, {"left_arm", "right_arm"}, "both arms must hold the utensil")
                self.assertEqual(f.in_zone[task.utensil], "utensil_zone")
                self.assertEqual(f.in_zone["cup"], "cup_zone")
                other = "spoon" if task.utensil == "fork" else "fork"
                self.assertIsNone(f.in_zone[other])
                for item in ("cup", task.utensil):
                    self.assertEqual(f.touching[item], set())
                    self.assertLess(f.speed[item], 0.02)


if __name__ == "__main__":
    unittest.main()
