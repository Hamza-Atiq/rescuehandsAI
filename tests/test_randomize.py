"""Off-nominal starts must stay physically valid, or they are not demonstrations."""
import unittest

from rescuehandsai.auditor import compute_facts
from rescuehandsai.randomize import perturb_start, start_problems
from rescuehandsai.scene import SCENE_ITEMS
from rescuehandsai.sim import MujocoSimulation


class PerturbTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = MujocoSimulation()

    @classmethod
    def tearDownClass(cls):
        cls.sim.close()

    def test_perturbed_start_is_seeded_and_within_limits(self):
        self.sim.reset(11)
        first = perturb_start(self.sim, 11)
        positions_a = {i: compute_facts(self.sim).positions[i] for i in SCENE_ITEMS}
        for name, (low, high) in self.sim.limits.items():
            self.assertTrue(low <= self.sim.previous[name] <= high, name)
        self.sim.reset(11)
        second = perturb_start(self.sim, 11)
        positions_b = {i: compute_facts(self.sim).positions[i] for i in SCENE_ITEMS}
        self.assertEqual(first["joint_offsets"], second["joint_offsets"])
        for item in SCENE_ITEMS:
            for a, b in zip(positions_a[item], positions_b[item]):
                self.assertAlmostEqual(a, b, places=6)

    def test_items_stay_on_the_table_and_the_state_is_usable(self):
        for seed in (12, 13, 14):
            with self.subTest(seed=seed):
                self.sim.reset(seed)
                perturb_start(self.sim, seed)
                facts = compute_facts(self.sim)
                self.assertEqual(facts.out_of_bounds, set())
                self.assertTrue(all(facts.supported[i] for i in SCENE_ITEMS))

    def test_a_clean_start_has_no_problems_but_a_lifted_cup_does(self):
        self.sim.reset(15)
        self.sim.settle(5)  # contacts appear only after a physics step
        facts = compute_facts(self.sim)
        self.assertIsNone(start_problems(facts, self.sim.scene_params))
        position = list(facts.positions["cup"])
        position[2] += 0.05  # balanced above the table: not a valid start
        self.sim.set_item_pose("cup", position)
        self.assertIsNotNone(start_problems(compute_facts(self.sim), self.sim.scene_params))

    def test_start_pose_must_name_every_joint_and_respect_limits(self):
        self.sim.reset(16)
        with self.assertRaisesRegex(ValueError, "missing joints"):
            self.sim.set_start_pose({"left_arm/gripper": 0.0})
        name = self.sim.names[0]
        bad = dict(self.sim.home_targets, **{name: self.sim.limits[name][1] + 1.0})
        with self.assertRaisesRegex(ValueError, "outside its limits"):
            self.sim.set_start_pose(bad)


if __name__ == "__main__":
    unittest.main()
