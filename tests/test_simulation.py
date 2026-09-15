import unittest

import numpy as np

from rescuehandsai.contracts import BimanualAction
from rescuehandsai.sim import MujocoSimulation


class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = MujocoSimulation()

    @classmethod
    def tearDownClass(cls):
        cls.sim.close()

    def setUp(self):
        self.sim.reset(seed=42)

    def test_reset_repeats_object_placement_and_robot_state(self):
        before = self.sim.observe()
        obj = self.sim.privileged().object_position
        self.sim.step(BimanualAction(before.timestamp, dict(before.positions)))
        self.sim.reset(seed=42)
        self.assertEqual(self.sim.observe().positions, before.positions)
        self.assertEqual(self.sim.privileged().object_position, obj)
        self.sim.reset(seed=43)
        self.assertNotEqual(self.sim.privileged().object_position, obj)

    def test_both_arms_and_grippers_respond_by_name(self):
        sim = self.sim
        start = sim.observe()
        targets = dict(start.positions)
        targets['left_arm/shoulder_pan'] += 0.08
        targets['right_arm/shoulder_pan'] -= 0.08
        targets['left_arm/gripper'] += 0.08
        targets['right_arm/gripper'] -= 0.08
        for _ in range(80):
            sim.step(BimanualAction(sim.observe().timestamp, targets))
        after = sim.observe()
        for name, sign in [('left_arm/shoulder_pan', 1), ('right_arm/shoulder_pan', -1),
                           ('left_arm/gripper', 1), ('right_arm/gripper', -1)]:
            self.assertGreater(sign * (after.positions[name] - start.positions[name]), 0.03)
        self.assertEqual(len(after.positions), 12)

    def test_rejected_command_does_not_advance_physics(self):
        before = self.sim.observe()
        bad = dict(before.positions)
        bad['right_arm/wrist_roll'] = float('nan')
        with self.assertRaises(ValueError):
            self.sim.step(BimanualAction(before.timestamp, bad))
        self.assertEqual(before.positions, self.sim.observe().positions)
        self.assertEqual(before.timestamp, self.sim.observe().timestamp)

    def test_cameras_produce_different_nonempty_rgb_images(self):
        frames = self.sim.observe(images=True).images
        self.assertEqual(set(frames), {'front', 'overhead'})
        for frame in frames.values():
            self.assertEqual(frame.shape, (240, 320, 3))
            self.assertEqual(frame.dtype, np.uint8)
            self.assertGreater(float(frame.std()), 5)
        self.assertFalse(np.array_equal(frames['front'], frames['overhead']))

    def test_object_settles_on_table_through_physics(self):
        for _ in range(60):
            obs = self.sim.observe()
            self.sim.step(BimanualAction(obs.timestamp, self.sim.home_targets))
        state = self.sim.privileged()
        self.assertAlmostEqual(state.object_position[2], 0.026, delta=0.003)
        self.assertTrue(any('table' in pair and 'table_item' in pair
                            for pair in state.contacts))


if __name__ == '__main__':
    unittest.main()
