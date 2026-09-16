import unittest

import numpy as np

from rescuehandsai.contracts import BimanualAction
from rescuehandsai.sim import MujocoSimulation, POLICY_CAMERAS, VIDEO_CAMERAS, is_software_renderer


class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = MujocoSimulation()

    @classmethod
    def tearDownClass(cls):
        cls.sim.close()

    def setUp(self):
        self.sim.reset(seed=42)

    def hold(self, steps):
        for _ in range(steps):
            self.sim.step(BimanualAction(self.sim.observe().timestamp, self.sim.home_targets))

    def test_software_renderers_are_recognised(self):
        self.assertTrue(is_software_renderer("llvmpipe (LLVM 15.0.7, 256 bits)"))
        self.assertTrue(is_software_renderer("softpipe"))
        self.assertFalse(is_software_renderer("Tesla T4/PCIe/SSE2"))
        self.assertFalse(is_software_renderer("Intel(R) HD Graphics 520"))

    def test_reset_repeats_scene_and_robot_state(self):
        before = self.sim.observe()
        objects = self.sim.privileged().objects
        self.hold(2)
        self.sim.reset(seed=42)
        self.assertEqual(self.sim.observe().positions, before.positions)
        self.assertEqual(self.sim.privileged().objects, objects)
        self.sim.reset(seed=43)
        self.assertNotEqual(self.sim.privileged().objects["cup"].position, objects["cup"].position)

    def test_both_arms_and_grippers_respond_by_name(self):
        sim = self.sim
        start = sim.observe()
        targets = dict(start.positions)
        targets['left_arm/shoulder_pan'] += 0.08
        targets['right_arm/shoulder_pan'] -= 0.08
        targets['left_arm/gripper'] += 0.08
        targets['right_arm/gripper'] -= 0.08
        for _ in range(30):
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

    def test_policy_cameras_exclude_video_camera(self):
        frames = self.sim.observe(images=True).images
        self.assertEqual(set(frames), set(POLICY_CAMERAS))
        self.assertNotIn("front", frames)
        for frame in frames.values():
            self.assertEqual(frame.shape, (256, 256, 3))
            self.assertEqual(frame.dtype, np.uint8)
            self.assertGreater(float(frame.std()), 5)
        self.assertFalse(np.array_equal(frames['left_wrist'], frames['overhead']))
        video = self.sim.render(VIDEO_CAMERAS)
        self.assertGreater(float(video["front"].std()), 5)

    def test_items_settle_on_table_through_physics(self):
        self.hold(20)
        state = self.sim.privileged()
        cup = state.objects["cup"]
        self.assertAlmostEqual(cup.position[2], self.sim.scene_params.cup_half_height, delta=0.004)
        for item in ("fork", "spoon"):
            self.assertLess(state.objects[item].position[2], 0.02)
            self.assertLess(max(map(abs, state.objects[item].linear_velocity)), 0.01)
        self.assertTrue(any('table' in pair and 'cup_body' in pair for pair in state.contacts))


if __name__ == '__main__':
    unittest.main()
