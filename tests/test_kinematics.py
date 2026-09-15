"""Teacher IK: reach every task spot with a downward hand; never touch live state."""
import unittest

import mujoco
import numpy as np

from rescuehandsai.kinematics import IKSolver, hand_axes
from rescuehandsai.sim import MujocoSimulation


class KinematicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = MujocoSimulation(seed=0)
        cls.ik = IKSolver(cls.sim.model)
        cfg = cls.sim.scene_config
        z = cfg["zones"]
        cls.spots = {
            "right_arm": [(*cfg["cup"]["pos"], 0.04, (1, 0)), (*z["cup_zone"]["pos"], 0.06, (1, 0)),
                          (*cfg["utensil"]["slots"][0], 0.012, (1, 0)),
                          (*cfg["utensil"]["slots"][1], 0.012, (1, 0)),
                          (0.04, 0.17, 0.08, (0, 1))],
            "left_arm": [(-0.02, 0.17, 0.08, (0, 1)), (*z["utensil_zone"]["pos"], 0.03, (1, 0))],
        }

    @classmethod
    def tearDownClass(cls):
        cls.sim.close()

    def test_reaches_task_spots_pointing_down(self):
        for arm, spots in self.spots.items():
            for x, y, zpos, closing in spots:
                with self.subTest(arm=arm, spot=(x, y, zpos)):
                    q = self.ik.solve(arm, (x, y, zpos), closing_xy=closing)
                    self.assertIsNotNone(q)
                    pos, finger, close_axis = self.ik.forward(arm, q)
                    self.assertLess(np.linalg.norm(pos - (x, y, zpos)), 0.005)
                    self.assertGreater(-finger[2], 0.97)
                    self.assertGreater(abs(np.dot(close_axis[:2], closing)), 0.95)
                    for name, value in q.items():
                        low, high = self.sim.limits[name]
                        self.assertTrue(low <= value <= high, name)

    def test_unreachable_point_returns_none(self):
        self.assertIsNone(self.ik.solve("right_arm", (0.9, 0.9, 0.05), closing_xy=(1, 0)))

    def test_does_not_change_live_simulation(self):
        before = self.sim.data.qpos.copy()
        self.ik.solve("left_arm", (-0.1, 0.2, 0.05), closing_xy=(1, 0))
        np.testing.assert_array_equal(before, self.sim.data.qpos)

    def test_hand_axes_match_measured_zero_pose(self):
        model = mujoco.MjModel.from_xml_path(str(self.sim.asset_path))
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        finger, closing = hand_axes(data.site_xmat[model.site("gripperframe").id].reshape(3, 3))
        np.testing.assert_allclose(finger, (1, 0, 0), atol=0.01)
        self.assertGreater(closing[2], 0.99)


if __name__ == "__main__":
    unittest.main()
