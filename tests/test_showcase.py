"""Presentation renders must never change what the policy sees or how physics runs."""
import unittest

import numpy as np

from rescuehandsai.scene import load_config, sample_params, world_xml
from rescuehandsai.showcase import ShowcaseRenderer
from rescuehandsai.sim import POLICY_CAMERAS, MujocoSimulation


class ShowcaseTests(unittest.TestCase):
    def test_default_scene_has_no_showcase_extras(self):
        config = load_config()
        params = sample_params(config, 3)
        plain, fancy = world_xml(params, config), world_xml(params, config, showcase=True)
        self.assertNotIn("showcase", plain)
        self.assertIn('camera name="showcase_hero"', fancy)
        self.assertIn('contype="0" conaffinity="0"', fancy[fancy.index("showcase_floor\" type"):])

    def test_twin_draws_the_live_state_without_touching_it(self):
        sim = MujocoSimulation()
        try:
            sim.reset(2)
            policy_view = sim.render(POLICY_CAMERAS)
            qpos, time = sim.data.qpos.copy(), float(sim.data.time)
            renderer = ShowcaseRenderer(320, 180)
            frame = renderer.draw(sim, sim.data.qpos, "hero")
            renderer.close()
            self.assertEqual(frame.shape, (180, 320, 3))
            self.assertGreater(frame.std(), 5)  # a real picture, not a blank buffer
            np.testing.assert_array_equal(sim.data.qpos, qpos)
            self.assertEqual(float(sim.data.time), time)
            for name, image in sim.render(POLICY_CAMERAS).items():
                np.testing.assert_array_equal(image, policy_view[name])
            with self.assertRaises(ValueError):
                ShowcaseRenderer(64, 64).draw(sim, qpos, "not_a_camera")
        finally:
            sim.close()


if __name__ == "__main__":
    unittest.main()
