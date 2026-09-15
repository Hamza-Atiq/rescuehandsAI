"""Dinner scene: names exist, randomization is seeded and stays in range."""
import json
import unittest

import mujoco

from rescuehandsai.scene import ROOT, SCENE_ITEMS, build_model, sample_params


class SceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "configs/scene.json").read_text())

    def test_same_seed_same_params_and_different_seed_differs(self):
        self.assertEqual(sample_params(self.config, 3), sample_params(self.config, 3))
        self.assertNotEqual(sample_params(self.config, 3), sample_params(self.config, 4))

    def test_sampled_values_stay_in_configured_ranges(self):
        cup, ut = self.config["cup"], self.config["utensil"]
        for seed in range(30):
            p = sample_params(self.config, seed)
            self.assertTrue(cup["radius"][0] <= p.cup_radius <= cup["radius"][1])
            self.assertTrue(cup["mass"][0] <= p.masses["cup"] <= cup["mass"][1])
            for item in ("fork", "spoon"):
                self.assertTrue(ut["mass"][0] <= p.masses[item] <= ut["mass"][1])
                self.assertTrue(ut["friction"][0] <= p.frictions[item] <= ut["friction"][1])
            self.assertEqual({p.slots["fork"], p.slots["spoon"]}, {0, 1})
            self.assertLessEqual(abs(p.poses["cup"][0] - cup["pos"][0]), cup["pos_noise"])

    def test_both_slot_orders_occur(self):
        orders = {sample_params(self.config, s).slots["fork"] for s in range(20)}
        self.assertEqual(orders, {0, 1})

    def test_model_contains_named_parts(self):
        model = build_model(sample_params(self.config, 0))
        for item in SCENE_ITEMS:
            model.body(item)
            model.joint(item + "_free")
        for name in ("plate", "cup_zone", "utensil_zone", "table"):
            model.geom(name)
        for name in ("overhead", "front", "left_arm/wrist_cam", "right_arm/wrist_cam"):
            model.camera(name)
        for arm in ("left_arm", "right_arm"):
            model.site(arm + "/gripperframe")
        self.assertEqual(model.nu, 12)

    def test_randomized_mass_reaches_compiled_model(self):
        params = sample_params(self.config, 5)
        model = build_model(params)
        self.assertAlmostEqual(float(model.body("cup").mass[0]), params.masses["cup"], places=6)


if __name__ == "__main__":
    unittest.main()
