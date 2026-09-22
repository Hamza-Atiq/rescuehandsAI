"""Physics version 3 = version 2 + MuJoCo NoSlip (experimental, grip-probe evidence 22 Sep)."""
import unittest

from rescuehandsai.pick_identity import config_hash, scene_hash
from rescuehandsai.scene import NOSLIP_ITERATIONS_V3, load_config, sample_params, world_xml
from rescuehandsai.sim import MujocoSimulation


class PhysicsV3Tests(unittest.TestCase):
    def test_v3_differs_from_v2_only_in_the_option_line(self):
        config = load_config()
        for seed in (0, 3, 3100000):
            params = sample_params(config, seed)
            v2 = world_xml(params, config, physics_version=2).splitlines()
            v3 = world_xml(params, config, physics_version=3).splitlines()
            self.assertEqual(len(v2), len(v3))
            changed = [(a, b) for a, b in zip(v2, v3) if a != b]
            self.assertEqual(len(changed), 1, changed)
            old, new = changed[0]
            self.assertIn("<option", old)
            self.assertEqual(new, old.replace("/>", f' noslip_iterations="{NOSLIP_ITERATIONS_V3}"/>'))

    def test_v1_and_v2_have_no_noslip(self):
        config = load_config()
        params = sample_params(config, 3)
        for version in (1, 2):
            self.assertNotIn("noslip", world_xml(params, config, physics_version=version))

    def test_noslip_survives_every_reset(self):
        sim = MujocoSimulation(physics_version=3)
        try:
            self.assertEqual(sim.model.opt.noslip_iterations, 3)
            for seed in (3100000, 3100001):
                sim.reset(seed)
                self.assertEqual(sim.model.opt.noslip_iterations, 3)
                self.assertEqual(sim.model.opt.impratio, 10)
                self.assertEqual(sim.model.opt.cone, 1)  # elliptic
        finally:
            sim.close()
        v2 = MujocoSimulation(physics_version=2)
        try:
            self.assertEqual(v2.model.opt.noslip_iterations, 0)
        finally:
            v2.close()

    def test_hashes_follow_the_version(self):
        config = load_config()
        params = sample_params(config, 3100000)
        self.assertNotEqual(scene_hash(params, config, 2), scene_hash(params, config, 3))
        sim = MujocoSimulation(physics_version=3)
        try:
            self.assertNotEqual(config_hash(2, sim.asset_path), config_hash(3, sim.asset_path))
        finally:
            sim.close()


if __name__ == "__main__":
    unittest.main()
