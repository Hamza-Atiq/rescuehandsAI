"""Physics version 2: the utensil's sampled friction governs its contacts (spec §4)."""
from dataclasses import replace
import hashlib
import json
import unittest

import numpy as np

from rescuehandsai.contracts import BimanualAction
from rescuehandsai.scene import ROOT, build_model, load_config, sample_params, world_xml
from rescuehandsai.showcase import scene_identity
from rescuehandsai.sim import MujocoSimulation


class PhysicsVersionPlumbingTests(unittest.TestCase):
    def test_version_1_default_and_explicit_match_reference(self):
        reference = json.loads((ROOT / "tests/data/physics_v1_reference.json").read_text())
        config = load_config()
        for seed, digest in reference["world_xml_sha256"].items():
            params = sample_params(config, int(seed))
            self.assertEqual(hashlib.sha256(world_xml(params, config, physics_version=1).encode()).hexdigest(), digest)

    def test_version_2_changes_only_utensil_geoms(self):
        config = load_config()
        params = sample_params(config, 3)
        v1 = world_xml(params, config).splitlines()
        v2 = world_xml(params, config, physics_version=2).splitlines()
        self.assertEqual(len(v1), len(v2))
        changed = [(a, b) for a, b in zip(v1, v2) if a != b]
        self.assertTrue(changed)
        for _, line in changed:
            self.assertTrue(any(f'name="{n}' in line for n in ("fork_", "spoon_")), line)
            self.assertIn('priority="2"', line)
            self.assertIn('condim="6"', line)

    def test_unknown_version_is_rejected(self):
        config = load_config()
        with self.assertRaises(ValueError):
            world_xml(sample_params(config, 0), config, physics_version=4)
        with self.assertRaises(ValueError):
            MujocoSimulation(physics_version=0)

    def test_simulation_records_version_and_scene_identity_follows_it(self):
        v1, v2 = MujocoSimulation(), MujocoSimulation(physics_version=2)
        try:
            self.assertEqual((v1.physics_version, v2.physics_version), (1, 2))
            self.assertNotEqual(scene_identity(v1), scene_identity(v2))
        finally:
            v1.close()
            v2.close()

    def test_reset_accepts_explicit_params_with_matching_seed(self):
        sim = MujocoSimulation(physics_version=2)
        try:
            config = sim.scene_config
            params = sample_params(config, 7)
            params = replace(params, frictions={**params.frictions, "fork": 0.2})
            sim.reset(7, params=params)
            self.assertEqual(sim.scene_params.frictions["fork"], 0.2)
            with self.assertRaises(ValueError):
                sim.reset(8, params=params)
        finally:
            sim.close()


class ContactFrictionTests(unittest.TestCase):
    """The friction MuJoCo actually uses in a jaw-utensil contact."""

    def jaw_contact_friction(self, physics_version, fork_friction):
        sim = MujocoSimulation(physics_version=physics_version)
        try:
            params = sample_params(sim.scene_config, 11)
            params = replace(params, frictions={**params.frictions, "fork": fork_friction})
            sim.reset(11, params=params)
            model, data = sim.model, sim.data
            import mujoco
            # MuJoCo's midphase broadphase caches a compile-time bounding structure per body;
            # it is not updated by a raw geom_pos write, so forcing a jaw geom onto the handle
            # here (to sample the contact friction MuJoCo would use) silently yields zero
            # contacts unless midphase is disabled on this throwaway model. This does not
            # touch production code or any friction/priority/condim value under test.
            model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MIDPHASE
            jaw = model.geom("right_arm/fixed_jaw_box5").id
            handle = model.geom("fork_handle").id
            body = int(model.geom_bodyid[jaw])
            target = data.geom_xpos[handle]
            model.geom_pos[jaw] = data.xmat[body].reshape(3, 3).T @ (target - data.xpos[body])
            mujoco.mj_forward(model, data)
            for contact in data.contact:
                if {int(contact.geom1), int(contact.geom2)} == {jaw, handle}:
                    return float(contact.friction[0])
            self.fail("no jaw-handle contact was created")
        finally:
            sim.close()

    def test_version_1_ignores_utensil_friction(self):
        self.assertAlmostEqual(self.jaw_contact_friction(1, 0.2), 1.0, places=6)
        self.assertAlmostEqual(self.jaw_contact_friction(1, 1.2), 1.0, places=6)

    def test_version_2_uses_utensil_friction(self):
        self.assertAlmostEqual(self.jaw_contact_friction(2, 0.2), 0.2, places=6)
        self.assertAlmostEqual(self.jaw_contact_friction(2, 1.2), 1.2, places=6)


from rescuehandsai.expert import ScriptedExpert
from rescuehandsai.task import make_task


def fork_in_gripper_frame(sim):
    site = sim.data.site("right_arm/gripperframe")
    rotation = site.xmat.reshape(3, 3)
    return rotation.T @ (sim.data.body("fork").xpos - site.xpos)


def lift_slip(physics_version, seed, friction, max_steps=160):
    """Teacher grasp on the fork; return (slip in gripper frame over the first lift, fork rise).

    Returns None when the teacher never reaches its first lift without staging."""
    sim = MujocoSimulation(physics_version=physics_version)
    try:
        params = sample_params(sim.scene_config, seed)
        params = replace(params, frictions={**params.frictions, "fork": friction})
        sim.reset(seed, params=params)
        expert = ScriptedExpert(sim, make_task(seed, utensil="fork", template=0), subtasks=("pick_utensil",))
        before = start_z = None
        for _ in range(max_steps):
            action = expert.act(sim.observe())
            if expert.phase.startswith("stage_"):  # the left arm is staging the utensil: not this test
                return None
            if expert.phase == "utensil_lift" and before is None:
                before, start_z = fork_in_gripper_frame(sim), float(sim.data.body("fork").xpos[2])
            if before is not None and expert.phase != "utensil_lift":
                break
            sim.step(action)
        if before is None:
            return None
        return float(np.linalg.norm(fork_in_gripper_frame(sim) - before)), float(sim.data.body("fork").xpos[2] - start_z)
    finally:
        sim.close()


class SlipTest(unittest.TestCase):
    """Same scene, same scripted grasp and lift; only the fork's friction differs (spec §4 b)."""

    def test_low_friction_slips_more_than_high_friction(self):
        for seed in range(20):
            high = lift_slip(2, seed, 1.2)
            low = lift_slip(2, seed, 0.05)
            if high is None or low is None:
                continue
            high_slip, high_rise = high
            low_slip, low_rise = low
            print(f"seed {seed}: slip high={high_slip:.4f} m rise={high_rise:.4f}; slip low={low_slip:.4f} m rise={low_rise:.4f}")
            self.assertGreater(high_rise, 0.02, "the high-friction grasp must lift the fork")
            self.assertGreater(low_slip, high_slip + 0.005)
            return
        self.fail("no seed in 0-19 reached a lift without staging; investigate the teacher before continuing")


if __name__ == "__main__":
    unittest.main()
