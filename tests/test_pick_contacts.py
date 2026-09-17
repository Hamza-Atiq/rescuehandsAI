"""Every forbidden contact class the rules claim must actually be detected (spec §5, §12)."""
import copy
import unittest

import mujoco
import numpy as np

from rescuehandsai.contracts import BimanualAction
from rescuehandsai.pick_config import load_contacts
from rescuehandsai.pick_contacts import (JAW_TABLE, JAW_UTENSIL, SCENE_NORMAL, VIOLATION, ContactClassifier,
                                         collides, geom_name)
from rescuehandsai.sim import MujocoSimulation


def body_collision_geoms(model, body):
    b = model.body(body).id
    return [g for g in range(model.ngeom) if model.geom_bodyid[g] == b and collides(model, g)]


def move_geom_to(sim, geom, world_point):
    """Test-only: put a geom's centre at a world point, then recompute contacts.

    MuJoCo 3.13.0 builds a per-body midphase bounding structure at model compile
    time; it is not invalidated by writing model.geom_pos afterwards, so a moved
    geom would otherwise never be tested for collisions. Disabling midphase on
    this test model makes the moved geom actually collide. Production code must
    never do this (it would blunt real broad-phase culling); it is confined to
    this test helper.
    """
    model, data = sim.model, sim.data
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MIDPHASE
    body = int(model.geom_bodyid[geom])
    model.geom_pos[geom] = data.xmat[body].reshape(3, 3).T @ (np.asarray(world_point, float) - data.xpos[body])
    mujoco.mj_forward(model, data)


# Test-only: for a few pairs (small box vs. large flat cylinder, mesh vs. box) placing the
# moving geom's centre exactly on the target geom's centre is a degenerate case: MuJoCo's
# convex-convex query reports the surfaces as exactly touching (dist == 0.0), not penetrating,
# so no contact is generated. Nudging the target 3 mm further along -z gives real, measurable
# penetration without changing which geoms are involved.
DEEPER = np.array([0.0, 0.0, -0.003])


class ContactClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = MujocoSimulation(physics_version=2)
        cls.config = load_contacts()

    @classmethod
    def tearDownClass(cls):
        cls.sim.close()

    def setUp(self):
        self.sim.reset(21)  # rebuilds the model, undoing any moved geom
        self.named = "fork"
        self.clf = ContactClassifier(self.sim.model, self.named, self.config)

    def g(self, name):
        return self.sim.model.geom(name).id

    def verdict_for(self, a, b):
        names = {geom_name(self.sim.model, a), geom_name(self.sim.model, b)}
        found = [v for v in self.clf.contacts(self.sim.data) if set(v.geoms) == names]
        self.assertTrue(found, f"no contact detected between {sorted(names)}")
        return found[0]

    # -- configuration checks ---------------------------------------------------
    def test_config_matches_the_model(self):
        self.assertEqual(self.clf.spare, "spoon")
        for name in self.config["decorative_geoms"]:
            self.assertFalse(collides(self.sim.model, self.g(name)), name)

    def test_collidable_shape_listed_as_decorative_is_rejected(self):
        bad = copy.deepcopy(self.config)
        bad["decorative_geoms"].append("table")
        with self.assertRaises(ValueError):
            ContactClassifier(self.sim.model, "fork", bad)

    def test_missing_geom_is_rejected(self):
        bad = copy.deepcopy(self.config)
        bad["jaw_grasp_geoms"]["fixed"].append("no_such_pad")
        with self.assertRaises(ValueError):
            ContactClassifier(self.sim.model, "fork", bad)

    # -- allowed contacts ----------------------------------------------------------
    def test_items_resting_on_the_table_are_normal(self):
        for _ in range(5):
            self.sim.step(BimanualAction(self.sim.observe().timestamp, self.sim.home_targets))
        verdicts = self.clf.contacts(self.sim.data)
        self.assertTrue(verdicts)
        self.assertTrue(all(v.kind == SCENE_NORMAL for v in verdicts), verdicts)

    def test_fixed_and_moving_grasp_shapes_on_the_named_utensil(self):
        handle = self.g("fork_handle")
        fixed = self.g("right_arm/fixed_jaw_box5")
        move_geom_to(self.sim, fixed, self.sim.data.geom_xpos[handle])
        verdict = self.verdict_for(fixed, handle)
        self.assertEqual((verdict.kind, verdict.jaw, verdict.labels), (JAW_UTENSIL, "fixed", ()))
        self.setUp()
        handle, moving = self.g("fork_handle"), self.g("right_arm/moving_jaw_box2")
        move_geom_to(self.sim, moving, self.sim.data.geom_xpos[handle])
        verdict = self.verdict_for(moving, handle)
        self.assertEqual((verdict.kind, verdict.jaw), (JAW_UTENSIL, "moving"))

    def test_grasp_shape_on_table_is_allowed_below_limit_and_judged_above(self):
        pad = self.g("right_arm/fixed_jaw_box5")
        x, y, _ = self.sim.data.geom_xpos[pad]
        move_geom_to(self.sim, pad, (x, y, -0.002))
        self.assertEqual(self.verdict_for(pad, self.g("table")).kind, JAW_TABLE)
        strict = dict(self.config, jaw_table_force_limit_n=-1.0)
        self.clf = ContactClassifier(self.sim.model, "fork", strict)
        self.assertEqual(self.verdict_for(pad, self.g("table")).labels, ("FORBIDDEN_CONTACT",))
        severe = dict(self.config, severe_force_limit_n=-1.0)
        self.clf = ContactClassifier(self.sim.model, "fork", severe)
        self.assertIn("EXCESS_FORCE", self.verdict_for(pad, self.g("table")).labels)

    # -- forbidden contacts: each one must be detectable ----------------------------
    def test_any_robot_shape_on_the_spare(self):
        spoon = self.g("spoon_handle")
        pad = self.g("right_arm/fixed_jaw_box5")
        move_geom_to(self.sim, pad, self.sim.data.geom_xpos[spoon])
        self.assertEqual(self.verdict_for(pad, spoon).labels, ("WRONG_ITEM_TOUCHED",))

    def test_housing_box_and_jaw_mesh_are_not_grasp_contacts(self):
        model = self.sim.model
        housing = next(g for g in body_collision_geoms(model, "right_arm/gripper")
                       if not model.geom(g).name and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX)
        handle = self.g("fork_handle")
        move_geom_to(self.sim, housing, self.sim.data.geom_xpos[handle])
        self.assertEqual(self.verdict_for(housing, handle).labels, ("FORBIDDEN_CONTACT",))
        self.setUp()
        model = self.sim.model
        mesh = next(g for g in body_collision_geoms(model, "right_arm/moving_jaw_so101_v1")
                    if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH)
        handle = self.g("fork_handle")
        move_geom_to(self.sim, mesh, self.sim.data.geom_xpos[handle] + DEEPER)
        self.assertEqual(self.verdict_for(mesh, handle).labels, ("FORBIDDEN_CONTACT",))
        box1 = self.g("right_arm/fixed_jaw_box1")
        self.setUp()
        handle = self.g("fork_handle")
        move_geom_to(self.sim, box1, self.sim.data.geom_xpos[handle])
        self.assertEqual(self.verdict_for(box1, handle).labels, ("FORBIDDEN_CONTACT",))

    def test_left_arm_touching_anything(self):
        left_pad = self.g("left_arm/fixed_jaw_box5")
        handle = self.g("fork_handle")
        move_geom_to(self.sim, left_pad, self.sim.data.geom_xpos[handle])
        self.assertEqual(self.verdict_for(left_pad, handle).labels, ("FORBIDDEN_CONTACT",))

    def test_robot_on_plate_and_cup(self):
        pad = self.g("right_arm/fixed_jaw_box5")
        plate = self.g("plate")
        move_geom_to(self.sim, plate, self.sim.data.geom_xpos[pad] + DEEPER)  # the plate is a world geom
        self.assertEqual(self.verdict_for(pad, plate).labels, ("FORBIDDEN_CONTACT",))
        self.setUp()
        pad, cup = self.g("right_arm/fixed_jaw_box5"), self.g("cup_body")
        move_geom_to(self.sim, pad, self.sim.data.geom_xpos[cup] + DEEPER)
        self.assertEqual(self.verdict_for(pad, cup).labels, ("FORBIDDEN_CONTACT",))

    def test_non_jaw_right_arm_shape_on_table(self):
        forearm = body_collision_geoms(self.sim.model, "right_arm/lower_arm")[0]
        x, y, _ = self.sim.data.geom_xpos[forearm]
        move_geom_to(self.sim, forearm, (x, y, -0.005))
        self.assertEqual(self.verdict_for(forearm, self.g("table")).labels, ("FORBIDDEN_CONTACT",))

    def test_self_collision_and_arm_arm_contact(self):
        model = self.sim.model
        upper = body_collision_geoms(model, "right_arm/upper_arm")[0]
        wrist = body_collision_geoms(model, "right_arm/wrist")[0]
        move_geom_to(self.sim, upper, self.sim.data.geom_xpos[wrist])
        self.assertEqual(self.verdict_for(upper, wrist).labels, ("SELF_COLLISION",))
        self.setUp()
        model = self.sim.model
        left = body_collision_geoms(model, "left_arm/lower_arm")[0]
        wrist = body_collision_geoms(model, "right_arm/wrist")[0]
        move_geom_to(self.sim, left, self.sim.data.geom_xpos[wrist])
        self.assertEqual(self.verdict_for(left, wrist).labels, ("ARM_ARM_CONTACT",))

    def test_unlisted_scene_shape_is_unknown(self):
        partial = copy.deepcopy(self.config)
        del partial["scene_geoms"]["plate"]
        clf = ContactClassifier(self.sim.model, "fork", partial)
        verdict = clf.classify(self.g("plate"), self.g("cup_body"), 0.0)
        self.assertEqual((verdict.kind, verdict.labels), (VIOLATION, ("UNKNOWN_CONTACT_PAIR",)))
        verdict = clf.classify(self.g("right_arm/fixed_jaw_box5"), self.g("plate"), 0.0)
        self.assertEqual(verdict.labels, ("UNKNOWN_CONTACT_PAIR",))


if __name__ == "__main__":
    unittest.main()
