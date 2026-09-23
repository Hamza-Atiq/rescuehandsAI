"""Every forbidden contact class the rules claim must actually be detected (spec §5, §12)."""
import copy
import unittest

import mujoco
import numpy as np

from rescuehandsai.contracts import BimanualAction
from rescuehandsai.pick_config import PICK_PHYSICS_VERSION, load_contacts
from rescuehandsai.pick_contacts import (JAW_UTENSIL, SCENE_NORMAL, VIOLATION, ContactClassifier,
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
        cls.sim = MujocoSimulation(physics_version=PICK_PHYSICS_VERSION)
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

    def test_non_colliding_shape_listed_as_a_scene_geom_is_rejected(self):
        bad = copy.deepcopy(self.config)
        bad["scene_geoms"]["table"].append("plate_rim")  # plate_rim is decorative: contype=0
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

    def test_grasp_shape_on_table_is_forbidden(self):
        # Owner decision 23 Sep: no robot shape may touch the table, grasp pads included.
        pad = self.g("right_arm/fixed_jaw_box5")
        x, y, _ = self.sim.data.geom_xpos[pad]
        move_geom_to(self.sim, pad, (x, y, -0.002))
        verdict = self.verdict_for(pad, self.g("table"))
        self.assertEqual((verdict.kind, verdict.labels), (VIOLATION, ("FORBIDDEN_CONTACT",)))
        for name in ("fixed_jaw_box3", "moving_jaw_box2", "moving_jaw_sph_tip1"):
            for a, b in ((self.g(f"right_arm/{name}"), self.g("table")), (self.g("table"), self.g(f"right_arm/{name}"))):
                self.assertEqual(self.clf.classify(a, b, 0.0).labels, ("FORBIDDEN_CONTACT",), name)
        severe = dict(self.config, severe_force_limit_n=-1.0)
        self.clf = ContactClassifier(self.sim.model, "fork", severe)
        self.assertIn("EXCESS_FORCE", self.verdict_for(pad, self.g("table")).labels)

    def test_the_removed_jaw_table_limit_is_rejected(self):
        # An old config must fail loudly, not quietly re-open a table permission.
        with self.assertRaises(ValueError):
            ContactClassifier(self.sim.model, "fork", dict(self.config, jaw_table_force_limit_n=None))

    # -- forbidden contacts: each one must be detectable ----------------------------
    def test_any_robot_shape_on_the_spare(self):
        spoon = self.g("spoon_handle")
        pad = self.g("right_arm/fixed_jaw_box5")
        move_geom_to(self.sim, pad, self.sim.data.geom_xpos[spoon])
        self.assertEqual(self.verdict_for(pad, spoon).labels, ("WRONG_ITEM_TOUCHED",))

    def test_left_arm_shape_on_the_spare_is_also_wrong_item_touched(self):
        spoon = self.g("spoon_handle")
        left_pad = self.g("left_arm/fixed_jaw_box5")
        move_geom_to(self.sim, left_pad, self.sim.data.geom_xpos[spoon])
        self.assertEqual(self.verdict_for(left_pad, spoon).labels, ("WRONG_ITEM_TOUCHED",))

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

    # -- jaw meshes: allowed on the named utensil only (owner decision 23 Sep) ---------
    def mesh_geom(self, arm, key):
        """The one collidable geom on `arm` matching a config selector {body, mesh}."""
        side = "moving" if key == "moving" else "fixed"
        sel = self.config["jaw_utensil_only_meshes"][side][0]
        model = self.sim.model
        found = [g for g in range(model.ngeom)
                 if model.body(int(model.geom_bodyid[g])).name == f"{arm}/{sel['body']}"
                 and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                 and model.mesh(int(model.geom_dataid[g])).name == f"{arm}/{sel['mesh']}"]
        self.assertEqual(len(found), 1, sel)
        return found[0]

    def test_jaw_mesh_selectors_resolve_to_the_evidenced_shapes(self):
        # Cross-check against the grip-probe evidence, which named these by compiled id.
        self.assertEqual(geom_name(self.sim.model, self.mesh_geom("right_arm", "moving")),
                         "right_arm/moving_jaw_so101_v1/geom_104")
        self.assertEqual(geom_name(self.sim.model, self.mesh_geom("right_arm", "fixed")),
                         "right_arm/gripper/geom_93")

    def test_jaw_meshes_support_the_named_utensil_in_both_orders(self):
        for side in ("fixed", "moving"):
            mesh = self.mesh_geom("right_arm", side)
            for part in ("fork_handle", "fork_neck"):
                for a, b in ((mesh, self.g(part)), (self.g(part), mesh)):
                    verdict = self.clf.classify(a, b, 5.0)
                    self.assertEqual((verdict.kind, verdict.jaw, verdict.labels), (JAW_UTENSIL, side, ()))

    def test_moving_jaw_mesh_real_contact_on_the_named_handle(self):
        mesh, handle = self.mesh_geom("right_arm", "moving"), self.g("fork_handle")
        move_geom_to(self.sim, mesh, self.sim.data.geom_xpos[handle] + DEEPER)
        verdict = self.verdict_for(mesh, handle)
        self.assertEqual((verdict.kind, verdict.jaw, verdict.labels), (JAW_UTENSIL, "moving", ()))

    def test_jaw_meshes_stay_forbidden_everywhere_else(self):
        for side in ("fixed", "moving"):
            mesh = self.mesh_geom("right_arm", side)
            cases = {"table": "FORBIDDEN_CONTACT", "plate": "FORBIDDEN_CONTACT", "cup_body": "FORBIDDEN_CONTACT",
                     "spoon_handle": "WRONG_ITEM_TOUCHED", "spoon_bowl": "WRONG_ITEM_TOUCHED"}
            for other, label in cases.items():
                for a, b in ((mesh, self.g(other)), (self.g(other), mesh)):
                    verdict = self.clf.classify(a, b, 0.0)
                    self.assertEqual((verdict.kind, verdict.labels), (VIOLATION, (label,)), (side, other))

    def test_jaw_mesh_table_contact_is_forbidden_in_a_real_contact(self):
        mesh = self.mesh_geom("right_arm", "fixed")
        x, y, _ = self.sim.data.geom_xpos[mesh]
        move_geom_to(self.sim, mesh, (x, y, -0.002))
        self.assertEqual(self.verdict_for(mesh, self.g("table")).labels, ("FORBIDDEN_CONTACT",))

    def test_jaw_mesh_permission_follows_the_named_utensil(self):
        clf = ContactClassifier(self.sim.model, "spoon", self.config)
        mesh = self.mesh_geom("right_arm", "moving")
        self.assertEqual(clf.classify(mesh, self.g("spoon_bowl"), 1.0).kind, JAW_UTENSIL)
        self.assertEqual(clf.classify(mesh, self.g("fork_handle"), 1.0).labels, ("WRONG_ITEM_TOUCHED",))

    def test_left_arm_jaw_meshes_get_no_permission(self):
        for side in ("fixed", "moving"):
            mesh = self.mesh_geom("left_arm", side)
            self.assertEqual(self.clf.classify(mesh, self.g("fork_handle"), 1.0).labels, ("FORBIDDEN_CONTACT",))

    def test_bad_jaw_mesh_selectors_are_rejected(self):
        for selector in ({"body": "moving_jaw_so101_v1", "mesh": "no_such_mesh"},
                         # the visual mesh on the same body cannot collide
                         {"body": "moving_jaw_so101_v1", "mesh": "moving_jaw_so101_v1"}):
            bad = copy.deepcopy(self.config)
            bad["jaw_utensil_only_meshes"]["moving"] = [selector]
            with self.assertRaises(ValueError):
                ContactClassifier(self.sim.model, "fork", bad)
        bad = copy.deepcopy(self.config)
        bad["jaw_utensil_only_meshes"]["fixed"] = bad["jaw_utensil_only_meshes"]["moving"]
        with self.assertRaises(ValueError):  # one shape cannot be both jaws
            ContactClassifier(self.sim.model, "fork", bad)

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
