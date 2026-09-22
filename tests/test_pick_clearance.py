import unittest

import mujoco
import numpy as np

from rescuehandsai.expert import GRASP_DEPTH, OPEN, RIGHT_SIGN, TABLE_CLEARANCE, UTENSIL_MARGIN, ScriptedExpert
from rescuehandsai.pick_cells import cell_params, make_pick_task
from rescuehandsai.pick_clearance import MAX_DIP_BOUND_M, Clearance, ClearanceChecker, _ancestor_offset
from rescuehandsai.scene import sample_params
from rescuehandsai.sim import MujocoSimulation
from rescuehandsai.task import TaskSpec


def reset_like_runner(sim, seed, cell):
    task = make_pick_task(seed, cell, "T1")
    sim.reset(seed, instruction=task.instruction,
              params=cell_params(sample_params(sim.scene_config, seed), cell, sim.scene_config))
    return task


class ClearanceCheckerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = MujocoSimulation(physics_version=2)

    @classmethod
    def tearDownClass(cls):
        cls.sim.close()

    def setUp(self):
        self.task = reset_like_runner(self.sim, 3100000, "F-A")
        self.checker = ClearanceChecker(self.sim.model, "right_arm")

    def old_reach_pose(self):
        """The unchanged full-task expert's reach pose (1.5 mm grasp centre).

        The reach is commanded with the jaw open: the teacher's `utensil_reach` Move
        carries `q` alone, and the jaw is opened by the preceding `utensil_approach`
        Move, so the jaw stays at OPEN through the reach rather than its start value.
        """
        expert = ScriptedExpert(self.sim, TaskSpec(task_id="t", instruction=self.task.instruction,
                                                   utensil=self.task.utensil, seed=3100000),
                                subtasks=("pick_utensil",))
        g = expert._handle_geometry(self.task.utensil)
        pos, axis = expert._utensil_frame(self.task.utensil)
        center = pos + g["end"] * axis
        center[2] = max(TABLE_CLEARANCE, pos[2] - GRASP_DEPTH)
        q, *_ = expert._grasp_with_clearance("right_arm", center, g["half_width"], 0.05, approach_xy=axis[:2],
                                             pitches=(1.3, 1.4, 1.2, 1.0), margin=UTENSIL_MARGIN,
                                             closing_sign=RIGHT_SIGN)
        return q | {"right_arm/gripper": OPEN}

    def test_checks_the_collidable_jaw_meshes_and_pads(self):
        for name in ("geom_93", "geom_104", "right_arm/fixed_jaw_box3", "right_arm/moving_jaw_box2"):
            self.assertIn(name, self.checker.shapes)
        self.assertFalse(any(s.startswith("left_arm/") for s in self.checker.shapes))

    def test_home_pose_is_well_above_the_table(self):
        c = self.checker.lowest(self.sim.data.qpos, self.sim.home_targets)
        self.assertIsInstance(c, Clearance)
        self.assertGreater(c.z, 0.03)

    def test_old_reach_pose_is_inside_the_table_by_the_fixed_jaw_mesh(self):
        # FINDINGS §3: about -4.4 mm, lowest shape geom_93.
        c = self.checker.lowest(self.sim.data.qpos, dict(self.sim.previous) | self.old_reach_pose())
        self.assertLess(c.z, -0.003)
        self.assertEqual(c.shape, "geom_93")

    def test_sphere_and_capsule_shapes_report_their_exact_lowest_point(self):
        # The old reach pose rotates the hand enough that sampling a handful of surface
        # points on round shapes sits above the true lowest point; check every sphere and
        # capsule collision shape against the closed-form minimum instead.
        targets = dict(self.sim.previous) | self.old_reach_pose()
        self.checker.lowest(self.sim.data.qpos, targets)
        m, d = self.sim.model, self.checker._data
        checked = 0
        for g in self.checker._geoms:
            t = m.geom_type[g]
            if t == mujoco.mjtGeom.mjGEOM_SPHERE:
                r = float(m.geom_size[g][0])
                expected = float(d.geom_xpos[g][2] - r)
            elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
                r, h = float(m.geom_size[g][0]), float(m.geom_size[g][1])
                R = d.geom_xmat[g].reshape(3, 3)
                c = d.geom_xpos[g]
                expected = float(min(c[2] + h * R[2, 2], c[2] - h * R[2, 2]) - r)
            else:
                continue
            self.assertAlmostEqual(self.checker._shape_lowest_z(g), expected, delta=1e-9)
            checked += 1
        self.assertGreater(checked, 0)

    def test_path_sampling_matches_the_dip_bound_formula_and_reports_its_spacing(self):
        start = dict(self.sim.previous)
        goal = start | self.old_reach_pose()
        names = sorted(set(start) | set(goal))
        a = np.array([start.get(n, goal.get(n)) for n in names])
        b = np.array([goal.get(n, start.get(n)) for n in names])
        weighted_range = sum(abs(b[i] - a[i]) * self.checker._radius[n] for i, n in enumerate(names))
        expected = max(2, int(np.ceil(weighted_range / (2 * MAX_DIP_BOUND_M))) + 1)
        c = self.checker.along(self.sim.data.qpos, start, goal)
        self.assertEqual(c.samples, expected)
        # The hand really did move between neighbouring checked poses.
        self.assertGreater(c.max_point_step_m, 0.0)
        # The new algorithm guarantees a measured chord is never longer than the path
        # length between the two samples, and that path length is 2*dip_bound (the
        # dip bound is half the per-interval path-length budget) -- so this is a proof
        # consequence of the R_j chain, not an assumption; a target spacing of at most
        # 1 mm is the owner's check (c) on that guarantee.
        self.assertLessEqual(c.max_point_step_m, 2 * c.dip_bound_m + 1e-12)
        self.assertLessEqual(c.max_point_step_m, 0.001)

    def test_dip_bound_is_within_the_target_on_the_old_reach_path(self):
        start = dict(self.sim.previous)
        goal = start | self.old_reach_pose()
        c = self.checker.along(self.sim.data.qpos, start, goal)
        self.assertGreater(c.dip_bound_m, 0.0)
        self.assertLessEqual(c.dip_bound_m, MAX_DIP_BOUND_M)

    def test_along_is_a_proven_lower_bound_checked_against_a_finer_resample(self):
        # along()'s reported z must never be ABOVE the true minimum along the path: verify
        # by brute-force resampling the same path 10x denser with the exact lowest().
        start = dict(self.sim.previous)
        goal = start | self.old_reach_pose()
        c = self.checker.along(self.sim.data.qpos, start, goal)
        names = sorted(set(start) | set(goal))
        a = np.array([start.get(n, goal.get(n)) for n in names])
        b = np.array([goal.get(n, start.get(n)) for n in names])
        fine_n = c.samples * 10
        fine_min = min(self.checker.lowest(self.sim.data.qpos, dict(zip(names, a + (b - a) * f))).z
                       for f in np.linspace(0.0, 1.0, fine_n))
        self.assertGreaterEqual(fine_min, c.z)

    def test_joint_radius_bound_is_never_smaller_than_the_actual_distance_from_its_axis(self):
        # Every right-arm hinge joint, not just the gripper: for an upstream joint (e.g.
        # shoulder_pan) the checked shapes it moves are on DESCENDANT bodies, not its own
        # body, so `_ancestor_offset` must actually walk the chain for this to exercise
        # anything. Checked at several different poses since the bound must hold at all
        # of them (it is configuration-independent by construction).
        model = self.sim.model
        joint_ids = [j for j in range(model.njnt)
                     if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
                     and model.joint(j).name.startswith("right_arm/")]
        self.assertTrue(joint_ids)
        poses = [
            dict(self.sim.previous) | self.old_reach_pose(),
            dict(self.sim.home_targets),
            dict(self.sim.previous) | self.old_reach_pose() | {"right_arm/gripper": 0.9},
        ]
        for pose in poses:
            self.checker.lowest(self.sim.data.qpos, pose)   # poses the checker's private data
            d = self.checker._data
            for j in joint_ids:
                name = model.joint(j).name
                anchor, axis = d.xanchor[j].copy(), d.xaxis[j].copy()
                body_j = int(model.jnt_bodyid[j])
                moved = [g for g in self.checker._geoms
                         if _ancestor_offset(model, body_j, int(model.geom_bodyid[g]))[0]]
                if not moved:
                    self.assertEqual(self.checker._radius[name], 0.0, name)
                    continue
                points = np.concatenate([self.checker._points[g] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
                                         for g in moved])
                rel = points - anchor
                distance = np.linalg.norm(rel - np.outer(rel @ axis, axis), axis=1)
                self.assertGreaterEqual(self.checker._radius[name], float(distance.max()), name)

    def test_shapes_are_found_by_collision_settings_not_a_fixed_list(self):
        m = self.sim.model
        bodies = {m.body(f"right_arm/{b}").id for b in ("wrist", "gripper", "camera_mount", "moving_jaw_so101_v1")}
        expected = {m.geom(g).name or f"geom_{g}" for g in range(m.ngeom)
                    if m.geom_bodyid[g] in bodies and (m.geom_contype[g] or m.geom_conaffinity[g])}
        self.assertEqual(set(self.checker.shapes), expected)
        visual_only = [m.geom(g).name or f"geom_{g}" for g in range(m.ngeom)
                       if m.geom_bodyid[g] in bodies and not (m.geom_contype[g] or m.geom_conaffinity[g])]
        self.assertTrue(visual_only)                      # the model has visual-only meshes...
        self.assertFalse(set(visual_only) & set(self.checker.shapes))   # ...and they are skipped

    def test_path_minimum_is_never_above_either_end(self):
        start = dict(self.sim.previous)
        goal = start | self.old_reach_pose()
        c = self.checker.along(self.sim.data.qpos, start, goal)
        self.assertLessEqual(c.z, self.checker.lowest(self.sim.data.qpos, goal).z + 1e-12)
        self.assertLessEqual(c.z, self.checker.lowest(self.sim.data.qpos, start).z + 1e-12)
        self.assertTrue(0.0 <= c.fraction <= 1.0)

    def test_checker_does_not_touch_the_live_simulation(self):
        d = self.sim.data
        before = (d.qpos.copy(), d.qvel.copy(), d.ctrl.copy(), d.xpos.copy(), float(d.time), dict(self.sim.previous))
        reach = dict(self.sim.previous) | self.old_reach_pose()
        self.checker.lowest(d.qpos, reach)
        self.checker.along(d.qpos, dict(self.sim.previous), reach)
        self.checker.site_z(d.qpos, reach)
        np.testing.assert_array_equal(d.qpos, before[0])
        np.testing.assert_array_equal(d.qvel, before[1])
        np.testing.assert_array_equal(d.ctrl, before[2])
        np.testing.assert_array_equal(d.xpos, before[3])
        self.assertEqual(float(d.time), before[4])
        self.assertEqual(dict(self.sim.previous), before[5])

    def test_unknown_joint_is_a_clear_error(self):
        with self.assertRaises(KeyError):
            self.checker.lowest(self.sim.data.qpos, {"right_arm/no_such_joint": 0.0})


if __name__ == "__main__":
    unittest.main()
