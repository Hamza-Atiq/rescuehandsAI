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


def _perp_distances(points: np.ndarray, anchor: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Perpendicular distance from the world-frame line (anchor, axis) to each point."""
    rel = points - anchor
    return np.linalg.norm(rel - np.outer(rel @ axis, axis), axis=1)


def _is_descendant(model, ancestor_body: int, body: int) -> bool:
    """Is `ancestor_body` `body` itself or a strict ancestor of it?

    Walks `model.body_parentid` directly. This is deliberately independent of
    `_ancestor_offset` (the function under test elsewhere in pick_clearance.py): the
    set of shapes a test checks a joint's radius against must not be selected by the
    same code whose correctness the test exists to prove, or a false-negative bug in
    that selection would make production and the test silently agree.
    """
    k = body
    while True:
        if k == ancestor_body:
            return True
        if k == 0:
            return False
        k = int(model.body_parentid[k])


def _exact_farthest_point_distance(checker, model, data, g: int, anchor: np.ndarray, axis: np.ndarray) -> float:
    """Exact farthest distance from the world-frame axis line to any point of geom `g`.

    Sphere: distance-to-a-line is a convex function of the point, so a sphere's
    farthest point from the line is along the centre's own perpendicular to it, offset
    outward by the radius -- exact, not a sample. Capsule: the same convexity argument
    applied to its central segment (a convex set): the farthest point of a segment
    under a convex function is always at one of its two endpoints, each offset by the
    radius. Box/mesh: `checker._points[g]` already holds the exact corner/vertex set
    MuJoCo collides against (its convex hull), so the sampled max over those points IS
    the exact max for these two types -- unlike for sphere/capsule, where that same
    array is only a handful of axis-pole samples that can under-report the true
    farthest point for an arbitrary axis.
    """
    t, s = model.geom_type[g], model.geom_size[g]
    R = data.geom_xmat[g].reshape(3, 3)
    center = data.geom_xpos[g]
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        r = float(s[0])
        return float(_perp_distances(center[None, :], anchor, axis)[0]) + r
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        r, h = float(s[0]), float(s[1])
        endpoints = np.array([center + h * R[:, 2], center - h * R[:, 2]])
        return float(_perp_distances(endpoints, anchor, axis).max()) + r
    world_points = checker._points[g] @ R.T + center
    return float(_perp_distances(world_points, anchor, axis).max())


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
        # body, so descendant selection must actually walk the chain for this to exercise
        # anything -- done via `_is_descendant`, which re-walks `body_parentid`
        # independently of `_ancestor_offset` (the function under test), so a
        # false-negative bug there could not make this test vacuously agree with
        # production. Checked at several different poses since the bound must hold at
        # all of them (it is configuration-independent by construction).
        #
        # The "ground truth" distance is the EXACT farthest point of each shape from the
        # joint's world axis line (`_exact_farthest_point_distance`), not a coarse
        # sample: the right hand has real sphere/capsule collision geoms (finger tips,
        # r ~ 0.75-1.1 mm) whose farthest point from an arbitrary axis is generally not
        # one of `_local_points`'s 6-8 axis-aligned sample points -- close enough to the
        # 0.25 mm dip bound and the 1 mm clearance margin that a coarse sample could hide
        # a real unsoundness in `_radius`.
        model = self.sim.model
        joint_ids = [j for j in range(model.njnt)
                     if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
                     and model.joint(j).name.startswith("right_arm/")]
        self.assertTrue(joint_ids)

        # Sanity check on the offset math itself, independent of qpos (body_pos is a
        # static model quantity): every joint upstream of the gripper must have a
        # genuinely non-zero chain offset to the hand -- i.e. `_ancestor_offset` really
        # is summing body offsets along the chain, not just returning `True` with an
        # accidental offset of 0.
        moving_jaw_body = model.body("right_arm/moving_jaw_so101_v1").id
        gripper_joint = model.joint("right_arm/gripper").id
        upstream = [j for j in joint_ids if j != gripper_joint]
        self.assertTrue(upstream)
        for j in upstream:
            is_ancestor, offset = _ancestor_offset(model, int(model.jnt_bodyid[j]), moving_jaw_body)
            self.assertTrue(is_ancestor, model.joint(j).name)
            self.assertGreater(offset, 0.0, model.joint(j).name)

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
                         if _is_descendant(model, body_j, int(model.geom_bodyid[g]))]
                # Non-vacuous: every right-arm joint moves at least one checked shape in
                # this model, so this assertion must never be trivially skipped.
                self.assertTrue(moved, name)
                distance = max(_exact_farthest_point_distance(self.checker, model, d, g, anchor, axis)
                               for g in moved)
                self.assertGreaterEqual(self.checker._radius[name], distance, name)

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
