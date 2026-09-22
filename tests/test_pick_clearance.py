import unittest

import numpy as np

from rescuehandsai.expert import GRASP_DEPTH, RIGHT_SIGN, TABLE_CLEARANCE, UTENSIL_MARGIN, ScriptedExpert
from rescuehandsai.pick_cells import cell_params, make_pick_task
from rescuehandsai.pick_clearance import Clearance, ClearanceChecker
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
        """The unchanged full-task expert's reach pose (1.5 mm grasp centre)."""
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
        return q

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

    def test_path_sampling_is_dense_enough_and_reports_its_spacing(self):
        start = dict(self.sim.previous)
        goal = start | self.old_reach_pose()
        c = self.checker.along(self.sim.data.qpos, start, goal)
        biggest = max(abs(goal[n] - start[n]) for n in goal)
        self.assertGreaterEqual(c.samples, int(np.ceil(biggest / 0.005)) + 1)
        # No hand point jumps more than 1 mm between neighbouring checked poses.
        self.assertGreater(c.max_point_step_m, 0.0)
        self.assertLessEqual(c.max_point_step_m, 0.001)

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
