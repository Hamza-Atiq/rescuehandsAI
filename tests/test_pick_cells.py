"""The four runs of one scene differ only in the word and the tray order (spec §3, §8)."""
from collections import Counter
from dataclasses import replace
import unittest

import numpy as np

from rescuehandsai.pick_cells import (CELLS, HELD_BACK_TEMPLATES, TRAIN_TEMPLATES, assign_templates, cell_params,
                                      check_scene, check_start, make_pick_task, named_slot, slot_jitter)
from rescuehandsai.scene import load_config, sample_params
from rescuehandsai.sim import MujocoSimulation


class CellTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.base = sample_params(self.config, 5)
        self.cells = {cell: cell_params(self.base, cell, self.config) for cell in CELLS}

    def test_properties_follow_the_item_and_everything_else_is_shared(self):
        for params in self.cells.values():
            for field in ("masses", "frictions", "utensil_scale", "cup_radius", "cup_half_height",
                          "table_rgb", "light_diffuse", "light_offset", "seed"):
                self.assertEqual(getattr(params, field), getattr(self.base, field), field)
            self.assertEqual(params.poses["cup"], self.base.poses["cup"])

    def test_jitter_follows_the_slot(self):
        a, b = self.cells["F-A"], self.cells["F-B"]
        np.testing.assert_allclose(b.poses["fork"], a.poses["spoon"], atol=1e-12)
        np.testing.assert_allclose(b.poses["spoon"], a.poses["fork"], atol=1e-12)
        self.assertEqual(self.cells["F-A"].poses, self.cells["S-A"].poses)
        self.assertEqual(self.cells["F-B"].poses, self.cells["S-B"].poses)
        for params in self.cells.values():
            for slot, jitter in slot_jitter(params, self.config).items():
                np.testing.assert_allclose(jitter, slot_jitter(self.base, self.config)[slot], atol=1e-12)

    def test_named_slots(self):
        self.assertEqual({c: named_slot(c) for c in CELLS}, {"F-A": 0, "F-B": 1, "S-A": 1, "S-B": 0})
        self.assertEqual(self.cells["F-A"].slots, {"fork": 0, "spoon": 1})
        self.assertEqual(self.cells["S-B"].slots, {"spoon": 0, "fork": 1})

    def test_tasks_differ_only_in_the_word(self):
        tasks = {cell: make_pick_task(5, cell, "T3") for cell in CELLS}
        texts = {task.instruction.replace(task.utensil, "{u}") for task in tasks.values()}
        self.assertEqual(texts, {TRAIN_TEMPLATES["T3"]})
        self.assertEqual((tasks["F-A"].utensil, tasks["F-A"].spare), ("fork", "spoon"))
        self.assertEqual((tasks["S-B"].utensil, tasks["S-B"].spare), ("spoon", "fork"))
        self.assertEqual(make_pick_task(5, "S-A", "H2").instruction, HELD_BACK_TEMPLATES["H2"].format(u="spoon"))
        with self.assertRaises(KeyError):
            make_pick_task(5, "X-Y", "T1")

    def test_no_template_names_a_side_or_slot(self):
        for text in list(TRAIN_TEMPLATES.values()) + list(HELD_BACK_TEMPLATES.values()):
            for word in ("left", "slot", "first", "second", "near", "far"):
                self.assertNotIn(word, text.lower())

    def test_template_rotation_is_fixed_and_balanced(self):
        main = assign_templates(100, sorted(TRAIN_TEMPLATES))
        self.assertEqual(main, assign_templates(100, sorted(TRAIN_TEMPLATES)))
        self.assertEqual(sorted(Counter(main).values()), [12, 12, 12, 12, 13, 13, 13, 13])
        self.assertEqual(set(Counter(assign_templates(24, sorted(HELD_BACK_TEMPLATES))).values()), {6})
        self.assertEqual(set(Counter(assign_templates(24, sorted(TRAIN_TEMPLATES))).values()), {3})


class StartCheckTests(unittest.TestCase):
    def test_normal_scene_is_valid_in_all_cells(self):
        results = check_scene(5)
        self.assertEqual(set(results), set(CELLS))
        for result in results.values():
            self.assertTrue(result.valid, result.reasons)

    def test_overlap_and_off_table_are_rejected(self):
        def overlap(params):
            return replace(params, poses={**params.poses, "spoon": params.poses["fork"]})

        result = check_start(5, "F-A", edit=overlap)
        self.assertFalse(result.valid)
        self.assertTrue(any(r.startswith("overlap:") for r in result.reasons), result.reasons)

        def off_table(params):
            x, y, yaw = params.poses["fork"]
            return replace(params, poses={**params.poses, "fork": (x + 2.0, y, yaw)})

        result = check_start(5, "F-A", edit=off_table)
        self.assertIn("off_table:fork", result.reasons)

    def test_check_runs_on_its_own_copy(self):
        sim = MujocoSimulation(physics_version=2)
        try:
            sim.reset(5)
            qpos, time = sim.data.qpos.copy(), float(sim.data.time)
            check_start(5, "F-A")
            np.testing.assert_array_equal(sim.data.qpos, qpos)
            self.assertEqual(float(sim.data.time), time)
        finally:
            sim.close()


if __name__ == "__main__":
    unittest.main()
