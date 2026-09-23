import unittest

from rescuehandsai.pick_config import PICK_PHYSICS_VERSION, StepCounts, derive_steps, load_contacts, load_rules, whole_steps


class PickConfigTests(unittest.TestCase):
    def test_steps_come_from_configured_time_steps(self):
        steps = derive_steps(load_rules(), physics_dt=0.005, control_dt=0.05)
        self.assertEqual(steps, StepCounts(substeps=10, hold=200, max_gap=20, final_speed=50, deadline_control=300))

    def test_other_time_steps_change_the_counts(self):
        steps = derive_steps(load_rules(), physics_dt=0.002, control_dt=0.05)
        self.assertEqual((steps.substeps, steps.hold, steps.max_gap, steps.final_speed), (25, 500, 50, 125))

    def test_non_whole_ratios_are_rejected(self):
        with self.assertRaises(ValueError):
            whole_steps(0.052, 0.005, "control_dt")
        rules = dict(load_rules(), hold_s=1.0025)
        with self.assertRaises(ValueError):
            derive_steps(rules, 0.005, 0.05)

    def test_configs_start_unfrozen(self):
        self.assertFalse(load_rules()["frozen"])
        contacts = load_contacts()
        self.assertFalse(contacts["force_limits_frozen"])
        self.assertIsNone(contacts["jaw_table_force_limit_n"])

    def test_rules_task_and_start_check_use_one_physics_version(self):
        # Adoption of v3 (23 Sep) must be coherent: a rules file on one version and task or
        # start-check defaults on another would raise CONTRACT_MISMATCH at run time.
        import inspect
        from rescuehandsai.pick_cells import check_start, make_pick_task

        self.assertEqual(PICK_PHYSICS_VERSION, 3)
        self.assertEqual(load_rules()["physics_version"], PICK_PHYSICS_VERSION)
        self.assertEqual(make_pick_task(5, "F-A", "T1").physics_version, PICK_PHYSICS_VERSION)
        default = inspect.signature(check_start).parameters["physics_version"].default
        self.assertEqual(default, PICK_PHYSICS_VERSION)


if __name__ == "__main__":
    unittest.main()
