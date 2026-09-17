import unittest

from rescuehandsai.pick_config import StepCounts, derive_steps, load_contacts, load_rules, whole_steps


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


if __name__ == "__main__":
    unittest.main()
