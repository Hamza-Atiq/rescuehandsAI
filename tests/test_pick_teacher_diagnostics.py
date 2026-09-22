import unittest

from rescuehandsai.pick_teacher_diagnostics import TableContactTally, lift_chain

PLAN = {"reach_site_z_m": 0.004, "lift_site_z_m": 0.052}


def s(label, site_z, utensil_z, in_hand=(0.0, 0.0, 0.0)):
    return {"label": label, "site_z": site_z, "utensil_z": utensil_z, "in_hand": list(in_hand)}


class LiftChainTests(unittest.TestCase):
    def test_separates_plan_gap_hand_rise_and_slip(self):
        samples = [s("utensil_squeeze", 0.008, 0.006),
                   s("utensil_lift", 0.008, 0.006, (0.0, 0.0, 0.0)),
                   s("utensil_lift", 0.050, 0.047, (0.0, 0.0, -0.001)),
                   s("hold", 0.051, 0.040, (0.0, 0.0, -0.008))]
        c = lift_chain(PLAN, samples, utensil_start_z=0.006)
        self.assertAlmostEqual(c["requested_rise_m"], 0.05)
        self.assertAlmostEqual(c["solved_rise_m"], 0.048)
        self.assertAlmostEqual(c["reach_gap_m"], 0.004)        # stopped 4 mm above plan
        self.assertAlmostEqual(c["hand_rise_m"], 0.043)
        self.assertAlmostEqual(c["utensil_rise_in_lift_m"], 0.041)
        self.assertAlmostEqual(c["utensil_final_rise_m"], 0.034)
        self.assertAlmostEqual(c["max_slip_m"], 0.008)

    def test_missing_phases_are_none_not_zero(self):
        c = lift_chain(PLAN, [s("utensil_reach", 0.01, 0.006)], utensil_start_z=0.006)
        for key in ("reach_gap_m", "hand_rise_m", "utensil_rise_in_lift_m", "max_slip_m"):
            self.assertIsNone(c[key], key)


class TableContactTallyTests(unittest.TestCase):
    def test_keeps_peak_force_deepest_overlap_and_first_move(self):
        t = TableContactTally()
        t.add("geom_93", 6.8, -0.0006, "utensil_reach")
        t.add("geom_93", 56.1, -0.0004, "utensil_reach")
        t.add("geom_93", 1.0, -0.0013, "utensil_close")
        t.add("right_arm/fixed_jaw_box3", 2.0, -0.0001, "utensil_close")
        rows = {r["shape"]: r for r in t.rows()}
        self.assertEqual(rows["geom_93"]["samples"], 3)
        self.assertAlmostEqual(rows["geom_93"]["peak_force_n"], 56.1)
        self.assertAlmostEqual(rows["geom_93"]["deepest_m"], -0.0013)
        self.assertEqual(rows["geom_93"]["first_label"], "utensil_reach")
        self.assertEqual(len(rows), 2)

    def test_empty_tally_has_no_rows(self):
        self.assertEqual(TableContactTally().rows(), [])


class UtensilContactTallyTests(unittest.TestCase):
    def test_peak_count_and_first_label(self):
        from rescuehandsai.pick_teacher_diagnostics import UtensilContactTally
        t = UtensilContactTally()
        t.add("geom_104", 3.0, "utensil_close")
        t.add("geom_104", 7.5, "hold")
        t.add("right_arm/moving_jaw_box2", 1.0, "utensil_close")
        rows = {r["shape"]: r for r in t.rows()}
        self.assertEqual(rows["geom_104"], {"shape": "geom_104", "samples": 2, "peak_force_n": 7.5,
                                            "first_label": "utensil_close"})
        self.assertEqual(rows["right_arm/moving_jaw_box2"]["samples"], 1)


if __name__ == "__main__":
    unittest.main()
