import unittest

from rescuehandsai.pick_cells import CELLS
from rescuehandsai.pick_stats import INCOMPLETE, MAIN_TEST_BARS, scored_entry, summarize


def entry(scene, cell, success=True, template="T1", labels=(), picked="named_first"):
    return {"scene": scene, "cell": cell, "template": template, "valid": True, "attempt": 1, "success": success,
            "picked": picked, "failure_labels": list(labels)}


def keys(scenes):
    return [f"{s}_{c}" for s in scenes for c in CELLS]


class SummaryTests(unittest.TestCase):
    def test_hand_worked_example(self):
        scored = [entry(1, c) for c in CELLS] + [entry(2, "F-A"), entry(2, "F-B"),
                                                 entry(2, "S-A", False, labels=["WRONG_ITEM_TOUCHED"], picked="spare_first"),
                                                 entry(2, "S-B", False, labels=["NO_LIFT"], picked="neither")]
        s = summarize(scored, planned_keys=keys([1, 2]))
        self.assertTrue(s["evaluation_valid"])
        self.assertEqual((s["episodes"]["successes"], s["episodes"]["episodes"]), (6, 8))
        self.assertEqual((s["scenes_all_four"]["passing"], s["scenes_all_four"]["complete_scenes"]), (1, 2))
        self.assertEqual(s["per_cell"]["S-A"], {"successes": 1, "episodes": 2})
        self.assertEqual(s["per_word"]["spoon"], {"successes": 2, "episodes": 4})
        self.assertEqual(s["per_slot"]["1"], {"successes": 3, "episodes": 4})  # F-B and S-A
        self.assertEqual(s["spare_touch_episodes"], 1)
        self.assertEqual(s["picked"], {"named_first": 6, "spare_first": 1, "neither": 1})
        self.assertEqual(s["failure_labels"], {"WRONG_ITEM_TOUCHED": 1, "NO_LIFT": 1})
        self.assertEqual(s["verdict"], "no pass bars for this run")

    def test_scene_grouping_widens_the_interval(self):
        scored = []
        for scene in range(10):
            scored += [entry(scene, c, success=scene < 5) for c in CELLS]
        s = summarize(scored, planned_keys=keys(range(10)), n_boot=4000)
        low, high = s["episodes"]["ci95"]
        self.assertAlmostEqual(s["episodes"]["rate"], 0.5)
        self.assertGreater(high - low, 0.45)  # 40 independent episodes would give about 0.31
        self.assertEqual(s, summarize(scored, planned_keys=keys(range(10)), n_boot=4000))

    def test_invalid_and_duplicate_entries_are_rejected(self):
        bad = dict(entry(1, "F-A"), valid=False)
        with self.assertRaises(ValueError):
            summarize([bad], planned_keys=keys([1]))
        with self.assertRaises(ValueError):
            summarize([entry(1, "F-A"), entry(1, "F-A")], planned_keys=keys([1]))
        with self.assertRaises(ValueError):
            summarize([entry(9, "F-A")], planned_keys=keys([1]))

    def test_incomplete_or_blocked_runs_are_not_eligible(self):
        scored = [entry(1, c) for c in CELLS]
        s = summarize(scored, planned_keys=keys([1, 2]), bars=MAIN_TEST_BARS)
        self.assertFalse(s["evaluation_valid"])
        self.assertEqual((s["verdict"], s["bars"]), (INCOMPLETE, None))
        self.assertEqual(len(s["missing_episodes"]), 4)
        s = summarize(scored, planned_keys=keys([1]), blocked_keys=["1_F-A"], bars=MAIN_TEST_BARS)
        self.assertEqual(s["verdict"], INCOMPLETE)

    def test_main_test_bars(self):
        scored = [entry(scene, c) for scene in range(100) for c in CELLS]
        s = summarize(scored, planned_keys=keys(range(100)), bars=MAIN_TEST_BARS, n_boot=200)
        self.assertEqual(s["verdict"], "pass")
        failing = [dict(e, success=False) if e["cell"] == "F-A" and e["scene"] < 9 else e for e in scored]
        s = summarize(failing, planned_keys=keys(range(100)), bars=MAIN_TEST_BARS, n_boot=200)
        self.assertEqual(s["bars"]["each_cell"]["pass"], False)  # F-A 91 < 92
        self.assertEqual(s["bars"]["episode_successes"]["pass"], True)  # 391 >= 380
        self.assertEqual(s["verdict"], "fail")

    def test_scored_entry_from_a_runner_record(self):
        record = {"seed": 3100001, "cell": "S-B", "template": "T2", "success": False,
                  "outcome": {"picked": "spare_first", "failures": [{"label": "WRONG_ITEM_TOUCHED"}]}}
        self.assertEqual(scored_entry(record, 2), {"scene": 3100001, "cell": "S-B", "template": "T2", "valid": True,
                                                   "attempt": 2, "success": False, "picked": "spare_first",
                                                   "failure_labels": ["WRONG_ITEM_TOUCHED"]})


if __name__ == "__main__":
    unittest.main()
