import unittest

from rescuehandsai.task import TEMPLATES, make_task, subtask_instruction, SUBTASKS


class TaskTests(unittest.TestCase):
    def test_every_instruction_names_only_the_requested_utensil(self):
        for seed in range(40):
            for utensil, other in (("fork", "spoon"), ("spoon", "fork")):
                task = make_task(seed, utensil)
                self.assertIn(utensil, task.instruction)
                self.assertNotIn(other, task.instruction)

    def test_seeded_and_both_utensils_occur(self):
        self.assertEqual(make_task(5), make_task(5))
        self.assertEqual({make_task(s).utensil for s in range(30)}, {"fork", "spoon"})
        self.assertGreater(len({make_task(s).instruction for s in range(30)}), len(TEMPLATES))

    def test_rejects_unknown_utensil(self):
        with self.assertRaises(ValueError):
            make_task(0, "knife")

    def test_subtask_text(self):
        for sub in SUBTASKS:
            self.assertIn("spoon" if sub != "place_cup" else "cup", subtask_instruction(sub, "spoon"))


if __name__ == "__main__":
    unittest.main()
