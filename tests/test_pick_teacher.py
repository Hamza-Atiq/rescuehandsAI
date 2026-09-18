import unittest

from rescuehandsai.pick_cells import make_pick_task
from rescuehandsai.pick_config import load_rules
from rescuehandsai.pick_runner import PickEpisodeRunner
from rescuehandsai.pick_teacher import PickTeacher
from rescuehandsai.sim import MujocoSimulation


class TeacherProtocolTests(unittest.TestCase):
    """The runner accepts only a policy with this exact shape."""

    def test_metadata_names_the_teacher_and_claims_no_checkpoint(self):
        meta = PickTeacher().metadata()
        self.assertEqual(meta["name"], "pick_teacher")
        self.assertEqual(meta["backend"], "python")
        self.assertEqual(meta["device"], "cpu")
        self.assertIsNone(meta["checkpoint"])

    def test_teacher_declares_privileged_state(self):
        # It reads exact object poses and uses IK; the learned policy never does.
        self.assertTrue(PickTeacher.uses_privileged_state)

    def test_teacher_needs_no_images(self):
        self.assertFalse(PickTeacher().wants_images())

    def test_act_before_reset_is_a_clear_error(self):
        with self.assertRaises(RuntimeError):
            PickTeacher().act(object())


class TeacherPickTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = MujocoSimulation(physics_version=2)
        cls.rules = load_rules()

    @classmethod
    def tearDownClass(cls):
        cls.sim.close()

    def test_teacher_lifts_the_named_utensil_and_keeps_holding_it(self):
        task = make_pick_task(3100000, "F-A", "T1")
        teacher = PickTeacher()
        record = PickEpisodeRunner(self.sim, teacher, rules=self.rules).run(task)
        self.assertTrue(record["success"], record["outcome"]["failures"])
        # PickOutcome.picked reports pick ORDER ("named_first"/"spare_first"/"simultaneous"/
        # "neither"), not the utensil name (verified in pick_outcome.py:218-226). The
        # pick-only teacher never touches the spare, so a success must show "named_first".
        self.assertEqual(record["outcome"]["picked"], "named_first")
        self.assertIsNotNone(teacher.finished_pick_at)

    def test_teacher_holds_position_after_the_pick_instead_of_going_home(self):
        """The expert's script ends with a home move that would put the utensil back."""
        task = make_pick_task(3100000, "F-A", "T1")
        teacher = PickTeacher()
        teacher.reset(self.sim, task)
        seen = []
        for _ in range(400):
            action = teacher.act(self.sim.observe(images=False))
            seen.append(dict(action.targets))
            if teacher.finished_pick_at is not None and len(seen) > teacher.finished_pick_at + 5:
                break
            self.sim.step(action, stop_on_cross_arm=False)
        self.assertIsNotNone(teacher.finished_pick_at, "the teacher never finished the pick")
        after = seen[teacher.finished_pick_at:]
        for targets in after[1:]:
            self.assertEqual(targets, after[0], "targets moved after the pick was finished")
        self.assertEqual(teacher.phase, "hold")

    def test_a_planning_failure_is_reported_as_a_policy_error_not_a_crash(self):
        from rescuehandsai.expert import PlanningError

        class Unreachable(PickTeacher):
            def act(self, obs):
                raise PlanningError("no reachable pose")

        record = PickEpisodeRunner(self.sim, Unreachable(),
                                   rules=self.rules).run(make_pick_task(3100000, "F-A", "T1"))
        self.assertEqual(record["outcome"]["first_failure"], "POLICY_ERROR")


if __name__ == "__main__":
    unittest.main()
