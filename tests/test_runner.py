"""Episode runner: supervisor recovery, watchdog and action guard."""
import unittest

from rescuehandsai.contracts import BimanualAction
from rescuehandsai.perturb import GripperGlitch
from rescuehandsai.policies.scripted import ScriptedPolicy
from rescuehandsai.runner import EpisodeLog, EpisodeRunner, clamp_action
from rescuehandsai.sim import MujocoSimulation
from rescuehandsai.task import make_task


class IdlePolicy:
    """Never tries anything: the watchdog must notice."""
    name = "idle"
    uses_privileged_state = False

    def reset(self, sim, task):
        self.targets = dict(sim.home_targets)

    def wants_images(self):
        return False

    def act(self, obs):
        return BimanualAction(obs.timestamp, self.targets)

    def after_recovery(self, sim, task, progress):
        pass

    def metadata(self):
        return {"name": self.name}


class ClampTests(unittest.TestCase):
    def test_limits_and_step_size_are_enforced(self):
        previous = {"a": 0.0, "b": 0.0}
        limits = {"a": (-1.0, 1.0), "b": (-0.1, 0.1)}
        safe, clamped = clamp_action(BimanualAction(0.0, {"a": 0.5, "b": -5.0}), previous, limits, 0.1)
        self.assertAlmostEqual(safe.targets["a"], 0.098)
        self.assertAlmostEqual(safe.targets["b"], -0.098)
        self.assertEqual(clamped, 2)
        with self.assertRaises(ValueError):
            clamp_action(BimanualAction(0.0, {"a": float("nan"), "b": 0.0}), previous, limits, 0.1)

    def test_missing_or_unexpected_joint_names_are_invalid(self):
        previous, limits = {"a": 0.0, "b": 0.0}, {"a": (-1.0, 1.0), "b": (-1.0, 1.0)}
        for targets in ({"a": 0.0}, {"a": 0.0, "b": 0.0, "nose": 0.0}):
            with self.subTest(targets=sorted(targets)), self.assertRaisesRegex(ValueError, "INVALID_ACTION"):
                clamp_action(BimanualAction(0.0, targets), previous, limits, 0.1)


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = MujocoSimulation()

    @classmethod
    def tearDownClass(cls):
        cls.sim.close()

    def test_idle_policy_triggers_failed_grasp_and_bounded_recovery(self):
        runner = EpisodeRunner(self.sim, IdlePolicy(), supervisor=True, stall_seconds=1.0, max_steps=600)
        log = runner.run(make_task(0))
        labels = [e["label"] for e in log.events]
        self.assertIn("FAILED_GRASP", labels)
        self.assertEqual(log.recoveries, 2)
        self.assertEqual((log.state, log.failure), ("FAILED", "RECOVERY_EXHAUSTED"))
        self.assertFalse(log.outcome["success"])

    def test_idle_policy_without_supervisor_times_out(self):
        runner = EpisodeRunner(self.sim, IdlePolicy(), supervisor=False, stall_seconds=1.0, max_steps=60)
        log = runner.run(make_task(0))
        self.assertIn("FAILED_GRASP", [e["label"] for e in log.events])
        self.assertEqual((log.state, log.failure, log.recoveries), ("FAILED", "TIMEOUT", 0))

    def test_recovery_motion_respects_the_step_budget(self):
        log = EpisodeRunner(self.sim, IdlePolicy(), supervisor=True, stall_seconds=0.05, max_steps=5).run(make_task(0))
        self.assertLessEqual(log.steps, 5)
        self.assertEqual((log.state, log.failure), ("FAILED", "TIMEOUT"))

    def test_task_limits_are_used_by_default(self):
        from dataclasses import replace
        task = replace(make_task(0), max_recoveries=0, timeout_s=3.0)
        log = EpisodeRunner(self.sim, IdlePolicy(), supervisor=True, stall_seconds=0.5).run(task)
        self.assertEqual((log.max_steps, log.max_recoveries, log.recoveries), (60, 0, 0))
        self.assertEqual(log.failure, "RECOVERY_EXHAUSTED")
        self.assertLessEqual(log.steps, 60)

    def test_fault_clock_runs_during_recovery_motion(self):
        glitch = GripperGlitch()
        glitch.reset(1)
        glitch.fired, glitch.arm, glitch._left = True, "right_arm", 7
        self.sim.reset(1)
        self.sim.set_actuator_fault("right_arm/gripper", glitch.open_value)
        runner = EpisodeRunner(self.sim, IdlePolicy(), supervisor=True, fault=glitch)
        log = EpisodeLog(1, {}, "", True, "GripperGlitch", max_steps=1000)
        runner._safe_pose(log, make_task(1))
        self.assertGreater(log.steps, 7)
        self.assertEqual(glitch._left, 0)
        self.assertIsNone(log.fault_step)  # recovery cannot start a new fault

    def test_a_policy_can_ask_for_recovery_instead_of_ending_the_episode(self):
        """The teacher raises FAILED_GRASP when the item is not in the hand it planned for."""
        class AsksForHelp(IdlePolicy):
            name = "asks_for_help"
            calls = 0

            def act(self, obs):
                type(self).calls += 1
                if type(self).calls == 5:
                    raise RuntimeError("FAILED_GRASP: the right hand is not holding the fork")
                return super().act(obs)

        log = EpisodeRunner(self.sim, AsksForHelp(), supervisor=True, stall_seconds=30.0,
                            max_steps=400).run(make_task(0))
        self.assertIn("FAILED_GRASP", [e["label"] for e in log.events])
        self.assertGreaterEqual(log.recoveries, 1)
        self.assertNotEqual(log.failure, "POLICY_ERROR")

    def test_an_unknown_policy_error_still_ends_the_episode(self):
        class Breaks(IdlePolicy):
            name = "breaks"

            def act(self, obs):
                raise RuntimeError("something unexpected")

        log = EpisodeRunner(self.sim, Breaks(), supervisor=True, max_steps=50).run(make_task(0))
        self.assertEqual((log.state, log.failure, log.recoveries), ("FAILED", "POLICY_ERROR", 0))

    def test_planning_failure_at_reset_is_a_logged_failure_not_a_crash(self):
        class CannotPlan(IdlePolicy):
            name = "cannot_plan"

            def reset(self, sim, task):
                raise RuntimeError("initial planning failed")

        log = EpisodeRunner(self.sim, CannotPlan(), supervisor=True, max_steps=50).run(make_task(0))
        self.assertEqual((log.state, log.failure, log.steps), ("FAILED", "POLICY_ERROR", 0))
        self.assertIn("initial planning failed", log.events[-1]["detail"])
        self.assertFalse(log.outcome["success"])

    def test_planning_failure_after_recovery_is_a_logged_failure_not_a_crash(self):
        class ReplanBreaks(IdlePolicy):
            name = "replan_breaks"

            def after_recovery(self, sim, task, progress):
                raise RuntimeError("planning failed after recovery")

        log = EpisodeRunner(self.sim, ReplanBreaks(), supervisor=True, stall_seconds=0.5,
                            max_steps=400).run(make_task(0))
        self.assertEqual((log.state, log.failure, log.recoveries), ("FAILED", "POLICY_ERROR", 1))
        self.assertIn("planning failed after recovery", log.events[-1]["detail"])

    def test_supervisor_recovers_from_gripper_glitch(self):
        task = make_task(1)
        failed = EpisodeRunner(self.sim, ScriptedPolicy(), supervisor=False, fault=GripperGlitch(),
                               max_steps=1500).run(task)
        self.assertEqual(failed.state, "FAILED")
        self.assertIn("OBJECT_DROPPED", [e["label"] for e in failed.events])
        recovered = EpisodeRunner(self.sim, ScriptedPolicy(), supervisor=True, fault=GripperGlitch(),
                                  max_steps=1500).run(task)
        self.assertEqual(recovered.state, "SUCCEEDED")
        self.assertEqual(recovered.recoveries, 1)
        self.assertTrue(recovered.outcome["success"])
        self.assertIsNotNone(recovered.fault_step)


if __name__ == "__main__":
    unittest.main()
