"""Episode runner: supervisor recovery, watchdog and action guard."""
import unittest

from rescuehandsai.contracts import BimanualAction
from rescuehandsai.perturb import GripperGlitch
from rescuehandsai.policies.scripted import ScriptedPolicy
from rescuehandsai.runner import EpisodeRunner, clamp_action
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
