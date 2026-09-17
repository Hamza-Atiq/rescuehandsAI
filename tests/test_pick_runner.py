import unittest
from unittest.mock import patch

import mujoco

from rescuehandsai.contracts import BimanualAction
from rescuehandsai.pick_cells import make_pick_task
from rescuehandsai.pick_config import load_rules
from rescuehandsai.pick_outcome import PickJudge
from rescuehandsai.pick_records import InvalidRun
from rescuehandsai.pick_runner import PickEpisodeRunner
from rescuehandsai.sim import MujocoSimulation


class HoldHome:
    uses_privileged_state = False

    def __init__(self, images=False):
        self.images, self.sim = images, None

    def reset(self, sim, task):
        self.sim = sim

    def wants_images(self):
        return self.images

    def act(self, obs):
        return BimanualAction(obs.timestamp, dict(self.sim.home_targets))

    def metadata(self):
        return {"name": "hold_home", "backend": "python", "device": "cpu", "checkpoint": None}


class Raising(HoldHome):
    def act(self, obs):
        raise RuntimeError("tensor shape mismatch")


class NaNAction(HoldHome):
    def act(self, obs):
        targets = dict(self.sim.home_targets)
        targets["right_arm/gripper"] = float("nan")
        return BimanualAction(obs.timestamp, targets)


class BadQacc(HoldHome):
    def act(self, obs):
        self.sim.data.warning[int(mujoco.mjtWarning.mjWARN_BADQACC)].number += 1
        return super().act(obs)


class SuccessThenCollision(PickJudge):
    """Hold completes at physics step 3; a collision would follow at step 5 of the same control step."""

    def update(self, f):
        if self.succeeded:
            return super().update(f)  # raises: the runner must have stopped
        if f.physics_step == 3:
            self.success_at = self.hold_ok_at = (f.physics_step, f.control_step)
            return []
        if f.physics_step == 5:
            self._fail("ARM_ARM_CONTACT", f, "late collision")
            return ["ARM_ARM_CONTACT"]
        return []


class SevereAtFour(PickJudge):
    def update(self, f):
        if f.physics_step == 4:
            self._fail("ARM_ARM_CONTACT", f, "collision")
            return ["ARM_ARM_CONTACT"]
        if f.physics_step > 4:
            raise AssertionError("physics continued after a severe violation")
        return []


class PickRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = MujocoSimulation(physics_version=2)
        cls.rules = dict(load_rules(), deadline_control_steps=5)
        cls.task = make_pick_task(21, "F-B", "T1")  # a test seed outside every data/test block

    @classmethod
    def tearDownClass(cls):
        cls.sim.close()

    def run_policy(self, policy):
        return PickEpisodeRunner(self.sim, policy, rules=self.rules).run(self.task)

    def test_holding_home_is_a_valid_no_lift_failure_at_the_deadline(self):
        record = self.run_policy(HoldHome())
        self.assertFalse(record["success"])
        self.assertEqual(record["outcome"]["stop"], "deadline")
        self.assertEqual(record["outcome"]["first_failure"], "NO_LIFT")
        self.assertEqual(record["outcome"]["picked"], "neither")
        self.assertEqual((record["control_steps"], record["physics_steps"]), (5, 5 * self.sim.substeps))
        self.assertEqual(record["scene"]["slots"], {"spoon": 0, "fork": 1})
        self.assertEqual(len(record["scene_sha256"]), 64)

    def test_policy_exception_is_a_valid_policy_error(self):
        record = self.run_policy(Raising())
        self.assertEqual(record["outcome"]["first_failure"], "POLICY_ERROR")
        self.assertEqual(record["outcome"]["stop"], "early_stop")
        self.assertEqual(record["outcome"]["unevaluable_rules"], ["R2", "R5"])

    def test_non_finite_action_is_invalid_action(self):
        record = self.run_policy(NaNAction())
        self.assertEqual(record["outcome"]["first_failure"], "INVALID_ACTION")

    def test_renderer_failure_is_sim_error_even_during_a_policy_call(self):
        def broken(*args, **kwargs):
            raise RuntimeError("GL context lost")

        self.sim.render = broken  # instance attribute shadows the method for this test only
        try:
            with self.assertRaises(InvalidRun) as ctx:
                self.run_policy(HoldHome(images=True))
        finally:
            del self.sim.render
        self.assertEqual(ctx.exception.label, "SIM_ERROR")
        self.assertEqual(ctx.exception.partial["outcome"]["unevaluable_rules"], ["R1", "R2", "R3", "R4", "R5"])

    def test_bad_acceleration_warning_is_sim_error(self):
        with self.assertRaises(InvalidRun) as ctx:
            self.run_policy(BadQacc())
        self.assertEqual(ctx.exception.label, "SIM_ERROR")
        self.assertIn("mjWARN_BADQACC", ctx.exception.detail)

    def test_success_stops_at_its_physics_step_before_a_later_collision(self):
        with patch("rescuehandsai.pick_runner.PickJudge", SuccessThenCollision):
            record = self.run_policy(HoldHome())
        self.assertTrue(record["success"])
        self.assertEqual((record["physics_steps"], record["control_steps"]), (3, 1))
        self.assertEqual(record["outcome"]["failures"], [])
        self.assertEqual(record["outcome"]["success_physics_step"], 3)

    def test_severe_violation_stops_at_its_physics_step(self):
        with patch("rescuehandsai.pick_runner.PickJudge", SevereAtFour):
            record = self.run_policy(HoldHome())
        self.assertFalse(record["success"])
        self.assertEqual((record["physics_steps"], record["outcome"]["stop"]), (4, "early_stop"))
        self.assertEqual(record["outcome"]["first_failure"], "ARM_ARM_CONTACT")

    def test_physics_version_mismatch_is_a_contract_mismatch(self):
        v1 = MujocoSimulation()
        try:
            with self.assertRaises(InvalidRun) as ctx:
                PickEpisodeRunner(v1, HoldHome(), rules=self.rules)
            self.assertEqual(ctx.exception.label, "CONTRACT_MISMATCH")
        finally:
            v1.close()


if __name__ == "__main__":
    unittest.main()
