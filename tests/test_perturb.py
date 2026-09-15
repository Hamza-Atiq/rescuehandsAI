"""Fault injection produces a real, detected drop and is seeded."""
import unittest

from rescuehandsai.auditor import FailureMonitor, compute_facts
from rescuehandsai.contracts import BimanualAction
from rescuehandsai.expert import ScriptedExpert
from rescuehandsai.perturb import GripperGlitch
from rescuehandsai.sim import MujocoSimulation
from rescuehandsai.task import make_task


class FaultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = MujocoSimulation()

    @classmethod
    def tearDownClass(cls):
        cls.sim.close()

    def test_fault_overrides_motor_but_not_command_history(self):
        self.sim.reset(0)
        self.sim.set_actuator_fault("right_arm/gripper", 1.0)
        for _ in range(15):
            self.sim.step(BimanualAction(self.sim.observe().timestamp, self.sim.home_targets))
        self.assertGreater(self.sim.observe().positions["right_arm/gripper"], 0.8)
        self.assertEqual(self.sim.previous, self.sim.home_targets)
        self.sim.set_actuator_fault("right_arm/gripper", None)
        with self.assertRaises(KeyError):
            self.sim.set_actuator_fault("nose", 0.0)

    def test_glitch_drops_held_utensil_and_monitor_reports_it(self):
        seed = 1
        task = make_task(seed)
        self.sim.reset(seed, task.instruction)
        expert = ScriptedExpert(self.sim, task, subtasks=("pick_utensil", "handoff"))
        glitch, monitor = GripperGlitch(), FailureMonitor()
        glitch.reset(seed)
        facts, events, fired, step = None, [], None, 0
        while step < 400 and (fired is None or step < fired + 20):
            if glitch.before_step(self.sim, facts, task, step):
                fired = step
            self.sim.step(expert.act(self.sim.observe()))
            facts = compute_facts(self.sim)
            expect = {task.utensil: "right_arm"} if expert.phase in ("utensil_lift", "handoff_carry") else {}
            events += monitor.update(facts, expect)
            step += 1
        self.assertIsNotNone(fired)
        self.assertEqual(facts.held_by[task.utensil], set())
        self.assertIn(("OBJECT_DROPPED", task.utensil, "right_arm"), [(e.label, e.item, e.arm) for e in events])

    def test_glitch_timing_is_seeded(self):
        a, b = GripperGlitch(), GripperGlitch()
        a.reset(3)
        b.reset(3)
        self.assertEqual(a.delay_steps, b.delay_steps)


if __name__ == "__main__":
    unittest.main()
