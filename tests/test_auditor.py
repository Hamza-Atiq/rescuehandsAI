"""Physics auditor facts and failure detection."""
import unittest

import mujoco

from rescuehandsai.auditor import AuditFacts, FailureMonitor, compute_facts
from rescuehandsai.contracts import BimanualAction
from rescuehandsai.sim import MujocoSimulation


def facts(time, held=None, supported=None, oob=(), cross=False):
    items = ("cup", "fork", "spoon")
    return AuditFacts(time=time,
                      held_by={i: set((held or {}).get(i, ())) for i in items},
                      touching={i: set((held or {}).get(i, ())) for i in items},
                      supported={i: (supported or {}).get(i, True) for i in items},
                      in_zone={i: None for i in items}, height={i: 0.01 for i in items},
                      speed={i: 0.0 for i in items}, out_of_bounds=set(oob),
                      cross_arm_contact=cross, up_z={i: 1.0 for i in items},
                      angular_speed={i: 0.0 for i in items})


class MonitorTests(unittest.TestCase):
    def test_drop_needs_prior_hold_and_debounce(self):
        m = FailureMonitor(debounce=3)
        expect = {"fork": "right_arm"}
        self.assertEqual(m.update(facts(0.0, {"fork": ["right_arm"]}, {"fork": False}), expect), [])
        lost = facts(0.05, {}, {"fork": False})
        self.assertEqual(m.update(lost, expect), [])
        self.assertEqual(m.update(facts(0.10, {}, {"fork": False}), expect), [])
        events = m.update(facts(0.15, {}, {"fork": False}), expect)
        self.assertEqual([e.label for e in events], ["OBJECT_DROPPED"])
        self.assertEqual((events[0].item, events[0].arm), ("fork", "right_arm"))
        # reported once, not every step
        self.assertEqual(m.update(facts(0.20, {}, {"fork": False}), expect), [])

    def test_grasp_that_never_forms_is_failed_grasp(self):
        m = FailureMonitor(debounce=2)
        expect = {"cup": "right_arm"}
        self.assertEqual(m.update(facts(0.0), expect), [])
        events = m.update(facts(0.05), expect)
        self.assertEqual([e.label for e in events], ["FAILED_GRASP"])

    def test_brief_flicker_is_not_a_drop(self):
        m = FailureMonitor(debounce=3)
        expect = {"cup": "right_arm"}
        m.update(facts(0.0, {"cup": ["right_arm"]}), expect)
        m.update(facts(0.05), expect)
        self.assertEqual(m.update(facts(0.10, {"cup": ["right_arm"]}), expect), [])
        self.assertEqual(m.update(facts(0.15, {"cup": ["right_arm"]}), expect), [])

    def test_collision_and_out_of_bounds_are_immediate(self):
        m = FailureMonitor()
        labels = [e.label for e in m.update(facts(0.0, oob={"spoon"}, cross=True), {})]
        self.assertEqual(sorted(labels), ["COLLISION", "OBJECT_OUT_OF_BOUNDS"])

    def test_reset_clears_hold_memory(self):
        m = FailureMonitor(debounce=1)
        m.update(facts(0.0, {"fork": ["left_arm"]}), {"fork": "left_arm"})
        m.reset_expectation("fork")
        self.assertEqual([e.label for e in m.update(facts(0.05), {"fork": "left_arm"})],
                         ["FAILED_GRASP"])


class ComputeFactsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = MujocoSimulation(seed=1)

    @classmethod
    def tearDownClass(cls):
        cls.sim.close()

    def settle(self, steps=10):
        for _ in range(steps):
            self.sim.step(BimanualAction(self.sim.observe().timestamp, self.sim.home_targets))

    def place(self, item, x, y, z):
        adr = self.sim.model.jnt_qposadr[self.sim.model.joint(item + "_free").id]
        self.sim.data.qpos[adr:adr + 3] = (x, y, z)
        mujoco.mj_forward(self.sim.model, self.sim.data)

    def test_resting_items_are_supported_not_held(self):
        self.sim.reset(1)
        self.settle()
        f = compute_facts(self.sim)
        for item in ("cup", "fork", "spoon"):
            self.assertTrue(f.supported[item], item)
            self.assertEqual(f.held_by[item], set())
        self.assertFalse(f.cross_arm_contact)
        self.assertEqual(f.out_of_bounds, set())

    def test_zone_membership_and_out_of_bounds(self):
        self.sim.reset(1)
        zx, zy = self.sim.scene_config["zones"]["cup_zone"]["pos"]
        self.place("cup", zx, zy, self.sim.scene_params.cup_half_height + 0.001)
        self.place("spoon", 0.0, -0.9, 0.01)
        self.settle()
        f = compute_facts(self.sim)
        self.assertEqual(f.in_zone["cup"], "cup_zone")
        self.assertIsNone(f.in_zone["fork"])
        self.assertIn("spoon", f.out_of_bounds)


if __name__ == "__main__":
    unittest.main()
