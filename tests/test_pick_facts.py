import unittest

import mujoco
import numpy as np

from rescuehandsai.contracts import BimanualAction
from rescuehandsai.pick_config import load_contacts
from rescuehandsai.pick_contacts import ContactClassifier
from rescuehandsai.pick_facts import FactReader, SimulatorFailure
from rescuehandsai.sim import MujocoSimulation


class FactReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = MujocoSimulation(physics_version=2)

    @classmethod
    def tearDownClass(cls):
        cls.sim.close()

    def setUp(self):
        self.sim.reset(21)
        self.reader = FactReader(self.sim, ContactClassifier(self.sim.model, "fork", load_contacts()))

    def hold(self, **kwargs):
        self.sim.step(BimanualAction(self.sim.observe().timestamp, self.sim.home_targets), **kwargs)

    def test_read_returns_poses_speeds_and_gripper_frame(self):
        f = self.reader.read(0, 0)
        self.assertEqual(set(f.positions), {"cup", "fork", "spoon"})
        self.assertEqual(f.rotations["fork"].shape, (3, 3))
        np.testing.assert_allclose(f.gripper_pos, self.sim.data.site("right_arm/gripperframe").xpos)
        self.assertEqual(f.linear_speed["fork"], 0.0)

    def test_hook_runs_after_every_physics_step(self):
        times = []
        taken = self.sim.step(BimanualAction(self.sim.observe().timestamp, self.sim.home_targets),
                              on_substep=lambda: times.append(float(self.sim.data.time)))
        self.assertEqual((len(times), taken), (self.sim.substeps, self.sim.substeps))
        self.assertTrue(all(b > a for a, b in zip(times, times[1:])))

    def test_hook_can_stop_at_the_triggering_physics_step(self):
        calls = []

        def stop_at_third():
            calls.append(float(self.sim.data.time))
            return len(calls) == 3

        start = float(self.sim.data.time)
        taken = self.sim.step(BimanualAction(self.sim.observe().timestamp, self.sim.home_targets),
                              on_substep=stop_at_third)
        self.assertEqual((taken, len(calls)), (3, 3))
        self.assertAlmostEqual(float(self.sim.data.time) - start, 3 * self.sim.config["physics_dt"])

    def test_brief_contact_between_control_updates_is_caught(self):
        model, data = self.sim.model, self.sim.data
        # MuJoCo 3.13.0 caches a per-body midphase structure at compile time; a post-compile
        # geom_pos write (below) needs this disabled or it produces no contact at all.
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MIDPHASE
        plate, pad = model.geom("plate").id, model.geom("right_arm/fixed_jaw_box5").id
        home = model.geom_pos[plate].copy()
        seen, count = [], [0]

        def hook():
            # After physics step 3 only, the plate touches the jaw; it is moved back before the
            # next physics step, so no step ever integrates that contact and the end of the
            # control step shows nothing. Only a per-physics-step read can see it.
            count[0] += 1
            if count[0] == 3:
                model.geom_pos[plate] = data.geom_xpos[pad].copy()
                mujoco.mj_forward(model, data)
            f = self.reader.read(count[0], 1)
            seen.append((count[0], [v.labels for v in f.verdicts if "plate" in v.geoms]))
            if count[0] == 3:
                model.geom_pos[plate] = home
                mujoco.mj_forward(model, data)

        self.hold(on_substep=hook)
        flagged = [step for step, labels in seen if ("FORBIDDEN_CONTACT",) in labels]
        self.assertEqual(flagged, [3])
        after = self.reader.read(99, 1)
        self.assertFalse([v for v in after.verdicts if "plate" in v.geoms])

    def test_sim_error_warnings_raise_and_other_warnings_are_recorded(self):
        self.sim.data.warning[int(mujoco.mjtWarning.mjWARN_INERTIA)].number += 2
        self.reader.read(1, 1)
        self.assertEqual(self.reader.other_warnings, {"mjWARN_INERTIA": 2})
        self.sim.data.warning[int(mujoco.mjtWarning.mjWARN_BADQACC)].number += 1
        with self.assertRaises(SimulatorFailure):
            self.reader.read(2, 1)

    def test_default_step_behaviour_is_unchanged(self):
        before = self.sim.contact_samples
        self.hold()
        self.assertGreaterEqual(self.sim.contact_samples, before)


if __name__ == "__main__":
    unittest.main()
