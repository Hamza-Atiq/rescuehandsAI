# tests/test_pick_expert.py
import unittest

from rescuehandsai.expert import PlanningError, ScriptedExpert
from rescuehandsai.motion import Move
from rescuehandsai.pick_cells import cell_params, make_pick_task
from rescuehandsai.pick_clearance import Clearance, ClearanceChecker
from rescuehandsai.pick_config import load_rules
from rescuehandsai.pick_expert import (MIN_TABLE_CLEARANCE_M, NoClearGraspError, PickExpert,
                                       StagingBlockedError)
from rescuehandsai.pick_runner import PickEpisodeRunner
from rescuehandsai.pick_teacher import PickTeacher
from rescuehandsai.scene import sample_params
from rescuehandsai.sim import MujocoSimulation
from rescuehandsai.task import TaskSpec


def reset_like_runner(sim, seed, cell):
    task = make_pick_task(seed, cell, "T1")
    sim.reset(seed, instruction=task.instruction,
              params=cell_params(sample_params(sim.scene_config, seed), cell, sim.scene_config))
    return task


def spec_for(task):
    return TaskSpec(task_id="t", instruction=task.instruction, utensil=task.utensil, seed=task.seed)


def right_arm_unreachable(expert):
    """Make every right-arm grasp plan fail, as when the utensil is out of the right arm's range."""
    original = expert._grasp_with_clearance

    def patched(arm, *args, **kwargs):
        if arm == "right_arm":
            raise PlanningError("right_arm cannot reach (forced by test)")
        return original(arm, *args, **kwargs)
    expert._grasp_with_clearance = patched


def first_labels(expert, limit=6, stop_at=None):
    """Collect up to `limit` move labels.

    With `stop_at` given: stop as soon as that label is collected (so a test double's own
    PlanningError raised right after yielding it is never even reached), but still
    propagate a PlanningError raised BEFORE `stop_at` is seen (so tests that expect a
    specific failure, without a `stop_at`, keep seeing it -- the default `stop_at=None`
    reproduces the old always-raise behaviour exactly).
    """
    labels = []
    for _ in range(limit):
        try:
            move = next(expert._plan)
        except PlanningError:
            if stop_at is not None and stop_at in labels:
                break
            raise
        labels.append(move.label)
        if stop_at is not None and move.label == stop_at:
            break
    return labels


class PickExpertPlanningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = MujocoSimulation(physics_version=2)

    @classmethod
    def tearDownClass(cls):
        cls.sim.close()

    def test_every_leg_of_the_chosen_grasp_clears_the_table_by_one_millimetre(self):
        for seed in (3100000, 3100001):
            for cell in ("F-A", "S-B"):
                with self.subTest(seed=seed, cell=cell):
                    task = reset_like_runner(self.sim, seed, cell)
                    expert = PickExpert(self.sim, spec_for(task))
                    first_labels(expert, 3)          # the grasp is planned before utensil_approach is yielded
                    plan = expert.grasp_plans[-1]
                    self.assertGreaterEqual(plan["min_clearance_m"], MIN_TABLE_CLEARANCE_M)
                    self.assertEqual(set(plan["legs"]),
                                     {"approach", "reach", "close", "lift", "regrasp_open", "regrasp_back_off"})
                    for leg, detail in plan["legs"].items():
                        self.assertGreaterEqual(detail["z_m"], MIN_TABLE_CLEARANCE_M, leg)
                    self.assertGreater(plan["center_z_m"], 0.0015)   # higher than the old 1.5 mm

    def test_clearance_is_rechecked_independently_on_the_solved_reach_pose(self):
        task = reset_like_runner(self.sim, 3100000, "F-A")
        expert = PickExpert(self.sim, spec_for(task))
        # settle, an optional home_start, approach, reach: four moves always include the reach
        moves = [next(expert._plan) for _ in range(4)]
        reach = next(m for m in moves if m.label == "utensil_reach")
        checker = ClearanceChecker(self.sim.model, "right_arm")
        targets = dict(self.sim.previous) | reach.goal | {"right_arm/gripper": 0.9}
        self.assertGreaterEqual(checker.lowest(self.sim.data.qpos, targets).z, MIN_TABLE_CLEARANCE_M)

    def test_no_clear_height_is_a_named_failure_not_a_staging_attempt(self):
        task = reset_like_runner(self.sim, 3100000, "F-A")
        expert = PickExpert(self.sim, spec_for(task))
        expert.clearance.along = lambda *a, **k: Clearance(-0.01, "geom_93", 1.0)
        with self.assertRaises(NoClearGraspError) as caught:
            first_labels(expert, 3)
        self.assertIn("above the table", str(caught.exception))
        self.assertNotIsInstance(caught.exception, StagingBlockedError)

    def test_pick_expert_blocks_left_arm_staging_with_a_clear_reason(self):
        task = reset_like_runner(self.sim, 3100000, "F-A")
        expert = PickExpert(self.sim, spec_for(task))
        right_arm_unreachable(expert)
        with self.assertRaises(StagingBlockedError) as caught:
            first_labels(expert, 3)
        self.assertIn("left-arm staging is disabled", str(caught.exception))

    def test_full_task_expert_still_stages_with_the_left_arm(self):
        """Preservation: the unchanged ScriptedExpert still calls _stage_utensil."""
        task = reset_like_runner(self.sim, 3100000, "F-A")
        expert = ScriptedExpert(self.sim, spec_for(task), subtasks=("pick_utensil",))
        right_arm_unreachable(expert)

        def spy():
            yield Move({}, 1, "stage_spy")
            raise PlanningError("spy stops here")
        expert._stage_utensil = spy
        # settle, an optional home_start, then stage_spy: allow room for either case and
        # stop as soon as stage_spy is reached (never even resuming the spy past its yield).
        self.assertIn("stage_spy", first_labels(expert, 5, stop_at="stage_spy"))


class PickTeacherUsesPickExpertTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = MujocoSimulation(physics_version=2)
        cls.rules = load_rules()

    @classmethod
    def tearDownClass(cls):
        cls.sim.close()

    def test_teacher_builds_a_pick_expert(self):
        task = reset_like_runner(self.sim, 3100000, "F-A")
        teacher = PickTeacher()
        teacher.reset(self.sim, task)
        self.assertIsInstance(teacher.expert, PickExpert)

    def test_blocked_staging_ends_the_episode_as_a_policy_error_and_no_staging_move_starts(self):
        phases = []

        class ForcedFallback(PickTeacher):
            def reset(self, sim, task):
                super().reset(sim, task)
                right_arm_unreachable(self.expert)

            def act(self, obs):
                try:
                    return super().act(obs)
                finally:
                    phases.append(self.expert.phase)

        record = PickEpisodeRunner(self.sim, ForcedFallback(), rules=self.rules).run(
            make_pick_task(3100000, "F-A", "T1"))
        self.assertEqual(record["outcome"]["first_failure"], "POLICY_ERROR")
        detail = record["outcome"]["failures"][0]["detail"]
        self.assertIn("StagingBlockedError", detail)
        self.assertIn("left-arm staging is disabled", detail)
        # home_start may move both arms to home; a left-arm grasp would start a "stage_" move.
        self.assertFalse([p for p in phases if str(p).startswith("stage_")], phases)


if __name__ == "__main__":
    unittest.main()
