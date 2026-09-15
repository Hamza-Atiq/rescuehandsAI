"""Scripted teacher: generates demonstrations with contact-based grasps.

It reads privileged object poses and uses IK. It exists to create training data
and a baseline; deployed control comes from the learned policy.
"""
import numpy as np

from .auditor import compute_facts
from .contracts import BimanualAction
from .kinematics import FIXED_JAW_OFFSET, IKSolver
from .motion import MotionFollower, Move
from .task import SUBTASKS, TaskSpec

OPEN = 0.9
CLOSE = -0.15
STEEP_PITCHES = (1.4, 1.2, 1.3, 1.0)


class PlanningError(RuntimeError):
    """The teacher could not find a reachable pose. Label: TARGET_MISSED."""


class ScriptedExpert:
    def __init__(self, sim, task: TaskSpec, subtasks=SUBTASKS):
        unknown = set(subtasks) - set(SUBTASKS)
        if unknown:
            raise ValueError(f"Unknown subtasks: {sorted(unknown)}")
        self.sim = sim
        self.task = task
        self.subtasks = tuple(subtasks)
        self.ik = IKSolver(sim.model)
        self.follower = MotionFollower(sim.previous, sim.config["max_command_delta"])
        self.subtask = None
        self.phase = "start"
        self.done = False
        self._move = None
        self._plan = self._script()

    # -- public ----------------------------------------------------------------
    def act(self, obs) -> BimanualAction:
        while self._move is None and not self.done:
            try:
                self._move = next(self._plan)
            except StopIteration:
                self.done = True
                break
            self.phase = self._move.label
            self.follower.begin(self._move)
        if self._move is not None and self.follower.advance(self._move):
            self._move = None
        return BimanualAction(obs.timestamp, dict(self.follower.targets))

    # -- geometry helpers --------------------------------------------------------
    def _solve(self, arm, target, *, approach_xy=None, pitches=STEEP_PITCHES, q_init=None):
        for pitch in pitches:
            q = self.ik.solve(arm, target, pitch=pitch, approach_xy=approach_xy, q_init=q_init)
            if q is not None:
                return q, pitch
        raise PlanningError(f"{arm} cannot reach {np.round(target, 3).tolist()}")

    def _pose(self, arm, target, pitch, approach_xy=None, q_init=None):
        q = self.ik.solve(arm, target, pitch=pitch, approach_xy=approach_xy, q_init=q_init)
        if q is None:
            raise PlanningError(f"{arm} cannot reach {np.round(target, 3).tolist()} at pitch {pitch}")
        return q

    def _grasp_site(self, arm, center, half_width, *, approach_xy=None, pitches=STEEP_PITCHES, q_init=None):
        """IK for a grasp whose jaws centre on `center` despite the offset fixed jaw."""
        q, pitch = self._solve(arm, center, approach_xy=approach_xy, pitches=pitches, q_init=q_init)
        _, finger, closing = self.ik.forward(arm, q)
        site = np.asarray(center) - (half_width - FIXED_JAW_OFFSET) * closing
        q = self._pose(arm, site, pitch, approach_xy, q)
        return q, site, finger, pitch

    def _grasp_with_clearance(self, arm, center, half_width, lift, *, approach_xy=None,
                              pitches=STEEP_PITCHES, q_init=None):
        """Grasp pose plus a pose `lift` metres above it, both reachable at one pitch."""
        for pitch in pitches:
            try:
                q, site, finger, _ = self._grasp_site(arm, center, half_width, approach_xy=approach_xy,
                                                      pitches=(pitch,), q_init=q_init)
                q_up = self._pose(arm, site + (0, 0, lift), pitch, approach_xy, q)
            except PlanningError:
                continue
            return q, q_up, site, finger, pitch
        raise PlanningError(f"{arm}: no pitch reaches {np.round(center, 3).tolist()} with {lift} m clearance")

    def _objects(self):
        return self.sim.privileged().objects

    # -- script ------------------------------------------------------------------
    def _script(self):
        yield Move({}, 4, "settle")
        if "place_cup" in self.subtasks:
            self.subtask = "place_cup"
            yield from self._place_cup()
        self.subtask = "home"
        yield Move(self._home("right_arm"), 20, "right_home")

    def _home(self, arm):
        return {n: v for n, v in self.sim.home_targets.items() if n.startswith(arm)}

    def _place_cup(self):
        arm, grip = "right_arm", "right_arm/gripper"
        sp = self.sim.scene_params
        r, hh = sp.cup_radius, sp.cup_half_height
        cup = np.array(self._objects()["cup"].position)
        grasp_z = min(hh, 0.035)
        q, q_lift, site, finger, pitch = self._grasp_with_clearance(arm, (cup[0], cup[1], grasp_z), r, 0.06)
        q_pre = self._pose(arm, site - 0.06 * finger, pitch, q_init=q)
        yield Move(q_pre | {grip: OPEN}, 25, "cup_approach")
        yield Move(q, 20, "cup_reach")
        yield Move({grip: CLOSE}, 12, "cup_close")
        yield Move({}, 4, "cup_squeeze")
        yield Move(q_lift, 20, "cup_lift")
        zone = self.sim.scene_config["zones"]["cup_zone"]["pos"]
        place_center = (zone[0], zone[1], grasp_z + 0.004)
        q_place, q_above, site_place, finger_p, pitch_p = self._grasp_with_clearance(
            arm, place_center, r, 0.06, pitches=(pitch,) + STEEP_PITCHES, q_init=q_lift)
        yield Move(q_above, 35, "cup_carry")
        yield Move(q_place, 20, "cup_lower")
        yield Move({grip: OPEN}, 10, "cup_release")
        q_back, _ = self._solve(arm, site_place - 0.05 * finger_p, q_init=q_place,
                                pitches=(pitch_p,) + STEEP_PITCHES)
        yield Move(q_back, 15, "cup_retreat")
