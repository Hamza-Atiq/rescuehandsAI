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
UTENSIL_MARGIN = 0.005
AIR_MARGIN = 0.008
# gripperframe sits at the fingertips; aim it below the handle centre so the jaw
# pads, not just the tips, close around the handle.
GRASP_DEPTH = 0.007
TABLE_CLEARANCE = 0.0015
# Jaw directions for the utensil (the wrist camera sits on the +closing side). The
# desired closing axis is the approach turned +90 degrees, so with these signs the
# right camera faces the robot at hand-off and the left camera faces away.
RIGHT_SIGN = 1.0
LEFT_SIGN = 1.0


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
    def _solve(self, arm, target, *, approach_xy=None, pitches=STEEP_PITCHES, q_init=None, closing_sign=None):
        for pitch in pitches:
            q = self.ik.solve(arm, target, pitch=pitch, approach_xy=approach_xy, q_init=q_init,
                              closing_sign=closing_sign)
            if q is not None:
                return q, pitch
        raise PlanningError(f"{arm} cannot reach {np.round(target, 3).tolist()}")

    def _pose(self, arm, target, pitch, approach_xy=None, q_init=None, closing_sign=None):
        q = self.ik.solve(arm, target, pitch=pitch, approach_xy=approach_xy, q_init=q_init,
                          closing_sign=closing_sign)
        if q is None:
            raise PlanningError(f"{arm} cannot reach {np.round(target, 3).tolist()} at pitch {pitch}")
        return q

    def _grasp_site(self, arm, center, half_width, *, approach_xy=None, pitches=STEEP_PITCHES, q_init=None,
                    margin=0.0, closing_sign=None):
        """IK for a grasp around `center` despite the offset fixed jaw.

        The fixed jaw is placed `margin` outside the object so it comes down beside
        it, not on top; closing the moving jaw then pushes the object onto it.
        """
        q, pitch = self._solve(arm, center, approach_xy=approach_xy, pitches=pitches, q_init=q_init,
                               closing_sign=closing_sign)
        _, finger, closing = self.ik.forward(arm, q)
        site = np.asarray(center) - (half_width + margin - FIXED_JAW_OFFSET) * closing
        sign = closing_sign
        if sign is None:  # keep the jaw direction the offset was computed for
            _, c = self.ik.desired_axes(arm, site, pitch, approach_xy=approach_xy)
            sign = 1.0 if np.dot(closing, c) >= 0 else -1.0
        q = self._pose(arm, site, pitch, approach_xy, q, closing_sign=sign)
        return q, site, finger, pitch

    def _grasp_with_clearance(self, arm, center, half_width, lift, *, approach_xy=None,
                              pitches=STEEP_PITCHES, q_init=None, margin=0.0, closing_sign=None):
        """Grasp pose plus a pose `lift` metres above it, both reachable at one pitch."""
        for pitch in pitches:
            try:
                q, site, finger, _ = self._grasp_site(arm, center, half_width, approach_xy=approach_xy,
                                                      pitches=(pitch,), q_init=q_init, margin=margin,
                                                      closing_sign=closing_sign)
                _, _, closing = self.ik.forward(arm, q)
                _, c = self.ik.desired_axes(arm, site, pitch, approach_xy=approach_xy)
                sign = 1.0 if np.dot(closing, c) >= 0 else -1.0
                q_up = self._pose(arm, site + (0, 0, lift), pitch, approach_xy, q, closing_sign=sign)
            except PlanningError:
                continue
            return q, q_up, site, finger, pitch
        raise PlanningError(f"{arm}: no pitch reaches {np.round(center, 3).tolist()} with {lift} m clearance")

    def _approach_pose(self, arm, site, finger, pitch, approach_xy, q_init, closing_sign=None):
        """A pose backed off along the fingers, preferring more clearance when reachable."""
        for back, up in ((0.05, 0.02), (0.05, 0.0), (0.035, 0.0), (0.025, 0.0)):
            q = self.ik.solve(arm, np.asarray(site) - back * finger + (0, 0, up), pitch=pitch,
                              approach_xy=approach_xy, q_init=q_init, closing_sign=closing_sign)
            if q is not None:
                return q
        raise PlanningError(f"{arm}: no approach pose near {np.round(site, 3).tolist()}")

    def _retreat_pose(self, arm, site, finger, pitch, approach_xy, q_init, closing_sign=None):
        """Back the open hand away along its fingers; fall back to smaller moves if needed."""
        options = ((0.05, 0.0), (0.035, 0.0), (0.02, 0.02), (0.0, 0.03), (0.02, 0.0))
        for back, up in options:
            target = np.asarray(site) - back * finger + (0, 0, up)
            for p in (pitch, 1.2, 1.4, 1.0):
                q = self.ik.solve(arm, target, pitch=p, approach_xy=approach_xy, q_init=q_init,
                                  closing_sign=closing_sign)
                if q is not None:
                    return q
        raise PlanningError(f"{arm}: no retreat pose near {np.round(site, 3).tolist()}")

    def _objects(self):
        return self.sim.privileged().objects

    # -- script ------------------------------------------------------------------
    def _script(self):
        yield Move({}, 4, "settle")
        for name in SUBTASKS:
            if name in self.subtasks:
                self.subtask = name
                yield from getattr(self, "_" + name)()
        self.subtask = "home"
        yield Move(self._home("right_arm") | self._home("left_arm"), 25, "home")

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
        q_back = self._retreat_pose(arm, site_place, finger_p, pitch_p, None, q_place)
        yield Move(q_back, 15, "cup_retreat")

    # -- utensil -----------------------------------------------------------------
    def _utensil_frame(self, item):
        body = self.sim.data.body(item)
        axis = body.xmat.reshape(3, 3)[:, 0].copy()   # handle end -> head
        axis[2] = 0.0
        return body.xpos.copy(), axis / np.linalg.norm(axis)

    def _handle_geometry(self, item):
        s = self.sim.scene_params.utensil_scale[item]
        hx, hy, hz = self.sim.scene_config["utensil"]["handle_half"]
        # body-frame x of the two grasp points on the handle
        return {"end": -hx * s + 0.02, "neck": hx * s - 0.014, "half_width": hy, "half_height": hz}

    def _pick_utensil(self):
        arm, grip, item = "right_arm", "right_arm/gripper", self.task.utensil
        g = self._handle_geometry(item)
        pos, axis = self._utensil_frame(item)
        center = pos + g["end"] * axis
        center[2] = max(TABLE_CLEARANCE, pos[2] - GRASP_DEPTH)
        q, q_lift, site, finger, pitch = self._grasp_with_clearance(
            arm, center, g["half_width"], 0.05, approach_xy=axis[:2], pitches=(1.3, 1.4, 1.2, 1.0),
            margin=UTENSIL_MARGIN, closing_sign=RIGHT_SIGN)
        self._right_pitch = pitch
        q_pre = self._approach_pose(arm, site, finger, pitch, axis[:2], q, RIGHT_SIGN)
        yield Move(q_pre | {grip: OPEN}, 25, "utensil_approach")
        yield Move(q, 18, "utensil_reach")
        yield Move({grip: CLOSE}, 12, "utensil_close")
        yield Move({}, 4, "utensil_squeeze")
        yield Move(q_lift, 18, "utensil_lift")
        self._q_right = q_lift

    def _handoff(self):
        right, left = "right_arm", "left_arm"
        rg, lg, item = "right_arm/gripper", "left_arm/gripper", self.task.utensil
        g = self._handle_geometry(item)
        hand = self.sim.scene_config["handoff"]
        pitch = getattr(self, "_right_pitch", 1.3)
        # Right hand carries the utensil flat with its head pointing at the left arm.
        head_dir = np.array([-1.0, 0.0])
        q_r, pitch_r = self._solve(right, hand["right_grasp"], approach_xy=head_dir, closing_sign=RIGHT_SIGN,
                                   pitches=(pitch, 1.3, 1.2, 1.4), q_init=getattr(self, "_q_right", None))
        q_r_high = self._pose(right, np.add(hand["right_grasp"], (0, 0, 0.03)), pitch_r, head_dir, q_r,
                              closing_sign=RIGHT_SIGN)
        yield Move(q_r_high, 30, "handoff_carry")
        yield Move(q_r, 15, "handoff_present")
        yield Move({}, 6, "handoff_settle")
        # Left grasp point comes from where the utensil really is now (physics may shift it).
        pos, axis = self._utensil_frame(item)
        center = pos + g["neck"] * axis
        center[2] -= GRASP_DEPTH
        approach = -axis[:2]   # left fingers point back towards the right hand
        q_l, q_l_up, site_l, finger_l, pitch_l = self._grasp_with_clearance(
            left, center, g["half_width"], 0.02, approach_xy=approach, pitches=(1.3, 1.2, 1.4, 1.0),
            margin=AIR_MARGIN, closing_sign=LEFT_SIGN)
        q_l_pre = self._approach_pose(left, site_l, finger_l, pitch_l, approach, q_l, LEFT_SIGN)
        yield Move(q_l_pre | {lg: OPEN}, 30, "handoff_left_approach")
        yield Move(q_l, 18, "handoff_left_reach")
        yield Move({lg: CLOSE}, 12, "handoff_left_close")
        yield Move({}, 5, "handoff_left_squeeze")
        # A normal opening: a very wide swing of the moving jaw knocks the utensil loose.
        yield Move({rg: OPEN}, 10, "handoff_right_release")
        yield Move({}, 4, "handoff_right_clear")
        _, finger_r, _ = self.ik.forward(right, q_r)
        site_r = np.asarray(hand["right_grasp"])
        q_r_up = q_r
        for up in (0.03, 0.02, 0.012):
            q = self.ik.solve(right, site_r + (0, 0, up), pitch=pitch_r, approach_xy=head_dir, q_init=q_r,
                              closing_sign=RIGHT_SIGN)
            if q is not None:
                q_r_up = q
                break
        yield Move(q_r_up, 12, "handoff_right_lift")
        q_r_back = self._retreat_pose(right, site_r, finger_r, pitch_r, head_dir, q_r_up, RIGHT_SIGN)
        yield Move(q_r_back, 15, "handoff_right_retreat")
        yield Move(self._home(right), 25, "right_home")
        self._left_pitch, self._q_left = pitch_l, q_l_up

    def _place_utensil(self):
        arm, grip, item = "left_arm", "left_arm/gripper", self.task.utensil
        g = self._handle_geometry(item)
        zone = self.sim.scene_config["zones"]["utensil_zone"]["pos"]
        approach = np.array([0.0, 1.0])  # utensil ends up lying along the table
        pitch = getattr(self, "_left_pitch", 1.3)
        yield Move(getattr(self, "_q_left", {}), 10, "utensil_lift_left")
        # Measure where the utensil really sits in the hand, then aim so its centre
        # lands on the zone (robust to slips and to how the hand-off went).
        site_pos, site_mat = self.sim.site_pose(f"{arm}/gripperframe")
        offset = site_mat.T @ (self.sim.data.body(item).xpos - site_pos)
        goal = np.array([zone[0], zone[1], g["half_height"] + 0.004])
        q_init = getattr(self, "_q_left", None)
        chosen = None
        for p in (pitch, 1.3, 1.2, 1.4):
            q = self.ik.solve(arm, goal, pitch=p, approach_xy=approach, q_init=q_init, closing_sign=LEFT_SIGN)
            if q is None:
                continue
            for _ in range(2):  # the hand orientation barely changes with a small shift
                self.ik.forward(arm, q)
                mat = self.ik.data.site_xmat[self.ik.model.site(f"{arm}/gripperframe").id].reshape(3, 3)
                site = goal - mat @ offset
                q = self.ik.solve(arm, site, pitch=p, approach_xy=approach, q_init=q, closing_sign=LEFT_SIGN)
                if q is None:
                    break
            if q is None:
                continue
            q_up = self.ik.solve(arm, site + (0, 0, 0.05), pitch=p, approach_xy=approach, q_init=q,
                                 closing_sign=LEFT_SIGN)
            if q_up is not None:
                _, finger, _ = self.ik.forward(arm, q)
                chosen = (q, q_up, site, finger, p)
                break
        if chosen is None:
            raise PlanningError(f"{arm}: cannot place {item} at {np.round(goal, 3).tolist()}")
        q, q_up, site, finger, pitch = chosen
        yield Move(q_up, 35, "utensil_carry")
        yield Move(q, 18, "utensil_lower")
        yield Move({grip: OPEN}, 10, "utensil_release")
        q_back = self._retreat_pose(arm, site, finger, pitch, approach, q, LEFT_SIGN)
        yield Move(q_back, 15, "utensil_retreat")
