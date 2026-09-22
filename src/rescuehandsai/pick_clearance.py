"""How far the right hand's collision shapes stay above the table on a commanded path.

The full-task expert checks one point (the gripperframe site) against the table. The
jaw meshes reach several millimetres past that point, so a pose can look clear and
still sit inside the table (docs/research/2026-09-22-collision-inspection/FINDINGS.md).
This checker looks at every collidable shape of the hand. It uses its own MjData and
kinematics only: it never steps physics and never changes the live simulation.
"""
import itertools
from dataclasses import dataclass

import mujoco
import numpy as np

HAND_BODIES = ("wrist", "gripper", "camera_mount", "moving_jaw_so101_v1")


# Sampling on a path: start with at most 0.005 rad of joint change between checked poses,
# then add poses until no hand shape point moves more than 1 mm between neighbours.
MAX_JOINT_STEP_RAD = 0.005
MAX_POINT_STEP_M = 0.001


@dataclass(frozen=True)
class Clearance:
    """Predicted clearance of the COMMANDED pose or path: a planning margin, not proof.

    The physical hand lags its command and is pushed by contacts, so only a physics
    measurement shows the real gap.
    """
    z: float                        # lowest world height of any checked shape (table top is z = 0)
    shape: str                      # the shape that is lowest
    fraction: float                 # where on the path (0 = start, 1 = goal)
    samples: int = 1                # poses checked
    max_point_step_m: float = 0.0   # largest move of any shape point between neighbouring samples


def _local_points(model, g) -> np.ndarray:
    t, s = model.geom_type[g], model.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        mid = model.geom_dataid[g]
        a, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        return model.mesh_vert[a:a + n].astype(float)
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        return np.array(list(itertools.product(*[(-x, x) for x in s[:3]])), dtype=float)
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        r = s[0]
        return np.array([[0, 0, -r], [0, 0, r], [r, 0, 0], [-r, 0, 0], [0, r, 0], [0, -r, 0]], dtype=float)
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        r, h = s[0], s[1]
        ring = [[r, 0], [-r, 0], [0, r], [0, -r]]
        return np.array([[0, 0, -h - r], [0, 0, h + r]] + [[x, y, z] for z in (-h, h) for x, y in ring], dtype=float)
    raise ValueError(f"unsupported collision shape type {int(t)} for geom {g}")


class ClearanceChecker:
    def __init__(self, model, arm: str = "right_arm"):
        self.model = model
        self.arm = arm
        self._data = mujoco.MjData(model)
        bodies = {model.body(f"{arm}/{b}").id for b in HAND_BODIES}
        self._geoms = [g for g in range(model.ngeom)
                       if model.geom_bodyid[g] in bodies and (model.geom_contype[g] or model.geom_conaffinity[g])]
        self._points = {g: _local_points(model, g) for g in self._geoms}
        self.shapes = tuple(model.geom(g).name or f"geom_{g}" for g in self._geoms)
        self._site = model.site(f"{arm}/gripperframe").id

    def _world_points(self) -> np.ndarray:
        d = self._data
        return np.concatenate([self._points[g] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
                               for g in self._geoms])

    def _pose(self, qpos, targets: dict):
        d = self._data
        d.qpos[:] = qpos
        for name, value in targets.items():
            d.qpos[self.model.jnt_qposadr[self.model.joint(name).id]] = value
        mujoco.mj_kinematics(self.model, d)

    def lowest(self, qpos, targets: dict) -> Clearance:
        self._pose(qpos, targets)
        best_z, best_g = np.inf, None
        for g in self._geoms:
            R = self._data.geom_xmat[g].reshape(3, 3)
            z = float((self._points[g] @ R.T + self._data.geom_xpos[g])[:, 2].min())
            if z < best_z:
                best_z, best_g = z, g
        return Clearance(best_z, self.model.geom(best_g).name or f"geom_{best_g}", 1.0)

    def along(self, qpos, start: dict, goal: dict, max_point_step: float = MAX_POINT_STEP_M) -> Clearance:
        """Lowest point on the straight joint-space line from start to goal (MotionFollower's path).

        Starts with at most MAX_JOINT_STEP_RAD of joint change between checked poses, then
        adds poses until no hand shape point moves more than `max_point_step` between
        neighbours, so a contact between two checked poses cannot hide in a big gap.
        """
        if max_point_step <= 0:
            raise ValueError("max_point_step must be positive")
        names = sorted(set(start) | set(goal))
        a = np.array([start.get(n, goal.get(n)) for n in names], dtype=float)
        b = np.array([goal.get(n, start.get(n)) for n in names], dtype=float)
        samples = max(2, int(np.ceil(float(np.max(np.abs(b - a), initial=0.0)) / MAX_JOINT_STEP_RAD)) + 1)
        while True:
            worst, previous, step = None, None, 0.0
            for f in np.linspace(0.0, 1.0, samples):
                c = self.lowest(qpos, dict(zip(names, a + (b - a) * f)))
                points = self._world_points()
                if previous is not None:
                    step = max(step, float(np.linalg.norm(points - previous, axis=1).max()))
                previous = points
                if worst is None or c.z < worst[0]:
                    worst = (c.z, c.shape, float(f))
            if step <= max_point_step:
                return Clearance(worst[0], worst[1], worst[2], samples, step)
            if samples > 100_000:
                raise RuntimeError(f"path sampling did not reach {max_point_step} m spacing")
            samples = (samples - 1) * 2 + 1

    def site_z(self, qpos, targets: dict) -> float:
        self._pose(qpos, targets)
        return float(self._data.site_xpos[self._site][2])
