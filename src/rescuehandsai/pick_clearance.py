"""How far the right hand's collision shapes stay above the table on a commanded path.

The full-task expert checks one point (the gripperframe site) against the table. The
jaw meshes reach several millimetres past that point, so a pose can look clear and
still sit inside the table (docs/research/2026-09-22-collision-inspection/FINDINGS.md).
This checker looks at every collidable shape of the hand. It uses its own MjData and
kinematics only: it never steps physics and never changes the live simulation.

`along()`'s reported height is a PROVEN lower bound on the true minimum along the
commanded path (see its docstring), not merely a sampled minimum. It is still only a
margin on the COMMANDED joint-space path -- the physical hand lags its command and is
pushed by contacts, so only a physics measurement shows the real gap.
"""
import itertools
from dataclasses import dataclass

import mujoco
import numpy as np

HAND_BODIES = ("wrist", "gripper", "camera_mount", "moving_jaw_so101_v1")


# along() picks enough samples that the proven dip bound (see _joint_radii and along())
# is at most this many metres.
MAX_DIP_BOUND_M = 0.00025


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
    max_point_step_m: float = 0.0   # largest MEASURED move of any shape point between neighbours
    dip_bound_m: float = 0.0        # PROVEN max dip below the lower sampled end (0.0 for lowest())


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


def _local_extent(model, g) -> float:
    """Max distance from geom `g`'s BODY-frame origin to any point on the geom.

    Exact for spheres (radius `r`) and capsules (`h + r`, the tip of the rounded cap --
    not the coarse ring samples `_local_points` uses for the unrelated lowest-z check,
    which would underestimate this). For mesh/box, the max vertex norm: exact, because a
    convex combination of vertices (any surface point) never has a larger norm than the
    largest vertex.
    """
    t, s = model.geom_type[g], model.geom_size[g]
    pos_norm = float(np.linalg.norm(model.geom_pos[g]))
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return pos_norm + float(s[0])
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return pos_norm + float(s[0]) + float(s[1])
    return pos_norm + float(np.max(np.linalg.norm(_local_points(model, g), axis=1)))


def _ancestor_offset(model, ancestor_body: int, body: int):
    """Whether `ancestor_body` is `body` itself or a strict ancestor of it, and the path cost.

    Returns `(True, offset)` where `offset` is the sum of `|body_pos[k]|` for every body
    `k` strictly below `ancestor_body` down to `body` (inclusive) -- a bound on the
    world-frame distance from `ancestor_body`'s frame origin to `body`'s frame origin,
    valid because a rotation never changes a vector's length (triangle inequality along
    the kinematic chain). Returns `(False, 0.0)` if the chain reaches the world body
    without ever reaching `ancestor_body`.
    """
    offset = 0.0
    k = body
    while True:
        if k == ancestor_body:
            return True, offset
        offset += float(np.linalg.norm(model.body_pos[k]))
        if k == 0:
            return False, 0.0
        k = int(model.body_parentid[k])


class ClearanceChecker:
    def __init__(self, model, arm: str = "right_arm"):
        self.model = model
        self.arm = arm
        self._data = mujoco.MjData(model)
        bodies = {model.body(f"{arm}/{b}").id for b in HAND_BODIES}
        self._geoms = [g for g in range(model.ngeom)
                       if model.geom_bodyid[g] in bodies and (model.geom_contype[g] or model.geom_conaffinity[g])]
        self._points = {g: _local_points(model, g) for g in self._geoms}
        self._extent = {g: _local_extent(model, g) for g in self._geoms}
        self.shapes = tuple(model.geom(g).name or f"geom_{g}" for g in self._geoms)
        self._site = model.site(f"{arm}/gripperframe").id
        self._radius = self._joint_radii()

    def _joint_radii(self) -> dict:
        """Per-joint bound R_j >= distance from joint j's rotation axis to any checked point.

        Configuration-independent: built once from body offsets and joint anchors (each
        in its own parent-relative local frame, summed via the triangle inequality,
        since rotations preserve vector length). A checked shape point moved only by
        joint j's rotation travels at a speed of at most |q_dot_j| * R_j; see `along()`.
        A joint that is not an ancestor of any checked shape's body gets R_j = 0 for it
        (it cannot move that shape at all); a joint's overall R_j is the max over the
        shapes it can move.
        """
        radii = {}
        for j in range(self.model.njnt):
            if self.model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            name = self.model.joint(j).name
            body_j = int(self.model.jnt_bodyid[j])
            anchor = float(np.linalg.norm(self.model.jnt_pos[j]))
            best = 0.0
            for g in self._geoms:
                is_ancestor, offset = _ancestor_offset(self.model, body_j, int(self.model.geom_bodyid[g]))
                if not is_ancestor:
                    continue
                best = max(best, anchor + offset + self._extent[g])
            radii[name] = best
        return radii

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

    def _shape_lowest_z(self, g) -> float:
        """Exact lowest world z of collision geom `g` at the checker's current internal pose.

        Spheres and capsules are round: sampling a handful of surface points (as the box
        and mesh shapes do, exactly, since a linear functional's minimum over a polytope
        is always at a vertex) sits up to a fraction of a millimetre ABOVE the true lowest
        point once the shape is rotated. Spheres and capsules use the closed-form minimum
        instead; boxes and meshes stay vertex-based (exact for those convex/flat shapes).
        """
        d = self._data
        t = self.model.geom_type[g]
        if t == mujoco.mjtGeom.mjGEOM_SPHERE:
            r = float(self.model.geom_size[g][0])
            return float(d.geom_xpos[g][2] - r)
        if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            r, h = float(self.model.geom_size[g][0]), float(self.model.geom_size[g][1])
            R = d.geom_xmat[g].reshape(3, 3)
            c = d.geom_xpos[g]
            return float(min(c[2] + h * R[2, 2], c[2] - h * R[2, 2]) - r)
        R = d.geom_xmat[g].reshape(3, 3)
        return float((self._points[g] @ R.T + d.geom_xpos[g])[:, 2].min())

    def lowest(self, qpos, targets: dict) -> Clearance:
        self._pose(qpos, targets)
        best_z, best_g = np.inf, None
        for g in self._geoms:
            z = self._shape_lowest_z(g)
            if z < best_z:
                best_z, best_g = z, g
        return Clearance(best_z, self.model.geom(best_g).name or f"geom_{best_g}", 1.0)

    def along(self, qpos, start: dict, goal: dict, max_dip_bound: float = MAX_DIP_BOUND_M) -> Clearance:
        """Lowest point on the straight joint-space line from start to goal (MotionFollower's path).

        The reported `z` is a PROVEN lower bound on the true minimum along this path, not
        just the sampled minimum. Between two adjacent sampled poses, a checked shape
        point moved by joint `j`'s change in angle `dq_j` travels at most `|dq_j| * R_j`
        (its distance from that joint's axis, bounded by `_joint_radii`); summed over all
        moving joints, its straight-line path length between the two samples is at most
        `L = sum_j |dq_j| * R_j`, so it can dip at most `L / 2` below whichever of the two
        sampled heights is lower. `samples` is chosen so this proven dip bound is at most
        `max_dip_bound` (default 0.25 mm) over the WHOLE path (so `L/2` per interval, with
        `samples - 1` intervals). This is still only a margin on the COMMANDED joint-space
        path, not proof about the physical hand.
        """
        if max_dip_bound <= 0:
            raise ValueError("max_dip_bound must be positive")
        names = sorted(set(start) | set(goal))
        a = np.array([start.get(n, goal.get(n)) for n in names], dtype=float)
        b = np.array([goal.get(n, start.get(n)) for n in names], dtype=float)
        weighted_range = float(sum(abs(b[i] - a[i]) * self._radius[n] for i, n in enumerate(names)))
        samples = max(2, int(np.ceil(weighted_range / (2 * max_dip_bound))) + 1)
        if samples > 100_000:
            raise RuntimeError(f"path would need {samples} samples to reach a {max_dip_bound} m dip bound")
        dip_bound = weighted_range / (2 * (samples - 1))
        worst, previous, max_step = None, None, 0.0
        for f in np.linspace(0.0, 1.0, samples):
            c = self.lowest(qpos, dict(zip(names, a + (b - a) * f)))
            points = self._world_points()
            if previous is not None:
                max_step = max(max_step, float(np.linalg.norm(points - previous, axis=1).max()))
            previous = points
            if worst is None or c.z < worst[0]:
                worst = (c.z, c.shape, float(f))
        return Clearance(worst[0] - dip_bound, worst[1], worst[2], samples, max_step, dip_bound)

    def site_z(self, qpos, targets: dict) -> float:
        self._pose(qpos, targets)
        return float(self._data.site_xpos[self._site][2])
