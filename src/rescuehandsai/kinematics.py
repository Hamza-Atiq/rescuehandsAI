"""Damped least-squares IK for the scripted teacher only.

Deployed control comes from the learned policy; this module exists to generate
demonstrations. It works on a private MjData and never changes live state.
"""
import mujoco
import numpy as np

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
DOWN = np.array([0.0, 0.0, -1.0])
# Measured on the SO-101 model: the fixed jaw sits 2 cm from gripperframe along
# the negative closing axis; the moving jaw opens towards the positive axis.
FIXED_JAW_OFFSET = 0.020


def hand_axes(site_xmat: np.ndarray):
    """Return (finger direction, jaw closing axis) of a gripperframe site."""
    return site_xmat[:, 0].copy(), site_xmat[:, 2].copy()


class IKSolver:
    def __init__(self, model: mujoco.MjModel, *, rot_weight=0.2, damping=0.01, seed=0):
        self.model = model
        self.data = mujoco.MjData(model)
        self.rot_weight = rot_weight
        self.damping = damping
        self._rng = np.random.default_rng(seed)
        self._arm = {}
        for arm in ("left_arm", "right_arm"):
            joints = [model.joint(f"{arm}/{j}") for j in ARM_JOINTS]
            self._arm[arm] = {
                "names": [f"{arm}/{j}" for j in ARM_JOINTS],
                "qpos": np.array([model.jnt_qposadr[j.id] for j in joints]),
                "dof": np.array([model.jnt_dofadr[j.id] for j in joints]),
                # Same limits as the simulator: joint range intersected with motor range.
                "low": np.array([max(model.jnt_range[j.id][0], model.actuator(j.name).ctrlrange[0])
                                 for j in joints]),
                "high": np.array([min(model.jnt_range[j.id][1], model.actuator(j.name).ctrlrange[1])
                                  for j in joints]),
                "site": model.site(f"{arm}/gripperframe").id,
            }

    def _set(self, arm, q):
        info = self._arm[arm]
        self.data.qpos[info["qpos"]] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        return self.data.site_xpos[info["site"]].copy(), self.data.site_xmat[info["site"]].reshape(3, 3)

    def forward(self, arm: str, q: dict):
        info = self._arm[arm]
        pos, mat = self._set(arm, np.array([q[n] for n in info["names"]]))
        finger, closing = hand_axes(mat)
        return pos, finger, closing

    def solve(self, arm: str, target, *, closing_xy=(1.0, 0.0), q_init: dict | None = None,
              tol=0.005, iterations=250):
        info = self._arm[arm]
        target = np.asarray(target, dtype=float)
        c = np.array([closing_xy[0], closing_xy[1], 0.0])
        c /= np.linalg.norm(c)
        starts = []
        if q_init is not None:
            starts.append(np.array([q_init[n] for n in info["names"]]))
        starts += [np.array(s, dtype=float) for s in
                   ([0, 0, 0, 1.2, 0], [0, -0.6, 0.9, 1.2, 0], [0, 0.6, -0.3, 1.2, 1.5],
                    [0, 0.6, -0.3, 1.2, -1.5])]
        starts += [self._rng.uniform(info["low"], info["high"]) for _ in range(4)]
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        best, best_err = None, np.inf
        for q in starts:
            q = np.clip(q, info["low"], info["high"])
            for _ in range(iterations):
                pos, mat = self._set(arm, q)
                finger, closing = hand_axes(mat)
                sign = 1.0 if np.dot(closing, c) >= 0 else -1.0
                e_pos = target - pos
                e_rot = np.cross(finger, DOWN) + np.cross(closing, sign * c)
                if np.linalg.norm(e_pos) < tol * 0.3 and np.linalg.norm(e_rot) < 0.01:
                    break
                mujoco.mj_jacSite(self.model, self.data, jacp, jacr, info["site"])
                J = np.vstack([jacp[:, info["dof"]], self.rot_weight * jacr[:, info["dof"]]])
                e = np.concatenate([e_pos, self.rot_weight * e_rot])
                dq = J.T @ np.linalg.solve(J @ J.T + self.damping ** 2 * np.eye(6), e)
                norm = np.linalg.norm(dq)
                if norm > 0.25:
                    dq *= 0.25 / norm
                q = np.clip(q + dq, info["low"], info["high"])
            pos, mat = self._set(arm, q)
            finger, closing = hand_axes(mat)
            err = np.linalg.norm(target - pos)
            ok = err < tol and -finger[2] > 0.97 and abs(np.dot(closing, c)) > 0.95
            if ok and err < best_err:
                best, best_err = q.copy(), err
                if err < tol * 0.3:
                    break
        if best is None:
            return None
        return dict(zip(info["names"], map(float, best)))
