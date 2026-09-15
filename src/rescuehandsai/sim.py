"""MuJoCo owns all physical state. No object attachment or pose rewrites in step."""
import json
import math
from pathlib import Path
import mujoco
import numpy as np

from .contracts import BimanualAction, Observation, PrivilegedState
from .control import validate_action

ROOT = Path(__file__).resolve().parents[2]
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
ARMS = ("left_arm", "right_arm")

WORLD = """<mujoco model="rescuehands_dinner_foundation">
  <option integrator="implicitfast" timestep="0.005" cone="elliptic" iterations="10" ls_iterations="20" impratio="10"/>
  <visual><global offwidth="640" offheight="480"/></visual>
  <worldbody>
    <light pos="0 -0.4 1.5" dir="0 0 -1"/>
    <light pos="0.5 0.5 1.0" dir="-0.3 -0.3 -1"/>
    <geom name="table" type="box" size="0.55 0.4 0.025" pos="0 0 -0.025"
          rgba="0.25 0.30 0.34 1"/>
    <geom name="target_zone" type="box" size="0.06 0.06 0.001" pos="0 0.15 0.001"
          contype="0" conaffinity="0" rgba="0.2 0.65 0.3 0.6"/>
    <body name="table_item_body" pos="0 0 0.08">
      <freejoint name="table_item_free"/>
      <geom name="table_item" type="box" size="0.025 0.025 0.025"
            mass="0.04" friction="1 0.005 0.0001" rgba="0.85 0.25 0.12 1"/>
    </body>
    <camera name="front" pos="0.8 -1 0.75" xyaxes="0.78 0.62 0 -0.3 0.38 0.88" fovy="45"/>
    <camera name="overhead" pos="0 0 1.3" xyaxes="1 0 0 0 1 0" fovy="50"/>
  </worldbody>
</mujoco>"""

class MujocoSimulation:
    def __init__(self, config_path=None):
        self.config = json.loads(Path(config_path or ROOT / "configs/simulation.json").read_text())
        cfg = self.config
        for key in ("physics_dt", "control_dt", "max_command_delta"):
            if not math.isfinite(cfg[key]) or cfg[key] <= 0:
                raise ValueError(f"{key} must be positive and finite")
        ratio = cfg["control_dt"] / cfg["physics_dt"]
        if not math.isclose(ratio, round(ratio), abs_tol=1e-9):
            raise ValueError("Control interval must contain whole physics steps")
        self.substeps = round(ratio)
        self.asset_path = (ROOT / cfg["asset_path"]).resolve()
        if not self.asset_path.is_file():
            raise FileNotFoundError("SO-101 model missing. Follow the asset setup in README.md.")
        spec = mujoco.MjSpec.from_string(WORLD)
        spec.option.timestep = cfg["physics_dt"]
        for arm, position, quaternion in (
            ("left_arm", [-0.27, -0.12, 0], [1, 0, 0, 0]),
            ("right_arm", [0.27, 0.12, 0], [0, 0, 0, 1]),
        ):
            child = mujoco.MjSpec.from_file(str(self.asset_path))
            child.meshdir = str(self.asset_path.parent / "assets")
            frame = spec.worldbody.add_frame(name=arm + "_mount", pos=position, quat=quaternion)
            spec.attach(child, frame=frame, prefix=arm + "/")
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.names = tuple(f"{arm}/{joint}" for arm in ARMS for joint in JOINTS)
        self._actuators = []
        self._qpos = []
        self._qvel = []
        self.limits = {}
        for name in self.names:
            act = self.model.actuator(name)
            joint = self.model.joint(name)
            if int(self.model.actuator_trnid[act.id, 0]) != joint.id:
                raise ValueError(f"Actuator/joint mapping mismatch: {name}")
            self._actuators.append(act.id)
            self._qpos.append(int(self.model.jnt_qposadr[joint.id]))
            self._qvel.append(int(self.model.jnt_dofadr[joint.id]))
            jl = self.model.jnt_range[joint.id]
            cl = self.model.actuator_ctrlrange[act.id]
            self.limits[name] = (float(max(jl[0], cl[0])), float(min(jl[1], cl[1])))
        if self.model.nu != 12 or len(cfg["home"]) != 6:
            raise ValueError("Expected two six-actuator SO-101 arms")
        self.home_targets = {f"{arm}/{joint}": float(cfg["home"][i])
                             for arm in ARMS for i, joint in enumerate(JOINTS)}
        validate_action(BimanualAction(0, self.home_targets), self.names, self.limits,
                        self.home_targets, now=0, max_age=0, max_delta=1)
        self._object_qpos = int(self.model.jnt_qposadr[self.model.joint("table_item_free").id])
        self._object_dof = int(self.model.jnt_dofadr[self.model.joint("table_item_free").id])
        self._renderer = None
        self.contact_history = []
        self.reset(0)

    def reset(self, seed: int):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._qpos] = [self.home_targets[n] for n in self.names]
        self.data.ctrl[self._actuators] = [self.home_targets[n] for n in self.names]
        rng = np.random.default_rng(seed)
        radius = self.config["randomization_xy"]
        self.data.qpos[self._object_qpos:self._object_qpos + 2] = rng.uniform(-radius, radius, 2)
        mujoco.mj_forward(self.model, self.data)
        self.previous = dict(self.home_targets)
        self.contact_history = []
        self.seed = seed

    def observe(self, *, images=False):
        frames = {}
        if images:
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=self.config["height"],
                                                 width=self.config["width"])
            for camera in ("front", "overhead"):
                self._renderer.update_scene(self.data, camera=camera)
                frames[camera] = self._renderer.render().copy()
        return Observation(float(self.data.time), self.config["instruction"],
                           dict(zip(self.names, map(float, self.data.qpos[self._qpos]))),
                           dict(zip(self.names, map(float, self.data.qvel[self._qvel]))), frames)

    def _geom_name(self, geom_id):
        geom = self.model.geom(geom_id)
        return geom.name or f"{self.model.body(int(self.model.geom_bodyid[geom_id])).name}/geom_{geom_id}"

    def privileged(self):
        contacts = tuple((self._geom_name(int(c.geom1)), self._geom_name(int(c.geom2)))
                         for c in self.data.contact if c.dist <= 0)
        return PrivilegedState(float(self.data.time),
                               tuple(map(float, self.data.body("table_item_body").xpos)),
                               tuple(map(float, self.data.qvel[self._object_dof:self._object_dof + 6])),
                               contacts)

    def step(self, action: BimanualAction):
        cfg = self.config
        values = validate_action(action, self.names, self.limits, self.previous,
                                 now=float(self.data.time), max_age=cfg["max_action_age"],
                                 max_delta=cfg["max_command_delta"])
        self.data.ctrl[self._actuators] = values
        self.previous = dict(zip(self.names, values))
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                raise RuntimeError("SIMULATION_ERROR: nonfinite physical state")
            state = self.privileged()
            if state.contacts:
                self.contact_history.append((state.timestamp, state.contacts))
            for a, b in state.contacts:
                cross_arm = ((a.startswith("left_arm/") and b.startswith("right_arm/"))
                             or (b.startswith("left_arm/") and a.startswith("right_arm/")))
                if cross_arm:
                    raise RuntimeError(f"COLLISION: {a} with {b}; simulation stopped")

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
