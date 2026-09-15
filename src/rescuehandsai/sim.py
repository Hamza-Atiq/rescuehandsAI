"""MuJoCo owns all physical state. No object attachment or pose rewrites in step."""
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from .contracts import BimanualAction, ObjectState, Observation, PrivilegedState
from .control import validate_action
from .scene import ROOT, SCENE_ITEMS, build_model, load_config, sample_params

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
ARMS = ("left_arm", "right_arm")
# Cameras the policy may see. "front" exists for videos only.
POLICY_CAMERAS = {"overhead": "overhead", "left_wrist": "left_arm/wrist_cam",
                  "right_wrist": "right_arm/wrist_cam"}
VIDEO_CAMERAS = {"front": "front"}


class MujocoSimulation:
    def __init__(self, config_path=None, scene_config_path=None, seed: int = 0):
        self.config = json.loads(Path(config_path or ROOT / "configs/simulation.json").read_text())
        cfg = self.config
        for key in ("physics_dt", "control_dt", "max_command_delta"):
            if not math.isfinite(cfg[key]) or cfg[key] <= 0:
                raise ValueError(f"{key} must be positive and finite")
        ratio = cfg["control_dt"] / cfg["physics_dt"]
        if not math.isclose(ratio, round(ratio), abs_tol=1e-9):
            raise ValueError("Control interval must contain whole physics steps")
        if len(cfg["home"]) != 6:
            raise ValueError("Home pose needs six joint values")
        self.substeps = round(ratio)
        self.scene_config = load_config(scene_config_path)
        self.asset_path = (ROOT / cfg["asset_path"]).resolve()
        if not self.asset_path.is_file():
            raise FileNotFoundError("SO-101 model missing. Follow the asset setup in README.md.")
        self.names = tuple(f"{arm}/{joint}" for arm in ARMS for joint in JOINTS)
        self.home_targets = {f"{arm}/{joint}": float(cfg["home"][i])
                             for arm in ARMS for i, joint in enumerate(JOINTS)}
        self._renderers = {}
        self.model = None
        self.reset(seed)

    # -- model binding -------------------------------------------------------
    def _bind(self):
        model = self.model
        if model.nu != 12:
            raise ValueError("Expected two six-actuator SO-101 arms")
        self._actuators, self._qpos, self._qvel, self.limits = [], [], [], {}
        for name in self.names:
            act = model.actuator(name)
            joint = model.joint(name)
            if int(model.actuator_trnid[act.id, 0]) != joint.id:
                raise ValueError(f"Actuator/joint mapping mismatch: {name}")
            self._actuators.append(act.id)
            self._qpos.append(int(model.jnt_qposadr[joint.id]))
            self._qvel.append(int(model.jnt_dofadr[joint.id]))
            jl, cl = model.jnt_range[joint.id], model.actuator_ctrlrange[act.id]
            self.limits[name] = (float(max(jl[0], cl[0])), float(min(jl[1], cl[1])))
        self._item_joint = {item: model.joint(item + "_free") for item in SCENE_ITEMS}
        # geom id -> owning arm name, or None for scene geoms
        self.geom_arm = []
        for g in range(model.ngeom):
            root = model.body(int(model.body_rootid[model.geom_bodyid[g]])).name
            self.geom_arm.append(next((arm for arm in ARMS if root.startswith(arm + "/")), None))

    def reset(self, seed: int, instruction: str | None = None):
        self.seed = seed
        self.instruction = instruction or self.config["instruction"]
        self.scene_params = sample_params(self.scene_config, seed)
        self.close()
        self.model = build_model(self.scene_params, self.scene_config, self.asset_path)
        self.model.opt.timestep = self.config["physics_dt"]
        self.data = mujoco.MjData(self.model)
        self._bind()
        validate_action(BimanualAction(0, self.home_targets), self.names, self.limits,
                        self.home_targets, now=0, max_age=0, max_delta=10)
        self.data.qpos[self._qpos] = [self.home_targets[n] for n in self.names]
        self.data.ctrl[self._actuators] = [self.home_targets[n] for n in self.names]
        mujoco.mj_forward(self.model, self.data)
        self.previous = dict(self.home_targets)
        self.contact_samples = 0
        self.faults = {}

    # -- observations --------------------------------------------------------
    def render(self, cameras: dict, width=None, height=None) -> dict:
        width = width or self.config["width"]
        height = height or self.config["height"]
        key = (width, height)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, height=height, width=width)
        renderer = self._renderers[key]
        frames = {}
        for out_name, camera in cameras.items():
            renderer.update_scene(self.data, camera=camera)
            # Shadows cost ~6x render time on integrated graphics. Training data and
            # deployment must use the same setting, so it lives in config.
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = bool(self.config["render_shadows"])
            frames[out_name] = renderer.render().copy()
        return frames

    def observe(self, *, images=False):
        frames = self.render(POLICY_CAMERAS) if images else {}
        return Observation(float(self.data.time), self.instruction,
                           dict(zip(self.names, map(float, self.data.qpos[self._qpos]))),
                           dict(zip(self.names, map(float, self.data.qvel[self._qvel]))), frames)

    def _geom_name(self, geom_id):
        geom = self.model.geom(geom_id)
        return geom.name or f"{self.model.body(int(self.model.geom_bodyid[geom_id])).name}/geom_{geom_id}"

    def privileged(self):
        contacts = tuple((self._geom_name(int(c.geom1)), self._geom_name(int(c.geom2)))
                         for c in self.data.contact if c.dist <= 0)
        objects = {}
        for item, joint in self._item_joint.items():
            adr = int(self.model.jnt_qposadr[joint.id])
            dof = int(self.model.jnt_dofadr[joint.id])
            objects[item] = ObjectState(tuple(map(float, self.data.qpos[adr:adr + 3])),
                                        tuple(map(float, self.data.qpos[adr + 3:adr + 7])),
                                        tuple(map(float, self.data.qvel[dof:dof + 3])))
        return PrivilegedState(float(self.data.time), objects, contacts)

    def site_pose(self, name: str):
        site = self.data.site(name)
        return site.xpos.copy(), site.xmat.reshape(3, 3).copy()

    # -- fault injection (evaluation only) ---------------------------------------
    def set_actuator_fault(self, name: str, value: float | None):
        """Simulated actuator fault: the named motor follows `value` instead of its
        command until cleared with None. Used only to inject failures in evaluation."""
        if name not in self.names:
            raise KeyError(name)
        if value is None:
            self.faults.pop(name, None)
        else:
            low, high = self.limits[name]
            self.faults[name] = min(high, max(low, float(value)))

    # -- stepping ------------------------------------------------------------
    def step(self, action: BimanualAction):
        cfg = self.config
        values = validate_action(action, self.names, self.limits, self.previous,
                                 now=float(self.data.time), max_age=cfg["max_action_age"],
                                 max_delta=cfg["max_command_delta"])
        self.previous = dict(zip(self.names, values))
        if self.faults:
            values = tuple(self.faults.get(n, v) for n, v in zip(self.names, values))
        self.data.ctrl[self._actuators] = values
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                raise RuntimeError("SIMULATION_ERROR: nonfinite physical state")
            for c in self.data.contact:
                if c.dist > 0:
                    continue
                self.contact_samples += 1
                a, b = self.geom_arm[c.geom1], self.geom_arm[c.geom2]
                if a is not None and b is not None and a != b:
                    raise RuntimeError(f"COLLISION: {self._geom_name(int(c.geom1))} with "
                                       f"{self._geom_name(int(c.geom2))}; simulation stopped")

    def close(self):
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers = {}
