"""Presentation renders: the same physical state drawn in a nicer twin of the scene.

The policy's cameras and scene are never touched. The twin adds only visual things
(sky, floor, a shadow light, presentation cameras) and no bodies or joints, so a
state (qpos) from the live simulation can be copied into it and drawn from any angle.
"""
import mujoco
import numpy as np

from .scene import SHOWCASE_CAMERAS, build_model


class ShowcaseRenderer:
    def __init__(self, width: int = 1280, height: int = 720, shadows: bool = True):
        self.width, self.height, self.shadows = width, height, shadows
        self._seed, self.model, self.data, self.renderer = None, None, None, None

    def _sync(self, sim):
        if self._seed == sim.scene_params.seed and self.model is not None:
            return
        if self.renderer is not None:
            self.renderer.close()
        self.model = build_model(sim.scene_params, sim.scene_config, sim.asset_path, showcase=True)
        if (self.model.nq, self.model.nv, self.model.nbody) != (sim.model.nq, sim.model.nv, sim.model.nbody):
            raise RuntimeError("showcase twin must have the same bodies and joints as the live scene")
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        self._seed = sim.scene_params.seed

    def draw(self, sim, qpos, camera: str = "hero") -> np.ndarray:
        """RGB frame of state `qpos` (from sim.data.qpos) seen by a showcase camera."""
        if camera not in SHOWCASE_CAMERAS:
            raise ValueError(f"unknown showcase camera {camera}; choose from {sorted(SHOWCASE_CAMERAS)}")
        self._sync(sim)
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=f"showcase_{camera}")
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = self.shadows
        return self.renderer.render().copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
