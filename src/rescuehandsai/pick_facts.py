"""Per-physics-step facts for the pick judge, read from MuJoCo after every substep (spec §5)."""
import mujoco
import numpy as np

from .pick_outcome import SubstepFacts
from .scene import SCENE_ITEMS

# Only these warnings make an episode invalid (spec §5); every other warning is recorded.
SIM_ERROR_WARNINGS = ("mjWARN_BADQACC", "mjWARN_CONTACTFULL", "mjWARN_CNSTRFULL")
WARNING_NAMES = {int(value): name for name, value in mujoco.mjtWarning.__members__.items() if name != "mjNWARNING"}


class SimulatorFailure(RuntimeError):
    """A simulator failure: the episode is invalid (SIM_ERROR), not a robot failure."""


class FactReader:
    def __init__(self, sim, classifier, gripper_site: str = "right_arm/gripperframe"):
        self.sim, self.classifier, self.site = sim, classifier, gripper_site
        model = sim.model
        self._dof = {item: int(model.jnt_dofadr[model.joint(item + "_free").id]) for item in SCENE_ITEMS}
        self._fatal = {int(getattr(mujoco.mjtWarning, name)) for name in SIM_ERROR_WARNINGS}
        self._baseline = self._counts()
        self.other_warnings = {}

    def _counts(self) -> list:
        return [int(w.number) for w in self.sim.data.warning]

    def read(self, physics_step: int, control_step: int) -> SubstepFacts:
        data = self.sim.data
        counts = self._counts()
        for index, (now, before) in enumerate(zip(counts, self._baseline)):
            if now <= before:
                continue
            name = WARNING_NAMES.get(index, f"warning_{index}")
            if index in self._fatal:
                raise SimulatorFailure(f"{name} at physics step {physics_step}")
            self.other_warnings[name] = now - before
        positions, rotations, linear, angular = {}, {}, {}, {}
        for item in SCENE_ITEMS:
            body = data.body(item)
            positions[item] = body.xpos.copy()
            rotations[item] = body.xmat.reshape(3, 3).copy()
            v = data.qvel[self._dof[item]:self._dof[item] + 6]
            linear[item] = float(np.linalg.norm(v[:3]))
            angular[item] = float(np.linalg.norm(v[3:]))
        site = data.site(self.site)
        return SubstepFacts(physics_step, control_step, positions, rotations, linear, angular,
                            site.xpos.copy(), site.xmat.reshape(3, 3).copy(), tuple(self.classifier.contacts(data)))
