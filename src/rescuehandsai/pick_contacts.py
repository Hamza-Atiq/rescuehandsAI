"""Shape-level contact classification for the pick milestone (spec §5).

Contacts are classified by collision shape (geom), never by whole body. Decorative
shapes cannot collide, so no rule may depend on them; the constructor enforces that.
"""
from dataclasses import dataclass

import mujoco
import numpy as np

from .scene import ARMS, UTENSILS

SCENE_NORMAL, JAW_UTENSIL, JAW_TABLE, VIOLATION = "scene_normal", "jaw_utensil", "jaw_table", "violation"


@dataclass(frozen=True)
class ContactVerdict:
    kind: str               # SCENE_NORMAL, JAW_UTENSIL, JAW_TABLE or VIOLATION
    labels: tuple            # failure labels; non-empty only for VIOLATION
    geoms: tuple             # (name1, name2)
    force: float             # contact normal force, N
    jaw: str | None = None  # "fixed" or "moving" for JAW_UTENSIL / JAW_TABLE


def geom_name(model, g: int) -> str:
    name = model.geom(g).name
    return name or f"{model.body(int(model.geom_bodyid[g])).name}/geom_{g}"


def collides(model, g: int) -> bool:
    return bool(model.geom_contype[g] or model.geom_conaffinity[g])


class ContactClassifier:
    def __init__(self, model, named: str, config: dict):
        if named not in UTENSILS:
            raise ValueError(f"unknown utensil: {named}")
        self.model, self.named = model, named
        self.spare = next(u for u in UTENSILS if u != named)
        self.grasp_arm = config["grasp_arm"]
        self.jaw_limit = config["jaw_table_force_limit_n"]
        self.severe_limit = config["severe_force_limit_n"]
        self.arm_of = []
        for g in range(model.ngeom):
            root = model.body(int(model.body_rootid[model.geom_bodyid[g]])).name
            self.arm_of.append(next((arm for arm in ARMS if root.startswith(arm + "/")), None))
        self.jaw = {}
        for side, names in config["jaw_grasp_geoms"].items():
            for name in names:
                self.jaw[self._geom(f"{self.grasp_arm}/{name}")] = side
        self.role = {}
        for role, names in config["scene_geoms"].items():
            for name in names:
                g = self._geom(name)
                if not collides(model, g):
                    raise ValueError(f"{name} is listed as a physical scene shape but cannot collide")
                self.role[g] = role
        for name in config["decorative_geoms"]:
            if collides(model, self._geom(name)):
                raise ValueError(f"{name} is listed as decorative but can collide")
        self._force = np.zeros(6)

    def _geom(self, name: str) -> int:
        try:
            return self.model.geom(name).id
        except KeyError as exc:
            raise ValueError(f"contact config names a geom the model does not have: {name}") from exc

    def classify(self, g1: int, g2: int, force: float) -> ContactVerdict:
        names = (geom_name(self.model, g1), geom_name(self.model, g2))
        a1, a2 = self.arm_of[g1], self.arm_of[g2]
        labels, kind, jaw = [], VIOLATION, None
        if a1 is not None and a2 is not None:
            labels.append("ARM_ARM_CONTACT" if a1 != a2 else "SELF_COLLISION")
        elif a1 is None and a2 is None:
            if g1 in self.role and g2 in self.role:
                kind = SCENE_NORMAL
            else:
                labels.append("UNKNOWN_CONTACT_PAIR")
        else:
            robot, other, arm = (g1, g2, a1) if a1 is not None else (g2, g1, a2)
            role, side = self.role.get(other), self.jaw.get(robot)
            if role is None:
                labels.append("UNKNOWN_CONTACT_PAIR")
            elif role == self.spare:
                labels.append("WRONG_ITEM_TOUCHED")
            elif arm != self.grasp_arm or role in ("plate", "cup"):
                labels.append("FORBIDDEN_CONTACT")
            elif side is not None and role == self.named:
                kind, jaw = JAW_UTENSIL, side
            elif side is not None and role == "table":
                kind, jaw = JAW_TABLE, side
                if self.jaw_limit is not None and force > self.jaw_limit:
                    kind, jaw = VIOLATION, None
                    labels.append("FORBIDDEN_CONTACT")
            else:
                labels.append("FORBIDDEN_CONTACT")
        if (a1 is not None or a2 is not None) and self.severe_limit is not None and force > self.severe_limit:
            kind, jaw = VIOLATION, None
            labels.append("EXCESS_FORCE")
        return ContactVerdict(kind, tuple(labels), names, float(force), jaw)

    def contacts(self, data) -> list:
        verdicts = []
        for i, contact in enumerate(data.contact):
            if contact.dist > 0:
                continue
            mujoco.mj_contactForce(self.model, data, i, self._force)
            verdicts.append(self.classify(int(contact.geom1), int(contact.geom2), abs(float(self._force[0]))))
        return verdicts
