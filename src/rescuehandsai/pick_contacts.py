"""Shape-level contact classification for the pick milestone (spec §5).

Contacts are classified by collision shape (geom), never by whole body. Decorative
shapes cannot collide, so no rule may depend on them; the constructor enforces that.
"""
from dataclasses import dataclass

import mujoco
import numpy as np

from .scene import ARMS, UTENSILS

# Owner decision 23 Sep 2026: every robot-table contact is forbidden. The jaw meshes sit below the
# pads at every reachable hand angle, so a pad-only table permission could never apply alone.
SCENE_NORMAL, JAW_UTENSIL, VIOLATION = "scene_normal", "jaw_utensil", "violation"


@dataclass(frozen=True)
class ContactVerdict:
    kind: str               # SCENE_NORMAL, JAW_UTENSIL or VIOLATION
    labels: tuple            # failure labels; non-empty only for VIOLATION
    geoms: tuple             # (name1, name2)
    force: float             # contact normal force, N
    jaw: str | None = None  # "fixed" or "moving" for JAW_UTENSIL


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
        if "jaw_table_force_limit_n" in config:
            raise ValueError("jaw_table_force_limit_n was removed: every robot-table contact is forbidden")
        self.severe_limit = config["severe_force_limit_n"]
        self.arm_of = []
        for g in range(model.ngeom):
            root = model.body(int(model.body_rootid[model.geom_bodyid[g]])).name
            self.arm_of.append(next((arm for arm in ARMS if root.startswith(arm + "/")), None))
        self.jaw = {}
        for side, names in config["jaw_grasp_geoms"].items():
            for name in names:
                self.jaw[self._geom(f"{self.grasp_arm}/{name}")] = side
        # Jaw meshes that carry real grip load (grip probe 22 Sep: geom_104 7-8 N, geom_93 up to
        # ~10 N). Owner decision 23 Sep: they may support the NAMED utensil only, like the pads
        # in jaw_grasp_geoms (no robot shape may touch the table). Selected by body + mesh name, never
        # by compiled geom id.
        self.utensil_only = {}
        for side, selectors in config.get("jaw_utensil_only_meshes", {}).items():
            for sel in selectors:
                g = self._mesh_geom(f"{self.grasp_arm}/{sel['body']}", f"{self.grasp_arm}/{sel['mesh']}")
                if g in self.jaw or self.utensil_only.get(g, side) != side:
                    raise ValueError(f"{sel} is listed for more than one jaw role")
                self.utensil_only[g] = side
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

    def _mesh_geom(self, body: str, mesh: str) -> int:
        model = self.model
        found = [g for g in range(model.ngeom)
                 if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                 and model.body(int(model.geom_bodyid[g])).name == body
                 and model.mesh(int(model.geom_dataid[g])).name == mesh]
        if len(found) != 1:
            raise ValueError(f"contact config mesh selector {body} / {mesh} matches {len(found)} geoms, not 1")
        if not collides(model, found[0]):
            raise ValueError(f"contact config mesh selector {body} / {mesh} names a shape that cannot collide")
        return found[0]

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
            elif robot in self.utensil_only and role == self.named:
                kind, jaw = JAW_UTENSIL, self.utensil_only[robot]
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
