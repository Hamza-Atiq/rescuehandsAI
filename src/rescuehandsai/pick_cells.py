"""The four runs of one pick scene (spec §3), the pick wordings (spec §8) and the start check."""
from dataclasses import dataclass, replace
import math

import numpy as np

from .contracts import BimanualAction
from .pick_config import PICK_PHYSICS_VERSION, load_rules
from .scene import SCENE_ITEMS, SceneParams, sample_params

# cell -> (named utensil, (utensil in slot 0, utensil in slot 1))
CELLS = {
    "F-A": ("fork", ("fork", "spoon")),
    "F-B": ("fork", ("spoon", "fork")),
    "S-A": ("spoon", ("fork", "spoon")),
    "S-B": ("spoon", ("spoon", "fork")),
}

# No wording names a side or a slot: the word is the only clue to which utensil.
TRAIN_TEMPLATES = {
    "T1": "pick up the {u} with the right hand",
    "T2": "grab the {u} with your right hand",
    "T3": "lift the {u} off the tray using the right arm",
    "T4": "take the {u} and hold it up",
    "T5": "use the right gripper to pick up the {u}",
    "T6": "get the {u} from the tray and hold it",
    "T7": "please raise the {u} with your right hand",
    "T8": "right hand: lift the {u}",
}
HELD_BACK_TEMPLATES = {
    "H1": "I need the {u}, lift it with your right hand",
    "H2": "hold up the {u} for me",
    "H3": "could you pick the {u} up?",
    "H4": "the {u}, please, in the right hand",
}


@dataclass(frozen=True)
class PickTask:
    seed: int
    cell: str
    utensil: str      # the named utensil
    spare: str
    template: str
    instruction: str
    physics_version: int = PICK_PHYSICS_VERSION


def named_slot(cell: str) -> int:
    named, order = CELLS[cell]
    return order.index(named)


def slot_jitter(base: SceneParams, config: dict) -> dict:
    """slot index -> (dx, dy, dyaw) that the scene drew for whichever utensil sits there."""
    slots = config["utensil"]["slots"]
    jitter = {}
    for item, slot in base.slots.items():
        x, y, yaw = base.poses[item]
        sx, sy = slots[slot]
        jitter[slot] = (x - sx, y - sy, yaw - math.pi / 2)
    return jitter


def cell_params(base: SceneParams, cell: str, config: dict) -> SceneParams:
    """Jitter stays with the slot; mass, friction and size stay with the item."""
    _, order = CELLS[cell]
    jitter = slot_jitter(base, config)
    poses, slots = dict(base.poses), {}
    for slot, item in enumerate(order):
        sx, sy = config["utensil"]["slots"][slot]
        dx, dy, dyaw = jitter[slot]
        poses[item] = (sx + dx, sy + dy, math.pi / 2 + dyaw)
        slots[item] = slot
    return replace(base, poses=poses, slots=slots)


def template_text(template_id: str) -> str:
    if template_id in TRAIN_TEMPLATES:
        return TRAIN_TEMPLATES[template_id]
    return HELD_BACK_TEMPLATES[template_id]


def make_pick_task(seed: int, cell: str, template: str, physics_version: int = PICK_PHYSICS_VERSION) -> PickTask:
    named, _ = CELLS[cell]
    spare = next(item for item in CELLS[cell][1] if item != named)
    return PickTask(seed, cell, named, spare, template, template_text(template).format(u=named), physics_version)


def assign_templates(count: int, template_ids) -> list:
    """Fixed rotation over the ordered scene list; the first templates take any extra scenes."""
    ids = list(template_ids)
    return [ids[i % len(ids)] for i in range(count)]


@dataclass(frozen=True)
class StartCheck:
    cell: str
    valid: bool
    reasons: tuple


def check_start(seed: int, cell: str, *, physics_version: int = PICK_PHYSICS_VERSION, rules: dict | None = None, edit=None) -> StartCheck:
    """Frozen start-validity rules on a separate simulation; the arms hold their home targets.

    `edit` (tests only) changes the cell's SceneParams before the scene is built."""
    from .sim import MujocoSimulation  # sim imports scene; keep this module importable without a model

    limits = (rules or load_rules())["start_check"]
    sim = MujocoSimulation(physics_version=physics_version)
    try:
        params = cell_params(sample_params(sim.scene_config, seed), cell, sim.scene_config)
        if edit is not None:
            params = edit(params)
        sim.reset(seed, params=params)
        model, data = sim.model, sim.data
        item_bodies = {model.body(item).id for item in SCENE_ITEMS}
        reasons = []
        for contact in data.contact:
            b1, b2 = int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])
            if contact.dist < -limits["overlap_depth_m"] and (b1 in item_bodies or b2 in item_bodies):
                reasons.append(f"overlap:{sim._geom_name(int(contact.geom1))}-{sim._geom_name(int(contact.geom2))}")
        start = {item: data.body(item).xpos.copy() for item in SCENE_ITEMS}
        for _ in range(limits["hold_control_steps"]):
            sim.step(BimanualAction(sim.observe().timestamp, sim.home_targets))
        table = sim.scene_config["table"]
        state = sim.privileged().objects
        for item in SCENE_ITEMS:
            position = data.body(item).xpos
            shift = float(np.linalg.norm(position - start[item]))
            speed = float(np.linalg.norm(state[item].linear_velocity))
            if shift > limits["max_shift_m"] or speed > limits["max_speed_mps"]:
                reasons.append(f"moving:{item}")
            if (abs(position[0] - table["center"][0]) > table["half_size"][0]
                    or abs(position[1] - table["center"][1]) > table["half_size"][1] or position[2] < 0):
                reasons.append(f"off_table:{item}")
        reasons = tuple(dict.fromkeys(reasons))
        return StartCheck(cell, not reasons, reasons)
    finally:
        sim.close()


def check_scene(seed: int, **kwargs) -> dict:
    """All four cells; a scene is usable only if every cell has a valid start."""
    return {cell: check_start(seed, cell, **kwargs) for cell in CELLS}
