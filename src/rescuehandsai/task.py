"""Structured dinner-table task and its natural-language instruction."""
from dataclasses import dataclass

import numpy as np

from .scene import UTENSILS

TASK_ID = "set_place_cup_and_handoff"

# Every template names the utensil; the tray holds both, so language decides.
TEMPLATES = (
    "Set the table: put the cup by the plate and pass the {u} to the left hand.",
    "Hand the {u} over to the left arm, then place the cup next to the plate.",
    "Put the cup on the right of the plate and give the {u} to the left arm to place.",
    "Set a place for dinner: cup beside the plate, and pass me the {u} with the other hand.",
    "Pick up the {u} and hand it to the left hand. Then move the cup to its spot by the plate.",
    "Cup next to the plate please, and the {u} goes left of the plate after a hand-off.",
)

# Utensil first: swinging a 20 cm utensil to the hand-off point would sweep through
# an already placed cup. The free right hand places the cup at the end.
SUBTASKS = ("pick_utensil", "handoff", "place_utensil", "place_cup")


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    instruction: str
    utensil: str
    seed: int
    max_recoveries: int = 2
    timeout_s: float = 75.0


def make_task(seed: int, utensil: str | None = None, template: int | None = None) -> TaskSpec:
    # separate stream from the scene randomization so the two stay independent
    rng = np.random.default_rng([seed, 7919])
    if utensil is None:
        utensil = UTENSILS[int(rng.integers(len(UTENSILS)))]
    if utensil not in UTENSILS:
        raise ValueError(f"Unknown utensil: {utensil}")
    index = int(rng.integers(len(TEMPLATES))) if template is None else template
    return TaskSpec(TASK_ID, TEMPLATES[index].format(u=utensil), utensil, seed)


def subtask_instruction(subtask: str, utensil: str) -> str:
    """Short step text, used when a VLM planner hands SmolVLA one step at a time."""
    return {
        "place_cup": "put the cup next to the plate",
        "pick_utensil": f"pick up the {utensil} with the right hand",
        "handoff": f"pass the {utensil} to the left hand",
        "place_utensil": f"place the {utensil} left of the plate",
    }[subtask]
