"""Policy contract shared by the scripted teacher and learned policies."""
from typing import Protocol

from ..contracts import BimanualAction, Observation
from ..task import TaskSpec


class Policy(Protocol):
    name: str
    # True only for the scripted teacher, which reads simulator ground truth.
    uses_privileged_state: bool

    def reset(self, sim, task: TaskSpec) -> None:
        """Start a new episode."""

    def wants_images(self) -> bool:
        """True when the next act() call will run inference and needs camera images."""

    def act(self, obs: Observation) -> BimanualAction:
        """Return full named joint targets for this control step."""

    def after_recovery(self, sim, task: TaskSpec, progress: dict) -> None:
        """The supervisor moved the arms to a safe pose; continue from the current scene."""

    def metadata(self) -> dict:
        """Name, backend, device, checkpoint; recorded with every result."""
