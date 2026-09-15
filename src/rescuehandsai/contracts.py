"""Simulator-independent contracts. Units: radians, metres, simulated seconds."""
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class BimanualAction:
    timestamp: float
    targets: Mapping[str, float]


@dataclass(frozen=True)
class Observation:
    timestamp: float
    instruction: str
    positions: Mapping[str, float]
    velocities: Mapping[str, float]
    # Images are RGB uint8 arrays. NumPy is deliberately not imported here.
    images: Mapping[str, object]


@dataclass(frozen=True)
class ObjectState:
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]
    linear_velocity: tuple[float, float, float]


@dataclass(frozen=True)
class PrivilegedState:
    """Simulator ground truth for the teacher, auditor and evaluation only."""
    timestamp: float
    objects: Mapping[str, ObjectState]
    contacts: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    policy: str
    control_check_passed: bool
    # None means manipulation success has not been evaluated.
    manipulation_success: bool | None = None
