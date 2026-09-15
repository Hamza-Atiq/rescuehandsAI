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
class PrivilegedState:
    timestamp: float
    object_position: tuple[float, float, float]
    object_velocity: tuple[float, ...]
    contacts: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    policy: str
    control_check_passed: bool
    # None means manipulation success has not been evaluated.
    manipulation_success: bool | None = None
