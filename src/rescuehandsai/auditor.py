"""Physics auditor: facts from privileged simulator state, and failure events.

Used by the teacher, the supervisor and evaluation. Never an input to the policy.
"""
from dataclasses import dataclass
import math

from .scene import ARMS, SCENE_ITEMS

FAILURE_LABELS = ("TIMEOUT", "COLLISION", "SELF_COLLISION", "FAILED_GRASP", "OBJECT_DROPPED",
                  "OBJECT_OUT_OF_BOUNDS", "TARGET_MISSED", "CONTROL_LIMIT", "INVALID_ACTION",
                  "POLICY_ERROR", "SIMULATION_ERROR", "RECOVERY_EXHAUSTED")
ZONE_MAX_HEIGHT = 0.06


@dataclass(frozen=True)
class AuditFacts:
    time: float
    held_by: dict        # item -> arms whose fixed AND moving jaw touch it
    touching: dict       # item -> arms with any gripper contact
    supported: dict      # item -> resting on table, plate or another item
    in_zone: dict        # item -> zone name or None
    height: dict         # item -> centre z (m)
    speed: dict          # item -> linear speed (m/s)
    out_of_bounds: set
    cross_arm_contact: bool
    positions: dict = None  # item -> (x, y, z)


@dataclass(frozen=True)
class FailureEvent:
    label: str
    time: float
    item: str | None = None
    arm: str | None = None

    def __post_init__(self):
        if self.label not in FAILURE_LABELS:
            raise ValueError(f"Unknown failure label: {self.label}")


def _gripper_bodies(model):
    fixed = {model.body(f"{arm}/gripper").id: arm for arm in ARMS}
    moving = {model.body(f"{arm}/moving_jaw_so101_v1").id: arm for arm in ARMS}
    return fixed, moving


def compute_facts(sim) -> AuditFacts:
    model, data = sim.model, sim.data
    fixed, moving = _gripper_bodies(model)
    item_body = {model.body(item).id: item for item in SCENE_ITEMS}
    touch_fixed = {i: set() for i in SCENE_ITEMS}
    touch_moving = {i: set() for i in SCENE_ITEMS}
    supported = dict.fromkeys(SCENE_ITEMS, False)
    cross = False
    for c in data.contact:
        if c.dist > 0:
            continue
        b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
        a1, a2 = sim.geom_arm[c.geom1], sim.geom_arm[c.geom2]
        if a1 is not None and a2 is not None and a1 != a2:
            cross = True
        for b, other, other_arm in ((b1, b2, a2), (b2, b1, a1)):
            item = item_body.get(b)
            if item is None:
                continue
            if other in fixed:
                touch_fixed[item].add(fixed[other])
            elif other in moving:
                touch_moving[item].add(moving[other])
            elif other_arm is None:  # world (table, plate) or another item
                supported[item] = True
    zones = sim.scene_config["zones"]
    thx, thy = sim.scene_config["table"]["half_size"]
    tcx, tcy = sim.scene_config["table"]["center"]
    in_zone, height, speed, oob, positions = {}, {}, {}, set(), {}
    state = sim.privileged().objects
    for item in SCENE_ITEMS:
        x, y, z = state[item].position
        positions[item] = (x, y, z)
        height[item] = z
        speed[item] = math.sqrt(sum(v * v for v in state[item].linear_velocity))
        in_zone[item] = next((name for name, zone in zones.items()
                              if abs(x - zone["pos"][0]) <= zone["half_size"][0]
                              and abs(y - zone["pos"][1]) <= zone["half_size"][1]
                              and z <= ZONE_MAX_HEIGHT), None)
        if abs(x - tcx) > thx or abs(y - tcy) > thy or z < -0.02:
            oob.add(item)
    return AuditFacts(
        time=float(data.time),
        held_by={i: touch_fixed[i] & touch_moving[i] for i in SCENE_ITEMS},
        touching={i: touch_fixed[i] | touch_moving[i] for i in SCENE_ITEMS},
        supported=supported, in_zone=in_zone, height=height, speed=speed,
        out_of_bounds=oob, cross_arm_contact=cross, positions=positions)


class FailureMonitor:
    """Turns fact streams into debounced, once-reported failure events.

    `expected_holds` maps item -> arm that should be holding it now.
    """

    def __init__(self, debounce: int = 3):
        if debounce < 1:
            raise ValueError("debounce must be at least 1")
        self.debounce = debounce
        self._was_held = {}
        self._missing = {}
        self._reported = set()

    def reset_expectation(self, item: str):
        for store in (self._was_held, self._missing):
            store.pop(item, None)
        self._reported = {r for r in self._reported if r[1] != item}

    def update(self, facts: AuditFacts, expected_holds: dict) -> list:
        events = []

        def report(label, item=None, arm=None):
            key = (label, item, arm)
            if key not in self._reported:
                self._reported.add(key)
                events.append(FailureEvent(label, facts.time, item, arm))

        if facts.cross_arm_contact:
            report("COLLISION")
        for item in sorted(facts.out_of_bounds):
            report("OBJECT_OUT_OF_BOUNDS", item)
        for item in list(self._missing):
            if item not in expected_holds:
                self.reset_expectation(item)
        for item, arm in expected_holds.items():
            if arm in facts.held_by.get(item, ()):
                self._was_held[item] = True
                self._missing[item] = 0
                continue
            self._missing[item] = self._missing.get(item, 0) + 1
            if self._missing[item] >= self.debounce:
                label = "OBJECT_DROPPED" if self._was_held.get(item) else "FAILED_GRASP"
                report(label, item, arm)
        return events
