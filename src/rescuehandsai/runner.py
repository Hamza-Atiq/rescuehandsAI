"""Episode runner: one state machine for every policy, with an optional supervisor.

Always on (bimanual controller): clamp each command to joint limits and the
per-step change limit, so a jumpy learned action is slowed down, not executed.
Supervisor on: the physics auditor watches every step; on a dropped item the
arms open and return to a safe pose, then the policy continues (bounded retries).
"""
from dataclasses import dataclass, field
import math
import time

from .auditor import FailureEvent, FailureMonitor, compute_facts
from .contracts import BimanualAction
from .evaluation import HandoffTracker, task_outcome
from .motion import MotionFollower, Move

STATES = ("IDLE", "EXECUTING", "RECOVERING", "SUCCEEDED", "FAILED", "ABORTED")
# TARGET_MISSED here comes from the progress watchdog (a stalled step); the
# "policy says done but physics disagrees" case ends the episode directly.
RECOVERABLE = {"OBJECT_DROPPED", "FAILED_GRASP", "TARGET_MISSED"}
TERMINAL = {"COLLISION", "OBJECT_OUT_OF_BOUNDS", "SIMULATION_ERROR", "INVALID_ACTION", "POLICY_ERROR"}


class EpisodeBudgetExceeded(Exception):
    """The episode's step budget ran out (also during recovery motion)."""


def clamp_action(action: BimanualAction, previous: dict, limits: dict, max_delta: float):
    """Return (safe action, number of joints that had to be clamped).

    The command must name exactly the robot's joints: a missing or unexpected name
    is an INVALID_ACTION, never silently dropped or filled in.
    """
    names, expected = set(action.targets), set(limits)
    if names != expected:
        raise ValueError(f"INVALID_ACTION: missing {sorted(expected - names)}, unexpected {sorted(names - expected)}")
    safe, clamped = {}, 0
    step = max_delta * 0.98
    for name, (low, high) in limits.items():
        value = float(action.targets[name])
        if not math.isfinite(value):
            raise ValueError(f"INVALID_ACTION: non-finite target for {name}")
        target = min(high, max(low, value))
        target = min(previous[name] + step, max(previous[name] - step, target))
        clamped += abs(target - value) > 1e-9
        safe[name] = target
    return BimanualAction(action.timestamp, safe), clamped


@dataclass
class EpisodeLog:
    seed: int
    policy: dict
    instruction: str
    supervisor: bool
    fault: str | None
    state: str = "IDLE"
    steps: int = 0
    events: list = field(default_factory=list)
    recoveries: int = 0
    clamped_joint_steps: int = 0
    inference_calls: int = 0
    inference_seconds: list = field(default_factory=list)
    fault_step: int | None = None
    max_steps: int = 0
    max_recoveries: int = 0
    outcome: dict = field(default_factory=dict)
    failure: str | None = None
    sim_seconds: float = 0.0
    wall_seconds: float = 0.0


class EpisodeRunner:
    def __init__(self, sim, policy, *, supervisor: bool, fault=None, max_steps: int | None = None,
                 max_recoveries: int | None = None, settle_steps: int = 10, stall_seconds: float = 15.0,
                 on_frame=None, on_command=None):
        """max_steps / max_recoveries default to the task's timeout_s / max_recoveries;
        explicit values are evaluation overrides and are recorded in the episode log."""
        self.sim, self.policy, self.supervisor = sim, policy, supervisor
        self.stall_seconds = stall_seconds
        self.fault, self.max_steps, self.max_recoveries = fault, max_steps, max_recoveries
        self.settle_steps, self.on_frame, self.on_command = settle_steps, on_frame, on_command

    # -- progress from physics (never from the policy's claims) -----------------
    @staticmethod
    def _progress(facts, task, holders, handoff, progress):
        u = task.utensil
        progress["pick_utensil"] |= "right_arm" in holders
        progress["handoff"] |= handoff.update(facts)
        placed = facts.in_zone[u] == "utensil_zone" and not facts.touching[u]
        progress["place_utensil"] = progress["handoff"] and placed
        progress["place_cup"] = facts.in_zone["cup"] == "cup_zone" and not facts.touching["cup"]
        return progress

    def _expected_holds(self, facts, task):
        expect = {}
        for item, zone in ((task.utensil, "utensil_zone"), ("cup", "cup_zone")):
            if facts.held_by[item]:
                self._last_holder[item] = sorted(facts.held_by[item])[0]
            if item in self._last_holder and facts.in_zone[item] != zone:
                holder = self._last_holder[item]
                expect[item] = sorted(facts.held_by[item])[0] if facts.held_by[item] else holder
        return expect

    def _step(self, action, log, facts, task, state):
        """The only place the episode advances physics: fault clock, budget, frame hook."""
        if log.steps >= log.max_steps:
            raise EpisodeBudgetExceeded
        if self.fault is not None and self.fault.before_step(self.sim, facts, task, log.steps):
            log.fault_step = log.steps
        if self.on_command:  # demonstration recording: every executed command, recovery included
            self.on_command(self.sim, action)
        self.sim.step(action)
        log.steps += 1
        if self.on_frame:
            self.on_frame(self.sim, state)

    def _safe_pose(self, log, task):
        """Open both hands, then return both arms to the start pose.

        The fault clock keeps running (a half-second glitch lasts half a second in
        every configuration) but a new fault cannot start: facts are withheld."""
        follower = MotionFollower(self.sim.previous, self.sim.config["max_command_delta"])
        opened = {n: 0.9 for n in self.sim.names if n.endswith("gripper")}
        for move in (Move(opened, 8, "open"), Move(self.sim.home_targets, 30, "home")):
            follower.begin(move)
            done = False
            while not done:
                done = follower.advance(move)
                action = BimanualAction(self.sim.observe().timestamp, dict(follower.targets))
                self._step(action, log, None, task, "RECOVERING")

    def _recover(self, log, task, events, monitor, holders, handoff, progress, policy, facts):
        """Bounded recovery: open both hands, return home, let the policy re-plan.

        Returns fresh facts, or None when the episode must stop (budget or a new failure)."""
        if log.recoveries >= log.max_recoveries:
            log.events.append({"label": "RECOVERY_EXHAUSTED", "time": facts.time})
            log.state, log.failure = "FAILED", "RECOVERY_EXHAUSTED"
            return None
        log.recoveries += 1
        log.state = "RECOVERING"
        for event in events:
            if event.item is not None:
                monitor.reset_expectation(event.item)
                self._last_holder.pop(event.item, None)
            if event.item == task.utensil:
                holders.clear()
                handoff.reset()
                progress["pick_utensil"] = progress["handoff"] = False
        try:
            self._safe_pose(log, task)
        except EpisodeBudgetExceeded:
            log.events.append({"label": "TIMEOUT", "time": float(self.sim.data.time)})
            log.state, log.failure = "FAILED", "TIMEOUT"
            return None
        except (ValueError, RuntimeError) as exc:
            label = "COLLISION" if "COLLISION" in str(exc) else "SIMULATION_ERROR"
            log.events.append({"label": label, "time": facts.time, "detail": str(exc)})
            log.state, log.failure = "FAILED", label
            return None
        policy.after_recovery(self.sim, task, progress)
        log.state = "EXECUTING"
        return compute_facts(self.sim)

    # -- episode -------------------------------------------------------------------
    def run(self, task) -> EpisodeLog:
        sim, policy = self.sim, self.policy
        sim.reset(task.seed, instruction=task.instruction)
        policy.reset(sim, task)
        if self.fault is not None:
            self.fault.reset(task.seed)
        log = EpisodeLog(task.seed, policy.metadata(), task.instruction, self.supervisor,
                         type(self.fault).__name__ if self.fault else None, state="EXECUTING")
        log.max_steps = (self.max_steps if self.max_steps is not None
                         else round(task.timeout_s / sim.config["control_dt"]))
        log.max_recoveries = self.max_recoveries if self.max_recoveries is not None else task.max_recoveries
        monitor = FailureMonitor(debounce=3)
        self._last_holder = {}
        holders, handoff = set(), HandoffTracker(task.utensil)
        progress = dict.fromkeys(("pick_utensil", "handoff", "place_utensil", "place_cup"), False)
        stall_limit = max(1, round(self.stall_seconds / sim.config["control_dt"]))
        stall, fingerprint = 0, None
        facts, stable, done_steps, started = compute_facts(sim), 0, 0, time.perf_counter()
        start_positions = dict(facts.positions)
        max_delta = sim.config["max_command_delta"]
        while log.state in ("EXECUTING", "RECOVERING"):
            if log.steps >= log.max_steps:
                log.events.append({"label": "TIMEOUT", "time": facts.time})
                log.state, log.failure = "FAILED", "TIMEOUT"
                break
            try:
                obs = sim.observe(images=policy.wants_images())
                t0 = time.perf_counter()
                raw = policy.act(obs)
                if obs.images:
                    log.inference_calls += 1
                    log.inference_seconds.append(time.perf_counter() - t0)
                action, clamped = clamp_action(raw, sim.previous, sim.limits, max_delta)
                log.clamped_joint_steps += clamped
                self._step(action, log, facts, task, log.state)
            except ValueError as exc:
                log.events.append({"label": "INVALID_ACTION", "time": facts.time, "detail": str(exc)})
                log.state, log.failure = "FAILED", "INVALID_ACTION"
                break
            except RuntimeError as exc:
                known = ("COLLISION", "SIMULATION_ERROR", *RECOVERABLE)
                label = str(exc).split(":")[0] if str(exc).startswith(known) else "POLICY_ERROR"
                log.events.append({"label": label, "time": facts.time, "detail": str(exc)})
                # A policy may report a recoverable situation it cannot handle itself
                # (the teacher does this when the item is no longer in the hand it planned for).
                if label in RECOVERABLE and self.supervisor:
                    asked = [FailureEvent(label, facts.time, task.utensil, None)]
                    facts = self._recover(log, task, asked, monitor, holders, handoff, progress, policy, facts)
                    if facts is None:
                        break
                    done_steps, stall, fingerprint, stable = 0, 0, None, 0
                    continue
                log.state, log.failure = "FAILED", label
                break
            facts = compute_facts(sim)
            for item in (task.utensil,):
                holders |= facts.held_by[item]
            progress = self._progress(facts, task, holders, handoff, progress)
            events = monitor.update(facts, self._expected_holds(facts, task))
            # Progress watchdog: catches a grasp that never starts or a step that stalls,
            # which hold-based checks cannot see (nothing was held yet).
            now = (tuple(progress.values()), tuple(sorted(facts.held_by[task.utensil])),
                   tuple(sorted(facts.held_by["cup"])))
            stall = 0 if now != fingerprint else stall + 1
            fingerprint = now
            if stall >= stall_limit:
                stall = 0
                utensil_pending = not progress["pick_utensil"] and not facts.held_by[task.utensil]
                cup_pending = progress["place_utensil"] and not facts.held_by["cup"] and not progress["place_cup"]
                if utensil_pending or cup_pending:
                    item = task.utensil if utensil_pending else "cup"
                    events.append(FailureEvent("FAILED_GRASP", facts.time, item, None))
                else:
                    events.append(FailureEvent("TARGET_MISSED", facts.time, None, None))
            for event in events:
                log.events.append({"label": event.label, "time": event.time, "item": event.item, "arm": event.arm})
            terminal = [e for e in events if e.label in TERMINAL]
            if terminal:
                log.state, log.failure = "FAILED", terminal[0].label
                break
            recoverable = [e for e in events if e.label in RECOVERABLE]
            if recoverable and self.supervisor:
                facts = self._recover(log, task, recoverable, monitor, holders, handoff, progress, policy, facts)
                if facts is None:
                    break
                done_steps, stall, fingerprint, stable = 0, 0, None, 0
                continue
            outcome = task_outcome(facts, task, handoff.done, sim.scene_params, start_positions)
            stable = stable + 1 if outcome["success"] else 0
            if stable >= self.settle_steps:
                log.state = "SUCCEEDED"
            elif getattr(policy, "done", False) and not outcome["success"]:
                done_steps += 1
                if done_steps > 3 * self.settle_steps:
                    # the policy believes it finished; physics disagrees
                    log.events.append({"label": "TARGET_MISSED", "time": facts.time})
                    log.state, log.failure = "FAILED", "TARGET_MISSED"
        sim.set_actuator_fault("right_arm/gripper", None)
        sim.set_actuator_fault("left_arm/gripper", None)
        log.outcome = task_outcome(compute_facts(sim), task, handoff.done, sim.scene_params, start_positions)
        log.sim_seconds = float(sim.data.time)
        log.wall_seconds = round(time.perf_counter() - started, 2)
        return log
