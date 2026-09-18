"""One pick episode: simulation, policy, per-physics-step facts and the judge (spec §5, §6).

Errors are labelled by where they come from: reset, observation and physics are the
simulator (invalid SIM_ERROR); policy calls and actions are the robot (valid failures)."""
from dataclasses import asdict
import time

import mujoco

from .control import validate_action
from .pick_cells import PickTask, cell_params
from .pick_config import derive_steps, load_contacts, load_rules
from .pick_contacts import ContactClassifier
from .pick_facts import FactReader, SimulatorFailure
from .pick_identity import config_hash, scene_hash, settings_hash, settings_record
from .pick_outcome import PickJudge
from .pick_records import InvalidRun
from .runner import clamp_action
from .scene import sample_params


class PickEpisodeRunner:
    def __init__(self, sim, policy, *, rules: dict | None = None, contacts: dict | None = None):
        self.sim, self.policy = sim, policy
        self.rules = rules if rules is not None else load_rules()
        self.contacts = contacts if contacts is not None else load_contacts()
        if sim.physics_version != self.rules["physics_version"]:
            raise InvalidRun("CONTRACT_MISMATCH", f"simulation physics_version {sim.physics_version} "
                                                  f"!= rules physics_version {self.rules['physics_version']}")
        self.steps = derive_steps(self.rules, sim.config["physics_dt"], sim.config["control_dt"])

    def run(self, task: PickTask) -> dict:
        sim, policy, steps = self.sim, self.policy, self.steps
        if task.physics_version != sim.physics_version:
            raise InvalidRun("CONTRACT_MISMATCH", f"task physics_version {task.physics_version} "
                                                  f"!= simulation {sim.physics_version}")
        record = {"seed": task.seed, "cell": task.cell, "utensil": task.utensil, "spare": task.spare,
                  "template": task.template, "instruction": task.instruction,
                  "physics_version": sim.physics_version, "policy": policy.metadata(), "rules": self.rules,
                  "clamped_joint_steps": 0, "timing": {"observe_s": [], "inference_s": []}}
        counters = {"physics": 0, "control": 0}
        judge = None

        def partial():
            out = dict(record, physics_steps=counters["physics"], control_steps=counters["control"])
            if judge is not None and judge.start is not None:
                out["outcome"] = asdict(judge.finish("crash", physics_step=counters["physics"],
                                                     control_step=counters["control"]))
            return out

        def simulator(what, fn):
            try:
                return fn()
            except InvalidRun:
                raise
            except Exception as exc:
                raise InvalidRun("SIM_ERROR", f"{what}: {type(exc).__name__}: {exc}", partial()) from exc

        def reset():
            params = cell_params(sample_params(sim.scene_config, task.seed), task.cell, sim.scene_config)
            sim.reset(task.seed, instruction=task.instruction, params=params)
            return params

        params = simulator("reset", reset)
        record.update(scene=settings_record(params), settings_sha256=settings_hash(params),
                      scene_sha256=scene_hash(params, sim.scene_config, sim.physics_version),
                      config_sha256=config_hash(sim.physics_version, sim.asset_path))
        reader = FactReader(sim, ContactClassifier(sim.model, task.utensil, self.contacts))
        table = sim.scene_config["table"]
        judge = PickJudge(task.utensil, task.spare, self.rules, steps, table["center"], table["half_size"])
        judge.start_from(simulator("read start", lambda: reader.read(0, 0)))
        stop = None
        try:
            policy.reset(sim, task)
        except Exception as exc:
            judge.record_error("POLICY_ERROR", 0, 0, f"reset: {type(exc).__name__}: {exc}")
            stop = "early_stop"
        severe = []

        def on_substep():
            # Stop at the physics step that decides the episode: success or a severe violation.
            counters["physics"] += 1
            severe.extend(judge.update(reader.read(counters["physics"], counters["control"])))
            return judge.succeeded or bool(severe)

        while stop is None and counters["control"] < steps.deadline_control:
            try:
                images = policy.wants_images()
            except Exception as exc:
                judge.record_error("POLICY_ERROR", counters["physics"], counters["control"],
                                   f"wants_images: {type(exc).__name__}: {exc}")
                stop = "early_stop"
                break
            t0 = time.perf_counter()
            obs = simulator("observe", lambda: sim.observe(images=images))
            record["timing"]["observe_s"].append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            try:
                raw = policy.act(obs)
            except Exception as exc:
                judge.record_error("POLICY_ERROR", counters["physics"], counters["control"],
                                   f"act: {type(exc).__name__}: {exc}")
                stop = "early_stop"
                break
            if images:
                record["timing"]["inference_s"].append(time.perf_counter() - t0)
            try:
                action, clamped = clamp_action(raw, sim.previous, sim.limits, sim.config["max_command_delta"])
                # clamp_action only fixes target values; it passes the policy's timestamp through.
                # sim.step checks the whole command again, so run that same check here: a malformed
                # or stale timestamp is the policy's fault (valid INVALID_ACTION), not a harness crash.
                validate_action(action, sim.names, sim.limits, sim.previous,
                                now=float(sim.data.time), max_age=sim.config["max_action_age"],
                                max_delta=sim.config["max_command_delta"])
            except (ValueError, AttributeError, TypeError) as exc:
                judge.record_error("INVALID_ACTION", counters["physics"], counters["control"], str(exc))
                stop = "early_stop"
                break
            record["clamped_joint_steps"] += clamped
            counters["control"] += 1
            try:
                sim.step(action, on_substep=on_substep, stop_on_cross_arm=False)
            except SimulatorFailure as exc:
                raise InvalidRun("SIM_ERROR", str(exc), partial()) from exc
            except mujoco.FatalError as exc:
                raise InvalidRun("SIM_ERROR", f"MuJoCo fatal error: {exc}", partial()) from exc
            except RuntimeError as exc:
                if str(exc).startswith("SIMULATION_ERROR"):
                    raise InvalidRun("SIM_ERROR", str(exc), partial()) from exc
                raise
            if judge.succeeded:
                stop = "success"
            elif severe:
                stop = "early_stop"
        stop = stop or "deadline"
        outcome = judge.finish(stop, physics_step=counters["physics"], control_step=counters["control"])
        record.update(success=outcome.success, outcome=asdict(outcome), control_steps=counters["control"],
                      physics_steps=counters["physics"], other_warnings=dict(reader.other_warnings))
        return record
