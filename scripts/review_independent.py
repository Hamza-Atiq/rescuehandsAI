"""Independent review probes; does not change training or simulator code.

Run with .venv-sim/Scripts/python.exe -B scripts/review_independent.py.
These probes report observations, including known defects; they do not certify
the complete learned policy. Object pose edits are test fixtures only.
"""
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import mujoco
import numpy as np
from rescuehandsai.auditor import compute_facts
from rescuehandsai.contracts import BimanualAction
from rescuehandsai.evaluation import task_outcome
from rescuehandsai.perturb import GripperGlitch
from rescuehandsai.runner import EpisodeRunner, EpisodeLog, clamp_action
from rescuehandsai.sim import MujocoSimulation
from rescuehandsai.task import make_task


def factory_shapes():
    """Execute the installed make_policy with only expensive model loading replaced."""
    source = ROOT / ".venv-pai/Lib/site-packages/lerobot/policies/factory.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "make_policy")
    class Model:
        @classmethod
        def from_pretrained(cls, **kwargs):
            return cls()
        def to(self, device):
            return self
    ft = SimpleNamespace(ACTION="action", STATE="state")
    data = {"observation.state": SimpleNamespace(type=ft.STATE, shape=(12,)),
            "action": SimpleNamespace(type=ft.ACTION, shape=(12,))}
    ns = {"get_policy_class": lambda _: Model, "dataset_to_policy_features": lambda _: data,
          "FeatureType": ft, "torch": SimpleNamespace(nn=SimpleNamespace(Module=Model)),
          "ACTION": "action"}
    # Future annotations avoid importing unrelated policy families.
    code = compile("from __future__ import annotations\n" + ast.unparse(fn), str(source), "exec")
    exec(code, ns)
    cfg = SimpleNamespace(type="smolvla", device="cpu", use_peft=False, pretrained_path="base",
                          input_features={"observation.state": SimpleNamespace(type=ft.STATE, shape=(6,))})
    ns["make_policy"](cfg=cfg, ds_meta=SimpleNamespace(features={}), rename_map={"a": "b"})
    return {"input_state": list(cfg.input_features["observation.state"].shape),
            "output_action": list(cfg.output_features["action"].shape),
            "scope": "real installed factory; model loading stubbed"}


class Idle:
    def reset(self, sim, task): self.targets = dict(sim.home_targets)
    def wants_images(self): return False
    def act(self, obs): return BimanualAction(obs.timestamp, self.targets)
    def metadata(self): return {"name": "independent_idle_probe"}
    def after_recovery(self, sim, task, progress): pass


def main():
    report = {"factory_shapes": factory_shapes()}
    sim = MujocoSimulation()
    try:
        task = make_task(3, "fork")
        sim.reset(task.seed, task.instruction)
        # Put final task objects down, with the cup upside down. Let physics settle.
        for item, zone in (("cup", "cup_zone"), (task.utensil, "utensil_zone")):
            adr = int(sim.model.jnt_qposadr[sim.model.joint(item + "_free").id])
            z = sim.scene_params.cup_half_height if item == "cup" else .006
            sim.data.qpos[adr:adr+3] = (*sim.scene_config["zones"][zone]["pos"], z + .001)
            sim.data.qpos[adr+3:adr+7] = (0, 1, 0, 0) if item == "cup" else (1, 0, 0, 0)
        mujoco.mj_forward(sim.model, sim.data)
        for _ in range(30):
            sim.step(BimanualAction(sim.observe().timestamp, sim.home_targets))
        facts = compute_facts(sim)
        report["upside_down_cup"] = {
            "cup_local_up_dot_world_up": float(sim.data.body("cup").xmat.reshape(3, 3)[2, 2]),
            "outcome": task_outcome(facts, task, True, sim.scene_params),
            "scope": "real MuJoCo fixture; historical holders supplied to evaluator"}
        sim.reset(1)
        raw = BimanualAction(0, dict(sim.previous) | {"unexpected_joint": 0})
        safe, _ = clamp_action(raw, sim.previous, sim.limits, .15)
        report["unknown_joint_guard"] = {"extra_joint_silently_removed": "unexpected_joint" not in safe.targets}
        runner = EpisodeRunner(sim, Idle(), supervisor=True, stall_seconds=.05, max_steps=2)
        result = runner.run(make_task(1))
        report["recovery_budget_steps"] = {"configured_max_steps": 2, "actual_steps": result.steps,
                                            "failure": result.failure}
        sim.reset(1)
        glitch = GripperGlitch()
        glitch.reset(1)
        glitch.fired, glitch.arm, glitch._left = True, "right_arm", 7
        sim.set_actuator_fault("right_arm/gripper", .9)
        runner = EpisodeRunner(sim, Idle(), supervisor=True, fault=glitch)
        log = EpisodeLog(1, {}, "", True, "GripperGlitch")
        runner._safe_pose(log)
        report["fault_during_recovery"] = {"recovery_steps": log.steps,
            "remaining_fault_steps_before": 7, "remaining_after": glitch._left,
            "fault_still_active": "right_arm/gripper" in sim.faults}
        sim.set_actuator_fault("right_arm/gripper", None)
        sim.reset(1)
        custom = make_task(1)
        from dataclasses import replace
        custom = replace(custom, max_recoveries=0, timeout_s=.1)
        log = EpisodeRunner(sim, Idle(), supervisor=True, stall_seconds=.1, max_steps=4).run(custom)
        report["task_limits"] = {"task_max_recoveries": 0, "actual_recoveries": log.recoveries,
                                 "task_timeout_s": .1, "actual_sim_seconds": log.sim_seconds}
    finally:
        sim.close()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
