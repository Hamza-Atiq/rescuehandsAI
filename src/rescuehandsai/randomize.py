"""Off-nominal starting states, so demonstrations cover more than the perfect path.

A policy trained only on flawless teacher runs sees one narrow band of states. The
moment it drifts — a slightly late grasp, a nudged cup — it is in a state no
demonstration ever visited, and the error grows. Measured on our own checkpoint:
prediction error against the teacher stays ~0.006-0.013 rad over a whole 50-step
chunk on dataset states, yet closed-loop runs still fail, which is the signature of
that state gap rather than a weak model.

These perturbations move the start into the band the policy actually visits: arms
nudged off the home pose, items shifted and spun on the table. The scripted teacher
then solves the task from there, so the recorded demonstration teaches recovery from
an off-track state without any change to the task, the success rule or the contract.
"""
import mujoco
import numpy as np

from .scene import SCENE_ITEMS


def perturb_start(sim, seed: int, *, joint_sigma: float = 0.10, item_shift: float = 0.05,
                  yaw_spread: float = 0.5, settle_steps: int = 30) -> dict:
    """Nudge arms and items after reset. Returns what was applied, for the episode log."""
    rng = np.random.default_rng([seed, 991])
    targets = {}
    for name in sim.names:
        low, high = sim.limits[name]
        delta = 0.0 if name.endswith("gripper") else float(rng.normal(0.0, joint_sigma))
        targets[name] = float(np.clip(sim.home_targets[name] + delta, low + 1e-3, high - 1e-3))
    sim.set_start_pose(targets)

    moved = {}
    half_x, half_y = sim.scene_config["table"]["half_size"]
    centre_x, centre_y = sim.scene_config["table"]["center"]
    for item in SCENE_ITEMS:
        dx, dy = rng.normal(0.0, item_shift, size=2)
        yaw = float(rng.normal(0.0, yaw_spread)) if item != "cup" else 0.0
        position = np.array(sim.privileged().objects[item].position)
        position[0] = float(np.clip(position[0] + dx, centre_x - half_x + 0.05, centre_x + half_x - 0.05))
        position[1] = float(np.clip(position[1] + dy, centre_y - half_y + 0.05, centre_y + half_y - 0.05))
        sim.set_item_pose(item, position, yaw_delta=yaw)
        moved[item] = {"position": [round(v, 4) for v in position], "yaw_delta": round(yaw, 3)}
    sim.settle(settle_steps)
    return {"joint_sigma": joint_sigma, "item_shift": item_shift, "items": moved,
            "joint_offsets": {n: round(targets[n] - sim.home_targets[n], 4) for n in sim.names}}


def start_problems(facts, scene_params, *, upright_min_up_z: float = 0.97) -> str | None:
    """Why this starting state is not a valid dinner table, or None when it is fine.

    A perturbed start can put the cup on the plate or leave an item balanced on
    another: the teacher would then be judged on an impossible scene, so such
    attempts are skipped instead of counted as teacher failures.
    """
    problems = []
    for item in SCENE_ITEMS:
        if not facts.supported[item]:
            problems.append(f"{item} is not resting on the table")
        if facts.held_by[item] or facts.touching[item]:
            problems.append(f"{item} starts in a gripper")
        if facts.in_zone[item] is not None:
            problems.append(f"{item} starts inside {facts.in_zone[item]}")
    if facts.up_z["cup"] < upright_min_up_z:
        problems.append("cup does not start upright")
    if abs(facts.height["cup"] - scene_params.cup_half_height) > 0.008:
        problems.append("cup does not start flat on the table")
    return "; ".join(problems) or None


def yaw_quaternion(yaw: float):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def quaternion_product(a, b):
    result = np.empty(4)
    mujoco.mju_mulQuat(result, np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return result
