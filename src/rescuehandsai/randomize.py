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


# Items must stay where two SO-101 arms can actually work. Measured: the right arm
# fails to reach a utensil beyond x approximately 0.45, and the mat/cup area sits
# between x 0.22 and 0.42, y 0.10 and 0.30.
REACH_X = (0.16, 0.42)
REACH_Y = (0.08, 0.32)
MAX_SHIFT = 0.035


def perturb_start(sim, seed: int, *, joint_sigma: float = 0.10, item_shift: float = MAX_SHIFT,
                  yaw_spread: float = 0.4, settle_steps: int = 30, max_scale: float = 1.0) -> dict:
    """Nudge arms and items after reset. Returns what was applied, for the episode log.

    The difficulty is itself random (a scale between 0.3 and 1): most demonstrations
    then start slightly off, a few start badly off, which is the spread a drifting
    policy actually meets. Items stay inside the measured reachable area, so a
    discarded attempt means the teacher failed, not that the scene was impossible.
    """
    rng = np.random.default_rng([seed, 991])
    scale = float(rng.uniform(0.3 * max_scale, max_scale))
    targets = {}
    for name in sim.names:
        low, high = sim.limits[name]
        delta = 0.0 if name.endswith("gripper") else float(rng.normal(0.0, joint_sigma * scale))
        targets[name] = float(np.clip(sim.home_targets[name] + delta, low + 1e-3, high - 1e-3))
    sim.set_start_pose(targets)

    moved = {}
    for item in SCENE_ITEMS:
        dx, dy = rng.normal(0.0, min(item_shift, MAX_SHIFT) * scale, size=2)
        yaw = float(rng.normal(0.0, yaw_spread * scale)) if item != "cup" else 0.0
        position = np.array(sim.privileged().objects[item].position)
        position[0] = float(np.clip(position[0] + dx, *REACH_X))
        position[1] = float(np.clip(position[1] + dy, *REACH_Y))
        sim.set_item_pose(item, position, yaw_delta=yaw)
        moved[item] = {"position": [round(v, 4) for v in position], "yaw_delta": round(yaw, 3)}
    sim.settle(settle_steps)
    return {"scale": round(scale, 3), "joint_sigma": joint_sigma, "item_shift": item_shift, "items": moved,
            "joint_offsets": {n: round(targets[n] - sim.home_targets[n], 4) for n in sim.names}}


# Heights and speeds, not contacts: MuJoCo only fills contacts after a step, and
# stepping the world before an episode measurably changes what the teacher then does
# (it cost two of ten successes when the runner did it).
UTENSIL_MAX_REST_HEIGHT = 0.02
STILL_SPEED = 0.01


def start_problems(facts, scene_params, *, upright_min_up_z: float = 0.97) -> str | None:
    """Why this starting state is not a valid dinner table, or None when it is fine.

    A perturbed start can leave the cup on the plate, an item balanced on another or
    something still rolling. The teacher would then be judged on an impossible scene,
    so such attempts are skipped instead of counted as teacher failures.
    """
    problems = []
    for item in SCENE_ITEMS:
        if facts.touching[item]:
            problems.append(f"{item} starts in a gripper")
        if facts.in_zone[item] is not None:
            problems.append(f"{item} starts inside {facts.in_zone[item]}")
        if facts.speed[item] > STILL_SPEED or facts.angular_speed[item] > 10 * STILL_SPEED:
            problems.append(f"{item} is still moving")
    if facts.up_z["cup"] < upright_min_up_z:
        problems.append("cup does not start upright")
    if abs(facts.height["cup"] - scene_params.cup_half_height) > 0.008:
        problems.append("cup does not start flat on the table")
    for utensil in (i for i in SCENE_ITEMS if i != "cup"):
        if facts.height[utensil] > UTENSIL_MAX_REST_HEIGHT:
            problems.append(f"{utensil} is not flat on the table")
    return "; ".join(problems) or None


def yaw_quaternion(yaw: float):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def quaternion_product(a, b):
    result = np.empty(4)
    mujoco.mju_mulQuat(result, np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return result
