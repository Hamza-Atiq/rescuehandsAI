"""Record episodes in LeRobot dataset format (needs the ML environment).

Policy-visible data only: three camera images, 12 joint positions, the
instruction text, and the 12 joint targets that were commanded.
"""
from pathlib import Path

import numpy as np

from .sim import POLICY_CAMERAS

FPS = 20


def features(joint_names, height: int, width: int) -> dict:
    names = list(joint_names)
    feats = {
        "observation.state": {"dtype": "float32", "shape": (len(names),), "names": names},
        "action": {"dtype": "float32", "shape": (len(names),), "names": names},
    }
    for cam in POLICY_CAMERAS:
        feats[f"observation.images.{cam}"] = {"dtype": "video", "shape": (height, width, 3),
                                              "names": ["height", "width", "channels"]}
    return feats


def fps_for(control_dt: float) -> int:
    """Dataset frame rate from the simulator's control interval; it must be a whole number."""
    rate = 1.0 / control_dt
    if not abs(rate - round(rate)) < 1e-9 or round(rate) < 1:
        raise ValueError(f"control_dt {control_dt} gives a non-integer frame rate {rate}")
    return round(rate)


class EpisodeRecorder:
    def __init__(self, root: Path, repo_id: str, joint_names, height: int, width: int,
                 fps: int = FPS, vcodec: str = "libsvtav1"):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # ML env only

        self.joint_names = tuple(joint_names)
        self.dataset = LeRobotDataset.create(
            repo_id=repo_id, fps=fps, root=root, robot_type="so101_bimanual_sim",
            features=features(self.joint_names, height, width), use_videos=True,
            image_writer_threads=4, vcodec=vcodec)
        self.frames = 0

    def add(self, obs, action) -> None:
        if set(obs.images) != set(POLICY_CAMERAS):
            raise ValueError(f"Expected cameras {sorted(POLICY_CAMERAS)}, got {sorted(obs.images)}")
        frame = {
            "observation.state": np.array([obs.positions[n] for n in self.joint_names], dtype=np.float32),
            "action": np.array([action.targets[n] for n in self.joint_names], dtype=np.float32),
            "task": obs.instruction,
        }
        for cam in POLICY_CAMERAS:
            frame[f"observation.images.{cam}"] = obs.images[cam]
        self.dataset.add_frame(frame)
        self.frames += 1

    def save(self) -> None:
        self.dataset.save_episode()
        self.frames = 0

    def discard(self) -> None:
        self.dataset.clear_episode_buffer()
        self.frames = 0

    def finalize(self) -> None:
        self.dataset.finalize()
