"""SmolVLA exported by Intel Physical AI Studio, run on OpenVINO (CPU/GPU) or Torch.

Policy-visible inputs only: three camera images, 12 joint positions, instruction.
Run in `.venv-pai`. The camera slot mapping must match training (see
training/kaggle_pipeline.py CAMERA_RENAME).
"""
import json
import time
from collections import deque
from pathlib import Path

import numpy as np

from ..contract import CAMERA_SLOTS, check_contract, load_contract
from ..contracts import BimanualAction


def to_chw_float(image: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 -> float32 1x3xHxW in [0, 1], as LeRobot datasets provide."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected uint8 HxWx3 image, got {image.dtype} {image.shape}")
    return (image.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]


class ExportedSmolVLAPolicy:
    uses_privileged_state = False

    def __init__(self, export_dir: Path, joint_names, *, device: str = "GPU", n_action_steps: int = 25):
        from physicalai.inference.model import InferenceModel  # .venv-pai only

        if isinstance(n_action_steps, bool) or not isinstance(n_action_steps, int) or n_action_steps < 1:
            raise ValueError(f"n_action_steps must be a positive integer, got {n_action_steps!r}")
        self.export_dir = Path(export_dir)
        self.contract = load_contract(self.export_dir, verify=True)  # refuses files changed since export
        check_contract(self.contract, joint_names, cameras=CAMERA_SLOTS)
        manifest = json.loads((self.export_dir / "manifest.json").read_text())
        self.manifest = manifest
        self.model = InferenceModel(self.export_dir, device=device)
        self.device = device
        self.joint_names = tuple(joint_names)
        self.n_action_steps = n_action_steps
        self.name = f"smolvla_{self.model.backend}_{device.lower()}"
        names = {f.name for f in self.model.input_features} if hasattr(self.model, "input_features") else None
        self._image_keys = self._resolve_image_keys(names)
        self._queue = deque()

    @staticmethod
    def _resolve_image_keys(names):
        keys = {}
        for slot, camera in CAMERA_SLOTS.items():
            candidates = [f"images.images.{slot}", f"images.{slot}", f"observation.images.{slot}"]
            found = next((c for c in candidates if names is None or c in names), None)
            if found is None:
                raise ValueError(f"exported model has no input for {slot}; inputs: {sorted(names)}")
            keys[camera] = found
        return keys

    def reset(self, sim, task):
        self._queue.clear()
        self.model.reset()
        check_contract(self.contract, sim.names, cameras=CAMERA_SLOTS,
                       control_hz=1 / sim.config["control_dt"])

    def wants_images(self) -> bool:
        return not self._queue

    def act(self, obs):
        if not self._queue:
            if not obs.images:
                raise RuntimeError("POLICY_ERROR: inference needs camera images")
            inputs = {"state": np.array([[obs.positions[n] for n in self.joint_names]], dtype=np.float32),
                      "task": [obs.instruction]}
            for camera, key in self._image_keys.items():
                inputs[key] = to_chw_float(obs.images[camera])
            chunk = np.asarray(self.model.predict_action_chunk(inputs))
            if chunk.ndim != 2 or chunk.shape[0] == 0 or chunk.shape[1] != len(self.joint_names):
                raise RuntimeError(f"POLICY_ERROR: action chunk shape {chunk.shape}")
            if not np.isfinite(chunk).all():
                raise RuntimeError("POLICY_ERROR: non-finite actions")
            self._queue.extend(chunk[: self.n_action_steps])
        values = self._queue.popleft()
        return BimanualAction(obs.timestamp, dict(zip(self.joint_names, map(float, values))))

    def after_recovery(self, sim, task, progress):
        self._queue.clear()
        self.model.reset()

    def metadata(self):
        return {"name": self.name, "backend": self.model.backend, "device": self.device,
                "checkpoint": str(self.export_dir), "n_action_steps": self.n_action_steps,
                "privileged_state": False, "camera_slots": CAMERA_SLOTS, "contract": self.contract}
