"""Independent smoke check of the existing SIX-joint spike, not the final VLA."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import openvino as ov
from physicalai.inference.model import InferenceModel
from rescuehandsai.sim import MujocoSimulation
from rescuehandsai.task import make_task
from rescuehandsai.policies.smolvla_exported import CAMERA_SLOTS, to_chw_float

sim = MujocoSimulation()
task = make_task(10)
sim.reset(task.seed, task.instruction)
obs = sim.observe(images=True)
names = sim.names
sim.close()
inputs = {"state": np.array([[obs.positions[n] for n in names[:6]]], dtype=np.float32),
          "task": [task.instruction]}
inputs.update({f"images.images.{slot}": to_chw_float(obs.images[camera])
               for slot, camera in CAMERA_SLOTS.items()})
t = time.perf_counter()
model = InferenceModel(ROOT / "artifacts/spike_openvino/openvino", device="GPU")
print(json.dumps({"stage": "loaded", "seconds": time.perf_counter()-t}), flush=True)
outputs, timings = [], []
for _ in range(2):
    t = time.perf_counter()
    outputs.append(model.predict_action_chunk(inputs))
    timings.append(time.perf_counter()-t)
print(json.dumps({"scope": "existing pretrained 6-joint spike; no trained bimanual checkpoint available",
    "device": ov.Core().get_property("GPU", "FULL_DEVICE_NAME"),
    "output_shape": list(outputs[0].shape), "finite": bool(all(np.isfinite(a).all() for a in outputs)),
    "repeat_max_abs_difference": float(np.max(np.abs(outputs[0]-outputs[1]))),
    "call_seconds": timings,
    "timing_caveat": "two smoke calls only, other review work active; not a controlled benchmark"}, indent=2))
