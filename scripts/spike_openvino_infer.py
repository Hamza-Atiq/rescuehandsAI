"""SPIKE (throwaway evidence): time the exported pretrained SmolVLA on OpenVINO
CPU and iGPU. Uses artifacts/spike_openvino/openvino from spike_openvino_smolvla.py.
Pretrained base weights on random inputs: latency evidence only, not task skill.
"""
import json
import time
from pathlib import Path

import numpy as np
import openvino as ov
from physicalai.inference.model import InferenceModel

OUT = Path("artifacts/spike_openvino")
report_path = OUT / "report.json"
report = json.loads(report_path.read_text())
core = ov.Core()
rng = np.random.default_rng(0)
obs = {"state": rng.standard_normal((1, 6), dtype=np.float32),
       "task": ["pick up the fork and pass it to the left hand"]}
for cam in ("camera1", "camera2", "camera3"):
    obs[f"images.images.{cam}"] = rng.random((1, 3, 256, 256), dtype=np.float32)

for device in core.available_devices:
    key = f"openvino_{device}"
    try:
        t = time.perf_counter()
        model = InferenceModel(OUT / "openvino", device=device)
        first = model.predict_action_chunk(obs)
        load_first = time.perf_counter() - t
        times = []
        for _ in range(5):
            t = time.perf_counter()
            model.predict_action_chunk(obs)
            times.append(time.perf_counter() - t)
        report[key] = {"device_name": core.get_property(device, "FULL_DEVICE_NAME"),
                       "load_and_first_s": round(load_first, 2), "mean_s": round(float(np.mean(times)), 3),
                       "p95_s": round(float(np.percentile(times, 95)), 3), "runs": len(times),
                       "chunk_shape": list(first.shape)}
        report.pop(key + "_error", None)
    except Exception as error:
        report[key + "_error"] = f"{type(error).__name__}: {error}"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(key, report.get(key) or report.get(key + "_error"), flush=True)
