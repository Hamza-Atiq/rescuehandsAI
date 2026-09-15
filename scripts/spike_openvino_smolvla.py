"""SPIKE (throwaway evidence): can pretrained SmolVLA export to OpenVINO and run
on this Intel CPU/iGPU through Physical AI Studio? Run with .venv-pai.

Not a benchmark of our trained policy. Findings go to docs/research.
"""
import json
import platform
import time
import traceback
from pathlib import Path

import numpy as np
import openvino as ov
import torch

OUT = Path("artifacts/spike_openvino")
OUT.mkdir(parents=True, exist_ok=True)
report = {"torch": torch.__version__, "openvino": ov.__version__,
          "cpu": platform.processor(),
          "devices": {d: ov.Core().get_property(d, "FULL_DEVICE_NAME") for d in ov.Core().available_devices}}


def log(**kw):
    report.update(kw)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(kw, default=str), flush=True)


try:
    from physicalai.policies.smolvla import SmolVLA
    from physicalai.inference.model import InferenceModel

    t = time.perf_counter()
    policy = SmolVLA(pretrained_name_or_path="lerobot/smolvla_base")
    policy.eval()
    log(load_s=round(time.perf_counter() - t, 1),
        inputs_schema=[(f.name, f.shape, str(f.dtype)) for f in policy.inputs_schema or []],
        outputs_schema=[(f.name, f.shape) for f in policy.outputs_schema or []])

    sample = policy.sample_input
    t = time.perf_counter()
    policy.export(OUT / "openvino", backend="openvino")
    log(export_openvino_s=round(time.perf_counter() - t, 1),
        files=sorted(p.name for p in (OUT / "openvino").rglob("*"))[:30])

    # Build a numpy observation from the schema.
    obs = {}
    for f in policy.inputs_schema:
        if str(f.dtype).endswith("STRING"):
            obs[f.name] = ["pick up the fork and pass it to the left hand"]
        else:
            obs[f.name] = np.random.default_rng(0).random((1, *f.shape), dtype=np.float32)

    for device in ["CPU", "GPU"]:
        if device not in ov.Core().available_devices:
            continue
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
            log(**{f"openvino_{device}": {"load_and_first_s": round(load_first, 2),
                                          "mean_s": round(float(np.mean(times)), 3),
                                          "chunk_shape": list(first.shape)}})
        except Exception as error:  # record and continue to the next device
            log(**{f"openvino_{device}_error": f"{type(error).__name__}: {error}"})

    # PyTorch CPU baseline through the same policy object.
    try:
        batch = {k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v) for k, v in sample.items()}
        with torch.inference_mode():
            policy.predict_action_chunk(batch)
            times = []
            for _ in range(3):
                t = time.perf_counter()
                policy.predict_action_chunk(batch)
                times.append(time.perf_counter() - t)
        log(pytorch_cpu_mean_s=round(float(np.mean(times)), 3))
    except Exception as error:
        log(pytorch_cpu_error=f"{type(error).__name__}: {error}", pytorch_trace=traceback.format_exc()[-2000:])
except Exception as error:
    log(fatal=f"{type(error).__name__}: {error}", trace=traceback.format_exc()[-3000:])
