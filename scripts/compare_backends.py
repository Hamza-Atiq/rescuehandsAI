"""Do the three ways of running our policy agree on one real observation? (.venv-pai)

  PYTHONPATH=src .venv-pai/Scripts/python.exe scripts/compare_backends.py \
      --checkpoint models/smolvla_rescuehands --export models/openvino/fp32 [--device GPU]

Compared on the same dinner-scene observation (seed 0) and the same starting noise:

  native   LeRobot SmolVLA in PyTorch, the code that trained the checkpoint
  torch    Intel Physical AI Studio SmolVLA in PyTorch
  openvino the exported model Intel runs on the CPU or iGPU

Physical AI Studio starts its denoising from zeros (`use_random_input_noise=False`),
so the native run is given zero noise too; otherwise the two differ by design and a
comparison would say nothing. Differences are reported in radians per joint command.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from rescuehandsai.policies.smolvla_exported import CAMERA_SLOTS, to_chw_float
from rescuehandsai.scene import ROOT
from rescuehandsai.sim import MujocoSimulation
from rescuehandsai.task import make_task

CAMERA_KEYS = {camera: f"observation.images.{slot}" for slot, camera in CAMERA_SLOTS.items()}


def observation(seed: int):
    sim = MujocoSimulation()
    task = make_task(seed)
    sim.reset(seed, task.instruction)
    obs = sim.observe(images=True)
    names = tuple(sim.names)
    sim.close()
    return obs, names


def native_chunk(checkpoint: Path, obs, names):
    import torch
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(str(checkpoint)).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint),
                                         preprocessor_overrides={"device_processor": {"device": "cpu"}})
    batch = {"observation.state": torch.tensor([[obs.positions[n] for n in names]], dtype=torch.float32),
             "task": [obs.instruction]}
    for camera, key in CAMERA_KEYS.items():
        batch[key] = torch.from_numpy(to_chw_float(obs.images[camera]))
    noise = torch.zeros(1, policy.config.chunk_size, policy.config.max_action_dim)
    started = time.perf_counter()
    with torch.no_grad():
        chunk = post(policy.predict_action_chunk(pre(batch), noise=noise))
    return np.asarray(chunk.squeeze(0)), round(time.perf_counter() - started, 2)


def studio_chunk(checkpoint: Path, obs, names):
    import torch
    from physicalai.policies.smolvla import SmolVLA

    policy = SmolVLA(pretrained_name_or_path=str(checkpoint)).eval()
    batch = {"state": torch.tensor([[obs.positions[n] for n in names]], dtype=torch.float32),
             "task": [obs.instruction]}
    for camera, key in CAMERA_KEYS.items():
        batch[key] = torch.from_numpy(to_chw_float(obs.images[camera]))
    started = time.perf_counter()
    with torch.inference_mode():
        chunk = policy.predict_action_chunk(batch)
    return np.asarray(chunk).reshape(-1, len(names)), round(time.perf_counter() - started, 2)


def exported_chunk(export: Path, obs, names, device: str):
    from rescuehandsai.policies.smolvla_exported import ExportedSmolVLAPolicy

    policy = ExportedSmolVLAPolicy(export, names, device=device)
    inputs = {"state": np.array([[obs.positions[n] for n in names]], dtype=np.float32),
              "task": [obs.instruction]}
    for camera, key in policy._image_keys.items():
        inputs[key] = to_chw_float(obs.images[camera])
    started = time.perf_counter()
    chunk = np.asarray(policy.model.predict_action_chunk(inputs))
    return chunk.reshape(-1, len(names)), round(time.perf_counter() - started, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--device", default="GPU", help="OpenVINO device for the exported model")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--executed-actions", type=int, default=25, help="actions the runner really executes")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "backend_agreement.json")
    args = parser.parse_args()

    obs, names = observation(args.seed)
    chunks, timings = {}, {}
    for label, fn in (("native", lambda: native_chunk(args.checkpoint, obs, names)),
                      ("torch", lambda: studio_chunk(args.checkpoint, obs, names)),
                      ("openvino", lambda: exported_chunk(args.export, obs, names, args.device))):
        chunk, seconds = fn()
        chunks[label], timings[label] = chunk[:args.executed_actions], seconds
        print(f"{label}: chunk {chunk.shape} in {seconds}s", flush=True)

    report = {"seed": args.seed, "instruction": obs.instruction, "joints": list(names),
              "executed_actions": args.executed_actions, "first_call_seconds": timings,
              "openvino_device": args.device, "noise": "zeros in every backend",
              "pairs": {}}
    for a, b in (("native", "torch"), ("torch", "openvino"), ("native", "openvino")):
        diff = np.abs(chunks[a] - chunks[b])
        report["pairs"][f"{a}_vs_{b}"] = {"max_abs_rad": round(float(diff.max()), 5),
                                          "mean_abs_rad": round(float(diff.mean()), 5)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["pairs"], indent=2))


if __name__ == "__main__":
    main()
