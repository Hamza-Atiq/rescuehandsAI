"""Intel benchmark for the exported SmolVLA policy (run in .venv-pai).

  PYTHONPATH=src .venv-pai/Scripts/python.exe scripts/benchmark_intel.py \
      --export models/openvino/fp32 [--torch-checkpoint outputs/.../pretrained_model] [--runs 5]

Uses a real observation from the dinner scene (seed 0), not random tensors.
Variants: OpenVINO FP32 on CPU, FP32 and FP16 execution on the iGPU, INT8
weight-compressed (NNCF) on CPU and iGPU, model caching, and optionally the
PyTorch CPU policy as the unoptimized baseline. For each: load time, first
call, mean/p95 latency, process memory and the action difference from the
FP32 CPU reference, so speed-ups are never reported without their accuracy cost.
"""
import argparse
import hashlib
from datetime import datetime, timezone
import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import openvino as ov
import psutil

from rescuehandsai.contract import CAMERA_SLOTS
from rescuehandsai.policies.smolvla_exported import to_chw_float
from rescuehandsai.scene import ROOT
from rescuehandsai.sim import MujocoSimulation
from rescuehandsai.task import make_task

EXECUTED_ACTIONS = 25  # the runner executes the first 25 actions of each chunk


def real_inputs(image_keys, joint_count_expected):
    sim = MujocoSimulation()
    task = make_task(0)
    sim.reset(0, task.instruction)
    obs = sim.observe(images=True)
    sim.close()
    state = np.array([[obs.positions[n] for n in sim.names]], dtype=np.float32)
    truncated = joint_count_expected is not None and state.shape[1] != joint_count_expected
    if truncated:
        state = state[:, :joint_count_expected]  # pretrained base model: 6-D (spike only), recorded
    inputs = {"state": state, "task": [obs.instruction], "_truncated": bool(truncated)}
    for camera, key in image_keys.items():
        inputs[key] = to_chw_float(obs.images[camera])
    return inputs


INT8_MODE = "INT8_ASYM"


def model_digest(export_dir: Path) -> str:
    """SHA-256 of the exported graph and weights, so derived models are tied to their source."""
    h = hashlib.sha256()
    for name in ("smolvla.xml", "smolvla.bin"):
        with open(export_dir / name, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 22), b""):
                h.update(block)
    return h.hexdigest()


INT8_DONE = ".complete"


def int8_complete(fp32_dir: Path, int8_dir: Path) -> bool:
    """A cached INT8 model counts only if compression finished and every file it needs is there."""
    needed = ["smolvla.xml", "smolvla.bin"] + [n for n in ("manifest.json", "tokenizer.xml", "tokenizer.bin")
                                              if (fp32_dir / n).exists()]
    return (int8_dir / INT8_DONE).is_file() and all((int8_dir / n).is_file() for n in needed)


def make_int8(fp32_dir: Path, int8_dir: Path):
    """Compress once per (source model, mode); int8_dir is named after both."""
    import nncf
    if int8_complete(fp32_dir, int8_dir):
        return
    if int8_dir.exists():  # an interrupted earlier compression: start again, never reuse half a model
        shutil.rmtree(int8_dir)
    int8_dir.mkdir(parents=True, exist_ok=True)
    model = ov.Core().read_model(fp32_dir / "smolvla.xml")
    compressed = nncf.compress_weights(model, mode=getattr(nncf.CompressWeightsMode, INT8_MODE))
    ov.save_model(compressed, int8_dir / "smolvla.xml")
    for name in ("manifest.json", "tokenizer.xml", "tokenizer.bin"):
        if (fp32_dir / name).exists():
            shutil.copy2(fp32_dir / name, int8_dir / name)
    (int8_dir / INT8_DONE).write_text("ok")


REFERENCE = "cpu_fp32"


def ordered_variants(names):
    """The accuracy reference always runs, and first, so every row has an accuracy number."""
    names = [n for n in names if n]
    if REFERENCE not in names:
        names.append(REFERENCE)
    return sorted(names, key=lambda n: n != REFERENCE)


def time_model(label, factory, inputs, runs):
    from physicalai.inference.model import InferenceModel  # noqa: F401 (import cost excluded)
    proc = psutil.Process()
    rss_before = proc.memory_info().rss
    t = time.perf_counter()
    model = factory()
    load_s = time.perf_counter() - t
    predict = getattr(model, "predict_action_chunk", model)  # InferenceModel returns the action chunk
    t = time.perf_counter()
    first = np.asarray(predict(inputs))
    first_s = time.perf_counter() - t
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        out = np.asarray(predict(inputs))
        times.append(time.perf_counter() - t)
    row = {"variant": label, "load_s": round(load_s, 2), "first_call_s": round(first_s, 2),
           "mean_s": round(float(np.mean(times)), 3), "p95_s": round(float(np.percentile(times, 95)), 3),
           "runs": runs, "rss_delta_mb": round((proc.memory_info().rss - rss_before) / 2**20, 1),
           "chunk_shape": list(out.shape)}
    del model
    return row, out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", type=Path, required=True, help="FP32 OpenVINO export directory")
    parser.add_argument("--torch-checkpoint", type=Path, help="LeRobot SmolVLA checkpoint for the PyTorch row")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--variants", default="cpu_fp32,gpu_fp32,gpu_fp16,cpu_int8,gpu_int8,gpu_fp16_cached")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    from physicalai.inference.model import InferenceModel

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = args.out or ROOT / "results" / f"benchmark_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    core = ov.Core()
    from rescuehandsai.contract import load_contract
    contract = load_contract(args.export, verify=True)  # time only the exact artifact the contract vouches for
    manifest = json.loads((args.export / "manifest.json").read_text())
    features = [f.get("init_args", f) for f in manifest["model"].get("input_features", [])]
    feature_names = {f["name"] for f in features} or None
    image_keys = {}
    for slot, camera in CAMERA_SLOTS.items():
        for key in (f"images.images.{slot}", f"images.{slot}", f"observation.images.{slot}"):
            if feature_names is None or key in feature_names:
                image_keys[camera] = key
                break
    state_dims = next((f["shape"][0] for f in features if f["name"] == "state"), None)
    inputs = real_inputs(image_keys, state_dims)
    truncated = inputs.pop("_truncated")
    source_sha256 = model_digest(args.export)

    wanted = ordered_variants(args.variants.split(","))
    int8_dir = (out.parent / f"openvino_int8_{INT8_MODE.lower()}_{source_sha256[:16]}"
                if "int8" in args.variants else None)
    if int8_dir:
        t = time.perf_counter()
        make_int8(args.export, int8_dir)
        compress_s = round(time.perf_counter() - t, 1)
    cache = out / "ov_cache"
    variants = {
        "cpu_fp32": lambda: InferenceModel(args.export, device="CPU", PERFORMANCE_HINT="LATENCY"),
        "gpu_fp32": lambda: InferenceModel(args.export, device="GPU", INFERENCE_PRECISION_HINT="f32"),
        "gpu_fp16": lambda: InferenceModel(args.export, device="GPU", INFERENCE_PRECISION_HINT="f16"),
        "cpu_int8": lambda: InferenceModel(int8_dir, device="CPU", PERFORMANCE_HINT="LATENCY"),
        "gpu_int8": lambda: InferenceModel(int8_dir, device="GPU", INFERENCE_PRECISION_HINT="f16"),
        "gpu_fp16_cached": lambda: InferenceModel(args.export, device="GPU", INFERENCE_PRECISION_HINT="f16",
                                                  CACHE_DIR=str(cache)),
    }
    rows, reference = [], None
    for name in wanted:
        if name.startswith("gpu") and "GPU" not in core.available_devices:
            rows.append({"variant": name, "skipped": "no OpenVINO GPU device"})
            continue
        try:
            if name == "gpu_fp16_cached":  # warm the cache once, then time a fresh load
                variants[name]()
            row, actions = time_model(name, variants[name], inputs, args.runs)
            actions = actions.reshape(-1, actions.shape[-1])[:EXECUTED_ACTIONS]
            if reference is None and name == REFERENCE:
                reference = actions
            if reference is not None:
                diff = np.abs(actions - reference)
                row["max_abs_action_diff_rad"] = round(float(diff.max()), 5)
                row["mean_abs_action_diff_rad"] = round(float(diff.mean()), 5)
        except Exception as exc:  # record the failure; never drop a row silently
            row = {"variant": name, "error": f"{type(exc).__name__}: {exc}"[:500]}
        rows.append(row)
        print(json.dumps(row), flush=True)

    if args.torch_checkpoint:
        try:
            import torch
            from physicalai.data.observation import Observation
            from physicalai.policies.smolvla import SmolVLA
            # The Intel PyTorch policy takes its own Observation, not the flat dict the
            # exported runtime accepts; Observation.to_dict() produces images.images.cameraN.
            batch = Observation(
                state=torch.from_numpy(inputs["state"]), task=list(inputs["task"]),
                images={f"images.{slot}": torch.from_numpy(inputs[image_keys[CAMERA_SLOTS[slot]]])
                        for slot in CAMERA_SLOTS})

            def torch_factory():  # loading happens inside the timer and memory window, like OpenVINO
                policy = SmolVLA(pretrained_name_or_path=str(args.torch_checkpoint)).eval()

                def call(_):
                    with torch.inference_mode():
                        return policy.predict_action_chunk(batch).numpy()
                return call
            row, actions = time_model("cpu_pytorch", torch_factory, inputs, max(2, args.runs // 2))
            actions = actions.reshape(-1, actions.shape[-1])[:EXECUTED_ACTIONS]
            if reference is not None:
                diff = np.abs(actions - reference)
                row["max_abs_action_diff_rad"] = round(float(diff.max()), 5)
                row["mean_abs_action_diff_rad"] = round(float(diff.mean()), 5)
            rows.insert(0, row)
        except Exception as exc:
            rows.insert(0, {"variant": "cpu_pytorch", "error": f"{type(exc).__name__}: {exc}"[:500]})

    report = {
        "created_utc": stamp, "export": str(args.export), "export_sha256": source_sha256, "contract": contract,
        "int8_dir": str(int8_dir) if int8_dir else None, "int8_mode": INT8_MODE if int8_dir else None,
        "state_dims_model": state_dims, "state_truncated_for_model": truncated,
        "openvino": ov.__version__,
        "devices": {d: core.get_property(d, "FULL_DEVICE_NAME") for d in core.available_devices},
        "cpu": platform.processor(), "platform": platform.platform(),
        "observation": "dinner scene seed 0, three 256x256 cameras, real joint state and instruction",
        "int8_compression_s": compress_s if int8_dir else None, "rows": rows,
        "accuracy_reference": REFERENCE if reference is not None else None,
        "note": "Accuracy column compares the first 25 executed actions to OpenVINO FP32 on CPU "
                "(run first); load_s and rss_delta_mb include model construction for every row, PyTorch too.",
    }
    (out / "benchmark.json").write_text(json.dumps(report, indent=2))
    lines = ["| Variant | Load s | First call s | Mean s | p95 s | Max action diff (rad) |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        if "mean_s" in r:
            lines.append(f"| {r['variant']} | {r['load_s']} | {r['first_call_s']} | {r['mean_s']} | {r['p95_s']} | "
                         f"{r.get('max_abs_action_diff_rad', '-')} |")
        else:
            lines.append(f"| {r['variant']} | - | - | - | - | {r.get('error') or r.get('skipped')} |")
    (out / "benchmark.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
