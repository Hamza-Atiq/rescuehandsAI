"""Read-only Kaggle hardware/checkpoint check. Does not train, upload, or delete.

Run with .venv-train/bin/python scripts/review_kaggle_preflight.py
  --checkpoint outputs/smolvla_smoke/checkpoints/last/pretrained_model
"""
import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from safetensors import safe_open

    report = {"packages": {}, "gpus": [], "checkpoint": str(args.checkpoint)}
    for name in ("torch", "lerobot", "accelerate", "transformers"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = "missing"
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            report["gpus"].append({"id": i, "name": torch.cuda.get_device_name(i),
                "capability": list(torch.cuda.get_device_capability(i)),
                "native_bf16": torch.cuda.is_bf16_supported(including_emulation=False),
                "memory_gib": round(torch.cuda.get_device_properties(i).total_memory/2**30, 2)})
    cfg_path = args.checkpoint / "config.json"
    if not cfg_path.is_file():
        raise SystemExit(f"Checkpoint config missing: {cfg_path}")
    cfg = json.loads(cfg_path.read_text())
    report["input_features"] = cfg.get("input_features")
    report["output_features"] = cfg.get("output_features")
    report["state_shape"] = cfg.get("input_features", {}).get("observation.state", {}).get("shape")
    report["action_shape"] = cfg.get("output_features", {}).get("action", {}).get("shape")
    report["bimanual_shapes_match"] = report["state_shape"] == report["action_shape"] == [12]
    weight_path = args.checkpoint / "model.safetensors"
    if weight_path.exists():
        with safe_open(str(weight_path), framework="pt", device="cpu") as weights:
            # Reads tensor headers, not all model data.
            report["weight_tensor_dtypes"] = dict(Counter(weights.get_slice(k).get_dtype() for k in weights.keys()))
    report["normalizer_files"] = sorted(p.name for p in args.checkpoint.glob("*normalizer*.safetensors"))
    report["scope"] = "hardware and saved metadata only; no CUDA training or policy-quality test"
    print(json.dumps(report, indent=2))
    return 0 if report["bimanual_shapes_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
