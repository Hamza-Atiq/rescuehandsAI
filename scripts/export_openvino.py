"""Export a fine-tuned SmolVLA checkpoint to OpenVINO with Intel Physical AI Studio.

  PYTHONPATH=src .venv-pai/Scripts/python.exe scripts/export_openvino.py \
      --checkpoint <hf_user>/smolvla_rescuehands  (or a local pretrained_model folder) \
      --out models/openvino/fp32

The checkpoint must have been trained on our dataset: 12-D state and action with
the joint order recorded in the dataset, and cameras renamed to camera1..3
(overhead, left wrist, right wrist). The script refuses anything else.
"""
import argparse
import json
import time
from pathlib import Path

from rescuehandsai.policies.smolvla_exported import CAMERA_SLOTS
from rescuehandsai.sim import MujocoSimulation


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (args.out / "manifest.json").exists():
        parser.error(f"{args.out} already contains an export")

    from physicalai.policies.smolvla import SmolVLA
    started = time.perf_counter()
    policy = SmolVLA(pretrained_name_or_path=args.checkpoint).eval()
    schema = {f.name: tuple(f.shape) for f in policy.inputs_schema or []}
    outputs = {f.name: tuple(f.shape) for f in policy.outputs_schema or []}
    joints = len(MujocoSimulation().names)
    problems = []
    if schema.get("state") != (joints,):
        problems.append(f"state shape {schema.get('state')} != ({joints},)")
    for slot in CAMERA_SLOTS:
        if not any(name.endswith(slot) for name in schema):
            problems.append(f"missing camera input {slot}")
    if outputs.get("action", (None, None))[1:] != (joints,):
        problems.append(f"action shape {outputs.get('action')} does not end in {joints}")
    if problems:
        raise SystemExit("Checkpoint does not match this task: " + "; ".join(problems))

    args.out.mkdir(parents=True, exist_ok=True)
    policy.export(args.out, backend="openvino")
    report = {"checkpoint": args.checkpoint, "inputs": schema, "outputs": outputs,
              "export_seconds": round(time.perf_counter() - started, 1),
              "files": sorted(p.name for p in args.out.iterdir())}
    (args.out / "export_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
