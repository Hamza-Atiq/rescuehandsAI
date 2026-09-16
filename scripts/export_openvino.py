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

from rescuehandsai.contract import CAMERA_SLOTS, build_contract, check_contract, file_sha256, load_contract, write_contract
from rescuehandsai.sim import MujocoSimulation


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (args.out / "manifest.json").exists():
        parser.error(f"{args.out} already contains an export")

    sim = MujocoSimulation()
    contract = load_contract(Path(args.checkpoint))  # local checkpoints only; refuses an unlabelled model
    check_contract(contract, sim.names, cameras=CAMERA_SLOTS, control_hz=1 / sim.config["control_dt"])

    from physicalai.policies.smolvla import SmolVLA
    started = time.perf_counter()
    policy = SmolVLA(pretrained_name_or_path=args.checkpoint).eval()
    schema = {f.name: tuple(f.shape) for f in policy.inputs_schema or []}
    outputs = {f.name: tuple(f.shape) for f in policy.outputs_schema or []}
    joints = len(sim.names)
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
    exported = build_contract(contract["joint_order"], dataset=contract["dataset"],
                              dataset_revision=contract.get("dataset_revision"),
                              control_hz=contract["control_hz"], source=str(args.checkpoint),
                              files={name: file_sha256(args.out / name) for name in ("smolvla.xml", "smolvla.bin")})
    write_contract(args.out, exported)
    report = {"checkpoint": args.checkpoint, "inputs": schema, "outputs": outputs,
              "export_seconds": round(time.perf_counter() - started, 1),
              "files": sorted(p.name for p in args.out.iterdir()), "contract": exported}
    (args.out / "export_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
