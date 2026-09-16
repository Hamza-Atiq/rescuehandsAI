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


EXPORT_FILES = ("smolvla.xml", "smolvla.bin", "tokenizer.xml", "tokenizer.bin", "manifest.json")


def write_export_contract(out: Path, contract: dict, checkpoint) -> dict:
    """Copy the checkpoint's verified contract to the export, hashing every export file."""
    missing = [name for name in ("smolvla.xml", "smolvla.bin", "manifest.json") if not (out / name).is_file()]
    if missing:
        raise SystemExit(f"export at {out} is incomplete: missing {missing}")
    exported = build_contract(contract["joint_order"], dataset=contract["dataset"],
                              dataset_revision=contract.get("dataset_revision"),
                              control_hz=contract["control_hz"], source=str(checkpoint),
                              files={name: file_sha256(out / name) for name in EXPORT_FILES if (out / name).is_file()},
                              cameras=contract.get("cameras"))
    write_contract(out, exported)
    return exported


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--contract-only", action="store_true",
                        help="label an existing export made from this checkpoint (exports made before "
                             "contracts existed); export_report.json must name the same checkpoint")
    args = parser.parse_args()
    if (args.out / "manifest.json").exists() and not args.contract_only:
        parser.error(f"{args.out} already contains an export")

    sim = MujocoSimulation()
    # local checkpoints only; refuses an unlabelled model or files changed since verification
    contract = load_contract(Path(args.checkpoint), verify=True)
    check_contract(contract, sim.names, cameras=CAMERA_SLOTS, control_hz=1 / sim.config["control_dt"])
    if args.contract_only:
        report_path = args.out / "export_report.json"
        if not report_path.is_file():
            parser.error(f"{report_path} is missing, so the export cannot be tied to a checkpoint")
        made_from = json.loads(report_path.read_text())["checkpoint"]
        if Path(made_from).resolve() != Path(args.checkpoint).resolve():
            parser.error(f"export was made from {made_from}, not {args.checkpoint}")
        write_export_contract(args.out, contract, args.checkpoint)
        print(json.dumps(load_contract(args.out, verify=True), indent=2))
        return

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
    exported = write_export_contract(args.out, contract, args.checkpoint)
    report = {"checkpoint": args.checkpoint, "inputs": schema, "outputs": outputs,
              "export_seconds": round(time.perf_counter() - started, 1),
              "files": sorted(p.name for p in args.out.iterdir()), "contract": exported}
    (args.out / "export_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
