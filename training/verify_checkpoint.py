"""Check a saved SmolVLA checkpoint before trusting a long run or exporting it.

    python training/verify_checkpoint.py --checkpoint outputs/<run>/checkpoints/last/pretrained_model \
        --dataset ABDHAM/rescuehands_table

Checks: saved state/action shapes are 12, the three camera slots are declared,
normalizer statistics have 12 values, joint names match the dataset, and the
policy loaded through its saved processors returns a finite (50, 12) action chunk
for a real dataset frame. With --write-contract it also writes task_contract.json
(ordered joint names, camera map, action units, control rate, dataset and weight
hash) next to the checkpoint, which inference then refuses to run without.
Exit code 1 on any problem. Nothing is uploaded.
"""
import argparse
import json
import sys
from pathlib import Path

JOINTS = 12
CAMERA_SLOTS = ("observation.images.camera1", "observation.images.camera2", "observation.images.camera3")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-root", help="local copy of the dataset, to avoid downloading it")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--write-contract", action="store_true",
                        help="write task_contract.json next to the checkpoint")
    args = parser.parse_args()

    import torch
    from safetensors.torch import load_file
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    problems, report = [], {"checkpoint": str(args.checkpoint)}
    cfg = json.loads((args.checkpoint / "config.json").read_text())
    inputs, outputs = cfg.get("input_features", {}), cfg.get("output_features", {})
    report["state_shape"] = inputs.get("observation.state", {}).get("shape")
    report["action_shape"] = outputs.get("action", {}).get("shape")
    report["cameras"] = sorted(k for k, v in inputs.items() if v.get("type") == "VISUAL")
    if report["state_shape"] != [JOINTS]:
        problems.append(f"saved state shape {report['state_shape']}")
    if report["action_shape"] != [JOINTS]:
        problems.append(f"saved action shape {report['action_shape']}")
    if report["cameras"] != sorted(CAMERA_SLOTS):
        problems.append(f"saved cameras {report['cameras']}")

    stats = {}
    for path in args.checkpoint.glob("*normalizer*.safetensors"):
        for key, tensor in load_file(str(path)).items():
            if key.startswith(("observation.state.", "action.")) and key.rsplit(".", 1)[1] in ("mean", "std"):
                stats[f"{path.name}:{key}"] = list(tensor.shape)
    report["normalizer_stats"] = stats
    if not stats or any(shape != [JOINTS] for shape in stats.values()):
        problems.append(f"normalizer stats {stats}")

    dataset = LeRobotDataset(args.dataset, root=args.dataset_root,
                             **({"video_backend": "pyav"} if args.device == "cpu" else {}))
    names = dataset.meta.features["action"].get("names")
    report["joint_order"] = names
    if cfg.get("action_feature_names") not in (None, names):
        problems.append(f"checkpoint joint order {cfg.get('action_feature_names')} != dataset {names}")

    policy = SmolVLAPolicy.from_pretrained(str(args.checkpoint)).to(args.device).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(args.checkpoint),
                                         preprocessor_overrides={"device_processor": {"device": args.device}})
    item = dataset[len(dataset) // 2]
    batch = {k: v.unsqueeze(0) for k, v in item.items()
             if isinstance(v, torch.Tensor) and k.startswith("observation.")}
    batch["task"] = [item["task"]]
    report["task"] = item["task"]
    with torch.no_grad():
        chunk = policy.predict_action_chunk(pre(batch))
        actions = post(chunk)
    report["chunk_shape"] = list(actions.shape)
    report["chunk_finite"] = bool(torch.isfinite(actions).all())
    if list(actions.shape)[-2:] != [policy.config.chunk_size, JOINTS] or not report["chunk_finite"]:
        problems.append(f"action chunk {report['chunk_shape']} finite={report['chunk_finite']}")
    else:
        target = item["action"].to(actions.device)
        report["first_action_abs_error_rad"] = round(float((actions[0, 0] - target).abs().mean()), 4)

    if args.write_contract and not problems:
        from rescuehandsai.contract import build_contract, file_sha256, write_contract
        weights = args.checkpoint / "model.safetensors"
        contract = build_contract(names, dataset=args.dataset, dataset_revision=dataset.revision,
                                  control_hz=args.control_hz, source=str(args.checkpoint),
                                  files={weights.name: file_sha256(weights)})
        report["contract"] = str(write_contract(args.checkpoint, contract))

    report["problems"] = problems
    print(json.dumps(report, indent=2))
    print("CHECKPOINT OK" if not problems else "CHECKPOINT PROBLEMS: " + "; ".join(problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
