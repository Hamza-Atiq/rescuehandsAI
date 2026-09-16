"""How far into its own 50-step plan can the learned policy be trusted? (.venv-pai)

  PYTHONPATH=src .venv-pai/Scripts/python.exe scripts/diagnose_horizon.py \
      --export models/openvino/fp32 --dataset ABDHAM/rescuehands_table \
      --dataset-root data/merged_local --frames 12

The runner executes the first N actions of every predicted chunk before asking
again. If the policy's error grows with the step index, executing 25 of them walks
the robot off the teacher's path and it never recovers. This measures that growth
directly: for real dataset frames it compares the predicted chunk with the teacher's
recorded commands for the same following steps, and reports error per step index.
Ground truth is the demonstration itself, so this is an open-loop check, not task success.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from rescuehandsai.contract import CAMERA_SLOTS
from rescuehandsai.scene import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-root")
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--frames", type=int, default=12, help="observations sampled across the dataset")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "horizon_diagnosis.json")
    args = parser.parse_args()

    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from physicalai.inference.model import InferenceModel

    dataset = LeRobotDataset(args.dataset, root=args.dataset_root, video_backend="pyav")
    joints = dataset.meta.features["action"]["names"]
    model = InferenceModel(args.export, device=args.device)
    image_keys = {slot: f"images.images.{slot}" for slot in CAMERA_SLOTS}

    episode_index = np.asarray(dataset.hf_dataset["episode_index"])
    actions = np.asarray(dataset.hf_dataset["action"], dtype=np.float32)
    starts = np.linspace(0, len(dataset) - 1, args.frames + 2)[1:-1].astype(int)

    per_frame, horizon = [], None
    for start in starts:
        item = dataset[int(start)]
        same_episode = np.flatnonzero(episode_index == episode_index[start])
        future = same_episode[same_episode >= start]
        inputs = {"state": item["observation.state"].numpy()[None].astype(np.float32),
                  "task": [item["task"]]}
        for slot, camera in CAMERA_SLOTS.items():
            image = item[f"observation.images.{camera}"]
            inputs[image_keys[slot]] = image.numpy()[None].astype(np.float32)
        chunk = np.asarray(model.predict_action_chunk(inputs)).reshape(-1, len(joints))
        steps = min(len(chunk), len(future))
        error = np.abs(chunk[:steps] - actions[future[:steps]]).mean(axis=1)
        horizon = error if horizon is None else horizon[:steps] + error[:steps]
        per_frame.append({"frame": int(start), "episode": int(episode_index[start]),
                          "steps_compared": steps, "mean_abs_rad": round(float(error.mean()), 4),
                          "error_at_0": round(float(error[0]), 4),
                          "error_at_10": round(float(error[min(10, steps - 1)]), 4),
                          "error_at_24": round(float(error[min(24, steps - 1)]), 4),
                          "error_at_49": round(float(error[steps - 1]), 4)})
        print(json.dumps(per_frame[-1]), flush=True)

    mean_curve = (horizon / len(per_frame)).tolist()
    report = {"export": str(args.export), "device": args.device, "dataset": args.dataset,
              "frames": len(per_frame), "joints": joints,
              "mean_abs_rad_by_step": [round(float(v), 4) for v in mean_curve],
              "per_frame": per_frame,
              "note": "open-loop: the policy is scored against the teacher's own commands, "
                      "so this measures prediction drift inside one chunk, not task success."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    curve = report["mean_abs_rad_by_step"]
    print("mean |error| (rad) at steps 0/5/10/15/20/25/35/49:",
          [curve[i] for i in (0, 5, 10, 15, 20, 25, 35, len(curve) - 1) if i < len(curve)])


if __name__ == "__main__":
    main()
