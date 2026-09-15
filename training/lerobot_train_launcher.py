"""Run LeRobot training with our task schema and T4-friendly weight precision.

Two corrections on top of `lerobot_train`, both applied where the policy is made:

1. Schema. LeRobot's factory replaces the output features from the dataset but keeps
   the pretrained input features when they are non-empty, so a SmolVLA base
   checkpoint would save a 6-D state description while training on our 12-D state.
   The data path pads the real tensor, so learning is unaffected, but Intel's export
   reads the saved description. We rebuild the input features from the dataset
   (after the camera rename) and refuse to train on anything but 12-D state/action
   and the three camera slots.
2. Precision. SmolVLA loads its VLM in bfloat16; Turing GPUs such as the Kaggle T4
   have no native bfloat16 kernels (7.4 s/step). Weights are cast to float32 and
   ACCELERATE_MIXED_PRECISION=fp16 enables autocast (1.33 s/step measured).

Arguments are passed to lerobot_train unchanged.
"""
import os
import sys

import torch
import lerobot.scripts.lerobot_train as lerobot_train
from lerobot.configs.types import FeatureType
from lerobot.datasets.feature_utils import dataset_to_policy_features

JOINTS = 12
CAMERA_SLOTS = ("observation.images.camera1", "observation.images.camera2", "observation.images.camera3")

_make_policy = lerobot_train.make_policy


def task_input_features(ds_meta, rename_map):
    features = dataset_to_policy_features(ds_meta.features)
    inputs = {(rename_map or {}).get(key, key): ft for key, ft in features.items() if ft.type is not FeatureType.ACTION}
    action = features.get("action")
    problems = []
    if tuple(inputs.get("observation.state").shape if "observation.state" in inputs else ()) != (JOINTS,):
        problems.append(f"state shape {inputs.get('observation.state')}")
    if action is None or tuple(action.shape) != (JOINTS,):
        problems.append(f"action shape {action}")
    visual = sorted(k for k, ft in inputs.items() if ft.type is FeatureType.VISUAL)
    if visual != sorted(CAMERA_SLOTS):
        problems.append(f"cameras {visual}")
    if problems:
        raise SystemExit("Dataset does not match the RescueHands policy schema: " + "; ".join(problems))
    return inputs


def make_policy_for_task(*args, **kwargs):
    cfg, ds_meta = kwargs["cfg"], kwargs.get("ds_meta")
    if ds_meta is not None:
        cfg.input_features = task_input_features(ds_meta, kwargs.get("rename_map"))
        print("input features:", {k: tuple(v.shape) for k, v in cfg.input_features.items()}, flush=True)
    policy = _make_policy(*args, **kwargs)
    if os.environ.get("RESCUEHANDS_WEIGHTS_DTYPE", "float32") == "float32":
        policy = policy.to(torch.float32)
    return policy


lerobot_train.make_policy = make_policy_for_task

if __name__ == "__main__":
    sys.argv = ["lerobot_train", *sys.argv[1:]]
    lerobot_train.main()
