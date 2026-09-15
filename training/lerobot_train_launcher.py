"""Run LeRobot training with the policy weights cast to float32.

SmolVLA loads its VLM in bfloat16. Turing GPUs such as the Kaggle T4 have no
native bfloat16 kernels, which made a training step take ~7 s. Casting the
weights to float32 (optionally with fp16 autocast via ACCELERATE_MIXED_PRECISION)
uses the kernels the T4 does have. Arguments are passed to lerobot_train unchanged.
"""
import os
import sys

import torch
import lerobot.scripts.lerobot_train as lerobot_train

_make_policy = lerobot_train.make_policy


def make_policy_float32(*args, **kwargs):
    policy = _make_policy(*args, **kwargs)
    if os.environ.get("RESCUEHANDS_WEIGHTS_DTYPE", "float32") == "float32":
        policy = policy.to(torch.float32)
    return policy


lerobot_train.make_policy = make_policy_float32

if __name__ == "__main__":
    sys.argv = ["lerobot_train", *sys.argv[1:]]
    lerobot_train.main()
