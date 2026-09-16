"""The policy contract: what a checkpoint or export expects, in writing.

A 12-number shape check cannot catch a swapped arm or a swapped camera, so the
ordered joint names, the camera map, the action units and the control rate are
written next to the checkpoint and copied into the OpenVINO export. Inference
loads the contract and refuses a simulator that does not match it.
"""
import hashlib
import json
import math
from pathlib import Path

CONTRACT_NAME = "task_contract.json"
ACTION_UNITS = "absolute joint position targets in radians"
# policy camera slot -> our camera name (same order as the training rename map)
CAMERA_SLOTS = {"camera1": "overhead", "camera2": "left_wrist", "camera3": "right_wrist"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def build_contract(joint_order, *, dataset: str, dataset_revision=None, control_hz: float,
                   source: str, files: dict | None = None, cameras: dict | None = None) -> dict:
    joint_order = list(joint_order)
    if len(joint_order) != len(set(joint_order)) or not joint_order:
        raise ValueError(f"joint order must be unique and non-empty: {joint_order}")
    if not _positive_finite(control_hz):
        raise ValueError(f"control rate must be a positive finite number, got {control_hz}")
    return {"joint_order": joint_order, "cameras": dict(cameras or CAMERA_SLOTS),
            "action_units": ACTION_UNITS, "control_hz": control_hz, "dataset": dataset,
            "dataset_revision": dataset_revision, "source": source, "files": files or {}}


def write_contract(directory: Path, contract: dict) -> Path:
    path = Path(directory) / CONTRACT_NAME
    path.write_text(json.dumps(contract, indent=2))
    return path


def _positive_finite(value) -> bool:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and value > 0


def verify_files(directory: Path, contract: dict) -> None:
    """Raise ValueError unless every hashed file is present and unchanged.

    A contract describes specific files; new weights next to an old contract must not pass."""
    files = contract.get("files") or {}
    if not files:
        raise ValueError("contract lists no file hashes, so it cannot vouch for these files")
    problems = []
    for name, expected in files.items():
        path = Path(directory) / name
        if not path.is_file():
            problems.append(f"{name} missing")
        elif file_sha256(path) != expected:
            problems.append(f"{name} changed since the contract was written")
    if problems:
        raise ValueError("Model files do not match their contract: " + "; ".join(problems))


def load_contract(directory: Path, *, verify: bool = False) -> dict:
    path = Path(directory) / CONTRACT_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing: this model was produced before the contract existed, or by other code. "
            "Write it with training/verify_checkpoint.py --write-contract before deploying.")
    contract = json.loads(path.read_text())
    if verify:
        verify_files(directory, contract)
    return contract


def check_contract(contract: dict, joint_names, *, cameras: dict | None = None, control_hz=None):
    """Raise ValueError unless the running simulator matches the trained contract."""
    problems = []
    if list(contract.get("joint_order", [])) != list(joint_names):
        problems.append(f"joint order {contract.get('joint_order')} != simulator {list(joint_names)}")
    if cameras is not None and dict(contract.get("cameras", {})) != dict(cameras):
        problems.append(f"camera map {contract.get('cameras')} != {cameras}")
    if contract.get("action_units") != ACTION_UNITS:
        problems.append(f"action units {contract.get('action_units')!r} != {ACTION_UNITS!r}")
    if not _positive_finite(contract.get("control_hz")):
        problems.append(f"control rate {contract.get('control_hz')!r} is not a positive finite number")
    elif control_hz is not None and abs(float(contract["control_hz"]) - control_hz) > 1e-6:
        problems.append(f"control rate {contract.get('control_hz')} Hz != {control_hz} Hz")
    if problems:
        raise ValueError("Model contract does not match this robot: " + "; ".join(problems))
