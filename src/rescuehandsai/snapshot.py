"""What actually ran: source, assets, runtime and model files (spec §11 snapshot record).

A hash of tracked files alone can miss code that runs, so every file under the snapshot
folders is hashed, tracked or not, and strict runs refuse uncommitted changes there."""
import hashlib
from importlib.metadata import PackageNotFoundError, version
import platform
import re
import subprocess
from pathlib import Path

SNAPSHOT_DIRS = ("src", "scripts", "training", "configs")
TEXT_SUFFIXES = {".py", ".json", ".jsonl", ".xml", ".sh", ".md", ".txt", ".toml", ".yaml", ".yml", ".cfg", ".ipynb"}
PACKAGES = ("mujoco", "numpy", "torch", "lerobot", "transformers", "openvino", "nncf")


class SnapshotRefused(RuntimeError):
    """Data generation, training and the final test need committed source."""


def file_digest(path) -> str:
    path = Path(path)
    data = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES:
        data = data.replace(b"\r\n", b"\n")  # Windows checkouts convert line endings; hash the content
    return hashlib.sha256(data).hexdigest()


def _source_files(root: Path) -> list:
    files = []
    for folder in SNAPSHOT_DIRS:
        base = root / folder
        if base.is_dir():
            files += [p for p in base.rglob("*")
                      if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    return sorted(files)


def source_hash(root) -> tuple:
    root = Path(root)
    files = {p.relative_to(root).as_posix(): file_digest(p) for p in _source_files(root)}
    digest = hashlib.sha256("\n".join(f"{name} {h}" for name, h in sorted(files.items())).encode()).hexdigest()
    return digest, files


def _git(root, *args) -> str:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, encoding="utf-8",
                          errors="replace", check=True).stdout


def uncommitted_files(root) -> list:
    out = _git(root, "status", "--porcelain", "--untracked-files=all", "--", *SNAPSHOT_DIRS)
    return sorted(line[3:].strip().strip('"') for line in out.splitlines()
                  if line.strip() and "__pycache__" not in line)


def asset_files(asset_xml) -> list:
    asset_xml = Path(asset_xml)
    text = asset_xml.read_text(encoding="utf-8")
    meshdir = re.search(r'<compiler[^>]*\bmeshdir="([^"]+)"', text)
    base = asset_xml.parent / (meshdir.group(1) if meshdir else "")
    files = [asset_xml] + [base / name for name in re.findall(r'<mesh[^>]*\bfile="([^"]+)"', text)]
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"asset files missing: {missing}")
    return files


def asset_hash(asset_xml) -> tuple:
    parent = Path(asset_xml).parent
    files = {p.relative_to(parent).as_posix(): file_digest(p) for p in asset_files(asset_xml)}
    digest = hashlib.sha256("\n".join(f"{name} {h}" for name, h in sorted(files.items())).encode()).hexdigest()
    return digest, files


def _devices() -> dict:
    devices = {}
    try:
        import openvino as ov
        core = ov.Core()
        for name in core.available_devices:
            info = {"name": str(core.get_property(name, "FULL_DEVICE_NAME"))}
            try:
                info["driver_version"] = str(core.get_property(name, "GPU_DRIVER_VERSION"))
            except Exception:
                info["driver_version"] = "unavailable"
            devices[f"openvino:{name}"] = info
    except ImportError:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            devices["cuda:0"] = {"name": torch.cuda.get_device_name(0), "cuda": str(torch.version.cuda)}
    except ImportError:
        pass
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
        if out.returncode == 0 and out.stdout.strip():
            devices["nvidia-smi"] = out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return devices


def runtime_versions() -> dict:
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "os": platform.platform(), "processor": platform.processor(),
            "packages": packages, "devices": _devices()}


def snapshot_record(root, asset_xml, *, strict: bool, model_files: dict | None = None, noise_file=None) -> dict:
    root = Path(root)
    dirty = uncommitted_files(root)
    if strict and dirty:
        raise SnapshotRefused(f"uncommitted or untracked files in {SNAPSHOT_DIRS}: {dirty}")
    source_digest, files = source_hash(root)
    asset_digest, assets = asset_hash(asset_xml)
    return {
        "git_revision": _git(root, "rev-parse", "HEAD").strip(),
        "source_sha256": source_digest,
        "uncommitted": {name: files.get(name) for name in dirty},  # None: deleted
        "asset_sha256": asset_digest,
        "asset_files": assets,
        "runtime": runtime_versions(),
        "model_files": {name: file_digest(path) for name, path in (model_files or {}).items()},
        "noise_sha256": file_digest(noise_file) if noise_file is not None else None,
    }
