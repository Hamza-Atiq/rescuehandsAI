"""Scene identity for the pick milestone (spec §4, §8): hashes, duplicate scenes, seed blocks."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .scene import ROOT, SceneParams, world_xml

SEED_BLOCKS_PATH = ROOT / "configs" / "pick_seed_blocks.json"
CONFIG_HASH_FILES = ("configs/scene.json", "configs/simulation.json", "configs/pick_contacts.json",
                     "src/rescuehandsai/scene.py")


def text_digest(path) -> str:
    """SHA-256 of a text file with CRLF normalised, so Windows and Linux checkouts agree."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def config_hash(physics_version: int, asset_path, root: Path = ROOT) -> str:
    digest = hashlib.sha256(f"physics_version={physics_version}\n".encode())
    for rel in CONFIG_HASH_FILES:
        digest.update(f"{rel} {text_digest(Path(root) / rel)}\n".encode())
    digest.update(f"asset {text_digest(asset_path)}\n".encode())
    return digest.hexdigest()


def scene_hash(params: SceneParams, config: dict, physics_version: int) -> str:
    return hashlib.sha256(world_xml(params, config, physics_version=physics_version).encode()).hexdigest()


def settings_record(params: SceneParams) -> dict:
    """JSON-ready generated scene settings (tuples become lists)."""
    return json.loads(json.dumps(asdict(params)))


def _rounded(value, digits):
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, list):
        return [_rounded(v, digits) for v in value]
    if isinstance(value, dict):
        return {k: _rounded(v, digits) for k, v in value.items()}
    return value


def settings_hash(params: SceneParams, digits: int = 5) -> str:
    record = settings_record(params)
    record.pop("seed")
    text = json.dumps(_rounded(record, digits), sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def find_duplicates(lists: dict) -> list:
    seen = {}
    for name, params_list in lists.items():
        for params in params_list:
            seen.setdefault(settings_hash(params), []).append([name, params.seed])
    return [{"settings_sha256": h, "members": members} for h, members in seen.items() if len(members) > 1]


def load_seed_blocks(path=None) -> dict:
    return json.loads(Path(path or SEED_BLOCKS_PATH).read_text())


def check_seed_blocks(config: dict) -> dict:
    intervals = [(name, start, stop) for name, (start, stop) in config["blocks"].items()]
    legacy = [(f"legacy_{i}", r["start"], r["stop"]) for i, r in enumerate(config["legacy_ranges"])]
    overlaps = []
    everything = intervals + legacy
    for i, (a, a0, a1) in enumerate(everything):
        for b, b0, b1 in everything[i + 1:]:
            if a.startswith("legacy_") and b.startswith("legacy_"):
                continue
            if a0 < b1 and b0 < a1:
                overlaps.append([a, b])
    return {"overlaps": overlaps,
            "legacy_provenance_complete": all(r["complete"] for r in config["legacy_ranges"]),
            "legacy_sources": [r["source"] for r in config["legacy_ranges"]]}


def block_of(seed: int, config: dict):
    return next((name for name, (start, stop) in config["blocks"].items() if start <= seed < stop), None)
