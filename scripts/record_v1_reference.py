"""Record the physics-version-1 reference before any pick-milestone code change (spec §12).

Writes tests/data/physics_v1_reference.json: the world XML SHA-256 for seeds 0-9, the
identity of the scene/simulation configs and the robot asset (XML and every mesh), and
the git revision it was taken at. Scripted outcomes come from
  scripts/evaluate.py --policy scripted --seeds 0:10 --supervisor on --name v1_reference_pre_pick
"""
import hashlib
import json
import subprocess

from rescuehandsai.scene import ROOT, load_config, sample_params, world_xml

TEXT_SUFFIXES = {".json", ".xml", ".py"}


def digest(path) -> str:
    data = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES:
        data = data.replace(b"\r\n", b"\n")  # this checkout converts line endings
    return hashlib.sha256(data).hexdigest()


def identity_files() -> list:
    sim_config = json.loads((ROOT / "configs/simulation.json").read_text())
    asset = ROOT / sim_config["asset_path"]
    meshes = sorted(p for p in (asset.parent / "assets").rglob("*") if p.is_file())
    return [ROOT / "configs/scene.json", ROOT / "configs/simulation.json", asset, *meshes]


def identity() -> dict:
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in identity_files()}


def main():
    config = load_config()
    hashes = {str(seed): hashlib.sha256(world_xml(sample_params(config, seed), config).encode()).hexdigest()
              for seed in range(10)}
    revision = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    out = ROOT / "tests" / "data" / "physics_v1_reference.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"git_revision": revision, "world_xml_sha256": hashes, "identity": identity()},
                              indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
