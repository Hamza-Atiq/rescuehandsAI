"""Compare a fresh scripted full-task evaluation with the pre-pick reference, episode by episode.

Usage (project root):
  scripts/check_v1_regression.py results/v1_reference_pre_pick results/<fresh run>
Exit code 0 only when the candidate matched the recorded results: every episode record field
(outcome, events, recoveries, steps, sim time, ...) except wall-clock timings, and the robot
asset and configuration identities in manifest.json. A match is evidence, not proof that all
behaviour is identical.
"""
import argparse
import json
from pathlib import Path

WALL_CLOCK_FIELDS = ("wall_seconds", "inference_seconds")  # machine speed, not behaviour
MANIFEST_IDENTITY = ("asset_sha256", "scene_config", "sim_config", "seeds", "args")


def _without_timing(record: dict) -> dict:
    return {k: v for k, v in record.items() if k not in WALL_CLOCK_FIELDS}


def compare(reference: Path, candidate: Path) -> list[str]:
    reference, candidate = Path(reference), Path(candidate)
    ref_files = sorted(reference.glob("episode_*.json"))
    if not ref_files:
        raise ValueError(f"{reference} has no episode files")
    problems = []
    ref_manifest, cand_manifest = reference / "manifest.json", candidate / "manifest.json"
    if not ref_manifest.is_file() or not cand_manifest.is_file():
        problems.append("manifest.json: missing from reference or candidate")
    else:
        ref_m, cand_m = json.loads(ref_manifest.read_text()), json.loads(cand_manifest.read_text())
        for key in MANIFEST_IDENTITY:
            ref_value, cand_value = ref_m.get(key), cand_m.get(key)
            if key == "args":  # the run name differs by design
                ref_value = {k: v for k, v in (ref_value or {}).items() if k != "name"}
                cand_value = {k: v for k, v in (cand_value or {}).items() if k != "name"}
            if ref_value != cand_value:
                problems.append(f"manifest.json: {key} differs")
    for ref_file in ref_files:
        cand_file = candidate / ref_file.name
        if not cand_file.is_file():
            problems.append(f"{ref_file.name}: missing from candidate")
            continue
        ref = _without_timing(json.loads(ref_file.read_text()))
        cand = _without_timing(json.loads(cand_file.read_text()))
        for field in sorted(set(ref) | set(cand)):
            if ref.get(field) != cand.get(field):
                problems.append(f"{ref_file.name}: {field} {ref.get(field)!r} != {cand.get(field)!r}")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    problems = compare(args.reference, args.candidate)
    for line in problems:
        print(line)
    print("v1 regression:", "FAILED" if problems else "matched recorded results")
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
