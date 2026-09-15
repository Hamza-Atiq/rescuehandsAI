"""Independent local dataset checks. Run in .venv-pai; no network or uploads."""
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rescuehandsai.sim import MujocoSimulation


def main():
    root = ROOT / "data/merged_local"
    info = json.loads((root / "meta/info.json").read_text())
    table = pq.read_table(root / "data/chunk-000/file-000.parquet").to_pydict()
    state = np.stack(table["observation.state"])
    action = np.stack(table["action"])
    eps = np.asarray(table["episode_index"])
    frame = np.asarray(table["frame_index"])
    ts = np.asarray(table["timestamp"])
    stats = json.loads((root / "meta/stats.json").read_text())
    sim = MujocoSimulation()
    names = list(sim.names)
    limits = np.array([sim.limits[n] for n in names])
    sim.close()
    within = eps[1:] == eps[:-1]
    metadata = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").to_pydict()
    task_table = pq.read_table(root / "meta/tasks.parquet").to_pydict()
    report = {
        "frames": len(eps), "episodes": len(set(eps)), "tasks": task_table,
        "state_shape": list(state.shape), "action_shape": list(action.shape),
        "state_finite": bool(np.isfinite(state).all()), "action_finite": bool(np.isfinite(action).all()),
        "joint_order_matches": info["features"]["action"]["names"] == names == info["features"]["observation.state"]["names"],
        "action_joint_limit_violations": int(((action < limits[:, 0]-1e-6) | (action > limits[:, 1]+1e-6)).sum()),
        "max_action_step": float(np.abs(np.diff(action, axis=0)[within]).max()),
        "timestamp_max_error": float(np.abs(ts-frame/info["fps"]).max()),
        "state_range_per_joint": np.ptp(state, axis=0).tolist(),
        "action_range_per_joint": np.ptp(action, axis=0).tolist(),
        "state_mean_stats_max_error": float(np.max(np.abs(state.mean(0)-np.array(stats["observation.state"]["mean"])))),
        "action_mean_stats_max_error": float(np.max(np.abs(action.mean(0)-np.array(stats["action"]["mean"])))),
        "episode_lengths_range": [min(Counter(eps).values()), max(Counter(eps).values())],
        "episode_metadata_columns": list(metadata),
        "camera_video_samples": {},
    }
    import av
    for path in sorted((root / "videos").rglob("*.mp4")):
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            count, shapes, first_std = 0, set(), None
            for vf in container.decode(stream):
                if count == 0:
                    first_std = float(vf.to_ndarray(format="rgb24").std())
                shapes.add((vf.width, vf.height))
                count += 1
            report["camera_video_samples"][str(path.relative_to(root))] = {
                "decoded_frames": count, "sizes": sorted(shapes), "first_frame_std": first_std}
    records = []
    for p in sorted((ROOT / "data").glob("local_w*_attempts.jsonl")):
        records += [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
    saved = [r for r in records if r["saved"]]
    report["attempt_logs"] = {"attempts": len(records), "saved": len(saved),
        "saved_utensils": dict(Counter(r["utensil"] for r in saved)),
        "eval_seed_overlap": sorted({r["seed"] for r in saved} & set(range(10))),
        "saved_without_support_field": sum("supported" not in r["outcome"] for r in saved)}
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
