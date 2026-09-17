"""Pick evaluation summaries (spec §11): the scene is the unit, and incomplete runs never pass."""
import numpy as np

from .pick_cells import CELLS, named_slot

MAIN_TEST_BARS = {"episode_successes_min": 380, "cell_successes_min": 92, "spare_touch_episodes_max": 4,
                  "scenes_all_four_min": 92}
INCOMPLETE = "incomplete — not eligible to pass"


def scored_entry(record: dict, attempt: int) -> dict:
    """The fields a summary needs from one valid runner record."""
    return {"scene": record["seed"], "cell": record["cell"], "template": record["template"], "valid": True,
            "attempt": attempt, "success": bool(record["success"]), "picked": record["outcome"]["picked"],
            "failure_labels": [f["label"] for f in record["outcome"]["failures"]]}


def _groups(scored, key) -> dict:
    out = {}
    for e in scored:
        group = out.setdefault(key(e), {"successes": 0, "episodes": 0})
        group["successes"] += int(e["success"])
        group["episodes"] += 1
    return out


def _count(values) -> dict:
    out = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return out


def _scene_bootstrap(scenes: dict, n_boot: int, seed: int):
    ids = sorted(scenes)
    if not ids:
        return None, None
    successes = np.array([sum(e["success"] for e in scenes[s]) for s in ids], float)
    episodes = np.array([len(scenes[s]) for s in ids], float)
    complete = np.array([len(scenes[s]) == len(CELLS) for s in ids], float)
    all_four = np.array([len(scenes[s]) == len(CELLS) and all(e["success"] for e in scenes[s]) for s in ids], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ids), size=(n_boot, len(ids)))  # resample whole scenes
    episode_rates = successes[idx].sum(1) / episodes[idx].sum(1)
    complete_counts = complete[idx].sum(1)
    scene_rates = np.where(complete_counts > 0, all_four[idx].sum(1) / np.maximum(complete_counts, 1), np.nan)
    ci = lambda x: [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]
    return ci(episode_rates), (ci(scene_rates) if complete.any() else None)


def summarize(scored, *, planned_keys, blocked_keys=(), invalid_counts=None, bars=None, n_boot: int = 2000,
              boot_seed: int = 0) -> dict:
    for e in scored:
        if not e.get("valid", False):
            raise ValueError("summaries take scored valid attempts only; invalid attempts never enter denominators")
    keys = [f"{e['scene']}_{e['cell']}" for e in scored]
    if len(set(keys)) != len(keys):
        raise ValueError("more than one scored attempt for an episode")
    planned = set(planned_keys)
    extra = sorted(set(keys) - planned)
    if extra:
        raise ValueError(f"scored episodes outside the plan: {extra}")
    missing = sorted(planned - set(keys))
    eligible = not missing and not blocked_keys
    scenes = {}
    for e in scored:
        scenes.setdefault(e["scene"], []).append(e)
    complete = {s: es for s, es in scenes.items() if len(es) == len(CELLS)}
    passing = sum(all(e["success"] for e in es) for es in complete.values())
    n, successes = len(scored), sum(e["success"] for e in scored)
    episode_ci, scene_ci = _scene_bootstrap(scenes, n_boot, boot_seed)
    per_cell = _groups(scored, lambda e: e["cell"])
    spare_touches = sum("WRONG_ITEM_TOUCHED" in e["failure_labels"] for e in scored)
    summary = {
        "evaluation_valid": eligible,
        "planned_episodes": len(planned), "scored_episodes": n,
        "missing_episodes": missing, "blocked_episodes": sorted(blocked_keys),
        "invalid_attempts": dict(invalid_counts or {}),
        "episodes": {"successes": successes, "episodes": n, "rate": successes / n if n else None, "ci95": episode_ci},
        "scenes_all_four": {"passing": passing, "complete_scenes": len(complete),
                            "rate": passing / len(complete) if complete else None, "ci95": scene_ci},
        "per_cell": per_cell,
        "per_slot": _groups(scored, lambda e: str(named_slot(e["cell"]))),
        "per_word": _groups(scored, lambda e: CELLS[e["cell"]][0]),
        "per_template": _groups(scored, lambda e: e["template"]),
        "spare_touch_episodes": spare_touches,
        "picked": _count(e["picked"] for e in scored),
        "failure_labels": _count(label for e in scored for label in e["failure_labels"]),
        "attempt_used": {f"{e['scene']}_{e['cell']}": e["attempt"] for e in scored},
    }
    if not eligible:
        summary["bars"], summary["verdict"] = None, INCOMPLETE
    elif bars is None:
        summary["bars"], summary["verdict"] = None, "no pass bars for this run"
    else:
        checks = {
            "episode_successes": {"value": successes, "min": bars["episode_successes_min"],
                                  "pass": successes >= bars["episode_successes_min"]},
            "each_cell": {"values": {c: per_cell.get(c, {"successes": 0})["successes"] for c in CELLS},
                          "min": bars["cell_successes_min"],
                          "pass": all(per_cell.get(c, {"successes": 0})["successes"] >= bars["cell_successes_min"]
                                      for c in CELLS)},
            "spare_touch_episodes": {"value": spare_touches, "max": bars["spare_touch_episodes_max"],
                                     "pass": spare_touches <= bars["spare_touch_episodes_max"]},
            "scenes_all_four": {"value": passing, "min": bars["scenes_all_four_min"],
                                "pass": passing >= bars["scenes_all_four_min"]},
        }
        summary["bars"] = checks
        summary["verdict"] = "pass" if all(c["pass"] for c in checks.values()) else "fail"
    return summary
