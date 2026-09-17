"""Run validity, attempts and resume rules for pick evaluations (spec §6, §11)."""
import json
from pathlib import Path

INVALID_LABELS = ("SIM_ERROR", "MODEL_LOAD_ERROR", "CONTRACT_MISMATCH")
RESUME_KEYS = ("model", "snapshot", "scene_list_sha256", "physics_version", "config_hash", "rules_sha256",
               "run_config")


class InvalidRun(Exception):
    """Broken test equipment: never scored. `partial` keeps the measurements taken before the failure."""

    def __init__(self, label: str, detail: str, partial: dict | None = None):
        if label not in INVALID_LABELS:
            raise ValueError(f"not an invalid-run label: {label}")
        super().__init__(f"{label}: {detail}")
        self.label, self.detail, self.partial = label, detail, partial


class EvaluationBlocked(RuntimeError):
    """A second invalid attempt for the same episode: stop until it is understood."""


class RetryNeedsDiagnosis(RuntimeError):
    """An invalid attempt may be retried once, only after its cause is written down."""


class AlreadyScored(RuntimeError):
    """This episode already has a valid result; valid failures are never replaced."""


class ResumeRefused(RuntimeError):
    """The experiment changed; use a new run directory."""


def load_model_or_invalid(factory):
    try:
        return factory()
    except Exception as exc:
        raise InvalidRun("MODEL_LOAD_ERROR", f"{type(exc).__name__}: {exc}") from exc


def check_run_contract(expected: dict, actual: dict) -> None:
    diff = {key: {"expected": value, "actual": actual.get(key)}
            for key, value in expected.items() if actual.get(key) != value}
    if diff:
        raise InvalidRun("CONTRACT_MISMATCH", json.dumps(diff, sort_keys=True, default=str))


def episode_key(seed: int, cell: str) -> str:
    return f"{seed}_{cell}"


def episode_filename(seed: int, cell: str, attempt: int) -> str:
    return f"episode_{seed}_{cell}_a{attempt}.json"


def write_episode(run_dir, seed: int, cell: str, attempt: int, record: dict) -> Path:
    path = Path(run_dir) / episode_filename(seed, cell, attempt)
    with path.open("x", encoding="utf-8") as handle:  # "x": an attempt file is never overwritten
        json.dump(record, handle, indent=2, default=str)
    return path


class AttemptLedger:
    def __init__(self, run_dir):
        self.path = Path(run_dir) / "attempts.jsonl"
        self.lines = ([json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
                      if self.path.is_file() else [])

    def _append(self, line: dict):
        self.lines.append(line)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, sort_keys=True) + "\n")

    def attempts(self, key: str) -> list:
        return [line for line in self.lines if line["type"] == "attempt" and line["key"] == key]

    def _diagnosed(self, key: str) -> set:
        return {line["attempt"] for line in self.lines if line["type"] == "diagnosis" and line["key"] == key}

    def next_attempt(self, key: str) -> int:
        attempts = self.attempts(key)
        if any(a["valid"] for a in attempts):
            raise AlreadyScored(f"{key} already has a valid result")
        invalid = [a for a in attempts if not a["valid"]]
        if len(invalid) >= 2:
            labels = ", ".join(a["label"] for a in invalid)
            raise EvaluationBlocked(f"{key}: two invalid attempts ({labels}); understand the cause before continuing")
        if invalid and invalid[-1]["attempt"] not in self._diagnosed(key):
            raise RetryNeedsDiagnosis(f"{key}: attempt {invalid[-1]['attempt']} was invalid "
                                      f"({invalid[-1]['label']}); write its diagnosis first")
        return len(attempts) + 1

    def add_diagnosis(self, key: str, attempt: int, text: str):
        if not text.strip():
            raise ValueError("a diagnosis must say what went wrong")
        if not any(a["attempt"] == attempt and not a["valid"] for a in self.attempts(key)):
            raise ValueError("a diagnosis must name an invalid attempt of this episode")
        self._append({"type": "diagnosis", "key": key, "attempt": attempt, "text": text})

    def record(self, key: str, attempt: int, *, valid: bool, label: str | None, filename: str):
        expected = len(self.attempts(key)) + 1
        if attempt != expected:
            raise ValueError(f"{key}: expected attempt {expected}, got {attempt}")
        if not valid and label not in INVALID_LABELS:
            raise ValueError(f"invalid attempt must have a label in {INVALID_LABELS}, got {label!r}")
        self._append({"type": "attempt", "key": key, "attempt": attempt, "valid": valid, "label": label,
                      "file": filename})

    def scored(self, key: str):
        return next((a for a in self.attempts(key) if a["valid"]), None)

    def blocked_keys(self) -> list:
        keys = sorted({line["key"] for line in self.lines if line["type"] == "attempt"})
        return [k for k in keys if sum(not a["valid"] for a in self.attempts(k)) >= 2]

    def invalid_counts(self) -> dict:
        counts = {}
        for line in self.lines:
            if line["type"] == "attempt" and not line["valid"]:
                counts[line["label"]] = counts.get(line["label"], 0) + 1
        return counts


def check_resume(run_dir, current: dict) -> None:
    missing = [key for key in RESUME_KEYS if key not in current]
    if missing:
        raise ValueError(f"the current run description lacks {missing}")
    manifest = Path(run_dir) / "manifest.json"
    if not manifest.is_file():
        raise ResumeRefused(f"no manifest.json in {run_dir}; nothing to resume")
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    now = json.loads(json.dumps({key: current[key] for key in RESUME_KEYS}, default=str))
    changed = [key for key in RESUME_KEYS if saved.get(key) != now[key]]
    if changed:
        raise ResumeRefused(f"experiment changed ({', '.join(changed)}); use a new run directory")
