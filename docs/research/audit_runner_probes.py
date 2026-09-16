"""Read-only audit probes. Run from repository root with the simulation Python."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from rescuehandsai.runner import EpisodeRunner
from rescuehandsai.task import make_task
from rescuehandsai.scene import sample_params, load_config
from rescuehandsai.evaluation import task_outcome, HandoffTracker
from test_outcome import good_facts

task = make_task(1)
params = sample_params(load_config(), 1)
facts = good_facts(params, task)

class Sim:
    config = {"control_dt": .05, "max_command_delta": .1}
    scene_params = params
    data = SimpleNamespace(time=0.)
    previous = {"joint": 0.}
    limits = {"joint": (-1., 1.)}
    def reset(self, *a, **k): pass
    def observe(self, **k): return SimpleNamespace(images={}, timestamp=0.)
    def set_actuator_fault(self, *a): pass

class Policy:
    def reset(self, *a): pass
    def metadata(self): return {}
    def wants_images(self): return False
    def act(self, obs): raise RuntimeError("FAILED_GRASP: cup is missing")
    def after_recovery(self, *a): raise RuntimeError("planning failed after recovery")

result = {}
with patch("rescuehandsai.runner.compute_facts", return_value=facts), patch.object(EpisodeRunner, "_safe_pose"):
    try:
        EpisodeRunner(Sim(), Policy(), supervisor=True).run(task)
    except Exception as exc:
        result["after_recovery_error_escapes_without_log"] = str(exc)

class ResetBreaks(Policy):
    def reset(self, *a): raise RuntimeError("initial planning failed")
try:
    EpisodeRunner(Sim(), ResetBreaks(), supervisor=True).run(task)
except Exception as exc:
    result["reset_error_escapes_without_log"] = str(exc)

other = "spoon" if task.utensil == "fork" else "fork"
spare_held = good_facts(params, task, held_by={other: {"right_arm"}},
                        touching={other: {"right_arm"}}, speed={other: 3.})
result["spare_held_and_moving_can_still_succeed"] = task_outcome(spare_held, task, True, params)["success"]

tracker = HandoffTracker(task.utensil)
for hands in ({"right_arm"}, {"right_arm", "left_arm"}, *([set()] * 100), {"left_arm"}):
    tracker.update(good_facts(params, task, held_by={task.utensil: hands}, supported={task.utensil: False}))
result["unbounded_airborne_contact_gap_still_counts_handoff"] = tracker.done
print(json.dumps(result, indent=2))
