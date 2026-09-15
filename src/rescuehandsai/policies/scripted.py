"""The scripted teacher as a policy: the baseline row (uses privileged state)."""
from ..expert import ScriptedExpert
from ..task import SUBTASKS


class ScriptedPolicy:
    name = "scripted_teacher"
    uses_privileged_state = True

    def reset(self, sim, task):
        self.expert = ScriptedExpert(sim, task)

    def wants_images(self) -> bool:
        return False

    def act(self, obs):
        return self.expert.act(obs)

    @property
    def done(self) -> bool:
        return self.expert.done

    def after_recovery(self, sim, task, progress):
        remaining = [s for s in SUBTASKS if not progress.get(s, False)]
        # a dropped utensil must be picked and handed over again
        if "place_utensil" in remaining:
            remaining = [s for s in SUBTASKS if s in ("pick_utensil", "handoff", "place_utensil")] + \
                        [s for s in remaining if s == "place_cup"]
        self.expert = ScriptedExpert(sim, task, subtasks=tuple(remaining))

    def metadata(self):
        return {"name": self.name, "backend": "python", "device": "cpu", "checkpoint": None,
                "privileged_state": True, "note": "IK teacher; baseline and demonstration source only"}
