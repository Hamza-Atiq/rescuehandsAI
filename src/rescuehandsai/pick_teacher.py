"""Pick-only teacher: approach the named utensil, grasp, lift, then hold (spec section 7).

It wraps PickExpert with only the `pick_utensil` subtask. The expert's script always
ends with a home move that would lower the utensil again, so the teacher stops feeding
the expert as soon as it reaches that phase and repeats the last command instead. IK
lives inside the expert only; the deployed policy never uses it.
"""
from .contracts import BimanualAction
from .pick_cells import PickTask
from .pick_expert import PickExpert
from .task import TaskSpec


class PickTeacher:
    uses_privileged_state = True

    def __init__(self):
        self.expert = None
        self.phase = "start"
        self.finished_pick_at = None
        self._targets = None
        self._steps = 0

    def reset(self, sim, task: PickTask):
        spec = TaskSpec(task_id=f"pick_{task.seed}_{task.cell}", instruction=task.instruction,
                        utensil=task.utensil, seed=task.seed)
        self.expert = PickExpert(sim, spec)
        self.phase = "pick"
        self.finished_pick_at = None
        self._targets = dict(sim.previous)
        self._steps = 0

    def wants_images(self) -> bool:
        return False

    def metadata(self) -> dict:
        return {"name": "pick_teacher", "backend": "python", "device": "cpu", "checkpoint": None,
                "subtasks": ["pick_utensil"], "uses_ik": True}

    def act(self, obs) -> BimanualAction:
        if self.expert is None:
            raise RuntimeError("PickTeacher.act called before reset")
        if self.phase == "pick":
            action = self.expert.act(obs)
            # `home` is the expert's own wind-down move and `done` means the script ran out.
            # Either way the pick is over: freeze the command so the lift is held.
            if self.expert.subtask == "home" or self.expert.done:
                self.phase = "hold"
                self.finished_pick_at = self._steps
            else:
                self._targets = dict(action.targets)
        self._steps += 1
        return BimanualAction(obs.timestamp, dict(self._targets))
