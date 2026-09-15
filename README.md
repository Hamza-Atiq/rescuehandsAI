# RescueHands AI

Two simulated SO-101 arms, with a plan to learn dinner-table tasks and recover
from mistakes. The Intel challenge corrections at the top of AGENTS.md take
priority over the older package-rescue examples.

## Current milestone

This first slice checks the real robot models, named motor commands, physics,
camera images and saved run evidence. **It does not yet pick up an item, run a
VLA, recover a failed task, or benchmark OpenVINO.** The test cube is a physical
test item, not the finished dinner-table scene.

Read [the research review](docs/research/2026-09-15-review.md) for the idea review,
official sources, required deliverables and what changed after reading Intel's
brief.

## Setup

Use Python 3.12 and uv. Keep the existing dependency lock file.

On this Windows machine, MuJoCo 3.13.0 failed to load under the Anaconda-based
environment. The same library loaded under standalone CPython 3.12.10. A separate
environment avoids changing the original one.

PowerShell:

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv-sim"
uv sync --frozen --python "C:/Users/HP/AppData/Local/Programs/Python/Python312/python.exe"
```

On another machine, replace the interpreter path with your standalone Python
3.12. Linux users can run `UV_PROJECT_ENVIRONMENT=.venv-sim uv sync --frozen --python 3.12`
and use `.venv-sim/bin/python` below.

## Get the real SO-101 model

Run from the project folder:

```text
git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git .cache/menagerie
git -C .cache/menagerie sparse-checkout set robotstudio_so101
git -C .cache/menagerie checkout 8161bba264d7fa7c99ca301e91e7fb44737676ad
```

If the repository is already downloaded, use the last two commands only.
Keep the downloaded LICENSE. The robot model belongs to its upstream authors
and uses Apache-2.0. See [asset notes](assets/README.md).
The application never silently substitutes a different robot.

## Run the checks

```powershell
.venv-sim/Scripts/python.exe -m unittest discover -s tests -v
.venv-sim/Scripts/python.exe -m rescuehandsai --seed 42
```

The second command creates a new folder under `artifacts/` containing:

- `result.json`: measured movement, versions, config, hashes and limitations.
- `steps.jsonl`: commands, robot state and physical contacts.
- Front and overhead PNG images.
- `control-preview.gif`: a short motor-check preview.

The result is called `control_check_passed`. `manipulation_success` is null
because this milestone does not attempt a manipulation task.

Rendering needs a working OpenGL driver. A rendering failure is reported as a
failed check; a blank image is not passed off as a working camera.

## Design

- `contracts.py`: shared observations, privileged state and named actions.
- `control.py`: reject bad names, numbers, time, limits and large command steps.
- `sim.py`: the only module that owns MuJoCo physics and robot mapping.
- `evaluation.py`: continuous stable-placement logic, awaiting a full physics auditor.
- `app.py`: run and record the motor-check episode.
- `configs/simulation.json`: timing, home pose, camera sizes and model location.

Units are metres, radians and simulated seconds. The state/action order is the
six named joints of the left arm, then the six named joints of the right arm.
Actual actuator IDs are resolved by name and checked against their joints.
Grippers are revolute joints in radians, not normalized 0–1 commands.

## Required next work

1. A real contact-based table-item grasp and release.
2. A hand-off or complementary two-arm action, with a multi-step task state machine.
3. Training data and a learned camera/state/language policy path.
4. A physics auditor and bounded recovery.
5. Ten randomized evaluation seeds and a task demonstration video.
6. OpenVINO conversion where supported, a real policy benchmark and Intel deployment.

No physical robot arms are needed. The supplied brief separately states a final
Intel Core Ultra Series 2/3 computer requirement. Keep that requirement visible
until a newer organizer clarification says otherwise. The linked Intel driver
setup is for Ubuntu and is not a Windows installation recipe.
