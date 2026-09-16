# lablab.ai submission draft — RescueHands AI

Fill every `⟨…⟩` from the final result files before submitting. Nothing in angle
brackets has been measured yet. Every other number in this file is backed by a
committed result or a Hub page.

## 📋 Basic information

**Project title** (max 50 characters)
RescueHandsAI: A dinner table that survives a drop   ← 50 characters

Alternative: RescueHands AI: Two Arms Recover a Dropped Fork   ← 47 characters

**Short description** (one line)
Two simulated SO-101 arms set a dinner place from a spoken-style instruction with a fine-tuned SmolVLA policy running on an Intel iGPU through OpenVINO, while a physics-aware supervisor catches drops and retries safely.

**Long description**

*The problem.* A vision-language-action policy always outputs an action, even when
reality has already gone wrong. In two-arm manipulation, one slip by one hand ruins
the other hand's work. RescueHands AI tests a simple idea: never trust an action just
because a model produced it. Check what physically happened, and recover.

*The task.* Two SO-101 arms in MuJoCo share a dinner table. The instruction names a
fork or a spoon, and both lie on a mat in random order, so the words decide which one
to move. The right arm picks up the named utensil, passes it to the left arm **in the
air**, the left arm places it beside the plate, and the right arm places the cup. Each
seed changes positions, sizes, mass, friction, lighting and table colour. Grasps are
real contacts: nothing is welded or teleported.

*The learned policy.* SmolVLA sees three cameras (overhead and both wrists), 12 joint
positions and the instruction, and outputs 12 joint targets in 50-step chunks. There
is no inverse kinematics in the learned control loop. A scripted IK teacher only makes
the demonstrations. We fine-tuned on free Kaggle T4 GPUs:
- v1: 101 demonstrations.
- v2: 569 demonstrations (336,729 frames). These add perturbed starting states and
  real drop-and-recover episodes, so the policy has seen what getting back on track
  looks like.

*Safety and recovery.* An action guard enforces joint limits and step size. A physics
auditor reads simulator state that the policy never sees: is the item held by both
jaws, supported, in its zone, dropped? When an item really drops, the supervisor opens
both hands, returns to a safe pose, and lets the policy try again, at most twice.
Success is measured from physics, never from the policy's claim. It requires both
items stable in their zones, released and upright, the spare utensil still near its start at the end, and an
ordered in-air hand-off.

*Intel optimization.* The fine-tuned policy is exported with Intel Physical AI Studio
to OpenVINO and runs on an Intel Core i5-6300U with HD Graphics 520. The organizers
allowed non-Core-Ultra Intel hardware. The benchmark compares PyTorch CPU, OpenVINO
FP32 on CPU, FP16 on the iGPU, INT8 weights and model caching. Every speed-up is
reported together with its action difference from the FP32 reference.

*Results (10 randomized seeds).*
- Scripted teacher: 8/10 clean. With a real gripper fault: 2/10 without the
  supervisor, 7/10 with it.
- SmolVLA v2 on the iGPU, same 10 seeds: ⟨x⟩/10 clean with the supervisor, ⟨y⟩/10 without it;
  with the gripper fault ⟨z⟩/10 with the supervisor vs ⟨w⟩/10 without it.
- Inference: ⟨a⟩ s per 50-action chunk on the iGPU vs ⟨b⟩ s in PyTorch on the CPU
  (⟨c⟩× faster), max action difference ⟨d⟩ rad.

*Honest limits.* Collision checks cover arm-to-arm contact only. Randomized item
friction does not reach the jaw contacts. Simulation pauses during inference, so this
is not real-time 20 Hz control.

*Reproducibility.* The repository contains:
- the pinned MuJoCo scene;
- data generation, training, export, evaluation and benchmark code;
- per-run manifests (code revision, arguments, package versions);
- model contracts (joint order, cameras, rate, file hashes);
- public dataset and model links;
- 99 tests.

**Technology & category tags**
MuJoCo · SmolVLA · LeRobot · Vision-Language-Action · Bimanual Manipulation · SO-101 · OpenVINO · Intel Physical AI Studio · Intel iGPU · NNCF · Imitation Learning · Robotics Simulation · Safety Supervisor · Python · Kaggle

## 📸 Cover image and presentation

- **Cover image:** `docs/submission/cover.jpg` (1920×1080). A rendered frame of the
  scripted teacher's in-air hand-off from the presentation camera.
- **Video presentation (about 3 min):** storyboard in
  `docs/presentation/submission-story-and-plan.md`. Footage:
  - ⟨the 10 seeds of the supervised fault run of SmolVLA v2 on the iGPU⟩
  - the showcase renders from `scripts/render_showcase.py`
  - one benchmark table
- **Slide presentation:** `docs/submission/rescuehands_slides.pdf` (10 slides: problem, task,
  architecture, learned policy and data, safety, results, Intel optimization, limits,
  reproducibility, links). ⟨Fill the v2 numbers in make_slides.py RESULTS and rebuild.⟩

## 💻 App hosting and repository

- **Public GitHub repository:** https://github.com/Hamza-Atiq/rescuehandsAI (branch `dinner-table`)
- **Demo application platform:** Hugging Face Hub. The model and dataset are public;
  the demo itself is the recorded simulation video. No live web app, because the
  policy must run on an Intel device.
- **Application URL:** https://huggingface.co/ABDHAM/smolvla_rescuehands_v2
  ⟨confirm it is public after the upload⟩. Dataset:
  https://huggingface.co/datasets/ABDHAM/rescuehands_table_v2
