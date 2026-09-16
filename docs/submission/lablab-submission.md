# lablab.ai submission — copy and paste

## Project title (max 50 characters)
RescueHandsAI: A dinner table that survives a drop

## Short description
Two simulated SO-101 arms set a dinner table from a spoken instruction. A fine-tuned SmolVLA policy runs on an Intel iGPU through OpenVINO, and a physics-aware supervisor catches drops and missed grasps and retries safely.

## Long description (project write-up)

**Problem.** A vision-language-action policy always outputs an action, even after
reality has gone wrong. With two arms, one slip ruins the other hand's work.
RescueHands AI never trusts an action just because a model produced it: it checks
what physically happened, then recovers.

**Task.** Two SO-101 arms in MuJoCo set a dinner place. The instruction names a fork or
a spoon, which lie in random order. The right arm picks the named one, hands it to the
left arm in the air, the left arm places it beside the plate, and the right arm places
the cup. Every seed changes positions, sizes, mass, lighting and table colour. Grasps
are real contacts: nothing is welded or teleported.

**Architecture.**
- SmolVLA sees 3 cameras, 12 joint positions and the instruction, and outputs 12 joint
  targets in 50-action chunks.
- An action guard enforces joint limits and step size.
- A physics auditor reads simulator ground truth that the policy never sees: held,
  dropped, placed.
- On a drop or a stalled grasp, the supervisor opens both hands, moves to a safe pose,
  and lets the policy retry, at most twice.
- Success is measured from physics: items in zones, upright, released and still, plus
  an ordered in-air hand-off.
- Inverse kinematics exists only in the scripted teacher that generated the 569
  training demonstrations, never in the deployed control loop.

**Workload placement.**
- Kaggle Tesla T4:
  - demonstration generation, with GPU EGL rendering: 14 s per episode instead of
    229 s on the CPU;
  - SmolVLA fine-tuning (6,000 steps, fp16);
  - OpenVINO export with Intel Physical AI Studio (195 s).
- The export was reproduced on the Intel laptop with byte-identical weights.
- Deployed on an **Intel Core i5-6300U with Intel HD Graphics 520:** policy inference
  (OpenVINO on the iGPU), MuJoCo simulation, the supervisor and all evaluations.

**Hardware optimization.**
- **OpenVINO on the iGPU:** 4.53 s mean per 50-action chunk (p95 5.26 s, 341 calls).
  PyTorch on the same laptop's CPU took about 190 s per chunk in a smoke benchmark.
- **Action chunking:** 25 executed actions per inference.
- **Shadow-free rendering:** 6× faster, and identical in training and deployment.
- **Hash-checked model contracts:** inference refuses a mismatched model.

**Results on 10 randomized seeds.**
- Scripted teacher: 8/10 clean. With a real gripper fault it scores 2/10 without the
  supervisor and 7/10 with it.
- SmolVLA v2 on the Intel iGPU, with the fault and the supervisor: 1/10 full task
  successes.
  - The hand-off was completed in 4/10, the utensil placed in 4/10, the cup placed in 2/10.
  - Seed 4 recovered from a real drop and set the whole table.
  - Every failure was detected and labelled.
  - The main weakness is grasp precision: the policy saw each new frame only 0.28 times
    in training.

**Honest limits.**
- Collision checks cover arm-to-arm contact only.
- Item friction does not reach the jaw contacts.
- Simulation pauses during inference.

**Reproducibility.**
- A public GitHub repo with the pinned scene and asset revision.
- Data, training, export, evaluation and benchmark code.
- 100 tests.
- A per-run manifest for every result, and hash-checked contracts.
- The public model and dataset on Hugging Face.

## Technology & category tags
Intel Physical AI Studio, OpenVINO, SO-101, Anomalib, MuJoCo, SmolVLA, LeRobot, Vision-Language-Action, Bimanual Manipulation, Intel iGPU

> Anomalib is on the organizers' required tag list, but **this project does not use
> Anomalib**, so the write-up does not claim it. Add the tag only because the checklist
> requires it.

## Cover image
`docs/submission/cover.jpg`

## Video presentation
A screen recording of the working closed-loop pipeline on the Intel laptop:
1. Show `results/smolvla_v2_sup-on_fault-glitch` and the terminal log of the learned
   policy on the Intel iGPU.
2. Play `artifacts/demo/episode_4_hero.mp4`: the learned policy with the gripper fault,
   supervisor recovery, and the table set.
3. Show the README results table.

## Slide presentation
`docs/submission/rescuehands_slides.pdf`

## Public GitHub repository
https://github.com/Hamza-Atiq/rescuehandsAI

## Demo application platform
Hugging Face (model + dataset) and a simulation video. No live web app, because the
policy must run on Intel hardware.

## Application URL
https://huggingface.co/ABDHAM/smolvla_rescuehands_v2
