# RescueHands AI — after the hackathon

Submitted to the lablab.ai Intel Physical AI track on 2026-09-16, before 23:30 PKT.
This roadmap turns the submission into a strong portfolio project. Its steps are ordered
by what the measured evidence says will help most.

## Where the learned policy stands (SmolVLA v2, Intel iGPU, seeds 0–9)

| Gripper fault | Supervisor | Success | What happened |
|---|---|---:|---|
| yes | on | 1/10 | hand-off 4/10, utensil placed 4/10, cup placed 2/10; no item left the table |
| yes | off | 0/10 (seeds 0–8: 0/9; seed 9 pending at time of writing) | 5 items knocked off the table, 4 timeouts |
| none | on / off | not run yet | |

**Diagnosis (measured):**
- **Grasp precision is the bottleneck.** In the failed seeds the utensil rose at most
  1.1 cm, compared with 8.7 cm when the task succeeded.
- **The policy trained too little on v2:** 6,000 steps × batch 16 = 0.28 passes over
  336,729 frames.
- **The supervisor works:** it caught every failure, and without it items fell off the
  table.

## Improvement plan

### 1. Train long enough (highest expected gain)
- Continue from `ABDHAM/smolvla_rescuehands_v2` for about 3 epochs (roughly 60k
  steps at batch 16), or use a larger batch on 2 GPUs.
- Keep checkpoint backups on the Hub (keep-alive plus upload loop, or Kaggle "Save & Run All").
- Pick checkpoints on **development seeds 20–39**, never on evaluation seeds 0–9.

### 2. Fix the drift the robot creates itself (DAgger-style data)
- Run the learned policy, save the states it actually visits, and let the scripted
  teacher label the correct action from those states.
- Add those labels to the dataset and retrain. This targets exactly the "states the
  demonstrations never showed" failure.

### 3. Close the open experiments
- **Chunk length:** `--n-action-steps` 10 vs 25 on development seeds.
- **Clean runs:** the matched clean pair (no fault, supervisor on/off).
- **More seeds:** 30–50 evaluation seeds, reported with confidence intervals.

### 4. Physics honesty (held during the hackathon)
- **Jaw friction:** item friction does not reach the grasp, because the pads have
  contact priority 1. Fix it, then regenerate data.
- **Collisions:** detect and count arm–table and arm–plate contacts (record first, then
  decide on stop thresholds).
- **Grasp possession:** judge it from contact history plus coherent motion, not lift
  height alone.

### 5. Intel optimization story
- Full v2 benchmark on an idle laptop: PyTorch CPU, OpenVINO CPU FP32, iGPU FP16,
  INT8 (NNCF), and model caching.
- Report latency together with accuracy cost.
- Asynchronous inference (plan while moving), to get closer to real-time control.
- Compare the Kaggle and laptop exports by action outputs, not only file hashes.

### 6. Presentation
- A 2–3 minute video with voice-over, showing learned runs drawn with the showcase cameras.
- A blog post: "What 569 demonstrations taught a VLA, and what they did not."
- Update the README results after each milestone, and keep every number tied to a
  results folder.

## CV line (accurate today)

> Built a bimanual robot-manipulation system: two SO-101 arms in MuJoCo, a fine-tuned
> SmolVLA vision-language-action policy exported to OpenVINO and run on an Intel iGPU,
> and a physics-aware safety supervisor. Generated 569 demonstrations with GPU
> rendering (16× faster), and made evaluation reproducible with run manifests,
> hash-checked model contracts and 100 tests. Measured a scripted-baseline recovery
> gain from 2/10 to 7/10 under injected gripper faults.

Update the learned-policy line once step 1 raises the success rate.
