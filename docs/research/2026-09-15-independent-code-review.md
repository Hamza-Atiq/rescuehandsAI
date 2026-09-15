# Independent review: RescueHands AI

## Main answer

Keep the project and keep SmolVLA. The robot foundation is working. Fix the
training-to-export contract before spending many GPU hours. The new T4 precision
change is a sensible experiment, but its speed is not yet independently measured.

This review reads the application, scripts, training code, configuration, and
tests, plus the installed LeRobot and Intel code at the important boundaries.
The main application snapshot is `2710e709484ac9a624b934cf0d5a94a1ea9e8ddd`.
The later `026f0a3` commit adds the independent probe, not another training change.
Production code was not changed by this review. The new review scripts are
separate from the robot and training code.

## What I ran myself

| Check | Independent result |
| --- | --- |
| Existing test suite | All 50 tests passed, 267.977 seconds |
| Fresh 10-seed scripted baseline, supervisor on, no injected fault | 8/10 succeeded; seed 6 failed during recovery planning, seed 9 exhausted recovery attempts; no collision episodes |
| Extra learned-adapter tests | All 3 passed; checked twelve named outputs, three camera slots, image scaling, action queue, and reset after recovery with a spy model |
| Installed LeRobot factory | Reproduced state metadata `[6]` and action metadata `[12]` with a twelve-joint dataset; expensive model loading was replaced by a stub |
| Physical upside-down cup fixture | MuJoCo cup up-axis was approximately -1, but `cup_upright` returned true; the complete fixture did not pass because its settled check was false |
| Recovery time bound | `max_steps=2` produced 40 executed steps before timeout |
| Task-specific limits | A task allowing zero recoveries still received one; a 0.1-second task reached 2.05 simulated seconds |
| Fault countdown during recovery | After 38 recovery steps, a fault with 7 steps left still had 7 steps left and remained active |
| Unknown action name | The runner silently removed an unexpected joint instead of rejecting the command |
| Local merged dataset | Checked every state/action row and decoded every frame of all three video files |
| Existing OpenVINO model structure | Actual stored model has six state inputs and six action outputs; CPU and GPU are discoverable; no random operators in this export |
| Actual OpenVINO GPU inference | Two calls on Intel HD Graphics 520 returned finite `(50, 6)` chunks with identical outputs for identical input; 7.258 and 5.496 seconds in a busy session, not a controlled benchmark |

The spy tests prove the adapter's plumbing, not learned robot skill. The factory
probe executes the installed factory code, but does not train or load a large
model. Physical test fixtures deliberately set starting object poses; this does
not mean the deployed robot teleports objects.

Commands:

```text
.venv-sim/Scripts/python.exe -B -m unittest discover -s tests -v
.venv-sim/Scripts/python.exe -B scripts/review_independent.py
.venv-sim/Scripts/python.exe -B scripts/review_policy_contract.py
.venv-pai/Scripts/python.exe -B scripts/review_data.py
.venv-pai/Scripts/python.exe -B scripts/review_openvino.py
```

The ten-scene baseline is saved in
`results/independent_review_clean/summary.json` and its per-episode JSON files.
I did not repeat the whole ten-seed injected-fault comparison: the fresh suite
reproduced its seed-1 failure/recovery pair, and the separate countdown probe
found that the fault duration needs fixing before a fair comparison is rerun.

## Fix before the long training run

### 1. The checkpoint can keep the wrong state size

Relevant code:

- `training/lerobot_train_launcher.py`, `make_policy_float32`
- Installed `lerobot/policies/factory.py`, lines 509-511
- `scripts/export_openvino.py`, state-schema guard

LeRobot replaces the action feature using the dataset, but replaces input
features only when they are empty. The pretrained SmolVLA config already has
an input state of six. Our dataset has twelve. The new launcher changes weight
precision, but does not correct those input features.

I executed the installed factory with a small stand-in model. It left state
metadata at six and changed the output to twelve. Intel's schema builder reads
the state shape from that saved metadata, so our exporter rejects it.

This does **not** prove that training used only six numbers. LeRobot's actual
state preparation pads the received tensor to 32; a twelve-number tensor can
still be used during training. A checkpoint may be repairable without repeating
training if its real tensors and normalization statistics are correct.

Required fix:

1. Build the policy's input and output feature definitions from this dataset,
   applying the existing camera rename map before model creation.
2. Assert twelve state values, twelve actions, and the three expected cameras.
3. Verify the checkpoint saved after a short run has those same definitions.
4. Verify state/action means and standard deviations each have twelve values.
5. Record the exact joint order and camera order with the checkpoint/export.
6. Load it in the real inference path and request one finite `(50, 12)` chunk.

`scripts/review_kaggle_preflight.py` prints GPU details, checkpoint shapes and
weight types without training, uploading or deleting anything. It is syntax
checked locally; its real CUDA checks must be run on Kaggle.

### 2. Speed: the model update is the bottleneck

The supplied Kaggle log reports:

- 101 episodes, 57,173 frames.
- Batch size 16, one process.
- Update time: 6.929 seconds.
- Data time: 0.135 seconds.
- Progress display: about 7.2-7.4 seconds per step.
- 20 steps, 320 training samples. This is a setup check, not learned task success.

At that rate 12,000 steps take roughly 24 hours, before final evaluation and
export. The data loader is a small part of the measured step. Changing video
downloads or adding many workers is not the first fix.

The installed SmolVLA code loads its vision/language backbone as bfloat16.
The owner confirmed Kaggle T4 x2. T4 is a Turing GPU; native bfloat16 support
belongs to newer architectures. The new launcher converts the policy to
float32 and lets Accelerate use float16 for suitable operations. This is a
reasonable way to test a format better suited to T4.

The code correctly passes `ACCELERATE_MIXED_PRECISION` to Accelerate. Do not
decide that AMP is off just because the earlier config says `policy.use_amp=false`:
this installed trainer uses `accelerator.autocast()`.

However, the original log alone does not prove the exact slowdown mechanism or
the speedup. No new Kaggle measurements are available to this review. The browser
could not access the supplied private notebook page. There is no local CUDA GPU.

#### Small experiment before choosing the long command

Use the same dataset and camera settings in each run:

| Trial | GPUs | Batch per GPU | Total batch | Purpose |
| --- | ---: | ---: | ---: | --- |
| A | 1 | 16 | 16 | float32 weights + float16 mixed precision |
| B | 1 | 16 | 16 | float32 without mixed precision, comparison/fallback |
| C | 2 | 8 | 16 | same total batch, check whether two GPUs really help |
| D, only if useful | 2 | 16 | 32 | measure greater throughput with a larger total batch |

Use at least 10 warm-up steps and about 30 measured steps, with unique output
folders. Record examples/second, update/data time, GPU memory, finite loss,
finite gradients, and whether optimizer updates are being skipped. Test save
and reload once. The existing 12-step probe is a quick filter, not a stable
performance benchmark. Do not assume two GPUs are twice as fast.

Keep the backbone frozen, keep all three camera views, and keep the current
image processing until precision and multi-GPU behavior are measured. Changing
image sizes, layers and precision together would hide which change helped or
damaged learning. Reducing inference denoising steps is not a fix for training
update time.

Do not copy 12,000 steps blindly after changing the total batch. The original
plan is 192,000 sampled frames (12,000 x 16), about 3.36 passes over this dataset.
At total batch 32, 6,000 steps see the same number of samples, but the learning
behavior is not guaranteed to be identical. Choose the final duration using
robot rollouts, not only loss or a sample-count formula.

### 3. Stay with T4 for this deadline

A TPU can train supported PyTorch models through PyTorch/XLA, but it is not a
drop-in replacement for this CUDA launcher. This repository hard-codes CUDA
for training and NVIDIA/EGL for GPU data generation. A TPU path would need a
separate environment and launch/device checks, and testing of SmolVLA's custom
attention, saving and export.

Recommendation: use the two T4s after the short measurements above. Do not spend
the remaining time moving to TPU just because it has eight devices.

Primary references checked:

- [PyTorch/XLA documentation](https://docs.pytorch.org/xla/master/)
- [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/en/index)
- [NVIDIA floating-point support](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html)

## Robot correctness and recovery

### 4. Fault duration changes when recovery is enabled

`EpisodeRunner._safe_pose` advances physics directly. It does not call
`GripperGlitch.before_step`, which advances the fault countdown. A nominal
half-second fault therefore lasts longer in a recovering run. This is a real
comparison problem, not just a naming issue.

Use one shared step function for ordinary and recovery motion. It should advance
the fault clock, enforce the time budget, run the physical checks, and record the
issued action. Prefer fault expiry based on simulated time rather than how often
one outer loop happens to call a method. Check equal physical fault duration in
supervisor-on and supervisor-off runs before publishing their comparison.

### 5. Episode and task limits are not enforced everywhere

The main loop checks `max_steps`, but recovery can execute many extra steps
without checking it. The runner also has its own defaults instead of honoring
`TaskSpec.timeout_s` and `TaskSpec.max_recoveries`.

Use the task limits unless an explicitly recorded evaluation override is given.
Check the budget before every physics/control step, including recovery. Reset
the placement-stability counter after a recovery. Preserve the correct failure
label if recovery ends in a simulator error.

### 6. Success still needs a stronger physical definition

Earlier support/spare-item issues have been improved. Remaining gaps:

- **Cup upright:** height alone cannot distinguish an upright cup from an
  upside-down cup. My physical fixture reproduced that wrong subcheck. Use the
  cup's orientation relative to the table normal.
- **Handoff:** a set saying both arms held the item at some point does not prove
  an in-air transfer. Right grab, table drop, left pickup also fills that set.
  Track ordered possession and support events through the transfer.
- **Settled:** linear speed alone misses a rotating item. Check angular speed too.
- **Dataset filtering:** it checks the final instant. Require a continuous stable
  placement interval, using the same evaluator as deployed episodes.
- **Support:** any non-arm contact currently counts as support. A side contact
  against another object is not necessarily load-bearing support. Use contact
  direction/force or a justified support rule for the simple scene.

Do not replace the current dataset immediately. First audit actual demonstrations
with the strengthened criteria, then keep the good episodes.

### 7. The runner bypasses part of the action contract

The lower-level validator correctly rejects unknown or missing joints, but
`clamp_action` rebuilds the command from expected names first. It silently
removes extra names; missing names can escape as uncaught `KeyError`.

Check the exact name set before clamping. Record bad policy schemas as structured
failures instead of crashing the evaluation and losing the remaining seeds.

### 8. Recovery training examples are still missing

The current generator calls the ordinary scripted expert directly. It does not
inject faults or run recovery. It also does not record a recovery label. The
approved design promised successful recovery demonstrations.

First prove the learned clean task. Then add a small, clearly tagged set of
successful re-grasp/recovery examples. Keep clean and recovery evaluation
separate. Scripted recovery results do not prove that SmolVLA can recover.

## Dataset findings

The local merged dataset matches the counts in the supplied training log. This
is not a cryptographic comparison against the current remote Hub revision.

- 101 episodes, 57,173 rows, 12 distinct instruction strings.
- State and action arrays both have shape `(57173, 12)`.
- All values are finite; all twelve joints vary across the dataset.
- Names match the simulator's left-six/right-six order.
- Zero action joint-limit violations.
- Largest within-episode command change: 0.14203012 rad, below 0.15 rad.
- Timestamp error from frame-index / 20 Hz: less than 0.000001 seconds.
- Recomputed state/action means agree with the stored means to about 0.000003.
- All stored state/action standard deviations are positive; their minima are
  approximately 0.160265 and 0.160724.
- Every frame of all three AV1 files decoded successfully at 256 x 256.
- Each video has 57,220 frames. The 47 additional frames are accounted for by
  longer video spans in 12 episode metadata entries; all episode starts align
  with the 20 Hz grid. Do not use a single global row number as the video index.
  LeRobot uses the per-episode `from_timestamp` offsets.
- Current local attempt logs contain 123 saved attempts across the available
  shards, more than the merged dataset's 101. Those are different sets, not
  evidence that the merged set has 123 episodes. None of those saved attempt
  seeds is in 0-9. All saved logs include the newer support-check field.

The schema/timing/numeric checks are good. They do not independently prove that
every saved demonstration contains a valid in-air handoff. Source seeds, source
revision and scene configuration are not stored in the merged episode metadata;
keep a provenance manifest when publishing the final dataset.

I also inspected the first decoded overhead, left-wrist and right-wrist frames.
The utensils and their different shapes are visible. The cup is near the right
edge of the overhead view. A little more camera margin could help later, but do
not change the deployment camera after training without matching the training
images. The three inspected PNGs are in `artifacts/independent_review/`.

## Export and benchmark checks

### 9. Prevent stale INT8 models from being benchmarked

`scripts/benchmark_intel.py` uses a shared `results/openvino_int8` folder and skips
compression whenever `smolvla.xml` exists. A later benchmark of a different
checkpoint can silently reuse the earlier INT8 weights.

Use an output folder keyed by a hash of the source model and compression
settings, or generate the INT8 model inside that benchmark run. Record source
hashes for every row. `make_int8` also uses `ov.save_model` without explicitly
preserving FP32 for remaining floating-point weights; record the actual saved
precision rather than implying that only INT8 compression changed.

### 10. Prove trained-model agreement before the final demo

The existing stored export really is a pretrained six-joint spike. It is useful
runtime evidence, but cannot control the twelve-joint robot as the final model.

After the schema fix and a saved short checkpoint:

1. Check all weights expected by the Intel loader are loaded; do not ignore
   missing/unexpected-weight warnings.
2. Compare native LeRobot and Intel PyTorch outputs using the same real images,
   instruction, normalization and **same initial action noise**. Intel currently
   defaults to zero initial noise; native LeRobot normally samples noise.
3. Compare Intel PyTorch and OpenVINO on that same input and noise choice.
4. Run the same task seeds through both paths and compare actual task success.
5. Measure latency in an otherwise idle run. Five samples and a busy CPU are not
   enough for a strong p95 or speedup claim.

The adapter's current reset check compares simulator names against the names
passed from that same simulator. It does not read the training joint order from
the checkpoint. Save and check the training order explicitly.

## What to build next, in order

1. Correct/verify the twelve-joint checkpoint schema and normalization.
2. Run the short T4 precision and two-GPU measurements; choose a working setting.
3. Save an early trained checkpoint and run one real learned-policy episode.
   A few hundred steps are a wiring check; do not expect task mastery that early.
4. Continue training with checkpoints every 500-1,000 steps. Evaluate intermediate
   checkpoints on separate development seeds (for example 100-109).
5. Fix the shared recovery clock, task budgets and success checks in parallel
   with training; avoid changing the observation/action contract mid-run.
6. Add recovery examples after the clean learned behavior starts working.
7. Run paired evaluations on the final seeds and measure Intel inference fairly.
8. Finish the submission video, slides and README using only measured results.

The existing seeds 0-9 have already guided teacher debugging. Keep that visible;
they are not a pristine unseen test set. Also try a separately fixed unseen set
before claiming broad generalization. Test both fork and spoon instructions on
the same scene to show that language changes behavior.

## Limits of this review

No access to the live Kaggle kernel was available. The new CUDA speed, actual
20-step saved checkpoint, distributed optimizer behavior and trained bimanual
task success could not be independently tested here. No final learned checkpoint
was present locally. A 20-step training log is useful setup evidence, not evidence
of learning, recovery, Intel optimization quality, or challenge completion.

The OpenVINO smoke check took about 621.16 seconds to load the existing model
while other review work was active. Its two subsequent calls took 7.258 and
5.496 seconds. This is a useful warning to test cold start, warm start and model
caching on an idle laptop before the demo; it is not a controlled comparison
with the earlier 4.68-second claim.
