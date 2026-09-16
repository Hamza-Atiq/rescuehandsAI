# RescueHands AI: independent release review

Reviewed on September 16, 2026, starting from commit `b2b6dcc`.
This is a review of the current files and new fixes. It builds on the earlier
independent review; it does not claim that every possible robot situation was tested.

## Decision

The project is moving in the right direction. It is not yet ready to claim a
reliable learned robot. **My full learned run on seed 0 failed with no injected
fault.** The robot picked up the spoon and reached the two-arm transfer area, but
the auditor reported a drop and two recovery attempts did not finish the task.
This is real learned movement, but not a completed learned table-setting demo.

The best next investment is evidence: learned-policy task completion, an honest
Intel benchmark, and a clear video. Avoid a new dashboard, a new large model, or a
TPU migration before this is done.

## What I checked myself

- Ran the current test suite in `.venv-sim`: **59 tests passed in 233.626 seconds**.
  Log: `artifacts/review_latest_sim_tests.log`.
- The default `.venv` failed to load MuJoCo with Windows DLL error 1114. That is
  a separate environment problem. Use the tested simulation environment.
- Read the revised control loop, fault injection, success checks, data recording,
  training launcher, checkpoint verifier, export adapter and benchmark code.
- Checked the downloaded checkpoint header: 500 saved tensors, 1,800,257,992 bytes,
  and complete data offsets. This checks file completeness, not learning quality.
- Confirmed saved input and output shapes are both 12. The saved normalizer
  state/action statistics also have 12 entries. Camera rename rules are saved.
- Independently decoded the saved normalizer tensors and compared state/action
  mean and standard deviation against `data/merged_local/meta/stats.json`.
  All values were finite; maximum difference was under `4.7e-8`.
- Recounted all 30 `scripted_v2` episode files. They agree with their summaries:
  clean + supervisor 8/10; fault without recovery 0/10; fault with recovery 7/10.
  No recorded episode exceeds its saved step budget. These are saved results,
  not 30 new independent reruns.
- Reproduced a remaining hand-off false positive with a separate fact sequence.
- Scanned 110 current tracked text files for common token/private-key patterns.
  No matches. This is not a full history scan or dependency security audit.
- Inspected the dataset camera image and the video-writing code. At review start,
  the result folders contained no final learned-policy demo videos.
- Ran a new scripted fault/recovery episode on seed 0: **success**, one detected
  spoon drop, one recovery, 658 steps / 32.9 simulated seconds. This independently
  confirms one baseline case, not the whole ten-seed score. Saved video has 329
  frames at 10 fps; six sampled frames were decoded and visually inspected.
  Evidence: `results/independent_recovery_video_20260916/` and
  `artifacts/review_recovery_contact_sheet.jpg`.
- Ran the actual trained export on the Intel GPU for 50 simulation steps: two
  finite action chunks, no recorded collision or invalid action, two clamped joint
  steps. Calls took 6.752 and 5.045 seconds under concurrent work. The deliberately
  short run timed out at 2.5 simulated seconds; this is a successful integration
  smoke test, not a task-success result. Evidence:
  `results/independent_trained_smoke_20260916/`.
- Then ran a **full learned episode**, seed 0, supervisor on, no injected fault,
  default task budget: **FAILED / RECOVERY_EXHAUSTED** after 993 steps. The log
  reports a left-arm spoon drop at 15.35 simulated seconds, then failed grasps at
  32.50 and 49.65 seconds. Two retries were used. The cup and utensil were not in
  their target zones, and the spare utensil had also moved. No cross-arm collision
  was recorded. This is one seed, not a ten-seed success estimate.
  Evidence: `results/independent_trained_full_20260916/episode_0.json` and its MP4.
- The full learned run made 39 inference calls (mean 4.568 seconds; p95 4.883).
  Episode execution took 300.06 wall seconds, excluding model construction.
  These are observed timings under concurrent work, not a controlled benchmark.
  The video has 496 frames at 10 fps. Six sampled frames were decoded and viewed:
  `artifacts/review_trained_full_frames.jpg`.
- Also inspected six frames around 13–16.5 simulated seconds. They show the spoon
  leaving the left gripper near the plate before the recovery move. The exact
  cause still needs action/contact traces; video alone cannot tell whether the
  model opened too early or lost its grip. Contact sheet:
  `artifacts/review_learned_failure_closeup.jpg`.

Repeat the small read-only audit with:

```powershell
$env:PYTHONPATH='src'
& .venv-sim/Scripts/python.exe -B scripts/review_release_evidence.py
```

Machine-readable evidence: `artifacts/independent_release_review.json`.

## Fixes that hold up

1. **Model shape:** the old six-joint state description is now 12 in the actual
   downloaded checkpoint. The training launcher rebuilds input features.
2. **Recovery time:** the shared `_step` method advances the fault clock during
   recovery and enforces the step budget. New regression tests pass.
3. **Task limits:** normal runs now use the task's timeout and retry limit.
4. **Cup direction:** an upside-down cup no longer passes just because its centre
   is at the right height. Angular speed is also checked.
5. **Joint names in commands:** missing and extra names are rejected before clamping.
6. **INT8 cache:** the cache name now includes a hash of the source graph and weights.
7. **Backend comparison:** the new comparison script uses the same scene and zero
   starting noise in all three backends. This is the right comparison design.
   An input-type bug was reproduced; another session corrected the comparison
   input during this review. A completed comparison is still required.

## Remaining issues, in order

### P1 — PyTorch benchmark input bug; comparison fix needs a completed rerun

The reviewed version of `scripts/compare_backends.py:71` passed a Python dictionary to Intel
`SmolVLA.predict_action_chunk`. The installed Physical AI API expects its
`Observation` object and calls `batch.to(self.device).to_dict()`.
The current comparison log ends with `AttributeError: 'dict' object has no attribute 'to'`.
I checked the installed method itself to confirm this contract. The optional
PyTorch row in `scripts/benchmark_intel.py` repeats the same mistake; it records
an error instead of producing a real baseline. The exported `InferenceModel`
accepts dictionaries, which explains why the OpenVINO smoke test can still work.
I also independently executed the installed method's input-handling code with a
minimal initialized stand-in and reproduced the same dictionary error, without
loading another large model. That proves this input-contract failure; it is not a
completed three-backend numerical comparison.

**Live update:** while this review was running, another session changed
`studio_chunk` to construct `physicalai.data.observation.Observation`, with
`images.images.camera1..3` after flattening. This addresses the identified input
type and naming problem. I read the diff. Do not treat the old traceback as proof
the new version still crashes. The optional benchmark row still uses a dictionary
at the last source check, and no completed agreement report was available.

**Fix:** construct the documented Physical AI `Observation` for Intel PyTorch,
using the exact camera feature names that the policy declares. Keep the LeRobot
and exported-runtime input adapters separate. Add a small input-contract test,
then rerun the same-input, same-noise comparison before quoting an optimization
speedup. Do not silently skip the failed baseline row.

### P1 — Learned task completion and learned recovery need proof

This is now a reproduced failure, not just missing evidence: the full learned
seed-0 run described above did not complete even without an injected fault.
Treat this as the release's first priority.

The committed success rows are `scripted_teacher`, which uses simulator positions
and inverse kinematics. The learned adapter receives images, joint positions and
the instruction; it does not receive object truth. This separation is good.
But the teacher's success rate cannot be used as the AI's success rate.

**Fix:** run one full clean learned episode first, then a small development set.
If it works, run 10 seeds with saved episode JSON and video. Pair fault-on tests
with the same seeds, task, model and fault settings. Count how many faults actually
fired; an arm that never picked anything up did not survive a drop test.

**Immediate debugging order for this failed episode:**

1. Inspect the 13–16 second interval and log the spoon pose, jaw contacts, actual
   joint positions and commanded targets around the loss. The label alone cannot
   distinguish a premature release, bad placement, or a grasp-monitor false alarm.
2. Finish the corrected native/Intel/export comparison on these observations,
   not just the initial home pose. If predictions disagree, fix the conversion
   before collecting more training data.
3. If they agree, compare the learned motion with a successful teacher episode.
   Test a shorter executed chunk on a small development set, with the same
   checkpoint. This means looking again sooner; it costs more inference time.
   Do not change ten knobs at once or claim that shorter chunks are already a fix.
4. If the policy still misses placement or cannot reacquire the dropped spoon,
   add demonstrations of those states and mixed clean/recovery fine-tuning.
5. Freeze the chosen version and run the complete evaluation. Do not silently
   widen target zones, remove the hand-off requirement, or replace learned actions
   with IK to turn failures into successes.

The training config says 12,000 requested steps. That file alone does not prove
12,000 updates finished. Save the final training log, checkpoint step, dataset
revision, model revision and checkpoint hash with the release.

### P1 — Hand-off checker still accepts a table-assisted transfer

File: `src/rescuehandsai/evaluation.py`, `HandoffTracker.update`.

Independently reproduced sequence:

1. Right hand holds the fork in the air: stage `right`.
2. Both hands hold it in the air: stage `shared`.
3. Left hand still touches it while it rests on the table: stage stays `shared`.
4. Left hand lifts it: stage becomes `done`.

This breaks the checker's stated rule that a hand-off stays in the air.
The existing tests do not cover support while a hand is still holding it.

**Fix:** before a transfer is complete, cancel the shared stage whenever support
returns. Preserve the valid right-hand pickup from the table. Add this exact
regression case and rerun both normal and fault evaluations. Do not assume the old
8/10 and 7/10 are unchanged after changing a success rule.

### P1 — Joint order is still not stored in the model contract

Files: `training/verify_checkpoint.py`,
`src/rescuehandsai/policies/smolvla_exported.py`, `scripts/export_openvino.py`.

The checkpoint has no `action_feature_names`. The verifier accepts this absence.
The adapter compares current simulator names with names supplied from the same
simulator. That does not check the order used during training. An accidental arm
swap could still pass a 12-number shape check.

**Fix:** save a small contract next to the checkpoint and export: ordered state and
action names, camera map, absolute-radian action units, 20 Hz, dataset revision and
model hash. Recover these names from the exact training dataset, not from a guess.
Compare that contract with the simulator before loading inference. Deliberately
swap two names in a test and require rejection.

### P1 — Recovery behaviour is outside the saved training demonstration path

The data generator directly runs the clean scripted expert. It does not record
faults or recovery episodes. The learned `after_recovery` only clears queued actions
and resets model state. The supervisor supplies a fixed open-and-home movement.
It does not teach the VLA how to pick up a newly dropped item.

**Fix:** test this before promising learned recovery. If it fails, collect a small
set of successful recovery demonstrations from the actual failure positions and
fine-tune on a mix of clean and recovery data. Keep deployment learned; do not
quietly substitute the IK teacher. If time runs out, label teacher recovery and
learned control as separate evidence.

### P2 — Dataset acceptance is weaker than episode acceptance

`scripts/generate_dataset.py` checks the final instant once. The runner requires
10 consecutive successful steps. A demonstration can pass the former without
showing that the object stays settled for the required time.

**Fix:** keep stepping and recording the final held command until the same stability
window passes. Record seed, generation revision and outcome with each episode.
Audit the current data before choosing to regenerate or retrain; do not throw away
a trained model simply because a filter needs improvement.

### P2 — Support is any non-arm contact, not verified upward support

`compute_facts` marks an item supported when it touches another non-arm object.
It does not identify a valid support surface or upward force. README says table or
plate, while code also accepts another item. A side contact is not proof of resting.

**Fix:** distinguish table/plate support from other contacts and check the support
direction or force. Add a side-contact rejection test. This is an evaluation gap,
not proof that the recorded successes were all false.

### P2 — Benchmark and comparison need stronger release evidence

- Run the trained model, not the older six-joint spike.
- The agreement script reports differences but has no finite-value check,
  acceptance threshold or failing exit status. Add these and compare several real
  observations, including grasp, hand-off and placement. One initial scene cannot
  establish behaviour preservation.
- The benchmark compares action chunks, not task success after quantization. Run
  task evaluation for the precision actually used in the video.
- For INT8, hash the manifest and tokenizer too; they change model meaning. Write
  cache files atomically and verify all required files before reusing a cache.
- `ov.save_model` in the INT8 helper uses default FP16 compression for remaining
  floating-point weights. Record this mixed storage choice or set it explicitly.
- Time one model at a time on an idle machine. Report load, first call and warm
  calls separately. These review runs overlap other work and are not benchmarks.
- Video advances in simulated time while inference pauses physics. Label that.
  Twenty simulator steps per second does not establish live 20 Hz AI inference.
- Inspected the actual exported XML: constant types include 2,072 f32 and 1,131
  bf16 entries. Thus `fp32` is not a description of every stored weight. The Intel
  SmolVLA exporter explicitly disables FP16 save compression, but the imported
  model still includes bf16 constants. Record storage types separately from device
  execution precision; do not infer one from a directory name.

Reference for the save default:
[OpenVINO save_model documentation](https://docs.openvino.ai/2025/api/ie_python_api/_autosummary/openvino.save_model.html).

### P2 — Full challenge coverage is deliberately smaller than the brief

Current task: choose fork/spoon, hand it over, place it and the cup. Plate placement,
drawer opening and pouring are not implemented. The cup/utensil size, mass,
friction, positions, light and table colour vary. This is useful randomization,
but not arbitrary object shapes or real-world testing.

**Fix:** describe the submitted workflow precisely. Make this part reliable before
adding another difficult action. Add a paired same-scene fork-versus-spoon test to
prove that changing the words changes the chosen object. Current seeded variety
alone is weaker evidence of language use. No need to add a separate VLM just for
its name; it would need its own grounded task and tests.

### P2 — Safety claims must match the implemented checks

The simulator stops on cross-arm contact and non-finite state. It validates joint
limits and command changes before stepping. It does not implement a full predicted
collision check, contact-force limit, or general same-arm/table collision guard.
The fixed recovery move still has the simulator checks, but does not run the full
failure auditor after every recovery step.

**Fix:** describe these as specific simulation safeguards. If extending recovery,
audit its intermediate states and terminate on new terminal failures. Do not call
this a certified safety controller or claim that all collisions are prevented.

## Security and release hygiene

- No common credential patterns found in the current tracked text scan. It did
  not search all history, all large files, or every kind of secret.
- `.gitignore` does not cover `.env`, `.env.*`, `.venv-train/`, or common private-key
  files. Add those before sharing more notebook and setup files. Keep `.env.example`
  if it contains placeholders only. A missing ignore rule is a risk, not evidence
  that a secret leaked.
- The Intel loader dynamically imports classes named in an export manifest.
  Treat model/export folders as trusted executable inputs. Do not accept arbitrary
  user-uploaded manifests. Pin the official export and check its hash before use.
- Inspected subprocess calls use argument lists, not `shell=True`. Safetensors
  avoids the ordinary pickle checkpoint path. These are good choices, not a full
  security guarantee.
- Pin Hugging Face dataset/model revisions and publish hashes. Current training
  config has a null dataset revision.
- No dependency vulnerability scan was completed. Do not label the project
  'security audited' or 'production safe'. No remote service/authentication layer
  was found that needs a new login system for this local demo.
- Replace the downloaded model card's stock ACT/physical-SO100 instructions with
  the actual simulated dual-SO101/SmolVLA instructions. Keep upstream notices and
  distinguish code and model licenses.

## Pictures and video

The scene is readable and the two yellow arms stand out. The overview is too tight
on the right: the cup is partly clipped in the inspected training frame. Do not
change trained policy cameras just to make a prettier video. Use a separate front
or wider presentation camera.

The current MP4 writer adds a coloured strip. Viewers cannot tell the instruction,
seed, policy or failure from that strip alone. Add captions in the presentation
edit or a video-only overlay. Show these five things:

1. Instruction and seed.
2. `Learned SmolVLA` or `Scripted teacher`, always visible.
3. `Normal / drop detected / recovering / succeeded / failed`.
4. Simulated time and any playback speed change.
5. Final physical checks and the matching run ID.

Do not use the old motor-check GIF as proof of table setting. The final package
needs actual manipulation clips and the 10-seed results, including failures.

## What to do next

See `docs/presentation/submission-story-and-plan.md` for the timed plan, video
storyboard, slide outline and a claim checklist. Freeze the evidence before
polishing the final numbers and narration.
