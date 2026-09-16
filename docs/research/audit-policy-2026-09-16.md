# Policy, training, and Intel export audit — 2026-09-16

This is a read-only source audit. No product code was changed. Every line of the assigned 21 source files was read, plus `tests/test_contract.py`, `scripts/review_policy_contract.py`, the saved model configuration/processors/training configuration, available policy result JSON, the base-model Intel spike report, and relevant installed LeRobot/Physical AI source. Binary weights were not semantically inspected. Video decoding of two training shards was attempted, but PyAV returned PermissionError; no visual claim is made about those videos.

## Verdict

The design direction is sensible: a clear 12-joint action format, three cameras, a scripted teacher separated from learned deployment, and explicit Intel inference. The current learned system is not ready to be called reliable. The saved raw learned run has one success in nine completed episode JSON files (seed 4 succeeds; seeds 0,1,2,3,5,6,7,8 fail). Seed 9 has a video but no episode JSON, and the run has no summary. This is not a completed 10-seed result. The independent full supervised seed-0 run failed with RECOVERY_EXHAUSTED. No trained-policy benchmark or native-versus-export agreement output was found.

## Confirmed findings

### P1 — Existing learned artifacts cannot run with the current contract requirement

`src/rescuehandsai/policies/smolvla_exported.py:32` unconditionally calls `load_contract`; `src/rescuehandsai/contract.py:44-47` fails if the file is absent. Both `models/smolvla_rescuehands/task_contract.json` and `models/openvino/fp32/task_contract.json` are absent (checked on disk). `scripts/export_openvino.py:29` similarly refuses the saved checkpoint. Historical result JSON predates this requirement and does not prove current code runs. Migration needs verified contracts bound to the real existing artifacts; do not invent metadata simply to make checks pass.

### P1 — The requested fine-tuning learning rate does not reach the active optimizer

`training/kaggle_pipeline.py:168-169` appends `--optimizer.lr`, but does not disable `use_policy_training_preset`. Installed LeRobot `configs/train.py:137-139` replaces optimizer AND scheduler with policy presets during validation. SmolVLA `configuration_smolvla.py:135-146` reads `policy.optimizer_lr` for those presets. Therefore the intended smaller learning rate can be silently replaced (or an incomplete optimizer CLI configuration may fail earlier). Use the policy preset field, or explicitly define a complete optimizer/scheduler with presets disabled. Assert the effective saved learning rate before a long run.

### P1 — The automatic `all` pipeline does not verify its checkpoint

`training/kaggle_pipeline.py:261` only verifies for `train` and `verify`, omitting `all`, although `all` is the default stage. Thus the advertised all-in-one route bypasses the final schema/action/contract gate. Add the same gate to all, and retain its JSON output as an artifact.

### P1 — The training pipeline cannot reliably publish a deployable contract

Training pushes the model from the LeRobot process (`training/kaggle_pipeline.py:166-179`), then verification creates a local contract afterward (`186-189`, `261-262`). There is no later upload of the contract. A freshly downloaded trained model can therefore lack the file deployment requires. In addition, `setup()` installs only dependencies (`51-55`), and `verify()` supplies no PYTHONPATH, while `verify_checkpoint.py:89` imports the src-layout project. In a fresh setup with no inherited PYTHONPATH/editable install, that contract step raises ModuleNotFoundError. This import-path failure is inferred directly from the setup and invocation, not reproduced in Kaggle. Package/install the project in the training environment, then verify and publish the complete validated directory.

### P1 — No fresh closed-loop proof accompanies model/export changes

`training/kaggle_pipeline.py:163` disables evaluation. `training/verify_checkpoint.py:72-86` checks only one dataset frame, finite output, shape, and a reported error that has no pass/fail limit. This is a useful smoke test, not learned task validation. The current `results/horizon_diagnosis.json` samples ten frames from the training dataset and explicitly describes an open-loop comparison. It cannot explain away the eight failures among nine completed learned episodes. Add a development validation set, closed-loop checks, and preserve final seeds for the final report. Do not repeatedly select hyperparameters against the final ten seeds.

### P2 — Horizon diagnosis fails when sampled frames have unequal remaining lengths

`scripts/diagnose_horizon.py:60` truncates the accumulated array to the current frame horizon. If an early sample has four future frames and a later one has fifty, adding arrays of lengths 4 and 50 raises ValueError. Reproduced with the exact NumPy expression. Even in monotonic-shortening cases, later horizon statistics are dropped. Accumulate per-index sums and counts, or sample only frames with a complete chunk. Labels at `64-66` also use the last available error as if it were step 10/24/49 when fewer steps exist; use null or the real index.

### P2 — Contract hashes are recorded but never checked

`src/rescuehandsai/contract.py:51-63` validates names, camera map, units and rate only; it never reads `files`. Export and runtime call this function. Replacing model weights while leaving task_contract.json untouched passes the current gate. `scripts/export_openvino.py:51-55` creates new output hashes but does not first verify source hashes. Include config, processors/normalizer, tokenizer and model files in integrity validation, not just weights. This is an accidental stale-artifact risk, not an assertion of tampering.

### P2 — Contract verification asserts important facts without checking them

`training/verify_checkpoint.py:64-67` permits missing checkpoint action_feature_names and only copies action names from the chosen dataset. It never checks the dataset state-name order against action order. It writes the caller/default control rate (`29`, `91-93`) without checking dataset fps. Camera order is taken from a constant rather than verified against the saved rename processor. A contract can therefore certify the wrong dataset, state ordering, camera mapping or frame rate. Read these facts from the recorded training manifest and check them. Shape alone is insufficient, exactly as the contract module docstring says.

### P2 — Dataset/model versions are not immutable

`training/kaggle_pipeline.py:158-160` selects Hub names without commit revisions. `verify_checkpoint.py:91` records dataset.revision, which the installed LeRobot class defaults to its version tag (v3.0), not necessarily an immutable content commit. Existing train_config.json has dataset revision null. Generated dataset frames omit seed, scene parameters, code revision and artifact hashes (`recorder.py:42-49`); seed attempts live beside shards and are not included by `merge_and_push` (`85-96`). Keep an immutable data manifest, accepted-episode-to-seed map, scene/code version, and model provenance with the published dataset.

### P2 — Intel benchmark gives an unfair PyTorch load/memory comparison

`scripts/benchmark_intel.py:173` constructs the expensive Torch policy before `time_model` starts its clock/RSS baseline at `80-84`. Its measured factory merely makes TorchCall (`181-185`). OpenVINO construction occurs inside the clock. Therefore PyTorch load_s and rss_delta_mb exclude policy loading, unlike OpenVINO. Construct the policy inside the measured factory. Also report PyTorch-vs-OpenVINO action differences: the Torch output is discarded at line 185 although the docstring promises accuracy comparison for each variant.

### P2 — Benchmark accuracy comparison depends on caller ordering

`scripts/benchmark_intel.py:147-162` obtains a reference only when cpu_fp32 is encountered. A reordered list gives no accuracy result for earlier rows; a list without CPU has no reference at all. Run the reference independently or reject unsupported configurations. `make_int8` treats the XML file alone as a complete export (`67-68`), so an interrupted copy can permanently skip missing tokenizer/manifest/weights. Validate the full export before reuse. These are reproducibility defects; no claim of an existing bad benchmark is made because no trained benchmark output was found.

### P2 — Independent learned-adapter test is broken and outside normal test discovery

Ran `scripts/review_policy_contract.py`: 3 tests, 1 ERROR. At line 38 its temporary model lacks the newly mandatory contract. After that is fixed its reset fixture at line 39 also lacks sim.config, now required by adapter line 60. The ordinary seven `tests/test_contract.py` tests all pass; they do not exercise this adapter. Move/update meaningful adapter coverage in the normal suite and test action queue, resets, wrong/missing image slots, empty chunks and invalid horizon values.

### P2 — Local shard worker reports completion after failure and skips incomplete shards

`scripts/local_data_worker.sh:4` uses only set -u; a failing generator does not stop the loop and line 13 always writes done. Line 9 skips every existing directory, including a shard interrupted before finalization. This can hide missing data. Check exit codes and a verified finalized marker; emit a failed status otherwise.

### P3 — Invalid model horizon can crash or silently use the wrong queue length

`src/rescuehandsai/policies/smolvla_exported.py:28,39,78-79` accepts zero/negative n_action_steps. Zero yields an empty queue and IndexError; negative slicing quietly executes almost the entire chunk. Empty output with shape (0,12) also passes the existing shape check. Validate a positive integer and nonempty chunk before queueing.

### P3 — NaN control rates pass the contract

`src/rescuehandsai/contract.py:60` uses abs(delta) > tolerance; comparisons to NaN are false. A contract with control_hz NaN passed against 20 Hz in a direct test. Require finite positive rates at creation and validation.

### P3 — Recorder rate can drift from simulation rate

`recorder.py:12,29` hard-codes 20 fps. Both generators construct it without passing the simulator rate (`generate_dataset.py:59-60`, `generate_recovery_dataset.py:53-54`). Current data metadata is 20 fps, so no current mismatch is established; changing control_dt would silently create incorrect timestamps and chunk timing. Derive rate from the simulation and assert integral/valid fps.

### P3 — User-facing pipeline arguments do not match actual behavior

`--episodes` says demonstrations to keep (`kaggle_pipeline.py:223`) but generation chooses a guessed fixed attempt count (`115-119`) with no refill or truncation. Zero episodes, zero shards, empty or reversed seed ranges, nonpositive runs/frames/steps, and negative perturb scales lack consistent validation. Export usage shows a Hub repo example (`export_openvino.py:4`) but line 29 supports local checkpoint directories only. Tighten argument validation and update examples.

### P3 — Profiling can understate cost and omits video finalization

`scripts/profile_generation.py:66-73` may stop early but still divides by requested args.steps. The record profiling path discards frames, never measures encoding/finalization, and deletes the temporary directory without recorder.finalize (`74-78`). Correct the actual step count and separately measure complete episode save/finalize cost before extrapolating data throughput.

## Evidence and limits

- `tests/test_contract.py`: 7 passed in the local simulation environment.
- `scripts/review_policy_contract.py`: 2 passed, 1 errored, as described above.
- NaN control-rate acceptance and unequal-horizon broadcasting failure reproduced without loading any model.
- All nine completed raw learned result JSON files were read programmatically; 1 SUCCEEDED, 8 FAILED. No current full-model inference was started because the mandatory local contracts are missing.
- `artifacts/spike_openvino/report.json` proves base SmolVLA 6-D export/inference on Intel i5-6300U/HD Graphics 520. It reports CPU mean 30.791 s and GPU mean 4.678 s. It is NOT the trained 12-D model benchmark.
- Existing trained full supervised result reports mean inference 4.568 s per chunk and fails. Simulation pauses during inference, as its summary explicitly states. A 25-action chunk at 20 Hz covers only 1.25 simulated seconds; this is not real-time 20 Hz closed-loop execution.
- `data/merged_local/meta/info.json`: 101 episodes, 57,173 frames, 20 fps. `data/recovery_trial` has zero saved episodes; its two attempts both fail. Local perturbed trials have 1, 2 and 0 saved episodes respectively. This does not prove what a newer remote dataset contains.
- `models/smolvla_rescuehands/train_config.json` declares 12,000 steps, batch 16, no evaluation, image transforms disabled, and no pinned dataset revision. Configuration alone does not establish that all 12,000 steps finished.
- Current code separates privileged teacher from learned observations cleanly, explicitly maps left/right joint targets, validates uint8 image inputs, and rejects nonfinite predicted actions. These are good foundations.

## Recommended repair order

1. Recover trustworthy artifact provenance and make existing checkpoint/export compatible through a checked migration.
2. Fix learning-rate selection, all-stage verification, import setup, and publish-after-verification ordering.
3. Repair adapter tests and add small synthetic tests for horizon aggregation, contract integrity and pipeline control flow.
4. Establish native/Torch/OpenVINO agreement on multiple real phases and objects; do not infer it from shape alone.
5. Debug the closed-loop failure states, then regenerate current-teacher data and validate on development seeds.
6. Run the final ten-seed learned/raw and learned/supervised comparisons, then benchmark the exact same verified model.
