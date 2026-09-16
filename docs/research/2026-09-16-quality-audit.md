# RescueHands AI quality audit — September 16, 2026

Review snapshot: `3471bed09e210e2808877e8bb9ba145c8aecc712`. This report is being delivered at the owner's request before the session limit. It is a review, not a claim that every defect has been found or fixed. Product code was not changed.

## Decision

**The idea is sound. The current system is not ready to be called reliable or finished.** Keep the dinner-table task, SmolVLA, MuJoCo, Intel inference and bounded recovery. Change the work process: use measured quality gates instead of a deadline-driven checklist. A passing unit suite or a good scripted video is not enough to approve learned deployment.

The intended product is two SO-101 arms that read an instruction, choose the fork or spoon, pass it between hands, place it beside a plate, and place a cup. The learned model should choose joint movements. The supervisor should notice failures and allow limited retries. The scripted teacher is a separate source of demonstrations and a comparison baseline.

## What the evidence actually says

| Check | Result |
|---|---|
| Fresh standard test suite | **74 tests passed**, 172.394 seconds |
| Separate learned-adapter review test | **1 error out of 3 tests**; missing contract fixture |
| Saved learned raw evaluation | **1 success, 8 failures among 9 completed episode JSON files** |
| Tenth learned episode | Video exists; episode JSON and full-run summary are missing |
| Saved independent learned/supervised full seed 0 | Failed with `RECOVERY_EXHAUSTED` |
| Current learned checkpoint/export | Required `task_contract.json` missing from both directories |
| README scripted v2 results | Historical clean 8/10; fault/recovery on 7/10; fault/recovery off 0/10 |
| Final trained-model Intel comparison | No completed benchmark/backend agreement report found |
| Base-model Intel spike | Exists; it is a 6-D pretrained test, not the trained 12-D task benchmark |

The learned seed-4 success must be acknowledged. It also must not be turned into a claim that the complete ten-seed evaluation passed. Saved scene frames support real manipulation; sparse video frames cannot independently certify every contact or the full handoff sequence.

## Most important problems

### 1. Current model files and current code do not fit together — P1

The adapter now requires a task contract, but the existing local checkpoint and OpenVINO export do not contain it. Historical videos cannot prove today's command works. See `policies/smolvla_exported.py:32`, `contract.py:44`, and `scripts/export_openvino.py:29`. Migrate only after verifying actual joint order, cameras, rate, processors and model identity.

### 2. Training may ignore the requested smaller learning rate — P1

`training/kaggle_pipeline.py:168` supplies `optimizer.lr`, while LeRobot's enabled policy preset replaces the optimizer configuration. Verify the effective saved configuration, not just the command text. Detailed installed-library evidence is in the policy audit.

### 3. The main training route skips its verification gate — P1

The default `all` stage omits verification at `training/kaggle_pipeline.py:261`. Model publishing happens before the later local contract is created, with no subsequent contract upload. The fresh Kaggle verification path also lacks an explicit installation/PYTHONPATH for this src-layout project; that import failure is a code-based inference, not a fresh Kaggle reproduction.

### 4. Collision safety is narrower than the product story — P1

`sim.step()` stops contacts between different arms. It does not stop prohibited table, plate or same-arm contacts. A targeted within-limit pose probe advanced despite about **4.38 cm of table penetration**. This proves missing detection, not that every normal run hits the table. Existing collision totals should be called cross-arm collision totals.

### 5. Random object friction does not change grasp friction — P1

The jaw collision shapes have higher contact priority. A sampled cup friction of about 0.717 still produced jaw/cup friction of 1.0. Table friction may vary, but the current experiment does not establish robustness to slippery grasps. Test effective contacts, then regenerate results after fixing the physics.

### 6. Evaluation can crash without recording the failure — P1

`runner.py:170–173` resets/plans before an episode log and error handling exist. `runner.py:163` calls `after_recovery` outside its recovery try block. A planning exception escapes, and `scripts/evaluate.py:116–123` never writes that episode JSON or a final summary. Both paths were reproduced using small policy fixtures in `audit_runner_probes.py`. Preserve failures and partial summaries even when a policy or setup call throws.

### 7. Recovery is not fully watched — P2

`runner.py:119–132` executes every recovery move through `_step`, but does not compute facts or update the failure monitor during those moves. Only the narrower simulation checks remain. New drops, bounds failures and other audit conditions can be missed until recovery finishes. Returning both arms to a pose named 'safe' does not prove the path is safe.

### 8. Benchmark comparisons are not yet trustworthy — P2

`scripts/benchmark_intel.py` loads Torch before its load/memory timer, but loads OpenVINO inside that timer. Accuracy reference calculation depends on variant ordering. No final trained-model result was found. Fix these before publishing acceleration or memory claims.

### 9. Saved runs cannot always be reproduced exactly — P2

Evaluation records a Git revision, but not dirty source/config hashes, complete scene parameters, model/processor hashes or every runtime option. Different experiments share a revision even when behavior differs. Dataset seeds and accepted-episode mappings are not preserved through merging/publishing. Model/dataset names are mutable. Capture exact inputs at run start.

### 10. Model contracts provide incomplete protection — P2

Hashes are written but not checked. Verification copies important facts from a selected dataset or constants without fully proving state order, camera mapping and FPS against the trained artifact. NaN control rate passes the current comparison. A contract must verify facts, not just contain them.

### 11. Some teacher and evaluation rules are too weak — P2

- Teacher possession checks sometimes use height alone. A bounced or supported object can be high without being held.
- Teacher staging on its final pickup attempt can exhaust the loop without a successful pickup. This is a source-path finding, not a reproduced normal episode.
- Start validation omits some bounds and object-stacking checks.
- `evaluation.py:76–78` checks the spare utensil's final location/support, but not its contact, speed or history. A synthetic held-and-moving spare still passed. Therefore 'spare untouched' is stronger than the implemented check.
- `HandoffTracker` allows an unlimited airborne interval without either hand holding the item after shared contact. A synthetic 100-update gap still completed the handoff. Brief contact flicker is reasonable; the gap should be bounded. This probe is not evidence that a normal episode exploited it.

### 12. Some testing and tools give false comfort — P2/P3

- `test_expert.py` invokes unittest before defining the full-task class; direct execution skips that class. Discovery includes it.
- The learned-adapter test lives outside standard discovery and currently errors.
- `diagnose_horizon.py:60` can crash when successive samples have different available future lengths.
- `local_data_worker.sh` can write a done marker after a failed generator and skip incomplete existing folders.
- Zero/negative action horizons and some invalid CLI ranges are not rejected.
- Recorder FPS is hard-coded rather than derived from simulation settings.
- `watch.py:76–88` applies perturbation after the first action, while evaluation applies it before policy reset. The two commands do not test the same starting-state condition.

### 13. Claims and project memory need updating — P2

The README says learned results are pending, while a partial run and one success now exist. Earlier review text saying no learned task completed is stale. The approved design says cup first; code now correctly documents utensil first to avoid a sweep through the cup. Much of AGENTS.md still describes the old package task, despite clear top-level overrides. Asset docs still describe a motor scene. Old review scripts also contain outdated API calls. A reader should not have to guess which account is current.

## Is Claude Code working properly?

**It has built useful work, but its conclusions need stronger checks.** The robot physics, named joints, instruction templates, separation of teacher and learned inputs, physical outcome checks and 74 passing tests are real strengths. It has also responded to earlier findings with useful fixes.

The weak part is the approval process. Current artifacts no longer match current code, an adapter test is broken, the training gate can be skipped, and final learning/Intel evidence is unfinished. Project history also treats low prediction error on ten training frames as enough to reject shorter action chunks and identify unseen states as the sole cause. That conclusion is not established. Test chunk length and recovery-data changes in controlled closed-loop runs. Do not blame or credit an agent based on confidence in its prose.

## Quality-based plan

1. **Freeze and identify the current candidate.** Save exact code/config/model/data versions. Retain failed runs. Update one current-status note.
2. **Fix safety and physics semantics.** Test prohibited world/self contacts, recovery auditing, effective friction and truthful success checks.
3. **Make the model pipeline reproducible.** Fix learning-rate control, verification in all stages, project imports, contract validation and publish order. Run in a clean environment.
4. **Prove the teacher's data quality.** Use separate development seeds, count every attempted/accepted/rejected episode, inspect contact and recovery behavior, and preserve seed mappings.
5. **Diagnose learned failures with controlled tests.** Compare chunk lengths, native and exported outputs, clamping and clean/recovery data. Include both utensils and all task phases. Training loss is not task success.
6. **Prove learned improvement.** Freeze a candidate and run paired supervisor off/on trials with the same fault conditions. Do not substitute teacher recovery for learned recovery.
7. **Finish final evidence.** Complete all ten seeds, retain every result and video, benchmark the exact trained artifact on the actual Intel machine, and check a fresh setup from documented commands.
8. **Package the real result.** Update README and saved memory, then make the slides, cover and demo from verified evidence. Do not claim real-time execution when simulation pauses during inference.

Each step ends when its checks pass, not when a time allowance expires. This review does not set an arbitrary target success rate for the owner; it requires complete, reproducible measurements and honest limits.

## Coverage and limits

The review team read the project-owned Python/shell implementation and tests across runtime, physics, training, inference and review tools; configs, plans, prior reports, README and project memory were included. The main reviewer inspected all four existing visual contact sheets. Additional seed-4 learned video frames were inspected. Results JSON consistency was checked across available saved results.

**This is not a claim that all ~20,000 images or every frame of 85 videos were viewed by eye.** A media integrity scan was started; the evidence audit records its exact completion state. Dependency source was inspected where relevant, not every installed library. Binary weights and meshes were not semantically reviewed. Remote Kaggle/Hugging Face state was not re-run or certified. No current learned run was started because of the missing contracts.

A fresh ten-seed scripted clean evaluation was started in `results/quality_audit_clean_20260916`; at this report checkpoint seeds 0–2 had succeeded. This is provisional, not a completed score. See the folder for later results.

Detailed companion reports:

- [Physics and teacher](audit-physics-2026-09-16.md)
- [Policy, training and Intel](audit-policy-2026-09-16.md)
- [Evidence and history](audit-evidence-2026-09-16.md)
- [Reproducible runner/outcome probes](audit_runner_probes.py)
