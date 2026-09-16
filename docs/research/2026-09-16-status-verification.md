# Verification of STATUS.md and audit fixes

Reviewed snapshot: `b871640` (September 16, 2026). Source status file: `docs/STATUS.md`, dated 20:12 PKT. This is a new verification report; the earlier audit describes an older version and must not be read as the current verdict. No product code changed during this review.

## Verdict

**Yes, the direction is right, and many fixes are real. No, almost everything cannot yet be called verified or finished.** The implementation is better tested and its claims are more honest. Keep the dinner-table task, SmolVLA, real simulation contacts, Intel inference and separate scripted baseline.

The remaining priority is evidence from the learned robot. More training data and a falling training loss are useful, but they do not establish completed tasks or recovery. Safety limits that the owner has placed on hold remain limits; documenting them does not fix them.

## Fresh checks completed

| Check | Independent result |
|---|---|
| `unittest discover -s tests -v` | **99 passed**, 110.399 seconds |
| `scripts/review_policy_contract.py` | **3 passed**, 0.320 seconds; the previously broken adapter fixture is repaired |
| Local v1 checkpoint contract | All listed file hashes pass; robot joint order and 20 Hz rate pass |
| Local v1 OpenVINO contract | All five listed export/tokenizer/manifest file hashes pass |
| Saved v2 checkpoint `002000/train_config.json` | Optimizer LR **5e-5**, warm-up **300**, decay **6000**, requested steps **6000** |
| Final scripted clean results | 10 episode JSONs, **8 successes**, summary agrees, `complete=true` |
| Final scripted fault/no-supervisor results | 10 episode JSONs, **2 successes**, summary agrees, 10 faults injected |
| Final scripted fault/supervisor results | 10 episode JSONs, **7 successes**, summary agrees, 10 faults injected |
| Presentation isolation test | Passed: drawing the twin does not change robot state, simulation time or policy camera pixels |

The three saved result directories are `results/audit_final_*`. They were recounted, not all rerun in this verification. The full fresh test suite includes actual physics and handoff tests, but is not a new full ten-seed experiment. The final saved baseline revision and current runtime changes were compared; subsequent scene edits add an optional presentation mode and leave the default scene path intact.

The backup configuration proves the effective requested training settings reached that checkpoint. It does not prove that 6,000 steps have completed or that the model succeeds.

## Audit fixes confirmed in source and tests

- Learning-rate overrides now use the policy fields consumed by LeRobot's optimizer preset.
- The default `all` path now invokes checkpoint verification. Verification receives the project's src path and uploads the contract afterward.
- Deployment checks hashes; nonfinite/nonpositive contract rates and invalid action horizons are rejected. Empty action chunks fail clearly.
- Initial policy planning and replanning errors now become logged failed episodes. Tests cover both.
- Evaluation writes a run manifest before episodes and rewrites a partial summary after each completed episode.
- Hand-off contact loss is bounded. A long gap fails; short contact flicker is accepted.
- The spare utensil must have no gripper contact and low speed at the end.
- Teacher staging no longer consumes the final pickup attempt without a successful lift. New targeted tests cover that path.
- Recovery now stops if an item leaves the table during retreat. Its targeted test passes.
- The direct-execution test ordering, horizon aggregation and local worker completion handling are corrected.
- Generator recorder FPS comes from simulation timing. Profiling now discloses that final video encoding is excluded.
- The benchmark includes Torch construction in the load/memory window, keeps Torch actions, and checks more files before reusing INT8 output.
- README now reports 1 of 9 completed v1 learned episodes, the 2/10 teacher no-supervisor fault result, and the friction/collision limits.

## Important remaining findings

### P1 — The deployment plan misses the matching learned fault comparison

`scripts/deploy_learned.sh:50` runs `on glitch`, `on none`, and `off none`. It never runs **`off glitch`**. The submission draft mirrors these three rows.

To show that the supervisor helps after a drop, compare the same learned model on the same seeds with the same fault, with the supervisor both on and off. Comparing a fault run to a clean run changes two things at once. The teacher's 2/10 versus 7/10 comparison cannot establish the learned model's recovery benefit.

Add the missing experiment to the plan. Ideally keep all four combinations. Before broad evaluation, use development seeds to confirm the exported model can attempt the task and the output mapping is correct. Do not present an incomplete run as a full ten-seed score.

### P1 — Two major safety/robustness gaps remain deliberately on hold

`sim.py` and `auditor.py` still detect cross-arm contact only. They do not detect/count prohibited table, plate or same-arm impacts. Jaw priority still fixes grasp friction at 1.0, so the object-friction variation does not test slippery grasps.

STATUS and README acknowledge this correctly. Respect the owner's hold; do not silently alter physics under an active trained model. These findings are **deferred, not resolved**. A later physics change needs matching training/evaluation evidence. Also retain the documented height-only possession limitation: `expert.py:226`, `279`, `307` still use lift height rather than verified sustained possession.

### P2 — Error recording is improved but not complete

`runner.py:217–228` catches ValueError and RuntimeError from action generation, but not other ordinary exceptions. A fresh probe with a policy raising `KeyError('camera input missing')` escaped the runner instead of returning an episode log. `scripts/evaluate.py:177–188` has cleanup in `finally`, but no exception-to-result conversion around the whole episode.

Simulation reset, perturbation, some callbacks and initial model construction also remain outside the relevant protection. Earlier completed episodes now keep a partial summary, which is an improvement; the failing episode can still disappear. Add a final exception recording boundary that preserves the actual error and cleanup. Do not catch user interrupts as ordinary robot failures.

### P2 — Provenance is better but still not fully reproducible

- The backup v2 training config still has `dataset.revision=null`.
- Both checked v1 contracts say `dataset_revision='v3.0'`, not an immutable dataset content commit.
- `training/verify_checkpoint.py:103` still copies `dataset.revision` into the contract.
- `upload_provenance` at `training/kaggle_pipeline.py:99` gathers every matching attempt log in the data directory and records the revision at upload time. It does not record the exact aggregation root order/episode offsets, nor prove that the upload-time code is the code that generated older shards.
- A hash of a dirty Git diff helps distinguish runs but cannot reconstruct that diff. The manifest omits the scene config and actual upstream asset content hash. For a clean frozen checkout these can be recovered more easily; for edited runs they cannot be assumed.

Pin dataset/model content revisions and preserve generation manifests, shard order, accepted-episode mapping, source/config hashes and the actual patch if running dirty code. Treat 'provenance added' as progress, not a complete guarantee.

### P2 — Saved-state replay needs the original scene and timing

`StateRecorder.close` (`scripts/evaluate.py:120`) saves qpos, runner state, seed, instruction and policy name. `scripts/render_showcase.py:45–60` rebuilds the scene from the current checkout and assumes 20 Hz/0.05 seconds. It does not verify the source scene/config/asset identity against the saved run.

This can produce a different-looking scene around the same qpos if the Kaggle checkout/config/assets differ. Freeze the exact matching simulation scene when transferring states, and store/check a scene identity plus actual timestamps/control interval. Also fix `fps=20 // every`: at `--every 3` playback is 6 fps instead of 6.667 fps. Current default 2 is unaffected. The first recorded state is already after a step, but is labelled t=0.

### P2 — The benchmark is only partly verified

There is now a real **trained v1 smoke benchmark** at `results/audit_c1_benchmark_smoke/benchmark.json`. This is progress beyond the earlier base-model spike. It contains Torch CPU, OpenVINO CPU and GPU FP16 rows, with action differences. It uses only one timed sample for each OpenVINO row and two for Torch. It is not a final stable latency/p95 study, and it is not v2 evidence.

`ordered_variants` (`scripts/benchmark_intel.py:94`) sorts CPU first only when the caller includes it. A GPU-only request still runs without an accuracy reference. The benchmark directly loads inference models rather than using the contract-checking adapter. Add a mandatory reference or clearly reject accuracy claims when it is missing, and verify the artifact identity before timing. No new full native-LeRobot versus Studio versus OpenVINO agreement report was found.

### P2/P3 — Keep standalone media and wording honest

The cover and sampled gallery frames render clearly. README labels the gallery as the scripted teacher. The standalone cover has no teacher label while displaying SmolVLA and a recovery claim. Add that label or use verified learned footage once available, so a detached image does not imply learned recovery has already been proved.

The spare utensil check proves it is released, still and close to its start **at the end**, not that it was never touched earlier. Use that wording unless contact history is tracked. The submission text still has deliberate blanks, and its slide count says six while STATUS describes ten; resolve before publishing.

## What remains unverified externally

Public Hub checks failed here: the web lookup returned an error and direct access failed DNS resolution. Therefore this review does not independently confirm the remote 569-episode dataset, live Kaggle progress, final v2 upload, or export completion. STATUS correctly warns that its running-job snapshot may be stale. The local backup config supports the learning-rate fix, but is not a substitute for current job output.

No v2 learned evaluation directories were present in the inspected local results. No new full learned inference run, long benchmark or remote export was launched during this review. The complete slide/PDF deck and every media frame were not re-reviewed in this pass.

## Recommended next order — based on quality

1. Confirm the final training exit, checkpoint verification, complete contract upload and exact content revisions.
2. Verify export hashes and compare real model outputs across backends on useful task phases. Run a development-seed learned episode before the larger evaluation.
3. Add the missing learned **supervisor off + fault on** row; preserve all four combinations if making both clean and recovery claims.
4. Finish the exception-recording and replay-provenance gaps before relying on unattended runs and exported videos.
5. Complete the ten-seed measurements and the benchmark on the same frozen model. Keep failures and partial denominators visible.
6. Fill submission numbers only from those files. Keep held physics limitations explicit. Do not let a schedule replace these gates.

Claude Code has made meaningful improvements and corrected several earlier overclaims. The remaining review decision should depend on successful end-to-end evidence, not the number of completed checklist items.
