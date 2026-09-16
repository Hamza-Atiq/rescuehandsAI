# Plan: act on the Sept 16 Codex quality audit

Sources: `docs/research/2026-09-16-quality-audit.md` and its three companion reports,
at snapshot `3471bed`. Every finding below was checked against the code before
being planned. "Reproduced" means I ran it, and "Read" means I confirmed it in the
source.

Rules for this work (from the owner):
- Only add or correct code, and keep working behaviour.
- Do not shrink the plan.
- After any teacher or success-rule change, re-run scripted seeds 0-9 and compare
  against 8/10 clean and 7/10 with fault.
- One fix at a time, with its test, and the full suite before each commit.

The v2 dataset job running on Kaggle already has its code loaded, so edits here
do not change it. Its provenance note is item C4.

## Verification of findings

| # | Finding | Verdict | How checked |
|---|---|---|---|
| 1 | v1 checkpoint and export lack `task_contract.json`, so current adapter refuses them | Confirmed | Read: files absent on disk |
| 2 | `--optimizer.lr` is overwritten by the SmolVLA preset | Confirmed | Read: lerobot 0.5.1 `configs/train.py:137-139`, preset reads `policy.optimizer_lr` |
| 3 | `all` stage skips verify; contract never uploaded; verify lacks `PYTHONPATH` | Confirmed | Read: `kaggle_pipeline.py` main and `verify()` |
| 4 | Only cross-arm contacts stop the sim; table/self contacts undetected | Confirmed | Read: `sim.step`, `auditor.compute_facts` |
| 5 | Jaw geoms have `priority="1"` and friction 1, so object friction never reaches grasps | Confirmed | Read: `so101.xml:24-27`, `scene.py:91,154` |
| 6 | `policy.reset` and `after_recovery` exceptions escape; evaluate loses episode and summary | Reproduced | `audit_runner_probes.py` |
| 7 | Recovery moves are not audited (facts=None during `_safe_pose`) | Confirmed | Read: `runner._safe_pose` |
| 8 | Benchmark: Torch load outside timer; Torch actions discarded; reference order-dependent; INT8 reuse checks xml only | Confirmed | Read: `benchmark_intel.py` |
| 9 | Run provenance: revision read at end, no dirty flag, no per-run manifest, no incremental summary | Confirmed | Read: `evaluate.py` |
| 10 | Contract hashes never checked; NaN control rate passes; verify copies facts without checking | Confirmed | Read: `contract.py`, `verify_checkpoint.py` |
| 11a | Spare utensil check ignores contact and speed | Reproduced | probe: held + moving spare passes |
| 11b | Hand-off allows unlimited gap with nobody holding | Reproduced | probe: 100-update gap passes |
| 11c | Staging on the final pickup attempt exits without a grasp and reuses stale `q_lift` | Confirmed | Read: `expert._pick_utensil` |
| 11d | Possession judged by height only | Confirmed | Read; deliberate after the contact-flicker regression |
| 11e | Start validation ignores `out_of_bounds` and `on_item` | Confirmed | Read: `randomize.start_problems` |
| 12 | `test_expert.py` main guard before a class; `diagnose_horizon` unequal-length crash; `local_data_worker.sh` marks done after failure; `n_action_steps<=0` accepted; recorder fps hard-coded; profiler divides by requested steps | Confirmed | Read |
| 13 | Docs stale (README learned rows, assets README, design order, pyproject description) | Confirmed | Read |
| H | "Unseen states, not horizon, is the cause" was over-claimed from an open-loop test | Accepted | The saved report says open-loop itself; a closed-loop chunk comparison was never run |

## Decisions that need the owner (not changed without approval)

- **5, jaw friction.** Making object friction reach the jaws changes grasp physics.
  That would invalidate the v2 dataset being generated now and the teacher tuning.
  Default: keep the physics, and document honestly that friction randomisation
  affects table and object contacts but not jaw grasps. Add a measurement script
  that prints effective contact friction, so the claim is evidence-based.
- **4, table/self contacts.** A hard stop could end normal teacher grasps, since
  flat utensils are picked next to the table surface. Default: detect and record
  prohibited contacts, with a penetration depth, as events and metrics first. Then
  measure seeds 0-9 and decide on a stop threshold from the data. Rename the
  existing metric to cross-arm collisions.
- **11d, possession by height.** A contact-history rule caused a measured 2/10
  regression before. It stays as it is until a short-history rule can be measured
  on seeds 0-9.

## Work order

### A. Blocks the next steps (fine-tune, laptop evaluation) — do first
- A1 (#2) Pipeline: pass `--policy.optimizer_lr`, `--policy.scheduler_warmup_steps`
  and `--policy.scheduler_decay_steps` instead of `--optimizer.lr`. Verify by
  reading `train_config.json` after a 20-step smoke run. Unit-test the argument
  list.
- A2 (#3) `verify()` gets `PYTHONPATH=src`. The `all` stage also verifies. When
  pushing, upload `task_contract.json` to the model repo after verification.
- A3 (#1) Migrate v1 artifacts with a checked script: run
  `verify_checkpoint --write-contract` on the local checkpoint against the local
  v1 dataset copy. Write the export contract only if `export_report.json` names
  that checkpoint and the files hash cleanly.
- A4 (#10) Contract: reject non-finite or non-positive `control_hz`. `load_contract`
  optionally verifies file hashes, and the adapter and export call it with
  verification on. Adapter rejects `n_action_steps < 1` and empty chunks.
- A5 (#6, #9) Runner: `policy.reset` and `after_recovery` errors become logged
  `POLICY_ERROR` failures. Evaluate writes `manifest.json` (git revision, dirty
  flag, args, contract, versions) before the first episode, and rewrites
  `summary.json` after every episode.

### B. Truthful success and safety semantics — measured
- B1 (11b) Hand-off: bound the no-holder airborne gap after "shared". Pick the
  bound from teacher traces on seeds 0-9 (largest observed gap × 2, at least
  5 updates). Test that the synthetic 100-update gap fails.
- B2 (11a) Spare utensil: at the end it must also be untouched and settled.
  Probe fails, and seeds 0-9 must stay at 8/10.
- B3 (7) Recovery motion: compute facts every step and record terminal failures
  (out of bounds, cross-arm). New drops during retreat are reported after
  recovery, not mid-motion, so recovery keeps its bounded shape.
- B4 (4) Record prohibited arm/table and arm/plate contacts deeper than a
  threshold as `WORLD_CONTACT` events with depth. Use a per-episode metric, not a
  stop yet. Rename the summary field to `cross_arm_collision_episodes`, and keep
  the old key for comparison.
- B5 (11c) Teacher pickup: count grasp attempts separately from staging, and
  always leave through an explicit successful lift. Add a regression test with
  staging on the last attempt. Re-run seeds 0-9.
- B6 (11e) Start validation also rejects `out_of_bounds` and items resting on
  items, with rejection reasons already logged.

### C. Evidence tools and reproducibility
- C1 (#8) Benchmark: build the Torch policy inside the timed factory, keep Torch
  actions and compare them, compute the reference independently of variant order,
  and treat an INT8 directory as complete only with every expected file.
- C2 (12) `diagnose_horizon`: per-index sums/counts, and null labels past the
  available steps. `profile_generation`: divide by actual steps.
  `local_data_worker.sh`: stop on generator failure, no done marker.
  `test_expert.py`: main guard at the end. Recorder fps comes from `control_dt`.
  `review_policy_contract.py`: give the spy model a contract and a sim config.
- C3 (H) Closed-loop chunk-length check (`n_action_steps` 10 vs 25) on
  development seeds 20-24, not the final 0-9. It runs after v2 exists and must be
  written up before anyone claims a cause.
- C4 Dataset provenance: after the Kaggle job finishes, upload
  `data/*_attempts.jsonl` (seed → saved) to the v2 dataset repo under
  `provenance/`, and pin model and dataset revisions in contracts.

### D. Documents
- README learned rows (1 of 9 completed, seed 9 interrupted), collision wording,
  assets README, design order note, pyproject description, current-status table.
  Correct the memory's over-claim.

## Exit checks
- Full unit suite passes, and the audit probes print `false` for all four items.
- Scripted seeds 0-9: clean and fault results no worse than before, or any change
  is explained by a stricter success rule, with both numbers reported.
- The v1 contract migration loads through the adapter.
- A 20-step Kaggle smoke run shows the requested learning rate in `train_config.json`.
