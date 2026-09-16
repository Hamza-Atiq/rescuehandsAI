# Evidence, plans, memory and presentation audit

Reviewed September 16, 2026. Working snapshot: `3471bed`. This is a read-only source audit; only this report and inspection artifacts were added. Other reviewers cover robot/control correctness and training/export code. Old review findings are history, not automatically current bugs.

## What the project is trying to build

Two simulated SO-101 arms set a small dinner place. The instruction selects fork or spoon. The right arm picks it up, passes it to the left arm, and the left arm places it beside a static plate. The right arm places the cup. SmolVLA supplies learned joint commands. A separate monitor checks the physical outcome and allows bounded retries. The final policy runs through OpenVINO on the owner's Intel laptop. A scripted teacher makes training examples and supplies a separately labelled baseline.

That direction matches the corrected challenge and the owner's model choice. The main weakness is still reliable learned execution and proof that learned recovery helps. A bigger slide deck or more teacher demonstrations alone cannot prove that claim.

## Facts from saved results

I parsed all 384 JSON files under `results/`. Each available summary agrees with its saved episode count and success count. This checks file consistency, not whether every success rule is correct.

| Saved evidence | Result | Meaning |
| --- | --- | --- |
| `scripted_v2_sup-on_fault-none` | 8/10 | Teacher baseline, old recorded revision `484d2bd` |
| `scripted_v2_sup-off_fault-glitch` | 0/10 | Teacher with fault, no recovery |
| `scripted_v2_sup-on_fault-glitch` | 7/10 | Teacher with fault and recovery |
| `independent_trained_smoke_20260916` | 0/1, intentional short timeout | Integration check only |
| `independent_trained_full_20260916` | 0/1, recovery exhausted | Learned seed 0, supervisor on, no injected fault |
| `smolvla_ov-gpu_sup-off_fault-none` | 1 success in 9 completed records | Learned seeds 0–8; seed 4 succeeds; five out-of-bounds failures, three timeouts |

The learned folder has ten MP4 files but only nine episode JSON files and no summary. Seed 9 has 110 video frames but no completed outcome. It is an interrupted run, not ten evaluated seeds. Do not report 1/10 or treat the partial video as a result.

The saved seed-4 success is a real improvement over the earlier review's single failed seed. Five sampled seed-4 frames show pickup/transfer-area motion, left-side spoon placement, and the cup near its target at the end. The JSON reports all success checks true. These samples do not independently prove continuous airborne transfer or every stability interval; the physics reviewer must assess the success rule.

## Findings

### P1 — The headline learned reliability and recovery claims remain unproved

Evidence: `README.md:3–8`, `README.md:76–87`, `results/smolvla_ov-gpu_sup-off_fault-none/episode_0.json` through `episode_8.json`, and `results/independent_trained_full_20260916/episode_0.json`.

Only one of nine completed unsupervised learned runs succeeds. The independent supervised clean run fails after two retries. There is no completed paired learned fault/recovery comparison. The opening description can read like a working learned table-setting/recovery product, although the results section says work in progress. Put the actual learned result and limitation beside the opening claim. Keep teacher and learned scores separate.

Required proof: freeze model, evaluator and scene; run complete clean learned evaluation and paired fault runs, save every outcome and video, count faults that actually fired, and publish the denominator. Completion must survive the corrected success checks.

### P1 — Claude ruled out an alternative cause without the needed experiment

Evidence: latest assistant summary in local Claude session `7d05a736-a6a8-4299-b1e6-efad6a3e0392.jsonl`; project memory `build-learnings.md` and `session-status.md`; `results/horizon_diagnosis.json`.

Claude says that replanning more often would have been a waste and that unseen states are the cause. The evidence is low average action error on ten frames from the demonstration dataset. The saved report itself correctly says this is an open-loop test, not task success. This can show the model copies familiar examples reasonably well. It cannot rule out a shorter action chunk helping when an actual episode drifts, nor prove that data coverage is the only fault. Averaging across joints can also hide brief large errors at grasp/release.

Treat missing state coverage as a strong hypothesis. Compare chunk lengths on the same checkpoint and separate development seeds, inspect joint/contact traces at the first divergence, and finish numerical backend agreement on those observations. More recovery data is sensible, but its benefit must be measured after training.

### P1 — Run provenance is too weak to reproduce the experiments

Evidence: `scripts/evaluate.py:134–154`, summaries in `check_v3_*`, `check_revert_*`, `base_restored_*`, `group1_*`, and `lift_*`.

Several meaningfully different experiments all name revision `a699b995`. Source edits made without committing are not captured by a Git HEAD string. `git_revision()` is called after all episodes finish, so a commit during a long run can also attach the ending revision to earlier work. Individual episode JSON lacks a code revision, and an interrupted run loses the summary where provenance would otherwise live. Model metadata names a mutable local directory, not a hash. Full scene/randomization settings, dependency versions and dirty-tree changes are not saved with every run.

Write an immutable run manifest before the first episode. Include source revision and dirty diff/hash, model/export/tokenizer hashes, full config, package versions, seeds and intended matrix. Save outcome/status incrementally, including interruption. Finish evaluation in a fixed checkout while other sessions work elsewhere.

### P1 — Final Intel optimization evidence is absent

Evidence: `README.md:91–102`; `artifacts/spike_openvino/report.json`; `compare_backends.log` ending in the old dictionary-input exception; no final benchmark/agreement result under `results/`.

The six-joint pretrained spike proves export feasibility, not preservation or acceleration of the trained twelve-joint robot. Its CPU timing was measured with competing work. Actual trained iGPU inference and learned motion exist, but there is no completed fair native/Intel/export comparison or final optimization table here. Old tracebacks do not prove that newer source fixes still fail; they prove that the saved comparison is incomplete.

Run the fixed comparison and trained-model benchmark on an otherwise idle machine. Record actual stored/execution precision, load/first/warm timing, input agreement and task performance after optimization. Keep simulated video time separate from wall time.

### P2 — Current state is scattered across stale documents and private memory

Evidence: `docs/superpowers/plans/2026-09-15-dinner-table.md`, approved design, README, old reviews, Claude memory.

- Every implementation-plan checkbox remains unchecked, even for completed work. It still specifies LeRobot 0.6.1/OpenVINO 2026.3.1 and two environments, while the working README uses LeRobot 0.5.1/OpenVINO 2026.1 and a separate Intel environment.
- Approved design lists cup first; measured implementation does utensil first because a long utensil can knock the cup. Claude memory explains this good change, but the design was not updated.
- README learned rows still say pending and the release review describes only the earlier failed seed. Nine later completed runs and one success are now available.
- The release review says `.env`/key ignore rules are missing; current `.gitignore:25–33` already fixes this. Claude memory says results are not in Git, but all 33 `scripted_v2` JSON files are tracked. Repeating these as current bugs wastes work.
- Memory gives conflicting historical `use_random_input_noise` statements and repeatedly mixes old status with current status. The newest session-status file is clearer, but still carries some fixed issues as pending.
- AGENTS correctly puts challenge corrections first, but its large historical body still says package rescue/optional Intel optimization in places. Clear precedence helps; a shorter current requirements document would lower mistake risk.
- The presentation plan is built around a 12-hour deadline budget. The owner's latest request explicitly prioritizes quality and says time is not the limit. The next plan should have measured exit conditions, not assumed success by a clock time.

Keep old reviews dated as history. Add a single current status table: requirement, evidence, current issue, next validation. Put durable decisions in the repository, not only private Claude memory.

### P2 — Reproduction instructions are not yet a clean-machine recipe

Evidence: `README.md:122–151`, `pyproject.toml`, `uv.lock`.

The core dependency lock is present and names MuJoCo 3.13.0. The Intel setup remains an ad hoc pip command with some unpinned packages and no full environment lock. README mixes Bash environment assignment and line continuations with Windows `Scripts/python.exe` paths. Those commands do not run unchanged in PowerShell, while native Linux environments use `bin/python`. Public model/dataset links and fixed revisions are absent from the quick start; placeholders describe retraining but do not give judges a direct verified-model reproduction path. The known Anaconda/default-environment DLL issue is not explained clearly in the main guide.

Provide separate tested Windows PowerShell and Kaggle/Linux commands, exact environment manifests, pinned model/dataset downloads, the asset revision, and one short verified demonstration command. Test from a new checkout/environment. The project description is still the scaffold text `Add your description here` (`pyproject.toml:4`).

### P2 — Required submission assets are plans rather than finished files

Evidence: `docs/presentation/submission-story-and-plan.md:133–141`, local inventory.

The repository contains a storyboard/slide outline, not a finished deck or cover. No combined labelled ten-seed demonstration or final Intel report was found. Thirteen per-episode videos exist locally, including the interrupted seed 9. The architecture Mermaid in README is useful; it does not fill missing final evidence. Local availability does not prove public links work without login. The MIT project license exists, and README credits the separately downloaded Apache-2.0 robot assets.

Finish the model/evaluation evidence first, then build the video/deck/cover from real frames. Verify repo/data/model/video/slides links from a logged-out view. Do not claim the public package was verified by this local audit.

### P2 — Visual evidence needs labels and better presentation framing

The existing policy overhead sample visibly clips the cup at the right border. Keep that camera consistent with training; use the wider front camera for presentation. Result frames show only a thin green/orange strip. That does not tell the viewer the policy, seed, instruction, event, final status, or simulated time. A viewer can easily confuse teacher recovery with learned recovery.

Add a video-only overlay identifying policy, seed/run ID, instruction, state and simulated playback time. Show all final outcomes, including failures. Do not use the old motor-check GIF to imply learned table setting.

### P2 — Review helpers are useful probes, but not general release gates

All six `scripts/review_*.py` files were read. Their scope is narrower than a whole-product audit:

- `review_data.py` hard-codes the first merged parquet data/episode files. It would miss later chunks after a larger merge, although it iterates all videos.
- `review_release_evidence.py` checks only `scripted_v2_*`; its current-secret scan skips files over 5 MB and all history, as its output honestly states. It does not fail exit status when a false handoff or count mismatch is reported.
- `review_independent.py` is a historical reproducer that calls private runner internals. Stronger current validation may make it stop early. Keep it dated or update it into isolated regression probes.
- `review_openvino.py` is deliberately a six-joint spike check. It cannot approve the final twelve-joint model.
- `review_policy_contract.py` uses a spy model. It proves plumbing, not learning or numerical export agreement.
- `review_kaggle_preflight.py` returns success based on shapes alone; it reports hardware/headers but does not prove trained quality or completed training.

Use these as scoped evidence. For release, combine all data chunks, explicit pass/fail criteria, final model contracts and real rollouts.

## Assessment of Claude's work

The logs show real engineering progress, not an empty project: a functioning teacher, learned export/inference, fault tests, measured regression checks and a diagnosed rendering bottleneck. Claude openly acknowledged two regressions and used before/after teacher measurements. The latest memory records v1's poor learned result rather than hiding it. Preserving v1 and writing v2 to a new model name is good.

The main process weaknesses are overconfident causal claims, stale memory, development/evaluation changes in the same checkout, and advice not verified in the target notebook before handing it to the owner. Recent assistant text first gave a Kaggle `!nohup ... &` command, then corrected it after the notebook rejected background execution. It also supplied broad cleanup globs for datasets/logs before the new launch, which conflicts with the owner's wish to preserve prior work; use new output directories instead. This audit did not execute those commands or delete anything.

The latest memory says v2 data generation was running on Kaggle, with a first check of 8 saved out of 10 attempts and a Tesla T4 renderer. This is a historical reported snapshot, not a fresh remote job check. Neither the current Kaggle job status nor v2 quality is proved locally. A better teacher yield is not a learned-policy success result.

## Coverage and limits

Read README, CLAUDE, gitignore, pyproject, all eight pre-existing docs, and all six review helpers. Checked uv.lock package metadata/structure, result JSON consistency, root log endings and relevant error/training/evaluation records. Inspected all five Claude project-memory files and recent assistant text from the two newest sessions. Did not dump credential values or arbitrary tool arguments. Did not read every line of every historical conversation or claim every dependency source/binary was inspected.

Visually inspected the learned-failure and scripted-recovery contact sheets, the overhead dataset frame, five newly extracted learned seed-4 frames, plus late frames from learned seeds 0 and 9. New images: `artifacts/audit-evidence/seed4_*.jpg` and `learned-seed4-contact-sheet.jpg`. The whole result-video/image integrity scan is saved separately at `artifacts/audit-evidence/integrity.json`; automated decoding is not naked-eye review. The other reviewers inspect additional images, source and tests.

I did not visually examine every one of the thousands of training images or every video frame. I did not rerun training, all evaluations, network downloads, a clean install, or the final Intel benchmark in this evidence audit. Saved logs and memory are distinguished from fresh independent tests. No report can promise that no possible bug remains.
