# RescueHands AI — current status

**Snapshot: 2026-09-16, 20:12 PKT** (Pakistan time; Kaggle logs use UTC = PKT − 5 h).
Submission deadline: 23:30 PKT on lablab.ai. Organizers may extend it; this is unconfirmed.

This is the single "where are we" file. Anything below that describes a running
job may have finished or failed since, so check it before acting.

---

## 1. Done (all pushed to GitHub, branch `dinner-table`)

### Robot system
- Two SO-101 arms in MuJoCo set a dinner place. The right arm picks the named fork
  or spoon, hands it to the left arm in the air, the left arm places it, and the
  right arm places the cup.
- Physics auditor, supervisor with bounded recovery (max 2), action guard,
  per-episode success from physics.
- Scripted IK teacher, used only for demonstrations and the baseline. It handles
  messy starts and re-grasps a dropped utensil.
- 99 unit and physics tests pass.

### Data and models (Hugging Face, user `ABDHAM`)
| Item | Status |
|---|---|
| Dataset v1 `ABDHAM/rescuehands_table` | 101 episodes, 57,173 frames |
| Dataset v2 `ABDHAM/rescuehands_table_v2` | **569 episodes, 336,729 frames** (101 v1 + 192 clean + 194 messy starts + 82 drop-and-recover); checked on the Hub; content commit `ca9e5592855c322446c3be87c2f1e6ef1ff29a32` (the running Kaggle verify writes only the tag `v3.0`, so cite this commit) |
| Model v1 `ABDHAM/smolvla_rescuehands` | 12k steps; OpenVINO export in `models/openvino/fp32` with a contract |
| v2 checkpoint backup `ABDHAM/smolvla_rescuehands_v2_checkpoints/002000` | Checked on the Hub; also downloaded to `models/_backup_v2_checkpoints/002000` (1.8 GB in 18.4 min) |

### Quality audit (Codex, 2026-09-16) — acted on
Plan and verdicts: `docs/superpowers/plans/2026-09-16-audit-fixes.md`. Fixed and
measured:
- **Learning rate:** `--lr` now really reaches the optimizer.
- **Verify stage:** has the right import path and uploads the model contract.
- **Contracts:** check file hashes and a finite control rate.
- **Runner:** logs planning errors instead of crashing.
- **Evaluations:** write `manifest.json` and a per-episode `summary.json`.
- **Hand-off check:** the no-hands gap is bounded at 5 steps (measured teacher
  maximum: 1).
- **Spare utensil check:** at the end it must have no gripper contact and be still (earlier contact is not tracked).
- **Teacher:** fixed the last-attempt pickup bug.
- **Recovery retreat:** is now watched for items leaving the table.
- **Benchmark:** made fair.
- **Tools:** small bugs fixed.
- **Dataset provenance:** added.
- **README:** matches the evidence.

Every change was re-run on scripted seeds 0–9 and results were identical per episode.

### Measured results (committed in `results/audit_final_*`)
| Policy | Supervisor | Gripper fault | Success |
|---|---|---|---|
| Scripted teacher | on | none | 8/10 |
| Scripted teacher | off | yes | 2/10 |
| Scripted teacher | on | yes | 7/10 |
| SmolVLA v1, iGPU | off | none | 1 of 9 completed (interrupted run; see `results/smolvla_ov-gpu_sup-off_fault-none/PROVENANCE.md`) |

### Presentation and submission kit
- **Showcase renders** (`scripts/render_showcase.py`): sky, floor, shadows and
  camera angles facing the arms. The policy's cameras are byte-for-byte unchanged.
- README gallery: `docs/media/` (4 images of the scripted teacher, labelled).
- **Cover:** `docs/submission/cover.jpg`.
- **Form text:** `docs/submission/lablab-submission.md`, with v2 numbers as ⟨blanks⟩.
- **Slides:** `docs/submission/rescuehands_slides.pptx` and `.pdf`, built by
  `make_slides.py`; v2 numbers are orange blanks in its `RESULTS` dict.
- **Repo tidy:** junk and superseded results removed from the tree (still in history).

### Infrastructure fixes worth remembering
- **Kaggle drew camera images on the CPU:** 229 s per episode.
  `training/kaggle_gpu_render.sh` makes the T4 draw them: 14 s per episode.
- **Kaggle rejects `!cmd &`:** start long jobs with `subprocess.Popen` from a Python cell.
- **Kaggle stops an interactive session that looks idle,** which killed the first
  v2 training run. Keep a cell running (the keep-alive loop) or use Save & Run All.
- **Evaluations save robot states** (`--save-states`); video is drawn afterwards
  with `render_showcase.py --states`. On the laptop that takes 0.84 s per frame,
  so draw learned videos on the Kaggle T4.

---

## 2. Running right now

**Kaggle: v2 fine-tune, 6,000 steps** (interactive notebook, `train_job` Popen, log `train_v2.log`)
```
kaggle_pipeline.py --stage train --steps 6000 --lr 5e-5 --warmup-steps 300 --save-freq 2000
  --init-from ABDHAM/smolvla_rescuehands --dataset-name rescuehands_table_v2
  --model-repo ABDHAM/smolvla_rescuehands_v2 --run-name smolvla_rescuehands_v2
```
- Started about 18:46 PKT and runs at 1.49 s/step.
- Loss: 0.041 at step 400, 0.025 at about step 2,190. It falls smoothly, and the
  learning rate follows its schedule.
- A **keep-alive cell** loops every 10 min. It prints progress and uploads each
  checkpoint to `ABDHAM/smolvla_rescuehands_v2_checkpoints/<step>`.
- When the job finishes, the pipeline itself:
  - pushes the model to `ABDHAM/smolvla_rescuehands_v2`;
  - runs `verify_checkpoint` (expect `CHECKPOINT OK`);
  - uploads `task_contract.json`.

---

## 3. Expected in the next few hours

| PKT | Step | Command / check |
|---|---|---|
| ~20:26 | Checkpoint 004000 backed up | Keep-alive cell prints `backed up checkpoint 004000` |
| ~21:05 | Training ends | Exit code `0`, `CHECKPOINT OK`, `uploaded task_contract.json` |
| ~21:05 | Kaggle: get new code | `!cd /kaggle/working/rescuehandsAI && timeout 60 git pull && git log --oneline -1` (expect `7bb12ee` or later) |
| ~21:05–21:30 | **Kaggle: OpenVINO export** (new, untested there) | `!python training/kaggle_pipeline.py --hf-user ABDHAM --stage export --run-name smolvla_rescuehands_v2 --model-repo ABDHAM/smolvla_rescuehands_v2` → expect `export contract hashes OK` and `uploaded openvino_fp32/` |
| ~21:30–21:40 | Laptop: fetch the export (~0.8 GB, ~8 min) | `bash scripts/deploy_learned.sh ABDHAM/smolvla_rescuehands_v2 v2 fetch-export` |
| **Fallback** if the Kaggle export fails | Laptop download (18 min) + export (24 min) | `bash scripts/deploy_learned.sh ABDHAM/smolvla_rescuehands_v2 v2 download export` |
| ~21:40 → | Laptop: 10-seed evaluations on the iGPU (~4–5 min per seed), most important first | `bash scripts/deploy_learned.sh ABDHAM/smolvla_rescuehands_v2 v2 eval`, which runs sup-on+glitch → sup-off+glitch → sup-on+none → sup-off+none (matched pairs), all with `--save-states` |
| in parallel | Laptop: full checkpoint download, needed for the PyTorch benchmark row | `bash scripts/deploy_learned.sh ABDHAM/smolvla_rescuehands_v2 v2 download` |
| after evals | Intel benchmark on the v2 export | `bash scripts/deploy_learned.sh ABDHAM/smolvla_rescuehands_v2 v2 bench` (run on an idle laptop) |

Before 23:30 there is time for about 18 learned episodes after the export arrives.
All 40 (4 matched runs), plus the benchmark, fit only if the deadline is extended.
The first two runs form the key recovery pair: same seeds and fault, supervisor on vs off.

---

## 4. Pending (in priority order)

1. **Evaluate v2** as in section 3. Report exact denominators and keep failed runs.
2. **Fill the numbers** in `docs/submission/lablab-submission.md`, `make_slides.py`
   (`RESULTS`) and the README results table. Rebuild the slides with
   `uv run --no-project --with python-pptx==1.0.2 python docs/submission/make_slides.py`.
3. **Demo video** (10 seeds, labelled). Draw learned episodes from the saved
   states on a GPU with `render_showcase.py --states results/<run>/episode_<seed>_states.npz --video <dir>`,
   then edit about 3 minutes following `docs/presentation/submission-story-and-plan.md`.
4. **Intel benchmark** on the v2 export (PyTorch CPU vs OpenVINO CPU/iGPU/INT8),
   on an idle laptop.
5. **Submit on lablab.ai**: title, short and long description, tags, cover,
   video, slides, GitHub URL, demo platform (Hugging Face), application URL.
6. **Nice to have:** README learned-policy images from showcase replays; commit
   final result folders (summaries, manifests, episode JSONs, but no bulk videos).

### Open decisions — owner said HOLD
- **Jaw friction:** gripper pads have contact priority 1 and friction 1.0, so item
  friction randomization never reaches grasps. Fixing it changes physics and
  invalidates dataset v2. The limit is stated in the README.
- **Arm–table contacts:** detected nowhere; only arm-to-arm collisions stop the
  simulation. The proposal is to record them first, measure, then decide.
- **Closed-loop chunk-length check** (`--n-action-steps` 10 vs 25) on development
  seeds 20–24, before claiming why learned runs fail.

---

## 5. Working rules for any session
- Explain in simple English. Measure before claiming. Keep changes additive and
  never shrink the owner's plan.
- After touching the teacher or success rules, re-run scripted seeds 0–9 and compare per episode.
- Never report a number without its result file, and keep teacher and learned results separate.
- Commit only small curated media; no bulk videos or logs.
