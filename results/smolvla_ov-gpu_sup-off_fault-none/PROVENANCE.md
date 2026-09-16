# Provenance of this run (written afterwards, 2026-09-16)

This run was **interrupted** and was made before `scripts/evaluate.py` wrote a
`manifest.json` or a per-episode `summary.json`. Nothing below was recorded by
the run itself; it was reconstructed afterwards and is marked as such.

| Fact | Value | Source |
| --- | --- | --- |
| Policy | SmolVLA v1, OpenVINO, device GPU (Intel HD Graphics 520), 25 executed actions per chunk | `policy` field in each episode JSON (recorded) |
| Export | `models/openvino/fp32`, `smolvla.bin` SHA-256 starts `47070eb338ff8119` | hash of the local file today; the run did not record a hash |
| Checkpoint | `ABDHAM/smolvla_rescuehands` (12,000 steps) | export_report.json of that export |
| Supervisor / fault | off / none | folder name and episode JSON (recorded) |
| Max steps | 1500 per episode | episode JSON (recorded) |
| Code revision | probably `b2b6dcc` (last commit before the episode files, written 08:33–09:07 PKT) | file times only, **not verified**; uncommitted edits unknown |
| Episodes | seeds 0–8 have results; seed 9 has a partial video and no result | files present |

Result: 1 success (seed 4) in 9 completed episodes; 5 × `OBJECT_OUT_OF_BOUNDS`,
3 × `TIMEOUT`. This is **not** a 10-seed result and must not be reported as 1/10.
A complete re-run with the current evaluator is required for final numbers.
