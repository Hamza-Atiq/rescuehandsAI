# RescueHands AI: story and remaining-work plan

## One sentence

**A robot should notice when its plan goes wrong.**

RescueHands AI studies this with two simulated arms setting a dinner table:
a learned policy suggests movements, physics checks the result, and a bounded
supervisor stops or retries when execution fails.

This is a simulation research prototype. Do not imply proven safety around people.

## Why someone should care

A robot demo is easy to like when nothing goes wrong. The useful question is what
happens after a missed grasp or dropped fork. This project makes that failure
visible and measures whether a recovery attempt helps.

The intended users are robotics developers testing learned manipulation. The
deliverable is a reproducible small test bed, an Intel inference path, and honest
failure evidence. Do not invent customers, market size, money saved, or deployments.

## What to emphasise

1. Two arms really share an object.
2. The instruction chooses the fork or spoon.
3. Physics, not the model's own words, decides success.
4. The same seeded failure can be tested with recovery off and on.
5. The learned model can be run on the owner's modest Intel computer, once the
   completed inference run proves it. Give the actual CPU/GPU names.

Presentation cannot substitute for the robot. In the supplied rubric, task
completion, reasoning, robustness and Intel work account for 85/100 points.
Use the story to make those results clear.

## A 12-hour plan with stop rules

**Current starting point:** the independent full learned seed-0 run failed after
a spoon-drop report and two unsuccessful retries. The robot did pick up the spoon
and reach the transfer area. Start by diagnosing that run, not by claiming the
learned task is complete. Its video and JSON are in
`results/independent_trained_full_20260916/`.

This is a work budget, not a guarantee. At 08:11 Pakistan time on September 16,
the saved 23:30 deadline was about 15 hours away. Recheck the actual clock before
starting. Keep at least two hours for upload problems and final review.

| Work budget | Work | Exit condition |
| --- | --- | --- |
| 0–2 hours | Finish backend comparison; run one full learned clean episode; inspect its video | Know whether model can pick, hand over and place; save failures too |
| 2–4 hours | Fix the reproduced hand-off rule; save/check joint contract; diagnose first learned failure | One trustworthy learned manipulation, or a precisely named blocker |
| 4–7 hours | Freeze candidate; run ten learned seeds and paired fault evaluation; benchmark final Intel runtime | JSON, model hash, videos and honest score table |
| 7–9 hours | Edit main demo; make six slides and cover from real frames | Story uses only verified claims |
| 9–10 hours | Clean README/model card, exact setup commands and limitations | Another person can follow the demo path |
| 10–12 hours | Rehearse, check links and licenses, upload and review submission | All required files open and agree with results |

If one learned clean run fails, do not spend hours running ten identical failures
or six benchmark variants. Find whether the problem is input/output mismatch,
training quality, clamping, or scene mismatch. Correct that first. If teacher and
learned recovery differ, do not mix their scores.

If time gets tight, cut a fancy dashboard, new VLM, drawer/pouring expansion and
extra quantization variants. Keep one learned task, ten measured seeds, one honest
Intel comparison, a clear video and reproducible instructions.

If the learned workflow still cannot finish by the release freeze, use a truthful
fallback story: a reproducible test bed that exposes failures in learned control
and demonstrates a separate scripted recovery baseline on Intel. State clearly
that learned task completion remains unfinished. This will limit task-completion
points, but mixing teacher footage into an AI-success claim would be misleading.

## Three-minute video storyboard

Adjust to the event's confirmed video limit before uploading.

| Time | Picture | Spoken point |
| --- | --- | --- |
| 0:00–0:15 | A real recorded drop; freeze on the fallen spoon | 'A robot can follow a plan. But what happens when the spoon falls?' |
| 0:15–0:35 | Show instruction, two arms, fork and spoon | 'We built a small test: set this place, using both arms.' |
| 0:35–1:15 | Best verified learned run; keep policy label visible | 'SmolVLA sees three cameras and joint positions. It chooses the movements.' |
| 1:15–1:50 | Same-seed fault test, recovery off/on | 'Our supervisor checks what physically happened and allows a limited retry.' |
| 1:50–2:15 | Ten-seed grid, including failures | 'Here are all the runs, not just the best one.' |
| 2:15–2:40 | Actual Intel device and measured timing | 'This is the computer and runtime we tested. Simulation pauses during inference.' |
| 2:40–3:00 | Real success frame, repo link, one limitation | 'The next step is stronger learned recovery. Our code and evidence are reproducible.' |

Only use the off/on comparison as a learned-recovery claim if it uses the learned
policy in both panels. Otherwise label it **scripted baseline recovery** and state
that learned recovery remains under test. Do not hide that label in small text.

Available independently recorded baseline clip:
`results/independent_recovery_video_20260916/episode_0.mp4`.
It uses a **spoon**. Fault injection starts at step 79 (3.95 simulated seconds);
the log reports a drop at 4.10 seconds. One recovery follows and the episode
finishes at 32.9 simulated seconds. These are simulation timestamps, not live
response-time claims. Use the actual object name in the narration.

## Six slides

1. **When the fork falls, the task should not become invisible.** Real failure
   picture and the one-sentence problem.
2. **One small dinner task. Two cooperating arms.** Instruction, camera views,
   and scope. Say the plate is already placed.
3. **Predict → check → act → check again.** Four-box architecture. Show that exact
   object positions go to the teacher/evaluator, not the learned policy.
4. **Show the evidence.** Learned run plus separately labelled teacher baseline.
   Use actual per-seed outcomes; leave missing results visibly unmeasured.
5. **Intel deployment.** Actual hardware, final model precision, load/warm latency,
   task success before/after optimization. Do not reuse the six-joint spike as the
   trained-model benchmark.
6. **What works, what fails, what comes next.** Honest scope, repo/model/data links,
   and recovery-data improvement. End with the product thesis.

## Cover image

Use a sharp real frame of the hand-off. Crop the presentation camera, not a policy
camera. Title: **RescueHands AI**. Subtitle: **When the plan fails, notice it.**
Keep the fork and both grippers large. Avoid a generic glowing AI brain, fake robot
hardware, unearned 'production ready' badges, or invented performance numbers.

## Claims checklist

| Claim | Minimum proof |
| --- | --- |
| 'The AI sets the table' | Learned policy completes the stated small workflow; episode JSON + video |
| 'Language changes behaviour' | Same scene, fork versus spoon instruction, correct different target |
| 'Recovery helps' | Same-policy, paired seeds, faults actually fired; both successes and failures |
| 'Runs on Intel' | Completed final-model inference on named Intel device |
| 'Faster with OpenVINO' | Matched input/model settings, warm timing and behaviour check |
| 'Robust' | Explicit variation ranges and all ten outcomes, not a universal claim |
| '12,000 training steps' | Completed training/checkpoint-step log, not only requested config |
| 'Safe' | Avoid this broad claim; state the specific tested checks and simulation limits |

## Release package

- README opens with one real demo frame, the exact task, and the honest result table.
- All evidence rows identify code revision, model hash, seeds, supervisor/fault mode.
- Model and dataset downloads use fixed revisions; no tokens in notebook outputs.
- Video has readable labels and working audio, with captions for key facts.
- Slides, cover, repo, model and dataset links work without the owner's login.
- Main video and ten-seed evidence are clearly distinguished.
- Failures are named. A missing result is called missing.

The strongest pitch is not 'our AI does everything'. It is a small claim that a
judge can see, check and remember.
