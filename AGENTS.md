# AGENTS.md — RescueHands AI

## Communication preference (applies in every session)

Always reply in simple English, as if explaining to a 10-year-old. Use short
sentences and everyday examples. Explain technical words when they are needed.
Keep the facts accurate, clearly say what works and what does not, and never
hide an important problem behind complicated language.

## Challenge corrections — September 15, 2026 (take priority below)

The user supplied the Intel online challenge brief during the first research
session. These confirmed requirements override older assumptions in this file:

- The submission scenario is **setting up a dinner table** using dual simulated
  SO-101 arms in MuJoCo. Package rescue is no longer the primary submission task.
  Keep the RescueHands name and safety/recovery thesis in this table-setting setting.
- Begin with a small table-setting action, then build the multi-step workflow.
  Include a hand-off or complementary two-arm action. A motor check alone is
  an engineering milestone, never a completed manipulation demo.
- OpenVINO optimization where supported and an Intel benchmark script are
  submission requirements, not optional stretch goals. Plan the model export
  path early while preserving simulation/control correctness as the first step.
- The brief's deployment section requires the final simulation and AI inference
  on Intel Core Ultra Series 2/3. An earlier section calls this hardware preferred
  and allows an Intel CPU/iGPU environment. Record this wording tension and use
  the stricter final-demo rule until the organizers clarify it.
- Simulation-first means physical SO-101 arms are not required. It does not by
  itself remove the final computer requirement. Do not claim hardware compliance
  without checking the actual machine or a newer organizer clarification.
- Provide evaluation and demonstration evidence across **10 randomized seeds**.
  Include object placement, weight, friction, shape, lighting and background
  changes in the final evaluation. Add them gradually during development.
- Training may use local or cloud hardware; Intel does not provide training
  infrastructure in the supplied brief.
- Scoring: task completion/bimanual manipulation 30; VLA/multimodal reasoning 20;
  robustness 15; OpenVINO/Intel optimization 20; reproducibility 10; innovation 5.
- Required package: reproducible repository and scene, training/fine-tuning,
  inference and evaluation code, Intel benchmark, 10-seed video, technical README.
- The linked installation guide targets Ubuntu 24.04 and suitable Intel hardware.
  Its driver installers are not instructions to modify this Windows computer.

### Decisions and organizer clarifications — September 15, 2026

- Deadline: 2026-09-16 23:30 Pakistan Standard Time (lablab.ai schedule).
  Submission: public MIT GitHub repo, video, slides, cover image.
- Organizer Discord reply (relayed by the owner): training may happen on any
  machine, but the VLA/VLM policy must be executed on Intel XPUs; a non-Core-Ultra
  Intel CPU + iGPU was accepted as an alternative. No remote Intel system is
  provided. Deploy on the owner's Intel Core i5-6300U CPU + HD Graphics 520.
- Organizer: IK must not dominate robot control. IK is allowed only inside the
  scripted demonstration teacher; deployed control comes from the VLA.
- Organizer: a simpler VLA combined with a more complex VLM is welcome.
- Owner decision: SmolVLA is the only learned policy. Train on free Kaggle GPU.
- Approved design: docs/superpowers/specs/2026-09-15-dinner-table-design.md

Sources: user-supplied Intel Physical AI Online Challenge brief, and
https://docs.openedgeplatform.intel.com/dev/edge-ai-suites/robotics-ai-suite/resources/hackathon_resources.html
The brief was pasted in the conversation; its original document URL is not yet
available. Do not present the installation page as containing the full rubric.

## 1. Purpose of this document

This file is the authoritative product and architecture handoff for Codex and any other coding agent working on **RescueHands AI**.

The project is intended for the **online Intel bimanual VLA manipulation challenge** associated with the AI Infra Summit Hackathon 2026. The core challenge direction discussed with the project owner is a **simulated dual-arm SO-101 manipulation system in MuJoCo using Vision-Language-Action (VLA) and multimodal reasoning**.

This document deliberately focuses on:

- the problem we are solving,
- the value proposition,
- system architecture,
- component responsibilities,
- interfaces between components,
- data and control flow,
- model strategy,
- simulation strategy,
- safety and recovery design,
- evaluation strategy,
- observability,
- project boundaries,
- decision rules for Codex,
- what should and should not be optimized.

It intentionally does **not** contain implementation commands, shell commands, copy-paste code, or a procedural installation tutorial. Codex should translate this architecture into a working repository while preserving the intent and boundaries described here.

---

# 2. Product vision

## Working name

**RescueHands AI**

## One-sentence description

A safety-aware bimanual Vision-Language-Action system that allows two simulated SO-101 robot arms to understand natural-language manipulation instructions, coordinate their actions in MuJoCo, detect physical execution failures, and recover without restarting the task.

## Core demo concept

A human gives a natural-language instruction such as:

> “Use both arms to move the large package from the danger zone to the safe platform.”

The system should:

1. observe the scene,
2. understand the instruction,
3. determine the manipulation objective,
4. coordinate two robot arms,
5. execute the task in simulation,
6. monitor the physical outcome,
7. detect failures such as slip, collision, failed grasp, or invalid placement,
8. recover or replan when possible,
9. report whether the task succeeded.

The project must feel like an **embodied AI system**, not a chatbot attached to a robot animation.

---

# 3. Problem statement

Most basic robotic manipulation demos are brittle.

A policy may work when:

- the object starts in the expected position,
- the instruction is phrased exactly as seen during training,
- the robot grasps correctly on the first attempt,
- there are no unexpected collisions,
- the object does not slip,
- the environment does not change.

Real robotic systems cannot assume this.

Bimanual manipulation is harder because two arms must share one workspace and coordinate timing, contact, grasping, and motion. A mistake by one arm can invalidate the action of the other arm.

The project therefore addresses two linked problems:

### Problem A — Bimanual instruction-following manipulation

How can two simulated robot arms jointly perform a task specified using natural language and visual observations?

### Problem B — Fragility of learned manipulation policies

How can the system identify when execution is going wrong and recover instead of blindly continuing?

---

# 4. Proposed solution

RescueHands AI uses a layered architecture rather than giving a VLA policy unrestricted control.

The system is divided into four major intelligence layers:

1. **Task Understanding**
2. **VLA / Manipulation Policy**
3. **Safety and Recovery Supervisor**
4. **MuJoCo Execution Environment**

The VLA is responsible for learned manipulation behavior.

The supervisor is responsible for deciding whether the current physical execution is acceptable.

This produces a hybrid architecture:

```text
Natural-language instruction
            +
Camera observations
            +
Robot state
            |
            v
   Task Understanding
            |
            v
      VLA Policy
            |
            v
 Candidate bimanual actions
            |
            v
Safety & Recovery Supervisor
       /            \
   approved        unsafe/failed
      |                |
      v                v
 Execute          stop/recover/replan
      |                |
      +-------> MuJoCo <------+
                  |
                  v
       new observations/state
```

The most important architectural principle is:

> The VLA proposes behavior; the execution layer and supervisor decide whether that behavior is physically acceptable.

---

# 5. Challenge alignment

The project should remain tightly aligned with the bimanual VLA manipulation theme.

The following are considered in scope:

- two simulated SO-101 arms,
- MuJoCo physics simulation,
- natural-language task instructions,
- one or more simulated cameras,
- robot proprioceptive state,
- bimanual manipulation,
- VLA-based or learned action generation,
- multimodal reasoning,
- coordinated grasping and placement,
- collision awareness,
- task success detection,
- failure detection,
- recovery or replanning,
- measurable evaluation.

The following are explicitly not the main project:

- multi-robot navigation,
- mobile robot search and rescue,
- SLAM,
- drone control,
- warehouse routing,
- generic chatbot agents,
- purely symbolic task planning with no learned manipulation,
- a web application whose robotics component is secondary.

The “rescue” framing is a use-case narrative. The technical centerpiece must remain **bimanual manipulation using VLA-style perception-language-action learning**.

---

# 6. Why this project can stand out

A basic entry could demonstrate a single pick-and-place task.

RescueHands AI should differentiate itself through the combination of:

### 6.1 Bimanual coordination

Both arms must participate meaningfully.

Examples:

- cooperative lifting,
- handover,
- hold-and-place,
- stabilize-and-manipulate,
- coordinated placement.

### 6.2 Language-conditioned behavior

The system should not merely replay a fixed animation.

The instruction should influence the target object, target location, or requested manipulation.

### 6.3 Physical failure awareness

The system should understand when execution did not produce the intended outcome.

### 6.4 Recovery

Instead of requiring a full environment reset after every failure, the system should attempt a bounded recovery when practical.

### 6.5 Measurable results

The final demo should include quantitative results such as success rate, collision rate, completion time, retry count, and recovery success.

### 6.6 Modular engineering

The architecture should make it possible to replace the VLA model, controller, planner, or simulator adapter without rewriting the entire system.

---

# 7. Target demo scenarios

The project should prioritize reliability over the number of tasks.

A polished system with two or three strong demonstrations is better than many partially working tasks.

## Scenario A — Cooperative transport

Instruction example:

> “Move the large package to the safe platform using both arms.”

Expected behavior:

- identify package,
- approach with both arms,
- achieve a stable bimanual grasp,
- lift cooperatively,
- transport,
- place on target,
- release,
- verify success.

This is the primary recommended demo because it clearly requires two arms.

## Scenario B — Hold and place

Instruction example:

> “Hold the container steady with the left arm and place the red item inside it with the right arm.”

Expected behavior:

- assign different roles to each arm,
- stabilize one object,
- manipulate another,
- maintain coordination,
- verify final containment.

This demonstrates asymmetric bimanual coordination.

## Scenario C — Handover

Instruction example:

> “Pass the object from the right arm to the left arm and place it in the target area.”

Expected behavior:

- right-arm pickup,
- handover pose,
- shared possession interval or carefully sequenced transfer,
- left-arm acquisition,
- right-arm release,
- final placement.

This demonstrates temporal coordination.

## Scenario D — Recovery demo

A task should intentionally encounter one recoverable problem:

- slightly shifted object,
- failed initial grasp,
- object slip,
- partially blocked approach,
- unexpected object displacement.

The system should detect the failure and attempt recovery.

This scenario may be more valuable to judges than adding another ordinary manipulation task.

---

# 8. System architecture

## 8.1 Architectural style

Use a modular, ports-and-adapters style.

Core logic should depend on abstractions rather than directly depending on MuJoCo, a particular VLA checkpoint, or a specific vision model.

Recommended conceptual modules:

```text
+----------------------------------------------------------+
|                    Application Layer                     |
|  Episode Orchestrator | Task Manager | Evaluation Runner |
+----------------------------------------------------------+
|                    Intelligence Layer                    |
| Language/Task Parser | VLA Policy | Recovery Planner     |
+----------------------------------------------------------+
|                     Safety Layer                         |
| Action Validator | Collision Guard | State Auditor       |
+----------------------------------------------------------+
|                   Robotics Layer                         |
| Bimanual Controller | Action Adapter | Robot State Model |
+----------------------------------------------------------+
|                  Perception Layer                        |
| Camera Interface | Scene State | Observation Builder     |
+----------------------------------------------------------+
|                  Simulation Adapter                      |
| MuJoCo World | SO-101 Models | Sensors | Physics         |
+----------------------------------------------------------+
|                  Observability Layer                     |
| Metrics | Episode Logs | Video | Traces | Artifacts      |
+----------------------------------------------------------+
```

---

# 9. Component specification

## 9.1 Episode Orchestrator

### Responsibility

Own the lifecycle of one manipulation episode.

### It should coordinate

- environment reset,
- task selection,
- instruction creation/loading,
- initial observation,
- policy inference,
- action execution,
- safety checks,
- recovery,
- success/failure detection,
- episode termination,
- metric collection,
- artifact saving.

### It should not

- perform low-level physics,
- contain model-specific tensor logic,
- implement grasp heuristics directly,
- know MuJoCo internals beyond the simulator interface.

The orchestrator is the central state machine of the application.

---

## 9.2 Task Specification

Every task should be represented as structured data rather than hidden inside prompt text.

A task conceptually contains:

- task ID,
- natural-language instruction,
- participating objects,
- target object,
- destination,
- required arm roles,
- success conditions,
- failure conditions,
- timeout,
- allowed recovery attempts,
- optional environment randomization parameters.

This structured task representation gives the system a clean separation between:

- what the human asks,
- what success means,
- how the robot chooses to accomplish it.

---

## 9.3 Task Understanding / Language Layer

### Purpose

Translate natural language into a normalized task intent.

For example:

> “Can you move the big red package over to the green safe area with both hands?”

could normalize into something conceptually similar to:

- action: transport
- object: large_red_package
- target: green_safe_zone
- required_coordination: bimanual

### Design goal

Do not make this layer responsible for joint-level control.

It should reason at the level of:

- object,
- destination,
- semantic constraints,
- arm roles,
- subgoal intent.

### Possible implementation strategies

The architecture should allow either:

- deterministic task templates,
- a lightweight language parser,
- a multimodal/VLM reasoning model,
- structured prompting around a language model.

For hackathon reliability, known demo tasks may use controlled instructions while still accepting paraphrases.

---

# 10. Observation model

The VLA and supervisor need a stable, explicit observation contract.

An observation may include:

## Visual observations

- front/global RGB camera,
- optional left-side camera,
- optional right-side camera,
- optional wrist camera if supported by the chosen simulation setup.

## Robot state

For both arms:

- joint positions,
- joint velocities,
- gripper state,
- actuator state where useful,
- end-effector pose if exposed or computed.

## Environment state

Available internally to the simulator/supervisor:

- object poses,
- object velocities,
- contacts,
- collision state,
- target-zone state,
- grasp relationship or proximity indicators.

Important distinction:

> Privileged simulator state may be used by evaluation and safety logic without necessarily being given to the learned policy.

This is useful because MuJoCo provides exact ground-truth state while a realistic policy can still operate primarily on camera/state observations.

---

# 11. VLA policy layer

## 11.1 Purpose

Convert multimodal observations and a language instruction into robot actions.

Conceptual input:

```text
camera images
+
current robot state
+
natural-language instruction
```

Conceptual output:

```text
action chunk for left arm
+
action chunk for right arm
+
gripper commands
```

The exact action representation is model-dependent and must be isolated behind an adapter.

---

# 12. Recommended VLA direction

## Primary candidate: SmolVLA / LeRobot-compatible workflow

SmolVLA is a strong architectural candidate because current LeRobot documentation describes it as a lightweight robotics foundation model using:

- multiple camera views,
- current sensorimotor state,
- natural-language instruction,

to generate action chunks.

Advantages:

- purpose-built robotics ecosystem,
- substantially smaller than very large VLA models,
- LeRobot dataset compatibility,
- appropriate for manipulation experiments,
- existing SO-101 ecosystem,
- suitable for fine-tuning,
- easier to experiment with under hackathon constraints than extremely large models.

This does not mean Codex must hard-wire SmolVLA everywhere.

The policy must be abstracted.

Recommended interface concept:

```text
VlaPolicy
  observe(...)
  predict_action(...)
  reset(...)
  metadata(...)
```

Concrete implementations may include:

- SmolVLA adapter,
- scripted baseline adapter,
- future OpenVLA-style adapter,
- replay/debug policy.

---

# 13. Baseline policy

A deterministic baseline is required.

This is not optional.

The project needs a baseline because without one we cannot tell whether the learned system actually improves anything.

The baseline may use:

- known simulator object positions,
- predefined target poses,
- inverse-kinematics or waypoint control,
- deterministic gripper sequencing,
- simple synchronized arm trajectories.

It should complete at least one core task under easy conditions.

Purpose of the baseline:

- validate the environment,
- validate success metrics,
- validate arm mapping,
- catch coordinate/action bugs,
- generate comparison metrics,
- provide a fallback demo if the learned policy is unstable.

The baseline is an engineering tool and an evaluation reference.

It should not be presented as the core AI innovation.

---

# 14. Bimanual control layer

The policy should not directly manipulate MuJoCo internals.

A **Bimanual Controller** translates policy-level actions into simulator control.

Responsibilities:

- map policy action dimensions to the correct arm joints,
- distinguish left-arm and right-arm commands,
- normalize/denormalize actions,
- handle gripper commands,
- enforce action rate limits,
- optionally interpolate actions,
- synchronize both arms,
- apply actuator commands through the simulation adapter.

This module must make arm mapping explicit.

Silent left/right action ordering errors are extremely dangerous because the simulation can still “run” while learning or inference is wrong.

---

# 15. Action representation

Codex must avoid assuming one universal action format.

Possible formats include:

- absolute joint positions,
- joint deltas,
- end-effector Cartesian deltas,
- poses,
- actuator targets,
- chunked sequences.

The architecture must therefore define a canonical internal action type and use adapters.

Conceptual internal bimanual action:

```text
timestamp
left_arm_action
right_arm_action
left_gripper
right_gripper
confidence / policy metadata
```

The simulator adapter then converts this to the exact control representation required by the SO-101 MuJoCo model.

---

# 16. Safety and Recovery Supervisor

This is the main differentiating component.

## Purpose

Observe the proposed actions and physical outcomes to detect unsafe, invalid, or failed behavior.

The supervisor should be deterministic where possible.

It does not need to be another large language model.

### Pre-action checks

Before executing a proposed action:

- joint-limit check,
- workspace-bound check,
- action-magnitude check,
- obvious arm-arm collision risk,
- obviously invalid gripper command,
- invalid numerical values,
- stale observation check.

### During/after-action checks

After execution:

- collision detected,
- excessive contact force if available,
- object unexpectedly dropped,
- object did not move when expected,
- grasp never occurred,
- target distance increasing unexpectedly,
- arm stuck or motion stalled,
- object left valid workspace,
- robot exceeded allowed task time.

---

# 17. Physics Auditor

The supervisor should contain or delegate to a **Physics Auditor**.

The Physics Auditor uses MuJoCo's privileged state to answer questions such as:

- Is the object still on the table?
- Is either gripper in meaningful contact with the target?
- Did a collision occur?
- Is the object moving with the robot after grasp?
- Was the object dropped?
- Has the object reached the target zone?
- Are both arms converging on an impossible configuration?
- Has the scene become unstable?

This is not cheating for internal monitoring.

The project should clearly distinguish:

- **policy observations**, and
- **evaluation/safety ground truth**.

---

# 18. Recovery system

Recovery should be bounded and explainable.

Do not create an uncontrolled infinite “agent loop.”

Every task should define a maximum number of recovery attempts.

## Recovery classes

### Failed approach

Possible response:

- return to a safe pose,
- re-observe,
- generate a new approach.

### Failed grasp

Possible response:

- open gripper,
- retreat slightly,
- re-observe object pose,
- retry from a corrected grasp approach.

### Object slip

Possible response:

- stop coordinated transport,
- determine new object location,
- reacquire object.

### Collision

Possible response:

- stop both arms,
- retreat to safe configuration,
- replan with additional separation.

### Incorrect placement

Possible response:

- reacquire object if safe,
- correct final position.

### Irrecoverable state

Terminate episode safely and mark failure.

Recovery must be visible in logs and metrics so it can be demonstrated to judges.

---

# 19. Recovery Planner

The Recovery Planner consumes a structured failure event.

Conceptual input:

```text
failure_type
current_scene_state
task_goal
recent_actions
retry_count
```

Conceptual output:

```text
recovery_strategy
optional subgoal
whether to continue or abort
```

The planner can begin rule-based.

If there is sufficient time, a reasoning model may be used for higher-level recovery selection, but low-level safety must remain deterministic.

A language model must never be the only mechanism preventing collisions.

---

# 20. MuJoCo simulation layer

MuJoCo is the physics authority.

The simulation layer should contain:

- dual SO-101 robot model,
- shared workspace,
- manipulable objects,
- target zones,
- obstacles if used,
- cameras,
- actuators,
- sensors,
- contacts,
- lighting and scene assets,
- deterministic reset configuration,
- randomized reset configuration.

The simulator adapter should expose a clean API to the rest of the application.

Conceptual capabilities:

```text
reset
step
get_observation
get_privileged_state
apply_action
render
check_contacts
get_camera_frame
is_terminal
```

No intelligence layer should directly scatter MuJoCo-specific calls throughout the codebase.

---

# 21. Robot model

The simulated robot should be treated as two named SO-101 instances:

- `left_arm`
- `right_arm`

Every joint, actuator, sensor, and camera mapping must be explicitly namespaced.

Avoid implicit numeric ordering whenever possible.

The system should know:

- which actuators belong to each arm,
- which joints belong to each arm,
- which gripper belongs to each arm,
- what the home pose is,
- what the safe pose is,
- joint limits,
- shared-workspace bounds.

---

# 22. Camera strategy

The minimum useful setup should provide enough visual information for manipulation.

Recommended logical configuration:

### Global/front camera

Purpose:

- identify scene layout,
- observe target object,
- observe target destination,
- support language grounding.

### Side or elevated camera

Purpose:

- improve depth/occlusion robustness,
- observe bimanual interaction.

### Optional arm-specific or wrist views

Only add if they materially improve policy performance and are supported cleanly.

More cameras increase:

- model input cost,
- training complexity,
- storage,
- synchronization risk.

Therefore, do not add cameras only for visual impressiveness.

---

# 23. Scene randomization

A strong demo should not rely on exactly one scene layout.

However, randomization must be introduced gradually.

Useful variations:

- target object x/y location,
- target-zone location,
- object orientation,
- small camera variation,
- distractor object placement,
- instruction paraphrase,
- mild obstacle placement.

Do not initially randomize everything.

The goal is to show reasonable robustness, not create an impossible research benchmark.

---

# 24. Dataset strategy

If the selected VLA requires task-specific fine-tuning, the project needs a dataset abstraction.

Each episode should logically contain:

- one or more camera streams,
- robot state trajectory,
- action trajectory,
- language instruction,
- episode/task metadata,
- timestamps,
- success status.

The data format should be compatible with the chosen VLA framework where practical.

If LeRobot/SmolVLA is selected, prefer alignment with LeRobot dataset conventions rather than inventing a proprietary format.

---

# 25. Demonstration data

For simulated training data, the architecture should support multiple generation sources.

Possible sources:

### Scripted expert

Use the deterministic controller as an expert to generate successful demonstrations.

Advantages:

- reproducible,
- scalable,
- no physical teleoperation hardware,
- precise simulator state.

### Human teleoperation

Optional if a suitable interface is available.

### Synthetic augmentation

Vary:

- object positions,
- trajectories,
- camera pose,
- target configuration,
- instruction wording.

The first dataset should be small and task-specific.

Quality is more important than raw volume.

---

# 26. Model training strategy

Do not train a foundation VLA from scratch.

The architecture should support:

1. pretrained policy,
2. task-specific fine-tuning,
3. evaluation,
4. optional optimization for Intel hardware.

The primary experimental question should be:

> Can a lightweight pretrained VLA be adapted to reliable bimanual manipulation in our simulation?

Secondary question:

> Does the safety/recovery supervisor improve completion reliability compared with the raw policy?

---

# 27. Intel optimization path

Because the challenge is Intel-aligned, the project should leave a clean path for Intel-specific inference optimization.

Potential path:

```text
trained/fine-tuned policy
        |
        v
exportable model representation
        |
        v
OpenVINO-compatible inference
        |
        v
Intel CPU / iGPU / NPU where supported
```

This must be treated as an optimization layer, not allowed to destabilize the core demo.

Architecture requirement:

The policy interface must make it possible to swap:

- native PyTorch inference,
- optimized inference runtime,

without changing the rest of the application.

If time is limited, a reliable end-to-end manipulation demo has priority over aggressive optimization.

---

# 28. OpenVINO adapter

If an OpenVINO version of the policy becomes practical, implement it behind the same policy contract.

The application should not know whether inference runs through:

- PyTorch,
- OpenVINO,
- another backend.

Record inference latency so an Intel optimization story can be quantified.

Potential metrics:

- mean policy inference latency,
- p95 inference latency,
- actions per second,
- memory consumption,
- task success after optimization.

Do not claim acceleration without measured before/after results.

---

# 29. Control loop

The runtime control loop conceptually behaves as follows:

```text
1. Receive current task.
2. Observe current scene.
3. Build multimodal policy input.
4. Ask policy for next action/action chunk.
5. Validate candidate action.
6. Execute valid action.
7. Advance physics.
8. Audit physical state.
9. Update task progress.
10. Recover/replan if necessary.
11. Repeat until success, failure, or timeout.
```

Different parts of the system may run at different frequencies.

For example:

- physics simulation can run rapidly,
- robot control at a lower control frequency,
- VLA inference at a still lower decision frequency,
- audit checks at each control step or chunk boundary.

These timing concerns should be explicit in configuration.

---

# 30. State machine

Recommended high-level task states:

```text
IDLE
PREPARING
OBSERVING
PLANNING
EXECUTING
VERIFYING
RECOVERING
SUCCEEDED
FAILED
ABORTED
```

State transitions must be explicit.

Do not create hidden transitions inside unrelated functions.

This makes debugging and demo telemetry much easier.

---

# 31. Success detection

Success must be defined mathematically/physically, not by visual impression.

Examples:

### Placement success

Object center is within target region and object velocity is below a stability threshold for a required duration.

### Cooperative transport success

Object reaches target zone while satisfying task-specific constraints.

### Handover success

Ownership/contact transitions from one arm to the other and the final object reaches the destination.

### Containment success

Target object lies inside the target container volume and is stable.

A task should not be marked successful merely because the policy says it is done.

---

# 32. Failure taxonomy

Use structured failure labels.

Recommended categories:

- `TIMEOUT`
- `COLLISION`
- `SELF_COLLISION`
- `FAILED_GRASP`
- `OBJECT_DROPPED`
- `OBJECT_OUT_OF_BOUNDS`
- `TARGET_MISSED`
- `CONTROL_LIMIT`
- `INVALID_ACTION`
- `POLICY_ERROR`
- `SIMULATION_ERROR`
- `RECOVERY_EXHAUSTED`

These labels should appear in evaluation reports.

---

# 33. Evaluation framework

The project requires repeatable evaluation.

Each policy variant should be tested across multiple seeded episodes.

Primary metrics:

## Task success rate

Fraction of episodes completed successfully.

## Collision rate

Fraction of episodes containing prohibited collision events.

## Recovery success rate

Fraction of recoverable failures that eventually result in task completion.

## Mean completion time

Elapsed simulated or wall-clock time for successful episodes.

## Mean retry count

Average number of recovery attempts.

## Path/action efficiency

Optional measure of unnecessary motion.

## Policy inference latency

Important if demonstrating Intel optimization.

---

# 34. Experimental comparisons

At minimum, target the following comparison:

| Variant | Learned policy | Safety checks | Recovery |
|---|---|---|---|
| Scripted baseline | No | Basic | No |
| VLA | Yes | Minimal | No |
| VLA + Supervisor | Yes | Yes | Yes |

Only include rows that were actually implemented.

Never invent benchmark values.

Every number shown to judges must be reproducible from stored run artifacts.

---

# 35. Observability

Every episode should produce enough evidence to understand why it succeeded or failed.

Recommended artifacts:

- episode metadata,
- task instruction,
- random seed,
- policy name/version,
- action logs,
- failure/recovery events,
- success result,
- metrics,
- rendered video,
- optional key frames,
- timing information.

A failure without logs is not useful.

---

# 36. Demo dashboard

A minimal optional dashboard or visual overlay may display:

- current instruction,
- task state,
- current subgoal,
- recovery count,
- policy latency,
- collision status,
- current success condition,
- left/right arm status.

This should remain secondary to the robotics system.

Do not spend excessive hackathon time building a polished web UI while the manipulation policy is unstable.

A clean simulation viewer with informative overlays is sufficient.

---

# 37. Project structure boundaries

Codex should organize the repository around responsibilities rather than one monolithic script.

Suggested conceptual structure:

```text
project/
  app/
      orchestration
      task_management

  sim/
      mujoco_adapter
      scene
      robot
      cameras
      sensors

  robotics/
      bimanual_controller
      action_types
      kinematics
      control_limits

  policies/
      base_policy
      scripted_policy
      smolvla_policy
      optimized_policy

  perception/
      observation_builder
      scene_representation

  reasoning/
      task_parser
      subgoal_representation

  safety/
      action_validator
      physics_auditor
      failure_detector
      recovery_planner

  data/
      dataset_adapter
      episode_recorder

  evaluation/
      metrics
      benchmark_runner
      reports

  observability/
      logging
      video
      telemetry

  configs/
      simulation
      robots
      tasks
      policies
      evaluation

  tests/
      unit
      integration
      simulation
```

Exact folder names may change, but separation of concerns should remain.

---

# 38. Configuration philosophy

Avoid important constants hidden in code.

Configuration should control:

- simulation timestep,
- control frequency,
- policy frequency,
- camera resolution,
- camera selection,
- task definitions,
- randomization ranges,
- action limits,
- joint limits,
- safety thresholds,
- recovery count,
- model/checkpoint,
- inference backend,
- seed,
- logging level,
- video recording.

There should be sensible defaults for the hackathon demo.

---

# 39. Reproducibility

Every experiment should have a seed.

Store:

- code version,
- model version,
- configuration,
- random seed,
- task ID,
- simulator version where practical.

If a run cannot be reproduced, it should not be used as evidence in the final presentation.

---

# 40. Testing strategy

Testing is especially important because robotics systems can fail silently.

## Unit tests

Test:

- action mapping,
- left/right arm indexing,
- normalization,
- task success functions,
- joint-limit checks,
- failure classification,
- recovery limits.

## Simulation integration tests

Test:

- scene loads,
- both arms respond,
- grippers respond,
- cameras render,
- object contacts occur,
- reset is deterministic,
- success detector works.

## Policy integration tests

Test:

- observation shape,
- model input schema,
- action output shape,
- valid ranges,
- reset behavior.

## Regression tests

Maintain known seeds for:

- a guaranteed baseline success,
- a known failed grasp,
- a known collision,
- a recovery scenario.

---

# 41. High-risk technical issues

Codex should actively guard against these.

## Action ordering bugs

The most dangerous silent error.

Explicitly validate which action dimension controls which joint.

## Coordinate-frame confusion

Clearly define:

- world frame,
- left-arm base frame,
- right-arm base frame,
- end-effector frames,
- camera frames,
- object frames.

## Incorrect action normalization

Training and inference normalization must match.

## Camera mismatch

Training and inference must agree on:

- camera names,
- resolution,
- ordering,
- preprocessing.

## State mismatch

The robot-state vector must use the same ordering during training and inference.

## Timing mismatch

A policy trained at one control frequency can fail at another.

## Over-randomization

Do not make early training data too broad.

## Model dependency explosion

Keep the simulation environment usable independently from the heavy ML environment where practical.

---

# 42. Hardware assumptions

The primary project is simulation-first.

No physical SO-101 arms are required for the intended online prototype.

The local development machine may or may not have sufficient GPU resources for VLA fine-tuning.

Therefore the architecture must distinguish:

- local simulation,
- model training,
- model inference,
- optional remote/cloud GPU execution.

Do not design the repository so simulation only runs on a machine with CUDA.

---

# 43. Dependency isolation

The project should aim to keep heavy ML dependencies from contaminating all modules.

Suggested logical environments:

### Core simulation/runtime

Contains:

- MuJoCo,
- robotics utilities,
- numerical tools,
- rendering,
- core project modules.

### ML policy environment

Contains:

- PyTorch,
- Transformers/LeRobot,
- VLA-specific packages,
- training stack.

These can be combined if compatibility is clean, but the architecture should not require it.

---

# 44. Security and reliability rules

Although this is simulation, coding agents must follow basic robustness rules.

- Never execute untrusted generated shell commands automatically.
- Never place API keys in source files.
- Never commit tokens.
- Validate external model/config paths.
- Fail clearly when a model or asset is missing.
- Avoid silent fallback that changes experimental meaning.
- Log which policy and model actually ran.
- Avoid downloading arbitrary assets without an explicit source.
- Pin important versions for reproducibility once the environment works.

---

# 45. Codex autonomy rules

Codex is expected to implement the system proactively, but must preserve architectural intent.

Codex may:

- choose reasonable internal class names,
- refactor for clarity,
- create tests,
- improve error handling,
- add configuration validation,
- add logging,
- create adapters,
- replace temporary internals,
- improve modularity,
- create synthetic test scenes,
- implement a scripted baseline.

Codex must not silently:

- replace bimanual manipulation with single-arm manipulation,
- remove the VLA objective,
- turn the project into a generic agent framework,
- substitute a mobile-robot problem,
- remove safety/recovery because it is difficult,
- claim evaluation results that were not measured,
- claim Intel acceleration without benchmarking,
- use privileged simulator state as hidden policy input unless explicitly configured,
- make external paid APIs mandatory for core operation,
- hard-code one model so deeply that it cannot be replaced.

---

# 46. Priority order

When tradeoffs are necessary, use this order:

1. Correct simulation
2. Correct two-arm control
3. Reliable task success detection
4. Scripted baseline
5. Correct observation/action contracts
6. VLA inference
7. One reliable learned bimanual task
8. Safety monitoring
9. Recovery
10. Evaluation
11. Intel optimization
12. Additional tasks
13. UI polish

Do not reverse this order merely to produce a visually impressive interface.

---

# 47. Minimum viable project

The minimum serious submission should include:

- dual simulated SO-101 environment,
- one meaningful bimanual task,
- natural-language task instruction,
- camera observations,
- robot-state observations,
- working deterministic baseline,
- working VLA or learned policy path,
- task success detector,
- episode video,
- quantitative evaluation.

---

# 48. Strong submission target

The desired final system should include:

- at least two bimanual tasks,
- instruction paraphrasing,
- scene variation,
- SmolVLA or equivalent fine-tuned policy,
- supervisor-based safety checks,
- at least one demonstrated recovery behavior,
- repeated seeded evaluation,
- VLA vs VLA+supervisor comparison,
- polished demo video,
- architecture diagram,
- measured inference latency,
- OpenVINO/Intel optimization if feasible and stable.

---

# 49. Stretch goals

Only attempt after the strong submission target works.

Possible stretch goals:

- additional camera views,
- richer scene randomization,
- learned recovery selection,
- automatic subgoal generation,
- policy ensemble,
- confidence-based fallback,
- domain randomization,
- synthetic-data scaling,
- optimized OpenVINO inference,
- voice instruction input,
- live telemetry dashboard,
- more complex object geometry.

---

# 50. What not to build first

Do not begin with:

- a web dashboard,
- voice control,
- complex multi-agent LLM orchestration,
- a large cloud backend,
- reinforcement learning from scratch,
- a custom foundation model,
- an elaborate digital twin,
- ROS unless the challenge stack clearly requires it,
- more than one complicated task.

These may distract from the central manipulation objective.

---

# 51. Recommended narrative for judges

The final story should be simple.

### Problem

Bimanual VLA policies are powerful but physically fragile.

### Observation

A manipulation policy can generate plausible actions while the real physical outcome diverges because of slip, collision, failed grasp, or object displacement.

### Solution

RescueHands AI combines:

- language-conditioned VLA manipulation,
- coordinated dual-arm control,
- a deterministic physics-aware safety supervisor,
- bounded recovery.

### Demonstration

The same manipulation task is evaluated:

- without recovery,
- with safety/recovery.

### Evidence

Show actual measured:

- success rate,
- collisions,
- recovery success,
- completion time,
- inference performance.

### Impact

The architecture makes learned manipulation more dependable for future assistive, industrial, disaster-response, and service robotics.

---

# 52. Technical narrative

A technically strong explanation should emphasize that this is not simply:

```text
LLM -> robot
```

It is:

```text
language + vision + proprioception
              |
              v
          VLA policy
              |
              v
       bimanual action
              |
              v
     safety validation
              |
              v
      physics execution
              |
              v
       outcome auditing
              |
        +-----+-----+
        |           |
     success      recovery
```

This closed-loop structure is the heart of the project.

---

# 53. Primary research questions

The implementation should make it possible to answer:

1. Can a compact VLA perform the selected bimanual manipulation task in simulation?

2. How robust is the policy to moderate scene variation?

3. What are the most common failure modes?

4. Does a physics-aware supervisor reduce catastrophic failures?

5. Can bounded recovery increase task success?

6. Can the policy be optimized for Intel inference without materially reducing task success?

These questions create a stronger technical story than simply saying “we built a robot demo.”

---

# 54. Data contracts

Codex should define explicit typed contracts for at least:

## Observation

Contains policy-visible multimodal input.

## PrivilegedState

Contains simulator ground truth used by safety/evaluation.

## BimanualAction

Contains synchronized left/right control intent.

## TaskSpec

Defines instruction and success conditions.

## FailureEvent

Defines structured detected failures.

## RecoveryDecision

Defines what corrective action should happen.

## EpisodeResult

Contains success/failure and metrics.

Typed contracts should be the main boundaries between modules.

---

# 55. Policy-visible vs privileged data

This distinction must remain explicit.

## Policy-visible data

Normally:

- camera images,
- language instruction,
- robot sensorimotor state.

## Privileged simulation data

Normally:

- exact object transforms,
- exact contacts,
- target-region geometry,
- collision metadata,
- hidden simulator truth.

Privileged data is acceptable for:

- evaluation,
- safety,
- scripted expert,
- dataset generation.

Do not quietly feed it into the VLA and then describe the result as vision-based manipulation.

---

# 56. Versioning

Track versions for:

- MuJoCo,
- robot assets,
- LeRobot,
- VLA checkpoint,
- training dataset,
- inference backend,
- evaluation configuration.

Current MuJoCo documentation should be treated as the authoritative simulator reference for the implemented version.

Once a working environment is stable, prefer reproducibility over chasing every new release.

---

# 57. Current external technical references

These references inform the architecture and should be checked again if APIs change.

## MuJoCo

Official documentation:
https://mujoco.readthedocs.io/

Relevant concepts:

- native Python bindings,
- physics stepping,
- actuators,
- sensors,
- cameras,
- contacts,
- rendering,
- simulation state.

At the time this architecture was prepared, MuJoCo 3.13.0 had been released in September 2026.

## Hugging Face LeRobot / SmolVLA

Official documentation:
https://huggingface.co/docs/lerobot/

SmolVLA documentation:
https://huggingface.co/docs/lerobot/v0.5.0/smolvla

The documented SmolVLA architecture consumes:

- multiple camera views,
- robot sensorimotor state,
- natural-language instruction,

and generates action chunks.

The documentation recommends task-specific fine-tuning and collecting demonstrations for the variations the task should handle.

## OpenVINO

Official documentation should be used if Intel model optimization is implemented.

Do not rely on third-party snippets for export or quantization behavior when official tooling is available.

---

# 58. Assumptions that must remain visible

The following are project assumptions, not universal truths.

1. The online challenge is simulation-based and does not require physical robot hardware.
2. Dual SO-101 manipulation in MuJoCo is the relevant target environment.
3. A LeRobot-compatible lightweight VLA is a practical model direction.
4. The exact organizer-provided starter repository, robot asset, evaluation rules, or required Intel APIs may impose additional constraints.
5. The final implementation must adapt to official challenge materials if they conflict with this document.

When official challenge documentation and this file disagree:

> official challenge requirements win.

Update this file rather than silently diverging.

---

# 59. Decision log philosophy

Important architectural decisions should be recorded.

Examples:

- selected SO-101 simulator asset,
- selected policy,
- selected action representation,
- selected control frequency,
- selected camera views,
- dataset format,
- definition of success,
- safety thresholds,
- recovery strategy,
- inference backend.

Each decision should state:

- choice,
- reason,
- alternatives considered,
- consequences.

This prevents Codex or future contributors from repeatedly revisiting settled choices without context.

---

# 60. Definition of done

The project is not “done” because the simulation opens or the arms move.

The final hackathon-quality definition of done is:

- the two-arm environment loads reliably,
- at least one meaningful task genuinely requires coordination,
- instructions influence behavior,
- the policy receives visual/state/language observations,
- a learned/VLA path produces robot actions,
- safety logic observes execution,
- failure conditions are measurable,
- at least one recovery behavior is demonstrated if feasible,
- success is automatically evaluated,
- multiple seeded runs are recorded,
- quantitative metrics are generated,
- demo artifacts are reproducible,
- claims shown to judges are supported by measurements.

---

# 61. Final instruction to Codex

Build this as a **robotics system first, AI demo second, UI product third**.

Do not optimize for lines of code or number of features.

Optimize for:

- correctness,
- clarity,
- modularity,
- measurable reliability,
- challenge alignment,
- a convincing live/video demonstration.

The key project thesis is:

> A VLA should not be trusted merely because it generated an action. A useful bimanual system must observe what physically happened, detect when reality diverged from intent, and recover safely.

Preserve that thesis throughout the implementation.
