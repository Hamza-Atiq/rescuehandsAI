# Physics and teacher audit — 2026-09-16

This is a read-only source review, apart from this report. Source and tests were read in full for `sim.py`, `scene.py`, `randomize.py`, `expert.py`, `kinematics.py`, `motion.py`, `control.py`, `perturb.py`, both JSON configurations, `assets/README.md`, and their eight test files. The cached upstream SO-101 XML was also read in full. Binary STL meshes were not individually inspected. The existing recovery contact sheet was opened and viewed. The parent review owns the complete test run and wider result/media audit.

## What is sound

- The scene really contains two namespaced SO-101 arms and free-moving objects. Normal `step()` updates motors and advances MuJoCo. It does not weld, attach, or teleport the objects.
- IK uses a separate `MjData`; it does not rewrite the live robot state. Its file clearly identifies it as a scripted teacher.
- Actions name all 12 joints. Validation checks mapping, finite numbers, joint and actuator limits, stale/future timestamps and command jumps before changing physics.
- Demonstrations perform actual cup transport and a utensil handoff. This is much more than a motor animation. The contact sheet visually shows both arms participating.
- Random seeds alter positions, slot order, dimensions, mass, friction configuration, table colour and lighting. These are real model parameters, with the important friction exception below.

## Findings

### P1: Safety only catches contacts between the two arms

`src/rescuehandsai/sim.py:202–213` checks nonfinite state and cross-arm contacts at each physics step. It does not stop arm/table impacts, arm/plate impacts, or self-collisions within one arm. `compute_facts` in `auditor.py` has the same narrow collision definition. Therefore a zero collision count is not evidence that the robot avoided every prohibited impact. Recovery motions that rely only on `sim.step()` have this same hole.

Independent probe: using a seeded random joint pose within all configured joint limits, the right gripper penetrated the table by about 0.0438 m. `step()` accepted a hold command and advanced to 0.05 seconds without raising a collision error. This probe deliberately created the bad starting pose; it proves missing detection, not that a normal episode necessarily reaches that pose.

Fix direction: define permitted contacts explicitly; detect prohibited self and world contacts, using depth/force/duration thresholds where needed to avoid false alarms. Apply the same audit to recovery moves. Add table/self collision tests, and label existing collision metrics as cross-arm-only until fixed.

### P1: Randomized object friction does not change the jaw grasp friction

`src/rescuehandsai/scene.py:91` and `:154` put randomized friction on object geoms with default priority 0. The cached upstream `so101.xml:24–27` gives all jaw collision geoms priority 1 and fixed friction. The higher priority wins in contacts. This weakens the claimed friction robustness test at exactly the place where slipping matters most.

Independent seed-0 cup probe: sampled cup friction was `0.7167927876527321`; every recorded cup/jaw contact used `[1.0, 1.0, 0.005, 0.0005, 0.0005]`. The contact priorities were `[0, 1]` or `[1, 0]`. Thus these are measured effective values, not just a suspicion from the XML. Table contact friction can still vary, so this is not a claim that all friction randomization is inactive. `perturb.py` says a slippery utensil did not dislodge the grasp; overridden friction is a plausible contributing reason and should be checked before attributing that entirely to strong jaws.

Fix direction: deliberately choose contact priority/mixing or explicit contact pairs so requested grasp friction becomes the effective contact friction. Test actual `data.contact[].friction` across low/high cases, not just sampled JSON values. Re-evaluate robustness after the physics change; it can materially reduce teacher success.

### P2: Direct execution silently skips the full handoff tests

`tests/test_expert.py:56–57` invokes `unittest.main()` before `ExpertFullTaskTests` is defined at line 60. Running the file directly exits after the cup tests, so the more important bimanual test never exists in that run. Unittest discovery imports the module and does include it. Move the main guard to the end. Current full-task coverage also only uses seeds 0, 1 and 2, while the required evaluation uses ten seeds.

### P2: A final-attempt staging branch can leave pickup without a successful grasp

In `expert.py:253–281`, each planning failure may call `_stage_utensil()` then `continue`, consuming a pickup attempt. If staging first becomes necessary on the final attempt, the loop ends without checking that the right hand picked up the item. It then assigns the previous attempt's `q_lift` to `_q_right`. The default three-attempt path can reach this after two failed grasps and a moved item; a one-attempt invocation would instead have an unbound `q_lift`. This is a source-path finding, not reproduced in a normal episode in this audit.

Fix direction: count actual pickup attempts separately from staging and finish only through an explicit successful-grasp path. Add a targeted test where staging occurs on the last attempt.

### P2: Lift height alone is not proof of possession

`expert.py:226`, `:274`, and `:301` treat height above 0.025 m as enough evidence for successful pickup or continued right-hand possession. A bounced utensil or one balanced on another object can be high without being held. The code already comments that single-step contacts flicker, which is a valid concern; the answer should be a short history of contact plus coherent motion, not height alone. The end-of-task evaluator is separate, so this does not automatically produce false final success, but it can create bad plans and demonstrations.

### P2: Start-state validity is less complete than its explanation claims

`randomize.py:68–94` rejects motion, high objects, cup tilt, gripper contact and starting inside a target. It does not inspect `facts.out_of_bounds` or `facts.on_item`; a low utensil partly stacked on another can pass. It checks maximum utensil height only, not low/negative height or flatness. Sampling currently clamps nominal XY positions, which reduces risk but does not prove validity after settling. The text also says rejected scenes would be impossible, which has not been established: some are merely inconvenient or excluded by the current start contract.

Fix direction: check the actual settled state's bounds and inter-item contact/support, distinguish invalid from merely difficult states, and report rejection counts by reason. Avoid claiming every teacher planning failure occurs on a guaranteed feasible scene.

### P3: Asset documentation is stale

`assets/README.md:8–9` says this is still a control-test scene rather than a dinner-table benchmark. The current scene and expert do implement dinner-table actions. Update this text and describe the simple procedural cup/fork/spoon honestly.

### P3: Scene-edit helpers rely on caller discipline

`sim.py:93–125` exposes start-pose edits, object pose rewrites and settling at any simulation time, although the comment says dataset-start-only. Current source callers found by search use these during `perturb_start`, so this review found no deployed teleporting cheat. Still, a later caller can violate the stated contract without an error. `set_item_pose` also lacks finite-value and shape validation. Restrict these helpers to an explicit initialization phase if this boundary is meant to be guaranteed.

## Evidence limits and quality gates

- The existing original scripted summaries are historical: clean 8/10, fault with supervisor 6/10, fault without supervisor 0/10. They reference three different git revisions. They must not be treated as a controlled comparison of one frozen implementation. Newer result directories exist and the main audit examines them.
- The inspected recovery contact sheet has six sparse frames. It shows plausible manipulation but cannot prove every movement is collision-free or that final placement remained stable.
- Geometric variation is mostly size variation of the same primitive models; background variation is table colour, with a fixed black surrounding area. State that narrow scope rather than implying broad unseen shape/background generalization.
- No live source edits or training changes were made for this audit. Probes used short-lived local simulator instances. The full test result must be taken from the parent audit, not inferred from the fact these probes ran.

Recommended gate order: fix contact/safety semantics; validate effective randomization; add the targeted regression cases above; freeze a code revision; regenerate teacher/evaluation evidence on all ten seeds; then judge learned policy and recovery results against that exact simulator version.
