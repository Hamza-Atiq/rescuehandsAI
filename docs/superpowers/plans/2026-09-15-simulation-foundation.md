# Simulation foundation implementation plan

Execute inline from the user-approved AGENTS.md architecture.

**Goal:** A verified two-arm SO-101 control check, before dinner-table manipulation.

**Architecture:** The simulator owns physics. Named actions pass through a
validator. Observations expose robot state and cameras; separate privileged
state exposes object poses and contacts. The application records the run.

**Stack:** Python 3.12, MuJoCo 3.13.0, NumPy, imageio, unittest.

**Spec:** AGENTS.md, including the September 15 challenge corrections.

## Files and checks

- [ ] Resolve native MuJoCo load failure with a minimal import test.
- [ ] configs/simulation.json: timings, camera size, home pose and source path.
- [ ] contracts.py: named absolute joint targets in radians and copied observations.
- [ ] control.py: reject wrong names, NaN, limits, stale timestamps and large steps.
- [ ] sim.py: compose real left_arm/right_arm SO-101 models; reset, step, render,
  read contacts and privileged table-item state.
- [ ] tests/test_foundation.py: write tests first; verify mapping, rejection without
  motion, independent arm/gripper response, deterministic reset and RGB frames.
- [ ] app.py: run a small control sequence and save seed, versions, actions,
  contacts, images and measured movement. Never label it manipulation success.
- [ ] README.md: runnable setup, source attribution, actual checks and limitations.
- [ ] Run the tests and command-line demo; inspect the camera output.

## Next milestones

Physical table-item grasp/release; complementary two-arm sequence; scene-based
task success; demonstrations and learned policy; bounded recovery; 10 randomized
seeds; OpenVINO benchmark and final Intel demonstration. These are required work
beyond this foundation, not completed features.
