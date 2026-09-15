# OpenVINO spike: pretrained SmolVLA on this laptop

Date: 2026-09-15. Throwaway evidence from `scripts/spike_openvino_smolvla.py` and
`scripts/spike_openvino_infer.py`. Raw numbers: `artifacts/spike_openvino/report.json`
(not committed; regenerate with the scripts).

## Question

Can SmolVLA be exported to OpenVINO and run on this Intel CPU and iGPU?

## Answer

Yes, through Intel Physical AI Studio (`physicalai-train` 0.1.0).

- Environment: `.venv-pai` with `physicalai-train[smolvla,cpu]==0.1.0`,
  `lerobot[dataset]==0.5.1`, `transformers==5.3.0`, torch 2.10 CPU,
  OpenVINO 2026.1. LeRobot 0.6.1 is incompatible (moved
  `dataset_to_policy_features`), so this environment stays separate.
- `SmolVLA(pretrained_name_or_path="lerobot/smolvla_base")` loaded, and
  `policy.export(dir, backend="openvino")` produced `smolvla.xml/.bin`,
  `tokenizer.xml/.bin` and `manifest.json`.
- The pretrained base model expects 6-D state, three 256x256 cameras and a
  48-token instruction; it outputs a 50x6 action chunk. Our fine-tuned model
  will be 12-D (two arms).
- `physicalai.inference.model.InferenceModel(dir, device=...)` ran the export.

## Measurements (FP32, random inputs, 5 timed runs after one warm-up)

| Device | Device name | Mean per chunk | p95 | Load + first call |
| --- | --- | ---: | ---: | ---: |
| CPU | Intel Core i5-6300U | 30.8 s | 43.7 s | 262 s |
| GPU | Intel HD Graphics 520 (iGPU) | 4.68 s | 4.95 s | 248 s |

**Caveat:** the CPU run overlapped with MuJoCo teacher runs on the same 4-thread
CPU, so the CPU number is inflated. Re-measure alone before reporting.
Export took about 33 minutes and first model download about 37 minutes.

## Consequences

- The iGPU is the practical deployment device on this laptop.
- Inference is slower than real time (a 50-step chunk is 2.5 s of motion at
  20 Hz). The evaluation runner pauses simulation time during inference and
  reports wall-clock latency separately. This must be stated in the README.
- Optimizations to measure next: FP16 on iGPU, INT8 weight compression (NNCF),
  fewer denoising steps, fewer executed actions per chunk, and PyTorch CPU as
  the baseline row.
