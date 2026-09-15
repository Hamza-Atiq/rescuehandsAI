"""Kaggle GPU pipeline: generate demonstrations, then fine-tune SmolVLA.

Run inside a Kaggle notebook (GPU T4/P100, Internet ON, secret HF_TOKEN set):

    !git clone https://github.com/<you>/rescuehandsAI.git
    %cd rescuehandsAI
    !python training/kaggle_pipeline.py --hf-user <you> --stage setup
    !python training/kaggle_pipeline.py --hf-user <you> --stage train --steps 12000

Stages: setup|data|probe|train|verify|all. `train` verifies the saved checkpoint
(12-D schema, normalizer stats, one real (50, 12) action chunk) when it finishes. Rendering uses
EGL on the GPU, which is hundreds of times faster than the laptop's iGPU.
Training data may be produced anywhere; the final demo and benchmark run on the
Intel laptop (organizer rule), with the same render settings (shadows off).
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv-train"
PY = VENV / "bin" / "python"
MENAGERIE_REV = "8161bba264d7fa7c99ca301e91e7fb44737676ad"
CAMERA_RENAME = {  # our semantic names -> SmolVLA base camera slots
    "observation.images.overhead": "observation.images.camera1",
    "observation.images.left_wrist": "observation.images.camera2",
    "observation.images.right_wrist": "observation.images.camera3",
}


def run(cmd, env=None, cwd=ROOT, stdout=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True, cwd=cwd, env={**os.environ, **(env or {})},
                   stdout=stdout, stderr=subprocess.STDOUT if stdout else None)


def hf_token():
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:  # Kaggle secrets
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as exc:
        raise SystemExit(f"Set a Kaggle secret named HF_TOKEN (write access): {exc}")


def setup():
    run([sys.executable, "-m", "pip", "install", "-q", "uv"])
    run(["uv", "venv", VENV, "--python", "3.12"])
    run(["uv", "pip", "install", "--python", PY, "lerobot[smolvla]==0.5.1", "transformers==5.3.0",
         "mujoco==3.13.0", "imageio", "numpy<2.3"])
    cache = ROOT / ".cache" / "menagerie"
    if not (cache / "robotstudio_so101" / "so101.xml").is_file():
        run(["git", "clone", "--filter=blob:none", "--sparse",
             "https://github.com/google-deepmind/mujoco_menagerie.git", cache])
        run(["git", "-C", cache, "sparse-checkout", "set", "robotstudio_so101"])
        run(["git", "-C", cache, "checkout", MENAGERIE_REV])
    run(["nvidia-smi"])


def data(hf_user: str, episodes: int, shards: int, first_seed: int):
    env = {"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "PYTHONPATH": str(ROOT / "src")}
    repo = f"{hf_user}/rescuehands_table"
    per = -(-int(episodes * 1.3) // shards)  # ~20% of teacher attempts fail and are skipped
    procs = []
    for i in range(shards):
        start = first_seed + i * per
        cmd = [PY, "scripts/generate_dataset.py", "--root", f"data/shard{i}", "--repo-id", repo,
               "--seeds", f"{start}:{start + per}"]
        log = open(ROOT / f"data_shard{i}.log", "w")
        procs.append(subprocess.Popen(list(map(str, cmd)), cwd=ROOT, env={**os.environ, **env},
                                      stdout=log, stderr=subprocess.STDOUT))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"shard failures: {codes}; see data_shard*.log")
    merge = (
        "from lerobot.datasets.aggregate import aggregate_datasets\n"
        f"aggregate_datasets(repo_ids=[{', '.join(repr(repo) for _ in range(shards))}],"
        f" roots=[{', '.join(repr(str(ROOT / f'data/shard{i}')) for i in range(shards))}],"
        f" aggr_repo_id={repo!r}, aggr_root={str(ROOT / 'data/merged')!r})\n"
        "from lerobot.datasets.lerobot_dataset import LeRobotDataset\n"
        f"ds = LeRobotDataset({repo!r}, root={str(ROOT / 'data/merged')!r})\n"
        "print('episodes', ds.num_episodes, 'frames', ds.num_frames)\n"
        "ds.push_to_hub(private=False)\n"
    )
    run([PY, "-c", merge], env={"HF_TOKEN": hf_token()})


def train(hf_user: str, steps: int, batch_size: int, save_freq: int, push: bool,
          precision: str = "fp16", gpus: int = 1, num_workers: int = 4, name: str | None = None,
          log_freq: int = 100, save: bool = True, stdout=None):
    """precision: bf16 = SmolVLA default weights; fp32 = float32 weights;
    fp16 = float32 weights with fp16 autocast (T4 has fast fp16 kernels, no bf16)."""
    repo = f"{hf_user}/rescuehands_table"
    name = name or run_name(push)
    out = ROOT / "outputs" / name
    if not push:  # smoke runs are disposable; LeRobot refuses an existing output dir
        import shutil
        shutil.rmtree(out, ignore_errors=True)
    args = ["--policy.path=lerobot/smolvla_base",
            f"--dataset.repo_id={repo}",
            f"--rename_map={json.dumps(CAMERA_RENAME)}",
            f"--batch_size={batch_size}", f"--steps={steps}", f"--num_workers={num_workers}",
            f"--save_checkpoint={str(save).lower()}", f"--save_freq={save_freq}",
            f"--log_freq={min(log_freq, steps)}", "--eval_freq=0",
            f"--output_dir={out}", f"--job_name={name}",
            "--policy.device=cuda", "--wandb.enable=false",
            f"--policy.push_to_hub={str(push).lower()}", f"--policy.repo_id={hf_user}/smolvla_rescuehands"]
    env = {"HF_TOKEN": hf_token(),
           "RESCUEHANDS_WEIGHTS_DTYPE": "native" if precision == "bf16" else "float32",
           "ACCELERATE_MIXED_PRECISION": "fp16" if precision == "fp16" else "no"}
    launcher = ROOT / "training" / "lerobot_train_launcher.py"
    if gpus > 1:
        cmd = [PY, "-m", "accelerate.commands.launch", "--multi_gpu", f"--num_processes={gpus}",
               f"--mixed_precision={env['ACCELERATE_MIXED_PRECISION']}", launcher, *args]
    else:
        cmd = [PY, launcher, *args]
    run(cmd, env=env, stdout=stdout)


def run_name(push: bool) -> str:
    return "smolvla_rescuehands" if push else "smolvla_smoke"


def verify(hf_user: str, push: bool):
    checkpoint = ROOT / "outputs" / run_name(push) / "checkpoints" / "last" / "pretrained_model"
    run([PY, ROOT / "training" / "verify_checkpoint.py", "--checkpoint", checkpoint,
         "--dataset", f"{hf_user}/rescuehands_table"])


def probe(hf_user: str, batch_size: int):
    """Time a few steps per precision on one GPU, without checkpoints or hub pushes."""
    import re
    import shutil
    report = []
    for precision in ("fp16", "fp32"):
        name = f"probe_{precision}"
        shutil.rmtree(ROOT / "outputs" / name, ignore_errors=True)
        log = ROOT / f"probe_{precision}.log"
        try:
            with open(log, "w") as fh:
                train(hf_user, steps=12, batch_size=batch_size, save_freq=12, push=False,
                      precision=precision, name=name, log_freq=4, save=False, stdout=fh)
            status = "ok"
        except subprocess.CalledProcessError:
            status = "FAILED"
        text = log.read_text(errors="replace")
        steps = re.findall(r"step:\d+ .*", text)
        oom = "OutOfMemory" in text or "out of memory" in text
        report.append(f"[{precision}] {status}{' (out of GPU memory)' if oom else ''}")
        report.extend(f"   {line}" for line in steps)
        if status == "FAILED" and not oom:
            report.extend(f"   {line}" for line in text.strip().splitlines()[-8:])
    print("\n===== SPEED PROBE (batch %d, 1 GPU) =====" % batch_size)
    print("\n".join(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hf-user", required=True)
    parser.add_argument("--stage", choices=["setup", "data", "probe", "train", "verify", "all"], default="all")
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--first-seed", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-freq", type=int, default=3000)
    parser.add_argument("--no-push", action="store_true", help="smoke test: keep the checkpoint local")
    parser.add_argument("--precision", choices=["fp16", "fp32", "bf16"], default="fp16",
                        help="fp16 measured 1.33 s/step on a T4 at batch 16; fp32 4.7 s; bf16 7.4 s")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if args.first_seed < 10:
        parser.error("seeds 0-9 are reserved for evaluation")
    if args.stage in ("setup", "all"):
        setup()
    if args.stage in ("data", "all"):
        data(args.hf_user, args.episodes, args.shards, args.first_seed)
    if args.stage == "probe":
        probe(args.hf_user, args.batch_size)
    if args.stage in ("train", "all"):
        train(args.hf_user, args.steps, args.batch_size, args.save_freq, not args.no_push,
              args.precision, args.gpus, args.num_workers)
    if args.stage in ("train", "verify"):
        verify(args.hf_user, not args.no_push)


if __name__ == "__main__":
    main()
