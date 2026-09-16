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


SIM_ENV = {"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "PYTHONPATH": str(ROOT / "src")}
NVEGL = ROOT / ".cache" / "nvegl"  # written by training/kaggle_gpu_render.sh
if (NVEGL / "10_nvidia.json").is_file():  # draw on the T4; without it EGL renders on the CPU
    SIM_ENV["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(NVEGL / "10_nvidia.json")
    SIM_ENV["LD_LIBRARY_PATH"] = f"{NVEGL}:{os.environ.get('LD_LIBRARY_PATH', '')}"


def shard_run(script: str, roots_and_seeds, tag: str, repo: str, extra=()):
    """Run one generator per shard in parallel; every shard writes its own log."""
    procs = []
    for root, (start, stop) in roots_and_seeds:
        cmd = [PY, script, "--root", root, "--repo-id", repo, "--seeds", f"{start}:{stop}", *extra]
        log = open(ROOT / f"{Path(root).name}.log", "w")
        procs.append(subprocess.Popen(list(map(str, cmd)), cwd=ROOT,
                                      env={**os.environ, **SIM_ENV}, stdout=log, stderr=subprocess.STDOUT))
    codes = [proc.wait() for proc in procs]
    if any(codes):
        raise SystemExit(f"{tag} shard failures: {codes}; see {tag}*.log")


def merge_and_push(repo: str, roots, out_root: Path):
    code = (
        "from lerobot.datasets.aggregate import aggregate_datasets\n"
        "from lerobot.datasets.lerobot_dataset import LeRobotDataset\n"
        f"roots = {[str(r) for r in roots]!r}\n"
        f"aggregate_datasets(repo_ids=[{repo!r}] * len(roots), roots=roots,"
        f" aggr_repo_id={repo!r}, aggr_root={str(out_root)!r})\n"
        f"ds = LeRobotDataset({repo!r}, root={str(out_root)!r})\n"
        "print('episodes', ds.num_episodes, 'frames', ds.num_frames)\n"
        "ds.push_to_hub(private=False)\n"
    )
    run([PY, "-c", code], env={"HF_TOKEN": hf_token()})


def data(hf_user: str, episodes: int, shards: int, first_seed: int, recovery_episodes: int = 0,
         repo_name: str = "rescuehands_table", base_dataset: str | None = None,
         perturbed_episodes: int = 0, perturb_scale: float = 0.6):
    """Generate clean demonstrations, optionally recovery demonstrations, and push the merge.

    Three kinds of demonstration, because a policy trained only on flawless runs has
    never seen the states it reaches once it drifts:
      clean      the tidy table, teacher succeeds first time
      perturbed  arms nudged off home and items shifted/spun, teacher solves from there
      recovery   an injected gripper fault, the supervisor's stop-and-retreat, second attempt
    Measured teacher success at perturb scale 0.6: 11/16 with the arms nudged, 8/16 with
    items moved, so expect roughly half of perturbed attempts to be kept.
    `base_dataset` is an existing Hub dataset that is downloaded and merged in as well.
    """
    repo = f"{hf_user}/{repo_name}"
    roots = []
    per = -(-int(episodes * 1.3) // shards) if episodes else 0  # ~20-30% of attempts are discarded
    if episodes:
        clean = [(f"data/clean{i}", (first_seed + i * per, first_seed + (i + 1) * per)) for i in range(shards)]
        shard_run("scripts/generate_dataset.py", clean, "data_clean", repo)
        roots += [ROOT / root for root, _ in clean]
    if perturbed_episodes:
        per_p = -(-int(perturbed_episodes * 2.0) // shards)  # about half of these attempts are kept
        start = first_seed + 50000
        rough = [(f"data/perturbed{i}", (start + i * per_p, start + (i + 1) * per_p)) for i in range(shards)]
        shard_run("scripts/generate_dataset.py", rough, "data_perturbed", repo,
                  extra=["--perturb", "--perturb-scale", str(perturb_scale)])
        roots += [ROOT / root for root, _ in rough]
    if recovery_episodes:
        per_rec = -(-int(recovery_episodes * 1.6) // shards)  # recovery attempts fail more often
        start = first_seed + 100000
        rec = [(f"data/recovery{i}", (start + i * per_rec, start + (i + 1) * per_rec)) for i in range(shards)]
        shard_run("scripts/generate_recovery_dataset.py", rec, "data_recovery", repo)
        roots += [ROOT / root for root, _ in rec]
    if base_dataset:
        base_root = ROOT / "data" / "base_download"
        code = ("from lerobot.datasets.lerobot_dataset import LeRobotDataset\n"
                f"ds = LeRobotDataset({base_dataset!r}, root={str(base_root)!r})\n"
                "print('base episodes', ds.num_episodes)\n")
        run([PY, "-c", code], env={"HF_TOKEN": hf_token()})
        roots.append(base_root)
    merge_and_push(repo, roots, ROOT / "data" / "merged")


def train(hf_user: str, steps: int, batch_size: int, save_freq: int, push: bool,
          precision: str = "fp16", gpus: int = 1, num_workers: int = 4, name: str | None = None,
          log_freq: int = 100, save: bool = True, stdout=None, init_from: str = "lerobot/smolvla_base",
          dataset_repo: str | None = None, model_repo: str | None = None, lr: float | None = None):
    """precision: bf16 = SmolVLA default weights; fp32 = float32 weights;
    fp16 = float32 weights with fp16 autocast (T4 has fast fp16 kernels, no bf16).

    init_from is the starting policy: the SmolVLA base, or one of our own checkpoints
    when fine-tuning further on a larger or recovery-augmented dataset."""
    repo = dataset_repo or f"{hf_user}/rescuehands_table"
    name = name or run_name(push)
    out = ROOT / "outputs" / name
    if not push:  # smoke runs are disposable; LeRobot refuses an existing output dir
        import shutil
        shutil.rmtree(out, ignore_errors=True)
    args = [f"--policy.path={init_from}",
            f"--dataset.repo_id={repo}",
            f"--rename_map={json.dumps(CAMERA_RENAME)}",
            f"--batch_size={batch_size}", f"--steps={steps}", f"--num_workers={num_workers}",
            f"--save_checkpoint={str(save).lower()}", f"--save_freq={save_freq}",
            f"--log_freq={min(log_freq, steps)}", "--eval_freq=0",
            f"--output_dir={out}", f"--job_name={name}",
            "--policy.device=cuda", "--wandb.enable=false",
            f"--policy.push_to_hub={str(push).lower()}",
            f"--policy.repo_id={model_repo or f'{hf_user}/smolvla_rescuehands'}"]
    if lr is not None:
        args.append(f"--optimizer.lr={lr}")
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


def verify(hf_user: str, push: bool, dataset_repo: str | None = None, name: str | None = None):
    checkpoint = ROOT / "outputs" / (name or run_name(push)) / "checkpoints" / "last" / "pretrained_model"
    run([PY, ROOT / "training" / "verify_checkpoint.py", "--checkpoint", checkpoint,
         "--dataset", dataset_repo or f"{hf_user}/rescuehands_table", "--write-contract"])


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
    parser.add_argument("--episodes", type=int, default=120, help="clean demonstrations to keep")
    parser.add_argument("--recovery-episodes", type=int, default=0,
                        help="demonstrations with an injected drop and the teacher's recovery")
    parser.add_argument("--perturbed-episodes", type=int, default=0,
                        help="demonstrations that start off-nominal (arms nudged, items moved)")
    parser.add_argument("--perturb-scale", type=float, default=0.6)
    parser.add_argument("--dataset-name", default="rescuehands_table")
    parser.add_argument("--base-dataset", help="existing Hub dataset to merge into the new one")
    parser.add_argument("--init-from", default="lerobot/smolvla_base",
                        help="starting policy: the base model, or our checkpoint to fine-tune further")
    parser.add_argument("--model-repo", help="Hub repo for the trained policy")
    parser.add_argument("--run-name", help="output folder under outputs/")
    parser.add_argument("--lr", type=float, help="override the learning rate (lower when fine-tuning)")
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
        data(args.hf_user, args.episodes, args.shards, args.first_seed, args.recovery_episodes,
             args.dataset_name, args.base_dataset, args.perturbed_episodes, args.perturb_scale)
    if args.stage == "probe":
        probe(args.hf_user, args.batch_size)
    dataset_repo = f"{args.hf_user}/{args.dataset_name}"
    if args.stage in ("train", "all"):
        train(args.hf_user, args.steps, args.batch_size, args.save_freq, not args.no_push,
              args.precision, args.gpus, args.num_workers, name=args.run_name,
              init_from=args.init_from, dataset_repo=dataset_repo, model_repo=args.model_repo, lr=args.lr)
    if args.stage in ("train", "verify"):
        verify(args.hf_user, not args.no_push, dataset_repo, args.run_name)


if __name__ == "__main__":
    main()
