"""Kaggle GPU pipeline: generate demonstrations, then fine-tune SmolVLA.

Run inside a Kaggle notebook (GPU T4/P100, Internet ON, secret HF_TOKEN set):

    !git clone https://github.com/<you>/rescuehandsAI.git
    %cd rescuehandsAI
    !python training/kaggle_pipeline.py --hf-user <you> --episodes 120 --steps 12000

Stages can be run separately with --stage setup|data|train|all. Rendering uses
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


def run(cmd, env=None, cwd=ROOT):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True, cwd=cwd, env={**os.environ, **(env or {})})


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


def train(hf_user: str, steps: int, batch_size: int):
    repo = f"{hf_user}/rescuehands_table"
    out = ROOT / "outputs" / "smolvla_rescuehands"
    run([PY, "-m", "lerobot.scripts.lerobot_train",
         "--policy.path=lerobot/smolvla_base",
         f"--dataset.repo_id={repo}",
         f"--rename_map={json.dumps(CAMERA_RENAME)}",
         f"--batch_size={batch_size}", f"--steps={steps}",
         "--save_freq=2000", "--log_freq=100", "--eval_freq=0",
         f"--output_dir={out}", "--job_name=smolvla_rescuehands",
         "--policy.device=cuda", "--wandb.enable=false",
         "--policy.push_to_hub=true", f"--policy.repo_id={hf_user}/smolvla_rescuehands"],
        env={"HF_TOKEN": hf_token()})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hf-user", required=True)
    parser.add_argument("--stage", choices=["setup", "data", "train", "all"], default="all")
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--first-seed", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.first_seed < 10:
        parser.error("seeds 0-9 are reserved for evaluation")
    if args.stage in ("setup", "all"):
        setup()
    if args.stage in ("data", "all"):
        data(args.hf_user, args.episodes, args.shards, args.first_seed)
    if args.stage in ("train", "all"):
        train(args.hf_user, args.steps, args.batch_size)


if __name__ == "__main__":
    main()
