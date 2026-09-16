#!/usr/bin/env bash
# Laptop side after Kaggle training: download -> contract hash check -> OpenVINO export -> seeded evaluations.
#
#   bash scripts/deploy_learned.sh <hub_model_repo> <name> [stage...]
#   bash scripts/deploy_learned.sh ABDHAM/smolvla_rescuehands_v2 v2 download export eval
#
# Stages (each is skipped when its output already exists, and stops on the first failure):
#   download  models/smolvla_rescuehands_<name>   (needs task_contract.json uploaded by the Kaggle verify stage)
#   export    models/openvino/<name>_fp32         (export_openvino checks every contract hash first)
#   eval      results/smolvla_<name>_sup-{on,off}_fault-{none,glitch} on seeds 0-9, iGPU
#   bench     results/benchmark_<name>
# Windows Git Bash paths (.venv-pai/Scripts); run from the project root.
set -euo pipefail
repo=$1; name=$2; shift 2
stages=${*:-download export eval}
PY=.venv-pai/Scripts/python.exe
ckpt=models/smolvla_rescuehands_${name}
export_dir=models/openvino/${name}_fp32
export PYTHONPATH=src PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_XET=1
log() { echo "[$(date +%H:%M:%S)] $*"; }

for stage in $stages; do
  case $stage in
    download)
      if [ -f "$ckpt/task_contract.json" ] && [ -f "$ckpt/model.safetensors" ]; then log "download: $ckpt exists"; continue; fi
      log "download: $repo -> $ckpt"
      $PY -c "from huggingface_hub import snapshot_download; snapshot_download('$repo', local_dir='$ckpt')"
      [ -f "$ckpt/task_contract.json" ] || { log "download: $repo has no task_contract.json (did the Kaggle verify stage upload it?)"; exit 1; }
      $PY -c "from pathlib import Path; from rescuehandsai.contract import load_contract; c = load_contract(Path('$ckpt'), verify=True); print('contract hashes OK:', sorted(c['files']), 'dataset', c['dataset'])"
      ;;
    export)
      if [ -f "$export_dir/task_contract.json" ]; then log "export: $export_dir exists"; continue; fi
      log "export: $ckpt -> $export_dir (about 25 min)"
      $PY scripts/export_openvino.py --checkpoint "$ckpt" --out "$export_dir"
      ;;
    eval)
      # most important first: the supervised fault run shows drops and recoveries (the demo);
      # states are saved for showcase video drawn later (on the iGPU a video costs minutes per seed)
      for combo in "on glitch" "on none" "off none"; do
        set -- $combo
        out=results/smolvla_${name}_sup-$1_fault-$2
        if [ -f "$out/summary.json" ] && grep -q '"complete": true' "$out/summary.json"; then log "eval: $out complete"; continue; fi
        [ -d "$out" ] && { log "eval: $out is incomplete; move it aside to re-run"; exit 1; }
        log "eval: $out (about 4-5 min per seed)"
        $PY scripts/evaluate.py --policy smolvla --export "$export_dir" --device GPU --seeds 0:10             --supervisor "$1" --fault "$2" --name "smolvla_${name}_sup-$1_fault-$2" --save-states
      done
      ;;
    bench)
      out=results/benchmark_${name}
      if [ -f "$out/benchmark.json" ]; then log "bench: $out exists"; continue; fi
      log "bench: $out"
      $PY scripts/benchmark_intel.py --export "$export_dir" --torch-checkpoint "$ckpt" --runs 3 --out "$out"
      ;;
    *) log "unknown stage $stage"; exit 1 ;;
  esac
done
log "done: $stages"
