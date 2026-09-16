#!/usr/bin/env bash
# Backup data generation on the laptop: small finalized shards so progress is never lost.
# usage: scripts/local_data_worker.sh <worker_id> <first_seed> <shards> <seeds_per_shard>
# A shard is reused only if it carries a .finalized marker (written after the generator
# exited successfully). A leftover folder without one is reported and the worker stops,
# because the generator refuses to overwrite it and silently skipping it would lose data.
set -u
worker=$1; first=$2; shards=$3; per=$4
failed=0
for ((i = 0; i < shards; i++)); do
  start=$((first + i * per)); stop=$((start + per))
  root="data/local_w${worker}_s${i}"
  if [ -f "$root/.finalized" ]; then continue; fi
  if [ -d "$root" ]; then
    echo "incomplete shard $root (no .finalized marker): inspect or remove it" >&2
    failed=1; break
  fi
  if PYTHONPATH=src PYTHONIOENCODING=utf-8 .venv-pai/Scripts/python.exe scripts/generate_dataset.py \
      --root "$root" --repo-id ABDHAM/rescuehands_table --seeds "${start}:${stop}" > "data_local_w${worker}_s${i}.log" 2>&1; then
    touch "$root/.finalized"
  else
    echo "generator failed for $root; see data_local_w${worker}_s${i}.log" >&2
    failed=1; break
  fi
done
if [ "$failed" -eq 0 ]; then echo done > "data_local_w${worker}.done"; else exit 1; fi
