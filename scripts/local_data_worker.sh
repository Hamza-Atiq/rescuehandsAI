#!/usr/bin/env bash
# Backup data generation on the laptop: small finalized shards so progress is never lost.
# usage: scripts/local_data_worker.sh <worker_id> <first_seed> <shards> <seeds_per_shard>
set -u
worker=$1; first=$2; shards=$3; per=$4
for ((i = 0; i < shards; i++)); do
  start=$((first + i * per)); stop=$((start + per))
  root="data/local_w${worker}_s${i}"
  [ -d "$root" ] && continue
  PYTHONPATH=src PYTHONIOENCODING=utf-8 .venv-pai/Scripts/python.exe scripts/generate_dataset.py \
    --root "$root" --repo-id ABDHAM/rescuehands_table --seeds "${start}:${stop}" > "data_local_w${worker}_s${i}.log" 2>&1
done
echo done > "data_local_w${worker}.done"
