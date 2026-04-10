#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0") -c <config> [-n <num_shards>] [-o <output>] [-l <log_dir>]

Run a benchmark in parallel shards (dev mode) and merge results.
Each shard writes its stdout/stderr to a separate log file.
GPU allocation is read from docker.gpus in the config file.

Options:
  -c <config>       Config YAML file (required)
  -n <num_shards>   Number of shards (default: 4)
  -o <output>       Output file for merged results (default: results/<config_name>.json)
  -l <log_dir>      Directory for per-shard log files (default: logs)
  -h                Show this help
EOF
  exit "${1:-0}"
}

CONFIG=""
NUM_SHARDS=4
OUTPUT=""
LOG_DIR="logs"

while getopts "c:n:o:l:h" opt; do
  case "$opt" in
    c) CONFIG="$OPTARG" ;;
    n) NUM_SHARDS="$OPTARG" ;;
    o) OUTPUT="$OPTARG" ;;
    l) LOG_DIR="$OPTARG" ;;
    h) usage 0 ;;
    *) usage 1 ;;
  esac
done

if [[ -z "$CONFIG" ]]; then
  echo "Error: -c <config> is required." >&2
  usage 1
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "Error: config file not found: $CONFIG" >&2
  exit 1
fi

# Derive output name from config filename if not specified
if [[ -z "$OUTPUT" ]]; then
  config_name="$(basename "$CONFIG" .yaml)"
  config_name="$(basename "$config_name" .yml)"
  OUTPUT="results/${config_name}.json"
fi

mkdir -p "$LOG_DIR"

cleanup() {
  echo "Cleaning up background processes..."
  kill -- -$$ 2>/dev/null || true
}
trap cleanup EXIT

echo "Config:     $CONFIG"
echo "Shards:     $NUM_SHARDS"
echo "Output:     $OUTPUT"
echo "Log dir:    $LOG_DIR"
echo "Mode:       --dev (mounting local src/)"
echo ""

# Check for existing shard results
existing=$(CONFIG="$CONFIG" NUM_SHARDS="$NUM_SHARDS" python3 -c "
import os, yaml, re
from pathlib import Path
with open(os.environ['CONFIG']) as f:
    cfg = yaml.safe_load(f)
num_shards = os.environ['NUM_SHARDS']
output_dir = Path(cfg.get('output_dir', './results'))
found = []
seen = set()
for b in cfg.get('benchmarks', []):
    name = b.get('name') or b['benchmark'].rsplit(':', 1)[-1]
    sub = b.get('subname')
    if sub:
        name = f'{name}_{sub}'
    safe = re.sub(r'[^\w\-.]', '_', name)
    if safe in seen:
        continue
    seen.add(safe)
    found.extend(output_dir.glob(f'{safe}_shard*of{num_shards}.json'))
if found:
    print(f'{len(found)} existing shard file(s) found, e.g.: {found[0]}')
")
if [[ -n "$existing" ]]; then
  echo "Error: $existing" >&2
  echo "Remove existing results or use a different output_dir." >&2
  exit 1
fi

echo "Launching ${NUM_SHARDS} shards..."

pids=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  log_file="${LOG_DIR}/shard${i}of${NUM_SHARDS}.log"
  echo "  shard $i -> $log_file"
  vla-eval run --dev -c "$CONFIG" --shard-id "$i" --num-shards "$NUM_SHARDS" \
    > "$log_file" 2>&1 &
  pids+=($!)
done

echo ""
echo "Waiting for all shards to finish..."
failed=0
for idx in "${!pids[@]}"; do
  pid="${pids[$idx]}"
  if ! wait "$pid"; then
    failed=$((failed + 1))
    echo "  shard $idx FAILED (see ${LOG_DIR}/shard${idx}of${NUM_SHARDS}.log)" >&2
  else
    echo "  shard $idx ok"
  fi
done

if [[ "$failed" -gt 0 ]]; then
  echo "ERROR: $failed of $NUM_SHARDS shards failed." >&2
  exit 1
fi

echo ""
echo "Merging results..."
vla-eval merge -c "$CONFIG" -o "$OUTPUT"

echo "Done. Results saved to $OUTPUT"


# 用法：
# ./scripts/run_sharded_dev.sh -c configs/robotwin_eval.yaml -n 4 -o results/my_run.json -l logs/robotwin_$(date +%Y%m%d)
