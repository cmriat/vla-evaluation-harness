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

# Derive run directory and output name
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
config_name="$(basename "$CONFIG" .yaml)"
config_name="$(basename "$config_name" .yml)"
RUN_DIR="results/${config_name}_${TIMESTAMP}"
if [[ -z "$OUTPUT" ]]; then
  OUTPUT="${RUN_DIR}/${config_name}.json"
fi

# Create a temporary config with output_dir pointing to the timestamped run directory
ORIG_CONFIG="$CONFIG"
RUN_CONFIG=$(mktemp /tmp/vla-eval-sharded-XXXXXX.yaml)
ORIG_CONFIG="$ORIG_CONFIG" RUN_DIR="$RUN_DIR" RUN_CONFIG="$RUN_CONFIG" python3 -c "
import os, yaml
with open(os.environ['ORIG_CONFIG']) as f:
    cfg = yaml.safe_load(f)
cfg['output_dir'] = os.environ['RUN_DIR']
with open(os.environ['RUN_CONFIG'], 'w') as f:
    yaml.safe_dump(cfg, f)
"
# Use the timestamped config for all subsequent commands
CONFIG="$RUN_CONFIG"

mkdir -p "$LOG_DIR"

cleanup() {
  echo "Cleaning up background processes..."
  kill -- -$$ 2>/dev/null || true
  rm -f "$RUN_CONFIG" 2>/dev/null || true
}
trap cleanup EXIT

echo "Config:     $CONFIG"
echo "Shards:     $NUM_SHARDS"
echo "Run dir:    $RUN_DIR"
echo "Output:     $OUTPUT"
echo "Log dir:    $LOG_DIR"
echo "Mode:       --dev (mounting local src/)"
echo ""

mkdir -p "$RUN_DIR"

echo "Preparing tasks (running expert check once with ${NUM_SHARDS} workers)..."
vla-eval prepare-tasks --dev -c "$CONFIG" -y --workers "$NUM_SHARDS"
echo "Task preparation complete."
echo ""

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

# Clean up prepared tasks cache (may be owned by root from Docker)
rm -rf "${RUN_DIR}/.prepared_tasks" 2>/dev/null || docker run --rm -v "$(cd "$(dirname "$RUN_DIR")" && pwd)/$(basename "$RUN_DIR"):/cleanup" alpine rm -rf /cleanup/.prepared_tasks

echo "Done. Results saved to $OUTPUT"


# 用法：
# ./scripts/run_sharded_dev.sh -c configs/robotwin_eval.yaml -n 4 -o results/my_run.json -l logs/robotwin_$(date +%Y%m%d)
