#!/usr/bin/env bash
set -euo pipefail

BIN="../bin/dmlp"
LAYERS=("512,512,512" "1024,2048,1024" "2048,2048,2048")
BATCHES=(64 128 256 512)
IMPLS=(baseline activation_fused)
ACTIVATION="relu"

mkdir -p ../data
LOG="../data/$(date +%Y%m%d_%H%M%S)_mlp_sweep.csv"
echo "impl,layers,batch,activation,time_ms,gflops" > "$LOG"

for layers in "${LAYERS[@]}"; do
  for batch in "${BATCHES[@]}"; do
    for impl in "${IMPLS[@]}"; do
      echo "Running $impl layers=$layers batch=$batch"

      # Run and capture output
      out="$("$BIN" --layers "$layers" --batch "$batch" --activation "$ACTIVATION" --impl "$impl" --no-verify)"

      # Extract the performance line (the one that starts with "Impl=")
      perf_line="$(printf '%s\n' "$out" | awk '/^Impl=/{print; exit}')"

      # Parse Time(ms) and GFLOP/s from that line
      time_ms="$(printf '%s\n' "$perf_line" | sed -n 's/.*Time(ms)=\([0-9.]*\).*/\1/p')"
      gflops="$(printf '%s\n' "$perf_line"   | sed -n 's/.*GFLOP\/s=\([0-9.]*\).*/\1/p')"

      if [[ -z "${time_ms}" || -z "${gflops}" ]]; then
        echo "ERROR: failed to parse output for impl=$impl layers=$layers batch=$batch" >&2
        echo "Full output was:" >&2
        echo "$out" >&2
        exit 1
      fi

      # Append CSV row
      echo "$impl,$layers,$batch,$ACTIVATION,$time_ms,$gflops" >> "$LOG"
    done
  done
done

echo "Results stored in $LOG"
