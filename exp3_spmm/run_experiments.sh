#!/bin/bash
set -euo pipefail

# Binaries
BASELINE="./spmm_baseline"
OPT="./spmm_opt"

# Outputs
OUTTXT="spmm_results.txt"
OUTCSV="spmm_results.csv"

# Sweeps (edit as you like)
MS=(256 512 1024 2048)
KS=(256 512 1024 2048)
NS=(32 64 128 256)
DENSITY=0.01
SEED=1234

# Fresh files
: > "$OUTTXT"
: > "$OUTCSV"

echo "========================================" >> "$OUTTXT"
echo "SpMM Benchmark Sweep - $(date)" >> "$OUTTXT"
echo "========================================" >> "$OUTTXT"
echo "" >> "$OUTTXT"

# CSV header
echo "impl,M,K,N,density,nnz,time_ms,gflops" >> "$OUTCSV"

parse_perf_line() {
  # Reads PERF line from stdin and prints: nnz time_ms gflops
  # Expected format:
  #   PERF M=... K=... N=... density=... nnz=... time_ms=... gflops=...
  awk '
    /^PERF / {
      for (i=1;i<=NF;i++) {
        split($i, kv, "=")
        if (kv[1]=="nnz") nnz=kv[2]
        else if (kv[1]=="time_ms") t=kv[2]
        else if (kv[1]=="gflops") g=kv[2]
      }
      if (nnz!="" && t!="" && g!="") {
        print nnz, t, g
        exit
      }
    }
  '
}

run_one() {
  local impl="$1"
  local bin="$2"
  local M="$3"
  local K="$4"
  local N="$5"
  local density="$6"
  local seed="$7"

  echo "----- ${impl} M=${M} K=${K} N=${N} density=${density} seed=${seed} -----" >> "$OUTTXT"
  # Capture stdout+stderr
  local out
  out="$("$bin" --M "$M" --K "$K" --N "$N" --density "$density" --seed "$seed" 2>&1 | tee -a "$OUTTXT")"
  echo "" >> "$OUTTXT"

  # Extract PERF line fields
  local nnz time_ms gflops
  read -r nnz time_ms gflops < <(printf "%s\n" "$out" | parse_perf_line)

  # Append CSV row
  echo "${impl},${M},${K},${N},${density},${nnz},${time_ms},${gflops}" >> "$OUTCSV"
}

echo "Sweeping sizes..." | tee -a "$OUTTXT"
echo "" >> "$OUTTXT"

for M in "${MS[@]}"; do
  for K in "${KS[@]}"; do
    for N in "${NS[@]}"; do
      echo "Running M=$M K=$K N=$N density=$DENSITY seed=$SEED" | tee -a "$OUTTXT"

      run_one "baseline" "$BASELINE" "$M" "$K" "$N" "$DENSITY" "$SEED"
      run_one "optimized" "$OPT"      "$M" "$K" "$N" "$DENSITY" "$SEED"
    done
  done
done

echo "========================================" >> "$OUTTXT"
echo "Sweep complete - $(date)" >> "$OUTTXT"
echo "Text log: $OUTTXT" | tee -a "$OUTTXT"
echo "CSV: $OUTCSV" | tee -a "$OUTTXT"
echo "========================================" >> "$OUTTXT"

echo "Run complete. Results saved to $OUTTXT and $OUTCSV"
