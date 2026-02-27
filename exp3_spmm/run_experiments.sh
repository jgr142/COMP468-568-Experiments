#!/bin/bash

# Output file
OUTFILE="spmm_results.txt"

echo "========================================" >> $OUTFILE
echo "SpMM Benchmark Run - $(date)" >> $OUTFILE
echo "========================================" >> $OUTFILE
echo "" >> $OUTFILE

echo "Running Baseline Implementation..."
echo "----- Baseline -----" >> $OUTFILE
./spmm_baseline >> $OUTFILE 2>&1
echo "" >> $OUTFILE

echo "Running Optimized Implementation..."
echo "----- Optimized -----" >> $OUTFILE
./spmm_opt >> $OUTFILE 2>&1
echo "" >> $OUTFILE

echo "Run complete. Results saved to $OUTFILE"
