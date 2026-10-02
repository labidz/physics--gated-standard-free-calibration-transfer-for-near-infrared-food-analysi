#!/bin/bash
# Confirmatory runs (PROTOCOL Amendment A): run once, after the amendment commit.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/claude/spectwin
nice -n 5 python3 experiments/run_e12.py conf "$1" > "logs/e12_conf_$2.log" 2>&1
