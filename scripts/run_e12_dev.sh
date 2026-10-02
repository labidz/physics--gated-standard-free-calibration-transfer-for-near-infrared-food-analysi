#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/claude/spectwin
nice -n 5 python3 experiments/run_e12.py dev "$1" > "logs/e12_dev_$2.log" 2>&1
