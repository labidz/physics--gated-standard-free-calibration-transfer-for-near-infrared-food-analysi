#!/usr/bin/env bash
cd /home/claude/spectwin; export OMP_NUM_THREADS=1
while kill -0 3047 2>/dev/null; do sleep 30; done
python3 experiments/run_e7_e8.py e7,e8
python3 experiments/run_e4_na.py
