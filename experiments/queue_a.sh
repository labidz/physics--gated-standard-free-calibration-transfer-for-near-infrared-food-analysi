#!/usr/bin/env bash
cd /home/claude/spectwin; export OMP_NUM_THREADS=1
while kill -0 2752 2>/dev/null; do sleep 30; done
python3 experiments/run_e1_twin.py real
python3 experiments/run_e2.py
python3 experiments/run_e6_e10.py e6,e10
