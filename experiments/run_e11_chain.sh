#!/usr/bin/env bash
# Ablation / scaling pretraining (1000 steps each, same corpus and seed), then evaluation.
cd /home/claude/spectwin
P=$(cat logs/classical_b.pid)
while kill -0 $P 2>/dev/null; do sleep 30; done
export OMP_NUM_THREADS=1
python3 experiments/pretrain_twin.py abl_full 1000 modality=vnir
python3 experiments/pretrain_twin.py abl_nophys 1000 modality=vnir use_physics=0
python3 experiments/pretrain_twin.py abl_noinv 1000 modality=vnir use_invariance=0
python3 experiments/pretrain_twin.py abl_noinst 1000 modality=vnir use_instrument=0
python3 experiments/pretrain_twin.py abl_noscatter 1000 modality=vnir use_scatter=0
python3 experiments/pretrain_twin.py abl_noprep 1000 modality=vnir use_prep_op=0
python3 experiments/pretrain_twin.py abl_sim 1000 modality=vnir sim=1
python3 experiments/pretrain_twin.py abl_frac25 1000 modality=vnir frac=0.25
python3 experiments/pretrain_twin.py abl_frac10 1000 modality=vnir frac=0.10
python3 experiments/run_e11.py abl_full,abl_nophys,abl_noinv,abl_noinst,abl_noscatter,abl_noprep,abl_sim,abl_frac25,abl_frac10,real
