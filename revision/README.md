# Supplementary data and code for the revision (post hoc analyses)
All analyses here were added in revision and are post hoc; the frozen configuration and Table 3 are unchanged.
- `out2/T1_master.csv`: one row per method and run (511 runs; includes MeanOnly_refit and NoOperator_refit for 130 confirmatory runs).
- `out2/T2_diagnostics.csv`: frozen-TTPA gates and estimates per run.
- `out3/*.csv`: same-instrument draws, recovery simulation, di-PLS penalty, subspace/baseline sensitivity, corn paired-spectrum validation.
- `make_tables.py`: regenerates `tex/tables_appendixD.tex` (Appendix D tables) from the CSVs; run from this folder (pandas, statsmodels, scipy).
- `build_master.py`: rebuilds T1 and T2 from the per-run result JSONs of the main repository.
- `add_meanonly_variants.py`: adds the two operator-ablation variants to the repository code; then run `E12_UPDATE=1 python experiments/run_e12.py conf <family>` (stored splits, raw data required).
- `extra_jobs.py`: recovery, same-instrument, di-PLS penalty and sensitivity jobs (run from the repository's `experiments/` folder).
- `paired_validation.py`: spectrum-level validation on the paired corn spectra (`python paired_validation.py <repository folder>`).
