# Pre-registered experimental protocol (frozen before any test-set evaluation)

Frozen: 2026-10-01 (Asia/Dhaka). Any later change is logged in the "Deviations" section with a reason.

## Benchmark (FoodSpecBench v0.1)

Built by `src/spectwin/bench.py` from GitHub/PyPI mirrors. Food/agrifood datasets: mango, plum, ch2021,
ch2018, corn, wheat, cassava, sorghum, forages, pears, grapes, tecator, cookie, bioethanol, ligno,
mayonnaise, oregano, fishoil, emulsions_nir, emulsions_raman, carbs, potato, cereal. Non-food controls:
tablets (IDRC 2002, instrument transfer), minerals (Raman mixtures with known fractions).

Excluded at data QC (documented, not used): caltransfer (no axis/provenance), honey (inconsistent axis
header and label coding), coffee ATR-FTIR and coffee UCR (no spectral axis), nirpyresearch peach /
plums / milk / milk powder / coffee-NIR / avocado (axis or label meaning undocumented).

Splits: official splits as recorded in each data card. Without an official split: n >= 300 -> fixed
random 75/25 holdout (seeded, stored in the npz); n < 300 -> outer repeated 5-fold CV (2 repeats,
grouped by `groups`). PlumSPECTRA uses its frozen outer folds. Datasets with n < 30 (carbs, potato,
cereal) are mechanism checks only (LOOCV) and are excluded from accuracy rankings. Hyper-parameters
are always chosen by inner 5-fold CV on training data only (grouped when groups exist).

## E1 in-domain tasks
Regression: mango DM; plum SSC, pH, SRF, MFF, LS; ch2021 y (unit 1); ch2018 protein; corn (m5)
moisture/oil/protein/starch; wheat protein; cassava TBC; sorghum CP/NDF/ADF; forages DM/NDF; pears Brix;
tecator moisture/fat/protein; cookie fat/sucrose/dry_flour/water; bioethanol glucose/ethanol; fishoil
iodine; emulsions_nir PUFAtot/PUFAfat; emulsions_raman PUFAtot/PUFAfat.
Classification: ch2018 product; grapes variety; mayonnaise oil.

## Methods and fixed tuning budgets
Preprocessing set P = {raw, SNV, MSC, SG1(w11,p2), SG2(w15,p2), SNV+SG1, SNV+SG2}.
- PLS / PLS-DA: P x LV 1..min(25, n-1); inner-CV minimum RMSE (accuracy for PLS-DA).
- Ridge: P x alpha in logspace(-6, 3, 10); mean-centred.
- SVR / SVC (RBF): best-PLS preprocessing, PCA scores (99.9% var, <= 50), C in {1,10,100,1000},
  gamma in {0.01, 0.1, 1}/n_pc; tuning on <= 2000 samples.
- GPR: RBF + white noise on PCA scores (<= 30); fitted on <= 2000 samples.
- kNN-LWPLS: global-PLS score space, k in {30, 60, 120, 240}, local LV in {3, 6, 10}.
- LightGBM: best-PLS preprocessing; num_leaves {15, 63} x min_child_samples {5, 20}; 2000 trees,
  lr 0.03, early stopping on a 10% inner split.
- kNN: PCA scores (whitened), k in {1,3,5,10,20}, distance weights.
- CNN1D (Cui & Fearn / Bjerrum style) and patch Transformer (no physics, no pretraining): SNV input,
  Adam 1e-3, batch 64, <= 300 epochs, early stopping (patience 30) on 15% inner split, offset/slope/
  multiplicative augmentation; 3 seeds.
- SpecTwin variants: SpecTwin-PLS (PLS on twin-canonicalised spectra), SpecTwin-FT (fine-tuned
  pretrained encoder, 3 seeds), SpecTwin-Lin (ridge on physical latents). Same inner-CV rules.
Not run: TabPFN (checkpoint hosted on Hugging Face, unreachable from the compute environment),
ACT (no verified public code), UltraIR (weights unreachable; 100M parameters).

## E2 label efficiency
mango, plum SSC, ch2018 protein, sorghum CP, pears, wheat, corn m5 moisture; n_train in
{10, 20, 50, 100, 200}; 5 random draws; fixed test sets. Methods: PLS, Ridge, CNN1D, SpecTwin-PLS, SpecTwin-FT.

## E3 transfer without standards
Scenarios: corn (6 directed instrument pairs x 4 properties; 10 random partitions 40 source-cal /
20 target-adaptation / 20 target-test); ch2021 (unit 1 -> units 2..7; adaptation = 245 standards on
the target unit used unpaired; test = validation spectra on that unit); tablets (control, 1 -> 2);
mango season 1-3 -> 4 and fruit temperature; wheat 2-month drift; cassava years; grapes harvest
dates; plum leave-one-cultivar-out (SSC); ch2018 leave-one-product-out (protein); ligno off-line ->
on-line (adaptation = first 10% of the run); SimFood-RT unit -> unit. Target data are split into an
unlabeled adaptation half and a test half (10 random repeats) unless stated.
Methods: no correction; domain mean-centring; di-PLS; CORAL; SpecTwin-TTPA (no labels, no standards).
With paired standards (reference only): DS, PDS, prediction-based slope/bias.
Metrics: target RMSEP, bias, R2, gap closed vs target-trained oracle, accuracy (classification).

## E4-E11
E4 identifiability (SimFood-RT ground truth; carbs pure spectra; minerals and cookie fractions):
spectral angle, matched correlation, fraction RMSE, instrument-parameter error, Spearman(kappa(J), error),
recovery vs number of pooled systems. E5 band-centre deviation vs literature NIR assignments; NIR-Raman
latent agreement (emulsions). E6 decimation x2/x4/x8 and range truncation: delta-RMSEP and prediction
RMSD. E7 PICP at 80/90/95%, normalised width, interval score, worst-group coverage; split, residual-
scaled and weighted conformal vs PLS leverage intervals, deep ensemble, LightGBM CQR. E8 AUROC, AUPRC,
FPR@95TPR, realised FDR and power (BH, alpha = 0.1) on oregano, mayonnaise leave-one-oil-out, ch2018
leave-one-product-out and simulated instrument faults; baselines Hotelling T2+Q, DD-SIMCA,
Mahalanobis, kNN distance. E9 ligno on-line RMSE vs HPLC. E10 parameters, train time, CPU latency,
avoided reference analyses (parametric). E11 ablations and corpus-size scaling.

## Statistics
Per task: randomization t-test on paired squared errors (van der Voet). Across tasks: Friedman test,
Holm-corrected Wilcoxon signed-rank vs SpecTwin, critical-difference diagram (Nemenyi), Bayesian
signed-rank with ROPE = 1% relative RMSEP. Bootstrap 95% CIs. Alpha = 0.05.

## Hypotheses
H1 (E1) SpecTwin is non-inferior to tuned PLS on average rank (no claim of superiority in-domain).
H2 (E2) SpecTwin has lower RMSEP than PLS at n_train <= 50.
H3 (E3) SpecTwin-TTPA has lower target RMSEP than every other standard-free method.
H4 (E4) Recovery error increases with kappa(J) and decreases with the number of pooled systems.
H5 (E7) Residual-scaled conformal keeps nominal coverage and improves worst-group coverage under shift.
H6 (E8) Physics residual gives higher AUROC than Hotelling T2+Q.

## Deviations
1. 2026-10-01: PLS/PLS-DA latent-variable cap raised from 25 to 50 (all tasks). Reason: on mango the
   inner-CV optimum hit the cap (k = 25), which would have handicapped the PLS baseline. Change favours
   the baseline; the only result computed before the change (mango PLS/kNN) was discarded and re-run.
2. 2026-10-01: PlumSPECTRA conventional traits screened with pre-specified plausibility bounds
   (pH 2-7, SSC 3-35 %, weight 5-300 g); values outside are set to NaN. Reason: the public release
   contains transcription errors (pH = 4352, SSC = 123.8 %, weight = 14132 g). Plum results computed
   before the screen were discarded and re-run.
3. 2026-10-01 (before any deep-model test evaluation): compute budget for deep baselines on the
   2-core CPU: max 200 epochs, patience 20, max 4000 (CNN) / 2000 (Transformer) optimiser steps;
   3 seeds when n_train < 2000, 1 seed otherwise (mango, plum, ch2018, ch2021).
4. 2026-10-01 (data QC, before any SpecTwin evaluation): signal-type corrections found when inspecting
   value ranges: ch2021 and ch2018 were relative reflectance -> converted to log10(1/R); grapes were raw
   detector counts -> pseudo-absorbance -log10(I/I_ref) (I_ref = per-wavelength max of training spectra),
   400-1100 nm; sorghum and cereal are published first derivatives; potato is SNV; ligno cropped to
   900-1700 cm-1 (ATR cut-off noise outside). Baseline results already computed on ch2021, ch2018 and
   grapes were discarded and re-run on the corrected data.
5. 2026-10-01: SpecTwin-FT (end-to-end fine-tuning) replaced by SpecTwin-MLP (small MLP head on the frozen
   pretrained encoder + physical latents, 3 seeds; 1 seed when n_train >= 2000) for CPU budget reasons.
6. 2026-10-01: TTPA design was developed in a pilot on the CORN dataset (random partitions overlapping with
   the final corn evaluation). Corn transfer results are therefore NOT a blind test. Design frozen before
   any other transfer scenario was run: (i) instrument operator estimated from the shape of the centred
   target covariance (signal subspace), (ii) forward simulation (re-fit the source calibration on
   source spectra mapped to the target instrument), (iii) two modes fixed by scenario type, not by
   results: 'fleet' (instrument or measurement condition changed: corn, ch2021, tablets, wheat drift,
   mango fruit temperature, SimFood-RT units) and 'chem' (population/process changed on the same
   instrument: cassava years, grapes dates, mango season, plum cultivars, ch2018 products, ligno).
   Both modes are reported for every scenario.
7. 2026-10-01: di-PLS reference uses diPLSlib (heuristic regularisation) patched for NumPy 2
   (scalar extraction in functions.dipals) and scikit-learn >= 1.6 (force_all_finite keyword).
8. 2026-10-01 (DISCLOSURE): a first SpecTwin version (v0: K = 32 atoms, calibration on the low-dimensional
   twin reconstruction) was checked on the official test sets of tecator, cookie, wheat, forages and mango
   and was much worse than PLS (e.g. cookie fat RMSEP 3.89 vs 0.64); its per-sample instrument terms also
   saturated (gauge freedom). It was revised before any other SpecTwin evaluation: v1 uses K = 64 atoms,
   q = 8, smoothing 1.0, an L2 penalty pinning per-sample instrument deviations of clean spectra to zero,
   and SpecTwin-PLS calibrates on the NATIVE-grid spectrum after removing twin-estimated offset, baseline
   and multiplicative scatter (no information loss). Because v0 test results informed this revision,
   E1 results on those five datasets are not fully blind. v0 numbers are reported in the supplement.
9. 2026-10-01: twin v1 -> v2 (training-data diagnostics only, no test evaluation): the stray-light
   parameterisation initialised at ~4e-4 capped apparent absorbance at ~3.4 AU and prevented the twin from
   fitting high-absorbance NIT spectra (tecator, 2-5.5 AU; training reconstruction error 0.48). Stray light is
   now initialised at ~5e-6 (max 2e-3 per dataset), augmentation stray light <= 5e-4. Retrained from scratch.
10. 2026-10-01: unlabelled per-fold twin adaptation for datasets outside the pretraining corpus uses 100 steps (CPU budget).
11. 2026-10-01: ablation/scaling variants pretrained for 1000 steps each (CPU budget); 'abl_full' (1000 steps) is the reference; corpus-size scaling uses 10%, 25%, 100%.
12. 2026-10-01: deep baselines use 3 seeds only for single-split tasks with n_train < 2000; cross-validated tasks (5/10 outer folds) use 1 seed per fold (CPU budget). Already-completed results unchanged.
13. 2026-10-01: inner CV falls back to sample-level folds when a training set has fewer groups than folds (only reachable in E2 tiny-n draws).

## Amendment A — E12: standard-free transfer against the prior art, with fresh confirmatory data

Status: written after the E12 development runs and BEFORE any confirmatory family was evaluated; frozen by the
git commit that adds it to PROTOCOL.md. Confirmatory data (`data/raw/conf/`) had been downloaded and parsed for
QC (shapes, signal type, sample matching) but no method had been run on it.

### Why
A prior-art review (Oct 2026) identified standard-free calibration-transfer methods missing from E3:
CT-VPdtw (Zou et al., Anal. Methods 2019), uDOP (Fonseca Diaz et al., Anal. Chim. Acta 2022), TCA/TCR
(Pan et al. 2011; Malli et al. 2017) and CTAI (Zhao et al., Molecules 2019), plus generic ML domain adaptation
(optimal transport; DANN). E3's pooled replicate statistics were dominated by corn, the TTPA development pilot.

### Design
* Development families: the 12 E3 families (corn reduced to replicates 0-4). Used to develop TTPA v4.
* Confirmatory families (never used before; run once, after this amendment is committed):
  C1 IDRC-2016 wheat, manufacturer A fleet (A1 -> A2, A3, A.Test; same 248 samples; S/A/T = 124/62/62; 10 reps);
  C2 the same for manufacturer B; C3 IDRC-2016 new units + new samples (A1 -> vliadA, B1 -> vliadB,
  C.Test -> vliadC; S = 248 master spectra, A/T = 75/75 of 150; 10 reps); C4 ICRISAT cereals, benchtop FOSS
  DS2500 -> handheld Hone EVT5 on the handheld grid within 1350-2498 nm (302 matched samples, 50/25/25 %;
  10 reps); C5 sugarcane, ASD LabSpec -> 7 portable spectrometers on each device's own grid, 4 properties
  (50/25/25 % of samples with a reference value; 5 reps); C6 grain, harvest year 2022 -> 2023 on two handheld
  spectrometers (A/T = halves of the 2023 sample IDs; 10 reps); C7 cereals, sorghum <-> pearl millet on the
  FOSS DS2500 (A/T = halves of the target species; 5 reps per direction).
  Deployment context (fixed by the design, not by results): C1-C5 'fleet', C6-C7 'chem'.
  Published protocols (reported separately): CTAI's Kennard-Stone protocols for IDRC-2016 B1->B2, B1->B3,
  B3->B2 (198/50) and corn m5->mp6 (64/16), next to the RMSEPs published by Zhao et al. 2019.
* Methods (identical splits and source calibration: PLS with preprocessing and LVs by source inner CV):
  NC; DMC; CORAL; di-PLS (diPLSlib, heuristic); uDOP (pycaltransfer, 1 component, 20 SVD components);
  TCR (pycaltransfer linear TCA, mu = 1, a = source LVs <= 20, OLS); OT-DA (POT exact EMD mapping of target
  to source in a pooled 20-component PCA space); DANN (1-D CNN, gradient reversal, 500 steps, fixed
  hyper-parameters); CT-VPdtw (authors' R code; SG width 21, order 2, derivative order 1 or 2 by master CV,
  span 5, S 1, <= 20 LVs; reference built from master + unlabelled target pool; robustness changes: >= 2 LVs and
  NA-filling of unmatched warp points, without which the released code stops on most datasets); TTPA v3 (E3);
  TTPA v4 (docs/THEORY_TTPA.md) and its ablations; DS/PDS/SBC with paired standards and the target-label
  oracle where the design provides them (references only).
* TTPA v4 (src/spectwin/transfer2.py, docs/THEORY_TTPA.md): operator (shift, warp; broadening by a nested test
  in the 'fleet' context) estimated from the shape of the target covariance; out-of-span and in-span mean shift;
  every component applied only if its Monte-Carlo gate fires (19 null draws from the source, matched sizes);
  'chem' context: goodness-of-fit gate for the operator and no in-span correction.
  Development history (all on development families): v4.0 compared a full-source statistic with smaller-source
  nulls (false-adaptation rate 0.95 on corn pseudo-targets) -> replaced by matched-size statistics; unbounded
  target-side broadening was degenerate (it smooths away noise) -> source-side only, 0-5 points; always-on
  broadening failed on ligno -> nested test, and no broadening in the 'chem' context; the GOF gate and the
  in-span gate were added after the plum and SimFood FT-NIR replicates showed spurious adaptation.
* Selection rule for the confirmatory configuration (applied mechanically to the development results; all
  candidates share one fit): candidates A = TTPA4 (gof in 'chem', in-span gate on), B = TTPA4-nogof,
  C = TTPA4-abstain (no operator in 'chem'), D = TTPA4-pargate-off. Primary = the candidate with the lowest mean
  development family-level log ratio vs NC (12 families; grapes on the error rate) among candidates with no
  development family more than 5% worse than NC; if none qualifies, fewest such families, then lowest mean;
  differences < 0.5% resolved in the order A, C, B, D. The chosen name is recorded below before the
  confirmatory run; the remaining variants are reported as ablations.
  First application (12 development families): no candidate qualified; D had the fewest families > 5% worse
  than NC (6, tied with C) and the lowest mean (-15.5% vs NC). Because every candidate still degraded half of the
  development families, two more conservative candidates were added before any confirmatory run and the same rule
  was re-applied to A-F: E = TTPA4-safe (goodness-of-fit gate for the operator in both contexts; 'fleet': in-span
  mean shift always corrected, out-of-span gated; 'chem': no mean-shift correction) and F = TTPA4-safe-abstain
  (as E, but no operator in the 'chem' context either, i.e. the source model is returned unchanged).
  Second application (results/tables/e12_dev_selection.md): F = TTPA4-safe-abstain, mean change vs NC -15.4%
  over the 12 development families, 2 families > 5% worse than NC (worst +17.6%); E: -14.6%, 3 families.
* **Frozen confirmatory configuration: `TTPA4-safe-abstain` (reported as "TTPA").** In the 'fleet' context:
  operator (shift, warp; broadening by nested test) applied iff the improvement gate and the goodness-of-fit
  gate both fire; in-span mean shift always corrected (unchanged population); out-of-span mean shift corrected
  iff its gate fires. In the 'chem' context: the source model is returned unchanged (Proposition 2: neither the
  in-span mean shift nor a line-shape change is identifiable when the population may have changed, and the
  operator was spurious in every development population family).
* Compute caps (memory): TCR, OT-DA and CT-VPdtw use at most 2,000 random source spectra (mango and ch2018
  development families; plum ran before the cap was introduced and used all source spectra).

### Endpoints
* Primary: replicate-level log(RMSEP_TTPA4 / RMSEP_comparator), linear mixed model with random intercepts
  for family and for scenario within family; one test per standard-free comparator (NC, DMC, CORAL, di-PLS,
  uDOP, TCR, OT-DA, DANN, CT-VPdtw, TTPA v3), Holm-adjusted, two-sided alpha = 0.05.
  Regression families only (all confirmatory families are regression).
* Secondary: family-level wins (one geometric-mean ratio per family) with sign tests; average rank of all
  standard-free methods across families; gate behaviour per family.
* Safety: number of confirmatory families in which TTPA4 is more than 5% worse than no correction
  (target: zero in 'chem' families).
* Gate validity (E12-size): false-adaptation rate on same-instrument pseudo-targets of the development
  datasets (target <= 5% per gate on average).
