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
