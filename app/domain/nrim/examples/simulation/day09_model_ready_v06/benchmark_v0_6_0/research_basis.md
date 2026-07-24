# NRIM v0.6 Research Basis

## Design decision

NRIM v0.6 uses a support-first, dual-path incident gate:

1. A support sentinel maps unsupported observations to `UNKNOWN` or
   `ESCALATE`; unsupported windows cannot enter Stage 2.
2. A fast path requires classifier/residual evidence plus independently
   observed service impact, signal breadth, and causal consistency.
3. A slow path applies a one-sided CUSUM to healthy-conditioned Stage 1
   log-odds residuals.
4. Validation risk is counted once per independent topology group and
   bounded with a one-sided exact Clopper-Pearson interval.

## Why this statistical contract

- Sequential change detectors expose an explicit false-alarm/detection-delay
  tradeoff. Robust Bayesian online change-point detection provides
  non-asymptotic false-alarm and delay analysis, while confidence-sequence
  methods provide an alternative under weak distributional assumptions:
  [Alami et al., ICML 2020](https://proceedings.mlr.press/v119/alami20a.html)
  and
  [Shekhar and Ramdas, ICML 2024](https://proceedings.mlr.press/v235/shekhar24a.html).
- Distributionally robust CUSUM is useful when the post-change law is
  uncertain, but it still requires a stated uncertainty model:
  [Xie, Liang, and Veeravalli, AISTATS 2024](https://proceedings.mlr.press/v238/xie24a.html).
- Point-wise metrics can misrepresent temporal incidents. The benchmark
  therefore reports episode recall, delay, recovery, and fragmentation:
  [Tatbul et al., NeurIPS 2018](https://proceedings.neurips.cc/paper/2018/hash/8f468c873a32bb0619eaeb2050ba45d1-Abstract.html).
- Conformal risk control is relevant when its exchangeability/calibration
  assumptions are defensible:
  [Angelopoulos et al., 2022](https://arxiv.org/abs/2208.02814).
  NRIM does not claim conformal coverage here because its overlapping
  temporal windows and registered topology shifts do not establish those
  assumptions.

## Claims deliberately not made

NRIM does **not** claim a distribution-free CUSUM false-alarm guarantee.
Windows overlap, residuals are estimated, and scenario observations are
dependent. The defensible claim is empirical and finite-sample:

- the calibration unit is an independent topology group;
- the locked test was sealed before calibration and opened once;
- no simulator context or cohort identifier enters an inference tensor;
- Stage 2 uses its frozen v0.5 feature view and weights;
- all evidence remains synthetic until prospective real-data shadow testing.

## Final v0.6 interpretation

Mandatory validation, development-test, and locked-test operating criteria
passed. The benchmark remains a candidate because persistence-only already
achieved perfect short-episode recall, so the preregistered fast-path
increment cannot be identified on this synthetic cohort. This is a benchmark
ceiling, not evidence that the fast path adds value. The next experiment
should introduce realistic partial observability and delayed/degraded service
probes, then test the fast-path increment prospectively without reopening the
v0.6 locked test.
