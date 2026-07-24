# NRIM Focused Diagnostic Experiment

## Stage 1

- Selected threshold: `0.597188`
- Feasible: `False`
- Validation healthy false-selection: `0.3212`
- Validation recall: `0.7183`
- Validation precision: `0.5368`

## Ranking without Stage 1 gating

| Split | All-faulty MRR | Accepted-faulty MRR | Stage 1 recall | End-to-end Hits@1 |
| :--- | ---: | ---: | ---: | ---: |
| validation | 0.6937 | 0.6454 | 0.7183 | 0.3239 |
| test | 0.6622 | 0.6515 | 0.8800 | 0.3333 |
| ood_test | 0.8451 | 0.8958 | 0.6038 | 0.4906 |

## Topology validation

- Seeds: `[11, 23, 37, 53, 71]`
- Retain topology under the registered rule: `False`
- validation: mean delta RR `0.03005`, 95% clustered CI `[-0.00379, 0.08519]`, P(delta>0) `0.749`
- test: mean delta RR `-0.00600`, 95% clustered CI `[-0.00918, 0.00000]`, P(delta>0) `0.000`
- ood_test: mean delta RR `-0.08333`, 95% clustered CI `[-0.08631, -0.08000]`, P(delta>0) `0.000`

## OOD audit

- OOD AUROC: `0.4215`
- OOD average precision: `0.2212`
- KS(OOD, train): `0.1673`
- Distance aligned with OOD definition: `False`

Detailed cohorts, feature contributions, drift, calibration, ablations, and operating curves are available in `diagnostic_experiments.json`.
