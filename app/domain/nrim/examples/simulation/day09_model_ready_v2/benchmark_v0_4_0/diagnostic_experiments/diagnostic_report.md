# NRIM Focused Diagnostic Experiment

## Stage 1

- Selected threshold: `0.646460`
- Feasible: `False`
- Stage 1 release status: `candidate`
- Validation healthy false-selection: `0.0878`
- Maximum confounder-family false-selection: `0.1538`
- Validation recall: `0.7000`
- Validation precision: `0.4590`

## Ranking without Stage 1 gating

| Split | All-faulty MRR | Accepted-faulty MRR | Stage 1 recall | End-to-end Hits@1 |
| :--- | ---: | ---: | ---: | ---: |
| validation | 0.7667 | 0.7976 | 0.7000 | 0.4500 |
| test | 0.7278 | 0.7246 | 0.7667 | 0.4000 |
| ood_test | 0.8280 | 0.8423 | 0.8889 | 0.6190 |

## Topology validation

- Seeds: `[11, 23, 37, 53, 71]`
- Retain topology under the registered rule: `True`
- validation: mean delta RR `0.00000`, 95% clustered CI `[-0.10526, 0.10000]`, P(delta>0) `0.404`
- test: mean delta RR `0.10000`, 95% clustered CI `[0.00000, 0.20588]`, P(delta>0) `0.943`
- ood_test: mean delta RR `-0.01164`, 95% clustered CI `[-0.04524, 0.05263]`, P(delta>0) `0.273`

## OOD audit

- OOD AUROC: `0.4430`
- OOD average precision: `0.1926`
- KS(OOD, train): `0.1406`
- Distance aligned with OOD definition: `False`

Detailed cohorts, feature contributions, drift, calibration, ablations, and operating curves are available in `diagnostic_experiments.json`.
