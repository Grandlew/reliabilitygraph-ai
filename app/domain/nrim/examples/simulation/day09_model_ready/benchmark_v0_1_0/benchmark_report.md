# NRIM Benchmark Freeze Report

**Benchmark:** NRIM IPTV Reliability Benchmark

**Version:** 0.1.0

**Domain:** iptv_synthetic

**Status:** FROZEN

## Fingerprint

```text
993b38b0135103f24b93e73db4ee61590df8c72f62f63a921d1f30818d54a972
```

## Split Summary

| Split | Windows | Mean nodes | Mean edges | Mean missingness |
|---|---:|---:|---:|---:|
| ood_test | 178 | 18.47 | 18.19 | 0.9470 |
| test | 100 | 24.48 | 25.48 | 0.9452 |
| train | 448 | 17.85 | 17.53 | 0.9467 |
| validation | 138 | 20.68 | 20.20 | 0.9504 |

## Audit Findings

- **WARNING SEVERE_CLASS_IMBALANCE:** Failure-class imbalance exceeds 4:1 in train.
- **WARNING SEVERE_CLASS_IMBALANCE:** Failure-class imbalance exceeds 4:1 in validation.
- **WARNING SEVERE_CLASS_IMBALANCE:** Failure-class imbalance exceeds 4:1 in ood_test.

## Shortcut Audit

| Shortcut | Target | Baseline | Shortcut | Improvement | Suspicious |
|---|---|---:|---:|---:|---|
| node_count | future_incident | 0.5000 | 0.4398 | -0.0602 | no |
| edge_count | future_incident | 0.5000 | 0.4398 | -0.0602 | no |
| node_feature_count | future_incident | 0.5000 | 0.5000 | 0.0000 | no |
| edge_feature_count | future_incident | 0.5000 | 0.5000 | 0.0000 | no |
| missing_feature_fraction | future_incident | 0.5000 | 0.4195 | -0.0805 | no |

## Limitations

- The benchmark is entirely synthetic.
- The failure catalogue is limited to four initial IPTV fault families.
- The operating models are simplified representations of real deployments.
- No real NetUP engineer-confirmed incident outcomes are included.
- The benchmark contains no ISP broadband scenarios.
- Synthetic benchmark performance does not prove production performance.

## Interpretation

A frozen status means the synthetic benchmark passed the defined development audits.

It does not establish production performance or synthetic-to-real transfer.
