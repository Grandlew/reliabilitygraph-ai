# NRIM Benchmark Freeze Report

**Benchmark:** NRIM IPTV Reliability Benchmark

**Version:** 0.3.0

**Domain:** iptv_synthetic

**Status:** FROZEN

## Fingerprint

```text
7ad72ddd27c6b38d5449859d226fdbd14fae04f87b7d4f33468bfa9f9700ae15
```

## Split Summary

| Split | Windows | Mean nodes | Mean edges | Mean missingness |
|---|---:|---:|---:|---:|
| ood_test | 296 | 22.97 | 23.46 | 0.5514 |
| test | 208 | 26.00 | 25.00 | 0.4643 |
| train | 624 | 18.50 | 18.17 | 0.5095 |
| validation | 208 | 15.00 | 14.00 | 0.4643 |

## Audit Findings

- No structural or distribution findings.

## Shortcut Audit

| Shortcut | Target | Baseline | Shortcut | Improvement | Suspicious |
|---|---|---:|---:|---:|---|
| node_count | future_incident | 0.5000 | 0.5000 | 0.0000 | no |
| edge_count | future_incident | 0.5000 | 0.5000 | 0.0000 | no |
| node_feature_count | future_incident | 0.5000 | 0.5000 | 0.0000 | no |
| edge_feature_count | future_incident | 0.5000 | 0.5000 | 0.0000 | no |
| missing_feature_fraction | future_incident | 0.5000 | 0.3646 | -0.1354 | no |
| root_cause__node_type | root_cause_node | 0.0679 | 0.5000 | 0.4321 | no |
| root_cause__static_criticality | root_cause_node | 0.0679 | 0.5000 | 0.4321 | no |
| root_cause__node_degree | root_cause_node | 0.0679 | 0.0000 | -0.0679 | no |
| root_cause__node_position | root_cause_node | 0.0679 | 0.5000 | 0.4321 | no |
| root_cause__node_identifier | root_cause_node | 0.0679 | 0.5000 | 0.4321 | no |
| root_cause__topology_size | root_cause_node | 0.0679 | 0.5000 | 0.4321 | no |
| root_cause__scenario_identifier | root_cause_node | 0.0679 | 0.5000 | 0.4321 | no |

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
