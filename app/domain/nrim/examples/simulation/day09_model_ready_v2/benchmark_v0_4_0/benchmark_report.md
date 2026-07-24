# NRIM Benchmark Freeze Report

**Benchmark:** NRIM IPTV Reliability Benchmark

**Version:** 0.4.0

**Domain:** iptv_synthetic

**Status:** FROZEN

## Fingerprint

```text
bc0eca8e18e9befa17e11c806dbe47a71cf47e80cf712d2b9c690368a7e8daf5
```

## Split Summary

| Split | Windows | Mean nodes | Mean edges | Mean missingness |
|---|---:|---:|---:|---:|
| ood_test | 450 | 19.64 | 19.64 | 0.5324 |
| test | 416 | 20.50 | 19.50 | 0.4643 |
| train | 1040 | 17.70 | 17.40 | 0.5086 |
| validation | 416 | 17.25 | 16.75 | 0.4821 |

## Audit Findings

- No structural or distribution findings.

## Shortcut Audit

| Shortcut | Target | Baseline | Shortcut | Improvement | Suspicious |
|---|---|---:|---:|---:|---|
| node_count | future_incident | 0.5000 | 0.5569 | 0.0569 | no |
| edge_count | future_incident | 0.5000 | 0.5711 | 0.0711 | no |
| node_feature_count | future_incident | 0.5000 | 0.5000 | 0.0000 | no |
| edge_feature_count | future_incident | 0.5000 | 0.5000 | 0.0000 | no |
| missing_feature_fraction | future_incident | 0.5000 | 0.4858 | -0.0142 | no |
| root_cause__node_type | root_cause_node | 0.0610 | 0.5000 | 0.4390 | no |
| root_cause__static_criticality | root_cause_node | 0.0610 | 0.5000 | 0.4390 | no |
| root_cause__node_degree | root_cause_node | 0.0610 | 0.0000 | -0.0610 | no |
| root_cause__node_position | root_cause_node | 0.0610 | 0.5000 | 0.4390 | no |
| root_cause__node_identifier | root_cause_node | 0.0610 | 0.5000 | 0.4390 | no |
| root_cause__topology_size | root_cause_node | 0.0610 | 0.5000 | 0.4390 | no |
| root_cause__scenario_identifier | root_cause_node | 0.0610 | 0.5625 | 0.5015 | no |

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
