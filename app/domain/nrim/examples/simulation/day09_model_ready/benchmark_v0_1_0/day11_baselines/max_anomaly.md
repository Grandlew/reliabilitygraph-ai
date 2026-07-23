# Evaluation Result: max_anomaly

**Benchmark Fingerprint:** `993b38b0135103f24b93e73db4ee61590df8c72f62f63a921d1f30818d54a972`
**Abstention Threshold:** `disabled (no feasible validation threshold)`

## Metrics

| Split | MRR | Hits@1 | Hits@3 | Mean Rank | Healthy False-Selection | Faulty Coverage | Mean Runtime (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Validation | 0.8261 | 0.6522 | 1.0000 | 1.348 | 1.0000 | 1.0000 | 0.2794 |
| Test | 0.5350 | 0.3000 | 0.8600 | 2.460 | 1.0000 | 1.0000 | 0.2094 |
| OOD Test | 0.5824 | 0.2809 | 1.0000 | 2.067 | 1.0000 | 1.0000 | 0.1598 |

## Configuration
```json
{
  "abstention_threshold_feasible": false,
  "maximum_healthy_false_selection_rate": 0.1,
  "minimum_faulty_coverage": 0.7,
  "random_seed": 42
}
```

> **Note on Synthetic Data:** These metrics are derived from simulated IPTV environments. 
> Performance on real-world production systems may vary based on topology complexity and noise profile.