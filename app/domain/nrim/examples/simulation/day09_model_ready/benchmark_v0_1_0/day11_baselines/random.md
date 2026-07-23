# Evaluation Result: random

**Benchmark Fingerprint:** `993b38b0135103f24b93e73db4ee61590df8c72f62f63a921d1f30818d54a972`
**Abstention Threshold:** `disabled (no feasible validation threshold)`

## Metrics

| Split | MRR | Hits@1 | Hits@3 | Mean Rank | Healthy False-Selection | Faulty Coverage | Mean Runtime (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Validation | 0.1785 | 0.0435 | 0.1449 | 10.928 | 1.0000 | 1.0000 | 0.1365 |
| Test | 0.1659 | 0.0600 | 0.1200 | 12.860 | 1.0000 | 1.0000 | 0.0677 |
| OOD Test | 0.1833 | 0.0562 | 0.1573 | 10.124 | 1.0000 | 1.0000 | 0.0480 |

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