# Evaluation Result: hybrid_engineering

**Benchmark Fingerprint:** `7ad72ddd27c6b38d5449859d226fdbd14fae04f87b7d4f33468bfa9f9700ae15`
**Abstention Threshold:** `0.4155114090168612`
**Incident Threshold:** `0.5963176417056761`

## Metrics

| Split | MRR | Hits@1 | Hits@3 | Mean Rank | Healthy False-Selection | Faulty Coverage | Incident Precision | Incident Recall | Mean Runtime (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Validation | 0.3560 | 0.2535 | 0.3662 | 4.706 | 0.3139 | 0.7183 | 0.5368 | 0.7183 | 4.7447 |
| Test | 0.3890 | 0.3067 | 0.3467 | 5.639 | 0.6241 | 0.8133 | 0.4430 | 0.8800 | 6.4708 |
| OOD Test | 0.3816 | 0.3491 | 0.3585 | 3.121 | 0.2263 | 0.5472 | 0.5688 | 0.5849 | 5.4696 |

## Configuration
```json
{
  "abstention_threshold_feasible": true,
  "incident_feature_count": 76,
  "incident_model": "standard_library_logistic",
  "incident_threshold": 0.5963176417056761,
  "incident_threshold_feasible": false,
  "incident_validation_coverage": 0.7183098591549296,
  "incident_validation_false_selection_rate": 0.32116788321167883,
  "maximum_healthy_false_selection_rate": 0.1,
  "minimum_faulty_coverage": 0.7,
  "ood_distance_threshold": 2.2218611421379615,
  "ood_escalation_enabled": false,
  "propagation_diagnostics": {
    "mean_pairwise_inversions": 0.19951832434157096,
    "mean_root_rank_change": -0.08653846153846154,
    "mean_topology_increment": 0.007699774576349019,
    "score_saturation_rate": 0.0,
    "top1_change_rate": 0.0,
    "window_count": 208
  },
  "random_seed": 42
}
```

> **Note on Synthetic Data:** These metrics are derived from simulated IPTV environments. 
> Performance on real-world production systems may vary based on topology complexity and noise profile.