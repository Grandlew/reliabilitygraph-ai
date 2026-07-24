# Evaluation Result: learned_fusion

**Benchmark Fingerprint:** `7ad72ddd27c6b38d5449859d226fdbd14fae04f87b7d4f33468bfa9f9700ae15`
**Abstention Threshold:** `0.8392380634160074`
**Incident Threshold:** `0.5963176417056761`

## Metrics

| Split | MRR | Hits@1 | Hits@3 | Mean Rank | Healthy False-Selection | Faulty Coverage | Incident Precision | Incident Recall | Mean Runtime (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Validation | 0.4636 | 0.3239 | 0.5634 | 2.137 | 0.3212 | 0.7183 | 0.5368 | 0.7183 | 4.5913 |
| Test | 0.5533 | 0.3200 | 0.7733 | 1.906 | 0.5940 | 0.8533 | 0.4430 | 0.8800 | 6.8753 |
| OOD Test | 0.5220 | 0.4717 | 0.5660 | 1.274 | 0.2474 | 0.5849 | 0.5688 | 0.5849 | 6.7318 |

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
  "random_seed": 42,
  "topology_candidate_validation_mrr": 0.47417840375586856,
  "topology_incremental_validation_mrr": 0.010563380281690182,
  "topology_retained": false,
  "topology_retention_reason": "Five-seed clustered test/OOD ablation did not satisfy the registered repeatability rule."
}
```

> **Note on Synthetic Data:** These metrics are derived from simulated IPTV environments. 
> Performance on real-world production systems may vary based on topology complexity and noise profile.