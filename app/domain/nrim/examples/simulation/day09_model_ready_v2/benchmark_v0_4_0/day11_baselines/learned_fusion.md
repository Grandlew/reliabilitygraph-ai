# Evaluation Result: learned_fusion

**Benchmark Fingerprint:** `bc0eca8e18e9befa17e11c806dbe47a71cf47e80cf712d2b9c690368a7e8daf5`
**Abstention Threshold:** `0.8650730344537793`
**Incident Threshold:** `0.6464604789452578`

## Metrics

| Split | MRR | Hits@1 | Hits@3 | Mean Rank | Healthy False-Selection | Faulty Coverage | Incident Precision | Incident Recall | Mean Runtime (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Validation | 0.5583 | 0.4500 | 0.7000 | 1.500 | 0.0824 | 0.7000 | 0.4590 | 0.7000 | 3.5473 |
| Test | 0.5444 | 0.4000 | 0.7333 | 1.636 | 0.1010 | 0.7333 | 0.3485 | 0.7667 | 4.3181 |
| OOD Test | 0.7487 | 0.6190 | 0.8889 | 1.339 | 0.3127 | 0.8889 | 0.3164 | 0.8889 | 4.1417 |

## Configuration
```json
{
  "abstention_threshold_feasible": true,
  "confounder_topology_balancing": true,
  "counterfactual_pairwise_training": true,
  "incident_feature_count": 111,
  "incident_model": "standard_library_logistic",
  "incident_threshold": 0.6464604789452578,
  "incident_threshold_feasible": false,
  "incident_validation_cohort_false_selection_rates": {
    "harmless_worker_restart+retention_increase": 0.024390243902439025,
    "high_healthy_workload": 0.15217391304347827,
    "high_healthy_workload+transient_recording_errors": 0.15384615384615385,
    "none": 0.038461538461538464,
    "retention_increase": 0.125,
    "temporary_latency_spike": 0.0,
    "temporary_latency_spike+transient_recording_errors": 0.12195121951219512,
    "transient_recording_errors": 0.0851063829787234
  },
  "incident_validation_coverage": 0.7,
  "incident_validation_false_selection_rate": 0.08776595744680851,
  "incident_validation_maximum_cohort_false_selection_rate": 0.15384615384615385,
  "maximum_healthy_false_selection_rate": 0.1,
  "minimum_faulty_coverage": 0.7,
  "ood_distance_threshold": 2.2984062038172217,
  "ood_escalation_enabled": false,
  "propagation_diagnostics": {
    "mean_pairwise_inversions": 0.19160578006286128,
    "mean_root_rank_change": -0.0625,
    "mean_topology_increment": 0.007118534041094827,
    "score_saturation_rate": 0.0,
    "top1_change_rate": 0.0,
    "window_count": 416
  },
  "random_seed": 42,
  "topology_ablation_validation_mrr": 0.42916666666666664,
  "topology_incremental_validation_mrr": 0.1291666666666667,
  "topology_retained": true,
  "topology_retention_reason": "Five-seed clustered test ablation satisfied the registered non-negative confidence rule."
}
```

> **Note on Synthetic Data:** These metrics are derived from simulated IPTV environments. 
> Performance on real-world production systems may vary based on topology complexity and noise profile.