# Evaluation Result: max_anomaly

**Benchmark Fingerprint:** `7ad72ddd27c6b38d5449859d226fdbd14fae04f87b7d4f33468bfa9f9700ae15`
**Abstention Threshold:** `0.42515790458365754`
**Incident Threshold:** `0.4890231244701022`

## Metrics

| Split | MRR | Hits@1 | Hits@3 | Mean Rank | Healthy False-Selection | Faulty Coverage | Incident Precision | Incident Recall | Mean Runtime (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Validation | 0.4296 | 0.3239 | 0.5493 | 1.860 | 0.3650 | 0.6056 | 0.4624 | 0.6056 | 0.7852 |
| Test | 0.3822 | 0.2400 | 0.5333 | 2.109 | 0.5714 | 0.6133 | 0.3770 | 0.6133 | 1.2824 |
| OOD Test | 0.3365 | 0.3019 | 0.3774 | 1.636 | 0.3316 | 0.4151 | 0.4112 | 0.4151 | 1.0753 |

## Configuration
```json
{
  "abstention_threshold_feasible": true,
  "incident_threshold": 0.4890231244701022,
  "incident_threshold_feasible": false,
  "incident_validation_coverage": 0.6056338028169014,
  "incident_validation_false_selection_rate": 0.36496350364963503,
  "maximum_healthy_false_selection_rate": 0.1,
  "minimum_faulty_coverage": 0.7,
  "random_seed": 42
}
```

> **Note on Synthetic Data:** These metrics are derived from simulated IPTV environments. 
> Performance on real-world production systems may vary based on topology complexity and noise profile.