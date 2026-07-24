# NRIM Scenario-Clustered Temporal Gate Freeze Report

- Benchmark version: `0.5.0`
- Release status: `candidate`
- Dataset fingerprint: `bc0eca8e18e9befa17e11c806dbe47a71cf47e80cf712d2b9c690368a7e8daf5`
- Reference-model checkpoint: `3909770769f03c1bd62fb9a7e9d6f6cc7640861482566d5dcec1ed0526ac6457`
- Decision metric unit: `scenario`
- Independent risk/bootstrap unit: `topology group`
- Window overlap treated as dependent: `true`

## Frozen policy

```json
{
  "window_threshold": 0.6464604789452578,
  "evidence_center": 0.5171683831562063,
  "decay": 0.5,
  "entry_threshold": 0.5,
  "exit_threshold": 0.1,
  "impact_weight": 0.5,
  "recovery_weight": 0.5,
  "minimum_suspect_windows": 3,
  "minimum_incident_windows": 1,
  "minimum_recovery_windows": 2,
  "maximum_suspect_windows": 4,
  "cooldown_windows": 1,
  "severity_override_probability": 0.95,
  "unsupported_escalation_probability": 0.8
}
```

## Scenario-level results

| Split | Selection / escalation | Risk UCB | Episode recall | Median delay (h) | Fragmentation | Unsupported semantics |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 0.1875 | 0.9024 | 0.8000 | 2.0 | 0.0000 | 1.0000 |
| test | 0.2500 | 1.0000 | 0.6250 | 0.0 | 0.0000 | 1.0000 |
| ood_test | 1.0000 | 1.0000 | 1.0000 | 0.0 | 0.2143 | 1.0000 |

## Shift-support results

- OOD scenarios marked unsupported: `1.0000`
- In-distribution validation scenarios marked unsupported: `0.0000`
- Unsupported OOD window decisions: `UNKNOWN=0.7090`, `ESCALATE=0.2910`

## Registered acceptance checks

- reference_reproduction: `PASS` - threshold=0.646460478945, reference=0.646460478945
- validation_global_risk_ucb: `FAIL` - UCB=0.9024 <= 0.10
- validation_confounder_risk_ucb: `FAIL` - worst UCB=0.9747 <= 0.15
- validation_episode_recall: `PASS` - recall=0.8000 >= 0.70
- test_point_risk: `FAIL` - risk=0.2500 <= 0.12
- test_episode_recall: `FAIL` - recall=0.6250 >= 0.70
- ood_safe_semantics: `PASS` - all unsupported decisions are UNKNOWN or ESCALATE
- temporal_stability: `PASS` - fewer healthy false episodes, no added fragmentation, and median delay <= 3 hours
- stage2_preserved: `PASS` - learned-fusion weights frozen in checkpoint
- split_and_shortcut_audits: `PASS` - sequence_leakage_free=True, prior_findings_clean=True

## Statistical interpretation

The selected point is `not feasible` under the registered finite-sample bounds. A zero observed false-episode count is not reported as zero risk; the exact one-sided binomial upper bound is used.

## OOD semantics

Unsupported low-score observations become `UNKNOWN`; unsupported high-score observations become `ESCALATE`. Neither is silently emitted as `HEALTHY`, and Stage 2 output on unsupported inputs is explicitly uncalibrated evidence.

## Stage 2

Learned-fusion ranking is frozen in the reference checkpoint. The temporal gate does not refit or alter Stage 2 scores.

## Limitations

- The benchmark is entirely synthetic.
- Exact group-risk bounds can remain inconclusive when a confounder cohort has few independent scenarios.
- The shift sentinel detects support mismatch, not semantic novelty or guaranteed model failure.
- Synthetic benchmark performance does not establish production performance.
