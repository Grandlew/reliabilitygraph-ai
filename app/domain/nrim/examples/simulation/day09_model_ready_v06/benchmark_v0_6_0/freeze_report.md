# NRIM v0.6 Dual-Path Episode Gate

- Status: **CANDIDATE**
- Locked test: **OPENED_ONCE**
- Stage 2 frozen checkpoint: `b6cd76db2489b09f3fe16c712a7ca37092d8a049b3f699f37439e24213f0f9ff`

## Validation

- Healthy scenario risk / 95% topology UCB: 0.0000 / 0.0487
- Episode recall: 1.0000
- Median / p90 delay: 0.0 / 0.0 hours

## Development test

- Healthy scenario risk: 0.0250
- Episode recall: 1.0000

## Locked test

- Healthy scenario risk: 0.0083
- Episode recall: 1.0000

## Mandatory validation decision

- PASS — scenario_false_selection_at_most_0_10
- PASS — global_95_ucb_at_most_0_10
- PASS — worst_confounder_point_at_most_0_10
- PASS — worst_confounder_ucb_at_most_0_15
- PASS — episode_recall_at_least_0_80
- PASS — each_family_recall_at_least_0_70
- PASS — median_delay_at_most_2h
- PASS — p90_delay_at_most_4h
- PASS — capacity_median_delay_at_most_4h
- PASS — fragmentation_at_most_0_05
- PASS — unsupported_safe_semantics

## Architectural attribution

- FAIL — fast_short_recall_increment_at_least_0_15
- PASS — fast_risk_increment_at_most_0_02
- PASS — slow_false_episode_reduction_at_least_50pct
- PASS — slow_persistent_recall_at_least_0_75
- PASS — impact_confirmation_is_incremental
- PASS — residual_not_worse_at_matched_recall
- PASS — registered_ood_detection_at_least_0_95
- PASS — id_support_false_alarm_at_most_0_05

The release remains a candidate because the architectural increment is not fully identified, even though mandatory validation, development, and locked-test operating criteria passed. Persistence-only already achieved 1.0 recall for 1–2-window incidents, leaving no headroom to demonstrate the preregistered +0.15 fast-path increment.

## Interpretation

The slow path is an empirically calibrated CUSUM over healthy-conditioned residuals. The report makes no distribution-free false-alarm claim for overlapping windows; risk is measured on independent topology groups with a one-sided exact binomial upper bound.
The fast path cannot open on classifier probability alone: it requires independent observable impact breadth and causal consistency. Unsupported inputs are UNKNOWN or ESCALATE and never enter Stage 2.

## Limitations

- Synthetic evidence is not production evidence.
- The system remains a candidate until prospective shadow-mode evaluation on real operations data.
