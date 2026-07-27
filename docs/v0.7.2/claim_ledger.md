# ReliabilityGraph AI v0.7.2 claim ledger

Baseline commit: `98a7aa0eab267a3828d2b32797afb5ae7e3c24c3`  
Starting test baseline: 297 passing tests  
Candidate state: `CANDIDATE`

## Frozen identities

- Bundle SHA-256: `9767bc9f3184a9e68fe99e0a31c5593a610c9dbd3b524dfa155c8bb53091ecf7`
- Bundle manifest SHA-256: `07bfab9d935e205e9372121622d674f03ab51e81c65f0cd4e063387b7baa4fb9`
- Feature schema SHA-256: `b2cb5de814d319934e80f4eb9d474ac2831fed9b8db18077cc61d56185c6e9e7`
- Policy SHA-256: `f78e0bb65a39364b53cf5888d24455158090d056a6fd31a90df116f7fb85c036`
- Prediction-envelope schema SHA-256: `a6d6a276bf89a76a8d4418664462583aeb3a7d060ad589baa817abaf8953f75c`

The v0.7.2 layer wraps these identities. It does not modify or reinterpret
their probabilities, ranks, thresholds, labels, support rules, temporal
policies, activation paths, or conservative safety states.

## Authorized software scope

- Immutable decision-event contracts and hash chains.
- Explicit candidate-set and observational causal-evidence records.
- Append-only persistence in the existing SQLite test adapter.
- Original, as-known-at, latest, and comparison projections.
- Lossless v0.6 compatibility wrapping.
- Deterministic synthetic golden replay and schema export.
- Read-only GET retrieval, export, and resumable summary streaming.

## Forbidden capabilities

- Operational telemetry queries or active diagnostics.
- Ticket, notification, configuration, suppression, restart, remediation, or
  customer-facing mutations.
- Automated action authority or new credentials/connectors.
- Model, threshold, policy, feature, label, support, temporal, or ranking
  changes.
- GNNs, graph databases, message brokers, Kubernetes, MPC, or LLM decision
  authority.

## Non-claims

All checked-in v0.7.2 fixtures are synthetic software evidence only. This slice
does not establish real IPTV accuracy, calibration, alert burden, causal
validity, human utility, production efficacy, outage prevention, remediation
safety, ISP transfer, cybersecurity readiness, or cross-sector generality.
