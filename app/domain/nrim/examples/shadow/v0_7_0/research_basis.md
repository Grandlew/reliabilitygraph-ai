# NRIM v0.7 research and engineering basis

## Truth boundary

NRIM v0.7 is an evidence system around the frozen v0.6 candidate. It is not a
new model release and it does not establish operational efficacy by itself.
Only a preregistered, prospective, non-interventional pilot on real IPTV
deployments can move the system from `CANDIDATE` to `SHADOW-VALIDATED`.

The release gate enforces this distinction in code: synthetic and retrospective
evidence origins cannot promote the model even when every numerical metric is
perfect.

## Primary research used

1. [NIST AI Risk Management Framework 1.0](https://www.nist.gov/itl/ai-risk-management-framework)
   motivates explicit governance, measurement, monitoring, independent review,
   documented knowledge limits, and fail-safe behavior throughout the
   lifecycle.
2. [NIST AI 800-4, Challenges to the Monitoring of Deployed AI Systems](https://www.nist.gov/publications/challenges-monitoring-deployed-ai-systems-center-ai-standards-and-innovation)
   distinguishes functionality, operational, human-factors, security,
   compliance, and impact monitoring. v0.7 records these as separate evidence
   classes instead of treating model accuracy as system reliability.
3. [Breck et al., The ML Test Score](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/)
   and
   [Rosen et al., Validating Data and Models in Continuous ML Pipelines](https://research.google/pubs/validating-data-and-models-in-continuous-ml-pipelines/)
   support treating schemas, datasets, models, serving behavior, and monitoring
   as linked production artifacts.
4. [Sculley et al., Hidden Technical Debt in Machine Learning Systems](https://research.google/pubs/hidden-technical-debt-in-machine-learning-systems/)
   motivates one shared online/replay feature path, explicit data dependencies,
   and the prohibition on configuration or threshold changes during the pilot.
5. [Corbin et al., DEPLOYR](https://arxiv.org/abs/2303.06269) reports that
   prospective silent-deployment performance can differ from retrospective
   estimates. This directly supports freezing v0.6 and measuring it silently
   before exposing its output to operators.
6. [Tonekaboni et al., Silent trial protocol](https://proceedings.mlr.press/v174/tonekaboni22a.html)
   supports real-time silent evaluation, systematic ground-truth collection,
   and user-centred evaluation before workflow integration.
7. [Apache Beam’s event-time and watermark model](https://beam.apache.org/documentation/basics/)
   informs the separation of event time, ingestion time, decision watermarks,
   and immutable late-data replay.
8. [OpenTelemetry semantic conventions](https://opentelemetry.io/docs/specs/semconv/general/)
   motivate structured, correlated operational events instead of unstructured
   log-only monitoring.
9. [NIST SP 800-122](https://csrc.nist.gov/pubs/sp/800/122/final) supports
   minimizing and protecting personally identifiable information. v0.7 uses
   domain-separated keyed HMAC pseudonyms and excludes direct identities from
   contracts and inference.

## Research-derived implementation decisions

### One serving path

`ServingFeatureBuilder` is the only feature implementation used for replay and
prospective inference. Golden snapshots and exact feature hashes detect
training-serving skew. Online-only and replay-only features are not permitted.

### Event time is not ingestion time

A prospective prediction includes only records whose event time is at or before
the cutoff and whose ingestion time is at or before the registered watermark.
Late observations remain stored but produce a separately identified replay
prediction. They never rewrite the original prospective prediction.

### Data failure is not health evidence

Contract failure, collector outage, insufficient freshness, and insufficient
coverage produce `DATA_QUALITY_ESCALATION`. Distribution shift produces
`UNKNOWN` or `ESCALATE`. Neither state is converted to `HEALTHY`, and neither
can expose Stage 2 ranking.

### Identity is retained for grouping, excluded from inference

Deployment pseudonyms are required for clustered uncertainty, drift
localization, and audit provenance. The shared feature builder never places
them in a model tensor. Profile software/collector versions are monitoring
dimensions, not inference features.

### Temporal state has a semantic epoch

The temporal sequence resets when topology version or operational-profile
version changes. Carrying a residual CUSUM across a changed graph or workload
contract would mix incompatible semantics. The transition itself remains
observable and is monitored.

### Negative and missed cases must be reviewed

Reviewing only NRIM alerts would create verification bias and could not estimate
recall or healthy-hour burden. The protocol therefore requires all known
incidents, NRIM-positive cases, a stratified negative sample, and random healthy
deployment-hours. Reviewers make an initial assessment before seeing NRIM where
operationally possible.

### Complexity must earn retention

Fast-only, slow-only, and dual policies run counterfactually on identical,
in-support snapshots. Their outputs are stored but remain hidden during the
silent phase. The preregistered simplification rule removes a path that does not
provide independent delay, recall, severe-rescue, or weak-impact value within
the burden budget.

## Statistical contract

- Incident recall uses incident episodes as units and reports an exact
  one-sided binomial bound plus a deployment-cluster bootstrap.
- Alert burden is episodes per 100 healthy deployment-hours. Its report includes
  an exact Poisson-model upper bound and a deployment-cluster bootstrap; the
  conservative reported upper bound is the larger value.
- Zero observed false episodes therefore never becomes a claim of zero risk.
- The Poisson bound is conditional on a Poisson event-count assumption. The
  cluster bootstrap preserves whole-deployment dependence but needs enough
  independent deployments. Neither calculation is a universal guarantee.
- Overlapping windows are never counted as independent trials.
- Cohen's kappa is reported only when double-review sampling is adequate. An
  unresolved label remains a legitimate outcome.

## Security and operational limitations

- The signed bundle uses Ed25519 and verifies every artifact and registered
  inference-source file at startup. Its public-key fingerprint must be
  committed outside the bundle; a public key shipped only inside its own bundle
  would not be a trust anchor.
- The local append-only SQLite repository is a deterministic pilot/test adapter.
  A real pilot should use PostgreSQL/object storage with organizational IAM,
  encrypted backups, retention policy, multi-process transaction testing, and
  independent audit export.
- Pseudonymization is not anonymization. The HMAC key must remain in an approved
  secret manager, linkage access must be separated, and the deployment privacy
  review remains an organizational gate.
- The minimal bearer-token review API is an integration seam, not a replacement
  for the organization's identity provider, SSO, role lifecycle, or security
  assessment.
- No real deployment, historical replay, engineer usability study, or
  prospective incident has been supplied in this workspace. Those gates remain
  unpassed by design.
