# NRIM v0.7.1 OELR research and engineering basis

## Decision

The highest-value next step is not another model component. It is an
independently verifiable evidence system that can determine whether the frozen
candidate is safe, semantically correct, operationally reproducible, and useful
on real deployments. The implementation therefore strengthens measurement,
provenance, review design, and staged release without claiming performance that
has not been observed.

## Primary sources

1. [NIST AI 800-3, A Statistical Framework for Evaluating AI](https://www.nist.gov/publications/expanding-ai-evaluation-toolbox-statistical-models)
   motivates explicit measurement goals, estimands, sampling assumptions, and
   uncertainty rather than a collection of context-free point metrics.
2. [NIST AI 800-4, Challenges to the Monitoring of Deployed AI Systems](https://www.nist.gov/publications/challenges-monitoring-deployed-ai-systems-center-ai-standards-and-innovation)
   supports separating functionality, operations, human factors, security,
   compliance, and impact evidence. OELR preserves those as distinct gates.
3. [NIST SP 800-92, Guide to Computer Security Log Management](https://csrc.nist.gov/pubs/sp/800/92/final)
   supports durable collection, protection, review, and analysis of operational
   evidence. OELR adds chained records and external checkpoint receipts.
4. [SLSA v1.2 artifact verification](https://slsa.dev/spec/v1.2/verifying-artifacts)
   supports verification against an expected identity and trust root rather
   than trusting metadata merely because it travels beside the artifact.
5. [Tonekaboni et al., A Silent Trial Framework for AI](https://proceedings.mlr.press/v174/tonekaboni22a.html)
   supports real-time silent evaluation, systematic ground-truth collection,
   technical-readiness testing, and later user-centred evaluation. OELR keeps
   human utility out of the hidden-output phase.
6. [OpenTelemetry event semantic conventions](https://opentelemetry.io/docs/specs/semconv/general/events/)
   support structured events with stable semantic meaning instead of
   unstructured text as the evidence interface.
7. [NIST exact binomial interval reference](https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/propconf.htm)
   documents exact binomial interval construction. OELR uses a one-sided exact
   lower bound for recall and records the confidence direction.
8. [Breck et al., The ML Test Score](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/)
   supports testing data, features, models, serving infrastructure, and
   monitoring as a connected production system.
9. [Sculley et al., Hidden Technical Debt in Machine Learning Systems](https://research.google/pubs/hidden-technical-debt-in-machine-learning-systems/)
   motivates explicit data dependencies, controlled configuration, and
   separation between model behavior and surrounding system behavior.

## Research-derived controls

### Estimands before dashboards

Every registered measure names its target population, primary unit,
denominator, inclusion rule, adjudication state, evidence phase, threshold,
uncertainty method, confidence direction, and assumptions. The compiler rejects
incomplete confirmatory estimands and derives feasibility facts before data are
examined.

### Exact bounds, deployment-aware sensitivity

Recall is an incident-level quantity and uses exact one-sided binomial bounds.
False-alert burden is an exposure rate and uses an exact one-sided Poisson-model
bound. The Poisson assumption is stated, never hidden. Deployment intervals are
reported individually; clustered or random-effects analyses are sensitivity
analyses and cannot erase low exposure or limited deployment count.

### Review design independent of model output

All known incidents and all NRIM-positive episodes are included. Healthy
records use a signed, stratified, deterministic sample with recorded inclusion
probabilities. This prevents an alert-only review loop from estimating recall
or burden. Unresolved labels remain unresolved rather than being forced into a
favorable binary label.

### Semantic acceptance, not schema-only acceptance

A syntactically valid number can still carry the wrong unit, time meaning,
quality meaning, applicability state, aggregation, or topology direction.
Collector and topology harnesses therefore require explicit semantic contracts
and mutation tests. A mapping passes only when intended fixtures match and
meaning-changing mutations fail closed.

### Independent trust is an infrastructure property

The application signs its current evidence-chain state. A separately keyed
custodian verifies it and issues a chained receipt. The included filesystem
sink proves protocol behavior only and labels itself a test adapter. It cannot
claim independent administration, retention lock, or organizational separation.

### Silent, advisory, and operational evidence are different

Silent predictions stay hidden, so no human-utility claim is possible.
Advisory validation requires a preregistered paired, randomized, or
rollout-aware workflow comparison and automation-bias measurement.
Operational candidacy additionally requires security, resilience, privacy,
change-management, deployment-owner, and independent-review approvals.

## Knowledge limits

- Unit tests and synthetic fixtures do not validate real source semantics.
- Historical replay can establish readiness and find defects but cannot replace
  prospective evidence.
- Five deployments make a pilot possible but do not guarantee transportability
  to a broader deployment population.
- Exact intervals quantify sampling uncertainty under their stated models; they
  do not correct label bias, missing incidents, source drift, or protocol
  deviations.
- Cryptographic integrity cannot make an administrator independent. That
  property requires separate infrastructure, credentials, custody, and policy.
- No evidence in this workspace justifies the labels “world-class,”
  “production-ready,” or “operationally effective.”

