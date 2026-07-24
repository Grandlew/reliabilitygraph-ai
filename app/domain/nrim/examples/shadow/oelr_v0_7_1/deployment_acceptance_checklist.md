# NRIM v0.7.1 deployment acceptance checklist

This checklist must be completed with signed artifacts. Checking a box in a
document is not evidence unless the referenced artifact exists and its hash is
registered.

## Gate A — contracts and privacy

- [ ] Five or more deployment pseudonyms and topology families approved.
- [ ] Collector contract signed for every deployment.
- [ ] All required canonical metrics mapped with unit and aggregation lineage.
- [ ] Event/ingestion clocks, precision, skew, delay, quality, and applicability
      semantics validated.
- [ ] Positive fixtures match 100%; semantic mutations fail closed 100%.
- [ ] Topology endpoint roles, edge direction, validity, and change cutoffs
      registered.
- [ ] Topology reconstruction is deterministic at every test cutoff; reversed,
      stale, and meaning-mutated controls fail closed.
- [ ] Privacy owner approves fields, pseudonymization, retention, linkage, and
      secret custody; direct-identifier audit is clean.

## Gate B — evidence workflow

- [ ] Distinct protocol, evidence, review, release, operations, privacy, and
      security owners assigned.
- [ ] External custodian has a separate key, credentials, administrator, and
      trust domain.
- [ ] Checkpoint/receipt chain round trip and tamper tests pass.
- [ ] Pilot storage passes concurrent append, conflict, atomicity, retention,
      deletion denial, backup/restore, chain parity, access, and immutable
      export tests.
- [ ] Blinded-first review workflow passes an engineer usability exercise.
- [ ] Signed sampling plan covers all known incidents, all NRIM positives, and
      stratified healthy negatives with inclusion probabilities.

## Gate C — real historical readiness

- [ ] Replay plan signed before outcome inspection.
- [ ] Approved real export, incident inventory, label availability inventory,
      and topology history supplied.
- [ ] Required feature reconstruction ≥95%; incident alignment ≥90%; root-cause
      mapping ≥95%.
- [ ] Future leakage is zero; replay determinism and topology reconstruction are
      100%.
- [ ] Every semantic deviation has versioned lineage and the data-gap report is
      complete.
- [ ] Model, thresholds, features, support rules, watermarks, and episode
      grouping were not tuned.
- [ ] Replay is reported as readiness evidence, not promotion evidence.

## Gate D — seven-day dress rehearsal

- [ ] Read-only boundary independently attested; no operational write
      credential or reachable mutation endpoint exists.
- [ ] Seven consecutive days of real, hidden-output execution completed.
- [ ] Schema ≥99%; invalid-category quarantine attribution 100%.
- [ ] Envelope reproduction, evidence completeness/continuity, external
      checkpoints, and restart parity are 100%.
- [ ] Registered availability floor met; unsafe Stage 2 outputs and emergency
      stops are zero.
- [ ] Dress rehearsal is reported as operational readiness, not efficacy.

## Gate E — eight-week silent pilot

- [ ] Protocol compiled, signed, and externally anchored before outcomes.
- [ ] At least five deployments/families, 3,000 healthy deployment-hours,
      30 adjudicated incidents, 14 severe incidents, and eight weeks reached.
- [ ] Predictions remained hidden; automatic remediation remained disabled.
- [ ] Every incident, positive episode, sampled negative, unresolved label, and
      inclusion probability is accounted for.
- [ ] Exact recall and burden bounds, per-deployment intervals, heterogeneity,
      ranking coverage, support safety, and path attribution compiled as
      registered.
- [ ] Deviations, stop conditions, exclusions, missingness, and denominators are
      reported without post-hoc tuning.
- [ ] Independent review and release authority sign the transition decision.

## Gate F — advisory and operational candidacy

- [ ] Silent validation passed before operators see output.
- [ ] Human evaluation design preregistered and paired, randomized, or
      rollout-aware.
- [ ] Time, investigation scope, usefulness, overrides, and automation bias
      reported.
- [ ] Security, resilience/recovery, privacy, change-management, deployment
      owner, and independent-review approvals obtained.
- [ ] Automatic remediation remains a separate, unauthorized future decision.

