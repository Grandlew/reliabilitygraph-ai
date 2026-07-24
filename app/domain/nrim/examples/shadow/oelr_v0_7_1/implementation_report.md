# NRIM v0.7.1 OELR implementation and truth report

Date: 2026-07-24

## Outcome

The repository now implements the Operational Evidence and Learning Readiness
layer recommended for NRIM v0.7. This is a versioned evidence, acceptance,
governance, and promotion boundary around the frozen v0.6 inference candidate.
It does not train a new model, alter a threshold, authorize automatic
remediation, or establish real-world efficacy.

The candidate remains `CANDIDATE`. The five operational readiness gates remain
closed because no approved real replay, real collector/topology acceptance,
independently administered pilot storage, seven-day real dress rehearsal, or
eight-week silent pilot evidence was supplied.

The active inference bundle was not re-signed or modified:

- manifest SHA-256:
  `07bfab9d935e205e9372121622d674f03ab51e81c65f0cd4e063387b7baa4fb9`
- bundle SHA-256:
  `9767bc9f3184a9e68fe99e0a31c5593a610c9dbd3b524dfa155c8bb53091ecf7`
- policy SHA-256:
  `f78e0bb65a39364b53cf5888d24455158090d056a6fd31a90df116f7fb85c036`
- feature schema SHA-256:
  `b2cb5de814d319934e80f4eb9d474ac2831fed9b8db18077cc61d56185c6e9e7`

## Implemented recommendations

| Area | Implemented software control | Truth status |
|---|---|---|
| Evidence states | Enforced `CANDIDATE` → `REAL-REPLAY-READY` → `SHADOW-VALIDATED` → `ADVISORY-VALIDATED` → `OPERATIONAL-CANDIDATE`, with phase-specific predecessors and evidence origins. | Implemented; no transition claimed. |
| Statistical compiler | Strict signed protocol, explicit estimands, population/unit/denominator/inclusion/adjudication/phase/uncertainty, evidence floors, derived feasibility constraints, stop rules, and separation of duties. | Implemented; no protocol registered without real owners and dates. |
| Collector semantics | Unit conversion provenance, event/ingestion time, skew/delay budgets, quality/applicability mappings, aggregation semantics, pseudonymization, strict frozen-contract output, and semantic negative controls. | Harness implemented; real collectors unvalidated. |
| Topology lineage | Endpoint-role rules, direction/dependency semantics, validity provenance, deterministic cutoff reconstruction, snapshot hashes, and reversed/stale/semantic mutation controls. | Harness implemented; real inventory history unvalidated. |
| Independent evidence trust | Application-signed chain checkpoints, separately keyed custodian receipts, predecessor-chain verification, exclusive creation, and store-to-checkpoint verification. | Interface and test adapter implemented; independent infrastructure absent. |
| Pilot storage | Qualification contract covers concurrent writes, conflict behavior, atomicity, retention lock, deletion denial, encrypted backup/restore, chain parity, access control, and immutable audit export. | SQLite is explicitly ineligible; pilot infrastructure absent. |
| Review sampling | Signed seed commitment and frame, all known incidents, all NRIM positives, stratified healthy sampling, recorded inclusion probabilities, and inverse-probability estimates. | Implemented; no real review frame supplied. |
| Adjudication | Sampling assignments are append-only review events; unresolved labels and inclusion probabilities remain observable; free-text payloads are privacy scanned. | Implemented; usability and independent review remain external. |
| Statistical evidence | Exact one-sided binomial recall bounds, exact Poisson burden bounds, per-deployment intervals and exposure rules, deployment-cluster ranking uncertainty, limited random-effects sensitivity, and phase-appropriate agreement reporting. | Implemented; no prospective estimates claimed. |
| Historical replay | Gate requires real data, preregistration, semantic acceptance, label reporting, no leakage, deterministic replay, topology reconstruction, complete lineage/gap reporting, and zero tuning. | Implemented; not run. |
| Dress rehearsal | Seven consecutive real days, hidden output, independently attested read-only operation, evidence continuity, external checkpoints, restart parity, availability, support safety, and stop conditions. | Implemented; not run. |
| Promotion | Silent evaluation prohibits human-utility claims; advisory requires registered human evaluation; operational candidacy adds security, resilience, privacy, change, deployment-owner, and independent approvals. | Implemented; all real-evidence states closed. |
| Test provenance | Full JUnit evidence records exact node IDs, nested test-scope counts, environment versions, commit, requirements, source-state hash, and the limitation of software tests. | Generated after the final full-suite run. |

## Statistical feasibility made explicit

At the registered floor of 3,000 healthy deployment-hours, the exact
one-sided 95% Poisson upper-bound gate of 0.20 episodes per 100 hours permits
at most one false episode. At 30 adjudicated incidents, at least 28 detections
are needed for the exact one-sided 95% recall lower bound to reach 0.80. A
confirmatory severe-recall lower bound of 0.80 with a 100% observed point
estimate requires at least 14 severe incidents.

Five deployments are treated as a feasibility floor, not sufficient evidence
for a stable population-level heterogeneity claim. The evaluator reports exact
per-deployment intervals and a registered minimum-exposure rule; random-effects
sensitivity is unavailable below ten deployments.

## Verification

- OELR schemas exported: 13.
- Repository tests collected: 288.
- Final pass/fail counts and exact test identities are recorded in
  `test-results/test_evidence_manifest.json`.
- The signed v0.7 bundle is verified separately after the suite.
- Synthetic/local tests establish software behavior only. They do not establish
  collector correctness, population generalization, operational safety, human
  utility, or production efficacy.

## Deliberately deferred

The recommendation to acquire substantially larger v0.8 evidence is retained
as a future acquisition target, not silently converted into authorization to
train. A v0.8 modeling cycle should begin only after the registered real pilot
has enough incidents, deployments, failure families, and healthy exposure to
support the intended estimands. GNN training, threshold retuning, and simulator
expansion are outside this implementation.

## Required next execution

1. Assign distinct named owners and approve privacy, security, storage, review,
   deployment, and stop-condition responsibilities.
2. Instantiate and pass the collector and topology contracts against real
   source systems, including every negative control.
3. Qualify independently administered retention-locked evidence storage and
   an external checkpoint custodian.
4. Register and sign the real historical-replay protocol before inspecting
   replay outcomes; run it without tuning.
5. Pass the seven-day real dress rehearsal.
6. Register the eight-week silent protocol before outcome access, keep output
   hidden, review the signed sample, and compile evidence at the scheduled
   cutoff.
7. Move to advisory evaluation only after silent validation; keep automatic
   remediation disabled.
