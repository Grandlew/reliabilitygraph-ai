# NRIM v0.7 implementation and truth report

Date: 2026-07-24

## Outcome

The repository now contains the software boundary required to run the frozen
v0.6 candidate as a read-only, prospective shadow evidence system. It does not
establish real-world efficacy and it is not a production release. Model,
feature, policy, and threshold tuning are prohibited during v0.7.

The active signed-bundle commitment is:

- bundle manifest SHA-256:
  `07bfab9d935e205e9372121622d674f03ab51e81c65f0cd4e063387b7baa4fb9`
- bundle file SHA-256:
  `9767bc9f3184a9e68fe99e0a31c5593a610c9dbd3b524dfa155c8bb53091ecf7`
- frozen policy SHA-256:
  `f78e0bb65a39364b53cf5888d24455158090d056a6fd31a90df116f7fb85c036`
- feature schema SHA-256:
  `b2cb5de814d319934e80f4eb9d474ac2831fed9b8db18077cc61d56185c6e9e7`

The original signed bundle is retained as a superseded lineage artifact. The
active bundle reuses the verified model and policy bytes without refitting. It
adds observation quality to the strict telemetry contract because the frozen
model's `low_quality_fraction` features require it. Its provenance commits the
parent manifest, corrected source paths, and the fact that the locked test was
not read during reissue.

## Verification completed

- Active Ed25519 signature, artifact hashes, trust anchor, source hashes, and
  provenance verified.
- All five frozen golden model snapshots passed.
- Five development-only, topology-distinct, end-to-end serving feature cases
  reproduced the v0.6 exported tensors exactly: 5/5, parity 1.0.
- The parity gate excluded the locked test split.
- Shadow tests passed: 30/30.
- Existing NRIM tests passed: 260/260.
- Full repository tests passed: 277/277, with two pre-existing Pydantic
  serialization warnings outside the v0.7 shadow package.

Synthetic development parity proves code-path equality only. It does not prove
that real collectors provide the same semantics, timestamps, quality values,
topology, or availability.

## Backlog status

| # | Blueprint task | Truthful status |
|---:|---|---|
| 1 | Freeze and package v0.6 | Complete: signed bundle, external-key fingerprint, golden startup tests, and mutation refusal. |
| 2 | Register v0.7 protocol | Tooling/template complete; deliberately not registered without approved dates, deployments, owners, reviewers, and an externally stored trust anchor. |
| 3 | Define telemetry contract | Implemented and locally validated, including units, applicability, quality, event/ingestion time, and fail-closed categories. Real collector mapping remains unvalidated. |
| 4 | Define topology contract | Implemented and locally validated with versioned validity intervals, directions, profiles, and snapshot hashes. Real inventory mapping remains unvalidated. |
| 5 | Build privacy boundary | Implemented with domain-separated HMAC pseudonyms and leakage audits. Organizational privacy approval and secret-manager integration remain external. |
| 6 | Build append-only telemetry mirror | Implemented as a read-only ingestion boundary and append-only SQLite pilot/test adapter. Production storage/IAM/backup integration remains external. |
| 7 | Implement event-time watermarking | Implemented; prospective evidence is immutable and late data produces a separate replay. |
| 8 | Implement validation and quarantine | Implemented with safe redaction and data-quality escalation that suppresses inference. |
| 9 | Build deterministic replay engine | Implemented and covered by deterministic replay/restart tests. |
| 10 | Implement serving feature parity | Implemented with one online/replay builder; 5/5 development cases match exactly. Real parity remains unmeasured. |
| 11 | Implement hash-locked inference loader | Complete for the active candidate; signature, artifact, source, schema, policy, and golden checks fail closed. |
| 12 | Build shadow inference orchestrator | Implemented read-only for Stage 1, residual, support, F/S/D paths, safe states, and Stage 2 suppression. |
| 13 | Build episode state store | Implemented with append-only transitions, semantic-epoch resets, idempotency, and restart equivalence. |
| 14 | Build immutable evidence writer | Implemented transactionally with input hashes, evidence hashes, and a prediction hash chain. Production durability remains unvalidated. |
| 15 | Build support and drift monitor | Implemented with runtime, contract, support, feature, topology, and unsafe-output monitoring. Alert integration remains external. |
| 16 | Implement adjudication schema and API | Implemented with authenticated role seams, blinded-first/versioned review, reveal control, reviewer provenance, and preserved disagreements. Identity-provider integration remains external. |
| 17 | Build minimal review console | Same-origin console and backend workflow implemented with blinded-first controls and evidence drill-down; the required engineer usability test is not complete. |
| 18 | Build prospective metrics engine | Implemented for episode/deployment units, clustered bootstrap, exact bounds, burden, ranking, support, path attribution, agreement, and utility. |
| 19 | Run historical replay readiness gate | Not run: no approved real historical export, incident mapping, or topology history was supplied. Synthetic data cannot satisfy this gate. |
| 20 | Run silent pilot and release gate | Not started: it requires preregistration plus at least 8 weeks, 5 deployments/topology families, 3,000 healthy deployment-hours, and 30 adjudicated incidents. |

## Promotion boundary

The release evaluator rejects synthetic and retrospective evidence for
promotion. `SHADOW-VALIDATED` requires real prospective silent evidence and
every mandatory safety, performance, burden, ranking, support, adjudication,
operational, and evidence-floor criterion. No current artifact should be
described as "world-class", "production-ready", or operationally effective.

## Next authorized work

1. Approve real deployment/telemetry/topology access and the privacy/storage
   design.
2. Map real historical data and run Task 19 without tuning.
3. Select pilot deployments, owners, dates, independent reviewers, incident
   sampling, watermark limits, and stop-condition owners.
4. Store the bundle and preregistration public-key fingerprints outside the
   application trust domain, then sign the protocol before observing outcomes.
5. Run the silent pilot with predictions hidden and no automatic remediation.
