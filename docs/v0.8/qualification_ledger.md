# v0.8 Phase A qualification ledger

## Release truth

This release adds offline, read-only IPTV-P0 qualification infrastructure.
The checked-in pack is synthetic and its mandatory terminal decision is
`REAL_DATA_REQUIRED`. It demonstrates contract, adapter, topology, time,
feature-reconstruction and evidence-sealing behavior only.

No operator telemetry, customer identifiers, production topology, credentials,
trained model, threshold change, live poller, inference invocation or
operational write capability is present.

## Completed assignments

1. Consolidated release truth in `baseline_ledger.json`.
2. Froze the managed-live-multicast IPTV-P0 charter and claim ceiling.
3. Added versioned Domain Pack contracts and trust-registry verification.
4. Added Deployment Pack, ownership, privacy and compatibility contracts.
5. Added deterministic source identity, lineage, clock and retention inventory.
6. Added offline-read-only capability attestation and forbidden probes.
7. Classified every frozen v0.6 input exactly once in the signal registry.
8. Composed profile-driven acceptance around the frozen collector semantics.
9. Added a deterministic offline JSONL/CSV reader protocol.
10. Added exact dedupe, append-only correction and reason-coded quarantine.
11. Froze clock-quality, skew, lateness and deterministic watermark rules.
12. Added synthetic positives and registered semantic mutation controls.
13. Froze the directional IPTV node, edge, multicast and service-path ontology.
14. Added offline full/delta topology JSONL ingestion.
15. Added event-time and knowledge-time topology reconstruction controls.
16. Added source/schema/clock/topology/feature/outcome gap compilation.
17. Added aggregate Stage 1/residual reconstruction and component-local,
    topology-bound Stage 2 reconstruction without inference.
18. Added privacy/residency contracts and repository fixture hygiene scans.
19. Added deterministic measured-result compilation, qualification/schema
    export, externally trusted manifests and gate reasons.
20. Added a no-tuning preregistered historical replay protocol.

## Merge-blocker hardening

The pre-merge audit proved three defects with failing regression tests before
the fixes were applied:

- Stage 2 repeated the aggregate Stage 1 vector for every topology node.
- `QualificationRequest` accepted caller-authored coverage, alignment,
  leakage, mutation-rejection, determinism and tuning scalars.
- Ed25519 metadata carried the public key used to verify its own signature,
  which proved possession of a key but not an authorized operator identity.

Stage 2 now joins each topology node to an explicit observation-component
pseudonym and an exact frozen-signal applicability set. Each node/feature cell
uses only component-local records visible at both event-time and knowledge-time
cutoffs. Observed, missing, unavailable and not-applicable cells remain
distinct. Per-node lineage commits the source records, cutoffs, topology
snapshot, applied captures, applicability decision and temporal rule. Required
coverage is derived from this lineage; zero governed Stage 2 cells yields zero
coverage. No inference or tuning path is present.

`QualificationMetrics` has been removed. The request instead carries frozen,
immutable measured-result objects containing individual semantic mappings,
positive and mutation controls, topology cutoffs and mutations, feature
reconstruction evidence, outcome identifiers, leakage findings, prohibited
tuning events, repeated-run commitments and tamper controls. The qualification
compiler revalidates the complete request and derives every gate fraction and
count from those sealed records. It does not accept authoritative coverage,
leakage, alignment or mutation-rejection scalars.

Signature metadata no longer embeds a verification key. Domain, deployment,
replay and final-manifest signatures are checked against an out-of-band trusted
signer registry. Registry entries bind signer identity and Ed25519 public key
to an authorized role, optional deployment, half-open validity interval and
revocation time. Signer IDs, key IDs and public keys must all be unique. A real
qualification requires three distinct authorized artifact signers and a fourth
distinct qualification-manifest signer. The bundle includes a hash-bound copy
of the registry for audit, but verification requires an independently supplied
registry with the same commitment; the bundle cannot install its own trust
root.

## Standards interpretation

- ETSI TR 101 290 informs the separation of transport-stream priority,
  availability and clock/PCR-related measurement semantics.
- RFC 3550 informs explicit sequence, timestamp, loss, jitter and out-of-order
  handling; RTP itself does not guarantee delivery order or quality of service.
- RFC 3376 and RFC 4541 inform directional multicast membership, control and
  snooping/topology semantics.
- OpenTelemetry semantic conventions inform stable names, types and meanings at
  the adapter boundary.
- NIST deployed-AI monitoring work supports context-specific, continuing
  evidence rather than a one-time software claim.
- SLSA v1.2 informs digest binding, provenance, signature verification and
  tamper detection for qualification artifacts.

## Gate and claim ceiling

Synthetic fixtures cannot satisfy a lawful deployment criterion even if every
software check passes. The checked-in synthetic registry is empty and its
manifest is explicitly unsigned. Until a lawful pack and all four distinct
authorized signatures pass the external criteria, the only permitted claim is:

> Engineering qualification infrastructure only; no real IPTV deployment has
> passed the external gate.

After one lawful pack passes, the maximum claim becomes:

> Compatible with one qualified real IPTV deployment for preregistered
> historical replay.

That does not establish prospective incident recall, alert burden, root-cause
quality, outage prevention, intervention safety or production readiness.
