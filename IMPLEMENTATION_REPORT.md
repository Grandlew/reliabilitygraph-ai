# NRIM implementation and release ledger

Current implementation branch: `feat/v0.8-iptv-p0-qualification`

Current baseline commit: `03ec7e47ced1729e7fb24e8b0e5d44b377e8261b`

Current phase: v0.8 Phase A IPTV-P0 qualification infrastructure

Earlier branch, commit and validation statements below are retained as
historical release evidence and are not descriptions of the current worktree.

## v0.8 Phase A outcome

The repository now contains an offline, read-only IPTV-P0 qualification layer.
It freezes scope and exclusions, classifies every frozen input, validates
source/clock/privacy semantics, reconstructs bitemporal topology, performs a
no-inference feature dry run, compiles reason-coded gaps, seals deterministic
evidence and preregisters historical replay without tuning.

The supplied synthetic pack deterministically returns `REAL_DATA_REQUIRED`.
No real operator pack was supplied or accessed, so v0.8 remains engineering
readiness infrastructure and cannot claim real-deployment compatibility.

The machine-readable baseline is `docs/v0.8/baseline_ledger.json`; the exact
20-assignment ledger and claim boundary are in
`docs/v0.8/qualification_ledger.md`.

### Test-first evidence

Before implementation, the v0.8 red-phase test failed with
`KeyError: 'terminal_decision'`: the prior historical replay gate had no
explicit `REAL_DATA_REQUIRED` terminal state for synthetic evidence. The gate
now emits `REAL_DATA_REQUIRED`, `BLOCKED` or `REAL_REPLAY_READY` without
weakening any existing criterion.

The pre-change local baseline also exposed three checkout-policy failures:
canonical v0.7.2 LF schemas and JSONL had been converted to CRLF. Narrow
extension-scoped `eol=lf` rules restore their committed canonical bytes and
cover the new v0.8 JSON, JSONL and Markdown evidence. No seal or verifier was
changed.

### Authority and claim statement

No model training, threshold tuning, GNN, inference call, frontend, live
endpoint, credential, ticket write, alarm suppression, customer notification
or remediation authority was added. Frozen v0.6 and v0.7.2 semantic identities
remain unchanged.

Passing synthetic qualification tests permits only the claim: “Engineering
qualification infrastructure only; no real IPTV deployment has passed the
external gate.”

### v0.8 validation evidence

Commands were run from the repository root using the frozen Python 3.12
environment and repository-local uv cache:

```text
uv lock --check
Resolved 24 packages in 1ms

uv run python -m tools.project_checks imports
project check passed: imports

uv run python -m compileall app tests tools
Completed successfully

uv run pytest tests/domain/nrim/shadow/iptv_p0 \
  tests/domain/nrim/shadow/test_iptv_p0_red_phase.py -q
269 passed

uv run pytest -q
731 passed in 110.26s

uv run python -m app.domain.nrim.shadow.iptv_p0.qualification verify \
  --directory app/domain/nrim/examples/shadow/iptv_p0_v0_8/expected
decision=REAL_DATA_REQUIRED; claim_ceiling=engineering_readiness_only

git diff --check
No output; exit code 0
```

The v0.8 fixture manifest SHA-256 is
`bb29016ed56190ec5acde631103f17d301e7dff6f70c1d76ce2ae35b3e90f355`.
The qualification evidence manifest SHA-256 is
`01c422bf13e9eaa9634d25fd3616043c4701f5036acf92515cf28579db2ad5da`.

Date: 2026-07-24  
Branch: `feat/standalone-foundation`  
Baseline commit: `0a162b3c9470fe991100777372fb3d40f6ec7fbc`

## Outcome

The standalone reproducibility foundation is implemented without changing
NRIM models, thresholds, policies, scientific semantics, or conservative
`UNKNOWN`/`ESCALATE` behavior. No GNN, frontend, or presales dependency was
added. The 271-test baseline is preserved and 11 infrastructure tests were
added, for 282 passing tests.

After the local validation recorded below, the standalone foundation was
committed and pushed to `feat/standalone-foundation` as
`a5f2a003e222fb2566bc60b0722870368d9f43e9` (`a5f2a003`).

## Import and dependency audit

The initial audit parsed all 111 files under `app/domain/nrim` and all 74
baseline test files without parse errors. The final automated audit covers
every Python file under `app` and `tests`, including the new infrastructure
tests.

Top-level third-party imports in application source are:

- `pydantic`
- `fastapi`
- `cryptography`

Tests additionally import `pytest` and FastAPI's `TestClient`. There are no
absolute `app.*` imports outside `app.domain.nrim`, and relative imports from
NRIM source do not escape that package. Some frozen or signed historical
synthetic-data manifests contain absolute strings from the source checkout.
Those strings are provenance inside governed artifacts, not executable
dependencies; changing them would invalidate hashes and evidence semantics, so
they were intentionally preserved.

The runtime dependency set is limited to FastAPI 0.139.x, Pydantic 2.13+, and
cryptography 49.x. The development group contains pytest 9.x and `httpx2`
2.x. Starlette 1.3.1 is resolved through FastAPI. Installing `httpx2` removes
Starlette's deprecated `httpx` fallback at its source; no warning filter or
suppression is configured. Frozen synchronization removed inherited,
unused packages including NumPy, pandas, SciPy, and scikit-learn.

## Implemented foundation

1. Audited source and test imports with AST parsing.
2. Added an executable import-boundary check for absolute and relative escapes.
3. Confirmed no executable presales or frontend coupling.
4. Added a Python 3.12-only `pyproject.toml`.
5. Declared the three minimal runtime dependencies.
6. Declared a separate two-package development group.
7. Generated `uv.lock`.
8. Selected the supported FastAPI/Starlette/`httpx2` combination without
   warning suppression.
9. Added a complete Python, uv, editor, build, temporary, and generated-evidence
   `.gitignore`.
10. Added a standalone NRIM README.
11. Documented why synthetic evidence cannot establish production validity.
12. Documented PowerShell and Linux setup and validation commands.
13. Moved strict NRIM pytest configuration into `pyproject.toml`.
14. Added Ubuntu 24.04/Python 3.12 CI.
15. Added dependent foundation, collection, and full-test CI stages.
16. Added compileall and import-boundary checks.
17. Removed two tracked generated test-evidence files; CI now generates and
   uploads ignored evidence.
18. Bound the manifest to commit, Git tree, lock hash, deterministic
   environment/package inventory, pytest node IDs, and JUnit hash.
19. Added failure tests for dirty trees, missing/stale locks, absolute/relative
   boundary escapes, and changed JUnit, lock, test IDs, or environment.
20. Added this implementation report.

## Validation evidence

Commands were run from the repository root with uv 0.10.4 and CPython 3.12.0.
On this managed Windows workspace, `UV_CACHE_DIR=.uv-cache` was set so uv's
cache stayed inside the writable repository. That variable does not change the
lock or environment resolution.

```text
uv lock --check
Resolved 24 packages in 8ms

uv sync --frozen --group dev
Audited 23 installed packages in 2ms; completed successfully.

uv run pytest --collect-only -q
282 tests collected in 1.28s

uv run pytest -q
282 passed in 22.85s

uv run pytest tests/infrastructure -q
11 passed in 1.94s

uv run python -m tools.project_checks imports
project check passed: imports

uv run python -m compileall app
Completed successfully.

git diff --check
No output; exit code 0.
```

The validated foundation changes were subsequently committed and pushed as
`a5f2a003`. The commit contains the workflow, ignore file, README, report,
`pyproject.toml`, `uv.lock`, tools, and infrastructure tests; deletion of
`pytest.ini`; and deletion of the two formerly tracked generated evidence
files.

## Cross-platform locked-test evidence correction

The committed v0.6 seal and manifests retain absolute paths from their original
Windows checkout as historical provenance. A verifier incorrectly treated
those strings as live filesystem locations, so a relocated Ubuntu checkout
could not find the locked manifest or its scenario files. Local validation
masked the defect because the retired source repository still existed.

The merged correction resolves governed manifests, scenario sidecars, and
model-ready windows only from their current artifact location, declared split,
scenario or window identifier, and required filename. Recorded paths are
validated as metadata but are never dereferenced. Missing colocated files,
wrong filenames, traversal components, altered manifest content, altered
scenario content, and changed commitment hashes remain rejected. No governed
v0.6 JSON artifact was modified, regenerated, or re-signed.

A byte-level audit of all 480 locked-test scenario sidecars then identified a
second, independent Ubuntu failure. All 240 `*.observable.json` files and all
240 `*.hidden.json` files have LF-only Git blobs but were historically sealed
from a CRLF Windows working tree. For every mismatch, the seal-expected SHA-256
equals both the raw Windows working-tree SHA-256 and the SHA-256 produced by
reconstructing CRLF from the Git blob. Conversely, the Git blob SHA-256 equals
the SHA-256 produced by normalizing the working-tree bytes to LF. There were no
exceptions or non-newline byte differences.

The portable resolver independently mapped all 480 manifest records to 480
unique, existing files under the relocated `locked_test` directory, with zero
wrong targets. Path resolution therefore did not contribute to any hash
mismatch.

The root `.gitattributes` now restores the historically sealed working-tree
bytes on both Windows and Ubuntu using only these governed text-artifact rules:

```gitattributes
app/domain/nrim/examples/simulation/day08_dataset_v06/locked_test/*.observable.json text eol=crlf
app/domain/nrim/examples/simulation/day08_dataset_v06/locked_test/*.hidden.json text eol=crlf
```

No binary path is marked as text. Raw SHA-256 verification and the existing seal
remain unchanged; there is no normalized-hash or multiple-hash fallback.
Regression coverage verifies Git checkout conversion with both
`core.autocrlf=false` and `core.autocrlf=true`, rejection of genuine content
changes to CRLF-sealed bytes, and verification after physical relocation with
stale Windows and POSIX provenance paths. Final local validation passes with
297 tests, including the strict shadow parity test. Clean Ubuntu acceptance
still requires an actual successful GitHub Actions run.

## Risks and limitations

- The Ubuntu workflow is defined but cannot be executed by this local Windows
  session; GitHub Actions is the clean-Ubuntu acceptance environment.
- The project remains synthetic-data-only. Passing tests and integrity-bound
  artifacts demonstrate software reproducibility, not operational accuracy,
  calibration, causal validity, or deployment safety.
- Historical frozen/signed artifacts retain original checkout path strings to
  preserve their commitments. The automated boundary applies to executable
  Python imports, where coupling risk exists.
- Evidence generation deliberately refuses a dirty Git tree or stale lock.
  The committed foundation at `a5f2a003` supplies the clean, commit-bound input
  needed by CI. The portable resolver and governed-newline correction are now
  part of the merged baseline used by later releases.

## Acceptance status

All 20 requested implementation tasks are complete. All locally executable
validation criteria pass with zero test failures, zero collection errors, and
no hidden warnings. Clean Ubuntu execution remains pending the first CI run.

## v0.7.2 decision-event and causal-evidence core

Date: 2026-07-26

Branch: `feat/v0.7.2-decision-evidence-core`

Baseline commit: `98a7aa0eab267a3828d2b32797afb5ae7e3c24c3`

### Outcome

v0.7.2 adds immutable, typed decision-event streams around the frozen v0.6
candidate. Every stream binds its information cutoff, truth time, knowledge
time, payload hashes, predecessor chain, candidate universe, observational
evidence obligations, final decision, and frozen prediction identity.
Deterministic projections reconstruct original, as-known-at, latest, and
comparison views. Late observations and adjudications append new events without
rewriting original model output.

The existing SQLite evidence authority now has a narrow decision-event adapter
with transactional whole-stream append, idempotent retries, immutable triggers,
ordered lookup, and append-only stream commits. The v0.6 adapter preserves the
entire `PredictionEnvelope`, rank order, scalar precision, safety state,
activation path, feature/schema/model/policy hashes, and blocked-route
semantics. Candidate instrumentation is observational and does not enter the
ranker or alter tensors.

A deterministic replay command produces LF-only canonical JSONL, projections,
and a digest-bound manifest. The checked-in synthetic fixture, versioned JSON
Schemas, schema manifest, scope ledger, and frozen identities are independently
verifiable. The FastAPI router exposes GET-only list, detail, event, evidence,
export, and resumable SSE views; reads never invoke inference.

### Test-first evidence

The representative red-phase test
`test_canonical_json_rejects_unsafe_process_dependent_values` failed twice
before implementation because the prior canonicalizer silently converted bytes
and unordered sets to process-dependent strings. The hardened canonicalizer now
rejects both, normalizes aware datetimes to UTC, rejects naive/non-finite values,
and binds contract type and schema version into typed commitments.

The focused v0.7.2 suite covers strict contracts, payload variants, bitemporal
boundaries, hashing, chain tamper/splice rejection, candidate exclusions,
evidence alternatives, append-only storage, projections and amendments, v0.6
compatibility, replay/restart/relocation, schema drift, read-only HTTP/SSE, PII,
dependency direction, and forbidden capabilities.

### Validation evidence

Commands were run from the repository root with the frozen Python 3.12
environment:

```text
uv run pytest --collect-only -q
462 tests collected

uv run pytest -q
462 passed in 66.09s

uv run pytest tests/domain/nrim/shadow/test_real_feature_parity.py -q
1 passed in 11.32s

uv lock --check
Resolved 24 packages in 2ms

uv run python -m tools.project_checks imports
project check passed: imports

uv run python -m compileall app
Completed successfully
```

The v0.7.2 focused suite contributes 165 passing cases over the 297-test
baseline. The sealed synthetic fixture and generated exchange contracts have
these handoff identities:

- Fixture manifest: `56aa79401d77f00a4e933e0151c1b7637b618a65f83f1e73babceddb992a5c62`
- Schema manifest: `ddb37e2b4312e5aa30ea2f358ebe7ccb15dcf1cba28e6f8c8a3354c1e06233b7`
- Canonical event JSONL: `f2a931ee77e62bc61f41a1139d71fc3d2a2a78ca2ac71619f26e3832589023d8`
- Canonical snapshot JSONL: `82256f861211c1cd6881d018fdf4052480f050adafd10092f958e8a299b446ee`

### Compatibility and authority statement

No governed v0.6 artifact, bundle, threshold, label, temporal policy, support
rule, feature, tensor, probability, rank, or activation decision was changed.
No dependency was added. No operational query, ticket, notification,
configuration write, alarm suppression, restart, remediation, customer action,
GNN, graph database, broker, Kubernetes, MPC, or LLM authority was introduced.

This remains a read-only `CANDIDATE`. Passing software and synthetic replay
tests do not establish real IPTV calibration, causal validity, alert burden,
human utility, production efficacy, outage prevention, remedy safety, ISP
transfer, cybersecurity readiness, or cross-sector generality. Real collector
and topology acceptance, approved historical replay, independent evidence
custody, dress rehearsal, and prospective silent-pilot gates remain closed.
