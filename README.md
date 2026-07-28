# NRIM

NRIM is a standalone Python 3.12 research implementation of the Network
Reliability Intelligence Model. It contains deterministic domain models,
simulation and benchmark tooling, baseline methods, shadow-evaluation
governance, and conservative diagnostic behavior. This repository does not
contain a frontend, presales application, or GNN.

## Safety and scientific scope

The checked-in datasets and examples are synthetic. They are useful for
reproducible software tests, controlled experiments, and regression detection,
but they are not evidence of production accuracy or operational readiness.
Synthetic results cannot establish real-network calibration, causal validity,
generalization under deployment shift, incident prevalence, alert burden, or
the safety and business impact of interventions. Real deployments require
prospective evaluation against representative operational data, independent
adjudication, privacy and security review, and explicit acceptance criteria.

The existing models, thresholds, policies, scientific semantics, and safe
`UNKNOWN`/`ESCALATE` behavior are intentional. Reproducibility checks do not
weaken or bypass those safeguards.

## Setup

Install [uv](https://docs.astral.sh/uv/) and use Python 3.12. The lockfile is
the dependency authority for development and CI.

PowerShell:

```powershell
uv python install 3.12
uv sync --frozen --group dev
uv run pytest --collect-only -q
uv run pytest -q
```

Linux:

```bash
uv python install 3.12
uv sync --frozen --group dev
uv run pytest --collect-only -q
uv run pytest -q
```

To validate the complete project foundation:

```bash
uv lock --check
uv sync --frozen --group dev
uv run python -m tools.project_checks imports
uv run python -m compileall app
uv run pytest --collect-only -q
uv run pytest -q
git diff --check
git status --short
```

## Test evidence

JUnit and reproducibility evidence are generated only in CI and uploaded as a
workflow artifact. They are deliberately ignored rather than treated as
checked-in scientific evidence. The evidence manifest binds a successful run
to the Git commit and tree, `uv.lock`, Python/platform/package environment,
collected pytest node IDs, and JUnit file hash. The verifier rejects a dirty
tree, a stale lock, or any changed bound input.

Local development does not need to create evidence. To reproduce the CI flow
in a clean checkout:

```bash
mkdir -p artifacts/test-evidence
uv run pytest --collect-only -q > artifacts/test-evidence/test-ids.txt
uv run pytest -q --junitxml=artifacts/test-evidence/junit.xml
uv run python -m tools.test_evidence generate \
  --junit artifacts/test-evidence/junit.xml \
  --test-ids artifacts/test-evidence/test-ids.txt \
  --output artifacts/test-evidence/manifest.json
uv run python -m tools.test_evidence verify \
  --manifest artifacts/test-evidence/manifest.json
```

The PowerShell equivalent uses `New-Item -ItemType Directory -Force
artifacts/test-evidence` for the first command and backticks for line
continuation.

## Immutable decision evidence (v0.7.2)

The decision-evidence layer records why a frozen v0.6 decision existed without
changing how that candidate decides. A decision stream contains strict,
hash-chained events for the serving snapshot, support, data quality, frozen
prediction, legal candidate universe, Stage 2 ranking when permitted,
observational evidence obligations, and final seal. Later evidence is appended
as an amendment or adjudication link; it never rewrites the original stream.

`DecisionSnapshot` is a deterministic projection, not another mutable truth
store. It supports:

- `original`: the exact events sealed for the historical decision;
- `as_known_at`: only events recorded by an explicit UTC knowledge time;
- `latest`: the original plus all valid amendments;
- `comparison`: original and latest with an event-level delta.

Candidate scores and causal evidence remain separate. Evidence states distinguish
support, contradiction, missingness, inapplicability, and unresolved evidence;
they never claim observational evidence is causal proof. Unsupported or
data-quality-blocked decisions remain `UNKNOWN`/`ESCALATE` and expose no Stage 2
ranking.

The checked-in fixture can be verified and replayed without network or
operational access:

```bash
uv run python -m app.domain.nrim.shadow.golden_replay verify-fixture \
  --directory app/domain/nrim/examples/shadow/decision_v0_7_2

uv run python -m app.domain.nrim.shadow.golden_replay replay \
  --input app/domain/nrim/examples/shadow/decision_v0_7_2/input/events.jsonl \
  --output-dir /tmp/nrim-decision-replay
```

`create_decision_router` provides only read operations:

- `GET /shadow/decisions`
- `GET /shadow/decisions/{id}`
- `GET /shadow/decisions/{id}/events`
- `GET /shadow/decisions/{id}/evidence`
- `GET /shadow/decisions/{id}/export`
- `GET /shadow/decisions/stream`

Reads never run inference. The router contains no ticket, notification,
configuration, suppression, restart, remediation, active-query, or customer
mutation capability.

This release proves deterministic software behavior and synthetic replay only.
It does not establish real IPTV accuracy, calibration, causal validity, alert
burden, human utility, outage prevention, or production readiness.

## IPTV-P0 offline qualification (v0.8 Phase A)

The `shadow.iptv_p0` package qualifies whether a lawful historical export can
reconstruct the frozen inputs without changing or invoking the model. It
contains strict Domain and Deployment Packs, a complete signal-availability
registry, deterministic source identities, offline JSONL/CSV adapters,
clock/watermark rules, bitemporal IPTV topology epochs, gap analysis, a
no-inference feature dry run, repository hygiene checks, evidence sealing and a
preregistered no-tuning replay protocol.

The reference adapters are offline and read-only. They contain no network
client, credential field, live polling, mutation, ticket, alarm-suppression,
notification or remediation capability. Required, conditional, optional and
unavailable signals remain distinct; absent data is never interpreted as
healthy.

Verify the checked-in synthetic qualification evidence:

```bash
uv run python -m app.domain.nrim.shadow.iptv_p0.qualification verify \
  --directory app/domain/nrim/examples/shadow/iptv_p0_v0_8/expected
```

Regenerate the synthetic pack in a temporary directory:

```bash
uv run python -m app.domain.nrim.shadow.iptv_p0.synthetic_fixture \
  --output /tmp/iptv_p0_v0_8
```

The checked-in bundle must return `REAL_DATA_REQUIRED`. It proves software
behavior only and cannot support a real replay or prospective performance
claim.
