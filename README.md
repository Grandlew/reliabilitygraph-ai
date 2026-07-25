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
