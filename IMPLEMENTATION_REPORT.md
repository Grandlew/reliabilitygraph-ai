# NRIM standalone foundation implementation report

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

The uncommitted correction resolves governed manifests, scenario sidecars, and
model-ready windows only from their current artifact location, declared split,
scenario or window identifier, and required filename. Recorded paths are
validated as metadata but are never dereferenced. Missing colocated files,
wrong filenames, traversal components, altered manifest content, altered
scenario content, and changed commitment hashes remain rejected. No governed
v0.6 JSON artifact was modified, regenerated, or re-signed.

Regression coverage includes stale Windows and POSIX absolute provenance paths.
Current local validation passes with 296 tests, including the relocated strict
shadow parity test. Clean Ubuntu acceptance still requires an actual successful
GitHub Actions run.

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
  needed by CI. The current correction remains intentionally uncommitted.

## Acceptance status

All 20 requested implementation tasks are complete. All locally executable
validation criteria pass with zero test failures, zero collection errors, and
no hidden warnings. Clean Ubuntu execution remains pending the first CI run.
