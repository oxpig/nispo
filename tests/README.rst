NISPO release checks
====================

Install the test dependencies with ``pip install -e '.[test]'``. Set
``NISPO_OPSIN_JAR`` to the OPSIN 2.9.0 CLI JAR and, if necessary, ``JAVA`` to
the Java executable. Release checks require the external parser::

    python -m pytest tests/core tests/integration --nispo-mode=full --require-opsin
    python -m pytest tests/core tests/integration --nispo-mode=fast --require-opsin

Plain ``python -m pytest`` selects the focused full-mode ``core/`` suite.
Use ``--nispo-mode=fast`` for fast mode, or ``-m workers`` for worker contracts.
The settings and registered markers live in ``pyproject.toml``.

Mode selection happens before execution. The combined suites select 2,150
full-mode cases or 1,857 fast-mode cases: 1,694 are shared, 456 run only in full,
and 163 run only in fast. Pytest reports the other mode's cases as deselected.
With the required parser available, neither release run should contain skips.
Release CI fails if a selected check skips or a suite contains no tests.

``full_only`` and ``fast_only`` identify the interface a contract tests.
``modes(...)`` declares applicable interfaces or a record's unchanged frozen
cohort membership. These labels do not claim that another engine cannot name
the corresponding molecule. Shared SMILES mode dispatch is covered separately
by ``test_mode_api.py``. Cohort membership is fixed before naming runs; these
regression selections are not a common-cohort accuracy benchmark.

Self-contained telemetry contracts in ``core/test_api_contracts.py`` cover
aggregation, serialization and instrumented batch naming. Tests for a separate
benchmark runner belong with that runner; this checkout requires no external
benchmark harness.

Related contracts share modules. ``data/development_graphs.json`` retains the
complete original fixture cohorts and their source hashes; consolidation does
not select cases or change the graph assertions.

Priority mode, batch, timeout and fallback contracts::

    python -m pytest tests/test_mode_api.py tests/test_default_timeout.py \
        tests/test_parallel_timeout.py tests/test_surechembl_batch_pin_contract.py \
        tests/test_composition_audit_contract.py tests/test_fast_public_separation.py \
        --require-opsin

The integration adapter selects all historical single-name/parser/graph-equality
contracts by their assertion syntax, retaining their original records verbatim.
``test_api_contract.py`` is their preserved source; its older harness is not
imported by the adapter. Selection never depends on naming outcomes.

Tests protect API behavior, graph identity, determinism, batch safety, input
ownership and worker transport. Follow ``AGENTS.md``; do not add generated-name
snapshots or private chemical ranking assertions. Some graph checks use bounded,
independent name-only parser supplements documented in
``_name_oracles/provenance.json``. Raw-only SureChEMBL checks require OPSIN
itself. Test pass counts are not raw OPSIN or preferred-name recovery rates.

Preferred-construction obligations and parent-selection proofs remain separate
from pytest. Run all ten acceptance gates against an installed package from
the repository root, using a new output directory::

    python tools/check_release_acceptance.py /absolute/new/report \
        --source-root /absolute/site-packages

``development/`` retains the required frozen inputs, origins and acceptance
helper. Complete pre-cleanup source, historical audits, corpora and original
local logs were preserved separately before pruning. Keep their provenance and
prior evaluations intact; cleanup does not authorize new panel-based fixtures.

Release CI builds and installs both wheel and sdist, runs both modes and the
priority contracts outside the source checkout, and requires all acceptance
gates. Install tests use the same ``pyproject.toml`` configuration and validation
inputs as the source-tree commands.
