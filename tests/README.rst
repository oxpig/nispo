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

Mode selection happens before execution. ``full_only``, ``fast_only`` and
``modes(...)`` declare applicable interfaces; the other mode's checks are
deselected. With the required parser available, selected release checks must
run without skips. Shared SMILES mode dispatch is checked separately by
``test_mode_api.py``. Test counts are not accuracy measurements.

Self-contained telemetry contracts in ``core/test_api_contracts.py`` cover
aggregation, serialization and instrumented batch naming. Tests for a separate
benchmark runner belong with that runner; this checkout requires no external
benchmark harness.

Priority mode, batch, timeout and fallback contracts::

    python -m pytest tests/test_mode_api.py tests/test_default_timeout.py \
        tests/test_parallel_timeout.py \
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

The shared numbered-stereo verifier lives in ``_name_oracles/numbered_stereo.py``.
Self-contained graph regressions from the former corpus module and rule-engine
tests run in ``core/test_graph_contracts.py``, retaining their original inputs
and raw OPSIN/Delta checks. Tests requiring deleted corpus files are removed.

Preferred-construction obligations and parent-selection proofs remain separate
from pytest. Run all seven generated/control acceptance gates against an
installed package from the repository root, using a new output directory::

    python tools/check_release_acceptance.py /absolute/new/report \
        --source-root /absolute/site-packages

The public acceptance runner contains generated functional/ring/stereo
composition checks and independently declared fast parent-selection controls.
Stored molecule-specific naming ledgers and their audit machinery are not
included. Historical development evidence remains separately archived.

Stored development molecule corpora and their cohort loaders are excluded
from the repository. Ordinary hand-written API/graph contracts and generated
rule controls remain. Original development inputs, provenance and prior results
are preserved separately, outside the public Git history.

Release CI builds and installs both wheel and sdist, runs both modes and the
priority contracts outside the source checkout, and requires all acceptance
gates. Install tests use the same ``pyproject.toml`` configuration and validation
inputs as the source-tree commands.
