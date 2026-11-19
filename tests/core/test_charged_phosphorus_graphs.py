"""Small independent graph contracts around P(III)/P(V) functional classes.

Provenance: these graphs were constructed directly from valence, ligand type
and formal-charge patterns on 2026-11-11. The coverage report motivated the
functional classes, never molecule identities or preferred-name choices.
No panel examples, code-comment molecules or prior disagreements were used.
Each bounded group is named once and sent to a cached graph oracle once.
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.current_only


PHOSPHORUS_GRAPHS = (
    pytest.param(
        ("CP(=O)([O-])[O-]", "CP(C)(=O)[O-]", "CP(=O)(N)[O-]", "C[PH](=O)[O-]"),
        id="charged-oxo-parents",
    ),
    pytest.param(
        ("COP(=O)([O-])OC", "COP(=O)([O-])C", "CO[PH](=O)[O-]", "CSP(=O)([O-])C"),
        id="oxo-ester-anions",
    ),
    pytest.param(
        ("CP(C)[O-]", "CP([O-])[O-]", "CO[PH][O-]", "CSP([O-])[S-]", "CSP([S-])[S-]"),
        id="mixed-trivalent-ligands",
    ),
    pytest.param(
        ("CNP(=O)(O)O", "CN(C)P(=O)(O)O", "CNP(=O)([O-])O", "CN(C)P(=O)([O-])Cl", "CP(Cl)N(C)C", "N(P(=O)(O)OC)(P(=O)(O)OC)P(=O)(O)OC"),
        id="phosphorus-nitrogen-ligands",
    ),
    pytest.param(
        ("OP(=O)(O)CP(=O)(O)O", "CC(O)(P(=O)(O)O)P(=O)(O)O", "OP(=O)(O)CCP(=O)(O)O", "OP(=O)(O)CCCP(=O)(O)O"),
        id="geminal-and-spaced-bisphosphonic-acids",
    ),
)


@pytest.mark.parametrize("inputs", PHOSPHORUS_GRAPHS)
def test_phosphorus_ligands_charge_and_hydrogens_are_recovered(naming, assert_graphs, inputs):
    names = naming.names(inputs)
    assert len(names) == len(inputs)
    assert all(isinstance(name, str) and name for name in names)
    assert_graphs(inputs, names)


def test_excess_phosphorus_valence_is_not_named(naming):
    result = naming.result("CP(=O)(=O)(O)O")
    assert not result.name
    assert result.generation_status == "invalid-input"
