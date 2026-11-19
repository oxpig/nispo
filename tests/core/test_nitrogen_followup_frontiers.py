"""Small nitrogen-class boundaries authored before any naming execution.

Origin: ordinary imine/iminium valence, imidic hydrazide connectivity,
carbonic imido/hydrazono derivatives, extended semicarbazone connectivity,
N-chalcogen-N links and one O-O join between noncarbon acid centres. All
substituents were chosen independently as small alkyl/halogen groups. No
panel, reference-name, code example or disagreement molecule was copied.
All 23 independently authored graphs were proved available in the initial
frozen run. They must remain available and recover the original structure;
loss of naming capability and delivered wrong graphs both fail.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from rdkit import Chem


pytestmark = pytest.mark.full_only


# Definitions are mirrored and hashed in the frozen provenance artifact before
# the first call to NISPO. Group labels state chemistry, not producer internals.
NITROGEN_FRONTIERS = (
    ("tertiary-iminium", "CCC=[N+](C)C", "iminium"),
    ("secondary-iminium", "CC(=[NH+]CC)C", "iminium"),
    ("iminium-and-primary-imine", "CC(=[NH2+])CC=N", "mixed-imine"),
    ("iminium-and-substituted-imine", "CN=C(C)CC=[NH2+]", "mixed-imine"),
    ("two-iminium-one-imine", "[NH2+]=CCC(=N)CC=[NH2+]", "mixed-imine"),
    ("alkyl-imidohydrazide", "CCC(=N)NN", "imidic-hydrazide"),
    ("proximal-N-methyl-imidohydrazide", "CC(=N)N(C)N", "imidic-hydrazide"),
    ("distal-N-methyl-imidohydrazide", "CC(=N)NNC", "imidic-hydrazide"),
    ("hydroxy-imidohydrazide", "CC(O)C(=N)NN", "imidic-hydrazide"),
    ("mixed-carbonic-imido-diester", "COC(=N)OCC", "carbonic-imido-hydrazono"),
    ("mixed-carbonic-hydrazono-diester", "COC(=NN)OCC", "carbonic-imido-hydrazono"),
    ("N-methyl-carbamic-hydrazono-ester", "CNC(=NN)OC", "carbonic-imido-hydrazono"),
    ("N-methyl-ketone-semicarbazone", "CCC(=NN(C)C(=O)N)C", "extended-semicarbazone"),
    ("branched-N-methyl-thiosemicarbazone", "CC(C)C=NN(C)C(=S)N", "extended-semicarbazone"),
    ("cyclic-ketone-semicarbazone", "NC(=O)NN=C1CCCC1", "extended-semicarbazone"),
    ("halo-N-methyl-semicarbazone", "CC(Cl)C(=NN(C)C(N)=O)C", "extended-semicarbazone"),
    ("unequal-secondary-amines-oxide", "CNONCC", "linked-amines"),
    ("unequal-amine-peroxide", "CN(C)OONCC", "linked-amines"),
    ("unequal-amine-sulfide", "CCN(C)SNC", "linked-amines"),
    ("nitrate-sulfate-peroxy-join", "[O-][N+](=O)OOS(=O)(=O)O", "noncarbon-peroxy-join"),
    ("nitrite-phosphate-peroxy-join", "O=NOOP(=O)(O)O", "noncarbon-peroxy-join"),
    ("silylated-five-membered-aza-ring", "C[Si](C)(C)n1ccnc1", "silyl-N-ring"),
    ("alkyl-substituted-silyl-amine-ring", "CC[Si](C)(C)N1CCCC1", "silyl-N-ring"),
)


def _canonical(text):
    graph = Chem.MolFromSmiles(text)
    assert graph is not None, f"Independently declared control has invalid valence: {text}"
    return Chem.MolToSmiles(graph, isomericSmiles=True)


@pytest.fixture(scope="module")
def nitrogen_frontier_results(assert_graphs):
    from nispo import smiles_to_iupac_batch_results

    inputs = [text for _case, text, _group in NITROGEN_FRONTIERS]
    expected = [_canonical(text) for text in inputs]
    results = smiles_to_iupac_batch_results(inputs, timeout_seconds=None)
    assert len(results) == len(inputs)
    # One name-oracle batch for this module; individual tests retain failures.
    named = [result.name for result in results if result.name]
    parsed = iter(assert_graphs.parse_names(named))
    records = {}
    evidence = []
    for (case, text, group), canonical, result in zip(NITROGEN_FRONTIERS, expected, results, strict=True):
        recovered = next(parsed) if result.name else None
        records[case] = (canonical, result, recovered)
        evidence.append({"case": case, "smiles": text, "group": group,
                         "status": result.generation_status, "name": result.name,
                         "preferred_status": result.preferred_status,
                         "route": result.route, "recovered_graph": recovered,
                         "limitations": [{"stage": issue.stage, "rule": issue.rule, "reason": issue.reason}
                                         for issue in result.unresolved]})
    output = os.environ.get("NISPO_NITROGEN_OUTPUT")
    if output:
        Path(output).write_text(json.dumps(evidence, indent=2)+"\n")
    return records


@pytest.mark.parametrize("case", [item[0] for item in NITROGEN_FRONTIERS])
def test_nitrogen_frontier_remains_available_and_preserves_graph(nitrogen_frontier_results, case):
    expected, result, recovered = nitrogen_frontier_results[case]
    assert result.generation_status == "available", (case, result.unresolved)
    assert result.name, f"{case}: previously available naming capability was lost"
    assert recovered, f"{case}: delivered name has no independent graph recovery: {result.name}"
    assert _canonical(recovered) == expected, f"{case}: delivered name changes the declared graph"
