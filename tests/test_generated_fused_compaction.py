from __future__ import annotations

import json
from pathlib import Path

from rdkit import Chem

from nispo.fused_template_families import candidate_generated_fused_templates
from nispo.template_index import molecule_features


RESOURCE_ROOT = Path(__file__).resolve().parents[1] / "nispo" / "resources"


def _list_leaves(value):
    if isinstance(value, list):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _list_leaves(item)


def test_generated_fused_indexes_are_id_based():
    assert (RESOURCE_ROOT / "generated_fused_template_index_records.json").exists()
    for filename in (
        "generated_fused_templates.json",
        "generated_fused_template_index.json",
        "generated_fused_template_graph_index.json",
        "generated_fused_template_element_graph_index.json",
    ):
        data = json.loads((RESOURCE_ROOT / filename).read_text(encoding="utf-8"))
        leaves = list(_list_leaves(data))
        assert leaves
        assert all(all(isinstance(item, int) for item in leaf) for leaf in leaves)


def test_generated_fused_candidate_materialization_still_returns_templates():
    mol = Chem.MolFromSmiles("C1=CC2=CC=CC=C2O1")
    templates = candidate_generated_fused_templates(mol, Chem, features=molecule_features(mol, Chem))

    assert templates
    assert all("name" in template and "mol" in template and "locants" in template for template in templates)
