from __future__ import annotations

from rdkit import Chem

from nispo.scaffold_catalog import generated_fused_parent_templates, parent_scaffold_catalog
from nispo.template_store import CompiledTemplate, TemplateFeatures, TemplateRecord, template_store


def test_template_store_owns_parent_resource_counts():
    store = template_store(Chem)
    catalog = parent_scaffold_catalog(Chem)

    assert len(catalog.static_retained) == 79
    assert len(catalog.opsin_retained_fused) == 377
    assert len(catalog.fusion_component_parents) == 114
    assert len(catalog.fused) == 570
    assert len(catalog.exact) == 1142
    assert len(catalog.single_ring) == 246
    assert len(generated_fused_parent_templates(Chem)) == 65705

    compiled = store.compiled_template(catalog.static_retained[0])
    assert isinstance(compiled, CompiledTemplate)
    assert isinstance(compiled.record, TemplateRecord)
    assert isinstance(compiled.features, TemplateFeatures)
    assert compiled.scaffold_key


def test_template_store_candidate_selection_is_stable():
    store = template_store(Chem)
    mol = Chem.MolFromSmiles("c1ccc2ccccc2c1")
    catalog = parent_scaffold_catalog(Chem)

    candidates = store.candidate_templates(mol, catalog.fused)

    assert candidates
    assert any(template["name"] == "naphthalene" for template in candidates)


def test_template_index_does_not_mutate_template_records():
    store = template_store(Chem)
    mol = Chem.MolFromSmiles("c1ccc2ccccc2c1")
    catalog = parent_scaffold_catalog(Chem)
    template = catalog.static_retained[0]
    keys_before = set(template)

    assert store.candidate_templates(mol, catalog.fused)
    assert set(template) == keys_before
    assert not any(key.startswith("_nispo_") for key in template)
