from __future__ import annotations

from rdkit import Chem

from nispo.scaffold_catalog import parent_scaffold_catalog
from nispo.template_store import CompiledTemplate, TemplateFeatures, TemplateRecord, template_store


def test_template_store_compiles_parent_resources():
    store = template_store(Chem)
    catalog = parent_scaffold_catalog(Chem)

    assert catalog.static_retained
    assert catalog.opsin_retained_fused
    assert catalog.fusion_component_parents
    assert catalog.fused
    assert catalog.exact
    assert catalog.single_ring

    compiled = store.compiled_template(catalog.static_retained[0])
    assert isinstance(compiled, CompiledTemplate)
    assert isinstance(compiled.record, TemplateRecord)
    assert isinstance(compiled.features, TemplateFeatures)
    assert compiled.scaffold_key


def test_template_store_does_not_cache_candidates_by_molecule_id():
    store = template_store(Chem)

    assert "_template_candidates" not in store.__dict__


def test_template_index_does_not_mutate_template_records():
    store = template_store(Chem)
    mol = Chem.MolFromSmiles("c1ccc2ccccc2c1")
    catalog = parent_scaffold_catalog(Chem)
    template = catalog.static_retained[0]
    keys_before = set(template)

    assert store.candidate_templates(mol, catalog.fused)
    assert set(template) == keys_before
    assert not any(key.startswith("_nispo_") for key in template)
