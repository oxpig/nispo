from __future__ import annotations

from rdkit import Chem

from nispo.scaffold_catalog import parent_scaffold_catalog
from nispo.resource_templates import build_opsin_retained_ring_csv_templates
from nispo.template_access import template_record_name
from nispo.template_store import CompiledTemplate, TemplateFeatures, TemplateRecord, template_store


def test_template_store_compiles_parent_resources():
    store = template_store(Chem)
    catalog = parent_scaffold_catalog(Chem)

    assert catalog.static_retained
    assert catalog.opsin_retained_fused
    assert catalog.opsin_retained_ring_csv_fused
    assert catalog.opsin_retained_ring_csv_single_ring
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


def test_opsin_retained_ring_csv_templates_are_locanted():
    templates = build_opsin_retained_ring_csv_templates(Chem)

    assert len(templates) == 1605
    assert all(template["locants"] for template in templates)
    assert any(
        template["name"] == "1,5-naphthyridine"
        and template["locants"].get(0) == "1"
        and template["locants"].get(5) == "5"
        for template in templates
    )


def test_opsin_retained_ring_csv_mono_rings_feed_single_ring_catalog():
    catalog = parent_scaffold_catalog(Chem)
    csv_single_ring_names = {template_record_name(template) for template in catalog.opsin_retained_ring_csv_single_ring}
    single_ring_names = {template_record_name(template) for template in catalog.single_ring}

    assert {"pyran", "thiopyran"} <= csv_single_ring_names
    assert csv_single_ring_names <= single_ring_names
