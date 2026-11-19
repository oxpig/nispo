"""Independent ring frontiers protect supported graph delivery.

Origin: twenty graphs were constructed from general ring topology, sulfur
valence and hydrogenation categories, before evaluating any producer names.
They were not copied from source examples, panel records or disagreements.
The frozen definition/provenance is in coverage_frontiers_20261111/rings.
All twenty controls independently recovered their input graph in full and
fast modes before availability became a required regression contract.
No particular parent, producer, rank or name spelling is required.
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.current_only

RING_FRONTIERS = (
    ('sulfur-monoxide-five', 'O=S1CCCC1'),
    ('sulfur-dioxide-five', 'O=S1(=O)CCCC1'),
    ('sulfur-monoxide-six', 'O=S1CCCCC1'),
    ('sulfur-dioxide-six', 'O=S1(=O)CCCCC1'),
    ('unsaturated-sulfur-monoxide', 'O=S1C=CC=C1'),
    ('unsaturated-sulfur-dioxide', 'O=S1(=O)C=CC=C1'),
    ('exocyclic-thione-oxide-five', 'O=S=C1CCCC1'),
    ('exocyclic-thione-oxide-six', 'O=S=C1CCCCC1'),
    ('partially-hydrogenated-fused-six', 'C1CCc2ccccc2C1'),
    ('partially-hydrogenated-fused-ene', 'C1=Cc2ccccc2CC1'),
    ('linked-arene-five-ene', 'c1ccccc1C1=CCCC1'),
    ('linked-arene-six-ene', 'c1ccccc1C1=CCCCC1'),
    ('methyl-unequal-small-spiro', 'CC1CC12CCC2'),
    ('hydroxy-unequal-small-spiro', 'OC1CC12CCC2'),
    ('fluoro-unequal-spiro', 'FC1CC12CCCC2'),
    ('methyl-unequal-medium-spiro', 'CC1CCC12CCCC2'),
    ('small-oxygen-sulfur-cycle', 'C1COS1'),
    ('selenium-six-cycle', 'C1CC[Se]CC1'),
    ('unsaturated-oxygen-five-cycle', 'C1=COCC1'),
    ('oxygen-sulfur-six-cycle', 'C1COCCS1'),
)


@pytest.fixture(scope='module')
def ring_frontier_results(naming, assert_graphs):
    inputs = [text for _label, text in RING_FRONTIERS]
    results = naming.results(inputs)
    assert len(results) == len(inputs)
    # Populate the shared independent parser cache with one small batch.
    assert_graphs.parse_names([result.name for result in results if result.name])
    return dict(zip(inputs, results, strict=True))


@pytest.mark.parametrize('text', [pytest.param(text, id=label) for label, text in RING_FRONTIERS])
def test_supported_ring_frontier_remains_available_and_recovers_graph(text, ring_frontier_results, assert_graphs):
    result = ring_frontier_results[text]
    assert result.generation_status == 'available'
    assert result.name
    assert_graphs([text], [result.name])
