from __future__ import annotations

from nispo import smiles_to_iupac
from nispo.parent.rule_helpers import _fused_parent_carbonyl_rank


def test_fused_carbonyl_rank_avoids_carboxyl_like_side_chains():
    assert _fused_parent_carbonyl_rank("3-hydroxy-20-oxo-pregnane") == 0
    assert _fused_parent_carbonyl_rank("23-ethyl-7,24-dihydroxy-24-oxo-cholane") == 1
    assert _fused_parent_carbonyl_rank("24-amino-24-oxo-23-(4-azabutyl)-cholane") == 1


def test_stereospecified_steroid_topology_uses_explicit_polycyclic_parent():
    name = smiles_to_iupac(
        "C1CC[C@H]2[C@H](C1)CC[C@@H]1[C@@H]2CC[C@@H]2CCC[C@H]21"
    )

    assert "tetracyclo[" in name
    assert "gonane" not in name


def test_small_ring_carbonyl_suffix_parent_beats_aromatic_substituent_parent():
    name = smiles_to_iupac("O=c1c(-c2ccccc2)c1-c1ccccc1")

    assert name == "1,2-diphenyl-cycloprop-1-en-3-one"


def test_cyclobutenedione_suffix_parent_beats_cyclobutadiene_dioxo_prefix():
    name = smiles_to_iupac(
        "O=c1c(NCc2ccc(-c3noc(C(F)(F)F)n3)c(F)c2)c(Nc2ccc(F)cc2)c1=O"
    )

    assert "cyclobut-1-en-3,4-dione" in name
    assert "cyclobuta-1,3-dien" not in name


def test_equal_priority_ring_system_candidate_can_beat_functional_chain_candidate():
    name = smiles_to_iupac("[C-]#[N+]c1c(F)cc(C2=CCCO2)c2c1OCC2")

    assert "bicyclo[4.3.0]" in name
    assert "azoniamethane" not in name


def test_substituent_like_functional_chain_defers_to_spiro_parent():
    name = smiles_to_iupac("O=C(NC1CC12CCCCC2)c1ccnc(-n2ccnc2)c1")

    assert "spiro[2.5]octane" in name
    assert "diamylcyclopropyl" not in name


def test_carbamate_side_chain_defers_to_component_spiro_parent():
    name = smiles_to_iupac("O=CC1CN(C(=O)OC(C)(C)C)CCC12CCCCC2")

    assert "spiro[cyclohexane-1,4'-piperidin]" in name
    assert "carbonyloxy)-propane" not in name


def test_kekulized_polycyclic_parent_can_beat_short_functional_chain():
    name = smiles_to_iupac(
        "O=C1CCC(N2Cc3ccc(COC(=O)Nc4c(F)ccc(Cl)c4F)cc3C2=O)C(=O)N1"
    )

    assert "bicyclo[4.3.0]" in name
    assert "oxapropane" not in name


def test_retained_fused_substituent_chain_defers_to_generic_ring_parent():
    name = smiles_to_iupac("O=c1cc(C2CC2)[nH]c2nc(NCc3ccc(Cl)cc3Cl)nn12")

    assert "tetraazabicyclo[4.3.0]" in name
    assert "triazolo[1,5-a]pyrimidine" not in name
    assert not name.endswith("-methane")


def test_chain_dominant_retained_fused_name_can_defer_to_polycyclic_parent():
    name = smiles_to_iupac(
        "Cc1ccc(NC(=O)C[C@H]2CCC[C@@H](C(F)(F)F)C2)cc1Nc1ncccc1-c1ncnc2[nH]cnc12"
    )

    assert "tetraazabicyclo[4.3.0]" in name
    assert "purine-6-yl" not in name
