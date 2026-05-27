from __future__ import annotations

from rdkit import Chem

from nispo import smiles_to_iupac
from nispo.parent.rule_helpers import _fused_parent_carbonyl_rank
from nispo.polycyclic_rules import _polycyclic_numbering_preference_key


def test_polycyclic_numbering_prefers_low_hetero_locants():
    mol = Chem.MolFromSmiles("N1CCCCC1")
    atoms = set(range(mol.GetNumAtoms()))
    nitrogen_first = tuple(range(mol.GetNumAtoms()))
    nitrogen_second = (1, 0, 5, 4, 3, 2)

    assert _polycyclic_numbering_preference_key(
        mol, nitrogen_first, atoms
    ) < _polycyclic_numbering_preference_key(mol, nitrogen_second, atoms)


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


def test_fused_bracket_descriptor_in_short_chain_defers_to_ring_parent():
    name = smiles_to_iupac("COc1cc(OC)c(Cl)c(Nc2nc3cccnc3[nH]c2=O)c1Cl")

    assert "triazabicyclo[4.4.0]" in name
    assert "pyrazino[2,3-b]pyridine" not in name
    assert not name.endswith("-methane")


def test_thienopyridine_short_chain_defers_to_generic_parent():
    name = smiles_to_iupac("Cc1cccc(-c2c(O)c3scc(Br)c3[nH]c2=O)c1")

    assert "thiabicyclo[4.3.0]" in name
    assert "thieno[3,2-b]pyridine" not in name
    assert not name.endswith("-methane")


def test_systematic_polycyclic_parent_can_beat_fragile_retained_fused_parent():
    name = smiles_to_iupac("CC(C)(C)c1nc2ccc([N+](=O)[O-])cc2n1C(=O)c1ccccc1")

    assert "diazabicyclo[4.3.0]" in name
    assert "benzimidazole" not in name


def test_systematic_polycyclic_parent_can_beat_retained_fused_amide():
    name = smiles_to_iupac("CC(C)n1c(Nc2ccc(C(=O)NO)cc2)nc2ccc(Br)cc21")

    assert "diazabicyclo[4.3.0]" in name
    assert "benzimidazole" not in name


def test_functional_bicyclic_parent_defers_to_fused_isoindolin_parent():
    name = smiles_to_iupac("O=C(O)C1C2CC(N3Cc4c(Br)cccc4C3=O)CC21")

    assert "azabicyclo[4.3.0]" in name
    assert "isoindolin" not in name


def test_functional_bicyclic_parent_defers_to_fused_imide_parent():
    name = smiles_to_iupac(
        "CC1(C)C2CCC1(COC(=O)NCc1ccc3c(c1)C(=O)N(C1CCC(=O)NC1=O)C3)CC2"
    )

    assert "azabicyclo[4.3.0]" in name
    assert "isoindolin" not in name


def test_spiro_acetal_parent_beats_diamyl_acyclic_ketal_name():
    name = smiles_to_iupac("COC(=O)C1OC2(CCCCC2)O[C@H]1c1ccccc1[N+](=O)[O-]")

    assert "dioxaspiro[4.5]decane" in name
    assert "diamyl" not in name


def test_spiro_acetal_benzoyl_chain_uses_ring_system_parent():
    name = smiles_to_iupac("O=C(CCC1COC2(CCCCC2)O1)c1ccccc1")

    assert "dioxaspiro[4.5]decane" in name
    assert "diamyl" not in name


def test_spiro_azetidine_parent_beats_diamyl_acyclic_carbamate_name():
    name = smiles_to_iupac("O=C(OCc1ccccc1)N1CC2(CCCCC2)C1")

    assert "azaspiro[3.5]nonane" in name
    assert "diamylazetidin" not in name


def test_spiro_oxetane_parent_beats_methoxyl_methylol_acyclic_amide_name():
    name = smiles_to_iupac("O=C(N1CCC2(CC1)CO2)C1(C2CC2)CC1")

    assert "oxaspiro[2.5]octane" in name
    assert "methoxyl" not in name
    assert "methylol" not in name


def test_short_functional_chain_defers_to_spiro_ring_parent():
    name = smiles_to_iupac("CC1=CCCC2(CCCCC2)[C@H]1/C=C/C=O")

    assert "spiro[5.5]undec" in name
    assert "prop-1-en" in name


def test_cyclic_sulfonimidoyl_methane_defers_to_ring_parent():
    name = smiles_to_iupac(
        "CS(=O)(=Nc1ccc(-c2noc(C(F)(F)F)n2)cc1)c1ccc(F)cc1F"
    )

    assert name.endswith("-benzene")
    assert "S-methylsulfonimidoyl" in name
    assert not name.endswith("-methane")


def test_halogenated_sulfonimidoyl_methane_keeps_acyclic_parent():
    name = smiles_to_iupac(
        "O=S(=Nc1ccc(-c2noc(C(F)(F)F)n2)c(F)c1)(c1ccccn1)C(F)(F)F"
    )

    assert name.endswith("-methane")
    assert "trifluoro" in name
    assert "S-pyridin-2-ylsulfonimidoyl" in name
