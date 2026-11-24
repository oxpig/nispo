from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

try:
    from rdkit import Chem, RDLogger
except Exception:  # pragma: no cover - benchmark environment provides RDKit.
    Chem = None
else:
    RDLogger.DisableLog("rdApp.*")

from ..molecule_iteration import iter_atoms, iter_bonds
from ..constants import HALO, MULT, ROOT_LENGTHS, ROOTS
from ..formatting import alkyl_to_alkoxy as _alkyl_to_alkoxy
from ..formatting import format_prefixes as _format_prefixes
from ..molecule_cache import mol_cache_dict
from ..stereochemistry import assigned_atom_stereochemistry
from .functional_suffixes import suffix_prefix_term, suffix_supported_term

_HALOSULFUR_OXO_PREFIXES = {
    2: {"F": "fluorosulfonyl", "Cl": "chlorosulfonyl", "Br": "bromosulfonyl", "I": "iodosulfonyl"},
    1: {"F": "fluorosulfinyl", "Cl": "chlorosulfinyl", "Br": "bromosulfinyl", "I": "iodosulfinyl"},
}
_SULFUR_OXO_SUBSTITUENT_SUFFIX = {2: "sulfonyl", 1: "sulfinyl", 0: "sulfanyl"}
_GROUP14_SUBSTITUENT_SUFFIX = {"Si": "silyl", "Ge": "germyl", "Sn": "stannyl", "Pb": "plumbyl"}


def _branch_rules():
    from .. import branch_rules

    return branch_rules


def _oxygen_prefix(mol, idx: int, parent_idx: int) -> str:
    return _branch_rules()._oxygen_prefix(mol, idx, parent_idx)


def _nitrogen_prefix(mol, idx: int, parent_idx: int) -> str:
    return _branch_rules()._nitrogen_prefix(mol, idx, parent_idx)


def _ring_substituent_prefix(mol, attachment_idx: int, parent_idx: int) -> str:
    return _branch_rules()._ring_substituent_prefix(mol, attachment_idx, parent_idx)


def branch_prefix(
    mol,
    idx: int,
    parent_idx: int,
    *,
    assume_simple_external_prefix_miss: bool = False,
) -> str:
    return _branch_rules().branch_prefix(
        mol,
        idx,
        parent_idx,
        assume_simple_external_prefix_miss=assume_simple_external_prefix_miss,
    )


def can_be_in_parent_skeleton(mol, idx: int) -> bool:
    return _branch_rules().can_be_in_parent_skeleton(mol, idx)


def parent_chain_name(mol, path: Sequence[int], *, as_substituent: bool = False) -> str:
    return _branch_rules().parent_chain_name(mol, path, as_substituent=as_substituent)


def replacement_prefixes(mol, path: Sequence[int]) -> list[tuple[str, str]]:
    return _branch_rules().replacement_prefixes(mol, path)


def _sulfur_imidoyl_s_group(name: str) -> str:
    if not name:
        return ""
    if re.fullmatch(r"[A-Za-z]+", name):
        return name
    if name.startswith("(") and name.endswith(")"):
        return name
    return f"({name})"


def _sulfur_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if (atom.GetAtomicNum() == 16 and atom.GetFormalCharge() == 0
            and not atom.GetIsotope() and atom.GetDegree() == 2
            and parent_bond is not None and parent_bond.GetBondTypeAsDouble() == 1):
        distal = [n for n in atom.GetNeighbors() if n.GetIdx() != parent_idx]
        if (len(distal) == 1 and distal[0].GetAtomicNum() == 6
                and not distal[0].GetIsotope() and distal[0].GetDegree() == 2
                and mol.GetBondBetweenAtoms(idx, distal[0].GetIdx()).GetBondTypeAsDouble() == 1):
            nitrogen = [n for n in distal[0].GetNeighbors()
                        if n.GetAtomicNum() == 7 and n.GetDegree() == 1
                        and not n.GetIsotope() and not n.GetFormalCharge()
                        and mol.GetBondBetweenAtoms(distal[0].GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 3]
            if len(nitrogen) == 1 and not distal[0].GetFormalCharge():
                return 'thiocyanato'
    if (atom.GetAtomicNum() == 16 and parent_bond is not None
            and parent_bond.GetBondTypeAsDouble() == 1
            and any(n.GetAtomicNum() == 6 and n.IsInRing() and n.GetIdx() != parent_idx
                    for n in atom.GetNeighbors())):
        from .sulfonyl_components import mapped_sulfonyl_prefix
        atoms = frozenset(collect_subtree(mol, idx, parent_idx))
        preferred = mapped_sulfonyl_prefix(mol, Chem, idx, parent_idx, atoms)
        if preferred is not None and preferred[0] and preferred[1] == atoms:
            return preferred[0]
    if atom.GetFormalCharge() < 0:
        heavy = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() != parent_idx]
        if not heavy and parent_bond is not None and parent_bond.GetBondType() == Chem.BondType.SINGLE:
            return "sulfido"
        return ""
    if atom.GetFormalCharge() > 0:
        heavy = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() != parent_idx]
        if not heavy and parent_bond is not None and parent_bond.GetBondType() == Chem.BondType.SINGLE:
            return suffix_supported_term("ium", "sulfaniumyl")
        return ""
    if atom.GetFormalCharge() != 0:
        return ""
    hydrogen_rich = _hydrogen_rich_sulfur_prefix(mol, idx, parent_idx)
    if hydrogen_rich:
        return hydrogen_rich
    if parent_bond is not None and parent_bond.GetBondType() == Chem.BondType.DOUBLE:
        heavy = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() != parent_idx]
        if not heavy and atom.GetTotalValence() == 2 and not atom.GetTotalNumHs(includeNeighbors=True):
            return suffix_prefix_term("thione", "thioxo")
        sulfoximine = _lambda6_sulfanylidene_prefix(mol, idx, parent_idx)
        if sulfoximine:
            return sulfoximine
        # A double attachment cannot fall through to a singly attached
        # sulfanyl/sulfinyl word: that loses its original bond and ligands.
        return _lambda_sulfur_hydride_prefix(mol, idx, parent_idx)
    non_parent = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() != parent_idx]
    lambda6 = _lambda6_tetrafluorosulfanyl_prefix(mol, idx, parent_idx, non_parent)
    if lambda6:
        return lambda6
    halogens = [
        n.GetSymbol()
        for n in atom.GetNeighbors()
        if n.GetIdx() != parent_idx and n.GetSymbol() in HALO
    ]
    if len(halogens) == 5 and all(symbol == "F" for symbol in halogens):
        return "pentafluorosulfanyl"
    if len(halogens) == 3 and all(symbol == "F" for symbol in halogens) and len(non_parent) == 3:
        return "trifluorosulfanyl"
    double_oxygens = {
        n.GetIdx()
        for n in atom.GetNeighbors()
        if n.GetSymbol() == "O"
        and mol.GetBondBetweenAtoms(idx, n.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
    }
    double_sulfurs = {
        n.GetIdx()
        for n in atom.GetNeighbors()
        if n.GetSymbol() == "S"
        and mol.GetBondBetweenAtoms(idx, n.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
        and not [
            nbr.GetIdx()
            for nbr in n.GetNeighbors()
            if nbr.GetIdx() != idx and nbr.GetAtomicNum() > 1
        ]
    }
    substituent_thioxo_sulfurs = double_sulfurs if len(double_sulfurs) == 1 else set()
    double_o = len(double_oxygens)
    if _is_sulfonate_sulfur(mol, idx):
        return "sulfonato"
    sulfonimidoyl = _sulfonimidoyl_prefix(mol, idx, parent_idx, double_o)
    if sulfonimidoyl:
        return sulfonimidoyl
    halosulfur = _halosulfur_prefix(double_o, halogens, len(non_parent))
    if halosulfur:
        return halosulfur
    heavy = [
        n.GetIdx()
        for n in atom.GetNeighbors()
        if n.GetIdx() != parent_idx
        and n.GetIdx() not in double_oxygens
        and n.GetIdx() not in substituent_thioxo_sulfurs
        and not _is_terminal_carbonyl_oxygen(n)
    ]
    thioxo_prefix = _sulfur_thioxo_substituent_prefix(
        mol,
        idx,
        parent_idx,
        double_o,
        len(substituent_thioxo_sulfurs),
        heavy,
    )
    if thioxo_prefix:
        return thioxo_prefix
    if double_o == 2 and not heavy:
        return suffix_prefix_term("sulfone", "sulfonyl")
    if double_o == 2 and len(heavy) == 1:
        retained_sulfonyl = _retained_sulfonyl_prefix(mol, heavy[0], idx)
        if retained_sulfonyl:
            return retained_sulfonyl
    if double_o == 2 and len(heavy) == 1 and _is_nitrile_carbon(mol, heavy[0]):
        return "cyanosulfonyl"
    if double_o == 0:
        sulfimidoyl = _sulfimidoyl_prefix(mol, idx, parent_idx)
        if sulfimidoyl:
            return sulfimidoyl
        if len(heavy) == 1 and mol.GetAtomWithIdx(heavy[0]).GetSymbol() == "O":
            lambda6_dioxy = _lambda6_dioxythio_prefix(mol, idx, parent_idx, heavy)
            if lambda6_dioxy:
                return lambda6_dioxy
    if not heavy:
        return "sulfanyl"
    if len(heavy) == 1:
        if double_o == 0 and mol.GetAtomWithIdx(heavy[0]).GetAtomicNum() == 6:
            # The P-66 terminal amide writer proves a compound carbon-chain
            # ligand. Preserve its boundary when substituting sulfanyl.
            from .subordinate_amide_chain import subordinate_amide_chain_prefix
            from ..formatting import enclose_name_unit
            ligand = subordinate_amide_chain_prefix(mol, Chem, heavy[0], idx)
            if ligand is not None:
                return enclose_name_unit(ligand[0]) + "sulfanyl"
        if mol.GetAtomWithIdx(heavy[0]).IsInRing():
            ring = _ring_substituent_prefix(mol, heavy[0], idx)
            if not ring:
                return ""
            suffix = _sulfur_oxo_substituent_suffix(double_o)
            return _apply_sulfinyl_stereo(mol, idx, double_o, ring + suffix) if suffix else ""
        if mol.GetAtomWithIdx(heavy[0]).GetSymbol() == "N" and double_o == 2:
            host = mol.GetAtomWithIdx(parent_idx)
            if host.GetAtomicNum() in {6, 7}:
                from ..sulfur_prefixes import characteristic_sulfur_prefix

                scope = (frozenset(collect_subtree(mol, idx, parent_idx)) if host.GetAtomicNum() == 6
                         else frozenset({idx, heavy[0], *double_oxygens}))
                sulfamoyl = characteristic_sulfur_prefix(mol, Chem, idx, parent_idx, scope)
                if sulfamoyl and sulfamoyl[0]:
                    return sulfamoyl[0]
            amino = _nitrogen_prefix(mol, heavy[0], idx)
            if not amino:
                return ""
            # Keep substituents on the distal amino nitrogen inside its group.
            # Otherwise a sulfonamido contraction can move them onto the
            # proximal nitrogen or the molecular parent during parsing.
            return (f"({amino})" if amino != "amino" else amino) + "sulfonyl"
        if mol.GetAtomWithIdx(heavy[0]).GetSymbol() == "O" and double_o == 2:
            oxy = _oxygen_prefix(mol, heavy[0], idx)
            if not oxy:
                return ""
            return oxy + "sulfonyl"
        distal = mol.GetAtomWithIdx(heavy[0])
        if double_o == 0 and distal.GetAtomicNum() == 16:
            host = mol.GetAtomWithIdx(parent_idx)
            if host.IsInRing() or (host.GetAtomicNum() == 6 and atom.GetTotalValence() == 2
                    and distal.GetTotalValence() == distal.GetDegree() == 2
                    and all(n.GetAtomicNum() != 16 or n.GetIdx() == idx for n in distal.GetNeighbors())):
                sulfur = _sulfur_prefix(mol, heavy[0], idx)
                if not sulfur:
                    return ""
                if sulfur.endswith("sulfanyl"):
                    return sulfur[: -len("sulfanyl")] + "disulfanyl"
                return sulfur + "sulfanyl"
        alk = branch_prefix(mol, heavy[0], idx) or substituent_from_root(mol, heavy[0], idx)
        if not alk:
            return ""
        suffix = _sulfur_oxo_substituent_suffix(double_o)
        return _apply_sulfinyl_stereo(mol, idx, double_o, alk + suffix) if suffix else ""
    return ""


def _hydrogen_rich_sulfur_prefix(mol, idx: int, parent_idx: int) -> str:
    if mol.GetAtomWithIdx(idx).GetTotalNumHs(includeNeighbors=True):
        return _lambda_sulfur_hydride_prefix(mol, idx, parent_idx)
    return ""


def _lambda_sulfur_hydride_prefix(mol, idx: int, parent_idx: int) -> str:
    """P-14.1/P-21/P-29: retain S bonding number and attachment valence.

    A lambda4/lambda6 sulfane loses one or two H atoms for the outer cut;
    every other ligand consumes its own original bond order. The remaining
    H budget must equal the original S atom, including explicit neighbours.
    Ordinary divalent sulfur and established acid-derived words stay with
    their dedicated writers.
    """
    sulfur = mol.GetAtomWithIdx(idx)
    bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    valence = int(sulfur.GetTotalValence())
    if (
        sulfur.GetSymbol() != "S" or sulfur.IsInRing() or sulfur.GetIsotope()
        or sulfur.GetFormalCharge() or sulfur.GetNumRadicalElectrons()
        or int(sulfur.GetChiralTag()) or valence not in {4, 6}
        or bond is None or bond.GetBondTypeAsDouble() not in {1, 2}
        or int(bond.GetStereo())
    ):
        return ""
    ligands, demand = [], int(bond.GetBondTypeAsDouble())
    for neighbor in sulfur.GetNeighbors():
        nidx = neighbor.GetIdx()
        if nidx == parent_idx:
            continue
        if neighbor.GetAtomicNum() == 1:
            if neighbor.GetIsotope():
                return ""
            continue
        edge = mol.GetBondBetweenAtoms(idx, nidx)
        order = edge.GetBondTypeAsDouble()
        if neighbor.GetIsotope() or neighbor.GetNumRadicalElectrons():
            return ""
        if order == 2:
            if neighbor.GetFormalCharge():
                return ""
            if (neighbor.GetAtomicNum() in {8, 16} and neighbor.GetDegree() == 1
                    and not neighbor.GetTotalNumHs(includeNeighbors=True)):
                ligand = 'oxo' if neighbor.GetAtomicNum() == 8 else 'thioxo'
            elif (neighbor.GetAtomicNum() == 7 and neighbor.GetTotalValence() == 3
                    or neighbor.GetAtomicNum() == 16 and not neighbor.IsInRing()):
                ligand = branch_prefix(mol, nidx, idx)
            else:
                return ""
        elif order == 1:
            ligand = branch_prefix(mol, nidx, idx)
        else:
            return ""
        if not ligand:
            return ""
        ligands.append(ligand)
        demand += int(order)
    if valence - demand != sulfur.GetTotalNumHs(includeNeighbors=True):
        return ""
    text = _format_prefixes([('', ligand) for ligand in ligands], omit_locants=True,
                            hierarchical_enclosures=True, separate_unlocanted=True)
    ending = 'sulfanylidene' if bond.GetBondTypeAsDouble() == 2 else 'sulfanyl'
    return (text + '-' if text else '') + f'lambda{valence}-' + ending


def _lambda6_sulfanylidene_prefix(mol, sulfur_idx: int, parent_idx: int) -> str:
    """Render a neutral S(VI)=N attachment with its two single-bond ligands."""
    sulfur = mol.GetAtomWithIdx(sulfur_idx)
    parent = mol.GetAtomWithIdx(parent_idx)
    bond = mol.GetBondBetweenAtoms(sulfur_idx, parent_idx)
    if (
        sulfur.GetSymbol() != "S" or sulfur.GetFormalCharge() or sulfur.IsInRing()
        or sulfur.GetIsotope() or sulfur.GetTotalNumHs()
        or parent.GetSymbol() != "N" or parent.GetFormalCharge()
        or bond is None or bond.GetBondType() != Chem.BondType.DOUBLE
    ):
        return ""
    oxo_count = 0
    ligands = []
    for neighbor in sulfur.GetNeighbors():
        idx = neighbor.GetIdx()
        if idx == parent_idx:
            continue
        bond = mol.GetBondBetweenAtoms(sulfur_idx, idx)
        if neighbor.GetFormalCharge() or neighbor.GetIsotope():
            return ""
        if neighbor.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            if neighbor.GetDegree() != 1:
                return ""
            oxo_count += 1
        elif neighbor.GetSymbol() in {"C", "N", "O"} and bond.GetBondType() == Chem.BondType.SINGLE:
            ligand = branch_prefix(mol, idx, sulfur_idx)
            if not ligand:
                return ""
            ligands.append(ligand)
        else:
            return ""
    if oxo_count != 1 or len(ligands) != 2:
        return ""
    # Keep ligand boundaries explicit: methyl + amino is not methylamino.
    text = "dimethyl" if ligands == ["methyl", "methyl"] else "".join(
        f"({ligand})" for ligand in sorted(ligands)
    )
    return text + "(oxo)-lambda6-sulfanylidene"


def _apply_sulfinyl_stereo(mol, sulfur_idx: int, double_o: int, prefix: str) -> str:
    if double_o != 1 or not prefix:
        return prefix
    descriptor = _sulfinyl_stereo_descriptor(mol, sulfur_idx)
    return f"({descriptor})-{prefix}" if descriptor else prefix


def _apply_sulfur_imidoyl_stereo(mol, sulfur_idx: int, prefix: str) -> str:
    if not prefix:
        return prefix
    descriptor = _sulfinyl_stereo_descriptor(mol, sulfur_idx)
    return f"({descriptor})-{prefix}" if descriptor else prefix


def _sulfinyl_stereo_descriptor(mol, sulfur_idx: int) -> str:
    if Chem is None:
        return ""
    atom = mol.GetAtomWithIdx(sulfur_idx)
    if atom.GetSymbol() != "S" or atom.GetFormalCharge() != 0:
        return ""
    for atom_idx, descriptor in assigned_atom_stereochemistry(mol, Chem, atom_order="idx"):
        if atom_idx == sulfur_idx and descriptor in {"R", "S"}:
            return descriptor
    return ""


def _retained_sulfonyl_prefix(mol, ligand_idx: int, sulfur_idx: int) -> str:
    ligand = mol.GetAtomWithIdx(ligand_idx)
    if ligand.GetSymbol() == "C":
        if _is_methyl_ligand(mol, ligand_idx, sulfur_idx):
            return "methanesulfonyl"
        if _is_trifluoromethyl_ligand(mol, ligand_idx, sulfur_idx):
            return "trifluoromethanesulfonyl"
    if ligand.IsInRing():
        ring = _ring_substituent_prefix(mol, ligand_idx, sulfur_idx)
        if ring in {"p-tolyl", "4-methylphenyl"}:
            return "4-methylbenzenesulfonyl"
    return ""


def _is_methyl_ligand(mol, carbon_idx: int, parent_idx: int) -> bool:
    atom = mol.GetAtomWithIdx(carbon_idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
        return False
    bond = mol.GetBondBetweenAtoms(carbon_idx, parent_idx)
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return False
    return not [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1
    ]


def _is_trifluoromethyl_ligand(mol, carbon_idx: int, parent_idx: int) -> bool:
    atom = mol.GetAtomWithIdx(carbon_idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
        return False
    bond = mol.GetBondBetweenAtoms(carbon_idx, parent_idx)
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return False
    fluorines = []
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        nbr_bond = mol.GetBondBetweenAtoms(carbon_idx, nidx)
        if nbr.GetSymbol() == "F" and not nbr.GetIsotope() and nbr_bond.GetBondType() == Chem.BondType.SINGLE:
            fluorines.append(nidx)
        elif nbr.GetAtomicNum() > 1:
            return False
    return len(fluorines) == 3


def _sulfimidoyl_prefix(mol, idx: int, parent_idx: int) -> str:
    sulfur = mol.GetAtomWithIdx(idx)
    attachment = mol.GetBondBetweenAtoms(idx, parent_idx)
    # Sulfinimidoyl owns exactly S(IV): one imino double bond and two
    # single-bond attachments, with no residual sulfur hydrogen.  A further
    # double ligand belongs to a different bonding-number construction and
    # must not disappear when this shorter acid-derived word is considered.
    if (sulfur.GetAtomicNum() != 16 or sulfur.GetFormalCharge()
            or sulfur.GetNumRadicalElectrons() or sulfur.GetTotalValence() != 4
            or sulfur.GetTotalNumHs(includeNeighbors=True)
            or attachment is None or attachment.GetBondType() != Chem.BondType.SINGLE):
        return ""
    imino = []
    single_groups = []
    for nbr in sulfur.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if nbr.GetSymbol() == "N" and bond.GetBondType() == Chem.BondType.DOUBLE:
            imino.append(nidx)
        elif bond.GetBondType() == Chem.BondType.SINGLE:
            single_groups.append(nidx)
        else:
            return ""
    if len(imino) != 1 or len(single_groups) != 1:
        return ""
    s_group = branch_prefix(mol, single_groups[0], idx) or substituent_from_root(mol, single_groups[0], idx)
    if not s_group:
        return ""
    n_group = _imino_n_substituent_name(mol, imino[0], idx)
    if n_group is None:
        return ""
    if not n_group:
        return _apply_sulfur_imidoyl_stereo(mol, idx, f"{s_group}sulfinimidoyl")
    prefix = f"N-({n_group})-S-{_sulfur_imidoyl_s_group(s_group)}sulfinimidoyl"
    return _apply_sulfur_imidoyl_stereo(mol, idx, prefix)


def _metal_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    symbol = atom.GetSymbol()
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    heavy = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1
    ]
    if symbol in {"Li", "Na", "K"} and not heavy:
        return {"Li": "lithio", "Na": "sodio", "K": "potassio"}[symbol]
    if symbol == "Mg" and len(heavy) == 1:
        ligand = mol.GetAtomWithIdx(heavy[0])
        if ligand.GetSymbol() in HALO:
            return HALO[ligand.GetSymbol()] + "magnesio"
    return ""


def _phosphorus_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "P" or atom.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    from ..phosphorus_acyl_prefixes import phosphorus_acyl_prefix
    acyl = phosphorus_acyl_prefix(mol, Chem, idx, parent_idx)
    if acyl is not None:
        return acyl.name
    explicit_anion = _anionic_phosphorus_ligand_prefix(mol, idx, parent_idx)
    if explicit_anion:
        return explicit_anion
    mapped = _mapped_phosphorus_prefix(mol, idx, parent_idx)
    if mapped:
        return mapped
    double_oxygen_count = 0
    double_sulfur_count = 0
    hydroxy_count = 0
    oxido_count = 0
    other_heavy = []
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_oxygen_count += 1
        elif (
            nbr.GetSymbol() == "S"
            and bond.GetBondType() == Chem.BondType.DOUBLE
            and not [
                atom.GetIdx()
                for atom in nbr.GetNeighbors()
                if atom.GetIdx() != idx and atom.GetAtomicNum() > 1
            ]
        ):
            double_sulfur_count += 1
        elif (
            nbr.GetSymbol() == "O"
            and bond.GetBondType() == Chem.BondType.SINGLE
            and len(nbr.GetNeighbors()) == 1
        ):
            if nbr.GetFormalCharge() == 0 and nbr.GetTotalNumHs() == 1:
                hydroxy_count += 1
            elif nbr.GetFormalCharge() < 0:
                oxido_count += 1
            else:
                return ""
        elif nbr.GetAtomicNum() > 1:
            if nbr.GetSymbol() not in {"C", "N", "O", "F", "Cl", "Br", "I"}:
                return ""
            other_heavy.append(nidx)
    thioyl = _phosphorus_thioyl_prefix(
        mol,
        idx,
        parent_idx,
        double_oxygen_count,
        double_sulfur_count,
        hydroxy_count,
        oxido_count,
        other_heavy,
    )
    if thioyl:
        return thioyl
    if any(mol.GetAtomWithIdx(nidx).GetSymbol() == "N" for nidx in other_heavy):
        return ""
    if double_oxygen_count == 0 and hydroxy_count == 0 and oxido_count == 0:
        return _substituted_hydride_prefix(mol, other_heavy, idx, "phosphanyl")
    if (
        double_oxygen_count == 2
        and hydroxy_count == 0
        and oxido_count == 0
        and not other_heavy
        and atom.GetTotalNumHs() == 0
    ):
        if _phosphoryl_parent_atom_has_direct_carbon_double_bond(mol, idx, parent_idx):
            return ""
        return "dioxophosphanyl"
    if double_oxygen_count != 1:
        return ""
    if not other_heavy:
        if (double_oxygen_count == 1 and hydroxy_count == 0 and oxido_count == 0
                and atom.GetTotalNumHs() == 2 and atom.GetTotalValence() == 5
                and not atom.IsInRing() and not atom.GetIsotope()):
            return 'phosphinoyl'
        if hydroxy_count == 2 and oxido_count == 0:
            return "phosphono"
        if hydroxy_count == 0 and oxido_count == 2:
            return "phosphonato"
        return ""
    ligand_names = []
    for nidx in other_heavy:
        ligand = branch_prefix(mol, nidx, idx)
        if not ligand:
            return ""
        ligand_names.append(ligand)
    if hydroxy_count or oxido_count:
        if hydroxy_count + oxido_count != 1:
            return ""
        if not ligand_names:
            return ""
        if any(_subtree_has_assigned_stereo(mol, nidx, idx) for nidx in other_heavy):
            return ""
        modifier = "hydroxy" if hydroxy_count else "oxido"
        ligand_text = _phosphoryl_ligand_text(ligand_names)
        return ligand_text + modifier + "phosphoryl" if ligand_text else ""
    if len(ligand_names) < 2:
        return ""
    if mol.GetAtomWithIdx(parent_idx).GetSymbol() == "N":
        return ""
    if len(set(ligand_names)) > 1 and any(_is_simple_phosphoryl_alkyl_or_alkoxy(name) for name in ligand_names):
        return ""
    ligand_text = _phosphoryl_ligand_text(ligand_names)
    return ligand_text + "phosphoryl" if ligand_text else ""


def _mapped_phosphorus_prefix(mol, idx: int, parent_idx: int) -> str:
    """Keep two proved neutral P(III/V) ligands inside one P component.

    Ordinary carbon and organyloxy ligands use their mapped simple/compound
    nature. Hydroxy and terminal halogen ligands are explicit characteristic
    prefixes. Other valences, annotations and ligand grammars retain their
    existing writers.
    """
    atom = mol.GetAtomWithIdx(idx)
    bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if (atom.GetAtomicNum() != 15 or atom.IsInRing() or atom.GetFormalCharge()
            or atom.GetIsotope() or atom.GetNumRadicalElectrons()
            or atom.GetTotalValence() not in {3, 5} or atom.GetTotalNumHs()
            or atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
            or bond is None or bond.GetBondTypeAsDouble() != 1):
        return ''
    from ..organyl_prefix_components import organyl_prefix_nature
    from ..prefix_sequences import PrefixUnit, mononuclear_prefix_sequence
    units = []
    oxo = 0
    for neighbor in atom.GetNeighbors():
        nidx = neighbor.GetIdx()
        if nidx == parent_idx:
            continue
        link = mol.GetBondBetweenAtoms(idx, nidx)
        if (neighbor.GetFormalCharge() or neighbor.GetIsotope()
                or neighbor.GetNumRadicalElectrons()):
            return ''
        if (neighbor.GetAtomicNum() == 8 and link.GetBondTypeAsDouble() == 2
                and neighbor.GetDegree() == 1 and not neighbor.GetTotalNumHs()):
            oxo += 1
            continue
        if link.GetBondTypeAsDouble() != 1:
            return ''
        if neighbor.GetAtomicNum() == 6:
            name = branch_prefix(mol, nidx, idx)
            nature = organyl_prefix_nature(mol, nidx, collect_subtree(mol, nidx, idx), name)
        elif neighbor.GetAtomicNum() == 8:
            distal = [n for n in neighbor.GetNeighbors() if n.GetIdx() != idx and n.GetAtomicNum() > 1]
            if not distal and neighbor.GetTotalNumHs() == 1:
                name, nature = 'hydroxy', 'simple'
            elif len(distal) == 1 and distal[0].GetAtomicNum() == 6 and not neighbor.GetTotalNumHs():
                root = distal[0].GetIdx()
                organyl = branch_prefix(mol, root, nidx)
                nature = organyl_prefix_nature(mol, root, collect_subtree(mol, root, nidx), organyl)
                name = branch_prefix(mol, nidx, idx)
            else:
                return ''
        elif neighbor.GetSymbol() in HALO and neighbor.GetDegree() == 1:
            name, nature = HALO[neighbor.GetSymbol()], 'simple'
        else:
            return ''
        if not name or nature is None:
            return ''
        units.append(PrefixUnit(name, nature))
    valence = int(atom.GetTotalValence())
    if oxo != (1 if valence == 5 else 0) or len(units) != 2:
        return ''
    text = mononuclear_prefix_sequence(units)
    parent = 'phosphoryl' if valence == 5 else 'phosphanyl'
    return '(' + text.text + parent + ')' if text is not None else ''


def _anionic_phosphorus_ligand_prefix(mol, idx: int, parent_idx: int) -> str:
    """Keep anionic O ligands and the actual P bonding number explicit."""
    atom = mol.GetAtomWithIdx(idx)
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if (
        atom.GetSymbol() != "P" or atom.GetFormalCharge() or atom.GetIsotope() or atom.IsInRing()
        or parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE
        or int(atom.GetTotalValence()) not in {3, 5}
    ):
        return ""
    neighbors = [n for n in atom.GetNeighbors() if n.GetIdx() != parent_idx and n.GetAtomicNum() > 1]
    if not any(n.GetSymbol() == "O" and n.GetFormalCharge() == -1 for n in neighbors):
        return ""
    ligands = []
    for neighbor in neighbors:
        nidx = neighbor.GetIdx()
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if neighbor.GetIsotope() or neighbor.GetFormalCharge() not in {0, -1}:
            return ""
        if bond.GetBondType() == Chem.BondType.SINGLE:
            if neighbor.GetFormalCharge() and (
                neighbor.GetSymbol() != "O" or neighbor.GetDegree() != 1 or neighbor.GetTotalNumHs()
            ):
                return ""
            ligand = branch_prefix(mol, nidx, idx)
        elif neighbor.GetSymbol() == "O" and not neighbor.GetFormalCharge() and neighbor.GetDegree() == 1 and bond.GetBondType() == Chem.BondType.DOUBLE:
            ligand = "oxo"
        else:
            return ""
        if not ligand:
            return ""
        ligands.append(ligand)
    if ligands == ["oxidanidyl", "oxidanidyl"] and int(atom.GetTotalValence()) == 3:
        return "(bis(oxidanidyl)phosphanyl)"
    text = "".join(f"({ligand})" for ligand in sorted(ligands))
    hydride = "-lambda5-phosphanyl" if int(atom.GetTotalValence()) == 5 else "phosphanyl"
    return f"({text}{hydride})"


def _phosphorus_thioyl_prefix(
    mol,
    phosphorus_idx: int,
    parent_idx: int,
    double_oxygen_count: int,
    double_sulfur_count: int,
    hydroxy_count: int,
    oxido_count: int,
    other_heavy: Sequence[int],
) -> str:
    if double_sulfur_count != 1 or double_oxygen_count != 0:
        return ""
    if mol.GetAtomWithIdx(phosphorus_idx).GetTotalNumHs() != 0:
        return ""
    ligand_names = []
    ligand_names.extend(["hydroxy"] * hydroxy_count)
    ligand_names.extend(["oxido"] * oxido_count)
    for nidx in other_heavy:
        ligand = _phosphorus_thioyl_ligand_name(mol, nidx, phosphorus_idx)
        if not ligand:
            return ""
        ligand_names.append(ligand)
    if not ligand_names:
        return ""
    ligand_text = _phosphoryl_ligand_text(ligand_names)
    if not ligand_text:
        return ""
    return ligand_text + "phosphorothioyl"


def _phosphorus_thioyl_ligand_name(mol, ligand_idx: int, phosphorus_idx: int) -> str:
    ligand = mol.GetAtomWithIdx(ligand_idx)
    if ligand.GetSymbol() in HALO:
        bond = mol.GetBondBetweenAtoms(ligand_idx, phosphorus_idx)
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return ""
        if any(nbr.GetIdx() != phosphorus_idx and nbr.GetAtomicNum() > 1 for nbr in ligand.GetNeighbors()):
            return ""
        return HALO[ligand.GetSymbol()]
    if ligand.GetSymbol() == "N":
        return _nitrogen_prefix(mol, ligand_idx, phosphorus_idx)
    if ligand.GetSymbol() == "O":
        return _oxygen_prefix(mol, ligand_idx, phosphorus_idx)
    return branch_prefix(mol, ligand_idx, phosphorus_idx)


def _group14_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    suffix = _GROUP14_SUBSTITUENT_SUFFIX.get(atom.GetSymbol())
    if not suffix or atom.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    if atom.GetSymbol() == "Si":
        if _is_trimethylsilyl_group(mol, idx, parent_idx):
            return "tms"
        if _is_tbdms_group(mol, idx, parent_idx):
            return "tbdms"
    heavy = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1
    ]
    return _substituted_hydride_prefix(mol, heavy, idx, suffix)


def _is_trimethylsilyl_group(mol, silicon_idx: int, blocked_idx: int) -> bool:
    silicon = mol.GetAtomWithIdx(silicon_idx)
    if silicon.GetSymbol() != "Si" or silicon.GetFormalCharge() != 0:
        return False
    methyls = [
        nbr.GetIdx()
        for nbr in silicon.GetNeighbors()
        if nbr.GetIdx() != blocked_idx and nbr.GetSymbol() == "C"
    ]
    if len(methyls) != 3:
        return False
    return all(_is_terminal_methyl_ligand(mol, methyl_idx, silicon_idx) for methyl_idx in methyls)


def _is_tbdms_group(mol, silicon_idx: int, blocked_idx: int) -> bool:
    silicon = mol.GetAtomWithIdx(silicon_idx)
    if silicon.GetSymbol() != "Si" or silicon.GetFormalCharge() != 0:
        return False
    carbon_ligands = [
        nbr.GetIdx()
        for nbr in silicon.GetNeighbors()
        if nbr.GetIdx() != blocked_idx and nbr.GetSymbol() == "C"
    ]
    if len(carbon_ligands) != 3:
        return False
    methyls = [idx for idx in carbon_ligands if _is_terminal_methyl_ligand(mol, idx, silicon_idx)]
    tert_butyl = [idx for idx in carbon_ligands if _is_tert_butyl_ligand(mol, idx, silicon_idx)]
    return len(methyls) == 2 and len(tert_butyl) == 1


def _is_terminal_methyl_ligand(mol, carbon_idx: int, center_idx: int) -> bool:
    carbon = mol.GetAtomWithIdx(carbon_idx)
    if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0:
        return False
    if mol.GetBondBetweenAtoms(carbon_idx, center_idx).GetBondType() != Chem.BondType.SINGLE:
        return False
    heavy = [
        nbr.GetIdx()
        for nbr in carbon.GetNeighbors()
        if nbr.GetIdx() != center_idx and nbr.GetAtomicNum() > 1
    ]
    return not heavy


def _is_tert_butyl_ligand(mol, carbon_idx: int, center_idx: int) -> bool:
    carbon = mol.GetAtomWithIdx(carbon_idx)
    if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0:
        return False
    if mol.GetBondBetweenAtoms(carbon_idx, center_idx).GetBondType() != Chem.BondType.SINGLE:
        return False
    methyls = [
        nbr.GetIdx()
        for nbr in carbon.GetNeighbors()
        if nbr.GetIdx() != center_idx and nbr.GetSymbol() == "C"
    ]
    if len(methyls) != 3:
        return False
    return all(_is_terminal_methyl_ligand(mol, methyl_idx, carbon_idx) for methyl_idx in methyls)


def _selenium_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "Se" or atom.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None:
        return ""
    if parent_bond.GetBondType() == Chem.BondType.DOUBLE:
        heavy = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetIdx() != parent_idx]
        return "selenoxo" if not heavy else ""
    if parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    heavy = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1
    ]
    if not heavy:
        return "selanyl"
    ligand_text = _substituent_ligand_text(mol, heavy, idx)
    return ligand_text + "selanyl" if ligand_text else ""


def _substituted_hydride_prefix(mol, ligand_indices: Sequence[int], center_idx: int, suffix: str) -> str:
    ligand_text = _substituent_ligand_text(mol, ligand_indices, center_idx)
    if ligand_indices and not ligand_text:
        return ""
    return ligand_text + suffix


def _substituent_ligand_text(mol, ligand_indices: Sequence[int], center_idx: int) -> str:
    ligands = []
    for nidx in ligand_indices:
        ligand = branch_prefix(mol, nidx, center_idx)
        if not ligand:
            return ""
        ligands.append(ligand)
    return _ligand_text(ligands)


def _ligand_text(ligands: Sequence[str]) -> str:
    counts = Counter(ligands)
    parts = []
    for ligand in sorted(counts, key=_phosphoryl_ligand_sort_key):
        count = counts[ligand]
        unit = _phosphoryl_ligand_unit(ligand)
        if count == 1:
            parts.append(unit)
        elif _needs_complex_phosphoryl_ligand(ligand):
            parts.append(f"{MULT.get(count, '')}({unit})")
        else:
            parts.append(MULT.get(count, "") + ligand)
    return "".join(parts)


def _phosphoryl_ligand_text(ligands: Sequence[str]) -> str:
    return _ligand_text(ligands)


def _phosphoryl_parent_atom_has_direct_carbon_double_bond(mol, phosphorus_idx: int, parent_idx: int) -> bool:
    parent = mol.GetAtomWithIdx(parent_idx)
    if parent.GetSymbol() != "C":
        return False
    for bond in parent.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        other = bond.GetOtherAtom(parent)
        if other.GetIdx() != phosphorus_idx and other.GetSymbol() == "C":
            return True
    return False


def _subtree_has_assigned_stereo(mol, start_idx: int, blocked_idx: int) -> bool:
    atoms = collect_subtree(mol, start_idx, blocked_idx)
    for atom_idx in atoms:
        if mol.GetAtomWithIdx(atom_idx).GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            return True
    for bond in iter_bonds(mol):
        if bond.GetStereo() == Chem.BondStereo.STEREONONE:
            continue
        if bond.GetBeginAtomIdx() in atoms and bond.GetEndAtomIdx() in atoms:
            return True
    return False


def _phosphoryl_ligand_unit(ligand: str) -> str:
    return f"({ligand})" if _needs_complex_phosphoryl_ligand(ligand) else ligand


def _needs_complex_phosphoryl_ligand(ligand: str) -> bool:
    return "-" in ligand or ligand[:1].isdigit() or ligand.startswith("(")


def _phosphoryl_ligand_sort_key(ligand: str) -> tuple[int, str]:
    simple_alkyl = {
        "methyl",
        "ethyl",
        "propyl",
        "butyl",
        "pentyl",
        "hexyl",
        "heptyl",
        "octyl",
        "nonyl",
        "decyl",
    }
    if ligand in simple_alkyl:
        return (0, ligand)
    if _needs_complex_phosphoryl_ligand(ligand):
        return (2, ligand)
    return (1, ligand)


def _is_simple_phosphoryl_alkyl_or_alkoxy(ligand: str) -> bool:
    simple = {
        "methyl",
        "ethyl",
        "propyl",
        "butyl",
        "pentyl",
        "hexyl",
        "heptyl",
        "octyl",
        "nonyl",
        "decyl",
        "methoxy",
        "ethoxy",
        "propoxy",
        "butoxy",
        "pentoxy",
        "hexoxy",
        "heptoxy",
        "octoxy",
        "nonoxy",
        "decoxy",
    }
    return ligand in simple


def _halosulfur_prefix(double_o: int, halogens: Sequence[str], non_parent_count: int) -> str:
    if len(halogens) != 1:
        return ""
    if double_o == 0 and non_parent_count == 1:
        return HALO[halogens[0]] + "sulfanyl"
    return _HALOSULFUR_OXO_PREFIXES.get(double_o, {}).get(halogens[0], "")


def _lambda6_tetrafluorosulfanyl_prefix(mol, sulfur_idx: int, parent_idx: int, non_parent: Sequence[int]) -> str:
    parent_bond = mol.GetBondBetweenAtoms(sulfur_idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""

    fluorines: list[int] = []
    ligands: list[int] = []
    for nidx in non_parent:
        bond = mol.GetBondBetweenAtoms(sulfur_idx, nidx)
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return ""
        if _is_terminal_fluorine_ligand(mol, nidx, sulfur_idx):
            fluorines.append(nidx)
        else:
            ligands.append(nidx)
    if len(fluorines) != 4 or len(ligands) != 1:
        return ""

    ligand = _lambda6_tetrafluorosulfanyl_ligand_prefix(mol, ligands[0], sulfur_idx)
    return f"{ligand}tetrafluoro-lambda6-sulfanyl" if ligand else ""


def _lambda6_tetrafluorosulfanyl_ligand_prefix(mol, ligand_idx: int, sulfur_idx: int) -> str:
    ligand = mol.GetAtomWithIdx(ligand_idx)
    bond = mol.GetBondBetweenAtoms(ligand_idx, sulfur_idx)
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    if ligand.GetSymbol() in {"Cl", "Br", "I"}:
        return HALO[ligand.GetSymbol()] if _is_terminal_halogen_ligand(mol, ligand_idx, sulfur_idx) else ""
    if ligand.GetSymbol() == "C":
        return _lambda6_fluoroethyl_ligand_prefix(mol, ligand_idx, sulfur_idx)
    return ""


def _lambda6_dioxythio_prefix(mol, sulfur_idx: int, parent_idx: int, non_parent: Sequence[int]) -> str:
    sulfur = mol.GetAtomWithIdx(sulfur_idx)
    if sulfur.GetSymbol() != "S" or sulfur.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(sulfur_idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    if len(non_parent) != 1:
        return ""

    oxygen_idx = non_parent[0]
    oxygen = mol.GetAtomWithIdx(oxygen_idx)
    sulfur_oxygen_bond = mol.GetBondBetweenAtoms(sulfur_idx, oxygen_idx)
    if (
        oxygen.GetSymbol() != "O"
        or oxygen.GetFormalCharge() != 0
        or sulfur_oxygen_bond is None
        or sulfur_oxygen_bond.GetBondType() != Chem.BondType.SINGLE
    ):
        return ""

    distal_oxygens = []
    for neighbor in oxygen.GetNeighbors():
        nidx = neighbor.GetIdx()
        if nidx == sulfur_idx:
            continue
        bond = mol.GetBondBetweenAtoms(oxygen_idx, nidx)
        if (
            neighbor.GetSymbol() == "O"
            and neighbor.GetFormalCharge() == 0
            and bond is not None
            and bond.GetBondType() == Chem.BondType.SINGLE
        ):
            distal_oxygens.append(nidx)
        elif neighbor.GetAtomicNum() > 1:
            return ""
    if len(distal_oxygens) != 1:
        return ""

    ligand = _dioxythio_ligand_prefix(mol, distal_oxygens[0], oxygen_idx)
    return f"{_sulfur_imidoyl_s_group(ligand)}dioxy-lambda6-thio" if ligand else ""


def _dioxythio_ligand_prefix(mol, oxygen_idx: int, blocked_idx: int) -> str:
    oxygen = mol.GetAtomWithIdx(oxygen_idx)
    if oxygen.GetSymbol() != "O" or oxygen.GetFormalCharge() != 0:
        return ""
    ligands = []
    for neighbor in oxygen.GetNeighbors():
        nidx = neighbor.GetIdx()
        if nidx == blocked_idx:
            continue
        bond = mol.GetBondBetweenAtoms(oxygen_idx, nidx)
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return ""
        if neighbor.GetAtomicNum() > 1:
            ligands.append(nidx)
    if len(ligands) != 1:
        return ""

    ligand_idx = ligands[0]
    ligand = mol.GetAtomWithIdx(ligand_idx)
    if ligand.GetSymbol() in HALO:
        return HALO[ligand.GetSymbol()] if _is_terminal_halogen_ligand(mol, ligand_idx, oxygen_idx) else ""
    if ligand.IsInRing():
        return _ring_substituent_prefix(mol, ligand_idx, oxygen_idx)
    return branch_prefix(mol, ligand_idx, oxygen_idx) or substituent_from_root(mol, ligand_idx, oxygen_idx)


def _lambda6_fluoroethyl_ligand_prefix(mol, carbon_idx: int, sulfur_idx: int) -> str:
    alpha = mol.GetAtomWithIdx(carbon_idx)
    if alpha.GetSymbol() != "C" or alpha.GetFormalCharge() != 0 or alpha.IsInRing() or alpha.GetIsotope():
        return ""
    beta_carbons = []
    alpha_fluorines = []
    for neighbor in alpha.GetNeighbors():
        nidx = neighbor.GetIdx()
        if nidx == sulfur_idx:
            continue
        bond = mol.GetBondBetweenAtoms(carbon_idx, nidx)
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return ""
        if _is_terminal_fluorine_ligand(mol, nidx, carbon_idx):
            alpha_fluorines.append(nidx)
        elif neighbor.GetSymbol() == "C":
            beta_carbons.append(nidx)
        elif neighbor.GetAtomicNum() > 1:
            return ""
    if len(alpha_fluorines) != 2 or len(beta_carbons) != 1:
        return ""

    beta_idx = beta_carbons[0]
    beta = mol.GetAtomWithIdx(beta_idx)
    if beta.GetFormalCharge() != 0 or beta.IsInRing() or beta.GetIsotope():
        return ""
    beta_bond = mol.GetBondBetweenAtoms(carbon_idx, beta_idx)
    if beta_bond is None or beta_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""

    beta_fluorines = []
    terminal_halogen = ""
    for neighbor in beta.GetNeighbors():
        nidx = neighbor.GetIdx()
        if nidx == carbon_idx:
            continue
        bond = mol.GetBondBetweenAtoms(beta_idx, nidx)
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return ""
        if _is_terminal_fluorine_ligand(mol, nidx, beta_idx):
            beta_fluorines.append(nidx)
        elif neighbor.GetSymbol() in {"Cl", "Br", "I"} and _is_terminal_halogen_ligand(mol, nidx, beta_idx):
            if terminal_halogen:
                return ""
            terminal_halogen = HALO[neighbor.GetSymbol()]
        elif neighbor.GetAtomicNum() > 1:
            return ""
    if len(beta_fluorines) != 2:
        return ""
    if terminal_halogen:
        return f"2-{terminal_halogen}tetrafluoroethyl"
    return "1,1,2,2-tetrafluoroethyl"


def _is_terminal_fluorine_ligand(mol, atom_idx: int, parent_idx: int) -> bool:
    atom = mol.GetAtomWithIdx(atom_idx)
    return (
        atom.GetSymbol() == "F"
        and atom.GetFormalCharge() == 0
        and not atom.GetIsotope()
        and _is_terminal_halogen_ligand(mol, atom_idx, parent_idx)
    )


def _is_terminal_halogen_ligand(mol, atom_idx: int, parent_idx: int) -> bool:
    atom = mol.GetAtomWithIdx(atom_idx)
    if atom.GetSymbol() not in HALO or atom.GetFormalCharge() != 0 or atom.GetIsotope():
        return False
    bond = mol.GetBondBetweenAtoms(atom_idx, parent_idx)
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return False
    return not [
        neighbor.GetIdx()
        for neighbor in atom.GetNeighbors()
        if neighbor.GetIdx() != parent_idx and neighbor.GetAtomicNum() > 1
    ]


def _sulfur_oxo_substituent_suffix(double_o: int) -> str:
    if double_o == 2:
        return suffix_prefix_term("sulfone", "sulfonyl")
    if double_o == 1:
        return suffix_prefix_term("sulfoxide", "sulfinyl")
    return _SULFUR_OXO_SUBSTITUENT_SUFFIX.get(double_o, "sulfanyl")


def _sulfur_thioxo_substituent_prefix(
    mol,
    sulfur_idx: int,
    parent_idx: int,
    double_o: int,
    double_s: int,
    heavy: Sequence[int],
) -> str:
    if double_s != 1 or len(heavy) != 1:
        return ""
    bond = mol.GetBondBetweenAtoms(sulfur_idx, parent_idx)
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    ligand = _sulfur_thioxo_ligand_prefix(mol, heavy[0], sulfur_idx)
    if not ligand:
        return ""
    if double_o == 1:
        return ligand + "sulfonothioyl"
    if double_o == 0:
        return ligand + "sulfanethionyl"
    return ""


def _sulfur_thioxo_ligand_prefix(mol, ligand_idx: int, sulfur_idx: int) -> str:
    ligand = mol.GetAtomWithIdx(ligand_idx)
    if ligand.GetSymbol() in HALO:
        return HALO[ligand.GetSymbol()]
    if ligand.IsInRing():
        return _ring_substituent_prefix(mol, ligand_idx, sulfur_idx)
    if ligand.GetSymbol() == "N":
        return _nitrogen_prefix(mol, ligand_idx, sulfur_idx)
    if ligand.GetSymbol() == "O":
        return _oxygen_prefix(mol, ligand_idx, sulfur_idx)
    return branch_prefix(mol, ligand_idx, sulfur_idx) or substituent_from_root(mol, ligand_idx, sulfur_idx)


def _sulfonimidoyl_prefix(mol, idx: int, parent_idx: int, double_o: int) -> str:
    if double_o != 1:
        return ""
    imino = []
    single_groups = []
    for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx or nbr.GetSymbol() == "O":
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if nbr.GetSymbol() == "N" and bond.GetBondType() == Chem.BondType.DOUBLE:
            imino.append(nidx)
        elif bond.GetBondType() == Chem.BondType.SINGLE:
            single_groups.append(nidx)
    if len(imino) != 1 or len(single_groups) != 1:
        return ""

    single_atom = mol.GetAtomWithIdx(single_groups[0])
    if single_atom.GetSymbol() in HALO:
        s_group = HALO[single_atom.GetSymbol()]
    elif single_atom.IsInRing():
        s_group = _ring_substituent_prefix(mol, single_groups[0], idx)
    else:
        s_group = branch_prefix(mol, single_groups[0], idx) or substituent_from_root(mol, single_groups[0], idx)
    if not s_group:
        return ""
    n_group = _imino_n_substituent_name(mol, imino[0], idx)
    if n_group is None:
        return ""
    if not n_group:
        return _apply_sulfur_imidoyl_stereo(mol, idx, f"{s_group}sulfonimidoyl")
    prefix = f"N-({n_group})-S-{_sulfur_imidoyl_s_group(s_group)}sulfonimidoyl"
    return _apply_sulfur_imidoyl_stereo(mol, idx, prefix)


def _imino_n_substituent_name(mol, nidx: int, sulfur_idx: int) -> str | None:
    heavy = [n.GetIdx() for n in mol.GetAtomWithIdx(nidx).GetNeighbors() if n.GetIdx() != sulfur_idx]
    if not heavy:
        return ""
    if len(heavy) != 1:
        return None
    attached = heavy[0]
    carbamoyl = _carbamoyl_substituent_name(mol, attached, nidx)
    if carbamoyl:
        return carbamoyl
    if mol.GetAtomWithIdx(attached).IsInRing():
        name = _ring_substituent_prefix(mol, attached, nidx)
        return name if name else None
    name = branch_prefix(mol, attached, nidx) or substituent_from_root(mol, attached, nidx)
    if not name:
        return None
    return name


def _carbamoyl_substituent_name(
    mol,
    cidx: int,
    nidx: int,
    *,
    localize_ligand_stereo: bool = False,
) -> str:
    atom = mol.GetAtomWithIdx(cidx)
    if atom.GetSymbol() != "C":
        return ""
    has_double_o = False
    amine = None
    for nbr in atom.GetNeighbors():
        nbr_idx = nbr.GetIdx()
        if nbr_idx == nidx:
            continue
        bond = mol.GetBondBetweenAtoms(cidx, nbr_idx)
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            has_double_o = True
        elif nbr.GetSymbol() == "N" and bond.GetBondType() == Chem.BondType.SINGLE:
            amine = nbr_idx
        else:
            return ""
    if not has_double_o or amine is None:
        return ""
    if mol.GetAtomWithIdx(amine).GetFormalCharge() != 0:
        nitrogen = _nitrogen_prefix(mol, amine, cidx)
        return f"({nitrogen})carbonyl" if nitrogen else ""
    if mol.GetAtomWithIdx(amine).IsInRing():
        ring = _ring_substituent_prefix(mol, amine, cidx)
        if not ring:
            return ""
        # P-16.5 encloses the compound ring substituent as one unit before
        # concatenating carbonyl. Identify substitution on the original
        # ring system rather than inferring it from the rendered spelling.
        from ..mapped_ring_parents import ring_systems
        core = next((system for system in ring_systems(mol) if amine in system), None)
        if core and any(
            nbr.GetAtomicNum() > 1 and nbr.GetIdx() not in core and nbr.GetIdx() != cidx
            for i in core for nbr in mol.GetAtomWithIdx(i).GetNeighbors()
        ):
            from ..formatting import enclose_name_unit
            return enclose_name_unit(ring) + "carbonyl"
        if ring.startswith("("):
            return ring + "carbonyl"
        return f"{ring}carbonyl"
    nitrogen = mol.GetAtomWithIdx(amine)
    if (nitrogen.GetDegree() in {2, 3}
            or nitrogen.GetDegree() == 1 and mol.GetAtomWithIdx(nidx).GetAtomicNum() in {7, 8}):
        # P-65.1.6/P-65.2.1.5: construct the subordinate amide from its
        # original C/O/N and disjoint mapped N-ligands, after the cut is known.
        from ..carboxamide_prefixes import mapped_n_substituted_carbamoyl
        preferred = mapped_n_substituted_carbamoyl(mol, Chem, cidx, nidx, amine)
        if preferred is not None:
            return preferred[0]
    groups = []
    for nbr in mol.GetAtomWithIdx(amine).GetNeighbors():
        nbr_idx = nbr.GetIdx()
        if nbr_idx == cidx:
            continue
        if mol.GetBondBetweenAtoms(amine, nbr_idx).GetBondType() != Chem.BondType.SINGLE:
            # The amino/carbamoyl grammar below substitutes individual N-H
            # bonds. A cumulene such as N=C=O instead belongs to its complete
            # isocyanato construction in the ordinary nitrogen writer.
            return ""
        if _is_diazo_attachment(mol, nbr_idx, amine):
            name = "diazo"
        elif nbr.IsInRing():
            name = _ring_substituent_prefix(mol, nbr_idx, amine)
        else:
            name = branch_prefix(mol, nbr_idx, amine) or substituent_from_root(mol, nbr_idx, amine)
            if localize_ligand_stereo:
                alpha_stereo = _alpha_ligand_stereo_prefix(mol, nbr_idx, name)
                if alpha_stereo:
                    name = alpha_stereo + name
        if not name:
            return ""
        groups.append(name)
    if not groups:
        return "carbamoyl"
    if len(groups) == 1:
        return f"(({groups[0]})amino)carbonyl"
    counts = Counter(groups)
    parts = []
    for name in sorted(counts):
        count = counts[name]
        if count > 1:
            locants = ",".join("N" for _ in range(count))
            if "-" in name or name.startswith("("):
                parts.append(f"{locants}-{MULT.get(count, '')}({name})")
            else:
                parts.append(f"{locants}-{MULT.get(count, '')}{name}")
        else:
            if _carbamoyl_n_ligand_needs_parentheses(name):
                parts.append(f"N-({name})")
            else:
                parts.append(f"N-{name}")
    return "-".join(parts) + "carbamoyl"


def _carbamoyl_n_ligand_needs_parentheses(name: str) -> bool:
    return "-" in name or "(" in name or ")" in name


def substituent_from_root(mol, root_idx: int, blocked_idx: int) -> str:
    cache = mol_cache_dict(mol, "substituent_from_root_cache")
    cache_key = (
        root_idx,
        blocked_idx,
        mol.HasProp("_nispo_allow_fragment_substituent_fallback"),
        mol.GetProp("_nispo_fragment_substituent_depth") if mol.HasProp("_nispo_fragment_substituent_depth") else "",
    )
    if cache_key in cache:
        return cache[cache_key]
    if mol.GetAtomWithIdx(root_idx).GetSymbol() == "P":
        direct = _anionic_phosphorus_ligand_prefix(mol, root_idx, blocked_idx)
        if direct:
            cache[cache_key] = direct
            return direct
    if mol.GetAtomWithIdx(root_idx).GetSymbol() == "S":
        direct = _lambda6_sulfanylidene_prefix(mol, root_idx, blocked_idx)
        if direct:
            cache[cache_key] = direct
            return direct
    if mol.GetAtomWithIdx(root_idx).GetSymbol() == "C":
        direct = _carbonyl_substituent_prefix(mol, root_idx, blocked_idx)
        if direct:
            cache[cache_key] = direct
            return direct
        from .carbon import geminal_alkoxy_alkyl_prefix

        direct = geminal_alkoxy_alkyl_prefix(mol, root_idx, blocked_idx)
        if direct:
            cache[cache_key] = direct
            return direct
    atoms = collect_subtree(mol, root_idx, blocked_idx)
    paths = rooted_paths(mol, root_idx, atoms)
    if not paths:
        cache[cache_key] = ""
        return ""
    best = ""
    best_path: Sequence[int] = ()
    best_key = None
    for path in paths:
        name = _name_subtree_path(mol, path, atoms, blocked_idx)
        if not name:
            continue
        path_key = (-len(path), substituent_locants(mol, path, atoms, blocked_idx), name)
        if best_key is None or path_key < best_key:
            best_key = path_key
            best = name
            best_path = path
    if best:
        best = _with_substituent_atom_stereo_prefix(mol, best_path, best)
    cache[cache_key] = best
    return best


def _with_substituent_atom_stereo_prefix(mol, path: Sequence[int], name: str) -> str:
    """Decorate this rooted path; child descriptors belong to their children."""
    if not name:
        return name
    locant_by_atom = {idx: loc for loc, idx in enumerate(path, start=1)}
    assigned = assigned_atom_stereochemistry(mol, Chem, atom_order="idx", include_pseudo=True)
    pieces = {(locant_by_atom[idx], 'atom'): descriptor
              for idx, descriptor in assigned
              if idx in locant_by_atom and descriptor in {'R', 'S'}}
    from ..stereochemistry import _indexed_bond_stereochemistry
    for begin, end, descriptor in _indexed_bond_stereochemistry(mol, Chem):
        if begin not in locant_by_atom or end not in locant_by_atom:
            continue
        first, second = locant_by_atom[begin], locant_by_atom[end]
        if abs(first - second) == 1:
            pieces[min(first, second), 'bond'] = descriptor
    if not pieces:
        return name
    leading = re.match(r"^\((?P<fields>(?:\d+)?[RSrsEZ](?:,(?:\d+)?[RSrsEZ])*)\)-", name)
    if leading:
        for field in leading['fields'].split(','):
            match = re.fullmatch(r'(\d+)([RSrsEZ])', field)
            if match is None:
                return name  # The existing unlocanted operation owns its scope.
            key = (int(match[1]), 'bond' if match[2] in 'EZ' else 'atom')
            pieces.setdefault(key, match[2])
        name = name[leading.end():]
    text = ','.join(str(locant) + descriptor
                    for (locant, _kind), descriptor in sorted(pieces.items()))
    return '(' + text + ')-' + name


def _has_atom_stereo_prefix(name: str) -> bool:
    return bool(re.search(
        r"\((?=[^()]*[RS])(?:\d+[a-z]?'?)?[RSZE]"
        r"(?:,(?:\d+[a-z]?'?)?[RSZE])*\)-", name))


def _name_subtree_path(mol, path: Sequence[int], atoms: set[int], blocked_idx: int) -> str:
    path_set = set(path)
    prefixes = []
    for pos, idx in enumerate(path, start=1):
        atom = mol.GetAtomWithIdx(idx)
        if not can_be_in_parent_skeleton(mol, idx):
            return ""
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx == blocked_idx or nidx in path_set:
                continue
            if nbr.GetSymbol() == "H" and nbr.GetIsotope() == 0:
                continue
            if nidx not in atoms:
                continue
            prefix = branch_prefix(mol, nidx, idx)
            if not prefix:
                return ""
            prefixes.append((str(pos), prefix))
    if (len(path) == 3 and len(prefixes) == 1 and prefixes[0][0] == "3"
            and prefixes[0][1].endswith("yl")
            and tuple(mol.GetAtomWithIdx(i).GetAtomicNum() for i in path) == (6, 6, 8)
            and all(not mol.GetAtomWithIdx(i).IsInRing()
                    and not mol.GetAtomWithIdx(i).GetFormalCharge()
                    and not mol.GetAtomWithIdx(i).GetIsotope()
                    and not mol.GetAtomWithIdx(i).GetNumRadicalElectrons()
                    and not mol.GetAtomWithIdx(i).GetChiralTag()
                    and mol.GetAtomWithIdx(i).GetDegree() == 2
                    and all(b.GetBondTypeAsDouble() == 1 for b in mol.GetAtomWithIdx(i).GetBonds())
                    for i in path)):
        children = [n.GetIdx() for n in mol.GetAtomWithIdx(path[-1]).GetNeighbors()
                    if n.GetIdx() not in path_set]
        if len(children) == 1 and mol.GetAtomWithIdx(children[0]).IsInRing():
            from ..principal_carbonyl import configured_acyclic_methine_ring_hosts
            child_scope = frozenset(collect_subtree(mol, children[0], path[-1]))
            if (configured_acyclic_methine_ring_hosts(mol).intersection(child_scope)
                    and not child_scope.intersection(path_set)
                    and child_scope | path_set == set(atoms)):
                from .hetero_atoms import _organyloxy_component
                from ..prefix_presentation import enclose_complete_name_unit

                oxy = _organyloxy_component(prefixes[0][1], attachment_locanted=True)
                # P-29.4: the original two-carbon arm bears an organyloxy
                # substituent; a single ether oxygen does not extend its chain.
                return "2-" + enclose_complete_name_unit(oxy) + "ethyl"
    parent = parent_chain_name(mol, path, as_substituent=True)
    if not parent:
        return ""
    attachment_bond = mol.GetBondBetweenAtoms(path[0], blocked_idx)
    if attachment_bond is None:
        return ""
    attachment_order = attachment_bond.GetBondTypeAsDouble()
    if attachment_order in {2.0, 3.0}:
        # A rooted path describes the internal bonds of the substituent;
        # its bond to the parent is additional unsaturation.
        if not parent.endswith("yl"):
            return ""
        parent += "idene" if attachment_order == 2.0 else "idyne"
    elif attachment_order != 1.0:
        return ""
    repl = replacement_prefixes(mol, path)
    prefix_text = _format_prefixes(prefixes)
    if not repl and parent[0].isalpha() and any(
        name.endswith("ylidene") for _, name in prefixes
    ):
        # P-16.2.4.1: the ylidene prefix joins the letter-led chain root.
        # Locants already placed inside that root keep their own hyphens.
        return prefix_text + parent
    pieces = []
    if prefix_text:
        pieces.append(prefix_text)
    if repl:
        pieces.append(repl + parent)
    else:
        pieces.append(parent)
    return "-".join(pieces)


def substituent_locants(mol, path: Sequence[int], atoms: set[int], blocked_idx: int) -> list[int]:
    path_set = set(path)
    locs = []
    for pos, idx in enumerate(path, start=1):
        if mol.GetAtomWithIdx(idx).GetSymbol() != "C":
            locs.append(pos)
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nbr.GetSymbol() == "H" and nbr.GetIsotope() == 0:
                continue
            if nidx != blocked_idx and nidx in atoms and nidx not in path_set:
                locs.append(pos)
    return locs


def rooted_paths(mol, root_idx: int, atoms: set[int]) -> list[tuple[int, ...]]:
    cache = mol_cache_dict(mol, "rooted_paths_cache")
    cache_key = (root_idx, tuple(sorted(atoms)))
    cached = cache.get(cache_key)
    if cached is not None:
        return list(cached)
    if mol.GetAtomWithIdx(root_idx).IsInRing():
        path_atoms = set(atoms)
    else:
        path_atoms = {idx for idx in atoms if not mol.GetAtomWithIdx(idx).IsInRing()}
        path_atoms.add(root_idx)
    leaves, blocked_ends = [], set()
    for idx in path_atoms - {root_idx}:
        neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors() if n.GetIdx() in path_atoms]
        if len(neighbors) > 1:
            continue
        if can_be_in_parent_skeleton(mol, idx):
            leaves.append(idx)
        else:
            blocked_ends.update(neighbors)
    # A halogen or terminal oxo group belongs to the endpoint as a
    # detachable prefix. It does not extend the skeletal path.
    for idx in blocked_ends - {root_idx}:
        if can_be_in_parent_skeleton(mol, idx) and sum(
                n.GetIdx() in path_atoms and can_be_in_parent_skeleton(mol, n.GetIdx())
                for n in mol.GetAtomWithIdx(idx).GetNeighbors()) == 1:
            leaves.append(idx)
    if not leaves:
        leaves = [root_idx]
    paths = []
    best_len = 0
    for leaf in leaves:
        if leaf == root_idx:
            path = (root_idx,)
        else:
            path = Chem.rdmolops.GetShortestPath(mol, root_idx, leaf)
        if any(idx not in path_atoms or not can_be_in_parent_skeleton(mol, idx) for idx in path):
            continue
        if len(path) > best_len:
            best_len = len(path)
            paths.clear()
        if len(path) == best_len:
            paths.append(tuple(path))
    if not paths and can_be_in_parent_skeleton(mol, root_idx):
        paths.append((root_idx,))
    cache[cache_key] = tuple(paths)
    return paths


def collect_subtree(mol, start_idx: int, blocked_idx: int) -> set[int]:
    cache = mol_cache_dict(mol, "collect_subtree_cache")
    key = (start_idx, blocked_idx)
    cached = cache.get(key)
    if cached is not None:
        return set(cached)
    seen = {blocked_idx}
    stack = [start_idx]
    atoms = set()
    while stack:
        idx = stack.pop()
        if idx in seen:
            continue
        seen.add(idx)
        atoms.add(idx)
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nbr.GetSymbol() == "H" and nbr.GetIsotope() == 0:
                continue
            if nidx not in seen:
                stack.append(nidx)
    cache[key] = frozenset(atoms)
    return atoms


def _is_terminal_carbonyl_oxygen(atom) -> bool:
    if atom.GetSymbol() != "O":
        return False
    if atom.GetDegree() != 1:
        return False
    bond = atom.GetBonds()[0]
    return bond.GetBondType() == Chem.BondType.DOUBLE


def is_carbonyl_carbon(mol, idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C":
        return False
    return any(
        nbr.GetSymbol() == "O"
        and mol.GetBondBetweenAtoms(idx, nbr.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
        for nbr in atom.GetNeighbors()
    )


def _is_terminal_ketene_alpha_carbon(mol, idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0:
        return False
    single_neighbors = 0
    has_ketene_carbonyl = False
    for nbr in atom.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(idx, nbr.GetIdx())
        if bond.GetBondType() == Chem.BondType.SINGLE:
            single_neighbors += 1
            continue
        if bond.GetBondType() != Chem.BondType.DOUBLE or nbr.GetSymbol() != "C":
            return False
        has_double_o = any(
            onbr.GetSymbol() == "O"
            and mol.GetBondBetweenAtoms(nbr.GetIdx(), onbr.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
            for onbr in nbr.GetNeighbors()
            if onbr.GetIdx() != idx
        )
        if has_double_o:
            has_ketene_carbonyl = True
        else:
            return False
    return has_ketene_carbonyl and single_neighbors == 1


def _is_carboxylate_carbon(mol, idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C":
        return False
    has_double_o = False
    has_single_ominus = False
    for nbr in atom.GetNeighbors():
        if nbr.GetSymbol() != "O":
            continue
        bond = mol.GetBondBetweenAtoms(idx, nbr.GetIdx())
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            has_double_o = True
        elif bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetFormalCharge() < 0:
            has_single_ominus = True
    return has_double_o and has_single_ominus


def _acyl_halide_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C":
        return ""
    children = [nbr for nbr in atom.GetNeighbors() if nbr.GetIdx() != parent_idx]
    if len(children) != 2:
        return ""
    has_double_o = any(
        nbr.GetSymbol() == "O"
        and mol.GetBondBetweenAtoms(idx, nbr.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
        for nbr in children
    )
    if not has_double_o:
        return ""
    halogens = [
        nbr.GetSymbol()
        for nbr in children
        if nbr.GetSymbol() in HALO
        and not nbr.GetFormalCharge() and not nbr.GetNumRadicalElectrons()
        and nbr.GetTotalValence() == 1
        and not nbr.GetTotalNumHs(includeNeighbors=True)
        and mol.GetBondBetweenAtoms(idx, nbr.GetIdx()).GetBondType() == Chem.BondType.SINGLE
    ]
    if len(halogens) != 1:
        return ""
    return {
        "F": "fluorocarbonyl",
        "Cl": "chloroformyl",
        "Br": "bromocarbonyl",
        "I": "iodocarbonyl",
    }.get(halogens[0], "")


def _carbonyl_substituent_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    has_double_o = False
    heavy = []
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            has_double_o = True
            continue
        heavy.append(nidx)
    if has_double_o and not heavy:
        return "formyl" if _formyl_prefix_allowed(mol, idx, parent_idx) else ""
    if not has_double_o or len(heavy) != 1:
        return ""
    if _is_carboxylate_carbon(mol, idx) and _atom_has_multiple_bond_outside(mol, parent_idx, idx):
        return "1-oxido-1-oxo-methyl"
    distal = heavy[0]
    if mol.GetAtomWithIdx(distal).GetAtomicNum() == 6 and mol.GetAtomWithIdx(parent_idx).IsInRing():
        from ..carboxamide_prefixes import mapped_ring_oxamoyl

        oxamoyl = mapped_ring_oxamoyl(mol, Chem, idx, parent_idx)
        if oxamoyl is not None:
            return oxamoyl[0]
    if mol.GetAtomWithIdx(parent_idx).GetAtomicNum() == 8 and mol.GetAtomWithIdx(distal).GetAtomicNum() == 6:
        from .unsaturated_acyl import unsaturated_acyl_prefix
        acyl = unsaturated_acyl_prefix(mol, Chem, idx, parent_idx)
        if acyl is not None:
            return acyl.text
    from .preferred_acyl import (
        carboxyalkanoyl_prefix, n_attached_aminoacyl_prefix, n_attached_chain_acyl_prefix,
        n_attached_heterocyclylacetyl_prefix,
    )
    preferred_acyl = (
        n_attached_chain_acyl_prefix(mol, Chem, idx, parent_idx)
        or n_attached_aminoacyl_prefix(mol, idx, parent_idx)
        or n_attached_heterocyclylacetyl_prefix(mol, idx, parent_idx)
        or carboxyalkanoyl_prefix(mol, Chem, idx, parent_idx)
    )
    if preferred_acyl:
        return preferred_acyl
    aminoacyl = _retained_aminoacyl_prefix(mol, idx, distal, parent_idx=parent_idx)
    if aminoacyl:
        return aminoacyl
    nitrocarbonyl = _nitrocarbonyl_prefix(mol, distal, idx)
    if nitrocarbonyl:
        return nitrocarbonyl
    localize_ligand_stereo = _carbonyl_branch_stereo_context(mol, parent_idx)
    carbamoyl = _carbamoyl_substituent_name(
        mol,
        idx,
        parent_idx,
        localize_ligand_stereo=localize_ligand_stereo,
    )
    if carbamoyl:
        return carbamoyl
    oxido_carbamoyl = _oxido_carbamoyl_prefix(mol, distal, idx)
    if oxido_carbamoyl:
        return oxido_carbamoyl
    hydroxycarbamoyl = _hydroxycarbamoyl_prefix(mol, distal, idx)
    if hydroxycarbamoyl:
        return hydroxycarbamoyl
    alkoxycarbonyl = _alkoxycarbonyl_prefix(mol, distal, idx, parent_idx=parent_idx)
    if alkoxycarbonyl:
        return alkoxycarbonyl
    if mol.GetAtomWithIdx(distal).IsInRing():
        if mol.GetAtomWithIdx(distal).GetAtomicNum() == 6:
            from .preferred_acyl import ring_carbonyl_prefix
            ring_acyl = ring_carbonyl_prefix(mol, idx, parent_idx)
            if ring_acyl:
                return ring_acyl
        name = _ring_substituent_prefix(mol, distal, idx)
    else:
        name = branch_prefix(mol, distal, idx) or substituent_from_root(mol, distal, idx)
    if not name:
        return ""
    if localize_ligand_stereo:
        alpha_stereo = _alpha_ligand_stereo_prefix(mol, distal, name)
        if alpha_stereo:
            name = alpha_stereo + name
    retained = _retained_carbonyl_prefix(name)
    if retained and mol.GetAtomWithIdx(parent_idx).GetAtomicNum()==8:
        from ..component_vocabulary import preferred_acyl_component
        retained=preferred_acyl_component(retained)
    if retained:
        return retained
    # The complete organyl is attached to carbonyl. An internal enclosure
    # does not bind its other substituents to that organyl free valence.
    branched_attachment = sum(n.GetAtomicNum() > 1 and n.GetIdx() != idx
                             for n in mol.GetAtomWithIdx(distal).GetNeighbors()) > 1
    if branched_attachment and any(mark in name for mark in "([{ -"):
        from ..prefix_presentation import enclose_complete_name_unit

        name = enclose_complete_name_unit(name)
    return name + "carbonyl"


_AMINOACYL_BY_SIDECHAIN = {
    "": "glycyl",
    "methyl": "alanyl",
    "isopropyl": "valyl",
    "1-methyl-ethyl": "valyl",
    "propyl": "norvalyl",
    "butyl": "norleucyl",
    "t-butyl": "t-leucyl",
    "tert-butyl": "t-leucyl",
    "isobutyl": "leucyl",
    "2-methylpropyl": "leucyl",
    "2-methyl-propyl": "leucyl",
    "sec-butyl": "isoleucyl",
    "1-methyl-propyl": "isoleucyl",
    "benzyl": "phenylalanyl",
    "2-(phenyl)ethyl": "homophenylalanyl",
    "4-hydroxybenzyl": "tyrosyl",
    "3,4-dihydroxybenzyl": "dopyl",
    "(1H-imidazol-4-yl)methyl": "histidyl",
    "(indole-3-yl)methyl": "tryptophyl",
    "hydroxymethyl": "seryl",
    "1-hydroxyethyl": "threonyl",
    "sulfanylmethyl": "cysteinyl",
    "mercaptomethyl": "cysteinyl",
    "2-methylsulfanyl-ethyl": "methionyl",
    "carboxymethyl": "aspartyl",
    "2-carboxyethyl": "glutamyl",
    "carbamylmethyl": "asparaginyl",
    "carbamoylmethyl": "asparaginyl",
    "2-carbamylethyl": "glutaminyl",
    "2-carbamoylethyl": "glutaminyl",
    "4-azabutyl": "ornithyl",
    "4-aminobutyl": "lysyl",
    "5-amino-4,6-diaza5-hexen-1-yl": "arginyl",
}


def _retained_aminoacyl_prefix(
    mol,
    carbonyl_idx: int,
    alpha_idx: int,
    seen: frozenset[int] = frozenset(),
    parent_idx: int | None = None,
) -> str:
    if carbonyl_idx in seen:
        return ""
    alpha = mol.GetAtomWithIdx(alpha_idx)
    if alpha.GetSymbol() != "C" or alpha.GetFormalCharge() != 0 or alpha.IsInRing() or alpha.GetIsotope():
        return ""
    bond = mol.GetBondBetweenAtoms(carbonyl_idx, alpha_idx)
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    amino = []
    sidechains = []
    for nbr in alpha.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == carbonyl_idx or nbr.GetAtomicNum() <= 1:
            continue
        nbr_bond = mol.GetBondBetweenAtoms(alpha_idx, nidx)
        if nbr_bond is None or nbr_bond.GetBondType() != Chem.BondType.SINGLE:
            return ""
        if nbr.GetSymbol() == "N" and nbr.GetFormalCharge() == 0 and not nbr.IsInRing():
            amino.append(nidx)
        else:
            sidechains.append(nidx)
    if len(amino) != 1 or len(sidechains) > 1:
        return ""
    upstream = _alpha_amino_acyl_prefix(mol, amino[0], alpha_idx, seen | {carbonyl_idx})
    if upstream is None:
        return ""
    if upstream and parent_idx is not None and mol.GetAtomWithIdx(parent_idx).IsInRing():
        return ""
    sidechain = ""
    if sidechains:
        sidechain = branch_prefix(mol, sidechains[0], alpha_idx) or substituent_from_root(mol, sidechains[0], alpha_idx)
    from ..component_vocabulary import with_preferred_keys
    prefix = with_preferred_keys(_AMINOACYL_BY_SIDECHAIN).get(sidechain, "")
    if not prefix:
        return ""
    if prefix == "t-leucyl":
        # P-103.1.1.3: an amino acid outside the retained-name tables uses its
        # ordinary systematic parent. The existing acyl path supplies that map.
        return ""
    stereo_prefix = _aminoacyl_stereo_prefix(mol, alpha_idx, prefix)
    if stereo_prefix is None:
        return ""
    emitted = upstream + stereo_prefix + prefix
    if alpha.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED and alpha.HasProp("_CIPCode"):
        descriptor = alpha.GetProp("_CIPCode")
        if descriptor in {"R", "S"}:
            mol_cache_dict(mol, "rendered_retained_aminoacyl_stereo")[(alpha_idx, emitted)] = descriptor
    return emitted


def _alpha_amino_acyl_prefix(
    mol,
    nitrogen_idx: int,
    alpha_idx: int,
    seen: frozenset[int],
) -> str | None:
    nitrogen = mol.GetAtomWithIdx(nitrogen_idx)
    if nitrogen.GetFormalCharge() != 0:
        return None
    heavy = [
        nbr.GetIdx()
        for nbr in nitrogen.GetNeighbors()
        if nbr.GetIdx() != alpha_idx and nbr.GetAtomicNum() > 1
    ]
    if not heavy:
        return ""
    if len(heavy) != 1:
        return None
    carbonyl_idx = heavy[0]
    carbonyl = mol.GetAtomWithIdx(carbonyl_idx)
    if carbonyl.GetSymbol() != "C" or carbonyl.GetFormalCharge() != 0 or carbonyl.IsInRing():
        return None
    bond_to_n = mol.GetBondBetweenAtoms(nitrogen_idx, carbonyl_idx)
    if bond_to_n is None or bond_to_n.GetBondType() != Chem.BondType.SINGLE:
        return None
    double_o = []
    alpha_candidates = []
    for nbr in carbonyl.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == nitrogen_idx:
            continue
        bond = mol.GetBondBetweenAtoms(carbonyl_idx, nidx)
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_o.append(nidx)
        elif nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE:
            alpha_candidates.append(nidx)
        elif nbr.GetAtomicNum() > 1:
            return None
    if len(double_o) != 1 or len(alpha_candidates) != 1:
        return None
    upstream = _retained_aminoacyl_prefix(mol, carbonyl_idx, alpha_candidates[0], seen)
    return upstream or None


def _is_free_alpha_amino_group(mol, nitrogen_idx: int, alpha_idx: int) -> bool:
    nitrogen = mol.GetAtomWithIdx(nitrogen_idx)
    if nitrogen.GetFormalCharge() != 0:
        return False
    for nbr in nitrogen.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == alpha_idx or nbr.GetAtomicNum() <= 1:
            continue
        return False
    return True


def _aminoacyl_stereo_prefix(mol, alpha_idx: int, prefix: str) -> str | None:
    atom = mol.GetAtomWithIdx(alpha_idx)
    if atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED:
        # Retained aminoacyl words carry the natural alpha configuration.
        # An unassigned stereogenic alpha carbon needs a systematic prefix
        # that keeps it unassigned, rather than silently introducing L.
        return "" if prefix == "glycyl" else None
    try:
        Chem.AssignStereochemistry(mol, force=True, cleanIt=False)
    except Exception:
        return None
    code = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else ""
    if code not in {"R", "S"}:
        return None
    if prefix == "cysteinyl":
        return "D-" if code == "S" else ""
    return "D-" if code == "R" else ""


def _alpha_ligand_stereo_prefix(mol, alpha_idx: int, ligand_name: str) -> str:
    if not ligand_name:
        return ""
    if ligand_name.startswith("("):
        return ""
    atom = mol.GetAtomWithIdx(alpha_idx)
    if atom.GetSymbol() != "C" or atom.IsInRing() or atom.GetIsotope():
        return ""
    if atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED:
        return ""
    try:
        Chem.AssignStereochemistry(mol, force=True, cleanIt=False)
    except Exception:
        return ""
    code = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else ""
    if code not in {"R", "S"}:
        return ""
    return f"(1{code})-"


def _carbonyl_branch_stereo_context(mol, parent_idx: int) -> bool:
    parent = mol.GetAtomWithIdx(parent_idx)
    return (
        parent.GetSymbol() in {"C", "N"}
        and parent.IsInRing()
        and not parent.GetIsAromatic()
        and _ring_component_ring_count(mol, parent_idx) > 1
    )


def _ring_component_ring_count(mol, atom_idx: int) -> int:
    cache = mol_cache_dict(mol, "branch_ring_component_ring_count")
    cached = cache.get(atom_idx)
    if cached is not None:
        return cached
    atom_rings = [set(ring) for ring in mol.GetRingInfo().AtomRings()]
    seeds = [idx for idx, ring in enumerate(atom_rings) if atom_idx in ring]
    if not seeds:
        cache[atom_idx] = 0
        return 0
    seen = set(seeds)
    stack = list(seeds)
    while stack:
        current = stack.pop()
        current_ring = atom_rings[current]
        for idx, ring in enumerate(atom_rings):
            if idx in seen or current_ring.isdisjoint(ring):
                continue
            seen.add(idx)
            stack.append(idx)
    count = len(seen)
    cache[atom_idx] = count
    return count


def _retained_carbonyl_prefix(name: str) -> str:
    if name.endswith("phenyl"):
        return name[: -len("phenyl")] + "benzoyl"
    retained = {
        "eth-1-en-1-yl": "acryloyl",
        "difluoromethyl": "difluoroacetyl",
        "ethyl": "propionyl",
        "isopropyl": "isobutyryl",
        "methyl": "acetyl",
        "phenyl": "benzoyl",
        "propyl": "butyryl",
        "tert-butyl": "pivaloyl",
        "trifluoromethyl": "trifluoroacetyl",
        "piperidino": "piperidinoyl",
        "piperazinino": "piperazinoyl",
        "piperazino": "piperazinoyl",
        "pyrrolidino": "pyrrolidinoyl",
        "morpholino": "morpholinoyl",
        "2-piperidyl": "2-piperidinoyl",
        "3-piperidyl": "3-piperidinoyl",
        "4-piperidyl": "4-piperidinoyl",
        "2-pyrrolidyl": "2-pyrrolidinoyl",
        "3-pyrrolidyl": "3-pyrrolidinoyl",
        "2-azetidyl": "2-azetidinoyl",
        "3-azetidyl": "3-azetidinoyl",
    }
    if name in retained:
        return retained[name]
    arylacetyl = _retained_arylacetyl_prefix(name)
    if arylacetyl:
        return arylacetyl
    acyclic = _acyclic_acyl_prefix(name)
    if acyclic:
        return acyclic
    if name.endswith("-yl") and _locanted_ring_acyl_prefix_allowed(name):
        return name[:-2] + "carbonyl"
    return ""


def _retained_arylacetyl_prefix(name: str) -> str:
    if name == "benzyl":
        return "phenylacetyl"
    for suffix, replacement in (
        ("benzyl", "phenylacetyl"),
        ("cyclopropylmethyl", "cyclopropylacetyl"),
        ("cyclobutylmethyl", "cyclobutylacetyl"),
        ("cyclopentylmethyl", "cyclopentylacetyl"),
        ("cyclohexylmethyl", "cyclohexylacetyl"),
    ):
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)] + replacement
    for ring in ("pyridin", "pyrimidin", "pyrazin", "pyridazin", "furan", "thiophen", "thiofuran"):
        for locant in ("2", "3", "4", "5", "6"):
            suffix = f"{ring}-{locant}-ylmethyl"
            if name == suffix:
                stem = "thiophen" if ring == "thiofuran" else ring
                return f"{stem}-{locant}-ylacetyl"
            if name.endswith(suffix) and len(name) > len(suffix):
                stem = "thiophen" if ring == "thiofuran" else ring
                return name[: -len(suffix)] + f"{stem}-{locant}-ylacetyl"
    return ""


_ACYCLIC_ACYL_PARENT_BY_ALKYL = {
    "methyl": "acetyl",
    "ethyl": "propionyl",
    "propyl": "butyryl",
    "butyl": "valeryl",
    "pentyl": "hexanoyl",
    "hexyl": "heptanoyl",
    "heptyl": "octanoyl",
    "octyl": "nonanoyl",
    "nonyl": "decanoyl",
    "decyl": "undecanoyl",
    "undecyl": "dodecanoyl",
    "dodecyl": "tridecanoyl",
}


def _acyclic_acyl_prefix(name: str) -> str:
    """Convert an alkyl ligand on C=O into the corresponding acyl prefix."""

    retained = _ACYCLIC_ACYL_PARENT_BY_ALKYL.get(name)
    if retained or not name.endswith("yl"):
        return retained or ""
    length = ROOT_LENGTHS.get(name[:-2])
    stem = ROOTS.get(length + 1) if length is not None else None
    return stem + "anoyl" if stem else ""


def _shift_acyl_ligand_locants(prefix: str) -> str | None:
    """Shift ligand locants by one because the acyl carbon becomes locant 1."""

    if not prefix:
        return ""
    parts = prefix.split("-")
    shifted_parts = []
    for part in parts:
        if not part:
            return None
        match = re.fullmatch(r"\d+(?:,\d+)*", part)
        if match:
            shifted_parts.append(",".join(str(int(loc) + 1) for loc in part.split(",")))
        else:
            shifted_parts.append(part)
    return "-".join(shifted_parts) + "-"


def _locanted_ring_acyl_prefix_allowed(name: str) -> bool:
    if not re.search(r"-\d+[a-z]?'?-yl$", name):
        return False
    if any(term in name for term in ("cyclo", "bicyclo[", "tricyclo[", "spiro[", "[")):
        return False
    if not re.fullmatch(
        r"(?:"
        r"azetidin|furan|imidazol|isothiazol|isoxazol|morpholin|oxazol|"
        r"piperazin|piperidin|pyrazin|pyrazol|pyridazin|pyridin|pyrimidin|"
        r"pyrrolidin|tetrazol|thiazol|thiophen|triazol"
        r")-\d+[a-z]?'?-yl",
        name,
    ):
        return False
    return not name.startswith(("benzyl", "allyl", "vinyl"))


def _formyl_prefix_allowed(mol, carbonyl_idx: int, parent_idx: int) -> bool:
    if _has_isotopic_hydrogen(mol, carbonyl_idx):
        return False
    parent = mol.GetAtomWithIdx(parent_idx)
    if parent.GetSymbol() == "N" and any(
        nbr.GetIdx() != carbonyl_idx and nbr.GetSymbol() == "N" for nbr in parent.GetNeighbors()
    ):
        return False
    if _atom_has_multiple_bond_to_heteroatom_outside(mol, parent_idx, carbonyl_idx):
        return False
    if not parent.IsInRing() and _assigned_stereocenter_count(mol) >= 3:
        return False
    if parent.GetIsAromatic() and _aromatic_formyl_ring_has_multiple_arylalkoxy_substituents(
        mol, parent_idx, carbonyl_idx
    ):
        return False
    return True


def _assigned_stereocenter_count(mol) -> int:
    return sum(1 for atom in iter_atoms(mol) if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED)


def _atom_has_multiple_bond_to_heteroatom_outside(mol, atom_idx: int, excluded_idx: int) -> bool:
    atom = mol.GetAtomWithIdx(atom_idx)
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == excluded_idx or nbr.GetSymbol() == "C":
            continue
        bond = mol.GetBondBetweenAtoms(atom_idx, nidx)
        # RDKit reports aromatic ring bonds as 1.5. They are annular bonds,
        # not an exocyclic multiple-bonded characteristic group that should
        # block the direct P-66.6.1.3 formyl prefix on a heteroaromatic host.
        if (bond is not None and bond.GetBondTypeAsDouble() > 1.0
                and not (bond.GetIsAromatic() and bond.IsInRing())):
            return True
    return False


def _aromatic_formyl_ring_has_multiple_arylalkoxy_substituents(
    mol,
    parent_idx: int,
    carbonyl_idx: int,
) -> bool:
    for ring in mol.GetRingInfo().AtomRings():
        if parent_idx not in ring:
            continue
        ring_atoms = set(ring)
        count = 0
        for ring_idx in ring_atoms:
            atom = mol.GetAtomWithIdx(ring_idx)
            for nbr in atom.GetNeighbors():
                nidx = nbr.GetIdx()
                if nidx in ring_atoms or nidx == carbonyl_idx:
                    continue
                bond = mol.GetBondBetweenAtoms(ring_idx, nidx)
                if (
                    nbr.GetSymbol() == "O"
                    and nbr.GetFormalCharge() == 0
                    and bond is not None
                    and bond.GetBondType() == Chem.BondType.SINGLE
                    and _alkoxy_ligand_contains_ring(mol, nidx, ring_idx)
                ):
                    count += 1
                    if count >= 2:
                        return True
    return False


def _alkoxy_ligand_contains_ring(mol, oxygen_idx: int, ring_idx: int) -> bool:
    starts = [
        nbr.GetIdx()
        for nbr in mol.GetAtomWithIdx(oxygen_idx).GetNeighbors()
        if nbr.GetIdx() != ring_idx and nbr.GetAtomicNum() > 1
    ]
    seen = {oxygen_idx, ring_idx}
    stack = list(starts)
    while stack and len(seen) < 80:
        idx = stack.pop()
        if idx in seen:
            continue
        seen.add(idx)
        atom = mol.GetAtomWithIdx(idx)
        if atom.IsInRing():
            return True
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx not in seen and nbr.GetAtomicNum() > 1:
                stack.append(nidx)
    return False


def _oxido_carbamoyl_prefix(mol, nitrogen_idx: int, carbonyl_idx: int) -> str:
    nitrogen = mol.GetAtomWithIdx(nitrogen_idx)
    if nitrogen.GetSymbol() != "N" or nitrogen.GetFormalCharge() != 0:
        return ""
    bond = mol.GetBondBetweenAtoms(nitrogen_idx, carbonyl_idx)
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    oxido = []
    other = []
    for nbr in nitrogen.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == carbonyl_idx:
            continue
        n_bond = mol.GetBondBetweenAtoms(nitrogen_idx, nidx)
        if nbr.GetSymbol() == "O" and nbr.GetFormalCharge() < 0 and n_bond.GetBondType() == Chem.BondType.SINGLE:
            oxido.append(nidx)
        else:
            other.append(nidx)
    if len(oxido) == 1 and not other:
        return "1-oxido-1-azamethylcarbonyl"
    return ""


def _nitrocarbonyl_prefix(mol, nitrogen_idx: int, carbonyl_idx: int) -> str:
    nitrogen = mol.GetAtomWithIdx(nitrogen_idx)
    if nitrogen.GetSymbol() != "N" or nitrogen.GetFormalCharge() <= 0:
        return ""
    bond = mol.GetBondBetweenAtoms(nitrogen_idx, carbonyl_idx)
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    single_oxygens = []
    double_oxygens = []
    other = []
    for nbr in nitrogen.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == carbonyl_idx:
            continue
        n_bond = mol.GetBondBetweenAtoms(nitrogen_idx, nidx)
        if nbr.GetSymbol() == "O" and n_bond.GetBondType() == Chem.BondType.DOUBLE:
            double_oxygens.append(nidx)
        elif (
            nbr.GetSymbol() == "O"
            and nbr.GetFormalCharge() < 0
            and n_bond.GetBondType() == Chem.BondType.SINGLE
        ):
            single_oxygens.append(nidx)
        else:
            other.append(nidx)
    if len(double_oxygens) == 1 and len(single_oxygens) == 1 and not other:
        return "nitrocarbonyl"
    return ""


def _hydroxycarbamoyl_prefix(mol, nitrogen_idx: int, carbonyl_idx: int) -> str:
    nitrogen = mol.GetAtomWithIdx(nitrogen_idx)
    if nitrogen.GetSymbol() != "N" or nitrogen.GetFormalCharge() != 0:
        return ""
    bond = mol.GetBondBetweenAtoms(nitrogen_idx, carbonyl_idx)
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    hydroxy = []
    other = []
    for nbr in nitrogen.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == carbonyl_idx:
            continue
        n_bond = mol.GetBondBetweenAtoms(nitrogen_idx, nidx)
        if nbr.GetSymbol() == "O" and nbr.GetFormalCharge() == 0 and n_bond.GetBondType() == Chem.BondType.SINGLE:
            terminal = [nnbr for nnbr in nbr.GetNeighbors() if nnbr.GetIdx() != nitrogen_idx and nnbr.GetAtomicNum() > 1]
            if terminal:
                other.append(nidx)
            else:
                hydroxy.append(nidx)
        else:
            other.append(nidx)
    if len(hydroxy) == 1 and not other:
        return "(hydroxyamino)carbonyl"
    return ""


def _alkoxycarbonyl_prefix(mol, oxygen_idx: int, carbonyl_idx: int, *, parent_idx: int | None = None) -> str:
    oxygen = mol.GetAtomWithIdx(oxygen_idx)
    if oxygen.GetSymbol() != "O" or oxygen.GetFormalCharge() != 0:
        return ""
    bond = mol.GetBondBetweenAtoms(oxygen_idx, carbonyl_idx)
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    heavy = [nbr.GetIdx() for nbr in oxygen.GetNeighbors() if nbr.GetIdx() != carbonyl_idx]
    if not heavy:
        return "carboxy"
    if len(heavy) != 1:
        return ""
    attached = heavy[0]
    if mol.GetAtomWithIdx(attached).IsInRing():
        ring = _ring_substituent_prefix(mol, attached, oxygen_idx)
        if not ring:
            return ""
        oxy = "phenoxy" if ring == "phenyl" else f"({ring})oxy"
    else:
        from .contracted_alkoxy import mapped_contracted_alkoxy
        from ..formatting import enclose_name_unit

        alkoxy = mapped_contracted_alkoxy(mol, Chem, oxygen_idx, carbonyl_idx)
        if alkoxy is not None and any(c in alkoxy[0] for c in "-()[]{}"):
            return alkoxy[0] + "carbonyl"
        alk = branch_prefix(mol, attached, oxygen_idx) or substituent_from_root(mol, attached, oxygen_idx)
        if not alk:
            return ""
        if parent_idx is not None and _carbonyl_branch_stereo_context(mol, parent_idx):
            alpha_stereo = _alpha_ligand_stereo_prefix(mol, attached, alk)
            if alpha_stereo:
                alk = alpha_stereo + alk
        retained = _retained_alkoxycarbonyl_prefix(mol, alk, parent_idx=parent_idx, carbonyl_idx=carbonyl_idx)
        if retained:
            return retained
        # P-63.2.2.2 permits substitution on methoxy. The helper proves the
        # ring-CH2-O connection before contracting the complete unit.
        from .hetero_atoms import _ring_methoxy_prefix
        ring_methoxy = _ring_methoxy_prefix(mol, attached, oxygen_idx)
        if ring_methoxy:
            return enclose_name_unit(ring_methoxy) + "carbonyl"
        oxy = (
            f"({alk})oxy"
            if alk == "benzyl" and _atom_has_multiple_bond_outside(mol, parent_idx, carbonyl_idx)
            else _alkyl_to_alkoxy(alk)
        )
    return oxy + "carbonyl"


def _retained_alkoxycarbonyl_prefix(
    mol,
    alkyl: str,
    *,
    parent_idx: int | None = None,
    carbonyl_idx: int | None = None,
) -> str:
    if parent_idx is not None and carbonyl_idx is not None and _has_nonaromatic_ring_amine_attachment(
        mol, parent_idx, carbonyl_idx
    ):
        return ""
    return {
        "t-butyl": "tert-butoxycarbonyl",
        "tert-butyl": "tert-butoxycarbonyl",
        "benzyl": "benzyloxycarbonyl",
    }.get(alkyl, "")


def _has_nonaromatic_ring_amine_attachment(mol, atom_idx: int, carbonyl_idx: int) -> bool:
    atom = mol.GetAtomWithIdx(atom_idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0:
        return False
    for nbr in atom.GetNeighbors():
        if nbr.GetIdx() == carbonyl_idx:
            continue
        if nbr.GetAtomicNum() == 6 and nbr.IsInRing() and not nbr.GetIsAromatic():
            return True
    return False


def _has_assigned_stereo(mol) -> bool:
    return any(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for atom in iter_atoms(mol)) or any(
        bond.GetStereo() != Chem.BondStereo.STEREONONE for bond in iter_bonds(mol)
    )


def _atom_has_multiple_bond_outside(mol, atom_idx: int | None, excluded_idx: int | None) -> bool:
    if atom_idx is None:
        return False
    atom = mol.GetAtomWithIdx(atom_idx)
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == excluded_idx:
            continue
        bond = mol.GetBondBetweenAtoms(atom_idx, nidx)
        if bond is not None and bond.GetBondTypeAsDouble() > 1.0:
            return True
    return False


def _has_isotopic_hydrogen(mol, atom_idx: int) -> bool:
    return any(nbr.GetSymbol() == "H" and nbr.GetIsotope() for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors())


def _is_isocyano_carbon(mol, idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() >= 0:
        return False
    return any(
        nbr.GetSymbol() == "N"
        and nbr.GetFormalCharge() > 0
        and mol.GetBondBetweenAtoms(idx, nbr.GetIdx()).GetBondType() == Chem.BondType.TRIPLE
        for nbr in atom.GetNeighbors()
    )


def _is_nitrile_carbon(mol, idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0:
        return False
    return any(
        nbr.GetSymbol() == "N"
        and nbr.GetFormalCharge() == 0
        and mol.GetBondBetweenAtoms(idx, nbr.GetIdx()).GetBondType() == Chem.BondType.TRIPLE
        for nbr in atom.GetNeighbors()
    )


def _is_nitrile_nitrogen(mol, idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0:
        return False
    return any(
        nbr.GetSymbol() == "C"
        and nbr.GetFormalCharge() == 0
        and mol.GetBondBetweenAtoms(idx, nbr.GetIdx()).GetBondType() == Chem.BondType.TRIPLE
        for nbr in atom.GetNeighbors()
    )


def _is_isocyano_nitrogen(mol, idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() <= 0:
        return False
    return any(
        nbr.GetSymbol() == "C"
        and nbr.GetFormalCharge() < 0
        and mol.GetBondBetweenAtoms(idx, nbr.GetIdx()).GetBondType() == Chem.BondType.TRIPLE
        for nbr in atom.GetNeighbors()
    )


def _is_sulfonate_sulfur(mol, idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "S":
        return False
    double_o = 0
    single_ominus = 0
    for nbr in atom.GetNeighbors():
        if nbr.GetSymbol() != "O":
            continue
        bond = mol.GetBondBetweenAtoms(idx, nbr.GetIdx())
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            double_o += 1
        elif bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetFormalCharge() < 0:
            single_ominus += 1
    return double_o >= 2 and single_ominus >= 1


def _is_carbonyl_oxygen_idx(mol, idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    return _is_terminal_carbonyl_oxygen(atom)


def _is_nitro_nitrogen(mol, idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if (atom.GetSymbol() != "N" or atom.GetFormalCharge() != 1
            or atom.GetTotalNumHs(includeNeighbors=True)):
        return False
    neighbors = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
    if len(neighbors) != 3:
        return False
    fixed = []
    for neighbor in neighbors:
        if (neighbor.GetAtomicNum() != 8 or neighbor.GetDegree() != 1
                or neighbor.GetTotalNumHs(includeNeighbors=True)):
            continue
        order = mol.GetBondBetweenAtoms(idx, neighbor.GetIdx()).GetBondTypeAsDouble()
        if (neighbor.GetFormalCharge(), order) in {(0, 2.0), (-1, 1.0)}:
            fixed.append((neighbor.GetIdx(), neighbor.GetFormalCharge(), order))
    if sorted((charge, order) for _, charge, order in fixed) != [(-1, 1.0), (0, 2.0)]:
        return False
    fixed_atoms = {i for i, _, _ in fixed}
    return all(mol.GetBondBetweenAtoms(idx, n.GetIdx()).GetBondTypeAsDouble() == 1
               for n in neighbors if n.GetIdx() not in fixed_atoms)


def _is_azido_attachment(mol, idx: int, parent_idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N":
        return False
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return False
    for mid in atom.GetNeighbors():
        if mid.GetIdx() == parent_idx or mid.GetSymbol() != "N" or mid.GetFormalCharge() <= 0:
            continue
        mid_bond = mol.GetBondBetweenAtoms(idx, mid.GetIdx())
        if mid_bond is None or mid_bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        for terminal in mid.GetNeighbors():
            terminal_bond = mol.GetBondBetweenAtoms(mid.GetIdx(), terminal.GetIdx())
            if (
                terminal.GetIdx() != idx
                and terminal.GetSymbol() == "N"
                and terminal.GetFormalCharge() < 0
                and terminal_bond is not None
                and terminal_bond.GetBondType() == Chem.BondType.DOUBLE
                and len([nbr for nbr in terminal.GetNeighbors() if nbr.GetAtomicNum() > 1]) == 1
            ):
                return True
    return False


def _is_diazo_attachment(mol, idx: int, parent_idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() <= 0:
        return False
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.DOUBLE:
        return False
    return any(
        nbr.GetIdx() != parent_idx
        and nbr.GetSymbol() == "N"
        and nbr.GetFormalCharge() < 0
        and mol.GetBondBetweenAtoms(idx, nbr.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
        for nbr in atom.GetNeighbors()
    )


def _is_diazo_nitrogen(mol, idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() <= 0:
        return False
    has_terminal_n = False
    has_carbon = False
    for nbr in atom.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(idx, nbr.GetIdx())
        if nbr.GetSymbol() == "N" and nbr.GetFormalCharge() < 0 and bond.GetBondType() == Chem.BondType.DOUBLE:
            has_terminal_n = True
        elif nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.DOUBLE:
            has_carbon = True
    return has_terminal_n and has_carbon


def _is_diazonio_attachment(mol, idx: int, parent_idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() <= 0:
        return False
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return False
    return any(
        nbr.GetIdx() != parent_idx
        and nbr.GetSymbol() == "N"
        and mol.GetBondBetweenAtoms(idx, nbr.GetIdx()).GetBondType() == Chem.BondType.TRIPLE
        for nbr in atom.GetNeighbors()
    )


def _is_nitroso_attachment(mol, idx: int, parent_idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if (atom.GetSymbol() != "N" or atom.GetFormalCharge()
            or atom.GetTotalNumHs(includeNeighbors=True)):
        return False
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return False
    distal = [n for n in atom.GetNeighbors() if n.GetIdx() != parent_idx and n.GetAtomicNum() > 1]
    return (len(distal) == 1 and distal[0].GetAtomicNum() == 8
            and distal[0].GetDegree() == 1 and not distal[0].GetFormalCharge()
            and not distal[0].GetTotalNumHs(includeNeighbors=True)
            and mol.GetBondBetweenAtoms(idx, distal[0].GetIdx()).GetBondType() == Chem.BondType.DOUBLE)
