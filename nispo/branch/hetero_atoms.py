from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

try:
    from rdkit import Chem, RDLogger
except Exception:  # pragma: no cover - benchmark environment provides RDKit.
    Chem = None
else:
    RDLogger.DisableLog("rdApp.*")

from ..constants import HALO, MULT, ROOTS
from ..formatting import alkyl_to_alkoxy as _alkyl_to_alkoxy
from ..substituent_resource_rules import resource_substituent_match as _resource_substituent_match
from ..template_access import template_record_name
from .functional_groups import (
    _carbonyl_substituent_prefix,
    _is_azido_attachment,
    _is_diazo_attachment,
    _is_diazonio_attachment,
    _is_isocyano_nitrogen,
    _is_nitrile_carbon,
    _is_nitro_nitrogen,
    _is_nitroso_attachment,
    _is_tbdms_group,
    _is_trimethylsilyl_group,
    _ligand_text,
    _phosphorus_prefix,
    _sulfur_prefix,
    substituent_from_root,
)
from .functional_suffixes import suffix_prefix_term, suffix_supported_term
from .ring_substituents import _ring_substituent_prefix


def _branch_rules():
    from .. import branch_rules

    return branch_rules


def branch_prefix(mol, idx: int, parent_idx: int) -> str:
    return _branch_rules().branch_prefix(mol, idx, parent_idx)


def _alkylidene_prefix(mol, carbon_idx: int, parent_idx: int) -> str:
    return _branch_rules()._alkylidene_prefix(mol, carbon_idx, parent_idx)


def _alkynyl_substituent_prefix(mol, idx: int, parent_idx: int) -> str:
    return _branch_rules()._alkynyl_substituent_prefix(mol, idx, parent_idx)


def _enal_substituent_prefix(mol, idx: int, parent_idx: int) -> str:
    return _branch_rules()._enal_substituent_prefix(mol, idx, parent_idx)


def _enone_substituent_prefix(mol, idx: int, parent_idx: int) -> str:
    return _branch_rules()._enone_substituent_prefix(mol, idx, parent_idx)


def _haloethenyl_prefix(mol, idx: int, parent_idx: int) -> str:
    return _branch_rules()._haloethenyl_prefix(mol, idx, parent_idx)


def _ketenyl_prefix(mol, idx: int, parent_idx: int) -> str:
    return _branch_rules()._ketenyl_prefix(mol, idx, parent_idx)


def _linear_carbon_chain(mol, start_idx: int, blocked_idx: int, max_len: int) -> tuple[int, ...]:
    return _branch_rules()._linear_carbon_chain(mol, start_idx, blocked_idx, max_len)


def _nitro_branch_predicate(mol, idx: int, parent_idx: int) -> bool:
    return _branch_rules()._nitro_branch_predicate(mol, idx, parent_idx)

def _oxygen_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    parent = mol.GetAtomWithIdx(parent_idx)
    heavy = _nonplain_hydrogen_neighbor_indices(atom, parent_idx)
    if atom.GetFormalCharge() < 0 and parent.GetSymbol() == "N" and parent.GetFormalCharge() > 0:
        if bond is not None and bond.GetBondType() == Chem.BondType.SINGLE:
            return "oxido"
    if atom.GetFormalCharge() < 0 and not heavy:
        if bond is not None and bond.GetBondType() == Chem.BondType.SINGLE:
            return "oxido"
    if atom.GetFormalCharge() > 0 and not heavy:
        if bond is not None and bond.GetBondType() == Chem.BondType.SINGLE:
            return "oxidaniumyl"
    if atom.GetFormalCharge() > 0 and len(heavy) == 1:
        ylidene_oxidanium = _ylidene_oxidaniumyl_prefix(mol, heavy[0], idx)
        if ylidene_oxidanium:
            return ylidene_oxidanium
    if bond.GetBondType() == Chem.BondType.DOUBLE:
        return suffix_prefix_term("one", "oxo")
    if atom.GetFormalCharge() != 0:
        return ""
    if not heavy:
        return "hydroxy"
    if len(heavy) == 1:
        if _is_nitro_nitrogen(mol, heavy[0]):
            return "nitrooxy"
        if mol.GetAtomWithIdx(heavy[0]).GetSymbol() == "H":
            return _isotopic_hydrogen_oxy_prefix(mol.GetAtomWithIdx(heavy[0]))
        if mol.GetAtomWithIdx(heavy[0]).GetSymbol() in HALO:
            return HALO[mol.GetAtomWithIdx(heavy[0]).GetSymbol()] + "oxy"
        carbonyl = _carbonyl_substituent_prefix(mol, heavy[0], idx)
        if carbonyl:
            return carbonyl + "oxy"
        glycosyl = _glycosyloxy_prefix(mol, heavy[0], idx)
        if glycosyl:
            return glycosyl
        protected = _protected_alkoxy_prefix(mol, heavy[0], idx)
        if protected:
            return protected
        if mol.GetAtomWithIdx(heavy[0]).GetSymbol() == "S":
            if parent.GetSymbol() not in {"Si", "Ge", "Sn"}:
                sulfur_oxy = _sulfur_oxy_alias(mol, heavy[0], idx)
                if sulfur_oxy:
                    return sulfur_oxy
            sulfur = _sulfur_prefix(mol, heavy[0], idx)
            if (
                sulfur.endswith(("sulfonyl", "sulfonothioyl"))
                and _sulfur_has_ring_carbon_ligand(mol, heavy[0], idx)
                and not _sulfur_has_nitroaryl_ligand(mol, heavy[0], idx)
            ):
                return sulfur + "oxy"
        if mol.GetAtomWithIdx(heavy[0]).GetSymbol() == "P":
            phosphorus = _phosphorus_oxy_prefix(mol, heavy[0], idx)
            if phosphorus:
                return phosphorus
            phosphorus = _phosphorus_prefix(mol, heavy[0], idx)
            if phosphorus:
                return phosphorus + "oxy"
        if mol.GetAtomWithIdx(heavy[0]).IsInRing():
            ring = _ring_substituent_prefix(mol, heavy[0], idx)
            if not ring:
                return ""
            phenoxy = _phenoxy_substituent_prefix(ring)
            if phenoxy:
                return phenoxy
            return f"({ring})oxy"
        retained = _retained_oxygen_resource_prefix(mol, idx, parent_idx)
        if retained:
            return retained
        alk = ""
        if mol.GetAtomWithIdx(heavy[0]).GetSymbol() == "C":
            alk = _branch_rules().branch_prefix(mol, heavy[0], idx)
        if not alk:
            alk = substituent_from_root(mol, heavy[0], idx)
        return _alkyl_to_alkoxy(alk)
    return ""


def _phenoxy_substituent_prefix(ring: str) -> str:
    if ring == "phenyl":
        return "phenoxy"
    if ring.endswith("phenyl") and not ring.endswith("diphenyl"):
        return ring[:-2] + "oxy"
    return ""


def _protected_alkoxy_prefix(mol, carbon_idx: int, oxygen_idx: int) -> str:
    ethoxy = _silylethoxy_prefix(mol, carbon_idx, oxygen_idx)
    if ethoxy:
        return ethoxy
    tbdmsoxyethoxy = _tbdmsoxyethoxy_prefix(mol, carbon_idx, oxygen_idx)
    if tbdmsoxyethoxy:
        return tbdmsoxyethoxy
    return ""


def _phosphorus_oxy_prefix(mol, phosphorus_idx: int, oxygen_idx: int) -> str:
    phosphorus = mol.GetAtomWithIdx(phosphorus_idx)
    if phosphorus.GetSymbol() != "P" or phosphorus.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(phosphorus_idx, oxygen_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    chain_length = _neutral_phosphate_chain_length(mol, phosphorus_idx, oxygen_idx)
    if chain_length in {2, 3}:
        return {2: "diphosphooxy", 3: "triphosphooxy"}[chain_length]
    double_oxygen_count = 0
    hydroxy_count = 0
    oxido_count = 0
    ligand_records = []
    for nbr in phosphorus.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == oxygen_idx:
            continue
        bond = mol.GetBondBetweenAtoms(phosphorus_idx, nidx)
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_oxygen_count += 1
            continue
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.SINGLE and len(nbr.GetNeighbors()) == 1:
            if nbr.GetFormalCharge() == 0 and nbr.GetTotalNumHs() == 1:
                hydroxy_count += 1
                continue
            if nbr.GetFormalCharge() < 0:
                oxido_count += 1
                continue
            return ""
        if bond.GetBondType() != Chem.BondType.SINGLE or nbr.GetAtomicNum() <= 1:
            return ""
        if nbr.GetSymbol() not in {"C", "O"}:
            return ""
        ligand = branch_prefix(mol, nidx, phosphorus_idx)
        if not ligand:
            return ""
        ligand_records.append((_phosphorus_oxy_ligand_class(nbr), ligand))
    if double_oxygen_count != 1:
        return ""
    if not ligand_records:
        if hydroxy_count == 2 and oxido_count == 0:
            return "phosphonooxy"
        if hydroxy_count == 0 and oxido_count == 2:
            return "phosphonatooxy"
        return ""
    if hydroxy_count == 1 and oxido_count == 0 and len(ligand_records) == 1:
        _ligand_class, ligand = ligand_records[0]
        if _phosphorus_oxy_ligand_needs_parentheses(ligand):
            return ""
        return _phosphorus_oxy_ligand_unit(ligand) + "phosphonooxy"
    if hydroxy_count == 0 and oxido_count == 0 and len(ligand_records) == 2:
        if any(_phosphorus_oxy_ligand_needs_parentheses(ligand) for _ligand_class, ligand in ligand_records):
            return ""
        if {ligand_class for ligand_class, _ligand in ligand_records} == {0, 1}:
            return _phosphorus_oxy_distinct_ligand_text(ligand_records) + "phosphoryloxy"
        counts = Counter(ligand_records)
        parts = []
        for (_ligand_class, ligand) in sorted(counts):
            count = counts[(_ligand_class, ligand)]
            unit = _phosphorus_oxy_ligand_unit(ligand)
            parts.append(unit if count == 1 else MULT.get(count, "") + unit)
        return "".join(parts) + "phosphoryloxy"
    return ""


def _neutral_phosphate_chain_length(mol, phosphorus_idx: int, blocked_oxygen_idx: int) -> int:
    length = 0
    current_p = phosphorus_idx
    blocked_oxygen = blocked_oxygen_idx
    seen_phosphorus: set[int] = set()
    while current_p not in seen_phosphorus:
        seen_phosphorus.add(current_p)
        atom = mol.GetAtomWithIdx(current_p)
        if atom.GetSymbol() != "P" or atom.GetFormalCharge() != 0:
            return 0
        length += 1
        double_oxygen_count = 0
        hydroxy_count = 0
        bridge: tuple[int, int] | None = None
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx == blocked_oxygen:
                continue
            bond = mol.GetBondBetweenAtoms(current_p, nidx)
            if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
                double_oxygen_count += 1
                continue
            if nbr.GetSymbol() != "O" or bond.GetBondType() != Chem.BondType.SINGLE:
                return 0
            if len(nbr.GetNeighbors()) == 1:
                if nbr.GetFormalCharge() == 0 and nbr.GetTotalNumHs() == 1:
                    hydroxy_count += 1
                    continue
                return 0
            next_phosphorus = [
                other.GetIdx()
                for other in nbr.GetNeighbors()
                if other.GetIdx() != current_p and other.GetSymbol() == "P"
            ]
            if len(next_phosphorus) != 1 or bridge is not None:
                return 0
            bridge = (nidx, next_phosphorus[0])
        if double_oxygen_count != 1:
            return 0
        if bridge is None:
            return length if hydroxy_count == 2 else 0
        if hydroxy_count != 1:
            return 0
        blocked_oxygen, current_p = bridge
    return 0


def _phosphorus_oxy_ligand_unit(ligand: str) -> str:
    if _phosphorus_oxy_ligand_needs_parentheses(ligand):
        return f"({ligand})"
    return ligand


def _phosphorus_oxy_distinct_ligand_text(ligand_records: Sequence[tuple[int, str]]) -> str:
    first, second = sorted(ligand_records)
    return _phosphorus_oxy_ligand_unit(first[1]) + f"({_phosphorus_oxy_ligand_unit(second[1])})"


def _phosphorus_oxy_ligand_class(atom) -> int:
    if atom.GetSymbol() == "C" and not atom.GetIsAromatic():
        return 0
    if atom.GetSymbol() == "O":
        return 1
    return 2


def _phosphorus_oxy_ligand_needs_parentheses(ligand: str) -> bool:
    return "-" in ligand or ligand[:1].isdigit() or ligand.startswith("(")


def _silylethoxy_prefix(mol, carbon_idx: int, oxygen_idx: int) -> str:
    first = mol.GetAtomWithIdx(carbon_idx)
    if first.GetSymbol() != "C" or first.GetFormalCharge() != 0 or first.IsInRing():
        return ""
    second = _single_heavy_neighbor(mol, carbon_idx, oxygen_idx, "C")
    if second is None:
        return ""
    silicon = _single_heavy_neighbor(mol, second, carbon_idx, "Si")
    if silicon is None:
        return ""
    return "2-tms-ethoxy" if _is_trimethylsilyl_group(mol, silicon, second) else ""


def _tbdmsoxyethoxy_prefix(mol, carbon_idx: int, oxygen_idx: int) -> str:
    first = mol.GetAtomWithIdx(carbon_idx)
    if first.GetSymbol() != "C" or first.GetFormalCharge() != 0 or first.IsInRing():
        return ""
    second = _single_heavy_neighbor(mol, carbon_idx, oxygen_idx, "C")
    if second is None:
        return ""
    distal_oxygen = _single_heavy_neighbor(mol, second, carbon_idx, "O")
    if distal_oxygen is None:
        return ""
    silicon = _single_heavy_neighbor(mol, distal_oxygen, second, "Si")
    if silicon is None:
        return ""
    return "2-tbdmsoxyethoxy" if _is_tbdms_group(mol, silicon, distal_oxygen) else ""


def _single_heavy_neighbor(mol, atom_idx: int, blocked_idx: int, symbol: str) -> int | None:
    atom = mol.GetAtomWithIdx(atom_idx)
    heavy = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != blocked_idx and nbr.GetAtomicNum() > 1
    ]
    if len(heavy) != 1:
        return None
    neighbor_idx = heavy[0]
    neighbor = mol.GetAtomWithIdx(neighbor_idx)
    bond = mol.GetBondBetweenAtoms(atom_idx, neighbor_idx)
    if neighbor.GetSymbol() != symbol or bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return None
    return neighbor_idx


def _sulfur_oxy_alias(mol, sulfur_idx: int, oxygen_idx: int) -> str:
    sulfur = mol.GetAtomWithIdx(sulfur_idx)
    if sulfur.GetSymbol() != "S" or sulfur.GetFormalCharge() != 0:
        return ""
    double_oxygen_count = 0
    single_ligands = []
    for nbr in sulfur.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == oxygen_idx:
            continue
        bond = mol.GetBondBetweenAtoms(sulfur_idx, nidx)
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_oxygen_count += 1
        elif bond.GetBondType() == Chem.BondType.SINGLE:
            single_ligands.append(nidx)
        elif nbr.GetAtomicNum() > 1:
            return ""
    if double_oxygen_count != 2 or len(single_ligands) != 1:
        return ""
    ligand_idx = single_ligands[0]
    ligand = mol.GetAtomWithIdx(ligand_idx)
    if ligand.GetSymbol() == "F":
        return "fluorosulfonyloxy"
    if ligand.GetSymbol() == "O" and _is_terminal_hydroxy_oxygen(mol, ligand_idx, sulfur_idx):
        return "sulfooxy"
    if ligand.GetSymbol() == "C" and _is_methyl_sulfonyl_ligand(mol, ligand_idx, sulfur_idx):
        return "mesyloxy"
    if ligand.GetSymbol() == "C" and _is_trifluoromethyl(mol, ligand_idx, sulfur_idx):
        return "triflyloxy"
    if ligand.IsInRing():
        ring = _ring_substituent_prefix(mol, ligand_idx, sulfur_idx)
        if ring in {"p-tolyl", "4-methylphenyl"}:
            return "tosyloxy"
    return ""


def _is_terminal_hydroxy_oxygen(mol, oxygen_idx: int, sulfur_idx: int) -> bool:
    oxygen = mol.GetAtomWithIdx(oxygen_idx)
    if oxygen.GetSymbol() != "O" or oxygen.GetFormalCharge() != 0:
        return False
    return not [
        nbr.GetIdx()
        for nbr in oxygen.GetNeighbors()
        if nbr.GetIdx() != sulfur_idx and nbr.GetAtomicNum() > 1
    ]


def _is_methyl_sulfonyl_ligand(mol, carbon_idx: int, sulfur_idx: int) -> bool:
    carbon = mol.GetAtomWithIdx(carbon_idx)
    if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0 or carbon.IsInRing() or carbon.GetIsotope():
        return False
    bond = mol.GetBondBetweenAtoms(carbon_idx, sulfur_idx)
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return False
    return not [
        nbr.GetIdx()
        for nbr in carbon.GetNeighbors()
        if nbr.GetIdx() != sulfur_idx and nbr.GetAtomicNum() > 1
    ]


def _is_trifluoromethyl(mol, carbon_idx: int, sulfur_idx: int) -> bool:
    carbon = mol.GetAtomWithIdx(carbon_idx)
    if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0:
        return False
    fluorines = []
    for nbr in carbon.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == sulfur_idx:
            continue
        bond = mol.GetBondBetweenAtoms(carbon_idx, nidx)
        if nbr.GetSymbol() == "F" and bond.GetBondType() == Chem.BondType.SINGLE:
            fluorines.append(nidx)
        elif nbr.GetAtomicNum() > 1:
            return False
    return len(fluorines) == 3


def _retained_oxygen_resource_prefix(mol, oxygen_idx: int, parent_idx: int) -> str:
    template = _resource_substituent_match(mol, oxygen_idx, parent_idx, Chem)
    if not template:
        return ""
    name = template_record_name(template)
    if "siloxy" in name or "silanoxy" in name:
        return name
    return ""


def _glycosyloxy_prefix(mol, carbon_idx: int, oxygen_idx: int) -> str:
    if mol.GetAtomWithIdx(carbon_idx).GetSymbol() != "C":
        return ""
    template = _resource_substituent_match(mol, carbon_idx, oxygen_idx, Chem)
    if not template or template.get("source_file") != "carbohydrates.xml":
        return ""
    name = template_record_name(template)
    if not name.endswith(("osyl", "aminyl", "muramyl", "neuraminyl")):
        return ""
    return name + "oxy"


def _sulfur_has_ring_carbon_ligand(mol, sulfur_idx: int, oxygen_idx: int) -> bool:
    sulfur = mol.GetAtomWithIdx(sulfur_idx)
    return any(
        nbr.GetIdx() != oxygen_idx and nbr.GetSymbol() == "C" and nbr.IsInRing()
        for nbr in sulfur.GetNeighbors()
    )


def _sulfur_has_nitroaryl_ligand(mol, sulfur_idx: int, oxygen_idx: int) -> bool:
    sulfur = mol.GetAtomWithIdx(sulfur_idx)
    ring_starts = [
        nbr.GetIdx()
        for nbr in sulfur.GetNeighbors()
        if nbr.GetIdx() != oxygen_idx and nbr.GetSymbol() == "C" and nbr.IsInRing()
    ]
    seen: set[int] = set()
    stack = list(ring_starts)
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        atom = mol.GetAtomWithIdx(current)
        for nbr in atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx == sulfur_idx:
                continue
            if _is_nitro_nitrogen(mol, nbr_idx):
                return True
            if nbr.IsInRing() and nbr_idx not in seen:
                stack.append(nbr_idx)
    return False


def _nonplain_hydrogen_neighbor_indices(atom, parent_idx: int) -> list[int]:
    return [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx and not (nbr.GetSymbol() == "H" and nbr.GetIsotope() == 0)
    ]


def _isotopic_hydrogen_prefix(atom) -> str:
    if atom.GetSymbol() != "H":
        return ""
    if atom.GetIsotope() == 2:
        return "deuterio"
    if atom.GetIsotope() == 3:
        return "tritio"
    return ""


def _isotopic_hydrogen_oxy_prefix(atom) -> str:
    isotope = _isotopic_hydrogen_prefix(atom)
    return isotope + "oxy" if isotope else ""


def _ylidene_oxidaniumyl_prefix(mol, carbon_idx: int, oxygen_idx: int) -> str:
    carbon = mol.GetAtomWithIdx(carbon_idx)
    if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0:
        return ""
    bond = mol.GetBondBetweenAtoms(carbon_idx, oxygen_idx)
    if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
        return ""
    heavy = [nbr.GetIdx() for nbr in carbon.GetNeighbors() if nbr.GetIdx() != oxygen_idx]
    if not heavy:
        return "methylideneoxidaniumyl"
    if len(heavy) != 1:
        return ""
    chain = _linear_carbon_chain(mol, heavy[0], carbon_idx, max_len=9)
    if not chain:
        return ""
    root = ROOTS.get(len(chain) + 1)
    return root + "ylideneoxidaniumyl" if root else ""


def _nitrogen_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if _is_diazo_attachment(mol, idx, parent_idx):
        return "diazo"
    if parent_bond is not None and parent_bond.GetBondType() == Chem.BondType.DOUBLE:
        return _imino_prefix(mol, idx, parent_idx)
    cumulene = _isocyanato_prefix(mol, idx, parent_idx)
    if cumulene:
        return cumulene
    if _is_isocyano_nitrogen(mol, idx):
        return "isocyano"
    if _is_diazonio_attachment(mol, idx, parent_idx):
        return "diazonio"
    azoxy = _azoxy_prefix(mol, idx, parent_idx)
    if azoxy:
        return azoxy
    diazenyl = _diazenyl_prefix(mol, idx, parent_idx)
    if diazenyl:
        return diazenyl
    hydrazone = _hydrazone_prefix(mol, idx, parent_idx)
    if hydrazone:
        return hydrazone
    imino_amino = _imino_amino_prefix(mol, idx, parent_idx)
    if imino_amino:
        return imino_amino
    amidino_amino = _amidino_amino_prefix(mol, idx, parent_idx)
    if amidino_amino:
        return amidino_amino
    sulfur_ylidene_amino = _sulfur_ylidene_amino_prefix(mol, idx, parent_idx)
    if sulfur_ylidene_amino:
        return sulfur_ylidene_amino
    if _is_azido_attachment(mol, idx, parent_idx):
        return "azido"
    if _nitro_branch_predicate(mol, idx, parent_idx):
        return "nitro"
    if _is_nitroso_attachment(mol, idx, parent_idx):
        return "nitroso"
    if atom.GetFormalCharge() > 0:
        heavy = _nonplain_hydrogen_neighbor_indices(atom, parent_idx)
        if not heavy:
            return suffix_supported_term("ium", "azaniumyl")
        names = []
        for nidx in heavy:
            if mol.GetAtomWithIdx(nidx).IsInRing():
                name = _ring_substituent_prefix(mol, nidx, idx)
            else:
                name = branch_prefix(mol, nidx, idx) or substituent_from_root(mol, nidx, idx)
            if not name:
                return ""
            names.append(name)
        counts = Counter(names)
        parts = []
        for name in sorted(counts):
            count = counts[name]
            if count > 1:
                parts.append(MULT.get(count, "") + name)
            else:
                parts.append(name)
        return "".join(parts) + "azaniumyl"
    if atom.GetFormalCharge() < 0:
        return _azanidyl_prefix(mol, idx, parent_idx)
    if atom.GetFormalCharge() != 0:
        return ""
    heavy = _nonplain_hydrogen_neighbor_indices(atom, parent_idx)
    if heavy and all(mol.GetAtomWithIdx(nidx).GetSymbol() in HALO for nidx in heavy):
        names = [HALO[mol.GetAtomWithIdx(nidx).GetSymbol()] for nidx in heavy]
        counts = Counter(names)
        return "".join((MULT.get(count, "") if count > 1 else "") + name for name, count in sorted(counts.items())) + "amino"
    if not heavy:
        return "amino"
    names = []
    for nidx in heavy:
        name = _nitrogen_ligand_prefix(mol, nidx, idx)
        if not name:
            return ""
        names.append(name)
    return _amino_prefix_from_ligands(mol, idx, parent_idx, names)


def _nitrogen_ligand_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    symbol = atom.GetSymbol()
    if symbol == "H":
        return _isotopic_hydrogen_prefix(atom)
    if symbol in HALO:
        isotope = atom.GetIsotope()
        return f"[{isotope}{symbol}]{HALO[symbol]}" if isotope else HALO[symbol]
    if atom.IsInRing():
        return _ring_substituent_prefix(mol, idx, parent_idx)
    if symbol == "N":
        return _nitrogen_prefix(mol, idx, parent_idx)
    if symbol == "S":
        return _sulfur_prefix(mol, idx, parent_idx)
    return (
        suffix_prefix_term("nitrile", "cyano")
        if _is_nitrile_carbon(mol, idx)
        else _functional_carbon_substituent_name(mol, idx, parent_idx)
        or _carbonyl_substituent_prefix(mol, idx, parent_idx)
        or branch_prefix(mol, idx, parent_idx)
        or substituent_from_root(mol, idx, parent_idx)
    )


def _sulfur_ylidene_amino_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0 or atom.IsInRing() or atom.GetIsotope():
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    if not _parent_ring_system_has_adjacent_nitrogens(mol, parent_idx):
        return ""
    heavy = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1]
    if len(heavy) != 1:
        return ""
    sulfur_idx = heavy[0]
    sulfur = mol.GetAtomWithIdx(sulfur_idx)
    nitrogen_sulfur_bond = mol.GetBondBetweenAtoms(idx, sulfur_idx)
    if (
        sulfur.GetSymbol() != "S"
        or sulfur.GetFormalCharge() != 0
        or sulfur.IsInRing()
        or sulfur.GetIsotope()
        or nitrogen_sulfur_bond is None
        or nitrogen_sulfur_bond.GetBondType() != Chem.BondType.DOUBLE
    ):
        return ""

    double_oxygen_count = 0
    ligands = []
    for nbr in sulfur.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == idx:
            continue
        bond = mol.GetBondBetweenAtoms(sulfur_idx, nidx)
        if bond is None:
            return ""
        if (
            nbr.GetSymbol() == "O"
            and nbr.GetFormalCharge() == 0
            and not nbr.IsInRing()
            and not nbr.GetIsotope()
            and bond.GetBondType() == Chem.BondType.DOUBLE
        ):
            if any(extra.GetIdx() != sulfur_idx and extra.GetAtomicNum() > 1 for extra in nbr.GetNeighbors()):
                return ""
            double_oxygen_count += 1
            continue
        if bond.GetBondType() != Chem.BondType.SINGLE or nbr.GetSymbol() != "C":
            return ""
        if nbr.GetFormalCharge() != 0 or nbr.GetIsotope():
            return ""
        ligand = branch_prefix(mol, nidx, sulfur_idx) or substituent_from_root(mol, nidx, sulfur_idx)
        if not ligand:
            return ""
        ligands.append(ligand)
    if double_oxygen_count != 1 or len(ligands) != 2:
        return ""
    ligand_text = _ligand_text(ligands)
    return f"{ligand_text}(oxo)-lambda6-sulfanylideneamino" if ligand_text else ""


def _parent_ring_system_has_adjacent_nitrogens(mol, parent_idx: int) -> bool:
    parent = mol.GetAtomWithIdx(parent_idx)
    if not parent.IsInRing():
        return False
    rings = [set(ring) for ring in mol.GetRingInfo().AtomRings()]
    systems = []
    for ring in rings:
        merged = ring
        remaining = []
        for system in systems:
            if merged & system:
                merged |= system
            else:
                remaining.append(system)
        remaining.append(merged)
        systems = remaining
    parent_system = next((system for system in systems if parent_idx in system), set())
    if not parent_system:
        return False
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        if begin.GetIdx() not in parent_system or end.GetIdx() not in parent_system:
            continue
        if begin.GetSymbol() == "N" and end.GetSymbol() == "N" and begin.IsInRing() and end.IsInRing():
            return True
    return False


def _amino_prefix_from_ligands(mol, amino_idx: int, parent_idx: int, names: Sequence[str]) -> str:
    if len(names) == 1:
        retained = _retained_single_ligand_amino_prefix(mol, amino_idx, parent_idx, names[0])
        if retained:
            return retained
        if names[0].endswith("phosphorothioyl"):
            return f"({names[0]})amino"
        if mol.GetAtomWithIdx(parent_idx).GetSymbol() != "S" and _single_ligand_amino_needs_explicit_n(names[0]):
            return f"N-({names[0]})amino"
        if "-" in names[0] or names[0].startswith("("):
            return f"({names[0]})amino"
        return names[0] + "amino"
    names = tuple(_multiligand_amino_safe_name(name) for name in names)
    counts = Counter(names)
    if len(counts) > 1 and all(count == 1 for count in counts.values()) and any(
        _needs_explicit_n_ligand_name(name) for name in counts
    ):
        parts = []
        for name in sorted(counts):
            parts.append(f"N-({name})" if "-" in name or name.startswith("(") or ")" in name else f"N-{name}")
        return "-".join(parts) + "amino"
    if all(count == 1 for count in counts.values()) and all(
        "-" not in name and not name.startswith("(") and ")" not in name for name in counts
    ):
        ordered = sorted(counts)
        return ordered[0] + "".join(f"({name})" for name in ordered[1:]) + "amino"
    if all(count == 1 for count in counts.values()):
        parts = []
        for name in sorted(counts):
            if "-" in name or name.startswith("(") or ")" in name:
                parts.append(f"N-({name})")
            else:
                parts.append(f"N-{name}")
        return "-".join(parts) + "amino"
    parts = []
    for name in sorted(counts):
        count = counts[name]
        if count > 1 and name.endswith(("sulfonyl", "sulfinyl")):
            multi = {2: "bis", 3: "tris", 4: "tetrakis"}.get(count, MULT.get(count, "") + "kis")
            parts.append(f"{multi}({name})")
        elif count > 1 and (
            "-" in name or name.startswith("(") or _repeated_amino_ligand_needs_complex_multiplier(name, count)
        ):
            multi = {2: "bis", 3: "tris", 4: "tetrakis"}.get(count, MULT.get(count, "") + "kis")
            parts.append(f"{multi}({name})")
        elif count == 1 and (name.endswith("carbonyl") or name.endswith("sulfonyl")) and len(names) > 1:
            parts.append(f"({name})")
        elif "-" in name or name.startswith("("):
            parts.append(f"({name})")
        else:
            parts.append((MULT.get(count, "") if count > 1 else "") + name)
    return "".join(parts) + "amino"


def _repeated_amino_ligand_needs_complex_multiplier(name: str, count: int) -> bool:
    if name in {"methyl", "ethyl", "propyl", "butyl", "phenyl", "benzyl"}:
        return False
    return (count == 3 and name == "decyl") or name.endswith(("methyl", "ethyl", "propyl", "butyl"))


def _single_ligand_amino_needs_explicit_n(name: str) -> bool:
    if "(" not in name or ")" not in name:
        return False
    if name.startswith("("):
        return False
    if name.endswith(
        ("carbonyl", "sulfonyl", "sulfinyl", "sulfonimidoyl", "sulfinimidoyl", "sulfimidoyl")
    ):
        return False
    return True


def _retained_single_ligand_amino_prefix(mol, amino_idx: int, parent_idx: int, name: str) -> str:
    if name.endswith("sulfonyl") and len(name) > len("sulfonyl"):
        return name[: -len("sulfonyl")] + "sulfonamido"
    acyl_amides = {
        "acroyl": "acrylamido",
        "acryloyl": "acrylamido",
        "acetyl": "acetamido",
        "benzoyl": "benzamido",
        "butyryl": "butyramido",
        "cyanoacetyl": "cyanoacetamido",
        "cyanomethylcarbonyl": "cyanoacetamido",
        "cyclopropylcarbonyl": "cyclopropanecarboxamido",
        "cyclobutylcarbonyl": "cyclobutanecarboxamido",
        "cyclopentylcarbonyl": "cyclopentanecarboxamido",
        "cyclohexylcarbonyl": "cyclohexanecarboxamido",
        "difluoroacetyl": "difluoroacetamido",
        "formyl": "formamido",
        "isobutyryl": "isobutyramido",
        "methacryloyl": "methacrylamido",
        "phenylacetyl": "(phenylacetyl)amino",
        "pivaloyl": "pivalamido",
        "propionyl": "propionamido",
        "trifluoroacetyl": "trifluoroacetamido",
        "valeryl": "valeramido",
    }
    if name in acyl_amides:
        return acyl_amides[name]
    if name in {"boc", "tert-butoxycarbonyl"} and _bocamino_prefix_allowed(mol, amino_idx, parent_idx):
        return "bocamino"
    if name == "cbz":
        return "cbzamino"
    if name == "fmoc":
        return "fmocamino"
    if name == "carbamyl":
        return "ureido"
    return ""


def _multiligand_amino_safe_name(name: str) -> str:
    return {
        "boc": "tert-butoxycarbonyl",
        "cbz": "carbbenzoxy",
        "fmoc": "((fluorene-9-yl)methyl)oxycarbonyl",
    }.get(name, name)


def _bocamino_prefix_allowed(mol, amino_idx: int, parent_idx: int) -> bool:
    parent = mol.GetAtomWithIdx(parent_idx)
    if not parent.IsInRing() or mol.GetRingInfo().NumRings() < 2:
        return True
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "C" or not atom.IsInRing():
            continue
        cidx = atom.GetIdx()
        has_exocyclic_oxo = False
        has_ring_nitrogen = False
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(cidx, nidx)
            if (
                nbr.GetSymbol() == "O"
                and not nbr.IsInRing()
                and bond is not None
                and bond.GetBondType() == Chem.BondType.DOUBLE
            ):
                has_exocyclic_oxo = True
            elif nbr.GetSymbol() == "N" and nbr.IsInRing() and nidx != amino_idx:
                has_ring_nitrogen = True
        if has_exocyclic_oxo and has_ring_nitrogen:
            return False
    return True


def _needs_explicit_n_ligand_name(name: str) -> bool:
    return name.endswith(("carbonyl", "formyl")) or name in {"chloroformyl", "fluoroformyl", "bromoformyl", "iodoformyl"}


def _azanidyl_prefix(mol, idx: int, parent_idx: int) -> str:
    parent = mol.GetAtomWithIdx(parent_idx)
    if parent.GetFormalCharge() < 0:
        return ""
    heavy = _nonplain_hydrogen_neighbor_indices(mol.GetAtomWithIdx(idx), parent_idx)
    if not heavy:
        return "azanidyl"
    if len(heavy) != 1:
        return ""
    substituent_idx = heavy[0]
    substituent = mol.GetAtomWithIdx(substituent_idx)
    if substituent.IsInRing():
        name = _ring_substituent_prefix(mol, substituent_idx, idx)
    elif substituent.GetSymbol() in HALO:
        isotope = substituent.GetIsotope()
        symbol = substituent.GetSymbol()
        name = f"[{isotope}{symbol}]{HALO[symbol]}" if isotope else HALO[symbol]
    elif substituent.GetSymbol() == "S":
        name = _sulfur_prefix(mol, substituent_idx, idx)
    elif substituent.GetSymbol() == "C":
        name = (
            suffix_prefix_term("nitrile", "cyano")
            if _is_nitrile_carbon(mol, substituent_idx)
            else _functional_carbon_substituent_name(mol, substituent_idx, idx)
            or _carbonyl_substituent_prefix(mol, substituent_idx, idx)
            or substituent_from_root(mol, substituent_idx, idx)
        )
    elif substituent.GetSymbol() == "N":
        name = _nitrogen_prefix(mol, substituent_idx, idx)
    else:
        return ""
    if not name:
        return ""
    if "-" in name or name.startswith("("):
        return f"({name})azanidyl"
    return name + "azanidyl"


def _functional_carbon_substituent_name(mol, idx: int, parent_idx: int) -> str:
    if mol.GetAtomWithIdx(idx).GetSymbol() != "C":
        return ""
    return (
        _ketenyl_prefix(mol, idx, parent_idx)
        or _enone_substituent_prefix(mol, idx, parent_idx)
        or _enal_substituent_prefix(mol, idx, parent_idx)
        or _haloethenyl_prefix(mol, idx, parent_idx)
        or _alkynyl_substituent_prefix(mol, idx, parent_idx)
    )


def _imino_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N":
        return ""
    if atom.GetFormalCharge() == 1 and atom.GetTotalNumHs() == 2:
        heavy = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() != parent_idx and n.GetAtomicNum() > 1]
        if not heavy and not atom.IsInRing() and not atom.GetIsotope():
            return "azaniumylidene"
        return ""
    if atom.GetFormalCharge() == 1:
        return _substituted_azaniumylidene_prefix(mol, idx, parent_idx)
    if atom.GetFormalCharge() != 0:
        return ""
    heavy = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() != parent_idx]
    if not heavy:
        return "imino"
    if len(heavy) != 1:
        return ""
    attached = heavy[0]
    if _nitro_branch_predicate(mol, attached, idx):
        return "nitroimino" if _nitroimino_parent_site_supported(mol, parent_idx) else ""
    if mol.GetAtomWithIdx(attached).IsInRing():
        name = _ring_substituent_prefix(mol, attached, idx)
    else:
        name = (
            suffix_prefix_term("nitrile", "cyano")
            if _is_nitrile_carbon(mol, attached)
            else substituent_from_root(mol, attached, idx)
        )
    if not name:
        return ""
    if "-" in name or name.startswith("("):
        return f"({name})imino"
    return name + "imino"


def _substituted_azaniumylidene_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.IsInRing() or atom.GetIsotope():
        return ""
    for ring in mol.GetRingInfo().AtomRings():
        if parent_idx in ring and len(ring) <= 3:
            return ""
    names = []
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if (
            nbr.GetSymbol() != "C"
            or nbr.GetFormalCharge() != 0
            or bond is None
            or bond.GetBondType() != Chem.BondType.SINGLE
        ):
            return ""
        if nbr.IsInRing():
            name = _ring_substituent_prefix(mol, nidx, idx)
        else:
            name = branch_prefix(mol, nidx, idx) or substituent_from_root(mol, nidx, idx)
        if not name or " " in name or "[" in name or "]" in name:
            return ""
        names.append(name)
    if not names:
        return ""
    counts = Counter(names)
    parts = []
    for name in sorted(counts):
        count = counts[name]
        if count > 1:
            multiplier = MULT.get(count)
            if not multiplier:
                return ""
            parts.append(multiplier + name)
        else:
            parts.append(name)
    return "".join(parts) + "azaniumylidene"


def _nitroimino_parent_site_supported(mol, parent_idx: int) -> bool:
    parent = mol.GetAtomWithIdx(parent_idx)
    if not parent.IsInRing():
        return False
    for ring in mol.GetRingInfo().AtomRings():
        if parent_idx not in ring or len(ring) != 5:
            continue
        symbols = [mol.GetAtomWithIdx(idx).GetSymbol() for idx in ring]
        if Counter(symbols) == Counter({"C": 2, "N": 3}) and all(
            mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring
        ):
            return False
    return True


def _isocyanato_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    for nbr in atom.GetNeighbors():
        cidx = nbr.GetIdx()
        if cidx == parent_idx or nbr.GetSymbol() != "C":
            continue
        n_c_bond = mol.GetBondBetweenAtoms(idx, cidx)
        if n_c_bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        terminal = [
            other
            for other in nbr.GetNeighbors()
            if other.GetIdx() != idx
            and mol.GetBondBetweenAtoms(cidx, other.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
        ]
        if len(terminal) != 1:
            continue
        symbol = terminal[0].GetSymbol()
        if symbol == "O":
            return "isocyanato"
        if symbol == "S":
            return "isothiocyanato"
    return ""


def _diazenyl_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0:
        return ""
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx or nbr.GetSymbol() != "N" or nbr.GetFormalCharge() != 0:
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        distal = [n.GetIdx() for n in nbr.GetNeighbors() if n.GetIdx() != idx]
        if not distal:
            return "diazenyl"
        if len(distal) != 1:
            continue
        didx = distal[0]
        if mol.GetAtomWithIdx(didx).IsInRing():
            name = _ring_substituent_prefix(mol, didx, nidx)
        else:
            name = substituent_from_root(mol, didx, nidx)
        if name:
            return name + "diazenyl"
    return ""


def _azoxy_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx or nbr.GetSymbol() != "N" or nbr.GetFormalCharge() <= 0:
            continue
        n_bond = mol.GetBondBetweenAtoms(idx, nidx)
        if n_bond is None or n_bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        oxido = []
        organic = []
        for distal in nbr.GetNeighbors():
            didx = distal.GetIdx()
            if didx == idx:
                continue
            bond = mol.GetBondBetweenAtoms(nidx, didx)
            if (
                distal.GetSymbol() == "O"
                and distal.GetFormalCharge() < 0
                and bond is not None
                and bond.GetBondType() == Chem.BondType.SINGLE
            ):
                oxido.append(didx)
            elif bond is not None and bond.GetBondType() == Chem.BondType.SINGLE:
                organic.append(didx)
        if len(oxido) != 1 or len(organic) != 1:
            continue
        didx = organic[0]
        organic_atom = mol.GetAtomWithIdx(didx)
        if organic_atom.IsInRing():
            name = _ring_substituent_prefix(mol, didx, nidx)
            if name and not (name.startswith("(") or "-" in name):
                return f"({name}azoxy)"
        else:
            name = substituent_from_root(mol, didx, nidx)
        if not name:
            continue
        if "-" in name or name.startswith("("):
            return f"({name})azoxy"
        return name + "azoxy"
    return ""


def _hydrazone_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    heavy = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetIdx() != parent_idx]
    if len(heavy) != 1:
        return ""
    distal_n_idx = heavy[0]
    distal_n = mol.GetAtomWithIdx(distal_n_idx)
    if distal_n.GetSymbol() != "N" or distal_n.GetFormalCharge() != 0:
        return ""
    n_bond = mol.GetBondBetweenAtoms(idx, distal_n_idx)
    if n_bond is None or n_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    imine_carbons = [
        nbr.GetIdx()
        for nbr in distal_n.GetNeighbors()
        if nbr.GetIdx() != idx
        and nbr.GetSymbol() == "C"
        and mol.GetBondBetweenAtoms(distal_n_idx, nbr.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
    ]
    if len(imine_carbons) != 1:
        return ""
    ylidene = _imine_carbon_ylidene(mol, imine_carbons[0], distal_n_idx)
    return ylidene + "hydrazinyl" if ylidene else ""


def _imino_amino_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    imine_carbons = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx
        and nbr.GetSymbol() == "C"
        and mol.GetBondBetweenAtoms(idx, nbr.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
    ]
    if len(imine_carbons) != 1:
        return ""
    ylidene = _imine_carbon_ylidene(mol, imine_carbons[0], idx)
    return ylidene + "amino" if ylidene else ""


def _amidino_amino_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    heavy = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetIdx() != parent_idx]
    if len(heavy) != 1:
        return ""
    carbon_idx = heavy[0]
    carbon = mol.GetAtomWithIdx(carbon_idx)
    if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0:
        return ""
    n_c_bond = mol.GetBondBetweenAtoms(idx, carbon_idx)
    if n_c_bond is None or n_c_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    terminal_imino = []
    other_heavy = []
    for nbr in carbon.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == idx:
            continue
        bond = mol.GetBondBetweenAtoms(carbon_idx, nidx)
        if (
            nbr.GetSymbol() == "N"
            and nbr.GetFormalCharge() == 0
            and bond is not None
            and bond.GetBondType() == Chem.BondType.DOUBLE
            and not [nnbr for nnbr in nbr.GetNeighbors() if nnbr.GetIdx() != carbon_idx and nnbr.GetAtomicNum() > 1]
        ):
            terminal_imino.append(nidx)
        else:
            other_heavy.append(nidx)
    if len(terminal_imino) != 1 or other_heavy:
        return ""
    return "iminomethylamino"


def _imine_carbon_ylidene(mol, carbon_idx: int, nitrogen_idx: int) -> str:
    atom = mol.GetAtomWithIdx(carbon_idx)
    if atom.IsInRing():
        rings = [ring for ring in mol.GetRingInfo().AtomRings() if carbon_idx in ring]
        if len(rings) != 1:
            return ""
        ring = rings[0]
        if not all(mol.GetAtomWithIdx(idx).GetSymbol() == "C" for idx in ring):
            return ""
        ring_set = set(ring)
        for a, b in zip(ring, ring[1:] + ring[:1]):
            bond = mol.GetBondBetweenAtoms(a, b)
            if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
                return ""
        for ring_idx in ring:
            for nbr in mol.GetAtomWithIdx(ring_idx).GetNeighbors():
                nidx = nbr.GetIdx()
                if nidx not in ring_set and nidx != nitrogen_idx:
                    return ""
        root = ROOTS.get(len(ring))
        return f"cyclo{root}ylidene" if root else ""
    substituents = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetIdx() != nitrogen_idx]
    if len(substituents) == 1:
        sub_idx = substituents[0]
        sub_atom = mol.GetAtomWithIdx(sub_idx)
        sub_bond = mol.GetBondBetweenAtoms(carbon_idx, sub_idx)
        if (
            sub_atom.GetSymbol() == "N"
            and sub_atom.GetFormalCharge() == 0
            and sub_bond is not None
            and sub_bond.GetBondType() == Chem.BondType.SINGLE
            and not [nbr for nbr in sub_atom.GetNeighbors() if nbr.GetIdx() != carbon_idx and nbr.GetAtomicNum() > 1]
        ):
            return "aminomethylidene"
        if (
            sub_atom.GetSymbol() == "N"
            and sub_atom.GetFormalCharge() == 0
            and sub_bond is not None
            and sub_bond.GetBondType() == Chem.BondType.SINGLE
        ):
            amino = _nitrogen_prefix(mol, sub_idx, carbon_idx)
            if amino and amino.endswith("amino") and "-" not in amino and "(" not in amino and ")" not in amino:
                return amino + "methylidene"
        if (
            sub_atom.GetSymbol() == "N"
            and sub_atom.GetFormalCharge() == 0
            and sub_bond is not None
            and sub_bond.GetBondType() == Chem.BondType.DOUBLE
            and not [nbr for nbr in sub_atom.GetNeighbors() if nbr.GetIdx() != carbon_idx and nbr.GetAtomicNum() > 1]
        ):
            return "iminomethylidene"
        if mol.GetAtomWithIdx(sub_idx).IsInRing():
            name = _ring_substituent_prefix(mol, sub_idx, carbon_idx)
            if name and name.endswith("yl"):
                return name + "methylidene"
            return ""
        name = substituent_from_root(mol, sub_idx, carbon_idx)
        if name and "-" not in name and not name.startswith("("):
            return name + "methylidene"
        return ""
    if len(substituents) == 2:
        functional_ethylidene = _branch_rules()._functionalized_ethylidene_prefix(
            mol,
            substituents,
            carbon_idx,
        )
        if functional_ethylidene:
            return functional_ethylidene
        names = []
        for sub_idx in substituents:
            if mol.GetAtomWithIdx(sub_idx).IsInRing():
                name = _ring_substituent_prefix(mol, sub_idx, carbon_idx)
            else:
                name = substituent_from_root(mol, sub_idx, carbon_idx)
            if not name or "-" in name or name.startswith("("):
                return ""
            names.append(name)
        counts = Counter(names)
        parts = []
        for name in sorted(counts):
            count = counts[name]
            parts.append((MULT.get(count, "") if count > 1 else "") + name)
        return "".join(parts) + "methylidene"
    return _alkylidene_prefix(mol, carbon_idx, nitrogen_idx)
