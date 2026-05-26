from __future__ import annotations

from functools import lru_cache

from .name_parts import NameParts, Suffix
from .suffix_store import inverse_suffix_store


@lru_cache(maxsize=None)
def _suffix_value(value: str) -> str:
    specs = inverse_suffix_store().by_value.get(value)
    return specs[0].value if specs else value


class FragmentAssemblyRule:
    """First-class naming rule for disconnected fragments, salts, and adducts."""

    def __init__(self, Chem, engine):
        self.Chem = Chem
        self.engine = engine

    def name(self, fragments: tuple[object, ...]) -> str:
        named_fragments: list[str] = []
        cations: list[str] = []
        neutral_halide_suffixes: list[str] = []
        neutral_halide_names: list[str] = []
        anions: list[str] = []

        for fragment in fragments:
            halide_anion = single_atom_halide_anion_name(fragment)
            if halide_anion is not None:
                anions.append(halide_anion)
                continue
            neutral_halide = neutral_halide_name(fragment)
            if neutral_halide is not None:
                symbol, _charge = single_atom_halide(fragment)
                neutral_halide_suffixes.append(neutral_halide)
                neutral_halide_names.append(f"hydrogen {HALIDE_ANIONS[symbol]}")
                continue
            common_anion = common_single_atom_anion_name(fragment)
            if common_anion is not None:
                anions.append(common_anion)
                continue
            cation = single_atom_cation_name(fragment, self.Chem)
            if cation is not None:
                cations.append(cation)
                continue
            polyatomic = common_polyatomic_fragment_name(fragment, self.Chem)
            if polyatomic:
                anions.append(polyatomic)
                continue
            simple_name = simple_fragment_name(fragment, self.Chem)
            if simple_name:
                named_fragments.append(simple_name)
                continue
            name = self.engine.name(fragment)
            if not name:
                return ""
            named_fragments.append(name)

        if len(named_fragments) == 1 and neutral_halide_suffixes and not anions and not cations:
            return _salt_name(named_fragments[0], neutral_halide_suffixes)
        if len(named_fragments) == 1 and anions and not neutral_halide_suffixes and not cations:
            return _salt_name(named_fragments[0], anions)
        if len(cations) == 1 and anions and not named_fragments and not neutral_halide_suffixes:
            return _salt_name(cations[0], anions)

        standalone = cations + named_fragments + anions + neutral_halide_names
        if standalone:
            return " and ".join(sorted(standalone, key=lambda item: (len(item), item)))
        return ""


HALIDE_STEMS = {"F": "fluor", "Cl": "chlor", "Br": "brom", "I": "iod"}

NEUTRAL_HALIDE_SALTS = {
    symbol: "hydro" + stem + _suffix_value("ide")
    for symbol, stem in HALIDE_STEMS.items()
}

HALIDE_ANIONS = {symbol: stem + _suffix_value("ide") for symbol, stem in HALIDE_STEMS.items()}

COMMON_MONATOMIC_CATIONS = {
    "Li": "lithium",
    "Na": "sodium",
    "K": "potassium",
    "Rb": "rubidium",
    "Cs": "cesium",
    "Mg": "magnesium",
    "Ca": "calcium",
    "Sr": "strontium",
    "Ba": "barium",
    "Al": "aluminium",
    "Zn": "zinc",
    "Ag": "silver",
}

COMMON_SINGLE_ATOM_CATIONS = {
    ("N", 1, 4): "ammonium",
    ("O", 1, 3): "hydronium",
    ("P", 1, 4): "phosphonium",
    ("S", 1, 3): "sulfonium",
}

NEUTRAL_HYDRIDES = {
    ("B", 3): "borane",
    ("C", 4): "methane",
    ("N", 3): "ammonia",
    ("O", 2): "water",
    ("P", 3): "phosphane",
    ("S", 2): "sulfane",
    ("Si", 4): "silane",
}

MULTIPLIERS = {
    2: "di",
    3: "tri",
    4: "tetra",
    5: "penta",
    6: "hexa",
}

COMPLEX_MULTIPLIERS = {
    2: "bis",
    3: "tris",
    4: "tetrakis",
    5: "pentakis",
    6: "hexakis",
}

PERFLUORO_ANIONS = {
    ("B", 4, -1): "tetrafluoroborate",
    ("P", 6, -1): "hexafluorophosphate",
    ("As", 6, -1): "hexafluoroarsenate",
    ("Sb", 6, -1): "hexafluoroantimonate",
}

OXOANIONS = {
    ("C", 3, -2, 0): "carbonate",
    ("C", 3, -1, 1): "hydrogen carbonate",
    ("N", 2, -1, 0): "nitrite",
    ("N", 3, -1, 0): "nitrate",
    ("P", 4, -3, 0): "phosphate",
    ("P", 4, -2, 1): "hydrogen phosphate",
    ("P", 4, -1, 2): "dihydrogen phosphate",
    ("S", 3, -2, 0): "sulfite",
    ("S", 3, -1, 1): "hydrogen sulfite",
    ("S", 4, -2, 0): "sulfate",
    ("S", 4, -1, 1): "hydrogen sulfate",
    ("Cl", 1, -1, 0): "hypochlorite",
    ("Cl", 2, -1, 0): "chlorite",
    ("Cl", 3, -1, 0): "chlorate",
    ("Cl", 4, -1, 0): "perchlorate",
    ("Br", 1, -1, 0): "hypobromite",
    ("Br", 2, -1, 0): "bromite",
    ("Br", 3, -1, 0): "bromate",
    ("Br", 4, -1, 0): "perbromate",
    ("I", 1, -1, 0): "hypoiodite",
    ("I", 3, -1, 0): "iodate",
    ("I", 4, -1, 0): "periodate",
}


def single_atom_halide(mol) -> tuple[str, int] | None:
    if mol.GetNumAtoms() != 1:
        return None
    atom = mol.GetAtomWithIdx(0)
    symbol = atom.GetSymbol()
    if symbol not in HALIDE_ANIONS:
        return None
    return symbol, atom.GetFormalCharge()


def single_atom_halide_anion_name(mol) -> str | None:
    halide = single_atom_halide(mol)
    if halide is None:
        return None
    symbol, charge = halide
    return HALIDE_ANIONS[symbol] if charge < 0 else None


def neutral_halide_name(mol) -> str | None:
    halide = single_atom_halide(mol)
    if halide is None:
        return None
    symbol, charge = halide
    return NEUTRAL_HALIDE_SALTS[symbol] if charge == 0 else None


def common_single_atom_anion_name(mol) -> str | None:
    if mol.GetNumAtoms() != 1:
        return None
    atom = mol.GetAtomWithIdx(0)
    hydride = _hydride_anion_name(atom)
    if hydride:
        return hydride
    if atom.GetSymbol() == "B" and atom.GetFormalCharge() == -1 and atom.GetTotalNumHs() == 4:
        return "borohydride"
    if atom.GetSymbol() == "O" and atom.GetFormalCharge() < 0 and atom.GetTotalNumHs() > 0:
        return "hydroxide"
    return None


def single_atom_cation_name(mol, Chem) -> str | None:
    if mol.GetNumAtoms() != 1:
        return None
    atom = mol.GetAtomWithIdx(0)
    if atom.GetFormalCharge() <= 0:
        return None
    symbol = atom.GetSymbol()
    common = COMMON_SINGLE_ATOM_CATIONS.get((symbol, atom.GetFormalCharge(), atom.GetTotalNumHs()))
    if common:
        return common
    if symbol in COMMON_MONATOMIC_CATIONS:
        return COMMON_MONATOMIC_CATIONS[symbol]
    element_name = Chem.GetPeriodicTable().GetElementName(atom.GetAtomicNum()).lower()
    return NameParts(parent=element_name, charge=f"({atom.GetFormalCharge()}+)").render()


def single_atom_anion_name(mol, Chem) -> str | None:
    if mol.GetNumAtoms() != 1:
        return None
    atom = mol.GetAtomWithIdx(0)
    if atom.GetFormalCharge() >= 0:
        return None
    symbol = atom.GetSymbol()
    if symbol in HALIDE_ANIONS:
        return HALIDE_ANIONS[symbol]
    element_name = Chem.GetPeriodicTable().GetElementName(atom.GetAtomicNum()).lower()
    return NameParts(parent=element_name, charge=f"({abs(atom.GetFormalCharge())}-)").render()


def simple_fragment_name(mol, Chem) -> str:
    if mol.GetNumAtoms() != 1:
        return ""
    atom = mol.GetAtomWithIdx(0)
    symbol = atom.GetSymbol()
    charge = atom.GetFormalCharge()
    common_anion = common_single_atom_anion_name(mol)
    if common_anion:
        return common_anion
    anion = single_atom_anion_name(mol, Chem)
    if anion:
        return anion
    neutral_halide = neutral_halide_name(mol)
    if neutral_halide:
        return f"hydrogen {HALIDE_ANIONS[symbol]}"
    cation = single_atom_cation_name(mol, Chem)
    if cation:
        return cation
    neutral_hydride = NEUTRAL_HYDRIDES.get((symbol, atom.GetTotalNumHs()))
    if neutral_hydride and charge == 0:
        return neutral_hydride

    element_name = Chem.GetPeriodicTable().GetElementName(atom.GetAtomicNum()).lower()
    if not element_name:
        return ""
    if charge == 0:
        return element_name
    sign = "+" if charge > 0 else "-"
    return NameParts(parent=element_name, charge=f"({abs(charge)}{sign})").render()


def common_polyatomic_fragment_name(mol, Chem) -> str:
    """Name common standalone counterions by graph features, not row lookups."""

    if mol.GetNumAtoms() < 2:
        return ""
    return (
        _substituted_borohydride_name(mol, Chem)
        or _organotrifluoroborate_name(mol, Chem)
        or _tetraorganoborate_name(mol, Chem)
        or _azanide_name(mol, Chem)
        or _perfluoro_counterion_name(mol, Chem)
        or _oxoanion_name(mol, Chem)
        or _organophosphorus_oxoanion_name(mol, Chem)
        or _pseudohalide_name(mol, Chem)
        or _bis_trifluoromethanesulfonyl_azanide_name(mol, Chem)
        or _trifluoromethanesulfonate_name(mol, Chem)
    )


def _hydride_anion_name(atom) -> str:
    if atom.GetSymbol() != "H" or atom.GetFormalCharge() >= 0:
        return ""
    if atom.GetIsotope() == 2:
        return "deuteride"
    if atom.GetIsotope() == 3:
        return "tritide"
    return "hydride"


def _substituted_borohydride_name(mol, Chem) -> str:
    borons = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == "B" and atom.GetFormalCharge() == -1]
    if len(borons) != 1:
        return ""
    boron = borons[0]
    bidx = boron.GetIdx()
    if any(
        mol.GetBondBetweenAtoms(bidx, nbr.GetIdx()).GetBondType() != Chem.BondType.SINGLE
        for nbr in boron.GetNeighbors()
    ):
        return ""
    hydrogens = boron.GetTotalNumHs()
    neighbors = [nbr.GetIdx() for nbr in boron.GetNeighbors()]
    if hydrogens == 3 and len(neighbors) == 1 and _is_cyano_group(mol, Chem, neighbors[0], bidx):
        return "cyanoborohydride"
    if hydrogens == 1 and len(neighbors) == 3 and all(_is_acetoxy_group(mol, Chem, nidx, bidx) for nidx in neighbors):
        return "triacetoxyborohydride"
    return ""


def _is_cyano_group(mol, Chem, carbon_idx: int, boron_idx: int) -> bool:
    carbon = mol.GetAtomWithIdx(carbon_idx)
    if carbon.GetSymbol() != "C":
        return False
    nitrogen_neighbors = [
        nbr
        for nbr in carbon.GetNeighbors()
        if nbr.GetIdx() != boron_idx
        and nbr.GetSymbol() == "N"
        and mol.GetBondBetweenAtoms(carbon_idx, nbr.GetIdx()).GetBondType() == Chem.BondType.TRIPLE
    ]
    return len(nitrogen_neighbors) == 1 and carbon.GetDegree() == 2


def _is_acetoxy_group(mol, Chem, oxygen_idx: int, boron_idx: int) -> bool:
    oxygen = mol.GetAtomWithIdx(oxygen_idx)
    if oxygen.GetSymbol() != "O":
        return False
    carbon_neighbors = [
        nbr.GetIdx()
        for nbr in oxygen.GetNeighbors()
        if nbr.GetIdx() != boron_idx
        and nbr.GetSymbol() == "C"
        and mol.GetBondBetweenAtoms(oxygen_idx, nbr.GetIdx()).GetBondType() == Chem.BondType.SINGLE
    ]
    if len(carbon_neighbors) != 1:
        return False
    carbon_idx = carbon_neighbors[0]
    carbon = mol.GetAtomWithIdx(carbon_idx)
    double_oxygen_count = 0
    methyl_count = 0
    for nbr in carbon.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == oxygen_idx:
            continue
        bond = mol.GetBondBetweenAtoms(carbon_idx, nidx)
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_oxygen_count += 1
        elif nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetTotalNumHs() == 3:
            methyl_count += 1
        else:
            return False
    return double_oxygen_count == 1 and methyl_count == 1


def _organotrifluoroborate_name(mol, Chem) -> str:
    borons = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == "B" and atom.GetFormalCharge() == -1]
    if len(borons) != 1 or _total_charge(mol) != -1:
        return ""
    boron = borons[0]
    bidx = boron.GetIdx()
    carbon_neighbors = []
    fluorine_count = 0
    for nbr in boron.GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(bidx, nidx)
        if bond.GetBondType() != Chem.BondType.SINGLE:
            return ""
        if nbr.GetSymbol() == "F":
            fluorine_count += 1
        elif nbr.GetSymbol() == "C":
            carbon_neighbors.append(nidx)
        else:
            return ""
    if fluorine_count != 3 or len(carbon_neighbors) != 1:
        return ""
    from .branch_rules import branch_prefix

    prefix = branch_prefix(mol, carbon_neighbors[0], bidx)
    if not prefix:
        return ""
    if _needs_fragment_parentheses(prefix):
        prefix = f"({prefix})"
    return prefix + "trifluoroborate"


def _tetraorganoborate_name(mol, Chem) -> str:
    borons = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == "B" and atom.GetFormalCharge() == -1]
    if len(borons) != 1 or _total_charge(mol) != -1:
        return ""
    boron = borons[0]
    bidx = boron.GetIdx()
    names = []
    for nbr in boron.GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(bidx, nidx)
        if nbr.GetSymbol() != "C" or bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return ""
        from .branch_rules import branch_prefix

        prefix = branch_prefix(mol, nidx, bidx)
        if not prefix:
            return ""
        names.append(prefix)
    if len(names) != 4:
        return ""
    prefix_text = _substituted_borate_prefix_text(names)
    return prefix_text + "borate" if prefix_text else ""


def _substituted_borate_prefix_text(names: list[str]) -> str:
    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    terms = []
    for name, count in counts.items():
        is_complex = _needs_fragment_parentheses(name)
        if count > 1 and is_complex:
            multiplier = COMPLEX_MULTIPLIERS.get(count, "")
            term = f"{multiplier}({name})" if multiplier else ""
        elif count > 1:
            multiplier = MULTIPLIERS.get(count, "")
            term = multiplier + name if multiplier else ""
        elif is_complex:
            term = f"({name})"
        else:
            term = name
        if not term:
            return ""
        terms.append((0 if is_complex else 1, name, term))
    return "".join(term for _complex_order, _name, term in sorted(terms))


def _azanide_name(mol, Chem) -> str:
    nitrogens = [
        atom
        for atom in mol.GetAtoms()
        if atom.GetSymbol() == "N"
        and atom.GetFormalCharge() == -1
        and not atom.IsInRing()
    ]
    if len(nitrogens) != 1 or _total_charge(mol) != -1:
        return ""
    nitrogen = nitrogens[0]
    nidx = nitrogen.GetIdx()
    neighbors = [nbr.GetIdx() for nbr in nitrogen.GetNeighbors()]
    implicit_hydrogens = nitrogen.GetTotalNumHs()
    if len(neighbors) + implicit_hydrogens != 2 or len(neighbors) > 2:
        return ""
    names = []
    for nbr_idx in neighbors:
        bond = mol.GetBondBetweenAtoms(nidx, nbr_idx)
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return ""
        prefix = _azanide_substituent_prefix(mol, Chem, nbr_idx, nidx)
        if not prefix:
            return ""
        names.append(prefix)
    if not names:
        return "azanide"
    from .parent.rule_helpers import _substituted_onium_name

    return _substituted_onium_name(names, "azanide")


def _azanide_substituent_prefix(mol, Chem, atom_idx: int, nitrogen_idx: int) -> str:
    atom = mol.GetAtomWithIdx(atom_idx)
    if atom.GetSymbol() == "C":
        from .branch_rules import branch_prefix

        return branch_prefix(mol, atom_idx, nitrogen_idx)
    if atom.GetSymbol() != "Si" or atom.GetFormalCharge() != 0:
        return ""
    names = []
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == nitrogen_idx:
            continue
        bond = mol.GetBondBetweenAtoms(atom_idx, nidx)
        if nbr.GetSymbol() != "C" or bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return ""
        from .branch_rules import branch_prefix

        prefix = branch_prefix(mol, nidx, atom_idx)
        if not prefix:
            return ""
        names.append(prefix)
    if len(names) + atom.GetTotalNumHs() != 3:
        return ""
    from .parent.rule_helpers import _substituted_onium_name

    return _substituted_onium_name(names, "silyl")


def _perfluoro_counterion_name(mol, Chem) -> str:
    atoms = list(mol.GetAtoms())
    centers = [atom for atom in atoms if atom.GetSymbol() != "F"]
    fluorines = [atom for atom in atoms if atom.GetSymbol() == "F"]
    total_charge = _total_charge(mol)
    if len(centers) != 1 or len(fluorines) != len(atoms) - 1:
        return ""
    center = centers[0]
    if center.GetDegree() != len(fluorines):
        return ""
    for fluorine in fluorines:
        bond = mol.GetBondBetweenAtoms(center.GetIdx(), fluorine.GetIdx())
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return ""
    return PERFLUORO_ANIONS.get((center.GetSymbol(), len(fluorines), total_charge), "")


def _oxoanion_name(mol, Chem) -> str:
    atoms = list(mol.GetAtoms())
    centers = [atom for atom in atoms if atom.GetSymbol() != "O"]
    oxygens = [atom for atom in atoms if atom.GetSymbol() == "O"]
    if len(centers) != 1 or len(oxygens) != len(atoms) - 1:
        return ""
    center = centers[0]
    if any(oxygen.GetDegree() != 1 for oxygen in oxygens):
        return ""
    if any(mol.GetBondBetweenAtoms(center.GetIdx(), oxygen.GetIdx()) is None for oxygen in oxygens):
        return ""
    hydroxy_count = sum(oxygen.GetTotalNumHs() for oxygen in oxygens)
    return OXOANIONS.get((center.GetSymbol(), len(oxygens), _total_charge(mol), hydroxy_count), "")


def _organophosphorus_oxoanion_name(mol, Chem) -> str:
    phosphorus_atoms = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == "P" and atom.GetFormalCharge() == 0]
    if len(phosphorus_atoms) != 1:
        return ""
    phosphorus = phosphorus_atoms[0]
    pidx = phosphorus.GetIdx()
    carbon_neighbors = []
    double_oxygen_count = 0
    hydroxy_count = 0
    oxido_count = 0
    for nbr in phosphorus.GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(pidx, nidx)
        if nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE:
            carbon_neighbors.append(nidx)
        elif nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_oxygen_count += 1
        elif nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.SINGLE and len(nbr.GetNeighbors()) == 1:
            if nbr.GetFormalCharge() < 0:
                oxido_count += 1
            elif nbr.GetFormalCharge() == 0 and nbr.GetTotalNumHs() == 1:
                hydroxy_count += 1
            else:
                return ""
        else:
            return ""
    if len(carbon_neighbors) != 1 or double_oxygen_count != 1:
        return ""
    implicit_hydrogens = phosphorus.GetTotalNumHs()
    total_charge = _total_charge(mol)
    if implicit_hydrogens == 1 and oxido_count == 1 and hydroxy_count == 0 and total_charge == -1:
        suffix = "phosphinate"
        hydrogen_prefix = ""
    elif implicit_hydrogens == 0 and oxido_count == 2 and hydroxy_count == 0 and total_charge == -2:
        suffix = "phosphonate"
        hydrogen_prefix = ""
    elif implicit_hydrogens == 0 and oxido_count == 1 and hydroxy_count == 1 and total_charge == -1:
        suffix = "phosphonate"
        hydrogen_prefix = "hydrogen "
    else:
        return ""
    from .branch_rules import branch_prefix

    prefix = branch_prefix(mol, carbon_neighbors[0], pidx)
    if not prefix:
        return ""
    if _needs_fragment_parentheses(prefix):
        prefix = f"({prefix})"
    return hydrogen_prefix + prefix + suffix


def _pseudohalide_name(mol, Chem) -> str:
    counts = _element_counts(mol)
    total_charge = _total_charge(mol)
    if total_charge != -1:
        return ""
    if counts == {"C": 1, "N": 1}:
        return "cyanide" if _has_bond(mol, Chem, "C", "N", Chem.BondType.TRIPLE) else ""
    if counts == {"N": 3}:
        return "azide" if any(atom.GetDegree() == 2 for atom in mol.GetAtoms()) else ""
    if counts == {"C": 1, "N": 1, "O": 1}:
        if _has_bond(mol, Chem, "C", "N", Chem.BondType.TRIPLE):
            return "cyanate"
        if _has_bond(mol, Chem, "C", "N", Chem.BondType.DOUBLE) and _has_bond(
            mol, Chem, "C", "O", Chem.BondType.DOUBLE
        ):
            return "isocyanate"
    if counts == {"C": 1, "N": 1, "S": 1}:
        if _has_bond(mol, Chem, "C", "N", Chem.BondType.TRIPLE):
            return "thiocyanate"
        if _has_bond(mol, Chem, "C", "N", Chem.BondType.DOUBLE) and _has_bond(
            mol, Chem, "C", "S", Chem.BondType.DOUBLE
        ):
            return "isothiocyanate"
    return ""


def _trifluoromethanesulfonate_name(mol, Chem) -> str:
    counts = _element_counts(mol)
    if counts != {"C": 1, "F": 3, "O": 3, "S": 1} or _total_charge(mol) != -1:
        return ""
    carbon = next(atom for atom in mol.GetAtoms() if atom.GetSymbol() == "C")
    sulfur = next(atom for atom in mol.GetAtoms() if atom.GetSymbol() == "S")
    carbon_neighbors = {neighbor.GetSymbol() for neighbor in carbon.GetNeighbors()}
    sulfur_oxygen_count = sum(1 for neighbor in sulfur.GetNeighbors() if neighbor.GetSymbol() == "O")
    if carbon_neighbors == {"F", "S"} and sulfur_oxygen_count == 3:
        return "trifluoromethanesulfonate"
    return ""


def _bis_trifluoromethanesulfonyl_azanide_name(mol, Chem) -> str:
    counts = _element_counts(mol)
    if counts != {"C": 2, "F": 6, "N": 1, "O": 4, "S": 2} or _total_charge(mol) != -1:
        return ""
    nitrogens = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == "N" and atom.GetFormalCharge() < 0]
    if len(nitrogens) != 1:
        return ""
    nitrogen = nitrogens[0]
    sulfur_neighbors = [
        nbr
        for nbr in nitrogen.GetNeighbors()
        if nbr.GetSymbol() == "S"
        and mol.GetBondBetweenAtoms(nitrogen.GetIdx(), nbr.GetIdx()).GetBondType() == Chem.BondType.SINGLE
    ]
    if len(sulfur_neighbors) != 2:
        return ""
    if all(_is_trifluoromethanesulfonyl_group(mol, Chem, sulfur.GetIdx(), nitrogen.GetIdx()) for sulfur in sulfur_neighbors):
        return "bis(trifluoromethylsulfonyl)azanide"
    return ""


def _is_trifluoromethanesulfonyl_group(mol, Chem, sulfur_idx: int, nitrogen_idx: int) -> bool:
    sulfur = mol.GetAtomWithIdx(sulfur_idx)
    double_oxygen_count = 0
    carbon_neighbors = []
    for nbr in sulfur.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == nitrogen_idx:
            continue
        bond = mol.GetBondBetweenAtoms(sulfur_idx, nidx)
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_oxygen_count += 1
        elif nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE:
            carbon_neighbors.append(nidx)
        else:
            return False
    if double_oxygen_count != 2 or len(carbon_neighbors) != 1:
        return False
    carbon = mol.GetAtomWithIdx(carbon_neighbors[0])
    fluorines = [
        nbr
        for nbr in carbon.GetNeighbors()
        if nbr.GetIdx() != sulfur_idx
        and nbr.GetSymbol() == "F"
        and mol.GetBondBetweenAtoms(carbon.GetIdx(), nbr.GetIdx()).GetBondType() == Chem.BondType.SINGLE
    ]
    return len(fluorines) == 3 and carbon.GetDegree() == 4


def _has_bond(mol, Chem, left_symbol: str, right_symbol: str, bond_type) -> bool:
    symbols = {left_symbol, right_symbol}
    for bond in mol.GetBonds():
        if bond.GetBondType() != bond_type:
            continue
        if {bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()} == symbols:
            return True
    return False


def _element_counts(mol) -> dict[str, int]:
    counts: dict[str, int] = {}
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        counts[symbol] = counts.get(symbol, 0) + 1
    return counts


def _total_charge(mol) -> int:
    return sum(atom.GetFormalCharge() for atom in mol.GetAtoms())


def _salt_name(parent: str, fragments: list[str]) -> str:
    suffix = _multiplied_suffix(fragments)
    if not suffix:
        return ""
    return NameParts(parent=parent, suffixes=(Suffix(suffix, separator=" "),)).render()


def _multiplied_suffix(names: list[str]) -> str:
    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    parts = []
    for name in sorted(counts):
        count = counts[name]
        multiplied = _multiplied_fragment_suffix(name, count)
        if not multiplied:
            return ""
        parts.append(multiplied)
    return " ".join(parts)


def _multiplied_fragment_suffix(name: str, count: int) -> str:
    if count == 1:
        return name
    if _needs_complex_fragment_multiplier(name):
        multiplier = COMPLEX_MULTIPLIERS.get(count, "")
        return f"{multiplier}({name})" if multiplier else ""
    multiplier = MULTIPLIERS.get(count, "")
    return multiplier + name if multiplier else ""


def _needs_complex_fragment_multiplier(name: str) -> bool:
    if "(" in name or ")" in name:
        return True
    if " " in name:
        return True
    return name.startswith(("bis", "tris", "tetrakis", "tetra", "penta", "hexa")) or name.endswith(
        ("azanide", "phosphinate", "phosphonate")
    )


def _needs_fragment_parentheses(name: str) -> bool:
    return name[:1].isdigit() or "-" in name or "," in name or name.startswith("(")
