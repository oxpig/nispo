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
    if symbol == "O" and charge < 0 and atom.GetTotalNumHs() > 0:
        return "hydroxide"
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
        _perfluoro_counterion_name(mol, Chem)
        or _oxoanion_name(mol, Chem)
        or _pseudohalide_name(mol, Chem)
        or _trifluoromethanesulfonate_name(mol, Chem)
    )


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
        prefix = MULTIPLIERS.get(count, "") if count > 1 else ""
        if count > 1 and not prefix:
            return ""
        parts.append(prefix + name)
    return " ".join(parts)
