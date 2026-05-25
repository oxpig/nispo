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
        neutral_halides: list[str] = []
        halide_anions: list[str] = []

        for fragment in fragments:
            halide_anion = single_atom_halide_anion_name(fragment)
            if halide_anion is not None:
                halide_anions.append(halide_anion)
                continue
            neutral_halide = neutral_halide_name(fragment)
            if neutral_halide is not None:
                neutral_halides.append(neutral_halide)
                continue
            cation = single_atom_cation_name(fragment, self.Chem)
            if cation is not None:
                cations.append(cation)
                continue
            simple_name = simple_fragment_name(fragment, self.Chem)
            if simple_name:
                named_fragments.append(simple_name)
                continue
            name = self.engine.name(fragment)
            if not name:
                return ""
            named_fragments.append(name)

        if len(named_fragments) == 1 and neutral_halides and not halide_anions and not cations:
            return _salt_name(named_fragments[0], neutral_halides)
        if len(named_fragments) == 1 and halide_anions and not neutral_halides and not cations:
            return _salt_name(named_fragments[0], halide_anions)
        if len(cations) == 1 and halide_anions and not named_fragments and not neutral_halides:
            return _salt_name(cations[0], halide_anions)
        if neutral_halides or halide_anions:
            return ""
        return " and ".join(sorted(cations + named_fragments, key=lambda item: (len(item), item)))


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

MULTIPLIERS = {
    2: "di",
    3: "tri",
    4: "tetra",
    5: "penta",
    6: "hexa",
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
    anion = single_atom_anion_name(mol, Chem)
    if anion:
        return anion
    neutral_halide = neutral_halide_name(mol)
    if neutral_halide:
        return f"hydrogen {HALIDE_ANIONS[symbol]}"
    cation = single_atom_cation_name(mol, Chem)
    if cation:
        return cation
    if symbol == "O" and charge < 0 and atom.GetTotalNumHs() > 0:
        return "hydroxide"
    if symbol in {"C", "N", "O", "S", "B", "P", "Si"} and charge == 0:
        return ""

    element_name = Chem.GetPeriodicTable().GetElementName(atom.GetAtomicNum()).lower()
    if not element_name:
        return ""
    if charge == 0:
        return element_name
    sign = "+" if charge > 0 else "-"
    return NameParts(parent=element_name, charge=f"({abs(charge)}{sign})").render()


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

