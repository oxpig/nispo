from __future__ import annotations

from .name_parts import NameParts, Suffix


class FragmentAssemblyRule:
    """First-class naming rule for disconnected fragments, salts, and adducts."""

    def __init__(self, Chem, engine):
        self.Chem = Chem
        self.engine = engine

    def name(self, fragments: tuple[object, ...]) -> str:
        named_fragments: list[str] = []
        neutral_halides: list[str] = []
        halide_anions: list[str] = []

        for fragment in fragments:
            halide = single_atom_halide(fragment)
            if halide is not None:
                symbol, charge = halide
                if charge < 0:
                    halide_anions.append(HALIDE_ANIONS[symbol])
                    continue
                if charge == 0:
                    neutral_halides.append(NEUTRAL_HALIDE_SALTS[symbol])
                    continue
            simple_name = simple_fragment_name(fragment, self.Chem)
            if simple_name:
                named_fragments.append(simple_name)
                continue
            name = self.engine.name(fragment)
            if not name:
                return ""
            named_fragments.append(name)

        if len(named_fragments) == 1 and neutral_halides and not halide_anions:
            return _salt_name(named_fragments[0], neutral_halides)
        if len(named_fragments) == 1 and halide_anions and not neutral_halides:
            return _salt_name(named_fragments[0], halide_anions)
        if neutral_halides or halide_anions:
            return ""
        return " and ".join(sorted(named_fragments, key=lambda item: (len(item), item)))


NEUTRAL_HALIDE_SALTS = {
    "F": "hydrofluoride",
    "Cl": "hydrochloride",
    "Br": "hydrobromide",
    "I": "hydroiodide",
}

HALIDE_ANIONS = {
    "F": "fluoride",
    "Cl": "chloride",
    "Br": "bromide",
    "I": "iodide",
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


def simple_fragment_name(mol, Chem) -> str:
    if mol.GetNumAtoms() != 1:
        return ""
    atom = mol.GetAtomWithIdx(0)
    symbol = atom.GetSymbol()
    charge = atom.GetFormalCharge()
    if symbol in HALIDE_ANIONS:
        if charge < 0:
            return HALIDE_ANIONS[symbol]
        if charge == 0:
            return f"hydrogen {HALIDE_ANIONS[symbol]}"
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
