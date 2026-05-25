from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from .naming_context import MoleculeContext
from .rule_dispatch import compile_rule_plan, rule_applies
from .stereochemistry import apply_stereochemical_prefix


@dataclass(frozen=True, slots=True)
class NameCandidate:
    """A deterministic name proposed by one rule family."""

    name: str
    family: str
    priority: int
    sort_key: tuple = ()


class MoleculeRule(Protocol):
    family: str
    priority: int

    def candidates(self, ctx: MoleculeContext) -> Iterable[NameCandidate]:
        ...


class NamingEngine:
    """OPSIN-inspired dispatcher over independent nomenclature rule families."""

    def __init__(self, Chem, rules: Iterable[MoleculeRule]):
        self._Chem = Chem
        self._rules = tuple(sorted(rules, key=lambda rule: rule.priority))
        self._plans = tuple(compile_rule_plan(rule) for rule in self._rules)

    def name(self, mol) -> str:
        fragments = self._Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if len(fragments) > 1:
            return _name_fragment_assembly(fragments, self)
        simple_name = _simple_fragment_name(mol, self._Chem)
        if simple_name:
            return simple_name

        ctx = MoleculeContext(mol, self._Chem)
        best: NameCandidate | None = None
        best_key: tuple | None = None
        for plan in self._plans:
            if not rule_applies(plan, ctx):
                continue
            rule = plan.rule
            for candidate in rule.candidates(ctx):
                if not candidate.name:
                    continue
                candidate_key = _candidate_key(candidate)
                if best_key is None or candidate_key < best_key:
                    best = candidate
                    best_key = candidate_key
            if best is not None and best.priority == rule.priority:
                return apply_stereochemical_prefix(mol, self._Chem, best.name)
        return apply_stereochemical_prefix(mol, self._Chem, best.name) if best is not None else ""


def _candidate_key(candidate: NameCandidate) -> tuple:
    return (candidate.priority, candidate.sort_key, candidate.name)


_NEUTRAL_HALIDE_SALTS = {
    "F": "hydrofluoride",
    "Cl": "hydrochloride",
    "Br": "hydrobromide",
    "I": "hydroiodide",
}

_HALIDE_ANIONS = {
    "F": "fluoride",
    "Cl": "chloride",
    "Br": "bromide",
    "I": "iodide",
}

_MULTIPLIERS = {
    2: "di",
    3: "tri",
    4: "tetra",
    5: "penta",
    6: "hexa",
}


def _name_fragment_assembly(fragments: tuple[object, ...], engine: NamingEngine) -> str:
    named_fragments: list[str] = []
    neutral_halides: list[str] = []
    halide_anions: list[str] = []

    for fragment in fragments:
        halide = _single_atom_halide(fragment)
        if halide is not None:
            symbol, charge = halide
            if charge < 0:
                halide_anions.append(_HALIDE_ANIONS[symbol])
                continue
            if charge == 0:
                neutral_halides.append(_NEUTRAL_HALIDE_SALTS[symbol])
                continue
        simple_name = _simple_fragment_name(fragment, engine._Chem)
        if simple_name:
            named_fragments.append(simple_name)
            continue
        name = engine.name(fragment)
        if not name:
            return ""
        named_fragments.append(name)

    if len(named_fragments) == 1 and neutral_halides and not halide_anions:
        suffix = _multiplied_suffix(neutral_halides)
        return f"{named_fragments[0]} {suffix}" if suffix else ""
    if len(named_fragments) == 1 and halide_anions and not neutral_halides:
        suffix = _multiplied_suffix(halide_anions)
        return f"{named_fragments[0]} {suffix}" if suffix else ""
    if neutral_halides or halide_anions:
        return ""
    return " and ".join(sorted(named_fragments, key=lambda item: (len(item), item)))


def _single_atom_halide(mol) -> tuple[str, int] | None:
    if mol.GetNumAtoms() != 1:
        return None
    atom = mol.GetAtomWithIdx(0)
    symbol = atom.GetSymbol()
    if symbol not in _HALIDE_ANIONS:
        return None
    return symbol, atom.GetFormalCharge()


def _simple_fragment_name(mol, Chem) -> str:
    if mol.GetNumAtoms() != 1:
        return ""
    atom = mol.GetAtomWithIdx(0)
    symbol = atom.GetSymbol()
    charge = atom.GetFormalCharge()
    if symbol in _HALIDE_ANIONS:
        if charge < 0:
            return _HALIDE_ANIONS[symbol]
        if charge == 0:
            return f"hydrogen {_HALIDE_ANIONS[symbol]}"
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
    return f"{element_name}({abs(charge)}{sign})"


def _multiplied_suffix(names: list[str]) -> str:
    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    parts = []
    for name in sorted(counts):
        count = counts[name]
        prefix = _MULTIPLIERS.get(count, "") if count > 1 else ""
        if count > 1 and not prefix:
            return ""
        parts.append(prefix + name)
    return " ".join(parts)
