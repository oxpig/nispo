from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .fragment_rules import FragmentAssemblyRule, common_polyatomic_fragment_name, simple_fragment_name
from .naming_context import MoleculeContext
from .rule_dispatch import RuleSpec
from .stereochemistry import apply_stereochemical_prefix


@dataclass(frozen=True, slots=True)
class NameCandidate:
    """A deterministic name proposed by one rule family."""

    name: str
    family: str
    priority: int
    sort_key: tuple = ()
    stereo_prefix: str = ""


class NamingEngine:
    """OPSIN-inspired dispatcher over independent nomenclature rule families."""

    def __init__(self, Chem, rules: Iterable[RuleSpec]):
        self._Chem = Chem
        self._rules = tuple(sorted(rules, key=lambda rule: rule.priority))
        self._fragment_rule = FragmentAssemblyRule(self._Chem, self)

    def name(self, mol) -> str:
        if mol.GetNumAtoms() == 1:
            simple_name = simple_fragment_name(mol, self._Chem)
            if simple_name:
                return simple_name
        fragment_atom_sets = self._Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
        if len(fragment_atom_sets) > 1:
            fragments = self._Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            return self._fragment_rule.name(fragments)
        simple_name = simple_fragment_name(mol, self._Chem)
        if simple_name:
            return simple_name
        polyatomic_name = common_polyatomic_fragment_name(mol, self._Chem)
        if polyatomic_name:
            return polyatomic_name

        ctx = MoleculeContext(mol, self._Chem)
        best: NameCandidate | None = None
        best_key: tuple | None = None
        for rule in self._rules:
            if not rule.applies(ctx):
                continue
            for candidate in rule.candidates(ctx):
                if not candidate.name:
                    continue
                candidate_key = _candidate_key(candidate)
                if best_key is None or candidate_key < best_key:
                    best = candidate
                    best_key = candidate_key
            if best is not None and best.priority == rule.priority:
                return _finalize_candidate_name(mol, self._Chem, best)
        if best is None:
            return ""
        return _finalize_candidate_name(mol, self._Chem, best)


def _candidate_key(candidate: NameCandidate) -> tuple:
    return (candidate.priority, candidate.sort_key, candidate.name)


def _finalize_candidate_name(mol, Chem, candidate: NameCandidate) -> str:
    name = _normalize_terminal_parent_stem(candidate.name)
    if candidate.stereo_prefix and not name.startswith("("):
        return _apply_uniform_isotope_label(mol, f"({candidate.stereo_prefix})-{name}")
    if candidate.family in {"whole-molecule-retained-parent"}:
        return _apply_uniform_isotope_label(mol, name)
    return _apply_uniform_isotope_label(
        mol,
        apply_stereochemical_prefix(
            mol,
            Chem,
            name,
            atom_order=_fallback_stereo_atom_order(candidate.family),
        ),
    )


def _apply_uniform_isotope_label(mol, name: str) -> str:
    if not name:
        return name
    labels: list[tuple[str, int, int]] = []
    for symbol in sorted({atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetSymbol() != "H"}):
        atoms = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == symbol]
        isotopes = {atom.GetIsotope() for atom in atoms}
        if len(isotopes) != 1:
            continue
        isotope = next(iter(isotopes))
        if isotope:
            labels.append((symbol, isotope, len(atoms)))
    if len(labels) != 1:
        return name
    symbol, isotope, count = labels[0]
    if symbol == "C" and isotope in {13, 14}:
        count_text = str(count) if count > 1 else ""
        return f"{name}-{isotope}{symbol}{count_text}"
    if symbol in {"F", "N"} and count == 1 and not name.startswith("("):
        return f"[{isotope}{symbol}]{name}"
    return name


def _fallback_stereo_atom_order(family: str) -> str:
    idx_order_families = {
        "acyclic-parent",
        "fused-retained-parent",
        "functional-acyclic-skeleton-with-ring-substituents",
        "imino-functional-acyclic-skeleton-with-ring-substituents",
        "single-ring-parent",
        "skeletal-replacement-fused-parent",
    }
    return "idx" if family in idx_order_families else "idx_desc"


def _normalize_terminal_parent_stem(name: str) -> str:
    terminal_stems = {
        "androst": "androstane",
        "chol": "cholane",
        "cholest": "cholestane",
        "campest": "campestane",
        "ergost": "ergostane",
        "poriferast": "poriferastane",
        "stigmast": "stigmastane",
        "gon": "gonane",
        "pregn": "pregnane",
    }
    for stem, parent in terminal_stems.items():
        if name == stem:
            return parent
        suffix = "-" + stem
        if name.endswith(suffix):
            return name[: -len(stem)] + parent
    return name
