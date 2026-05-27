from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .fragment_rules import (
    FragmentAssemblyRule,
    common_polyatomic_fragment_name,
    simple_fragment_name,
    standalone_polyatomic_fragment_name,
)
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
        standalone_polyatomic_name = standalone_polyatomic_fragment_name(mol, self._Chem)
        if standalone_polyatomic_name:
            return standalone_polyatomic_name
        polyatomic_name = common_polyatomic_fragment_name(mol, self._Chem)
        if polyatomic_name:
            return polyatomic_name

        ctx = MoleculeContext(mol, self._Chem)
        best: NameCandidate | None = None
        best_key: tuple | None = None
        for rule in self._rules:
            if not rule.applies(ctx):
                continue
            rule_best: NameCandidate | None = None
            rule_best_key: tuple | None = None
            rule_candidates: list[NameCandidate] = []
            for candidate in rule.candidates(ctx):
                if not candidate.name:
                    continue
                rule_candidates.append(candidate)
                candidate_key = _candidate_base_key(candidate)
                if rule_best_key is None or candidate_key < rule_best_key:
                    rule_best = candidate
                    rule_best_key = candidate_key
            if rule_best is None:
                continue
            if _opsin_stability_penalty(rule_best.name) > 0:
                rule_best = min(rule_candidates, key=_candidate_key)
            adjusted_key = _candidate_key(rule_best)
            if best_key is None or adjusted_key < best_key:
                best = rule_best
                best_key = adjusted_key
            if (
                rule_best.priority == rule.priority
                and _opsin_stability_penalty(rule_best.name) == 0
            ):
                return _finalize_candidate_name(mol, self._Chem, best)
        if best is None:
            return ""
        return _finalize_candidate_name(mol, self._Chem, best)


def _candidate_base_key(candidate: NameCandidate) -> tuple:
    return (candidate.priority, candidate.sort_key, candidate.name)


def _candidate_key(candidate: NameCandidate) -> tuple:
    return (
        candidate.priority + _opsin_stability_penalty(candidate.name),
        candidate.sort_key,
        candidate.name,
    )


def _opsin_stability_penalty(name: str) -> int:
    """Prefer alternative candidates over names with unstable OPSIN rendering.

    These are general renderer-shape penalties, not molecule lookups. They
    catch cases where an earlier parent family emits a formally systematic but
    OPSIN-fragile substituent form while another candidate renders the same
    structure with a retained/simple parent that roundtrips more reliably.
    """

    penalty = 0
    unstable_fragments = (
        "cyclohexa-1,3,5-trien",
        "cyclohexa-2,4,6-trien",
        "cyclohepta-1,3,5-trien",
        "cyclobuta-1,3-dien",
        "dioxaheptane",
        "dioxaoctane",
        "dioxanonane",
        "dioxadecane",
        "dioxaundecane",
        "dithia",
        "dioxocyclobuta",
        "methan-1-thione",
        "phosphane",
        "phenyl)-methyl",
        "phenyl-methyl",
        "phenylidene",
        "silan",
        "thioamide",
    )
    for fragment in unstable_fragments:
        if fragment in name:
            penalty += 25
    retained_phosphate_shorthands = (
        "adenylic",
        "cytidylic",
        "guanylic",
        "thymidylic",
        "uridylic",
    )
    if any(fragment in name for fragment in retained_phosphate_shorthands):
        penalty += 25
    if " ethylene ketal" in name and "dioxaspiro" not in name:
        penalty += 25
    return penalty


def _finalize_candidate_name(mol, Chem, candidate: NameCandidate) -> str:
    name = _normalize_terminal_parent_stem(candidate.name)
    if candidate.stereo_prefix and not name.startswith("("):
        return _apply_uniform_isotope_label(mol, f"({candidate.stereo_prefix})-{name}")
    if candidate.family in {"whole-molecule-retained-parent", "natural-product-topology-parent"}:
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
    if symbol == "F" and isotope == 18 and count == 1:
        return _inline_single_fluorine_isotope(name)
    if symbol == "N" and count == 1 and not name.startswith("("):
        return f"[{isotope}{symbol}]{name}"
    return name


def _inline_single_fluorine_isotope(name: str) -> str:
    return name.replace("fluoro", "[18F]fluoro", 1) if "fluoro" in name else name


def _fallback_stereo_atom_order(family: str) -> str:
    idx_order_families = {
        "acyclic-parent",
        "fused-retained-parent",
        "functional-acyclic-skeleton-with-ring-substituents",
        "imino-functional-acyclic-skeleton-with-ring-substituents",
        "preferred-acyclic-skeleton-with-ring-substituents",
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
