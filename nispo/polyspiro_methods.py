"""Whole-parent proposal API for bounded linear polyspiro hydrocarbon cores."""
from __future__ import annotations

from dataclasses import dataclass

from .constants import MULT, ROOTS
from .polyspiro_frontier import PolyspiroNumbering, polyspiro_parent
from .naming_methods import MethodDecision


@dataclass(frozen=True, slots=True)
class PolyspiroProposal:
    name: str
    sequence: tuple[int, ...]
    atom_locants: tuple[tuple[int, str], ...]
    descriptor: tuple
    spiro_atoms: tuple[int, ...]
    spiro_locants: tuple[int, ...]
    rule: str


@dataclass(frozen=True, slots=True)
class PolyspiroMethodResult:
    status: str
    reason: str
    proposals: tuple[PolyspiroProposal, ...] = ()


@dataclass(frozen=True, slots=True)
class PolyspiroMethodDecision(MethodDecision):
    """Complete method decision retaining all equivalent original-atom maps."""

    equivalent_atom_locants: tuple[tuple[tuple[int, str], ...], ...] = ()


def _parent_component(numbering: PolyspiroNumbering, atom_count: int):
    spiro_count = len(numbering.spiro_atoms)
    multiplier = MULT.get(spiro_count)
    root = ROOTS.get(atom_count)
    if not multiplier or not root:
        return None
    descriptor_terms = [str(term.linking_atoms) +
                        (f"^{term.repeated_spiro_locant}"
                         if term.repeated_spiro_locant is not None else "")
                        for term in numbering.descriptor]
    return f"{multiplier}spiro[{'.'.join(descriptor_terms)}]{root}ane"


def polyspiro_parent_proposals(mol, Chem=None) -> PolyspiroMethodResult:
    """Compose whole spiro-parent components without selecting among tied maps."""
    frontier = polyspiro_parent(mol, Chem)
    if frontier.status != "complete":
        return PolyspiroMethodResult(frontier.status, frontier.reason, ())
    proposals = []
    for numbering in frontier.candidates:
        name = _parent_component(numbering, len(numbering.sequence))
        if not name:
            return PolyspiroMethodResult("unsupported",
                "the spiro multiplicative prefix or hydrocarbon root is unavailable", ())
        proposals.append(PolyspiroProposal(
            name=name, sequence=numbering.sequence,
            atom_locants=numbering.atom_locants,
            descriptor=numbering.descriptor,
            spiro_atoms=numbering.spiro_atoms,
            spiro_locants=numbering.spiro_locants,
            rule="P-24.2.2.1/P-24.2.2.2"))
    return PolyspiroMethodResult("complete", frontier.reason, tuple(proposals))


def polyspiro_decision(mol, Chem=None) -> PolyspiroMethodDecision | None:
    """Adapt a text-consensus polyspiro frontier to the shared decision type.

    Equivalent numbering maps are retained as metadata. One map is used for
    the ordinary decision fields only after all surviving rendered parent
    components have been shown to agree.
    """
    if Chem is None:
        try:
            from rdkit import Chem as rdkit_chem
        except Exception:
            return None
        Chem = rdkit_chem
    result = polyspiro_parent_proposals(mol, Chem)
    if result.status != "complete" or not result.proposals:
        return None
    names = {proposal.name for proposal in result.proposals}
    if len(names) != 1:
        return None
    heavy = frozenset(atom.GetIdx() for atom in mol.GetAtoms()
                      if atom.GetAtomicNum() > 1)
    if not heavy or heavy != frozenset(range(mol.GetNumAtoms())):
        return None
    mapped = []
    for proposal in result.proposals:
        atom_locants = tuple(proposal.atom_locants)
        if (len(atom_locants) != len(heavy)
                or frozenset(atom for atom, _locant in atom_locants) != heavy
                or len({locant for _atom, locant in atom_locants}) != len(heavy)):
            return None
        mapped.append(atom_locants)
    # The proposals are already source-tied. This representative is not used
    # to rank candidates or alter the agreed component text.
    base = result.proposals[0]
    base_map = tuple(base.atom_locants)
    return PolyspiroMethodDecision(
        name=next(iter(names)), method="substitutive",
        principal_centers=frozenset(), atoms=heavy,
        rule="P-24.2.0/P-24.2.2/P-24.2.2.1/P-24.2.2.2",
        atom_locants=base_map,
        component_roles=(("polyspiro-parent", heavy, base_map),),
        equivalent_atom_locants=tuple(dict.fromkeys(mapped)))
