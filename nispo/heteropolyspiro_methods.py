"""Bounded mixed-carbon heteropolyspiro parents by P-24.2.4.1."""
from __future__ import annotations

from dataclasses import dataclass

from .molecule_iteration import iter_atoms, iter_bonds
from .constants import MULT, ROOTS, STANDARD_REPLACEMENT_VALENCE
from .heteropolycyclic_parents import replacement_locants, replacement_prefix
from .naming_methods import MethodDecision
from .polyspiro_frontier import PolyspiroNumbering, polyspiro_parent


RULE = ("P-24.2.2/P-24.2.3/P-24.2.4.1.1/P-24.2.4.1.2(a)-(b)/"
        "P-15.4.1.2/P-15.4.2.1/P-51.4.2.3")
MAX_SKELETAL_ATOMS = 60


@dataclass(frozen=True, slots=True)
class HeteropolyspiroRecognition:
    status: str
    reason: str
    carbon_topology: object | None = None
    original_atomic_numbers: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class HeteropolyspiroProposal:
    name: str
    sequence: tuple[int, ...]
    atom_locants: tuple[tuple[int, str], ...]
    atomic_numbers: tuple[int, ...]
    descriptor: tuple
    spiro_atoms: tuple[int, ...]
    spiro_locants: tuple[int, ...]
    hetero_locants: tuple[int, ...]
    seniority_locants: tuple[tuple[int, ...], ...]
    criteria_key: tuple
    rule: str = RULE


@dataclass(frozen=True, slots=True)
class HeteropolyspiroResult:
    status: str
    reason: str
    proposals: tuple[HeteropolyspiroProposal, ...] = ()


@dataclass(frozen=True, slots=True)
class HeteropolyspiroMethodDecision(MethodDecision):
    """Shared method decision preserving all source-equivalent atom maps."""

    equivalent_atom_locants: tuple[tuple[tuple[int, str], ...], ...] = ()


def _unsupported(reason: str) -> HeteropolyspiroRecognition:
    return HeteropolyspiroRecognition("unsupported", reason)


def recognize_heteropolyspiro(mol, Chem=None) -> HeteropolyspiroRecognition:
    """Recognize a neutral, saturated mixed-carbon graph for spiro numbering.

    Carbon capping is private and strictly isomorphic: only atomic numbers are
    replaced in a graph with identical atom/bond indices, for hydrocarbon
    topology selection. All final hetero locants come from the original graph.
    """
    if Chem is None:
        try:
            from rdkit import Chem as rdkit_chem
        except Exception:
            return _unsupported("RDKit chemistry support is unavailable")
        Chem = rdkit_chem
    if (mol is None or mol.GetNumAtoms() > MAX_SKELETAL_ATOMS
            or mol.GetNumAtoms() != mol.GetNumHeavyAtoms()
            or mol.GetStereoGroups()):
        return _unsupported("requires a bounded complete heavy-atom graph without explicit hydrogen atoms")
    atoms = tuple(iter_atoms(mol))
    if any(atom.GetAtomicNum() not in STANDARD_REPLACEMENT_VALENCE
           or atom.GetFormalCharge() or atom.GetNumRadicalElectrons()
           or atom.GetTotalValence() != STANDARD_REPLACEMENT_VALENCE[atom.GetAtomicNum()]
           or atom.GetIsotope() or atom.GetIsAromatic()
           or int(atom.GetChiralTag())
           for atom in atoms):
        return _unsupported("requires neutral standard-valence unannotated skeletal atoms")
    if any(bond.GetBondTypeAsDouble() != 1 or bond.GetStereo()
           or int(bond.GetBondDir()) != 0 for bond in iter_bonds(mol)):
        return _unsupported("requires saturated single bonds without stereo or direction annotations")
    numbers = tuple(atom.GetAtomicNum() for atom in atoms)
    if 6 not in numbers or not any(number != 6 for number in numbers):
        return _unsupported("requires both skeletal carbon and at least one skeletal replacement atom")

    rw = Chem.RWMol()
    for _atom in atoms:
        rw.AddAtom(Chem.Atom(6))
    for bond in iter_bonds(mol):
        rw.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), Chem.BondType.SINGLE)
    topology = rw.GetMol()
    try:
        Chem.SanitizeMol(topology)
    except Exception:
        return _unsupported("isomorphic carbon topology could not be sanitized")
    return HeteropolyspiroRecognition("recognized", "complete bounded mixed-carbon heteropolyspiro graph",
                                      topology, numbers)


def _descriptor_text(numbering: PolyspiroNumbering) -> str:
    terms = []
    for term in numbering.descriptor:
        text = str(term.linking_atoms)
        if term.repeated_spiro_locant is not None:
            text += "^" + str(term.repeated_spiro_locant)
        terms.append(text)
    return ".".join(terms)


def heteropolyspiro_parent_proposals(mol, Chem=None) -> HeteropolyspiroResult:
    """Construct all P-24-tied mixed heteropolyspiro parent components."""
    recognition = recognize_heteropolyspiro(mol, Chem)
    if recognition.status != "recognized":
        return HeteropolyspiroResult("unsupported", recognition.reason)
    if Chem is None:
        from rdkit import Chem as rdkit_chem
        Chem = rdkit_chem
    numbers = recognition.original_atomic_numbers
    frontier = polyspiro_parent(recognition.carbon_topology, Chem)
    if frontier.status != "complete":
        return HeteropolyspiroResult("unsupported",
                                     "carbon-capped topology: " + frontier.reason)
    options = []
    for candidate in frontier.candidates:
        key = replacement_locants(candidate.sequence, numbers)
        prefix = replacement_prefix(numbers, key[1])
        if prefix is None:
            continue
        n_spiro = len(candidate.spiro_atoms)
        multiplier = MULT.get(n_spiro)
        root = ROOTS.get(len(candidate.sequence))
        if not multiplier or not root:
            continue
        name = prefix + multiplier + "spiro[" + _descriptor_text(candidate) + "]" + root + "ane"
        atom_locants = tuple(candidate.atom_locants)
        proposal = HeteropolyspiroProposal(
            name=name,
            sequence=candidate.sequence,
            atom_locants=atom_locants,
            atomic_numbers=tuple(numbers[atom_id] for atom_id in candidate.sequence),
            descriptor=candidate.descriptor,
            spiro_atoms=candidate.spiro_atoms,
            spiro_locants=candidate.spiro_locants,
            hetero_locants=key[0],
            seniority_locants=key[1],
            criteria_key=key,
        )
        options.append((key, proposal))
    if not options:
        return HeteropolyspiroResult("unsupported", "replacement prefix multiplicity or parent root is unavailable")
    best = min(key for key, _proposal in options)
    proposals = []
    seen = set()
    for key, proposal in options:
        if key == best and proposal.sequence not in seen:
            seen.add(proposal.sequence)
            proposals.append(proposal)
    names = {proposal.name for proposal in proposals}
    if len(names) != 1:
        return HeteropolyspiroResult("unsupported", "surviving source-tied replacement names do not agree")
    return HeteropolyspiroResult("complete", "P-24 hydrocarbon frontier then P-24 heteroatom criteria",
                                 tuple(proposals))


def heteropolyspiro_decision(mol, Chem=None) -> HeteropolyspiroMethodDecision | None:
    """Return a whole-parent method decision after full-map text consensus."""
    result = heteropolyspiro_parent_proposals(mol, Chem)
    if result.status != "complete" or not result.proposals:
        return None
    names = {proposal.name for proposal in result.proposals}
    if len(names) != 1:
        return None
    heavy = frozenset(atom.GetIdx() for atom in iter_atoms(mol) if atom.GetAtomicNum() > 1)
    if not heavy or heavy != frozenset(range(mol.GetNumAtoms())):
        return None
    maps = []
    for proposal in result.proposals:
        row = tuple(proposal.atom_locants)
        if (len(row) != len(heavy)
                or frozenset(atom for atom, _locant in row) != heavy
                or len({locant for _atom, locant in row}) != len(heavy)):
            return None
        maps.append(row)
    base = result.proposals[0]
    base_map = tuple(base.atom_locants)
    return HeteropolyspiroMethodDecision(
        name=next(iter(names)), method="skeletal-replacement",
        principal_centers=frozenset(), atoms=heavy, rule=RULE,
        atom_locants=base_map,
        component_roles=(("heteropolyspiro-parent", heavy, base_map),),
        equivalent_atom_locants=tuple(dict.fromkeys(maps)))
