"""Mapped skeletal-replacement parents for bounded linear dispiro cores.

The hydrocarbon P-24 frontier is established first. Heteroatom replacement
then selects among only that source-tied frontier, retaining original atom IDs.
"""

from .constants import STANDARD_REPLACEMENT_VALENCE
from .heteropolycyclic_parents import replacement_locants, replacement_prefix
from .dispiro_frontier import dispiro_frontier

RULES = {
    "skeletal-replacement-dispiro": (
        "P-24.2.2", "P-24.2.4.1.1", "P-24.2.4.1.2(a)",
        "P-24.2.4.1.2(b)", "P-51.4.2.3",
    )
}


def _carbon_capped_graph(graph, Chem):
    """Return a clean isomorphic all-carbon graph with the same atom indices."""
    if graph.GetNumAtoms() != graph.GetNumHeavyAtoms() or graph.GetNumAtoms() < 7:
        return None, "requires a complete heavy-atom core without explicit H atoms"
    if any(atom.GetAtomicNum() not in STANDARD_REPLACEMENT_VALENCE
           or atom.GetFormalCharge() or atom.GetNumRadicalElectrons()
           or atom.GetTotalValence() != STANDARD_REPLACEMENT_VALENCE[atom.GetAtomicNum()]
           for atom in graph.GetAtoms()):
        return None, "requires standard-valence neutral skeletal atoms"
    if any(bond.GetBondTypeAsDouble() != 1 for bond in graph.GetBonds()):
        return None, "only saturated single-bonded cores are supported"
    numbers = tuple(atom.GetAtomicNum() for atom in graph.GetAtoms())
    if 6 not in numbers or not any(number != 6 for number in numbers):
        return None, "requires both carbon and at least one skeletal heteroatom"

    rw = Chem.RWMol()
    for _ in graph.GetAtoms():
        rw.AddAtom(Chem.Atom(6))
    for bond in graph.GetBonds():
        rw.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), Chem.BondType.SINGLE)
    capped = rw.GetMol()
    try:
        Chem.SanitizeMol(capped)
    except Exception:
        return None, "the isomorphic all-carbon capped graph is not a valid saturated skeleton"
    return (capped, numbers), None


def _descriptor_text(candidate):
    terms = []
    for term in candidate.descriptor:
        count = str(term.linking_atoms)
        if term.repeated_spiro_locant is not None:
            count += "^" + str(term.repeated_spiro_locant)
        terms.append(count)
    return ".".join(terms)


def skeletons(graph, Chem=None):
    """Yield P-24.2.4.1 replacement parents using the mapped tuple contract.

    Each yielded row has the same shape as ``heteropolycyclic_parents.skeletons``:
    parent component, original atom-index sequence, locants, numbered atomic
    numbers, numbered edge pairs, completeness flag, comparison key, method.
    Atom isotopes and stereo annotations remain on the caller's original graph;
    this producer only supplies skeleton maps for downstream decoration.
    """
    if Chem is None:
        from rdkit import Chem
    capped, _reason = _carbon_capped_graph(graph, Chem)
    if capped is None:
        return
    carbon_graph, numbers = capped
    frontier = dispiro_frontier(carbon_graph, Chem)
    if frontier.status != "complete":
        return

    options = []
    for candidate in frontier.candidates:
        key = replacement_locants(candidate.sequence, numbers)
        prefix = replacement_prefix(numbers, key[1])
        if prefix is None:
            continue
        parent = (prefix + "dispiro[" + _descriptor_text(candidate) + "]"
                  + candidate.hydrocarbon_parent)
        locants = tuple(str(number) for number in range(1, len(candidate.sequence) + 1))
        inverse = {atom_idx: number for number, atom_idx in enumerate(candidate.sequence)}
        edges = tuple(sorted(
            tuple(sorted((inverse[bond.GetBeginAtomIdx()], inverse[bond.GetEndAtomIdx()])))
            for bond in graph.GetBonds()
        ))
        options.append((key, parent, candidate.sequence, locants,
                        tuple(numbers[idx] for idx in candidate.sequence), edges))
    if not options:
        return
    best = min(option[0] for option in options)
    seen = set()
    for key, parent, sequence, locants, numbered_numbers, edges in options:
        if key != best or sequence in seen:
            continue
        seen.add(sequence)
        yield (parent, sequence, locants, numbered_numbers, edges, True, key,
               "skeletal-replacement-dispiro")

