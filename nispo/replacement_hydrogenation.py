"""Neutral fused skeletal replacement with an explicit hydrogenation state.

Reference hydrides are constructed from the numbered parent vocabulary. Only
their hydrogen states are enumerated: the input is never tautomerized. This
bounded adapter covers C/N/O/S cores and at most six aromatic N-H choices.
"""
from functools import lru_cache

from .constants import MULT
from .template_access import template_record_locants, template_record_mol


@lru_cache(maxsize=256)
def _reference_hydrides(query, numbers):
    from itertools import product
    from rdkit import Chem
    from .fused_hydrogen_state import _reference_states

    graph = Chem.RWMol(query)
    choices = []
    for idx, number in enumerate(numbers):
        atom = graph.GetAtomWithIdx(idx)
        atom.SetAtomicNum(number)
        if number != 6:
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(False)
        if number == 7 and atom.GetIsAromatic() and atom.GetDegree() == 2:
            choices.append(idx)
    if len(choices) > 6:
        return ()
    references = {}
    for hydrogens in product((0, 1), repeat=len(choices)):
        candidate = Chem.Mol(graph)
        for idx, count in zip(choices, hydrogens):
            atom = candidate.GetAtomWithIdx(idx)
            atom.SetNumExplicitHs(count)
            atom.SetNoImplicit(True)
        try:
            Chem.SanitizeMol(candidate)
        except (ValueError, RuntimeError):
            continue
        # A valid numbered reference remains usable when an integral spiro
        # atom has degree four. The annular matching enumerator below is
        # intentionally restricted to degree-two/three fused atoms.
        population = tuple(a.GetTotalNumHs() for a in candidate.GetAtoms())
        references.setdefault(population, candidate)
        for reference in _reference_states(candidate):
            population = tuple(a.GetTotalNumHs() for a in reference.GetAtoms())
            references.setdefault(population, reference)
    if not references and len(numbers) <= 18:
        # A localized hydrocarbon Kekule drawing can place a double bond at
        # the position replaced by divalent O/S or pyrrolic N. Enumerate the
        # neutral replacement graph itself before deciding it has no hydride.
        # This bounded graph operation also admits integral degree-four
        # spiro carbons; it never changes the input's bond or H state.
        from .ring_state_parent import _reference_states as graph_reference_states
        edges = tuple((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in query.GetBonds())
        mancude = set().union(*(set(ring) for ring in query.GetRingInfo().AtomRings()
            if any(query.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble() > 1
                   for a, b in zip(ring, ring[1:] + ring[:1]))))
        saturated = tuple(i for i in range(len(numbers)) if i not in mancude)
        for population, _indicated, bonds in graph_reference_states(
                numbers, edges, allow_tetravalent_carbon=True, saturated_atoms=saturated):
            rw = Chem.RWMol()
            for number, count in zip(numbers, population):
                atom = Chem.Atom(number)
                atom.SetNumExplicitHs(count)
                atom.SetNoImplicit(True)
                rw.AddAtom(atom)
            for a, b, aromatic, order in bonds:
                rw.AddBond(a, b, {1.0: Chem.BondType.SINGLE,
                                  1.5: Chem.BondType.AROMATIC,
                                  2.0: Chem.BondType.DOUBLE}[order])
            reference = rw.GetMol()
            try:
                Chem.SanitizeMol(reference)
            except (ValueError, RuntimeError):
                continue
            references.setdefault(population, reference)
    if not references:
        return ()
    # Fixed topology and neutral valences make minimum H equivalent to the
    # maximum number of noncumulative annular double bonds (P-25.7).
    minimum = min(map(sum, references))
    return tuple(reference for population, reference in references.items()
                 if sum(population) == minimum)


def hydrogenated_replacement_parent(mol, Chem, template, match, parent_name):
    """Compose proved hydro operations on a numbered replacement parent.

    An unsupported reference or ambiguous capped pi graph declines this
    construction. It cannot justify the unhydrogenated name. No charge,
    isotope or radical operation is added by this adapter.
    """
    query = template_record_mol(template)
    locants = template_record_locants(template)
    from .locant_order import writing_locant_key as locant_key
    if (not all(a.GetAtomicNum() == 6 and not a.GetFormalCharge()
                for a in query.GetAtoms())
            or any(idx not in locants or locant_key(locants[idx]) is None
                   for idx in range(len(match)))):
        return ""
    actual = tuple(mol.GetAtomWithIdx(idx) for idx in match)
    if any(a.GetAtomicNum() not in {6, 7, 8, 16} or a.GetFormalCharge()
           or a.GetIsotope() or a.GetNumRadicalElectrons() for a in actual):
        return ""
    core = frozenset(match)
    numbers = tuple(a.GetAtomicNum() for a in actual)
    capped = tuple(a.GetTotalNumHs() + sum(
        b.GetBondTypeAsDouble() for b in a.GetBonds()
        if b.GetOtherAtomIdx(a.GetIdx()) not in core) for a in actual)
    if any(value != int(value) for value in capped):
        return ""
    capped = tuple(map(int, capped))
    references = _reference_hydrides(query, numbers)
    if not references:
        return ""
    from .hydrogen_accommodation import unique_capped_pi_graph
    edges = tuple((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in query.GetBonds())
    if not unique_capped_pi_graph(numbers, edges, capped):
        return ""
    from .hydrogen_accommodation import annular_pi_bonds, joint_hydrogen_states
    actual_pi = annular_pi_bonds(mol, Chem, match)
    mancude = set().union(*(set(ring) for ring in query.GetRingInfo().AtomRings()
        if any(query.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble() > 1
               for a, b in zip(ring, ring[1:] + ring[:1]))))
    candidates = []
    for reference in references:
        differences = tuple(capped[idx] - ref.GetTotalNumHs()
                            for idx, ref in enumerate(reference.GetAtoms()))
        if any(value < 0 for value in differences):
            continue
        if any(differences) and not joint_hydrogen_states(
                numbers, edges, tuple(a.GetTotalNumHs() for a in reference.GetAtoms()),
                capped, (0,) * len(match), actual_pi):
            continue
        hydro = []
        indicated = []
        for idx, ref in enumerate(reference.GetAtoms()):
            hydro.extend([locants[idx]] * (capped[idx] - ref.GetTotalNumHs()))
            if idx in mancude and ((ref.GetAtomicNum() == 7 and ref.GetTotalNumHs())
                    or ref.GetAtomicNum() == 6 and ref.GetTotalNumHs() == 2
                    and all(b.GetBondTypeAsDouble() == 1 for b in ref.GetBonds())):
                indicated.append(locants[idx])
        hydro.sort(key=locant_key)
        indicated.sort(key=locant_key)
        if hydro and len(hydro) not in MULT:
            continue
        text = (",".join(hydro) + "-" + MULT[len(hydro)] + "hydro-") if hydro else ""
        if indicated:
            text += ",".join(loc + "H" for loc in indicated) + "-"
        text += parent_name
        candidates.append(((len(hydro), tuple(map(locant_key, indicated)),
                            tuple(map(locant_key, hydro))), text))
    return min(candidates)[1] if candidates else ""
