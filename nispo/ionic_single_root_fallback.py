"""Mapped, preference-unresolved single-oxidanide names for bounded ions.

P-74.1.3 admits an anionic parent with a cationic substituent. P-72.6.3
permits the second anionic center as a systematic prefix. This general-name
producer does not claim that multiplication, prefix selection, or stereo is
preferred. The source graph is never modified.
"""
from .formatting import prefix_alphabetization_key
from .naming_methods import MethodDecision
from .naming_result import SelectionLimitation


def _cation_oxy(mol, Chem, oxygen_id, phosphorus_id, cation_id):
    from .polycyclic_rules import _bicyclic_parent_sequence_candidates, _ring_system_graph

    oxygen = mol.GetAtomWithIdx(oxygen_id)
    if (oxygen.GetAtomicNum() != 8 or oxygen.GetFormalCharge() or oxygen.GetDegree() != 2
            or any(b.GetBondTypeAsDouble() != 1 for b in oxygen.GetBonds())):
        return None
    neighbors = [a for a in oxygen.GetNeighbors() if a.GetIdx() != phosphorus_id]
    if (len(neighbors) != 1 or neighbors[0].GetAtomicNum() != 6
            or not neighbors[0].IsInRing()):
        return None
    attachment = neighbors[0].GetIdx()
    ring = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.IsInRing())
    if not 5 <= len(ring) <= 16 or cation_id not in ring or attachment not in ring:
        return None
    cation = mol.GetAtomWithIdx(cation_id)
    if (cation.GetAtomicNum() != 7 or cation.GetFormalCharge() != 1
            or cation.GetDegree() not in (2, 3)
            or cation.GetTotalNumHs() != 4 - cation.GetDegree()
            or any(a.GetIdx() not in ring for a in cation.GetNeighbors())):
        return None
    for i in ring:
        atom = mol.GetAtomWithIdx(i)
        if i != cation_id and (atom.GetAtomicNum() != 6 or atom.GetFormalCharge()):
            return None
        for neighbor in atom.GetNeighbors():
            j = neighbor.GetIdx()
            if (mol.GetBondBetweenAtoms(i, j).GetBondTypeAsDouble() != 1
                    or j not in ring and not (i == attachment and j == oxygen_id)):
                return None
    edge_count = sum(1 for b in mol.GetBonds()
                     if b.GetBeginAtomIdx() in ring and b.GetEndAtomIdx() in ring)
    if edge_count - len(ring) + 1 != 2:
        return None
    # Neutralization is confined to this temporary core and is used solely
    # for numbering. All ownership and charge evidence stays on the input.
    original = tuple(sorted(ring))
    rw = Chem.RWMol(mol)
    for i in sorted(set(range(mol.GetNumAtoms())) - ring, reverse=True):
        rw.RemoveAtom(i)
    hydride = rw.GetMol()
    n_site = original.index(cation_id)
    nitrogen = hydride.GetAtomWithIdx(n_site)
    nitrogen.SetFormalCharge(0)
    nitrogen.SetNumExplicitHs(3 - cation.GetDegree())
    nitrogen.SetNoImplicit(True)
    try:
        Chem.SanitizeMol(hydride)
    except (ValueError, RuntimeError):
        return None
    graph = _ring_system_graph(hydride, Chem, set(range(len(ring))))
    if not graph:
        return None
    rows = []
    for parent, seq, _ in _bicyclic_parent_sequence_candidates(hydride, set(graph), graph):
        if not parent.endswith('e'):
            continue
        n_locant = seq.index(n_site) + 1
        at_locant = seq.index(original.index(attachment)) + 1
        prefix = f'({parent[:-1]}-{n_locant}-ium-{at_locant}-yl)oxy'
        rows.append(((n_locant, at_locant), prefix, tuple(original[i] for i in seq)))
    if not rows:
        return None
    best = min(row[0] for row in rows)
    winners = [row for row in rows if row[0] == best]
    if len({row[1] for row in winners}) != 1:
        return None
    ranks = tuple(Chem.CanonicalRankAtoms(mol, breakTies=True))
    chosen = min(winners, key=lambda row: tuple(ranks[i] for i in row[2]))
    mapping = tuple((i, str(loc)) for loc, i in enumerate(chosen[2], 1))
    return chosen[1], ring | {oxygen_id}, mapping


def _amino(mol, nitrogen_id, phosphorus_id):
    from .prefix_sequences import PrefixUnit, mononuclear_prefix_sequence
    from .simple_prefixes import simple_external_prefix_candidates

    n = mol.GetAtomWithIdx(nitrogen_id)
    neighbors = [a for a in n.GetNeighbors() if a.GetIdx() != phosphorus_id]
    if (n.GetAtomicNum() != 7 or n.GetFormalCharge() or n.IsInRing() or len(neighbors) > 2
            or n.GetTotalNumHs() != 2 - len(neighbors)
            or any(b.GetBondTypeAsDouble() != 1 for b in n.GetBonds())):
        return None
    owned, units = {nitrogen_id}, []
    for atom in neighbors:
        candidates = simple_external_prefix_candidates(mol, atom.GetIdx(), nitrogen_id)
        simple = [c for c in candidates if c.kind in {'linear-alkyl', 'branched-alkyl'}]
        if len(simple) != 1 or owned & set(simple[0].atoms):
            return None
        owned.update(simple[0].atoms)
        units.append(PrefixUnit(simple[0].text, 'simple'))
    if not units:
        return 'amino', frozenset(owned)
    sequence = mononuclear_prefix_sequence(units)
    return (sequence.text + 'amino', frozenset(owned)) if sequence else None


def single_root_phosphorus_zwitterion(mol, Chem):
    """Return a fully owned general name with three explicit limitations."""
    if (mol.GetStereoGroups() or len(Chem.GetMolFrags(mol)) != 1
            or any(a.GetIsotope() or a.GetNumRadicalElectrons() or int(a.GetChiralTag())
                   or a.GetAtomicNum() not in (6, 7, 8, 15) for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) for b in mol.GetBonds())):
        return None
    charged = [a for a in mol.GetAtoms() if a.GetFormalCharge()]
    neg = [a for a in charged if a.GetFormalCharge() == -1 and a.GetAtomicNum() == 8]
    pos = [a for a in charged if a.GetFormalCharge() == 1 and a.GetAtomicNum() == 7]
    if (len(neg) != 2 or len(pos) != 1 or len(charged) != 3
            or any(a.GetDegree() != 1 or a.GetTotalNumHs() for a in neg)):
        return None
    p = neg[0].GetNeighbors()[0]
    p_id = p.GetIdx()
    if (p.GetAtomicNum() != 15 or p.GetFormalCharge() or p.IsInRing() or p.GetDegree() != 4
            or p.GetTotalNumHs() != 1 or p.GetTotalValence() != 5
            or neg[1].GetNeighbors()[0].GetIdx() != p_id
            or any(b.GetBondTypeAsDouble() != 1 for b in p.GetBonds())):
        return None
    neg_ids = frozenset(a.GetIdx() for a in neg)
    remaining = [a for a in p.GetNeighbors() if a.GetIdx() not in neg_ids]
    nitrogens = [a for a in remaining if a.GetAtomicNum() == 7]
    oxygens = [a for a in remaining if a.GetAtomicNum() == 8]
    if len(nitrogens) != 1 or len(oxygens) != 1:
        return None
    amino = _amino(mol, nitrogens[0].GetIdx(), p_id)
    oxy = _cation_oxy(mol, Chem, oxygens[0].GetIdx(), p_id, pos[0].GetIdx())
    if amino is None or oxy is None:
        return None
    amino_text, amino_atoms = amino
    oxy_text, oxy_atoms, oxy_locants = oxy
    parts = (neg_ids, frozenset({p_id}), amino_atoms, oxy_atoms)
    if (sum(map(len, parts)) != mol.GetNumAtoms()
            or frozenset().union(*parts) != frozenset(range(mol.GetNumAtoms()))):
        return None
    prefixes = [(oxy_text, '[' + oxy_text + ']'),
                (amino_text, '[' + amino_text + ']' if '(' in amino_text
                 else '(' + amino_text + ')'), ('oxidanidyl', '(oxidanidyl)')]
    prefixes.sort(key=lambda item: prefix_alphabetization_key(item[0]))
    name = '{' + ''.join(text for _, text in prefixes) + '-lambda5-phosphanyl}oxidanide'
    ranks = tuple(Chem.CanonicalRankAtoms(mol, breakTies=True))
    root = min(neg_ids, key=lambda i: ranks[i])
    other = next(iter(neg_ids - {root}))
    roles = (("anionic_parent", frozenset({root}), ((root, ''),)),
             ("anionic_prefix", frozenset({other}), ()),
             ("phosphanyl", frozenset({p_id}), ()),
             ("amino_branch", amino_atoms, ()),
             ("cationic_oxy_branch", oxy_atoms, oxy_locants))
    return MethodDecision(
        name=name, method='substitutive', principal_centers=frozenset({root}),
        atoms=frozenset(range(mol.GetNumAtoms())),
        rule='P-72.6.3/P-73.6/P-74.0/P-74.1.3/P-16.3.4',
        atom_locants=((root, ''),), component_roles=roles,
        preference_issues=(
            SelectionLimitation('anionic-parent', 'P-72.5.1.1/P-72.7',
                'A singly rooted general name is mapped; multiplicative naming '
                'of both equivalent anions and preferred-parent comparison remain unresolved.'),
            SelectionLimitation('anionic-prefix', 'P-72.6.2/P-72.6.3',
                'The systematic oxidanidyl prefix is used; the preselected oxido '
                'alternative and its charge-preserving interpretation remain unresolved.'),
            SelectionLimitation('stereochemistry', 'P-91.2.1.2.1/P-91.3',
                'The original graph lacks a complete preferred stereodescriptor '
                'census or proof of descriptor omission.'),
        ))
