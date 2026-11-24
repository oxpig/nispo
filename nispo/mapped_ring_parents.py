"""P-44 mapped ring proposals and P-29 rooted prefixes on the original graph.

Hydrogen-capped, unannotated cores supply vocabulary/numbering only. All suffix
and branch operations and CIP decoration use original atoms in original scope.
The admitted graph operations are heteromonocycles, retained fused systems,
carbon monocycles, and saturated carbon or heterobicyclic/spiro systems.
"""
from dataclasses import dataclass
from functools import lru_cache
from threading import RLock
from types import SimpleNamespace

from .molecule_iteration import iter_atoms, iter_bonds
from .component_mapping import hydrogen_bindings
from .molecule_cache import mol_cache_dict, mol_cache_get
from .fast_ring_scope import (
    fast_carbon_ring_numbering_enabled,
    without_fast_carbon_ring_numbering,
)
from .parent_selection import graph_parent_facts, locant_set, replace_parent_facts as replace
from .ring_state_parent import (
    _inventory, _reference_states, _render, _skeletons, _VALENCE, _RingNumberingFrontier,
    _skeleton_references, is_supported_hw_chalcogen_core,
)


def _fixed_graph_atoms(graph):
    if type(graph).__module__ == "rdkit.Chem.rdchem" and type(graph).__name__ in {"Mol", "RWMol"}:
        return iter_atoms(graph)
    return graph.GetAtoms()


def _fixed_graph_bonds(graph):
    if type(graph).__module__ == "rdkit.Chem.rdchem" and type(graph).__name__ in {"Mol", "RWMol"}:
        return iter_bonds(graph)
    return graph.GetBonds()


@dataclass(frozen=True, slots=True)
class _RingAttachment:
    """Hashable synthetic suffix facts shared by rooted ring numberings."""

    center: int
    owner: int
    atoms: tuple
    kind: str = 'yl'


def ring_systems(mol):
    # Observe native ring information on every call. The pure grouping can be
    # shared by exact ring tuples without retaining a molecule or hiding an
    # input whose ring information was subsequently initialized or changed.
    return _ring_systems(tuple(mol.GetRingInfo().AtomRings()))


@lru_cache(maxsize=512)
def _ring_systems(rings):
    systems=[]
    for ring in rings:
        merged=set(ring)
        for existing in systems[:]:
            if existing & merged:
                merged.update(existing);systems.remove(existing)
        systems.append(merged)
    return tuple(frozenset(core) for core in systems)


def _core_graph(mol,Chem,core):
    # A lazy first-witness probe and a later complete numbering search use
    # the same capped ring. Store only its binary; each consumer still gets
    # a private scratch graph and the original, uncanonicalized atom order.
    cache = mol_cache_dict(mol, 'mapped_ring_core_graphs')
    key = (Chem, frozenset(core))
    if key in cache:
        entry = cache[key]
        return None if entry is None else (Chem.Mol(entry[0]), entry[1])
    prepared = _build_core_graph(mol, Chem, core)
    cache[key] = (None if prepared is None else
                  (prepared[0].ToBinary(Chem.PropertyPickleOptions.AllProps), prepared[1]))
    return prepared


def _build_core_graph(mol,Chem,core):
    # P-22 permits ordinary divalent Se/Te in the bounded Hantzsch-Widman
    # monocycle. This proof does not broaden fused or polycyclic valences.
    hw_chalcogen_core = is_supported_hw_chalcogen_core(mol, core)
    original=tuple(sorted(core));copy=Chem.Mol(mol)
    try:Chem.Kekulize(copy,clearAromaticFlags=True)
    except (ValueError,RuntimeError):return None
    graph=Chem.RWMol();inverse={i:j for j,i in enumerate(original)}
    capped_hydrogens = {}
    spiro_core = None
    for i in original:
        source=mol.GetAtomWithIdx(i);number=source.GetAtomicNum()
        if source.GetFormalCharge():
            return None
        if number not in _VALENCE and not (hw_chalcogen_core and number in {34, 52}):
            from .constants import STANDARD_REPLACEMENT_VALENCE
            from .heteropolycyclic_parents import free_spiro_core
            if spiro_core is None:
                rings = [r for r in mol.GetRingInfo().AtomRings() if set(r) <= core]
                spiro_core = free_spiro_core(rings)
            if (spiro_core != core or number not in STANDARD_REPLACEMENT_VALENCE
                    or source.GetTotalValence() != STANDARD_REPLACEMENT_VALENCE[number]):
                return None
        atom=Chem.Atom(number)
        capped=source.GetTotalNumHs(includeNeighbors=True)+sum(b.GetBondTypeAsDouble()
            for b in source.GetBonds() if b.GetOtherAtomIdx(i) not in core and mol.GetAtomWithIdx(b.GetOtherAtomIdx(i)).GetAtomicNum()>1)
        if int(capped)!=capped:return None
        capped_hydrogens[i] = capped
        atom.SetNumExplicitHs(int(capped));atom.SetNoImplicit(True);graph.AddAtom(atom)
    for b in iter_bonds(copy):
        a,c=b.GetBeginAtomIdx(),b.GetEndAtomIdx()
        if a in core and c in core:graph.AddBond(inverse[a],inverse[c],b.GetBondType())
    result=graph.GetMol()
    try:Chem.SanitizeMol(result)
    except (ValueError,RuntimeError):return None
    # This is the same original-atom cap used by the later renderer. Only a
    # complete, successfully sanitized core supplies reusable H evidence.
    mol_cache_dict(mol, 'ring_capped_hydrogens').setdefault(
        frozenset(core), capped_hydrogens)
    return result,original


def capped_subtree_fragment(mol, Chem, atoms):
    """Make a sanitized parent fragment without losing annular bond orders.

    Removing the external bond of a neutral pyrrolic N introduces its parent
    hydride's N-H bond before P-29 later removes that H for the yl prefix.
    RDKit cannot infer that H on an aromatic ``n``. A failed sanitization may
    already have changed bond orders, so no partially sanitized graph escapes.
    """
    atoms = frozenset(atoms)
    ordered = tuple(sorted(atoms))
    inverse = {old: local for local, old in enumerate(ordered)}
    graph = Chem.RWMol()
    for i in ordered:
        source = mol.GetAtomWithIdx(i)
        atom = Chem.Atom(source)
        atom.SetAtomMapNum(0)
        removed = tuple(bond for bond in source.GetBonds()
                        if bond.GetOtherAtomIdx(i) not in atoms
                        and bond.GetOtherAtom(source).GetAtomicNum() > 1)
        if source.GetAtomicNum() == 7 and source.GetIsAromatic() and removed:
            # An ordinary N-substituted pyrrole-type atom has three single/
            # aromatic heavy neighbors and no original H. Charged, radical,
            # or multiple free-valence operations require another writer.
            if (source.GetFormalCharge() or source.GetNumRadicalElectrons()
                    or source.GetDegree() != 3 or source.GetTotalValence() != 3
                    or source.GetTotalNumHs(includeNeighbors=True)
                    or len(removed) != 1
                    or removed[0].GetBondTypeAsDouble() != 1):
                return None
            atom.SetNumExplicitHs(atom.GetNumExplicitHs() + 1)
            atom.SetNoImplicit(True)
        graph.AddAtom(atom)
    for bond in _fixed_graph_bonds(mol):
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a not in inverse or b not in inverse:
            continue
        graph.AddBond(inverse[a], inverse[b], bond.GetBondType())
        graph.GetBondBetweenAtoms(inverse[a], inverse[b]).SetIsAromatic(bond.GetIsAromatic())
    fragment = graph.GetMol()
    try:
        Chem.SanitizeMol(fragment)
    except (ValueError, RuntimeError):
        return None
    return fragment, inverse


def skeletons(mol,Chem,core):
    """Replay numbering witnesses, preserving lazy first-witness consumers."""
    core = frozenset(core)
    cache = mol_cache_dict(mol, 'mapped_ring_skeletons')
    key = (id(Chem), core)
    entry = cache.setdefault(key, [[], False])
    prefix = tuple(entry[0])
    def source_allows(row):
        if len(row) > 7 and row[7] == 'component-fused-benzofuran-pyridine':
            from .benzo_fused_pyridine import source_supported
            # The capped core has no source annotations or substituents. The
            # original molecule must independently pass the bounded guard.
            return source_supported(mol)
        if len(row) > 7 and row[7] == 'component-fused-three-face-azine':
            from .three_face_aza_fusion import source_supported
            if not source_supported(mol, core):
                return False
            # A locant-bearing branch must identify the same original atom
            # in the source P-25 drawing and the independent numbered
            # component graph. Bare hydrides have no attachment locant.
            if any(n.GetIdx() not in core and n.GetAtomicNum() > 1
                   for i in core for n in mol.GetAtomWithIdx(i).GetNeighbors()):
                from .fused_numbering_witness import numbering_matches
                return numbering_matches(mol, row[1], row[2], row[0])
            return True
        return True
    for row in prefix:
        if source_allows(row):
            yield row
    if entry[1]:
        return
    # Cache values, never a suspended generator holding the owning molecule.
    # A first-witness scope probe therefore remains lazy; a later full search
    # fills the rest of the same sequence in its original order.
    for index, row in enumerate(_uncached_skeletons(mol, Chem, core)):
        if index < len(prefix):
            continue
        if index == len(entry[0]):
            entry[0].append(row)
        if source_allows(row):
            yield row
    entry[1] = True


def _uncached_skeletons(mol,Chem,core):
    prepared=_core_graph(mol,Chem,core)
    if prepared is None:return
    graph,original=prepared
    source_aromatic = any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in core)
    force_locants = False
    if (graph.GetNumAtoms() > 10 and graph.GetNumBonds() == graph.GetNumAtoms()
            and any(a.GetAtomicNum() != 6 for a in iter_atoms(graph))):
        force_locants = _large_ring_needs_locants(mol, core)
    entry = _core_numbering_cache(Chem, graph.ToBinary(), source_aromatic, force_locants)
    with entry[2]:
        prefix = tuple(entry[0])
        complete = entry[1]
    for row in prefix:
        yield (row[0], tuple(original[i] for i in row[1]), *row[2:])
    if complete:
        return
    for index, row in enumerate(_local_skeletons(graph, Chem, source_aromatic, force_locants)):
        if index < len(prefix):
            continue
        with entry[2]:
            if index == len(entry[0]):
                entry[0].append(row)
        yield (row[0], tuple(original[i] for i in row[1]), *row[2:])
    with entry[2]:
        entry[1] = True


def assembly_unit_skeletons(mol, Chem, core):
    """Supply complete hydride words, including a saturated unit's hydro state.

    Ordinary skeletons deliberately separate retained vocabulary from its
    hydrogenation operation. An assembly cites a complete unit, so it must
    first render that operation instead of repeating the bare mancude word.
    Original atom maps and all equivalent numberings are retained.
    """
    rows = tuple(skeletons(mol, Chem, core))
    prepared = _core_graph(mol, Chem, core)
    if prepared is None:
        return
    graph, original = prepared
    if (not all(b.GetBondTypeAsDouble() == 1 for b in _fixed_graph_bonds(graph))
            or all(row[5] for row in rows)):
        yield from rows
        return
    from .naming_context import MoleculeContext
    from .ring_state_parent import _render
    inverse = {idx: local for local, idx in enumerate(original)}
    ctx = MoleculeContext(graph, Chem)
    inventory = _inventory(graph, Chem)
    if not inventory.supported or inventory.groups:
        return
    for row in rows:
        if row[5]:
            yield row
            continue
        local = (row[0], tuple(inverse[i] for i in row[1]), *row[2:])
        names = set()
        for reference in _reference_states(local[3], local[4]):
            for rendered in _render(ctx, local, reference, inventory):
                if (not rendered.principal_centers and not rendered.prefixes
                        and rendered.construction is not None
                        and rendered.construction.atoms == frozenset(range(graph.GetNumAtoms()))):
                    names.add(rendered.name)
        if len(names) == 1:
            yield (names.pop(), *row[1:])



@lru_cache(maxsize=512)
def _core_numbering_cache(Chem, graph_binary, source_aromatic, force_locants):
    # Bounded vocabulary witnesses only: no original molecule, annotations,
    # branch names, or suspended generator survives in this cache. Binary
    # graph identity preserves the original enumeration and its limits.
    return [[], False, RLock()]



def _large_ring_needs_locants(mol, core):
    # P-14.3.3: the otherwise implicit heteroatom locant belongs in a
    # unit whose substitution or isotope/stereo operation needs locants.
    force_locants = any(a.GetIdx() not in core and a.GetAtomicNum() > 1
                        for a in iter_atoms(mol))
    populations = {}
    for i in core:
        atom = mol.GetAtomWithIdx(i)
        populations.setdefault(atom.GetAtomicNum(), []).append(atom.GetIsotope())
    force_locants |= any(len(masses) > 1 and any(masses)
                          and not all(mass and mass == masses[0] for mass in masses)
                          for masses in populations.values())
    force_locants |= any(int(mol.GetAtomWithIdx(i).GetChiralTag())
                          or any(n.GetAtomicNum() == 1 and n.GetIsotope()
                                 for n in mol.GetAtomWithIdx(i).GetNeighbors())
                          for i in core)
    return force_locants



def _local_skeletons(graph, Chem, source_aromatic, force_locants):
    from .naming_context import MoleculeContext
    from .retained_von_baeyer_parents import skeletons as retained_cage_skeletons
    annular_yne = (graph.GetNumBonds() == graph.GetNumAtoms()
                   and any(b.GetBondTypeAsDouble() == 3 for b in _fixed_graph_bonds(graph)))
    if annular_yne:
        # RDKit can mark the other bonds of a strained annul-yne aromatic.
        # A mancude retained word would erase the triple bond. Keep one
        # complete Kekule graph for P-31's explicit ene/yne construction;
        # the absolute H vector is checked again on the original molecule.
        graph = Chem.Mol(graph)
        try:
            Chem.Kekulize(graph, clearAromaticFlags=True)
        except (ValueError, RuntimeError):
            return
        source_aromatic = False
    else:
        retained=tuple(retained_cage_skeletons(graph,Chem))
        if retained:
            yield from retained
            return
    found=False
    for skeleton in (() if annular_yne else _skeletons(MoleculeContext(graph,Chem))):
        found=True
        yield (skeleton[0],tuple(skeleton[1]),*skeleton[2:7],
            skeleton[7] if len(skeleton)>7 else
            'Hantzsch-Widman' if graph.GetRingInfo().NumRings()==1 else 'retained-fused',
            *skeleton[8:])
    if found:return  # P-52 retained/HW vocabulary precedes other constructions.
    if graph.GetNumBonds() - graph.GetNumAtoms() + 1 in {2, 3}:
        from .localized_bicyclic_states import fallback_skeletons
        explicit = tuple(fallback_skeletons(graph, Chem))
        if explicit:
            yield from explicit
            return
    if graph.GetNumBonds()-graph.GetNumAtoms()+1 in {3, 4}:
        from .compound_spiro_parents import skeletons as compound_spiro_skeletons, unsaturated_skeletons
        compound = tuple(compound_spiro_skeletons(graph, Chem) if not source_aromatic
                         else unsaturated_skeletons(graph, Chem))
        if compound:
            yield from compound
            return
    if (graph.GetNumBonds() == graph.GetNumAtoms()
            and any(b.GetBondTypeAsDouble() > 1 for b in _fixed_graph_bonds(graph))
            and not source_aromatic):
        from .unsaturated_ring_parents import unsaturated_monocycle_frontier
        adjacency = {a.GetIdx(): {n.GetIdx() for n in a.GetNeighbors()}
                     for a in _fixed_graph_atoms(graph)}
        elements = {a.GetIdx(): a.GetAtomicNum() for a in _fixed_graph_atoms(graph)}
        orders = {frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())):
                  int(b.GetBondTypeAsDouble()) for b in _fixed_graph_bonds(graph)}
        frontier = unsaturated_monocycle_frontier(adjacency, elements, orders)
        if frontier.status != 'complete':
            return
        for numbering in frontier.candidates:
            order = numbering.sequence
            inverse = {idx: j for j, idx in enumerate(order)}
            edges = tuple((inverse[b.GetBeginAtomIdx()], inverse[b.GetEndAtomIdx()])
                          for b in _fixed_graph_bonds(graph))
            method = ('skeletal-replacement-unsaturated-monocycle'
                      if numbering.hetero_locants else 'unsaturated-monocycle')
            yield (numbering.parent_component, tuple(order),
                   tuple(map(str, range(1, len(order) + 1))),
                   tuple(elements[i] for i in order), edges, False, (), method,
                   numbering)
        return
    if any(a.GetAtomicNum()!=6 for a in _fixed_graph_atoms(graph)):
        if (graph.GetNumAtoms() > 10
                and graph.GetNumBonds() == graph.GetNumAtoms()
                and all(b.GetBondTypeAsDouble() == 1 for b in _fixed_graph_bonds(graph))):
            from .large_heteromonocycle_methods import large_heteromonocycle_frontier
            adjacency = {a.GetIdx(): {n.GetIdx() for n in a.GetNeighbors()}
                         for a in _fixed_graph_atoms(graph)}
            elements = {a.GetIdx(): a.GetAtomicNum() for a in _fixed_graph_atoms(graph)}
            frontier = large_heteromonocycle_frontier(adjacency, elements)
            if frontier.status != 'complete':
                return
            for numbering in frontier.candidates:
                parts = []
                for term in numbering.replacement_terms:
                    locations = term.locants
                    if force_locants and term.omit_locant:
                        locations = numbering.hetero_locants
                    parts.append((','.join(map(str, locations)) + '-' if locations else '')
                                 + term.term)
                parent = '-'.join(parts) + numbering.base_hydride
                order = numbering.sequence
                inverse = {idx: j for j, idx in enumerate(order)}
                edges = tuple((inverse[b.GetBeginAtomIdx()], inverse[b.GetEndAtomIdx()])
                              for b in _fixed_graph_bonds(graph))
                yield (parent, tuple(order),
                       tuple(map(str, range(1, len(order) + 1))),
                       tuple(elements[i] for i in order), edges, True, (),
                       'skeletal-replacement-monocycle')
            return
        if graph.GetNumBonds() - graph.GetNumAtoms() + 1 == 3:
            from .heterodispiro_parents import skeletons as heterodispiro_skeletons
            dispiro = tuple(heterodispiro_skeletons(graph, Chem))
            for skeleton in dispiro:
                yield (skeleton[0], tuple(skeleton[1]), *skeleton[2:])
            if dispiro:
                return
        from .polyspiro_frontier import MAX_RINGS
        if 4 <= graph.GetNumBonds() - graph.GetNumAtoms() + 1 <= MAX_RINGS:
            from .heteropolyspiro_methods import heteropolyspiro_parent_proposals
            for row in heteropolyspiro_parent_proposals(graph, Chem).proposals:
                inverse = {i: j for j, i in enumerate(row.sequence)}
                edges = tuple((inverse[b.GetBeginAtomIdx()], inverse[b.GetEndAtomIdx()])
                              for b in _fixed_graph_bonds(graph))
                yield (row.name, row.sequence, tuple(loc for _, loc in row.atom_locants),
                       row.atomic_numbers, edges, True, row.criteria_key, 'skeletal-replacement-polyspiro')
            return
        from .heteropolycyclic_parents import skeletons as heteropolycyclic_skeletons
        for skeleton in heteropolycyclic_skeletons(graph):
            yield (skeleton[0],tuple(skeleton[1]),*skeleton[2:])
        return
    rings=graph.GetRingInfo().AtomRings();size=graph.GetNumAtoms()
    saturated=all(b.GetBondTypeAsDouble()==1 for b in _fixed_graph_bonds(graph))
    if len(rings)==1:
        from .constants import ROOTS
        if size not in ROOTS or not saturated and not (size==6 and all(a.GetIsAromatic() for a in _fixed_graph_atoms(graph))):return
        parent='cyclo'+ROOTS[size]+'ane' if saturated else 'benzene'
        ring=tuple(rings[0])
        for direction in (ring,tuple(reversed(ring))):
            for start in range(size):
                seq=direction[start:]+direction[:start]
                yield (parent,tuple(seq),tuple(map(str,range(1,size+1))),
                    (6,)*size,tuple((j,(j+1)%size) for j in range(size)),saturated,(),'monocyclic')
    elif graph.GetNumBonds() - size + 1 == 2 and saturated:
        # P-23/P-24 saturated carbon constructions; the retained fused
        # vocabulary admitted by this producer has already been considered.
        shared=set(rings[0])&set(rings[1])
        if len(shared) != 1:
            from .von_baeyer_frontier import bicyclic_theta_frontier
            from .constants import ROOTS
            adjacency = {a.GetIdx(): {n.GetIdx() for n in a.GetNeighbors()}
                         for a in _fixed_graph_atoms(graph)}
            frontier = bicyclic_theta_frontier(adjacency)
            if frontier.status == 'incomplete':
                return
            if frontier.status == 'complete':
                if size not in ROOTS:
                    return
                for numbering in frontier.candidates:
                    order = numbering.sequence
                    parent = ('bicyclo[' + '.'.join(map(str, numbering.bridge_lengths))
                              + ']' + ROOTS[size] + 'ane')
                    inverse = {idx: j for j, idx in enumerate(order)}
                    edges = tuple((inverse[b.GetBeginAtomIdx()], inverse[b.GetEndAtomIdx()])
                                  for b in _fixed_graph_bonds(graph))
                    yield (parent, tuple(order),
                           tuple(map(str, range(1, size + 1))), (6,) * size,
                           edges, True, (), 'von-Baeyer')
                return
        from .polycyclic_rules import bicyclic_parent_candidate_records,spiro_parent_candidate_records
        records=(spiro_parent_candidate_records if len(shared)==1 else bicyclic_parent_candidate_records)(graph,Chem)
        seen=set()
        for parent,seq,*extras in records:
            if extras or set(seq)!=set(range(size)):continue
            # Every automorphism of the fixed numbered hydride is a valid
            # placement on the original core. Extrinsic isotope H must never
            # prune an orientation before suffix numbering (P-82.5).
            for symmetry in graph.GetSubstructMatches(graph,uniquify=False,maxMatches=10000):
                order=tuple(symmetry[i] for i in seq)
                if (parent,order) in seen:continue
                seen.add((parent,order));inverse={idx:j for j,idx in enumerate(order)}
                edges=tuple((inverse[b.GetBeginAtomIdx()],inverse[b.GetEndAtomIdx()]) for b in _fixed_graph_bonds(graph))
                yield (parent,tuple(order),tuple(map(str,range(1,size+1))),
                    (6,)*size,edges,True,(),'spiro' if len(shared)==1 else 'von-Baeyer')
    elif graph.GetNumBonds() - size + 1 == 3 and saturated:
        from .dispiro_frontier import dispiro_frontier
        frontier = dispiro_frontier(graph, Chem)
        if frontier.status == 'incomplete':
            return
        if frontier.status == 'complete' and frontier.candidates:
            for numbering in frontier.candidates:
                terms = [str(term.linking_atoms)
                         + (f'^{term.repeated_spiro_locant}'
                            if term.repeated_spiro_locant is not None else '')
                         for term in numbering.descriptor]
                parent = 'dispiro[' + '.'.join(terms) + ']' + numbering.hydrocarbon_parent
                order = numbering.sequence
                inverse = {idx: j for j, idx in enumerate(order)}
                edges = tuple((inverse[b.GetBeginAtomIdx()], inverse[b.GetEndAtomIdx()])
                              for b in _fixed_graph_bonds(graph))
                yield (parent, tuple(order),
                       tuple(map(str, range(1, size + 1))), (6,) * size,
                       edges, True, (), 'dispiro')
            return
        # Dispiro vocabulary has had first refusal. This bounded P-23 consumer
        # admits only the same neutral saturated all-carbon rank-three core.
        if all(a.GetAtomicNum() == 6 and a.GetFormalCharge() == 0
               and a.GetNumRadicalElectrons() == 0 for a in _fixed_graph_atoms(graph)):
            from .tricyclic_von_baeyer_frontier import tricyclic_von_baeyer_frontier
            adjacency = {a.GetIdx(): {n.GetIdx() for n in a.GetNeighbors()}
                         for a in _fixed_graph_atoms(graph)}
            tricyclic = tricyclic_von_baeyer_frontier(adjacency)
            if tricyclic.status != 'complete':
                return
            from .constants import ROOTS
            if size not in ROOTS:
                return
            for numbering in tricyclic.candidates:
                parent = numbering.bridge_descriptor + ROOTS[size] + 'ane'
                order = numbering.sequence
                inverse = {idx: j for j, idx in enumerate(order)}
                edges = tuple((inverse[b.GetBeginAtomIdx()], inverse[b.GetEndAtomIdx()])
                              for b in _fixed_graph_bonds(graph))
                yield (parent, tuple(order),
                       tuple(map(str, range(1, size + 1))), (6,) * size,
                       edges, True, (), 'von-Baeyer')


def _site(mol,group,core=None):
    if group.kind in {'alcohol','ketone','lactam','lactone','pseudoketone'}:return group.center
    if group.kind in {'imine','oxime','oxime-ether'}:return group.center
    if group.kind=='peroxol':return group.owner
    if group.kind=='amine':
        from .amine_components import carbon_hosts
        hosts=carbon_hosts(mol,group,core)
        return hosts[0] if len(hosts)==1 else None
    return group.owner


def records(mol,Chem,*,scope=None,attachment=None,core_filter=None,principal=None,
            nitrogen_attachment=None,ester_arms=(),general_branches=False,local_inventory=False):
    # Parent orientations repeatedly request the same original rooted scope.
    # Cache immutable witnesses on the API-owned molecule, never globally or
    # on a capped fragment with different CIP priorities.
    # A core filter is redundant when this scope contains exactly that one
    # complete ring system.  Rooted and explicitly proved fused consumers
    # otherwise rebuild the same numbering and branch constructions.
    if (type(scope) in (set, frozenset, list, tuple)
            and type(core_filter) in (set, frozenset)):
        scoped = frozenset(scope)
        wanted = frozenset(core_filter)
        if (principal != () and tuple(core for core in ring_systems(mol) if core <= scoped) == (wanted,)):
            core_filter = None
    key=(None if scope is None else frozenset(scope),attachment,
         None if core_filter is None else frozenset(core_filter),principal,nitrogen_attachment,
         ester_arms,general_branches,local_inventory)
    cache=mol_cache_dict(mol,'mapped_ring_records')
    if key not in cache:
        cache[key]=_records(mol,Chem,scope=scope,attachment=attachment,core_filter=core_filter,
                            principal=principal,nitrogen_attachment=nitrogen_attachment,
                            ester_arms=ester_arms,general_branches=general_branches,
                            local_inventory=local_inventory)
    return cache[key]


def proposal_plans(mol, Chem, principal):
    """Expose one graph-bound construction per physical ring system.

    Rooted-prefix callers retain the eager records contract. The cross-route
    pool can defer complete parent construction until its coverage and parent
    seniority can still affect selection.
    """
    from .proposal_plans import ParentProposalPlan, graph_proposal_bound
    from .cross_route_selection import ParentProposal

    def construct(core):
        aggregate_key = (None, None, None, principal, None)
        cache = mol_cache_dict(mol, 'mapped_ring_records')
        if aggregate_key in cache:
            rows = tuple(record for record in cache[aggregate_key]
                         if frozenset(record.parent_atoms) == core)
        else:
            rows = records(mol, Chem, core_filter=core, principal=principal)
        proposals = []
        for record in rows:
            facts = parent_facts(mol, record, principal[0].seniority)
            if facts is not None:
                proposals.append(ParentProposal(
                    record.name, 'mapped-ring:' + record.method, principal[0].kind,
                    record.principal_centers, record.parent_atoms, record.atom_locants,
                    facts, True, ring_state=record, construction=record.construction,
                ))
        return tuple(proposals)

    for core in ring_systems(mol):
        groups = tuple(group for group in principal if _site(mol, group, core) in core)
        if not groups:
            continue
        yield ParentProposalPlan(
            'mapped-ring', tuple(sorted(core)), frozenset(group.center for group in groups),
            graph_proposal_bound(mol, core, groups, cyclic=True),
            lambda core=core: construct(core), ring_route=True,
        )


def _requires_relative_ring_writer(mol, Chem, core):
    """Keep a coupled relative monocycle with its established P-93 writer.

    The mapped CIP writer has no proof for the coupled units when ordinary
    tetrahedral assignment does not identify both tagged ring atoms. The
    existing cis/trans writer can preserve their relative configuration.
    """
    if any(sum(n.GetIdx() in core for n in mol.GetAtomWithIdx(i).GetNeighbors()) != 2
           for i in core):
        return False
    from .stereochemistry import (
        _simple_relative_ring_branch_tagged_atoms, assigned_atom_stereochemistry,
    )
    tagged = _simple_relative_ring_branch_tagged_atoms(mol, Chem, core)
    if not tagged:
        return False
    assigned = dict(assigned_atom_stereochemistry(mol, Chem, atom_order='idx'))
    return not tagged <= assigned.keys()


def _fast_rooted_carbon_ring(mol, scope, attachment, core, principal, nitrogen_attachment):
    """Prove a single, unannotated carbon ring with one rooted single bond.

    For a saturated or wholly aromatic carbon monocycle, rotating the parent
    to put its free valence at position 1 preserves its ring and hydrogen
    state. Numberings rooted elsewhere cannot beat that suffix locant. The
    remaining orientations still go through the ordinary numbering frontier.
    """
    if (attachment is None or scope is None or principal is not None
            or nitrogen_attachment is not None or mol.GetStereoGroups()):
        return False
    scope = frozenset(scope)
    if attachment not in core or attachment not in scope:
        return False
    if any(atom.GetAtomicNum() == 1 or atom.GetFormalCharge() or atom.GetIsotope()
           or atom.GetNumRadicalElectrons() or atom.GetAtomMapNum() or atom.HasQuery()
           or atom.HasProp('_UnknownStereo')
           or int(atom.GetChiralTag()) and atom.GetIdx() in scope
           for atom in iter_atoms(mol)):
        return False
    cuts, inner = [], []
    for bond in iter_bonds(mol):
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (bond.HasQuery() or (int(bond.GetStereo()) or int(bond.GetBondDir()))
                and (left in scope or right in scope)):
            return False
        if (left in scope) != (right in scope):
            cuts.append(bond)
        if left in core and right in core:
            inner.append(bond)
    if (len(cuts) != 1 or cuts[0].GetBondTypeAsDouble() != 1
            or attachment not in {cuts[0].GetBeginAtomIdx(), cuts[0].GetEndAtomIdx()}):
        return False
    if tuple(system for system in ring_systems(mol) if system <= scope) != (core,):
        return False
    rings = tuple(frozenset(ring) for ring in mol.GetRingInfo().AtomRings()
                  if frozenset(ring) <= core)
    if len(rings) != 1 or rings[0] != core or len(inner) != len(core):
        return False
    saturated = all(bond.GetBondTypeAsDouble() == 1 and not bond.GetIsAromatic()
                    for bond in inner)
    aromatic = (all(bond.GetIsAromatic() for bond in inner)
                and all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in core))
    return saturated or aromatic


def _records(mol,Chem,*,scope=None,attachment=None,core_filter=None,principal=None,
             nitrogen_attachment=None,ester_arms=(),general_branches=False,local_inventory=False):
    from .naming_context import MoleculeContext
    if mol.GetStereoGroups() or any(a.GetNumRadicalElectrons()
        or int(a.GetChiralTag()) and a.GetAtomicNum()!=6 for a in iter_atoms(mol)):return ()
    if any(atom.GetFormalCharge() for atom in iter_atoms(mol)):
        from .extended_inventory import selection_inventory
        from .neutral_charge_components import validate_neutral_charge_role_proofs
        from .mapped_principal_general import selected_acid_anions

        source_inventory = selection_inventory(mol)
        anion = (general_branches and attachment is None and principal is not None
                 and tuple(principal) == selected_acid_anions(mol, source_inventory)
                 and (scope is None or all(set(group.atoms) <= scope for group in principal)))
        if not anion and not validate_neutral_charge_role_proofs(mol, source_inventory.charge_role_proofs):
            return ()
    ctx=MoleculeContext(mol,Chem);inventory=_inventory(mol,Chem)
    if not inventory.supported:
        if (attachment is None or scope is None or not general_branches
                or not (local_inventory and principal == () or mol_cache_get(mol, 'numbered_polycycle_recovery'))):
            return ()
        # A numbered annular parent can own an unsupported group outside
        # this branch. Admit a group-free branch only after local recognition.
        capped = capped_subtree_fragment(mol, Chem, scope)
        if capped is None:
            return ()
        inventory = _inventory(capped[0], Chem)
        if not inventory.supported or inventory.groups:
            return ()
    needed=frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum()>1 or a.GetIsotope() or int(a.GetChiralTag()))
    scope=needed if scope is None else frozenset(scope)
    source_groups=inventory.principal if principal is None else principal
    if attachment is None and not source_groups and core_filter is None:return ()
    output=[]
    for core in ring_systems(mol):
        if not core<=scope or core_filter is not None and core!=core_filter:continue
        if attachment is not None and _requires_relative_ring_writer(mol, Chem, core):
            continue
        if attachment is not None:
            if attachment not in core:continue
            atom=mol.GetAtomWithIdx(attachment)
            nitrogen_site = (atom.GetAtomicNum()==7 and not atom.GetFormalCharge()
                             and atom.GetTotalValence()==3
                             and (atom.GetIsAromatic() or
                                  all(b.GetBondTypeAsDouble()==1 for b in atom.GetBonds())))
            # A pyrrolic N attachment consumes the indicated H of the
            # numbered parent just like an ordinary N-yl single valence.
            # The capped graph and H-state renderer prove that operation;
            # excluding aromatic N here prevents otherwise complete ring
            # assemblies from being constructed through those atoms.
            if atom.GetAtomicNum()!=6 and not nitrogen_site:continue
            cut_bonds = [b for b in atom.GetBonds() if b.GetOtherAtomIdx(attachment) not in scope
                         and b.GetOtherAtom(atom).GetAtomicNum() > 1]
            double_cut = len(cut_bonds)==1 and cut_bonds[0].GetBondTypeAsDouble()==2
            if double_cut and atom.GetAtomicNum()!=6:continue
            groups=(_RingAttachment(attachment, attachment, (attachment,),
                                   'ylidene' if double_cut else 'yl'),)
        else:groups=tuple(g for g in source_groups if _site(mol,g,core) in core)
        if not groups and (source_groups or core_filter is None):continue
        # Only scope-local functions can affect prefix vocabulary. Functions
        # outside a rooted branch belong to its enclosing component.
        local=SimpleNamespace(principal=groups,groups=tuple(g for g in inventory.groups if set(g.atoms)<=scope),
                              ester_arms=ester_arms,general_branches=general_branches)
        candidates=[]
        fast_rooted_carbon = (fast_carbon_ring_numbering_enabled()
            and all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in core)
            and _fast_rooted_carbon_ring(mol, scope, attachment, core, principal, nitrogen_attachment))
        all_parents, kept_parents = (set(), set()) if fast_rooted_carbon else (None, None)
        # The N-attachment writer applies a different key after rendering.
        numbering_frontier = (_RingNumberingFrontier.for_context(ctx)
                              if nitrogen_attachment is None else None)
        for skeleton in skeletons(mol,Chem,core):
            if fast_rooted_carbon:
                all_parents.add(skeleton[0])
                if attachment not in skeleton[1]:
                    continue
                root_position = skeleton[1].index(attachment)
                if (root_position >= len(skeleton[2])
                        or str(skeleton[2][root_position]) != '1'):
                    continue
                kept_parents.add(skeleton[0])
            # P-31 absolute ene/yne states already carry the actual weighted
            # hydrogen vector. They do not use mancude hydrogen accommodation.
            if nitrogen_attachment is not None and len(skeleton) > 8:
                continue  # A rooted N-site numbering proof remains separate.
            for reference in _skeleton_references(skeleton):
                candidates.extend(_render(ctx,skeleton,reference,local,principal=groups,scope=scope,
                                          numbering_frontier=numbering_frontier))
        if numbering_frontier is not None:
            candidates.extend(numbering_frontier.finish())
        if fast_rooted_carbon and (all_parents != kept_parents or not candidates
                or len({r.parent for r in candidates}) != 1):
            # If no complete rooted candidate survived, or pruning removed an
            # entire vocabulary, retain the original exhaustive behavior.
            return without_fast_carbon_ring_numbering(
                _records, mol, Chem, scope=scope, attachment=attachment,
                core_filter=core_filter, principal=principal,
                nitrogen_attachment=nitrogen_attachment,ester_arms=ester_arms,
                general_branches=general_branches, local_inventory=local_inventory)
        if not candidates or len({r.parent for r in candidates})!=1:continue
        def numbering(record):
            key=record.numbering_key
            if nitrogen_attachment is None:return key
            from .parent_selection import locant_key
            loc=locant_key(dict(record.atom_locants)[nitrogen_attachment])
            return key[:8]+(tuple(sorted(key[8]+(loc,))),(loc,))+key[10:]
        if nitrogen_attachment is not None:
            candidates=[r for r in candidates if nitrogen_attachment in r.principal_centers and not r.prefixes]
            if not candidates:continue
        key=min(numbering(r) for r in candidates);finalists=[r for r in candidates if numbering(r)==key]
        if len({r.name for r in finalists})==1:output.append(finalists[0])
    return tuple(output)


def parent_facts(mol,record,principal_class):
    from .ring_parent_selection import ring_type_evidence
    from .extended_inventory import selection_inventory
    mapping=dict(record.atom_locants)
    facts=graph_parent_facts(mol,record.parent_atoms,mapping,principal_class=principal_class,
        suffix_centers=record.principal_centers,suffix_locants=record.suffix_locants,
        prefixes=record.prefixes,kind='ring',name=record.name,
        indicated_h=record.indicated_h_locants,allow_annotations=True,
        charge_role_proofs=selection_inventory(mol).charge_role_proofs,
        prefix_branch_ownership=getattr(record.construction,'prefix_branch_ownership',()),
        replacement=record.method.startswith('skeletal-replacement-'))
    if facts is None:return None
    kind,details,identity=ring_type_evidence(mol,record.parent_atoms,mapping,record.parent)
    fusion = getattr(record, 'fusion_construction', None)
    if fusion is not None:
        from .fusion_components import component_list_key, fusion_descriptor_evidence
        letters, numbers = fusion_descriptor_evidence(fusion)
        details = (*details[:2], letters, numbers, component_list_key(fusion))
    # Other mapped ring producers supply the same record protocol without an
    # absolute unsaturation state. Its absence carries the existing unknown
    # locant evidence; it must not make those saturated producers unusable.
    unsaturation_state = getattr(record, 'unsaturation_state', None)
    def multiple_locations(order=None):
        if unsaturation_state is None:
            return ()
        return locant_set(str(a) if b - a == 1 else f'{a}({b})'
                          for a, b, degree in unsaturation_state.edge_locants
                          if order is None or degree == order)
    # Intrinsic graph and fixed vocabulary prove ties within the same parent;
    # absent explicit unsaturation locants are not a tie across different
    # unproved vocabularies. Each record's own numbering was solved earlier.
    return replace(facts,ring_type=kind,ring_type_details=details,ring_identity=identity,
        fusion_construction=fusion,
        fusion_scope=frozenset(record.parent_atoms) if fusion is not None else None,
        fusion_locants=(),
        unsaturation_locants=multiple_locations(), double_locants=multiple_locations(2))


def rooted_prefix(mol,Chem,root,blocked,atoms):
    order = mol.GetBondBetweenAtoms(root,blocked).GetBondTypeAsDouble()
    if order not in {1, 2}:return '',frozenset()
    if order == 2:
        # The first two-free-valence construction covers undecorated junction
        # stereochemistry. Cross-component E/Z needs its own locant proof.
        from .component_mapping import has_annotations
        if (mol.GetAtomWithIdx(root).GetAtomicNum()!=6
                or mol.GetBondBetweenAtoms(root,blocked).GetStereo()!=Chem.BondStereo.STEREONONE
                or has_annotations(mol, atoms | {blocked})):
            return '',frozenset()
    candidates=records(mol,Chem,scope=atoms,attachment=root)
    if len(candidates)!=1:return '',frozenset()
    # This compatibility pair carries text and ownership only. New alternative
    # P-31 fused constructions must not lose their pending preferred-fusion
    # evidence when embedded by a parent that would otherwise claim resolved.
    if candidates[0].method in {'component-fused-benzofuran-pyridine',
                                'component-fused-three-face-azine',
                                'component-fused-general',
                                'skeletal-replacement-explicit-bicyclic',
                                'skeletal-replacement-localized-tricyclic'}:
        return '',frozenset()
    return candidates[0].name,frozenset(atoms)


def rooted_benzo_imidazothiazole(mol, Chem, root, atoms, core):
    """Use the complete fused branch on an unannotated three-face scope."""
    from .component_mapping import has_annotations

    if (len(core) != 12 or sorted(mol.GetAtomWithIdx(i).GetAtomicNum()
                                 for i in core) != [6] * 9 + [7, 7, 16]
            or not core <= atoms or has_annotations(mol, atoms)):
        return ''
    rows = records(mol, Chem, scope=atoms, attachment=root, core_filter=core)
    if (len(rows) == 1 and rows[0].method == 'component-fused-benzo-imidazothiazole'
            and rows[0].hydrogen_budget_proved and rows[0].annotation_coverage
            and rows[0].construction is not None
            and rows[0].construction.atoms | rows[0].construction.embedded_atoms == atoms):
        return rows[0].name
    return ''


def proved_fused_rooted_prefix(mol, Chem, root, blocked, atoms):
    """Supply a complete H-state prefix at a proved aromatic fused site.

    A local oxo word cannot be inserted into a separately named fused parent:
    its indicated-H state and numbering belong to the whole fused system.
    This path instead renders every prefix and the rooted free valence in one
    mapped P-25/P-29/P-54/P-58 construction. Capping external bonds must prove
    one annular pi graph on the original vertices; localized pi isomers with
    the same hydrogen population require the separate explicit P-31 route.
    An aromatic carbon uses an ordinary substitutable C-H free valence. An
    aromatic nitrogen is admitted only when capping the removed attachment
    gives one neutral N-H and the joined construction cites that same site as
    indicated hydrogen. Reduced-carbon attachment still requires a separate
    joined hydrogen-state proof.
    """
    from .hydrogen_accommodation import unique_capped_pi_graph

    bond = mol.GetBondBetweenAtoms(root, blocked)
    root_atom = mol.GetAtomWithIdx(root)
    reduced_nitrogen = (root_atom.GetAtomicNum() == 7 and not root_atom.GetIsAromatic()
                        and root_atom.GetDegree() == 3 and root_atom.GetTotalValence() == 3)
    if (bond is None or bond.GetBondTypeAsDouble() != 1
            or root_atom.GetAtomicNum() not in {6, 7}
            or not (root_atom.GetIsAromatic() or reduced_nitrogen)):
        return '', frozenset()
    core = next((system for system in ring_systems(mol) if root in system), None)
    if core is None or blocked in core:
        return '', frozenset()
    adjacency = {i: {n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors()
                     if n.GetIdx() in core} for i in core}
    junctions = tuple(i for i, neighbors in adjacency.items() if len(neighbors) == 3)
    if (len(junctions) != 2 or junctions[1] not in adjacency[junctions[0]]
            or any(len(neighbors) not in {2, 3} for neighbors in adjacency.values())
            or sum(map(len, adjacency.values())) != 2 * (len(core) + 1)):
        return '', frozenset()
    prepared = _core_graph(mol, Chem, core)
    if prepared is None:
        return '', frozenset()
    graph, original = prepared
    if root_atom.GetAtomicNum() == 7:
        capped = graph.GetAtomWithIdx(original.index(root))
        if (root_atom.GetFormalCharge() or root_atom.GetIsotope()
                or root_atom.GetNumRadicalElectrons() or root_atom.GetDegree() != 3
                or root_atom.GetTotalNumHs(includeNeighbors=True) != 0
                or sum(neighbor.GetIdx() in core for neighbor in root_atom.GetNeighbors()) != 2
                or not (capped.GetIsAromatic() or reduced_nitrogen) or capped.GetFormalCharge()
                or capped.GetTotalNumHs(includeNeighbors=True) != 1):
            return '', frozenset()
    numbers = tuple(atom.GetAtomicNum() for atom in _fixed_graph_atoms(graph))
    edges = tuple((edge.GetBeginAtomIdx(), edge.GetEndAtomIdx())
                  for edge in _fixed_graph_bonds(graph))
    hydrogen = tuple(atom.GetTotalNumHs() for atom in _fixed_graph_atoms(graph))
    if not unique_capped_pi_graph(numbers, edges, hydrogen):
        return '', frozenset()
    candidates = records(mol, Chem, scope=atoms, attachment=root, core_filter=core)
    if len(candidates) != 1:
        return '', frozenset()
    record = candidates[0]
    if not (record.indicated_h_locants or record.added_h_locants or record.hydro_locants):
        return '', frozenset()
    if root_atom.GetAtomicNum() == 7:
        locant = dict(record.atom_locants).get(root)
        if (reduced_nitrogen and not (record.parent.endswith('-naphthyridine')
                                     and locant in record.hydro_locants)
                or not reduced_nitrogen and locant not in record.indicated_h_locants):
            return '', frozenset()
    construction = record.construction
    if (not record.hydrogen_budget_proved or not record.annotation_coverage
            or record.preference_issues or construction is None
            or construction.atoms | construction.embedded_atoms != frozenset(atoms)):
        return '', frozenset()
    return record.name, frozenset(atoms)


def heteropolycyclic_hydride(mol,Chem):
    """The same mapped construction for a function-free hydride and its prefixes.

    Each complete ring core supplies its eligible vocabulary and numbering.
    The renderer proves ownership of every remaining atom through mapped
    substituents, then P-44 compares the distinct physical parents. Alternative
    fused vocabulary never competes against retained vocabulary by name score.
    """
    if mol.GetStereoGroups() or any(a.GetFormalCharge() or a.GetNumRadicalElectrons()
            or int(a.GetChiralTag()) and a.GetAtomicNum()!=6 for a in iter_atoms(mol)):return None
    systems=ring_systems(mol)
    if not systems:return None
    inventory=_inventory(mol,Chem)
    if not inventory.supported or inventory.groups:return None
    from .naming_context import MoleculeContext
    ctx=MoleculeContext(mol,Chem)
    scope=frozenset(a.GetIdx() for a in iter_atoms(mol)
                    if a.GetAtomicNum()>1 or a.GetIsotope())
    parents=[]
    for core in systems:
        if _requires_relative_ring_writer(mol, Chem, core):return None
        candidates=[]
        numbering_frontier = _RingNumberingFrontier.for_context(ctx)
        for skeleton in skeletons(mol,Chem,core):
            method=skeleton[7]
            if not (method.startswith('skeletal-replacement-') or method in {
                    'dispiro','unsaturated-monocycle','monocyclic',
                    'Hantzsch-Widman','retained-fused','component-fused',
                    'component-fused-benzofuran-pyridine','component-fused-three-face-azine','component-fused-general',
                    'component-spiro','benzo-heterocycle'}):
                continue  # Established P-23/P-24 writers own other methods.
            references=((None,) if skeleton[5] or len(skeleton)>8 else
                        _reference_states(skeleton[3],skeleton[4]))
            for reference in references:
                candidates.extend(_render(ctx,skeleton,reference,inventory,scope=scope,
                                          numbering_frontier=numbering_frontier))
        # Vocabulary eligibility and the fixed-parent numbering are decided
        # within this physical system, before comparing distinct P-44 parents.
        # An unrendered competing system leaves preference open.
        if numbering_frontier is not None:
            candidates.extend(numbering_frontier.finish())
        if not candidates or len({r.parent for r in candidates})!=1:return None
        key=min(r.numbering_key for r in candidates)
        finalists=[r for r in candidates if r.numbering_key==key]
        if len({r.name for r in finalists})!=1:return None
        # P-22.1.3/P-53 retain the exact unmodified mono-/dimethylbenzene
        # hydrides. Their existing ring-chain parent writer owns that
        # vocabulary; a bare benzene construction must not bypass it.
        record=finalists[0]
        if (len(systems)==1 and record.parent=='benzene'
                and len(record.prefixes) in {1,2}
                and all(text=='methyl' for _,text in record.prefixes)
                and not any(a.GetIsotope() or int(a.GetChiralTag()) for a in iter_atoms(mol))):
            return None
        parents.append(finalists[0])
    from .parent_selection import parent_frontier
    selected=parent_frontier(parents,lambda r:parent_facts(mol,r,()))
    return selected[0] if len(selected)==1 else None
