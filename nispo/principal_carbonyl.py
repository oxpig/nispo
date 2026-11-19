"""Conservative graph evidence for principal ring carbonyl suffixes.

Functional-class coverage precedes presentation length. This module does not
infer tautomer states or renumber fused systems; those remain with the existing
template and polycycle renderers.
"""

from .molecule_iteration import iter_atoms, iter_bonds
from .molecule_cache import mol_cache_get, mol_cache_set
from .formatting import (
    complete_parent_name,
    format_prefixes,
    join_prefix_parent,
    multiplied_suffix,
)


def localize_acyclic_alkenyl_stereo(mol, root_idx, parent_idx, prefix):
    """Keep a two-carbon alkenyl branch's geometry with that component."""
    import re

    if not prefix.endswith(("ethenyl", "ethen-1-yl")):
        return prefix
    root = mol.GetAtomWithIdx(root_idx)
    if root.GetAtomicNum() != 6 or root.IsInRing():
        return prefix
    neighbors = [
        n
        for n in root.GetNeighbors()
        if n.GetIdx() != parent_idx and n.GetAtomicNum() > 1
    ]
    if (
        len(neighbors) != 1
        or neighbors[0].GetAtomicNum() != 6
        or neighbors[0].IsInRing()
    ):
        return prefix
    bond = mol.GetBondBetweenAtoms(root_idx, neighbors[0].GetIdx())
    if bond.GetBondTypeAsDouble() != 2:
        return prefix
    # In this proved two-carbon branch the enclosure already delimits
    # the substituent on the alkenyl stem (P-16.2.4).
    prefix = re.sub(r"([)\]\}])-(ethenyl|ethen-1-yl)$", r"\1\2", prefix)
    if re.search(r"\([^)]*[EZ][^)]*\)", prefix):
        return prefix
    descriptor = {
        "STEREOE": "E",
        "STEREOTRANS": "E",
        "STEREOZ": "Z",
        "STEREOCIS": "Z",
    }.get(str(bond.GetStereo()))
    return f"(1{descriptor})-{prefix}" if descriptor else prefix


def principal_carbonyl_sites(mol):
    """Return (suffix, (carbonyl, ring-owner) pairs) when no senior class competes."""
    from .functional_inventory import functional_inventory
    inventory = functional_inventory(mol)
    if inventory.supported and inventory.principal and inventory.principal[0].kind not in {
         "aldehyde", "ketone", "pseudoketone", "lactam", "lactone", "alcohol"
    }:
        return ("", ())
    cached = mol_cache_get(mol, "principal_carbonyl_sites")
    if cached is not None:
        return cached
    result = _principal_carbonyl_sites(mol)
    return mol_cache_set(mol, "principal_carbonyl_sites", result)


def supported_ring_carbonyl_selection(mol, suffix, sites):
    """Avoid evaluating relaxed heterocyclic templates as exact suffix parents."""
    if not suffix or not any(
        mol.GetAtomWithIdx(owner).IsInRing() for _, owner in sites
    ):
        return False
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    if suffix == "one":
        nitrogen = any(a.IsInRing() and a.GetAtomicNum() == 7 for a in iter_atoms(mol))
        if nitrogen:
            if len(sites) != 1:
                return False
            carbonyl, _ = sites[0]
            # Exact benzo-fused five-membered N lactams have a validated
            # oxindole prefix/parent H-state conversion. Other heterocyclic
            # tautomer families retain their established selectors.
            for first in rings:
                for second in rings:
                    core = first | second
                    if (
                        len(first) == 5
                        and len(second) == 6
                        and len(first & second) == 2
                        and carbonyl in first
                        and len(core) == 9
                        and all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in second)
                        and sum(mol.GetAtomWithIdx(i).GetAtomicNum() == 7 for i in core)
                        == 1
                    ):
                        return True
            return False
        if len(rings) == 1:
            return False  # Existing monocycle suffix/numbering grammar.
        if len(rings) == 2:
            from .polycyclic_rules import _simple_bicyclic_oxygen_suffix_facts

            if _simple_bicyclic_oxygen_suffix_facts(mol, frozenset().union(*rings)):
                return False  # Existing bridged/spiro suffix numbering.
    if (
        suffix == "carbaldehyde"
        and len(rings) == 1
        and any(int(a.GetChiralTag()) for a in iter_atoms(mol))
    ):
        return False
    return True


def _principal_carbonyl_sites(mol):
    if not mol.GetRingInfo().NumRings() or not any(
        a.GetAtomicNum() == 8 for a in iter_atoms(mol)
    ):
        return "", ()
    aldehydes, ketones = [], []
    for atom in iter_atoms(mol):
        if atom.GetAtomicNum() not in {1, 6, 7, 8, 9, 17, 35, 53}:
            return "", ()
        if atom.GetIsotope() or atom.GetNumRadicalElectrons():
            return "", ()
        if atom.GetFormalCharge() and not _is_isocyano_atom(mol, atom):
            return "", ()
        if atom.GetAtomicNum() != 6:
            continue
        # A neutral nitrile is senior to either aldehyde or ketone.
        if any(
            n.GetAtomicNum() == 7
            and not n.GetFormalCharge()
            and mol.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble()
            == 3
            for n in atom.GetNeighbors()
        ):
            return "", ()
        oxygens = [
            n
            for n in atom.GetNeighbors()
            if n.GetAtomicNum() == 8
            and mol.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble()
            == 2
        ]
        if not oxygens:
            continue
        if len(oxygens) != 1 or oxygens[0].GetDegree() != 1:
            return "", ()
        single_hetero = [
            n
            for n in atom.GetNeighbors()
            if n.GetAtomicNum() in {7, 8}
            and mol.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble()
            == 1
        ]
        # Noncyclic esters/acids, ordinary amides and hydrazides need their
        # functional-class routes. A ring N-linked carbonyl is a pseudoketone.
        if any(not n.IsInRing() for n in single_hetero):
            return "", ()
        if any(n.GetAtomicNum() == 8 and not atom.IsInRing() for n in single_hetero):
            return "", ()
        if atom.GetTotalNumHs(includeNeighbors=True):
            owners = [
                n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() in {6, 7}
            ]
            if len(owners) != 1:
                return "", ()
            aldehydes.append((atom.GetIdx(), owners[0]))
        else:
            ketones.append((atom.GetIdx(), atom.GetIdx()))
    if aldehydes:
        return "carbaldehyde", tuple(aldehydes)
    if ketones:
        return "one", tuple(ketones)
    return "", ()


def _is_isocyano_atom(mol, atom):
    charge, number = atom.GetFormalCharge(), atom.GetAtomicNum()
    if (number, charge) not in {(6, -1), (7, 1)}:
        return False
    return any(
        n.GetAtomicNum() == (7 if number == 6 else 6)
        and n.GetFormalCharge() == -charge
        and mol.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble()
        == 3
        for n in atom.GetNeighbors()
    )


def ring_carbonyl_suffix_name(mol, parent_atoms, atom_locants, prefixes, parent):
    """Promote literal oxo/formyl groups on saturated carbon parent positions.

    This is shared by retained carbon skeletons and generic polycycles. Mancude
    heterocyclic carbonyl grammar is deliberately left to the fused renderer.
    """
    if mol.HasProp("_nispo_principal_prefix_mode"):
        return ""
    suffix, sites = principal_carbonyl_sites(mol)
    if not suffix:
        return ""
    parent = complete_parent_name(parent)
    core = set(parent_atoms)
    if any(
        set(ring) & core and not set(ring) <= core
        for ring in mol.GetRingInfo().AtomRings()
    ):
        # A spiro component is still a parent-hydride component in an
        # assembled name. Its suffix must be applied to the complete assembly.
        return ""
    chosen = [(c, owner) for c, owner in sites if owner in core]
    if not chosen:
        return ""
    if suffix == "one" and (
        not all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in core)
        or not parent.endswith(("ane", "ene", "yne"))
    ):
        return ""
    # For aldehydes, every group must be accommodated. Side-chain aldehydes
    # are handled by the acyclic route instead of disappearing into a prefix.
    if suffix == "carbaldehyde" and len(chosen) != len(sites):
        return ""
    locants = [str(atom_locants.get(owner, "")) for _, owner in chosen]
    if not all(loc.isdigit() for loc in locants):
        return ""
    prefix = "oxo" if suffix == "one" else "formyl"
    removed = {(loc, prefix) for loc in locants}
    if not removed <= set(prefixes):
        return ""
    remaining = [item for item in prefixes if item not in removed]
    locants.sort(key=int)
    count = len(locants)
    if count > 6:
        return ""
    if suffix == "one":
        stem = parent.removesuffix("e") if count == 1 else parent
    else:
        stem = parent
    suffix_text = multiplied_suffix(count, suffix) if count > 1 else suffix
    core_name = f"{stem}-{','.join(locants)}-{suffix_text}"
    if suffix == "carbaldehyde" and parent == "benzene" and locants == ["1"]:
        core_name = "benzaldehyde"
    return join_prefix_parent(format_prefixes(remaining), core_name)


def principal_ring_carbonyl_key(mol, candidate):
    """Chemical ranking only for candidates which already express the suffix."""
    suffix, sites = principal_carbonyl_sites(mol)
    if (
        not suffix
        or not candidate.parent_atoms
        or not candidate.name.endswith(
            (suffix, "benzaldehyde") if suffix == "carbaldehyde" else suffix
        )
    ):
        return ()
    core = set(candidate.parent_atoms)
    covered = sum(owner in core for _, owner in sites)
    if not covered or (suffix == "carbaldehyde" and covered != len(sites)):
        return ()
    if (
        suffix == "one"
        and sum(not mol.GetAtomWithIdx(c).IsInRing() for c, _ in sites) > covered
    ):
        return ()
    rings = [set(r) for r in mol.GetRingInfo().AtomRings() if set(r) <= core]
    if not rings:
        return ()
    if (
        suffix == "carbaldehyde"
        and len(rings) == 1
        and any(
            int(a.GetChiralTag()) and a.GetIdx() not in core for a in iter_atoms(mol)
        )
    ):
        return ()  # Complex off-parent stereo still needs its specialized route.
    atoms = set().union(*rings)
    numbers = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in atoms]
    # Earlier element criteria tie within N/C skeletons; oxygen-bearing rings
    # stay with their established lactone and retained-composition routes.
    if any(n not in {6, 7} for n in numbers):
        return ()
    if (
        suffix == "one"
        and 7 in numbers
        and (len(sites) != 1 or not candidate.fused_lactam_seniority)
    ):
        return ()
    return (-covered, -int(7 in numbers), -len(rings), -len(atoms), -numbers.count(7))


def higher_carbopolycycle_prefix(mol, Chem, attachment_idx, parent_idx):
    """Render a larger carbon ring component in prefix mode under a senior suffix.

    Reuse the bounded polycycle topology/numbering route. Unspecified stereo
    stays unspecified; explicit stereochemistry retains the established branch
    renderer until it can be mapped through fragment atom/bond ordering.
    """
    if mol.HasProp("_nispo_principal_prefix_mode") or mol.GetRingInfo().NumRings() < 4:
        return ""
    from .branch_rules import collect_subtree
    from .branch.ring_substituents import (
        _fragment_from_subtree_atoms,
        _parent_name_to_substituent_prefix,
    )
    from .polycyclic_rules import kekulized_polycyclic_parent_candidate_records

    atoms = set(collect_subtree(mol, attachment_idx, parent_idx))
    rings = [set(r) for r in mol.GetRingInfo().AtomRings() if set(r) <= atoms]
    if not 4 <= len(rings) <= 6 or not 17 <= len(atoms) <= 60:
        return ""
    ring_atoms = set().union(*rings)
    if attachment_idx not in ring_atoms or any(
        mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in ring_atoms
    ):
        return ""
    if any(
        mol.GetAtomWithIdx(i).GetAtomicNum() not in {1, 6, 8, 9, 17, 35, 53}
        or mol.GetAtomWithIdx(i).GetFormalCharge()
        or mol.GetAtomWithIdx(i).GetIsotope()
        or mol.GetAtomWithIdx(i).GetNumRadicalElectrons()
        or int(mol.GetAtomWithIdx(i).GetChiralTag())
        for i in atoms
    ):
        return ""
    if any(
        int(b.GetStereo()) and {b.GetBeginAtomIdx(), b.GetEndAtomIdx()} <= atoms
        for b in iter_bonds(mol)
    ):
        return ""
    fragment = _fragment_from_subtree_atoms(mol, atoms)
    if fragment is None:
        return ""
    frag, indices = fragment
    frag.SetProp("_nispo_principal_prefix_mode", "1")
    records = kekulized_polycyclic_parent_candidate_records(frag, Chem, limit=1)
    for record in records:
        name, seq = record[:2]
        if indices[attachment_idx] in seq and not any(record[2:]):
            return _parent_name_to_substituent_prefix(
                name, str(seq.index(indices[attachment_idx]) + 1)
            )
    return ""


def _ring_carbonyl_comparison_sites(mol):
    """P-43 class 48: compare oxo with proved annular thioxo prefixes."""
    from .mapped_principal_general import senior_sites

    rebased = mol_cache_get(mol, 'source_ring_carbonyl_comparison_sites')
    if rebased is not None:
        return rebased
    cached = mol_cache_get(mol, 'ring_carbonyl_comparison_sites')
    if cached is not None:
        return cached
    ordinary = senior_sites(mol)
    if ordinary:
        return mol_cache_set(mol, 'ring_carbonyl_comparison_sites', ordinary)
    from .functional_inventory import _uncached_inventory

    rings = mol.GetRingInfo().AtomRings()
    oxo_centers = {
        atom.GetNeighbors()[0].GetIdx()
        for atom in iter_atoms(mol)
        if (atom.GetAtomicNum() == 8 and atom.GetDegree() == 1
            and not atom.GetFormalCharge() and atom.GetBonds()[0].GetBondTypeAsDouble() == 2
            and atom.GetNeighbors()[0].GetAtomicNum() == 6
            and atom.GetNeighbors()[0].IsInRing())
    }
    thioxo = frozenset(
        atom.GetIdx()
        for atom in iter_atoms(mol)
        if (atom.GetAtomicNum() == 16 and atom.GetDegree() == 1
            and not atom.GetFormalCharge() and not atom.GetIsotope()
            and atom.GetBonds()[0].GetBondTypeAsDouble() == 2
            and atom.GetNeighbors()[0].GetAtomicNum() == 6
            and any(atom.GetNeighbors()[0].GetIdx() in ring and oxo_centers.intersection(ring)
                    for ring in rings))
    )
    groups = ()
    if thioxo:
        inventory = _uncached_inventory(mol, extended=True, subordinate_thioxo=thioxo)
        if inventory.supported and inventory.source_frontier.status != 'barrier':
            groups = inventory.principal
    return mol_cache_set(mol, 'ring_carbonyl_comparison_sites', groups)


def record_cited_ring_suffix(mol, core, name, groups):
    """Bind a successful ring suffix operation to its original site IDs."""
    from .molecule_cache import mol_cache_dict

    centers = frozenset(group.center for group in groups)
    if centers:
        mol_cache_dict(mol, 'cited_ring_suffix_centers')[(frozenset(core), name)] = centers


def cited_ring_suffix_centers(mol, core, name):
    cache = mol_cache_get(mol, 'cited_ring_suffix_centers')
    return cache.get((frozenset(core), name), frozenset()) if cache else frozenset()


def configured_acyclic_methine_ring_hosts(mol):
    """Original ring endpoints of configured exocyclic methines."""
    cached = mol_cache_get(mol, 'configured_acyclic_methine_ring_hosts')
    if cached is not None:
        return cached
    hosts = set()
    for bond in iter_bonds(mol):
        if bond.GetBondTypeAsDouble() != 2 or not bond.GetStereo():
            continue
        ring_atom, methine = bond.GetBeginAtom(), bond.GetEndAtom()
        if not ring_atom.IsInRing():
            ring_atom, methine = methine, ring_atom
        if (ring_atom.IsInRing() and ring_atom.GetAtomicNum() == 6
                and not methine.IsInRing() and methine.GetAtomicNum() == 6
                and methine.GetDegree() == 2 and methine.GetTotalNumHs() == 1
                and all(n.GetAtomicNum() == 6 for n in methine.GetNeighbors())):
            hosts.add(ring_atom.GetIdx())
    return mol_cache_set(mol, 'configured_acyclic_methine_ring_hosts', frozenset(hosts))


def configured_methine_ring_junction(mol, core):
    """Configured exocyclic C=C joining this ring to an acyclic methine."""
    return not configured_acyclic_methine_ring_hosts(mol).isdisjoint(core)


def mapped_ring_oxo_suffix(mol, core, locants, prefixes, parent, *, allow_hetero=False):
    """Close the bounded fused C=O suffix operation on a proved ring map."""
    if not configured_methine_ring_junction(mol, core):
        return ''
    groups = _ring_carbonyl_comparison_sites(mol)
    if not groups or any(group.kind not in {'ketone', 'pseudoketone', 'lactam', 'lactone'}
                         or group.center not in core for group in groups):
        return ''
    if not allow_hetero and any(mol.GetAtomWithIdx(idx).GetAtomicNum() != 6 for idx in core):
        return ''
    if not all(mol.GetAtomWithIdx(group.center).GetAtomicNum() == 6
               and not mol.GetAtomWithIdx(group.center).GetIsAromatic()
               for group in groups):
        return ''
    positions = {str(locants[group.center]) for group in groups if group.center in locants}
    suffix_prefixes = {(position, 'oxo') for position in positions}
    if (len(positions) != len(groups) or not all(p.isdigit() for p in positions)
            or not suffix_prefixes <= set(prefixes)):
        return ''
    parent = complete_parent_name(parent)
    if not parent.endswith(('an', 'ane', 'en', 'ene', 'yne')):
        return ''
    kept = [item for item in prefixes if item not in suffix_prefixes]
    suffix = 'one' if len(positions) == 1 else multiplied_suffix(len(positions), 'one')
    if len(positions) == 1:
        parent = parent.removesuffix('e')
    elif parent.endswith(('an', 'en')):
        parent += 'e'
    name = join_prefix_parent(format_prefixes(kept),
                              f"{parent}-{','.join(sorted(positions,key=int))}-{suffix}")
    record_cited_ring_suffix(mol, core, name, groups)
    return name
