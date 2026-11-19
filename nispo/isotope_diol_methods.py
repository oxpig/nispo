"""P-82.2.2.2 suffix/prefix partition for a bounded acyclic diol family.

This producer selects between two unlike isotope-modified OH characteristic
groups only after retaining the ordinary all-OH parent numbering frontier.
The parent integration is handled by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

from .characteristic_isotope_selection import (
    CharacteristicGroupMap,
    select_characteristic_isotope_frontier,
)
from .constants import ROOTS
from .component_mapping import MappedComponent, hydrogen_bindings, decorate_component
from .naming_methods import (
    MethodDecision,
    NamingMethodOutcome,
    NamingMethodStatus,
)


RULE = "P-82.2.2.2/P-63.1.2/P-63.1.4/P-82.5.1/P-82.5.2"


@dataclass(frozen=True, slots=True)
class DiolHydroxySite:
    host: int
    oxygen: int
    isotope_hydrogens: tuple[int, ...]
    selector_map: CharacteristicGroupMap


@dataclass(frozen=True, slots=True)
class IsotopeDiolRecognition:
    carbon_atoms: tuple[int, ...]
    sites: tuple[DiolHydroxySite, DiolHydroxySite]
    heavy_atoms: frozenset[int]
    isotope_atoms: frozenset[int]


@dataclass(frozen=True, slots=True)
class IsotopeDiolDecision(MethodDecision):
    """Decision with site-role maps and every source-tied parent orientation."""

    isotope_selection: object | None = None
    suffix_site: int | None = None
    prefix_site: int | None = None
    hydroxy_prefix: object | None = None
    equivalent_atom_locant_maps: tuple[tuple[tuple[int, str], ...], ...] = ()
    role_maps: tuple[CharacteristicGroupMap, ...] = ()
    suffix_isotope_atom_ids: frozenset[int] = frozenset()
    parent_isotope_atom_ids: frozenset[int] = frozenset()
    parent_isotope_claims: tuple = ()


def recognize_unbranched_diol(mol) -> IsotopeDiolRecognition | None:
    """Recognize two ordinary OH sites on one saturated, unbranched C chain.

    Isotope labels are allowed only on parent C/H and on the two OH O/H
    units. Charged, radical, bridged, unsaturated, stereochemical, and other
    characteristic-group graphs are outside this producer.
    """
    try:
        if mol is None or mol.GetNumHeavyAtoms() > 64 or mol.GetStereoGroups():
            return None
        atoms = tuple(mol.GetAtoms())
        if any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in atoms):
            return None
        if any(int(a.GetChiralTag()) for a in atoms) or any(
            int(b.GetStereo()) or int(b.GetBondDir()) for b in mol.GetBonds()
        ):
            return None
        heavy = frozenset(a.GetIdx() for a in atoms if a.GetAtomicNum() > 1)
        carbons = tuple(a.GetIdx() for a in atoms if a.GetAtomicNum() == 6)
        oxygens = tuple(a.GetIdx() for a in atoms if a.GetAtomicNum() == 8)
        if (len(carbons) < 2 or len(carbons) not in ROOTS or len(oxygens) != 2
                or heavy != frozenset(carbons) | frozenset(oxygens)
                or any(a.GetAtomicNum() not in {1, 6, 8} for a in atoms)):
            return None

        carbon_set = set(carbons)
        adjacency = {i: [] for i in carbon_set}
        for bond in mol.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            aa, bb = mol.GetAtomWithIdx(a), mol.GetAtomWithIdx(b)
            if aa.GetAtomicNum() == 1 or bb.GetAtomicNum() == 1:
                if bond.GetBondTypeAsDouble() != 1.0:
                    return None
                continue
            if bond.GetBondTypeAsDouble() != 1.0:
                return None
            if aa.GetAtomicNum() == bb.GetAtomicNum() == 6:
                adjacency[a].append(b)
                adjacency[b].append(a)
            elif {aa.GetAtomicNum(), bb.GetAtomicNum()} == {6, 8}:
                oxygen_idx = a if aa.GetAtomicNum() == 8 else b
                carbon_idx = b if oxygen_idx == a else a
                oxygen = mol.GetAtomWithIdx(oxygen_idx)
                heavy_neighbors = [n.GetIdx() for n in oxygen.GetNeighbors()
                                   if n.GetAtomicNum() > 1]
                if (heavy_neighbors != [carbon_idx] or oxygen.GetFormalCharge()
                        or oxygen.GetNumRadicalElectrons() or oxygen.GetTotalValence() != 2
                        or oxygen.GetTotalNumHs(includeNeighbors=True) != 1):
                    return None
            else:
                return None

        if any(len(neighbors) > 2 for neighbors in adjacency.values()):
            return None
        endpoints = tuple(i for i, neighbors in adjacency.items() if len(neighbors) == 1)
        if len(endpoints) != 2 or any(not adjacency[i] for i in adjacency):
            return None
        ordered = [endpoints[0]]
        previous = None
        current = endpoints[0]
        while True:
            next_atoms = [i for i in adjacency[current] if i != previous]
            if not next_atoms:
                break
            if len(next_atoms) != 1 or next_atoms[0] in ordered:
                return None
            previous, current = current, next_atoms[0]
            ordered.append(current)
        if set(ordered) != carbon_set:
            return None

        sites = []
        for oxygen_idx in oxygens:
            oxygen = mol.GetAtomWithIdx(oxygen_idx)
            host_ids = [n.GetIdx() for n in oxygen.GetNeighbors()
                        if n.GetAtomicNum() == 6]
            if len(host_ids) != 1:
                return None
            host = host_ids[0]
            isotope_h = tuple(sorted(
                n.GetIdx() for n in oxygen.GetNeighbors()
                if n.GetAtomicNum() == 1 and n.GetIsotope()
            ))
            atoms_in_group = frozenset((oxygen_idx, *isotope_h))
            group_map = CharacteristicGroupMap(
                group_id=f"OH:{host}", atoms=atoms_in_group,
                roles=(("hydroxyl_oxygen", (oxygen_idx,)),
                       ("hydroxyl_hydrogen", isotope_h)),
            )
            sites.append(DiolHydroxySite(host, oxygen_idx, isotope_h, group_map))
        if len({site.host for site in sites}) != 2:
            return None

        isotope_atoms = frozenset(a.GetIdx() for a in atoms if a.GetIsotope())
        allowed_isotope_atoms = set(carbons)
        for site in sites:
            allowed_isotope_atoms.add(site.oxygen)
            allowed_isotope_atoms.update(site.isotope_hydrogens)
        allowed_isotope_atoms.update(
            a.GetIdx() for a in atoms
            if a.GetAtomicNum() == 1 and a.GetIsotope()
            and any(n.GetAtomicNum() == 6 for n in a.GetNeighbors())
        )
        for atom in atoms:
            if atom.GetAtomicNum() == 1 and atom.GetIsotope():
                neighbors = tuple(atom.GetNeighbors())
                if len(neighbors) != 1 or neighbors[0].GetIdx() not in carbon_set | {s.oxygen for s in sites}:
                    return None
            elif atom.GetAtomicNum() == 8 and atom.GetIsotope() and atom.GetIdx() not in allowed_isotope_atoms:
                return None
        if not isotope_atoms <= allowed_isotope_atoms:
            return None
        return IsotopeDiolRecognition(
            tuple(ordered), tuple(sites), heavy, isotope_atoms,
        )
    except (AttributeError, IndexError, TypeError, ValueError):
        return None


def _stereo_isotope_gate(mol, Chem):
    """Decline isotope-created stereogenic units until their rules are mapped."""
    if not any(a.GetIsotope() for a in mol.GetAtoms()):
        return True
    try:
        original = frozenset((str(x.type), int(x.centeredOn))
                             for x in Chem.FindPotentialStereo(mol))
        cleared = Chem.Mol(mol)
        for atom in cleared.GetAtoms():
            atom.SetIsotope(0)
        Chem.SanitizeMol(cleared)
        unmodified = frozenset((str(x.type), int(x.centeredOn))
                               for x in Chem.FindPotentialStereo(cleared))
        return original == unmodified
    except Exception:
        return False


def _profile(mol, site):
    labels = []
    oxygen = mol.GetAtomWithIdx(site.oxygen)
    if oxygen.GetIsotope():
        labels.append((8, int(oxygen.GetIsotope())))
    labels.extend((1, int(mol.GetAtomWithIdx(i).GetIsotope()))
                  for i in site.isotope_hydrogens)
    return tuple(sorted(labels))


def _ordinary_orientations(mol, recognition):
    path = recognition.carbon_atoms
    candidates = []
    for oriented in (path, tuple(reversed(path))):
        locants = {atom: str(i + 1) for i, atom in enumerate(oriented)}
        suffix_locants = tuple(sorted(int(locants[s.host]) for s in recognition.sites))
        candidates.append((suffix_locants, locants, oriented))
    lowest = min(row[0] for row in candidates)
    return tuple(row for row in candidates if row[0] == lowest)


def _isotope_numbering_key(mol, recognition, locants):
    """P-82.5.2 key over mapped modified groups and parent C/H atoms."""
    modified = []
    for atom in mol.GetAtoms():
        mass = int(atom.GetIsotope())
        if not mass:
            continue
        z = int(atom.GetAtomicNum())
        if z == 6:
            modified.append((int(locants[atom.GetIdx()]), -z, -mass))
        elif z == 1:
            host = next((n.GetIdx() for n in atom.GetNeighbors()
                         if n.GetAtomicNum() == 6), None)
            if host is not None:
                modified.append((int(locants[host]), -z, -mass))
            else:
                oxygen = next((n.GetIdx() for n in atom.GetNeighbors()
                               if n.GetAtomicNum() == 8), None)
                site = next((s for s in recognition.sites if s.oxygen == oxygen), None)
                if site is None:
                    return None
                modified.append((int(locants[site.host]), -z, -mass))
        elif z == 8:
            site = next((s for s in recognition.sites if s.oxygen == atom.GetIdx()), None)
            if site is None:
                return None
            modified.append((int(locants[site.host]), -z, -mass))
        else:
            return None
    return tuple(sorted(modified))


def isotope_diol_outcome(mol, Chem) -> NamingMethodOutcome:
    """Return a source-status result for the unlike-isotope diol operation."""
    recognition = recognize_unbranched_diol(mol)
    if recognition is None:
        return NamingMethodOutcome(NamingMethodStatus.OUTSIDE_SCOPE,
                                   reason="not-bounded-unbranched-diol")
    if not _stereo_isotope_gate(mol, Chem):
        return NamingMethodOutcome(
            NamingMethodStatus.OUTSIDE_SCOPE, rule="P-82.4/P-82.5.1",
            method="isotope-diol-method", reason="isotope-created-stereo-outside-scope",
        )

    sites = tuple(sorted(recognition.sites,
                         key=lambda site: (recognition.carbon_atoms.index(site.host), site.host)))
    profiles = tuple(_profile(mol, site) for site in sites)
    if not any(profiles) or profiles[0] == profiles[1]:
        return NamingMethodOutcome(NamingMethodStatus.OUTSIDE_SCOPE,
                                   reason="no-unlike-isotope-profiles")
    selected = select_characteristic_isotope_frontier(
        mol, tuple(site.selector_map for site in sites),
        frozenset().union(*(site.selector_map.atoms for site in sites)),
    )
    if selected is None:
        return NamingMethodOutcome(NamingMethodStatus.PENDING, rule=RULE,
            method="isotope-diol-method", reason="invalid-or-incomplete-OH-role-map")
    if selected.barrier or len(selected.principal_frontier) != 1:
        return NamingMethodOutcome(NamingMethodStatus.PENDING, rule=RULE,
            method="isotope-diol-method", reason="unresolved-isotope-profile-frontier",
            evidence=(("criterion", selected.criterion),
                      ("counts", selected.modified_counts)))
    cluster = selected.principal_frontier[0]
    winning_group_indices = selected.multiplicity_groups[cluster]
    # The bounded unequal-profile scope has exactly one winning group. A
    # multi-unit winning cluster would require a separate multiplicative
    # isotope prefix/suffix proof.
    if len(winning_group_indices) != 1:
        return NamingMethodOutcome(NamingMethodStatus.PENDING, rule=RULE,
            method="isotope-diol-method", reason="winning-profile-has-multiplicity")
    winning_map = sites[winning_group_indices[0]].selector_map.group_id
    winner = next(site for site in sites if site.selector_map.group_id == winning_map)
    loser = next(site for site in sites if site is not winner)

    orientations = _ordinary_orientations(mol, recognition)
    isotope_ranked = []
    for _ordinary_suffix_key, locants, oriented in orientations:
        key = _isotope_numbering_key(mol, recognition, locants)
        if key is None:
            return NamingMethodOutcome(NamingMethodStatus.PENDING, rule=RULE,
                method="isotope-diol-method", reason="incomplete-P82-numbering-map")
        isotope_ranked.append((key, locants, oriented))
    best_key = min(row[0] for row in isotope_ranked)
    finalist_maps = tuple(row[1] for row in isotope_ranked if row[0] == best_key)
    suffix_locants = {mapping[winner.host] for mapping in finalist_maps}
    prefix_locants = {mapping[loser.host] for mapping in finalist_maps}
    if len(suffix_locants) != 1 or len(prefix_locants) != 1:
        return NamingMethodOutcome(NamingMethodStatus.PENDING, rule=RULE,
            method="isotope-diol-method", reason="numbering-frontier-does-not-fix-site-locants")

    # Rendering is assembled from source-proven components. The parent C/H
    # descriptor is delegated to the common original-map writer.
    from types import SimpleNamespace
    from .oxygen_characteristic_suffixes import oxygen_ending
    from .polyol_hydroxy_isotopes import polyol_hydroxy_isotope_prefix

    suffix_locant = next(iter(suffix_locants))
    prefix_locant = next(iter(prefix_locants))
    parent = ROOTS[len(recognition.carbon_atoms)] + "ane"
    suffix_group = SimpleNamespace(kind="alcohol", center=winner.host)
    ending, suffix_owned = oxygen_ending(
        mol, Chem, parent, (suffix_group,), (suffix_locant,), suffix="ol",
    )
    decoration = polyol_hydroxy_isotope_prefix(
        mol, Chem, host_atom_id=loser.host, oxygen_atom_id=loser.oxygen,
        host_locant=prefix_locant, component_id=f"diol-hydroxy-{prefix_locant}",
    )
    if not ending or decoration is None:
        return NamingMethodOutcome(NamingMethodStatus.PENDING, rule=RULE,
            method="isotope-diol-method", reason="typed-suffix-or-prefix-renderer-declined")

    prefix_rows = ((prefix_locant, decoration.text),)
    carbon_map = tuple((idx, loc) for idx, loc in
                       sorted(next(row[1] for row in isotope_ranked if row[0] == best_key).items(),
                              key=lambda pair: int(pair[1])))
    # The OH suffix O is locally represented in ``ending``; the hydroxy
    # prefix record owns its O/H separately. Only parent atoms and suffix O
    # enter this isotope-decorated component.
    component_atoms = frozenset((*recognition.carbon_atoms, winner.oxygen))
    carbon_bindings = dict(carbon_map)
    parent_hydrogens = {}
    for carbon in recognition.carbon_atoms:
        for neighbor in mol.GetAtomWithIdx(carbon).GetNeighbors():
            if neighbor.GetAtomicNum() == 1 and neighbor.GetIsotope():
                parent_hydrogens[neighbor.GetIdx()] = carbon_bindings[carbon]
    component_locants = tuple((*carbon_map, (winner.oxygen, "")))
    suffix_h = tuple((i, "") for i in winner.isotope_hydrogens)
    component_locants = tuple((*component_locants, *suffix_h,
                               *parent_hydrogens.items()))
    component_atoms = (component_atoms | frozenset(winner.isotope_hydrogens)
                       | frozenset(parent_hydrogens))
    main = MappedComponent(
        parent=ending, atom_locants=component_locants, atoms=component_atoms,
        prefixes=prefix_rows, embedded_atoms=decoration.component_atom_ids,
        systematic_parent=True, local_isotope_atoms=frozenset(suffix_owned),
    )
    name = decorate_component(mol, Chem, main)
    if not name:
        return NamingMethodOutcome(NamingMethodStatus.PENDING, rule=RULE,
            method="isotope-diol-method", reason="mapped-parent-decoration-declined")
    from .component_mapping import _mapped_isotope_descriptor_plan
    parent_plan = _mapped_isotope_descriptor_plan(mol, Chem, main, dict(main.atom_locants))
    if parent_plan is None:
        return NamingMethodOutcome(NamingMethodStatus.PENDING, rule=RULE,
            method="isotope-diol-method", reason="parent-isotope-role-map-declined")
    _parent_tokens, parent_claims = parent_plan
    parent_isotope_atom_ids = frozenset(row[0] for row in parent_claims)

    # Retain every map tied by ordinary and isotope criteria. Their rendered
    # output must be the same; atom-index order cannot select the PIN.
    rendered_by_map = set()
    for map_row in finalist_maps:
        mapped_suffix, owned = oxygen_ending(
            mol, Chem, parent, (suffix_group,), (map_row[winner.host],), suffix="ol",
        )
        mapped_prefix = polyol_hydroxy_isotope_prefix(
            mol, Chem, host_atom_id=loser.host, oxygen_atom_id=loser.oxygen,
            host_locant=map_row[loser.host],
            component_id=f"diol-hydroxy-{map_row[loser.host]}",
        )
        if not mapped_suffix or mapped_prefix is None:
            return NamingMethodOutcome(NamingMethodStatus.PENDING, rule=RULE,
                method="isotope-diol-method", reason="tied-numbering-render-declined")
        row_main = MappedComponent(
            parent=mapped_suffix,
            atom_locants=tuple((idx, map_row[idx]) for idx in recognition.carbon_atoms)
                         + ((winner.oxygen, ""),)
                         + tuple((i, "") for i in winner.isotope_hydrogens)
                         + tuple((h.GetIdx(), map_row[host.GetIdx()])
                                 for host in (mol.GetAtomWithIdx(c) for c in recognition.carbon_atoms)
                                 for h in host.GetNeighbors()
                                 if h.GetAtomicNum() == 1 and h.GetIsotope()),
            atoms=frozenset((*recognition.carbon_atoms, winner.oxygen,
                             *winner.isotope_hydrogens,
                             *(h.GetIdx() for c in recognition.carbon_atoms
                               for h in mol.GetAtomWithIdx(c).GetNeighbors()
                               if h.GetAtomicNum() == 1 and h.GetIsotope()))),
            prefixes=((map_row[loser.host], mapped_prefix.text),),
            embedded_atoms=mapped_prefix.component_atom_ids,
            systematic_parent=True, local_isotope_atoms=frozenset(owned),
        )
        rendered = decorate_component(mol, Chem, row_main)
        if not rendered:
            return NamingMethodOutcome(NamingMethodStatus.PENDING, rule=RULE,
                method="tied-numbering-decoration-declined")
        rendered_by_map.add(rendered)
    if len(rendered_by_map) != 1 or name not in rendered_by_map:
        return NamingMethodOutcome(NamingMethodStatus.PENDING, rule=RULE,
            method="isotope-diol-method", reason="source-tied-maps-render-differently")

    chosen_map = next(row[1] for row in isotope_ranked if row[0] == best_key)
    atom_locants = tuple((idx, chosen_map[idx]) for idx in recognition.carbon_atoms)
    atom_locants += tuple((h.GetIdx(), chosen_map[host.GetIdx()])
                          for host in (mol.GetAtomWithIdx(c) for c in recognition.carbon_atoms)
                          for h in host.GetNeighbors()
                          if h.GetAtomicNum() == 1 and h.GetIsotope())
    atom_locants += ((winner.oxygen, ""),) + tuple(
        (idx, chosen_map[loser.host] if idx in loser.isotope_hydrogens else "")
        for site in (winner, loser) for idx in site.isotope_hydrogens
    )
    roles = (
        ("diol-carbon-parent", frozenset(recognition.carbon_atoms),
         tuple((idx, chosen_map[idx]) for idx in recognition.carbon_atoms)),
        ("selected-hydroxy-suffix", frozenset({winner.oxygen}), ((winner.oxygen, ""),)),
        ("other-hydroxy-prefix", frozenset({loser.oxygen}), ((loser.oxygen, prefix_locant),)),
    )
    decision = IsotopeDiolDecision(
        name=name, method="substitutive", principal_centers=frozenset({winner.host}),
        atoms=recognition.heavy_atoms, rule=RULE, atom_locants=atom_locants,
        component_roles=roles, isotope_selection=selected,
        suffix_site=winner.host, prefix_site=loser.host,
        hydroxy_prefix=decoration,
        equivalent_atom_locant_maps=tuple(
            tuple(sorted(mapping.items(), key=lambda row: row[1]))
            for mapping in finalist_maps
        ), role_maps=tuple(site.selector_map for site in sites),
        suffix_isotope_atom_ids=frozenset(suffix_owned),
        parent_isotope_atom_ids=parent_isotope_atom_ids,
        parent_isotope_claims=parent_claims,
    )
    return NamingMethodOutcome(NamingMethodStatus.COMPLETE, decision,
        rule=RULE, method="isotope-diol-method", reason="rendered")
