"""Typed, bounded hydroxylamine substitution methods (Blue Book P-68.3.1.1.1).

This module recognizes neutral hydroxylamine connectivity separately from
method admission. It proposes mapped acyclic carbon amine parents for the
N-hydroxy and N-alkoxy methods; O-substituted hydroxylamine is typed as its
functional-parent method for a separate caller.
"""

from dataclasses import dataclass

from .molecule_iteration import iter_atoms

RULE = "P-41/P-43/P-44/P-62/P-68.3.1.1.1.1-.3/P-16.9/P-82"


@dataclass(frozen=True, slots=True)
class HydroxylamineSite:
    nitrogen: int
    oxygen: int
    carbon_substituents: tuple[int, ...]
    oxygen_substituent: int | None
    kind: str  # "N-hydroxy", "O-hydrocarbyl", or "N-alkoxy"


@dataclass(frozen=True, slots=True)
class MethodDecision:
    accepted: bool
    method: str
    rule: str
    site: HydroxylamineSite | None = None
    reason: str = ""


def hydroxylamine_sites(mol):
    """Return only neutral, acyclic, valence-closed N–O hydroxylamine sites."""
    found = []
    for n in iter_atoms(mol):
        if n.GetAtomicNum() != 7 or n.IsInRing() or n.GetFormalCharge() or n.GetNumRadicalElectrons():
            continue
        if any(b.GetBondTypeAsDouble() != 1 for b in n.GetBonds()) or n.GetTotalValence() != 3:
            continue
        heavy = [a for a in n.GetNeighbors() if a.GetAtomicNum() > 1]
        oxygens = [a for a in heavy if a.GetAtomicNum() == 8]
        carbons = tuple(a.GetIdx() for a in heavy if a.GetAtomicNum() == 6)
        if len(oxygens) != 1 or len(carbons) > 2 or any(a.GetAtomicNum() not in {6, 8} for a in heavy):
            continue
        o = oxygens[0]
        if o.IsInRing() or o.GetFormalCharge() or o.GetNumRadicalElectrons() or o.GetIsAromatic():
            continue
        if any(b.GetBondTypeAsDouble() != 1 for b in o.GetBonds()) or o.GetTotalValence() != 2:
            continue
        oheavy = [a for a in o.GetNeighbors() if a.GetAtomicNum() > 1]
        if len(oheavy) == 1 and o.GetTotalNumHs(includeNeighbors=True) == 1 and carbons:
            site = HydroxylamineSite(n.GetIdx(), o.GetIdx(), carbons, None, "N-hydroxy")
        elif (len(oheavy) == 1 and o.GetTotalNumHs(includeNeighbors=True) == 1
              and not carbons and n.GetTotalNumHs(includeNeighbors=True) == 2):
            site = HydroxylamineSite(n.GetIdx(), o.GetIdx(), (), None, "hydroxylamine")
        elif len(oheavy) == 2 and o.GetTotalNumHs(includeNeighbors=True) == 0:
            substituents = [a.GetIdx() for a in oheavy if a.GetIdx() != n.GetIdx()]
            if len(substituents) != 1 or mol.GetAtomWithIdx(substituents[0]).GetAtomicNum() != 6:
                continue
            if carbons:
                site = HydroxylamineSite(n.GetIdx(), o.GetIdx(), carbons, substituents[0], "N-alkoxy")
            elif n.GetTotalNumHs(includeNeighbors=True) == 2:
                site = HydroxylamineSite(n.GetIdx(), o.GetIdx(), (), substituents[0], "O-hydrocarbyl")
            else:
                continue
        else:
            continue
        # This producer needs an acyclic carbon branch that the generic mapped
        # branch grammar can render. Recognition remains useful when admission
        # declines a ring/complex branch.
        found.append(site)
    return tuple(found)


def recognize(mol, nitrogen=None):
    sites = tuple(s for s in hydroxylamine_sites(mol)
                  if nitrogen is None or s.nitrogen == nitrogen)
    return sites[0] if len(sites) == 1 else None


def aromatic_amine_decision(mol, Chem):
    """P-68.3.1.1.1.3 on a source-closed substituted aniline parent.

    The ring may have methyl or halogen substituents and nitrogen may bear a
    second methyl group. The oxygen-side organyl contains only carbon; this
    proves no competing senior characteristic group is hidden in the name.
    """
    if (mol.GetNumHeavyAtoms() > 64 or mol.GetStereoGroups()
            or len(Chem.GetMolFrags(mol)) != 1 or mol.GetRingInfo().NumRings() not in {1, 2}
            or any(a.GetAtomicNum() not in {6, 7, 8, 9, 17, 35, 53}
                   or a.GetFormalCharge() or a.GetIsotope() or a.GetNumRadicalElectrons()
                   or int(a.GetChiralTag()) or a.HasQuery() for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) or b.HasQuery()
                   for b in mol.GetBonds())):
        return None
    sites = hydroxylamine_sites(mol)
    if (len(sites) != 1 or sites[0].kind != 'N-alkoxy'
            or not amine_derivative_eligible(mol, sites[0])):
        return None
    site = sites[0]
    rings = [frozenset(row) for row in mol.GetRingInfo().AtomRings()
             if len(row) == 6 and all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6
                                      and mol.GetAtomWithIdx(i).GetIsAromatic() for i in row)
             and sum(i in row for i in site.carbon_substituents) == 1]
    if len(rings) != 1:
        return None
    ring = rings[0]
    anchors = [i for i in site.carbon_substituents if i in ring]
    if len(anchors) != 1:
        return None
    anchor = anchors[0]
    from .component_mapping import mapped_branch
    from .formatting import format_prefixes, join_prefix_parent, prefix_alphabetization_key
    from .naming_methods import MethodDecision as NamingDecision

    oxygen_text, oxygen_atoms = mapped_branch(mol, Chem, site.oxygen, site.nitrogen)
    if (not oxygen_text or not oxygen_atoms or site.oxygen not in oxygen_atoms
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in oxygen_atoms if i != site.oxygen)):
        return None
    seen = set(ring) | {site.nitrogen} | set(oxygen_atoms)
    nitrogen_prefixes = [('N', oxygen_text)]
    for root in site.carbon_substituents:
        if root == anchor:
            continue
        atom = mol.GetAtomWithIdx(root)
        if atom.GetDegree() != 1 or atom.GetTotalNumHs(includeNeighbors=True) != 3 or root in seen:
            return None
        nitrogen_prefixes.append(('N', 'methyl'))
        seen.add(root)
    halogens = {9: 'fluoro', 17: 'chloro', 35: 'bromo', 53: 'iodo'}
    branches = []
    for host in ring:
        for atom in mol.GetAtomWithIdx(host).GetNeighbors():
            root = atom.GetIdx()
            if root in ring or root == site.nitrogen:
                continue
            if root in seen or atom.GetDegree() != 1:
                return None
            if atom.GetAtomicNum() in halogens:
                term = halogens[atom.GetAtomicNum()]
            elif atom.GetAtomicNum() == 6 and atom.GetTotalNumHs(includeNeighbors=True) == 3:
                term = 'methyl'
            else:
                return None
            branches.append((host, root, term))
            seen.add(root)
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms())
    if frozenset(seen) != heavy:
        return None
    neighbors = {i: [n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in ring]
                 for i in ring}
    if any(len(row) != 2 for row in neighbors.values()):
        return None
    rows = []
    for first in neighbors[anchor]:
        seq, prev, current = [anchor], anchor, first
        while current != anchor and len(seq) <= len(ring):
            seq.append(current)
            remaining = [i for i in neighbors[current] if i != prev]
            if len(remaining) != 1:
                break
            prev, current = current, remaining[0]
        if current != anchor or len(seq) != len(ring) or frozenset(seq) != ring:
            continue
        mapping = {idx: str(i) for i, idx in enumerate(seq, 1)}
        ring_prefixes = [(mapping[host], text) for host, _root, text in branches]
        prefixes = nitrogen_prefixes + ring_prefixes
        text_order = sorted({text for _loc, text in ring_prefixes}, key=prefix_alphabetization_key)
        score = (tuple(sorted(int(loc) for loc, _text in ring_prefixes)),
                 tuple(tuple(sorted(int(loc) for loc, value in ring_prefixes if value == text))
                       for text in text_order))
        name = join_prefix_parent(format_prefixes(prefixes), 'aniline')
        rows.append((score, name, mapping))
    if not rows:
        return None
    best = min(row[0] for row in rows)
    winners = [row for row in rows if row[0] == best]
    if len({row[1] for row in winners}) != 1:
        return None
    _score, name, mapping = winners[0]
    atom_mapping = tuple(mapping.items()) + ((site.nitrogen, 'N'),)
    roles = [('aniline-parent', ring | {site.nitrogen}, atom_mapping),
             ('oxygen-substituent', frozenset(oxygen_atoms), ())]
    used = set(ring) | {site.nitrogen} | set(oxygen_atoms)
    roles.extend(('substituent', frozenset((i,)), ()) for i in heavy - used)
    return NamingDecision(name, 'substitutive', frozenset({site.nitrogen}), heavy,
                          'P-68.3.1.1.1.3/P-41/P-62.4', atom_mapping,
                          component_roles=tuple(roles))


def method_decision(mol, site=None):
    site = recognize(mol) if site is None else site
    if site is None:
        return MethodDecision(False, "unsupported", RULE, reason="no unique neutral hydroxylamine site")
    if site.kind in {"O-hydrocarbyl", "hydroxylamine"}:
        return MethodDecision(True, "functional-parent", RULE, site,
                              "P-68 functional parent; no amine-chain proposal is supplied here")
    if not amine_derivative_eligible(mol, site):
        return MethodDecision(False, "redirect", RULE, site,
                              "N-acyl or N-sulfonyl higher-function derivative; use its amide method")
    return MethodDecision(True, "amine-derivative", RULE, site)


def amine_derivative_eligible(mol, site):
    """Keep N-hydroxy amides/sulfonamides outside the amine suffix route."""
    n = mol.GetAtomWithIdx(site.nitrogen)
    for atom in n.GetNeighbors():
        if atom.GetIdx() == site.oxygen:
            continue
        z = atom.GetAtomicNum()
        if z == 6 and any(
            b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(atom).GetAtomicNum() == 8
            for b in atom.GetBonds()
        ):
            return False
        if z == 16 and sum(
            b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(atom).GetAtomicNum() == 8
            for b in atom.GetBonds()
        ) >= 1:
            return False
    return True


def _core_isotope_unsupported(mol, site):
    core = {site.nitrogen, site.oxygen}
    return any(
        mol.GetAtomWithIdx(i).GetIsotope()
        or any(n.GetAtomicNum() == 1 and n.GetIsotope()
               for n in mol.GetAtomWithIdx(i).GetNeighbors())
        for i in core
    )


def principal_scope(mol, groups):
    """Admit only recognized N-substituted hydroxylamines at amine-class sites."""
    groups = tuple(groups)
    if len(groups) != 1 or any(g.kind != "amine" for g in groups):
        return False
    sites = hydroxylamine_sites(mol)
    if len(sites) != len(groups) or {s.nitrogen for s in sites} != {g.center for g in groups}:
        return False
    return all(s.kind in {"N-hydroxy", "N-alkoxy"} and s.carbon_substituents
               and amine_derivative_eligible(mol, s) and not _core_isotope_unsupported(mol, s)
               for s in sites)


def functional_parent_component(mol, Chem, site):
    """Render the bounded O-hydrocarbyl functional-parent construction.

    Returns ``(name, MappedComponent)`` for a simple mapped O-carbon branch;
    the caller still owns P-41 method competition. O-acyl and complex/ring
    branches are deliberately declined unless the generic branch renderer can
    prove their complete spelling and atom ownership.
    """
    if site.kind not in {"O-hydrocarbyl", "hydroxylamine"}:
        return "", None
    from .component_mapping import MappedComponent, decorate_component, hydrogen_bindings, mapped_branch
    from .cross_route_selection import _annotation_ownership

    prefixes, owned = (), frozenset()
    if site.kind == "O-hydrocarbyl":
        if site.oxygen_substituent is None:
            return "", None
        branch, owned = mapped_branch(mol, Chem, site.oxygen_substituent, site.oxygen)
        if not branch or not owned or site.nitrogen in owned or site.oxygen in owned:
            return "", None
        prefixes = (("O", branch),)
    if _core_isotope_unsupported(mol, site):
        return "", None
    mapping = hydrogen_bindings(mol, {site.nitrogen: "", site.oxygen: ""})
    needed = {a.GetIdx() for a in iter_atoms(mol)
              if a.GetAtomicNum() > 1 or a.GetIsotope() or int(a.GetChiralTag())}
    if set(mapping) | owned != needed or not _annotation_ownership(mol, mapping):
        return "", None
    construction = MappedComponent(
        "hydroxylamine", tuple(mapping.items()), frozenset(mapping),
        prefixes, frozenset(owned),
    )
    name = decorate_component(mol, Chem, construction)
    return (name, construction) if name else ("", None)


def hydroxylamine_decision(mol, Chem):
    """Return a complete class-21 decision for the bounded O-only family.

    N-carbon-substituted sites belong to the class-19 mapped amine route and
    are left to :func:`chain_proposals`. This whole-molecule wrapper accepts
    only H2N–OH and H2N–O–organyl with exactly one N and O and a C/H-only
    organyl branch. The caller's generic method selector can compare the
    returned class-21 decision against other naming methods.
    """
    sites = hydroxylamine_sites(mol)
    if len(sites) != 1 or sites[0].kind not in {"hydroxylamine", "O-hydrocarbyl"}:
        return None
    site = sites[0]
    atoms = tuple(iter_atoms(mol))
    if (any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in atoms)
            or mol.GetStereoGroups()
            or any(a.GetAtomicNum() not in {1, 6, 7, 8} for a in atoms)
            or sum(a.GetAtomicNum() == 7 for a in atoms) != 1
            or sum(a.GetAtomicNum() == 8 for a in atoms) != 1
            or any(a.GetAtomicNum() == 1 and a.GetIsotope() for a in atoms)):
        return None
    arm = frozenset()
    if site.kind == "O-hydrocarbyl":
        from .branch_rules import collect_subtree
        arm = frozenset(collect_subtree(mol, site.oxygen_substituent, site.oxygen))
        if (not arm or any(mol.GetAtomWithIdx(i).GetAtomicNum() not in {1, 6} for i in arm)
                or any(mol.GetAtomWithIdx(i).IsInRing()
                       and mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in arm)):
            return None
        # The designated O-organyl arm must account for every carbon; this
        # excludes additional functional groups, hidden N/O, and assemblies.
        if {a.GetIdx() for a in atoms if a.GetAtomicNum() == 6} != {
            i for i in arm if mol.GetAtomWithIdx(i).GetAtomicNum() == 6
        }:
            return None
    elif any(a.GetAtomicNum() == 6 for a in atoms):
        return None
    name, construction = functional_parent_component(mol, Chem, site)
    if not name or construction is None:
        return None
    heavy = frozenset(a.GetIdx() for a in atoms if a.GetAtomicNum() > 1)
    arm_heavy = frozenset(i for i in arm if mol.GetAtomWithIdx(i).GetAtomicNum() > 1)
    owned = construction.atoms | construction.embedded_atoms
    if owned != frozenset(a.GetIdx() for a in atoms if a.GetAtomicNum() > 1
                          or a.GetIsotope() or int(a.GetChiralTag())):
        return None
    from .naming_methods import MethodDecision as NamingMethodDecision
    return NamingMethodDecision(
        name=name,
        method="substitutive",
        principal_centers=frozenset({site.nitrogen}),
        atoms=heavy,
        rule="P-41 class 21/P-68.3.1.1.1.1-.2/P-82",
        atom_locants=construction.atom_locants,
        component_roles=(("hydroxylamine-core", frozenset({site.nitrogen, site.oxygen}), ()),
                         ("O-organyl", arm_heavy, ())),
    )


def chain_proposals(mol, Chem, inventory, *, scope=None, principal=None):
    """Yield mapped chain proposals for supported P-68 N-hydroxy/N-alkoxy amines.

    The caller must inventory each N–O nitrogen as a class-19 ``amine`` site
    and pass the P-41 principal group set. This route itself does not rank
    hydroxylamine against other characteristic groups.
    """
    groups = tuple(inventory.principal if principal is None else principal)
    if (not groups or not inventory.principal or inventory.principal[0].kind != "amine"
            or not principal_scope(mol, groups)):
        return
    sites = {s.nitrogen: s for s in hydroxylamine_sites(mol)}
    from .multifunctional_parent import _paths
    from .component_mapping import MappedComponent, decorate_component, hydrogen_bindings, mapped_branch
    from .cross_route_selection import ParentProposal, _annotation_ownership
    from .branch_rules import parent_chain_name
    from .formatting import complete_parent_name, attach_suffix
    from .parent_selection import graph_parent_facts

    carbon = {a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() == 6
              and not a.IsInRing() and (scope is None or a.GetIdx() in scope)}
    for path in _paths(mol, carbon):
        for seq in (path, tuple(reversed(path))):
            chosen = tuple(g for g in groups if any(
                n in seq for n in sites[g.center].carbon_substituents))
            if len(chosen) != len(groups):
                continue
            # One acyclic host is the principal amine parent attachment. Other
            # N-carbon groups are ordinary N-substituents and stay mapped.
            if any(sum(n in seq for n in sites[g.center].carbon_substituents) != 1 for g in chosen):
                continue
            mapping = {i: str(j) for j, i in enumerate(seq, 1)}
            nmap = {}
            for g in chosen:
                site = sites[g.center]
                nmap[site.nitrogen] = "N" if len(groups) == 1 else "N" + mapping[next(
                    n for n in site.carbon_substituents if n in seq)]
            mapping.update(nmap)
            prefixes, embedded, failed = [], set(), False
            for idx, locant in tuple(mapping.items()):
                for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
                    root = neighbor.GetIdx()
                    if root in mapping or neighbor.GetAtomicNum() <= 1 or (scope is not None and root not in scope):
                        continue
                    if mol.GetBondBetweenAtoms(idx, root).GetBondTypeAsDouble() != 1:
                        failed = True
                        break
                    text, owned = mapped_branch(mol, Chem, root, idx)
                    if not text or owned & (mapping.keys() | embedded) or any(c.isspace() for c in text):
                        failed = True
                        break
                    prefixes.append((locant, text))
                    embedded.update(owned)
                if failed:
                    break
            mapping = hydrogen_bindings(mol, mapping)
            needed = {a.GetIdx() for a in iter_atoms(mol)
                      if a.GetAtomicNum() > 1 or a.GetIsotope() or int(a.GetChiralTag())}
            target = needed if scope is None else needed & set(scope)
            if failed or set(mapping) | embedded != target or not _annotation_ownership(mol, mapping):
                continue
            parent = complete_parent_name(parent_chain_name(mol, seq))
            if not parent:
                continue
            host_locs = tuple(sorted((mapping[next(n for n in sites[g.center].carbon_substituents if n in seq)]
                                      for g in chosen), key=int))
            carbon_prefixes = any(loc.isdigit() for loc, _text in prefixes)
            carbon_isotopes = {mol.GetAtomWithIdx(i).GetIsotope() for i in seq
                               if mol.GetAtomWithIdx(i).GetAtomicNum() == 6}
            mixed_carbon_labels = len(carbon_isotopes) > 1 and any(carbon_isotopes)
            omit = (len(chosen) == 1 and len(seq) <= 2
                    and not carbon_prefixes and not mixed_carbon_labels)
            suffix_locs = () if omit else host_locs
            ending = attach_suffix(parent, "amine", suffix_locs)
            if not ending:
                continue
            construction = MappedComponent(
                ending, tuple(mapping.items()), frozenset(mapping), tuple(prefixes),
                frozenset(embedded), systematic_parent=True,
                nitrogen_site_locants=True,
                isotope_unlocanted_elements=(7,),
            )
            name = decorate_component(mol, Chem, construction)
            if not name or len(name) > 260:
                continue
            facts = graph_parent_facts(
                mol, seq, mapping, principal_class=groups[0].seniority,
                suffix_centers=[g.center for g in chosen], suffix_locants=host_locs,
                prefixes=prefixes, kind="chain", name=name, allow_annotations=True,
            )
            if facts is not None:
                yield ParentProposal(
                    name, "mapped-chain-hydroxylamine", groups[0].kind,
                    frozenset(g.center for g in chosen), tuple(seq),
                    tuple(mapping.items()), facts, True, construction=construction,
                )
