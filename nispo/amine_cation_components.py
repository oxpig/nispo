"""Mapped acyclic aminium cation components under P-62.6.1 and P-77.1.2.

This producer names one tetravalent, nonannular nitrogen cation from a
saturated acyclic carbon framework, including a bounded terminal-primary-
polyamine monosalt family. It returns the cation component only; counterion
naming and stoichiometry remain the ionic assembler's job.
"""

from dataclasses import dataclass
from functools import lru_cache

from .component_mapping import (
    MappedComponent,
    decorate_component,
    fully_covered,
    hydrogen_bindings,
    mapped_branch,
    mapped_general_branch,
)
from .branch_rules import parent_chain_name
from .formatting import attach_suffix, complete_parent_name
from .parent_selection import ParentFacts, locant_set, locant_key, parent_frontier
from .parent_annotations import (isotope_seniority, isotope_numbering_key,
                                 stereo_seniority, stereo_identity,
                                 stereo_numbering, substituent_isotopes)
from .multifunctional_parent import _paths
from .branch_rules import collect_subtree


RULE = "P-62.6.1(1)/P-62.2.2.1/P-44.3/P-44.4/P-14.4/P-45.2/P-82/P-9"


@dataclass(frozen=True, slots=True)
class AmineCationComponent:
    """A fully mapped organic aminium cation suitable for ionic assembly."""

    name: str
    atoms: frozenset[int]
    atom_locants: tuple[tuple[int, str], ...]
    charge: int
    rule: str
    proof: tuple
    component_roles: tuple


@lru_cache(maxsize=1)
def ring_alkyl_aminium_query(Chem):
    return Chem.MolFromSmarts('[N+;X4;H3;D1;!R]-[C;X4;!R]')


def ring_alkyl_aminium(mol, Chem):
    """Fully mapped primary aminium on a saturated chain with one aryl branch."""
    matches = mol.GetSubstructMatches(ring_alkyl_aminium_query(Chem), maxMatches=2)
    if (len(matches) != 1 or not mol.GetRingInfo().NumRings()
            or mol.GetNumHeavyAtoms() > 32 or mol.GetStereoGroups()
            or any(a.GetAtomicNum() == 1 or a.GetIsotope() or a.GetNumRadicalElectrons()
                   or a.GetChiralTag() or a.HasQuery() for a in mol.GetAtoms())
            or any(b.GetStereo() or b.HasQuery() for b in mol.GetBonds())):
        return None
    nitrogen, idx = matches[0]
    if any(a.GetFormalCharge() for a in mol.GetAtoms() if a.GetIdx() != nitrogen):
        return None
    path, previous = [], nitrogen
    while not mol.GetAtomWithIdx(idx).IsInRing():
        carbon = mol.GetAtomWithIdx(idx)
        if (carbon.GetAtomicNum() != 6 or carbon.GetTotalNumHs() != 2
                or carbon.GetDegree() != 2):
            return None
        path.append(idx)
        neighbor = next(n.GetIdx() for n in carbon.GetNeighbors() if n.GetIdx() != previous)
        if mol.GetBondBetweenAtoms(idx, neighbor).GetBondType() != Chem.BondType.SINGLE:
            return None
        previous, idx = idx, neighbor
    if not mol.GetAtomWithIdx(idx).GetIsAromatic():
        return None
    mapping = {atom: str(locant) for locant, atom in enumerate(path, 1)}
    mapping[nitrogen] = "N"
    prefix, owned = mapped_general_branch(mol, Chem, idx, previous)
    if not prefix or set(mapping) & owned or not fully_covered(mol, mapping, owned):
        return None
    hydride = complete_parent_name(parent_chain_name(mol, path))
    if not hydride:
        return None
    locants = ("1",) if len(path) > 2 else ()
    parent = attach_suffix(hydride, "aminium", locants)
    component = MappedComponent(parent, tuple(mapping.items()), frozenset(mapping),
                                ((mapping[previous], prefix),), owned)
    name = decorate_component(mol, Chem, component)
    if not name:
        return None
    roles = (("aminium-parent", component.atoms, component.atom_locants),
             ("aryl-substituent", owned, ()))
    return AmineCationComponent(name, component.atoms | owned, component.atom_locants,
                                1, RULE, (("parent_path", tuple(path)),), roles)


def _carbon_components(mol, nitrogen, roots):
    carbon_atoms = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6}
    remaining = set(carbon_atoms)
    components = []
    for root in roots:
        found, stack = set(), [root]
        while stack:
            idx = stack.pop()
            if idx in found:
                continue
            found.add(idx)
            stack.extend(a.GetIdx() for a in mol.GetAtomWithIdx(idx).GetNeighbors()
                         if a.GetAtomicNum() == 6)
        if not found <= remaining:
            return None
        remaining -= found
        components.append((root, frozenset(found)))
    if remaining:
        return None
    return tuple(components)


def _site(mol, Chem):
    if mol.GetNumHeavyAtoms() > 64 or len(Chem.GetMolFrags(mol)) != 1:
        return None
    nitrogens = [a for a in mol.GetAtoms()
                 if a.GetAtomicNum() == 7 and a.GetFormalCharge() == 1]
    if len(nitrogens) != 1:
        return None
    nitrogen = nitrogens[0]
    if (nitrogen.IsInRing() or nitrogen.GetIsAromatic()
            or nitrogen.GetIsotope() or int(nitrogen.GetChiralTag())
            or nitrogen.GetNumRadicalElectrons()
            or nitrogen.GetTotalValence() != 4
            or sum(a.GetFormalCharge() for a in mol.GetAtoms()) != 1
            or any(a.GetNumRadicalElectrons() for a in mol.GetAtoms())):
        return None
    roots = []
    for neighbor in nitrogen.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(nitrogen.GetIdx(), neighbor.GetIdx())
        if neighbor.GetAtomicNum() == 1:
            if neighbor.GetIsotope():
                return None
            continue
        if (neighbor.GetAtomicNum() != 6 or bond is None
                or bond.GetBondType() != Chem.BondType.SINGLE):
            return None
        roots.append(neighbor.GetIdx())
    if not 1 <= len(roots) <= 4:
        return None
    if nitrogen.GetTotalNumHs(includeNeighbors=True) + len(roots) != 4:
        return None
    if any(a.GetAtomicNum() not in {1, 6, 7} for a in mol.GetAtoms()):
        return None
    if any(a.GetAtomicNum() == 1 and a.GetIsotope() for a in mol.GetAtoms()):
        return None
    carbons = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 6]
    if not carbons or len(carbons) > 40:
        return None
    for atom in carbons:
        if atom.IsInRing() or atom.GetFormalCharge() or atom.GetNumRadicalElectrons():
            return None
        for bond in atom.GetBonds():
            if bond.GetBondType() != Chem.BondType.SINGLE or int(bond.GetStereo()):
                return None
    # Nitrogen may have attached N-H represented explicitly, but the carbon
    # framework must be a forest rooted at one N-C bond per component.
    components = _carbon_components(mol, nitrogen.GetIdx(), roots)
    if components is None:
        return None
    neutral_nitrogens = [a for a in mol.GetAtoms()
                         if a.GetAtomicNum() == 7 and a.GetIdx() != nitrogen.GetIdx()]
    # P-77.1.2 monosalts of polyamines use a substitutive name.  This
    # bounded extension recognizes only neutral terminal primary amino groups
    # on the carbon tree; secondary/tertiary nitrogens, N-N bonds and other
    # nitrogen chemistry need separate source-backed component methods.
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 7 or atom.GetIdx() == nitrogen.GetIdx():
            continue
        heavy_neighbors = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
        if (atom.GetFormalCharge() or atom.IsInRing() or atom.GetIsAromatic()
                or atom.GetIsotope() or int(atom.GetChiralTag())
                or atom.GetNumRadicalElectrons() or len(heavy_neighbors) != 1
                or heavy_neighbors[0].GetAtomicNum() != 6
                or atom.GetTotalNumHs(includeNeighbors=True) != 2
                or any(n.GetAtomicNum() not in {1, 6} for n in atom.GetNeighbors())):
            return None
        bond = mol.GetBondBetweenAtoms(atom.GetIdx(), heavy_neighbors[0].GetIdx())
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return None
    if neutral_nitrogens:
        # The closed producer below handles a single unbranched carbon
        # framework. Branched aminoalkyl prefixes need a separate mapped
        # prefix proof and are left outside this family.
        if (len(components) != 1
                or any(sum(n.GetAtomicNum() == 6 for n in atom.GetNeighbors()) > 2
                       for atom in carbons)):
            return None
    for root, component in components:
        if sum(a.GetIdx() in component for a in nitrogen.GetNeighbors()) != 1:
            return None
        if len(_paths(mol, component)) < 1:
            return None
    from .stereochemistry import assigned_atom_stereochemistry
    assigned = dict(assigned_atom_stereochemistry(
        mol, Chem, atom_order="idx", include_pseudo=True
    ))
    if any(int(a.GetChiralTag()) and a.GetIdx() not in assigned
           for a in mol.GetAtoms()):
        return None
    return nitrogen.GetIdx(), tuple(components)


def _make_candidate(mol, Chem, nitrogen, root, parent_path, other_roots):
    parent_path = tuple(parent_path)
    if root not in parent_path:
        return None
    chain_name = complete_parent_name(parent_chain_name(mol, parent_path))
    if not chain_name.endswith("e"):
        return None
    host_locant = parent_path.index(root) + 1
    neutral_amino_sites = tuple(
        atom.GetIdx() for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 7 and atom.GetIdx() != nitrogen
    )
    amino_hosts = {next(n.GetIdx() for n in mol.GetAtomWithIdx(site).GetNeighbors()
                        if n.GetAtomicNum() == 6) for site in neutral_amino_sites}
    # A terminal neutral amino group is rendered as a direct `amino` prefix.
    # Until mapped aminoalkyl branch vocabulary is proved, every such host
    # must be on the selected parent chain.
    if not amino_hosts <= set(parent_path):
        return None
    if len(parent_path) == 1:
        suffix_name = chain_name[:-1] + "aminium"
    elif (host_locant == 1 and len(parent_path) == 2 and not neutral_amino_sites
          and not _partial_parent_carbon_isotope(mol, parent_path)):
        suffix_name = chain_name[:-1] + "aminium"
    elif host_locant == 1:
        suffix_name = chain_name[:-1] + "-1-aminium"
    else:
        suffix_name = chain_name[:-1] + f"-{host_locant}-aminium"

    mapping = {idx: str(locant) for locant, idx in enumerate(parent_path, 1)}
    mapping[nitrogen] = "N"
    embedded = set()
    # Each row is retained from the original graph even when the component
    # vocabulary is not proved.  Its locant and count still participate in
    # P-45 selection; the missing citation facts remain unknown.
    prefixes = []
    prefix_natures = {}
    renderable = True
    roles = [("aminium-parent", frozenset(parent_path) | {nitrogen}, ())]
    parent_role_locants = tuple(sorted(mapping.items(), key=lambda item: _locant_key(item[1])))
    roles[0] = ("aminium-parent", frozenset(parent_path) | {nitrogen}, parent_role_locants)

    # All carbon substituents not in the selected principal path remain mapped
    # prefixes at their actual parent-carbon locants.
    for idx in parent_path:
        for atom in mol.GetAtomWithIdx(idx).GetNeighbors():
            other = atom.GetIdx()
            if other in neutral_amino_sites:
                prefixes.append((mapping[idx], "amino"))
                prefix_natures["amino"] = "simple"
                embedded.add(other)
                roles.append(("amino-prefix", frozenset((other,)),
                              ((other, mapping[idx]),)))
                continue
            if atom.GetAtomicNum() != 6 or other in mapping or other in embedded:
                continue
            atoms = frozenset(collect_subtree(mol, other, idx))
            text, mapped_atoms = mapped_branch(mol, Chem, other, idx)
            if not atoms or atoms & (set(mapping) | embedded):
                return None
            # The original graph defines eligibility and ownership. A mapper
            # that declines or returns only part of that scope removes the
            # spelling, not the parent candidate or its graph facts.
            if set(mapped_atoms) != set(atoms):
                text = ""
            nature = (_mapped_alkyl_nature(mol, Chem, text, atoms, other, idx)
                      if text else None)
            prefixes.append((mapping[idx], text if nature else None))
            if nature is None:
                renderable = False
            elif text in prefix_natures and prefix_natures[text] != nature:
                return None
            elif nature is not None:
                prefix_natures[text] = nature
            embedded.update(atoms)
            roles.append(("carbon-branch", frozenset(atoms), ()))

    # The remaining N-bound carbon components are N-substituents. A complete
    # branch renderer keeps their atom ownership separate from the parent map.
    for branch_root in other_roots:
        if branch_root == root:
            continue
        atoms = frozenset(collect_subtree(mol, branch_root, nitrogen))
        text, mapped_atoms = mapped_branch(mol, Chem, branch_root, nitrogen)
        if not atoms or atoms & (set(mapping) | embedded):
            return None
        if set(mapped_atoms) != set(atoms):
            text = ""
        nature = (_mapped_alkyl_nature(mol, Chem, text, atoms, branch_root, nitrogen)
                  if text else None)
        prefixes.append(("N", text if nature else None))
        if nature is None:
            renderable = False
        elif text in prefix_natures and prefix_natures[text] != nature:
            return None
        elif nature is not None:
            prefix_natures[text] = nature
        embedded.update(atoms)
        roles.append(("N-substituent", frozenset(atoms), ()))

    mapping = hydrogen_bindings(mol, mapping)
    if not fully_covered(mol, mapping, embedded):
        return None
    component = None
    name = None
    if renderable:
        component = MappedComponent(
            suffix_name,
            tuple(mapping.items()),
            frozenset(mapping),
            tuple(prefixes),
            frozenset(embedded),
            # The shared typed-prefix multiplier API intentionally does not
            # claim amino-group semantics. The bounded polyamine path uses the
            # general locanted prefix writer, while existing mapped alkyl
            # prefixes retain their explicit simple/compound metadata.
            prefix_natures=(None if neutral_amino_sites else
                            tuple(sorted(prefix_natures.items()))),
        )
        name = decorate_component(mol, Chem, component) or None

    heavy_atoms = frozenset(a.GetIdx() for a in mol.GetAtoms()
                            if a.GetAtomicNum() > 1)
    if {idx for _, atoms, _ in roles for idx in atoms} != set(heavy_atoms):
        return None
    mapped_locants = (tuple((idx, loc) for idx, loc in component.atom_locants
                            if idx in heavy_atoms) if component is not None else ())
    if len({idx for idx, _ in mapped_locants}) != len(mapped_locants):
        return None
    # Compare candidates only from explicit P-44/P-45 evidence. In particular,
    # an isotope/stereo flag is not a substitute for the rule's full locant,
    # count, and identity comparisons.
    facts = _candidate_facts(mol, nitrogen, parent_path, mapping, prefixes, name)
    proof = (
        ("method", "P-62.6.1(1) substitutive aminium component"),
        ("polyamine_source_rules", ("P-77.1.2", "P-62.2.3", "P-14.3.4.2(b)")
         if neutral_amino_sites else ()),
        ("nitrogen", nitrogen),
        ("parent_carbon_root", root),
        ("parent_path", parent_path),
        ("suffix_carbon_locant", str(host_locant)),
        ("neutral_amino_prefix_sites", tuple(neutral_amino_sites)),
        ("n_substituent_roots", tuple(i for i in other_roots if i != root)),
        ("prefixes", tuple(prefixes)),
    )
    component = (AmineCationComponent(
        name, heavy_atoms, mapped_locants, 1,
        RULE + ("/P-77.1.2/P-62.2.3/P-14.3.4.2(b)"
                if neutral_amino_sites else ""),
        proof, tuple(roles)
    ) if name is not None else None)
    return facts, component


def _partial_parent_carbon_isotope(mol, path):
    masses = tuple(mol.GetAtomWithIdx(i).GetIsotope() for i in path)
    return any(masses) and not (len(set(masses)) == 1 and masses[0] != 0)


def _mapped_alkyl_nature(mol, Chem, text, atoms, root, blocked):
    """Classify only after the complete original carbon tree is rendered."""
    from .component_mapping import has_annotations
    from .substituent_multiplication import multiply_substituent
    from .substituent_selection import mapped_saturated_carbon_prefix

    if has_annotations(mol, atoms):
        # Preserve the already proved Job 136 annotated-simple grammar.  This
        # task does not extend annotated compound component vocabulary.
        return ("simple" if multiply_substituent(
            text, 1, nature="simple") is not None else None)
    rendered = mapped_saturated_carbon_prefix(mol, Chem, root, blocked, atoms)
    if rendered is None or rendered[0] != text or set(rendered[1]) != set(atoms):
        return None
    carbon_atoms = {idx for idx in atoms
                    if mol.GetAtomWithIdx(idx).GetAtomicNum() == 6}
    if not carbon_atoms or len(carbon_atoms) != len(atoms):
        return None
    degrees = [sum(n.GetIdx() in carbon_atoms
                   for n in mol.GetAtomWithIdx(idx).GetNeighbors())
               for idx in carbon_atoms]
    # Unsubstituted linear alkyls and the retained tert-butyl prefix are
    # simple. Every other source-rendered saturated carbon tree is substituted
    # and therefore compound under P-16.3.2.
    root_degree = sum(n.GetIdx() in carbon_atoms
                      for n in mol.GetAtomWithIdx(root).GetNeighbors())
    retained_star = (len(carbon_atoms) == 4 and root_degree == 3 and all(
        sum(n.GetIdx() in carbon_atoms
            for n in mol.GetAtomWithIdx(idx).GetNeighbors()) == 1
        for idx in carbon_atoms if idx != root
    ))
    simple_graph = max(degrees, default=0) <= 2 or retained_star
    nature = "simple" if simple_graph else "compound"
    if multiply_substituent(text, 1, nature=nature) is None:
        return None
    return nature


def _candidate_facts(mol, nitrogen, path, mapping, prefixes, name):
    from .formatting import prefix_alphabetization_key

    atoms = frozenset(path)
    prefix_locs = locant_set(loc for loc, _ in prefixes)
    grouped = {}
    for loc, text in prefixes:
        if text is None:
            continue
        grouped.setdefault(text, []).append(locant_key(loc))
    citation = (tuple(loc for text in sorted(grouped, key=prefix_alphabetization_key)
                      for loc in sorted(grouped[text]))
                if prefix_locs is not None and all(text is not None for _, text in prefixes)
                else None)
    mapped = {idx: loc for idx, loc in mapping.items()
              if idx in atoms or idx == nitrogen}
    suffix_locant = str(path.index(next(
        idx for idx in path if mol.GetBondBetweenAtoms(idx, nitrogen))) + 1)
    return ParentFacts(
        principal_class=(6, 0), suffix_count=1, kind="chain",
        elements=((6, len(path)),), ring_count=0, multiple_count=0,
        double_count=0, nonstandard_key=(), indicated_h_locants=(),
        replacement_locants=(), replacement_element_locants=(), fusion_locants=(),
        suffix_locants=locant_set((suffix_locant,)), attachment_locants=(),
        unsaturation_locants=(), double_locants=(),
        isotope_key=isotope_seniority(mol, mapped),
        equivalent_isotope_key=isotope_numbering_key(
            mol, mapped, include_substituent_groups=True),
        stereo_key=stereo_seniority(mol, mapped), multiplied_parents=1,
        prefix_count=len(prefixes), prefix_locants=prefix_locs,
        prefix_citation_locants=citation, substituent_nonstandard_key=(),
        substituent_isotope_key=substituent_isotopes(mol, mapped),
        alphanumerical_key=(tuple(sorted(
            prefix_alphabetization_key(text) for _, text in prefixes))
            if all(text is not None for _, text in prefixes) else None),
        numbering_scope=(atoms, frozenset((nitrogen,))),
        stereo_identity=stereo_identity(mol, mapped),
        stereo_numbering_key=stereo_numbering(mol, mapped),
    )


def _locant_key(locant):
    if locant == "N":
        return (0, 0)
    try:
        return (1, int(locant))
    except (TypeError, ValueError):
        return (2, str(locant))


def amine_cation_component(mol, Chem):
    """Return one mapped P-62.6 aminium component, or ``None`` if unproved."""
    site = _site(mol, Chem)
    if site is None:
        return ring_alkyl_aminium(mol, Chem) if mol.GetRingInfo().NumRings() else None
    nitrogen, components = site
    neutral_amino_sites = tuple(
        atom.GetIdx() for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 7 and atom.GetIdx() != nitrogen
    )
    # The new path is deliberately limited to mono-protonated polyamines with
    # a single carbon arm at N+. Existing annotated tetraalkyl aminium cases
    # remain on the established route; annotations on this new polyamine
    # family are not claimed by this source-derived implementation.
    if neutral_amino_sites:
        charged_heavy = [a for a in mol.GetAtomWithIdx(nitrogen).GetNeighbors()
                         if a.GetAtomicNum() > 1]
        if (len(components) != 1 or len(charged_heavy) != 1
                or charged_heavy[0].GetAtomicNum() != 6
                or len([a for a in mol.GetAtoms()
                        if a.GetAtomicNum() == 6 and (a.GetIsotope() or int(a.GetChiralTag()))])
                or any(int(b.GetStereo()) for b in mol.GetBonds())
                or mol.GetStereoGroups()):
            return None
    candidates = []
    all_roots = tuple(root for root, _ in components)
    for root, atoms in components:
        for path in _paths(mol, atoms):
            if neutral_amino_sites and set(path) != set(atoms):
                continue
            if root not in path:
                continue
            for oriented in (path, tuple(reversed(path))):
                candidate = _make_candidate(
                    mol, Chem, nitrogen, root, oriented, all_roots
                )
                if candidate is not None:
                    candidates.append(candidate)
    if not candidates:
        return None
    frontier = parent_frontier(candidates, lambda item: item[0])
    winners = [component for _, component in frontier]
    if any(component is None for component in winners):
        return None
    if len({component.name for component in winners}) != 1:
        return None
    return min(winners, key=lambda component: component.proof)
