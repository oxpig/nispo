"""Mapped monoanions of bounded acyclic polyhydroxy carbon compounds.

P-77.2.2/P-72.2.2.2.2 use hydroxy prefixes and an olate suffix for a
monosalt of a polyhydroxy compound. This module returns the organic anion
component only; ionic stoichiometry and cation naming belong to the assembler.
"""

from dataclasses import dataclass, replace

from .isotope_component_coverage import IsotopeComponentRecord


@dataclass(frozen=True, slots=True)
class PolyolAnionComponent:
    """A mapped organic monoanion suitable for binary salt assembly."""

    name: str
    atoms: frozenset[int]
    atom_locants: tuple[tuple[int, str], ...]
    charge: int
    rule: str
    proof: tuple
    anionic_oxygen: int
    hydroxy_oxygens: tuple[int, ...]
    parent_atoms: frozenset[int]
    embedded_atoms: frozenset[int]
    # Primary original-isotope accounting for the mapped parent and its
    # independently rendered plain branch component.  ``None`` preserves
    # construction compatibility for callers that only consume the fields
    # above; returned values from this producer always carry a validated tuple.
    isotope_coverage: tuple[IsotopeComponentRecord, ...] | None = None


def polyol_anion_component(mol, Chem):
    """Return a bounded polyhydroxy monoanion component, otherwise ``None``.

    Scope is one connected, saturated acyclic C/O graph with a carbon-tree
    parent, one terminal O− on that parent, at least one neutral OH on that
    parent, and optional unlabelled hydrocarbon branches. Oxygen isotope
    placement in olate/hydroxy language is not established here, so labelled
    oxygen graphs decline. Isotope and CIP ranking use the original charged
    graph, never a neutralized selection copy.
    """
    if (mol.GetNumHeavyAtoms() > 32 or len(Chem.GetMolFrags(mol)) != 1
            or mol.GetStereoGroups()):
        return None
    if any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()):
        return None
    if any(atom.GetAtomicNum() not in {1, 6, 8} for atom in mol.GetAtoms()):
        return None
    if any(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()):
        return None

    carbons = {atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6}
    oxygens = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8]
    if len(carbons) < 2 or len(carbons) > 10 or not oxygens:
        return None
    if any(atom.GetFormalCharge() not in {0, -1} for atom in mol.GetAtoms()):
        return None
    if sum(atom.GetFormalCharge() for atom in mol.GetAtoms()) != -1:
        return None
    anionic = [atom for atom in oxygens if atom.GetFormalCharge() == -1]
    if len(anionic) != 1 or anionic[0].GetDegree() != 1:
        return None
    anion_o = anionic[0]
    anion_owner = anion_o.GetNeighbors()[0].GetIdx()
    if anion_owner not in carbons:
        return None
    if mol.GetBondBetweenAtoms(anion_o.GetIdx(), anion_owner).GetBondType() != Chem.BondType.SINGLE:
        return None

    hydroxy = []
    for oxygen in oxygens:
        if oxygen.GetIsotope():
            return None  # No proved role-local oxygen label placement for olates.
        if oxygen.GetIdx() == anion_o.GetIdx():
            continue
        if oxygen.GetFormalCharge() != 0 or oxygen.GetDegree() != 1:
            return None
        owner = oxygen.GetNeighbors()[0]
        bond = mol.GetBondBetweenAtoms(oxygen.GetIdx(), owner.GetIdx())
        if (owner.GetAtomicNum() != 6 or bond.GetBondType() != Chem.BondType.SINGLE
                or oxygen.GetTotalNumHs(includeNeighbors=True) != 1):
            return None
        hydroxy.append(oxygen.GetIdx())
    if not hydroxy:
        return None

    adjacency = {idx: set() for idx in carbons}
    edge_count = 0
    for bond in mol.GetBonds():
        left, right = bond.GetBeginAtom(), bond.GetEndAtom()
        if left.GetAtomicNum() == right.GetAtomicNum() == 6:
            if bond.GetBondType() != Chem.BondType.SINGLE:
                return None
            adjacency[left.GetIdx()].add(right.GetIdx())
            adjacency[right.GetIdx()].add(left.GetIdx())
            edge_count += 1
        elif bond.GetBondType() != Chem.BondType.SINGLE:
            return None
    if edge_count != len(carbons) - 1:
        return None
    seen, stack = set(), [next(iter(carbons))]
    while stack:
        idx = stack.pop()
        if idx in seen:
            continue
        seen.add(idx)
        stack.extend(adjacency[idx] - seen)
    if seen != carbons or any(mol.GetAtomWithIdx(i).IsInRing() for i in carbons):
        return None
    if any(atom.IsInRing() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1):
        return None

    # Only parent-carbon 13C annotations and ordinary carbon stereocenters
    # are handled by this bounded writer. Any off-parent annotation needs a
    # separately mapped branch component and is conservatively declined.
    for atom in mol.GetAtoms():
        if atom.GetIsotope() and (atom.GetAtomicNum() != 6 or atom.GetIsotope() != 13):
            return None
        if atom.GetChiralTag() and atom.GetAtomicNum() != 6:
            return None
    if any(bond.GetStereo() != Chem.BondStereo.STEREONONE for bond in mol.GetBonds()):
        return None

    from .acyclic.rules import _path_substituent_prefixes
    from .component_mapping import MappedComponent
    from .constants import ROOTS
    from .multifunctional_parent import _paths
    from .parent_annotations import (
        isotope_numbering_key, isotope_seniority, stereo_identity,
        stereo_numbering, stereo_seniority, substituent_isotopes,
    )
    from .parent_selection import graph_parent_facts, parent_frontier
    from .isotope_component_coverage import (
        component_record_from_atom_map, validate_isotope_component_coverage,
    )
    from .isotope_component_writer import render_mapped_component_with_coverage

    parent_paths = [tuple(path) for path in _paths(mol, carbons)
                    if anion_owner in path and len(path) >= 2]
    if not parent_paths:
        return None

    candidates = []
    for path in parent_paths:
        for oriented in (path, tuple(reversed(path))):
            # Remaining hydroxy centers must be locanted on the carbon parent;
            # an OH hidden in an embedded branch has no supported map here.
            if any(mol.GetAtomWithIdx(oxygen).GetNeighbors()[0].GetIdx()
                   not in oriented for oxygen in hydroxy):
                continue
            path_locants = {idx: str(pos) for pos, idx in enumerate(oriented, 1)}
            # Plain unlabelled hydrocarbon branches are named as prefixes and
            # embedded as opaque atoms, without assigning them parent locants.
            branch_carbons = carbons - set(oriented)
            if any(mol.GetAtomWithIdx(idx).GetIsotope()
                   or mol.GetAtomWithIdx(idx).GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
                   for idx in branch_carbons):
                continue

            # Build a neutral, unannotated structural view only for ordinary
            # parent facts. Annotation ranking is independently taken from the
            # original charged molecule below.
            facts_mol = Chem.Mol(mol)
            facts_oxygen = facts_mol.GetAtomWithIdx(anion_o.GetIdx())
            facts_oxygen.SetFormalCharge(0)
            facts_oxygen.SetNoImplicit(False)
            facts_oxygen.SetNumExplicitHs(0)
            for atom in facts_mol.GetAtoms():
                atom.SetIsotope(0)
                atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            for bond in facts_mol.GetBonds():
                bond.SetStereo(Chem.BondStereo.STEREONONE)
                bond.SetBondDir(Chem.BondDir.NONE)
            try:
                Chem.SanitizeMol(facts_mol)
            except (ValueError, RuntimeError):
                continue

            bare = Chem.Mol(mol)
            for atom in bare.GetAtoms():
                atom.SetIsotope(0)
                atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            for bond in bare.GetBonds():
                bond.SetStereo(Chem.BondStereo.STEREONONE)
                bond.SetBondDir(Chem.BondDir.NONE)
            prefixes = _path_substituent_prefixes(bare, oriented, skip_plain_hydrogen=True)
            if prefixes is None or sum(term == "oxido" for _, term in prefixes) != 1:
                continue
            if sum(term == "hydroxy" for _, term in prefixes) != len(hydroxy):
                continue
            anion_locant = path_locants[anion_owner]
            stem = ROOTS.get(len(oriented))
            if stem is None:
                continue
            parent = f"{stem}an-{anion_locant}-olate"
            mapped = dict(path_locants)
            for oxygen in (anion_o.GetIdx(), *hydroxy):
                host = mol.GetAtomWithIdx(oxygen).GetNeighbors()[0].GetIdx()
                if host not in path_locants:
                    break
                mapped[oxygen] = path_locants[host]
            else:
                # All locanted groups have an actual parent-carbon context.
                prefix_facts = tuple((loc, term) for loc, term in prefixes if term != "oxido")
                atom_set = frozenset((*oriented, anion_o.GetIdx(), *hydroxy))
                embedded = frozenset(branch_carbons)
                atom_map = tuple((idx, mapped[idx]) for idx in sorted(atom_set))
                candidate_component = MappedComponent(
                    parent=parent,
                    atom_locants=atom_map,
                    atoms=atom_set,
                    prefixes=prefix_facts,
                    embedded_atoms=embedded,
                    systematic_parent=True,
                )
                written = render_mapped_component_with_coverage(
                    mol, Chem, candidate_component,
                    component_id="polyol-anion-parent",
                )
                if written is None:
                    continue
                rendered = written.text
                facts = graph_parent_facts(
                    facts_mol, oriented, path_locants.items(),
                    # P-41 class 4: these are anionic characteristic groups;
                    # neutral hydroxy groups are subordinate prefixes.
                    principal_class=(4,), suffix_centers=(anion_owner,),
                    suffix_locants=(anion_locant,), prefixes=prefix_facts,
                    kind="chain", name=rendered,
                )
                if facts is None:
                    continue
                # Use isotope/stereo evidence from original charged graph only.
                isotope_key = isotope_seniority(mol, path_locants)
                equivalent_isotope_key = isotope_numbering_key(
                    mol, path_locants, include_substituent_groups=True)
                stereo_key = stereo_seniority(mol, path_locants)
                branch_isotope_key = substituent_isotopes(mol, path_locants)
                if (isotope_key is None or equivalent_isotope_key is None
                        or stereo_key is None or branch_isotope_key is None):
                    continue
                facts = replace(
                    facts,
                    isotope_key=isotope_key,
                    equivalent_isotope_key=equivalent_isotope_key,
                    stereo_key=stereo_key,
                    substituent_isotope_key=branch_isotope_key,
                    numbering_scope=(frozenset(oriented), frozenset((anion_owner,))),
                    stereo_identity=stereo_identity(mol, path_locants),
                    stereo_numbering_key=stereo_numbering(mol, path_locants),
                )
                candidates.append({
                    "path": oriented, "name": rendered, "path_locants": path_locants,
                    "atom_locants": mapped, "component": candidate_component,
                    "coverage": written.coverage,
                    "facts": facts,
                })

    frontier = parent_frontier(candidates, lambda row: row["facts"])
    if not frontier:
        return None
    names = {row["name"] for row in frontier}
    if len(names) != 1:
        return None
    # This tie affects returned maps only; every frontier candidate has the
    # same independently rendered name.
    selected = min(frontier, key=lambda row: tuple(row["path"]))
    parent_atoms = frozenset(selected["path"])
    oxygen_atoms = frozenset((anion_o.GetIdx(), *hydroxy))
    atoms = parent_atoms | oxygen_atoms
    embedded_atoms = frozenset(carbons - parent_atoms)
    if atoms & embedded_atoms or atoms | embedded_atoms != frozenset(
            atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1):
        return None
    component = selected["component"]
    if component.atoms != atoms or component.embedded_atoms != embedded_atoms:
        return None
    name = selected["name"]
    coverage_records = [selected["coverage"]]
    if embedded_atoms:
        # Branch atoms are rendered only as ordinary, unlabelled prefix
        # components by the same typed prefix operation. The component map
        # deliberately gives these embedded atoms no parent locants. Their
        # record owns source atoms without claiming any isotope descriptor.
        embedded_terms = tuple(
            f"{loc}-{term}" if loc else term
            for loc, term in component.prefixes if term != "hydroxy"
        )
        embedded_text = ";".join(embedded_terms)
        if not embedded_text:
            return None
        try:
            embedded_record = component_record_from_atom_map(
                mol,
                component_id="polyol-anion-embedded-branches",
                component_text=embedded_text,
                atom_ids=embedded_atoms,
                atom_locants=tuple((idx, None) for idx in sorted(embedded_atoms)),
                descriptor_atom_ids=(),
            )
        except (TypeError, ValueError, IndexError, KeyError):
            return None
        coverage_records.append(embedded_record)
    coverage = tuple(coverage_records)
    coverage_result = validate_isotope_component_coverage(mol, coverage)
    if not coverage_result.complete:
        return None
    return PolyolAnionComponent(
        name=name,
        atoms=atoms,
        atom_locants=component.atom_locants,
        charge=-1,
        rule="P-77.2.2/P-72.2.2.2.2/P-44/P-45/P-82",
        proof=("one terminal carbon-bound O−", "one or more parent-bound OH groups",
               "acyclic saturated carbon tree", "original-graph isotope/stereo facts"),
        anionic_oxygen=anion_o.GetIdx(),
        hydroxy_oxygens=tuple(sorted(hydroxy)),
        parent_atoms=parent_atoms,
        embedded_atoms=embedded_atoms,
        isotope_coverage=coverage,
    )
