"""P-65.5.1 sulfonyl halides with source-closed benzene ownership."""

from dataclasses import replace

from .bounded_retained_carboxylic_parents import substituted_benzene_paths
from .component_mapping import decorate_component
from .constants import MULT
from .extended_inventory import selection_inventory
from .molecule_iteration import iter_atoms, iter_bonds
from .naming_methods import MethodDecision
from .neutral_charge_components import neutral_charge_role_proofs


def sulfonyl_halide_site(sulfur):
    """Map one neutral C–S(=O)₂–halogen site."""
    if (sulfur.GetAtomicNum() != 16 or sulfur.GetDegree() != 4 or sulfur.IsInRing()
            or sulfur.GetFormalCharge() or sulfur.GetIsotope() or sulfur.GetNumRadicalElectrons()):
        return None
    oxygens, halides, carbons = [], [], []
    for bond in sulfur.GetBonds():
        atom = bond.GetOtherAtom(sulfur)
        element, order = atom.GetAtomicNum(), bond.GetBondTypeAsDouble()
        if atom.GetFormalCharge() or atom.GetNumRadicalElectrons():
            return None
        if element == 8 and order == 2 and atom.GetDegree() == 1 and not atom.GetIsotope():
            oxygens.append(atom.GetIdx())
        elif element in {9, 17, 35, 53} and order == 1 and atom.GetDegree() == 1 and not atom.GetIsotope():
            halides.append(atom.GetIdx())
        elif element == 6 and order == 1:
            carbons.append(atom.GetIdx())
    if len(oxygens) == 2 and len(halides) == len(carbons) == 1:
        return (*oxygens, halides[0], carbons[0])
    return None


def mapped_sulfonyl_halide_decision(mol, Chem):
    """Render a selected sulfonyl halide with a numbered aromatic acid side."""
    from .branch_rules import collect_subtree
    from .formatting import attach_suffix
    from .mapped_ring_parents import records
    from .naming_result import SelectionLimitation

    inventory = selection_inventory(mol)
    if not inventory.supported or not inventory.principal or mol.GetStereoGroups():
        return None
    if len(inventory.principal) > 1:
        return _ring_polysulfonyl_halide(mol, Chem, inventory.principal)
    group = inventory.principal[0]
    if group.kind != 'sulfonyl-halide':
        return None
    if not mol.GetAtomWithIdx(group.owner).IsInRing():
        return _chain_sulfonyl_halide(mol, Chem, group)
    if not mol.GetAtomWithIdx(group.owner).GetIsAromatic():
        return None
    scope = frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1)
    acid_side = frozenset(collect_subtree(mol, group.owner, group.center))
    halide = {9: 'fluoride', 17: 'chloride', 35: 'bromide', 53: 'iodide'}[
        mol.GetAtomWithIdx(group.ligand).GetAtomicNum()]
    if acid_side & set(group.atoms) or acid_side | set(group.atoms) != scope:
        return None
    candidates = []
    for row in records(mol, Chem, scope=acid_side, attachment=group.owner, general_branches=True):
        component = row.construction
        locant = dict(row.atom_locants).get(group.owner)
        if (component is None or not locant or not row.hydrogen_budget_proved or not row.annotation_coverage
                or component.atoms | component.embedded_atoms != acid_side or row.added_h_locants):
            continue
        locants = () if row.parent == 'benzene' and not component.prefixes else (locant,)
        parent = attach_suffix(row.hydride or row.parent, 'sulfonyl', locants)
        if not parent:
            continue
        name = decorate_component(mol, Chem, replace(component, parent=parent,
                                  embedded_atoms=component.embedded_atoms | set(group.atoms)))
        if name:
            candidates.append((name, row.atom_locants))
    if len({name for name, _ in candidates}) != 1:
        return None
    name, locants = candidates[0]
    rule = 'P-41/P-65.5.1'
    return MethodDecision(f'{name} {halide}', 'functional-class', frozenset({group.center}), scope,
        rule, locants, preference_issues=(SelectionLimitation(
            'sulfonyl-halide-parent', rule, 'Complete acid-side parent comparison remains open.'),))


def _ring_polysulfonyl_halide(mol, Chem, groups):
    """Cite identical sulfonyl halides on the same otherwise bare ring."""
    from .mapped_ring_parents import records
    from .naming_result import SelectionLimitation

    if any(group.kind != "sulfonyl-halide" for group in groups) or len(groups) not in MULT:
        return None
    rings = mol.GetRingInfo().AtomRings()
    if len(rings) != 1:
        return None
    ring = frozenset(rings[0])
    owners = frozenset(group.owner for group in groups)
    units = frozenset().union(*(group.atoms for group in groups))
    if len(owners) != len(groups) or not owners <= ring or ring & units:
        return None
    scope = frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1)
    if ring | units != scope or any(a.GetIsotope() or a.GetChiralTag() or a.GetFormalCharge()
                                    for a in iter_atoms(mol)):
        return None
    elements = {mol.GetAtomWithIdx(group.ligand).GetAtomicNum() for group in groups}
    if len(elements) != 1:
        return None
    candidates = []
    for owner in owners:
        for row in records(mol, Chem, scope=ring, attachment=owner, core_filter=ring):
            mapping = dict(row.atom_locants)
            if not row.hydrogen_budget_proved or not row.annotation_coverage or row.added_h_locants:
                continue
            if not all(mapping.get(i, "").isdecimal() for i in owners):
                continue
            candidates.append((tuple(sorted(int(mapping[i]) for i in owners)),
                               row.hydride or row.parent, row.atom_locants))
    if not candidates:
        return None
    positions = min(row[0] for row in candidates)
    best = [row for row in candidates if row[0] == positions]
    if len({row[1] for row in best}) != 1:
        return None
    _, parent, mapping = best[0]
    halide = {9: "fluoride", 17: "chloride", 35: "bromide", 53: "iodide"}[elements.pop()]
    multiplier = MULT[len(groups)]
    name = f"{parent}-{','.join(map(str, positions))}-{multiplier}sulfonyl {multiplier}{halide}"
    rule = "P-41/P-65.5.1"
    return MethodDecision(name, "functional-class", frozenset(group.center for group in groups),
                          scope, rule, mapping, preference_issues=(SelectionLimitation(
                              "sulfonyl-halide-parent", rule, "The mapped ring cites all sulfur sites."),))


def _chain_sulfonyl_halide(mol, Chem, group):
    from .mapped_principal_general import _carbon_proposals
    from .naming_result import SelectionLimitation
    from .parent_selection import parent_frontier

    if any(a.GetIsotope() or a.GetNumRadicalElectrons()
           or a.GetChiralTag() for a in iter_atoms(mol)) or any(b.GetStereo() for b in iter_bonds(mol)):
        return None
    if any(a.GetFormalCharge() for a in iter_atoms(mol)) and not neutral_charge_role_proofs(mol):
        return None
    capped = Chem.RWMol(mol)
    capped.ReplaceAtom(group.ligand, Chem.Atom(8))
    capped = capped.GetMol()
    try:
        Chem.SanitizeMol(capped)
    except (ValueError, RuntimeError):
        return None
    inventory = selection_inventory(capped)
    if (not inventory.supported or len(inventory.principal) != 1
            or inventory.principal[0].kind != "sulfonic-acid"
            or inventory.principal[0].center != group.center):
        return None
    rows = list(_carbon_proposals(capped, Chem, inventory.principal, inventory, 0))
    if not rows or any(facts is None for _, facts in rows):
        return None
    best = parent_frontier(rows, lambda row: row[1])
    names = {row.name for row, _ in best}
    if len(names) != 1 or not (acid := names.pop()).endswith("sulfonic acid"):
        return None
    name = acid.removesuffix("sulfonic acid") + "sulfonyl"
    halide = {9: "fluoride", 17: "chloride", 35: "bromide", 53: "iodide"}[
        mol.GetAtomWithIdx(group.ligand).GetAtomicNum()]
    rule = "P-41/P-65.5.1"
    return MethodDecision(f"{name} {halide}", "functional-class", frozenset({group.center}),
        frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1),
        rule, best[0][0].atom_locants, preference_issues=(SelectionLimitation(
            "sulfonyl-halide-parent", rule, "Complete acid-side parent comparison remains open."),))


def benzene_sulfonyl_halide_decision(mol, Chem):
    """Name one unannotated benzene sulfonyl halide and simple ring prefixes.

    The aromatic six-ring can carry terminal methyl, halogen or proved
    isocyano substituents. An acid-capped private copy
    independently confirms the sole sulfur source class, while all emitted
    atom ownership and the acyl suffix are mapped back to the source graph.
    More than one functional group, annotations and other ring systems decline.
    """
    atoms = tuple(iter_atoms(mol))
    if (mol.GetStereoGroups() or len(Chem.GetMolFrags(mol)) != 1
            or any(a.GetNumRadicalElectrons() or a.GetIsotope()
                   or int(a.GetChiralTag()) or a.GetAtomicNum() == 1 for a in atoms)
            or any(int(b.GetStereo()) or int(b.GetBondDir()) for b in iter_bonds(mol))):
        return None
    sulfur = [a for a in atoms if a.GetAtomicNum() == 16]
    rings = mol.GetRingInfo().AtomRings()
    if len(sulfur) != 1 or len(rings) != 1 or len(rings[0]) != 6:
        return None
    s = sulfur[0]
    if s.IsInRing() or s.GetDegree() != 4:
        return None
    ring = frozenset(rings[0])
    if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6
           or not mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
        return None
    ring_bonds = [b for b in iter_bonds(mol)
                  if b.GetBeginAtomIdx() in ring and b.GetEndAtomIdx() in ring]
    if len(ring_bonds) != 6 or any(not b.GetIsAromatic() for b in ring_bonds):
        return None
    oxygens = [n for n in s.GetNeighbors() if n.GetAtomicNum() == 8
               and n.GetDegree() == 1
               and mol.GetBondBetweenAtoms(s.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 2]
    halides = [n for n in s.GetNeighbors() if n.GetAtomicNum() in {9, 17, 35, 53}
               and n.GetDegree() == 1
               and mol.GetBondBetweenAtoms(s.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 1]
    owners = [n for n in s.GetNeighbors() if n.GetIdx() in ring
              and mol.GetBondBetweenAtoms(s.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 1]
    if len(oxygens) != 2 or len(halides) != 1 or len(owners) != 1:
        return None
    halide, owner = halides[0].GetIdx(), owners[0].GetIdx()
    suffix = {s.GetIdx(), halide, *(a.GetIdx() for a in oxygens)}
    outside = {a.GetIdx() for a in atoms} - ring - suffix
    labels = {9: 'fluoro', 17: 'chloro', 35: 'bromo', 53: 'iodo'}
    prefixes = {}
    charged = any(a.GetFormalCharge() for a in atoms)
    charge_roles = neutral_charge_role_proofs(mol) if charged else ()
    consumed = set()
    if charged and not charge_roles:
        return None
    for proof in charge_roles:
        if (proof.kind != 'isocyano' or proof.attachment_atom not in ring - {owner}
                or not set(proof.atoms) <= outside):
            return None
        prefixes[proof.atoms[0]] = proof.kind
        consumed.update(proof.atoms)
    for idx in outside - consumed:
        atom = mol.GetAtomWithIdx(idx)
        if (atom.GetDegree() != 1 or atom.GetNeighbors()[0].GetIdx() not in ring
                or atom.GetNeighbors()[0].GetIdx() == owner
                or atom.GetBonds()[0].GetBondTypeAsDouble() != 1):
            return None
        if atom.GetAtomicNum() in labels:
            prefixes[idx] = labels[atom.GetAtomicNum()]
        elif atom.GetAtomicNum() == 6 and atom.GetTotalNumHs() == 3:
            prefixes[idx] = 'methyl'
        else:
            return None
    if any(sum(n.GetIdx() in outside for n in mol.GetAtomWithIdx(i).GetNeighbors()) > 1
           for i in ring):
        return None

    capped = Chem.RWMol(mol)
    capped.GetAtomWithIdx(halide).SetAtomicNum(8)
    capped = capped.GetMol()
    try:
        Chem.SanitizeMol(capped)
    except (ValueError, RuntimeError):
        return None
    inventory = selection_inventory(capped)
    if (not inventory.supported or len(inventory.groups) != 1
            or len(inventory.principal) != 1
            or inventory.principal[0].kind != 'sulfonic-acid'
            or inventory.principal[0].center != s.GetIdx()):
        return None
    group = inventory.principal[0]
    paths = substituted_benzene_paths(mol, ring, owner, prefixes)
    from .complete_functional_parent import _render_functional_candidate

    candidates = []
    for path in paths:
        candidate = _render_functional_candidate(
            capped, Chem, (group,), 'sulfonic-acid', path, True, True,
            set(group.atoms) - {group.center}, [],
            mapped_ring_prefixes=True, charge_role_proofs=inventory.charge_role_proofs,
        )
        if candidate is None:
            return None
        construction = candidate.construction
        if not construction.parent.endswith('sulfonic acid'):
            return None
        # This changes a typed acid suffix to its corresponding acyl suffix,
        # never a finished full name; the departing halide is a separate role.
        acyl_parent = construction.parent[:-len('sulfonic acid')] + 'sulfonyl'
        locants = tuple((i, locant) for i, locant in construction.atom_locants
                        if i != halide)
        original = replace(construction, parent=acyl_parent, atom_locants=locants,
                           atoms=construction.atoms - {halide})
        text = decorate_component(mol, Chem, original)
        if (not text or original.atoms | original.embedded_atoms | {halide}
                != {a.GetIdx() for a in atoms}
                or original.atoms & original.embedded_atoms
                or halide in original.atoms | original.embedded_atoms):
            return None
        candidates.append((text, locants, original))
    if not candidates or len({candidate[0] for candidate in candidates}) != 1:
        return None
    text, locants, original = candidates[0]
    halide_name = {9: 'fluoride', 17: 'chloride', 35: 'bromide', 53: 'iodide'}[
        mol.GetAtomWithIdx(halide).GetAtomicNum()]
    acyl_atoms = original.atoms | original.embedded_atoms
    return MethodDecision(
        text + ' ' + halide_name, 'functional-class', frozenset({s.GetIdx()}),
        frozenset(a.GetIdx() for a in atoms), 'P-65.5.1/P-65.3.1', locants,
        component_roles=(('sulfonyl', acyl_atoms, locants),
                         ('halide', frozenset({halide}), ())),
    )
