"""General substitutive delivery from complete mapped cyclic constructions.

P-15.1/P-29 permits a substituted cyclic parent even when preference among
P-44/P-45 parents or P-51 methods remains unresolved. Every construction uses
the original graph's mapped parent, branches, hydrogen state and annotations.
No fragment naming, structure parser, or preferred-name claim is involved.
"""

from .naming_methods import MethodDecision
from .naming_result import SelectionLimitation


RULE = 'P-15.1/P-29/P-41/P-63/P-64/P-65/P-66/P-82/P-9'


def mapped_ring_construction(mol, Chem):
    """Return a whole-graph general name after the preferred routes fail."""
    if (not mol.GetRingInfo().NumRings() or mol.GetStereoGroups()
            or len(Chem.GetMolFrags(mol)) != 1
            or any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in mol.GetAtoms())):
        return None
    from .extended_inventory import selection_inventory
    inventory = selection_inventory(mol)
    if (not inventory.supported or not inventory.principal
            or len({g.kind for g in inventory.principal}) != 1):
        return None
    if len(inventory.principal) == 1 and inventory.principal[0].kind == 'sulfonamide':
        from .nested_aza_sulfonamide import nested_aza_sulfonamide_decision
        nested = nested_aza_sulfonamide_decision(mol, Chem)
        if nested is not None:
            return nested
    from .mapped_ring_parents import records
    atoms = frozenset(a.GetIdx() for a in mol.GetAtoms())
    rows = []
    for row in records(mol, Chem, principal=inventory.principal):
        construction = row.construction
        if (row.name and row.principal_centers and construction is not None
                and construction.atoms | construction.embedded_atoms == atoms):
            rows.append(row)
    if not rows:
        return None
    # This orders only complete general constructions. Missing preferred
    # alternatives cannot be excluded from this local pool; keep that gap in
    # the result instead of certifying its winner as a PIN.
    row = min(rows, key=lambda r: (-len(r.principal_centers), -len(r.parent_atoms), r.name))
    return MethodDecision(
        name=row.name, method='substitutive', principal_centers=row.principal_centers,
        atoms=atoms, rule=RULE, atom_locants=row.atom_locants,
        component_roles=(('cyclic-parent', frozenset(row.parent_atoms), row.atom_locants),
                         ('mapped-prefixes-and-suffixes', row.construction.embedded_atoms, ())),
        preference_issues=(SelectionLimitation(
            'mapped-cyclic-construction', 'P-44/P-45/P-51',
            'The original mapped cyclic parent, functions and branches form a complete general name; global preferred parent/method comparison remains unresolved.'),),
    )


def mapped_substituted_ring(mol, Chem):
    """Recover a complete substituted cyclic hydride without a principal group."""
    if (not mol.GetRingInfo().NumRings() or mol.GetStereoGroups()
            or len(Chem.GetMolFrags(mol)) != 1
            or any(a.HasQuery() for a in mol.GetAtoms())
            or any(b.HasQuery() for b in mol.GetBonds())):
        return None
    from .component_mapping import _general_branch_stereo_is_cited
    from .extended_inventory import selection_inventory
    from .mapped_ring_parents import records, ring_systems
    from .neutral_charge_components import validate_neutral_charge_role_proofs

    inventory = selection_inventory(mol)
    proofs = inventory.charge_role_proofs
    if (not inventory.supported or inventory.principal
            or any(proof.kind != 'azido' for proof in proofs)
            or not validate_neutral_charge_role_proofs(mol, proofs)):
        return None
    scope = frozenset(a.GetIdx() for a in mol.GetAtoms()
                      if a.GetAtomicNum() > 1 or a.GetIsotope() or a.GetChiralTag())
    best = None
    for core in sorted(ring_systems(mol), key=len, reverse=True):
        if best is not None and len(core) < len(best.parent_atoms):
            break
        for row in records(mol, Chem, scope=scope, core_filter=core,
                           principal=(), general_branches=True):
            component = row.construction
            if (not row.name or not row.hydrogen_budget_proved or not row.annotation_coverage
                    or component is None or component.atoms | component.embedded_atoms != scope):
                continue
            branches = component.prefix_branch_ownership
            if (any(not any(set(proof.atoms) <= owned and proof.attachment_atom in owned | {host}
                            for _, host, owned, _ in branches) for proof in proofs)
                    or any(not _general_branch_stereo_is_cited(mol, Chem, owned, text)
                           for _, _, owned, text in branches)):
                continue
            if best is None or row.name < best.name:
                best = row
    return best


def azido_fused_construction(mol, Chem):
    """Deliver a mapped fused parent when azide charges are fully owned."""
    from .extended_inventory import selection_inventory
    from .mapped_ring_parents import records, ring_systems
    from .neutral_charge_components import validate_neutral_charge_role_proofs

    inventory = selection_inventory(mol)
    proofs = inventory.charge_role_proofs
    if (not inventory.supported or not any(proof.kind == 'azido' for proof in proofs)
            or not validate_neutral_charge_role_proofs(mol, proofs)
            or len(Chem.GetMolFrags(mol)) != 1 or mol.GetStereoGroups()
            or any(a.GetIsotope() or a.HasQuery() for a in mol.GetAtoms())):
        return None
    scope = frozenset(range(mol.GetNumAtoms()))
    candidates = []
    for core in ring_systems(mol):
        if len(core) < 8:
            continue
        principal = tuple(g for g in inventory.principal if g.center in core)
        if principal:
            if not (all(g.kind == 'alcohol' for g in principal)
                    and set(principal) == set(inventory.principal)):
                # A cyclic imide must own both carbonyls on its annular N.
                if len(principal) != 2 or any(g.kind != 'lactam' for g in principal):
                    continue
                carbonyls = {g.center for g in principal}
                shared = set.intersection(*[{n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors()
                                             if n.GetAtomicNum() == 7 and n.GetIdx() in core}
                                            for i in carbonyls])
                if len(shared) != 1 or any(g.kind not in {'lactam', 'pseudoketone'}
                                           for g in inventory.principal):
                    continue
        elif (not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in core)
              or any(g.kind != 'pseudoketone' for g in inventory.principal)):
            continue
        for row in records(mol, Chem, scope=scope, core_filter=core, principal=principal,
                           general_branches=True):
            component = row.construction
            if (row.principal_centers == {g.center for g in principal}
                    and row.hydrogen_budget_proved and row.annotation_coverage and component is not None
                    and component.atoms | component.embedded_atoms == scope):
                candidates.append(row)
    if len(candidates) != 1:
        return None
    return candidates[0]
