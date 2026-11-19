"""Two identical benzoate anions with a shared divalent linker."""

from .branch_rules import collect_subtree
from .formatting import enclose_name_unit
from .naming_methods import MethodDecision, _Unit, _connector_name, _fragment
from .naming_result import SelectionLimitation


def linked_carboxylate_scope(mol, groups):
    """Allow only the two acid charges and proved neutral nitro components."""
    from .neutral_charge_components import _raw_neutral_charge_role_proofs

    charges = {a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge()}
    acid_charges = {g.ligand for g in groups if g.ligand >= 0
                    and mol.GetAtomWithIdx(g.ligand).GetFormalCharge() == -1}
    if len(acid_charges) != len(groups):
        return False
    if acid_charges == charges:
        return True
    neutral = set()
    for proof in _raw_neutral_charge_role_proofs(mol):
        if (proof.kind != 'nitro' or proof.net_charge or proof.charge_atom_ids & neutral
                or proof.charge_atom_ids & acid_charges):
            return False
        neutral.update(proof.charge_atom_ids)
    return charges == acid_charges | neutral


def linked_carboxylate_decision(mol, Chem, groups, scope):
    if (len(groups) != 2 or any(g.kind != 'carboxylate' or g.owner < 0 for g in groups)
            or not linked_carboxylate_scope(mol, groups)
            or any(a.GetIsotope() or a.GetChiralTag() for a in mol.GetAtoms())
            or any(b.GetStereo() for b in mol.GetBonds())):
        return None
    rings = mol.GetRingInfo().AtomRings()
    cores = [next((frozenset(r) for r in rings if g.owner in r), frozenset()) for g in groups]
    if (cores[0] & cores[1] or any(len(core) != 6 or any(
            not mol.GetAtomWithIdx(i).GetIsAromatic() or mol.GetAtomWithIdx(i).GetAtomicNum() != 6
            for i in core) for core in cores)):
        return None
    from .extended_inventory import selection_inventory
    from .mapped_ring_parents import records

    path = Chem.GetShortestPath(mol, groups[0].owner, groups[1].owner)
    linker = tuple(i for i in path if i not in cores[0] | cores[1])
    elements = tuple(mol.GetAtomWithIdx(i).GetAtomicNum() for i in linker)
    if (not linker or (elements != (16, 16) and any(number != 6 for number in elements))
            or any(mol.GetAtomWithIdx(i).IsInRing() for i in linker)
            or any(mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble() != 1
                   for a, b in zip(linker, linker[1:]))):
        return None
    occupied = set(linker)
    units = []
    for group, core, host in zip(groups, cores, (linker[0], linker[-1]), strict=True):
        attachments = [n.GetIdx() for n in mol.GetAtomWithIdx(host).GetNeighbors() if n.GetIdx() in core]
        if len(attachments) != 1:
            return None
        root = attachments[0]
        if mol.GetBondBetweenAtoms(root, host).GetBondTypeAsDouble() != 1:
            return None
        atoms = frozenset(collect_subtree(mol, root, host))
        if atoms & occupied or not core | set(group.atoms) <= atoms:
            return None
        fragment, original = _fragment(mol, Chem, atoms)
        if fragment is None:
            return None
        if group.ligand not in original or group.center not in original:
            return None
        # Number the capped acid while retaining any original neutral nitro
        # resonance pair. The two anionic oxygens are independently owned.
        anion = Chem.Mol(fragment)
        acid = fragment.GetAtomWithIdx(original.index(group.ligand))
        acid.SetFormalCharge(0)
        acid.SetNumExplicitHs(1)
        acid.SetNoImplicit(True)
        try:
            Chem.SanitizeMol(fragment)
        except (ValueError, RuntimeError):
            return None
        acids = selection_inventory(fragment).principal
        if (len(acids) != 1 or acids[0].kind != 'acid'
                or acids[0].center != original.index(group.center)):
            return None
        rows = records(fragment, Chem, principal=acids, general_branches=True)
        if len(rows) != 1 or not rows[0].hydrogen_budget_proved or not rows[0].annotation_coverage:
            return None
        row = rows[0]
        local_root = original.index(root)
        locant = dict(row.atom_locants).get(local_root)
        if not locant or row.parent != 'benzene' or not row.name.endswith('benzoic acid'):
            return None
        for atom in anion.GetAtoms():
            atom.SetAtomMapNum(0)
        anion.GetAtomWithIdx(local_root).SetAtomMapNum(1)
        signature = Chem.MolToSmiles(anion, canonical=True, isomericSmiles=True)
        name = row.name.removesuffix('benzoic acid') + 'benzoate'
        units.append(_Unit(atoms, root, host, signature, name, locant,
                           frozenset({group.center}), (), bool(row.prefixes)))
        occupied.update(atoms)
    left, right = units
    if (occupied != scope or (left.signature, left.parent, left.locant) !=
            (right.signature, right.parent, right.locant)):
        return None
    bridge = _connector_name(mol, frozenset(linker), units)
    if not bridge:
        return None
    repeated = ('bis' + enclose_name_unit(left.parent)) if left.substituted else 'di' + left.parent
    name = f"{left.locant},{right.locant}'-{enclose_name_unit(bridge)}{repeated}"
    rule = 'P-51.3/P-72.2.2.2.1'
    locants = tuple((unit.root, unit.locant) for unit in units)
    roles = tuple(('carboxylate', unit.atoms, ((unit.root, unit.locant),)) for unit in units)
    return MethodDecision(name, 'multiplicative', left.centers | right.centers, scope, rule,
                          locants, component_roles=(*roles, ('connector', frozenset(linker), ())),
                          preference_issues=(SelectionLimitation(
                              'principal-parent-and-method', rule,
                              'The two anion sites and linker are mapped. Complete preferred comparison remains open.'),))
