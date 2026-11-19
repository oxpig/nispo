"""Functional-class names for neutral sulfuric diamides (P-67.1.2.6.1)."""

from .molecule_iteration import iter_atoms, iter_bonds
from .formatting import format_prefixes


def sulfuric_diamide_name(mol):
    if not any(a.GetAtomicNum() == 16 for a in iter_atoms(mol)):
        return ""
    if any(
        a.GetAtomicNum() not in {1, 6, 7, 8, 9, 16, 17, 35, 53}
        or a.GetFormalCharge()
        or a.GetIsotope()
        or a.GetNumRadicalElectrons()
        or int(a.GetChiralTag())
        for a in iter_atoms(mol)
    ):
        return ""
    if any(int(b.GetStereo()) for b in iter_bonds(mol)):
        return ""
    sulfurs = [a for a in iter_atoms(mol) if a.GetAtomicNum() == 16]
    if len(sulfurs) != 1:
        return ""
    sulfur = sulfurs[0]
    oxygens, nitrogens = [], []
    for n in sulfur.GetNeighbors():
        order = mol.GetBondBetweenAtoms(
            sulfur.GetIdx(), n.GetIdx()
        ).GetBondTypeAsDouble()
        if n.GetAtomicNum() == 8 and order == 2 and n.GetDegree() == 1:
            oxygens.append(n)
        elif n.GetAtomicNum() == 7 and order == 1 and not n.IsInRing():
            nitrogens.append(n)
        else:
            return ""
    if len(oxygens) != 2 or len(nitrogens) != 2:
        return ""
    # Do not promote this noncarbon-acid amide over a senior ester or an
    # ordinary carboxamide. Cyclic lactams remain junior pseudoketones.
    for atom in iter_atoms(mol):
        if atom.GetAtomicNum() != 6 or not any(
            n.GetAtomicNum() == 8
            and mol.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble()
            == 2
            for n in atom.GetNeighbors()
        ):
            continue
        if any(
            mol.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble()
            == 1
            and (n.GetAtomicNum() == 8 or n.GetAtomicNum() == 7 and not n.IsInRing())
            for n in atom.GetNeighbors()
        ):
            return ""
    from .parent_rules import _external_substituent_names

    sides = [
        _external_substituent_names(mol, n.GetIdx(), sulfur.GetIdx()) for n in nitrogens
    ]
    if any(side is None or len(side) > 2 for side in sides):
        return ""
    # Number the equivalent N sites using the first differing alphabetical
    # substituent, then alphabetize all substituents in the complete name.
    import re

    def key(name):
        return re.sub(r"[^a-z]", "", name.lower())

    sides.sort(key=lambda side: (not bool(side), tuple(sorted(map(key, side)))))
    prefixes = [
        (locant, name) for locant, side in zip(("N", "N'"), sides) for name in side
    ]
    text = format_prefixes(prefixes)
    return text + "sulfuric diamide"


def mapped_annular_sulfamide_decision(mol, Chem, group):
    """Cite -SO2NH2 on its original numbered ring nitrogen (P-66.1.1.2)."""
    from dataclasses import replace
    from .branch.preferred_acyl import ring_acid_derivative_parent
    from .branch_rules import collect_subtree
    from .component_mapping import (
        _general_branch_stereo_is_cited, decorate_component, fully_covered,
    )
    from .mapped_ring_parents import records
    from .naming_methods import MethodDecision
    from .naming_result import SelectionLimitation

    atoms = [mol.GetAtomWithIdx(i) for i in group.atoms]
    nitrogens = [a for a in atoms if a.GetAtomicNum() == 7]
    annular = [a.GetIdx() for a in nitrogens if a.IsInRing() and a.GetDegree() == 3]
    terminal = [a for a in nitrogens
                if a.GetDegree() == 1 and a.GetTotalNumHs(includeNeighbors=True) == 2]
    if len(annular) != 1 or len(terminal) != 1:
        return None
    nitrogen, = annular
    scope = frozenset(collect_subtree(mol, nitrogen, group.center))
    suffix_atoms = frozenset(group.atoms) - {nitrogen}
    if (scope & suffix_atoms or not fully_covered(mol, scope, suffix_atoms)
            or any(mol.GetAtomWithIdx(i).GetIsotope() for i in suffix_atoms)):
        return None
    choices = records(mol, Chem, scope=scope, attachment=nitrogen, general_branches=True)
    if len(choices) != 1:
        return None
    row, = choices
    component = row.construction
    if (component is None or not row.hydrogen_budget_proved or not row.annotation_coverage
            or component.atoms | component.embedded_atoms != scope):
        return None
    parent = ring_acid_derivative_parent(row, nitrogen, 'sulfonamide')
    name = decorate_component(mol, Chem, replace(component, parent=parent)) if parent else ''
    if not name or not _general_branch_stereo_is_cited(mol, Chem, scope, name):
        return None
    rule = 'P-41/P-66.1.1.2/P-67.1.2.6.1'
    return MethodDecision(
        name, 'substitutive', frozenset({group.center}), scope | suffix_atoms,
        rule=rule, atom_locants=row.atom_locants,
        component_roles=(('cyclic-parent-and-branches', scope, row.atom_locants),
                         ('sulfonamide-suffix', suffix_atoms, ())),
        preference_issues=(SelectionLimitation(
            'annular-sulfamide', rule,
            'The original ring nitrogen and external sulfonamide are mapped; preferred method comparison remains unresolved.'),))
