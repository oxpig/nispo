"""Functional-class names for mapped carbonyl oxides (C=O⁺–O⁻)."""

from dataclasses import replace
from functools import lru_cache

from .naming_result import SelectionLimitation


@lru_cache(maxsize=1)
def carbonyl_oxide_query(Chem):
    return Chem.MolFromSmarts("[CX3;+0]=[OX2;+1]-[OX1;-1]")


def carbonyl_oxide_decision(mol, Chem):
    """Render the carbonyl parent after accounting for the complete oxide unit."""
    matches = mol.GetSubstructMatches(carbonyl_oxide_query(Chem))
    if len(matches) != 1 or mol.GetStereoGroups() or len(Chem.GetMolFrags(mol)) != 1:
        return None
    carbon, inner, outer = matches[0]
    if ({(a.GetIdx(), a.GetFormalCharge()) for a in mol.GetAtoms() if a.GetFormalCharge()}
            != {(inner, 1), (outer, -1)} or any(
                a.GetIsotope() or a.GetNumRadicalElectrons() or a.GetChiralTag() or a.HasQuery()
                for a in mol.GetAtoms()) or any(b.GetStereo() or b.HasQuery() for b in mol.GetBonds())):
        return None
    # The functional-class suffix restores exactly this O–O bond and its
    # charges. Preserve the atom correspondence when the terminal O is removed.
    parent = Chem.RWMol(mol)
    parent.GetAtomWithIdx(inner).SetFormalCharge(0)
    parent.RemoveAtom(outer)
    Chem.SanitizeMol(parent)
    source_ids = tuple(i for i in range(mol.GetNumAtoms()) if i != outer)
    from .mapped_principal_general import mapped_principal_decision

    decision = mapped_principal_decision(parent, Chem)
    center = source_ids.index(carbon)
    if (decision is None or decision.principal_centers != {center}
            or decision.atoms != frozenset(range(len(source_ids)))
            or not decision.name.endswith(('one', 'aldehyde', 'al'))):
        return None
    return replace(decision, name=decision.name + ' oxide', method='functional-class',
                   rule='P-64/P-72',
                   atoms=frozenset(range(mol.GetNumAtoms())), principal_centers=frozenset({carbon}),
                   atom_locants=tuple((source_ids[i], loc) for i, loc in decision.atom_locants),
                   component_roles=(('carbonyl-parent', source_ids, ()), ('oxide', (outer,), ())),
                   preference_issues=(SelectionLimitation(
                       'carbonyl-oxide-parent', 'P-74.2.2.1.6',
                       'The functional-class oxide name preserves the original carbonyl oxide graph. '
                       'The intact dipolar parent required for a preferred name remains unresolved.'),))
