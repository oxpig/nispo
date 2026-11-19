"""P-66/P-68 carboxamide suffixes on a mapped acyclic hydrazine parent.

The carbonic diamide motif alone does not select urea: the N-N parent and
the external amide have different numbering and construction boundaries.
"""
from .component_mapping import MappedComponent, decorate_component, fully_covered, mapped_branch
from .group14_mononuclear_methods import _carbon_organyl
from .molecule_iteration import iter_atoms
from .naming_methods import MethodDecision


RULE = 'P-44/P-66.1.1.1.1.3/P-66.1.1.3/P-68.3.1.2.4'
_SUFFIX = {8: 'carboxamide', 16: 'carbothioamide',
           34: 'carboselenoamide', 52: 'carbotelluroamide'}


def hydrazine_carboxamide_decision(mol, Chem):
    """Own one C(=X)(N)(N-N) and disjoint carbon/halo-carbon N arms.

    Core isotope/stereo, imine, second acyl, and carbohydrazide operations
    need their own mapped suffix constructions and decline this producer.
    """
    if mol.GetNumAtoms() > 128 or mol.GetStereoGroups():
        return None
    atoms = tuple(iter_atoms(mol))
    if any(a.GetAtomicNum() == 1 or a.GetFormalCharge()
           or a.GetNumRadicalElectrons() or int(a.GetChiralTag()) for a in atoms):
        return None
    nitrogens = [a for a in atoms if a.GetAtomicNum() == 7]
    if len(nitrogens) != 3 or any(a.IsInRing() or a.GetIsotope()
                                or a.GetTotalValence() != 3 for a in nitrogens):
        return None
    pairs = [b for b in mol.GetBonds() if b.GetBeginAtom().GetAtomicNum() == 7
             and b.GetEndAtom().GetAtomicNum() == 7]
    if len(pairs) != 1 or pairs[0].GetBondType() != Chem.BondType.SINGLE:
        return None
    hydrazine = frozenset({pairs[0].GetBeginAtomIdx(), pairs[0].GetEndAtomIdx()})
    sites = []
    for carbon in atoms:
        if (carbon.GetAtomicNum() != 6 or carbon.IsInRing() or carbon.GetDegree() != 3
                or carbon.GetTotalValence() != 4 or carbon.GetTotalNumHs()):
            continue
        ci = carbon.GetIdx()
        terminal = [a for a in carbon.GetNeighbors() if a.GetAtomicNum() in _SUFFIX
                    and a.GetDegree() == 1 and a.GetTotalValence() == 2
                    and mol.GetBondBetweenAtoms(ci, a.GetIdx()).GetBondType() == Chem.BondType.DOUBLE]
        ns = [a.GetIdx() for a in carbon.GetNeighbors() if a.GetAtomicNum() == 7
              and mol.GetBondBetweenAtoms(ci, a.GetIdx()).GetBondType() == Chem.BondType.SINGLE]
        if len(terminal) == 1 and len(ns) == 2 and len(hydrazine & set(ns)) == 1:
            sites.append((ci, terminal[0].GetIdx(), tuple(ns)))
    if len(sites) != 1:
        return None
    ci, xi, ns = sites[0]
    host = next(i for i in ns if i in hydrazine)
    other = next(i for i in hydrazine if i != host)
    amide = next(i for i in ns if i != host)
    core = hydrazine | {ci, xi, amide}
    if any(mol.GetAtomWithIdx(i).GetIsotope() for i in core):
        return None
    mapping = {host: '1', other: '2', amide: 'N', ci: '', xi: ''}
    prefixes, branches, branch_roles, ownership = [], set(), [], []
    parent_substituted = False
    for ni in (host, other, amide):
        nitrogen = mol.GetAtomWithIdx(ni)
        for neighbor in nitrogen.GetNeighbors():
            root = neighbor.GetIdx()
            if root in core:
                continue
            group = _carbon_organyl(mol, root, ni)
            if not group or group & (core | branches):
                return None
            # Carbon isotope descriptors are constructed locally by the
            # mapped branch writer. Other isotope-bearing arms stay separate.
            if any(mol.GetAtomWithIdx(i).GetIsotope()
                   and mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in group):
                return None
            text, owned = mapped_branch(mol, Chem, root, ni)
            if not text or frozenset(owned) != group:
                return None
            prefixes.append((mapping[ni], text))
            parent_substituted |= ni in hydrazine
            branches.update(group)
            branch_roles.append(('nitrogen-substituent', group, ()))
            ownership.append((root, ni, group, text))
    if not fully_covered(mol, core, branches):
        return None
    # P-68.3.1.2.4 uses the bare suffix for an unsubstituted hydrazine
    # skeleton, and explicit 1/2 locants when that skeleton is substituted.
    ending = ('hydrazine-1-' if parent_substituted else 'hydrazine') + _SUFFIX[mol.GetAtomWithIdx(xi).GetAtomicNum()]
    construction = MappedComponent(ending, tuple(mapping.items()), core,
        tuple(prefixes), frozenset(branches), systematic_parent=True,
        nitrogen_site_locants=True, prefix_branch_ownership=tuple(ownership))
    name = decorate_component(mol, Chem, construction)
    if not name:
        return None
    roles = (('hydrazine-parent', hydrazine, ((host, '1'), (other, '2'))),
             ('external-carboxamide', frozenset({ci, xi, amide}),
              ((ci, ''), (xi, ''), (amide, 'N'))), *branch_roles)
    return MethodDecision(name, 'substitutive', frozenset({ci}),
        core | frozenset(branches), RULE, tuple(mapping.items()), component_roles=roles)
