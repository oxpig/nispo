"""Retained fused parents whose ring states need source-specific numbering."""

from functools import lru_cache

from .naming_methods import MethodDecision


@lru_cache(maxsize=1)
def _templates():
    from rdkit import Chem
    from .parent_rules import _curated_fused_templates

    # P-25.2.1 Table 2.8 fixes both these parent graphs and indicated H.
    parents = {
        'perimidine': ('1H-perimidine', 'C1=Nc2cccc3cccc(c23)N1'),
        'selenoxanthene': ('9H-selenoxanthene', 'c1ccc2c(c1)Cc1ccccc1[Se]2'),
        'telluroxanthene': ('9H-telluroxanthene', 'c1ccc2c(c1)Cc1ccccc1[Te]2'),
    }
    templates = []
    for name, (preferred, smiles) in parents.items():
        canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
        for row in _curated_fused_templates(Chem):
            query, locants = row['mol'], row['locants']
            if (row['name'] == name and Chem.MolToSmiles(query) == canonical
                    and set(locants) == set(range(query.GetNumAtoms()))):
                templates.append((preferred, query, dict(locants)))
                break
    return tuple(templates)


def retained_fused_closed_decision(mol, Chem):
    """Admit fixed parent graphs and simple substitution at proved H sites.

    Each source atom's hydrogen budget and all external single bonds are
    checked. Other modifications and characteristic groups are left to the
    normal ring-state selection machinery.
    """
    if (mol.GetNumHeavyAtoms() > 48 or mol.GetStereoGroups()
            or len(Chem.GetMolFrags(mol)) != 1
            or any(a.GetAtomicNum() not in {6, 7, 9, 17, 34, 35, 52, 53} or a.GetFormalCharge()
                   or a.GetIsotope() or a.GetNumRadicalElectrons()
                   or int(a.GetChiralTag()) or a.HasQuery() for a in mol.GetAtoms())
            or any(int(b.GetStereo()) or int(b.GetBondDir()) or b.HasQuery() for b in mol.GetBonds())):
        return None
    from .formatting import format_prefixes, join_prefix_parent, prefix_alphabetization_key
    from .parent_selection import locant_key

    prefixes = {6: 'methyl', 9: 'fluoro', 17: 'chloro', 35: 'bromo', 53: 'iodo'}
    records = []
    for parent, query, locants in _templates():
        for match in mol.GetSubstructMatches(query, uniquify=False, maxMatches=1000):
            core = frozenset(match)
            components, terms = [], []
            valid = True
            for qidx, idx in enumerate(match):
                qatom, atom = query.GetAtomWithIdx(qidx), mol.GetAtomWithIdx(idx)
                external = [a for a in atom.GetNeighbors() if a.GetIdx() not in core]
                if (atom.GetTotalNumHs(includeNeighbors=True) + len(external)
                        != qatom.GetTotalNumHs(includeNeighbors=True)):
                    valid = False
                    break
                for branch in external:
                    allowed_host = (atom.GetAtomicNum() == 6
                                    or (parent == '1H-perimidine' and qatom.GetAtomicNum() == 7
                                        and qatom.GetTotalNumHs(includeNeighbors=True) == 1))
                    if (not allowed_host or branch.GetAtomicNum() not in prefixes
                            or branch.GetDegree() != 1 or branch.IsInRing()
                            or mol.GetBondBetweenAtoms(idx, branch.GetIdx()).GetBondTypeAsDouble() != 1
                            or (branch.GetAtomicNum() == 6
                                and branch.GetTotalNumHs(includeNeighbors=True) != 3)):
                        valid = False
                        break
                    components.append(branch.GetIdx())
                    terms.append((locants[qidx], prefixes[branch.GetAtomicNum()]))
                if not valid:
                    break
            if (not valid or len(terms) > 2
                    or core | frozenset(components) != frozenset(range(mol.GetNumAtoms()))):
                continue
            citation = sorted({text for _, text in terms}, key=prefix_alphabetization_key)
            score = (tuple(sorted((locant_key(loc) for loc, _ in terms))),
                     tuple(tuple(sorted(locant_key(loc) for loc, name in terms if name == kind))
                           for kind in citation))
            name = join_prefix_parent(format_prefixes(terms), parent) if terms else parent
            records.append((score, name, core, tuple((idx, locants[qidx]) for qidx, idx in enumerate(match)),
                            tuple(components)))
    if not records:
        return None
    best = min(row[0] for row in records)
    winners = [row for row in records if row[0] == best]
    if len({row[1] for row in winners}) != 1:
        return None
    _, name, core, mapping, components = winners[0]
    heavy = frozenset(range(mol.GetNumAtoms()))
    roles = [('retained-fused-parent', core, mapping)]
    roles.extend(('substituent', frozenset((idx,)), ()) for idx in components)
    return MethodDecision(name, 'retained-parent', core, heavy, 'P-25.2.1/Table 2.8', mapping,
                          component_roles=tuple(roles))
