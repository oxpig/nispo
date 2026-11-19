"""Mapped, bounded naming of neutral N-substituted guanidines."""

from __future__ import annotations

from collections import defaultdict
from itertools import permutations

from .molecule_iteration import iter_atoms, iter_bonds
from .component_mapping import mapped_branch
from .formatting import prefix_alphabetization_key
from .naming_methods import MethodDecision
from .substituent_multiplication import multiply_substituent


RULE = "P-66.4.1.2.1.1/P-66.4.1.2.1.2/P-14.4/P-14.5/P-16"
_N_LOCANTS = ("N", "N′", "N′′")
_LOCANT_RANK = {locant: index for index, locant in enumerate(_N_LOCANTS)}


def guanidine_decision(mol, Chem):
    """Return a complete decision for one neutral, explicitly drawn guanidine.

    The C=N bond in the input is retained and labels the imino site N″ under
    the guanidine-specific locant pattern. The two amino sites receive N/N′
    by a lowest-locant comparison in substituent citation order. No tautomer
    is inferred, generated, or normalized.
    """
    if mol is None or mol.GetNumHeavyAtoms() > 128 or mol.GetStereoGroups():
        return None
    if any(atom.GetAtomicNum() == 1 or atom.GetFormalCharge()
           or atom.GetNumRadicalElectrons() or atom.GetIsotope()
           or int(atom.GetChiralTag()) for atom in iter_atoms(mol)):
        return None
    if any(bond.GetStereo() != Chem.BondStereo.STEREONONE
           or bond.GetBondDir() != Chem.BondDir.NONE for bond in iter_bonds(mol)):
        return None

    sites = []
    for carbon in iter_atoms(mol):
        if carbon.GetAtomicNum() != 6 or carbon.IsInRing():
            continue
        n_single, n_double, other = [], [], []
        for neighbor in carbon.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(carbon.GetIdx(), neighbor.GetIdx())
            if neighbor.GetAtomicNum() == 7 and bond.GetBondType() == Chem.BondType.SINGLE:
                n_single.append(neighbor.GetIdx())
            elif neighbor.GetAtomicNum() == 7 and bond.GetBondType() == Chem.BondType.DOUBLE:
                n_double.append(neighbor.GetIdx())
            else:
                other.append(neighbor.GetIdx())
        if len(n_single) == 2 and len(n_double) == 1:
            sites.append((carbon.GetIdx(), tuple(n_single), n_double[0], tuple(other)))
    if len(sites) != 1:
        return None
    carbon, amino_sites, imino_site, other = sites[0]
    if other:
        return None

    core = {carbon, imino_site, *amino_sites}
    branch_rows = []
    branch_owners = set()
    branch_by_n = defaultdict(list)
    for nidx in (imino_site, *amino_sites):
        nitrogen = mol.GetAtomWithIdx(nidx)
        if nitrogen.GetAtomicNum() != 7 or nitrogen.GetFormalCharge() or nitrogen.IsInRing():
            return None
        roots = [neighbor.GetIdx() for neighbor in nitrogen.GetNeighbors()
                 if neighbor.GetIdx() != carbon and neighbor.GetAtomicNum() > 1]
        max_substituents = 1 if nidx == imino_site else 2
        if len(roots) > max_substituents:
            return None
        if nitrogen.GetTotalNumHs(includeNeighbors=True) + len(roots) != max_substituents:
            return None
        for root in roots:
            name, atoms = _simple_organyl(mol, Chem, root, nidx)
            if not name or not atoms or atoms & (core | branch_owners):
                return None
            branch_owners.update(atoms)
            branch_by_n[nidx].append(name)
            branch_rows.append((nidx, name, frozenset(atoms)))

    heavy = frozenset(atom.GetIdx() for atom in iter_atoms(mol)
                      if atom.GetAtomicNum() > 1)
    if core | branch_owners != set(heavy) or not branch_rows:
        return None

    component_names = sorted({name for _, name, _ in branch_rows},
                             key=prefix_alphabetization_key)
    assignments = []
    for amino_locants in permutations(("N", "N′")):
        locant_by_n = {imino_site: "N′′",
                       amino_sites[0]: amino_locants[0],
                       amino_sites[1]: amino_locants[1]}
        score = tuple(tuple(sorted(_LOCANT_RANK[locant_by_n[nidx]]
                                   for nidx, name, _ in branch_rows if name == component))
                      for component in component_names)
        rendered = []
        for component in component_names:
            locants = sorted((locant_by_n[nidx] for nidx, name, _ in branch_rows
                              if name == component), key=_LOCANT_RANK.__getitem__)
            operation = multiply_substituent(component, len(locants), nature="simple")
            if operation is None:
                return None
            rendered.append(",".join(locants) + "-" + operation.text)
        assignments.append((score, "-".join(rendered) + "guanidine", locant_by_n))

    best_score = min(row[0] for row in assignments)
    winners = [row for row in assignments if row[0] == best_score]
    winner_names = {row[1] for row in winners}
    if len(winner_names) != 1:
        return None
    name = winner_names.pop()
    if len(winners) == 1:
        locant_by_n = winners[0][2]
    else:
        # Symmetric amino sites may leave two equivalent mapped assignments.
        # Use canonical ranks only to stabilize the returned map; the cited
        # name has already been proven identical for both assignments.
        ranks = tuple(Chem.CanonicalRankAtoms(mol, breakTies=True))
        ordered_amino = sorted(amino_sites, key=lambda index: ranks[index])
        locant_by_n = {imino_site: "N′′", ordered_amino[0]: "N", ordered_amino[1]: "N′"}

    atom_locants = ((imino_site, "N′′"),
                    (amino_sites[0], locant_by_n[amino_sites[0]]),
                    (amino_sites[1], locant_by_n[amino_sites[1]]))
    roles = [("guanidine-core", frozenset(core), atom_locants)]
    for nidx, component, atoms in branch_rows:
        roles.append(("N-substituent-at-" + locant_by_n[nidx], atoms, ()))
    return MethodDecision(name, "substitutive", frozenset({carbon}), heavy, RULE,
                          atom_locants, component_roles=tuple(roles))


def _simple_organyl(mol, Chem, root, nitrogen):
    text, atoms = mapped_branch(mol, Chem, root, nitrogen)
    if not text or not atoms or any(mol.GetAtomWithIdx(index).GetAtomicNum() != 6
                                    for index in atoms):
        return "", frozenset()
    ring_atoms = {index for index in atoms if mol.GetAtomWithIdx(index).IsInRing()}
    if ring_atoms:
        if (len(atoms) != 6 or len(ring_atoms) != 6
                or not all(mol.GetAtomWithIdx(index).GetIsAromatic() for index in atoms)):
            return "", frozenset()
        for index in atoms:
            for neighbor in mol.GetAtomWithIdx(index).GetNeighbors():
                if neighbor.GetIdx() not in atoms | {nitrogen} and neighbor.GetAtomicNum() > 1:
                    return "", frozenset()
        if any(mol.GetBondBetweenAtoms(left, right).GetBondType() != Chem.BondType.AROMATIC
               for left in atoms for right in atoms if left < right
               and mol.GetBondBetweenAtoms(left, right) is not None):
            return "", frozenset()
        if text != "phenyl":
            return "", frozenset()
        return text, frozenset(atoms)

    degrees = {index: sum(neighbor.GetIdx() in atoms
                          for neighbor in mol.GetAtomWithIdx(index).GetNeighbors())
               for index in atoms}
    if (any(degree > 2 for degree in degrees.values())
            or degrees.get(root) != (0 if len(atoms) == 1 else 1)
            or len(atoms) > 1 and sum(degree == 1 for degree in degrees.values()) != 2
            or any(mol.GetBondBetweenAtoms(left, right).GetBondType() != Chem.BondType.SINGLE
                   for left in atoms for right in atoms if left < right
                   and mol.GetBondBetweenAtoms(left, right) is not None)):
        return "", frozenset()
    return text, frozenset(atoms)
