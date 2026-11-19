"""Original-graph geminal amide N locants (Blue Book P-66.1.1.3.1.2).

Substitution count precedes prefix citation. Isotope numbering is considered
only after those chemical choices tie. Remaining distinct labelled arm graphs
are an unresolved choice, never an atom-index tie-breaker.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import permutations

from .branch_rules import collect_subtree
from .component_mapping import mapped_branch
from .formatting import prefix_alphabetization_key
from .parent_annotations import isotope_numbering_key


@dataclass(frozen=True, slots=True)
class AmideNitrogenLocants:
    nitrogen_locants: tuple[tuple[int, str], ...]
    equivalent_geminal_orders: tuple[tuple[tuple[int, ...], ...], ...] = ()
    rule: str = "P-66.1.1.3.1.1/P-66.1.1.3.1.2/P-14.5/P-82.5.2"


def external_amide_nitrogen_locants(mol, Chem, groups, parent_locants, *, locant_series=None):
    """Assign source-bounded N/N′ parent-site locants, or return None.

    Each selected group is an ordinary external carboxamide on a mapped
    carbon host. The caller owns parent selection and complete atom coverage.
    Nongeminal sites preserve their established unrendered N+position form.
    """
    groups = tuple(groups)
    mapping = dict(parent_locants)
    if not groups or any(g.kind != 'amide' or g.owner not in mapping for g in groups):
        return None
    by_host = defaultdict(list)
    for group in groups:
        by_host[group.owner].append(group)
    labels, equivalent = {}, []
    for host, attached in by_host.items():
        locant = str(mapping[host])
        if len(attached) == 1:
            labels[attached[0].ligand] = 'N' + (locant if len(groups) > 1 else '')
            continue
        if len(attached) > 4:
            return None
        arms = []
        for group in attached:
            nitrogen = mol.GetAtomWithIdx(group.ligand)
            if (nitrogen.GetAtomicNum() != 7 or nitrogen.IsInRing()
                    or nitrogen.GetFormalCharge() or nitrogen.GetNumRadicalElectrons()
                    or nitrogen.GetTotalValence() != 3
                    or mol.GetBondBetweenAtoms(group.center, group.ligand).GetBondTypeAsDouble() != 1):
                return None
            # Unlike isotope profiles must first be partitioned under P-82.
            # An identical-profile set has a separate typed suffix writer;
            # this operation only assigns its N substitution locants.
            if any(mol.GetAtomWithIdx(i).GetIsotope()
                    or any(n.GetAtomicNum() == 1 and n.GetIsotope()
                           for n in mol.GetAtomWithIdx(i).GetNeighbors()) for i in group.atoms):
                from .amide_isotopes import amide_isotope_partition
                partition = amide_isotope_partition(mol, attached)
                if (partition is None or partition.selection.barrier
                        or len(partition.selection.multiplicity_groups) != 1):
                    return None
            roots = tuple(n.GetIdx() for n in nitrogen.GetNeighbors()
                          if n.GetIdx() != group.center and n.GetAtomicNum() > 1)
            if len(roots) > 2 or any(mol.GetBondBetweenAtoms(group.ligand, i).GetBondTypeAsDouble() != 1 for i in roots):
                return None
            prefixes, owned = [], set(group.atoms)
            owned.update(n.GetIdx() for i in group.atoms
                for n in mol.GetAtomWithIdx(i).GetNeighbors()
                if n.GetAtomicNum() == 1 and n.GetIsotope())
            for root in roots:
                text, atoms = mapped_branch(mol, Chem, root, group.ligand)
                if not text or root not in atoms or owned & atoms:
                    return None
                prefixes.append(prefix_alphabetization_key(text)[:3])
                owned.update(atoms)
            expected = frozenset(collect_subtree(mol, group.center, host))
            if set(expected) != owned or expected & set(mapping):
                return None
            scratch = Chem.Mol(mol)
            for atom in scratch.GetAtoms():
                atom.SetAtomMapNum(0)
            identity = Chem.MolFragmentToSmiles(scratch, atomsToUse=sorted(expected),
                rootedAtAtom=group.ligand, canonical=True, isomericSmiles=True)
            arms.append((group, tuple(prefixes), expected, identity))
        if any(arms[i][2] & arms[j][2] for i in range(len(arms)) for j in range(i + 1, len(arms))):
            return None
        words = sorted({word for _group, prefixes, _owned, _identity in arms for word in prefixes})
        choices = []
        for order in permutations(range(len(arms))):
            occupancy = tuple(-len(arms[index][1]) for index in order)
            citation = tuple(tuple(position for position, index in enumerate(order)
                for word in arms[index][1] if word == target) for target in words)
            choices.append(((occupancy, citation), order))
        chemical_key = min(key for key, _order in choices)
        live = [order for key, order in choices if key == chemical_key]
        numbered = []
        for order in live:
            n_mapping = {arms[index][0].ligand: 'N' + "'" * prime + locant
                         for prime, index in enumerate(order)}
            isotope_mapping = dict(mapping)
            isotope_mapping.update(n_mapping)
            for group, _prefixes, _atoms, _identity in arms:
                for idx in group.atoms:
                    isotope_mapping.setdefault(idx, locant)
            extra = {} if locant_series is None else {'locant_series':locant_series}
            key = isotope_numbering_key(mol, isotope_mapping, include_substituent_groups=True, **extra)
            if key is None:
                return None
            numbered.append((key, order, n_mapping))
        isotope_key = min(key for key, _order, _mapping in numbered)
        finalists = [(order, n_mapping) for key, order, n_mapping in numbered if key == isotope_key]
        # A choice between distinct remaining physical/annotated arms needs
        # another proved criterion. Only equivalent rooted graphs may merge.
        if any(len({arms[order[position]][3] for order, _mapping in finalists}) != 1
               for position in range(len(arms))):
            return None
        labels.update(finalists[0][1])
        equivalent.append(tuple(tuple(arms[index][0].ligand for index in order)
                                for order, _mapping in finalists))
    return AmideNitrogenLocants(tuple(labels.items()), tuple(equivalent))
