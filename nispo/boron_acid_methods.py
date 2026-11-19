"""Mapped retained functional-parent decisions for simple boron oxoacids.

This bounded method supports neutral, mononuclear boronic and borinic acids
with pure hydrocarbon organyl branches: acyclic saturated carbon trees and
simple phenyl-type branches.  P-67/P-68 define the acid parent and organyl
substitution; mapped branch assembly preserves original C-isotope/stereo
facts.  It does not claim general boron nomenclature.
"""

from dataclasses import dataclass

from .molecule_iteration import iter_atoms, iter_bonds
from .naming_methods import MethodDecision

RULE = "P-41/P-42.3/P-67.1.1.2/P-67.1.2.1/P-68.1.4.1"


@dataclass(frozen=True, slots=True)
class BoronAcidSite:
    boron: int
    carbon_roots: tuple[int, ...]
    carbon_atoms: tuple[int, ...]
    kind: str
    hydroxy: tuple[int, ...]


def _carbon_branch(mol, root, boron):
    """Collect and classify one B-bound hydrocarbon organyl graph."""
    pending, atoms = [root], set()
    while pending:
        idx = pending.pop()
        if idx in atoms:
            continue
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() != 6:
            return None
        atoms.add(idx)
        for neighbor in atom.GetNeighbors():
            nidx = neighbor.GetIdx()
            if nidx == boron or neighbor.GetAtomicNum() == 1:
                continue
            if neighbor.GetAtomicNum() != 6:
                return None
            pending.append(nidx)
    if any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in (mol.GetAtomWithIdx(i) for i in atoms)):
        return None
    if any(mol.GetAtomWithIdx(i).GetIsotope() and mol.GetAtomWithIdx(i).GetAtomicNum() != 6
           for i in atoms):
        return None
    if any(n.GetAtomicNum() == 1 and n.GetIsotope()
           for i in atoms for n in mol.GetAtomWithIdx(i).GetNeighbors()):
        return None
    ring_atoms = {i for i in atoms if mol.GetAtomWithIdx(i).IsInRing()}
    if ring_atoms:
        rings = [tuple(i for i in ring if i in atoms)
                 for ring in mol.GetRingInfo().AtomRings() if set(ring) & atoms]
        if (len(rings) != 1 or len(ring_atoms) != 6 or len(rings[0]) != 6
                or set(rings[0]) != ring_atoms
                or any(not mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring_atoms)):
            return None
        if any(mol.GetAtomWithIdx(i).GetIsAromatic() != (i in ring_atoms) for i in atoms):
            return None
    else:
        if any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in atoms):
            return None
    for idx in atoms:
        atom = mol.GetAtomWithIdx(idx)
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            if other.GetIdx() == boron:
                if idx != root or bond.GetBondTypeAsDouble() != 1:
                    return None
            elif other.GetAtomicNum() == 1:
                continue
            elif bond.GetIsAromatic():
                if idx not in ring_atoms or other.GetIdx() not in ring_atoms:
                    return None
            elif bond.GetBondTypeAsDouble() != 1:
                return None
        if idx not in ring_atoms and atom.GetHybridization().name != "SP3":
            return None
    edge_count = sum(1 for bond in iter_bonds(mol)
                     if bond.GetBeginAtomIdx() in atoms and bond.GetEndAtomIdx() in atoms)
    if edge_count != len(atoms) - 1 + (1 if ring_atoms else 0):
        return None
    return frozenset(atoms)


def _site(mol):
    if (mol.GetNumAtoms() > 64 or mol.GetStereoGroups()
            or any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in iter_atoms(mol))
            or any(a.GetAtomicNum() == 1 and a.GetIsotope() for a in iter_atoms(mol))):
        return None
    borons = [a for a in iter_atoms(mol) if a.GetAtomicNum() == 5]
    if len(borons) != 1:
        return None
    b = borons[0]
    if (b.GetFormalCharge() or b.IsInRing() or b.GetIsotope()
            or int(b.GetChiralTag()) or b.GetTotalValence() != 3):
        return None
    carbons, hydroxy = [], []
    for a in b.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(b.GetIdx(), a.GetIdx())
        if bond.GetBondTypeAsDouble() != 1:
            return None
        if a.GetAtomicNum() == 6:
            carbons.append(a.GetIdx())
        elif a.GetAtomicNum() in {8, 16, 34, 52}:
            if (a.GetFormalCharge() or a.IsInRing() or a.GetDegree() != 1
                    or a.GetIsotope() or int(a.GetChiralTag())
                    or a.GetTotalNumHs(includeNeighbors=True) != 1):
                return None
            hydroxy.append(a.GetIdx())
        else:
            return None
    if (len(carbons), len(hydroxy)) not in {(1, 2), (2, 1)}:
        return None
    kind = "boronic acid" if len(hydroxy) == 2 else "borinic acid"
    replacements = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in hydroxy
                    if mol.GetAtomWithIdx(i).GetAtomicNum() != 8]
    if replacements:
        from collections import Counter
        from .constants import MULT
        counts = Counter(replacements)
        words = {16: 'thio', 34: 'seleno', 52: 'telluro'}
        infix = ''.join(('' if count == 1 else MULT[count]) + words[element]
                        for element, count in sorted(counts.items(), key=lambda row: words[row[0]]))
        kind = ('borono' if len(hydroxy) == 2 else 'borino') + infix + 'ic acid'
    if b.GetTotalNumHs(includeNeighbors=True) != 0:
        return None
    branches = []
    covered = {b.GetIdx(), *hydroxy}
    for root in carbons:
        chain = _carbon_branch(mol, root, b.GetIdx())
        if chain is None or any(chain & prior[0] for prior in branches):
            return None
        branches.append((chain, root))
        covered.update(chain)
    if covered != {a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1}:
        return None
    return BoronAcidSite(b.GetIdx(), tuple(root for _, root in branches),
                         tuple(sorted(covered - {b.GetIdx(), *hydroxy})), kind,
                         tuple(hydroxy)), branches


def recognize(mol):
    found = _site(mol)
    return None if found is None else found[0]


def _branch_signature(mol, Chem, atoms, root, *, isotopes, stereo):
    """Canonical rooted branch identity, independent of input atom order."""
    rw = Chem.RWMol(mol)
    keep = set(atoms)
    remove = [a.GetIdx() for a in iter_atoms(mol) if a.GetIdx() not in keep]
    root_after = root - sum(idx < root for idx in remove)
    for idx in reversed(remove):
        rw.RemoveAtom(idx)
    for atom in rw.GetAtoms():
        atom.SetAtomMapNum(0)
        if not isotopes:
            atom.SetIsotope(0)
        if not stereo:
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            if atom.HasProp('_CIPCode'):
                atom.ClearProp('_CIPCode')
    if not stereo:
        for bond in rw.GetBonds():
            bond.SetStereo(Chem.BondStereo.STEREONONE)
            bond.SetBondDir(Chem.BondDir.NONE)
    anchor = rw.AddAtom(Chem.Atom(0))
    rw.AddBond(root_after, anchor, Chem.BondType.SINGLE)
    branch = rw.GetMol()
    Chem.SanitizeMol(branch)
    # `stereo` has already been cleared explicitly when it is excluded; keep
    # isomeric SMILES enabled so isotope masses remain in the signature.
    return Chem.MolToSmiles(branch, canonical=True, isomericSmiles=True)


def _ordered_rendered(mol, Chem, rendered, branches, prefix_alphabetization_key):
    """Order two organyl units only when Blue Book rules settle their tie.

    P-14.5 supplies the ordinary alphabetic key but excludes isotope and
    stereochemical descriptors. P-82.2.2.1 specifically puts one otherwise
    identical isotope-modified substituent before its unmodified analogue.
    Other equal-key, nonidentical cases remain outside this method's proof.
    """
    if len(rendered) < 2:
        return rendered
    if len(rendered) != 2:
        return None
    (name_a, atoms_a), (name_b, atoms_b) = rendered
    root_a = next(root for atoms, root in branches if atoms == atoms_a)
    root_b = next(root for atoms, root in branches if atoms == atoms_b)
    base_a = _branch_signature(mol, Chem, atoms_a, root_a,
                               isotopes=False, stereo=False)
    base_b = _branch_signature(mol, Chem, atoms_b, root_b,
                               isotopes=False, stereo=False)
    full_a = _branch_signature(mol, Chem, atoms_a, root_a,
                               isotopes=True, stereo=True)
    full_b = _branch_signature(mol, Chem, atoms_b, root_b,
                               isotopes=True, stereo=True)
    alpha_a = prefix_alphabetization_key(name_a)[:3]
    alpha_b = prefix_alphabetization_key(name_b)[:3]
    if base_a != base_b:
        if alpha_a == alpha_b:
            return None
        return sorted(rendered, key=lambda row: prefix_alphabetization_key(row[0])[:3])
    if full_a == full_b:
        # Identical groups can be combined. Their relative role order has no
        # effect on the cited name.
        return rendered
    iso_a = _branch_signature(mol, Chem, atoms_a, root_a,
                              isotopes=True, stereo=False)
    iso_b = _branch_signature(mol, Chem, atoms_b, root_b,
                              isotopes=True, stereo=False)
    has_iso_a = any(mol.GetAtomWithIdx(i).GetIsotope() for i in atoms_a)
    has_iso_b = any(mol.GetAtomWithIdx(i).GetIsotope() for i in atoms_b)
    has_stereo_a = any(int(mol.GetAtomWithIdx(i).GetChiralTag()) for i in atoms_a)
    has_stereo_b = any(int(mol.GetAtomWithIdx(i).GetChiralTag()) for i in atoms_b)
    if (iso_a != iso_b and has_iso_a != has_iso_b
            and not has_stereo_a and not has_stereo_b):
        # Specifically supported by P-82.2.2.1: modified first.
        return rendered if has_iso_a else list(reversed(rendered))
    return None


def boron_acid_decision(mol, Chem=None):
    """Return a whole-graph substitutive decision or ``None``.

    This public decision wrapper is suitable for a caller to place in the
    principal-method pipeline.  The input graph is already parsed/sanitized;
    `Chem` is accepted for consistency with other method APIs.
    """
    found = _site(mol)
    if found is None:
        return None
    if Chem is None:
        from rdkit import Chem
    site, branches = found
    from .component_mapping import MappedComponent, decorate_component, mapped_branch
    from .formatting import format_prefixes, prefix_alphabetization_key

    core_atoms = frozenset({site.boron, *site.hydroxy})
    core = decorate_component(mol, Chem, MappedComponent(
        site.kind, tuple((idx, "") for idx in core_atoms), core_atoms))
    if not core:
        return None
    rendered = []
    for atoms, root in branches:
        prefix, owned = mapped_branch(mol, Chem, root, site.boron)
        if not prefix or frozenset(i for i in owned if mol.GetAtomWithIdx(i).GetAtomicNum() > 1) != atoms:
            return None
        rendered.append((prefix, atoms))
    rendered = _ordered_rendered(mol, Chem, rendered, branches,
                                 prefix_alphabetization_key)
    if rendered is None:
        return None
    if len(rendered) == 1:
        # The organyl name is substituted for B-H in the retained acid name;
        # it is not a locanted prefix on a separate carbon parent.
        substituents = rendered[0][0]
    else:
        if (len(rendered) == 2 and rendered[0][0] != rendered[1][0]
                and _branch_signature(mol, Chem, rendered[0][1],
                                      next(root for atoms, root in branches if atoms == rendered[0][1]),
                                      isotopes=False, stereo=False)
                == _branch_signature(mol, Chem, rendered[1][1],
                                     next(root for atoms, root in branches if atoms == rendered[1][1]),
                                     isotopes=False, stereo=False)):
            # Here the order is explicitly fixed by P-82.2.2.1; the ordinary
            # prefix formatter would re-sort by displayed spelling.
            substituents = "".join(text for text, _ in rendered)
        else:
            substituents = format_prefixes([("", text) for text, _ in rendered],
                                          omit_locants=True)
    if not substituents:
        return None
    name = substituents + core
    heavy = frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1)
    roles = [("retained-functional-parent", core_atoms, ())]
    roles.extend((f"organyl:{index}", atoms, ())
                 for index, (_, atoms) in enumerate(rendered, 1))
    return MethodDecision(name, "substitutive", frozenset({site.boron}), heavy,
                          RULE, (), (), tuple(roles))


standardMethodDecision = boron_acid_decision
