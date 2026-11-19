"""Mapped decisions for simple neutral phosphonic and phosphinic acids.

The proposal covers one mononuclear phosphorus acid with one or two
hydrocarbon organyl groups substituted at phosphorus. It composes the named
acid core with independently mapped organyl branches; unsupported core
annotations and non-hydrocarbon branches decline.
"""

from dataclasses import dataclass

from .molecule_iteration import iter_atoms
from .naming_methods import MethodDecision


RULE = ("P-42.3/P-67.1.1.1/P-67.1.1.2/P-82.2.1/P-82.2.2.1/"
        "P-82.4/P-92.1.2.1.1/P-93.4")


@dataclass(frozen=True, slots=True)
class PhosphorusAcidSite:
    phosphorus: int
    double_oxygen: int
    hydroxy_oxygens: tuple[int, ...]
    carbon_roots: tuple[int, ...]
    kind: str


def _hydrocarbon_branch(mol, root, phosphorus):
    """Return the complete carbon-only organyl component, or decline."""
    pending, atoms = [root], set()
    while pending:
        idx = pending.pop()
        if idx in atoms:
            continue
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() == 8:
            neighbors = tuple(atom.GetNeighbors())
            if (len(neighbors) != 1 or neighbors[0].GetIdx() != root
                    or mol.GetBondBetweenAtoms(idx, root).GetBondTypeAsDouble() != 2
                    or atom.GetFormalCharge() or atom.GetIsotope()):
                return None
            atoms.add(idx)
            continue
        if atom.GetAtomicNum() != 6 or atom.GetFormalCharge() or atom.GetNumRadicalElectrons():
            return None
        atoms.add(idx)
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            j = other.GetIdx()
            if j == phosphorus or other.GetAtomicNum() == 1:
                continue
            if other.GetAtomicNum() != 6 and not (idx == root and other.GetAtomicNum() == 8
                    and other.GetDegree() == 1 and bond.GetBondTypeAsDouble() == 2):
                return None
            pending.append(j)
    if not atoms:
        return None
    # Every carbon edge in the component must be ordinary or aromatic.
    for idx in atoms:
        atom = mol.GetAtomWithIdx(idx)
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            if other.GetAtomicNum() == 1:
                continue
            if other.GetIdx() == phosphorus:
                if idx != root or bond.GetBondTypeAsDouble() != 1:
                    return None
            elif other.GetIdx() in atoms:
                if not (bond.GetIsAromatic() or bond.GetBondTypeAsDouble() == 1
                        or bond.GetBondTypeAsDouble() == 2):
                    return None
            else:
                return None
    return frozenset(atoms)


def _recognize(mol):
    if (mol.GetNumAtoms() > 96 or mol.GetStereoGroups()
            or any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in iter_atoms(mol))):
        return None
    phosphorus_atoms = [a for a in iter_atoms(mol) if a.GetAtomicNum() == 15]
    if len(phosphorus_atoms) != 1:
        return None
    p = phosphorus_atoms[0]
    pidx = p.GetIdx()
    if p.IsInRing() or p.GetIsotope() or int(p.GetChiralTag()):
        return None
    carbon_roots, hydroxy, double_oxygens = [], [], []
    core_atoms = {pidx}
    for neighbor in p.GetNeighbors():
        idx = neighbor.GetIdx()
        bond = mol.GetBondBetweenAtoms(pidx, idx)
        if neighbor.GetAtomicNum() == 6 and bond.GetBondTypeAsDouble() == 1:
            carbon_roots.append(idx)
            continue
        if neighbor.GetAtomicNum() != 8 or neighbor.IsInRing():
            return None
        if neighbor.GetIsotope() or int(neighbor.GetChiralTag()):
            return None
        if bond.GetBondTypeAsDouble() == 2 and neighbor.GetDegree() == 1:
            double_oxygens.append(idx)
            core_atoms.add(idx)
        elif (bond.GetBondTypeAsDouble() == 1 and neighbor.GetDegree() == 1
              and neighbor.GetTotalNumHs(includeNeighbors=True) == 1):
            # Core O/H isotope syntax is not established by this operation.
            if any(n.GetAtomicNum() == 1 for n in neighbor.GetNeighbors()):
                return None
            hydroxy.append(idx)
            core_atoms.add(idx)
        else:
            return None
    if len(double_oxygens) != 1:
        return None
    if (len(carbon_roots), len(hydroxy)) == (1, 2):
        kind = "phosphonic acid"
    elif (len(carbon_roots), len(hydroxy)) == (2, 1):
        kind = "phosphinic acid"
    else:
        return None
    if len(set(carbon_roots)) != len(carbon_roots):
        return None
    branches = []
    covered = set(core_atoms)
    for root in carbon_roots:
        atoms = _hydrocarbon_branch(mol, root, pidx)
        if atoms is None or any(atoms & prior for prior in branches):
            return None
        branches.append(atoms)
        covered.update(atoms)
    heavy = {a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1}
    if covered != heavy:
        return None
    return PhosphorusAcidSite(pidx, double_oxygens[0], tuple(hydroxy),
                              tuple(carbon_roots), kind), tuple(branches), frozenset(core_atoms)


def recognize(mol):
    """Return a source-bounded structural site, without rendering a name."""
    found = _recognize(mol)
    return None if found is None else found[0]


def _acyl_branch_name(mol, Chem, root, atoms):
    """Select the acyl carbon parent, keeping its own nuclide descriptors."""
    if any(int(mol.GetAtomWithIdx(i).GetChiralTag()) for i in atoms) or any(
            int(b.GetStereo()) for b in mol.GetBonds()
            if b.GetBeginAtomIdx() in atoms and b.GetEndAtomIdx() in atoms):
        return ''
    remove = set(range(mol.GetNumAtoms())) - set(atoms)
    rw = Chem.RWMol(mol)
    for index in sorted(remove, reverse=True):
        rw.RemoveAtom(index)
    carbon = root - sum(index < root for index in remove)
    hydroxy = rw.AddAtom(Chem.Atom(8))
    rw.AddBond(carbon, hydroxy, Chem.BondType.SINGLE)
    capped = rw.GetMol()
    try:
        Chem.SanitizeMol(capped)
    except (ValueError, RuntimeError):
        return ''
    from .api import mol_to_iupac
    acid = mol_to_iupac(capped)
    retained = {'formic acid': 'formyl', 'acetic acid': 'acetyl', 'benzoic acid': 'benzoyl'}
    if acid in retained:
        return retained[acid]
    if acid.endswith('carboxylic acid'):
        parent = acid[:-len('carboxylic acid')]
        # P-82.6.3.2 gives the separately identified carbonyl nuclide
        # square brackets in the complete systematic acyl unit. Preserve
        # any other parent nuclides and change only this mapped carbon.
        mass = mol.GetAtomWithIdx(root).GetIsotope()
        local = '(' + str(mass) + 'C)'
        if mass and parent.endswith(local):
            parent = parent[:-len(local)] + '[' + str(mass) + 'C]'
        return parent + 'carbonyl'
    if acid.endswith('oic acid'):
        return acid[:-len('oic acid')] + 'oyl'
    return ''


def phosphorus_acid_decision(mol, Chem=None):
    """Return a whole-graph mapped substitutive decision, or ``None``."""
    found = _recognize(mol)
    if found is None:
        return None
    if Chem is None:
        from rdkit import Chem
    site, branches, core_atoms = found
    from .component_mapping import MappedComponent, decorate_component, mapped_branch
    from .formatting import format_prefixes, prefix_alphabetization_key

    rendered = []
    for root, atoms in zip(site.carbon_roots, branches):
        acyl = any(n.GetAtomicNum() == 8 and mol.GetBondBetweenAtoms(root, n.GetIdx()).GetBondTypeAsDouble() == 2
                   for n in mol.GetAtomWithIdx(root).GetNeighbors())
        if acyl:
            text = _acyl_branch_name(mol, Chem, root, atoms)
            owned = atoms
        else:
            text, owned = mapped_branch(mol, Chem, root, site.phosphorus)
        owned_heavy = frozenset(i for i in owned if mol.GetAtomWithIdx(i).GetAtomicNum() > 1)
        if not text or owned_heavy != atoms:
            return None
        rendered.append((text, atoms, frozenset(owned), root))
    # P-82.2.2.1 gives a modified substituent precedence over its otherwise
    # identical unmodified counterpart. No such special ordering rule is
    # established here for stereo-only differences between identical arms.
    if len(rendered) == 2 and _same_unmodified_organyl(mol, Chem, site,
                                                        rendered[0], rendered[1]):
        first, second = rendered
        first_isotope = _has_isotope(mol, first[2])
        second_isotope = _has_isotope(mol, second[2])
        first_stereo = _has_stereo(mol, Chem, first[1])
        second_stereo = _has_stereo(mol, Chem, second[1])
        if first_stereo != second_stereo or (first_stereo and first[0] != second[0]):
            return None
        if first_isotope != second_isotope:
            rendered.sort(key=lambda row: not _has_isotope(mol, row[2]))
        elif first_isotope and first[0] != second[0]:
            return None
    rendered.sort(key=lambda row: prefix_alphabetization_key(row[0]))
    # Re-apply the explicit P-82 precedence after the general alphabetic
    # comparison, whose keys can be equal after stripping isotope descriptors.
    if len(rendered) == 2 and _same_unmodified_organyl(mol, Chem, site,
                                                        rendered[0], rendered[1]):
        left, right = rendered
        if _has_isotope(mol, left[2]) != _has_isotope(mol, right[2]):
            rendered.sort(key=lambda row: not _has_isotope(mol, row[2]))
    prefixes = tuple(("", text) for text, _, _, _ in rendered)
    # P-16.5.1.3.1: separately attached organyls must remain separate units.
    # A cyclic organyl followed by methyl otherwise becomes a single
    # cycloalkylmethyl arm, moving the P-C attachment and hydrogen budget.
    prefix_text = format_prefixes(prefixes, omit_locants=True,
                                  hierarchical_enclosures=True, separate_unlocanted=True)
    core_name = decorate_component(mol, Chem, MappedComponent(
        prefix_text + site.kind,
        tuple((idx, "") for idx in sorted(core_atoms)),
        core_atoms,
        embedded_atoms=frozenset().union(*(owned for _, _, owned, _ in rendered)),
        systematic_parent=True,
        prefix_mode="unlocanted",
    ))
    if not core_name:
        return None
    heavy = frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1)
    roles = [("phosphorus-acid-core", core_atoms,
              tuple((idx, "") for idx in sorted(core_atoms)))]
    roles.extend((f"organyl:{n}", owned, ())
                 for n, (_, _, owned, _) in enumerate(rendered, 1))
    return MethodDecision(core_name, "substitutive", frozenset({site.phosphorus}),
                          heavy, RULE, tuple((idx, "") for idx in sorted(core_atoms)),
                          component_roles=tuple(roles))


standardMethodDecision = phosphorus_acid_decision


def _has_isotope(mol, atoms):
    return any(mol.GetAtomWithIdx(i).GetIsotope() for i in atoms)


def _has_stereo(mol, Chem, atoms):
    from .stereochemistry import assigned_atom_stereochemistry

    atoms = set(atoms)
    return any(idx in atoms for idx, _ in assigned_atom_stereochemistry(
        mol, Chem, atom_order="idx", include_pseudo=True))


def _same_unmodified_organyl(mol, Chem, site, left, right):
    """Compare constitutional rooted branches after removing isotope/stereo."""
    def signature(root, atoms):
        copy = Chem.Mol(mol)
        for atom in copy.GetAtoms():
            atom.SetIsotope(0)
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        for bond in copy.GetBonds():
            bond.SetStereo(Chem.BondStereo.STEREONONE)
            bond.SetBondDir(Chem.BondDir.NONE)
        return Chem.MolFragmentToSmiles(copy, atomsToUse=sorted(atoms),
            rootedAtAtom=root, canonical=True, isomericSmiles=False)
    return signature(left[3], left[1]) == signature(right[3], right[1])
