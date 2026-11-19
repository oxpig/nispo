"""Mapped NH–NO and NH–NO2 prefix units from Blue Book P-67.1.4.3.2.

Only the one-core, unannotated, neutral-NH branch with its single open bond
to a parent is admitted. The helper describes the original component; it does
not infer chain extension through the terminal oxo-nitrogen.
"""
from __future__ import annotations

from dataclasses import dataclass


RULE = "P-67.1.4.3.2/P-66.1.1.4.3/P-16"


@dataclass(frozen=True, slots=True)
class NitrogenOxoAmidePrefix:
    name: str
    kind: str
    parent_nitrogen: int
    terminal_nitrogen: int
    oxygens: tuple[int, ...]
    atoms: frozenset[int]
    rule: str = RULE


@dataclass(frozen=True, slots=True)
class DirectNitrogenOxoSubstituent:
    """A terminal nitroso/nitro group attached directly to a parent N.

    This is deliberately distinct from :class:`NitrogenOxoAmidePrefix`: the
    latter is the P-67 ``–NH–NOx`` unit rooted at its NH, whereas this record
    represents only the terminal ``–NOx`` atoms when the bond cut is between
    the parent nitrogen and its N-oxo substituent.
    """

    name: str
    kind: str
    parent_nitrogen: int
    terminal_nitrogen: int
    oxygens: tuple[int, ...]
    atoms: frozenset[int]
    rule: str = "P-61.5.1/P-61.5.2/P-66.1.1.3.1.1/P-66.1.6.1.1.2"


def _plain(atom):
    return (not atom.GetIsotope() and not int(atom.GetChiralTag())
            and not atom.GetNumRadicalElectrons() and atom.GetFormalCharge() == 0)


def _terminal_oxo_nitrogen(mol, nitrogen_id: int, linked_nitrogen_id: int):
    """Return the exact P-67 terminal N=O or N(=O)O− local pattern."""
    if (not 0 <= nitrogen_id < mol.GetNumAtoms()
            or not 0 <= linked_nitrogen_id < mol.GetNumAtoms()
            or nitrogen_id == linked_nitrogen_id):
        return None
    nitrogen = mol.GetAtomWithIdx(nitrogen_id)
    linked = mol.GetAtomWithIdx(linked_nitrogen_id)
    if (nitrogen.GetAtomicNum() != 7 or linked.GetAtomicNum() != 7
            or linked.GetFormalCharge() != 0 or linked.GetNumRadicalElectrons()
            or nitrogen.GetIsotope() or int(nitrogen.GetChiralTag())
            or nitrogen.GetNumRadicalElectrons() or nitrogen.IsInRing()):
        return None
    connecting = mol.GetBondBetweenAtoms(nitrogen_id, linked_nitrogen_id)
    if connecting is None or connecting.GetBondTypeAsDouble() != 1:
        return None
    oxygens = [n for n in nitrogen.GetNeighbors() if n.GetAtomicNum() == 8]
    nonoxygen = [n.GetIdx() for n in nitrogen.GetNeighbors() if n.GetAtomicNum() != 8]
    if nonoxygen != [linked_nitrogen_id] or len(oxygens) not in {1, 2}:
        return None
    for oxygen in oxygens:
        if (oxygen.GetIsotope() or int(oxygen.GetChiralTag())
                or oxygen.GetNumRadicalElectrons() or oxygen.GetDegree() != 1):
            return None
    oxygen_ids = tuple(sorted(o.GetIdx() for o in oxygens))
    doubles = [o for o in oxygens
               if mol.GetBondBetweenAtoms(nitrogen_id, o.GetIdx()).GetBondTypeAsDouble() == 2]
    if len(oxygen_ids) == 1:
        oxygen = oxygens[0]
        if (nitrogen.GetFormalCharge() != 0 or nitrogen.GetDegree() != 2
                or nitrogen.GetTotalValence() != 3 or nitrogen.GetTotalNumHs() != 0
                or oxygen.GetFormalCharge() != 0 or len(doubles) != 1):
            return None
        kind = "nitrous"
    else:
        singles = [o for o in oxygens
                   if mol.GetBondBetweenAtoms(nitrogen_id, o.GetIdx()).GetBondTypeAsDouble() == 1]
        if (nitrogen.GetFormalCharge() != 1 or nitrogen.GetDegree() != 3
                or nitrogen.GetTotalValence() != 4 or nitrogen.GetTotalNumHs() != 0
                or len(doubles) != 1 or len(singles) != 1
                or doubles[0].GetFormalCharge() != 0
                or singles[0].GetFormalCharge() != -1):
            return None
        kind = "nitric"
    for atom_id in (nitrogen_id, linked_nitrogen_id, *oxygen_ids):
        for bond in mol.GetAtomWithIdx(atom_id).GetBonds():
            if bond.GetStereo() != 0 or bond.GetBondDir() != 0:
                return None
    return kind, oxygen_ids


def nitrogen_oxo_amide_prefix(mol, root: int, blocked: int,
                              subtree_atoms) -> NitrogenOxoAmidePrefix | None:
    """Recognize an original-graph –NH–NO/NO2 prefix rooted at its NH.

    The parent NH has exactly one bond to ``blocked`` and one to a terminal
    nitrogen; it has one hydrogen and no other substituents. The terminal
    nitrogen is either neutral N=O or charge-separated [N+](=O)[O-].
    """
    if (not isinstance(root, int) or not isinstance(blocked, int)
            or root == blocked or not 0 <= root < mol.GetNumAtoms()
            or not 0 <= blocked < mol.GetNumAtoms()):
        return None
    atoms = frozenset(subtree_atoms)
    if not atoms or root not in atoms or blocked in atoms:
        return None
    parent_n = mol.GetAtomWithIdx(root)
    host = mol.GetAtomWithIdx(blocked)
    if (parent_n.GetAtomicNum() != 7 or not _plain(parent_n)
            or parent_n.IsInRing() or parent_n.GetDegree() != 2
            or parent_n.GetTotalValence() != 3 or parent_n.GetTotalNumHs() != 1
            or host.GetAtomicNum() != 6):
        return None
    host_bond = mol.GetBondBetweenAtoms(root, blocked)
    if host_bond is None or host_bond.GetBondTypeAsDouble() != 1:
        return None
    children = [n.GetIdx() for n in parent_n.GetNeighbors()
                if n.GetIdx() != blocked]
    if len(children) != 1:
        return None
    terminal = mol.GetAtomWithIdx(children[0])
    terminal_core = _terminal_oxo_nitrogen(mol, terminal.GetIdx(), root)
    if terminal_core is None:
        return None
    kind = terminal_core[0]
    oxygen_ids = terminal_core[1]
    name = "nitramido" if kind == "nitric" else "nitrosoamino"
    expected_atoms = frozenset((root, terminal.GetIdx(), *oxygen_ids))
    if atoms != expected_atoms:
        return None
    # Reject specified bond stereochemistry on the typed unit or its parent cut.
    for atom_id in expected_atoms:
        atom = mol.GetAtomWithIdx(atom_id)
        for bond in atom.GetBonds():
            if bond.GetStereo() != 0 or bond.GetBondDir() != 0:
                return None
    return NitrogenOxoAmidePrefix(
        name=name, kind=kind, parent_nitrogen=root,
        terminal_nitrogen=terminal.GetIdx(), oxygens=oxygen_ids, atoms=atoms)


def direct_nitrogen_oxo_substituent(mol, root: int, blocked: int,
                                    subtree_atoms) -> DirectNitrogenOxoSubstituent | None:
    """Recognize terminal ``–NO``/``–NO2`` on an original parent nitrogen.

    Recognition delegates to the exact functional-inventory motif, then binds
    the cut and complete terminal subtree to original atom IDs.  It does not
    infer this as a carbon-rooted ``nitrosoamino``/``nitramido`` group.
    Isotope, stereochemistry, radicals, alternative charge placement, extra
    terminal-N attachments, and incomplete ownership are outside this record.
    """
    if (not isinstance(root, int) or not isinstance(blocked, int)
            or root == blocked or not 0 <= root < mol.GetNumAtoms()
            or not 0 <= blocked < mol.GetNumAtoms()):
        return None
    terminal = mol.GetAtomWithIdx(root)
    parent = mol.GetAtomWithIdx(blocked)
    if terminal.GetAtomicNum() != 7 or parent.GetAtomicNum() != 7:
        return None
    from .functional_inventory import nitrogen_oxoamide_sites

    matches = [row for row in nitrogen_oxoamide_sites(mol)
               if row[1] == root and row[2] == blocked]
    if len(matches) != 1:
        return None
    kind, terminal_n, parent_n, oxygen_ids = matches[0]
    expected = frozenset((terminal_n, *oxygen_ids))
    if frozenset(subtree_atoms) != expected:
        return None
    # The central P-67 motif and the parent N are unannotated in this local
    # grammar. The charge-separated nitramide form is the only charge allowed.
    for idx in (terminal_n, parent_n, *oxygen_ids):
        atom = mol.GetAtomWithIdx(idx)
        if (atom.GetIsotope() or int(atom.GetChiralTag())
                or atom.GetNumRadicalElectrons()):
            return None
    if parent.GetFormalCharge():
        return None
    for atom_id in expected:
        for bond in mol.GetAtomWithIdx(atom_id).GetBonds():
            if int(bond.GetStereo()) or int(bond.GetBondDir()):
                return None
    connecting = mol.GetBondBetweenAtoms(terminal_n, parent_n)
    if connecting is None or connecting.GetBondTypeAsDouble() != 1:
        return None
    if kind == "nitrous-amide":
        name, label = "nitroso", "nitrous"
    elif kind == "nitramide":
        name, label = "nitro", "nitric"
    else:
        return None
    return DirectNitrogenOxoSubstituent(
        name=name, kind=label, parent_nitrogen=parent_n,
        terminal_nitrogen=terminal_n, oxygens=tuple(oxygen_ids), atoms=expected,
    )


def has_unsupported_nitrogen_oxo_chain(mol, root: int, subtree_atoms) -> bool:
    """Block only a proved N–N–oxo-N extension from amino fallback."""
    atoms = set(subtree_atoms)
    if not 0 <= root < mol.GetNumAtoms() or mol.GetAtomWithIdx(root).GetAtomicNum() != 7:
        return False
    for idx in atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() != 7:
            continue
        if idx == root:
            continue
        adjacent_nitrogens = [n.GetIdx() for n in atom.GetNeighbors()
                              if n.GetAtomicNum() == 7]
        if any(other != root
               and _terminal_oxo_nitrogen(mol, idx, other) is not None
               for other in adjacent_nitrogens):
            return True
    return False


def has_unresolved_direct_nitrogen_oxo_substituent(mol, root: int, blocked: int) -> bool:
    """Prevent an unproved N–N–O branch from falling into generic N naming."""
    if (not 0 <= root < mol.GetNumAtoms() or not 0 <= blocked < mol.GetNumAtoms()
            or mol.GetAtomWithIdx(root).GetAtomicNum() != 7
            or mol.GetAtomWithIdx(blocked).GetAtomicNum() != 7):
        return False
    bond = mol.GetBondBetweenAtoms(root, blocked)
    if bond is None or bond.GetBondTypeAsDouble() != 1:
        return False
    atom = mol.GetAtomWithIdx(root)
    return any(neighbor.GetAtomicNum() == 8 for neighbor in atom.GetNeighbors())
