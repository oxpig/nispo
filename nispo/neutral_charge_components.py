"""Exact charge-role proofs for neutral charge-separated characteristic groups.

The recognizers operate on the caller's original atom indices. They neither
normalize resonance forms nor mutate the input graph. A proof is admissible
only when its recognized charge atoms account for the complete charged-atom
set of a net-neutral, nonradical molecule.
"""

from dataclasses import dataclass

from .molecule_iteration import iter_atoms


@dataclass(frozen=True, slots=True)
class NeutralChargeRoleProof:
    kind: str
    attachment_atom: int
    atoms: tuple[int, ...]
    roles: tuple[tuple[str, int], ...]
    formal_charges: tuple[tuple[int, int], ...]
    bond_orders: tuple[tuple[int, int, float], ...]

    @property
    def charge_atom_ids(self):
        return frozenset(idx for idx, charge in self.formal_charges if charge)

    @property
    def net_charge(self):
        return sum(charge for _idx, charge in self.formal_charges)


def _bond_order(mol, a, b):
    bond = mol.GetBondBetweenAtoms(a, b)
    return None if bond is None else float(bond.GetBondTypeAsDouble())


def _motif_atoms_are_plain(mol, atoms):
    for idx in atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetNumRadicalElectrons() or atom.GetIsotope() or int(atom.GetChiralTag()):
            return False
    return True


def _nitro_at_host(mol, host_idx, nitrogen_idx, *, kind, carbon_idx, bridge=None):
    host = mol.GetAtomWithIdx(host_idx)
    nitrogen = mol.GetAtomWithIdx(nitrogen_idx)
    if (_bond_order(mol, host_idx, nitrogen_idx) != 1.0
            or host.GetFormalCharge()
            or nitrogen.GetAtomicNum() != 7 or nitrogen.GetFormalCharge() != 1
            or nitrogen.GetDegree() != 3 or nitrogen.GetTotalValence() != 4):
        return None
    oxygens = [n for n in nitrogen.GetNeighbors()
               if n.GetIdx() != host_idx and n.GetAtomicNum() == 8]
    other_heavy = [n.GetIdx() for n in nitrogen.GetNeighbors()
                   if n.GetAtomicNum() > 1 and n.GetIdx() != host_idx
                   and n.GetAtomicNum() != 8]
    if len(oxygens) != 2 or other_heavy:
        return None
    double_o = [o for o in oxygens if _bond_order(mol, nitrogen_idx, o.GetIdx()) == 2.0]
    single_o = [o for o in oxygens if _bond_order(mol, nitrogen_idx, o.GetIdx()) == 1.0]
    if len(double_o) != 1 or len(single_o) != 1:
        return None
    oxo, anionic = double_o[0], single_o[0]
    if (oxo.GetDegree() != 1 or oxo.GetFormalCharge() != 0 or oxo.GetTotalValence() != 2
            or anionic.GetDegree() != 1 or anionic.GetFormalCharge() != -1
            or anionic.GetTotalValence() != 1):
        return None
    atoms = (nitrogen_idx, oxo.GetIdx(), anionic.GetIdx())
    if bridge is not None:
        atoms = (bridge, *atoms)
    if not _motif_atoms_are_plain(mol, atoms):
        return None
    roles = (("nitro-nitrogen", nitrogen_idx), ("nitro-oxo-oxygen", oxo.GetIdx()),
             ("nitro-anionic-oxygen", anionic.GetIdx()))
    charges = tuple((idx, mol.GetAtomWithIdx(idx).GetFormalCharge()) for idx in atoms)
    bonds = ((host_idx, nitrogen_idx, 1.0), (nitrogen_idx, oxo.GetIdx(), 2.0),
             (nitrogen_idx, anionic.GetIdx(), 1.0))
    if bridge is not None:
        roles = (("nitrate-bridge-oxygen", bridge), *roles)
        bonds = ((carbon_idx, bridge, 1.0), *bonds)
    return NeutralChargeRoleProof(kind, carbon_idx, atoms, roles, charges, bonds)


def _nitrate_at_carbon(mol, carbon_idx, oxygen_idx):
    oxygen = mol.GetAtomWithIdx(oxygen_idx)
    if oxygen.GetDegree() != 2 or oxygen.GetFormalCharge():
        return None
    nitrogen = next(n for n in oxygen.GetNeighbors() if n.GetIdx() != carbon_idx)
    if nitrogen.GetAtomicNum() != 7 or nitrogen.GetFormalCharge() != 1:
        return None
    carbon = mol.GetAtomWithIdx(carbon_idx)
    if (oxygen.GetTotalValence() != 2 or _bond_order(mol, carbon_idx, oxygen_idx) != 1.0
            or any(b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(carbon).GetAtomicNum() == 8
                   for b in carbon.GetBonds())):
        return None
    return _nitro_at_host(mol, oxygen_idx, nitrogen.GetIdx(), kind="nitrooxy",
                          carbon_idx=carbon_idx, bridge=oxygen_idx)


def _isocyano_at_carbon(mol, carbon_idx, nitrogen_idx):
    nitrogen = mol.GetAtomWithIdx(nitrogen_idx)
    if (nitrogen.GetDegree() != 2 or nitrogen.GetTotalValence() != 4
            or _bond_order(mol, carbon_idx, nitrogen_idx) != 1.0):
        return None
    terminal = next(n for n in nitrogen.GetNeighbors() if n.GetIdx() != carbon_idx)
    last = terminal.GetIdx()
    if (terminal.GetAtomicNum() != 6 or terminal.GetFormalCharge() != -1
            or terminal.GetDegree() != 1 or _bond_order(mol, nitrogen_idx, last) != 3.0
            or not _motif_atoms_are_plain(mol, (nitrogen_idx, last))):
        return None
    return NeutralChargeRoleProof(
        "isocyano", carbon_idx, (nitrogen_idx, last),
        (("isocyano-nitrogen", nitrogen_idx), ("isocyano-carbon", last)),
        ((nitrogen_idx, 1), (last, -1)),
        ((carbon_idx, nitrogen_idx, 1.0), (nitrogen_idx, last, 3.0)),
    )


def _azido_at_host(mol, host_idx, first_idx):
    host = mol.GetAtomWithIdx(host_idx)
    first = mol.GetAtomWithIdx(first_idx)
    if (_bond_order(mol, host_idx, first_idx) != 1.0
            or not (host.GetAtomicNum() == 6 or host.GetAtomicNum() == 7 and host.IsInRing()
                    and host.GetTotalValence() == 3) or host.GetFormalCharge() or host.GetNumRadicalElectrons()
            or first.GetAtomicNum() != 7 or first.GetFormalCharge() != 0):
        return None
    middle_neighbors = [n for n in first.GetNeighbors() if n.GetAtomicNum() == 7
                        and n.GetIdx() != host_idx
                        and _bond_order(mol, first_idx, n.GetIdx()) == 2.0]
    if len(middle_neighbors) != 1 or first.GetDegree() != 2 or first.GetTotalValence() != 3:
        return None
    middle = middle_neighbors[0]
    if (middle.GetFormalCharge() != 1 or middle.GetDegree() != 2
            or middle.GetTotalValence() != 4):
        return None
    terminal = [n for n in middle.GetNeighbors() if n.GetIdx() != first_idx]
    if (len(terminal) != 1 or terminal[0].GetAtomicNum() != 7
            or _bond_order(mol, middle.GetIdx(), terminal[0].GetIdx()) != 2.0):
        return None
    last = terminal[0]
    if (last.GetFormalCharge() != -1 or last.GetDegree() != 1
            or last.GetTotalValence() != 2):
        return None
    atoms = (first_idx, middle.GetIdx(), last.GetIdx())
    if not _motif_atoms_are_plain(mol, atoms):
        return None
    roles = (("azido-attachment-nitrogen", first_idx),
             ("azido-central-positive-nitrogen", middle.GetIdx()),
             ("azido-terminal-negative-nitrogen", last.GetIdx()))
    charges = tuple((idx, mol.GetAtomWithIdx(idx).GetFormalCharge()) for idx in atoms)
    bonds = ((host_idx, first_idx, 1.0), (first_idx, middle.GetIdx(), 2.0),
             (middle.GetIdx(), last.GetIdx(), 2.0))
    return NeutralChargeRoleProof("azido", host_idx, atoms, roles, charges, bonds)


def _nitramide_charge_at_nitrogen(mol, n0_idx, n1_idx):
    """Validate charge evidence for the exact existing P-67 nitramide core.

    This proof is for charge accounting only. It does not recognize or emit a
    substituent prefix; the existing nitric/nitrous amide inventory remains
    authoritative for class membership.
    """
    n0, n1 = mol.GetAtomWithIdx(n0_idx), mol.GetAtomWithIdx(n1_idx)
    if (n0.GetAtomicNum() != 7 or n1.GetAtomicNum() != 7
            or n0.GetFormalCharge() != 1 or n0.GetDegree() != 3
            or n0.GetTotalValence() != 4 or n0.IsInRing() or n0.GetIsAromatic()
            or n0.GetNumRadicalElectrons()
            or n1.GetFormalCharge() != 0 or n1.GetNumRadicalElectrons()
            or n1.IsInRing() or n1.GetIsAromatic() or n1.GetTotalValence() != 3
            or _bond_order(mol, n0_idx, n1_idx) != 1.0
            or any(b.GetBondTypeAsDouble() != 1.0 for b in n1.GetBonds())):
        return None
    oxygens = [a for a in n0.GetNeighbors() if a.GetAtomicNum() == 8]
    other = [a for a in n0.GetNeighbors() if a.GetIdx() != n1_idx
             and a.GetAtomicNum() != 8]
    if len(oxygens) != 2 or other:
        return None
    doubles = [o for o in oxygens if _bond_order(mol, n0_idx, o.GetIdx()) == 2.0]
    singles = [o for o in oxygens if _bond_order(mol, n0_idx, o.GetIdx()) == 1.0]
    if len(doubles) != 1 or len(singles) != 1:
        return None
    oxo, anionic = doubles[0], singles[0]
    if (oxo.GetDegree() != 1 or oxo.GetFormalCharge() != 0 or oxo.GetTotalValence() != 2
            or oxo.GetNumRadicalElectrons()
            or anionic.GetDegree() != 1 or anionic.GetFormalCharge() != -1
            or anionic.GetTotalValence() != 1 or anionic.GetNumRadicalElectrons()):
        return None
    external = [a for a in n1.GetNeighbors() if a.GetIdx() != n0_idx and a.GetAtomicNum() > 1]
    if any(a.GetAtomicNum() not in {6, 16} for a in external):
        return None
    atoms = (n0_idx, oxo.GetIdx(), anionic.GetIdx())
    roles = (("nitramide-nitrogen", n0_idx), ("nitramide-oxo-oxygen", oxo.GetIdx()),
             ("nitramide-anionic-oxygen", anionic.GetIdx()),
             ("nitramide-attached-nitrogen", n1_idx))
    charges = tuple((idx, mol.GetAtomWithIdx(idx).GetFormalCharge()) for idx in atoms)
    bonds = ((n0_idx, oxo.GetIdx(), 2.0), (n0_idx, anionic.GetIdx(), 1.0),
             (n0_idx, n1_idx, 1.0))
    return NeutralChargeRoleProof("nitramide-charge", n1_idx, atoms, roles, charges, bonds)


def _recognize_at_carbon(mol, carbon_idx, attached_idx):
    if attached_idx < 0:
        return None
    host = mol.GetAtomWithIdx(carbon_idx)
    if host.GetAtomicNum() != 6 or host.GetFormalCharge() or host.GetNumRadicalElectrons():
        return None
    attached = mol.GetAtomWithIdx(attached_idx)
    if attached.GetAtomicNum() == 7 and attached.GetFormalCharge() == 1:
        if attached.GetDegree() == 2:
            return _isocyano_at_carbon(mol, carbon_idx, attached_idx)
        return _nitro_at_host(mol, carbon_idx, attached_idx, kind="nitro", carbon_idx=carbon_idx)
    if attached.GetAtomicNum() == 7 and attached.GetFormalCharge() == 0:
        return _azido_at_host(mol, carbon_idx, attached_idx)
    if attached.GetAtomicNum() == 8:
        return _nitrate_at_carbon(mol, carbon_idx, attached_idx)
    return None


def _raw_neutral_charge_role_proofs(mol, nitramide_sites=()):
    proofs = []
    for kind, n0, n1, _oxygens in nitramide_sites:
        if kind == "nitramide":
            proof = _nitramide_charge_at_nitrogen(mol, n0, n1)
            if proof is not None:
                proofs.append(proof)
    for host in iter_atoms(mol):
        number = host.GetAtomicNum()
        if number not in {6, 7} or host.GetFormalCharge() or host.GetNumRadicalElectrons():
            continue
        if number == 6:
            for neighbor in host.GetNeighbors():
                if neighbor.GetAtomicNum() in {7, 8}:
                    proof = _recognize_at_carbon(mol, host.GetIdx(), neighbor.GetIdx())
                    if proof is not None:
                        proofs.append(proof)
        elif host.IsInRing():
            for neighbor in host.GetNeighbors():
                if neighbor.GetAtomicNum() == 7:
                    proof = _azido_at_host(mol, host.GetIdx(), neighbor.GetIdx())
                    if proof is not None:
                        proofs.append(proof)
    return tuple(sorted(proofs, key=lambda row: (row.kind, row.attachment_atom, row.atoms)))


def validate_neutral_charge_role_proofs(mol, proofs):
    """Validate exact proof identity and complete net-zero charge coverage.

    With no charge present, an empty proof tuple is valid. Any radical or
    uncovered/overlapping charged atom invalidates the entire proof set.
    """
    proofs = tuple(proofs)
    if any(atom.GetNumRadicalElectrons() for atom in iter_atoms(mol)):
        return False
    charged = frozenset(atom.GetIdx() for atom in iter_atoms(mol)
                        if atom.GetFormalCharge())
    if sum(atom.GetFormalCharge() for atom in iter_atoms(mol)) != 0:
        return False
    if not charged:
        return not proofs
    covered = set()
    covered_motif_atoms = set()
    for proof in proofs:
        if not isinstance(proof, NeutralChargeRoleProof):
            return False
        if proof.charge_atom_ids & covered or set(proof.atoms) & covered_motif_atoms:
            return False
        if proof.kind == "nitramide-charge":
            current = _nitramide_charge_at_nitrogen(
                mol, dict(proof.roles)["nitramide-nitrogen"], proof.attachment_atom,
            )
        elif proof.kind == "azido" and proof.atoms:
            current = _azido_at_host(mol, proof.attachment_atom, proof.atoms[0])
        else:
            current = _recognize_at_carbon(mol, proof.attachment_atom,
                                           proof.atoms[0] if proof.atoms else -1)
        if current != proof or proof.net_charge != 0:
            return False
        covered.update(proof.charge_atom_ids)
        covered_motif_atoms.update(proof.atoms)
    return bool(proofs) and frozenset(covered) == charged


def neutral_charge_role_proofs(mol, nitramide_sites=()):
    """Return recognized proofs only when they cover every graph charge.

    ``nitramide_sites`` must come from the existing exact nitric/nitrous
    amide recognizer. The resulting ``nitramide-charge`` records are evidence
    only and are never surfaced as nitro/azido detachable prefix facts.
    """
    proofs = _raw_neutral_charge_role_proofs(mol, nitramide_sites)
    if proofs and validate_neutral_charge_role_proofs(mol, proofs):
        return proofs
    return ()


def neutral_charge_prefix_facts(mol, proofs=None):
    """Return source-typed detachable prefix facts after whole-graph proof.

    Each record is ``(kind, attachment_atom, motif_atom_ids)``. The host is
    a parent atom and is outside the motif ownership.
    """
    proofs = neutral_charge_role_proofs(mol) if proofs is None else tuple(proofs)
    if not validate_neutral_charge_role_proofs(mol, proofs):
        return None
    return tuple((proof.kind, proof.attachment_atom, proof.atoms) for proof in proofs
                 if proof.kind in {"nitro", "azido"})


def neutral_charge_atom_ids(proofs):
    return frozenset(idx for proof in proofs for idx in proof.charge_atom_ids)


def nested_neutral_prefix_owned(mol, parent_atoms, mapping, prefixes, motif,
                                branch_ownership, *, resolve_branch=None):
    """Validate a neutral charge unit inside an actually rendered child prefix.

    The complete net-zero motif proof is checked by the caller. A nested unit
    must additionally belong to one original bridge-cut component whose mapped
    writer reproduces the recorded text and exact atom scope. Charge elsewhere,
    a multiply attached component or a suffix-only scope cannot qualify.
    """
    from rdkit import Chem
    from .branch_rules import collect_subtree
    from .component_mapping import mapped_branch

    resolve_branch = resolve_branch or mapped_branch
    _kind, motif_host, motif_atoms = motif
    needed = set(motif_atoms) | {motif_host}
    emitted = {(str(loc), str(text)) for loc, text in prefixes}
    for row in branch_ownership:
        if not isinstance(row, (tuple, list)) or len(row) != 4:
            continue
        root, host, atoms, text = row
        if (not isinstance(root, int) or not isinstance(host, int)
                or host not in parent_atoms or root in parent_atoms
                or not 0 <= root < mol.GetNumAtoms() or not isinstance(text, str)):
            continue
        atoms = frozenset(atoms)
        if (not needed <= atoms or atoms & parent_atoms
                or (str(mapping.get(host)), text) not in emitted):
            continue
        bond = mol.GetBondBetweenAtoms(root, host)
        if (bond is None or bond.GetBondTypeAsDouble() != 1
                or frozenset(collect_subtree(mol, root, host)) != atoms):
            continue
        rebuilt, owned = resolve_branch(mol, Chem, root, host)
        if rebuilt == text and frozenset(owned) == atoms:
            return True
    return False
