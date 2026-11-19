"""Mapped, bounded nitric/nitrous acid amide names (Blue Book P-67.1.2.6.3).

This producer handles one complete neutral nitramide or nitrous-amide unit
with zero, one, or two carbon organyl groups on the amide nitrogen.
Unsaturated, cyclic and halogen-substituted carbon components are admitted
only with a complete rooted map from the existing organyl writer.
Hydrazides, multiple oxoacid cores, isotopes, stereochemistry,
and other heteroatom chemistry are outside this small closed grammar.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .molecule_iteration import iter_atoms, iter_bonds
from .naming_methods import MethodDecision

RULE = "P-41/P-43/P-44/P-67.1.2.6.3/P-14.5/P-16.3/P-16.5.1.1/P-16.5.1.3.2"
_HEAVY_LIMIT = 64
_HALOGENS = frozenset({9, 17, 35, 53})
# P-41 ranks amides in the order of their corresponding acids.  These are the
# mapped class-11 groups currently exposed by selection_inventory whose acid
# precedes nitric/nitrous acid in P-42: carboxylic/sulfonic acids (7a) and
# carbonic acid (7b).  Unknown class-11 kinds deliberately remain unresolved.
_P42_SENIOR_AMIDE_KINDS = frozenset({"amide", "sulfonamide", "urea"})


@dataclass(frozen=True, slots=True)
class NitrousAmideRecognition:
    status: Literal["supported", "not-applicable", "deferred"]
    reason: str
    acid_nitrogen: int | None = None
    amide_nitrogen: int | None = None
    oxygens: tuple[int, ...] = ()
    organyl_roots: tuple[int, ...] = ()
    organyl_atoms: tuple[frozenset[int], ...] = ()
    kind: str = ""  # nitric or nitrous


@dataclass(frozen=True, slots=True)
class NitrousAmideProposal:
    name: str
    acid_nitrogen: int
    amide_nitrogen: int
    oxygens: tuple[int, ...]
    organyl_roots: tuple[int, ...]
    organyl_atoms: tuple[frozenset[int], ...]
    organyl_names: tuple[str, ...]
    organyl_natures: tuple[str, ...]
    kind: str
    atoms: frozenset[int]
    prefix_evidence: tuple = ()
    rule: str = RULE

    def as_method_decision(self) -> MethodDecision:
        core = frozenset((self.acid_nitrogen, self.amide_nitrogen, *self.oxygens))
        roles = [("nitric-or-nitrous-amide-unit", core, ())]
        roles.extend(("nitrogen-organyl", atoms, ()) for atoms in self.organyl_atoms)
        if (len(set().union(*(r[1] for r in roles))) != sum(len(r[1]) for r in roles)
                or frozenset().union(*(r[1] for r in roles)) != self.atoms):
            raise ValueError("nitric/nitrous amide roles are not a disjoint complete heavy-atom cover")
        return MethodDecision(
            name=self.name,
            method="functional-class",
            principal_centers=frozenset((self.acid_nitrogen, self.amide_nitrogen)),
            atoms=self.atoms,
            rule=self.rule,
            component_roles=tuple(roles),
        )


@dataclass(frozen=True, slots=True)
class NitrousAmideDisposition:
    """Caller guidance for source-shaped cores that do not get this name."""
    status: Literal["not-applicable", "renderable", "higher-class-route", "pending"]
    reason: str
    recognition: NitrousAmideRecognition
    higher_class_centers: tuple[tuple[str, int, tuple[int, int]], ...] = ()
    projected_inventory_supported: bool = False
    core_sites: tuple[tuple[int, int, tuple[int, ...], str], ...] = ()


def _bond(mol, Chem, i: int, j: int, order: float) -> bool:
    bond = mol.GetBondBetweenAtoms(i, j)
    return bond is not None and bond.GetBondTypeAsDouble() == order


def _carbon_organyl(mol, Chem, root: int, n_amide: int) -> frozenset[int] | None:
    """Prove one neutral carbon/halogen component with a single N attachment."""
    if not _bond(mol, Chem, root, n_amide, 1):
        return None
    pending, seen = [root], {root}
    while pending:
        idx = pending.pop()
        atom = mol.GetAtomWithIdx(idx)
        if (atom.GetAtomicNum() != 6 or atom.GetIsotope() or int(atom.GetChiralTag())
                or atom.GetFormalCharge()
                or atom.GetNumRadicalElectrons() or atom.GetTotalValence() != 4):
            return None
        for bond in atom.GetBonds():
            nbr = bond.GetOtherAtom(atom)
            j, z = nbr.GetIdx(), nbr.GetAtomicNum()
            if j == n_amide or z == 1:
                continue
            if z == 6:
                if bond.GetBondTypeAsDouble() not in {1, 1.5, 2, 3}:
                    return None
                if j not in seen:
                    seen.add(j)
                    pending.append(j)
            elif z in _HALOGENS:
                if (bond.GetBondTypeAsDouble() != 1 or nbr.GetDegree() != 1
                        or nbr.GetFormalCharge() or nbr.GetIsotope()
                        or int(nbr.GetChiralTag()) or nbr.GetNumRadicalElectrons()):
                    return None
                seen.add(j)
            else:
                return None
    # Cycles and unsaturation do not change the P-67 N-organyl operation.
    # Component rendering must still prove the same complete atom ownership.
    external = {(i, n.GetIdx()) for i in seen for n in mol.GetAtomWithIdx(i).GetNeighbors()
                if n.GetIdx() not in seen}
    if external != {(root, n_amide)}:
        return None
    return frozenset(seen)


def _recognized_local_cores(mol, Chem):
    """Return source-shaped local NO–N/NO2–N amide cores without judging N-substitution."""
    found = []
    for n0 in (a for a in iter_atoms(mol) if a.GetAtomicNum() == 7):
        for n1 in (a for a in n0.GetNeighbors() if a.GetAtomicNum() == 7):
            o_atoms = [a for a in n0.GetNeighbors() if a.GetAtomicNum() == 8]
            if not _bond(mol, Chem, n0.GetIdx(), n1.GetIdx(), 1):
                continue
            if len(o_atoms) == 1:
                oxygen = o_atoms[0]
                if (n0.GetFormalCharge() == 0 and n0.GetTotalValence() == 3
                        and n0.GetDegree() == 2 and _bond(mol, Chem, n0.GetIdx(), oxygen.GetIdx(), 2)
                        and oxygen.GetFormalCharge() == 0 and oxygen.GetDegree() == 1):
                    found.append((n0.GetIdx(), n1.GetIdx(), (oxygen.GetIdx(),), "nitrous"))
            elif len(o_atoms) == 2:
                doubles = [o.GetIdx() for o in o_atoms if _bond(mol, Chem, n0.GetIdx(), o.GetIdx(), 2)]
                singles = [o.GetIdx() for o in o_atoms if _bond(mol, Chem, n0.GetIdx(), o.GetIdx(), 1)]
                if (n0.GetFormalCharge() == 1 and n0.GetTotalValence() == 4
                        and n0.GetDegree() == 3 and len(doubles) == 1 and len(singles) == 1
                        and mol.GetAtomWithIdx(doubles[0]).GetFormalCharge() == 0
                        and mol.GetAtomWithIdx(doubles[0]).GetDegree() == 1
                        and mol.GetAtomWithIdx(singles[0]).GetFormalCharge() == -1
                        and mol.GetAtomWithIdx(singles[0]).GetDegree() == 1):
                    found.append((n0.GetIdx(), n1.GetIdx(), tuple(sorted((doubles[0], singles[0]))), "nitric"))
    return tuple(found)


def recognize_nitrous_amide(mol, Chem=None) -> NitrousAmideRecognition:
    """Recognize one exact NO2–N or NO–N amide core and its full graph."""
    if Chem is None:
        from rdkit import Chem as RDKitChem
        Chem = RDKitChem
    candidates = _recognized_local_cores(mol, Chem)
    if not candidates:
        return NitrousAmideRecognition("not-applicable", "no candidate oxo-nitrogen amide core")
    if len(candidates) != 1:
        return NitrousAmideRecognition("deferred", "requires exactly one oxo-nitrogen amide core")
    n0, n1, oids, kind = candidates[0]

    def decline(reason, kind=""):
        return NitrousAmideRecognition("deferred", reason, n0, n1, oids, kind=kind)

    if (mol.GetNumHeavyAtoms() > _HEAVY_LIMIT or len(Chem.GetMolFrags(mol)) != 1
            or mol.GetNumAtoms() != mol.GetNumHeavyAtoms() or mol.GetStereoGroups()):
        return decline("requires one connected heavy-atom graph within the declared bound")
    if any(a.GetIsotope() or int(a.GetChiralTag()) or a.GetNumRadicalElectrons()
           for a in iter_atoms(mol)):
        return decline("isotope, atom stereo, or radical annotations are outside this grammar")
    if any(int(b.GetStereo()) or b.GetBondDir() != Chem.BondDir.NONE for b in iter_bonds(mol)):
        return decline("bond stereochemistry/direction is outside this grammar")
    acid_n, amide_n = mol.GetAtomWithIdx(n0), mol.GetAtomWithIdx(n1)
    if acid_n.IsInRing() or amide_n.IsInRing() or acid_n.GetIsAromatic() or amide_n.GetIsAromatic():
        return decline("oxo-nitrogen amide core atoms must be acyclic and nonaromatic")
    if (acid_n.GetNumRadicalElectrons() or amide_n.GetNumRadicalElectrons()
            or amide_n.GetFormalCharge()):
        return decline("amide nitrogen and core must be closed-shell; N-acyl/charged cases decline")
    if len(oids) == 1:
        o = mol.GetAtomWithIdx(oids[0])
        if (acid_n.GetFormalCharge() != 0 or acid_n.GetTotalValence() != 3
                or acid_n.GetDegree() != 2 or o.GetFormalCharge() != 0 or o.GetDegree() != 1
                or not _bond(mol, Chem, n0, oids[0], 2)
                or not _bond(mol, Chem, n0, n1, 1)):
            return decline("nitrous-amide core must be neutral O=N–N")
    else:
        doubles = [i for i in oids if _bond(mol, Chem, n0, i, 2)]
        singles = [i for i in oids if _bond(mol, Chem, n0, i, 1)]
        if (len(doubles) != 1 or len(singles) != 1 or acid_n.GetFormalCharge() != 1
                or acid_n.GetTotalValence() != 4 or acid_n.GetDegree() != 3
                or not _bond(mol, Chem, n0, n1, 1)):
            return decline("nitramide core must be the neutral charge-separated O2N–N form")
        od, os = mol.GetAtomWithIdx(doubles[0]), mol.GetAtomWithIdx(singles[0])
        if (od.GetFormalCharge() or od.GetDegree() != 1 or os.GetFormalCharge() != -1
                or os.GetDegree() != 1):
            return decline("nitramide oxido/oxo oxygen assignments are not the closed form")
    if (amide_n.GetIsAromatic() or amide_n.GetTotalValence() != 3
            or any(b.GetBondTypeAsDouble() != 1 for b in amide_n.GetBonds())):
        return decline("amide nitrogen must have ordinary neutral single-bond valence")
    heavy_neighbors = [a for a in amide_n.GetNeighbors() if a.GetAtomicNum() > 1]
    roots = tuple(a.GetIdx() for a in heavy_neighbors if a.GetAtomicNum() == 6)
    if (len(roots) > 2 or sum(a.GetIdx() == n0 for a in heavy_neighbors) != 1
            or len(heavy_neighbors) != len(roots) + 1):
        return decline("amide N permits only the oxo-nitrogen core and zero to two carbon organyls", kind)
    arm_sets = []
    for root in roots:
        branch = _carbon_organyl(mol, Chem, root, n1)
        if branch is None:
            return decline("N substituent is not a supported complete carbon organyl", kind)
        arm_sets.append(branch)
    core = {n0, n1, *oids}
    all_heavy = {a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1}
    if (set().union(core, *(set(s) for s in arm_sets)) != all_heavy
            or sum(len(s) for s in arm_sets) + len(core) != len(all_heavy)):
        return decline("unassigned atoms or additional functional groups are outside the family", kind)
    try:
        Chem.SanitizeMol(Chem.Mol(mol))
    except Exception:
        return decline("RDKit valence validation failed", kind)
    return NitrousAmideRecognition("supported", "complete one-core nitric/nitrous amide graph",
                                   n0, n1, oids, roots, tuple(arm_sets), kind)


def _prefix_text(names: tuple[str, ...], stem: str,
                 natures: tuple[str, ...] | None = None):
    """Render P-67 N-organyls through the shared typed P-16 prefix writer."""
    if not names:
        return stem, ()
    if natures is None or len(natures) != len(names):
        return None
    from .prefix_sequences import PrefixUnit, mononuclear_prefix_sequence
    sequence = mononuclear_prefix_sequence(
        PrefixUnit(component=name, nature=nature)
        for name, nature in zip(names, natures)
    )
    if sequence is None:
        return None
    return sequence.text + stem, sequence.units


def nitrous_amide_proposal(mol, Chem=None) -> NitrousAmideProposal | None:
    if Chem is None:
        from rdkit import Chem as RDKitChem
        Chem = RDKitChem
    site = recognize_nitrous_amide(mol, Chem)
    if site.status != "supported":
        return None
    from .component_mapping import mapped_branch
    from .organyl_prefix_components import organyl_prefix_nature
    names = []
    natures = []
    for root in site.organyl_roots:
        name, mapped = mapped_branch(mol, Chem, root, site.amide_nitrogen)
        arm_atoms = site.organyl_atoms[len(names)]
        if not name or frozenset(mapped) != arm_atoms:
            return None
        names.append(name)
        nature = organyl_prefix_nature(mol, root, arm_atoms, name)
        if nature is None:
            return None
        natures.append(nature)
    stem = "nitramide" if site.kind == "nitric" else "nitrous amide"
    rendered = _prefix_text(tuple(names), stem, tuple(natures))
    if rendered is None:
        return None
    prefix, prefix_evidence = rendered
    atoms = frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1)
    return NitrousAmideProposal(prefix, site.acid_nitrogen, site.amide_nitrogen,
                                site.oxygens, site.organyl_roots, site.organyl_atoms,
                                tuple(names), tuple(natures), site.kind, atoms,
                                prefix_evidence)


def nitrous_amide_decision(mol, Chem=None) -> MethodDecision | None:
    proposal = nitrous_amide_proposal(mol, Chem)
    return proposal.as_method_decision() if proposal is not None else None


def nitrous_amide_disposition(mol, Chem=None) -> NitrousAmideDisposition:
    """Return whether a source-shaped core is named, deferred, or outranked.

    External P-41 groups are inspected on a private graph with only the
    recognized oxo-nitrogen core removed. This makes the ordinary functional
    inventory usable for net-neutral nitramides whose formal N+/O− pair would
    otherwise stop inventory construction before a cofunction is seen. The
    projected inventory is evidence of group candidates only; it never proves
    coverage or a name for the original molecule.
    """
    if Chem is None:
        from rdkit import Chem as RDKitChem
        Chem = RDKitChem
    recognition = recognize_nitrous_amide(mol, Chem)
    core_sites = _recognized_local_cores(mol, Chem)
    if recognition.status == "not-applicable":
        return NitrousAmideDisposition("not-applicable", recognition.reason, recognition)
    if recognition.status == "supported" and nitrous_amide_proposal(mol, Chem) is not None:
        return NitrousAmideDisposition("renderable", "complete P-67.1.2.6.3 proposal is available",
                                       recognition, core_sites=core_sites)

    cores = core_sites
    if not cores:
        return NitrousAmideDisposition("not-applicable", "no exact P-67 nitric/nitrous amide core", recognition)
    removed = {idx for n0, _n1, oxygens, _kind in cores for idx in (n0, *oxygens)}
    rw = Chem.RWMol(Chem.Mol(mol))
    for idx in sorted(removed, reverse=True):
        rw.RemoveAtom(idx)
    projected_map = {old: old - sum(deleted < old for deleted in removed)
                     for old in range(mol.GetNumAtoms()) if old not in removed}
    try:
        projected = rw.GetMol()
        Chem.SanitizeMol(projected)
        from .extended_inventory import selection_inventory
        inventory = selection_inventory(projected)
    except Exception:
        return NitrousAmideDisposition("pending", "source-shaped core is unrendered and external class inventory is unavailable",
                                       recognition, core_sites=core_sites)
    core_nitrogens = {n for _n0, n, _os, _kind in cores}
    higher = []
    for group in inventory.groups:
        original_center = next((old for old, new in projected_map.items()
                                if new == group.center), None)
        if original_center is None or original_center in core_nitrogens:
            continue
        rank = group.seniority
        # P-41 says amides are ordered by their corresponding acids.  P-42
        # places carboxylic/sulfonic acids (7a) and carbonic acid (7b) ahead
        # of nitric/nitrous acid (7e), so the corresponding carbon-derived
        # amide, sulfonamide, and urea classes can supply a senior route.
        # Do not generalize to an unknown class-11 kind without its acid proof.
        if (rank[0] < 11
                or (rank[0] == 11 and group.kind in _P42_SENIOR_AMIDE_KINDS)):
            higher.append((group.kind, original_center, rank))
    if higher:
        return NitrousAmideDisposition(
            "higher-class-route",
            "a separately recognized P-41/P-42 class senior to nitric/nitrous amides is present; projected inventory does not prove complete naming",
            recognition, tuple(sorted(higher, key=lambda item: (item[2], item[1], item[0]))),
            bool(inventory.supported),
            core_sites,
        )
    return NitrousAmideDisposition(
        "pending",
        "source-shaped nitric/nitrous amide has no proven senior P-41/P-42 competitor; lower-class fallback is blocked",
        recognition,
        (),
        bool(inventory.supported),
        core_sites,
    )
