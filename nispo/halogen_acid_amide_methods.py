"""Mapped neutral amides of terminal halogen acids.

The original bounded ``R-NH-X`` family is retained.  The typed extension
distinguishes it from ``R2N-X`` and the carbon-free ``NH2-X`` parent case, and
from N-bound higher oxoacid groups such as ``N-BrO2``.  Unsupported structures
are deferred rather than sent through ordinary halo-prefix logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .naming_methods import MethodDecision


RULE = "P-41/P-61.3.2.2/P-67.1.2.2/P-68.5.2/P-68.5.3/P-44/P-16"
EXTENSION_RULE = "P-10/P-11/P-12.2/P-41/P-42.5/P-61.3.2.2/P-62.4/P-67.1.2.2/P-67.1.2.6/P-68.5.2/P-68.5.3/P-44/P-16"
CYCLIC_IMIDE_RULE = "P-66.2.1/P-64.3/P-61.3.2.2/P-41"
_ACID_ADJECTIVE = {9: "hypofluorous", 17: "hypochlorous",
                   35: "hypobromous", 53: "hypoiodous"}
_HEAVY_LIMIT = 64


@dataclass(frozen=True, slots=True)
class HypohalousAmideRecognition:
    status: Literal["supported", "not-applicable", "deferred"]
    reason: str
    nitrogen: int | None = None
    halogen: int | None = None
    carbon_root: int | None = None
    organyl_atoms: frozenset[int] = frozenset()


@dataclass(frozen=True, slots=True)
class HalogenAcidAmideDisposition:
    """Class-level result consumed by selection code without guessing names.

    ``preselected-parent`` is intentionally not an organic ``MethodDecision``:
    P-10/P-11 require carbon for organic PIN status.  ``higher-class-route``
    carries the original centers which must be allowed to continue through
    ordinary principal-group selection. ``cyclic-imide-route`` instead
    requires a completed mapped P-66.2.1 pseudoketone construction.
    """
    status: Literal["not-applicable", "renderable", "preselected-parent",
                    "higher-class-route", "cyclic-imide-route", "pending"]
    reason: str
    nitrogen: int | None = None
    halogen: int | None = None
    organyl_roots: tuple[int, ...] = ()
    organyl_atoms: tuple[frozenset[int], ...] = ()
    centers: frozenset[int] = frozenset()
    acid_adjective: str | None = None
    confidence: Literal["printed-example", "rule-composed", "inference", "unresolved"] = "unresolved"
    class11_subrank: int | None = None
    higher_groups: tuple["SeniorGroupEvidence", ...] = ()
    rule: str = EXTENSION_RULE


@dataclass(frozen=True, slots=True)
class SeniorGroupEvidence:
    """Minimum-ranked original characteristic groups that outrank N–X."""
    kind: str
    rank: tuple[int, int]
    centers: frozenset[int]


@dataclass(frozen=True, slots=True)
class HalogenAcidAmideProposal:
    name: str
    nitrogen: int
    halogen: int
    organyl_roots: tuple[int, ...]
    organyl_atoms: tuple[frozenset[int], ...]
    organyl_names: tuple[str, ...]
    organyl_natures: tuple[str, ...]
    atoms: frozenset[int]
    confidence: str
    prefix_evidence: tuple[object, ...] = ()
    rule: str = EXTENSION_RULE

    def as_method_decision(self) -> MethodDecision:
        roles = tuple((f"organic-organyl-{i + 1}", atoms, ())
                      for i, atoms in enumerate(self.organyl_atoms)) + (
            ("halogen-acid-amide-unit", frozenset((self.nitrogen, self.halogen)), ()),
        )
        covered = frozenset().union(*(role[1] for role in roles))
        if len(covered) != sum(len(role[1]) for role in roles) or covered != self.atoms:
            raise ValueError("halogen-acid-amide roles are not a disjoint full cover")
        return MethodDecision(name=self.name, method="functional-class",
                              principal_centers=frozenset((self.nitrogen,)),
                              atoms=self.atoms, rule=self.rule, component_roles=roles)


@dataclass(frozen=True, slots=True)
class HypohalousAmideProposal:
    name: str
    nitrogen: int
    halogen: int
    carbon_root: int
    organyl_atoms: frozenset[int]
    organyl_name: str
    atoms: frozenset[int]
    rule: str = RULE

    def as_method_decision(self) -> MethodDecision:
        roles = (
            ("organic-organyl", self.organyl_atoms, ()),
            ("hypohalous-acid-amide-unit", frozenset((self.nitrogen, self.halogen)), ()),
        )
        if roles[0][1] & roles[1][1] or frozenset().union(*(r[1] for r in roles)) != self.atoms:
            raise ValueError("hypohalous-amide component roles are not a disjoint full cover")
        return MethodDecision(
            name=self.name,
            method="functional-class",
            principal_centers=frozenset((self.nitrogen,)),
            atoms=self.atoms,
            rule=self.rule,
            component_roles=roles,
        )


def _single(mol, left: int, right: int, Chem) -> bool:
    bond = mol.GetBondBetweenAtoms(left, right)
    return bond is not None and bond.GetBondType() == Chem.BondType.SINGLE


def _cyclic_imide_centers(mol, nitrogen_idx: int, halogen_idx: int, Chem):
    """P-66.2.1 core for the existing mapped heteromonocyclic writer.

    This is a class handoff, not a completed naming construction. Only the
    later whole-graph ring writer can discharge it. Other characteristic
    groups, polycycles and nonordinary N/X valence retain their old routing.
    """
    rings = mol.GetRingInfo().AtomRings()
    if (len(Chem.GetMolFrags(mol)) != 1 or len(rings) != 1
            or not 3 <= len(rings[0]) <= 10
            or any(a.GetFormalCharge() or a.GetNumRadicalElectrons()
                   for a in mol.GetAtoms())):
        return None
    ring = frozenset(rings[0])
    nitrogen, halogen = mol.GetAtomWithIdx(nitrogen_idx), mol.GetAtomWithIdx(halogen_idx)
    if (nitrogen_idx not in ring or nitrogen.GetIsAromatic()
            or nitrogen.GetTotalValence() != 3
            or nitrogen.GetTotalNumHs(includeNeighbors=True)
            or halogen.GetDegree() != 1 or halogen.GetTotalValence() != 1
            or not _single(mol, nitrogen_idx, halogen_idx, Chem)):
        return None
    if any(mol.GetAtomWithIdx(idx).GetAtomicNum() != (7 if idx == nitrogen_idx else 6)
           for idx in ring):
        return None
    centers = frozenset(a.GetIdx() for a in nitrogen.GetNeighbors()
                        if a.GetIdx() != halogen_idx and a.GetAtomicNum() > 1)
    if len(centers) != 2 or not centers <= ring:
        return None
    oxygens = set()
    for idx in centers:
        carbon = mol.GetAtomWithIdx(idx)
        oxo = [a for a in carbon.GetNeighbors() if a.GetAtomicNum() == 8
               and mol.GetBondBetweenAtoms(idx, a.GetIdx()).GetBondType() == Chem.BondType.DOUBLE]
        if (carbon.GetAtomicNum() != 6 or carbon.GetTotalValence() != 4
                or not _single(mol, nitrogen_idx, idx, Chem) or len(oxo) != 1
                or oxo[0].GetDegree() != 1 or oxo[0].GetTotalValence() != 2):
            return None
        oxygens.add(oxo[0].GetIdx())
    # Keep this integration to the one imide functional parent. A competing
    # characteristic-group parent needs its own shared comparison, rather
    # than being forced into this ring handoff.
    if len(oxygens) != 2 or any(
            a.GetAtomicNum() not in {1, 6, 9, 17, 35, 53}
            and a.GetIdx() not in ring | oxygens for a in mol.GetAtoms()):
        return None
    return centers


def _principal_class_evidence(mol, halogen_z: int, n_x_center: int, *, terminal_hypohalous: bool):
    """Return source-ranked higher groups, considering only the best rank."""
    from .extended_inventory import selection_inventory
    from .monobasic_oxoacid_seniority import class11_monobasic_oxoamide_subrank

    target_subrank = (class11_monobasic_oxoamide_subrank(halogen_z, 0)
                      if terminal_hypohalous else None)
    target = (11, target_subrank) if target_subrank is not None else None
    inventory = selection_inventory(mol)
    ranked = []
    for group in inventory.groups:
        # The N-X functional-class unit itself is not an independent senior
        # group, even if the generic inventory describes the same nitrogen as
        # an amine/ammonium center.
        if group.center == n_x_center:
            continue
        kind = group.kind
        rank = group.seniority
        if kind == "nitramide":
            # Nitric acid has two terminal oxos; P-42.5 assigns its exact
            # corresponding amide rank. The shared base inventory's legacy
            # common nitramide/nitrous-amide slot is not sufficiently precise.
            subrank = class11_monobasic_oxoamide_subrank(7, 2)
            if subrank is None:
                continue
            rank = (11, subrank)
        elif kind == "nitrous-amide":
            subrank = class11_monobasic_oxoamide_subrank(7, 1)
            if subrank is None:
                continue
            rank = (11, subrank)
        if rank[0] < 11 or (target is not None and rank < target):
            ranked.append((rank, kind, group.center))
    if not ranked:
        return target, ()
    minimum = min(row[0] for row in ranked)
    grouped = {}
    for rank, kind, center in ranked:
        if rank != minimum:
            continue
        grouped.setdefault((kind, rank), set()).add(center)
    evidence = tuple(SeniorGroupEvidence(kind, rank, frozenset(centers))
                     for (kind, rank), centers in sorted(grouped.items()))
    return target, evidence


def _organyl(mol, Chem, root: int, nitrogen: int) -> frozenset[int] | None:
    """Return a whole saturated carbon tree/plain phenyl with one N bond."""
    if not _single(mol, root, nitrogen, Chem):
        return None
    pending, seen = [root], {root}
    while pending:
        current = pending.pop()
        atom = mol.GetAtomWithIdx(current)
        if atom.GetAtomicNum() != 6 or atom.GetIsotope() or int(atom.GetChiralTag()):
            return None
        if not atom.GetIsAromatic() and atom.GetTotalValence() != 4:
            return None
        for bond in atom.GetBonds():
            neighbor = bond.GetOtherAtom(atom)
            idx = neighbor.GetIdx()
            if idx == nitrogen or neighbor.GetAtomicNum() == 1:
                continue
            if neighbor.GetAtomicNum() != 6 or (
                    not atom.GetIsAromatic() and bond.GetBondType() != Chem.BondType.SINGLE):
                return None
            if idx not in seen:
                seen.add(idx)
                pending.append(idx)
    external = {(idx, n.GetIdx()) for idx in seen
                for n in mol.GetAtomWithIdx(idx).GetNeighbors() if n.GetIdx() not in seen}
    if external != {(root, nitrogen)}:
        return None
    aromatic = {idx for idx in seen if mol.GetAtomWithIdx(idx).GetIsAromatic()}
    if aromatic:
        rings = [tuple(r) for r in mol.GetRingInfo().AtomRings() if set(r) & seen]
        if (aromatic != seen or len(seen) != 6 or len(rings) != 1 or set(rings[0]) != seen
                or len(rings[0]) != 6
                or not all(mol.GetBondBetweenAtoms(a, b).GetIsAromatic()
                           for a, b in zip(rings[0], rings[0][1:] + rings[0][:1]))):
            return None
        return frozenset(seen)
    if any(mol.GetAtomWithIdx(idx).IsInRing() for idx in seen):
        return None
    edge_count = sum(1 for bond in mol.GetBonds()
                     if bond.GetBeginAtomIdx() in seen and bond.GetEndAtomIdx() in seen)
    if edge_count != len(seen) - 1:
        return None
    # Branch rendering supports the carbon-tree cases admitted here, but its
    # public text result and mapped atom set are both checked by the caller.
    return frozenset(seen)


def recognize_hypohalous_amide(mol, Chem=None) -> HypohalousAmideRecognition:
    """Classify a complete graph for the ordinary R-NH-X amide route."""
    if Chem is None:
        from rdkit import Chem as RDKitChem
        Chem = RDKitChem
    nx = [(bond.GetBeginAtom(), bond.GetEndAtom()) for bond in mol.GetBonds()
          if {bond.GetBeginAtom().GetAtomicNum(), bond.GetEndAtom().GetAtomicNum()}
          and 7 in {bond.GetBeginAtom().GetAtomicNum(), bond.GetEndAtom().GetAtomicNum()}
          and (bond.GetBeginAtom().GetAtomicNum() in _ACID_ADJECTIVE
               or bond.GetEndAtom().GetAtomicNum() in _ACID_ADJECTIVE)]
    if not nx:
        return HypohalousAmideRecognition("not-applicable", "no N-halogen bond")
    pairs = []
    for left, right in nx:
        n, x = (left, right) if left.GetAtomicNum() == 7 else (right, left)
        pairs.append((n.GetIdx(), x.GetIdx()))
    nidx, xidx = pairs[0]
    def decline(reason):
        return HypohalousAmideRecognition("deferred", reason, nidx, xidx)
    if len(pairs) != 1 or len(Chem.GetMolFrags(mol)) != 1:
        return decline("requires one connected N-hypohalous-acid amide unit")
    if mol.GetNumHeavyAtoms() > _HEAVY_LIMIT or mol.GetNumAtoms() != mol.GetNumHeavyAtoms():
        return decline("explicit hydrogen or graph over the declared bound")
    if mol.GetStereoGroups() or any(int(b.GetStereo()) or b.GetBondDir() != Chem.BondDir.NONE
                                    for b in mol.GetBonds()):
        return decline("stereo/directional annotations are outside the bounded renderer")
    if any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        return decline("charged or radical graph")
    if any(a.GetIsotope() or int(a.GetChiralTag()) for a in mol.GetAtoms()):
        return decline("isotope or atom stereo requires a separate mapped grammar")
    nitrogen, halogen = mol.GetAtomWithIdx(nidx), mol.GetAtomWithIdx(xidx)
    if (halogen.GetAtomicNum() not in _ACID_ADJECTIVE or halogen.GetDegree() != 1
            or halogen.GetTotalValence() != 1
            or not _single(mol, nidx, xidx, Chem)):
        return decline("halogen is not a terminal single-bonded hypohalous-acid N ligand")
    neighbors = list(nitrogen.GetNeighbors())
    carbon_neighbors = [a for a in neighbors if a.GetAtomicNum() == 6]
    if (nitrogen.GetIsAromatic() or nitrogen.IsInRing() or nitrogen.GetTotalValence() != 3
            or nitrogen.GetTotalNumHs(includeNeighbors=True) != 1
            or len(neighbors) != 2 or len(carbon_neighbors) != 1
            or sum(a.GetAtomicNum() in _ACID_ADJECTIVE for a in neighbors) != 1):
        return decline("requires ordinary neutral R-NH-X with one carbon organyl and one N-H")
    root = carbon_neighbors[0].GetIdx()
    atoms = _organyl(mol, Chem, root, nidx)
    if atoms is None:
        return decline("N-bound group is not a saturated carbon tree")
    owned = set(atoms) | {nidx, xidx}
    heavy = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
    if owned != heavy:
        return decline("additional functions or unassigned heavy atoms are outside the family")
    if not _single(mol, nidx, root, Chem):
        return decline("organyl attachment is not a single N-C bond")
    try:
        Chem.SanitizeMol(Chem.Mol(mol))
    except Exception:
        return decline("RDKit valence validation failed")
    return HypohalousAmideRecognition("supported", "complete neutral R-NH-X graph",
                                      nidx, xidx, root, atoms)


def classify_halogen_acid_amide(mol, Chem=None) -> HalogenAcidAmideDisposition:
    """Recognize N-terminal-hypohalous-acid amide cores with class routing.

    This disposition is deliberately separate from the legacy R-NH-X API.
    It admits NH2-X only as a preselected noncarbon parent construction and
    treats R2N-X as a transparent rule-composed inference.  Higher P-41
    classes are returned with their original centers so callers can continue
    selection; unsupported or ambiguous cases remain pending.
    """
    if Chem is None:
        from rdkit import Chem as RDKitChem
        Chem = RDKitChem
    if sum(atom.GetAtomicNum() == 7 for atom in mol.GetAtoms()) == 2:
        from .hydrazine_methods import halogenated_hydrazine_decision
        if halogenated_hydrazine_decision(mol, Chem) is not None:
            return HalogenAcidAmideDisposition('not-applicable',
                'P-14.3.4.2(b) retains the hydrazine parent for ordinary terminal-halogen substitution',
                rule='P-14.3.4.2(b)')
    nx_bonds = []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        nums = {a.GetAtomicNum(), b.GetAtomicNum()}
        if 7 in nums and nums & set(_ACID_ADJECTIVE):
            n, x = (a, b) if a.GetAtomicNum() == 7 else (b, a)
            nx_bonds.append((n.GetIdx(), x.GetIdx()))
    if not nx_bonds:
        return HalogenAcidAmideDisposition("not-applicable", "no N-halogen bond")
    nidx, xidx = nx_bonds[0]
    acid_subrank = None
    higher_groups = ()
    # Principal-group routing is decided from the original graph before
    # renderer-only annotation, ring, or organyl restrictions.
    early_x = mol.GetAtomWithIdx(xidx)
    terminal_hypohalous = (early_x.GetDegree() == 1 and early_x.GetTotalValence() == 1
                            and _single(mol, nidx, xidx, Chem))
    target_rank, higher_groups = _principal_class_evidence(
        mol, early_x.GetAtomicNum(), nidx, terminal_hypohalous=terminal_hypohalous)
    def pending(reason, *, roots=(), branches=(), status="pending", centers=frozenset()):
        return HalogenAcidAmideDisposition(status, reason, nidx, xidx,
                                           tuple(roots), tuple(branches), centers,
                                           (_ACID_ADJECTIVE.get(mol.GetAtomWithIdx(xidx).GetAtomicNum())
                                            if terminal_hypohalous else None),
                                           class11_subrank=(target_rank[1] if target_rank else None),
                                           higher_groups=higher_groups)
    if higher_groups:
        high_centers = frozenset().union(*(item.centers for item in higher_groups))
        return pending("a source-ranked superior characteristic group owns principal selection",
                       status="higher-class-route", centers=high_centers)
    if len(nx_bonds) != 1 or len(Chem.GetMolFrags(mol)) != 1:
        return pending("requires one connected N-halogen-acid amide core")
    imide_centers = _cyclic_imide_centers(mol, nidx, xidx, Chem)
    if imide_centers:
        # P-66.2.1 explicitly prefers a cyclic pseudoketone for N-bromo
        # imides. Other terminal halogens can use the same mapped general
        # construction, but retain the open P-61/P-66 method preference.
        return HalogenAcidAmideDisposition(
            "cyclic-imide-route", "cyclic imide requires the mapped heterocyclic pseudoketone writer",
            nidx, xidx, centers=imide_centers,
            confidence="rule-composed" if early_x.GetAtomicNum() == 35 else "inference",
            rule=CYCLIC_IMIDE_RULE)
    if mol.GetNumHeavyAtoms() > _HEAVY_LIMIT or mol.GetNumAtoms() != mol.GetNumHeavyAtoms():
        return pending("explicit hydrogen or graph over the declared bound")
    if (mol.GetStereoGroups() or any(int(b.GetStereo()) or b.GetBondDir() != Chem.BondDir.NONE
                                     for b in mol.GetBonds())):
        return pending("stereo/directional annotations are outside this renderer")
    if any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        return pending("charged or radical graph")
    if any(a.GetIsotope() or int(a.GetChiralTag()) for a in mol.GetAtoms()):
        return pending("isotope or atom stereo requires a separate mapped grammar")
    nitrogen, halogen = mol.GetAtomWithIdx(nidx), mol.GetAtomWithIdx(xidx)
    acid_subrank = target_rank[1] if target_rank else None
    # Distinguish N-BrO2 and related higher oxoacid amides from the terminal
    # hypohalous-acid unit.  They remain pending for their own P-68.5.3 grammar.
    if not terminal_hypohalous:
        return pending("N-X is not a terminal hypohalous-acid ligand; higher oxoacid families are separate")
    if (nitrogen.GetIsAromatic() or nitrogen.IsInRing() or nitrogen.GetTotalValence() != 3
            or any(b.GetBondType() != Chem.BondType.SINGLE for b in nitrogen.GetBonds())):
        return pending("N-X nitrogen is not an ordinary acyclic trivalent amide nitrogen")
    # Apply the clear P-41 senior classes before restricting the N substituent
    # grammar.  A higher acid/ester/anhydride/acid-halide graph must be allowed
    # to proceed to its own renderer even when the N-bound carbon arm contains
    # that characteristic group and is not an admitted alkyl arm here.
    try:
        from .functional_inventory import functional_inventory
        inventory = functional_inventory(mol)
        if inventory.supported:
            higher = [g for g in inventory.groups if g.seniority[0] < 11
                      and g.center != nidx]
            if higher:
                return pending("a P-41-superior characteristic group owns principal selection",
                               status="higher-class-route",
                               centers=frozenset(g.center for g in higher))
    except Exception:
        return pending("principal-class comparison is unavailable")
    neighbors = list(nitrogen.GetNeighbors())
    carbon_neighbors = [a for a in neighbors if a.GetIdx() != xidx and a.GetAtomicNum() == 6]
    other_heavy = [a for a in neighbors if a.GetIdx() != xidx and a.GetAtomicNum() > 1]
    if len(carbon_neighbors) != len(other_heavy) or any(a.GetIdx() not in {c.GetIdx() for c in carbon_neighbors}
                                                        for a in other_heavy):
        return pending("N-bound groups are not ordinary carbon organyls")
    hydrogen_count = int(nitrogen.GetTotalNumHs(includeNeighbors=True))
    if len(carbon_neighbors) + hydrogen_count != 2:
        return pending("requires two N valence positions besides terminal X")
    if len(carbon_neighbors) == 0:
        if hydrogen_count != 2:
            return pending("carbon-free parent is not NH2-X")
        heavy = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
        if heavy != {nidx, xidx}:
            return pending("NH2-X has unassigned heavy atoms")
        return pending("preselected noncarbon parent; not an organic whole-molecule PIN",
                       status="preselected-parent", centers=frozenset((nidx,)))
    if len(carbon_neighbors) not in {1, 2}:
        return pending("only R-NH-X and R2N-X are covered")
    if hydrogen_count != 2 - len(carbon_neighbors):
        return pending("N-H count does not match ordinary RnN-X valence")
    roots = tuple(a.GetIdx() for a in carbon_neighbors)
    branches = []
    for root in roots:
        branch = _organyl(mol, Chem, root, nidx)
        if branch is None:
            return pending("an N-bound group is not a saturated carbon tree or plain phenyl",
                           roots=roots)
        branches.append(branch)
    if len(set().union(*branches)) != sum(map(len, branches)):
        return pending("organyl branches overlap")
    owned = {nidx, xidx} | set().union(*(set(b) for b in branches))
    heavy = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
    if owned != heavy:
        return pending("additional functions or unassigned heavy atoms are outside the family",
                       roots=roots, branches=branches)
    try:
        Chem.SanitizeMol(Chem.Mol(mol))
    except Exception:
        return pending("RDKit valence validation failed", roots=roots, branches=branches)
    published_methyl_chloride = (len(roots) == 1 and len(branches[0]) == 1
                                 and halogen.GetAtomicNum() == 17)
    published_ethyl_chloride = (len(roots) == 1 and len(branches[0]) == 2
                                and halogen.GetAtomicNum() == 17)
    conf = ("printed-example" if published_methyl_chloride or published_ethyl_chloride
            else "rule-composed" if len(roots) == 1 else "inference")
    return HalogenAcidAmideDisposition("renderable", "complete ordinary neutral N-terminal-X graph",
                                       nidx, xidx, roots, tuple(branches),
                                       frozenset((nidx,)), _ACID_ADJECTIVE[halogen.GetAtomicNum()],
                                       conf, acid_subrank)


def halogen_acid_amide_proposal(mol, Chem=None) -> HalogenAcidAmideProposal | None:
    """Render bounded R-NH-X/R2N-X graphs; NH2-X is a parent-only result."""
    if Chem is None:
        from rdkit import Chem as RDKitChem
        Chem = RDKitChem
    result = classify_halogen_acid_amide(mol, Chem)
    if result.status != "renderable":
        return None
    from .component_mapping import mapped_branch
    names, maps = [], []
    natures = []
    for root in result.organyl_roots:
        name, atom_map = mapped_branch(mol, Chem, root, result.nitrogen)
        atom_set = frozenset(atom_map)
        expected = result.organyl_atoms[len(names)]
        if not name or atom_set != expected:
            return None
        names.append(name)
        maps.append(atom_set)
        from .organyl_prefix_components import organyl_prefix_nature
        nature = organyl_prefix_nature(mol, root, atom_set, name)
        if nature is None:
            return None
        natures.append(nature)
    adjective = result.acid_adjective
    if len(names) == 1:
        name = f"{names[0]}{adjective} amide"
        prefix_evidence = ()
    else:
        from .prefix_sequences import PrefixUnit, mononuclear_prefix_sequence
        sequence = mononuclear_prefix_sequence(
            PrefixUnit(component=component, nature=nature)
            for component, nature in zip(names, natures)
        )
        if sequence is None:
            return None
        name = f"{sequence.text}{adjective} amide"
        prefix_evidence = sequence.units
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    return HalogenAcidAmideProposal(name, result.nitrogen, result.halogen,
                                    result.organyl_roots, tuple(maps), tuple(names),
                                    tuple(natures), heavy, result.confidence,
                                    prefix_evidence)


def hypohalous_amide_proposal(mol, Chem=None) -> HypohalousAmideProposal | None:
    """Render the mapped preferred hypohalous-amide name if fully supported."""
    if Chem is None:
        from rdkit import Chem as RDKitChem
        Chem = RDKitChem
    recognition = recognize_hypohalous_amide(mol, Chem)
    if recognition.status != "supported":
        return None
    from .component_mapping import mapped_branch
    organyl_name, mapped = mapped_branch(
        mol, Chem, recognition.carbon_root, recognition.nitrogen)
    if not organyl_name or frozenset(mapped) != recognition.organyl_atoms:
        return None
    adjective = _ACID_ADJECTIVE[mol.GetAtomWithIdx(recognition.halogen).GetAtomicNum()]
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    return HypohalousAmideProposal(
        name=f"{organyl_name}{adjective} amide",
        nitrogen=recognition.nitrogen,
        halogen=recognition.halogen,
        carbon_root=recognition.carbon_root,
        organyl_atoms=recognition.organyl_atoms,
        organyl_name=organyl_name,
        atoms=heavy,
    )


def hypohalous_amide_decision(mol, Chem=None) -> MethodDecision | None:
    proposal = hypohalous_amide_proposal(mol, Chem)
    return proposal.as_method_decision() if proposal is not None else None


def mononuclear_haloazane_name(mol, Chem):
    """A mapped general P-21/P-68.5 construction for one neutral NH2-X.

    This does not resolve the preferred noncarbon-parent comparison. The
    complete two-atom graph gives each isotope its original element site,
    rather than inheriting a numerical locant from a retained template.
    """
    from .component_mapping import MappedComponent, decorate_component
    from .constants import HALO

    if mol.GetNumAtoms() != 2 or mol.GetNumHeavyAtoms() != 2:
        return ''
    nitrogens = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7]
    if len(nitrogens) != 1:
        return ''
    nitrogen = nitrogens[0]
    neighbors = tuple(nitrogen.GetNeighbors())
    if len(neighbors) != 1:
        return ''
    halogen = neighbors[0]
    if (halogen.GetAtomicNum() not in _ACID_ADJECTIVE
            or halogen.GetIsotope()
            or nitrogen.GetTotalValence() != 3 or nitrogen.GetTotalNumHs() != 2
            or halogen.GetTotalValence() != 1 or halogen.GetTotalNumHs()
            or mol.GetBondBetweenAtoms(nitrogen.GetIdx(), halogen.GetIdx()).GetBondTypeAsDouble() != 1
            or any(atom.GetFormalCharge() or atom.GetNumRadicalElectrons()
                   or int(atom.GetChiralTag()) for atom in mol.GetAtoms())
            or mol.GetStereoGroups()):
        return ''
    component = MappedComponent(
        parent='azane',
        atom_locants=((nitrogen.GetIdx(), '1'),),
        atoms=frozenset((nitrogen.GetIdx(),)),
        embedded_atoms=frozenset((halogen.GetIdx(),)),
        prefixes=(('', HALO[halogen.GetSymbol()]),),
        prefix_mode='unlocanted',
        prefix_natures=((HALO[halogen.GetSymbol()], 'simple'),),
        isotope_unlocanted_elements=(7,))
    return decorate_component(mol, Chem, component)
