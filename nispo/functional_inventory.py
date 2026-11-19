"""Shared graph inventory for supported neutral organic characteristic groups.

The keys implement the applicable P-41 class order, including the distinction
between ordinary esters/amides and cyclic or hidden amides (pseudoketones).
They are not a universal ordering for ions, radicals or replacement chemistry.
Each site retains input atom identities; a renderer must prove its own coverage.
"""

from collections import OrderedDict
from dataclasses import dataclass, replace
from functools import cached_property, lru_cache
from threading import RLock
from typing import Literal

from .molecule_iteration import iter_atoms
from .molecule_cache import mol_cache_get, mol_cache_set
from .monobasic_oxoacid_seniority import class11_monobasic_oxoamide_subrank


@lru_cache(maxsize=1)
def _neutral_carboxyl_query(Chem):
    return Chem.MolFromSmarts("[#6][C;X3;!R](=[O;X1;+0])[O;X2;H1;+0]")


_SENIORITY = {
    "carboxylate": (4, 0),
    "carbamate-anion": (4, 1),
    "sulfonate": (4, 1),
    "sulfinate": (4, 2),
    "phosphonate": (4, 3),
    "phosphinate": (4, 4),
    "phosphate": (4, 5),
    "sulfur-oxoacid-ester-anion": (4, 6),
    "ammonium": (6, 0),
    "sulfonic-acid": (7, 1),
    "sulfinic-acid": (7, 2),
    "phosphonic-acid": (7, 4),
    "phosphinic-acid": (7, 5),
    "phosphoric-acid": (7, 6),
    "phosphate-acid-ester": (9, 3),
    "sulfur-oxoacid-ester": (9, 3),
    "sulfonamide": (11, 1),
    "sulfamide": (11, 2),
    "thiosemicarbazone": (11, 2),
    "sulfonimidamide": (11, 1, 1),
    # Compatibility projection only. Source-principal selection uses the
    # mapped P-41/P-42 semantic facts attached to each inventory.
    "sulfinamide": (11, 1, 1),
    "thiol": (17, 1),
    "acid": (7, 0),
    "carbonic-acid": (7, 3),
    "anhydride": (8, 0),
    "ester": (9, 0),
    "carbamate": (9, 1),
    "carbonate": (9, 2),
    "acid-halide": (10, 0),
    "sulfonyl-halide": (10, 1),
    "amide": (11, 0),
    # P-41 orders amides by their corresponding acids; P-42.5 places nitric
    # acid above nitrous acid by oxidation state. Preserve the existing class
    # rank tuple while using the literal source-table order for both.
    "nitramide": (11, class11_monobasic_oxoamide_subrank(7, 2)),
    "nitrous-amide": (11, class11_monobasic_oxoamide_subrank(7, 1)),
    "urea": (11, 2),
    "amidine": (11, 0, 1),
    "amidoxime": (11, 0, 1),
    "guanidine": (11, 2, 1),
    "cyanamide": (11, 3),
    "hydrazide": (12, 0),
    "cyanohydrazide": (12, 3),
    "nitrile": (14, 0),
    "aldehyde": (15, 0),
    "ketone": (16, 0),
    "lactone": (16, 0),
    "lactam": (16, 0),
    "pseudoketone": (16, 0),
    "alcohol": (17, 0),
    "amine": (19, 0),
    "peroxol": (18, 0),
    "imine": (20, 0),
    "oxime": (20, 0),
    "oxime-ether": (20, 0),
}


@dataclass(frozen=True, slots=True)
class CharacteristicGroup:
    kind: str
    center: int
    atoms: tuple[int, ...]
    owner: int = -1
    ligand: int = -1
    seniority_rank: int = 0

    @property
    def seniority(self):
        if self.kind in {"peroxol", "hydrazide"}:
            return (18 if self.kind == "peroxol" else 12, self.seniority_rank)
        return _SENIORITY[self.kind]


@dataclass(frozen=True)
class FunctionalInventory:
    groups: tuple[CharacteristicGroup, ...]
    supported: bool
    charge_role_proofs: tuple = ()
    source_group_facts: tuple = ()
    source_class_evidence: tuple = ()

    @cached_property
    def source_frontier(self):
        """Return the bounded source-based principal-group result.

        Only classes with an implemented, source-closed within-class order are
        opted in. Other classes retain the historical principal projection
        and are explicitly marked outside this semantic consumer's scope.
        """
        if not self.supported or not self.groups:
            return SourcePrincipalFrontier("unavailable", (), reason="No supported group inventory.")
        if len(self.source_group_facts) != len(self.groups):
            return SourcePrincipalFrontier(
                "barrier", self.groups,
                reason="Source priority facts are missing or not aligned with inventory groups.",
            )
        if any(fact is None or fact.class_rank is None for fact in self.source_group_facts):
            return SourcePrincipalFrontier(
                "barrier", self.groups,
                reason="At least one recognized group lacks its P-41 class rank.",
            )
        min_class = min(fact.class_rank for fact in self.source_group_facts)
        from .group_priority import SOURCE_FRONTIER_CLASS_RANKS

        if min_class not in SOURCE_FRONTIER_CLASS_RANKS:
            return SourcePrincipalFrontier(
                "outside-scope", self.principal_legacy,
                reason=f"P-41 class {min_class} is outside the implemented source-comparison scope.",
            )
        fact_by_group_id = {id(group): fact for group, fact in zip(self.groups, self.source_group_facts)}
        class_groups = tuple(
            group for group in self.groups
            if fact_by_group_id[id(group)].class_rank == min_class
        )
        class_facts = tuple(fact_by_group_id[id(group)] for group in class_groups)
        # Class 11 retains its historical inventory-completeness contract,
        # including a singleton. Other source classes resolve pairwise: a
        # missing detail in a known lower class cannot block a proved winner,
        # while an unknown same-class comparison remains in the frontier.
        if min_class == 11 and any(not _complete_priority_facts(fact, min_class)
                                   for fact in class_facts):
            return SourcePrincipalFrontier(
                "barrier", class_groups,
                reason=f"At least one minimum-class-{min_class} group lacks complete applicable source facts.",
            )
        from .group_priority import priority_frontier_analysis

        analysis = priority_frontier_analysis(
            class_groups, lambda group: fact_by_group_id.get(id(group)),
        )
        return SourcePrincipalFrontier(
            analysis.status, analysis.items, analysis.unknown_pairs, analysis.cycle,
            analysis.reason,
        )

    @cached_property
    def principal(self):
        result = self.source_frontier
        if result.status in {"selected", "tie", "barrier"}:
            return result.groups
        return self.principal_legacy

    @cached_property
    def principal_legacy(self):
        """Historical two-field tuple projection for compatibility."""
        if not self.supported or not self.groups:
            return ()
        first = min(g.seniority for g in self.groups)
        return tuple(g for g in self.groups if g.seniority == first)


@dataclass(frozen=True, slots=True)
class SourcePrincipalFrontier:
    """Typed status for the bounded opt-in P-41/P-42 source comparison."""

    status: Literal["selected", "tie", "barrier", "outside-scope", "unavailable"]
    groups: tuple[CharacteristicGroup, ...]
    unknown_pairs: tuple = ()
    cycle: bool = False
    reason: str = ""


def _complete_priority_facts(facts, class_rank):
    """Check applicable within-class source facts for the frontier class."""
    if facts is None or facts.class_rank != class_rank:
        return False
    if class_rank not in {7, 10, 11, 12, 17}:
        return True
    if class_rank == 17:
        return facts.hydroxy_analogue in {"O", "S"}
    if class_rank == 10 and facts.leaving_group not in {
        "F", "Cl", "Br", "I", "N3", "CN", "NC", "NCO", "NCS", "NCSe", "NCTe", "CNO",
    }:
        return False
    if facts.corresponding_acid is None:
        return False
    acid = facts.corresponding_acid
    if acid.category == "7a":
        known = (acid.p42_1_family in {
            "carboxylic", "sulfonic", "sulfinic", "selenonic", "seleninic",
            "telluronic", "tellurinic",
        } and acid.p42_1_form in {"ordinary", "peroxy", "imidic", "hydrazonic"})
        if not known:
            return False
        if acid.p42_1_form != "ordinary":
            replacement = facts.replacement
            if replacement is None:
                return False
            if (replacement.replaceable_atom_counts is None
                    or len(replacement.replaceable_atom_counts) != 6
                    or replacement.peroxy_group_counts is None
                    or len(replacement.peroxy_group_counts) != 4
                    or replacement.acidic_hydroxy_atom_counts is None
                    or len(replacement.acidic_hydroxy_atom_counts) != 4):
                return False
        return True
    if acid.category == "7b":
        return acid.p42_2_family in {
            "polycarbonic", "tetracarbonic", "tricarbonic", "dicarbonic",
            "carbonic", "cyanic",
        }
    if acid.category == "7c":
        return (acid.central_element in {"N", "P", "As", "Sb", "B"}
                and all(value is not None for value in (
            acid.central_element, acid.central_atom_count, acid.homopolyacid,
            acid.contiguous_central_atoms, acid.acidic_hydroxyl_count,
            acid.oxidation_number,
        )))
    if acid.category == "7d":
        return (acid.central_element in {"P", "As", "Sb", "Si", "B", "S", "Se", "Te"}
                and all(value is not None for value in (
            acid.central_element, acid.central_atom_count, acid.homopolyacid,
            acid.contiguous_central_atoms, acid.acidic_hydroxyl_count,
            acid.oxidation_number,
        )))
    if acid.category == "7e":
        return acid.central_element in {"N", "F", "Cl", "Br", "I"} and acid.oxidation_number is not None
    return False


def functional_inventory(mol):
    cached = mol_cache_get(mol, "functional_inventory")
    if cached is not None:
        return cached
    result = _inventory(mol)
    return mol_cache_set(mol, "functional_inventory", result)


_INVENTORY_CACHE = OrderedDict()
_INVENTORY_CACHE_LOCK = RLock()
_INVENTORY_CACHE_MAXSIZE = 2048


def _inventory(mol, *, extended=False):
    """Share immutable classification facts across exact graph copies.

    Proof builders deliberately use fresh molecules. Their classification can
    be shared only with identical native state and the same inventory scope;
    compatibility and extended recognition are never interchangeable.
    """
    from rdkit import Chem

    try:
        state = mol.ToBinary(Chem.PropertyPickleOptions.AllProps)
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return _uncached_inventory(mol, extended=extended)
    key = (state, extended, _inventory_build, _uncached_inventory)
    with _INVENTORY_CACHE_LOCK:
        cached = _INVENTORY_CACHE.get(key)
        if cached is not None:
            _INVENTORY_CACHE.move_to_end(key)
            return cached
    result = _uncached_inventory(mol, extended=extended)
    # A hit must not skip a native side effect of the original recognizer.
    try:
        unchanged = mol.ToBinary(Chem.PropertyPickleOptions.AllProps) == state
    except (AttributeError, TypeError, ValueError, RuntimeError):
        unchanged = False
    if unchanged:
        with _INVENTORY_CACHE_LOCK:
            _INVENTORY_CACHE[key] = result
            _INVENTORY_CACHE.move_to_end(key)
            if len(_INVENTORY_CACHE) > _INVENTORY_CACHE_MAXSIZE:
                _INVENTORY_CACHE.popitem(last=False)
    return result


def _uncached_inventory(mol, *, extended=False, subordinate_thioxo=()):
    """Build an inventory and attach aligned semantic priority evidence."""
    result = _inventory_build(mol, extended=extended, subordinate_thioxo=subordinate_thioxo)
    if any(group.kind in {'acid', 'carbonic-acid'} for group in result.groups):
        from rdkit import Chem
        from .carbon_acid_derivative_methods import carbon_acid_source_class

        source = carbon_acid_source_class(mol, Chem)
        if source is not None and any(
                group.center == source.center
                and group.kind in {'acid', 'carbonic-acid'}
                for group in result.groups):
            # P-65.1.8 distinguishes permitted substitution of formic acid
            # from the listed carbonic-acid functional replacements. The
            # same superficial C(=O)(OH) pattern does not fix its acid class.
            groups = tuple(
                replace(group, kind=source.principal_kind)
                if group.center == source.center
                and group.kind in {'acid', 'carbonic-acid'} else group
                for group in result.groups)
            result = replace(result, groups=groups,
                             source_class_evidence=(source,))
    from .group_priority import functional_group_priority_facts

    facts = tuple(functional_group_priority_facts(group, mol) for group in result.groups)
    return replace(result, source_group_facts=facts)


def nitrogen_oxoamide_sites(mol):
    """Recognize exact acyclic nitric/nitrous amide cores by original IDs.

    This is a class-recognition operation only. It does not choose a name or
    decide principal-group seniority. Core isotopes are deliberately ignored
    here; naming and isotope coverage are separate obligations. A hydrazide
    chain, cyclic amide nitrogen, arbitrary charge, or noncanonical charged
    nitramide form is not admitted.
    """
    sites = []
    for n0 in (a for a in iter_atoms(mol) if a.GetAtomicNum() == 7):
        if n0.IsInRing() or n0.GetIsAromatic():
            continue
        for n1 in (a for a in n0.GetNeighbors() if a.GetAtomicNum() == 7):
            if (n1.IsInRing() or n1.GetIsAromatic() or n1.GetFormalCharge()
                    or n1.GetTotalValence() != 3
                    or any(b.GetBondTypeAsDouble() != 1 for b in n1.GetBonds())):
                continue
            if mol.GetBondBetweenAtoms(n0.GetIdx(), n1.GetIdx()).GetBondTypeAsDouble() != 1:
                continue
            # Class membership is determined by the non-hydrogen substituents
            # on the amide nitrogen. Explicit protium/deuterium atoms are
            # annotations on that nitrogen, not additional organyl arms.
            external = [a for a in n1.GetNeighbors()
                        if a.GetIdx() != n0.GetIdx() and a.GetAtomicNum() > 1]
            if any(a.GetAtomicNum() not in {6, 16} for a in external):
                continue  # Hydrazides are a separate P-67 class.
            oxygens = [a for a in n0.GetNeighbors() if a.GetAtomicNum() == 8]
            if len(oxygens) == 1:
                oxygen = oxygens[0]
                if (not n0.GetFormalCharge() and n0.GetDegree() == 2
                        and n0.GetTotalValence() == 3
                        and oxygen.GetDegree() == 1 and not oxygen.GetFormalCharge()
                        and mol.GetBondBetweenAtoms(n0.GetIdx(), oxygen.GetIdx()).GetBondTypeAsDouble() == 2):
                    sites.append(("nitrous-amide", n0.GetIdx(), n1.GetIdx(),
                                  (oxygen.GetIdx(),)))
            elif len(oxygens) == 2:
                doubles = [o for o in oxygens
                           if mol.GetBondBetweenAtoms(n0.GetIdx(), o.GetIdx()).GetBondTypeAsDouble() == 2]
                singles = [o for o in oxygens
                           if mol.GetBondBetweenAtoms(n0.GetIdx(), o.GetIdx()).GetBondTypeAsDouble() == 1]
                if (n0.GetFormalCharge() == 1 and n0.GetDegree() == 3
                        and n0.GetTotalValence() == 4 and len(doubles) == len(singles) == 1
                        and not doubles[0].GetFormalCharge() and doubles[0].GetDegree() == 1
                        and singles[0].GetFormalCharge() == -1 and singles[0].GetDegree() == 1):
                    sites.append(("nitramide", n0.GetIdx(), n1.GetIdx(),
                                  tuple(sorted(o.GetIdx() for o in oxygens))))
    return tuple(sites)


def _ordinary_annular_cn_bond(mol, carbon, nitrogen, *, allow_polycycle=False):
    """A localized C=N bond in an ordinary P-22/P-25/P-54 skeleton.

    Admit ordinary C/N/O/S monocycles and small edge-fused systems with
    standard neutral valence and noncumulative annular bonds. The extended
    inventory also permits up to four edge-fused faces in a branch.
    """
    bond = mol.GetBondBetweenAtoms(carbon.GetIdx(), nitrogen.GetIdx())
    if bond is None or not bond.IsInRing() or not nitrogen.IsInRing():
        return False
    rings = mol.GetRingInfo().AtomRings()
    candidates = [ring for ring in rings if carbon.GetIdx() in ring
                  and nitrogen.GetIdx() in ring and 3 <= len(ring) <= 7]
    if len(candidates) != 1:
        return False
    component = {frozenset(candidates[0])}
    core = set(candidates[0])
    pending = set(map(frozenset, rings)) - component
    while attached := {ring for ring in pending if ring & core}:
        for ring in attached:
            component.add(ring)
            core.update(ring)
        pending -= attached
    if len(component) == 2:
        left, right = tuple(component)
        shared = left & right
        if (len(left) not in {5,6} or len(right) not in {5,6}
                or len(shared) != 2 or mol.GetBondBetweenAtoms(*sorted(shared)) is None
                or any(mol.GetAtomWithIdx(i).GetAtomicNum() not in {6,7} for i in shared)):
            return False
    elif len(component) != 1:
        if not allow_polycycle or len(component) > 4:
            return False
        faces = tuple(component)
        if any(len(face) not in {5, 6, 7} for face in faces):
            return False
        for j, face in enumerate(faces):
            for other in faces[:j]:
                shared = face & other
                if shared and (len(shared) != 2 or mol.GetBondBetweenAtoms(*sorted(shared)) is None):
                    return False
    valence = {6:4,7:3,8:2,16:2}
    for idx in core:
        atom = mol.GetAtomWithIdx(idx)
        if (atom.GetAtomicNum() not in valence
                or atom.GetTotalValence() != valence[atom.GetAtomicNum()]
                or atom.GetFormalCharge() or atom.GetNumRadicalElectrons()):
            return False
        annular = [edge for edge in atom.GetBonds() if edge.GetOtherAtomIdx(idx) in core]
        if (len(annular) not in {2,3}
                or any(edge.GetBondTypeAsDouble() not in {1,1.5,2} for edge in annular)
                or sum(edge.GetBondTypeAsDouble()==2 for edge in annular)>1):
            return False
    return True


def _simple_annular_peroxide_carbonyl(mol, carbon, oxygen):
    """Prove a complete unannotated monocyclic C/O peroxide carbonyl graph.

    This is an inventory admission proof, not a claim of preferred naming.
    All heavy atoms must be in the sole ring or be the terminal carbonyl O.
    """
    atoms = tuple(mol.GetAtoms())
    if any(a.GetAtomicNum() not in {1, 6, 8} or a.GetFormalCharge()
           or a.GetNumRadicalElectrons() or a.GetIsotope() or int(a.GetChiralTag())
           for a in atoms):
        return False
    if any(int(b.GetStereo()) for b in mol.GetBonds()):
        return False
    rings = mol.GetRingInfo().AtomRings()
    if len(rings) != 1:
        return False
    ring = set(rings[0])
    c, o = carbon.GetIdx(), oxygen.GetIdx()
    if c not in ring or o not in ring:
        return False
    ring_oxygens = [i for i in ring if mol.GetAtomWithIdx(i).GetAtomicNum() == 8]
    if len(ring_oxygens) != 2 or o not in ring_oxygens:
        return False
    other = next(i for i in ring_oxygens if i != o)
    bond = mol.GetBondBetweenAtoms(o, other)
    if bond is None or not bond.IsInRing() or bond.GetBondTypeAsDouble() not in {1, 1.5}:
        return False
    extra = [a for a in atoms if a.GetAtomicNum() > 1 and a.GetIdx() not in ring]
    if len(extra) != 1 or extra[0].GetAtomicNum() != 8 or extra[0].GetDegree() != 1:
        return False
    bond = mol.GetBondBetweenAtoms(c, extra[0].GetIdx())
    return bond is not None and bond.GetBondTypeAsDouble() == 2


def _spectator_silicon(atom):
    """Ordinary organosilane or silyl ether, with no acid-derived Si role."""
    if (atom.GetFormalCharge() or atom.GetTotalValence() != 4
            or atom.GetNumRadicalElectrons() or atom.IsInRing()
            or any(bond.GetBondTypeAsDouble() != 1 for bond in atom.GetBonds())):
        return False
    for neighbor in atom.GetNeighbors():
        if neighbor.GetAtomicNum() in {1, 6}:
            continue
        if (neighbor.GetAtomicNum() != 8 or neighbor.GetDegree() != 2
                or neighbor.GetFormalCharge()):
            return False
        other = next(a for a in neighbor.GetNeighbors() if a.GetIdx() != atom.GetIdx())
        if (other.GetAtomicNum() != 6 or any(
                bond.GetBondTypeAsDouble() > 1
                and bond.GetOtherAtom(other).GetAtomicNum() not in {1, 6}
                for bond in other.GetBonds())):
            return False
    return True


def _inventory_build(mol, *, extended=False, subordinate_thioxo=()):
    atoms = tuple(iter_atoms(mol))
    from rdkit import Chem

    # These admission facts are harmless native reads, even before ring or
    # valence initialization. Keep duck-typed/custom graphs on their original
    # recognizer path. Buckets retain the native atom ordering.
    native_atoms = type(mol) is Chem.Mol
    atom_numbers = tuple(atom.GetAtomicNum() for atom in atoms) if native_atoms else None
    atoms_by_number = {}
    if atom_numbers is not None:
        for atom, number in zip(atoms, atom_numbers):
            atoms_by_number.setdefault(number, []).append(atom)
    oxoamide_sites = (nitrogen_oxoamide_sites(mol)
                     if atom_numbers is None or 7 in atoms_by_number else ())
    from .neutral_charge_components import (
        neutral_charge_atom_ids, neutral_charge_role_proofs,
    )
    # Carbon-bound P-61.5/P-61.7 groups and existing exact P-67 nitramides
    # receive typed charge evidence only when the whole graph is net-neutral
    # and every charged atom is covered by those recognized motifs.
    # Every available charge-role proof owns a charged atom. A native graph
    # with no charges cannot supply one. On an uninitialized valence cache,
    # preserve the recognizer's original reads and error behavior.
    charge_role_proofs = (neutral_charge_role_proofs(mol, oxoamide_sites)
                         if atom_numbers is None or mol.NeedsUpdatePropertyCache()
                         or any(atom.GetFormalCharge() for atom in atoms) else ())
    charge_exceptions = {
        idx for kind, n0, _n1, oxygens in oxoamide_sites if kind == "nitramide"
        for idx in (n0, *[o for o in oxygens
                          if mol.GetAtomWithIdx(o).GetFormalCharge() == -1])
    }
    charge_exceptions.update(neutral_charge_atom_ids(charge_role_proofs))
    from .peroxol_components import recognize_peroxols
    # A C-X-Y-H site necessarily contains two distinct chalcogen atoms.
    peroxol_groups = (recognize_peroxols(mol)
                      if atom_numbers is None
                      or sum(len(atoms_by_number.get(number, ()))
                             for number in (8, 16, 34, 52)) >= 2 else ())
    peroxol_sulfur = {i for group in peroxol_groups for i in group.atoms
                       if mol.GetAtomWithIdx(i).GetAtomicNum() == 16}
    peroxol_se_te = {i for group in peroxol_groups for i in group.atoms
                     if mol.GetAtomWithIdx(i).GetAtomicNum() in {34, 52}}
    all_sulfur = {a.GetIdx() for a in (atoms_by_number.get(16, ())
                  if atom_numbers is not None else atoms) if a.GetAtomicNum() == 16}
    all_se_te = {a.GetIdx() for a in (tuple(atoms_by_number.get(34, ()))
                 + tuple(atoms_by_number.get(52, ()))
                 if atom_numbers is not None else atoms) if a.GetAtomicNum() in {34, 52}}
    # A neighboring ordinary thiol is admitted only to compare its class-17
    # seniority against a recognized class-18 chalcogen-peroxol site. Broader
    # sulfur selection remains with the extended inventory.
    thiol_groups = []
    for idx in sorted(all_sulfur - peroxol_sulfur):
        atom = mol.GetAtomWithIdx(idx)
        carbon_neighbors = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 6]
        if (peroxol_groups and not atom.GetFormalCharge()
                and not atom.GetNumRadicalElectrons() and not atom.GetIsotope()
                and not atom.IsInRing() and atom.GetDegree() == 1
                and len(carbon_neighbors) == 1 and atom.GetTotalNumHs(includeNeighbors=True) == 1
                and all(b.GetBondTypeAsDouble() == 1 for b in atom.GetBonds())):
            thiol_groups.append(CharacteristicGroup(
                "thiol", idx, (idx,), carbon_neighbors[0].GetIdx(), idx))
    eligible_sulfur = peroxol_sulfur | {g.center for g in thiol_groups}
    # P-63.3.1 C–S–S–C bridges are typed substituent units, not suffix
    # groups. Allow their sulfur atoms through ordinary-group recognition
    # only when the complete neutral bridge recognizer owns them.
    if all_sulfur:
        from .peroxide_components import peroxide_bridge
        for idx in all_sulfur:
            atom = mol.GetAtomWithIdx(idx)
            for neighbor in atom.GetNeighbors():
                if neighbor.GetAtomicNum() != 6:
                    continue
                bridge = peroxide_bridge(mol, idx, neighbor.GetIdx())
                if bridge is None or bridge[0] != 16:
                    continue
                eligible_sulfur.add(idx)
                eligible_sulfur.update(n.GetIdx() for n in atom.GetNeighbors()
                                       if n.GetAtomicNum() == 16)
    # P-63.4.2 sulfur atoms enter the ordinary group inventory only when
    # every sulfur is owned by a recognized neutral chalcogen-peroxol motif.
    # Other S/P chemistry keeps its existing extended-inventory gate.
    allow_peroxol_sulfur = bool(all_sulfur) and all_sulfur <= eligible_sulfur
    # Selenium and tellurium enter the ordinary inventory only as atoms of a
    # complete, recognized neutral P-63.4.2 C–X–Y–H unit. This does not admit
    # unrelated selenium/tellurium chemistry globally.
    allow_peroxol_se_te = bool(all_se_te) and all_se_te <= peroxol_se_te
    allowed = ({1, 6, 7, 8, 9, 17, 35, 53}
               | ({15, 16} if extended else set())
               | ({16} if allow_peroxol_sulfur else set())
               | ({34, 52} if allow_peroxol_se_te else set()))
    silicon = (atoms_by_number.get(14, ()) if atom_numbers is not None else
               tuple(a for a in atoms if a.GetAtomicNum() == 14))
    if extended and silicon and all(map(_spectator_silicon, silicon)):
        allowed.add(14)
    if any(
        a.GetAtomicNum() not in allowed
        or a.GetNumRadicalElectrons()
        or (not extended and a.GetFormalCharge() and a.GetIdx() not in charge_exceptions)
        or (not extended and a.GetIsotope())
        or a.GetAtomicNum() == 7
        and int(a.GetChiralTag())
        for a in atoms
    ):
        return FunctionalInventory((), False)
    groups, used_nitrogens = [], set()
    subordinate_heterones = set()
    for kind, n0, n1, oxygens in oxoamide_sites:
        groups.append(CharacteristicGroup(
            kind, n1, tuple((n0, n1, *oxygens)), n0, n0,
        ))
        used_nitrogens.add(n1)
    if extended:
        from .extended_inventory import additional_sites

        extra, covered, additional_used_nitrogens, supported = additional_sites(
            mol, charge_role_proofs=charge_role_proofs, subordinate_thioxo=subordinate_thioxo,
            subordinate_heterones=subordinate_heterones,
        )
        if not supported:
            return FunctionalInventory((), False)
        groups.extend(extra)
        used_nitrogens.update(additional_used_nitrogens)
    groups.extend(thiol_groups)
    if not any(a.GetAtomicNum() in {7, 8} for a in atoms) and not peroxol_groups:
        return FunctionalInventory(tuple(groups), True, charge_role_proofs)
    carbonyl_centers = set()
    deferred_cn_units = []
    carbon_atoms = atoms
    if atom_numbers is not None:
        # Only bonds incident to N/O can supply the existing C=O/C=N/C#N
        # admission signal. Inspect their native bonds once, then classify
        # carbon centers in the original atom order.
        carbon_group_centers = set()
        for number in (7, 8):
            for heteroatom in atoms_by_number.get(number, ()):
                hetero_idx = heteroatom.GetIdx()
                for bond in heteroatom.GetBonds():
                    if bond.GetBondTypeAsDouble() not in {2, 3}:
                        continue
                    other = bond.GetOtherAtomIdx(hetero_idx)
                    if atom_numbers[other] == 6:
                        carbon_group_centers.add(other)
        carbon_atoms = (atoms[idx] for idx in sorted(carbon_group_centers))
    for carbon in carbon_atoms:
        if carbon.GetAtomicNum() != 6:
            continue
        idx = carbon.GetIdx()
        if carbon.GetFormalCharge() and idx in charge_exceptions:
            # The terminal carbon in a proved isocyano unit is a prefix
            # atom, not the center of a carbon-bound nitrile.
            continue
        # This loop emits carbon groups only for C=O, C=N or C#N bonds.
        # Ordinary skeletal carbons cannot reach any of those constructions;
        # avoid rebuilding their complete neighbor/order inventories.
        if atom_numbers is None and not any(
                bond.GetBondTypeAsDouble() in {2, 3}
                and bond.GetOtherAtom(carbon).GetAtomicNum() in {7, 8}
                for bond in carbon.GetBonds()):
            continue
        neighbors = tuple(carbon.GetNeighbors())
        order = {
            n.GetIdx(): mol.GetBondBetweenAtoms(idx, n.GetIdx()).GetBondTypeAsDouble()
            for n in neighbors
        }
        oxygens = [
            n for n in neighbors if n.GetAtomicNum() == 8 and order[n.GetIdx()] == 2
        ]
        carbons = [
            n for n in neighbors if n.GetAtomicNum() == 6 and order[n.GetIdx()] == 1
        ]
        owner = carbons[0].GetIdx() if len(carbons) == 1 else -1
        triple_n = [
            n for n in neighbors if n.GetAtomicNum() == 7 and order[n.GetIdx()] == 3
        ]
        if triple_n:
            if len(triple_n) != 1 or triple_n[0].GetDegree() != 1:
                return FunctionalInventory(tuple(groups), False)
            terminal = triple_n[0].GetIdx()
            if owner >= 0:
                groups.append(CharacteristicGroup('nitrile', idx, (idx, terminal), owner))
            elif extended and carbon.GetDegree() == 2:
                host = next(n for n in neighbors if n.GetIdx() != terminal)
                if (host.GetAtomicNum() != 7 or host.GetFormalCharge()
                        or host.GetTotalValence() != 3 or order[host.GetIdx()] != 1):
                    return FunctionalInventory(tuple(groups), False)
                # An N-cyano imino substituent stays inside the C=N unit.
                # An annular N bearing cyano can carry a carbonitrile suffix.
                if all(b.GetBondTypeAsDouble() == 1 for b in host.GetBonds()):
                    host_idx = host.GetIdx()
                    if host.IsInRing():
                        groups.append(CharacteristicGroup('nitrile', idx, (idx, terminal), host_idx))
                    else:
                        distal = tuple(n.GetIdx() for n in host.GetNeighbors()
                                       if n.GetAtomicNum() == 7 and not n.GetIsAromatic()
                                       and not n.GetFormalCharge() and n.GetTotalValence() == 3)
                        if len(distal) > 1:
                            return FunctionalInventory(tuple(groups), False)
                        kind = 'cyanohydrazide' if distal else 'cyanamide'
                        groups.append(CharacteristicGroup(kind, idx,
                            (idx, terminal, host_idx, *distal), host_idx, host_idx))
                        used_nitrogens.update(distal)
                    used_nitrogens.add(host_idx)
                elif (host.IsInRing() or not any(b.GetBondTypeAsDouble() == 2
                          and b.GetOtherAtom(host).GetAtomicNum() == 6 for b in host.GetBonds())):
                    return FunctionalInventory(tuple(groups), False)
            else:
                return FunctionalInventory(tuple(groups), False)
            used_nitrogens.add(terminal)
            continue
        double_n = [
            n for n in neighbors if n.GetAtomicNum() == 7 and order[n.GetIdx()] == 2
        ]
        if double_n:
            # Ordinary P-62 imines are neutral carbon=N sites. N–O (oxime),
            # N–N (hydrazone), cumulene, charged and overvalent forms need
            # separate rules and must not leak into the ordinary imine class.
            if oxygens or len(double_n) != 1:
                return FunctionalInventory(tuple(groups), False)
            nitrogen = double_n[0]
            if _ordinary_annular_cn_bond(mol, carbon, nitrogen, allow_polycycle=extended):
                # An endocyclic skeleton operation is named by its original
                # H-W/hydro state. It is not an extra exocyclic =NH suffix.
                used_nitrogens.add(nitrogen.GetIdx())
                continue
            if extended and nitrogen.GetIdx() in additional_used_nitrogens:
                sulfur = next((a for a in nitrogen.GetNeighbors() if a.GetAtomicNum() == 16), None)
                if sulfur is not None:
                    from .sulfonylguanidine import sulfonylguanidine

                    site = sulfonylguanidine(mol, sulfur.GetIdx())
                    if site is not None and site.carbon == idx and site.imino == nitrogen.GetIdx():
                        continue
            from .oxime_components import oxime_site
            oxime = oxime_site(mol, idx, allow_carboxymethyl=extended, allow_cyclic_ether=extended)
            if oxime is not None:
                amino = next((n for n in neighbors if n.GetAtomicNum() == 7
                              and order[n.GetIdx()] == 1 and not n.IsInRing()
                              and not n.GetFormalCharge() and not n.GetIsAromatic()
                              and n.GetTotalValence() == 3 and n.GetTotalNumHs() in {1, 2}
                              and all(ligand.GetIdx() == idx or ligand.GetAtomicNum() == 6
                                      and mol.GetBondBetweenAtoms(n.GetIdx(), ligand.GetIdx()).GetBondTypeAsDouble() == 1
                                      for ligand in n.GetNeighbors())), None)
                if amino is not None and len(carbons) == 1 and not carbon.IsInRing():
                    # P-66.4.4 treats this as an N'-substituted amidine.
                    # Its amino nitrogen belongs to the same characteristic group.
                    groups.append(CharacteristicGroup('amidoxime', idx,
                        (idx, oxime.nitrogen, oxime.oxygen, amino.GetIdx()), owner, oxime.nitrogen))
                    used_nitrogens.update((oxime.nitrogen, amino.GetIdx()))
                    continue
                groups.append(CharacteristicGroup(
                    oxime.kind, idx, (idx, oxime.nitrogen, oxime.oxygen),
                    owner, oxime.nitrogen,
                ))
                used_nitrogens.add(oxime.nitrogen)
                continue
            amino = tuple(n for n in neighbors if n.GetAtomicNum() == 7
                          and order[n.GetIdx()] == 1)
            if (amino and not carbon.IsInRing() and not nitrogen.IsInRing()
                    and len(amino) in {1, 2} and len(neighbors) == 1 + len(amino) + len(carbons)
                    and all(not n.GetFormalCharge() and n.GetTotalValence() == 3
                            and not n.GetIsAromatic()
                            and all(child.GetAtomicNum() not in {7, 8}
                                    for child in n.GetNeighbors())
                            for n in (*amino, nitrogen))):
                if not extended:
                    return FunctionalInventory(tuple(groups), False)
                kind = 'guanidine' if len(amino) == 2 else 'amidine'
                nitrogens = (nitrogen.GetIdx(), *(n.GetIdx() for n in amino))
                groups.append(CharacteristicGroup(kind, idx, (idx, *nitrogens), owner,
                                                   amino[0].GetIdx()))
                used_nitrogens.update(nitrogens)
                continue
            n_single_heavy = [
                n for n in nitrogen.GetNeighbors()
                if n.GetIdx() != idx and n.GetAtomicNum() > 1
            ]
            if (not nitrogen.GetFormalCharge() and not nitrogen.GetNumRadicalElectrons()
                    and not nitrogen.IsInRing() and nitrogen.GetTotalValence() == 3
                    and len(n_single_heavy) == 1
                    and mol.GetBondBetweenAtoms(nitrogen.GetIdx(), n_single_heavy[0].GetIdx()).GetBondTypeAsDouble() == 1):
                attached = n_single_heavy[0]
                hydrazone = (attached.GetAtomicNum() == 7 and not attached.GetFormalCharge()
                             and not attached.GetNumRadicalElectrons())
                oxime_acyl = (attached.GetAtomicNum() == 8 and not attached.GetFormalCharge()
                    and any(n.GetAtomicNum() == 6 and n.GetIdx() != idx and any(
                        o.GetAtomicNum() == 8 and
                        mol.GetBondBetweenAtoms(n.GetIdx(), o.GetIdx()).GetBondTypeAsDouble() == 2
                        for o in n.GetNeighbors()) for n in attached.GetNeighbors()))
                if hydrazone or oxime_acyl:
                    deferred_cn_units.append((idx, nitrogen.GetIdx(), attached.GetIdx()))
                    used_nitrogens.add(nitrogen.GetIdx())
                    continue
            if (
                nitrogen.GetFormalCharge()
                or nitrogen.GetNumRadicalElectrons()
                or nitrogen.IsInRing()
                or any(n.GetAtomicNum() != 6 for n in n_single_heavy)
                or len(n_single_heavy) > 1
                or any(mol.GetBondBetweenAtoms(nitrogen.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() != 1
                       for n in n_single_heavy)
                or nitrogen.GetTotalValence() != 3
            ):
                return FunctionalInventory(tuple(groups), False)
            groups.append(CharacteristicGroup(
                "imine", idx, (idx, nitrogen.GetIdx()), owner, nitrogen.GetIdx()
            ))
            used_nitrogens.add(nitrogen.GetIdx())
            if not oxygens:
                continue
        if not oxygens:
            continue
        if len(oxygens) != 1 or oxygens[0].GetDegree() != 1:
            return FunctionalInventory(tuple(groups), False)
        carbonyl_centers.add(idx)
        single_o = [
            n
            for n in neighbors
            if n.GetAtomicNum() == 8 and order[n.GetIdx()] in {1, 1.5}
        ]
        single_n = [
            n
            for n in neighbors
            if n.GetAtomicNum() == 7 and order[n.GetIdx()] in {1, 1.5}
        ]
        annular_s = [n for n in neighbors if n.GetAtomicNum() == 16
            and order[n.GetIdx()] == 1 and n.GetTotalValence() == 2
            and not n.GetFormalCharge() and not n.GetNumRadicalElectrons()
            and mol.GetBondBetweenAtoms(idx, n.GetIdx()).IsInRing()]
        used_nitrogens.update(n.GetIdx() for n in single_n)
        kind, ligand = "", -1
        if annular_s:
            # P-41 class 16 / P-64.1.2: a carbonyl whose divalent sulfur
            # neighbour is part of the same ring is a pseudoketone. This
            # recognizes the original graph; no O-for-S copy fixes its class.
            kind, ligand = 'pseudoketone', annular_s[0].GetIdx()
        elif single_o:
            oxygen = single_o[0]
            ligand = oxygen.GetIdx()
            if (
                len(single_o) == 2
                and not single_n
                and all(
                    mol.GetBondBetweenAtoms(idx, o.GetIdx()).IsInRing()
                    for o in single_o
                )
            ):
                kind = "lactone"  # Cyclic carbonate is a class-16 pseudoketone.
            elif len(single_o) == 2 and not single_n:
                kind = (
                    "carbonic-acid"
                    if any(o.GetTotalNumHs() for o in single_o)
                    else "carbonate"
                )
            elif len(single_o) != 1:
                return FunctionalInventory(tuple(groups), False)
            elif (
                extended
                and oxygen.GetDegree() == 1
                and oxygen.GetFormalCharge() == -1
            ):
                if not single_n and (owner >= 0 or carbon.GetTotalNumHs() == 1):
                    kind = "carboxylate"
                elif (owner < 0 and len(single_n) == 1 and single_n[0].IsInRing()
                      and not single_n[0].GetFormalCharge() and single_n[0].GetTotalValence() == 3
                      and order[single_n[0].GetIdx()] == 1 and carbon.GetDegree() == 3):
                    kind, owner = "carbamate-anion", single_n[0].GetIdx()
            elif (
                sum(n.GetAtomicNum() > 1 for n in oxygen.GetNeighbors()) == 1
                and oxygen.GetTotalNumHs(includeNeighbors=True) == 1
            ):
                kind = (
                    "acid"
                    if not single_n and (owner >= 0 or carbon.GetTotalNumHs())
                    else "carbonic-acid"
                )
            elif oxygen.GetDegree() == 2:
                if (any(n.GetAtomicNum() not in {6, 7} for n in oxygen.GetNeighbors())
                        and not _simple_annular_peroxide_carbonyl(mol, carbon, oxygen)):
                    return FunctionalInventory(tuple(groups), False)
                if mol.GetBondBetweenAtoms(idx, ligand).IsInRing():
                    kind = "lactone"
                elif any(
                    n.GetIdx() != idx
                    and any(
                        x.GetAtomicNum() == 8
                        and mol.GetBondBetweenAtoms(
                            n.GetIdx(), x.GetIdx()
                        ).GetBondTypeAsDouble()
                        == 2
                        for x in n.GetNeighbors()
                    )
                    for n in oxygen.GetNeighbors()
                ):
                    kind = "anhydride"
                elif single_n:
                    kind = "carbamate" if len(single_n) == 1 else ""
                elif owner >= 0 or carbon.GetTotalNumHs():
                    kind = "ester"
        elif single_n:
            nitrogen = single_n[0]
            ligand = nitrogen.GetIdx()
            if carbon.IsInRing():
                kind = "lactam"
            elif len(single_n) == 2:
                if (any(n.IsInRing() for n in single_n)
                        and (not extended or carbon.GetDegree() != 3
                             or any(n.GetTotalValence() != 3 or order[n.GetIdx()] != 1
                                    for n in single_n))):
                    return FunctionalInventory(tuple(groups), False)
                kind = "urea"
            elif len(single_n) == 1:
                if nitrogen.IsInRing():
                    kind = "pseudoketone"
                elif (any(n.GetAtomicNum() == 7 and not n.GetIsAromatic()
                          and order[ligand] == 1 and mol.GetBondBetweenAtoms(
                              ligand, n.GetIdx()).GetBondTypeAsDouble() == 1
                          for n in nitrogen.GetNeighbors())
                      and nitrogen.GetIdx() not in {site[2] for site in oxoamide_sites}):
                    kind = "hydrazide"
                elif owner >= 0 or carbon.GetTotalNumHs():
                    kind = "amide"
        elif any(n.GetAtomicNum() in {9, 17, 35, 53} for n in neighbors):
            kind = "acid-halide"
        elif carbon.GetTotalNumHs(includeNeighbors=True):
            kind = "aldehyde"
        elif sum(n.GetAtomicNum() == 6 for n in neighbors) == 2:
            kind = "ketone"
        elif extended and owner >= 0 and carbon.GetDegree() == 3:
            sulfur = next((n for n in neighbors if n.GetAtomicNum() == 16
                           and order[n.GetIdx()] == 1 and n.GetDegree() == 4
                           and not n.GetFormalCharge() and not n.IsInRing()), None)
            if sulfur is not None:
                ligands = sorted((b.GetBondTypeAsDouble(), b.GetOtherAtom(sulfur).GetAtomicNum(),
                                  b.GetOtherAtom(sulfur).GetDegree() if b.GetBondTypeAsDouble() == 2 else 0)
                                 for b in sulfur.GetBonds() if b.GetOtherAtomIdx(sulfur.GetIdx()) != idx)
                if ligands == [(1.0, 6, 0), (2.0, 8, 1), (2.0, 8, 1)]:
                    from rdkit import Chem

                    # The complete sulfone is an acylsulfonyl branch under a
                    # separately recognized neutral carboxylic acid.
                    if mol.HasSubstructMatch(_neutral_carboxyl_query(Chem)):
                        continue
        if not kind:
            return FunctionalInventory(tuple(groups), False)
        hydrazide_n = ()
        if kind == 'hydrazide':
            hydrazide_n = tuple(n.GetIdx() for n in nitrogen.GetNeighbors()
                                if n.GetAtomicNum() == 7 and not n.GetFormalCharge()
                                and n.GetTotalValence() == 3 and not n.GetIsAromatic()
                                and mol.GetBondBetweenAtoms(ligand, n.GetIdx()).GetBondTypeAsDouble() == 1)
            if len(hydrazide_n) != 1:
                return FunctionalInventory(tuple(groups), False)
            used_nitrogens.update(hydrazide_n)
        groups.append(
            CharacteristicGroup(
                kind,
                idx,
                tuple(
                    [idx, oxygens[0].GetIdx()]
                    + [n.GetIdx() for n in single_o + single_n + annular_s] + list(hydrazide_n)
                ),
                owner,
                ligand,
            )
        )
    for oxygen in (atoms_by_number.get(8, ()) if atom_numbers is not None else atoms):
        if (
            oxygen.GetAtomicNum() != 8
            or sum(n.GetAtomicNum() > 1 for n in oxygen.GetNeighbors()) != 1
            or oxygen.GetTotalNumHs(includeNeighbors=True) != 1
        ):
            continue
        neighbor = next((n for n in oxygen.GetNeighbors() if n.GetAtomicNum() == 6), None)
        if neighbor is not None and neighbor.GetIdx() not in carbonyl_centers:
            groups.append(
                CharacteristicGroup(
                    "alcohol", neighbor.GetIdx(), (neighbor.GetIdx(), oxygen.GetIdx())
                )
            )
    for nitrogen in (atoms_by_number.get(7, ()) if atom_numbers is not None else atoms):
        if (
            nitrogen.GetAtomicNum() == 7
            and nitrogen.GetIdx() not in used_nitrogens
            and not nitrogen.IsInRing()
            and all(b.GetBondTypeAsDouble() == 1 for b in nitrogen.GetBonds())
        ):
            groups.append(
                CharacteristicGroup("amine", nitrogen.GetIdx(), (nitrogen.GetIdx(),))
            )
    # P-63.4.1's neutral carbon-bound C–O–O–H operation is separate from
    # carbonyl/acid oxygen recognition. The recognizer itself excludes C(=O)OOH.
    groups.extend(peroxol_groups)
    if subordinate_heterones and not any(group.seniority[0] < 16 for group in groups):
        # P-41: a heterone outranks alcohols and amines. Do not hide it as
        # a prefix merely to admit a lower-class ordinary parent writer.
        return FunctionalInventory(tuple(groups), False)
    if deferred_cn_units and not any(group.seniority < (19, 0) for group in groups):
        return FunctionalInventory(tuple(groups), False)
    return FunctionalInventory(tuple(groups), True, charge_role_proofs)
