"""Evidence-based parent/suffix competition across compatible mapped routes.

P-51 method decisions remain authoritative before this pool. It admits supported
neutral chemistry and explicitly mapped chain/monocyclic/polycyclic parents. A
candidate is chosen only when no applicable comparison with a different name
is unknown. Structural parsing is never used to rank preferred names.
"""
from dataclasses import dataclass, field
from itertools import chain

from .molecule_iteration import iter_atoms, iter_bonds
from .molecule_cache import mol_cache_dict, mol_cache_get
from .graph_model import mapped_graph_components
from .component_mapping import (MappedComponent, decorate_component, fully_covered,
                                hydrogen_bindings, mapped_branch)
from .extended_inventory import selection_inventory
from .parent_selection import compare_parents, graph_parent_facts, parent_frontier, replace_parent_facts
from .selection_evidence import (CandidateCompleteness, parent_justification,
                                 carbon_tree_method_proved, proposal_identity)


def _site_key(site):
    return (site.kind, site.principal_center, site.host_atom, tuple(site.group_atoms))


def _proposal_site_key(proposal):
    if proposal.selected_site_evidence:
        return tuple(_site_key(site) for site in proposal.selected_site_evidence)
    return tuple(
        (proposal.principal_kind, center, center, (center,))
        for center in sorted(proposal.principal_centers)
    )


@dataclass(frozen=True, slots=True)
class ParentProposal:
    name: str
    route: str
    principal_kind: str
    principal_centers: frozenset[int]
    parent_atoms: tuple[int, ...]
    atom_locants: tuple
    parent_selection: object
    annotation_coverage: bool
    prepared: object = None
    ring_state: object = None
    construction: MappedComponent | None = None
    # Additive P-44 site evidence. ``principal_centers`` remains the legacy
    # compatibility projection; these keys distinguish multiple groups on
    # one host and preserve selected/omitted roles by original atom identity.
    principal_site_keys: tuple = ()
    selected_site_evidence: tuple = ()
    omitted_site_evidence: tuple = ()
    # Preserve compatibility enumeration after deferred legacy construction.
    # This key is never supplied to the chemical preference comparator.
    enumeration_key: tuple | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class CrossRouteSelection:
    selected: ParentProposal
    candidates: tuple[ParentProposal, ...]
    frontier: tuple[ParentProposal, ...]
    comparisons: tuple
    admission_comparisons: tuple = ()
    functional_parent_frontier: object = None
    principal_site_frontier: object = None
    deferred_proposals: tuple = ()
    candidate_completeness: CandidateCompleteness | None = None
    justification: object = None

    @property
    def functional_parent_admission(self):
        """Typed original-graph admission evidence from the shared frontier."""
        return getattr(self.functional_parent_frontier, "admission_evidence", None)


@dataclass(frozen=True, slots=True)
class CrossRouteOutcome:
    """Keep a supplied but unresolved parent pool distinct from no pool."""

    status: str
    selection: CrossRouteSelection | None = None
    candidates: tuple[ParentProposal, ...] = ()
    functional_parent_frontier: object = None
    reason: str = ""
    principal_site_frontier: object = None
    deferred_proposals: tuple = ()
    justification: object = None
    general_proposal: ParentProposal | None = None

    @property
    def selected(self):
        """The delivered mapped parent, including a pending general choice.

        Local group/ownership consumers can inspect the actual construction.
        Only ``status == 'complete'`` establishes a completed shared selection.
        """
        return self.selection.selected if self.selection is not None else self.general_proposal


def cross_route_scope(mol):
    from .molecule_cache import mol_cache_dict, naming_cache_state

    cache = mol_cache_dict(mol, "cross_route_scope")
    key = naming_cache_state(mol)
    if key not in cache:
        cache[key] = _cross_route_scope(mol)
    return cache[key]



def _cross_route_scope(mol):
    # P-63.4 admits Se/Te only inside a complete, neutral, typed peroxol
    # operation. Other selenium/tellurium chemistry needs its own proof.
    from .peroxol_components import recognize_peroxols
    peroxol_se_te = {i for group in recognize_peroxols(mol) for i in group.atoms
        if mol.GetAtomWithIdx(i).GetAtomicNum() in {34, 52}}
    from .neutral_charge_components import (
        validate_neutral_charge_role_proofs,
        neutral_charge_prefix_facts,
    )
    charge_roles = selection_inventory(mol).charge_role_proofs
    if not validate_neutral_charge_role_proofs(mol, charge_roles):
        return None
    if mol.GetNumHeavyAtoms() > 64:
        from rdkit import Chem
        from .tree_ring_assemblies import tree_assembly_consumer_scope
        if not tree_assembly_consumer_scope(mol,Chem):return None
    if any(
        (a.GetAtomicNum() not in {1, 6, 7, 8, 9, 16, 17, 35, 53}
         and a.GetIdx() not in peroxol_se_te)
        or a.GetNumRadicalElectrons()
        for a in iter_atoms(mol)
    ):
        return None
    # Relative/enhanced stereo and unsupported CIP descriptor types require
    # their own writing rules. Never silently flatten them to ordinary R/S.
    from rdkit import Chem
    from .stereochemistry import assigned_atom_stereochemistry
    if mol.GetStereoGroups():
        return None
    assigned = dict(assigned_atom_stereochemistry(mol, Chem, atom_order='idx', include_pseudo=True))
    if any(int(a.GetChiralTag()) and a.GetIdx() not in assigned for a in iter_atoms(mol)):
        return None
    # A chain parent does not excuse unproved construction of an attached ring
    # prefix. Admit the whole molecule only within the mapped vocabulary,
    # including every independent ring system (P-52/P-29).
    from .mapped_ring_parents import ring_systems,skeletons
    if any(not next(skeletons(mol,Chem,core),None) for core in ring_systems(mol)):
        return None
    from .ring_state_parent import _inventory
    inventory = _inventory(mol,Chem)
    if not inventory.supported or not inventory.principal:
        return None
    if any(g.kind == 'sulfinamide' for g in inventory.groups):
        # P-41/P-66.1.1.4.2 admits this subordinate prefix only with exact
        # original-site roles and a proved senior principal-group frontier.
        from .functional_parent_admission import parent_admission_evidence
        admission = parent_admission_evidence(
            mol, Chem, inventory, inventory.principal)
        if admission.status != 'admitted':
            return None
    if any(g.kind not in {'acid', 'amide', 'nitrile', 'ester', 'aldehyde', 'ketone',
                          'alcohol', 'amine', 'peroxol', 'imine','oxime','oxime-ether','hydrazide','lactam','lactone','pseudoketone',
                          'sulfonic-acid','sulfinic-acid','sulfonamide','thiol',
                          'nitramide','nitrous-amide','sulfinamide'} for g in inventory.groups):
        return None
    if inventory.principal[0].kind not in {'acid', 'amide', 'nitrile', 'ester',
                                          'aldehyde', 'ketone', 'alcohol','amine','peroxol','imine','oxime','oxime-ether','hydrazide','lactam','lactone','pseudoketone',
                                          'sulfonic-acid','sulfinic-acid','sulfonamide','thiol'}:
        return None
    if inventory.principal[0].kind=='amine':
        from .amine_components import principal_scope
        from .hydroxylamine_methods import principal_scope as hydroxylamine_scope
        if not (principal_scope(mol,inventory.principal)
                or hydroxylamine_scope(mol,inventory.principal)):
            return None
    if inventory.principal[0].kind in {'imine', 'oxime', 'oxime-ether'}:
        # P-68.3.1.1.2 treats oximes and their ethers as N-substituted
        # imines. They share the class-20 suffix rather than competing as
        # unrelated characteristic groups. Ordinary ring imines retain the
        # separate, proved ring-state construction.
        if all(g.kind == 'imine' for g in inventory.principal):
            from .imine_components import principal_scope
        else:
            from .imine_family_components import principal_scope
        if not principal_scope(mol, inventory.principal):
            return None
    if inventory.principal[0].kind=='peroxol':
        from .peroxol_components import principal_scope
        if not principal_scope(mol,inventory.principal):return None
    if inventory.principal[0].kind=='hydrazide':
        from .hydrazide_components import recognize_hydrazides
        recognized={site.center for site in recognize_hydrazides(mol, annular_terminal=True)}
        if any(group.center not in recognized for group in inventory.principal):return None
    # The bounded inventory does not recognize every exocyclic heteroatom
    # multiple bond. Unknown characteristic groups need their own suffix or
    # prefix rules before entering this complete construction proof (P-41).
    known=[set(g.atoms) for g in inventory.groups]
    # P-61 nitro/azido units are detachable prefixes, not suffix groups.
    # Their internal multiple bonds remain original-graph typed evidence;
    # they must not make a complete neutral parent look unrecognized.
    known.extend(set(atoms) for _kind, _host, atoms
                 in neutral_charge_prefix_facts(mol, charge_roles))
    if any(b.GetBondTypeAsDouble()>1 and not b.IsInRing()
        and any(mol.GetAtomWithIdx(i).GetAtomicNum() not in {1,6} for i in (b.GetBeginAtomIdx(),b.GetEndAtomIdx()))
        and not any({b.GetBeginAtomIdx(),b.GetEndAtomIdx()}<=atoms for atoms in known)
        for b in iter_bonds(mol)):
        return None
    # P-82.2.2.2 is a suffix-selection operation after class recognition.
    # Unlike labelled hydroxy units can be partitioned without changing the
    # inventory: the selected unit supplies the suffix, the others prefixes.
    groups = inventory.principal
    if groups and all(group.kind == 'amide' for group in groups):
        from .amide_isotopes import amide_isotope_partition
        partition = amide_isotope_partition(mol, groups)
        if partition is not None and partition.active:
            if partition.selection.barrier or not partition.selected_groups:
                return None
            from types import SimpleNamespace
            return SimpleNamespace(groups=inventory.groups, principal=partition.selected_groups,
                supported=True, isotope_partition_rule='P-82.2.2.2', amide_isotope_partition=partition)
    signatures = set()
    for group in groups:
        characteristic = set(group.atoms) - ({group.center} if group.kind == 'alcohol' else set())
        if group.kind == 'peroxol':
            characteristic = set(group.atoms)
        if group.kind in {'ketone','lactam','lactone','pseudoketone'}:
            # These cyclic amides/esters are class-16 heterocyclic -one
            # constructions (P-41), not amide/ester suffixes. The annular C/N/O
            # atoms are skeletal parent atoms; a 13C/15N label there does not
            # modify the suffix oxygen or invoke P-82.2.2.2 partitioning.
            characteristic={i for i in group.atoms if mol.GetAtomWithIdx(i).GetAtomicNum()==8
                and mol.GetBondBetweenAtoms(group.center,i) is not None
                and mol.GetBondBetweenAtoms(group.center,i).GetBondTypeAsDouble()==2}
        labels = []
        for idx in characteristic:
            atom = mol.GetAtomWithIdx(idx)
            bond = mol.GetBondBetweenAtoms(group.center, idx) if idx != group.center else None
            role = 0 if idx == group.center else bond.GetBondTypeAsDouble() if bond is not None else 3
            if atom.GetIsotope():
                labels.append((role, atom.GetAtomicNum(), atom.GetIsotope()))
            labels.extend((role, 1, h.GetIsotope()) for h in atom.GetNeighbors()
                          if h.GetAtomicNum() == 1 and h.GetIsotope())
        signatures.add(tuple(sorted(labels)))
    if len(signatures) > 1:
        if all(group.kind == 'alcohol' for group in groups):
            from .oxygen_characteristic_suffixes import prioritize_oxygen_groups
            from types import SimpleNamespace
            priority = prioritize_oxygen_groups(mol, groups)
            centers = {unit.center for unit in priority.suffix_groups}
            selected = tuple(group for group in groups if group.center in centers)
            if not selected:
                return None
            return SimpleNamespace(groups=inventory.groups, principal=selected,
                                   supported=True, isotope_partition_rule='P-82.2.2.2')
        return None  # Other classes require their own complete prefix proof.
    return inventory


def _facts(mol, groups, path, mapping, prefixes, name, *, cyclic, construction=None):
    charge_roles = selection_inventory(mol).charge_role_proofs
    from .nonstandard_parent_evidence import build_substituent_nonstandard_proof
    nonstandard_proof = build_substituent_nonstandard_proof(
        mol, path, construction, prefixes, mapping, groups,
    )
    return graph_parent_facts(mol, path, mapping, principal_class=groups[0].seniority,
                             suffix_centers=[g.center for g in groups],
                             suffix_locants=[mapping[g.owner if g.center not in path else g.center]
                                             for g in groups],
                             prefixes=prefixes, kind='ring' if cyclic else 'chain', name=name,
                             allow_annotations=True, charge_role_proofs=charge_roles,
                             prefix_branch_ownership=getattr(construction,'prefix_branch_ownership',()),
                             substituent_nonstandard_proof=nonstandard_proof)


def _annotation_ownership(mol, mapping):
    # Numbering changes the locant values, but ownership uses only the set of
    # mapped original atoms on this API-owned molecule.
    cache = mol_cache_dict(mol, "annotation_ownership")
    key = frozenset(mapping)
    if key not in cache:
        cache[key] = _uncached_annotation_ownership(mol, mapping)
    return cache[key]


def _uncached_annotation_ownership(mol, mapping):
    """Every annotation belongs to one parent or singly attached branch scope.

    MappedComponent/mapped_branch validate the actual rendered local map. This
    independent partition also rejects stereo bonds crossing those scopes.
    """
    from .stereochemistry import _indexed_bond_stereochemistry
    from rdkit import Chem

    owners = {i: 0 for i in mapping}
    if mol_cache_get(mol, "graph_view") is not None:
        for count, (component, attached) in enumerate(mapped_graph_components(
                mol, mapping, include_isotopic_hydrogens=True), 1):
            if len(frozenset(attached)) != 1:
                return False
            owners.update(dict.fromkeys(component, count))
    else:
        # Preserve native initialization order for a direct call on a graph
        # whose primitive view has not yet been constructed.
        remaining = {a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1 or a.GetIsotope()} - mapping.keys()
        count = 0
        while remaining:
            count += 1
            attached, stack = set(), [next(iter(remaining))]
            while stack:
                i = stack.pop()
                if i not in remaining:
                    continue
                remaining.remove(i)
                owners[i] = count
                for atom in mol.GetAtomWithIdx(i).GetNeighbors():
                    n = atom.GetIdx()
                    if n in mapping:
                        attached.add(n)
                    elif n in remaining:
                        stack.append(n)
            if len(attached) != 1:
                return False
    if any((a.GetIsotope() or int(a.GetChiralTag())) and a.GetIdx() not in owners for a in iter_atoms(mol)):
        return False
    return all(a in owners and b in owners and owners[a] == owners[b]
               for a, b, _ in _indexed_bond_stereochemistry(mol, Chem))


def functional_proposals(mol, Chem, inventory, *, structural_frontier=None,
                         nitrogen_oxo_higher_class=False, defer=False):
    from .complete_functional_parent import functional_parent_candidates, FunctionalParentPlan

    groups = inventory.principal
    # A subordinate nitrile is one cyano prefix, including its carbon. It may
    # not be absorbed into an alkane parent and leave a triple-bonded N to be
    # interpreted as an amino prefix.
    prefix_atoms = {g.center for g in inventory.groups if g.kind == 'nitrile' and g not in groups}
    if structural_frontier is not None and structural_frontier.status in {"blocked", "conflict"}:
        return
    bounded = structural_frontier is not None and structural_frontier.status in {"complete", "tie"}
    candidates = (structural_frontier.candidates
                  if bounded
                  else functional_parent_candidates(mol, Chem, include_single=True,
                       excluded_skeletal_atoms=prefix_atoms, mapped_ring_prefixes=True,
                       principal=groups, nitrogen_oxo_higher_class=nitrogen_oxo_higher_class,
                       defer_render=defer))
    for record in candidates:
        if isinstance(record, FunctionalParentPlan):
            from .proposal_plans import ParentProposalPlan
            def construct(record=record):
                rendered = record.construct()
                if rendered is None:
                    return ()
                proposal = _functional_proposal(mol, groups, rendered, legacy=True)
                return () if proposal is None else (proposal,)
            yield ParentProposalPlan(
                'complete-functional', record.parent_atoms, record.principal_centers,
                record.bound, construct,
            )
            continue
        proposal = _functional_proposal(mol, groups, record, legacy=not bounded)
        if proposal is not None:
            yield proposal


def _functional_proposal(mol, groups, record, *, legacy):
    if not record.name:
        return None
    path, mapping = record.parent_atoms, dict(record.atom_locants)
    cyclic = all(mol.GetAtomWithIdx(i).IsInRing() for i in path)
    selected = tuple(group for group in groups if group.center in record.principal_centers)
    if not selected or len(selected) != len(record.principal_centers):
        return None
    facts = _facts(mol, selected, path, mapping, record.prefixes, record.name,
                   cyclic=cyclic, construction=record.construction)
    if facts is not None and _annotation_ownership(mol, mapping):
        return ParentProposal(record.name, 'complete-functional', groups[0].kind,
                              record.principal_centers, path, record.atom_locants, facts, True,
                              construction=record.construction,
                              enumeration_key=record.numbering_key if legacy else None)
    return None


def partial_functional_proposals(mol, Chem, inventory, *, defer=False):
    """P-44.1.1 chain suffixes when other principal sites are elsewhere.

    A complete-group producer cannot propose these parents: not every source
    site belongs to the chain. Enumerate the ordinary carbon-tree paths and
    retain the exact selected subset. Every omitted site must be owned by a
    fully mapped prefix. The common comparator and unrendered witnesses still
    decide coverage/parent seniority; renderer availability supplies no rank.
    """
    groups = tuple(inventory.principal)
    if (len(groups) < 2 or groups[0].kind not in {'acid', 'ester', 'amide', 'nitrile'}
            or any(group.kind != groups[0].kind for group in groups)):
        return
    acyclic_polyacid = None
    if not any(atom.IsInRing() for atom in iter_atoms(mol)):
        from .acyclic_chalcogen_polyacid import acyclic_chalcogen_polyacid_scope

        acyclic_polyacid = acyclic_chalcogen_polyacid_scope(mol, Chem, inventory)
        if acyclic_polyacid is None:
            return
    kind = groups[0].kind
    from .multifunctional_parent import _paths
    from .proposal_plans import ParentProposalPlan, graph_proposal_bound

    prefix_centers = {group.center for group in inventory.groups
                      if group.kind == 'nitrile' and group not in groups}
    carbon = {atom.GetIdx() for atom in iter_atoms(mol)
              if atom.GetAtomicNum() == 6 and not atom.IsInRing()
              and atom.GetIdx() not in prefix_centers}
    if len(carbon) > 40:
        return
    for path in _paths(mol, carbon):
        covered = tuple(group for group in groups if group.center in {path[0], path[-1]})
        if not covered or len(covered) == len(groups):
            continue
        if acyclic_polyacid is not None and len(covered) != 2:
            continue
        # All source groups on this chain must be terminal suffix sites.
        if any(group.center in path and group not in covered for group in groups):
            continue
        # Each N remains primary in this finite extension. Existing complete
        # and ring writers retain their wider N-substitution contracts.
        if kind == 'amide' and any(
                mol.GetAtomWithIdx(group.ligand).GetTotalNumHs(includeNeighbors=True) != 2
                for group in covered):
            continue
        for oriented in (path, tuple(reversed(path))):
            if len(covered) == 1 and oriented[0] != covered[0].center:
                continue
            def construct(oriented=tuple(oriented), covered=covered):
                proposals = tuple(_render_partial_functional_proposal(
                    mol, Chem, covered, kind, oriented))
                if acyclic_polyacid is None:
                    return proposals
                from .acyclic_chalcogen_polyacid import mapped_polyacid_proposal_complete

                return tuple(proposal for proposal in proposals
                             if mapped_polyacid_proposal_complete(
                                 mol, Chem, acyclic_polyacid, proposal))
            if defer:
                yield ParentProposalPlan(
                    'mapped-chain-partial-functional', tuple(oriented),
                    frozenset(group.center for group in covered),
                    graph_proposal_bound(mol, oriented, covered), construct,
                    key=('partial-functional', tuple(oriented), tuple(
                        (group.kind, group.center, tuple(group.atoms)) for group in covered)),
                    single_proposal=(mol_cache_get(mol, 'candidate_telemetry') is None
                                     and _render_partial_functional_proposal
                                     is _DEFAULT_PARTIAL_FUNCTIONAL_RENDER),
                )
            else:
                yield from construct()


def _render_partial_functional_proposal(mol, Chem, covered, kind, path):
    from .complete_functional_parent import _render

    skipped = {idx for group in covered for idx in group.atoms if idx != group.center}
    owners = []
    if kind == 'ester':
        for group in covered:
            roots = [atom.GetIdx() for atom in mol.GetAtomWithIdx(group.ligand).GetNeighbors()
                     if atom.GetIdx() != group.center and atom.GetAtomicNum() > 1]
            if len(roots) != 1:
                return
            text, atoms = mapped_branch(mol, Chem, roots[0], group.ligand)
            if text not in {'methyl', 'ethyl', 'propyl', 'butyl', 'pentyl', 'hexyl'}:
                return
            owners.append((text, atoms))
            skipped.update(atoms)
        if len({text for text, _atoms in owners}) != 1:
            return
    row = _render(mol, Chem, covered, kind, path, False, False, skipped, owners,
                  mapped_ring_prefixes=True, return_component=True)
    if not row:
        return
    rendered, construction = row
    name, _locants, _prefix_locs, mapping, prefixes, _component, _owners = rendered
    if not _annotation_ownership(mol, mapping):
        return
    facts = _facts(mol, covered, path, mapping, prefixes, name,
                   cyclic=False, construction=construction)
    if facts is not None:
        yield ParentProposal(name, 'mapped-chain-partial-functional', kind,
            frozenset(group.center for group in covered), tuple(path),
            tuple(mapping.items()), facts, True, construction=construction)


# The native generator returns zero or one row. A replacement callback has
# no cardinality contract and keeps the conservative proposal-pool path.
_DEFAULT_PARTIAL_FUNCTIONAL_RENDER = _render_partial_functional_proposal


def sulfur_chain_proposals(mol, Chem, inventory, *, site_frontier=None, defer=False):
    """P-44.1.1 coverage when sulfur groups cannot all share one parent.

    An uncovered group remains a complete typed prefix. Comparing these
    proposals with ring parents uses the same published criteria as complete
    suffix constructions, rather than a ring-first or longest-chain fallback.
    """
    groups=inventory.principal
    kind=groups[0].kind
    if kind not in {'sulfonic-acid','sulfinic-acid','sulfonamide','thiol'}:
        return
    from .multifunctional_parent import _paths
    excluded={g.center for g in inventory.groups if g.kind=='nitrile'}
    carbon={a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum()==6
            and not a.IsInRing() and a.GetIdx() not in excluded}
    if len(carbon)>40:return
    site_candidates = ({tuple(row.parent_atoms): row for row in site_frontier.candidates}
                       if site_frontier is not None else {})
    for raw_path in _paths(mol,carbon):
      for path in (raw_path, tuple(reversed(raw_path))):
        graph_candidate = site_candidates.get(tuple(path)) if site_frontier is not None else None
        if site_frontier is not None and graph_candidate is None:
            continue
        covered=(tuple(site.source_group for site in graph_candidate.selected_sites)
                 if graph_candidate is not None else tuple(g for g in groups if g.owner in path))
        if not covered or (graph_candidate is None and len(covered)==len(groups)):continue
        skipped={i for g in covered for i in g.atoms if i!=g.center}
        for oriented in ((path,) if graph_candidate is not None else
                         (path,tuple(reversed(path)))):
            if defer:
                from .proposal_plans import ParentProposalPlan, graph_proposal_bound
                def construct(oriented=tuple(oriented), covered=covered,
                              skipped=frozenset(skipped), graph_candidate=graph_candidate):
                    return tuple(_render_sulfur_proposal(
                        mol, Chem, covered, kind, oriented, skipped, graph_candidate))
                yield ParentProposalPlan(
                    'mapped-chain-sulfur', tuple(oriented),
                    frozenset(g.center for g in covered),
                    graph_proposal_bound(mol, oriented, covered), construct,
                    key=('sulfur', tuple(oriented), tuple(
                        (g.kind, g.center, g.owner, tuple(g.atoms)) for g in covered)),
                )
            else:
                yield from _render_sulfur_proposal(
                    mol, Chem, covered, kind, oriented, skipped, graph_candidate)


def _render_sulfur_proposal(mol, Chem, covered, kind, oriented, skipped, graph_candidate):
    from .complete_functional_parent import _render
    row=_render(mol,Chem,covered,kind,oriented,True,False,skipped,(),
                mapped_ring_prefixes=True,return_component=True)
    if not row:return
    rendered,construction=row
    name,_,_,mapping,prefixes,_,_=rendered
    if not _annotation_ownership(mol,mapping):return
    facts=_facts(mol,covered,oriented,mapping,prefixes,name,cyclic=False,
                 construction=construction)
    if facts is not None:
        selected_sites = graph_candidate.selected_sites if graph_candidate is not None else ()
        omitted_sites = graph_candidate.omitted_sites if graph_candidate is not None else ()
        all_sites = tuple(site for site in (*selected_sites, *omitted_sites))
        yield ParentProposal(name,'mapped-chain-sulfur',kind,
            frozenset(g.center for g in covered),tuple(oriented),tuple(mapping.items()),
            facts,True,construction=construction,
            principal_site_keys=tuple(_site_key(site) for site in all_sites),
            selected_site_evidence=tuple(selected_sites),
            omitted_site_evidence=tuple(omitted_sites))


def oxygen_proposals(mol, Chem, inventory, *, site_frontier=None, defer=False, branch_resolver=None):
    """Mapped chain/monocycle suffix renderings sharing the same obligations."""
    from .multifunctional_parent import _paths
    from .constants import ROOTS

    groups = inventory.principal
    kind = groups[0].kind
    if kind in {'lactam','lactone','pseudoketone'}:
        # P-41 class 16 shares the -one construction. Recognition kind does
        # not restrict carbonyl coverage to rings or to acyclic ketones.
        kind='ketone'
    if kind not in {'aldehyde', 'ketone', 'alcohol'}:
        return
    carbon = {a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() == 6 and not a.IsInRing()}
    skeletons = [(p, False) for path in _paths(mol, carbon) for p in (path, tuple(reversed(path)))]
    for ring in mol.GetRingInfo().AtomRings():
        if any(mol.GetAtomWithIdx(i).GetAtomicNum()!=6 for i in ring):
            continue
        from .mapped_ring_parents import ring_systems
        if frozenset(ring) not in ring_systems(mol):
            continue
        aromatic = all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
        saturated = all(mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble() == 1
                        for a, b in zip(ring, ring[1:]+ring[:1]))
        if not saturated and not (aromatic and len(ring) == 6 and kind == 'alcohol'):
            continue
        if kind == 'aldehyde':
            continue
        for direction in (tuple(ring), tuple(reversed(ring))):
            for start in range(len(ring)):
                skeletons.append((direction[start:]+direction[:start], True))
    site_candidates = ({tuple(row.parent_atoms): row for row in site_frontier.candidates}
                       if site_frontier is not None else {})
    for path, cyclic in skeletons:
        graph_candidate = site_candidates.get(tuple(path)) if site_frontier is not None and not cyclic else None
        if site_frontier is not None and not cyclic and graph_candidate is None:
            continue
        if len(path) not in ROOTS:
            continue
        covered = ([site.source_group for site in graph_candidate.selected_sites]
                   if graph_candidate is not None else [g for g in groups if g.center in path])
        if not covered or (graph_candidate is None and
                           len({g.center for g in covered}) != len(covered)):
            continue
        if kind == 'aldehyde' and any(path.index(g.center) not in {0, len(path)-1} for g in covered):
            continue
        if defer:
            from .proposal_plans import ParentProposalPlan, graph_proposal_bound
            def construct(path=tuple(path), cyclic=cyclic, covered=tuple(covered),
                          graph_candidate=graph_candidate):
                proposal = _render_oxygen_proposal(
                    mol, Chem, inventory, path, cyclic, covered, graph_candidate, branch_resolver)
                return () if proposal is None else (proposal,)
            yield ParentProposalPlan(
                'mapped-ring-oxygen' if cyclic else 'mapped-chain-oxygen',
                tuple(path), frozenset(g.center for g in covered),
                graph_proposal_bound(mol, path, covered, cyclic=cyclic), construct,
                key=('oxygen', tuple(path), tuple(
                    (g.kind, g.center, g.owner, tuple(g.atoms)) for g in covered)),
                single_proposal=mol_cache_get(mol, 'candidate_telemetry') is None,
            )
        else:
            proposal = _render_oxygen_proposal(
                mol, Chem, inventory, path, cyclic, covered, graph_candidate, branch_resolver)
            if proposal is not None:
                yield proposal


def _render_oxygen_proposal(mol, Chem, inventory, path, cyclic, covered, graph_candidate,
                            branch_resolver=None):
    groups = inventory.principal
    kind = groups[0].kind
    if kind in {'lactam', 'lactone', 'pseudoketone'}:
        kind = 'ketone'
    from .constants import ROOTS
    from .branch_rules import parent_chain_name
    from .formatting import (attach_suffix, complete_parent_name, format_prefixes,
                             join_prefix_parent, multiplied_suffix)

    mapping = {i: str(j) for j, i in enumerate(path, 1)}
    suffix_atoms = {i for g in covered for i in g.atoms if i != g.center}
    # Suffix oxygens keep a source-atom binding at their host position.
    # A typed suffix writer owns their isotope descriptor, while the
    # host binding supplies isotope numbering and annotation coverage.
    for group in covered:
        for idx in group.atoms:
            if idx != group.center:
                mapping[idx] = mapping[group.center]
    resolve_branch = branch_resolver or mapped_branch
    embedded, prefixes, branch_ownership = set(), [], []
    failed = False
    for idx in path:
        for atom in mol.GetAtomWithIdx(idx).GetNeighbors():
            root = atom.GetIdx()
            if root in mapping or root in suffix_atoms or atom.GetAtomicNum() <= 1:
                continue
            text, owned = resolve_branch(mol, Chem, root, idx)
            if not text and kind == 'alcohol' and not cyclic and atom.IsInRing():
                from .branch_rules import collect_subtree
                from .component_mapping import _complete_rooted_ring

                scope = frozenset(collect_subtree(mol, root, idx))
                text = _complete_rooted_ring(mol, Chem, root, scope)
                if text:
                    owned = scope
            if not text or owned & (mapping.keys() | embedded) or any(c.isspace() for c in text):
                failed = True
                break
            prefixes.append((mapping[idx], text))
            # Preserve the original cut and the exact ownership supplied by
            # the branch writer. The component's P-14 presentation proof
            # revalidates these records before omitting any prefix locants.
            branch_ownership.append((root, idx, frozenset(owned), text))
            embedded.update(owned)
        if failed:
            break
    mapping = hydrogen_bindings(mol, mapping)
    if failed or not fully_covered(mol, mapping, embedded):
        return None
    locants = sorted(int(mapping[g.center]) for g in covered)
    aromatic = cyclic and all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in path)
    parent = ('benzene' if aromatic else 'cyclo'+ROOTS[len(path)]+'ane') if cyclic else complete_parent_name(parent_chain_name(mol, path))
    if not parent:
        return None
    suffix = {'aldehyde': 'al', 'ketone': 'one', 'alcohol': 'ol'}[kind]
    from .oxygen_characteristic_suffixes import oxygen_ending, oxygen_group
    units = tuple(oxygen_group(mol, group) for group in covered)
    modified_suffix = any(unit is not None and any(mol.GetAtomWithIdx(i).GetIsotope()
        for i in unit.oxygen_atoms + unit.attached_hydrogens) for unit in units)
    if aromatic and len(covered) == 1 and not modified_suffix:
        ending = 'phenol'
    elif kind == 'aldehyde':
        if len(covered) == 1 and len(path) == 1:
            ending = 'formaldehyde'
        elif len(covered) == 1 and parent == 'ethane':
            ending = 'acetaldehyde'
        else:
            ending = attach_suffix(parent, 'al' if len(covered) == 1 else 'dial')
    else:
        ending = attach_suffix(parent, multiplied_suffix(len(covered), suffix), map(str, locants))
        if not prefixes and len(path) == 2 and len(covered) == 1 and not any(a.GetIsotope() for a in iter_atoms(mol)):
            ending = attach_suffix(parent, suffix)
    suffix_owned = frozenset()
    if modified_suffix or len(path) == 1 and kind == 'alcohol':
        carbon_masses = [mol.GetAtomWithIdx(i).GetIsotope() for i in path]
        numbered_parent_isotopes = (
            len(path) > 1 and any(carbon_masses) and len(set(carbon_masses)) > 1
        ) or any(n.GetAtomicNum() == 1 and n.GetIsotope()
                 for i in path for n in mol.GetAtomWithIdx(i).GetNeighbors())
        omit_ethanol = (kind == 'alcohol' and len(path) == 2 and
                        len(covered) == 1 and not prefixes and not numbered_parent_isotopes)
        suffix_locs = (() if kind == 'aldehyde' or len(path) == 1 or omit_ethanol else
                       tuple(map(str, locants)))
        ending, suffix_owned = oxygen_ending(mol, Chem, parent, covered,
                                             suffix_locs, suffix=suffix)
        if not ending:
            return None
    construction = MappedComponent(
        ending, tuple(mapping.items()), frozenset(mapping), tuple(prefixes),
        frozenset(embedded), local_isotope_atoms=suffix_owned,
        prefix_branch_ownership=tuple(branch_ownership))
    name = decorate_component(mol, Chem, construction)
    if not name or len(name) > 260 or not _annotation_ownership(mol, mapping):
        return None
    facts = _facts(mol, covered, path, mapping, prefixes, name, cyclic=cyclic,
                   construction=construction)
    if facts is not None:
        selected_sites = graph_candidate.selected_sites if graph_candidate is not None else ()
        omitted_sites = graph_candidate.omitted_sites if graph_candidate is not None else ()
        all_sites = tuple((*selected_sites, *omitted_sites))
        return ParentProposal(name, 'mapped-ring-oxygen' if cyclic else 'mapped-chain-oxygen',
                             kind, frozenset(g.center for g in covered), tuple(path),
                             tuple(mapping.items()), facts, True, construction=construction,
                             principal_site_keys=tuple(_site_key(site) for site in all_sites),
                             selected_site_evidence=tuple(selected_sites),
                             omitted_site_evidence=tuple(omitted_sites))



def cross_route_selection(mol, Chem, prepared=(), *, nitrogen_oxo_higher_class=False):
    """Compatibility projection of the status-bearing shared selection."""
    return cross_route_selection_outcome(
        mol, Chem, prepared,
        nitrogen_oxo_higher_class=nitrogen_oxo_higher_class).selection


def _prefer_annotation_parent_ownership(mol, Chem, proposals, ring_proposals):
    """Let a fresh complete annotation proof own an equivalent ring parent."""
    proofed = [proposal for proposal in proposals
               if getattr(proposal.construction, "parent_annotation_proof", None) is not None]
    if not proofed:
        return "ready", proposals, ring_proposals, ""

    from .parent_annotation_proof import (
        ParentAnnotationProof, ParentAnnotationStatus,
        validate_monocycle_parent_annotation_proof,
    )
    from .component_mapping import MappedComponent
    from .extended_inventory import selection_inventory as fresh_selection_inventory
    fresh_inventory = fresh_selection_inventory(Chem.Mol(mol))
    source_rows = tuple(getattr(getattr(fresh_inventory, "source_frontier", None),
                                "groups", ()))
    source_identity = {
        (str(g.kind), int(g.center), tuple(int(i) for i in g.atoms),
         int(g.owner), int(g.ligand)) for g in source_rows
    }
    principal_identity = {
        (str(g.kind), int(g.center), tuple(int(i) for i in g.atoms),
         int(g.owner), int(g.ligand)) for g in getattr(fresh_inventory, "principal", ())
    }
    for proposal in proofed:
        construction = proposal.construction
        if not isinstance(construction, MappedComponent):
            return "pending", proposals, ring_proposals, "Annotation evidence is not attached to a mapped construction."
        proof = construction.parent_annotation_proof
        if not isinstance(proof, ParentAnnotationProof):
            return "pending", proposals, ring_proposals, "A mapped construction carried conflicting annotation evidence."
        proposal_map = dict(proposal.atom_locants)
        construction_map = dict(construction.atom_locants)
        proof_groups = tuple(
            group for group in fresh_inventory.groups
            if (str(group.kind), int(group.center), tuple(int(i) for i in group.atoms),
                int(group.owner), int(group.ligand)) in set(proof.principal_identity)
        )
        proof_locants = tuple(
            (atom_id, proposal_map[atom_id]) for atom_id in proof.parent_atoms
            if atom_id in proposal_map
        )
        required_original_ownership = {
            atom.GetIdx() for atom in iter_atoms(mol)
            if atom.GetAtomicNum() > 1 or atom.GetIsotope() or int(atom.GetChiralTag())
        }
        construction_owned = frozenset(construction.atoms)
        construction_embedded = frozenset(construction.embedded_atoms)
        if (len(proposal.atom_locants) != len(proposal_map)
                or len(construction.atom_locants) != len(construction_map)
                or proposal_map != construction_map
                or construction_owned != frozenset(construction_map)
                or construction_owned & construction_embedded
                or not (construction_owned | construction_embedded) >= required_original_ownership
                or not proof.parent_atoms <= construction_owned
                or proof.status is not ParentAnnotationStatus.PROVED
                or source_identity != principal_identity
                or set(proof.principal_identity) != source_identity
                or frozenset(proposal.parent_atoms) != proof.parent_atoms
                or set(dict(proof_locants)) != set(proof.parent_atoms)
                or frozenset(proposal.principal_centers)
                != frozenset(identity[1] for identity in source_identity)
                or not validate_monocycle_parent_annotation_proof(
                    mol, Chem, proof, parent_locants=proof_locants,
                    principal_groups=proof_groups,
                )):
            return "pending", proposals, ring_proposals, "A mapped parent annotation proof is stale, incomplete, or conflicting."

    suppressed_ring_ids = set()
    for proposal in proofed:
        equivalent = [ring for ring in ring_proposals
                      if frozenset(ring.parent_atoms) == frozenset(proposal.parent_atoms)
                      and frozenset(ring.principal_centers)
                      == frozenset(proposal.principal_centers)]
        if not equivalent:
            continue
        peers = [row for row in proofed
                 if frozenset(row.parent_atoms) == frozenset(proposal.parent_atoms)
                 and frozenset(row.principal_centers) == frozenset(proposal.principal_centers)]
        # Every peer has already been revalidated against its OWN atom-locant
        # map above. Equivalent numberings of one physical parent naturally
        # have different proof records (including isotope locants); comparing
        # the records for equality mistakes alternatives for contradictions.
        # Keep those alternatives for the normal P-14/P-45 comparator.
        suppressed_ring_ids.update(id(ring) for ring in equivalent)
        peer_ids = {id(row) for row in peers}
        proposals = [row for row in proposals
                     if id(row) in peer_ids or not (
                         frozenset(row.parent_atoms) == frozenset(proposal.parent_atoms)
                         and frozenset(row.principal_centers)
                         == frozenset(proposal.principal_centers)
                     )]
    ring_proposals = [row for row in ring_proposals if id(row) not in suppressed_ring_ids]
    return "ready", proposals, ring_proposals, ""


def _oxygen_suffix_site_join(mol, Chem, inventory):
    """Join existing oxygen-isotope writers to constitutional P-44 maxima.

    The carbon-tree frontier does not prove isotope/stereo numbering. Its
    structural maxima can nevertheless be retained when P-82.2.2.2 leaves
    the source-group set unchanged. Each maximum is subsequently rendered
    on the ORIGINAL graph by oxygen_proposals, including its suffix oxygen
    roles, isotope placement, CIP, full coverage and numbering comparisons.
    """
    groups = tuple(inventory.principal)
    if not groups or groups[0].kind not in {'alcohol', 'ketone'}:
        return None
    from .oxygen_characteristic_suffixes import oxygen_group
    units = tuple(oxygen_group(mol, group) for group in groups)
    if any(unit is None for unit in units):
        return None
    suffix_oxygen = {idx for unit in units for idx in unit.oxygen_atoms}
    if not any(mol.GetAtomWithIdx(idx).GetIsotope() for idx in suffix_oxygen):
        return None
    # Other noncarbon annotations need their own role writer. Parent carbon
    # labels and assigned carbon stereo retain the existing mapped decorator.
    if any(atom.GetIsotope() and atom.GetAtomicNum() not in {1, 6}
           and atom.GetIdx() not in suffix_oxygen for atom in iter_atoms(mol)):
        return None
    base = Chem.Mol(mol)
    Chem.RemoveStereochemistry(base)
    for atom in base.GetAtoms():
        atom.SetIsotope(0)
    from .principal_site_parent_frontier import principal_site_parent_frontier
    frontier = principal_site_parent_frontier(base, Chem)
    if (getattr(frontier, 'status', None) != 'proved' or frontier.subordinate_groups
            or {site.source_group for site in frontier.sites} != set(groups)):
        return None
    return frontier


def cross_route_selection_outcome(mol, Chem, prepared=(), *, nitrogen_oxo_higher_class=False, lazy=True,
                                  deliver_annotated_general=False, deliver_owned_general=False):
    from .proposal_plans import DeferredCompatibilityCandidates

    inventory = cross_route_scope(mol)
    if inventory is None:
        if isinstance(prepared, DeferredCompatibilityCandidates):
            prepared.restore()
        return CrossRouteOutcome('outside-scope', reason='No proved shared parent scope.')
    from .principal_site_parent_frontier import principal_site_parent_frontier
    site_frontier = principal_site_parent_frontier(mol, Chem)
    if getattr(inventory, 'isotope_partition_rule', ''):
        # A constitutional all-site proof cannot undo the prior isotope
        # suffix/prefix partition. The selected class inventory is unchanged.
        site_frontier = None
    if getattr(site_frontier, 'status', None) != 'proved':
        site_frontier = _oxygen_suffix_site_join(mol, Chem, inventory) or site_frontier
    if getattr(site_frontier, 'status', None) == 'proved':
        principal_groups = tuple(site.source_group for site in site_frontier.sites)
        # Only the proved ordinary tree universe may replace the local
        # principal projection. Annotated/out-of-scope graphs retain the
        # preexisting cross_route_scope partition (including P-82.2 isotope
        # group selection) without a new global class-17 barrier.
        from types import SimpleNamespace
        inventory = SimpleNamespace(groups=inventory.groups, principal=principal_groups,
                                    supported=inventory.supported)
    else:
        site_frontier = None

    from .complete_functional_parent import functional_parent_frontier
    prefix_atoms = frozenset(g.center for g in inventory.groups
                             if g.kind == 'nitrile' and g not in inventory.principal)
    structural_frontier = functional_parent_frontier(
        mol, Chem, excluded_skeletal_atoms=prefix_atoms,
        mapped_ring_prefixes=True, principal=inventory.principal,
        nitrogen_oxo_higher_class=nitrogen_oxo_higher_class)
    if structural_frontier.status in {"blocked", "conflict"}:
        # P-44 selects the structural parent before writing its name. The
        # complete-group maximum cannot disappear into a smaller route pool.
        if isinstance(prepared, DeferredCompatibilityCandidates):
            prepared.restore()
        return CrossRouteOutcome('pending', functional_parent_frontier=structural_frontier,
                                 reason=structural_frontier.reason,
                                 principal_site_frontier=site_frontier)
    special_principal = inventory.principal[0].kind in {'imine', 'peroxol', 'oxime', 'oxime-ether', 'hydrazide'}
    proposals = list(chain(() if special_principal else functional_proposals(
                               mol, Chem, inventory, structural_frontier=structural_frontier,
                               nitrogen_oxo_higher_class=nitrogen_oxo_higher_class, defer=lazy),
                           partial_functional_proposals(mol, Chem, inventory, defer=lazy),
                           oxygen_proposals(mol, Chem, inventory, site_frontier=site_frontier, defer=lazy),
                           sulfur_chain_proposals(mol,Chem,inventory,site_frontier=site_frontier,defer=lazy)))
    from .amine_components import chain_proposals
    proposals.extend(chain_proposals(mol,Chem,inventory,site_frontier=site_frontier,defer=lazy))
    from .hydroxylamine_methods import chain_proposals as hydroxylamine_chain_proposals
    proposals.extend(hydroxylamine_chain_proposals(mol,Chem,inventory))
    from .imine_family_components import chain_proposals as imine_chain_proposals
    proposals.extend(imine_chain_proposals(mol,Chem,inventory))
    from .peroxol_components import chain_proposals as peroxol_chain_proposals
    proposals.extend(peroxol_chain_proposals(mol,Chem,inventory))
    if inventory.principal[0].kind == 'hydrazide':
        from .hydrazide_components import hydrazide_proposals
        proposals.extend(row.as_parent_proposal() for row in hydrazide_proposals(mol,Chem,inventory))
    from .proposal_plans import ParentProposalPlan, RouteProposalPool
    if lazy:
        from .mapped_ring_parents import proposal_plans as ring_proposal_plans
        proposals.extend(ring_proposal_plans(mol, Chem, inventory.principal))
    else:
        proposals.append(ParentProposalPlan(
            'mapped-ring', (), frozenset(), None,
            lambda: _eager_ring_proposals(mol, Chem, inventory), ring_route=True,
        ))

    pool = RouteProposalPool(proposals)
    pool.realize(lazy=lazy)
    pool.ensure_cycle_safety()
    ordinary, rings = pool.materialized()
    outcome = _finish_cross_route_selection(
        mol, Chem, inventory, prepared, structural_frontier, site_frontier, ordinary, rings)
    # A provisional winner never licenses dropping a live competitor. The
    # final validated winner must discharge every skipped plan directly.
    # Unknown/tied/failed selections restore the eager pool, including every
    # unresolved construction and all original completeness barriers.
    if pool.evidence and (outcome.status != 'complete' or not pool.discharged(outcome.selection)):
        pool.realize_remaining()
        ordinary, rings = pool.materialized()
        outcome = _finish_cross_route_selection(
            mol, Chem, inventory, prepared, structural_frontier, site_frontier, ordinary, rings)
    # Only the principal pipeline opts into immediate named delivery. Every
    # deferred compatibility candidate is excluded by annotation or by exact
    # typed ownership of its physical parent. Loading cannot change this
    # pending proposal or its explicit justification.
    # Other callers retain restoration and its fallback preparation effects.
    compatible_general_delivery = (
        isinstance(prepared, DeferredCompatibilityCandidates)
        and ((deliver_annotated_general and prepared.annotation_excluded)
             or (deliver_owned_general and prepared.ownership_excluded))
        and outcome.status == 'pending'
        and outcome.general_proposal is not None
        and bool(outcome.general_proposal.name)
    )
    if (isinstance(prepared, DeferredCompatibilityCandidates)
            and not prepared.loaded and outcome.status != 'complete'
            and not compatible_general_delivery):
        # Missing/failed/unknown typed construction cannot discharge a live
        # compatibility pool or erase its contribution to later fallbacks.
        prepared.restore()
        outcome = _finish_cross_route_selection(
            mol, Chem, inventory, prepared, structural_frontier, site_frontier, ordinary, rings)
    if pool.evidence:
        from dataclasses import replace
        outcome = replace(outcome, deferred_proposals=pool.evidence,
                          selection=(replace(outcome.selection, deferred_proposals=pool.evidence)
                                     if outcome.selection is not None else None))
    return outcome


def _eager_ring_proposals(mol, Chem, inventory):
    from .mapped_ring_parents import records,parent_facts
    ring_proposals=[]
    for record in records(mol,Chem,principal=inventory.principal):
        facts=parent_facts(mol,record,inventory.principal[0].seniority)
        if facts is not None:
            ring_proposals.append(ParentProposal(record.name,'mapped-ring:'+record.method,
                inventory.principal[0].kind,record.principal_centers,record.parent_atoms,
                record.atom_locants,facts,True,ring_state=record,construction=record.construction))

    return tuple(ring_proposals)


def _finish_cross_route_selection(mol, Chem, inventory, prepared, structural_frontier,
                                  site_frontier, proposals, ring_proposals):
    from .proposal_plans import DeferredCompatibilityCandidates
    if isinstance(prepared, DeferredCompatibilityCandidates):
        prepared.ownership_excluded = False
    ownership_status, proposals, ring_proposals, ownership_reason = (
        _prefer_annotation_parent_ownership(mol, Chem, proposals, ring_proposals)
    )
    if ownership_status != "ready":
        return CrossRouteOutcome('pending', candidates=tuple(proposals),
                                 reason=ownership_reason)
    # A proved typed rendering owns equivalent numberings of that physical
    # parent (P-14/P-52/P-58); compatibility spellings do not compete as new
    # physical parents. Other parent structures still enter P-44 comparison.
    owned={(frozenset(p.parent_atoms),p.principal_centers) for p in ring_proposals}
    proposals=[p for p in proposals if (frozenset(p.parent_atoms),p.principal_centers) not in owned]
    proposals.extend(ring_proposals)
    # A typed chain producer also fixes the preferred spelling of that
    # physical parent. Legacy renderings are not independent P-44 parents.
    owned.update((frozenset(p.parent_atoms), p.principal_centers) for p in proposals
                 if p.route in {'mapped-chain-oxygen', 'mapped-chain-imine-family', 'mapped-chain-imine',
                                'mapped-chain-oxime', 'mapped-chain-peroxol',
                                'mapped-chain-hydroxylamine',
                                'mapped-hydrazide'})
    # Established acyclic oxygen and ring renderers participate only with
    # explicit comparable facts. Their unknown scope cannot establish a tie.
    from .principal_oxygen import principal_oxygen_record
    legacy = principal_oxygen_record(mol)
    if legacy.name and legacy.parent_selection is not None:
        mapping = dict(legacy.stereo_locants)
        groups = [g for g in inventory.principal if g.center in mapping]
        if groups and (frozenset(mapping),frozenset(g.center for g in groups)) not in owned:
            # Reuse its proven prefix evidence; opt-in annotation evidence is
            # absent here because this established producer declines isotopes.
            from dataclasses import replace
            from .parent_annotations import (isotope_seniority, isotope_numbering_key, stereo_seniority,
                                             substituent_isotopes, stereo_identity, stereo_numbering)
            facts = replace_parent_facts(legacy.parent_selection,
                            numbering_scope=(frozenset(mapping), frozenset(g.center for g in groups)),
                            isotope_key=isotope_seniority(mol, mapping), stereo_key=stereo_seniority(mol, mapping),
                            substituent_isotope_key=substituent_isotopes(mol, mapping),
                            equivalent_isotope_key=isotope_numbering_key(
                                mol, mapping, include_substituent_groups=True),
                            stereo_identity=stereo_identity(mol, mapping),
                            stereo_numbering_key=stereo_numbering(mol, mapping))
            proposals.append(ParentProposal(legacy.name, 'established-acyclic-oxygen', groups[0].kind,
                                           frozenset(g.center for g in groups), tuple(mapping),
                                           legacy.stereo_locants, facts, True))
    from .proposal_plans import DeferredCompatibilityCandidates
    if isinstance(prepared, DeferredCompatibilityCandidates):
        prepared = prepared.resolve(owned, inventory.principal)
    for item in prepared:
        candidate = item.candidate
        facts = candidate.parent_selection
        mapping = dict(candidate.stereo_locants)
        groups = [g for g in inventory.principal
                  if g.center in candidate.parent_atoms or g.owner in candidate.parent_atoms]
        if facts is None or facts.suffix_count != len(groups) or not groups or not candidate.parent_atoms <= mapping.keys():
            continue
        if (frozenset(candidate.parent_atoms),frozenset(g.center for g in groups)) in owned:
            continue
        if any(a.GetIsotope() or int(a.GetChiralTag()) for a in iter_atoms(mol)) or any(int(b.GetStereo()) for b in iter_bonds(mol)):
            continue
        from dataclasses import replace
        from .parent_annotations import (isotope_seniority, isotope_numbering_key,
                                         stereo_seniority, stereo_identity, stereo_numbering)
        facts = replace_parent_facts(facts, numbering_scope=(candidate.parent_atoms, frozenset(g.center for g in groups)),
                        isotope_key=isotope_seniority(mol, mapping), stereo_key=stereo_seniority(mol, mapping),
                        equivalent_isotope_key=isotope_numbering_key(
                            mol, mapping, include_substituent_groups=True),
                        stereo_identity=stereo_identity(mol, mapping),
                        stereo_numbering_key=stereo_numbering(mol, mapping))
        proposals.append(ParentProposal(item.output_name or item.name, 'established-ring:'+candidate.family,
                                       groups[0].kind, frozenset(g.center for g in groups),
                                       tuple(sorted(candidate.parent_atoms)), tuple(mapping.items()), facts, True, item))
    if site_frontier is not None:
        proposals, missing_site_maxima = _qualify_site_frontier_proposals(
            proposals, site_frontier,
        )
        if missing_site_maxima:
            return CrossRouteOutcome(
                'pending', candidates=tuple(proposals),
                reason='A source-proved maximum principal-site parent did not receive a complete mapped rendering.',
                principal_site_frontier=site_frontier,
            )
    # Collapse identical output/evidence copies before the partial comparison.
    unique = {}
    for proposal in proposals:
        unique.setdefault((proposal.name, proposal.parent_atoms, proposal.atom_locants), proposal)
    proposals = tuple(unique.values())
    if not proposals:
        required_parent = bool(getattr(structural_frontier, 'required_parent_atoms', ()))
        status = ('pending' if required_parent or (
                  structural_frontier.status in {'complete', 'tie'}
                  and structural_frontier.candidates) else 'outside-scope')
        return CrossRouteOutcome(status, functional_parent_frontier=structural_frontier,
                                 reason=('The source-proved required parent has no complete mapped proposal.'
                                         if required_parent else 'No eligible rendered parent proposal was supplied.'),
                                 principal_site_frontier=site_frontier)
    from .method_parent_frontier import select_method_parent
    if site_frontier is not None and site_frontier.candidate_universe_complete:
        completeness = CandidateCompleteness(
            'ordinary-principal-site-carbon-tree', site_frontier.selection_rules,
            tuple(tuple(row.parent_atoms) for row in site_frontier.candidates),
            tuple(tuple(row.parent_atoms) for row in proposals),
            'Every oriented source-qualified principal-site maximum has a mapped construction.',
            proposal_keys=tuple(proposal_identity(row) for row in proposals),
        )
    else:
        completeness = getattr(structural_frontier, 'candidate_completeness', None)
        if completeness is not None:
            # The original compared records remain in structural_frontier;
            # bind the joined pool separately after its source eliminations.
            from dataclasses import replace
            completeness = replace(completeness,
                proposal_keys=tuple(proposal_identity(row) for row in proposals))
    ranked = select_method_parent(
        proposals, lambda proposal: proposal.parent_selection,
        physical_parent_identity=lambda proposal: (
            frozenset(proposal.parent_atoms), _proposal_site_key(proposal)),
        completeness=completeness,
    )
    frontier = ranked.candidates
    potential=(() if site_frontier is not None else
               _unrendered_parent_facts(mol,inventory,proposals))
    chosen = ranked.selected
    if chosen is not None:
        comparisons = tuple(compare_parents(chosen.parent_selection, p.parent_selection)
                            for p in proposals)
        admission=tuple(compare_parents(chosen.parent_selection,facts) for facts in potential)
        if all(c.preference==-1 for c in admission):
            evidence_candidates = (*proposals, *getattr(structural_frontier, 'compared_candidates', ()))
            justification = parent_justification(
                completeness, evidence_candidates, chosen,
                method_rule='P-51.1/P-45.1.1',
                method_proved=carbon_tree_method_proved(mol, inventory.principal, chosen),
            )
            selection = CrossRouteSelection(chosen, proposals, frontier, comparisons,admission,
                                            structural_frontier, site_frontier,
                                            candidate_completeness=completeness,
                                            justification=justification)
            return CrossRouteOutcome('complete', selection, proposals, structural_frontier,
                                     'All supplied competitors and unrendered witnesses were resolved.',
                                     site_frontier, justification=justification)
    # A complete mapped construction can remain available when ONLY the
    # broader universe is unproved. Its local source comparison does not
    # discharge that missing global evidence. Retain the exact typed writer,
    # rather than forcing a later legacy route to reconstruct its branches.
    local = (ranked if chosen is not None else select_method_parent(
        proposals, lambda proposal: proposal.parent_selection,
        proposal_set_complete=True))
    general = local.selected
    pending_reason = (ranked.reason if chosen is None else
                      'An unrendered parent witness was not proved junior.')
    if completeness is None and structural_frontier.reason:
        pending_reason += ' Source-frontier limit: ' + structural_frontier.reason
    return CrossRouteOutcome('pending', candidates=proposals,
                             functional_parent_frontier=structural_frontier,
                             reason=pending_reason,
                             principal_site_frontier=site_frontier,
                             general_proposal=general,
                             justification=parent_justification(
                                 completeness, proposals, general,
                                 method_rule='P-51.1/P-45.1.1',
                                 method_proved=carbon_tree_method_proved(mol, inventory.principal, general),
                                 reason=pending_reason))


def _unrendered_parent_facts(mol,inventory,proposals):
    """A missing renderer is not proof that a competing parent is ineligible.

    Upper coverage bounds and graph facts can prove an admitted parent senior
    before unavailable numbering/vocabulary. Otherwise decline the pool.
    These witnesses have no invented output or locants.
    """
    from collections import Counter
    from .parent_selection import ParentFacts
    from .mapped_ring_parents import ring_systems,_site
    from .multifunctional_parent import _paths
    represented={(frozenset(p.parent_atoms),p.principal_centers) for p in proposals}
    groups=inventory.principal;kind=groups[0].kind
    candidates=[]
    for core in ring_systems(mol):
        centers=frozenset(g.center for g in groups if _site(mol,g,core) in core)
        if centers:candidates.append((core,centers,'ring'))
    # A subordinate nitrile remains an indivisible cyano prefix, as in the
    # rendered functional proposals above; it cannot extend an alkane parent.
    excluded={g.center for g in inventory.groups if g.kind=='nitrile' and g not in groups}
    carbon={a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum()==6 and not a.IsInRing() and a.GetIdx() not in excluded}
    for path in _paths(mol,carbon):
        centers=frozenset(g.center for g in groups if (g.center in path or
            kind in {'amine','peroxol','imine','sulfonic-acid','sulfinic-acid','sulfonamide','thiol'} and _site(mol,g,path) in path) and
            (kind not in {'acid','amide','nitrile','ester','aldehyde'} or path.index(g.center) in {0,len(path)-1}))
        if centers:candidates.append((frozenset(path),centers,'chain'))
    facts=[];seen=set()
    for atoms,centers,parent_kind in candidates:
        key=(atoms,centers)
        if key in represented or key in seen:continue
        seen.add(key)
        edges=sum(1 for b in iter_bonds(mol) if {b.GetBeginAtomIdx(),b.GetEndAtomIdx()}<=atoms)
        facts.append(ParentFacts(principal_class=groups[0].seniority,suffix_count=len(centers),kind=parent_kind,
            elements=tuple(sorted(Counter(mol.GetAtomWithIdx(i).GetAtomicNum() for i in atoms).items())),
            ring_count=edges-len(atoms)+1))
    return tuple(facts)


def _qualify_site_frontier_proposals(proposals, frontier):
    """Restrict proposals to exact graph maxima and retain source-site roles.

    The graph proof is independent of rendering. This adapter makes a failed
    mapped writer observable and verifies original group identity, selected
    versus omitted sites, host locants, and the actual suffix occurrence
    count before a candidate enters parent comparison.
    """
    all_sites = tuple(frontier.sites)
    all_keys = tuple(_site_key(site) for site in all_sites)
    candidates = {tuple(row.parent_atoms): row for row in frontier.candidates}
    qualified = []
    for proposal in proposals:
        candidate = candidates.get(tuple(proposal.parent_atoms))
        if candidate is None or proposal.parent_selection is None:
            continue
        selected = tuple(candidate.selected_sites)
        omitted = tuple(candidate.omitted_sites)
        selected_keys = tuple(_site_key(site) for site in selected)
        if proposal.principal_site_keys:
            proposal_keys = tuple(proposal.principal_site_keys)
            if set(proposal_keys) != set(all_keys):
                continue
            proposal_selected = tuple(proposal.selected_site_evidence)
            proposal_omitted = tuple(proposal.omitted_site_evidence)
            if ({_site_key(site) for site in proposal_selected} != set(selected_keys)
                    or {_site_key(site) for site in proposal_omitted}
                    != {_site_key(site) for site in omitted}):
                continue
        else:
            # Legacy primary-amine proposals do not yet expose site fields.
            # Their original center is unique within this admitted class, so
            # attach the exact source records only after matching all centers.
            selected_centers = frozenset(site.principal_center for site in selected)
            if proposal.principal_centers != selected_centers:
                continue
            if len(selected) != proposal.parent_selection.suffix_count:
                continue
        mapping = dict(proposal.atom_locants)
        site_locants = {(center, host): str(locant)
                        for center, host, locant in candidate.site_locants}
        if any(mapping.get(site.host_atom) != site_locants.get(
                (site.principal_center, site.host_atom)) for site in selected):
            continue
        if proposal.parent_selection.suffix_count != len(selected):
            continue
        from dataclasses import replace
        qualified.append(replace(
            proposal,
            principal_site_keys=all_keys,
            selected_site_evidence=selected,
            omitted_site_evidence=omitted,
        ))

    # Every oriented P-44 maximum is a live obligation. A single successful
    # rendering of another orientation cannot hide a writer failure here.
    qualified_paths = {
        (tuple(proposal.parent_atoms),
         tuple(_site_key(site) for site in proposal.selected_site_evidence))
        for proposal in qualified
    }
    missing = tuple(
        candidate for candidate in frontier.candidates
        if (tuple(candidate.parent_atoms),
            tuple(_site_key(site) for site in candidate.selected_sites)) not in qualified_paths
    )
    return qualified, missing
