"""Independent fast general naming based on public NISPO 0.1.7.

Fast mode uses the bundled writer and bounded shared component producers.
The comprehensive engine is a last resort. OPSIN is never invoked while
naming. Graph recovery and preferred-name selection are not guaranteed.
"""
from __future__ import annotations

from functools import lru_cache
import re

from ._fast_engine import api as _public
from ._fast_engine.formatting import complete_selected_name
from ._fast_engine.input_stereochemistry import unsupported_relative_stereochemistry
from .amidine_methods import _amidoxime_query, amidoxime_decision
from .amine_cation_components import ring_alkyl_aminium, ring_alkyl_aminium_query
from .carbonyl_oxide_methods import carbonyl_oxide_decision, carbonyl_oxide_query
from .fast_parent_construction import (bounded_ester_name, bounded_fused_carbonyl_name,
                                       principal_acid_name, repaired_parent_name)
from .locant_presentation import placement_proof_scope
from .naming_result import NamingResult, SelectionLimitation
from .prefix_presentation import (normalize_enclosing_marks, numbered_biphenyl_ligand,
                                  simple_phenylalkyl_prefixes)
from .ring_template_presentation import locanted_fused_component_presentation

_LIMITATION = SelectionLimitation(
    'explicit-fast-mode', 'P-44/P-45/P-51',
    'Fast mode does not establish full preferred-parent/method comparison; '
    'this general name is not a verified PIN.',
)
_RELATIVE_LIMITATION = SelectionLimitation(
    'input-stereochemistry', 'P-91/P-93',
    'Input carries relative, mixture, or non-absolute CX stereochemistry; '
    'the ordinary molecular writer cannot safely assert one absolute configuration. '
    'Use an applicable explicitly declared stereo ensemble or subgroup operation.',
)
_FUSED_ARENE_LIMITATION = SelectionLimitation(
    'fast-ring-construction', 'P-23/P-25',
    'A graph-based von Baeyer construction was used; comparison with preferred '
    'fused parents and all numbering and unsaturation alternatives remains unresolved.',
)
_FUSED_HYDRIDE_METHODS = frozenset({
    'component-fused', 'component-fused-benzofuran-pyridine', 'component-fused-three-face-azine',
    'component-fused-general', 'retained-fused',
})


@lru_cache(maxsize=2)
def _component_queries(Chem):
    return tuple(Chem.MolFromSmarts(query) for query in (
        '[#6]-[S;!R](=[O;D1])(=[N])-[N]',
        '[O;R]1[#6;R][#6;R][N;R][#6;R][#6;R]1',
        '[#6;R]-[CX3;!R](=[OX1])[OX2H]',
        '[C;!R;X3;H1](=[N;!R;X2]-[c])-[N;!R;X3;H0](-[C;!R])-[C;!R]',
        '[#6]-[CX3;!R](=[OX1])[OX2H]',
        '[P;X4;+0](=[O;D1])(-[O;X2])-[#6]',
        '[N;R]-[C;!R](=[O;D1])-[C;X4;H2]-[N]=[N+]=[N-]',
        '[P;X4;+0](=[O;D1])(-[O;X2;H0]-[#6])(-[O;X2;H0]-[#6])-[N;X3]',
        '[N;R]-[C;R]-[C;X4;H2]-[N]=[N+]=[N-]',
        '[#6]-[O;X2;!R;+0]-[#6]',
    ))


def _fast_component_branch(mol, _Chem, root, blocked):
    from ._fast_engine.branch_rules import collect_subtree
    from ._fast_engine.parent_rules import _simple_or_branch_prefix

    scope = frozenset(collect_subtree(mol, root, blocked))
    if not scope or blocked in scope or any(
            mol.GetAtomWithIdx(i).GetChiralTag() or mol.GetAtomWithIdx(i).GetFormalCharge()
            for i in scope):
        return '', frozenset()
    if {(i, a.GetIdx()) for i in scope for a in mol.GetAtomWithIdx(i).GetNeighbors()
        if a.GetAtomicNum() > 1 and a.GetIdx() not in scope} != {(root, blocked)}:
        return '', frozenset()
    if any(b.GetStereo() for i in scope for b in mol.GetAtomWithIdx(i).GetBonds()):
        return '', frozenset()
    name = _simple_or_branch_prefix(mol, root, blocked)
    return (name, scope) if name else ('', frozenset())


_EXTENDED_ELEMENTS = re.compile(r'P|B(?!r)|\[(?:Si|Ge|As|Se|Sn|Sb|Te|Pb|Al|Ga|In|Tl)')
_HALOGEN = re.compile('F|Cl|Br|I')
_GROUP13 = re.compile(r'B(?!r)|\[(?:Al|Ga|In|Tl)')
_GROUP14 = re.compile(r'\[(?:Si|Ge|Sn|Pb)')
_GROUP14_PAIR = re.compile(r'\[(Si|Ge|Sn|Pb)[^]]*\].*?\[\1')
_N_OR_O = re.compile('[NnOo]')


@lru_cache(maxsize=2)
def _alcohol_site_query(Chem):
    return Chem.MolFromSmarts('[O;H1;D1;+0]-[#6;!$(C=O)]')


def _shared_ring_name(mol, Chem, rings, size, canonical):
    # The two-unit exception has a central square. All other members have
    # at least seven faces, including any further fused or pendant rings.
    if (len(rings) >= 5 and size >= 20 and size >= 4 * (len(rings) - 1)
            and (len(rings) >= 7 or any(len(r) == 4 for r in rings))):
        from .polynaphthylenes import polynaphthylene_decision

        decision = polynaphthylene_decision(mol, Chem, rings)
        return decision.name if decision else ''
    if len(rings) > 2 or len(rings) == 1 and canonical and 'c' in canonical:
        return ''
    ring = rings[0]
    bare = len(rings) == 1 and len(ring) == size
    extended = canonical is None or _EXTENDED_ELEMENTS.search(canonical)
    if bare or extended:
        element = mol.GetAtomWithIdx(ring[0]).GetAtomicNum()
        if bare and element != 6 and all(mol.GetAtomWithIdx(i).GetAtomicNum() == element for i in ring):
            from .saturated_heterocycle_methods import homogeneous_ring_parent

            decision = homogeneous_ring_parent(mol, Chem, ring, element)
            if decision:
                return decision.name
        if extended or bare and any(c in canonical for c in 'NOS'):
            from .saturated_heterocycle_methods import saturated_heteromonocycle_decision

            decision = saturated_heteromonocycle_decision(mol, Chem)
            if decision:
                return decision.name
    if (len(rings) == 1 and len(ring) in (size, size - 1)
            and (canonical is None or '=' in canonical or '#' in canonical or _HALOGEN.search(canonical)
                 or 'O' in canonical or 'S' in canonical)):
        from .homogeneous_hydrides import simple_carbocycle

        return simple_carbocycle(mol, Chem, ring)
    return ''


def _shared_component(mol, Chem, canonical=None):
    """Try bounded graph scopes with complete shared constructions."""
    size = mol.GetNumHeavyAtoms()
    rings = mol.GetRingInfo().AtomRings()
    if (canonical is None or ('B' in canonical or '[' in canonical)
            and (_GROUP13.search(canonical) or rings and 'c' in canonical and _GROUP14.search(canonical))):
        from .homogeneous_hydrides import organyl_hydride_parent

        name = organyl_hydride_parent(mol, Chem)
        if name:
            return name
    if (not rings and (2 <= size <= 3 or canonical is not None and 'C' not in canonical)
            and (canonical is None or '=' in canonical or '#' in canonical)):
        from .homogeneous_hydrides import monounsaturated_chain

        name = monounsaturated_chain(mol)
        if name:
            return name
    if not rings and 2 <= size <= 8 and (canonical is None or _HALOGEN.search(canonical)):
        from .homogeneous_hydrides import halogenated_short_parent

        name = halogenated_short_parent(mol)
        if name:
            return name
    if not rings and size >= 4 and (canonical is None or '[' in canonical):
        # Screen for a substituted heavier group-14 chain. The shared method
        # checks that the nuclei form one path and accounts for every ligand.
        spelling = canonical if canonical is not None else Chem.MolToSmiles(mol)
        if _HALOGEN.search(spelling) and _GROUP14_PAIR.search(spelling):
            from .rules45_97_chains_locants import homogeneous_parent

            decision = homogeneous_parent(mol, Chem)
            if decision:
                return decision.name
    if rings:
        name = _shared_ring_name(mol, Chem, rings, size, canonical)
        if name:
            return name
    # Every remaining component contains nitrogen or oxygen. In particular,
    # other hydrocarbons can go directly to the existing fast writer.
    if canonical is not None and not _N_OR_O.search(canonical):
        return ''
    if (len(rings) <= 2 and (canonical is None or canonical.count('O') >= 2)
            and len(mol.GetSubstructMatches(_alcohol_site_query(Chem), maxMatches=2)) == 2):
        if len(rings) == 1 and canonical and 'c' in canonical:
            from .retained_hydrocarbon_derivatives import benzene_polyol

            decision = benzene_polyol(mol, Chem, rings[0])
            if decision:
                return decision.name
        if size >= 5 and (canonical is None or 'S' in canonical or _EXTENDED_ELEMENTS.search(canonical)
                          or len(rings) == 2):
            from .multiplicative_parent_extensions import linked_alcohol_decision

            decision = linked_alcohol_decision(mol, Chem)
            if decision:
                return decision.name
        if not rings:
            from .multiplicative_parent_extensions import straight_polyol

            decision = straight_polyol(mol, Chem, allow_branches=True)
            if decision:
                return decision.name
    (sulfur, oxazine, acid, formamidine, carbon_acid, phosphorus,
     azido_acyl, phosphoramidate, polycyclic_azidomethyl, ether) = _component_queries(Chem)
    if rings and size <= 32 and mol.HasSubstructMatch(ring_alkyl_aminium_query(Chem)):
        component = ring_alkyl_aminium(mol, Chem)
        if component is not None:
            return component.name
    if (size >= 30 and len(rings) >= 5 and max(map(len, rings)) >= 12
            and mol.HasSubstructMatch(acid)):
        from .extended_inventory import selection_inventory
        from .mapped_ring_acid_general import mapped_bridged_acid_decision

        inventory = selection_inventory(mol)
        if (inventory.supported and len(inventory.principal) == 1
                and inventory.principal[0].kind == 'acid'):
            decision = mapped_bridged_acid_decision(
                mol, Chem, inventory.principal[0], frozenset(range(mol.GetNumAtoms())))
            if decision is not None:
                return decision.name
    bounded_acid = size <= 32 and len(rings) <= 3
    acyclic_replacement = not rings and 10 <= size <= 128
    if (bounded_acid or acyclic_replacement) and mol.HasSubstructMatch(carbon_acid):
        if (acyclic_replacement
                and len({match[1] for match in mol.GetSubstructMatches(ether, maxMatches=4)}) == 4):
            from .extended_inventory import selection_inventory
            from .naming_methods import _replacement_parent

            inventory = selection_inventory(mol)
            groups = inventory.principal
            if inventory.supported and 1 <= len(groups) <= 2 and all(g.kind == 'acid' for g in groups):
                proof = _replacement_parent(mol, Chem, groups, evidence=True)
                if proof:
                    return proof.decision.name
        if bounded_acid:
            name = principal_acid_name(mol, Chem, carbon_acid)
            if name:
                return name
    if size <= 32 and len(rings) <= 3:
        name = bounded_ester_name(mol, Chem)
        if name:
            return name
    if size <= 48 and mol.HasSubstructMatch(_amidoxime_query(Chem)):
        decision = amidoxime_decision(mol, Chem)
        if decision is not None:
            return decision.name
        if mol.HasSubstructMatch(phosphorus):
            from .oxoacid_ester_methods import oxoacid_ester_decision

            decision = oxoacid_ester_decision(mol, Chem, amidoxime_arm_only=True)
            if decision is not None:
                return decision.name
    if (2 <= len(rings) <= 4 and size <= 40 and mol.HasSubstructMatch(azido_acyl)
            or len(rings) >= 5 and size <= 64 and mol.HasSubstructMatch(polycyclic_azidomethyl)):
        from .mapped_ring_construction import azido_fused_construction

        mapped = azido_fused_construction(mol, Chem)
        if mapped is not None:
            return mapped.name
    if 2 <= len(rings) <= 4 and size <= 54 and mol.HasSubstructMatch(phosphoramidate):
        from .oxoacid_ester_methods import oxoacid_ester_decision

        decision = oxoacid_ester_decision(mol, Chem)
        if decision is not None:
            return decision.name
    if size > 40:
        return ''
    if mol.HasSubstructMatch(sulfur):
        from .extended_inventory import selection_inventory
        from .nitrogen_acid_derivatives import sulfonimidamide_decision

        inventory = selection_inventory(mol)
        decision = (None if _unproved_fast_graph_scope(mol) else
                    sulfonimidamide_decision(mol, Chem, inventory,
                                            branch_resolver=_fast_component_branch))
        if decision is None and _unproved_fast_graph_scope(mol):
            decision = sulfonimidamide_decision(mol, Chem, inventory)
        if decision is None and rings and any(atom.GetChiralTag() for atom in mol.GetAtoms()):
            decision = sulfonimidamide_decision(mol, Chem, inventory)
        if decision is not None:
            return decision.name
    if mol.HasSubstructMatch(formamidine):
        from ._fast_engine.parent_rules import _formamidine_parent_names

        names = _formamidine_parent_names(mol, Chem, systematic=True)
        if len(names) == 1:
            return names[0]
    if (len(rings) == 2 and size <= 20 and len(set(rings[0]).intersection(rings[1])) == 2
            and mol.HasSubstructMatch(oxazine)):
        from .fusion_state_skeletons import benzo_heterocycle_skeletons
        from .naming_context import MoleculeContext
        from .ring_state_parent import select_ring_state

        selected = select_ring_state(MoleculeContext(mol, Chem),
                                     skeleton_source=benzo_heterocycle_skeletons)
        if selected is not None:
            return selected.selected.name
    return ''


def _bounded_fused_pair(mol, Chem, name):
    if name.count("cyclo[") != 2 or mol.GetNumHeavyAtoms() > 32:
        return ''
    if not name.endswith(('ene', 'en')) or "spiro" in name:
        return ''
    from .mapped_ring_parents import heteropolycyclic_hydride, ring_systems

    cores = tuple(core for core in ring_systems(mol) if len(core) >= 9)
    if len(cores) != 2 or not any(mol.GetAtomWithIdx(i).GetAtomicNum() in {7, 8, 16}
                                  for core in cores for i in core):
        return ''
    if not any(n.GetIdx() in cores[1] for i in cores[0]
               for n in mol.GetAtomWithIdx(i).GetNeighbors()):
        return ''
    record = heteropolycyclic_hydride(mol, Chem)
    if record is not None and record.method in _FUSED_HYDRIDE_METHODS:
        return record.name
    return ''


def _bounded_retained_aza_arene(mol, Chem, name):
    """Join a small, singly attached retained ring in a fast-engine bridge name."""
    if 'azabicyclo[4.4.0]' not in name or 'pentaen-' not in name:
        return name
    from .mapped_ring_parents import ring_systems
    from .polycyclic_rules import retained_aza_arene_substituent
    from ._fast_engine.polycyclic_rules import polycyclic_substituent_prefix as fast_prefix

    replacements = []
    for core in ring_systems(mol):
        if len(core) != 10:
            continue
        for root in core:
            atom = mol.GetAtomWithIdx(root)
            if atom.GetAtomicNum() != 6 or not atom.GetIsAromatic():
                continue
            for host in atom.GetNeighbors():
                blocked = host.GetIdx()
                if blocked in core or not host.IsInRing() or host.GetIsAromatic():
                    continue
                old = fast_prefix(mol, Chem, root, blocked)
                if not old or f'({old})' not in name:
                    continue
                new = retained_aza_arene_substituent(mol, Chem, root, blocked)
                if new:
                    replacements.append((f'({old})', f'({new})'))
    if len(replacements) == 1 and name.count(replacements[0][0]) == 1:
        old, new = replacements[0]
        return name.replace(old, new, 1)
    return name


def _full_after_failure(smiles, *, structured):
    """Explicit last resort: only called after fast mode has no usable name."""
    from . import api
    from ._smiles_dispatch import _fast_handoff_keywords
    settings = _fast_handoff_keywords()
    if not structured:
        return api.smiles_to_iupac(smiles, fallback_to_fast=False, **settings)
    result = api.smiles_to_iupac_result(smiles, fallback_to_fast=False, **settings)
    if not result.name:
        return result
    return NamingResult(result.name, result.generation_status, 'unresolved',
                        (_LIMITATION, *result.unresolved),
                        'fast:full-after-failure:' + result.route)


def _unproved_fast_graph_scope(mol):
    """Decline fast-engine routes with reproduced charge/H or bridge-map loss.

    These are structural eligibility checks, not molecule/name exceptions.
    Shared bounded producers run first; the full API is the normal handoff.
    A timeout's fast-only attempt must instead report unavailable.
    """
    for atom in mol.GetAtoms():
        number = atom.GetAtomicNum()
        if atom.GetFormalCharge() < 0:
            if number in {5, 6, 15, 16}:
                return True
            if number == 8 and any(n.GetAtomicNum() in {5, 15} and n.GetFormalCharge() == 0
                                   for n in atom.GetNeighbors()):
                return True
            if number == 7 and any(n.GetAtomicNum() == 7 and n.GetFormalCharge() > 0
                                   and n.IsInRing() for n in atom.GetNeighbors()):
                return True
        if number == 16 and atom.GetTotalValence() > 2:
            if atom.GetTotalNumHs() or (not atom.IsInRing() and any(
                    b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(atom).GetAtomicNum() == 7
                    for b in atom.GetBonds())):
                return True
    rings = tuple(map(frozenset, mol.GetRingInfo().AtomRings()))

    def elements(core):
        return [mol.GetAtomWithIdx(i).GetAtomicNum() for i in core]

    def carbonyl_count(core):
        return sum(bond.GetBondTypeAsDouble() == 2
                   and bond.GetOtherAtomIdx(i) not in core
                   and bond.GetOtherAtom(mol.GetAtomWithIdx(i)).GetAtomicNum() == 8
                   for i in core if mol.GetAtomWithIdx(i).GetAtomicNum() == 6
                   for bond in mol.GetAtomWithIdx(i).GetBonds())

    for ring in rings:
        numbers = elements(ring)
        # The fast-engine thiadiazolone map loses the N/S correspondence. Other
        # isolated lactams and annular P/S oxides retain their own writers.
        if (len(ring) == 5 and numbers.count(7) == 2 and numbers.count(16) == 1
                and carbonyl_count(ring)):
            return True
    for index, first in enumerate(rings):
        for second in rings[index + 1:]:
            shared = first & second
            core = first | second
            numbers = elements(core)
            # Reproduced nine-atom diaza bridge mappings can reverse the
            # bridge lengths. Spiro and other bridge families are separate.
            if len(shared) >= 3 and len(core) == 9 and numbers.count(7) == 2:
                return True
            if len(shared) == 2 and min(len(first), len(second)) == 4:
                small = first if len(first) == 4 else second
                if not any(mol.GetAtomWithIdx(i).GetChiralTag() for i in small):
                    return True
            if (len(shared) != 2 or set(numbers) - {6, 7, 8}
                    or any(other not in (first, second) and len(other & core) >= 2
                           for other in rings)):
                continue
            # Limit the hydrogen-state handoff to the reproduced two-face
            # carbonyl families. Fully unsaturated monoaza oxindole units,
            # polyoxo parents and larger fused systems have distinct maps.
            saturated_carbon = any(mol.GetAtomWithIdx(i).GetAtomicNum() == 6
                                   and mol.GetAtomWithIdx(i).GetHybridization()
                                   == _public.Chem.HybridizationType.SP3 for i in core)
            if carbonyl_count(core) == 1 and (
                    numbers.count(7) >= 2
                    or (numbers.count(7) == 1 and saturated_carbon)
                    or (numbers.count(7) == 0 and numbers.count(8) == 1
                        and min(len(first), len(second)) == 6)):
                return 'fused-carbonyl'
            if (numbers.count(7) == numbers.count(8) == 1
                    and any(mol.GetAtomWithIdx(i).GetAtomicNum() == 7
                            and not mol.GetAtomWithIdx(i).GetIsAromatic() for i in core)):
                return True
    geometric = [bond for bond in mol.GetBonds() if bond.GetStereo()
                 and (bond.GetBeginAtom().IsInRing() or bond.GetEndAtom().IsInRing())]
    if geometric:
        from rdkit.Chem import rdCIPLabeler
        copy = _public.Chem.Mol(mol)
        try:
            rdCIPLabeler.AssignCIPLabels(copy, maxRecursiveIterations=1000000)
        except (RuntimeError, ValueError):
            return True
        for bond in geometric:
            endpoints = {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()}
            # One endocyclic alkene in a carbon ring has a separate legacy
            # orientation problem. Macrocyclic heteroatom parents are outside
            # this boundary and retain their established explicit bond maps.
            if any(endpoints <= ring and set(elements(ring)) == {6}
                   and sum(b.GetBondTypeAsDouble() == 2 for b in mol.GetBonds()
                           if {b.GetBeginAtomIdx(), b.GetEndAtomIdx()} <= ring) == 1
                   for ring in rings):
                return True
            modern = copy.GetBondWithIdx(bond.GetIdx())
            legacy = {_public.Chem.BondStereo.STEREOE: 'E',
                      _public.Chem.BondStereo.STEREOZ: 'Z'}.get(bond.GetStereo())
            if legacy and modern.HasProp('_CIPCode') and modern.GetProp('_CIPCode') != legacy:
                return True
            for atom in (bond.GetBeginAtom(), bond.GetEndAtom()):
                if atom.GetAtomicNum() == 7 and any(n.GetAtomicNum() == 7 for n in atom.GetNeighbors()):
                    return True
                if atom.GetAtomicNum() != 6 or atom.IsInRing() or atom.GetDegree() != 3:
                    continue
                neighbors = [n for n in atom.GetNeighbors()
                             if n.GetIdx() != bond.GetOtherAtomIdx(atom.GetIdx())]
                cyano = any(any(b.GetBondTypeAsDouble() == 3 and b.GetOtherAtom(n).GetAtomicNum() == 7
                                for b in n.GetBonds()) for n in neighbors)
                carbonyl = any(any(b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(n).GetAtomicNum() == 8
                                   for b in n.GetBonds()) for n in neighbors)
                if cyano and carbonyl:
                    return True
    return False


def _fast_dispatch(smiles, *, structured, allow_full_fallback=True, _provisional_callback=None):
    Chem = _public.Chem
    if Chem is None:
        return NamingResult('', 'unavailable', 'not-assessed') if structured else ''
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        mol = None
    if mol is None or not mol.GetNumAtoms():
        return NamingResult('', 'invalid-input', 'not-assessed') if structured else ''
    return _fast_molecule(mol, smiles=smiles, structured=structured,
                          allow_full_fallback=allow_full_fallback,
                          _provisional_callback=_provisional_callback)


@placement_proof_scope()
def _fast_molecule(mol, *, smiles, structured, allow_full_fallback, _provisional_callback=None):
    Chem = _public.Chem

    def unavailable():
        if allow_full_fallback:
            return _full_after_failure(smiles, structured=structured)
        return (NamingResult('', 'unavailable', 'not-assessed', (), 'fast:no-full-handoff')
                if structured else '')

    if unsupported_relative_stereochemistry(mol, Chem):
        if structured:
            return NamingResult('', 'unavailable', 'unresolved', (_RELATIVE_LIMITATION,),
                                'unsupported-relative-stereochemistry')
        return ''
    # The public writer can silently discard isotope labels or radical
    # hydrogens. Decline those annotated graphs instead of emitting a name
    # for a different molecule. Inspect the parsed graph so byte-string and
    # CX inputs receive the same annotation protection as ordinary text.
    # The fast-engine peroxide writer can neutralize a peroxolate oxygen and
    # describe the counterion as an uncharged fragment. The full writer
    # retains this ionic construction; a fast-only timeout fallback must
    # decline it instead of delivering a different charge/H state.
    annotated_component = ''
    if any(atom.GetIsotope() or atom.GetNumRadicalElectrons()
           or (atom.GetAtomicNum() == 8 and atom.GetFormalCharge() == -1
               and any(neighbor.GetAtomicNum() == 8 for neighbor in atom.GetNeighbors()))
           for atom in mol.GetAtoms()):
        rings = mol.GetRingInfo().AtomRings()
        if rings and any(a.GetIsotope() for a in mol.GetAtoms()):
            try:
                annotated_component = _shared_ring_name(mol, Chem, rings, mol.GetNumHeavyAtoms(), None)
            except (ValueError, RuntimeError):
                return unavailable()
        elif not rings and 2 <= mol.GetNumAtoms() <= 8 and any(
                a.GetAtomicNum() in (9, 17, 35, 53) for a in mol.GetAtoms()):
            from .homogeneous_hydrides import halogenated_short_parent

            annotated_component = halogenated_short_parent(mol)
        if not annotated_component and any(a.GetAtomicNum() == 1 and a.GetIsotope() in (2, 3)
                                           for a in mol.GetAtoms()):
            from .oxygen_characteristic_suffixes import hydroxy_hydrogen_parent

            decision = hydroxy_hydrogen_parent(mol, Chem)
            if decision:
                annotated_component = decision.name
        if not annotated_component:
            return unavailable()
    canonical = None
    try:
        # Public 0.1.7 can depend on atom traversal. Normalize ordinary SMILES
        # so the writer sees a deterministic graph. Preserve CX metadata on
        # its original graph; canonical SMILES would discard that metadata.
        if not annotated_component and not mol.HasProp('_CXSMILES_Data') and not mol.GetStereoGroups():
            canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            if smiles is None or canonical != smiles:
                normalized = Chem.MolFromSmiles(canonical)
                if normalized is not None:
                    mol = normalized
    except Exception:
        return unavailable()
    oxide = mol.HasSubstructMatch(carbonyl_oxide_query(Chem))
    try:
        if annotated_component:
            component = annotated_component
        elif oxide:
            decision = carbonyl_oxide_decision(mol, Chem)
            component = decision.name if decision is not None else ''
        else:
            component = _shared_component(mol, Chem, canonical)
    except Exception:
        component = ''
    if component:
        component = normalize_enclosing_marks(simple_phenylalkyl_prefixes(complete_selected_name(component)))
        result = NamingResult(component, 'available', 'unresolved', (_LIMITATION,),
                              'fast:mapped-component')
        if _provisional_callback is not None:
            _provisional_callback(result)
        return result if structured else component
    if oxide:
        # The fast-engine carbonyl writer can omit the terminal charged oxygen.
        return unavailable()
    guard = _unproved_fast_graph_scope(mol)
    if guard == 'fused-carbonyl':
        try:
            component = bounded_fused_carbonyl_name(mol, Chem)
        except Exception:
            component = ''
        if component:
            component = normalize_enclosing_marks(simple_phenylalkyl_prefixes(complete_selected_name(component)))
            result = NamingResult(component, 'available', 'unresolved', (_LIMITATION,),
                                  'fast:mapped-component')
            if _provisional_callback is not None:
                _provisional_callback(result)
            return result if structured else component
    if guard:
        return unavailable()
    rescue = ''
    if mol.GetNumBonds() >= mol.GetNumAtoms() + 3:
        try:
            from .fused_arene_rescue import candidate_name
            rescue = candidate_name(mol, Chem)
        except Exception:
            # An unavailable structural route must not suppress fast naming.
            pass
    if rescue and _provisional_callback is not None:
        _provisional_callback(NamingResult(rescue, 'available', 'unresolved',
            (_LIMITATION, _FUSED_ARENE_LIMITATION), 'fast:structural-von-baeyer'))
    try:
        name = _public._engine().name(mol)
    except Exception:
        name = ''
    # Preserve established retained/fused names. A generic bridge name on this
    # narrow graph scope is instead built from the graph and checked against
    # its own numbered topology; fast-engine bridge candidates can have a reversed
    # main bridge and describe a different connectivity.
    prefix = name.split('cyclo[', 1)[0] if 'cyclo[' in name else ''
    use_rescue = bool(rescue and (not name or prefix.isalpha()))
    if use_rescue:
        name = rescue
    if not name:
        return unavailable()
    fused = _bounded_fused_pair(mol, Chem, name)
    mapped = bool(fused)
    if fused:
        name = fused
    retained = _bounded_retained_aza_arene(mol, Chem, name)
    if retained != name:
        name = retained
        mapped = True
    if "spirocyclo" in name and "oxo" in name and "sulfonyl" in name:
        from .spiro_component_rules import sulfonyl_monospiro_ketone

        decision = sulfonyl_monospiro_ketone(mol, Chem)
        if decision is not None:
            name = decision.name
            mapped = True
    if "spiro[" in name and "thia" in name and "tetraoxo" in name:
        from .spiro_component_rules import sultam_spiro_dione

        decision = sultam_spiro_dione(mol, Chem)
        if decision is not None:
            name = decision.name
            mapped = True
    repaired = repaired_parent_name(mol, Chem, name)
    if repaired:
        name = repaired
        mapped = True
    name = complete_selected_name(name)
    name = locanted_fused_component_presentation(mol, name)
    name = normalize_enclosing_marks(simple_phenylalkyl_prefixes(numbered_biphenyl_ligand(name)))
    if structured:
        limitations = (_LIMITATION, _FUSED_ARENE_LIMITATION) if use_rescue else (_LIMITATION,)
        route = ('fast:structural-von-baeyer' if use_rescue else
                 'fast:mapped-component' if mapped else 'fast:public-0.1.7')
        return NamingResult(name, 'available', 'unresolved', limitations, route)
    return name


def _fast_mol_only_result(mol, *, _provisional_callback=None):
    """Try the independent fast writer without re-entering full mode."""
    Chem = _public.Chem
    if Chem is None or mol is None:
        return NamingResult('', 'unavailable', 'not-assessed')
    try:
        copied = Chem.Mol(mol)
        if not copied.GetNumAtoms():
            return NamingResult('', 'invalid-input', 'not-assessed')
        # Canonical SMILES cannot represent query predicates or substance
        # groups. Do not turn such caller-supplied graphs into ordinary
        # molecules merely because the comprehensive search timed out.
        if (any(atom.HasQuery() for atom in copied.GetAtoms())
                or any(bond.HasQuery() for bond in copied.GetBonds())
                or Chem.GetMolSubstanceGroups(copied)):
            return NamingResult('', 'unavailable', 'not-assessed', (), 'fast:unsupported-mol')
    except Exception:
        return NamingResult('', 'invalid-input', 'not-assessed')
    return _fast_molecule(copied, smiles=None, structured=True, allow_full_fallback=False,
                          _provisional_callback=_provisional_callback)


def smiles_to_iupac_fast_result(smiles):
    """Return a general name and explicit preferred-name uncertainty."""
    return _fast_dispatch(smiles, structured=True)


def smiles_to_iupac_fast(smiles):
    """Return a fast general name, or an empty string if unavailable."""
    return _fast_dispatch(smiles, structured=False)
