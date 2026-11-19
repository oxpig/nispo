"""Mapped direct cyclic assemblies and their principal suffixes.

P-28 fixes the assembly junctions before P-14/P-45 selects among equivalent
functionalized numberings. A missing writer retains that original obligation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from collections import Counter
from itertools import product
import re

from .molecule_iteration import iter_atoms, iter_bonds
from .component_mapping import MappedComponent, decorate_component, fully_covered, hydrogen_bindings
from .extended_inventory import selection_inventory
from .formatting import multiplied_suffix, prefix_alphabetization_key
from .locant_order import writing_locant_key


_RULE = 'P-28.1/P-28.2.1/P-28.3/P-44.1/P-44.2.1/P-45/P-52.2.7/P-16.5.2.1'
_DOUBLE_RULE = 'P-28.1/P-28.2.1/P-28.2.2/P-29.3.3/P-44.1/P-45/P-16.5.1.1/P-16.5.2.1'
_LATIN = {2:'bi',3:'ter',4:'quater',5:'quinque',6:'sexi'}
_MAX_NUMBERINGS = 8192


@dataclass(frozen=True, slots=True)
class FunctionalAssemblyProof:
    components: tuple[frozenset[int], ...]
    path: tuple[int, ...]
    connectors: tuple[tuple[int, int], ...]
    parent_atoms: frozenset[int]
    principal_centers: frozenset[int]
    principal_kind: str
    maximum_disjoint_copies: int | None = 1
    topology_issue: str = ''
    rule: str = _RULE
    junction_orders: tuple[int, ...] = ()
    isotope_context: object | None = None


@dataclass(frozen=True, slots=True)
class FunctionalAssemblyCandidate:
    name: str
    parent_atoms: frozenset[int]
    principal_centers: frozenset[int]
    atom_locants: tuple
    suffix_locants: tuple[str, ...]
    junction_locants: tuple[tuple[int, int], ...]
    component_roles: tuple
    construction: MappedComponent
    chemical_key: tuple
    method: str = 'functional-ring-assembly'
    rule: str = _RULE
    junction_orders: tuple[int, ...] = ()
    stereo_key: tuple = ()
    isotope_key: tuple = ()

    @property
    def atoms(self):
        return self.construction.atoms | self.construction.embedded_atoms


@dataclass(frozen=True, slots=True)
class FunctionalAssemblyOutcome:
    status: str  # complete, pending, outside-scope
    reason: str
    decision: FunctionalAssemblyCandidate | None = None
    proof: FunctionalAssemblyProof | None = None
    candidates: tuple[FunctionalAssemblyCandidate, ...] = ()
    rule: str = _RULE


def _outside(reason):
    return FunctionalAssemblyOutcome('outside-scope', reason)


def _suffix_site(group):
    return group.center if group.kind == 'alcohol' else group.owner


def _scope(mol, Chem):
    bare = _bare_double_scope(mol, Chem)
    if bare is not None:
        return bare
    inventory = selection_inventory(mol)
    groups = tuple(inventory.principal)
    if (not inventory.supported or not groups or groups[0].kind not in {'acid','amide','nitrile','alcohol'}
            or any(g.kind != groups[0].kind for g in groups)
            or any(a.GetNumRadicalElectrons() for a in iter_atoms(mol))):
        return None
    from .neutral_charge_components import validate_neutral_charge_role_proofs

    if not validate_neutral_charge_role_proofs(mol, inventory.charge_role_proofs):
        return None
    source = inventory.source_frontier
    if (source.status not in {'selected','tie'}
            or set(groups) != set(source.groups)
            or len(inventory.groups) != len(inventory.source_group_facts)):
        return None
    from .complete_functional_parent import _ring_site_source_class_proved
    facts = dict(zip(inventory.groups,inventory.source_group_facts))
    if groups[0].kind == 'alcohol':
        from .oxygen_characteristic_suffixes import oxygen_group
        if (any(a.GetIsotope() or int(a.GetChiralTag()) for a in iter_atoms(mol))
                or any(int(b.GetStereo()) for b in iter_bonds(mol))
                or not all(oxygen_group(mol, g) is not None
                           and getattr(facts.get(g), 'class_rank', None) == 17 for g in groups)):
            return None
    elif not all(_ring_site_source_class_proved(mol,Chem,g,facts.get(g),g.kind) for g in groups):
        return None
    from .mapped_ring_parents import ring_systems, _core_graph
    components = ring_systems(mol)
    if len(components) < 2:
        return None
    ring_for_atom = {a:i for i,component in enumerate(components) for a in component}
    owner_components={ring_for_atom.get(_suffix_site(g)) for g in groups}
    if None in owner_components or any(g.kind != 'alcohol' and g.center in ring_for_atom for g in groups):
        return None
    identities = []
    for component in components:
        capped = _core_graph(mol,Chem,component)
        identities.append(None if capped is None else Chem.MolToSmiles(capped[0],isomericSmiles=False))
    adjacency = {i:[] for i in range(len(components))}
    for bond in iter_bonds(mol):
        a,b=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
        i,j=ring_for_atom.get(a),ring_for_atom.get(b)
        if (i is None or j is None or i==j or identities[i] is None
                or identities[i]!=identities[j] or bond.GetBondTypeAsDouble() not in {1,2}):
            continue
        adjacency[i].append((j,a,b));adjacency[j].append((i,b,a))
    first=next(iter(owner_components))
    connected,stack=set(),[first]
    while stack:
        i=stack.pop()
        if i in connected:continue
        connected.add(i);stack.extend(j for j,_a,_b in adjacency[i])
    if len(connected)<2 or not owner_components<=connected:
        return None
    if any(len(adjacency[i]) not in {1,2} for i in connected):
        selected=tuple(components[i] for i in sorted(connected))
        proof=FunctionalAssemblyProof(selected,(),(),frozenset().union(*selected),
            frozenset(g.center for g in groups),groups[0].kind,None,
            'P-28.6 needs a principal unbranched assembly and complete branch comparison.')
        return inventory,groups,proof,False
    ends=[i for i in connected if len(adjacency[i])==1]
    if len(ends)!=2:
        return None
    path,previous,current=[],None,ends[0]
    connectors=[]
    while True:
        if current in path:return None
        path.append(current)
        following=[row for row in adjacency[current] if row[0]!=previous]
        if not following:break
        if len(following)!=1:return None
        nxt,a,b=following[0];connectors.append((a,b))
        previous,current=current,nxt
    if set(path)!=connected:return None
    selected_components=tuple(components[i] for i in path)
    core=frozenset().union(*selected_components)
    proof=FunctionalAssemblyProof(selected_components,tuple(range(len(path))),tuple(connectors),
        core,frozenset(g.center for g in groups),groups[0].kind)
    orders=tuple(int(mol.GetBondBetweenAtoms(a,b).GetBondTypeAsDouble()) for a,b in connectors)
    proof=replace(proof,junction_orders=orders)
    if any(order!=1 for order in orders):
        if len(selected_components)==2 and orders==(2,):
            proof=replace(proof,rule=_DOUBLE_RULE)
        else:
            proof=replace(proof,maximum_disjoint_copies=None,rule=_DOUBLE_RULE,
                topology_issue='P-28.2.2 requires another method (P-31) for three or more cyclic systems with double junctions.')
    # The source gate is geometric, before a bounded parent writer. Extra
    # cyclic systems plus indirect links may form a larger phane parent.
    possible_phane=(len(components)>=4 and len(connected)!=len(components))
    return inventory,groups,proof,possible_phane


def _bare_double_scope(mol,Chem):
    """Recognize complete cyclic hydrides before any suffix inventory.

    Internal ring heteroatoms are part of their parent hydride. No external
    characteristic group or prefix is swallowed by this hydride-only proof.
    Larger identical assemblies retain the explicit P-28.2.2 method boundary.
    """
    junctions=tuple(b for b in iter_bonds(mol) if not b.IsInRing()
        and b.GetBeginAtom().IsInRing() and b.GetEndAtom().IsInRing())
    if not any(b.GetBondTypeAsDouble()==2 for b in junctions):return None
    if any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in iter_atoms(mol)):return None
    from .mapped_ring_parents import ring_systems,_core_graph
    components=ring_systems(mol)
    if len(components)<2:return None
    core=frozenset().union(*components)
    if core!=frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum()>1):return None
    identities=[]
    for component in components:
        capped=_core_graph(mol,Chem,component)
        if capped is None:return None
        identities.append(Chem.MolToSmiles(capped[0],isomericSmiles=False))
    if len(set(identities))!=1 or len(junctions)!=len(components)-1:return None
    owners={a:i for i,component in enumerate(components) for a in component}
    adjacency={i:set() for i in range(len(components))}
    for bond in junctions:
        if bond.GetBondTypeAsDouble() not in {1,2}:return None
        i,j=owners[bond.GetBeginAtomIdx()],owners[bond.GetEndAtomIdx()]
        if i==j or j in adjacency[i]:return None
        adjacency[i].add(j);adjacency[j].add(i)
    seen,stack=set(),[0]
    while stack:
        i=stack.pop()
        if i in seen:continue
        seen.add(i);stack.extend(adjacency[i])
    if len(seen)!=len(components):return None
    connectors=tuple((b.GetBeginAtomIdx(),b.GetEndAtomIdx()) for b in junctions)
    orders=tuple(int(b.GetBondTypeAsDouble()) for b in junctions)
    if len(components)==2:
        a,b=connectors[0]
        if a not in components[0]:connectors=((b,a),)
        proof=FunctionalAssemblyProof(components,(0,1),connectors,core,frozenset(),'',
            rule=_DOUBLE_RULE,junction_orders=orders)
    else:
        proof=FunctionalAssemblyProof(components,(),connectors,core,frozenset(),'',None,
            'P-28.2.2 requires another method (P-31) for three or more cyclic systems with double junctions.',
            _DOUBLE_RULE,orders)
    return None,(),proof,False


def _unit_numberings(mol,Chem,core,attachments,*,double_junction=False):
    from .mapped_ring_parents import assembly_unit_skeletons,_core_graph
    rings=tuple(r for r in mol.GetRingInfo().AtomRings() if set(r)&set(core))
    simple=len(rings)==1 and frozenset(rings[0])==core
    if not simple and not double_junction:return None
    capped=_core_graph(mol,Chem,core)
    if capped is None:return None
    bare=capped[0]
    saturated=all(b.GetBondTypeAsDouble()==1 for b in bare.GetBonds())
    aromatic=all(a.GetIsAromatic() for a in bare.GetAtoms())
    if double_junction and not saturated:return None
    if (not saturated and not aromatic or any(a.GetIsAromatic() and a.GetAtomicNum()==7
            and a.GetTotalNumHs(includeNeighbors=True) for a in bare.GetAtoms())):
        return None
    rows=tuple(assembly_unit_skeletons(mol,Chem,core))
    if not rows:return None
    fixed=min(row[6] for row in rows)
    rows=tuple(row for row in rows if row[6]==fixed)
    parents={row[0] for row in rows}
    if len(parents)!=1:return None
    # The owning ring route supplies source-preferred simple hydrides and
    # all their equivalent numberings. No assembly-specific atom order wins.
    parent=next(iter(parents))
    options=[]
    for row in rows:
        if any(not str(loc).isdigit() for loc in row[2]):return None
        mapping={atom:int(loc) for atom,loc in zip(row[1],row[2])}
        key=tuple(sorted(mapping[a] for a in attachments))
        options.append((key,tuple(row[1]),mapping))
    best=min(key for key,_seq,_mapping in options)
    unique={seq:(seq,mapping) for key,seq,mapping in options if key==best}
    return parent,tuple(unique.values())


def _numberings(mol,Chem,proof):
    attachments={a:[] for a in range(len(proof.components))}
    for i,(a,b) in enumerate(proof.connectors):
        attachments[i].append(a);attachments[i+1].append(b)
    units=tuple(_unit_numberings(mol,Chem,core,attachments[i],
                    double_junction=proof.junction_orders==(2,))
                for i,core in enumerate(proof.components))
    if any(unit is None for unit in units):return None,'Assembly unit numbering/indicated-H grammar is unavailable.'
    if len({unit[0] for unit in units})!=1:return None,'Identical units lack one source-preferred parent hydride.'
    parent=units[0][0]
    count=len(units)
    product_size=1
    for unit in units:product_size*=len(unit[1])
    if product_size*2>_MAX_NUMBERINGS:return None,'Complete assembly-numbering frontier exceeds its explicit resource bound.'
    numbered=[]
    for order in (tuple(range(count)),tuple(reversed(range(count)))):
        for rows in product(*(units[i][1] for i in order)):
            local_maps=[row[1] for row in rows]
            junctions=[];all_junctions=[]
            mapping={};roles=[]
            for occurrence,(index,(seq,local)) in enumerate(zip(order,rows),1):
                written={atom:(str(loc)+("'" if occurrence==2 else '') if count==2
                               else f'{occurrence}^{loc}') for atom,loc in local.items()}
                mapping.update(written)
                roles.append((f'ring-{occurrence}',proof.components[index],tuple(written.items())))
            for i in range(count-1):
                left,right=proof.components[order[i]],proof.components[order[i+1]]
                edges=[(b.GetBeginAtomIdx(),b.GetEndAtomIdx()) for b in iter_bonds(mol)
                       if b.GetBeginAtomIdx() in left and b.GetEndAtomIdx() in right]
                edges += [(b.GetEndAtomIdx(),b.GetBeginAtomIdx()) for b in iter_bonds(mol)
                          if b.GetEndAtomIdx() in left and b.GetBeginAtomIdx() in right]
                if len(edges)!=1:return None,'The numbered assembly has no unique original junction.'
                a,b=edges[0]
                pair=(local_maps[i][a],local_maps[i+1][b]);junctions.append(pair)
                all_junctions.extend((writing_locant_key(mapping[a]),writing_locant_key(mapping[b])))
            fixed_key=(tuple(sorted(all_junctions)),tuple(junctions))
            numbered.append((fixed_key,parent,mapping,tuple(junctions),tuple(roles)))
    best=min(row[0] for row in numbered)
    return tuple(row for row in numbered if row[0]==best),''


def _parent_name(parent,count,junctions,*,junction_order=1):
    if junction_order==2:
        if count!=2 or not parent.endswith('e'):return ''
        # P-29.3.3 contracts saturated carbon monocycles; other cyclic
        # hydrides retain their full stem, including von Baeyer descriptors.
        unit=(parent[:-3]+'ylidene' if re.fullmatch(r'cyclo[a-z]+ane',parent)
              else parent[:-1]+'ylidene')
        # P-16.5.1.1 encloses a multiplied unit consisting of a parent
        # hydride and a substituent suffix, including a divalent prefix.
        term='bi('+unit+')'
    elif parent=='benzene':
        term=_LATIN[count]+'phenyl'
    else:
        # bi(cyclo...) avoids a von Baeyer ambiguity. Numeric hydride
        # components retain their own enclosing parentheses at any count.
        enclosed=(count==2 and parent.startswith('cyclo') or bool(re.search(r'\d',parent))
                  or 'hydro' in parent)
        term=_LATIN[count]+('('+parent+')' if enclosed else parent)
    if count==2:
        return f"{junctions[0][0]},{junctions[0][1]}'-{term}"
    return ':'.join(f'{i+1}^({a}),{i+2}^({b})'
                    for i,(a,b) in enumerate(junctions))+'-'+term


def _written_suffix_locant(locant):
    # Keep typed comparison/atom maps unchanged; serialize the two levels
    # of a P-28.3 locant with one explicit superscript convention.
    match = re.fullmatch(r'([1-9]\d*)\^([1-9]\d*)', locant)
    return f'{match[1]}^({match[2]})' if match else locant


def _construction_parent(groups,proof,numbering,suffix_locs,*,mol=None):
    _fixed,parent,_mapping,junctions,_roles=numbering
    hydride=_parent_name(parent,len(proof.components),junctions,
        junction_order=2 if proof.junction_orders==(2,) else 1)
    if not hydride:return ''
    if not groups:return hydride
    suffix={'acid':'carboxylic acid','amide':'carboxamide','nitrile':'carbonitrile','alcohol':'ol'}[proof.principal_kind]
    suffix=multiplied_suffix(len(groups),suffix)
    if proof.principal_kind == 'alcohol' and len(groups) == 1:
        if hydride.endswith('e'):
            hydride = hydride[:-1]
        elif hydride.endswith('e)'):
            hydride = hydride[:-2] + ')'
    if proof.isotope_context is not None:
        from .assembly_isotopes import suffix_isotopes
        if mol is None:return ''
        typed=suffix_isotopes(mol,proof.isotope_context)
        suffix=typed.suffix if typed is not None else ''
    return '['+hydride+']-'+','.join(map(_written_suffix_locant,suffix_locs))+'-'+suffix if suffix else ''


def _chemical_numbering_key(suffix_locs,prefixes,*,annotated=False):
    # P-14.4(f)/(g) must not use a rendered child's R/S as an earlier
    # chemical citation tie. The final name is only a formatter fallback.
    prefix_order=lambda text:(prefix_alphabetization_key(text)[:3] if annotated
                              else prefix_alphabetization_key(text))
    return (tuple(writing_locant_key(loc) for loc in suffix_locs),-len(prefixes),
        tuple(sorted(writing_locant_key(loc) for loc,_text in prefixes)),
        tuple(writing_locant_key(loc) for loc,text in sorted(prefixes,
            key=lambda item:(prefix_order(item[1]),writing_locant_key(item[0])))))


def _construct(mol,Chem,inventory,groups,proof,numbering):
    from .naming_context import MoleculeContext
    from .ring_state_parent import _build_ring_preparation,_ring_render_branches
    _fixed,parent,mapping,junctions,roles=numbering
    mapping=dict(mapping)
    isotope_context=proof.isotope_context
    if isotope_context is not None:
        from .assembly_isotopes import assembly_isotope_preparation
        prepared=assembly_isotope_preparation(mol,Chem,inventory,isotope_context) if groups else None
    else:
        prepared=(_build_ring_preparation(mol,Chem,proof.parent_atoms,inventory.groups,groups,None,(),())
                  if groups else None)
    if groups and prepared is None:return None
    nmap={}
    if proof.principal_kind=='amide':
        from .amide_site_locants import external_amide_nitrogen_locants
        extra={}
        if isotope_context is not None:
            from .assembly_isotopes import assembly_locant_series
            extra['locant_series']=assembly_locant_series
        assigned=external_amide_nitrogen_locants(mol,Chem,groups,mapping,**extra)
        if assigned is None:return None
        nmap.update(assigned.nitrogen_locants)
    for target,host,prefix in (prepared.aliases if prepared is not None else ()):
        mapping[target]=prefix+mapping[host]
    mapping.update(nmap)
    needed=frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum()>1 or a.GetIsotope())
    if isotope_context is not None and groups:
        from .assembly_isotopes import assembly_isotope_branches
        branches=assembly_isotope_branches(mol,Chem,isotope_context,tuple(sorted(proof.parent_atoms)),
            nmap,mapping,needed,prepared)
    else:
        branches=(_ring_render_branches(MoleculeContext(mol,Chem),tuple(sorted(proof.parent_atoms)),
            nmap,mapping,proof.parent_atoms,needed,None,prepared) if groups else
            ((),frozenset(),(),hydrogen_bindings(mol,mapping)))
    if branches is None:return None
    prefixes,embedded,ownership,bindings=branches
    suffix_locs=tuple(sorted((mapping[_suffix_site(g)] for g in groups),key=writing_locant_key))
    ending=_construction_parent(groups,proof,numbering,suffix_locs,mol=mol)
    if not ending:return None
    construction=MappedComponent(ending,tuple(bindings.items()),frozenset(bindings),tuple(prefixes),
        frozenset(embedded),systematic_parent=True,nitrogen_site_locants=proof.principal_kind=='amide',
        prefix_branch_ownership=tuple(ownership))
    isotope_key=()
    if isotope_context is not None:
        from .assembly_isotopes import suffix_isotopes,build_assembly_isotope_evidence
        suffix=suffix_isotopes(mol,isotope_context)
        if suffix is None:return None
        construction=replace(construction,local_isotope_atoms=suffix.local_atoms,
                             amide_isotope_evidence=suffix.amide_evidence)
        evidence=build_assembly_isotope_evidence(mol,Chem,isotope_context,construction,
            {i:mapping[i] for i in proof.parent_atoms})
        if evidence is None:return None
        construction=replace(construction,assembly_isotope_evidence=evidence,
                             prefix_citation_order=evidence.prefix_citation_order)
        isotope_key=evidence.isotope_key
    from .assembly_stereochemistry import (has_assembly_annotations,
        build_assembly_stereo_evidence,assembly_stereo_numbering)
    annotated=has_assembly_annotations(mol)
    stereo_key=()
    if annotated:
        evidence=build_assembly_stereo_evidence(mol,Chem,proof.components,proof.parent_atoms,
            dict(construction.atom_locants),construction.atoms,proof.connectors,
            allow_isotopes=isotope_context is not None)
        if evidence is None:return None
        construction=replace(construction,assembly_stereo_evidence=evidence)
        stereo_key=assembly_stereo_numbering(mol,Chem,construction)
        if stereo_key is None:return None
    if (construction.atoms&construction.embedded_atoms
            or not fully_covered(mol,construction.atoms,construction.embedded_atoms)):
        return None
    name=decorate_component(mol,Chem,construction)
    if not name:return None
    key=(isotope_context.chemical_key if isotope_context is not None else
         _chemical_numbering_key(suffix_locs,prefixes,annotated=annotated))
    return FunctionalAssemblyCandidate(name,proof.parent_atoms,proof.principal_centers,
        tuple(bindings.items()),suffix_locs,junctions,roles,construction,key,
        method='functional-ring-assembly' if groups else 'ring-assembly',rule=proof.rule,
        junction_orders=proof.junction_orders,stereo_key=stereo_key,isotope_key=isotope_key)


def _valid_candidate(mol,Chem,groups,proof,numbering,candidate):
    """Require the actual writer to retain original roles and numbering."""
    if not isinstance(candidate,FunctionalAssemblyCandidate):return False
    _fixed,_parent,original,junctions,roles=numbering
    mapping=dict(candidate.atom_locants)
    construction=candidate.construction
    if (candidate.parent_atoms!=proof.parent_atoms
            or candidate.principal_centers!=proof.principal_centers
            or candidate.junction_locants!=junctions or candidate.component_roles!=roles
            or candidate.junction_orders!=proof.junction_orders
            or proof.junction_orders!=tuple(int(mol.GetBondBetweenAtoms(a,b).GetBondTypeAsDouble())
                                           for a,b in proof.connectors)
            or len(mapping)!=len(candidate.atom_locants)
            or any(mapping.get(i)!=original[i] for i in proof.parent_atoms)
            or dict(construction.atom_locants)!=mapping
            or construction.atoms&construction.embedded_atoms
            or not proof.parent_atoms<=construction.atoms
            or not fully_covered(mol,construction.atoms,construction.embedded_atoms)):
        return False
    suffixes=tuple(sorted((original[_suffix_site(g)] for g in groups),key=writing_locant_key))
    if candidate.suffix_locants!=suffixes:return False
    if construction.parent!=_construction_parent(groups,proof,numbering,suffixes,mol=mol):return False
    if proof.principal_kind=='amide':
        from .amide_site_locants import external_amide_nitrogen_locants
        extra={}
        if proof.isotope_context is not None:
            from .assembly_isotopes import assembly_locant_series
            extra['locant_series']=assembly_locant_series
        assignment=external_amide_nitrogen_locants(mol,Chem,groups,original,**extra)
        if assignment is None or any(mapping.get(n)!=loc for n,loc in assignment.nitrogen_locants):
            return False
    from .branch_rules import collect_subtree
    prefixes=[];owned=set()
    for root,host,atoms,text in construction.prefix_branch_ownership:
        if (root not in atoms or host not in mapping or atoms&construction.atoms
                or atoms&owned or frozenset(collect_subtree(mol,root,host))!=atoms):
            return False
        owned.update(atoms);prefixes.append((mapping[host],text))
    if Counter(prefixes)!=Counter(construction.prefixes):return False
    suffix_atoms={i for g in groups for i in g.atoms if i not in construction.atoms}
    if owned|suffix_atoms!=set(construction.embedded_atoms):return False
    isotope_context=proof.isotope_context
    if isotope_context is not None:
        from .assembly_isotopes import validate_assembly_isotope_evidence
        if (set(isotope_context.selected_groups)!=set(groups)
                or not validate_assembly_isotope_evidence(mol,Chem,construction)
                or construction.assembly_isotope_evidence.context!=isotope_context
                or candidate.isotope_key!=construction.assembly_isotope_evidence.isotope_key):return False
    elif construction.assembly_isotope_evidence is not None or candidate.isotope_key:return False
    from .assembly_stereochemistry import (has_assembly_annotations,
        build_assembly_stereo_evidence,assembly_stereo_numbering)
    annotated=has_assembly_annotations(mol)
    if annotated:
        expected=build_assembly_stereo_evidence(mol,Chem,proof.components,proof.parent_atoms,
            mapping,construction.atoms,proof.connectors,allow_isotopes=isotope_context is not None)
        if expected is None or construction.assembly_stereo_evidence!=expected:return False
        stereo_key=assembly_stereo_numbering(mol,Chem,construction)
        if stereo_key is None or candidate.stereo_key!=stereo_key:return False
    elif construction.assembly_stereo_evidence is not None or candidate.stereo_key:return False
    expected_key=(isotope_context.chemical_key if isotope_context is not None else
                  _chemical_numbering_key(suffixes,prefixes,annotated=annotated))
    if candidate.chemical_key!=expected_key:return False
    return decorate_component(mol,Chem,construction)==candidate.name


def functional_assembly_outcome(mol,Chem):
    """Return completed or explicit pending source evidence for this family."""
    scope=_scope(mol,Chem)
    if scope is None:return _outside('No supported full-principal direct assembly was proved.')
    inventory,groups,proof,possible_phane=scope
    pending=lambda reason,rows=():FunctionalAssemblyOutcome('pending',reason,None,proof,tuple(rows),proof.rule)
    if proof.topology_issue:
        if proof.topology_issue.startswith('P-28.6'):
            from .tree_ring_assemblies import functional_tree_assembly_outcome
            return functional_tree_assembly_outcome(mol,Chem,groups,proof)
        return pending(proof.topology_issue)
    if len(proof.components)>6:
        return pending('P-28.5/P-52.2.7.3 requires the phane method for seven or more components.')
    if possible_phane:
        return pending('Additional cyclic systems leave a larger P-52 phane-parent comparison open.')
    from .assembly_stereochemistry import has_assembly_annotations,assembly_stereo_census
    annotated=has_assembly_annotations(mol)
    isotopes=any(a.GetIsotope() for a in iter_atoms(mol))
    if isotopes:
        from .assembly_isotopes import prepare_assembly_isotopes
        context=prepare_assembly_isotopes(mol,Chem,inventory,groups,proof)
        if context.issue:return pending(context.issue)
        groups=context.selected_groups
        proof=replace(proof,principal_centers=frozenset(g.center for g in groups),
            isotope_context=context,rule=proof.rule+'/P-82.2/P-82.3/P-82.4/P-82.5/P-82.6')
    if annotated:
        census=assembly_stereo_census(mol,Chem,allow_isotopes=isotopes)
        if census.issue:return pending(census.issue)
        proof=replace(proof,rule=proof.rule+'/P-14.4(j)/P-93.5.7.2/P-93.5.1.4.2/P-16.5.4.1.3')
    numberings,reason=_numberings(mol,Chem,proof)
    if numberings is None:return pending(reason)
    if isotopes:
        allowed=set(proof.isotope_context.ordinary_maps)
        numberings=tuple(row for row in numberings if tuple(sorted(row[2].items())) in allowed)
        if not numberings:return pending('No original numbering retains the complete ordinary chemical frontier.')
    rows=[];missing=0
    for numbering in numberings:
        candidate=_construct(mol,Chem,inventory,groups,proof,numbering)
        if candidate is None or not _valid_candidate(mol,Chem,groups,proof,numbering,candidate):missing+=1
        else:rows.append(candidate)
    if missing:return pending('An assembly-junction maximum lacks complete mapped suffix/prefix construction.',rows)
    if not rows:return pending('No complete assembly construction is available.')
    key=min(row.chemical_key for row in rows)
    frontier=tuple(row for row in rows if row.chemical_key==key)
    if isotopes:
        key=min(row.isotope_key for row in frontier)
        frontier=tuple(row for row in frontier if row.isotope_key==key)
    if annotated:
        locations=min(row.stereo_key[0] for row in frontier)
        frontier=tuple(row for row in frontier if row.stereo_key[0]==locations)
        if len({row.stereo_key[1] for row in frontier})!=1:
            return pending('Equivalent chemical numberings need another stereodescriptor-family comparison.',frontier)
        key=min(row.stereo_key[2] for row in frontier)
        frontier=tuple(row for row in frontier if row.stereo_key[2]==key)
    if len({row.name for row in frontier})!=1:
        return pending('Equivalent assembly numberings leave a preferred-prefix choice unresolved.',frontier)
    return FunctionalAssemblyOutcome('complete','Complete original assembly, fixed junctions and functional numbering.',
        frontier[0],proof,tuple(rows),proof.rule)
