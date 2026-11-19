"""P-62 ordinary amine sites and original-graph N-substituent ownership.

Recognition, method admission, suffix attachment and prefix construction are
separate. Nitrogen is never a carbon-chain continuation through two N bonds.
"""

from .molecule_iteration import iter_atoms
from .molecule_cache import mol_cache_dict, mol_cache_get
from itertools import permutations, product

RULES=('P-62.2.1.2/P-62.2.2.1','P-62.2.2.2','P-62.2.3','P-62.2.4.1.2',
       'P-62.2.6.1','P-16.9-N-sites','P-62.2.4.1.3')


def carbon_hosts(mol,group,parent=None):
    return tuple(n.GetIdx() for n in mol.GetAtomWithIdx(group.center).GetNeighbors()
                 if n.GetAtomicNum()==6 and (parent is None or n.GetIdx() in parent))


def ordinary_amine(mol,idx):
    n=mol.GetAtomWithIdx(idx)
    heavy=[a for a in n.GetNeighbors() if a.GetAtomicNum()>1]
    return (n.GetAtomicNum()==7 and not n.IsInRing() and not n.GetFormalCharge()
        and not n.GetNumRadicalElectrons() and not int(n.GetChiralTag())
        and n.GetTotalValence()==3 and 1<=len(heavy)<=3
        and all(a.GetAtomicNum()==6 for a in heavy)
        and all(b.GetBondTypeAsDouble()==1 for b in n.GetBonds())
        and not any(not b.GetIsAromatic() and b.GetBondTypeAsDouble()>=2 and b.GetOtherAtom(a).GetAtomicNum() not in {1,6}
                    for a in heavy for b in a.GetBonds()))


def principal_scope(mol,groups):
    """Ordinary amine trees, with explicit barriers for unproved methods."""
    if any(not ordinary_amine(mol,g.center) for g in groups):return False
    from .polyamine_methods import substitutive_scope
    return substitutive_scope(mol,groups)


def common_framework(mol,groups):
    hosts=[set(carbon_hosts(mol,g)) for g in groups]
    from .mapped_ring_parents import ring_systems
    frameworks=list(ring_systems(mol))
    remaining={a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum()==6}
    while remaining:
        component=set();stack=[next(iter(remaining))]
        while stack:
            i=stack.pop()
            if i not in remaining:continue
            remaining.remove(i);component.add(i)
            stack.extend(n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors()
                         if n.GetIdx() in remaining)
        frameworks.append(component)
    return any(all(h&framework for h in hosts) for framework in frameworks)


def suffix_sites(mol,groups,parent,mapping):
    return next(suffix_site_maps(mol,groups,parent,mapping),None)


def suffix_site_maps(mol,groups,parent,mapping):
    """P-16.9.3: enumerate N/N' at each geminal host independently.

    N'3 denotes a prime on N; N3' is reserved for a prime on the numerical
    superscript of a repeated parent. Neither representation loses atom IDs.
    The caller compares alternatives chemically, never by input atom order.
    """
    sites=[]
    by_site={}
    for group in groups:
        hosts=carbon_hosts(mol,group,parent)
        if len(hosts)!=1 or not ordinary_amine(mol,group.center):return
        sites.append(hosts[0])
        by_site.setdefault(hosts[0],[]).append(group.center)
    for assignments in product(*(permutations(ids) for ids in by_site.values())):
        nmap={}
        for (site,_),ids in zip(by_site.items(),assignments):
            for prime,idx in enumerate(ids):
                nmap[idx]='N'+"'"*prime+(mapping[site] if len(groups)>1 else '')
        yield tuple(sites),nmap


def unlocanted_nitrogen(mol,scope=None):
    atoms=iter_atoms(mol) if scope is None else (mol.GetAtomWithIdx(i) for i in scope)
    return (7,) if sum(a.GetAtomicNum()==7 for a in atoms)==1 else ()


def amine_ending(mol,Chem,parent,groups,locants,*,one_word_isotope=False):
    """Build an amine ending, optionally leaving N isotope roles for the name.

    P-82.2.5 places isotope descriptors before a one-word name. In that
    mode, uniformly modified principal amine N and N-bound isotope-H atoms remain
    mapped on the caller's complete component; they are not decorated inside
    the suffix. Unlike or incomplete N-isotope profiles remain outside this
    bounded operation.
    """
    from .component_mapping import MappedComponent,decorate_component,hydrogen_bindings
    from .formatting import attach_suffix,multiplied_suffix,enclose_name_unit
    from .constants import MULT
    isotope_centers = tuple(
        g.center for g in groups
        if mol.GetAtomWithIdx(g.center).GetIsotope()
        or any(n.GetAtomicNum() == 1 and n.GetIsotope()
               for n in mol.GetAtomWithIdx(g.center).GetNeighbors())
    )
    if one_word_isotope and isotope_centers:
        nitrogen_count = sum(a.GetAtomicNum() == 7 for a in iter_atoms(mol))
        profiles={(
            mol.GetAtomWithIdx(g.center).GetIsotope(),
            tuple(sorted(n.GetIsotope() for n in mol.GetAtomWithIdx(g.center).GetNeighbors()
                         if n.GetAtomicNum()==1 and n.GetIsotope())),
        ) for g in groups}
        if (len(isotope_centers) != len(groups) or nitrogen_count != len(groups)
                or len(profiles) != 1
                or len(groups) > 1 and any(
                    len([n for n in mol.GetAtomWithIdx(g.center).GetNeighbors()
                         if n.GetAtomicNum() > 1]) != 1 for g in groups)):
            return '', frozenset()
        suffix = multiplied_suffix(len(groups), 'amine')
        text = attach_suffix(parent, suffix, locants)
        return text, frozenset()
    units=[];owned=set()
    for g in groups:
        bindings=hydrogen_bindings(mol,{g.center:''});owned.update(bindings)
        units.append(decorate_component(mol,Chem,MappedComponent('amine',tuple(bindings.items()),
            frozenset(bindings),isotope_unlocanted_elements=(7,))))
    if len(set(units))!=1 or not units[0]:return '',frozenset()
    suffix=multiplied_suffix(len(groups),'amine')
    text=attach_suffix(parent,suffix,locants)
    if units[0]!='amine':
        complete=units[0] if len(groups)==1 else MULT[len(groups)].removesuffix('a')+enclose_name_unit(units[0])
        text=text[:-len(suffix)]+complete
    return text,frozenset(owned)


def _anilino_ring_witness(mol, Chem, root, atoms, candidates, text):
    """Carry only original-ring ownership proved by every aniline finalist."""
    from .branch_rules import collect_subtree
    from .component_mapping import mapped_branch, mapped_branch_ring_atoms

    finalists = [record for record in candidates
                 if record.parent == 'benzene' and record.name.endswith('aniline')
                 and record.name[:-len('aniline')] + 'anilino' == text]
    if not finalists:
        return frozenset()
    witnesses = []
    for record in finalists:
        component = record.construction
        parent = frozenset(record.parent_atoms)
        if (component is None or not record.annotation_coverage
                or not record.hydrogen_budget_proved
                or record.principal_centers != {root}
                or not parent <= component.atoms
                or component.atoms & component.embedded_atoms
                or component.atoms | component.embedded_atoms != atoms
                or len(component.prefix_branch_ownership) != len(component.prefixes)):
            return frozenset()
        owned = set()
        rings = {i for i in parent if mol.GetAtomWithIdx(i).IsInRing()}
        for child, anchor, branch, branch_text in component.prefix_branch_ownership:
            branch = frozenset(branch)
            boundary = {(i, n.GetIdx()) for i in branch
                        for n in mol.GetAtomWithIdx(i).GetNeighbors()
                        if n.GetAtomicNum() > 1 and n.GetIdx() not in branch}
            if (anchor not in component.atoms or child not in branch
                    or branch & owned or boundary != {(child, anchor)}
                    or frozenset(collect_subtree(mol, child, anchor)) != branch
                    or mapped_branch(mol, Chem, child, anchor) != (branch_text, branch)):
                return frozenset()
            owned.update(branch)
            rings.update(mapped_branch_ring_atoms(
                mol, Chem, child, anchor, branch_text, branch))
        if owned != set(component.embedded_atoms):
            return frozenset()
        witnesses.append(frozenset(rings))
    return frozenset.intersection(*witnesses)


def amino_prefix(mol,Chem,root,blocked,atoms):
    """One amino component, rooted on N, including its isotope H and branches."""
    from .component_mapping import (MappedComponent, decorate_component, hydrogen_bindings,
        mapped_branch, mapped_branch_ring_atoms, _record_branch_ring_atoms)
    if not ordinary_amine(mol,root) or mol.GetAtomWithIdx(blocked).GetAtomicNum()!=6:return '',frozenset()
    key=(root,blocked,frozenset(atoms));cache=mol_cache_dict(mol,'mapped_amino_prefix')
    if key in cache:return cache[key]
    # Mark in-progress recursion conservatively. Every successful recursive
    # branch loses the cut edge, so an ordinary acyclic amino tree terminates.
    cache[key]=('',frozenset())
    roots=[n.GetIdx() for n in mol.GetAtomWithIdx(root).GetNeighbors()
           if n.GetAtomicNum()>1 and n.GetIdx()!=blocked]
    # P-62.2.1.1.1/P-62.2.3: anilino is a preferred complete component.
    from .mapped_ring_parents import records,ring_systems,parent_facts
    from .functional_inventory import CharacteristicGroup
    aryl=[];other_heteroring=False
    for core in ring_systems(mol):
        if not core.intersection(roots):continue
        if len(core)==6 and all(mol.GetAtomWithIdx(i).GetAtomicNum()==6 and mol.GetAtomWithIdx(i).GetIsAromatic() for i in core):
            aryl.append(core)
        else:other_heteroring=True
    if aryl and not other_heteroring:
        group=CharacteristicGroup('amine',root,(root,));candidates=[]
        for core in aryl:
            candidates.extend(records(mol,Chem,scope=atoms,core_filter=core,principal=(group,)))
        from .parent_selection import parent_frontier
        candidates=parent_frontier(candidates,lambda r:parent_facts(mol,r,group.seniority))
        names={r.name[:-len('aniline')]+'anilino' for r in candidates
               if r.parent=='benzene' and r.name.endswith('aniline')}
        if len(names)==1:
            name = names.pop()
            scope = frozenset(atoms)
            rings = _anilino_ring_witness(mol, Chem, root, scope, candidates, name)
            cache[key]=(name,scope)
            _record_branch_ring_atoms(mol, Chem, root, blocked, name, scope, rings)
            return cache[key]
        return cache[key]
    terms=[];embedded=set();natures={};typed=True;proved_ring_atoms=set()
    for neighbor in roots:
        text,owned=mapped_branch(mol,Chem,neighbor,root)
        if not text or owned&embedded:return cache[key]
        terms.append(text);embedded.update(owned)
        proved_ring_atoms.update(mapped_branch_ring_atoms(
            mol,Chem,neighbor,root,text,owned))
        from .organyl_prefix_components import (organyl_prefix_nature,
                                                multiplied_organyl_prefix)
        from .component_vocabulary import SUBSTITUENT_VOCABULARY
        nature=organyl_prefix_nature(mol,neighbor,owned,text)
        if (nature is None
                and multiplied_organyl_prefix(mol,neighbor,owned,text,1) == text):
            # The original carbon-rooted ligand can own neutral acyclic
            # N/O/S modifications as one complete compound component.
            # Preserve that graph-proved nature in the shared multiplier:
            # two methoxymethyl arms are bis(methoxymethyl), while the
            # concatenated dimethoxymethyl denotes a different carbon graph.
            nature='compound'
        retained=SUBSTITUENT_VOCABULARY.get(text)
        if (nature is None or text in natures and natures[text]!=nature
                or retained is not None and retained.source_status=='preferred-retained'
                and nature=='compound'):
            typed=False
        else:natures[text]=nature
    mapping=hydrogen_bindings(mol,{root:'N'})
    if set(mapping)|embedded!=set(atoms):return cache[key]
    prefixes=tuple(('',text) for text in terms)
    # Carry the original organyl topology to the multiplier. In particular,
    # two cycloalkylmethyl arms are bis(cycloalkylmethyl), not one
    # dicycloalkylmethyl arm. Other component grammars keep their own writer.
    if typed and natures:
        from collections import Counter
        from .prefix_sequences import PrefixUnit,mononuclear_prefix_sequence
        typed=mononuclear_prefix_sequence(
            PrefixUnit(text,natures[text],count) for text,count in Counter(terms).items()) is not None
    else:typed=False
    text=decorate_component(mol,Chem,MappedComponent('amino',tuple(mapping.items()),frozenset(mapping),
        prefixes,frozenset(embedded),isotope_unlocanted_elements=(7,),prefix_mode='amino',
        prefix_natures=tuple(sorted(natures.items())) if typed else None))
    cache[key]=(text,frozenset(atoms)) if text else ('',frozenset())
    if text:
        _record_branch_ring_atoms(mol,Chem,root,blocked,text,atoms,proved_ring_atoms)
    return cache[key]


def chain_proposals(mol,Chem,inventory,*,scope=None,principal=None,site_frontier=None,defer=False):
    from .multifunctional_parent import _paths
    if inventory.principal[0].kind!='amine':return
    groups=inventory.principal if principal is None else principal
    carbon={a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum()==6 and not a.IsInRing() and (scope is None or a.GetIdx() in scope)}
    if site_frontier is not None:
        # The shared P-44 frontier has already retained all globally maximal
        # oriented paths. Keep the writer on those exact original atom paths;
        # attached branches remain prefixes whose complete ownership is
        # checked below. A failed/incomplete frontier cannot be replaced by a
        # writer-local path ranking.
        if (getattr(site_frontier,'status',None)!='proved'
                or not getattr(site_frontier,'candidate_universe_complete',False)):
            return
        candidates=tuple(getattr(site_frontier,'candidates',()))
        for candidate in candidates:
            seq=tuple(candidate.parent_atoms)
            if not seq or not set(seq)<=carbon:continue
            covered=tuple(site.source_group for site in candidate.selected_sites)
            if not covered or any(g not in groups for g in covered):continue
            mapping={i:str(j) for j,i in enumerate(seq,1)}
            for sites,nmap in suffix_site_maps(mol,covered,seq,mapping):
                yield from _defer_chain_proposal(
                    mol,Chem,groups,covered,seq,dict(mapping),sites,nmap,scope,
                    site_frontier=site_frontier,candidate=candidate,defer=defer,
                )
        return
    for path in _paths(mol,carbon):
        for seq in (path,tuple(reversed(path))):
            covered=tuple(g for g in groups if carbon_hosts(mol,g,seq))
            if not covered:continue
            mapping={i:str(j) for j,i in enumerate(seq,1)}
            for sites,nmap in suffix_site_maps(mol,covered,seq,mapping):
                yield from _defer_chain_proposal(
                    mol,Chem,groups,covered,seq,dict(mapping),sites,nmap,scope,defer=defer)


def _defer_chain_proposal(mol,Chem,groups,covered,seq,mapping,sites,nmap,scope,*,
                         site_frontier=None,candidate=None,defer=False):
    if not defer:
        yield from _chain_proposal(
            mol,Chem,groups,covered,seq,mapping,sites,nmap,scope,
            site_frontier=site_frontier,candidate=candidate)
        return
    from .proposal_plans import ParentProposalPlan, graph_proposal_bound

    def construct():
        return tuple(_chain_proposal(
            mol,Chem,groups,covered,seq,dict(mapping),sites,nmap,scope,
            site_frontier=site_frontier,candidate=candidate))

    yield ParentProposalPlan(
        'mapped-chain-amine', tuple(seq), frozenset(g.center for g in covered),
        graph_proposal_bound(mol, seq, covered), construct,
        key=('amine', tuple(seq), tuple(sorted(nmap.items())),
             None if scope is None else frozenset(scope), tuple(
                 (g.kind, g.center, g.owner, tuple(g.atoms)) for g in covered)),
        single_proposal=(mol_cache_get(mol, 'candidate_telemetry') is None
                         and _chain_proposal is _DEFAULT_CHAIN_PROPOSAL),
    )


def _chain_proposal(mol,Chem,groups,covered,seq,mapping,sites,nmap,scope,*,site_frontier=None,candidate=None):
    from .cross_route_selection import ParentProposal,_annotation_ownership
    from .component_mapping import (
        MappedComponent, decorate_component, hydrogen_bindings, mapped_branch, mapped_general_branch,
    )
    from .formatting import complete_parent_name
    from .branch_rules import parent_chain_name
    from .parent_selection import graph_parent_facts
    sites_with_azido = mol_cache_get(mol, 'nested_azido_sites')
    resolver = mapped_general_branch if sites_with_azido else mapped_branch
    ownership = [] if sites_with_azido else None
    mapping.update(nmap);prefixes=[];embedded=set();failed=False
    for idx in tuple(mapping):
        for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
            root=neighbor.GetIdx()
            if root in mapping or neighbor.GetAtomicNum()<=1 or (scope is not None and root not in scope):continue
            # A cut C=C/C#C needs an ylidene/ylidyne component,
            # rather than the single-free-valence prefix below.
            order=mol.GetBondBetweenAtoms(idx,root).GetBondTypeAsDouble()
            if order!=1 and not(order==2 and neighbor.GetAtomicNum()==8):
                failed=True;break
            text,owned=resolver(mol,Chem,root,idx)
            if not text or owned&(mapping.keys()|embedded) or any(c.isspace() for c in text):
                failed=True;break
            prefixes.append((mapping[idx],text));embedded.update(owned)
            if ownership is not None:
                ownership.append((root, idx, owned, text))
        if failed:break
    mapping=hydrogen_bindings(mol,mapping)
    needed={a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum()>1 or a.GetIsotope() or int(a.GetChiralTag())}
    if failed or set(mapping)|embedded != (needed if scope is None else needed & set(scope)) or not _annotation_ownership(mol,mapping):return
    if (any(mol.GetAtomWithIdx(i).GetAtomicNum() == 16
            and mol.GetAtomWithIdx(i).GetIsAromatic() for i in embedded)
            and any(mol.GetAtomWithIdx(g.center).GetIsotope()
                    or any(n.GetAtomicNum() == 1 and n.GetIsotope()
                           for n in mol.GetAtomWithIdx(g.center).GetNeighbors())
                    for g in covered)):
        # The existing one-word N-isotope construction is not independently
        # parseable with this aromatic sulfur branch. Keep it out of newly
        # admitted proposals until that annotation grammar is established.
        return
    parent=complete_parent_name(parent_chain_name(mol,seq))
    if not parent:return
    locants=tuple(sorted((mapping[i] for i in sites),key=int))
    carbon_prefixes=any(loc.isdigit() for loc,text in prefixes)
    carbon_isotopes={mol.GetAtomWithIdx(i).GetIsotope() for i in seq}
    carbon_labels=len(carbon_isotopes)>1 and any(carbon_isotopes)
    # P-14.3.4.2(a): the suffix cannot occupy a second skeletal position
    # when the mapped carbon parent has a single atom. Keep carbon-prefix
    # locants intact, since they may distinguish substitution of the carbon
    # from substitution of the amine nitrogen. For two-carbon parents the
    # existing narrower monosubstitution rule still applies.
    omit=(len(sites)==1 and (len(seq)==1 or
          len(seq)==2 and not carbon_prefixes and not carbon_labels))
    ending,suffix_owned=amine_ending(
        mol,Chem,parent,covered,() if omit else locants,
        one_word_isotope=True,
    )
    if not ending:return
    if any(mol.GetAtomWithIdx(i).GetIsotope() for i in embedded):
        # An embedded isotope has its own local descriptor. The one-word
        # parent-front rule for simultaneous embedded and principal-N labels
        # is not established by this component contract.
        if any(mol.GetAtomWithIdx(g.center).GetIsotope()
               or any(n.GetAtomicNum() == 1 and n.GetIsotope()
                      for n in mol.GetAtomWithIdx(g.center).GetNeighbors())
               for g in covered):
            return
    if any(mol.GetAtomWithIdx(g.center).GetIsotope()
           or any(n.GetAtomicNum() == 1 and n.GetIsotope()
                  for n in mol.GetAtomWithIdx(g.center).GetNeighbors())
           for g in covered):
        from .formatting import format_prefixes,join_prefix_parent
        prefix_text = format_prefixes(prefixes, hierarchical_enclosures=True)
        complete_one_word = join_prefix_parent(prefix_text, ending)
        component = MappedComponent(
            complete_one_word, tuple(mapping.items()), frozenset(mapping),
            embedded_atoms=frozenset(embedded), systematic_parent=True,
            nitrogen_site_locants=True,
            isotope_unlocanted_elements=unlocanted_nitrogen(mol),
        )
    else:
        component = MappedComponent(ending,tuple(mapping.items()),frozenset(mapping),
            tuple(prefixes),frozenset(embedded),systematic_parent=True,nitrogen_site_locants=True,
            isotope_unlocanted_elements=unlocanted_nitrogen(mol),local_isotope_atoms=suffix_owned)
    name=decorate_component(mol,Chem,component)
    if not name or len(name)>260:return
    facts=graph_parent_facts(mol,seq,mapping,principal_class=groups[0].seniority,
        suffix_centers=[g.center for g in covered],suffix_locants=locants,prefixes=prefixes,
        kind='chain',name=name,allow_annotations=True,
        charge_role_proofs=tuple(sites_with_azido.values()) if sites_with_azido else (),
        prefix_branch_ownership=ownership or (), resolve_branch=resolver)
    if facts is not None:
        selected_sites=tuple(getattr(candidate,'selected_sites',()))
        omitted_sites=tuple(getattr(candidate,'omitted_sites',()))
        all_sites=tuple(getattr(site_frontier,'sites',())) if site_frontier is not None else ()
        site_keys=tuple((s.kind,s.principal_center,s.host_atom,s.group_atoms) for s in all_sites)
        yield ParentProposal(name,'mapped-chain-amine','amine',frozenset(g.center for g in covered),
            tuple(seq),tuple(mapping.items()),facts,True,
            construction=component,
            principal_site_keys=site_keys,
            selected_site_evidence=selected_sites,
            omitted_site_evidence=omitted_sites)


# A custom producer may return multiple rows; only the native generator has
# the zero-or-one proposal contract used for memo-alias cycle cardinality.
_DEFAULT_CHAIN_PROPOSAL = _chain_proposal
