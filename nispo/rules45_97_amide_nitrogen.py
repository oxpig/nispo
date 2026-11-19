"""Mapped nitrogen-functional parents and complete hydrazide prefixes.

The branch writer derives names from complete independently capped functional
parents. The whole-graph writers partition their atoms before selecting a
method; unknown annotations defer to the ordinary full-mode pipeline.
"""
from dataclasses import replace
from itertools import permutations

from .bluebook_lambda_amine_methods import _plain, _out, _sanitize
from .constants import MULT, ROOTS
from .fast_ring_scope import fast_carbon_ring_numbering_enabled
from .formatting import format_prefixes, enclose_name_unit, prefix_alphabetization_key
from .molecule_iteration import iter_atoms, iter_bonds


def _order(mol, a, b):
    bond=mol.GetBondBetweenAtoms(a,b)
    return bond.GetBondTypeAsDouble() if bond else 0


def _side(mol, start, blocked):
    result=set();todo=[start]
    while todo:
        i=todo.pop()
        if i==blocked or i in result:continue
        result.add(i)
        todo.extend(a.GetIdx() for a in mol.GetAtomWithIdx(i).GetNeighbors())
    return frozenset(result)


def _ordinary_scope(mol, atoms):
    return (all(mol.GetAtomWithIdx(i).GetAtomicNum() in {6,7,8,9,16,17,35,53}
                and not (mol.GetAtomWithIdx(i).GetIsotope() or mol.GetAtomWithIdx(i).GetFormalCharge()
                         or mol.GetAtomWithIdx(i).GetNumRadicalElectrons() or int(mol.GetAtomWithIdx(i).GetChiralTag()))
                for i in atoms)
            and all(not int(b.GetStereo()) and not int(b.GetBondDir()) for b in iter_bonds(mol)
                    if b.GetBeginAtomIdx() in atoms and b.GetEndAtomIdx() in atoms))


def _polyazanyl(mol,Chem,root,blocked,scope):
    ns={i for i in scope if mol.GetAtomWithIdx(i).GetAtomicNum()==7}
    if len(ns)<3 or len(ns) not in MULT or root not in ns:
        return None
    path=[];previous=blocked;current=root
    while current not in path:
        path.append(current)
        others=[a.GetIdx() for a in mol.GetAtomWithIdx(current).GetNeighbors() if a.GetIdx() in ns and a.GetIdx()!=previous]
        if not others:break
        if len(others)!=1 or _order(mol,current,others[0])!=1:return None
        previous,current=current,others[0]
    if set(path)!=ns or any(mol.GetAtomWithIdx(i).IsInRing() or mol.GetAtomWithIdx(i).GetTotalValence()!=3 for i in ns):
        return None
    from .component_mapping import mapped_branch
    from .phosphorus_ester_methods import _carbon_component
    prefixes=[];owned=set(ns)
    for j,i in enumerate(path,1):
        for a in mol.GetAtomWithIdx(i).GetNeighbors():
            if a.GetIdx() in ns or a.GetIdx()==blocked:continue
            group=_carbon_component(mol,a.GetIdx(),i)
            text,mapped=mapped_branch(mol,Chem,a.GetIdx(),i)
            if group is None or not text or group!=mapped or group&owned:return None
            prefixes.append((str(j),text));owned.update(group)
    if owned!=set(scope):return None
    return format_prefixes(prefixes,hierarchical_enclosures=True)+MULT[len(path)]+'azan-1-yl'


def _oxamoyl(mol,root,blocked,scope):
    n=mol.GetAtomWithIdx(root)
    if n.GetAtomicNum()!=7 or _order(mol,root,blocked) not in {1,2}:return None
    cs=[a for a in n.GetNeighbors() if a.GetIdx()!=blocked]
    if len(cs)!=1 or cs[0].GetAtomicNum()!=6 or _order(mol,root,cs[0].GetIdx())!=1:return None
    c=cs[0]
    ox=[a for a in c.GetNeighbors() if a.GetAtomicNum()==8 and a.GetDegree()==1 and _order(mol,c.GetIdx(),a.GetIdx())==2]
    nxt=[a for a in c.GetNeighbors() if a.GetAtomicNum()==6 and _order(mol,c.GetIdx(),a.GetIdx())==1]
    if len(ox)!=1 or len(nxt)!=1 or c.GetDegree()!=3:return None
    end=nxt[0]
    eo=[a for a in end.GetNeighbors() if a.GetAtomicNum()==8 and a.GetDegree()==1 and _order(mol,end.GetIdx(),a.GetIdx())==2]
    en=[a for a in end.GetNeighbors() if a.GetAtomicNum()==7 and a.GetDegree()==1 and a.GetTotalNumHs()==2 and _order(mol,end.GetIdx(),a.GetIdx())==1]
    if len(eo)!=1 or len(en)!=1 or end.GetDegree()!=3:return None
    if scope!={root,c.GetIdx(),ox[0].GetIdx(),end.GetIdx(),eo[0].GetIdx(),en[0].GetIdx()}:return None
    return 'oxamoylimino' if _order(mol,root,blocked)==2 else 'oxamoylamino'


def _clear_cut_geometry(probe,Chem,boundaries):
    """Clear only configurations invalidated by the artificial branch cut."""
    ends=frozenset().union(*boundaries)
    for bond in iter_bonds(probe):
        pair=frozenset((bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()))
        if pair in boundaries:
            bond.SetStereo(Chem.BondStereo.STEREONONE)
            if bond.HasProp('_CIPCode'):bond.ClearProp('_CIPCode')
        if pair & ends and bond.GetBondDir() in {
                Chem.BondDir.ENDUPRIGHT,Chem.BondDir.ENDDOWNRIGHT}:
            bond.SetBondDir(Chem.BondDir.NONE)


def _subordinate_probe(mol,Chem):
    """Prove supported C=N boundary geometry before making a private probe.

    The eligibility graph retains every original atom and bond. Actual
    descriptors are assigned on the unchanged source graph after restoring
    the complete preferred branch.
    """
    allowed={6,7,8,9,16,17,35,53}
    if _plain(mol,Chem,allowed,charges=(-1,0,1)):
        return mol,frozenset()
    boundaries=set()
    for bond in iter_bonds(mol):
        if bond.GetStereo()==Chem.BondStereo.STEREONONE:continue
        if bond.GetStereo() not in {Chem.BondStereo.STEREOE,Chem.BondStereo.STEREOZ,
                Chem.BondStereo.STEREOCIS,Chem.BondStereo.STEREOTRANS}:return None
        ends=[mol.GetAtomWithIdx(i) for i in (bond.GetBeginAtomIdx(),bond.GetEndAtomIdx())]
        if sorted(a.GetAtomicNum() for a in ends)!=[6,7] or bond.IsInRing():return None
        root=next(a.GetIdx() for a in ends if a.GetAtomicNum()==7)
        host=next(a.GetIdx() for a in ends if a.GetAtomicNum()==6)
        scope=_side(mol,root,host)
        if host in scope or _oxamoyl(mol,root,host,scope)!='oxamoylimino':return None
        boundaries.add(frozenset((root,host)))
    if len(boundaries)!=1:return None
    ends=frozenset().union(*boundaries)
    for bond in iter_bonds(mol):
        if bond.GetBondDir()==Chem.BondDir.NONE:continue
        if (bond.GetBondDir() not in {Chem.BondDir.ENDUPRIGHT,Chem.BondDir.ENDDOWNRIGHT}
                or bond.GetBondType()!=Chem.BondType.SINGLE
                or not {bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()} & ends):return None
    from .stereochemistry import assigned_bond_stereochemistry
    assigned=assigned_bond_stereochemistry(mol,Chem)
    if {frozenset((a,b)) for a,b,_ in assigned}!=boundaries:return None
    probe=Chem.Mol(mol)
    _clear_cut_geometry(probe,Chem,boundaries)
    if not _plain(probe,Chem,allowed,charges=(-1,0,1)):return None
    return probe,frozenset(boundaries)


def _name_hydrazido_fragment(fragment, Chem):
    from .api import _IN_DIRECT_NAMING, mol_to_iupac
    from .rule_engine import _HYDRAZIDO_FRAGMENT_NAMES

    names = _HYDRAZIDO_FRAGMENT_NAMES.get()
    if names is None or not _IN_DIRECT_NAMING.get():
        return mol_to_iupac(fragment)

    # Recursive fragment naming shares one outer delivery scope. Atom order,
    # source annotations and flags can all affect the delivered spelling.
    try:
        key = (fragment.ToBinary(Chem.PropertyPickleOptions.AllProps),
               fast_carbon_ring_numbering_enabled())
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return mol_to_iupac(fragment)
    cached = names.get(key)
    if cached is not None:
        return cached
    name = mol_to_iupac(fragment)
    # The API can also return an empty result after a construction error.
    if name and len(names) < 128:
        names[key] = name
    return name


def _hydrazido(mol,Chem,root,blocked,scope):
    n=mol.GetAtomWithIdx(root)
    if n.GetAtomicNum()!=7 or n.IsInRing() or n.GetTotalValence()!=3 or _order(mol,root,blocked)!=1:return None
    ns=[a for a in n.GetNeighbors() if a.GetIdx()!=blocked and a.GetAtomicNum()==7 and _order(mol,root,a.GetIdx())==1]
    if len(ns)!=1:return None
    proximal=ns[0]
    cs=[a for a in proximal.GetNeighbors() if a.GetIdx()!=root and a.GetAtomicNum()==6 and _order(mol,proximal.GetIdx(),a.GetIdx())==1]
    if len(cs)!=1 or proximal.IsInRing() or proximal.GetTotalValence()!=3:return None
    c=cs[0]
    double=[a for a in c.GetNeighbors() if a.GetAtomicNum() in {7,8} and a.GetDegree()==1
            and _order(mol,c.GetIdx(),a.GetIdx())==2]
    if len(double)!=1 or c.IsInRing():return None
    if double[0].GetAtomicNum()==7 and double[0].GetTotalNumHs()!=1:return None
    # Rooted polycarbonic hydrazides cite the other hydrazide through its
    # complete hydrazinecarbonyl group, rather than changing a dihydrazide
    # ending into an ambiguously divalent prefix.
    if n.GetDegree()==2 and proximal.GetDegree()==2:
        rest=[a for a in c.GetNeighbors() if a.GetIdx() not in {proximal.GetIdx(),double[0].GetIdx()}]
        if len(rest)==1 and rest[0].GetAtomicNum()==8 and rest[0].GetDegree()==2:
            other=[a for a in rest[0].GetNeighbors() if a.GetIdx()!=c.GetIdx()]
            if len(other)==1 and other[0].GetAtomicNum()==6:
                end=other[0]
                eo=[a for a in end.GetNeighbors() if a.GetAtomicNum()==8 and a.GetDegree()==1 and _order(mol,end.GetIdx(),a.GetIdx())==2]
                en=[a for a in end.GetNeighbors() if a.GetAtomicNum()==7 and a.GetDegree()==2 and _order(mol,end.GetIdx(),a.GetIdx())==1]
                if len(eo)==1 and len(en)==1:
                    terminal=[a for a in en[0].GetNeighbors() if a.GetIdx()!=end.GetIdx()]
                    if (len(terminal)==1 and terminal[0].GetAtomicNum()==7 and terminal[0].GetDegree()==1
                            and terminal[0].GetTotalNumHs()==2 and _order(mol,en[0].GetIdx(),terminal[0].GetIdx())==1
                            and scope=={root,proximal.GetIdx(),c.GetIdx(),double[0].GetIdx(),rest[0].GetIdx(),end.GetIdx(),eo[0].GetIdx(),en[0].GetIdx(),terminal[0].GetIdx()}):
                        tail='methanimidohydrazido' if double[0].GetAtomicNum()==7 else 'formohydrazido'
                        return '[(hydrazinecarbonyl)oxy]'+tail
    # Ordinary hydrazides and their imido analogues retain one complete
    # functional parent. Acyl C is not joined to another N-N hydrazide loop.
    from .naming_methods import _fragment
    fragment,original=_fragment(mol,Chem,scope)
    if fragment is None:return None
    imido=double[0].GetAtomicNum()==7
    complete = _name_hydrazido_fragment(fragment, Chem)
    if complete.endswith('hydrazide'):
        return complete[:-1]+'o'
    if imido:
        rw=Chem.RWMol(fragment);a=rw.GetAtomWithIdx(original.index(double[0].GetIdx()))
        a.SetAtomicNum(8);a.SetNumExplicitHs(0);a.SetNoImplicit(True)
        fragment=_sanitize(rw,Chem)
        if fragment is None:return None
        from .full_bluebook_followup import _imidohydrazide_from_hydrazide
        complete = _imidohydrazide_from_hydrazide(_name_hydrazido_fragment(fragment, Chem), 1)
    if not complete or not complete.endswith('hydrazide'):
        # Derive the complete hydrazide from its mapped acid cap if a
        # generic renderer has not exposed the functional parent suffix.
        if n.GetDegree()!=2 or proximal.GetDegree()!=2:return None
        if any(a.GetIdx()!=c.GetIdx() and a.GetAtomicNum()==6 and any(
                x.GetAtomicNum()==8 and x.GetTotalNumHs()==1 and _order(mol,a.GetIdx(),x.GetIdx())==1
                for x in a.GetNeighbors()) and any(x.GetAtomicNum()==8 and _order(mol,a.GetIdx(),x.GetIdx())==2
                for x in a.GetNeighbors()) for a in (mol.GetAtomWithIdx(i) for i in scope)):
            return None
        rw=Chem.RWMol(mol)
        cap=rw.GetAtomWithIdx(proximal.GetIdx());cap.SetAtomicNum(8);cap.SetNumExplicitHs(1);cap.SetNoImplicit(True)
        if imido:
            oxygen=rw.GetAtomWithIdx(double[0].GetIdx());oxygen.SetAtomicNum(8);oxygen.SetNumExplicitHs(0);oxygen.SetNoImplicit(True)
        remove=(set(range(mol.GetNumAtoms()))-set(scope))|{root}
        for i in sorted(remove,reverse=True):rw.RemoveAtom(i)
        acid=_sanitize(rw,Chem)
        if acid is None:return None
        ordinary = _name_hydrazido_fragment(acid, Chem)
        if imido:
            retained={'formic acid':'methanimidohydrazide','acetic acid':'ethanimidohydrazide','benzoic acid':'benzenecarboximidohydrazide'}
            if ordinary in retained:complete=retained[ordinary]
            elif ordinary.endswith('carboxylic acid'):complete=ordinary[:-len('carboxylic acid')]+'carboximidohydrazide'
            elif ordinary.endswith('oic acid'):complete=ordinary[:-len('oic acid')]+'imidohydrazide'
        else:
            from .hydrazide_components import _replace_acid_ending
            complete=_replace_acid_ending(ordinary)
    if not complete or not complete.endswith('hydrazide'):return None
    return complete[:-1]+'o'


def branch_prefix(mol,Chem,root,blocked):
    """Return a graph-complete preferred nitrogen prefix or decline."""
    atom=mol.GetAtomWithIdx(root)
    if atom.GetAtomicNum() not in {7,16}:return None
    if atom.GetAtomicNum()==16:
        if (atom.GetDegree()==1 and atom.GetTotalNumHs()==1 and _order(mol,root,blocked)==1
                and _ordinary_scope(mol,{root})):
            return 'sulfanyl',frozenset({root})
        return None
    if atom.IsInRing() or atom.GetFormalCharge() or atom.GetTotalValence()!=3:return None
    scope=_side(mol,root,blocked)
    if atom.GetAtomicNum()==7 and not atom.GetFormalCharge() and _order(mol,root,blocked)==1:
        if any(mol.GetAtomWithIdx(i).GetFormalCharge() for i in scope):
            from .naming_methods import _fragment
            fragment,original=_fragment(mol,Chem,scope)
            if fragment is not None:
                nitramide=isocyanato_nitramide(fragment,Chem)
                if nitramide is not None and nitramide.name.endswith('nitramide'):
                    return nitramide.name[:-1]+'o',scope
    if blocked in scope:return None
    boundary=mol.GetBondBetweenAtoms(root,blocked)
    if (not _ordinary_scope(mol,scope) or boundary is None
            or boundary.GetStereo()!=Chem.BondStereo.STEREONONE):
        # An internal directional single bond can encode geometry at the
        # external N=C bond; its descriptor belongs to the selected parent.
        eligibility=_subordinate_probe(mol,Chem)
        if eligibility is None or frozenset((root,blocked)) not in eligibility[1]:return None
        if not _ordinary_scope(eligibility[0],scope):return None
    text=_oxamoyl(mol,root,blocked,scope)
    if not text and _order(mol,root,blocked)==1:
        text=_polyazanyl(mol,Chem,root,blocked,scope)
    if not text:
        text=_hydrazido(mol,Chem,root,blocked,scope)
    return (text,scope) if text else None


def _rebase_component(mol,Chem,record,back,marked,prefixes,*,probe=None,boundaries=frozenset()):
    """Replace artificial F branches with their original complete owners."""
    from .component_mapping import decorate_component
    component=record.construction
    if component is None:return None
    mapping=tuple((back[i],loc) for i,loc in component.atom_locants)
    core=frozenset(back[i] for i in component.atoms)
    embedded=set(back[i] for i in component.embedded_atoms)
    ownership=[];compound=[]
    for root,host,atoms,text in component.prefix_branch_ownership:
        original=back[root]
        if original in marked:
            name,scope=marked[original]
            embedded.update(scope);atoms=scope;root=original;host=back[host];text=name
            if name not in {'sulfanyl','carbamimidoyl'} and not name.endswith('azan-1-yl'):
                if sum(mol.GetAtomWithIdx(i).GetAtomicNum()==7 for i in scope)>=3 or name in {'oxamoylimino','oxamoylamino'}:
                    compound.append(name)
        else:
            atoms=frozenset(back[i] for i in atoms);root=back[root];host=back[host]
        ownership.append((root,host,atoms,text))
    component=replace(component,atom_locants=mapping,atoms=core,prefixes=tuple(prefixes),
        embedded_atoms=frozenset(embedded),prefix_branch_ownership=tuple(ownership),
        compound_prefix_names=tuple(dict.fromkeys((*component.compound_prefix_names,*compound))),retained_acid_locant_evidence=None)
    if record.construction.retained_acid_locant_evidence is not None:
        # P-14.3.4.6: the two-carbon acid's alpha carbon is its only
        # substitutable site. The source-mapped original graph proves the
        # same omission after restoring the complete noncarbon branch.
        old=record.construction.retained_acid_locant_evidence
        if len(prefixes)!=1:return None
        loc,text=prefixes[0]
        if loc!=old.alpha_locant:return None
        from .retained_acid_rules import retained_acid_locant_evidence
        proof=retained_acid_locant_evidence(mol,Chem,parent_class=old.parent_class,
            principal_kind=old.principal_kind,parent_atoms=frozenset(back[i] for i in old.parent_atoms),
            atom_locants=mapping,prefixes=prefixes)
        if proof is None:return None
        prefix=enclose_name_unit(text) if text in compound else text
        name=prefix+component.parent
    else:
        name=decorate_component(probe if boundaries else mol,Chem,component)
    if not name:return None
    if boundaries:
        # P-93.4.2.1.2/.3 uses the carbon chain locant for this exocyclic
        # C=N. The functional prefix owns N, so the companion mapped writer
        # decorates that boundary on the unchanged complete source graph.
        hosts=dict(mapping)
        for pair in boundaries:
            if len(pair & core)!=1:return None
            anchor=next(iter(pair & core));other=next(iter(pair-{anchor}))
            if (mol.GetAtomWithIdx(anchor).GetAtomicNum()!=6 or other not in marked
                    or marked[other][0]!='oxamoylimino' or not hosts.get(anchor)):return None
        from .stereochemistry import apply_mapped_branch_bond_stereochemistry
        name=apply_mapped_branch_bond_stereochemistry(mol,Chem,name,mapping)
    if record.ester_owners:
        # The native record owns the complete organyl words, including
        # distributive locants for unlike ester owners. The branch change
        # affects only the separately recorded acid-component boundary.
        if not record.component_name or not record.name.endswith(record.component_name):return None
        owner=record.name[:-len(record.component_name)]
        if not owner or not owner.endswith(' '):return None
        name=owner+name
    return name


def subordinate_prefix_decision(mol,Chem):
    eligibility=_subordinate_probe(mol,Chem)
    if eligibility is None:return None
    probe,boundaries=eligibility
    # Root each potential cut explicitly. Only a carbon or amide nitrogen
    # host can use these prefixes in this whole-molecule construction.
    possibilities=[]
    for atom in iter_atoms(mol):
        if atom.GetAtomicNum() not in {7,16}:continue
        for host in atom.GetNeighbors():
            if host.GetAtomicNum() not in {6,7}:continue
            row=branch_prefix(probe,Chem,atom.GetIdx(),host.GetIdx())
            if row:possibilities.append((atom.GetIdx(),host.GetIdx(),*row))
    from .branch.carbamimidoyl import carbamimidoyl_prefix
    for atom in iter_atoms(mol):
        if atom.GetAtomicNum()!=6:continue
        for host in atom.GetNeighbors():
            if host.GetAtomicNum()!=7 or _order(mol,atom.GetIdx(),host.GetIdx())!=1:continue
            text=carbamimidoyl_prefix(mol,Chem,atom.GetIdx(),host.GetIdx())
            if text:possibilities.append((atom.GetIdx(),host.GetIdx(),text,_side(mol,atom.GetIdx(),host.GetIdx())))
    for root,host,text,scope in possibilities:
        # The branch cut must be a bridge, and no original annotation is
        # changed. F is used solely as a numbered proposal marker.
        if host in scope or not scope or len(scope)>=mol.GetNumAtoms():continue
        if boundaries and boundaries!={frozenset((root,host))}:continue
        if any(a.GetFormalCharge() for a in iter_atoms(mol) if a.GetIdx() not in scope):continue
        rw=Chem.RWMol(mol)
        if boundaries:_clear_cut_geometry(rw,Chem,boundaries)
        marker=rw.GetAtomWithIdx(root)
        marker.SetAtomicNum(9);marker.SetNumExplicitHs(0);marker.SetNoImplicit(True)
        old_order=_order(mol,root,host)
        rw.GetBondBetweenAtoms(root,host).SetBondType(Chem.BondType.SINGLE)
        if old_order==2 and rw.GetAtomWithIdx(host).GetNoImplicit():
            rw.GetAtomWithIdx(host).SetNumExplicitHs(rw.GetAtomWithIdx(host).GetNumExplicitHs()+1)
        removed=scope-{root}
        for i in sorted(removed,reverse=True):rw.RemoveAtom(i)
        proxy=_sanitize(rw,Chem)
        if proxy is None:continue
        back=tuple(i for i in range(mol.GetNumAtoms()) if i not in removed)
        from .extended_inventory import selection_inventory
        inv=selection_inventory(proxy)
        allowed={'acid','carboxylate','ester','amide','sulfonic-acid','sulfinic-acid','sulfonamide'}
        if not inv.supported or not inv.principal or any(g.kind not in allowed for g in inv.principal):continue
        from .complete_functional_parent import functional_parent_candidates
        from .parent_selection import parent_frontier,replace_parent_facts,_prefix_comparison_values
        candidates=[r for r in functional_parent_candidates(proxy,Chem,include_single=True)
                    if r.principal_centers==frozenset(g.center for g in inv.principal)]
        restored=[]
        for r in candidates:
            if r.construction is None:continue
            mark=back.index(root);hosts=dict(r.construction.atom_locants)
            if back.index(host) not in hosts:continue
            loc=hosts[back.index(host)]
            if sum(rt==mark and ht==back.index(host) and txt=='fluoro' for rt,ht,_ats,txt in r.construction.prefix_branch_ownership)!=1:continue
            new=[(l,text if l==loc and p=='fluoro' else p) for l,p in r.construction.prefixes]
            name=_rebase_component(mol,Chem,r,back,{root:(text,scope)},new,
                probe=probe,boundaries=boundaries)
            if name and r.parent_selection is not None:
                prefix_locs,citation=_prefix_comparison_values(tuple(new))
                facts=replace_parent_facts(r.parent_selection,prefix_locants=prefix_locs,
                    prefix_citation_locants=citation,alphanumerical_key=prefix_alphabetization_key(name)[:3])
                restored.append((facts,name,frozenset(back[i] for i in r.principal_centers)))
        preferred=parent_frontier(restored,lambda r:r[0])
        names={row[1] for row in preferred}
        if len(names)==1:
            centers=frozenset().union(*(row[2] for row in preferred))
            return _out(mol,names.pop(),centers,'P-58.3.2/P-66.3.5.3/P-66.4.2.3.6/P-66.1.1.4.5.1')
    return None


def isocyanato_nitramide(mol,Chem):
    if (mol.GetStereoGroups() or any(a.GetIsotope() or a.GetNumRadicalElectrons() or int(a.GetChiralTag()) for a in iter_atoms(mol))
            or any(int(b.GetStereo()) or int(b.GetBondDir()) for b in iter_bonds(mol))):return None
    from .nitrogen_oxo_amide_prefixes import _terminal_oxo_nitrogen
    from .component_mapping import mapped_branch
    from .phosphorus_ester_methods import _carbon_component
    for n in iter_atoms(mol):
        if n.GetAtomicNum()!=7 or n.GetFormalCharge() or n.GetTotalValence()!=3 or n.IsInRing():continue
        nox=[a for a in n.GetNeighbors() if a.GetAtomicNum()==7 and _terminal_oxo_nitrogen(mol,a.GetIdx(),n.GetIdx()) is not None]
        if len(nox)!=1 or _terminal_oxo_nitrogen(mol,nox[0].GetIdx(),n.GetIdx())[0]!='nitric':continue
        core={n.GetIdx(),nox[0].GetIdx(),*(a.GetIdx() for a in nox[0].GetNeighbors() if a.GetAtomicNum()==8)}
        prefixes=[];isocyanates=0
        for a in n.GetNeighbors():
            if a.GetIdx() in core:continue
            scope=_side(mol,a.GetIdx(),n.GetIdx())
            if scope&core:return None
            if a.GetAtomicNum()==7 and a.GetDegree()==2 and _order(mol,n.GetIdx(),a.GetIdx())==1:
                cs=[c for c in a.GetNeighbors() if c.GetIdx()!=n.GetIdx()]
                if len(cs)!=1 or cs[0].GetAtomicNum()!=6 or _order(mol,a.GetIdx(),cs[0].GetIdx())!=2:return None
                os=[o for o in cs[0].GetNeighbors() if o.GetIdx()!=a.GetIdx()]
                if len(os)!=1 or os[0].GetAtomicNum()!=8 or os[0].GetDegree()!=1 or _order(mol,cs[0].GetIdx(),os[0].GetIdx())!=2:return None
                if scope!={a.GetIdx(),cs[0].GetIdx(),os[0].GetIdx()}:return None
                text='isocyanato';isocyanates+=1
            else:
                atoms=_carbon_component(mol,a.GetIdx(),n.GetIdx())
                text,owned=mapped_branch(mol,Chem,a.GetIdx(),n.GetIdx())
                if atoms is None or atoms!=owned or not text:return None
            prefixes.append(('N',text));core.update(scope)
        if isocyanates and core==set(range(mol.GetNumAtoms())):
            return _out(mol,format_prefixes(prefixes,hierarchical_enclosures=True)+'nitramide',(n.GetIdx(),),'P-58.3.2/P-67.1.4')
    return None


def imido_polycarbonic_acid(mol,Chem):
    if not _plain(mol,Chem,{6,7,8}):return None
    carbons=set()
    for a in iter_atoms(mol):
        if a.GetAtomicNum()!=6 or a.IsInRing() or a.GetDegree()!=3:continue
        doubles=[n for n in a.GetNeighbors() if n.GetAtomicNum() in {7,8} and _order(mol,a.GetIdx(),n.GetIdx())==2]
        singles=[n for n in a.GetNeighbors() if n.GetAtomicNum() in {7,8} and _order(mol,a.GetIdx(),n.GetIdx())==1]
        if len(doubles)==1 and len(singles)==2:carbons.add(a.GetIdx())
    if len(carbons)<2 or len(carbons) not in MULT:return None
    backbone=set(carbons);oxo={};ends=[];owned=set(carbons);n_arms=[]
    for c in carbons:
        atom=mol.GetAtomWithIdx(c)
        double=[a for a in atom.GetNeighbors() if a.GetAtomicNum() in {7,8} and _order(mol,c,a.GetIdx())==2]
        if len(double)!=1:return None
        d=double[0]
        if d.GetAtomicNum()==8:
            if d.GetDegree()!=1:return None
        else:
            if d.IsInRing() or d.GetTotalValence()!=3:return None
            extra=[a for a in d.GetNeighbors() if a.GetIdx()!=c]
            if len(extra)>1 or len(extra)+d.GetTotalNumHs()!=1:return None
            n_arms.extend((d.GetIdx(),c,a.GetIdx(),False) for a in extra)
        oxo[c]=d.GetIdx();owned.add(d.GetIdx())
        for a in atom.GetNeighbors():
            if a.GetIdx()==d.GetIdx():continue
            if _order(mol,c,a.GetIdx())!=1 or a.GetAtomicNum() not in {7,8}:return None
            if a.GetDegree()==1 and a.GetAtomicNum()==8 and a.GetTotalNumHs()==1:
                ends.append(c);owned.add(a.GetIdx())
            else:
                joined=[n for n in a.GetNeighbors() if n.GetIdx() in carbons and _order(mol,a.GetIdx(),n.GetIdx())==1]
                if len(joined)!=2 or a.IsInRing():return None
                if a.GetAtomicNum()==8:
                    if a.GetDegree()!=2 or a.GetTotalNumHs():return None
                else:
                    extra=[n for n in a.GetNeighbors() if n.GetIdx() not in carbons]
                    if a.GetTotalValence()!=3 or len(extra)>1 or len(extra)+a.GetTotalNumHs()!=1:return None
                    n_arms.extend((a.GetIdx(),a.GetIdx(),n.GetIdx(),True) for n in extra)
                backbone.add(a.GetIdx());owned.add(a.GetIdx())
    if len(ends)!=2 or ends[0]==ends[1]:return None
    from .component_mapping import mapped_branch
    from .phosphorus_ester_methods import _carbon_component
    branches=[]
    for nitrogen,position,root,in_chain in dict.fromkeys(n_arms):
        group=_carbon_component(mol,root,nitrogen)
        text,mapped=mapped_branch(mol,Chem,root,nitrogen)
        if group is None or not text or group!=mapped or group&owned:return None
        branches.append((position,in_chain,text));owned.update(group)
    if owned!=set(range(mol.GetNumAtoms())):return None
    from .naming_methods import _linear_order
    path=_linear_order(mol,backbone,ends)
    if not path or len(path)!=2*len(carbons)-1:return None
    choices=[]
    for seq in (path,path[::-1]):
        positions={i:j for j,i in enumerate(seq,1)}
        locs=[positions[c] for c,i in oxo.items() if mol.GetAtomWithIdx(i).GetAtomicNum()==7]
        locs.extend(positions[i] for i in backbone-carbons if mol.GetAtomWithIdx(i).GetAtomicNum()==7)
        if not locs:continue
        locs=tuple(sorted(locs));mult=MULT[len(locs)] if len(locs)>1 else ''
        prefixes=[(str(positions[p]) if chain else 'N^'+str(positions[p]),text) for p,chain,text in branches]
        name=','.join(map(str,locs))+'-'+mult+'imido'+MULT[len(carbons)]+'carbonic acid'
        if prefixes:name=format_prefixes(prefixes,hierarchical_enclosures=True)+'-'+name
        key=(locs,tuple(sorted(positions[p] for p,_chain,_text in branches)),
             tuple((prefix_alphabetization_key(text),positions[p]) for p,_chain,text in sorted(branches,key=lambda r:prefix_alphabetization_key(r[2]))))
        choices.append((key,name))
    return _out(mol,min(choices)[1],carbons,'P-65.2.3.1.2') if choices else None


def sulfinohydrazonamide(mol,Chem):
    if not _plain(mol,Chem,{6,7,8,9,16,17,35,53}):return None
    ss=[a for a in iter_atoms(mol) if a.GetAtomicNum()==16]
    if len(ss)!=1:return None
    s=ss[0]
    imines=[a for a in s.GetNeighbors() if a.GetAtomicNum()==7 and _order(mol,s.GetIdx(),a.GetIdx())==2]
    amino=[a for a in s.GetNeighbors() if a.GetAtomicNum()==7 and _order(mol,s.GetIdx(),a.GetIdx())==1]
    roots=[a for a in s.GetNeighbors() if a.GetAtomicNum()==6 and _order(mol,s.GetIdx(),a.GetIdx())==1]
    if len(imines)!=1 or len(amino)!=1 or len(roots)!=1 or s.GetDegree()!=3:return None
    imine,amide=imines[0],amino[0]
    outer=[a for a in imine.GetNeighbors() if a.GetIdx()!=s.GetIdx()]
    if (len(outer)!=1 or outer[0].GetAtomicNum()!=7 or _order(mol,imine.GetIdx(),outer[0].GetIdx())!=1
            or imine.GetTotalNumHs() or imine.GetDegree()!=2):return None
    terminal=outer[0]
    from .component_mapping import mapped_branch
    from .phosphorus_ester_methods import _carbon_component
    carbon=_carbon_component(mol,roots[0].GetIdx(),s.GetIdx())
    if carbon is None:
        carbon=_side(mol,roots[0].GetIdx(),s.GetIdx())
        if any(mol.GetAtomWithIdx(i).GetAtomicNum()!=6 and not (
                mol.GetAtomWithIdx(i).GetAtomicNum()==8 and mol.GetAtomWithIdx(i).GetDegree()==1
                and mol.GetAtomWithIdx(i).GetTotalNumHs()==1 and all(b.GetBondTypeAsDouble()==1 for b in mol.GetAtomWithIdx(i).GetBonds())
                or mol.GetAtomWithIdx(i).GetAtomicNum() in {9,17,35,53} and mol.GetAtomWithIdx(i).GetDegree()==1)
                for i in carbon):return None
    core={s.GetIdx(),imine.GetIdx(),amide.GetIdx(),terminal.GetIdx()};owned=core|set(carbon);prefixes=[];n_branches=[]
    for n,blocked,loc in ((amide,s.GetIdx(),'N'),(terminal,imine.GetIdx(),"N'")):
        if n.GetTotalValence()!=3 or n.IsInRing():return None
        for root in n.GetNeighbors():
            if root.GetIdx()==blocked:continue
            group=_carbon_component(mol,root.GetIdx(),n.GetIdx())
            text,atoms=mapped_branch(mol,Chem,root.GetIdx(),n.GetIdx())
            if group is None or group!=atoms or not text or group&owned:return None
            prefixes.append((loc,text));n_branches.append((root.GetIdx(),n.GetIdx(),group,text));owned.update(group)
    if owned!=set(range(mol.GetNumAtoms())):return None
    rw=Chem.RWMol(mol)
    rw.GetAtomWithIdx(imine.GetIdx()).SetAtomicNum(8)
    rw.GetAtomWithIdx(imine.GetIdx()).SetNoImplicit(True);rw.GetAtomWithIdx(imine.GetIdx()).SetNumExplicitHs(0)
    rw.GetAtomWithIdx(amide.GetIdx()).SetAtomicNum(8)
    rw.GetAtomWithIdx(amide.GetIdx()).SetNumExplicitHs(1);rw.GetAtomWithIdx(amide.GetIdx()).SetNoImplicit(True)
    keep=set(carbon)|{s.GetIdx(),imine.GetIdx(),amide.GetIdx()}
    for i in sorted(set(range(mol.GetNumAtoms()))-keep,reverse=True):rw.RemoveAtom(i)
    proxy=_sanitize(rw,Chem)
    if proxy is None:return None
    from .api import mol_to_iupac
    acid=mol_to_iupac(proxy)
    if not acid.endswith('sulfinic acid'):return None
    base=acid[:-len('sulfinic acid')]+'sulfinohydrazonamide'
    if prefixes:
        from .complete_functional_parent import functional_parent_candidates
        from .parent_selection import parent_frontier
        from .component_mapping import decorate_component
        back=tuple(i for i in range(mol.GetNumAtoms()) if i in keep)
        candidates=parent_frontier(list(functional_parent_candidates(proxy,Chem,include_single=True)),lambda r:r.parent_selection)
        names=set()
        for record in candidates:
            comp=record.construction
            if comp is None or not comp.parent.endswith('sulfinic acid'):continue
            labels=[(back[i],loc) for i,loc in comp.atom_locants if back[i] not in {amide.GetIdx(),imine.GetIdx()}]
            labels.extend(((amide.GetIdx(),'N'),(terminal.GetIdx(),"N'"),(imine.GetIdx(),"N''")))
            ownership=tuple((back[r],back[h],frozenset(back[i] for i in atoms),text) for r,h,atoms,text in comp.prefix_branch_ownership)+tuple(n_branches)
            component=replace(comp,parent=comp.parent[:-len('sulfinic acid')]+'sulfinohydrazonamide',
                atom_locants=tuple(labels),atoms=frozenset(back[i] for i in comp.atoms)|{terminal.GetIdx()},
                prefixes=comp.prefixes+tuple(prefixes),embedded_atoms=frozenset(back[i] for i in comp.embedded_atoms)|frozenset().union(*(row[2] for row in n_branches)),
                prefix_branch_ownership=ownership,nitrogen_site_locants=True)
            name=decorate_component(mol,Chem,component)
            if name:names.add(name)
        if len(names)!=1:return None
        return _out(mol,names.pop(),(s.GetIdx(),),'P-66.4.2.1/P-66.1.1.3')
    return _out(mol,format_prefixes(prefixes,hierarchical_enclosures=True)+base,(s.GetIdx(),),'P-66.4.2.1')


def equivalent_amide_assembly(mol,Chem):
    if not _plain(mol,Chem,{6,7,8}):return None
    ns=[a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum()==7]
    if len(ns)!=4:return None
    from .naming_methods import _fragment
    from .api import mol_to_iupac
    for bridge in iter_bonds(mol):
        a,b=bridge.GetBeginAtomIdx(),bridge.GetEndAtomIdx()
        if a not in ns or b not in ns or bridge.GetBondTypeAsDouble()!=1:continue
        central={a,b};units=[]
        for n in (a,b):
            outside=[x for x in mol.GetAtomWithIdx(n).GetNeighbors() if x.GetIdx() not in central]
            if len(outside)!=1 or outside[0].GetAtomicNum()!=7 or _order(mol,n,outside[0].GetIdx())!=1:break
            root=outside[0].GetIdx();scope=_side(mol,root,n)
            c=[x for x in outside[0].GetNeighbors() if x.GetAtomicNum()==6 and _order(mol,root,x.GetIdx())==1
               and any(y.GetAtomicNum()==8 and _order(mol,x.GetIdx(),y.GetIdx())==2 for y in x.GetNeighbors())]
            if len(c)!=1 or central&scope:break
            fragment,original=_fragment(mol,Chem,scope)
            if fragment is None:break
            local=original.index(root);fragment.GetAtomWithIdx(local).SetAtomMapNum(1)
            identity=Chem.MolToSmiles(fragment);fragment.GetAtomWithIdx(local).SetAtomMapNum(0)
            name=mol_to_iupac(fragment)
            if not name.endswith('amide'):break
            units.append((scope,identity,name,c[0].GetIdx()))
        else:
            if len(units)!=2 or units[0][0]&units[1][0] or central|units[0][0]|units[1][0]!=set(range(mol.GetNumAtoms())):continue
            if units[0][1:3]!=units[1][1:3]:continue
            parent=units[0][2]
            multiplied='di'+parent if parent[0].isalpha() and '-' not in parent and '(' not in parent else 'bis('+parent+')'
            return _out(mol,"N,N'-(hydrazine-1,2-diyl)"+multiplied,(row[3] for row in units),'P-15.3.2.1/P-58.3.2',method='multiplicative')
    return None


def hydrazine_ylidene_assembly(mol,Chem):
    if not _plain(mol,Chem,{6,7}):return None
    ns=[a for a in iter_atoms(mol) if a.GetAtomicNum()==7]
    if len(ns)!=4:return None
    units=[]
    from .component_mapping import mapped_branch
    from .phosphorus_ester_methods import _carbon_component
    from .naming_methods import _fragment,_linear_order
    for n in ns:
        carbon=[a for a in n.GetNeighbors() if a.GetAtomicNum()==6 and _order(mol,n.GetIdx(),a.GetIdx())==2]
        terminal=[a for a in n.GetNeighbors() if a.GetAtomicNum()==7 and _order(mol,n.GetIdx(),a.GetIdx())==1]
        if len(carbon)!=1 or len(terminal)!=1 or n.GetDegree()!=2:continue
        t=terminal[0];scope=_side(mol,n.GetIdx(),carbon[0].GetIdx())
        if carbon[0].GetIdx() in scope or t.GetTotalValence()!=3:continue
        owned={n.GetIdx(),t.GetIdx()};prefixes=[]
        for root in t.GetNeighbors():
            if root.GetIdx()==n.GetIdx():continue
            group=_carbon_component(mol,root.GetIdx(),t.GetIdx())
            text,atoms=mapped_branch(mol,Chem,root.GetIdx(),t.GetIdx())
            if group is None or group!=atoms or not text or group&owned:break
            prefixes.append(('2',text));owned.update(group)
        else:
            if owned!=set(scope):continue
            fragment,original=_fragment(mol,Chem,scope)
            if fragment is None:continue
            fragment.GetAtomWithIdx(original.index(n.GetIdx())).SetAtomMapNum(1)
            identity=Chem.MolToSmiles(fragment)
            units.append((scope,carbon[0].GetIdx(),identity,format_prefixes(prefixes,hierarchical_enclosures=True)+'hydrazine'))
    if len(units)!=2 or units[0][0]&units[1][0] or units[0][2:]!=units[1][2:]:return None
    central=set(range(mol.GetNumAtoms()))-units[0][0]-units[1][0]
    if not central or len(central) not in ROOTS or any(mol.GetAtomWithIdx(i).GetAtomicNum()!=6 for i in central):return None
    rings=tuple(r for r in mol.GetRingInfo().AtomRings() if set(r)<=central)
    if rings:
        if len(rings)!=1 or set(rings[0])!=central:return None
        from .ring_rules import _carbocycle_parent
        from .formatting import complete_parent_name
        ring=tuple(rings[0]);choices=[]
        for direction in (ring,ring[::-1]):
            for j in range(len(ring)):
                seq=direction[j:]+direction[:j]
                locs=tuple(sorted(seq.index(u[1])+1 for u in units))
                ene=tuple(i for i,(a,b) in enumerate(zip(seq,seq[1:]+seq[:1]),1) if _order(mol,a,b)==2)
                parent=complete_parent_name(_carbocycle_parent(mol,seq))
                if parent:choices.append(((locs,ene),parent+'-'+','.join(map(str,locs))+'-diylidene'))
        if not choices:return None
        linker=min(choices)[1]
    else:
        path=(next(iter(central)),) if len(central)==1 else _linear_order(mol,central,[units[0][1],units[1][1]])
        if not path or any(_order(mol,a,b)!=1 for a,b in zip(path,path[1:])):return None
        linker=ROOTS[len(path)]+'ane'+('diylidene' if len(path)<=2 else '-1,'+str(len(path))+'-diylidene')
    if '-' in linker:linker=enclose_name_unit(linker)
    return _out(mol,"1,1'-"+linker+'bis('+units[0][3]+')',(row[1] for row in units),'P-68.3.1.2.2/P-15.3.2.1',method='multiplicative')


def carbamimidoyl_urea(mol,Chem):
    if not _plain(mol,Chem,{6,7,8}):return None
    cs=[a for a in iter_atoms(mol) if a.GetAtomicNum()==6 and any(n.GetAtomicNum()==8 and _order(mol,a.GetIdx(),n.GetIdx())==2 for n in a.GetNeighbors())]
    if len(cs)!=1:return None
    c=cs[0];ox=[a for a in c.GetNeighbors() if a.GetAtomicNum()==8 and a.GetDegree()==1 and _order(mol,c.GetIdx(),a.GetIdx())==2]
    ns=[a for a in c.GetNeighbors() if a.GetAtomicNum()==7 and _order(mol,c.GetIdx(),a.GetIdx())==1]
    if len(ox)!=1 or len(ns)!=2 or c.GetDegree()!=3:return None
    from .component_mapping import mapped_branch
    from .branch.carbamimidoyl import carbamimidoyl_prefix
    from .phosphorus_ester_methods import _carbon_component
    choices=[]
    for arrangement in permutations(ns):
        prefixes=[];owned={c.GetIdx(),ox[0].GetIdx(),*(n.GetIdx() for n in ns)};count=0
        for n,loc in zip(arrangement,('N',"N'")):
            for root in n.GetNeighbors():
                if root.GetIdx()==c.GetIdx():continue
                scope=_side(mol,root.GetIdx(),n.GetIdx())
                text=carbamimidoyl_prefix(mol,Chem,root.GetIdx(),n.GetIdx()) if root.GetAtomicNum()==6 else ''
                if text:count+=1
                else:
                    atoms=_carbon_component(mol,root.GetIdx(),n.GetIdx());text,mapped=mapped_branch(mol,Chem,root.GetIdx(),n.GetIdx())
                    if atoms is None or atoms!=mapped:break
                if not text or scope&owned:break
                prefixes.append((loc,text));owned.update(scope)
            else:continue
            break
        else:
            if count and owned==set(range(mol.GetNumAtoms())):
                key=tuple(loc for loc,text in sorted(prefixes,key=lambda r:prefix_alphabetization_key(r[1])))
                choices.append((key,format_prefixes(prefixes,hierarchical_enclosures=True)+'urea'))
    return _out(mol,min(choices)[1],(c.GetIdx(),),'P-66.4.1.2.1.3') if choices else None


def select(mol,Chem,numbers):
    if not numbers[7]:return None
    producers=[]
    if numbers[7]>=3 and numbers[8]>=3:producers.append(isocyanato_nitramide)
    if numbers[7] and numbers[8]>=2 and numbers[6]>=2:producers.append(imido_polycarbonic_acid)
    if numbers[7]>=3 and numbers[16]:producers.append(sulfinohydrazonamide)
    if numbers[7]>=4 and numbers[8]>=1:producers.extend((carbamimidoyl_urea,equivalent_amide_assembly))
    if numbers[7]==4 and not numbers[8]:producers.append(hydrazine_ylidene_assembly)
    if numbers[8] and (numbers[7]>=2 or numbers[16]):producers.append(subordinate_prefix_decision)
    for producer in producers:
        result=producer(mol,Chem)
        if result:return result
    return None
