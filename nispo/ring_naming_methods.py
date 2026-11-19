"""Bounded P-24/P-28/P-52 construction choices from actual ring topology."""
from .constants import ROOTS
from .naming_methods import MethodDecision


def ring_naming_decision(mol,Chem):
    from .bridged_fused_methods import select_methano_fused_decision
    bridged = select_methano_fused_decision(mol, Chem)
    if bridged is not None:
        return bridged
    from .simple_fused_bridges import select_simple_fused_bridge
    bridged = select_simple_fused_bridge(mol, Chem)
    if bridged is not None:
        return bridged
    from .extended_fused_bridges import select_extended_fused_bridge
    bridged = select_extended_fused_bridge(mol, Chem)
    if bridged is not None:
        return bridged
    from .multiple_fused_bridges import select_multiple_fused_bridges
    bridged = select_multiple_fused_bridges(mol, Chem)
    if bridged is not None:
        return bridged
    from .mapped_ring_parents import heteropolycyclic_hydride
    record=heteropolycyclic_hydride(mol,Chem)
    if record is not None:
        from .heteropolycyclic_parents import RULES
        if record.method == 'skeletal-replacement-monocycle':
            rules = ('P-22.2.3.1', 'P-22.2.3.2', 'P-51.4.2', 'P-14.3.3')
        elif record.method in {'unsaturated-monocycle', 'skeletal-replacement-unsaturated-monocycle'}:
            rules = ('P-31.1.1', 'P-31.1.3', 'P-22.2.3', 'P-14.4')
        elif record.method == 'dispiro':
            rules = ('P-24.2.2', 'P-24.2.2.1', 'P-24.2.2.2', 'P-14.4')
        elif record.method == 'skeletal-replacement-dispiro':
            from .heterodispiro_parents import RULES as DISPIRO_RULES
            rules = DISPIRO_RULES[record.method]
        elif record.method in {'retained-fused', 'Hantzsch-Widman', 'monocyclic'}:
            rules = ('P-12.1', 'P-25.2.1', 'P-29', 'P-44', 'P-53')
        elif record.method in {'component-fused', 'component-fused-benzofuran-pyridine',
                               'component-fused-three-face-azine', 'component-fused-general'}:
            rules = ('P-25.3.2', 'P-25.3.3.1', 'P-25.7.1.3', 'P-54.4', 'P-58.2')
        elif record.method == 'component-spiro':
            rules = ('P-24.5.1', 'P-24.5.2', 'P-23.3', 'P-14.4')
        elif record.method == 'benzo-heterocycle':
            rules = ('P-25.2.2.4', 'P-25.3.5', 'P-54.4', 'P-58.2')
        elif record.method in {'skeletal-replacement-localized-bicyclic',
                               'skeletal-replacement-explicit-bicyclic'}:
            rules = ('P-23.3', 'P-31.1.4', 'P-25.7.1.2')
        elif record.method == 'skeletal-replacement-localized-tricyclic':
            rules = ('P-23.2', 'P-31.1.4', 'P-25.7.1.2')
        else:
            rules = RULES.get(record.method)
            if rules is None:return None
        heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
        required = frozenset(a.GetIdx() for a in mol.GetAtoms()
                             if a.GetAtomicNum() > 1 or a.GetIsotope())
        construction = record.construction
        if construction is None:
            return None
        mapped = frozenset(dict(record.atom_locants))
        embedded = construction.embedded_atoms
        if (mapped != construction.atoms or mapped & embedded
                or mapped | embedded != required):
            return None
        roles = (('parent-hydride', mapped, record.atom_locants),)
        if embedded:
            roles += (('cited-substituents', embedded, ()),)
        return MethodDecision(record.name,record.method,frozenset(),heavy,
            '/'.join(rules),record.atom_locants,
            component_roles=roles,preference_issues=record.preference_issues)
    if any(a.GetAtomicNum()!=6 or a.GetFormalCharge() or a.GetIsotope()
           or a.GetNumRadicalElectrons() or int(a.GetChiralTag()) for a in mol.GetAtoms()):return None
    if any(int(b.GetStereo()) for b in mol.GetBonds()):return None
    rings=tuple(map(tuple,mol.GetRingInfo().AtomRings()))
    if len(rings)!=2:return None
    a,b=rings;sa,sb=set(a),set(b);shared=sa&sb
    if sa|sb!=set(range(mol.GetNumAtoms())):return None
    saturated=all(b.GetBondTypeAsDouble()==1 for b in mol.GetBonds())
    atoms=frozenset(sa|sb)
    if len(shared)==1 and saturated:
        sizes=sorted((len(a)-1,len(b)-1))
        if len(atoms) not in ROOTS:return None
        name='spiro['+'.'.join(map(str,sizes))+']'+ROOTS[len(atoms)]+'ane'
        return MethodDecision(name,'spiro',frozenset(),atoms,'P-24.2.1')
    if not shared:
        links=[bond for bond in mol.GetBonds() if (bond.GetBeginAtomIdx() in sa)!=(bond.GetEndAtomIdx() in sa)]
        if len(links)!=1 or links[0].GetBondTypeAsDouble()!=1 or len(a)!=len(b):return None
        if saturated and len(a) in ROOTS:
            name="1,1'-bi(cyclo"+ROOTS[len(a)]+"ane)"
        elif len(a)==6 and all(atom.GetIsAromatic() for atom in mol.GetAtoms()):
            name="1,1'-biphenyl"
        else:return None
        return MethodDecision(name,'ring-assembly',frozenset(),atoms,'P-28/P-52.2.7.1')
    if len(shared)==2 and mol.GetBondBetweenAtoms(*sorted(shared)) is not None and min(map(len,rings))<5:
        from .polycyclic_rules import bicyclic_parent_candidate_records
        records=bicyclic_parent_candidate_records(mol,Chem)
        for record in records:
            name,sequence=record[:2]
            if set(sequence)==atoms and 'bicyclo[' in name and not any(record[2:]):
                return MethodDecision(name,'von-baeyer',frozenset(),atoms,'P-52.2.4.1',
                                      tuple((idx,str(j)) for j,idx in enumerate(sequence,1)))
    return None
