"""Original-H O/C alkyl and aryl joins for P-104 retained parents."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from .carbohydrates import _SugarInput,_SugarScope
from .carbohydrate_modifications import _ROLE,_OWNER,_consume


@dataclass(frozen=True,slots=True)
class CyclitolSubstituent:
    position: int
    attachment: str
    group_smiles: str
    attachment_atom: int = 0


@dataclass(frozen=True,slots=True)
class CyclitolSubstituentBinding:
    position: int
    attachment: str
    original_parent_atom_map: int
    substituent_root_map: int
    atom_maps: tuple[int,...]
    organyl_name: str
    has_annotations: bool
    unassigned_stereo_atom_maps: tuple[int,...] = ()


def _source_graph(request,Chem):
    graph=Chem.MolFromSmiles(request.group_smiles)
    if graph is None or len(Chem.GetMolFrags(graph))!=1:raise _SugarInput('Each alkyl/aryl source must be one valid connected graph.')
    if not 0<=request.attachment_atom<graph.GetNumAtoms():raise _SugarInput('The source attachment atom is absent.')
    if (graph.GetStereoGroups() or any(a.GetFormalCharge() or a.GetNumRadicalElectrons()
            or a.GetAtomicNum() not in {1,6,9,17,35,53} or a.GetIsotope() and a.GetAtomicNum()!=6
            or a.GetChiralTag() not in {Chem.ChiralType.CHI_UNSPECIFIED,Chem.ChiralType.CHI_TETRAHEDRAL_CW,Chem.ChiralType.CHI_TETRAHEDRAL_CCW}
            for a in graph.GetAtoms()) or any(b.GetBondType()!=Chem.BondType.SINGLE and not b.GetIsAromatic()
            or int(b.GetStereo()) for b in graph.GetBonds())):
        raise _SugarScope('This operation requires a neutral alkyl/aryl source with ordinary tetrahedral and carbon-isotope states.')
    root=graph.GetAtomWithIdx(request.attachment_atom)
    if root.GetAtomicNum()!=6 or root.GetTotalNumHs()<1:raise _SugarInput('The source carbon must have an available attachment hydrogen.')
    for atom in graph.GetAtoms():atom.SetAtomMapNum(0)
    return graph


def substitute(base,requests,Chem):
    if not isinstance(requests,tuple):raise _SugarInput('Substituents must be an immutable tuple of typed original-site requests.')
    sites=[]
    for request in requests:
        if (not isinstance(request,CyclitolSubstituent) or type(request.position) is not int or not 1<=request.position<=6
                or not isinstance(request.attachment,str) or request.attachment not in {'O','C'}
                or not isinstance(request.group_smiles,str) or not request.group_smiles or type(request.attachment_atom) is not int):
            raise _SugarInput('Declare an original inositol site, O/C H port, source graph and source carbon index.')
        sites.append((request.position,request.attachment))
    if len(set(sites))!=len(sites):raise _SugarInput('Two substitutions cannot consume the same O-H or C-H port.')
    graph=Chem.RWMol(base);joins=[]
    for request in sorted(requests,key=lambda r:(r.position,r.attachment)):
        role=('O' if request.attachment=='O' else '')+str(request.position)
        hosts=[a for a in graph.GetAtoms() if a.HasProp(_ROLE) and a.GetProp(_ROLE)==role]
        if len(hosts)!=1:raise _SugarInput('The requested original host is unavailable.')
        host=hosts[0]
        if request.attachment=='O':
            if host.GetAtomicNum()!=8 or host.GetDegree()!=1 or host.GetTotalNumHs()!=1:raise _SugarInput('O substitution requires the unconsumed original OH H.')
        elif (host.GetAtomicNum()!=6 or host.GetTotalNumHs()!=1
                or host.GetChiralTag() not in {Chem.ChiralType.CHI_TETRAHEDRAL_CW,Chem.ChiralType.CHI_TETRAHEDRAL_CCW}):
            raise _SugarScope('C substitution requires the original configured one-H carbon port.')
        piece=_source_graph(request,Chem);offset=graph.GetNumAtoms();root=offset+request.attachment_atom
        owner=f'sub:{request.attachment}:{request.position}'
        for atom in piece.GetAtoms():atom.SetProp(_OWNER,owner)
        # The appended bond occupies the original final implicit-H port in
        # each native ordered tetrahedron. Retain physical tag parity.
        _consume(host);_consume(piece.GetAtomWithIdx(request.attachment_atom))
        graph.InsertMol(piece);graph.AddBond(host.GetIdx(),root,Chem.BondType.SINGLE)
        graph.UpdatePropertyCache(strict=False);joins.append((request,host.GetIdx(),root,owner))
    graph=graph.GetMol();Chem.SanitizeMol(graph);Chem.AssignStereochemistry(graph,cleanIt=True,force=True)
    from rdkit.Chem import rdCIPLabeler
    rdCIPLabeler.AssignCIPLabels(graph,maxRecursiveIterations=1000000)
    from .component_mapping import mapped_branch
    potential=Chem.FindPotentialStereo(graph);bindings=[];terms=[]
    for request,host,root,owner in joins:
        owned=frozenset(a.GetIdx() for a in graph.GetAtoms() if a.GetProp(_OWNER)==owner)
        word,claimed=mapped_branch(graph,Chem,root,host)
        if not word or claimed!=owned:raise _SugarScope('The complete joined organyl lacks a proved mapped component name.')
        annotated=any(graph.GetAtomWithIdx(i).GetIsotope() or int(graph.GetAtomWithIdx(i).GetChiralTag()) for i in owned)
        unspecified=tuple(s.centeredOn+1 for s in potential if s.type==Chem.StereoType.Atom_Tetrahedral
            and s.centeredOn in owned and s.specified==Chem.StereoSpecified.Unspecified)
        bindings.append(CyclitolSubstituentBinding(request.position,request.attachment,host+1,root+1,
            tuple(i+1 for i in sorted(owned)),word,annotated,unspecified))
        terms.append((request.position,word,request.attachment))
    return graph,terms,tuple(bindings)


def prefix_terms(terms):
    from .constants import MULT
    from .formatting import prefix_alphabetization_key,format_prefixes
    grouped=defaultdict(list);markers=defaultdict(set)
    for position,word,marker in terms:grouped[word,marker].append(position);markers[word].add(marker)
    if any(len(values)>1 for values in markers.values()):raise _SugarScope('Equal-word O/C attachment classes need a joint citation comparison.')
    rendered=[]
    for (word,marker),positions in sorted(grouped.items(),key=lambda item:(prefix_alphabetization_key(item[0][0]),item[0][1])):
        unit=format_prefixes([('',word)],omit_locants=True,hierarchical_enclosures=True);count=len(positions)
        if count>1 and unit!=word:raise _SugarScope('Repeated complex organyl units need a proved O/C multiplication construction.')
        multiplier=MULT[count] if count>1 else ''
        unit=multiplier+('-'+marker+'-' if multiplier else marker+'-')+unit if marker else multiplier+unit
        rendered.append(','.join(map(str,sorted(positions)))+'-'+unit)
    return '-'.join(rendered)
