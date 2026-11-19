"""P-101 numbered steroid construction with original physical atom roles."""
from __future__ import annotations

from collections import defaultdict,deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re

from .naming_result import NamingResult,SelectionLimitation
from .constants import MULT


@dataclass(frozen=True,slots=True)
class NaturalProductHomo:
    site: str
    count: int = 1


@dataclass(frozen=True,slots=True)
class NaturalProductReplacement:
    site: str
    element: str


@dataclass(frozen=True,slots=True)
class NaturalProductSpec:
    parent: str = 'androstane'
    c5_configuration: str = 'alpha'
    nor: tuple[str,...] = ()
    homo: tuple[NaturalProductHomo,...] = ()
    apo: tuple[str,...] = ()
    seco: tuple[tuple[str,str],...] = ()
    cyclo: tuple[tuple[str,str],...] = ()
    abeo: tuple[tuple[str,str,str],...] = ()
    des: tuple[str,...] = ()
    replacements: tuple[NaturalProductReplacement,...] = ()
    double_bonds: tuple[tuple[str,str],...] = ()
    epimers: tuple[str,...] = ()
    stereochemistry: tuple[tuple[str,str],...] = ()


@dataclass(frozen=True,slots=True)
class NaturalProductNamingResult(NamingResult):
    canonical_smiles: str = ''
    mapped_smiles: str = ''
    requested_parent: str = ''
    selected_parent: str = ''
    constructed_name: str = ''
    parent_atom_maps: tuple[tuple[str,int],...] = ()
    selected_parent_atom_maps: tuple[tuple[str,int],...] = ()
    deleted_locants: tuple[str,...] = ()
    inserted_locants: tuple[str,...] = ()
    skeletal_operation_count: int = 0
    cycle_rank: int = 0
    formula: str = ''
    construction_limitations: tuple[SelectionLimitation,...] = ()
    minimum_operation_status: str = 'not-compared'
    unassigned_stereo_locants: tuple[str,...] = ()
    preferred_name_scope: str = 'P-100: Chapter 10 does not identify PINs'


class _Input(ValueError):pass
class _Scope(ValueError):pass


_STEMS={'gonane':'gon','estrane':'estr','androstane':'androst','pregnane':'pregn','cholestane':'cholest'}
_SITE='_naturalProductSite'
# Standard steroid drawing; z is assigned only to the alpha/beta H port.
_XY={1:(-3,1),2:(-4,.5),3:(-4,-.5),4:(-3,-1),5:(-2,-.5),6:(-1,-1),7:(0,-.5),8:(0,.5),
     9:(-1,1),10:(-2,.5),11:(-1,2),12:(0,2.5),13:(1,2),14:(1,1),15:(2,.75),16:(2.75,1.5),17:(2,2.25)}


def _key(site):
    parts=re.fullmatch(r'(\d+)(?:\((\d+)\))?([a-z]*)',site)
    return (int(parts[1]),int(parts[2] or 0),parts[3]) if parts else (100000,0,site)


def _sites(graph):return {a.GetProp(_SITE):a.GetIdx() for a in graph.GetAtoms()}


def _local_h_tag(graph,site,side,Chem):
    """Original ordered carbon neighbours plus the explicit alpha/beta H port."""
    mapping=_sites(graph);atom=graph.GetAtomWithIdx(mapping[str(site)])
    roots=[int(a.GetProp(_SITE)) for a in atom.GetNeighbors()]
    if len(roots)!=3 or any(i not in _XY for i in roots):raise _Scope('The source H-port orientation is outside the steroid diagram.')
    probe=Chem.MolFromSmiles('C(F)(Cl)(Br)I');conf=Chem.Conformer(5);conf.Set3D(True)
    x,y=_XY[site];conf.SetAtomPosition(0,(x,y,0))
    for i,root in enumerate(roots,1):conf.SetAtomPosition(i,(*_XY[root],0))
    conf.SetAtomPosition(4,(x,y,side));probe.AddConformer(conf)
    Chem.AssignAtomChiralTagsFromStructure(probe)
    return probe.GetAtomWithIdx(0).GetChiralTag()


@lru_cache(maxsize=5)
def _parent_data(parent):
    from .api import Chem
    import xml.etree.ElementTree as ET
    path=Path(__file__).resolve().parent/'resources'/'opsin_tokens'/'naturalProducts.xml'
    matches=[token for token in ET.parse(path).iter('token') if token.text and _STEMS[parent] in token.text.split('|')]
    if len(matches)!=1:raise _Scope('The numbered fundamental steroid resource is ambiguous or absent.')
    token=matches[0];mol=Chem.MolFromSmiles(token.attrib['value']);labels=token.attrib['labels'].split('/')
    if mol is None or len(labels)!=mol.GetNumAtoms() or any(not label.isdigit() for label in labels):
        raise _Scope('The fundamental parent does not have complete original numbering.')
    for atom,label in zip(mol.GetAtoms(),labels,strict=True):
        atom.SetProp(_SITE,label);atom.SetNumExplicitHs(0);atom.SetNoImplicit(False)
    mol.UpdatePropertyCache(strict=True)
    mapping=_sites(mol);anchor=mol.GetAtomWithIdx(mapping['14']).GetChiralTag()
    if anchor not in {Chem.ChiralType.CHI_TETRAHEDRAL_CW,Chem.ChiralType.CHI_TETRAHEDRAL_CCW}:
        raise _Scope('The numbered fundamental parent has no committed C14 stereo anchor.')
    # The established fundamental C14 H is alpha. Calibrate the orientation
    # of the independent planar diagram against that original physical port.
    normal=-1 if _local_h_tag(mol,14,-1,Chem)==anchor else 1
    if _local_h_tag(mol,14,normal,Chem)!=anchor:raise _Scope('The source C14 alpha-H anchor cannot orient the drawing.')
    return mol,normal


def _parent(spec,Chem):
    stored,alpha=_parent_data(spec.parent);graph=Chem.Mol(stored);mapping=_sites(graph)
    c5=graph.GetAtomWithIdx(mapping['5'])
    c5.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED if spec.c5_configuration=='unspecified'
        else _local_h_tag(graph,5,alpha if spec.c5_configuration=='alpha' else -alpha,Chem))
    return graph


@lru_cache(maxsize=5)
def _steroid_connectivity(parent, Chem):
    # Match the numbered carbon frame first. The decorated graph below is
    # the witness for bond orders and every retained stereogenic unit.
    parameters = Chem.AdjustQueryParameters.NoAdjustments()
    parameters.makeBondsGeneric = True
    return Chem.AdjustQueryProperties(_parent_data(parent)[0], parameters)


def mapped_steroid_ketone(mol, Chem):
    """Use a fundamental steroid parent when its ketone and OH groups reconstruct the source."""
    from .extended_inventory import selection_inventory
    from .formatting import attach_suffix, format_prefixes, multiplied_suffix
    from .naming_methods import MethodDecision

    if (mol.GetRingInfo().NumRings() != 4 or not 18 <= mol.GetNumHeavyAtoms() <= 36
            or mol.GetStereoGroups() or any(a.GetAtomicNum() not in {6, 8}
                or a.GetIsotope() or a.GetFormalCharge() or a.GetNumRadicalElectrons()
                or a.GetAtomMapNum() or a.HasQuery() for a in mol.GetAtoms())
            or any(b.GetStereo() or b.HasQuery() for b in mol.GetBonds())):
        return None
    inventory = selection_inventory(mol)
    groups = inventory.groups
    if (not inventory.supported or not inventory.principal
            or any(g.kind not in {'ketone', 'alcohol'} for g in groups)
            or any(g.kind != 'ketone' for g in inventory.principal)):
        return None
    carbon = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
    oxygens = {i for g in groups for i in g.atoms if i not in carbon}
    scope = frozenset(range(mol.GetNumAtoms()))
    if carbon | oxygens != scope or len(oxygens) != len(groups):
        return None
    edges = [b for b in mol.GetBonds() if {b.GetBeginAtomIdx(), b.GetEndAtomIdx()} <= carbon]
    multiple = [b for b in edges if b.GetBondType() != Chem.BondType.SINGLE]
    if (len(edges) != len(carbon) + 3 or len(multiple) != 1
            or multiple[0].GetBondType() != Chem.BondType.DOUBLE or not multiple[0].IsInRing()):
        return None
    original = Chem.MolToSmiles(mol, isomericSmiles=True)
    bond = multiple[0]
    candidates = []
    for parent, stem in _STEMS.items():
        if len(carbon) != _parent_data(parent)[0].GetNumAtoms():
            continue
        query = _steroid_connectivity(parent, Chem)
        # P-101.6 uses the lower number on a sequential steroid double bond.
        # This writer only covers unsaturation that removes the C5 descriptor.
        for match in mol.GetSubstructMatches(query, uniquify=False, maxMatches=16):
            if set(match) != carbon:
                continue
            locants = {idx: a.GetProp(_SITE) for a, idx in zip(query.GetAtoms(), match, strict=True)}
            a, b = sorted((locants[bond.GetBeginAtomIdx()], locants[bond.GetEndAtomIdx()]), key=int)
            if '5' not in {a, b} or int(b) != int(a) + 1:
                continue

            def decorate(editable):
                sites = _sites(editable)
                editable.GetBondBetweenAtoms(sites[a], sites[b]).SetBondType(Chem.BondType.DOUBLE)
                for group in groups:
                    locant = locants[group.center]
                    oxygen = Chem.Atom(8)
                    oxygen.SetProp(_SITE, 'O' + locant)
                    order = Chem.BondType.DOUBLE if group.kind == 'ketone' else Chem.BondType.SINGLE
                    editable.AddBond(sites[locant], editable.AddAtom(oxygen), order)

            witness = _edit(_parent(NaturalProductSpec(parent=parent), Chem), decorate, Chem)
            Chem.AssignStereochemistry(witness, cleanIt=True, force=True)
            if Chem.MolToSmiles(witness, isomericSmiles=True) != original:
                continue
            ketones = sorted((locants[g.center] for g in inventory.principal), key=int)
            prefixes = [(locants[g.center], 'hydroxy') for g in groups if g.kind == 'alcohol']
            hydride = f'{stem}-{a}-ene'
            suffix = multiplied_suffix(len(ketones), 'one')
            if not suffix:
                return None
            name = format_prefixes(prefixes) + attach_suffix(hydride, suffix, ketones)
            candidates.append((name, tuple(sorted(locants.items()))))
    if len(candidates) != 1:
        return None
    name, mapping = candidates[0]
    rule = 'P-101.2/P-101.6/P-101.7/P-64'
    return MethodDecision(name, 'retained-stereoparent', frozenset(g.center for g in inventory.principal),
        scope, rule, mapping, preference_issues=(SelectionLimitation('natural-product-parent', rule,
            'The numbered stereoparent and its modifications reconstruct the complete original graph.'),))


def mapped_steroid_yl(mol, Chem, root, host, scope):
    """Use a numbered fundamental steroid graph for a complete rooted prefix."""
    if (mol.GetRingInfo().NumRings() < 4 or not 17 <= len(scope) <= 48
            or mol.GetStereoGroups()):
        return "", frozenset()
    from .mapped_ring_parents import ring_systems

    ring = next((core for core in ring_systems(mol) if root in core), frozenset())
    if (len(ring) != 17 or host in ring or not ring <= scope
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in ring)):
        return "", frozenset()
    if any(a.GetIsotope() or a.GetFormalCharge() or a.GetNumRadicalElectrons()
           for a in (mol.GetAtomWithIdx(i) for i in scope)):
        return "", frozenset()
    if {(i, n.GetIdx()) for i in scope for n in mol.GetAtomWithIdx(i).GetNeighbors()
            if n.GetAtomicNum() > 1 and n.GetIdx() not in scope} != {(root, host)}:
        return "", frozenset()
    if any(b.GetBondTypeAsDouble() != 1 for i in ring for b in mol.GetAtomWithIdx(i).GetBonds()):
        return "", frozenset()
    if any(int(b.GetStereo()) and {b.GetBeginAtomIdx(), b.GetEndAtomIdx()} & scope
           for b in mol.GetBonds()):
        return "", frozenset()
    from .branch_rules import collect_subtree
    from .component_mapping import (MappedComponent, decorate_component, mapped_general_branch,
                                    _record_branch_ring_atoms)
    from .formatting import attach_suffix
    from .template_match_utils import match_has_extra_core_bonds

    # Keep the fundamental parent intact. A separate ring joined through one
    # bond is a substituent even if a retained alias includes that ring.
    carbon = set(ring)
    pending = list(ring)
    while pending:
        for atom in mol.GetAtomWithIdx(pending.pop()).GetNeighbors():
            i = atom.GetIdx()
            if i in scope and i not in carbon and atom.GetAtomicNum() == 6 and not atom.IsInRing():
                carbon.add(i)
                pending.append(i)
    carbon = frozenset(carbon)
    decorations, embedded = [], set()
    for i in carbon:
        for bond in mol.GetAtomWithIdx(i).GetBonds():
            j = bond.GetOtherAtomIdx(i)
            if j == host or j in carbon:
                continue
            if bond.GetBondTypeAsDouble() != 1 or j in embedded:
                return "", frozenset()
            atom = mol.GetAtomWithIdx(j)
            if atom.GetAtomicNum() == 8 and atom.GetDegree() == atom.GetTotalNumHs() == 1:
                word, owned = "hydroxy", frozenset({j})
            elif atom.GetAtomicNum() == 8 and atom.GetDegree() == 2 and not atom.IsInRing():
                word, owned = mapped_general_branch(mol, Chem, j, i)
                if not word or owned & (carbon | embedded) or not owned <= scope:
                    return "", frozenset()
            elif atom.GetAtomicNum() == 6 and atom.IsInRing():
                owned = frozenset(collect_subtree(mol, j, i))
                if (owned & (carbon | embedded) or not owned <= scope
                        or len(owned) > 10 or any(int(mol.GetAtomWithIdx(k).GetChiralTag()) for k in owned)):
                    return "", frozenset()
                word, proved = _mapped_steroid_ring_branch(mol, Chem, j, i, owned)
                if not word or proved != owned:
                    return "", frozenset()
            else:
                return "", frozenset()
            decorations.append((i, word))
            embedded.update(owned)
    if carbon | embedded != scope:
        return "", frozenset()

    options = set()
    for parent in _STEMS:
        query, _ = _parent_data(parent)
        if query.GetNumAtoms() != len(carbon):
            continue
        for match in mol.GetSubstructMatches(query, uniquify=False):
            if set(match) != carbon or match_has_extra_core_bonds(mol, query, match):
                continue
            if any(int(atom.GetChiralTag()) and not int(mol.GetAtomWithIdx(i).GetChiralTag())
                   for atom, i in zip(query.GetAtoms(), match, strict=True)):
                continue
            locants = {i: atom.GetProp(_SITE) for atom, i in zip(query.GetAtoms(), match, strict=True)}
            prefixes = tuple((locants[i], word) for i, word in decorations)
            name = decorate_component(mol, Chem, MappedComponent(
                attach_suffix(parent, "yl", (locants[root],)), tuple(locants.items()), carbon,
                prefixes, frozenset(embedded), systematic_parent=True,
            ))
            if name:
                options.add(name)
    if len(options) != 1:
        return "", frozenset()
    name = options.pop()
    _record_branch_ring_atoms(mol, Chem, root, host, name, scope,
                              (i for i in scope if mol.GetAtomWithIdx(i).IsInRing()))
    return name, scope


def _mapped_steroid_ring_branch(mol, Chem, root, host, scope):
    from .mapped_ring_parents import rooted_prefix

    text, atoms = rooted_prefix(mol, Chem, root, host, scope)
    if text:
        return text, atoms
    # A pyrone's exocyclic oxo and its indicated-H parent must be
    # transferred together. Match the complete ring and terminal oxygen.
    if len(scope) != 7:
        return "", frozenset()
    return _pyrone_yl(mol, Chem, root, host, scope)


@lru_cache(maxsize=1)
def _pyrone_queries(Chem):
    return ((Chem.MolFromSmiles("O=c1cccco1"), ("O", "2", "3", "4", "5", "6", "1"), "2H"),
            (Chem.MolFromSmiles("O=c1ccocc1"), ("O", "4", "3", "2", "1", "6", "5"), "4H"))


def _pyrone_yl(mol, Chem, root, host, scope):
    from .template_match_utils import match_has_extra_core_bonds

    if {(i, n.GetIdx()) for i in scope for n in mol.GetAtomWithIdx(i).GetNeighbors()
            if n.GetAtomicNum() > 1 and n.GetIdx() not in scope} != {(root, host)}:
        return "", frozenset()
    words = []
    for query, labels, hydrogen in _pyrone_queries(Chem):
        for match in mol.GetSubstructMatches(query, uniquify=False):
            if set(match) != scope or match_has_extra_core_bonds(mol, query, match):
                continue
            locant = labels[match.index(root)]
            if locant in {"O", "1", hydrogen[0]}:
                continue
            words.append((int(locant), f"{hydrogen[0]}-oxo-{hydrogen}-pyran-{locant}-yl"))
    return (min(words)[1], scope) if words else ("", frozenset())


@lru_cache(maxsize=1)
def _unmodified_candidates():
    from .api import Chem
    rows={}
    for parent in _STEMS:
        for state,word in (('alpha','α'),('beta','β'),('unspecified','ξ')):
            graph=_parent(NaturalProductSpec(parent,state),Chem)
            rows[Chem.MolToSmiles(graph,canonical=True,isomericSmiles=True)]=(parent,'5'+word+'-'+parent,graph)
    return rows


def _edit(graph,operation,Chem):
    """Transfer each old tetrahedron through one literal neighbour exchange."""
    original={}
    for atom in graph.GetAtoms():
        if atom.GetChiralTag() not in {Chem.ChiralType.CHI_TETRAHEDRAL_CW,Chem.ChiralType.CHI_TETRAHEDRAL_CCW}:continue
        reference=tuple(n.GetProp(_SITE) for n in atom.GetNeighbors())
        if len(reference)==3 and atom.GetTotalNumHs()==1:reference+=(None,)
        if len(reference)!=4:raise _Scope('An original tetrahedron lacks four committed ports.')
        original[atom.GetProp(_SITE)]=(atom.GetChiralTag(),reference)
    editable=Chem.RWMol(graph);operation(editable);result=editable.GetMol()
    for atom in result.GetAtoms():atom.SetNumExplicitHs(0);atom.SetNoImplicit(False)
    result.UpdatePropertyCache(strict=True)
    from .glycosyl_construction import _reordered_tag
    for atom in result.GetAtoms():
        old=original.get(atom.GetProp(_SITE))
        if old is None:continue
        actual=tuple(n.GetProp(_SITE) for n in atom.GetNeighbors())
        if len(actual)==3 and atom.GetTotalNumHs()==1:actual+=(None,)
        if len(actual)!=4:
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED);continue
        tag,reference=old;removed=set(reference)-set(actual);added=set(actual)-set(reference)
        if len(removed)!=len(added) or len(removed)>1:raise _Scope('Several original tetrahedral ports change in one operation.')
        if removed:
            previous=next(iter(removed));replacement=next(iter(added));reference=tuple(replacement if i==previous else i for i in reference)
        atom.SetChiralTag(_reordered_tag(tag,reference,actual,Chem))
    Chem.SanitizeMol(result)
    return result


def _pair(value):
    if not isinstance(value,tuple) or len(value)!=2 or any(not isinstance(s,str) or not s for s in value) or value[0]==value[1]:
        raise _Input('An edge operation needs two distinct original string locants.')
    return tuple(sorted(value,key=_key))


def _edge(graph,pair):
    mapping=_sites(graph)
    if any(site not in mapping for site in pair):raise _Input('An operation refers to a removed or nonexistent atom.')
    return graph.GetBondBetweenAtoms(*(mapping[s] for s in pair))


def _delete_nor(graph,locants,Chem):
    deleted=set(locants);mapping=_sites(graph)
    if not deleted<=mapping.keys():raise _Input('A nor site is absent.')
    if any(graph.GetAtomWithIdx(mapping[s]).GetAtomicNum()!=6 or graph.GetAtomWithIdx(mapping[s]).GetDegree()>2 for s in deleted):
        raise _Scope('Nor requires an unsubstituted degree-one/two carbon in this saturated scope.')
    # Delete a component of adjacent atoms as one path; its at-most-two
    # surviving endpoints are joined. Terminal deletions have one endpoint.
    components=[];pending=set(deleted)
    while pending:
        part=set();stack=[pending.pop()]
        while stack:
            site=stack.pop();part.add(site)
            for neighbor in graph.GetAtomWithIdx(mapping[site]).GetNeighbors():
                other=neighbor.GetProp(_SITE)
                if other in pending:pending.remove(other);stack.append(other)
        outside={n.GetProp(_SITE) for site in part for n in graph.GetAtomWithIdx(mapping[site]).GetNeighbors() if n.GetProp(_SITE) not in deleted}
        if not 1<=len(outside)<=2:raise _Scope('The deleted nor path has no valid surviving endpoint pair.')
        if len(outside)==2 and _edge(graph,tuple(outside)) is not None:raise _Scope('Nor would collapse an existing ring edge.')
        components.append(outside)
    rank=graph.GetNumBonds()-graph.GetNumAtoms()+1
    def operation(editable):
        for idx in sorted((mapping[s] for s in deleted),reverse=True):editable.RemoveAtom(idx)
        current=_sites(editable)
        for outside in components:
            if len(outside)==2:
                a,b=outside;editable.AddBond(current[a],current[b],Chem.BondType.SINGLE)
    result=_edit(graph,operation,Chem)
    if len(Chem.GetMolFrags(result))!=1 or result.GetNumBonds()-result.GetNumAtoms()+1!=rank:raise _Scope('Nor does not preserve the connected ring count.')
    return result


def _homo_edge(graph,site):
    mapping=_sites(graph);compound=re.fullmatch(r'(\d+)\((\d+)\)',site)
    if compound:
        pair=tuple(sorted(compound.groups(),key=_key));bond=_edge(graph,pair)
        if site!=pair[0]+'('+pair[1]+')':raise _Input('A compound homo site cites the lower endpoint first.')
        if bond is None or any(graph.GetAtomWithIdx(mapping[s]).GetDegree()<3 for s in pair):
            raise _Scope('A compound homo site must identify an existing bond connector.')
        return pair,site
    if site not in mapping:raise _Input('A homo connector site is absent.')
    atom=graph.GetAtomWithIdx(mapping[site])
    if atom.GetDegree()!=2 or not atom.IsInRing():raise _Scope('This simple homo site requires a saturated cyclic atomic connector.')
    component=set();stack=[site]
    while stack:
        current=stack.pop()
        if current in component:continue
        component.add(current)
        stack.extend(n.GetProp(_SITE) for n in graph.GetAtomWithIdx(mapping[current]).GetNeighbors() if n.GetDegree()==2)
    if site!=max(component,key=_key):raise _Scope('A simple homo locant must follow the highest numbered atom of its atomic connector.')
    outside=[n.GetProp(_SITE) for n in atom.GetNeighbors() if n.GetProp(_SITE) not in component]
    if len(outside)!=1:raise _Scope('The highest connector atom has no unique following junction.')
    return (site,outside[0]),site


def _insert_homo(graph,homo,Chem):
    pair,stem=_homo_edge(graph,homo.site);mapping=_sites(graph);inserted=tuple(stem+chr(97+i) for i in range(homo.count))
    if set(inserted)&mapping.keys():raise _Input('A homo inserted locant is already occupied.')
    def operation(editable):
        editable.RemoveBond(*(mapping[s] for s in pair));previous=mapping[pair[0]]
        for site in inserted:
            atom=Chem.Atom(6);atom.SetProp(_SITE,site);idx=editable.AddAtom(atom)
            editable.AddBond(previous,idx,Chem.BondType.SINGLE);previous=idx
        editable.AddBond(previous,mapping[pair[1]],Chem.BondType.SINGLE)
    return _edit(graph,operation,Chem),inserted


def _remove_set(graph,deleted,Chem):
    mapping=_sites(graph)
    def operation(editable):
        for idx in sorted((mapping[s] for s in deleted),reverse=True):editable.RemoveAtom(idx)
    result=_edit(graph,operation,Chem)
    if not result.GetNumAtoms() or len(Chem.GetMolFrags(result))!=1:raise _Scope('Removal must retain one nonempty connected parent structure.')
    return result


def _apo_set(graph,site):
    mapping=_sites(graph)
    if site not in mapping:raise _Input('An apo side-chain site is absent.')
    origin=graph.GetAtomWithIdx(mapping[site])
    if origin.IsInRing():raise _Scope('This apo scope requires an original acyclic side-chain atom.')
    distance={a.GetIdx():0 for a in graph.GetAtoms() if a.IsInRing()};queue=deque(distance)
    while queue:
        idx=queue.popleft()
        for neighbor in graph.GetAtomWithIdx(idx).GetNeighbors():
            other=neighbor.GetIdx()
            if other not in distance:distance[other]=distance[idx]+1;queue.append(other)
    forward=[a.GetIdx() for a in origin.GetNeighbors() if distance[a.GetIdx()]>distance[origin.GetIdx()]]
    if not forward:raise _Scope('No side-chain atom lies beyond the apo locant.')
    deleted=set();stack=forward[:]
    while stack:
        idx=stack.pop()
        if idx==origin.GetIdx() or idx in deleted:continue
        deleted.add(idx);stack.extend(a.GetIdx() for a in graph.GetAtomWithIdx(idx).GetNeighbors())
    return {graph.GetAtomWithIdx(i).GetProp(_SITE) for i in deleted}


def _set_RS(graph,site,descriptor,Chem):
    from rdkit.Chem import rdCIPLabeler
    mapping=_sites(graph)
    if site not in mapping:raise _Input('A stereochemical site is absent.')
    atom=graph.GetAtomWithIdx(mapping[site])
    if descriptor=='unspecified':atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED);return
    for tag in (Chem.ChiralType.CHI_TETRAHEDRAL_CW,Chem.ChiralType.CHI_TETRAHEDRAL_CCW):
        atom.SetChiralTag(tag)
        if atom.HasProp('_CIPCode'):atom.ClearProp('_CIPCode')
        rdCIPLabeler.AssignCIPLabels(graph,maxRecursiveIterations=1000000)
        if atom.HasProp('_CIPCode') and atom.GetProp('_CIPCode')==descriptor:return
    raise _Scope('The requested R/S configuration is not established on the modified original graph.')


def _prefix(locants,word):
    return ','.join(sorted(locants,key=_key))+'-'+('' if len(locants)==1 else MULT[len(locants)])+word


def _failure(status,reason):
    return NaturalProductNamingResult('',status,'not-assessed',
        (SelectionLimitation('natural-product-construction','P-101',reason),),'explicit-natural-product')


def natural_product_to_iupac_result(spec:NaturalProductSpec) -> NaturalProductNamingResult:
    from .api import Chem
    if Chem is None:return _failure('unavailable','RDKit is unavailable.')
    try:
        if (not isinstance(spec,NaturalProductSpec) or not isinstance(spec.parent,str) or spec.parent not in _STEMS
                or not isinstance(spec.c5_configuration,str) or spec.c5_configuration not in {'alpha','beta','unspecified'}):
            raise _Input('Declare a covered fundamental steroid and explicit alpha/beta/unspecified C5 state.')
        fields=('nor','homo','apo','seco','cyclo','abeo','des','replacements','double_bonds','epimers','stereochemistry')
        if any(not isinstance(getattr(spec,key),tuple) for key in fields):raise _Input('Natural-product operations must be immutable tuples.')
        for key in ('nor','apo','epimers'):
            values=getattr(spec,key)
            if any(not isinstance(s,str) or not s for s in values) or len(set(values))!=len(values):raise _Input('Original operation locants must be distinct nonempty strings.')
        if len(spec.des)>1 or any(s not in {'A','D'} for s in spec.des):raise _Scope('One terminal steroid ring A or D is covered by this des operation.')
        if spec.des and any(getattr(spec,key) for key in ('nor','homo','apo','seco','cyclo','abeo')):
            raise _Scope('Combining des with other skeletal modifications requires separate source ordering.')
        if any(not isinstance(h,NaturalProductHomo) or not isinstance(h.site,str) or isinstance(h.count,bool) or not isinstance(h.count,int) or not 1<=h.count<=26 for h in spec.homo):raise _Input('Declare a source homo site and one-to-twenty-six inserted carbon atoms.')
        if len({h.site for h in spec.homo})!=len(spec.homo):raise _Input('Combine repeated homo operations at one source site.')
        if any(not isinstance(r,NaturalProductReplacement) or not isinstance(r.site,str) or r.element not in {'N','O','S','Si','Ge','Se','Te'} for r in spec.replacements):raise _Input('Declare an original ordinary-valence replacement site and covered element.')
        if len({r.site for r in spec.replacements})!=len(spec.replacements):raise _Input('A skeletal site cannot be replaced twice.')
        seco=tuple(sorted((_pair(p) for p in spec.seco),key=lambda p:tuple(map(_key,p))))
        cyclo=tuple(sorted((_pair(p) for p in spec.cyclo),key=lambda p:tuple(map(_key,p))))
        doubles=tuple(sorted((_pair(p) for p in spec.double_bonds),key=lambda p:tuple(map(_key,p))))
        if any(len(set(values))!=len(values) for values in (seco,cyclo,doubles)):raise _Input('Do not repeat an edge operation.')
        if set(seco)&set(cyclo):raise _Scope('Cancelling cyclo/seco edits are not a valid minimal construction.')
        if len(spec.abeo)>1 or spec.abeo and any(getattr(spec,k) for k in ('nor','homo','apo','seco','cyclo','des')):
            raise _Scope('Abeo combination ordering needs a separate source comparison.')
        for triplet in spec.abeo:
            if not isinstance(triplet,tuple) or len(triplet)!=3 or any(not isinstance(s,str) or not s for s in triplet) or len(set(triplet))!=3:
                raise _Input('Abeo requires distinct stationary, old and new endpoint locants.')
        for row in spec.stereochemistry:
            if not isinstance(row,tuple) or len(row)!=2 or not isinstance(row[0],str) or row[1] not in {'R','S','unspecified'}:raise _Input('Declare a mapped R/S or unspecified stereochemical site.')
        if len({s for s,d in spec.stereochemistry})!=len(spec.stereochemistry):raise _Input('A stereochemical site cannot be declared twice.')
        graph=_parent(spec,Chem);deleted=set();inserted=[]
        if spec.des:
            private={'A':{'1','2','3','4'},'D':{'15','16','17'}}[spec.des[0]]
            # Side chains belonging exclusively to the removed terminal ring
            # are removed with it, never left as disconnected components.
            if spec.des[0]=='D':
                original=_sites(graph);stack=[original['17']];found=set()
                while stack:
                    idx=stack.pop()
                    if idx in found:continue
                    found.add(idx)
                    stack.extend(n.GetIdx() for n in graph.GetAtomWithIdx(idx).GetNeighbors()
                        if n.GetProp(_SITE) not in {'13','14','18'})
                private|={graph.GetAtomWithIdx(i).GetProp(_SITE) for i in found}
            graph=_remove_set(graph,private,Chem);deleted|=private
        if spec.nor:graph=_delete_nor(graph,spec.nor,Chem);deleted.update(spec.nor)
        for homo in sorted(spec.homo,key=lambda h:_key(h.site)):
            graph,added=_insert_homo(graph,homo,Chem);inserted.extend(added)
        for site in sorted(spec.apo,key=_key):
            removed=_apo_set(graph,site);graph=_remove_set(graph,removed,Chem);deleted|=removed
        opened=set()
        for pair in seco:
            bond=_edge(graph,pair)
            if bond is None or not bond.IsInRing():raise _Scope('Seco must cleave an existing ring bond.')
            def operation(editable,pair=pair):
                mapping=_sites(editable);editable.RemoveBond(*(mapping[s] for s in pair))
            graph=_edit(graph,operation,Chem);opened.update(pair)
            if len(Chem.GetMolFrags(graph))!=1:raise _Scope('Seco cannot split this explicit parent into multiple components.')
        for pair in cyclo:
            bond=_edge(graph,pair);mapping=_sites(graph)
            if bond is not None or any(graph.GetAtomWithIdx(mapping[s]).GetTotalNumHs()<1 for s in pair):raise _Scope('Cyclo requires two nonadjacent sites with replaceable hydrogen.')
            def operation(editable,pair=pair):
                current=_sites(editable);editable.AddBond(*(current[s] for s in pair),Chem.BondType.SINGLE)
            graph=_edit(graph,operation,Chem)
        for stationary,old,new in spec.abeo:
            if _edge(graph,(stationary,old)) is None or _edge(graph,(stationary,new)) is not None:raise _Scope('Abeo requires an existing old bond and an unbonded new endpoint.')
            mapping=_sites(graph)
            if graph.GetAtomWithIdx(mapping[new]).GetTotalNumHs()<1:raise _Scope('The new abeo endpoint has no available hydrogen.')
            def operation(editable,a=stationary,b=old,c=new):
                current=_sites(editable);editable.RemoveBond(current[a],current[b]);editable.AddBond(current[a],current[c],Chem.BondType.SINGLE)
            graph=_edit(graph,operation,Chem);opened.update((old,new))
            if len(Chem.GetMolFrags(graph))!=1:raise _Scope('The endpoint migration disconnects the parent.')
        # Modifications precede ordinary-valence skeletal replacement.
        mapping=_sites(graph)
        for replacement in spec.replacements:
            if replacement.site not in mapping:raise _Input('A replacement site is absent.')
            atom=graph.GetAtomWithIdx(mapping[replacement.site])
            if int(atom.GetChiralTag()) or atom.GetDegree()>2:raise _Scope('Replacement at a stereogenic/substituted junction needs a joint valence/configuration writer.')
            atom.SetAtomicNum(Chem.GetPeriodicTable().GetAtomicNumber(replacement.element))
        graph.UpdatePropertyCache(strict=True);Chem.SanitizeMol(graph)
        double_locs=[]
        for pair in doubles:
            bond=_edge(graph,pair)
            if bond is None or bond.GetBondType()!=Chem.BondType.SINGLE:raise _Scope('Unsaturation needs an existing original single bond.')
            mapping=_sites(graph)
            if any(graph.GetAtomWithIdx(mapping[s]).GetAtomicNum()!=6 or graph.GetAtomWithIdx(mapping[s]).GetTotalNumHs()<1 for s in pair):raise _Scope('The original carbon bond has no complete dehydrogenation budget.')
            if not bond.IsInRing():raise _Scope('New acyclic E/Z configuration needs an explicit original-bond writer.')
            bond.SetBondType(Chem.BondType.DOUBLE)
            for site in pair:graph.GetAtomWithIdx(mapping[site]).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            first,second=sorted(pair,key=_key)
            double_locs.append(first if first.isdigit() and second.isdigit() and int(second)==int(first)+1 else first+'('+second+')')
            graph.UpdatePropertyCache(strict=True)
        Chem.SanitizeMol(graph)
        mapping=_sites(graph)
        for site in spec.epimers:
            if site not in mapping:raise _Input('An epimer site is absent.')
            atom=graph.GetAtomWithIdx(mapping[site])
            if atom.GetChiralTag() not in {Chem.ChiralType.CHI_TETRAHEDRAL_CW,Chem.ChiralType.CHI_TETRAHEDRAL_CCW}:raise _Scope('An epi site has no established fundamental configuration.')
            atom.InvertChirality()
        from rdkit.Chem import rdCIPLabeler
        rdCIPLabeler.AssignCIPLabels(graph,maxRecursiveIterations=1000000)
        for site,descriptor in spec.stereochemistry:
            if site=='5':raise _Scope('Declare C5 through its source alpha/beta/unspecified field or an explicit epimer operation.')
            if site in mapping:
                atom=graph.GetAtomWithIdx(mapping[site])
                if atom.HasProp('_CIPCode') and atom.GetProp('_CIPCode') in {'R','S'} and descriptor in {'R','S'} and atom.GetProp('_CIPCode')!=descriptor:
                    raise _Scope('An existing fundamental configuration must be inverted with the mapped epi operation.')
            _set_RS(graph,site,descriptor,Chem)
        # Deleting an angular carbon can make its old quaternary neighbour
        # the first atom in the native graph. Commit its new H count before
        # mapped SMILES traversal: an implicit H at that root otherwise has
        # a different native first-atom stereo interpretation on reparse.
        for atom in graph.GetAtoms():
            if int(atom.GetChiralTag()) and atom.GetNumImplicitHs():
                atom.SetNumExplicitHs(atom.GetTotalNumHs());atom.SetNoImplicit(True)
        graph.UpdatePropertyCache(strict=True)
        Chem.AssignStereochemistry(graph,cleanIt=True,force=True)
        rdCIPLabeler.AssignCIPLabels(graph,maxRecursiveIterations=1000000)
        potential=Chem.FindMolChiralCenters(graph,includeUnassigned=True,useLegacyImplementation=False)
        unknown=tuple(sorted((graph.GetAtomWithIdx(i).GetProp(_SITE) for i,code in potential if code=='?'),key=_key))
        c5=''
        if '5' in mapping and any(i==mapping['5'] for i,code in potential):c5='5'+{'alpha':'α','beta':'β','unspecified':'ξ'}[spec.c5_configuration]+'-'
        if spec.double_bonds:
            n=len(double_locs);stem=spec.parent[:-3]
            parent=stem+('' if n==1 else 'a')+'-'+','.join(sorted(double_locs,key=_key))+'-'+('' if n==1 else MULT[n])+'ene'
        else:parent=spec.parent
        name=c5+parent
        for values,word in ((spec.nor,'nor'),(tuple(inserted),'homo'),(spec.apo,'apo')):
            if values:name=_prefix(values,word)+'-'+name
        if seco:name=':'.join(','.join(p) for p in seco)+'-'+('' if len(seco)==1 else MULT[len(seco)])+'seco-'+name
        if cyclo:name=':'.join(','.join(p) for p in cyclo)+'-'+('' if len(cyclo)==1 else MULT[len(cyclo)])+'cyclo-'+name
        if spec.abeo:name=':'.join(a+'('+b+'→'+c+')' for a,b,c in spec.abeo)+'-abeo-'+name
        if spec.des:name='des-'+spec.des[0]+'-'+name
        if spec.epimers:name=_prefix(spec.epimers,'epi')+'-'+name
        if spec.replacements:
            vocabulary={'O':(0,'oxa'),'S':(1,'thia'),'Se':(2,'selena'),'Te':(3,'tellura'),'N':(4,'aza'),'Si':(5,'sila'),'Ge':(6,'germa')}
            groups=defaultdict(list)
            for r in spec.replacements:groups[r.element].append(r.site)
            name='-'.join(_prefix(groups[e],vocabulary[e][1]) for e in sorted(groups,key=lambda e:vocabulary[e][0]))+'-'+name
        extra=dict(spec.stereochemistry)
        for site in opened:
            if site in mapping:
                atom=graph.GetAtomWithIdx(mapping[site])
                if atom.HasProp('_CIPCode') and atom.GetProp('_CIPCode') in {'R','S'}:extra.setdefault(site,atom.GetProp('_CIPCode'))
        if extra:name='('+','.join(site+('Ξ' if extra[site]=='unspecified' else extra[site]) for site in sorted(extra,key=_key))+')-'+name
        operation_count=len(spec.nor)+sum(h.count for h in spec.homo)+len(spec.apo)+len(seco)+len(cyclo)+len(spec.abeo)+len(spec.des)
        issues=[]
        modified=any(getattr(spec,key) for key in fields)
        if modified:issues.append(SelectionLimitation('natural-product-parent-comparison','P-101.3.7.1','The supplied construction has not completed the global fewest-operation and detachable-prefix comparison.'))
        if unknown:issues.append(SelectionLimitation('natural-product-configuration','P-101.2.6/P-101.8','Original or new stereogenic sites remain unspecified: '+','.join(unknown)))
        identity=Chem.MolToSmiles(graph,canonical=True,isomericSmiles=True)
        constructed_name=name;selected_parent=spec.parent;selected_mapping=();minimum='not-compared' if modified else 'unmodified-parent'
        match=_unmodified_candidates().get(identity)
        if match is not None:
            selected_parent,name,template=match
            correspondence=graph.GetSubstructMatch(template,useChirality=True)
            if len(correspondence)!=graph.GetNumAtoms():raise _Scope('The exact unmodified parent has no complete original atom correspondence.')
            selected_mapping=tuple((a.GetProp(_SITE),correspondence[a.GetIdx()]+1) for a in template.GetAtoms())
            minimum='exact-unmodified-parent'
            issues=[issue for issue in issues if issue.stage!='natural-product-parent-comparison']
        for atom in graph.GetAtoms():atom.SetAtomMapNum(atom.GetIdx()+1)
        mapped=Chem.MolToSmiles(graph,canonical=True,isomericSmiles=True);restored=Chem.MolFromSmiles(mapped)
        if restored is None:raise _Scope('The modified graph cannot be serialized with its original atom roles.')
        for atom in restored.GetAtoms():atom.SetAtomMapNum(0)
        if Chem.MolToSmiles(restored,canonical=True,isomericSmiles=True)!=identity:raise _Scope('Mapped serialization changes the original isomeric graph.')
        from rdkit.Chem.rdMolDescriptors import CalcMolFormula
        return NaturalProductNamingResult(name,'available','not-assessed',route='explicit-natural-product',canonical_smiles=identity,mapped_smiles=mapped,
            requested_parent=spec.parent,selected_parent=selected_parent,constructed_name=constructed_name,
            parent_atom_maps=tuple(sorted(((a.GetProp(_SITE),a.GetAtomMapNum()) for a in graph.GetAtoms()),key=lambda row:_key(row[0]))),
            selected_parent_atom_maps=tuple(sorted(selected_mapping,key=lambda row:_key(row[0]))),
            deleted_locants=tuple(sorted(deleted,key=_key)),inserted_locants=tuple(sorted(inserted,key=_key)),skeletal_operation_count=operation_count,
            cycle_rank=graph.GetNumBonds()-graph.GetNumAtoms()+1,formula=CalcMolFormula(graph),construction_limitations=tuple(issues),
            minimum_operation_status=minimum,unassigned_stereo_locants=unknown)
    except _Input as error:return _failure('invalid-input',str(error))
    except (_Scope,ValueError,RuntimeError,OverflowError,TypeError,KeyError) as error:return _failure('unavailable',str(error))


def natural_product_to_iupac(spec:NaturalProductSpec) -> str:return natural_product_to_iupac_result(spec).name
