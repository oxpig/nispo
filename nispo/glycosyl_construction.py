"""Source glycosyl components and original-ligand oxygen joins."""
from __future__ import annotations

from .carbohydrates import (
    CarbohydrateSpec,_SugarInput,_SugarScope,_parent_facts,_parent_word,_build_sugar,_reordered_tag,
)
from .carbohydrate_modifications import (
    _ROLE,_OWNER,_atom,_validate,_modify,_derived_parent,_name,
)


def sugar_component(spec,interpretation,Chem,*,context,substance_anomer=None,additional_contexts=()):
    if (not isinstance(spec,CarbohydrateSpec) or not isinstance(spec.ring,str)
            or spec.ring not in {'furanose','pyranose'} or not isinstance(spec.modifications,tuple)
            or interpretation not in {'alpha','beta','unspecified'} or substance_anomer not in {None,'alpha,beta'}):
        raise _SugarInput('The glycosyl component requires one cyclic sugar and an explicit anomer interpretation.')
    count,carbonyl,sides=_parent_facts(spec.parent);_parent_word(count,carbonyl,sides,spec.ring)
    _validate(spec.modifications,count,prefix_context=True)
    base,closure=_build_sugar(count,carbonyl,sides,spec.ring,interpretation,Chem)
    for atom in base.GetAtoms():
        idx=atom.GetIdx();atom.SetProp(_ROLE,str(idx+1) if idx<count else 'O'+str(idx-count+1))
    graph,terms,esters,phosphates=_modify(base,count,carbonyl,spec.modifications,Chem)
    parent,remaining,anomer,terms=_derived_parent(count,carbonyl,sides,spec.ring,interpretation,spec.modifications,terms)
    name=_name(parent,substance_anomer or anomer,terms,esters,phosphates,spec.modifications,base,count,carbonyl,Chem,context=context)
    citation_name=None
    if context=='glycosyl':
        citation_parent=parent[2:] if parent.startswith(('D-','L-')) else parent
        citation_name=_name(citation_parent,'unspecified',terms,esters,phosphates,spec.modifications,base,count,carbonyl,Chem,context=context)
    oxygen=_atom(graph,'O'+str(carbonyl))
    if oxygen.GetAtomicNum()!=8 or oxygen.GetDegree()!=1 or oxygen.GetTotalNumHs()!=1:
        raise _SugarScope('The anomeric site has no available oxygen OH for this glycosyl condensation.')
    words={context:name}
    for other in additional_contexts:
        words[other]=_name(parent,substance_anomer or anomer,terms,esters,phosphates,spec.modifications,base,count,carbonyl,Chem,context=other)
    return graph,name,dict(count=count,carbonyl=carbonyl,source_sides=sides,remaining_sides=remaining,
        anomer=anomer,closure=closure,phosphate_sites=tuple(p for p,_idx,_acids in phosphates),citation_name=citation_name,
        context_names=words,parent_name=parent)


def join_alcohol_oxygen(sugar,carbonyl,alcohol,oxygen_index,Chem):
    """Keep the alcohol oxygen; replace the sugar OH ligand with its port.

    Returns final joined graph and C-anomer / bridge-O / aglycone-C indices.
    The native old/new ordered-neighbour permutation proves configuration.
    """
    graph=Chem.RWMol(sugar);anomer=_atom(graph,str(carbonyl));leaving=_atom(graph,'O'+str(carbonyl))
    removed=leaving.GetIdx();original_tag=anomer.GetChiralTag();anomer_index=anomer.GetIdx()
    neighbors=tuple(n.GetIdx() for n in anomer.GetNeighbors())
    if leaving.GetAtomicNum()!=8 or leaving.GetDegree()!=1 or leaving.GetTotalNumHs()!=1:
        raise _SugarInput('The glycosyl anomeric OH is unavailable.')
    piece=Chem.Mol(alcohol);oxygen=piece.GetAtomWithIdx(oxygen_index)
    if oxygen.GetAtomicNum()!=8 or oxygen.GetDegree()!=1 or oxygen.GetTotalNumHs()!=1:
        raise _SugarInput('The aglycone site requires one available OH.')
    carbon=oxygen.GetNeighbors()[0]
    if carbon.GetAtomicNum()!=6:raise _SugarInput('The alcohol OH is not attached to an organic carbon.')
    for atom in graph.GetAtoms():atom.SetProp(_OWNER,'sugar:'+atom.GetProp(_OWNER))
    for atom in piece.GetAtoms():atom.SetProp(_OWNER,'aglycone')
    if oxygen.GetNumExplicitHs():oxygen.SetNumExplicitHs(oxygen.GetNumExplicitHs()-1)
    offset=graph.GetNumAtoms();bridge=offset+oxygen_index;root=offset+carbon.GetIdx()
    reference=tuple(bridge if i==removed else i for i in neighbors)
    graph.InsertMol(piece);graph.RemoveAtom(removed)
    translate=lambda i:i-int(i>removed)
    anomer_index=translate(anomer_index);bridge=translate(bridge);root=translate(root)
    reference=tuple(translate(i) for i in reference)
    graph.AddBond(anomer_index,bridge,Chem.BondType.SINGLE);graph.UpdatePropertyCache(strict=False)
    atom=graph.GetAtomWithIdx(anomer_index);actual=tuple(n.GetIdx() for n in atom.GetNeighbors())
    if len(reference)==3:reference+=(None,);actual+=(None,)
    if original_tag!=Chem.ChiralType.CHI_UNSPECIFIED:
        atom.SetChiralTag(_reordered_tag(original_tag,reference,actual,Chem))
    else:atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    result=graph.GetMol();Chem.SanitizeMol(result);Chem.AssignStereochemistry(result,cleanIt=True,force=True)
    from rdkit.Chem import rdCIPLabeler
    rdCIPLabeler.AssignCIPLabels(result,maxRecursiveIterations=1000000)
    return result,(anomer_index,bridge,root)
