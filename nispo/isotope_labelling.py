"""Explicit P-83 labelled-substance semantics and mapped descriptor writing."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal

from .input_stereochemistry import unsupported_relative_stereochemistry
from .naming_result import NamingResult, SelectionLimitation


@dataclass(frozen=True, slots=True)
class Nuclide:
    element: str
    mass: int


@dataclass(frozen=True, slots=True)
class IsotopeLabelling:
    """A labelled substance, distinct from a fixed isotopic molecular graph.

    Heavy-atom maps align precursor/forms. Specific/selective inputs supply
    labelled isotopologues; the unmodified precursor is implicitly present.
    Other modes declare nuclides and an undefined distribution/depletion, or
    all-position G/U labelling. U may instead declare a subset of atom maps.
    No mixing proportions or enrichment levels are invented.
    """
    unmodified_smiles: str
    kind: Literal['specific','selective','nonselective','deficient','general','uniform']
    isotopologues: tuple[str,...] = ()
    nuclides: tuple[Nuclide,...] = ()
    subset_atom_maps: tuple[int,...] = ()


@dataclass(frozen=True, slots=True)
class LabelledSite:
    atom_map: int
    element: str
    mass: int
    maximum_count: int


@dataclass(frozen=True, slots=True)
class IsotopeLabellingNamingResult(NamingResult):
    kind: str = ""
    unmodified_smiles: str = ""
    isotopologues: tuple[str,...] = ()
    labelled_sites: tuple[LabelledSite,...] = ()
    includes_unmodified: bool | None = None
    position_ratios: str = "unspecified"


def _failure(status,reason):
    return IsotopeLabellingNamingResult(name='', generation_status=status,
        preferred_status='unresolved',
        unresolved=(SelectionLimitation('isotope-labelling','P-83',reason),),
        route='explicit-isotope-labelling')


def _unmodified_identity(mol,Chem):
    copy=Chem.Mol(mol)
    for atom in copy.GetAtoms():
        atom.SetIsotope(0)
        if atom.GetAtomicNum()==1:
            atom.SetAtomMapNum(0)
    Chem.SanitizeMol(copy)
    copy=Chem.RemoveHs(copy)
    Chem.AssignStereochemistry(copy,cleanIt=True,force=True)
    return Chem.MolToSmiles(copy,canonical=True,isomericSmiles=True)


def _sites(mol):
    sites=Counter()
    for atom in mol.GetAtoms():
        if not atom.GetIsotope():
            continue
        if atom.GetAtomicNum()==1:
            if atom.GetDegree()!=1:
                return None
            host=atom.GetNeighbors()[0]
            if host.GetAtomicNum()!=6:
                return None  # O/N-bound labelled hydrogen needs a local part.
            key=(host.GetAtomMapNum(),'H',atom.GetIsotope())
        else:
            key=(atom.GetAtomMapNum(),atom.GetSymbol(),atom.GetIsotope())
        sites[key]+=1
    return sites


def _mapped_chain_component(mol,Chem,kind,subset):
    """One complete ordinary carbon-chain component with acid/OH/halogens.

    All carbon atoms form one simple saturated path. Terminal COOH outranks
    alcohol; every OH is otherwise a suffix. All explicit H atoms retain the
    locant of their carbon host. Other labelled components decline.
    """
    if (mol.GetRingInfo().NumRings() or mol.GetStereoGroups()
            or any(a.GetFormalCharge() or a.GetNumRadicalElectrons()
                   or a.GetAtomicNum() not in {1,6,8,9,17,35,53} for a in mol.GetAtoms())):
        return None
    carbon=frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum()==6)
    from .constants import ROOTS
    if len(carbon) not in ROOTS:
        return None
    graph={i:frozenset(n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in carbon) for i in carbon}
    if any(len(values)>2 for values in graph.values()) or sum(map(len,graph.values()))!=2*(len(carbon)-1):
        return None
    if any(mol.GetBondBetweenAtoms(i,j).GetBondType()!=Chem.BondType.SINGLE for i in carbon for j in graph[i]):
        return None
    ends=tuple(i for i in carbon if len(graph[i])<=1)
    if not ends:
        return None
    path=[];previous=None;current=ends[0]
    while current is not None:
        path.append(current)
        onward=graph[current]-{previous}
        previous,current=current,(next(iter(onward)) if onward else None)
    if set(path)!=set(carbon):
        return None
    acid_hosts=[];hydroxy=[];halogens=[];bound={}
    for atom in mol.GetAtoms():
        idx=atom.GetIdx();number=atom.GetAtomicNum()
        if number==6:
            if atom.GetTotalValence()!=4:
                return None
            continue
        if atom.GetDegree()!=1 or atom.GetNeighbors()[0].GetIdx() not in carbon:
            return None
        host=atom.GetNeighbors()[0].GetIdx();bound[idx]=host
        order=mol.GetBondBetweenAtoms(idx,host).GetBondTypeAsDouble()
        if number==1:
            if not atom.GetIsotope() or order!=1:
                return None
        elif number==8:
            if atom.GetIsotope():
                return None
            if order==2:
                other=[n for n in mol.GetAtomWithIdx(host).GetNeighbors()
                       if n.GetAtomicNum()==8 and n.GetIdx()!=idx
                       and mol.GetBondBetweenAtoms(host,n.GetIdx()).GetBondTypeAsDouble()==1
                       and n.GetDegree()==1 and n.GetTotalNumHs(includeNeighbors=True)==1]
                if len(other)!=1 or host not in ends:
                    return None
                acid_hosts.append(host)
            elif order==1 and atom.GetTotalNumHs(includeNeighbors=True)==1:
                hydroxy.append(host)
            else:
                return None
        elif order==1 and atom.GetTotalNumHs(includeNeighbors=True)==0 and not atom.GetIsotope():
            halogens.append((host,{9:'fluoro',17:'chloro',35:'bromo',53:'iodo'}[number]))
        else:
            return None
    if len(acid_hosts)>1:
        return None
    for host in acid_hosts:
        hydroxy.remove(host)
    from .component_mapping import MappedComponent,decorate_component
    from .formatting import attach_suffix,multiplied_suffix,prefix_alphabetization_key
    from .parent_annotations import isotope_numbering_key,stereo_numbering
    candidates=[]
    for seq in (tuple(path),tuple(reversed(path))):
        mapping={idx:str(i) for i,idx in enumerate(seq,1)}
        if acid_hosts and mapping[acid_hosts[0]]!='1':
            continue
        suffix_locs=tuple(sorted(int(mapping[i]) for i in hydroxy))
        prefixes=[(mapping[i],word) for i,word in halogens]
        parent=ROOTS[len(seq)]+'ane'
        if acid_hosts:
            parent=({1:'formic acid',2:'acetic acid'}.get(len(seq))
                    or attach_suffix(parent,'oic acid'))
            prefixes += [(mapping[i],'hydroxy') for i in hydroxy]
        elif hydroxy:
            parent=attach_suffix(parent,multiplied_suffix(len(hydroxy),'ol'),
                () if len(seq)==1 and len(hydroxy)==1 else [str(i) for i in suffix_locs])
        prefix_locs=tuple(sorted(int(loc) for loc,_ in prefixes))
        citation=tuple(int(loc) for loc,_ in sorted(prefixes,key=lambda row:(prefix_alphabetization_key(row[1]),int(row[0]))))
        chemical=(suffix_locs if not acid_hosts else (),prefix_locs,citation)
        for atom,host in bound.items():
            source=mol.GetAtomWithIdx(atom)
            mapping[atom]=('O'+mapping[host] if source.GetAtomicNum()==8
                           and source.GetTotalNumHs(includeNeighbors=True) else mapping[host])
        isotope=isotope_numbering_key(mol,mapping)
        stereo=stereo_numbering(mol,mapping)
        if isotope is None or stereo is None:
            return None
        component=MappedComponent(parent,tuple(mapping.items()),frozenset(mapping),tuple(prefixes),
            prefix_mode='unlocanted' if len(seq)==1 else 'locanted',
            isotope_labelling_kind=kind,isotope_labelling_subset=subset)
        name=decorate_component(mol,Chem,component)
        if not name:
            return None
        candidates.append(((chemical,isotope,stereo),name))
    if not candidates:
        return None
    rank=min(row[0] for row in candidates)
    names={name for key,name in candidates if key==rank}
    return next(iter(names)) if len(names)==1 else None


def isotope_labelling_to_iupac_result(substance:IsotopeLabelling) -> IsotopeLabellingNamingResult:
    from .api import Chem
    if Chem is None:
        return _failure('unavailable','RDKit is unavailable.')
    kinds={'specific','selective','nonselective','deficient','general','uniform'}
    if (not isinstance(substance,IsotopeLabelling) or not isinstance(substance.kind,str)
            or substance.kind not in kinds
            or not isinstance(substance.unmodified_smiles,str)
            or not isinstance(substance.isotopologues,tuple)
            or not isinstance(substance.nuclides,tuple)
            or not isinstance(substance.subset_atom_maps,tuple)):
        return _failure('invalid-input','An explicit IsotopeLabelling input is required.')
    try:
        base=Chem.MolFromSmiles(substance.unmodified_smiles)
    except (TypeError,ValueError):
        base=None
    if (base is None or not base.GetNumAtoms() or base.GetNumHeavyAtoms()>64
            or len(Chem.GetMolFrags(base))!=1 or any(a.GetIsotope() for a in base.GetAtoms())):
        return _failure('invalid-input','The precursor must be one unmodified molecular graph.')
    atom_maps=[a.GetAtomMapNum() for a in base.GetAtoms() if a.GetAtomicNum()>1]
    if any(i<=0 for i in atom_maps) or len(set(atom_maps))!=len(atom_maps):
        return _failure('invalid-input','Every precursor heavy atom needs a unique positive map for cross-form alignment.')
    if base.GetStereoGroups() or unsupported_relative_stereochemistry(base, Chem):
        return _failure('unavailable','Relative or enhanced-stereo substances need a combined ensemble operation.')
    kind=substance.kind;maximum=Counter();members=[];forms=[]
    periodic=Chem.GetPeriodicTable()
    if kind in {'specific','selective'}:
        if not substance.isotopologues or substance.nuclides or substance.subset_atom_maps:
            return _failure('invalid-input','Specific/selective labelling requires explicit forms and no distribution-only fields.')
        identity=_unmodified_identity(base,Chem)
        for smiles in substance.isotopologues:
            try:
                mol=Chem.MolFromSmiles(smiles)
                # Validate every form before plain identity serialization or
                # a joint overlay can discard its stereo relationship.
                if mol is not None and (mol.GetStereoGroups()
                        or unsupported_relative_stereochemistry(mol, Chem)):
                    return _failure('unavailable',
                        'Relative or enhanced-stereo isotopologues need a combined ensemble operation.')
                matches=mol is not None and _unmodified_identity(mol,Chem)==identity
            except (TypeError,ValueError,RuntimeError):
                matches=False
            if not matches:
                return _failure('invalid-input','Every labelled form must have the same mapped unmodified structure and configuration.')
            site_counts=_sites(mol)
            if not site_counts:
                return _failure('invalid-input','Each labelled form needs a supported isotope-bearing position.')
            maximum |= site_counts
            forms.append(mol)
            copy=Chem.Mol(mol)
            for atom in copy.GetAtoms():atom.SetAtomMapNum(0)
            members.append(Chem.MolToSmiles(copy,canonical=True,isomericSmiles=True))
        members=sorted(set(members))
        if (kind=='specific' and len(members)!=1) or (kind=='selective' and len(members)<2):
            return _failure('invalid-input','Specific labelling has one distinct isotopologue; selective labelling requires several.')
    else:
        if substance.isotopologues or not substance.nuclides:
            return _failure('invalid-input','Distribution/depletion modes require nuclides rather than fixed isotopologue forms.')
        subset=substance.subset_atom_maps
        if (subset and kind!='uniform'
                or any(type(i) is not int or i not in atom_maps for i in subset)
                or len(set(subset))!=len(subset)):
            return _failure('invalid-input','Only uniform labelling may declare a valid subset of precursor atom maps.')
        if any(not isinstance(n,Nuclide) or not isinstance(n.element,str)
               or type(n.mass) is not int or n.mass<=0 for n in substance.nuclides):
            return _failure('invalid-input','Valid Nuclide objects are required.')
        if len(set(substance.nuclides))!=len(substance.nuclides):
            return _failure('invalid-input','Duplicate nuclide declarations are not permitted.')
        for nuclide in substance.nuclides:
            if not isinstance(nuclide,Nuclide) or type(nuclide.mass) is not int or nuclide.mass<=0:
                return _failure('invalid-input','Valid Nuclide objects are required.')
            possible=[]
            for atom in base.GetAtoms():
                if atom.GetAtomicNum()==1:continue
                if subset and atom.GetAtomMapNum() not in subset:continue
                count=(atom.GetTotalNumHs(includeNeighbors=True) if nuclide.element=='H'
                       else int(atom.GetSymbol()==nuclide.element))
                if count:
                    possible.append(atom.GetAtomMapNum())
                    maximum[(atom.GetAtomMapNum(),nuclide.element,nuclide.mass)]=count
            if not possible or subset and set(possible)!=set(subset):
                return _failure('invalid-input','The declared nuclide has no matching position or does not cover the declared subset.')
            if kind=='nonselective' and len(set(possible))<2:
                return _failure('invalid-input','Nonselective labelling requires more than one possible position.')
            if kind in {'general','uniform'} and nuclide.element!='H' and sum(
                    a.GetSymbol()==nuclide.element for a in base.GetAtoms())<2:
                return _failure('invalid-input','With only one atom of the element, only specific labelling is defined.')
    by_element={}
    for (_,symbol,mass),count in maximum.items():
        if symbol not in {'C','H'}:
            return _failure('unavailable','This first mapped component writer covers carbon and carbon-bound hydrogen labels.')
        try:
            valid=periodic.GetMassForIsotope(periodic.GetAtomicNumber(symbol),mass)>0
        except (RuntimeError,OverflowError):
            valid=False
        if not valid:
            return _failure('invalid-input','The nuclide mass is not recognized for its element.')
        by_element.setdefault(symbol,set()).add(mass)
    if any(len(masses)>1 for masses in by_element.values()):
        return _failure('unavailable','Different masses of one element need a joint multi-nuclide component operation.')
    overlay=Chem.Mol(forms[0] if kind=='specific' else base)
    if kind!='specific':
        rw=Chem.RWMol(overlay)
        lookup={a.GetAtomMapNum():a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum()>1}
        for (atom_map,symbol,mass),count in maximum.items():
            host=rw.GetAtomWithIdx(lookup[atom_map])
            if symbol!='H':
                host.SetIsotope(mass)
            else:
                if host.GetAtomicNum()!=6 or count>host.GetTotalNumHs(includeNeighbors=True):
                    return _failure('unavailable','Hydrogen distribution includes an unsupported local heteroatom position.')
                total=host.GetTotalNumHs(includeNeighbors=True)
                host.SetNumExplicitHs(total-count);host.SetNoImplicit(True)
                for _ in range(count):
                    hydrogen=Chem.Atom(1);hydrogen.SetIsotope(mass)
                    index=rw.AddAtom(hydrogen);rw.AddBond(host.GetIdx(),index,Chem.BondType.SINGLE)
        overlay=rw.GetMol()
        try:
            Chem.SanitizeMol(overlay)
        except (ValueError,RuntimeError):
            return _failure('unavailable','The joint position map does not admit a valid maximum-occupancy reference graph.')
    for atom in overlay.GetAtoms():atom.SetAtomMapNum(0)
    name=_mapped_chain_component(overlay,Chem,kind,bool(substance.subset_atom_maps))
    if not name:
        return _failure('unavailable','A complete single mapped carbon-chain naming component is not available.')
    sites=tuple(LabelledSite(atom_map,symbol,mass,count) for (atom_map,symbol,mass),count in sorted(maximum.items()))
    return IsotopeLabellingNamingResult(
        name=name, generation_status='available', preferred_status='resolved',
        route='explicit-isotope-labelling', kind=kind,
        unmodified_smiles=Chem.MolToSmiles(base,canonical=True,isomericSmiles=True),
        isotopologues=tuple(members), labelled_sites=sites,
        includes_unmodified=True if kind in {'specific','selective'} else None,
        position_ratios='equal-at-declared-positions' if kind=='uniform' else 'unspecified')


def isotope_labelling_to_iupac(substance:IsotopeLabelling) -> str:
    return isotope_labelling_to_iupac_result(substance).name
