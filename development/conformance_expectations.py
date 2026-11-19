"""Independent closed-family PIN derivations and local rule obligations.

These formulas use the declared graph operations and primary-rule component
vocabulary. Complete expected names are composed only in memory, never stored
as per-molecule fixtures. They do not import production naming/ranking code.
"""
import re

ROOT={1:'meth',2:'eth',3:'prop',4:'but',5:'pent',6:'hex',7:'hept',8:'oct',
      9:'non',10:'dec',11:'undec',12:'dodec',13:'tridec',14:'tetradec'}
ACID={1:'formic',2:'acetic',3:'propanoic',4:'butanoic',5:'pentanoic'}
ACYL={1:'formyl',2:'acetyl',3:'propanoyl',4:'butanoyl',5:'pentanoyl'}
ANION={1:'formate',2:'acetate',3:'propanoate',4:'butanoate',5:'pentanoate'}
ORGANYL={1:'methyl',2:'ethyl',3:'propyl'}
OL={1:'ol',2:'diol',3:'triol',4:'tetrol',5:'pentol',6:'hexol',7:'heptol'}


def independently_derived_pin(row,mol):
    family=row['family'];facts=row['obligations']
    if family=='ester-homologues':return ORGANYL[facts['alcohol_carbons']]+' '+ANION[facts['acid_carbons']]
    if family=='acyl-halide-homologues':
        halogen=next(a.GetAtomicNum() for a in mol.GetAtoms() if a.GetAtomicNum() in {9,17,35,53})
        return ACYL[facts['acid_carbons']]+' '+{9:'fluoride',17:'chloride',35:'bromide',53:'iodide'}[halogen]
    if family=='anhydride-homologues':
        a,b=facts['acid_carbons'];terms=[ACID[a]] if a==b else sorted([ACID[a],ACID[b]])
        return ' '.join(terms)+' anhydride'
    if family=='declared-spiro':
        a,b=facts['sizes'];size=a+b-1
        return 'spiro['+'.'.join(map(str,sorted((a-1,b-1))))+']'+ROOT[size]+'ane'
    if family=='declared-assembly':
        a,b=facts['sizes'];assert a==b
        return "1,1'-bi(cyclo"+ROOT[a]+'ane)'
    if family=='aromatic-ring-assembly':return ','.join(('1',"1'"))+'-'+'biphenyl' 
    if family=='declared-fused' and min(facts['sizes'])<5:
        a,b=facts['sizes'];return 'bicyclo['+'.'.join(map(str,sorted((a-2,b-2,0),reverse=True)))+']'+ROOT[a+b-2]+'ane'
    if family=='declared-bridged':
        bridges=facts['bridges'];size=sum(bridges)+2
        return 'bicyclo['+'.'.join(map(str,sorted(bridges,reverse=True)))+']'+ROOT[size]+'ane'
    if family=='acid-ketone-homologues':
        acid=next(a for a in mol.GetAtoms() if a.GetAtomicNum()==6 and any(
            n.GetAtomicNum()==8 and n.GetTotalNumHs()==1 for n in a.GetNeighbors()))
        keto=next(a for a in mol.GetAtoms() if a.GetAtomicNum()==6 and a.GetIdx()!=acid.GetIdx()
                  and any(n.GetAtomicNum()==8 for n in a.GetNeighbors()))
        # The declared structure is a single carbon path; distances specify
        # functional locants without consulting the production selector.
        from rdkit import Chem
        loc=len(Chem.GetShortestPath(mol,acid.GetIdx(),keto.GetIdx()))
        size=sum(a.GetAtomicNum()==6 for a in mol.GetAtoms())
        return str(loc)+'-oxo'+ROOT[size]+'anoic acid'
    if family in {'alcohol-amine-homologues','amino-polyol-chains'}:
        carbon=[a for a in mol.GetAtoms() if a.GetAtomicNum()==6]
        ends=[a for a in carbon if sum(n.GetAtomicNum()==6 for n in a.GetNeighbors())==1]
        from rdkit import Chem
        path=Chem.GetShortestPath(mol,ends[0].GetIdx(),ends[1].GetIdx())
        candidates=[]
        for p in (path,tuple(reversed(path))):
            locs=tuple(j for j,i in enumerate(p,1) if any(n.GetAtomicNum()==8 and n.GetTotalNumHs()==1
                for n in mol.GetAtomWithIdx(i).GetNeighbors()))
            amino=next(j for j,i in enumerate(p,1) if any(n.GetAtomicNum()==7 for n in mol.GetAtomWithIdx(i).GetNeighbors()))
            candidates.append((locs,amino))
        locs,amino=min(candidates)
        size=len(carbon);parent=ROOT[size]+('an' if len(locs)==1 else 'ane')
        return str(amino)+'-amino'+parent+'-'+','.join(map(str,locs))+'-'+OL[len(locs)]
    if family=='polyol-suffix-vowels':
        from rdkit import Chem
        carbon=[a for a in mol.GetAtoms() if a.GetAtomicNum()==6]
        ends=[a.GetIdx() for a in carbon if sum(n.GetAtomicNum()==6 for n in a.GetNeighbors())==1]
        path=Chem.GetShortestPath(mol,*ends)
        locs=min(tuple(j for j,i in enumerate(p,1) if any(n.GetAtomicNum()==8 for n in mol.GetAtomWithIdx(i).GetNeighbors()))
                 for p in (path,tuple(reversed(path))))
        return ROOT[len(carbon)]+'ane-'+','.join(map(str,locs))+'-'+OL[len(locs)]
    return None


def local_obligations(row,mol,name,principal,decision,trace):
    family=row['family'];facts=row['obligations'];checks={}
    if 'principal' in facts and facts['principal']!='none':
        checks['P-41-principal-class']=bool(principal) and all(g.kind==facts['principal'] for g in principal)
    if 'principal_count' in facts:checks['P-44.1.1-recognized-sites']=len(principal)==facts['principal_count']
    if 'replacement_eligible' in facts:
        checks['P-51.4.1.1-eligibility']=any(r['method']=='skeletal-replacement' and r['eligible']==facts['replacement_eligible'] for r in trace)
        if facts['replacement_eligible']:
            checks['P-51.4-selected-and-emitted']=decision is not None and decision.method=='skeletal-replacement' and decision.name==name
            checks['P-51.4-heterounit-count']=decision is not None and len(getattr(decision,'heterounits',()))==facts['heterounits']
    if 'multiplicative_eligible' in facts:
        checks['P-51.3-rooted-identity-and-symmetry']=(decision is not None and decision.method=='multiplicative')==facts['multiplicative_eligible']
    if family=='amino-polyol-rings':
        checks['P-41/P-44-ring-OH-suffix']=name.endswith('-1,3-diol')
    if row['workstream']==5:
        checks['P-77-binary-construction']=bool(name) and ' and ' not in name
        checks['P-77-cation-before-anion']=name.startswith(('sodium','disodium','calcium','bis(','methan','ethan','propan','butan')) or 'aminium chloride' in name
        if family=='partial-polyacid-neutralization':checks['P-77.3.2-residual-carboxy-prefix']='carboxy' in name and 'hydroxy' not in name
    if family=='ester-prefix-with-senior-acid':
        checks['P-65.6.3.2.3-senior-acid']=name.endswith(' acid')
        checks['P-65.1.7-preferred-acyl-component']=not any(alias in name for alias in ('propionyl','butyryl','acetoxy'))
    return checks
