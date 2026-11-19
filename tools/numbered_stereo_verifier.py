"""Independent numbered-stereochemistry and Delta-locant graph verification.

``verify_names`` receives names only. Raw OPSIN results remain separate. This
bounded supplement handles one leading tetrahedral R/S/r/s block or one
explicitly enclosed, rooted component block. Duplicate leading locants require
a terminal saturated carbomonocyclic parent and one unique annular assignment.
Independent E/Z blocks retain their parsed bond geometry. Other tetrahedral
blocks and ambiguous scopes decline; one unique reconstruction is required. It imports no
production naming/selection code and is never used by the naming engine.

``verify_delta_graph`` separately reconstructs a unique graph from the declared
Delta locants and an OPSIN-numbered parent, then compares it with a supplied
SMILES target. That target is not used to construct or select the graph.
"""
from collections import Counter
from itertools import product
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from xml.etree import ElementTree as ET

from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

NS = {'c': 'http://www.xml-cml.org/schema'}
BLOCK = re.compile(r'^\(([^()]*)\)-(.+)$')
DESCRIPTOR = re.compile(r"(\d+[a-z]*'*)([RSrs])$")


def _request(name,include_uppercase=False):
    match = BLOCK.fullmatch(name)
    if match is not None and not all(DESCRIPTOR.fullmatch(t) for t in match[1].split(',')):
        match=None  # An independent leading E/Z block stays on the constitution.
    unit=None
    if match is None:
        choices=[]
        for block in re.finditer(r'\(([^()]*)\)-',name):
            entries=[DESCRIPTOR.fullmatch(t) for t in block[1].split(',')]
            if entries and all(entries):choices.append((block,entries))
        if len(choices)!=1:return None
        block,entries=choices[0]
        if not block.start() or name[block.start()-1] not in '([{':return None
        opening=name[block.start()-1];closing={'(':')','[':']','{':'}'}[opening]
        stack=[opening];end=None
        for i in range(block.end(),len(name)):
            c=name[i]
            if c in '([{':stack.append(c)
            elif c in ')]}':
                if not stack or {'(':')','[':']','{':'}'}[stack.pop()]!=c:return None
                if not stack:end=i;break
        if end is None:return None
        unit=name[block.end():end]
        if not re.search(r'-\d+[a-z]*-yl$',unit):return None
        constitution=name[:block.start()]+name[block.end():]
        values=tuple((m[1],m[2]) for m in entries)
    else:
        entries = [DESCRIPTOR.fullmatch(text) for text in match[1].split(',')]
        if not entries or any(m is None for m in entries):return None
        values = tuple((m[1], m[2]) for m in entries)
        constitution=match[2]
    if not include_uppercase and not any(d in {'r', 's'} for _, d in values):
        return None
    if len(values) > 12 or len({loc for loc, _ in values}) != len(values):
        return None
    return constitution, values,unit


def _assignments(mol,locants,constraints,unit,parsed,constitution):
    if unit is None:
        if all(len(locants.get(loc,()))==1 for loc,_ in constraints):
            return (tuple(locants[loc][0] for loc,_ in constraints),)
        # A terminal saturated carbomonocyclic parent proves that these
        # leading descriptors have annular scope. A ring elsewhere in a chain
        # name cannot establish it. Other parent grammars remain unsupported.
        if not re.search(r'cyclo[a-z]+an(?:e)?(?:-(?:\d+,)*\d+-)?(?:ol|diol|triol|one|dione|carboxylic acid|carboxamide|carbonitrile)$',constitution):
            return ()
        systems=[]
        for ring in mol.GetRingInfo().AtomRings():
            core=set(ring)
            for old in systems[:]:
                if old&core:core.update(old);systems.remove(old)
            systems.append(core)
        if len(systems)!=1:return ()
        candidates=[set(locants.get(loc,()))&systems[0] for loc,_ in constraints]
        if any(len(c)!=1 for c in candidates):return ()
        return (tuple(next(iter(c)) for c in candidates),)
    record=parsed.get(unit)
    if record is None:return ()
    local,labels=record
    root=re.search(r'-(\d+[a-z]*)-yl$',unit)[1]
    required={root,*(loc for loc,_ in constraints)}
    if any(len(labels.get(loc,()))!=1 for loc in required):
        # A substituent such as methyl may carry its own CML locant 1.
        # The explicit terminal cycloalkanyl parent, not that nested branch,
        # owns this component's leading descriptors and its free valence.
        if not re.search(r'cyclo[a-z]+an-\d+-yl$',unit):return ()
        # OPSIN's separately parsed radical identifies the named free valence,
        # independently of duplicate branch locants or other ring systems.
        roots=[i for i in labels.get(root,()) if local.GetAtomWithIdx(i).GetNumRadicalElectrons()==1]
        if len(roots)!=1:return ()
        rings=[ring for ring in local.GetRingInfo().AtomRings() if roots[0] in ring and
            all(local.GetAtomWithIdx(i).GetAtomicNum()==6 and not local.GetAtomWithIdx(i).GetIsAromatic() for i in ring)]
        if len(rings)!=1:return ()
        core=set(rings[0])
        labels={loc:[i for i in values if i in core] for loc,values in labels.items()}
    if len(labels.get(root,()))!=1 or any(len(labels.get(loc,()))!=1 for loc,_ in constraints):return ()
    heavy=[a.GetIdx() for a in local.GetAtoms() if a.GetAtomicNum()>1]
    inverse={i:j for j,i in enumerate(heavy)};query=Chem.RWMol()
    for i in heavy:
        source=local.GetAtomWithIdx(i);atom=Chem.Atom(source.GetAtomicNum())
        atom.SetIsotope(source.GetIsotope());atom.SetIsAromatic(source.GetIsAromatic());query.AddAtom(atom)
    for bond in local.GetBonds():
        a,b=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
        if a in inverse and b in inverse:query.AddBond(inverse[a],inverse[b],bond.GetBondType())
    graph=query.GetMol();graph.UpdatePropertyCache(strict=False)
    results=set()
    for match in mol.GetSubstructMatches(graph,uniquify=False,maxMatches=10000):
        owned=set(match);attachment=match[inverse[labels[root][0]]]
        external=[(i,n.GetIdx()) for i in owned for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetAtomicNum()>1 and n.GetIdx() not in owned]
        if len(external)!=1 or external[0][0]!=attachment:continue
        results.add(tuple(match[inverse[labels[loc][0]]] for loc,_ in constraints))
    return tuple(sorted(results))


def _graph(row):
    graph, ids, locants = Chem.RWMol(), {}, {}
    geometric=[]
    for entry in row.findall('./c:atomArray/c:atom', NS):
        atom = Chem.Atom(entry.get('elementType'))
        atom.SetNoImplicit(True)
        atom.SetFormalCharge(int(entry.get('formalCharge', '0')))
        atom.SetIsotope(int(entry.get('isotopeNumber', '0')))
        idx = graph.AddAtom(atom)
        ids[entry.get('id')] = idx
        for label in entry.findall('c:label', NS):
            if label.get('dictRef') == 'cmlDict:locant':
                locants.setdefault(label.get('value'), []).append(idx)
    for entry in row.findall('./c:bondArray/c:bond', NS):
        a, b = entry.get('atomRefs2').split()
        order = {'S': Chem.BondType.SINGLE, 'D': Chem.BondType.DOUBLE,
                 'T': Chem.BondType.TRIPLE}.get(entry.get('order'))
        if order is None:
            return None
        graph.AddBond(ids[a], ids[b], order)
        stereo=entry.find('c:bondStereo',NS)
        if stereo is not None:
            refs=stereo.get('atomRefs4','').split()
            if order!=Chem.BondType.DOUBLE or len(refs)!=4 or stereo.text not in {'C','T'}:return None
            geometric.append((refs,stereo.text))
    # Nested stereo needs a separate component-scope decoder. Do not flatten it.
    if row.findall('.//c:atomParity', NS):
        return None
    mol = graph.GetMol()
    Chem.SanitizeMol(mol)
    for refs,value in geometric:
        first,a,b,last=(ids[r] for r in refs)
        bond=mol.GetBondBetweenAtoms(a,b)
        if bond is None:return None
        if bond.GetBeginAtomIdx()!=a:first,last=last,first
        bond.SetStereoAtoms(first,last)
        bond.SetStereo(Chem.BondStereo.STEREOCIS if value=='C' else Chem.BondStereo.STEREOTRANS)
    if geometric:
        Chem.SetDoubleBondNeighborDirections(mol)
        Chem.AssignStereochemistry(mol,cleanIt=True,force=True)
    return mol, locants


def verify_names(names,*,include_uppercase=False):
    """Name-only reconstruction; optionally audit modern uppercase CIP too.

    The opt-in uses the same exact-case constraints and unique graph proof;
    it does not alter raw OPSIN results or the default lower-case supplement.
    """
    requests = {name: _request(name,include_uppercase) for name in dict.fromkeys(names)}
    requests = {name: r for name, r in requests.items() if r is not None}
    result = {name: dict(graph=None, unique=False, mode='unsupported-CIP-name') for name in requests}
    if not requests:
        return result
    constitutions = list(dict.fromkeys([r[0] for r in requests.values()]+[r[2] for r in requests.values() if r[2]]))
    with tempfile.TemporaryDirectory(prefix='nispo-numbered-cip-') as temp:
        folder = Path(temp)
        (folder/'names.txt').write_text('\n'.join(constitutions)+'\n')
        jar = os.environ.get('NISPO_OPSIN_JAR', str(Path(__file__).resolve().parents[2]/'benchmark/opsin-cli-2.9.0-jar-with-dependencies.jar'))
        subprocess.run([os.environ.get('JAVA', 'java'), '-jar', jar,
            '-r','-o', 'cml', str(folder/'names.txt'), str(folder/'graphs.cml')],
            capture_output=True, text=True, check=True, timeout=240)
        rows = ET.parse(folder/'graphs.cml').getroot().findall('./c:molecule', NS)
    parsed = {}
    for row in rows:
        title = row.find('c:name', NS)
        if title is None or not title.text:
            continue
        try:
            parsed[title.text] = _graph(row)
        except (ValueError, RuntimeError, KeyError):
            parsed[title.text] = None
    for name, (constitution, constraints, unit) in requests.items():
        record = parsed.get(constitution)
        if record is None:
            continue
        mol, locants = record
        solutions = set()
        for atoms in _assignments(mol,locants,constraints,unit,parsed,constitution):
          if any(mol.GetAtomWithIdx(i).GetAtomicNum()!=6 or mol.GetAtomWithIdx(i).GetDegree()!=4 for i in atoms):continue
          for tags in product((Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                             Chem.ChiralType.CHI_TETRAHEDRAL_CCW), repeat=len(atoms)):
            candidate = Chem.Mol(mol)
            for idx, tag in zip(atoms, tags):
                candidate.GetAtomWithIdx(idx).SetChiralTag(tag)
            try:
                rdCIPLabeler.AssignCIPLabels(candidate)
            except RuntimeError:
                continue
            if all(candidate.GetAtomWithIdx(i).HasProp('_CIPCode') and
                   candidate.GetAtomWithIdx(i).GetProp('_CIPCode') == d
                   for i, (_, d) in zip(atoms, constraints)):
                solutions.add(Chem.MolToSmiles(Chem.RemoveHs(candidate)))
        result[name] = dict(graph=next(iter(solutions)) if len(solutions) == 1 else None,
            unique=len(solutions) == 1, solution_count=len(solutions), mode='component-scoped-CIP-constraints' if unit else 'numbered-CIP-constraints')
    return result


# Localized fusion verification retained from the independently numbered CML audit.
def _locant_key(locant):
    m = re.fullmatch(r'(\d+)([a-z]?)',locant)
    return (int(m[1]),m[2]) if m else None


def verify_delta_graph(smiles, name, *, java=None, jar=None):
    match = re.fullmatch(r'(?P<prefix>.*?)Δ\^(?:\(([^()]*(?:\([^()]+\)[^()]*)*)\)|\[([^\[\]]+)\])-(.+)',name)
    if match is None:return False
    descriptors = match.group(2) or match.group(3)
    leading = match.group('prefix')
    # Indicated H is itself hyphenated before the fused parent; ordinary
    # substituent text loses only the separator that introduced Delta.
    if not re.search(r'(?:\d+[a-z]?H,)*\d+[a-z]?H-$', leading):
        leading = leading.removesuffix('-')
    parent = leading + match.group(4)
    java = java or os.environ.get('JAVA') or shutil.which('java')
    jar = jar or os.environ.get('NISPO_OPSIN_JAR')
    if not java or not jar:return False
    run = subprocess.run([str(java),'-jar',str(jar),'-r','-o','cml'],
                         input=parent+'\n',capture_output=True,text=True,check=True,timeout=30)
    root=ET.fromstring(run.stdout);ns={'c':'http://www.xml-cml.org/schema'}
    atoms=root.findall('.//c:atomArray/c:atom',ns); bonds=root.findall('.//c:bondArray/c:bond',ns)
    heavy=[a for a in atoms if a.attrib['elementType']!='H'];ids=[a.attrib['id'] for a in heavy]
    index={name:i for i,name in enumerate(ids)}; labels={};elements=[];isotopes=[];hydrogen=Counter();edges=[];orders={}
    for atom in heavy:
        elements.append(atom.attrib['elementType'])
        isotopes.append(int(atom.attrib.get('isotopeNumber', 0)))
    for bond in bonds:
        left,right=bond.attrib['atomRefs2'].split()
        if left in index and right in index:
            edge=tuple(sorted((index[left],index[right])))
            edges.append(edge);orders[edge]=bond.attrib['order']
        elif left in index:hydrogen[index[left]]+=1
        elif right in index:hydrogen[index[right]]+=1
    # OPSIN gives substituent atoms their own local numeric locants. Determine
    # the fused ring topologically before interpreting the parent locants.
    skeleton=Chem.RWMol()
    for _ in elements:skeleton.AddAtom(Chem.Atom(6))
    for a,b in edges:skeleton.AddBond(a,b,Chem.BondType.SINGLE)
    topology=skeleton.GetMol();topology.UpdatePropertyCache(strict=False)
    ring={i for cycle in Chem.GetSymmSSSR(topology) for i in cycle}
    for i,atom in enumerate(heavy):
        if i not in ring:continue
        found=[label.attrib['value'] for label in atom.findall('c:label',ns)
               if _locant_key(label.attrib['value']) is not None]
        if len(found)!=1 or found[0] in labels:return False
        labels[found[0]]=i
    ordered=sorted(labels,key=_locant_key)
    terms=re.findall(r'\d+[a-z]?(?:\(\d+[a-z]?\))?',descriptors)
    if ','.join(terms)!=descriptors:return False
    witness=set()
    for term in terms:
        item=re.fullmatch(r'(\d+[a-z]?)(?:\((\d+[a-z]?)\))?',term)
        first,second=item.groups()
        if first not in labels:return False
        if second is None:
            pos=ordered.index(first)
            if pos+1>=len(ordered):return False
            second=ordered[pos+1]
        if second not in labels:return False
        edge=tuple(sorted((labels[first],labels[second])))
        if edge not in edges:return False
        witness.add(edge)
    degree=Counter(i for edge in edges for i in edge);valence={'C':4,'N':3,'O':2,'S':2}
    required=frozenset(i for i,z in enumerate(elements) if i in ring and valence[z]-degree[i]-hydrogen[i]==1)
    adjacency={i:{b if a==i else a for a,b in edges if i in (a,b)}&required for i in required}
    def matchings(rem):
        if not rem:yield frozenset();return
        first=min(rem)
        for nxt in adjacency[first]&rem:
            for rest in matchings(rem-{first,nxt}):yield rest|{tuple(sorted((first,nxt)))}
    recovered=set()
    for pairs in matchings(required):
        if not witness <= pairs:continue
        rw=Chem.RWMol()
        for i,z in enumerate(elements):
            atom=Chem.Atom(z);atom.SetIsotope(isotopes[i]);atom.SetNumExplicitHs(hydrogen[i]);atom.SetNoImplicit(True);rw.AddAtom(atom)
        for a,b in edges:
            if a in ring and b in ring:
                order=Chem.BondType.DOUBLE if (a,b) in pairs else Chem.BondType.SINGLE
            else:
                order={'S':Chem.BondType.SINGLE,'D':Chem.BondType.DOUBLE,'T':Chem.BondType.TRIPLE}.get(orders[(a,b)])
                if order is None:return False
            rw.AddBond(a,b,order)
        candidate=rw.GetMol()
        try:Chem.SanitizeMol(candidate)
        except (ValueError,RuntimeError):continue
        recovered.add(Chem.MolToSmiles(candidate,canonical=True,isomericSmiles=True))
    target=Chem.MolFromSmiles(smiles)
    return target is not None and recovered=={Chem.MolToSmiles(target,canonical=True,isomericSmiles=True)}
