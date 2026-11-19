"""Development-only name -> numbered constitution -> CIP-constraint proof.

No original molecule is an input. Raw OPSIN results remain separate. This
bounded supplement handles one leading tetrahedral R/S/r/s block or one
explicitly enclosed, rooted component block. Duplicate leading locants require
a terminal saturated carbomonocyclic parent and one unique annular assignment.
An independent leading E/Z block on a single carbon monocycle can also be
proved against modern CIP. Other E/Z blocks retain their parsed geometry. Other tetrahedral
blocks and ambiguous scopes decline; one unique reconstruction is required. It imports no
production naming/selection code and is never used by the naming engine.
"""
from itertools import product
import os
from pathlib import Path
import re
import subprocess
import tempfile
from xml.etree import ElementTree as ET

from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

NS = {'c': 'http://www.xml-cml.org/schema'}
BLOCK = re.compile(r'^\(([^()]*)\)-(.+)$')
DESCRIPTOR = re.compile(r"(\d+[a-z]*'*)([RSrs])$")


def _request(name,include_uppercase=False):
    # A numbered ol hydrogen is installed before solving the written CIP
    # constraints; assigning stereo on the unlabelled base could invert it.
    labelled = re.fullmatch(r'\((\d+[RS](?:,\d+[RS])*)\)-([a-z0-9,\-]+)-(\d+)-\(([23])H\)ol',name)
    if include_uppercase and labelled:
        constraints=tuple((m[1],m[2]) for m in
                          (DESCRIPTOR.fullmatch(v) for v in labelled[1].split(',')))
        return (labelled[2]+'-'+labelled[3]+'-ol',constraints,None,
                (labelled[3],int(labelled[4])))
    match = BLOCK.fullmatch(name)
    if include_uppercase and match is not None and re.fullmatch(r'\d+[EZ]', match[1]):
        alkene = re.search(r'cyclo[a-z]+-(\d+)-ene$', match[2])
        if alkene is not None:
            if match[1][:-1] != alkene[1]:
                return None
            return match[2], ((alkene[1], match[1][-1]),), None, None
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
    return constitution, values,unit,None


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
        # A nested substituent may carry the same CML locants as the
        # component parent. The independently parsed radical identifies
        # the free valence; its connected ring system identifies the scope.
        # OPSIN's separately parsed radical identifies the named free valence,
        # independently of duplicate branch locants or other ring systems.
        roots=[i for i in labels.get(root,()) if local.GetAtomWithIdx(i).GetNumRadicalElectrons()==1]
        if len(roots)!=1:return ()
        core={roots[0]}
        remaining=list(local.GetRingInfo().AtomRings())
        while any(core.intersection(ring) for ring in remaining):
            attached=[ring for ring in remaining if core.intersection(ring)]
            for ring in attached:core.update(ring);remaining.remove(ring)
        if len(core)==1:return ()
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


def _monocycle_geometry(mol, locants, constraints):
    """Own one numbered annular C=C; enumerate its two modern CIP states."""
    rings = mol.GetRingInfo().AtomRings()
    if (len(rings) != 1 or len(rings[0]) < 8 or len(constraints) != 1
            or any(a.GetAtomicNum() not in {1, 6} or a.GetFormalCharge()
                   or a.GetIsotope() or a.GetChiralTag() or a.GetNumRadicalElectrons()
                   for a in mol.GetAtoms())):
        return set()
    ring = set(rings[0]);locant, descriptor = constraints[0]
    numbered = set(locants.get(locant, ())) & ring
    doubles = [b for b in mol.GetBonds() if b.GetBondTypeAsDouble() == 2]
    if len(numbered) != 1 or len(doubles) != 1:
        return set()
    bond = doubles[0];a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    if not {a,b} <= ring or not numbered <= {a,b}:
        return set()
    left = [n.GetIdx() for n in mol.GetAtomWithIdx(a).GetNeighbors() if n.GetIdx() != b]
    right = [n.GetIdx() for n in mol.GetAtomWithIdx(b).GetNeighbors() if n.GetIdx() != a]
    if len(left) != 2 or len(right) != 2:
        return set()
    solutions = set()
    for geometry in (Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS):
        candidate = Chem.Mol(mol);configured = candidate.GetBondWithIdx(bond.GetIdx())
        configured.SetStereoAtoms(left[0],right[0]);configured.SetStereo(geometry)
        Chem.SetDoubleBondNeighborDirections(candidate)
        Chem.AssignStereochemistry(candidate,cleanIt=True,force=True)
        rdCIPLabeler.AssignCIPLabels(candidate,maxRecursiveIterations=1_000_000)
        if configured.HasProp('_CIPCode') and configured.GetProp('_CIPCode') == descriptor:
            solutions.add(Chem.MolToSmiles(Chem.RemoveHs(candidate)))
    return solutions


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
    for name, (constitution, constraints, unit, isotope) in requests.items():
        record = parsed.get(constitution)
        if record is None:
            continue
        mol, locants = record
        if isotope is not None:
            locant,mass=isotope
            hosts=locants.get(locant,())
            if len(hosts)!=1:continue
            host=mol.GetAtomWithIdx(hosts[0]);hydrogens=[]
            if host.GetAtomicNum()!=6:continue
            for o in host.GetNeighbors():
                if (o.GetAtomicNum()!=8 or o.GetDegree()!=2 or o.GetFormalCharge()
                        or any(b.GetBondTypeAsDouble()!=1 for b in o.GetBonds())):continue
                hydrogens.extend(a.GetIdx() for a in o.GetNeighbors() if a.GetAtomicNum()==1)
            if len(hydrogens)!=1 or any(a.GetIsotope() for a in mol.GetAtoms()):continue
            mol=Chem.Mol(mol);mol.GetAtomWithIdx(hydrogens[0]).SetIsotope(mass)
        if all(d in {'E','Z'} for _,d in constraints):
            solutions = _monocycle_geometry(mol,locants,constraints)
            result[name] = dict(graph=next(iter(solutions)) if len(solutions)==1 else None,
                unique=len(solutions)==1,solution_count=len(solutions),
                mode='numbered-monocycle-modern-CIP-geometry')
            continue
        solutions = set()
        assignments = _assignments(mol,locants,constraints,unit,parsed,constitution)
        # One written descriptor inside a multiplied component applies to
        # every copy. This bounded decoder only owns a single occurrence;
        # choosing one symmetric match would silently unassign the others.
        if unit is not None and len(assignments) != 1:
            continue
        for atoms in assignments:
          # A neutral, three-coordinate aziridine N has a specified
          # tetrahedral configuration with its lone pair as fourth ligand.
          # Other non-carbon stereocenters remain outside this proof scope.
          if any(not (mol.GetAtomWithIdx(i).GetAtomicNum()==6 and mol.GetAtomWithIdx(i).GetDegree()==4
                      or mol.GetAtomWithIdx(i).GetAtomicNum()==7
                      and mol.GetAtomWithIdx(i).GetDegree()==3
                      and not mol.GetAtomWithIdx(i).GetFormalCharge()
                      and mol.GetAtomWithIdx(i).IsInRingSize(3)) for i in atoms):continue
          for tags in product((Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                             Chem.ChiralType.CHI_TETRAHEDRAL_CCW), repeat=len(atoms)):
            candidate = Chem.Mol(mol)
            for idx, tag in zip(atoms, tags):
                candidate.GetAtomWithIdx(idx).SetChiralTag(tag)
            try:
                rdCIPLabeler.AssignCIPLabels(candidate, maxRecursiveIterations=1_000_000)
            except RuntimeError:
                continue
            if all(candidate.GetAtomWithIdx(i).HasProp('_CIPCode') and
                   candidate.GetAtomWithIdx(i).GetProp('_CIPCode') == d
                   for i, (_, d) in zip(atoms, constraints)):
                solutions.add(Chem.MolToSmiles(Chem.RemoveHs(candidate)))
        result[name] = dict(graph=next(iter(solutions)) if len(solutions) == 1 else None,
            unique=len(solutions) == 1, solution_count=len(solutions),
            mode=('numbered-ol-hydrogen-CIP-constraints' if isotope else
                  'component-scoped-CIP-constraints' if unit else 'numbered-CIP-constraints'))
    return result
