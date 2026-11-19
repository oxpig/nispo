"""Whole-graph constructions for P-44 and modified functional parents.

All cuts below are proposal cuts: names and annotations are emitted only
after the original graph has been partitioned into disjoint owned units.
"""
from collections import Counter
from dataclasses import replace
import re

from .bluebook_lambda_amine_methods import _plain, _out, _sanitize
from .component_mapping import mapped_branch
from .constants import MULT, ROOTS
from .formatting import format_prefixes, enclose_name_unit


def order(mol, a, b):
    bond = mol.GetBondBetweenAtoms(a, b)
    return bond.GetBondTypeAsDouble() if bond else 0


def side(mol, start, blocked):
    found, todo = set(), [start]
    while todo:
        i = todo.pop()
        if i == blocked or i in found:
            continue
        found.add(i)
        todo.extend(n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors())
    return frozenset(found)


def capped(mol, Chem, atoms):
    from .naming_methods import _fragment
    return _fragment(mol, Chem, atoms)


def hydrazonic_chalcogen_acid(mol, Chem):
    if not _plain(mol, Chem, {6, 7, 8, 16, 34, 52}):
        return None
    ns = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    xs = [a for a in mol.GetAtoms() if a.GetAtomicNum() in {16, 34, 52}]
    if len(ns) != 2 or len(xs) != 1:
        return None
    # An ordinary carboxylic acid is senior to its chalcogen analogue.
    if any(a.GetAtomicNum()==6 and any(n.GetAtomicNum()==8 and n.GetDegree()==1
            and order(mol,a.GetIdx(),n.GetIdx())==2 for n in a.GetNeighbors())
            and any(n.GetAtomicNum()==8 and n.GetDegree()==1 and n.GetTotalNumHs()==1
            and order(mol,a.GetIdx(),n.GetIdx())==1 for n in a.GetNeighbors()) for a in mol.GetAtoms()):
        return None
    x = xs[0]
    if x.GetDegree() != 1 or x.GetTotalNumHs() != 1:
        return None
    c = x.GetNeighbors()[0]
    if c.GetAtomicNum() != 6 or c.IsInRing() or order(mol, c.GetIdx(), x.GetIdx()) != 1:
        return None
    imines = [a for a in c.GetNeighbors() if a.GetAtomicNum() == 7 and order(mol, c.GetIdx(), a.GetIdx()) == 2]
    if len(imines) != 1:
        return None
    n = imines[0]
    terminal = [a for a in n.GetNeighbors() if a.GetIdx() != c.GetIdx()]
    if (len(terminal) != 1 or terminal[0].GetAtomicNum() != 7
            or terminal[0].GetDegree() != 1 or terminal[0].GetTotalNumHs() != 2
            or n.GetTotalNumHs() or order(mol, n.GetIdx(), terminal[0].GetIdx()) != 1):
        return None
    # The acid analogue retains the systematic carbon parent and its locants.
    rw = Chem.RWMol(mol)
    rw.GetAtomWithIdx(x.GetIdx()).SetAtomicNum(8)
    rw.GetAtomWithIdx(n.GetIdx()).SetAtomicNum(8)
    rw.RemoveAtom(terminal[0].GetIdx())
    proxy = _sanitize(rw, Chem)
    if proxy is None:
        return None
    from .api import mol_to_iupac
    acid = mol_to_iupac(proxy)
    acid = {'acetic acid': 'ethanoic acid', 'formic acid': 'methanoic acid',
            'benzoic acid': 'benzenecarboxylic acid'}.get(acid, acid)
    if not acid.endswith(('oic acid', 'carboxylic acid')):
        return None
    stem = acid[:-len('oic acid')] + 'e'
    # carboxylic -> carbohydrazono, rather than carboxylehydrazono.
    if acid.endswith('carboxylic acid'):
        stem = acid[:-len('carboxylic acid')] + 'carbo'
    infix = {16: 'thio', 34: 'seleno', 52: 'telluro'}[x.GetAtomicNum()]
    return _out(mol, stem + 'hydrazono' + infix + 'ic acid', (c.GetIdx(),), 'P-65.1.5.1')


def carbonic_halogen_imido_acid(mol, Chem):
    if not _plain(mol, Chem, {6, 7, 8, 9, 17, 35, 53}):
        return None
    cs = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 6]
    if len(cs) != 1:
        return None
    c = cs[0]
    oh, halogen, imino = [], [], []
    for a in c.GetNeighbors():
        o = order(mol, c.GetIdx(), a.GetIdx())
        if a.GetAtomicNum() == 8 and a.GetDegree() == 1 and a.GetTotalNumHs() == 1 and o == 1:
            oh.append(a)
        elif a.GetAtomicNum() in {9, 17, 35, 53} and a.GetDegree() == 1 and o == 1:
            halogen.append(a)
        elif a.GetAtomicNum() == 7 and o == 2:
            imino.append(a)
        else:
            return None
    if len(oh) != 1 or len(halogen) != 1 or len(imino) != 1:
        return None
    n = imino[0]
    others = [a for a in n.GetNeighbors() if a.GetIdx() != c.GetIdx()]
    if not others and n.GetTotalNumHs() == 1:
        infix = 'imid'
        used = {c.GetIdx(), oh[0].GetIdx(), halogen[0].GetIdx(), n.GetIdx()}
    elif (len(others) == 1 and others[0].GetAtomicNum() == 7 and others[0].GetDegree() == 1
          and others[0].GetTotalNumHs() == 2 and not n.GetTotalNumHs()):
        infix = 'hydrazon'
        used = {c.GetIdx(), oh[0].GetIdx(), halogen[0].GetIdx(), n.GetIdx(), others[0].GetIdx()}
    else:
        return None
    if used != set(range(mol.GetNumAtoms())):
        return None
    hal = {9: 'fluorid', 17: 'chlorid', 35: 'bromid', 53: 'iodid'}[halogen[0].GetAtomicNum()]
    return _out(mol, 'carbono' + hal + infix + 'ic acid', (c.GetIdx(),), 'P-65.2.1.5')


def carbamoyl_carbamic_acid(mol, Chem):
    if not _plain(mol, Chem, {6, 7, 8}):
        return None
    sites = []
    for c in mol.GetAtoms():
        if c.GetAtomicNum() != 6 or c.GetDegree() != 3:
            continue
        oxo = [a for a in c.GetNeighbors() if a.GetAtomicNum() == 8 and a.GetDegree() == 1 and order(mol, c.GetIdx(), a.GetIdx()) == 2]
        oh = [a for a in c.GetNeighbors() if a.GetAtomicNum() == 8 and a.GetDegree() == 1 and a.GetTotalNumHs() == 1]
        n = [a for a in c.GetNeighbors() if a.GetAtomicNum() == 7 and order(mol, c.GetIdx(), a.GetIdx()) == 1]
        if len(oxo) == len(oh) == len(n) == 1:
            sites.append((c, oxo[0], oh[0], n[0]))
    if len(sites) != 1:
        return None
    c, oxo, oh, n = sites[0]
    used, prefixes, amides = {c.GetIdx(), oxo.GetIdx(), oh.GetIdx(), n.GetIdx()}, [], 0
    for root in n.GetNeighbors():
        if root.GetIdx() == c.GetIdx():
            continue
        text, owned = mapped_branch(mol, Chem, root.GetIdx(), n.GetIdx())
        if not text or used & owned:
            return None
        # Only an ordinary carbamoyl acyl ligand has the required acid class.
        acyl_oxygen = [a for a in root.GetNeighbors() if a.GetAtomicNum()==8
                      and a.GetDegree()==1 and order(mol,root.GetIdx(),a.GetIdx())==2]
        acyl_nitrogen = [a for a in root.GetNeighbors() if a.GetAtomicNum()==7
                        and a.GetIdx()!=n.GetIdx() and order(mol,root.GetIdx(),a.GetIdx())==1]
        if (root.GetAtomicNum()==6 and root.GetDegree()==3 and len(acyl_oxygen)==1
                and len(acyl_nitrogen)==1 and all(mol.GetAtomWithIdx(i).GetAtomicNum() in {6,7,8}
                for i in owned)):
            amides += 1
            loc = ''
        else:
            if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in owned):
                return None
            loc = 'N'
        prefixes.append((loc, text))
        used.update(owned)
    if amides != 1 or used != set(range(mol.GetNumAtoms())):
        return None
    prefix = format_prefixes(prefixes, hierarchical_enclosures=True).lstrip('-')
    return _out(mol, prefix + 'carbamic acid',
                (c.GetIdx(),), 'P-66.1.6.1.1.4/P-44')


def mononuclear_hydride(mol, Chem, element):
    allowed = {6, 8, 14} if element == 14 else {6, 14, 15}
    # Only the phosphorus centre may carry a tetrahedral annotation here.
    probe = Chem.Mol(mol)
    centers = [a for a in mol.GetAtoms() if a.GetAtomicNum() == element]
    if len(centers) != 1:
        return None
    center = centers[0]
    if element == 15:
        probe.GetAtomWithIdx(center.GetIdx()).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    if not _plain(probe, Chem, allowed) or center.IsInRing() or center.GetTotalValence() != {14: 4, 15: 3}[element]:
        return None
    prefixes, used = [], {center.GetIdx()}
    from .prefix_sequences import PrefixUnit, mononuclear_prefix_sequence
    for a in center.GetNeighbors():
        if order(mol, center.GetIdx(), a.GetIdx()) != 1:
            return None
        text, atoms = mapped_branch(mol, Chem, a.GetIdx(), center.GetIdx())
        if not text or atoms & used:
            return None
        if element==15 and any(mol.GetAtomWithIdx(i).GetAtomicNum()==14 for i in atoms):
            # A linked silicon skeleton needs its own locanted disilanyl
            # construction. Only the independently mapped SiH3 unit is a
            # simple silyl ligand in this mononuclear route.
            if atoms!={a.GetIdx()} or a.GetAtomicNum()!=14 or a.GetTotalNumHs()!=3:
                return None
        # Alcohols and carbonyl derivatives are senior to a hydride; ether O
        # can be subordinate to Si after earlier P-44 criteria have tied.
        if any(mol.GetAtomWithIdx(i).GetAtomicNum() == 8 and (
                mol.GetAtomWithIdx(i).GetTotalNumHs() or any(b.GetBondTypeAsDouble() == 2
                for b in mol.GetAtomWithIdx(i).GetBonds())) for i in atoms):
            return None
        carbon_only = all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in atoms)
        simple = carbon_only and (text in {'phenyl','tert-butyl'} or all(
            sum(n.GetIdx() in atoms for n in mol.GetAtomWithIdx(i).GetNeighbors()) <= 2 for i in atoms))
        if center.GetAtomicNum() == 14 and a.GetAtomicNum() == 8:
            rings = [r for r in mol.GetRingInfo().AtomRings() if set(r) <= atoms]
            attached=[n for n in a.GetNeighbors() if n.GetIdx()!=center.GetIdx()]
            if (len(atoms) == 5 and len(rings) == 1 and len(rings[0]) == 3
                    and Counter(mol.GetAtomWithIdx(i).GetAtomicNum() for i in rings[0])=={6:2,8:1}
                    and Counter(mol.GetAtomWithIdx(i).GetAtomicNum() for i in atoms) == {6:3,8:2}
                    and len(attached)==1 and attached[0].GetIdx() not in rings[0]
                    and attached[0].GetAtomicNum()==6 and attached[0].GetTotalNumHs()==2
                    and any(n.GetIdx() in rings[0] for n in attached[0].GetNeighbors())):
                text = 'oxiranylmethoxy'; simple = True
        prefixes.append(PrefixUnit(text,'simple' if simple or text=='silyl' else 'compound')); used.update(atoms)
    if used != set(range(mol.GetNumAtoms())):
        return None
    parent = {14: 'silane', 15: 'phosphane'}[element]
    sequence = mononuclear_prefix_sequence(prefixes)
    if sequence is None:
        return None
    text = sequence.text + parent
    if int(center.GetChiralTag()):
        from rdkit.Chem import rdCIPLabeler
        rdCIPLabeler.AssignCIPLabels(mol)
        if not center.HasProp('_CIPCode') or center.GetProp('_CIPCode') not in {'R', 'S'}:
            return None
        text = '(' + center.GetProp('_CIPCode') + ')-' + text
    return _out(mol, text, (center.GetIdx(),), 'P-44.1.2.1/P-93.3.4.1')


def phosphorus_modified_acid(mol, Chem):
    if not _plain(mol, Chem, {6, 8, 15, 16, 34, 52}):
        return None
    ps = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 15]
    if len(ps) != 1 or ps[0].IsInRing():
        return None
    p = ps[0]
    oxo, hydroxyls, prefixes, used = [], [], [], {p.GetIdx()}
    from .phosphorus_ester_methods import _carbon_component
    for a in p.GetNeighbors():
        o = order(mol, p.GetIdx(), a.GetIdx())
        if a.GetAtomicNum() in {8, 16, 34, 52} and a.GetDegree() == 1:
            if o == 2 and not a.GetTotalNumHs():
                oxo.append(a)
            elif o == 1 and a.GetTotalNumHs() == 1:
                hydroxyls.append(a)
            else:
                return None
            used.add(a.GetIdx())
        elif a.GetAtomicNum() == 6 and o == 1:
            atoms = _carbon_component(mol, a.GetIdx(), p.GetIdx())
            text, owned = mapped_branch(mol, Chem, a.GetIdx(), p.GetIdx())
            if atoms is None or atoms != owned or atoms & used or not text:
                return None
            prefixes.append(('', text)); used.update(atoms)
        else:
            return None
    if (len(oxo) > 1 or not hydroxyls or used != set(range(mol.GetNumAtoms()))
            or not any(a.GetAtomicNum() != 8 for a in oxo + hydroxyls)):
        return None
    fixed = len(prefixes) + p.GetTotalNumHs()
    if fixed not in {0, 1, 2} or len(hydroxyls) != 3 - fixed - (0 if oxo else 1):
        # P(III) acids have the same number of acid OH sites as the
        # corresponding substituted phosphorous parent (3-fixed).
        if oxo or len(hydroxyls) != 3 - fixed:
            return None
    stem = ('phosphor', 'phosphon', 'phosphin')[fixed]
    modified = Counter(a.GetAtomicNum() for a in oxo + hydroxyls if a.GetAtomicNum() != 8)
    infix = ''.join((MULT[n] if n > 1 else '') + {16:'thio',34:'seleno',52:'telluro'}[e]
                    for e, n in sorted(modified.items()))
    tail = 'ic acid' if oxo else 'us acid'
    if oxo and len(hydroxyls) == 1 and hydroxyls[0].GetAtomicNum() != oxo[0].GetAtomicNum():
        symbol = {8:'O',16:'S',34:'Se',52:'Te'}[hydroxyls[0].GetAtomicNum()]
        if hydroxyls[0].GetAtomicNum() != 8:
            tail = 'ic '+symbol+'-acid'
    return _out(mol, format_prefixes(prefixes, omit_locants=True, hierarchical_enclosures=True,
                separate_unlocanted=True) + stem + 'o' + infix + tail, (p.GetIdx(),), 'P-67.1.4.1.1.4')


def heterocyclic_thione(mol, Chem):
    if not _plain(mol, Chem, {6, 7, 8, 16, 34, 52}) or mol.GetRingInfo().NumRings() != 1:
        return None
    ring = frozenset(mol.GetRingInfo().AtomRings()[0])
    if not any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in ring):
        return None
    sites = [(a.GetNeighbors()[0].GetIdx(), a.GetIdx()) for a in mol.GetAtoms()
             if a.GetAtomicNum() == 16 and a.GetDegree() == 1
             and a.GetNeighbors()[0].GetIdx() in ring and order(mol, a.GetIdx(), a.GetNeighbors()[0].GetIdx()) == 2]
    if not sites:
        return None
    rw = Chem.RWMol(mol)
    for _host, end in sites:
        rw.GetAtomWithIdx(end).SetAtomicNum(8)
    proxy = _sanitize(rw, Chem)
    if proxy is None:
        return None
    from .naming_context import MoleculeContext
    from .ring_state_parent import select_ring_state
    from .component_mapping import decorate_component
    selection = select_ring_state(MoleculeContext(proxy, Chem))
    if selection is None:
        return None
    record = selection.selected
    if record.principal_centers != frozenset(h for h, _e in sites) or record.construction is None:
        return None
    parent = record.construction.parent
    if not parent.endswith('one'):
        return None
    parent = parent[:-3] + 'thione'
    # thione begins with a consonant: restore the final e of the hydride.
    if len(sites) == 1:
        parent = re.sub(r'(?<!e)(?=-\d+(?:\([^)]*\))?-thione$)', 'e', parent)
    name = decorate_component(mol, Chem, replace(record.construction, parent=parent))
    return _out(mol, name, (h for h, _e in sites), 'P-64.6.1/P-52.2.2.2') if name else None


def phosphorus_modified_ester(mol, Chem):
    """P(III/V) O-esters with amido, halido and chalcogen replacements."""
    if not _plain(mol, Chem, {6,7,8,9,15,16,17,34,35,52,53}):
        return None
    ps = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 15]
    if len(ps) != 1 or ps[0].IsInRing() or ps[0].GetTotalValence() not in {3,5}:
        return None
    p = ps[0]; used = {p.GetIdx()}
    esters, p_arms, nitrogens, oxos, halides, pseudos = [], [], [], [], [], []
    from .phosphorus_ester_methods import _carbon_component, _organyl_words
    def arm(root, host):
        atoms = _carbon_component(mol, root, host)
        if atoms is None or atoms & used:
            return None
        used.update(atoms)
        return root, atoms
    for a in p.GetNeighbors():
        idx = a.GetIdx(); o = order(mol, p.GetIdx(), idx)
        if a.GetAtomicNum() in {8,16,34,52} and a.GetDegree() == 1 and o == 2:
            oxos.append(a); used.add(idx)
        elif a.GetAtomicNum() in {9,17,35,53} and a.GetDegree() == 1 and o == 1:
            halides.append(a.GetAtomicNum()); used.add(idx)
        elif a.GetAtomicNum() == 6 and o == 1:
            row = arm(idx, p.GetIdx())
            if row is None:
                return None
            p_arms.append(row)
        elif a.GetAtomicNum() == 8 and a.GetDegree() == 2 and o == 1:
            others = [n for n in a.GetNeighbors() if n.GetIdx() != p.GetIdx()]
            row = arm(others[0].GetIdx(), idx) if len(others) == 1 else None
            if row is None:
                return None
            used.add(idx); esters.append((idx, *row))
        elif a.GetAtomicNum() == 7 and o == 1:
            # Covalent isothiocyanate, distinct from an amido nitrogen.
            other = [n for n in a.GetNeighbors() if n.GetIdx() != p.GetIdx()]
            if len(other) == 1 and order(mol, idx, other[0].GetIdx()) == 2:
                c = other[0]
                ends = [n for n in c.GetNeighbors() if n.GetIdx() != idx]
                if (c.GetAtomicNum() != 6 or len(ends) != 1 or ends[0].GetAtomicNum() != 16
                        or ends[0].GetDegree() != 1 or order(mol, c.GetIdx(), ends[0].GetIdx()) != 2):
                    return None
                atoms = {idx,c.GetIdx(),ends[0].GetIdx()}
                if atoms & used:
                    return None
                used.update(atoms); pseudos.append('isothiocyanatid')
                continue
            if a.IsInRing() or a.GetTotalValence() != 3:
                return None
            rows = []
            for root in other:
                row = arm(root.GetIdx(), idx)
                if row is None or order(mol, idx, root.GetIdx()) != 1:
                    return None
                rows.append(row)
            if len(rows) + a.GetTotalNumHs() != 2:
                return None
            used.add(idx); nitrogens.append((idx, tuple(rows)))
        else:
            return None
    if not esters or len(oxos) > 1 or used != set(range(mol.GetNumAtoms())):
        return None
    if (len(oxos)==1 and oxos[0].GetAtomicNum()==8
            and not (nitrogens or halides or pseudos)):
        return None  # Existing ordinary phosphate/phosphonate ester writer.
    fixed = len(p_arms) + p.GetTotalNumHs()
    if fixed not in {0,1,2} or len(esters)+len(nitrogens)+len(halides)+len(pseudos) != 3-fixed:
        return None
    if bool(oxos) != (p.GetTotalValence() == 5):
        return None
    site = dict(p=p.GetIdx(), p_arms=tuple(p_arms), ester_oxygens=tuple(esters),
                nitrogens=tuple(nitrogens), kind=('phosphoric','phosphonic','phosphinic')[fixed])
    from .phosphorus_amide_methods import _substitution
    substitution = _substitution(mol, Chem, site)
    words = _organyl_words(mol, Chem, site)
    if substitution is None or words is None:
        return None
    prefix, _mapping = substitution
    stem = ('phosphor','phosphon','phosphin')[fixed]
    infixes = []
    if nitrogens:
        infixes.append('amid' if len(nitrogens) == 1 else 'o'+MULT[len(nitrogens)]+'amid')
    for element, count in sorted(Counter(halides).items()):
        infixes.append((MULT[count] if count > 1 else '')+{9:'fluorid',17:'chlorid',35:'bromid',53:'iodid'}[element])
    infixes.extend(pseudos)
    if oxos and oxos[0].GetAtomicNum() != 8:
        infixes.append({16:'thio',34:'seleno',52:'telluro'}[oxos[0].GetAtomicNum()])
    tail = 'ate' if oxos else 'ite'
    infix = 'o'.join(infixes)
    if infix and infix[0] not in 'aeiou':
        infix = 'o'+infix
    if not infix and fixed==0:
        stem = 'phosph'
    body = stem + infix + tail
    # Complex pseudohalide class terms retain their enclosing marks.
    if pseudos and len(infixes) == 2 and nitrogens:
        body = stem + 'amid(' + pseudos[0] + tail + ')'
    return _out(mol, ' '.join([*(w for _key,w,_rows in words), prefix+body]),
                (p.GetIdx(),), 'P-67.1.3.2/P-67.1.4.1', method='functional-class')


def mixed_chain_pseudoketone(mol, Chem):
    if not _plain(mol, Chem, {6,8,16,34,52}):
        return None
    from .chalcogen_chain_pseudoketones import _path, _simple_acyl, _chain_words, _parent_name
    xs = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in {8,16,34,52}
          and all(b.GetBondTypeAsDouble() == 1 for b in a.GetBonds())}
    if len(xs) < 3:
        return None
    path = _path(mol, xs)
    if not path or any(mol.GetAtomWithIdx(i).GetDegree() != 2 or mol.GetAtomWithIdx(i).IsInRing() for i in xs):
        return None
    arms = []
    for end in (path[0],path[-1]):
        roots = [a for a in mol.GetAtomWithIdx(end).GetNeighbors() if a.GetIdx() not in xs]
        if len(roots) != 1 or roots[0].GetAtomicNum() != 6:
            return None
        acyl = _simple_acyl(mol, roots[0].GetIdx(), end, {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6})
        if acyl is None:
            return None
        arms.append((roots[0].GetIdx(), *acyl))
    if arms[0][1] & arms[1][1] or xs | arms[0][1] | arms[1][1] != set(range(mol.GetNumAtoms())):
        return None
    elements = tuple(mol.GetAtomWithIdx(i).GetAtomicNum() for i in path)
    if len(set(elements)) == 1:
        return None  # The homogeneous method already owns these paths.
    if elements == elements[::-1] and arms[0][2] == arms[1][2]:
        if len(elements) == 3 and elements[0] == 16 and elements[1] == 8:
            return _out(mol, "1,1'-dithioxanediyldi("+_parent_name(arms[0][2])+')',
                        (a[0] for a in arms), 'P-65.7.5.1', method='multiplicative')
        return None
    from .formatting import prefix_alphabetization_key
    choices = []
    for reverse in (False, True):
        seq = elements[::-1] if reverse else elements
        parent, other = (arms[1], arms[0]) if reverse else arms
        # Build the remote acyl prefix inward through homogeneous blocks.
        acyl = {1:'formyl',2:'acetyl'}.get(other[2], ROOTS.get(other[2], '')+'anoyl')
        from itertools import groupby
        blocks = [(e,len(tuple(items))) for e,items in groupby(seq)]
        text = acyl
        for e,count in reversed(blocks):
            words = _chain_words(e,count) if count > 1 else {
                8:('oxidane','oxy','oxydiyl'),16:('sulfane','sulfanyl','sulfanediyl'),
                34:('selane','selanyl','selanediyl'),52:('tellane','tellanyl','tellanediyl')}[e]
            if words is None:
                break
            suffix = 'peroxy' if e == 8 and count == 2 else words[1]
            text = ('('+text+')' if text.endswith(('peroxy','sulfanyl','selanyl','tellanyl')) else text) + suffix
        else:
            choices.append(((-parent[2],prefix_alphabetization_key(text)), '1-'+enclose_name_unit(text)+_parent_name(parent[2])))
    if not choices:
        return None
    name = min(choices)[1]
    return _out(mol,name,(a[0] for a in arms),'P-65.7.5.1')


def isotopic_sulfanyl_acid(mol,Chem):
    probe = Chem.Mol(mol)
    isotopes = [(a.GetIdx(),a.GetIsotope()) for a in mol.GetAtoms() if a.GetIsotope()]
    if not isotopes or any(mol.GetAtomWithIdx(i).GetAtomicNum()!=16 for i,_mass in isotopes):
        return None
    for i,_mass in isotopes:
        probe.GetAtomWithIdx(i).SetIsotope(0)
    if not _plain(probe,Chem,{6,8,16}):
        return None
    sulfur={a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum()==16}
    from .chalcogen_chain_pseudoketones import _path
    path=_path(mol,sulfur)
    if not path or len(path)<2 or len(path) not in MULT or any(mol.GetAtomWithIdx(i).GetDegree()!=2
            or mol.GetAtomWithIdx(i).GetTotalValence()!=2 for i in path):
        return None
    from .multiplicative_parent_extensions import unit
    from .phosphorus_ester_methods import _carbon_component
    choices=[]
    for seq in (path,path[::-1]):
        near=[a.GetIdx() for a in mol.GetAtomWithIdx(seq[0]).GetNeighbors() if a.GetIdx() not in sulfur]
        remote=[a.GetIdx() for a in mol.GetAtomWithIdx(seq[-1]).GetNeighbors() if a.GetIdx() not in sulfur]
        if len(near)!=1 or len(remote)!=1:
            continue
        acid_atoms=side(mol,near[0],seq[0]); organyl=_carbon_component(mol,remote[0],seq[-1])
        if (organyl is None or acid_atoms&organyl or acid_atoms|organyl|sulfur!=set(range(mol.GetNumAtoms()))):
            continue
        parent=unit(mol,Chem,acid_atoms,near[0])
        if parent is None or not parent[1].endswith(' acid'):
            continue
        prefix,owned=mapped_branch(mol,Chem,remote[0],seq[-1])
        if not prefix or owned!=organyl:
            continue
        labels={i:j for j,i in enumerate(seq,1)}
        parts=[]
        for mass in sorted({mass for _i,mass in isotopes}):
            locs=sorted(labels[i] for i,m in isotopes if m==mass)
            if Chem.GetPeriodicTable().GetMassForIsotope(16,mass)<=0:
                return None
            parts.append(','.join(map(str,locs))+'-'+str(mass)+'S'+(str(len(locs)) if len(locs)>1 else ''))
        branch=prefix+'('+','.join(parts)+')'+MULT[len(path)]+'sulfanyl'
        choices.append(parent[2]+'-'+enclose_name_unit(branch)+parent[1])
    return _out(mol,choices[0],(),'P-82.6.3.3/P-44') if len(choices)==1 else None


def chalcone_parent(mol,Chem):
    # Ring substitution below ketone is admitted; the retained parent itself
    # specifies E, so unspecified or Z inputs cannot enter this route.
    probe=Chem.Mol(mol)
    for a in mol.GetAtoms():
        if not a.GetFormalCharge():
            continue
        # Nitro resonance charges are wholly subordinate to the ketone.
        if a.GetAtomicNum()==7 and a.GetFormalCharge()==1:
            os=[n for n in a.GetNeighbors() if n.GetAtomicNum()==8 and n.GetDegree()==1]
            if len(os)!=2 or sorted(order(mol,a.GetIdx(),n.GetIdx()) for n in os)!=[1,2] or sum(n.GetFormalCharge() for n in os)!=-1:
                return None
        elif not (a.GetAtomicNum()==8 and a.GetFormalCharge()==-1 and a.GetDegree()==1
                  and a.GetNeighbors()[0].GetAtomicNum()==7 and a.GetNeighbors()[0].GetFormalCharge()==1):
            return None
        probe.GetAtomWithIdx(a.GetIdx()).SetFormalCharge(0)
    for b in probe.GetBonds():
        b.SetStereo(Chem.BondStereo.STEREONONE); b.SetBondDir(Chem.BondDir.NONE)
    if not _plain(probe,Chem,{6,7,8,9,16,17,35,53}):
        return None
    rings=tuple(map(tuple,mol.GetRingInfo().AtomRings()))
    if len(rings)!=2 or any(len(r)!=6 or any(mol.GetAtomWithIdx(i).GetAtomicNum()!=6
        or not mol.GetAtomWithIdx(i).GetIsAromatic() for i in r) for r in rings) or set(rings[0])&set(rings[1]):
        return None
    from .extended_inventory import selection_inventory
    inventory=selection_inventory(mol)
    if not inventory.supported or len(inventory.principal)!=1 or inventory.principal[0].kind!='ketone':
        return None
    oxos=[a for a in mol.GetAtoms() if a.GetAtomicNum()==8 and a.GetDegree()==1
          and a.GetNeighbors()[0].GetAtomicNum()==6 and order(mol,a.GetIdx(),a.GetNeighbors()[0].GetIdx())==2]
    if len(oxos)!=1:
        return None
    c=oxos[0].GetNeighbors()[0]
    aryl=[a for a in c.GetNeighbors() if a.IsInRing()]
    alpha=[a for a in c.GetNeighbors() if a.GetAtomicNum()==6 and not a.IsInRing()]
    if len(aryl)!=1 or len(alpha)!=1 or c.GetDegree()!=3:
        return None
    a=alpha[0]; beta=[n for n in a.GetNeighbors() if n.GetIdx()!=c.GetIdx()]
    if len(beta)!=1 or a.GetTotalNumHs()!=1:
        return None
    b=beta[0]; db=mol.GetBondBetweenAtoms(a.GetIdx(),b.GetIdx())
    ends=[n for n in b.GetNeighbors() if n.GetIdx()!=a.GetIdx()]
    if (db.GetBondTypeAsDouble()!=2 or db.GetStereo()!=Chem.BondStereo.STEREOE
            or len(ends)!=1 or not ends[0].IsInRing() or b.GetTotalNumHs()!=1):
        return None
    fixed={c.GetIdx(),oxos[0].GetIdx(),a.GetIdx(),b.GetIdx()}
    options=[]
    for root,prime in ((ends[0].GetIdx(),''),(aryl[0].GetIdx(),"'")):
        ring=next(r for r in rings if root in r)
        rows=[]
        for direction in (ring,ring[::-1]):
            start=direction.index(root); seq=direction[start:]+direction[:start]
            prefixes=[];used=set(ring)
            for j,i in enumerate(seq,1):
                for n in mol.GetAtomWithIdx(i).GetNeighbors():
                    if n.GetIdx() in ring or n.GetIdx() in fixed:
                        continue
                    text,atoms=mapped_branch(mol,Chem,n.GetIdx(),i)
                    if not text or atoms&used:
                        break
                    # Lower-class substituents: carbon, halogen, ether,
                    # alcohol, amine and nitro. Any further carbonyl or
                    # nitrile changes the principal parent.
                    if any(mol.GetAtomWithIdx(k).GetAtomicNum() in {8,16} and any(
                        bd.GetBondTypeAsDouble()==2 and bd.GetOtherAtom(mol.GetAtomWithIdx(k)).GetAtomicNum()==6
                        for bd in mol.GetAtomWithIdx(k).GetBonds())
                        or mol.GetAtomWithIdx(k).GetAtomicNum()==7 and any(bd.GetBondTypeAsDouble()==3
                        for bd in mol.GetAtomWithIdx(k).GetBonds()) for k in atoms):
                        break
                    prefixes.append((str(j)+prime,text));used.update(atoms)
                else:
                    continue
                break
            else:
                rows.append((prefixes,used))
        if not rows:
            return None
        options.append(rows)
    candidates=[]
    from itertools import product
    from .formatting import prefix_alphabetization_key
    for left,right in product(*options):
        if left[1]&right[1] or fixed|left[1]|right[1]!=set(range(mol.GetNumAtoms())):
            continue
        prefixes=left[0]+right[0]
        key=(tuple(sorted(int(loc.rstrip("'")) for loc,_name in prefixes)),
             tuple((prefix_alphabetization_key(name),int(loc.rstrip("'")),loc.endswith("'"))
                   for loc,name in sorted(prefixes,key=lambda r:(prefix_alphabetization_key(r[1]),r[0]))))
        candidates.append((key,format_prefixes(prefixes)+'chalcone'))
    return _out(mol,min(candidates)[1],(c.GetIdx(),),'P-64.2.1.1') if candidates else None


def mancude_two_face_fusion(mol,Chem):
    if (not _plain(mol,Chem,{6,7,8,16,34,52}) or mol.GetRingInfo().NumRings()!=2
            or any(not a.IsInRing() or a.GetTotalNumHs()>1 or a.GetAtomicNum()!=6 and a.GetTotalNumHs()
                   for a in mol.GetAtoms())):
        return None
    from .fusion_components import _mancude
    if not _mancude(mol):
        return None
    from .fusion_state_skeletons import two_face_fusion_templates
    templates=two_face_fusion_templates(mol,Chem)
    names={t['name'] for t in templates}
    if len(names)!=1:
        return None
    return _out(mol,names.pop(),(),'P-25.3.2.4/P-25.7/P-52.2.4',method='fusion')


def select(mol, Chem, numbers):
    producers = []
    if numbers[16] and numbers[8] >= 2 and any(a.GetIsotope() for a in mol.GetAtoms()):
        producers.append(isotopic_sulfanyl_acid)
    if numbers[8] and mol.GetRingInfo().NumRings()==2 and numbers[6]>=15:
        producers.append(chalcone_parent)
    if mol.GetRingInfo().NumRings()==2 and any(numbers[e] for e in (34,52)):
        producers.append(mancude_two_face_fusion)
    if (numbers[8] and numbers[15]) or any(numbers[x] >= 2 for x in (16,34,52)) or (numbers[8] and any(numbers[x] for x in (16,34,52))):
        from .bluebook_hydride_assembly_methods import mixed_anhydride_decision
        producers.append(mixed_anhydride_decision)
    if numbers[7] == 2 and numbers[8] >= 2:
        from .multiplicative_parent_extensions import imine_assembly, hydroxylamine_assembly
        producers.extend((imine_assembly,hydroxylamine_assembly))
    if mol.GetRingInfo().NumRings() >= 2:
        from .multiplicative_parent_extensions import repeated_ring_parents
        producers.append(repeated_ring_parents)
    if mol.GetRingInfo().NumRings() >= 4:
        from .multiplicative_parent_extensions import replacement_spirobi
        producers.append(replacement_spirobi)
    if numbers[7] == 2 and any(numbers[x] for x in (16,34,52)):
        producers.append(hydrazonic_chalcogen_acid)
    if numbers[7] and numbers[8] and any(numbers[x] for x in (9,17,35,53)):
        producers.append(carbonic_halogen_imido_acid)
    if numbers[7] == 2 and numbers[8] == 3:
        producers.append(carbamoyl_carbamic_acid)
    if numbers[14] == 1:
        result = mononuclear_hydride(mol, Chem, 14)
        if result:
            return result
    if numbers[15] == 1:
        result = phosphorus_modified_ester(mol, Chem)
        if result:
            return result
        if numbers[8] or any(numbers[x] for x in (16,34,52)):
            producers.append(phosphorus_modified_acid)
        else:
            result = mononuclear_hydride(mol, Chem, 15)
            if result:
                return result
    if numbers[16] and mol.GetRingInfo().NumRings():
        producers.append(heterocyclic_thione)
    if numbers[8] >= 2 and any(numbers[x] for x in (16,34,52)):
        producers.append(mixed_chain_pseudoketone)
    for producer in producers:
        result = producer(mol, Chem)
        if result:
            return result
    return None
