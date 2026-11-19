"""Mapped sulfur radicals, modified acids and anhydrides (P-64/P-65/P-71).

The recognizers own the complete original graph. Carbon parents and every
separate branch retain their original atom indices for final annotation.
"""
from dataclasses import replace
from itertools import combinations
import re

from .molecule_iteration import iter_atoms, iter_bonds
from .naming_methods import MethodDecision
from .component_mapping import MappedComponent, decorate_component, mapped_branch
from .constants import MULT, ROOTS
from .formatting import complete_parent_name, enclose_name_unit, format_prefixes

_X = {8, 16, 34, 52}


def _order(mol, a, b):
    bond = mol.GetBondBetweenAtoms(a, b)
    return bond.GetBondTypeAsDouble() if bond else 0


def _side(mol, start, blocked):
    seen, todo = set(), [start]
    while todo:
        idx = todo.pop()
        if idx == blocked or idx in seen:
            continue
        seen.add(idx)
        todo.extend(n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors())
    return frozenset(seen)


def _clean(mol, Chem, allowed, *, radical=None):
    if (len(Chem.GetMolFrags(mol)) != 1 or mol.GetStereoGroups()
            or any(a.GetAtomicNum() not in allowed or a.GetFormalCharge() or a.HasQuery()
                   or (a.GetNumRadicalElectrons() and a.GetIdx() != radical)
                   or ((a.GetIsotope() or int(a.GetChiralTag())) and a.GetAtomicNum() != 6)
                   for a in iter_atoms(mol))):
        return False
    return not any(b.HasQuery() for b in iter_bonds(mol))


def _decision(mol, name, centers, rule, *, component=None, method='substitutive'):
    if not name:
        return None
    atoms = frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1)
    if component is not None and component.atoms | component.embedded_atoms != atoms:
        return None
    locants = component.atom_locants if component else ()
    return MethodDecision(name, method, frozenset(centers), atoms, rule, locants,
                          component_roles=(('complete-original-graph', atoms, locants),))


def _ordinary_acid_present(mol):
    for atom in iter_atoms(mol):
        if atom.GetAtomicNum() != 6:
            continue
        oxygens = [a for a in atom.GetNeighbors() if a.GetAtomicNum() == 8]
        if (any(a.GetDegree() == 1 and _order(mol, atom.GetIdx(), a.GetIdx()) == 2 for a in oxygens)
                and any(a.GetDegree() == 1 and a.GetTotalNumHs() == 1
                        and _order(mol, atom.GetIdx(), a.GetIdx()) == 1 for a in oxygens)):
            return True
    return False


def _tree_paths(mol, carbon, required):
    """All complete candidate carbon paths containing the suffix attachments."""
    adjacency = {i: tuple(n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors()
                          if n.GetIdx() in carbon) for i in carbon}
    if sum(map(len, adjacency.values())) != 2 * (len(carbon) - 1):
        return ()
    endpoints = [i for i, ns in adjacency.items() if len(ns) <= 1]
    if len(carbon) == 1:
        return (tuple(carbon),)
    result = []
    for start, end in combinations(endpoints, 2):
        def visit(current, previous, path):
            if current == end:
                return tuple(path)
            for nxt in adjacency[current]:
                if nxt != previous:
                    found = visit(nxt, current, path + [nxt])
                    if found:
                        return found
            return ()
        path = visit(start, None, [start])
        if path and set(required) <= set(path):
            result.extend((path, path[::-1]))
    return tuple(result)


def _prefixes(mol, Chem, path, core):
    labels = {i: str(j) for j, i in enumerate(path, 1)}
    used, prefixes, ownership = set(path) | set(core), [], []
    for host in path:
        for root in mol.GetAtomWithIdx(host).GetNeighbors():
            idx = root.GetIdx()
            if idx in used:
                continue
            text, atoms = mapped_branch(mol, Chem, idx, host)
            if not text or atoms & used:
                return None
            used.update(atoms)
            prefixes.append((labels[host], text))
            ownership.append((idx, host, atoms, text))
    if used != set(range(mol.GetNumAtoms())):
        return None
    return labels, tuple(prefixes), frozenset(used - set(path) - set(core)), tuple(ownership)


def _carbon_suffix(mol, Chem, hosts, core, suffix, *, require_ends=False,
                   omit_two_carbon_prefix_locants=False, extra_branches=(), direct_suffix=False):
    """Construct a suffix-bearing carbon parent with the original annotation map."""
    pool = {a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() == 6} - (set(core)-set(hosts))
    if not pool or not set(hosts) <= pool:
        return None
    carbon,todo=set(),[hosts[0]]
    while todo:
        i=todo.pop()
        if i in carbon:continue
        carbon.add(i)
        todo.extend(n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in pool)
    if not set(hosts)<=carbon:return None
    rings = [r for r in mol.GetRingInfo().AtomRings() if set(r)<=carbon]
    if rings:
        if (len(rings) != 1 or len(rings[0]) not in ROOTS or carbon != set(rings[0])
                or require_ends):
            return None
        ring = tuple(rings[0])
        paths = []
        for direction in (ring, ring[::-1]):
            for start in range(len(ring)):
                paths.append(direction[start:] + direction[:start])
        aromatic = all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
        if aromatic and len(ring) != 6:
            return None
        if not aromatic and any(b.GetBondTypeAsDouble() != 1 for b in iter_bonds(mol)
                                if b.GetBeginAtomIdx() in carbon and b.GetEndAtomIdx() in carbon):
            return None
        base = 'benzene' if aromatic else 'cyclo' + ROOTS[len(ring)] + 'ane'
    else:
        paths = _tree_paths(mol, carbon, hosts)
        base = None
    choices = []
    for path in paths:
        if any(host not in path for _root,host,_atoms,_text in extra_branches):
            continue
        if require_ends and set(hosts) != {path[0], path[-1]}:
            continue
        if len(path) not in ROOTS:
            continue
        record = _prefixes(mol, Chem, path, core)
        if record is None:
            continue
        labels, prefixes, embedded, ownership = record
        extra_atoms=frozenset(i for _r,_h,atoms,_t in extra_branches for i in atoms)
        prefixes+=tuple((labels[host],text) for _root,host,_atoms,text in extra_branches)
        embedded|=frozenset(core)-frozenset(path)
        ownership+=tuple(extra_branches)
        if base is None:
            from .branch_rules import parent_chain_name
            parent = complete_parent_name(parent_chain_name(mol, path))
        else:
            parent = base
        if not parent:
            continue
        suffix_locs = sorted(int(labels[i]) for i in hosts)
        if require_ends or direct_suffix:
            stem=parent.removesuffix('e') if direct_suffix and suffix[:1] in 'aeiouy' else parent
            body = stem + suffix
        else:
            omit = len(path) <= 2 and not prefixes or base is not None and not prefixes and len(hosts) == 1
            if suffix=='one':
                omit=False
                parent=parent.removesuffix('e')
            body = parent + ('' if omit else '-' + ','.join(map(str, suffix_locs)) + '-') + suffix
        mode = 'locanted'
        if omit_two_carbon_prefix_locants and len(path) == 2 and len(hosts) == 1:
            other = path[-1] if path[0] in hosts else path[0]
            if prefixes and {loc for loc, _text in prefixes} == {labels[other]}:
                # A C(=O)NH2 at the sole nonacid carbon has no alternative
                # prefix attachment in this two-carbon principal parent.
                mode = 'unlocanted'
        component = MappedComponent(body, tuple(labels.items()), frozenset(path),
            prefixes, embedded, systematic_parent=True, prefix_mode=mode,
            prefix_branch_ownership=ownership)
        if extra_branches:
            extra_words={text for _root,_host,_atoms,text in extra_branches}
            simple_words={text for root,_host,atoms,text in ownership
                          if len(atoms)==1 or all(mol.GetAtomWithIdx(i).GetAtomicNum()==6 for i in atoms)
                          and all(sum(n.GetIdx() in atoms for n in mol.GetAtomWithIdx(i).GetNeighbors())<=2 for i in atoms)}
            natures=tuple((text,'compound' if text in extra_words or text not in simple_words else 'simple')
                          for text in dict.fromkeys(text for _loc,text in prefixes))
            component=replace(component,prefix_natures=natures)
        name = decorate_component(mol, Chem, component)
        if not name:
            continue
        multiple = [j for j, (a, b) in enumerate(zip(path, path[1:]), 1) if _order(mol, a, b) > 1]
        double = [j for j, (a, b) in enumerate(zip(path, path[1:]), 1) if _order(mol, a, b) == 2]
        key = (-len(path), -len(multiple), -len(double), tuple(suffix_locs),
               tuple(multiple), tuple(double), -len(prefixes), tuple(sorted(int(loc) for loc, _t in prefixes)))
        choices.append((key, name, component))
    if not choices:
        return None
    best = min(key for key, _name, _component in choices)
    finalists = [row for row in choices if row[0] == best]
    if len({row[1] for row in finalists}) != 1:
        return None
    return finalists[0][1:]


def sulfur_radical(mol, Chem):
    """P-71.3.4 keeps the free electron on the terminal sulfur atom."""
    radicals = [a for a in iter_atoms(mol) if a.GetNumRadicalElectrons()]
    if len(radicals) != 1:
        return None
    radical = radicals[0]
    if (radical.GetAtomicNum() != 16 or radical.GetNumRadicalElectrons() != 1
            or radical.GetDegree() != 1 or radical.GetTotalNumHs() or radical.IsInRing()
            or not _clean(mol, Chem, {6, 7, 8, 9, 16, 17, 35, 53}, radical=radical.GetIdx())):
        return None
    chain, previous, current = [], None, radical.GetIdx()
    while mol.GetAtomWithIdx(current).GetAtomicNum() == 16:
        atom = mol.GetAtomWithIdx(current)
        if (current in chain or atom.IsInRing() or atom.GetTotalNumHs()
                or any(b.GetBondTypeAsDouble() != 1 for b in atom.GetBonds())):
            return None
        chain.append(current)
        others = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() != previous]
        if len(others) != 1:
            return None
        previous, current = current, others[0]
    if mol.GetAtomWithIdx(current).GetAtomicNum() != 6:
        return None
    if len(chain) > 1 and len(chain) not in MULT:
        return None
    root = mol.GetAtomWithIdx(current)
    thioacyl = [a.GetIdx() for a in root.GetNeighbors() if a.GetAtomicNum()==16
                and a.GetDegree()==1 and _order(mol,current,a.GetIdx())==2]
    if len(thioacyl)==1:
        # Systematic thioacyl prefixes supersede retained thioacetyl forms.
        if any(n.GetIdx() not in {previous,*thioacyl} and n.GetAtomicNum()!=6 for n in root.GetNeighbors()):
            return None
        result=_carbon_suffix(mol,Chem,[current],set(chain)|set(thioacyl),'thioyl',
                              direct_suffix=True,omit_two_carbon_prefix_locants=True)
        if result is None:
            return None
        text,_component=result
        text = enclose_name_unit(text)
    else:
        text, owned = mapped_branch(mol, Chem, current, previous)
        if not text or set(owned)|set(chain)!=set(range(mol.GetNumAtoms())) or set(owned)&set(chain):
            return None
        if re.search(r'\d',text) or any(c in text for c in '()[]'):
            text=enclose_name_unit(text)
    parent = (MULT[len(chain)] if len(chain) > 1 else '') + 'sulfanyl'
    return _decision(mol, text + parent, (radical.GetIdx(),), 'P-71.3.4', method='radical')


def dithioic_polyacid(mol, Chem):
    """P-65.1.5.1/2 preserves both C(=S)SH ends in one carbon parent."""
    if not _clean(mol, Chem, {6, 7, 8, 9, 16, 17, 35, 53}) or _ordinary_acid_present(mol):
        return None
    sites = []
    for atom in iter_atoms(mol):
        if atom.GetAtomicNum() != 6 or atom.IsInRing():
            continue
        double = [a for a in atom.GetNeighbors() if a.GetAtomicNum() == 16 and a.GetDegree() == 1
                  and _order(mol, atom.GetIdx(), a.GetIdx()) == 2 and not a.GetTotalNumHs()]
        single = [a for a in atom.GetNeighbors() if a.GetAtomicNum() == 16 and a.GetDegree() == 1
                  and _order(mol, atom.GetIdx(), a.GetIdx()) == 1 and a.GetTotalNumHs() == 1]
        if len(double) == len(single) == 1:
            sites.append((atom.GetIdx(), double[0].GetIdx(), single[0].GetIdx()))
    if len(sites) != 2:
        return None
    core = {i for _c, x, h in sites for i in (x, h)}
    if core != {a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() == 16}:
        return None
    # Lower-class alcohol/ether, ketone and amino/amide decoration can be
    # mapped as prefixes. Another acid-family replacement delegates.
    for a in iter_atoms(mol):
        if a.GetAtomicNum()==8:
            if a.GetDegree()==1 and a.GetNeighbors()[0].GetAtomicNum()==6 and (
                    a.GetTotalNumHs()==1 and a.GetBonds()[0].GetBondTypeAsDouble()==1
                    or not a.GetTotalNumHs() and a.GetBonds()[0].GetBondTypeAsDouble()==2):
                continue
            if a.GetDegree()==2 and all(n.GetAtomicNum()==6 for n in a.GetNeighbors()) and all(
                    b.GetBondTypeAsDouble()==1 for b in a.GetBonds()):
                continue
            return None
        if a.GetAtomicNum()==7 and (a.GetTotalValence()!=3 or any(n.GetAtomicNum()!=6 for n in a.GetNeighbors())
                                   or any(b.GetBondTypeAsDouble()!=1 for b in a.GetBonds())):
            return None
    result = _carbon_suffix(mol, Chem, [row[0] for row in sites], core, 'bis(dithioic acid)', require_ends=True)
    if result is None:
        return None
    name, component = result
    return _decision(mol, name, (row[0] for row in sites), 'P-65.1.5.1/P-65.1.5.2', component=component)


def symmetric_anhydride(mol, Chem):
    """P-65.7.1 multiplies the acid only after complete rooted identity."""
    if not _clean(mol, Chem, {6, 7, 8, 9, 16, 17, 34, 35, 52, 53}):
        return None
    from .acid_derivative_methods import _acid_component
    candidates = []
    for bridge in iter_atoms(mol):
        if bridge.GetAtomicNum() != 8 or bridge.GetDegree() != 2 or bridge.IsInRing():
            continue
        left, right = bridge.GetNeighbors()
        if any(_order(mol, bridge.GetIdx(), a.GetIdx()) != 1 for a in (left, right)):
            continue
        def endpoint(a):
            return (a.GetAtomicNum() in {6, 16, 34, 52} and any(n.GetAtomicNum() in _X and n.GetDegree() == 1
                    and _order(mol, a.GetIdx(), n.GetIdx()) == 2 for n in a.GetNeighbors()))
        if not endpoint(left) or not endpoint(right):
            continue
        scopes = [_side(mol, a.GetIdx(), bridge.GetIdx()) for a in (left, right)]
        if scopes[0] & scopes[1] or scopes[0] | scopes[1] | {bridge.GetIdx()} != set(range(mol.GetNumAtoms())):
            continue
        signatures = [Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(scope), rootedAtAtom=a.GetIdx(),
                        canonical=True, isomericSmiles=True) for a, scope in zip((left, right), scopes)]
        if signatures[0] != signatures[1]:
            continue
        parts = [_acid_component(mol, Chem, a.GetIdx(), bridge.GetIdx()) for a in (left, right)]
        if all(p is not None for p in parts) and parts[0].identity == parts[1].identity:
            acid = parts[0].name
            if acid.endswith(' acid'):
                candidates.append(acid[:-5] + ' anhydride')
            continue
        # The carbonyl-O component helper does not cover C=S or R-SO(x).
        # Those source families use unannotated capped acid components here.
        if any(a.GetIsotope() or int(a.GetChiralTag()) for a in iter_atoms(mol)) or any(int(b.GetStereo()) for b in iter_bonds(mol)):
            continue
        rw = Chem.RWMol(mol)
        rw.RemoveBond(bridge.GetIdx(), right.GetIdx())
        rw.GetAtomWithIdx(bridge.GetIdx()).SetNumExplicitHs(0)
        rw.GetAtomWithIdx(bridge.GetIdx()).SetNoImplicit(False)
        cap = rw.AddAtom(Chem.Atom(8)); rw.AddBond(right.GetIdx(), cap, Chem.BondType.SINGLE)
        proxy = rw.GetMol()
        try: Chem.SanitizeMol(proxy)
        except (ValueError, RuntimeError): continue
        from .bluebook_hydride_assembly_methods import _capped_acid_name
        names = [_capped_acid_name(fragment, Chem) for fragment in Chem.GetMolFrags(proxy, asMols=True)]
        if len(names) == 2 and names[0] == names[1] and names[0].endswith(' acid'):
            acid = re.sub(r' (?:O|S|Se|Te)-acid$', ' acid', names[0])
            candidates.append(acid[:-5] + ' anhydride')
    if len(set(candidates)) != 1:
        return None
    return _decision(mol, candidates[0], (), 'P-65.7.1', method='functional-class')


def sulfur_pseudoester(mol, Chem):
    if not _clean(mol, Chem, {6, 8, 9, 16, 17, 34, 35, 52, 53}):
        return None
    sites = []
    for carbon in iter_atoms(mol):
        if carbon.GetAtomicNum() != 6 or carbon.IsInRing():
            continue
        oxo = [a.GetIdx() for a in carbon.GetNeighbors() if a.GetAtomicNum() == 8 and a.GetDegree() == 1
               and _order(mol, carbon.GetIdx(), a.GetIdx()) == 2]
        linked = [a.GetIdx() for a in carbon.GetNeighbors() if a.GetAtomicNum() in _X
                  and _order(mol, carbon.GetIdx(), a.GetIdx()) == 1]
        if len(oxo) == len(linked) == 1:
            sites.append((carbon.GetIdx(), oxo[0], linked[0]))
    if len(sites) != 1:
        return None
    c, oxygen, start = sites[0]
    chain, previous, current = [], c, start
    while mol.GetAtomWithIdx(current).GetAtomicNum() in _X:
        atom = mol.GetAtomWithIdx(current)
        if (current in chain or atom.IsInRing() or atom.GetDegree() != 2
                or atom.GetTotalValence() != 2 or atom.GetIsotope()
                or any(b.GetBondTypeAsDouble() != 1 for b in atom.GetBonds())):
            return None
        chain.append(current)
        nxt = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() != previous]
        previous, current = current, nxt[0]
    if len(chain) < 2 or mol.GetAtomWithIdx(current).GetAtomicNum() != 6:
        return None
    remote = _side(mol, current, previous)
    from .phosphorus_ester_methods import _carbon_component
    if _carbon_component(mol, current, previous) != remote:
        return None  # A remote acyl has the senior anhydride class.
    prefix, owned = mapped_branch(mol, Chem, current, previous)
    if not prefix or owned != remote:
        return None
    from itertools import groupby
    blocks = [(e, len(tuple(items))) for e, items in groupby(mol.GetAtomWithIdx(i).GetAtomicNum() for i in chain)]
    for e, count in reversed(blocks):
        if count > 1 and count not in MULT:
            return None
        if e == 8:
            suffix = 'oxy' if count == 1 else 'peroxy' if count == 2 else MULT[count] + 'oxidanyl'
        else:
            suffix = (MULT[count] if count > 1 else '') + {16:'sulfanyl',34:'selanyl',52:'tellanyl'}[e]
        if prefix.endswith(('sulfanyl', 'selanyl', 'tellanyl', 'peroxy', 'oxidanyl')):
            prefix = enclose_name_unit(prefix)
        prefix += suffix
    core = set(chain) | set(remote) | {oxygen}
    result = _carbon_suffix(mol, Chem, [c], core, 'one', require_ends=False,
                            extra_branches=((start,c,frozenset(chain)|remote,prefix),))
    if result is None:
        return None
    name, component = result
    if len(component.atom_locants) < 2:
        return None
    return _decision(mol, name, (c,), 'P-65.6.3.4.2/P-64.3.2', component=component)


def peroxyacid_above_amide(mol, Chem):
    if not _clean(mol, Chem, {6, 7, 8, 9, 17, 35, 53}) or _ordinary_acid_present(mol):
        return None
    from .peroxyacid_components import recognize_peroxyacids
    sites = recognize_peroxyacids(mol)
    if len(sites) != 1:
        return None
    site = sites[0]
    # This extension resolves the missing acid/amide comparison. Ordinary
    # peroxyacids retain their established mapped producer.
    amides = [a for a in iter_atoms(mol) if a.GetAtomicNum() == 6 and any(
        n.GetAtomicNum() == 8 and n.GetDegree() == 1 and _order(mol,a.GetIdx(),n.GetIdx()) == 2 for n in a.GetNeighbors())
        and any(n.GetAtomicNum() == 7 and _order(mol,a.GetIdx(),n.GetIdx()) == 1 for n in a.GetNeighbors())]
    if not amides:
        return None
    core = {site.carbonyl_carbon, site.carbonyl_oxygen, site.proximal_oxygen, site.terminal_oxygen}
    if any(mol.GetAtomWithIdx(i).GetIsotope() for i in core - {site.carbonyl_carbon}):
        return None
    result = _carbon_suffix(mol, Chem, [site.carbonyl_carbon], core, 'peroxoic acid', direct_suffix=True,
                            omit_two_carbon_prefix_locants=True)
    if result is None:
        return None
    name, component = result
    return _decision(mol, name, (site.carbonyl_carbon,), 'P-65.1.4.1/P-41', component=component)


def modified_chalcogen_acid(mol, Chem):
    if not _clean(mol, Chem, {6, 7, 8, 9, 16, 17, 34, 35, 52, 53}):
        return None
    centers = [a for a in iter_atoms(mol) if a.GetAtomicNum() in {16,34,52}]
    if len(centers) != 1 or centers[0].IsInRing():
        return None
    center = centers[0]; eidx = center.GetIdx()
    oxo, acid, peroxo, hydrazono, carbon, core = [], [], [], [], [], {eidx}
    for atom in center.GetNeighbors():
        idx = atom.GetIdx(); bond = _order(mol, eidx, idx)
        if atom.GetAtomicNum() == 6 and bond == 1:
            carbon.append(idx); continue
        if atom.GetAtomicNum() == 8 and atom.GetDegree() == 1 and bond == 2:
            oxo.append(idx); core.add(idx); continue
        if atom.GetAtomicNum() == 8 and bond == 1:
            if atom.GetDegree() == 1 and atom.GetTotalNumHs() == 1:
                acid.append(idx); core.add(idx); continue
            ends = [a for a in atom.GetNeighbors() if a.GetIdx() != eidx]
            if (atom.GetDegree() == 2 and len(ends) == 1 and ends[0].GetAtomicNum() == 8
                    and ends[0].GetDegree() == 1 and ends[0].GetTotalNumHs() == 1
                    and _order(mol, idx, ends[0].GetIdx()) == 1):
                peroxo.append((idx, ends[0].GetIdx())); core.update(peroxo[-1]); continue
        if atom.GetAtomicNum() == 7 and bond == 2 and atom.GetDegree() == 2 and not atom.GetTotalNumHs():
            ends = [a for a in atom.GetNeighbors() if a.GetIdx() != eidx]
            if (len(ends) == 1 and ends[0].GetAtomicNum() == 7 and ends[0].GetDegree() == 1
                    and ends[0].GetTotalNumHs() == 2 and _order(mol, idx, ends[0].GetIdx()) == 1):
                hydrazono.append((idx, ends[0].GetIdx())); core.update(hydrazono[-1]); continue
        return None
    if len(carbon) != 1:
        return None
    root = carbon[0]
    if len(peroxo) == 1 and not acid and not hydrazono and len(oxo) in {1,2}:
        stem = {(16,1):'sulfino',(16,2):'sulfono',(34,1):'selenino',(34,2):'selenono',
                (52,1):'tellurino',(52,2):'tellurono'}[(center.GetAtomicNum(),len(oxo))]
        suffix, rule = stem + 'peroxoic acid', 'P-65.3.1.2'
    elif len(acid) == 1 and not peroxo and len(hydrazono) in {1,2} and len(hydrazono)+len(oxo) in {1,2}:
        units = len(hydrazono)+len(oxo)
        stem = {(16,1):'sulfino',(16,2):'sulfono',(34,1):'selenino',(34,2):'selenono',
                (52,1):'tellurino',(52,2):'tellurono'}[(center.GetAtomicNum(),units)]
        suffix = stem + (MULT[len(hydrazono)] if len(hydrazono)>1 else '') + 'hydrazonic acid'
        rule = 'P-65.3.1.4'
    else:
        return None
    if any(mol.GetAtomWithIdx(i).GetIsotope() or int(mol.GetAtomWithIdx(i).GetChiralTag()) for i in core):
        return None
    # The carbon-parent component maps its carbon stereochemistry, but the
    # modified acid atoms are embedded. It cannot cite an assigned E/Z unit
    # on E=N. Keep that original annotation for the general mapped writer.
    if any(int(b.GetStereo()) and
           (b.GetBeginAtomIdx() in core or b.GetEndAtomIdx() in core)
           for b in iter_bonds(mol)):
        return None
    if _ordinary_acid_present(mol):
        # P-65.3.2.3 retains the complete, direction-bearing modified
        # chalcogen function as a concatenated compound prefix.
        if hydrazono:
            prefix='hydroxy'+suffix[:-len('ic acid')]+'oyl'
        else:
            acyl={(16,1):'sulfinyl',(16,2):'sulfonyl',(34,1):'seleninyl',(34,2):'selenonyl',
                  (52,1):'tellurinyl',(52,2):'telluronyl'}[(center.GetAtomicNum(),len(oxo))]
            prefix='hydroperoxy'+acyl
        acids=[]
        for c in iter_atoms(mol):
            if c.GetAtomicNum()!=6 or c.IsInRing():continue
            double=[a.GetIdx() for a in c.GetNeighbors() if a.GetAtomicNum()==8 and a.GetDegree()==1 and _order(mol,c.GetIdx(),a.GetIdx())==2]
            oh=[a.GetIdx() for a in c.GetNeighbors() if a.GetAtomicNum()==8 and a.GetDegree()==1 and a.GetTotalNumHs()==1 and _order(mol,c.GetIdx(),a.GetIdx())==1]
            if len(double)==len(oh)==1 and any(a.GetAtomicNum()==6 for a in c.GetNeighbors()):
                acids.append((c.GetIdx(),double[0],oh[0]))
        if len(acids)!=1:return None
        acid_c,acid_o,acid_oh=acids[0]
        combined_core=core|{acid_c,acid_o,acid_oh}
        result=_carbon_suffix(mol,Chem,[acid_c],combined_core,'oic acid',direct_suffix=True,
            extra_branches=((eidx,root,frozenset(core),prefix),))
        if result is None:return None
        name,component=result
        return _decision(mol,name,(acid_c,),'P-65.3.2.3/P-41',component=component)
    # Remaining oxygen/nitrogen can be subordinate hydroxy, amino or oxo,
    # but not an ester or another acid in this complete functional family.
    for a in iter_atoms(mol):
        if a.GetIdx() not in core and a.GetAtomicNum() in {7,8}:
            if a.GetDegree()!=1 or a.GetNeighbors()[0].GetAtomicNum()!=6:
                return None
    result = _carbon_suffix(mol,Chem,[root],core,suffix)
    if result is None:
        return None
    name, component = result
    return _decision(mol,name,(eidx,),rule,component=component)


def thiocarbonyl_oxide(mol,Chem):
    if not _clean(mol,Chem,{6,8,9,16,17,35,53}):
        return None
    ss=[a for a in iter_atoms(mol) if a.GetAtomicNum()==16]
    os=[a for a in iter_atoms(mol) if a.GetAtomicNum()==8]
    if len(ss)!=1 or len(os)!=1:
        return None
    s,o=ss[0],os[0]
    carbon=[a for a in s.GetNeighbors() if a.GetAtomicNum()==6 and _order(mol,s.GetIdx(),a.GetIdx())==2]
    if (s.IsInRing() or s.GetDegree()!=2 or s.GetTotalValence()!=4 or len(carbon)!=1
            or o.GetDegree()!=1 or _order(mol,s.GetIdx(),o.GetIdx())!=2):
        return None
    c=carbon[0].GetIdx()
    prefix,owned=mapped_branch(mol,Chem,c,s.GetIdx())
    if not prefix or not prefix.endswith('ylidene') or owned|{s.GetIdx(),o.GetIdx()}!=set(range(mol.GetNumAtoms())):
        return None
    return _decision(mol,prefix+'-λ4-sulfanone',(s.GetIdx(),),'P-64.4.2')


def select(mol,Chem,numbers):
    chalcogens=sum(numbers[e] for e in (16,34,52))
    if numbers[16] and any(a.GetNumRadicalElectrons() for a in iter_atoms(mol)):
        result=sulfur_radical(mol,Chem)
        if result:return result
    if numbers[16]>=4:
        result=dithioic_polyacid(mol,Chem)
        if result:return result
    if (numbers[8]>=3 or numbers[8] and chalcogens>=2) and not set(numbers)-{6,7,8,9,16,17,34,35,52,53}:
        result=symmetric_anhydride(mol,Chem)
        if result:return result
    if numbers[6]>=3 and numbers[8] and (chalcogens>=2 or chalcogens and numbers[8]>=2):
        result=sulfur_pseudoester(mol,Chem)
        if result:return result
    if chalcogens==1 and (numbers[8]>=3 or numbers[7]>=2 and numbers[8]):
        result=modified_chalcogen_acid(mol,Chem)
        if result:return result
    if numbers[16]==1 and numbers[8]==1:
        result=thiocarbonyl_oxide(mol,Chem)
        if result:return result
    if numbers[7] and numbers[8]>=4:
        return peroxyacid_above_amide(mol,Chem)
    return None
