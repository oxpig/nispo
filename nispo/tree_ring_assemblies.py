"""Mapped P-28/P-29 unit trees with recursively constructed ring branches.

The selected parent is an unbranched assembly. Original characteristic-group
coverage and an actual prefix attachment constrain eligible paths before their
span is compared. Every tied junction numbering remains visible until later
suffix/prefix facts resolve it. No name-length or atom-index ranking is used.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import product
from math import prod
import re

from .molecule_cache import exact_graph_computation, mol_cache_get, naming_cache_state
from .molecule_iteration import iter_atoms, iter_bonds


RULE = 'P-28.1/P-28.2.1/P-28.3/P-28.6/P-29.2/P-29.3.5/P-44.1.1/P-45/P-52.2.7/P-16.3/P-16.5'
_MAX_HEAVY = 256
_MAX_UNITS = 32
_MAX_FRAMES = 8192


@dataclass(frozen=True, slots=True)
class AssemblyTree:
    scope: frozenset[int]
    systems: tuple[frozenset[int], ...]
    adjacency: tuple[tuple[int, ...], ...]
    connectors: tuple
    parent: str
    numberings: tuple
    census: tuple
    issue: str = ''


@dataclass(frozen=True, slots=True)
class AssemblyFrame:
    path: tuple[int, ...]
    parent_atoms: frozenset[int]
    atom_locants: tuple
    junctions: tuple
    junction_locants: tuple
    fixed_key: tuple


@dataclass(frozen=True, slots=True)
class AssemblyBranchClaim:
    root: int
    host: int
    atoms: frozenset[int]
    text: str
    nature: str
    enclosed: bool
    ring_atoms: frozenset[int]


@dataclass(frozen=True, slots=True)
class TreeAssemblyEvidence:
    tree: AssemblyTree
    frame: AssemblyFrame
    mode: str
    groups: tuple
    root: int | None
    blocked: int | None
    branches: tuple[AssemblyBranchClaim, ...]
    comparison_key: tuple


@dataclass(frozen=True, slots=True)
class TreeAssemblyProof:
    tree: AssemblyTree
    principal_centers: frozenset[int]
    rule: str = RULE
    maximum_disjoint_copies: int | None = 1

    @property
    def components(self):
        return self.tree.systems


@dataclass(frozen=True, slots=True)
class TreeAssemblyRecord:
    name: str
    construction: object
    parent_atoms: frozenset[int]
    principal_centers: frozenset[int]
    atom_locants: tuple
    suffix_locants: tuple
    component_roles: tuple
    numbering_key: tuple
    frame: AssemblyFrame
    tree: AssemblyTree
    method: str
    rule: str = RULE

    @property
    def atoms(self):
        return self.tree.scope


@dataclass(frozen=True, slots=True)
class TreeAssemblyOutcome:
    status: str
    reason: str
    decision: TreeAssemblyRecord | None = None
    proof: TreeAssemblyProof | None = None
    candidates: tuple = ()
    rule: str = RULE


def _census(mol):
    atoms = tuple((a.GetIdx(), a.GetAtomicNum(), a.GetIsotope(), a.GetFormalCharge(),
        a.GetNumRadicalElectrons(), int(a.GetChiralTag()), a.GetNumExplicitHs(),
        a.GetNoImplicit(), a.GetIsAromatic()) for a in iter_atoms(mol))
    bonds = tuple((b.GetBeginAtomIdx(), b.GetEndAtomIdx(), str(b.GetBondType()),
        b.GetBondTypeAsDouble(), b.GetIsAromatic(),
        int(b.GetStereo()), int(b.GetBondDir()), tuple(b.GetStereoAtoms())) for b in iter_bonds(mol))
    groups = tuple((str(g.GetGroupType()), tuple(a.GetIdx() for a in g.GetAtoms()))
                   for g in mol.GetStereoGroups())
    return atoms, bonds, groups


def _needed(mol):
    return frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum()>1 or a.GetIsotope())


@lru_cache(maxsize=8192)
def _location(value):
    from .locant_order import writing_locant_key
    match = re.fullmatch(r'(\d+)\^(\d+[a-z]?)', value)
    if match:
        local = writing_locant_key(match[2])
        return None if local is None else (2, int(match[1]), (3, local))
    return writing_locant_key(value)


def _series(values):
    keys = tuple(_location(value) for value in values)
    return None if any(key is None for key in keys) else tuple(sorted(keys))


def _component_locant(occurrence, local, count):
    if count > 2:
        return str(occurrence)+'^'+local
    if occurrence == 1:
        return local
    match = re.fullmatch(r'(\d+)([a-z]?)', local)
    return match[1]+"'"+match[2] if match else ''


def _tree_uncached(mol, Chem, scope):
    from .mapped_ring_parents import ring_systems, _core_graph, assembly_unit_skeletons
    systems = tuple(sorted((core for core in ring_systems(mol) if core <= scope),
                           key=lambda core:tuple(sorted(core))))
    if len(systems) < 2:
        return None
    if any(core & scope and not core <= scope for core in ring_systems(mol)):
        return None
    # Capping retains exactly the ring-system atoms and their elements. Unequal
    # element multisets therefore cannot yield identical capped unit SMILES.
    # Reject that necessary-condition failure before copying/kekulizing the
    # whole graph separately for each unit and constructing its vocabulary.
    element_counts = None
    for core in systems:
        counts = Counter(mol.GetAtomWithIdx(i).GetAtomicNum() for i in core)
        if element_counts is None:
            element_counts = counts
        elif counts != element_counts:
            return None
    owner = {i:n for n,core in enumerate(systems) for i in core}
    adjacency = [[] for _ in systems]
    connectors = []
    pairs = set()
    ordinary_junctions = True
    for bond in iter_bonds(mol):
        a,b = bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
        left,right = owner.get(a),owner.get(b)
        if left is None or right is None or left == right:
            continue
        if left > right:
            left,right,a,b = right,left,b,a
        if (left,right) in pairs:
            return None
        pairs.add((left,right));adjacency[left].append(right);adjacency[right].append(left)
        connectors.append((left,right,a,b,bond.GetBondTypeAsDouble()))
        ordinary_junctions &= (bond.GetBondType()==Chem.BondType.SINGLE and not bond.GetIsAromatic())
    if len(connectors) != len(systems)-1:
        return None
    seen,stack = set(),[0]
    while stack:
        current = stack.pop()
        if current in seen:continue
        seen.add(current);stack.extend(adjacency[current])
    if len(seen) != len(systems):return None
    base = Chem.Mol(mol)
    for atom in iter_atoms(base):atom.SetIsotope(0)
    Chem.RemoveStereochemistry(base)
    signatures,chemicals = [],[]
    # Unit identity is required regardless of vocabulary or numbering. Prove
    # it before constructing every unit's (potentially large) numbering pool.
    for core in systems:
        capped = _core_graph(base,Chem,core)
        if capped is None:return None
        chemical = Chem.Mol(capped[0])
        for atom in iter_atoms(chemical):atom.SetAtomMapNum(0)
        signatures.append(Chem.MolToSmiles(chemical,isomericSmiles=False))
        if signatures[-1] != signatures[0]:return None
        chemicals.append(chemical)
    parents,numberings = [],[]
    issue = ''
    for core,chemical in zip(systems,chemicals,strict=True):
        bonds = tuple(iter_bonds(chemical))
        saturated = all(b.GetBondTypeAsDouble()==1 for b in bonds)
        aromatic = all(a.GetIsAromatic() for a in iter_atoms(chemical))
        if not saturated and not aromatic:
            issue = 'P-31 partial-unsaturation assembly composition remains unproved.'
        if any(a.GetAtomicNum()==7 and a.GetIsAromatic() and a.GetTotalNumHs()
               for a in iter_atoms(chemical)):
            issue = 'P-28.2.3 indicated-hydrogen assembly composition remains unproved.'
        rows = tuple(assembly_unit_skeletons(base,Chem,core))
        if not rows:
            parents.append('');numberings.append(())
            issue = issue or 'A complete intrinsic cyclic-unit vocabulary/numbering frontier is unavailable.'
            continue
        fixed = min(row[6] for row in rows)
        rows = tuple(row for row in rows if row[6]==fixed)
        vocabulary = {row[0] for row in rows}
        if len(vocabulary) != 1:
            parents.append('');numberings.append(())
            issue = issue or 'Equivalent cyclic-unit numberings leave vocabulary eligibility open.'
            continue
        parent = next(iter(vocabulary))
        # P-31.1.1.1 / IUPAC von Baeyer VB-8: an unmodified von Baeyer
        # skeleton ending in -ane describes the saturated
        # hydride.  The skeleton frontier may supply that word for an
        # unsaturated unit; it is not a complete assembly component until
        # its multiple bonds have been expressed in the word.
        if (not saturated and re.search(r'(?:bi|tri|tetra|penta|hexa)cyclo\[[^]]+\][a-z]+ane$', parent)):
            issue = 'Unsaturated von Baeyer assembly unit lacks a proved unsaturation construction.'
        if re.search(r'\d+H-',parent):
            issue = 'P-28.2.3 indicated-hydrogen assembly composition remains unproved.'
        maps = tuple(sorted({tuple(sorted((i,str(loc)) for i,loc in zip(row[1],row[2],strict=True)))
                             for row in rows}))
        if any(frozenset(i for i,_ in row)!=core or any('^' in loc or _location(loc) is None
               for _,loc in row) for row in maps):
            issue = issue or 'A cyclic-unit map needs another intrinsic/composite locant grammar.'
        parents.append(parent);numberings.append(maps)
    if len(set(signatures)) != 1:return None
    if len(set(parents)) != 1:
        issue = issue or 'Identical cyclic units do not have one proved parent vocabulary.'
    if any(mol.GetAtomWithIdx(i).GetIsotope() or mol.GetAtomWithIdx(i).GetFormalCharge()
           or mol.GetAtomWithIdx(i).GetNumRadicalElectrons() or int(mol.GetAtomWithIdx(i).GetChiralTag())
           for i in scope) or any(int(b.GetStereo()) or int(b.GetBondDir()) for b in iter_bonds(mol)
               if {b.GetBeginAtomIdx(),b.GetEndAtomIdx()} & scope) or mol.GetStereoGroups():
        issue = 'Original-graph isotope/stereo/charge composition for the tree assembly remains unproved.'
    if not ordinary_junctions:
        issue = 'P-28/P-29 single-junction tree construction requires original ordinary single bonds.'
    if sum(mol.GetAtomWithIdx(i).GetAtomicNum()>1 for i in scope)>_MAX_HEAVY or len(systems)>_MAX_UNITS:
        issue = 'The proved assembly exceeds the explicit graph/unit construction bounds.'
    return AssemblyTree(scope,systems,tuple(tuple(sorted(row)) for row in adjacency),
        tuple(sorted(connectors)),parents[0],tuple(numberings),_census(mol),issue)


def _tree(mol, Chem, scope, *, validation=False):
    namespace = 'tree_assembly_validation_context' if validation else 'tree_assembly_context'
    return exact_graph_computation(mol,Chem,namespace,(scope,),lambda:_tree_uncached(mol,Chem,scope))


def _edge(tree, left, right):
    row = next(row for row in tree.connectors if {row[0],row[1]}=={left,right})
    return (row[2],row[3]) if row[0]==left else (row[3],row[2])


def _paths(tree, required):
    rows = []
    for start in range(len(tree.systems)):
        stack = [(start,(start,))]
        while stack:
            current,path = stack.pop()
            if required <= set(path) and len(path)>1:rows.append(path)
            stack.extend((other,path+(other,)) for other in tree.adjacency[current] if other not in path)
    if not rows:return ()
    maximum = max(map(len,rows))
    return tuple(sorted(path for path in rows if len(path)==maximum))


def _frames_uncached(tree, groups, root):
    owner = {a:i for i,core in enumerate(tree.systems) for a in core}
    required = {owner[g.owner] for g in groups}
    if root is not None:
        if root not in owner:return (), 'The actual free valence is not on a complete cyclic unit.'
        required.add(owner[root])
    paths = _paths(tree,required)
    if not paths:return (), 'No one assembly path contains all original principal groups; the P-51/other-parent comparison remains open.'
    if len(paths[0])>6:return (), 'P-28.5/P-52.2.7.3 requires phane construction for this principal assembly path.'
    junction_choices = {}
    options = []
    for path in paths:
        choices = []
        for position,unit in enumerate(path):
            sites = []
            if position:sites.append(_edge(tree,path[position-1],unit)[1])
            if position+1<len(path):sites.append(_edge(tree,unit,path[position+1])[0])
            key = unit, tuple(sites)
            if key not in junction_choices:
                # P-28.3.1 compares the lowest set before citation order.
                # All paths using these ordered sites share that comparison.
                ranked = []
                for row in tree.numberings[unit]:
                    mapping = dict(row)
                    cited = tuple(_location(mapping[i]) for i in sites)
                    ranked.append(((tuple(sorted(cited)), cited), row))
                minimum = min(rank for rank, _ in ranked)
                junction_choices[key] = tuple(row for rank, row in ranked if rank == minimum)
            choices.append(junction_choices[key])
        representative = [dict(maps[0]) for maps in choices]
        locants = tuple((representative[i][a],representative[i+1][b])
            for i in range(len(path)-1) for a,b in (_edge(tree,path[i],path[i+1]),))
        junction_key = tuple(_location(loc) for pair in locants for loc in pair)
        numbered_set = tuple(sorted(_location(_component_locant(position+1,
            representative[position][atom],len(path))) for position,unit in enumerate(path)
            for neighbor in tree.adjacency[unit] if neighbor in path
            for atom in (_edge(tree,unit,neighbor)[0],)))
        options.append(((numbered_set,junction_key),path,tuple(choices),locants))
    # Number each unbranched candidate in both directions before P-28.6
    # compares different parent paths. A free valence cannot override that
    # intrinsic numbering (P-29.3.5).
    intrinsic = {}
    for key,path,_choices,_locants in options:
        identity = min(path,tuple(reversed(path)))
        intrinsic[identity] = min(intrinsic.get(identity,key),key)
    numbered = []
    for key,path,choices,locants in options:
        if key!=intrinsic[min(path,tuple(reversed(path)))]:continue
        choices = list(choices)
        if root is not None:
            position = path.index(owner[root])
            minimum = min(_location(dict(row)[root]) for row in choices[position])
            choices[position] = tuple(row for row in choices[position]
                if _location(dict(row)[root])==minimum)
            root_key = (_location(_component_locant(position+1,
                dict(choices[position][0])[root],len(path))),)
        else:root_key = ()
        numbered.append(((key[1],root_key),path,tuple(choices),locants))
    options = numbered
    fixed = min(row[0] for row in options)
    options = [row for row in options if row[0]==fixed]
    if sum(prod(len(maps) for maps in row[2]) for row in options)>_MAX_FRAMES:
        return (), 'The complete tied assembly numbering frontier exceeds its explicit frame bound.'
    frames = []
    for key,path,choices,locants in options:
        for maps in product(*choices):
            mapping = tuple(sorted((atom,_component_locant(i+1,local,len(path)))
                for i,row in enumerate(maps) for atom,local in row))
            core = frozenset().union(*(tree.systems[i] for i in path))
            junctions = tuple(_edge(tree,a,b) for a,b in zip(path,path[1:]))
            frames.append(AssemblyFrame(path,core,mapping,junctions,locants,key))
    return tuple(dict.fromkeys(frames)), ''


def _frames(mol, Chem, tree, groups, root, *, validation=False):
    namespace = 'tree_assembly_validation_frames' if validation else 'tree_assembly_frames'
    return exact_graph_computation(mol,Chem,namespace,(tree,groups,root),
        lambda:_frames_uncached(tree,groups,root))


def _inventory(mol, Chem, tree, mode, groups):
    from .extended_inventory import selection_inventory
    inventory = selection_inventory(Chem.Mol(mol))
    if not inventory.supported:return None
    core = frozenset().union(*tree.systems)
    if mode=='functional':
        from .functional_ring_assemblies import _scope
        original = _scope(Chem.Mol(mol),Chem)
        if original is None or set(original[1])!=set(groups):return None
        if any(g.kind not in {'acid','amide','nitrile'} or g.owner not in core for g in groups):return None
    elif mode=='bare':
        if any(g.center not in core or not set(g.atoms)<=core for g in inventory.principal):return None
    return inventory


def _prepare(mol, Chem, tree, frame, groups, inventory):
    from .ring_state_parent import _build_ring_preparation
    from .component_mapping import hydrogen_bindings
    from .branch_rules import collect_subtree
    mapping = dict(frame.atom_locants)
    local_groups = tuple(g for g in inventory.groups if set(g.atoms)<=tree.scope)
    prepared = _build_ring_preparation(mol,Chem,frame.parent_atoms,local_groups,groups,
                                       tree.scope,(),())
    if prepared is None:return None
    nmap = {}
    if groups and groups[0].kind=='amide':
        from .amide_site_locants import external_amide_nitrogen_locants
        assignment = external_amide_nitrogen_locants(mol,Chem,groups,mapping,locant_series=_series)
        if assignment is None:return None
        nmap.update(assignment.nitrogen_locants)
    for target,host,prefix in prepared.aliases:mapping[target]=prefix+mapping[host]
    mapping.update(nmap)
    embedded = set(prepared.suffix_atoms)
    branches = []
    for host in sorted(frame.parent_atoms | nmap.keys()):
        for branch,allowed in prepared.branch_edges[host]:
            if branch in embedded:continue
            if not allowed:return None
            atoms = frozenset(collect_subtree(mol,branch,host))
            if not atoms or atoms & (mapping.keys() | embedded) or not atoms<=tree.scope:return None
            embedded.update(atoms);branches.append((branch,host,atoms))
    bindings = hydrogen_bindings(mol,mapping)
    if set(bindings) | embedded != tree.scope:return None
    suffixes = tuple(sorted((mapping[g.owner] for g in groups),key=_location))
    key = frame.fixed_key + (_series(suffixes),-len(branches),
                             _series(mapping[host] for _root,host,_atoms in branches))
    return tuple(sorted(bindings.items())),frozenset(embedded),tuple(branches),suffixes,key


def _branch(mol, Chem, root, host, atoms):
    from .mapped_ring_parents import ring_systems, records
    systems = tuple(core for core in ring_systems(mol) if core<=atoms)
    if len(systems)>1 and mol.GetAtomWithIdx(root).IsInRing():
        outcome = ring_assembly_prefix_outcome(mol,Chem,root,host,atoms)
        if outcome.status!='complete':return None
        record = outcome.decision
        return AssemblyBranchClaim(root,host,atoms,record.name,
            'compound' if record.construction.prefixes else 'simple',True,
            frozenset().union(*record.tree.systems))
    if len(systems)==1 and root in systems[0]:
        rows = records(mol,Chem,scope=atoms,attachment=root)
        if len(rows)!=1:return None
        record = rows[0]
        substituted = bool(record.prefixes)
        plain = (record.parent=='benzene' or re.fullmatch(r'cyclo[a-z]+ane',record.parent))
        return AssemblyBranchClaim(root,host,atoms,record.name,
            'compound' if substituted else 'simple',substituted or not plain,systems[0])
    from .component_mapping import mapped_branch
    text,owned = mapped_branch(mol,Chem,root,host)
    if not text or owned!=atoms:return None
    # Other established branch grammars keep their existing multiplier path;
    # no new simple/compound chemistry is inferred from their text.
    return AssemblyBranchClaim(root,host,atoms,text,'legacy',False,
                               frozenset(i for i in atoms if mol.GetAtomWithIdx(i).IsInRing()))


def _compose(mol, Chem, tree, frame, mode, groups, root, plan):
    from .component_mapping import MappedComponent
    from .formatting import multiplied_suffix, prefix_alphabetization_key
    from .functional_ring_assemblies import _parent_name
    if plan is None:return None
    bindings,embedded,specs,suffixes,key = plan
    claims = tuple(_branch(mol,Chem,a,b,atoms) for a,b,atoms in specs)
    if any(claim is None or any(c.isspace() for c in claim.text) for claim in claims):return None
    mapping = dict(bindings)
    prefixes = tuple((mapping[claim.host],claim.text) for claim in claims)
    metadata = {}
    for claim in claims:
        value = (claim.nature,bool(claim.enclosed))
        if claim.text in metadata and metadata[claim.text]!=value:return None
        metadata[claim.text]=value
    parent = tree.parent
    if mode=='prefix' and parent!='benzene':parent=parent.removesuffix('e')
    body = _parent_name(parent,len(frame.path),frame.junction_locants)
    if mode=='prefix':
        if root not in mapping:return None
        body = '['+body+']-'+mapping[root]+'-yl'
    elif groups:
        suffix = multiplied_suffix(len(groups),{'acid':'carboxylic acid',
            'amide':'carboxamide','nitrile':'carbonitrile'}[groups[0].kind])
        if not suffix:return None
        body = '['+body+']-'+','.join(suffixes)+'-'+suffix
    component = MappedComponent(body,bindings,frozenset(mapping),prefixes,embedded,
        systematic_parent=True,nitrogen_site_locants=bool(groups and groups[0].kind=='amide'),
        prefix_branch_ownership=tuple((c.root,c.host,c.atoms,c.text) for c in claims))
    citation = tuple(sorted(((prefix_alphabetization_key(text)[:3],_location(loc))
                             for loc,text in prefixes)))
    return component,claims,suffixes,key+(citation,)


def validate_tree_assembly_evidence(mol, Chem, component):
    return exact_graph_computation(mol,Chem,'tree_assembly_writer_validation',(component,),
        lambda:_validate_tree_assembly_evidence(mol,Chem,component))


def _validate_tree_assembly_evidence(mol, Chem, component):
    evidence = component.tree_assembly_evidence
    if not isinstance(evidence,TreeAssemblyEvidence):return False
    if evidence.tree.census!=_census(mol):return False
    tree = _tree(mol,Chem,evidence.tree.scope,validation=True)
    if tree is None or tree.issue or tree!=evidence.tree:return False
    if evidence.mode not in {'bare','functional','prefix'}:return False
    if evidence.mode=='prefix':
        if not _prefix_boundary(mol,tree.scope,evidence.root,evidence.blocked):return False
    elif evidence.root is not None or evidence.blocked is not None:return False
    frames,issue = _frames(mol,Chem,tree,evidence.groups,evidence.root,validation=True)
    if issue or evidence.frame not in frames:return False
    inventory = _inventory(mol,Chem,tree,evidence.mode,evidence.groups)
    if inventory is None:return False
    plan = _prepare(mol,Chem,tree,evidence.frame,evidence.groups,inventory)
    expected = _compose(mol,Chem,tree,evidence.frame,evidence.mode,evidence.groups,evidence.root,plan)
    return (expected is not None and replace(component,tree_assembly_evidence=None)==expected[0]
            and evidence.branches==expected[1] and evidence.comparison_key==expected[3])


def tree_assembly_prefix_text(component, writing_prefixes):
    """P-16 multiplication from actual mapped branch nature, in source order."""
    from .constants import MULT
    from .formatting import COMPLEX_MULT, enclose_name_unit, format_prefixes, prefix_alphabetization_key
    evidence = component.tree_assembly_evidence
    if not isinstance(evidence,TreeAssemblyEvidence):return None
    facts = {claim.text:(claim.nature,claim.enclosed) for claim in evidence.branches}
    grouped = defaultdict(list)
    for locant,text in writing_prefixes:
        if text not in facts or _location(locant) is None:return None
        grouped[text].append(locant)
    if set(grouped)!=set(facts):return None
    parts = []
    for text in sorted(grouped,key=prefix_alphabetization_key):
        locants = sorted(grouped[text],key=_location)
        nature,enclosed = facts[text]
        if nature=='legacy':
            rendered = format_prefixes([(loc,text) for loc in locants],hierarchical_enclosures=True)
            if not rendered:return None
            parts.append(rendered);continue
        if nature not in {'simple','compound'}:return None
        multiplier = (COMPLEX_MULT if nature=='compound' else MULT).get(len(locants),'') if len(locants)>1 else ''
        if len(locants)>1 and not multiplier:return None
        body = enclose_name_unit(text) if enclosed or nature=='compound' else text
        parts.append(','.join(locants)+'-'+multiplier+body)
    return '-'.join(parts)


def _record_for_frame(mol, Chem, tree, frame, mode, groups, root, blocked, plan):
    from .component_mapping import decorate_component
    value = _compose(mol,Chem,tree,frame,mode,groups,root,plan)
    if value is None:return None
    component,claims,suffixes,key = value
    evidence = TreeAssemblyEvidence(tree,frame,mode,groups,root,blocked,claims,key)
    component = replace(component,tree_assembly_evidence=evidence)
    name = decorate_component(mol,Chem,component)
    if not name:return None
    roles = tuple((f'ring-{n}',tree.systems[unit],tuple((i,dict(frame.atom_locants)[i])
        for i in sorted(tree.systems[unit]))) for n,unit in enumerate(frame.path,1))
    return TreeAssemblyRecord(name,component,frame.parent_atoms,frozenset(g.center for g in groups),
        component.atom_locants,suffixes,roles,key,frame,tree,
        'ring-assembly-prefix' if mode=='prefix' else 'functional-ring-assembly' if groups else 'ring-assembly')


def _valid_record(mol, Chem, record, tree, frame, mode, groups, root, blocked):
    from .component_mapping import decorate_component
    if not isinstance(record,TreeAssemblyRecord) or record.tree!=tree or record.frame!=frame:return False
    evidence = record.construction.tree_assembly_evidence
    if (not isinstance(evidence,TreeAssemblyEvidence) or evidence.tree!=tree or evidence.frame!=frame
            or (evidence.mode,evidence.groups,evidence.root,evidence.blocked)!=(mode,groups,root,blocked)
            or record.numbering_key!=evidence.comparison_key
            or record.parent_atoms!=frame.parent_atoms
            or record.principal_centers!=frozenset(g.center for g in groups)
            or record.atom_locants!=record.construction.atom_locants
            or record.construction.atoms | record.construction.embedded_atoms != tree.scope
            or not validate_tree_assembly_evidence(mol,Chem,record.construction)):
        return False
    return decorate_component(mol,Chem,record.construction)==record.name


def _render_uncached(mol, Chem, scope, mode, groups, root, blocked):
    tree = _tree(mol,Chem,scope)
    if tree is None:return TreeAssemblyOutcome('outside-scope','No complete identical cyclic-unit tree was proved.')
    proof = TreeAssemblyProof(tree,frozenset(g.center for g in groups))
    pending = lambda reason,rows=():TreeAssemblyOutcome('pending',reason,None,proof,tuple(rows))
    if tree.issue:return pending(tree.issue)
    if mode=='prefix' and not _prefix_boundary(mol,scope,root,blocked):
        return pending('The actual rooted assembly scope does not have one original single-bond attachment.')
    inventory = _inventory(mol,Chem,tree,mode,groups)
    if inventory is None:
        if mode=='bare':
            return TreeAssemblyOutcome('outside-scope',
                'A senior external characteristic group or another class-specific parent is required.')
        return pending('The assembly does not yet have a complete original characteristic-group comparison.')
    frames,issue = _frames(mol,Chem,tree,groups,root)
    if issue:return pending(issue)
    plans = tuple(_prepare(mol,Chem,tree,frame,groups,inventory) for frame in frames)
    if any(plan is None for plan in plans):return pending('A maximal assembly numbering lacks complete original suffix/N/branch ownership.')
    first_key = min(plan[4] for plan in plans)
    rows = []
    for frame, plan in zip(frames, plans, strict=True):
        if plan[4] != first_key:
            continue
        record = _record_for_frame(mol,Chem,tree,frame,mode,groups,root,blocked,plan)
        if not _valid_record(mol,Chem,record,tree,frame,mode,groups,root,blocked):
            return pending('A source-maximal assembly frame lacks a complete validated writer.',rows)
        rows.append(record)
    key = min(row.numbering_key for row in rows)
    winners = tuple(row for row in rows if row.numbering_key==key)
    if len({row.name for row in winners})!=1:
        return pending('Tied assembly source facts leave another prefix comparison unresolved.',rows)
    return TreeAssemblyOutcome('complete','Complete original tree, maximal path and recursive mapped branch construction.',
                               winners[0],proof,tuple(rows))


def _render(mol, Chem, scope, mode, groups=(), root=None, blocked=None):
    guards = frozenset(mol_cache_get(mol,'branch_prefix_in_progress',()))
    return exact_graph_computation(mol,Chem,'tree_assembly_outcome',
        (scope,mode,tuple(groups),root,blocked,naming_cache_state(mol),guards),
        lambda:_render_uncached(mol,Chem,scope,mode,tuple(groups),root,blocked))


def tree_ring_assembly_outcome(mol, Chem):
    if mol is None or mol.GetRingInfo().NumRings()<2:
        return TreeAssemblyOutcome('outside-scope','Fewer than two cyclic systems.')
    return _render(mol,Chem,_needed(mol),'bare')


def functional_tree_assembly_outcome(mol, Chem, groups, original_proof):
    from .functional_ring_assemblies import FunctionalAssemblyOutcome
    outcome = _render(mol,Chem,_needed(mol),'functional',tuple(groups))
    if outcome.status=='outside-scope':
        return FunctionalAssemblyOutcome('pending','The branched assembly still requires a larger or different parent comparison.',
                                         None,original_proof,(),RULE)
    return FunctionalAssemblyOutcome(outcome.status,outcome.reason,outcome.decision,outcome.proof,outcome.candidates,RULE)


def _prefix_boundary(mol, scope, root, blocked):
    if root not in scope or blocked in scope or blocked is None:return False
    bond = mol.GetBondBetweenAtoms(root,blocked)
    if (bond is None or str(bond.GetBondType())!='SINGLE' or bond.GetIsAromatic()):return False
    outside = {(i,n.GetIdx()) for i in scope for n in mol.GetAtomWithIdx(i).GetNeighbors()
               if n.GetIdx() not in scope and n.GetAtomicNum()>1}
    return outside=={(root,blocked)}


def ring_assembly_prefix_outcome(mol, Chem, root, blocked, scope):
    scope = frozenset(scope)
    if not mol.GetAtomWithIdx(root).IsInRing():
        return TreeAssemblyOutcome('outside-scope','The free valence is not on a cyclic unit.')
    return _render(mol,Chem,scope,'prefix',(),root,blocked)


_DEFAULT_RING_ASSEMBLY_PREFIX_OUTCOME = ring_assembly_prefix_outcome


def tree_assembly_consumer_scope(mol, Chem):
    """Allow an existing small parent to consume one independently proved tree.

    The parent/suffix pool keeps its existing 64-heavy-atom budget outside
    this one prefix. Every cyclic system must belong to the completed prefix,
    and every principal group and its parent attachment must remain outside.
    This grants no new parent comparison, annotation or vocabulary rule.
    """
    return exact_graph_computation(mol,Chem,'tree_assembly_consumer_scope',(),
        lambda:_tree_assembly_consumer_scope(mol,Chem))


def _tree_assembly_consumer_scope(mol, Chem):
    from .extended_inventory import selection_inventory
    from .mapped_ring_parents import ring_systems
    from .branch_rules import collect_subtree
    if mol.GetNumHeavyAtoms()>_MAX_HEAVY+64:return False
    systems=ring_systems(mol)
    if len(systems)<2:return False
    cyclic=frozenset().union(*systems)
    inventory=selection_inventory(Chem.Mol(mol))
    if not inventory.supported or not inventory.principal:return False
    for root in cyclic:
        for neighbor in mol.GetAtomWithIdx(root).GetNeighbors():
            blocked=neighbor.GetIdx()
            if blocked in cyclic or neighbor.GetAtomicNum()==1:continue
            scope=frozenset(collect_subtree(mol,root,blocked))
            if (not cyclic<=scope or any(g.center in scope or g.owner in scope
                    or set(g.atoms)&scope for g in inventory.principal)
                    or sum(a.GetAtomicNum()>1 and a.GetIdx() not in scope
                           for a in iter_atoms(mol))>64):continue
            outcome=ring_assembly_prefix_outcome(mol,Chem,root,blocked,scope)
            if outcome.status=='complete':return True
    return False
