"""Structural evidence for P-25 polycomponent and multiparent selection.

This parses component occurrences and shared paths, not complete PIN answers.
Unsupported or ambiguous bindings delegate. Numbering the completed molecule
remains the renderer's responsibility.
"""

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
import re

from .molecule_iteration import iter_atoms, iter_bonds
from .parent_selection import locant_key, locant_set


_MULTIPLIERS = {'di':2, 'tri':3, 'tetra':4, 'penta':5, 'hexa':6,
                'bis':2, 'tris':3, 'tetrakis':4, 'pentakis':5, 'hexakis':6}
_NUMBER = re.compile(r"(\d+[a-z]*)(\'*)$")
_LETTER = re.compile(r"([a-z])(\'*)")


@dataclass(frozen=True, slots=True)
class FusionLink:
    child: int
    target: int
    child_locants: tuple[str, ...]
    target_locants: tuple[str, ...]
    letters: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True, slots=True)
class FusionNetwork:
    components: tuple
    parents: tuple[int, ...]
    links: tuple[FusionLink, ...]
    orders: tuple[int, ...]
    interparents: frozenset[int]
    elements: tuple
    atom_count: int
    ring_count: int
    skeleton: str
    benzo_parent: bool = False
    citation: bool | None = True
    names: tuple[str, ...] = ()
    groupings: tuple = ()
    active: tuple = ()
    shared_junctions: int = 0
    grouped_parents: bool = False
    citation_groups: tuple = ()
    unit_multipliers: tuple = ()

    @property
    def parent(self):
        return self.components[self.parents[0]]


def construction_components(construction):
    if isinstance(construction, FusionNetwork):
        return construction.components
    return (construction.parent, construction.attached)


def skeleton_key(elements, bonds):
    """Element-labelled topology, independent of H state and atom order."""
    # Reversed or reordered undirected edges describe the same labelled
    # graph. Normalize their order so equivalent inputs share a cache entry.
    edges = tuple(sorted((a, b) if a <= b else (b, a) for a, b in bonds))
    return _skeleton_key_cached(tuple(elements), edges)


@lru_cache(maxsize=8192)
def _skeleton_key_cached(elements, bonds):
    from rdkit import Chem

    graph = Chem.RWMol()
    for element in elements:
        atom = Chem.Atom(element)
        atom.SetNoImplicit(True)
        graph.AddAtom(atom)
    for a,b in bonds:
        graph.AddBond(a,b,Chem.BondType.SINGLE)
    return Chem.MolToSmiles(graph.GetMol(),canonical=True)


def matches_skeleton(construction, mol, atoms):
    atoms = tuple(sorted(atoms))
    counts = tuple(sorted(Counter(mol.GetAtomWithIdx(i).GetAtomicNum() for i in atoms).items()))
    positions = {idx:j for j,idx in enumerate(atoms)}
    bonds = [(positions[b.GetBeginAtomIdx()],positions[b.GetEndAtomIdx()]) for b in iter_bonds(mol)
             if b.GetBeginAtomIdx() in positions and b.GetEndAtomIdx() in positions]
    if (len(atoms), len(bonds)-len(atoms)+1, counts) != (
            construction.atom_count, construction.ring_count, construction.elements):
        return False
    # A shared junction may be tetrahedral carbon, but a neutral graph alone
    # is not permission to assign aromatic/multiple bonds at that junction.
    for i in atoms:
        internal = [b for b in mol.GetAtomWithIdx(i).GetBonds()
                    if b.GetOtherAtomIdx(i) in positions]
        if len(internal)==4 and (mol.GetAtomWithIdx(i).GetAtomicNum()!=6
                or any(b.GetIsAromatic() or b.GetBondTypeAsDouble()!=1 for b in internal)):
            return False
    return not isinstance(construction,FusionNetwork) or construction.skeleton == skeleton_key(
        [mol.GetAtomWithIdx(i).GetAtomicNum() for i in atoms],bonds)


def _designation(text, resolve):
    """Resolve a primitive unit and its occurrence count, before splitting di-."""
    component = resolve(text)
    if component is not None:
        return component,1
    for prefix,count in _MULTIPLIERS.items():
        if text.startswith(prefix):
            remainder = text[len(prefix):]
            if remainder.startswith('(') and remainder.endswith(')'):
                remainder = remainder[1:-1]
            component = resolve(remainder)
            if component is not None:
                return component,count
    # P-25.3.6.3 / .7.3(b): parentheses delimit a component expression,
    # whose native component labels retain ownership within each copy.
    for prefix,count in _MULTIPLIERS.items():
        if prefix not in ('bis','tris','tetrakis','pentakis','hexakis'):
            continue
        if text.startswith(prefix+'(') and text.endswith(')'):
            inside = text[len(prefix)+1:-1]
            network = parse_network(inside,resolve,_fragment=True)
            if network is not None and len(network.parents)==1:
                return network,count
    return None


def _tokens(name, resolve):
    """Read component locant brackets separately from fusion descriptors."""
    result, start, i = [],0,0
    while i < len(name):
        if name[i] == '(':
            depth,j = 1,i+1
            while j < len(name) and depth:
                depth += (name[j]=='(')-(name[j]==')')
                j += 1
            if depth:
                return None
            i = j
            continue
        if name[i] != '[':
            i += 1
            continue
        end = name.find(']',i+1)
        if end < 0:
            return None
        # A bracket following a complete component designation is a fusion
        # descriptor. Leading HW/annulene locants belong to the designation.
        unit = _designation(name[start:i],resolve)
        if unit is not None:
            result.append((unit,name[i+1:end],name[start:i]))
            start = end+1
        i = end+1
    parent = _designation(name[start:],resolve)
    return (result,parent,name[start:]) if result and parent is not None else None


@lru_cache(maxsize=8192)
def _number_list(text):
    values = tuple(_NUMBER.fullmatch(part) for part in text.split(','))
    if not values or not all(values):
        return None
    primes = {len(value[2]) for value in values}
    return (tuple(value[1] for value in values),primes.pop()) if len(primes)==1 else None


@lru_cache(maxsize=2048)
def _numbered_path(component, values):
    """Expand only uniquely implied peripheral fusion-carbon locants."""
    labels = dict(component.label_elements)
    if len(values) < 2 or any(value not in labels for value in values):
        return None
    graph = {value:set() for value in labels}
    for edge in component.bonds:
        a,b = tuple(edge); graph[a].add(b); graph[b].add(a)
    result = [values[0]]
    for target in values[1:]:
        paths = []
        def walk(path):
            if len(paths)>1:
                return
            if path[-1] == target:
                paths.append(path)
                return
            for nxt in graph[path[-1]]-set(path):
                if nxt == target or labels[nxt]==6 and re.fullmatch(r'\d+[a-z]+',nxt):
                    walk(path+[nxt])
        walk([result[-1]])
        if len(paths)!=1:
            return None
        result.extend(paths[0][1:])
    return tuple(result) if len(set(result))==len(result) else None


@lru_cache(maxsize=2048)
def _perimeter(component):
    # Number order is accepted only when it actually forms a closed native
    # perimeter. Traditional exceptional numberings and interior atoms do not
    # acquire guessed letter bindings.
    labels = tuple(sorted(dict(component.label_elements),key=locant_key))
    if labels[:2] == ('1','2') and all(frozenset((a,b)) in component.bonds
                                      for a,b in zip(labels,labels[1:]+labels[:1])):
        return labels
    # P-25.3.3 retains the traditional numbering of acridine and certain
    # other parents. Recover their actual outer boundary from the component
    # graph rather than assuming that sorted native locants trace it.
    from rdkit import Chem
    positions = {label:i for i,label in enumerate(labels)}
    rw = Chem.RWMol()
    elements = dict(component.label_elements)
    for label in labels:
        atom = Chem.Atom(elements[label]); atom.SetNoImplicit(True); rw.AddAtom(atom)
    for edge in component.bonds:
        a,b = tuple(edge); rw.AddBond(positions[a],positions[b],Chem.BondType.SINGLE)
    graph = rw.GetMol()
    faces = tuple(tuple(ring) for ring in Chem.GetSymmSSSR(graph))
    if len(faces) != component.ring_count:
        return None
    owners = Counter(frozenset((a,b)) for face in faces
                     for a,b in zip(face,face[1:]+face[:1]))
    if any(count not in {1,2} for count in owners.values()):
        return None
    boundary = {i:set() for i in range(len(labels))}
    for edge,count in owners.items():
        if count == 1:
            a,b = tuple(edge); boundary[a].add(b); boundary[b].add(a)
    boundary = {i:neighbors for i,neighbors in boundary.items() if neighbors}
    if (labels[:2] != ('1','2') or any(len(neighbors) != 2 for neighbors in boundary.values())
            or positions['2'] not in boundary.get(positions['1'],set())):
        return None
    start, current, previous = positions['1'], positions['2'], positions['1']
    path = [start]
    while current != start:
        if current in path or current not in boundary:
            return None
        path.append(current)
        following = boundary[current]-{previous}
        if len(following) != 1:
            return None
        previous,current = current,next(iter(following))
    if len(path) != len(boundary):
        return None
    return tuple(labels[i] for i in path)


def _lettered_path(component, text):
    matches = tuple(_LETTER.finditer(text))
    if not matches or ''.join(match[0] for match in matches)!=text:
        return None
    primes = {len(match[2]) for match in matches}
    perimeter = _perimeter(component)
    if len(primes)!=1 or perimeter is None:
        return None
    sides = [ord(match[1])-ord('a') for match in matches]
    if any(side>=len(perimeter) for side in sides) or any(
            (a+1)%len(perimeter)!=b for a,b in zip(sides,sides[1:])):
        return None
    path = tuple(perimeter[side] for side in sides)+(perimeter[(sides[-1]+1)%len(perimeter)],)
    return path,primes.pop(),tuple((match[1],len(match[2])) for match in matches)


def _descriptors(text, copies):
    if '-' not in text and ':' not in text and ';' not in text:
        # Abbreviated first-order carbon descriptors: dibenzo[b,e].
        if re.fullmatch(r"[a-z]+'*(?:,[a-z]+'*)*",text):
            return [('1,2',part,'letter',True) for part in text.split(',')]
        return [(None,text,'number',True)]  # terminal carbon, higher order
    if ';' in text:
        parts = text.split(';')
        result = []
        for part in parts:
            if part.count(':')!=1:
                return None
            a,b = part.split(':'); result.append((a,b,'number',False))
        return result
    if '-' in text:
        result = []
        for part in text.split(':'):
            if part.count('-')!=1:
                return None
            a,b = part.split('-'); result.append((a,b,'letter',False))
        return result
    parts = text.split(':')
    if copies>1 and len(parts)==copies and all(_number_list(part) for part in parts):
        return [(None,part,'number',True) for part in parts]
    if len(parts)==2:
        return [(parts[0],parts[1],'number',False)]
    return None


def _tree_paths(graph, start, end):
    queue = deque([(start,(start,))])
    visited = {start}
    while queue:
        node,path = queue.popleft()
        if node==end:
            return path
        for nxt in graph[node]-visited:
            visited.add(nxt); queue.append((nxt,path+(nxt,)))
    return None


def _finish(components, parents, links, root_spelling, *, expected_shape=None):
    graph = {i:set() for i in range(len(components))}
    for link in links:
        graph[link.child].add(link.target); graph[link.target].add(link.child)
    if sum(map(len,graph.values()))!=2*(len(components)-1):
        return None  # Shared-component cycles require a different model.
    orders = [None]*len(components)
    queue = deque(parents)
    for i in parents: orders[i]=0
    while queue:
        i = queue.popleft()
        for j in graph[i]:
            if orders[j] is None:
                orders[j]=orders[i]+1; queue.append(j)
    if None in orders:
        return None
    interparents = set()
    if len(parents)>1:
        from itertools import combinations
        for a,b in combinations(parents,2):
            path = _tree_paths(graph,a,b)
            inner = path[1:-1]
            if len(inner)%2!=1 or any(components[i].identity!=components[j].identity
                    for i,j in zip(inner,reversed(inner))):
                return None
            interparents.update(inner)
        if len(parents)>2 and len(interparents)!=1:
            return None  # General branched extended interparent symmetry unproved.

    atoms, bonds, node_atoms = [],[],[]
    for component in components:
        mapping = {}
        for label,element in component.label_elements:
            mapping[label]=len(atoms); atoms.append(element)
        node_atoms.append(mapping)
        bonds.extend(tuple(mapping[v] for v in edge) for edge in component.bonds)
    union = list(range(len(atoms)))
    def representative(i):
        while union[i]!=i:
            union[i]=union[union[i]]; i=union[i]
        return i
    for link in links:
        if len(link.child_locants)!=len(link.target_locants):
            return None
        for a,b in zip(link.child_locants,link.target_locants):
            a,b = representative(node_atoms[link.child][a]),representative(node_atoms[link.target][b])
            if atoms[a]!=atoms[b]:
                return None  # P-25.3.2.5.1 shared heteroatoms belong to both units.
            union[a]=b
    sets = [{representative(i) for i in mapping.values()} for mapping in node_atoms]
    if any(len(values)!=component.atom_count for values,component in zip(sets,components)):
        return None
    # Nonoverlapping ring locations need not be atom-disjoint: adjacent
    # fusion sides can meet at one junction (P-25.3.7.1). Nonadjacent
    # component occurrences may share that vertex, never a ring path.
    if any(len(sets[a]&sets[b])>1 for a in graph for b in graph
           if a<b and b not in graph[a]):
        return None
    representatives = sorted({representative(i) for i in range(len(atoms))})
    positions = {old:i for i,old in enumerate(representatives)}
    elements = [atoms[i] for i in representatives]
    edges = {tuple(sorted((positions[representative(a)],positions[representative(b)]))) for a,b in bonds}
    if any(a==b for a,b in edges):
        return None
    degrees = Counter(i for edge in edges for i in edge)
    if expected_shape is not None:
        atom_count, bond_count, labelled_degrees = expected_shape
        if (len(elements) != atom_count or len(edges) != bond_count
                or Counter((element, degrees[i]) for i, element in enumerate(elements)) != labelled_degrees):
            return None
    junctions = {positions[old] for old in representatives
                 if sum(old in values for values in sets)>=3}
    if any(degrees[i]>(2 if element in (8,16,34,52) else
                       4 if element==6 and i in junctions else 3)
           for i,element in enumerate(elements)):
        return None
    rank = len(edges)-len(elements)+1
    if rank!=sum(component.ring_count for component in components):
        return None
    return FusionNetwork(tuple(components),tuple(parents),tuple(links),tuple(orders),
        frozenset(interparents),tuple(sorted(Counter(elements).items())),len(elements),rank,
        skeleton_key(elements,edges),len(parents)==1 and 'benzo' in root_spelling,
        shared_junctions=sum(degrees[i]==4 for i in junctions))


def parse_network(name, resolve, *, _fragment=False, expected_shape=None):
    """Parse primitive occurrences, including scoped composite groups.

    A grouped prefix exposes its final component's native labels, not the
    completed fused system's numbering. A grouped multiparent exposes its
    locally active interparent components. Both expand into ordinary native
    occurrences before graph, location or component seniority is compared.
    """
    name = name.replace('′',"'")
    if len(name)>2048 or name.count('(')>16:
        return None
    parsed = _tokens(name,resolve)
    if parsed is None:
        return None
    tokens,(parent_unit,count),spelling = parsed
    parent = parent_unit.parent if isinstance(parent_unit,FusionNetwork) else parent_unit
    components,parents,links,names,groups,groupings,multipliers = [],[],[],[],[],[],[]
    def singular(text,component,copies):
        if copies==1:
            return text
        for multiplier,number in _MULTIPLIERS.items():
            if number==copies and text.startswith(multiplier):
                rest = text[len(multiplier):]
                if rest.startswith('(') and rest.endswith(')'): rest=rest[1:-1]
                if isinstance(component,FusionNetwork) or resolve(rest)==component: return rest
        return None
    from .fusion_components import _HW_NAMES
    root_name = singular(spelling,parent_unit,count)
    if root_name is None:
        return None
    root_name = _HW_NAMES.get(root_name,root_name)
    def instantiate(unit,unit_name,copies,group_id,spelling,*,parent_group=False):
        """Copy local namespaces; return their roots and local active ports."""
        roots,ports = [],[]
        for copy in range(copies):
            first = len(components)
            if not isinstance(unit,FusionNetwork):
                components.append(unit); names.append(unit_name); groups.append(group_id)
                roots.append(first)
                continue
            components.extend(unit.components); names.extend(unit.names)
            groups.extend((group_id,copy,local) for local in unit.citation_groups)
            roots.append(first+unit.parents[0])
            # Internal links retain local labels here. Their instance primes
            # are rebound after the external port establishes order/instance.
            links.extend(FusionLink(first+l.child,first+l.target,l.child_locants,
                                    l.target_locants,l.letters) for l in unit.links)
            groupings.extend((tuple(first+i for i in nodes),kind)
                             for nodes,kind in unit.groupings)
            multipliers.extend((tuple(first+i for i in nodes),word)
                               for nodes,word in unit.unit_multipliers)
            ports.extend((first+i,depth,prime+copy) for i,depth,prime in unit.active
                         if not parent_group or depth==max(unit.orders))
        if isinstance(unit,FusionNetwork):
            groupings.append((tuple(roots),'parent' if parent_group else 'attached'))
        if copies>1:
            word = next(word for word,n in _MULTIPLIERS.items()
                        if n==copies and spelling.startswith(word))
            multipliers.append((tuple(roots),word))
        return tuple(roots),ports
    parents,active = instantiate(parent_unit,root_name,count,-1,spelling,parent_group=True)
    if len(components)>32:
        return None
    # Bind the expanded parent group's internal namespaces across its copies.
    if isinstance(parent_unit,FusionNetwork):
        width = len(parent_unit.components)
        links[:] = [FusionLink(l.child,l.target,
            tuple(v.rstrip("'")+"'"*(len(v)-len(v.rstrip("'"))+l.child//width) for v in l.child_locants),
            tuple(v.rstrip("'")+"'"*(len(v)-len(v.rstrip("'"))+l.target//width) for v in l.target_locants),
            tuple((letter,prime+l.target//width) for letter,prime in l.letters)) for l in links]
    # Most recently cited lower-order unit/group. Different branches may
    # reuse primes; only explicit current-group references establish binding.
    link_groups = [tuple(links)] if links else []
    for group_id,((unit,copies),text,prefix) in enumerate(reversed(tokens)):
        component = unit.parent if isinstance(unit,FusionNetwork) else unit
        group_start = len(links)
        descriptions = _descriptors(text,copies)
        if (not active and count>1 and component.ring_count==parent.ring_count==1
                and component.elements==((6,component.atom_count),)
                and parent.elements==((6,parent.atom_count),) and '-' not in text and ';' not in text):
            parts = text.split(':')
            if len(parts)==count and all(_number_list(part) for part in parts):
                descriptions = [(part,'a'+"'"*i,'letter',False) for i,part in enumerate(parts)]
        if not descriptions:
            return None
        first = len(components)
        if first+copies*(len(unit.components) if isinstance(unit,FusionNetwork) else 1)>32:
            return None
        unit_spelling = prefix
        prefix = singular(prefix,unit,copies)
        if prefix is None: return None
        from .resource_templates import _fusion_component_parent_name
        unit_name = (prefix[:-1] + 'e' if prefix.endswith(('aleno', 'apheno', 'aceno'))
                     else _fusion_component_parent_name(prefix))
        unit_name = _HW_NAMES.get(unit_name,unit_name)
        roots,_ = instantiate(unit,unit_name,copies,group_id,unit_spelling)
        internal_end = len(links)
        assigned = {}
        new_active = []
        for ordinal,(child_text,target_text,kind,implicit) in enumerate(descriptions):
            if implicit and (component.ring_count!=1 or component.elements!=((6,component.atom_count),)):
                return None
            child = _number_list(child_text) if child_text is not None else (('1','2'),None)
            if child is None:
                return None
            child_values,prime = child
            child_path = None if implicit else _numbered_path(component,child_values)
            if not implicit and child_path is None:
                return None
            letters = ()
            if kind=='letter':
                target = _lettered_path(parent,target_text)
                if target is None or target[1]>=count:
                    return None
                target_path,target_prime,letters = target
                target_node,depth = parents[target_prime],1
            else:
                target = _number_list(target_text)
                if target is None:
                    return None
                target_values,target_prime = target
                choices = [(node,depth) for node,depth,p in active if p==target_prime
                           and _numbered_path(components[node],target_values) is not None]
                if (_fragment and not choices and target_prime==0 and count==1
                        and _numbered_path(parent,target_values) is not None):
                    choices = [(parents[0],0)]
                if len(choices)!=1:
                    return None
                target_node,lower_depth = choices[0]
                target_path = _numbered_path(components[target_node],target_values)
                depth = lower_depth+1
            if implicit:
                # An abbreviated terminal monocyclic hydrocarbon has no
                # cited child locants. Its vertices are equivalent under
                # rotation/reflection; use a path of the target's length.
                perimeter = _perimeter(component)
                if (component.ring_count != 1 or perimeter is None
                        or any(element != 6 for _label, element in component.label_elements)
                        or not 2 <= len(target_path) < len(perimeter)):
                    return None
                child_path = tuple(perimeter[:len(target_path)])
            if prime is None:
                prime = max((p for _,_,p in active),default=depth-2)+1
            if implicit and copies>1:
                prime = depth-1+ordinal
            if not implicit and kind=='number' and prime<depth-1 and not (_fragment and prime==0):
                return None
            if copies==1:
                child_node = roots[0]
            else:
                if implicit:
                    copy = ordinal
                else:
                    minimum = min(_number_list(a)[1] for a,_,_,_ in descriptions)
                    copy = prime-minimum
                if not 0<=copy<copies:
                    return None
                child_node = roots[copy]
            if child_node in assigned and assigned[child_node]!=(depth,prime):
                return None
            assigned[child_node]=(depth,prime)
            links.append(FusionLink(child_node,target_node,
                tuple(value+"'"*prime for value in child_path),
                tuple(value+"'"*target_prime for value in target_path),letters))
            new_active.append((child_node,depth,prime))
        if len(assigned)!=copies:
            return None
        if isinstance(unit,FusionNetwork):
            width = len(unit.components)
            primes = {}
            for copy,root in enumerate(roots):
                depth,prime = assigned[root]
                for local in range(width):
                    node = first+copy*width+local
                    same_layer = [i for i in range(width) if unit.orders[i]==unit.orders[local]]
                    primes[node] = prime+unit.orders[local]+same_layer.index(local)
                new_active.extend((first+copy*width+i,depth+d,primes[first+copy*width+i])
                                  for i,d,_ in unit.active)
            links[group_start:internal_end] = [FusionLink(l.child,l.target,
                tuple(v.rstrip("'")+"'"*primes[l.child] for v in l.child_locants),
                tuple(v.rstrip("'")+"'"*primes[l.target] for v in l.target_locants))
                for l in links[group_start:internal_end]]
        # References in links keep primes for comparisons; graph union uses
        # native component labels, so strip only those instance primes there.
        minimum_depth = min(depth for depth,_ in assigned.values())
        active = [entry for entry in active if entry[1]<minimum_depth]+list(dict.fromkeys(new_active))
        # Higher components precede their receiving unit in citation.
        link_groups.append(tuple(links[group_start:]))
    if len(components)>32:
        return None
    native_links = tuple(FusionLink(link.child,link.target,
        tuple(value.rstrip("'") for value in link.child_locants),
        tuple(value.rstrip("'") for value in link.target_locants),link.letters) for link in links)
    result = _finish(components, parents, native_links, spelling, expected_shape=expected_shape)
    if result is None:
        return None
    from dataclasses import replace
    result = replace(result,links=tuple(link for group in reversed(link_groups) for link in group),
                     names=tuple(names),groupings=tuple(groupings),active=tuple(active),
                     grouped_parents=isinstance(parent_unit,FusionNetwork),citation_groups=tuple(groups),
                     unit_multipliers=tuple(multipliers))
    return replace(result,citation=_citation_proof(result,groups))


def _component_graph(network):
    graph = {i:set() for i in range(len(network.components))}
    for link in network.links:
        graph[link.child].add(link.target); graph[link.target].add(link.child)
    return graph


def _native_binding(network, child, target):
    link = next(l for l in network.links if {l.child,l.target}=={child,target})
    a,b = (link.child_locants,link.target_locants) if link.child==child else (link.target_locants,link.child_locants)
    return tuple(v.rstrip("'") for v in a),tuple(v.rstrip("'") for v in b)


def _branch_identity(network, node, previous, *, graph=None):
    """Identity of the entire attached unit, including internal fusion sites.

    Primes distinguish occurrences, not chemical identities. Keep native
    labels and their paired orientation; do not reduce groups to a formula,
    topology alone, or a list of constituent component names.
    """
    if graph is None:
        graph = _component_graph(network)
    def visit(i,parent):
        branches = []
        for child in graph[i]-{parent}:
            own,other = _native_binding(network,child,i)
            binding = min((own,other),(own[::-1],other[::-1]))
            branches.append((binding,visit(child,i)))
        return network.components[i].identity,tuple(sorted(branches))
    return visit(node,previous)


def _branch_citation_key(network, node, previous, *, graph=None):
    """P-25.3.4.2.3: multipart alphabet, internal sites, then heterolocants."""
    if graph is None:
        graph = _component_graph(network)
    def visit(i,parent):
        children = graph[i]-{parent}
        units = defaultdict(list)
        for child in children:
            units[_branch_identity(network, child, i, graph=graph)].append(child)
        parts,locants,hetero = [],[],[]
        for nodes in units.values():
            text,sites,positions = visit(nodes[0],i)
            if text is None or len(nodes)>6:
                return None,(),()
            if len(nodes)>1:
                # This multiplier is integral to the enclosing multipart
                # prefix, so P-25.3.4.2.3.2 includes it in the alphabet.
                declared = next((word for roots,word in network.unit_multipliers
                                 if set(roots)==set(nodes)),None)
                if declared is not None:
                    text = declared+text
                elif graph[nodes[0]]-{i}:
                    text = ('bis','tris','tetrakis','pentakis','hexakis')[len(nodes)-2]+text
                else:
                    multiplied = _multiplied(_prefix(network.names[nodes[0]]),len(nodes))
                    if multiplied is None: return None,(),()
                    text = re.sub(r'[^a-z]','',multiplied)
            parts.append((text,sites,positions,nodes))
        for text,sites,positions,nodes in sorted(parts,key=lambda item:item[:3]):
            locants.extend(sites); hetero.extend(positions)
            for child in nodes:
                own,other = _native_binding(network,child,i)
                locants.extend(locant_set(own+other))
        name = _prefix(network.names[i]) if network.names else None
        if name is None:
            return None,(),()
        alpha = ''.join(part[0] for part in sorted(parts,key=lambda item:item[:3]))+re.sub(r'[^a-z]','',name)
        positions = [label for label,e in network.components[i].label_elements if e!=6]
        hetero.extend(locant_set(positions))
        return alpha,tuple(locants),tuple(hetero)
    return visit(node,previous)


def _citation_proof(network, groups):
    """Prove sibling citation and exact primitive/composite multiplication."""
    graph = _component_graph(network)
    valid = True
    for i in graph:
        children = [j for j in graph[i] if network.orders[j]>network.orders[i]
                    and j not in network.interparents]
        if len(children)<2:
            continue
        cited = sorted(children,reverse=True)
        keys = [_branch_citation_key(network, j, i, graph=graph) for j in cited]
        if any(key[0] is None for key in keys):
            return None
        valid &= keys==sorted(keys)
        identities = defaultdict(list)
        for j in children:
            identities[_branch_identity(network, j, i, graph=graph)].append(j)
        for identical in identities.values():
            if len(identical)<2:
                continue
            if any(graph[j]-{i} for j in identical):
                valid &= any(kind=='attached' and set(nodes)==set(identical)
                             for nodes,kind in network.groupings)
            else:
                valid &= len({groups[j] for j in identical})==1
    return valid


def ordered_components(components):
    """Sort only when every required comparison is proved."""
    from .fusion_components import compare_components

    result = []
    for component in components:
        for i,previous in enumerate(result):
            comparison = compare_components(component,previous).preference
            if comparison is None:
                return None
            if comparison==-1:
                result.insert(i,component)
                break
        else:
            result.append(component)
    # Insertion comparisons alone cannot prove a partial comparator total.
    if any(compare_components(a,b).preference not in (-1,0)
           for i,a in enumerate(result) for b in result[i+1:]):
        return None
    return tuple(result)


def _as_network(construction):
    if isinstance(construction,FusionNetwork):
        return construction
    child = tuple(str(value[1])+value[2]+"'"*value[3] for value in construction.numbers)
    link = FusionLink(1,0,child,(),((construction.letter,0),))
    return FusionNetwork((construction.parent,construction.attached),(0,),(link,),(0,1),
        frozenset(),construction.elements,construction.atom_count,construction.ring_count,'',
        construction.benzo_parent)


def _component_rows(left, right, rule):
    from .fusion_components import component_criteria

    if sorted(c.identity for c in left)==sorted(c.identity for c in right):
        return
    a,b = ordered_components(left),ordered_components(right)
    if a is None or b is None:
        yield rule,None,None
        return
    for x,y in zip(a,b):
        for subrule,av,bv in component_criteria(x,y):
            yield rule+':'+subrule,av,bv
    if len(a)!=len(b):
        yield rule,None,None


def _multiply_count(network):
    graph = _component_graph(network)
    groups = defaultdict(list)
    for target in graph:
        for node in graph[target]:
            if (network.orders[node]>network.orders[target]
                    and node not in network.interparents):
                groups[(target, _branch_identity(network, node, target, graph=graph))].append(node)
    repeated_interparents = Counter(network.orders[i] for i in network.interparents)
    return (sum(len(nodes) for nodes in groups.values() if len(nodes)>1)
            + sum(count for count in repeated_interparents.values() if count>1))


def _benzo_exception(multi, single):
    """P-25.3.5.3 applies to explicit benzoheterocycle units, not retained ones."""
    if len(multi.parents)<2 or len(single.parents)!=1 or not single.benzo_parent:
        return False
    root = single.parent
    if root.ring_count!=2 or multi.parent.ring_count!=1:
        return False
    from rdkit import Chem
    def graph(component):
        labels = dict(component.label_elements)
        order = list(labels); positions = {label:i for i,label in enumerate(order)}
        mol = Chem.RWMol()
        for label in order: mol.AddAtom(Chem.Atom(labels[label]))
        for edge in component.bonds:
            a,b = tuple(edge); mol.AddBond(positions[a],positions[b],Chem.BondType.SINGLE)
        return mol.GetMol()
    mol = graph(root)
    rings = Chem.GetSymmSSSR(mol)
    if len(rings)!=2 or not any(len(r)==6 and all(mol.GetAtomWithIdx(i).GetAtomicNum()==6 for i in r)
                               for r in rings):
        return False
    key = multi.parent
    for ring in rings:
        if len(ring)<5 or len(ring)!=key.atom_count:
            continue
        elements = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in ring]
        if all(e==6 for e in elements):
            continue
        target = graph(key)
        positions = {idx:j for j,idx in enumerate(ring)}
        edges = [(positions[b.GetBeginAtomIdx()],positions[b.GetEndAtomIdx()]) for b in iter_bonds(mol)
                 if b.GetBeginAtomIdx() in positions and b.GetEndAtomIdx() in positions]
        target_edges = [(b.GetBeginAtomIdx(),b.GetEndAtomIdx()) for b in iter_bonds(target)]
        if skeleton_key(elements,edges)==skeleton_key([a.GetAtomicNum() for a in iter_atoms(target)],target_edges):
            return True
    return False


def network_criteria(left, right):
    """P-25 identity, location, layer selection, then descriptor criteria."""
    from .fusion_components import component_criteria

    a,b = _as_network(left),_as_network(right)
    if _benzo_exception(a,b) or _benzo_exception(b,a):
        yield 'P-25.3.5.3',-int(_benzo_exception(a,b)),-int(_benzo_exception(b,a))
    yield from component_criteria(a.parent,b.parent)
    yield 'P-25.3.7.1',-len(a.parents),-len(b.parents)
    yield 'P-25.3.4.2.1(a)',0,0  # Both are proved fusion-only constructions.
    yield 'P-25.3.4.2.1(b)',int(max(a.orders)>1),int(max(b.orders)>1)
    for depth in range(1,max(max(a.orders),max(b.orders))+1):
        yield 'P-25.3.4.2.1(c)',-a.orders.count(depth),-b.orders.count(depth)
    multiplied = [_multiply_count(n) for n in (a,b)]
    yield 'P-25.3.4.2.1(d)',*(None if value is None else -value for value in multiplied)
    for depth in range(1,max(max(a.orders),max(b.orders))+1):
        rows = [tuple(n.components[i] for i in n.interparents if n.orders[i]==depth) for n in (a,b)]
        yield from _component_rows(*rows,'P-25.3.4.2.1(e)')
    for depth in range(1,max(max(a.orders),max(b.orders))+1):
        rows = [tuple(c for i,c in enumerate(n.components) if n.orders[i]==depth
                      and i not in n.interparents) for n in (a,b)]
        yield from _component_rows(*rows,'P-25.3.4.2.1(f)/P-25.3.4.2.2(a)')

    yield 'P-52.2.4.3',int(a.grouped_parents),int(b.grouped_parents)

    if a.citation is None or b.citation is None or not a.citation and not b.citation:
        yield 'P-25.3.4.2.3/P-25.3.6.1',None,None
    else:
        yield 'P-25.3.4.2.3/P-25.3.6.1',int(not a.citation),int(not b.citation)

    # The component choices above must all tie before location descriptors
    # can distinguish constructions. Preserve primes and layer ownership.
    first = [tuple(link for link in n.links if link.target in n.parents) for n in (a,b)]
    numbers = [tuple(value for link in links for value in link.child_locants) for links in first]
    yield 'P-25.3.4.2.2(b)',locant_set(numbers[0]),locant_set(numbers[1])
    yield 'P-25.3.4.2.2(c)',tuple(map(locant_key,numbers[0])),tuple(map(locant_key,numbers[1]))
    letters = [tuple(value for link in links for value in link.letters) for links in first]
    yield 'P-25.3.4.2.4(a)',tuple(sorted(letters[0])),tuple(sorted(letters[1]))
    # In a multiparent construction, the connecting component establishes
    # the parent occurrences and their primes before additional attachments
    # to those parents are considered (P-25.3.7.2).
    citation_letters = []
    for network, links in zip((a, b), first):
        if len(network.parents) > 1:
            links = (tuple(link for link in links if link.child in network.interparents)
                     + tuple(link for link in links if link.child not in network.interparents))
        citation_letters.append(tuple(value for link in links for value in link.letters))
    yield 'P-25.3.4.2.4(b)',citation_letters[0],citation_letters[1]
    lower,upper = [[],[]],[[],[]]
    for depth in range(2,max(max(a.orders),max(b.orders))+1):
        for side,n in enumerate((a,b)):
            links = tuple(link for link in n.links if n.orders[link.child]==depth and not link.letters)
            lower[side].append(tuple(value for link in links for value in link.target_locants))
            upper[side].append(tuple(value for link in links for value in link.child_locants))
    for rule,values,as_set in [('e',lower,True),('f',lower,False),('g',upper,True),('h',upper,False)]:
        key = lambda rows: tuple(locant_set(row) if as_set else tuple(map(locant_key,row)) for row in rows)
        yield 'P-25.3.4.2.4('+rule+')',key(values[0]),key(values[1])


_RETAINED_PREFIXES = dict(zip(
    ('anthracene','naphthalene','benzene','phenanthrene','furan','imidazole','pyridine','pyrimidine','thiophene'),
    ('anthra','naphtho','benzo','phenanthro','furo','imidazo','pyrido','pyrimido','thieno')))


def _prefix(name):
    if name in _RETAINED_PREFIXES:
        return _RETAINED_PREFIXES[name]
    # Small carbocyclic prefixes cannot simply become parent names on rerooting.
    if name.startswith('cyclo') and name.endswith('a'):
        return name
    annulene = re.fullmatch(r'\[(\d+)\]annulene',name)
    if annulene is not None:
        from .constants import ROOTS
        root = ROOTS.get(int(annulene[1]))
        return 'cyclo'+root+'a' if root is not None else None
    return name[:-1]+'o' if name.endswith('e') else name+'o'


def _multiplied(name,count):
    if count==1: return name
    if not 2<=count<=6: return None
    if name.startswith('[') and not re.fullmatch(r'\[\d+\]annulene',name):
        return ('bis','tris','tetrakis','pentakis','hexakis')[count-2]+'('+name+')'
    return ('di','tri','tetra','penta','hexa')[count-2]+name


def _first_order_binding(a, b, parent):
    perimeter = _perimeter(parent)
    if perimeter is None: return None
    for x,y in ((a,b),(a[::-1],b[::-1])):
        for start in range(len(perimeter)):
            if tuple(perimeter[(start+j)%len(perimeter)] for j in range(len(y)))==y:
                letters = ''.join(chr(ord('a')+(start+j)%len(perimeter)) for j in range(len(y)-1))
                return x,letters
    return None


def _omit_peri_fusion_carbons(component, values):
    """P-25.3.8.4: omit recoverable internal peripheral fusion-C locants."""
    values = tuple(values)
    if len(values) <= 2:
        return values
    elements = dict(component.label_elements)
    degrees = Counter(label for edge in component.bonds for label in edge)
    perimeter = _perimeter(component)
    if perimeter is None:
        return values
    shortened = tuple(label for position, label in enumerate(values)
                      if position in (0, len(values)-1) or elements.get(label) != 6
                      or degrees[label] <= 2 or label not in perimeter)
    # Retain a full spelling if the omitted junctions are not uniquely
    # recoverable from the component's labelled graph.
    return shortened if _numbered_path(component, shortened) == values else values


def _implicit_terminal_hydrocarbon(component, values):
    """P-25.3.8.1/.2: terminal monocyclic hydrocarbon ortho-fusion."""
    return (2 <= len(values) < component.atom_count and component.ring_count == 1
            and all(element == 6 for _label, element in component.label_elements))


def _fusion_numbers(component, locants, prime):
    return ','.join(locant + "'" * prime for locant in _omit_peri_fusion_carbons(component, locants))


@lru_cache(maxsize=4096)
def _fusion_descriptor_text(child_component, target_component, own, other,
                            first_order, prime, target_prime, terminal):
    """Render one native attachment with the exact primes assigned by its caller."""
    if first_order:
        binding = _first_order_binding(own, other, target_component)
        if binding is None:
            return None
        own, letters = binding
        target = ''.join(letter + "'" * target_prime for letter in letters)
        separator = '-'
    else:
        own, other = min(((own, other), (own[::-1], other[::-1])),
                         key=lambda pair: tuple(map(locant_key, pair[1] + pair[0])))
        target = _fusion_numbers(target_component, other, target_prime)
        separator = ':'
    if terminal and _implicit_terminal_hydrocarbon(child_component, own):
        return target
    return _fusion_numbers(child_component, own, prime) + separator + target


def _fusion_descriptor(network, child, target, first_order, prime, target_prime, *, terminal=False):
    # The caller already has the edge in its component tree. Share its spelling
    # across complete numberings that select the same two native bindings.
    own, other = _native_binding(network, child, target)
    return _fusion_descriptor_text(network.components[child], network.components[target],
                                   own, other, first_order, prime, target_prime, terminal)


def _branched_multiparent(network, roots, center, graph, sibling_orders):
    """Render branches on parents and on a single interparent component."""
    def branch(node, previous, depth, prime, target_prime):
        child_nodes = graph[node]-{previous}
        prefix = _prefix(network.names[node])
        descriptor = _fusion_descriptor(network, node, previous, depth == 1, prime, target_prime,
                                        terminal=not child_nodes)
        if prefix is None or descriptor is None:
            return None
        children = siblings(node, child_nodes, depth+1, prime)
        return None if children is None else children+prefix+'['+descriptor+']'

    def sibling_parts(target, children, depth, target_prime):
        groups = defaultdict(list)
        for child in children:
            groups[_branch_identity(network, child, target, graph=graph)].append(child)
        parts = []
        for group in groups.values():
            if len(group) == 1:
                text = branch(group[0], target, depth, depth-1, target_prime)
            else:
                if graph[group[0]]-{target}:
                    return None
                prefix = _prefix(network.names[group[0]])
                prefix = _multiplied(prefix, len(group)) if prefix is not None else None
                if prefix is None:
                    return None
                order = next((tuple(node for node in candidate if node in group)
                              for indices, candidate in (sibling_orders or {}).items()
                              if frozenset(group) <= indices), tuple(sorted(group)))
                descriptions = []
                for ordinal, node in enumerate(order):
                    descriptor = _fusion_descriptor(network, node, target, depth == 1,
                                                    depth - 1 + ordinal, target_prime)
                    if descriptor is None:
                        return None
                    descriptions.append(descriptor)
                text = prefix+'['+(':' if depth == 1 else ';').join(descriptions)+']'
            key = _branch_citation_key(network, group[0], target, graph=graph)
            if text is None or key[0] is None:
                return None
            parts.append((text, key))
        return parts

    def siblings(target, children, depth, target_prime):
        parts = sibling_parts(target, children, depth, target_prime)
        return None if parts is None else ''.join(text for text, _ in sorted(parts, key=lambda item: item[1]))

    root_parts = []
    for prime, root in enumerate(roots):
        parts = sibling_parts(root, graph[root]-{center}, 1, prime)
        if parts is None:
            return None
        root_parts.extend(parts)
    root_text = ''.join(text for text, _ in sorted(root_parts, key=lambda item: item[1]))
    center_text = siblings(center, graph[center]-set(roots), 2, 0)
    prefix = _prefix(network.names[center])
    parent = _multiplied(network.names[roots[0]], len(roots))
    if center_text is None or prefix is None or parent is None:
        return None
    descriptions = []
    for prime, root in enumerate(roots):
        descriptor = _fusion_descriptor(network, center, root, True, 0, prime)
        if descriptor is None:
            return None
        descriptions.append(descriptor)
    return root_text+center_text+prefix+'['+':'.join(descriptions)+']'+parent


def _render_rerooted(network, roots, *, sibling_orders=None):
    """Construct component trees or bare symmetric interparent paths/stars.

    Component-local bindings come from the original network; no molecular
    SMILES or complete-name answer is a selection key.
    """
    graph = _component_graph(network)
    if len(roots)>1:
        centers = set.intersection(*(graph[root] for root in roots))-set(roots)
        if len(centers) == 1 and len(graph) > len(roots)+1:
            return _branched_multiparent(network, roots, next(iter(centers)), graph, sibling_orders)
        if len(roots)==2:
            path = _tree_paths(graph,*roots)
            if len(path)!=len(graph) or (len(path)-2)%2!=1:
                return None
            layers = [(path[i],path[-i-1]) for i in range(1,(len(path)-1)//2)]
            layers.append((path[len(path)//2],))
            if any(network.components[a].identity!=network.components[b].identity for a,b in layers[:-1]):
                return None
        else:
            centers = [i for i in graph if graph[i]==set(roots)]
            if len(centers)!=1 or len(graph)!=len(roots)+1:
                return None
            layers = [(centers[0],)]
        result = _multiplied(network.names[roots[0]],len(roots))
        if result is None: return None
        lower,lower_primes,offset = tuple(roots),tuple(range(len(roots))),0
        for depth,layer in enumerate(layers,1):
            prefix = _prefix(network.names[layer[0]])
            if prefix is None: return None
            prefix = _multiplied(prefix,len(layer))
            descriptions=[]
            primes = tuple(range(offset,offset+len(layer)))
            for side,target in enumerate(lower):
                child = layer[side] if len(layer)>1 else layer[0]
                p = primes[side] if len(layer)>1 else primes[0]
                descriptor = _fusion_descriptor(network, child, target, depth == 1, p, lower_primes[side])
                if descriptor is None: return None
                descriptions.append(descriptor)
            result = prefix+'['+(':' if depth==1 else ';').join(descriptions)+']'+result
            lower,lower_primes = layer,primes
            offset += len(layer)
        return result

    root = roots[0]
    if network.names[root].startswith('cyclo'):
        return None  # An attached -a prefix is not a parent designation.
    def branch(node,previous,depth,prime,target_prime):
        children = graph[node]-{previous}
        prefix = _prefix(network.names[node])
        descriptor = _fusion_descriptor(network, node, previous, depth == 1, prime, target_prime,
                                        terminal=not children)
        if prefix is None or descriptor is None: return None
        child_text = siblings(node,children,depth+1,prime)
        return None if child_text is None else child_text+prefix+'['+descriptor+']'
    def siblings(target,children,depth,target_prime):
        units = defaultdict(list)
        for child in children:
            units[_branch_identity(network, child, target, graph=graph)].append(child)
        prefixes=[]
        for group in units.values():
            if len(group)==1:
                text = branch(group[0],target,depth,depth-1,target_prime)
            else:
                # OPSIN 2.9 does not parse P-25.3.6.3's composite bis(...)
                # syntax. Keep its selection evidence, but do not invent a
                # roundtrip-capable output spelling for it.
                if graph[group[0]]-{target}:
                    return None
                prefix = _prefix(network.names[group[0]])
                if prefix is None: return None
                prefix = _multiplied(prefix,len(group))
                if prefix is None: return None
                descriptions=[]
                # Equivalent components are distinct occurrences: choosing a
                # prime assignment from atom indices would make the result
                # depend on input order. The caller can compare assignments.
                order = next((tuple(node for node in candidate if node in group)
                              for indices, candidate in (sibling_orders or {}).items()
                              if frozenset(group) <= indices), tuple(sorted(group)))
                for ordinal,node in enumerate(order):
                    descriptor = _fusion_descriptor(network, node, target, depth == 1,
                                                    depth - 1 + ordinal, target_prime)
                    if descriptor is None: return None
                    descriptions.append(descriptor)
                text = prefix+'['+(':' if depth==1 else ';').join(descriptions)+']'
            if text is None: return None
            key = _branch_citation_key(network, group[0], target, graph=graph)
            if key[0] is None: return None
            prefixes.append((text,key))
        return ''.join(text for text,_ in sorted(prefixes,key=lambda item:item[1]))
    text = siblings(root,graph[root],1,0)
    # A primitive prefix spelling is not necessarily a legal parent spelling.
    from .fusion_components import resolve_component
    if text is None or resolve_component(network.names[root])!=network.components[root]:
        return None
    return text+network.names[root]


@lru_cache(maxsize=4096)
def _alternative_names(name):
    from .fusion_components import fusion_construction
    network = fusion_construction(name)
    if not isinstance(network,FusionNetwork) or not network.names:
        return ()
    graph = {i:set() for i in range(len(network.components))}
    for link in network.links:
        graph[link.child].add(link.target); graph[link.target].add(link.child)
    root_sets = [(i,) for i in graph]
    leaves = tuple(i for i in graph if len(graph[i])==1)
    if len(leaves)>=2 and len({network.components[i].identity for i in leaves})==1:
        root_sets.append(leaves)
    result=[]
    for roots in root_sets:
        text = _render_rerooted(network,roots)
        # The installed parser accepts expanded fusion syntax but cannot
        # bind parenthesized multiplied designations to fusion descriptors.
        # Their P-25 evidence is still modeled; do not emit unparseable aliases.
        if text is not None and '(' in text:
            continue
        candidate = fusion_construction(text) if text else None
        if (isinstance(candidate,FusionNetwork) and candidate.skeleton==network.skeleton
                and candidate.citation is True):
            result.append(text)
    return tuple(dict.fromkeys(result))


def alternative_fusion_records(records):
    """Generate re-rootings while preserving established P-25.3.3 numbering.

    Numbering the completed systematic fused system is independent of the
    chosen component decomposition. Reuse a complete declared map only for
    a neutral, unannotated systematic source network. Shared tetrahedral
    junctions additionally require a proved maximum noncumulative double-bond
    state; their hydrogen-bearing C/N positions are retained explicitly.
    Retained whole-parent exceptions, hydrogenation and composite output
    spellings unsupported by the independent parser delegate.
    """
    from rdkit import Chem
    from .fusion_components import fusion_construction,fusion_input_scope,_mancude
    result,seen = list(records),{record.get('name') for record in records}
    for record in records:
        network = fusion_construction(str(record.get('name','')))
        if not isinstance(network,FusionNetwork): continue
        mol = Chem.MolFromSmiles(str(record.get('smiles','')))
        labels = {str(i):value for i,value in (record.get('locants') or {}).items()}
        if (mol is None or not fusion_input_scope(mol)
                or not (all(a.GetIsAromatic() for a in iter_atoms(mol))
                        or network.shared_junctions and _mancude(mol))
                or set(labels)!=set(map(str,range(mol.GetNumAtoms())))
                or len(set(labels.values()))!=mol.GetNumAtoms()
                or any(locant_key(value) is None for value in labels.values())
                or not matches_skeleton(network,mol,range(mol.GetNumAtoms()))):
            continue
        hydrogen = sorted((labels[str(a.GetIdx())] for a in iter_atoms(mol)
                           if a.GetTotalNumHs() and (a.GetAtomicNum()==7 or
                               a.GetAtomicNum()==6 and not a.GetIsAromatic() and
                               all(b.GetBondTypeAsDouble()==1 for b in a.GetBonds()))),key=locant_key)
        indicated = ','.join(value+'H' for value in hydrogen)+'-' if hydrogen else ''
        for name in _alternative_names(str(record['name'])):
            name = indicated+name
            if name not in seen:
                seen.add(name); result.append(dict(record,name=name))
    return result
