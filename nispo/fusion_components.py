"""P-25.3.2.4 component seniority, separate from whole-parent seniority.

Components come from ordinary OPSIN structural/numbering metadata. No molecule
answer list or complete-name substitutions are used. Unsupported constructions
and the first missing component criterion remain explicit unknowns.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
import re

from .molecule_iteration import iter_atoms, iter_bonds
from .parent_selection import ParentComparison, locant_key, locant_set


_FIRST_ELEMENT = (7, 9, 17, 35, 53, 8, 16, 34, 52, 15, 33, 51, 83,
                  14, 32, 50, 82, 5, 13, 31, 49, 81)
_COUNT_ELEMENTS = (9, 17, 35, 53, 8, 16, 34, 52, 7, 15, 33, 51, 83,
                   14, 32, 50, 82, 5, 13, 31, 49, 81)
_FUSION = re.compile(r'^(.+)\[([0-9a-z,]+)-([a-z])\](.+)$')
_HYDROCARBON_FUSION = re.compile(r'^(.+)\[([a-z])\](.+)$')
_INDICATED = re.compile(r'^(?:\d+[a-z]?H,)*\d+[a-z]?H-')
# P-25.3.2.1.2: these component identities use Hantzsch-Widman locants.
_HW_NAMES = {'oxazole': '[1,3]oxazole', 'isoxazole': '[1,2]oxazole',
             'thiazole': '[1,3]thiazole', 'isothiazole': '[1,2]thiazole',
             's-triazole': '[1,2,4]triazole', 'sym-triazole': '[1,2,4]triazole'}


def fusion_input_scope(mol):
    """Keep unsupported input states out of early resource-alias pruning.

    The resource scaffold itself can be neutral even when the input is an
    N-oxide, isotope or assigned stereoisomer. Its scope cannot stand in for
    the input's scope before renderable candidates have been established.
    """
    return (all(not a.GetFormalCharge() and not a.GetIsotope()
                and not a.GetNumRadicalElectrons() and not int(a.GetChiralTag())
                for a in iter_atoms(mol))
            and all(not int(b.GetStereo()) for b in iter_bonds(mol)))


@dataclass(frozen=True, slots=True)
class FusionComponent:
    identity: str
    # Min-is-senior values for P-25.3.2.4(a-j); None is unproved.
    criteria: tuple
    elements: tuple
    atom_count: int
    ring_count: int
    label_elements: tuple
    bonds: frozenset


@dataclass(frozen=True, slots=True)
class FusionConstruction:
    parent: FusionComponent
    attached: FusionComponent
    numbers: tuple
    letter: str
    elements: tuple
    atom_count: int
    ring_count: int
    benzo_parent: bool = False


def component_criteria(left, right):
    if left.identity == right.identity:
        return
    for letter, a, b in zip('abcdefghij', left.criteria, right.criteria):
        yield f'P-25.3.2.4({letter})', a, b


def compare_components(left, right):
    if left is None or right is None:
        return ParentComparison(None, 'P-25-component-unavailable', 'Component evidence is missing.')
    for rule, a, b in component_criteria(left, right):
        if a is None or b is None:
            return ParentComparison(None, rule, 'An earlier component criterion is unproved.')
        if a != b:
            return ParentComparison(-1 if a < b else 1, rule, 'First decisive component criterion.')
    return ParentComparison(0, 'P-25-component-tie', 'Component seniority ties.')


def construction_criteria(left, right):
    from .fusion_networks import FusionNetwork, network_criteria
    if isinstance(left,FusionNetwork) or isinstance(right,FusionNetwork):
        yield from network_criteria(left,right)
        return
    yield from component_criteria(left.parent, right.parent)
    # With exactly one attached component, location counts/order/multiplicity
    # tie. Component selection precedes the fusion descriptor tie-breaks.
    for rule, a, b in component_criteria(left.attached, right.attached):
        yield 'P-25.3.4.2.2(a):' + rule, a, b


def compare_constructions(left, right):
    if left is None or right is None:
        return ParentComparison(None, 'P-25-construction-unavailable', 'Construction evidence is missing.')
    for rule, a, b in construction_criteria(left, right):
        if a is None or b is None:
            return ParentComparison(None, rule, 'An earlier construction criterion is unproved.')
        if a != b:
            return ParentComparison(-1 if a < b else 1, rule, 'First decisive construction criterion.')
    return ParentComparison(0, 'P-25-construction-tie', 'Component selections tie.')


def _universal_winners(records, preference):
    """Return records that beat or tie every rival under the existing criteria."""
    records = tuple(records)
    if not records:
        return ()
    # A likely senior record makes an efficient first check. Every survivor
    # still has to satisfy the comparison against all remaining rivals.
    first = records[0]
    for record in records[1:]:
        if preference(record, first) == -1:
            first = record
    return tuple(record for record in records
                 if preference(record, first) in (-1, 0)
                 and all(preference(record, other) in (-1, 0)
                         for other in records if other is not first))


def _mancude(mol):
    """Prove maximum noncumulative double count for ordinary neutral units.

    Most components meet the cardinality upper bound immediately. A bounded
    exact matching proof handles odd/fused exceptions; an expensive proof is
    left unknown instead of being converted into a ranking penalty.
    """
    from rdkit import Chem

    copy = Chem.Mol(mol)
    try:
        Chem.Kekulize(copy, clearAromaticFlags=True)
    except (ValueError, RuntimeError):
        return False
    eligible = []
    for atom in iter_atoms(copy):
        if atom.GetAtomicNum() == 6 and atom.GetDegree() <= 3:
            eligible.append(atom.GetIdx())
        elif atom.GetAtomicNum() == 7 and atom.GetDegree() == 2:
            # N-H in the input is not proof of an obligatory donor site.
            # A mancude state can remove that H when placing a double bond.
            eligible.append(atom.GetIdx())
    multiple = [b for b in iter_bonds(copy) if b.GetBondTypeAsDouble() > 1]
    endpoints = [i for b in multiple for i in (b.GetBeginAtomIdx(), b.GetEndAtomIdx())]
    if any(b.GetBondTypeAsDouble() != 2 for b in multiple) or len(set(endpoints)) != len(endpoints):
        return False
    if not set(endpoints) <= set(eligible):
        return False
    if len(multiple) == len(eligible) // 2:
        return True
    if len(eligible) > 22:
        return False
    positions = {idx: j for j, idx in enumerate(eligible)}
    neighbors = []
    for idx in eligible:
        neighbors.append(sum(1 << positions[a.GetIdx()] for a in copy.GetAtomWithIdx(idx).GetNeighbors()
                             if a.GetIdx() in positions))

    @lru_cache(maxsize=None)
    def matching(mask):
        if not mask:
            return 0
        bit = mask & -mask
        i = bit.bit_length() - 1
        remainder = mask ^ bit
        best = matching(remainder)
        candidates = neighbors[i] & remainder
        while candidates:
            other = candidates & -candidates
            best = max(best, 1 + matching(remainder ^ other))
            if best == mask.bit_count() // 2:
                break
            candidates ^= other
        return best
    return len(multiple) == matching((1 << len(eligible)) - 1)


def component_from_graph(mol, labels):
    """Extract independent a-j facts from a numbered mancude component."""
    from rdkit import Chem
    from .ring_parent_selection import _horizontal_row, ring_type_evidence

    labels = dict(labels)
    if (not mol.GetNumAtoms() or set(labels) != set(range(mol.GetNumAtoms()))
            or len(set(labels.values())) != mol.GetNumAtoms()
            or any(locant_key(value) is None for value in labels.values())
            or any(not a.IsInRing() or a.GetAtomicNum() not in {6, 7, 8, 16, 34, 52}
                   or a.GetFormalCharge() or a.GetIsotope() or a.GetNumRadicalElectrons()
                   or int(a.GetChiralTag()) for a in iter_atoms(mol))
            or not _mancude(mol)):
        return None
    rings = tuple(tuple(ring) for ring in Chem.GetSymmSSSR(mol))
    rank = mol.GetNumBonds() - mol.GetNumAtoms() + 1
    if len(Chem.GetMolFrags(mol)) != 1 or rank < 1:
        return None
    if rank > 1 and ring_type_evidence(mol, range(mol.GetNumAtoms()), labels, 'component', include_components=False)[0] != 2:
        return None
    counts = Counter(a.GetAtomicNum() for a in iter_atoms(mol))
    first = next((j for j, e in enumerate(_FIRST_ELEMENT) if counts[e]), len(_FIRST_ELEMENT))
    hetero = [a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() != 6]
    sizes = tuple(-size for size in sorted(map(len, rings), reverse=True)) if len(rings) == rank else None
    row = 1 if rank == 1 else _horizontal_row(rings) if len(rings) == rank else None
    if row is None and 3 <= rank <= 4 and len(rings) == rank and any(len(r) != 6 for r in rings):
        row = _mixed_component_horizontal_row(Chem.MolToSmiles(mol, isomericSmiles=True))
    junctions = [a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() == 6 and a.GetDegree() > 2]
    peripheral = all(not all(n.GetDegree() > 2 for n in mol.GetAtomWithIdx(i).GetNeighbors()) for i in junctions)
    key = (first, -rank, sizes, -len(hetero), -len({mol.GetAtomWithIdx(i).GetAtomicNum() for i in hetero}),
           tuple(-counts[e] for e in _COUNT_ELEMENTS), -row if row is not None else None,
           locant_set(labels[i] for i in hetero),
           tuple(locant_set(labels[i] for i in hetero if mol.GetAtomWithIdx(i).GetAtomicNum() == e)
                 for e in _COUNT_ELEMENTS),
           locant_set(labels[i] for i in junctions) if peripheral else None)
    symbols = [f'[{a.GetSymbol()}:{labels[a.GetIdx()]}]' for a in iter_atoms(mol)]
    identity = Chem.MolFragmentToSmiles(mol, atomsToUse=list(labels), atomSymbols=symbols, canonical=True)
    return FusionComponent(identity, key, tuple(sorted(counts.items())), mol.GetNumAtoms(), rank,
        tuple(sorted((labels[a.GetIdx()], a.GetAtomicNum()) for a in iter_atoms(mol))),
        frozenset(frozenset((labels[b.GetBeginAtomIdx()], labels[b.GetEndAtomIdx()])) for b in iter_bonds(mol)))


@lru_cache(maxsize=2048)
def _mixed_component_horizontal_row(smiles):
    """Use P-25's source-oriented drawing when a mixed face tree is proved."""
    from rdkit import Chem
    from .fused_orientation import fused_orientation_frontier

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    frontier = fused_orientation_frontier(mol)
    rows = {candidate.horizontal_row_count for candidate in frontier.candidates}
    return next(iter(rows)) if frontier.admitted and len(rows) == 1 else None


@lru_cache(maxsize=1)
def _catalog():
    from rdkit import Chem
    from .opsin_resource_utils import iter_opsin_token_records
    from .resource_templates import (_template_from_token, _fusion_component_parent_name,
                                     PARENT_NAME_NORMALIZATIONS)

    result = defaultdict(dict)
    for record in iter_opsin_token_records(('fusionComponents.xml', 'arylGroups.xml', 'simpleCyclicGroups.xml')):
        token = record.token
        if set(token.attrib) - {'value', 'labels', 'fusedRingNumbering'}:
            continue  # Dynamic/front-locant and functional operations are not guessed.
        template = _template_from_token(Chem, record, min_rings=1, max_atoms=48,
                                        allow_dot_ring_closures=True)
        if template is None:
            continue
        facts = component_from_graph(template['mol'], template['locants'])
        if facts is None:
            continue
        for alias in (token.text or '').split('|'):
            if record.file_name == 'fusionComponents.xml':
                aliases = (alias, _fusion_component_parent_name(alias))
            else:
                normalized = PARENT_NAME_NORMALIZATIONS.get(alias, alias)
                aliases = (normalized, normalized + 'e' if normalized.endswith(('in','ol','en')) else normalized)
            for name in aliases:
                result[name][facts.identity] = facts
                if name in _HW_NAMES:
                    result[_HW_NAMES[name]][facts.identity] = facts
    return {name: tuple(values.values()) for name, values in result.items()}


@lru_cache(maxsize=2048)
def resolve_component(name):
    name = _INDICATED.sub('', name)
    if name.endswith(('aleno', 'apheno', 'aceno')):
        name = name[:-1] + 'e'
    # P-25.3.5 treats the numbered benzoheterocycle as one component.
    # Resolve the exact preferred fusion prefix to the independently numbered
    # 1-benzofuran graph; it is not a text-level whole-name substitution.
    if name == '[1]benzofuro':
        return resolve_component('[1]benzofuran')
    if name == '[1]benzothiopheno':
        return resolve_component('[1]benzothiophene')
    if name == '[1]benzothiophene':
        return _benzothiophene_component()
    candidates = _catalog().get(name, ())
    # An ambiguous resource spelling is not a declared component identity.
    if candidates:
        return candidates[0] if len(candidates) == 1 else None
    benzo = re.fullmatch(r'\[([0-9,]+)\](benzo.+)',name)
    if benzo is not None:
        component = resolve_component(benzo[2])
        if component is not None and component.ring_count==2:
            positions = tuple(label for element in _COUNT_ELEMENTS
                              for label,e in component.label_elements if e==element)
            if positions==tuple(benzo[1].split(',')):
                return component
    component = (_hw_component(name) or _hydrocarbon_component(name)
                 or _polyalene_component(name) or _systematic_fused_component(name))
    if component is not None:
        return component
    from .seven_member_fusion_components import seven_member_chalcogen_component

    component = seven_member_chalcogen_component(name)
    if component is not None:
        return component
    from .benzo_seven_member_fusion_components import benzo_seven_member_component

    return benzo_seven_member_component(name)


@lru_cache(maxsize=1)
def _benzothiophene_component():
    """P-25.2.2.4 sulfur analogue of the numbered 1-benzofuran unit."""
    from rdkit import Chem
    from .ring_state_parent import _reference_states

    reference = resolve_component('[1]benzofuran')
    if reference is None:
        return None
    elements = dict(reference.label_elements)
    labels = tuple(sorted(elements, key=locant_key))
    if sum(value == 8 for value in elements.values()) != 1:
        return None
    indices = {label: i for i, label in enumerate(labels)}
    numbers = tuple(16 if elements[label] == 8 else elements[label] for label in labels)
    edges = tuple(sorted(tuple(sorted(indices[label] for label in edge)) for edge in reference.bonds))
    candidates = {}
    for hydrogens, _indicated, bonds in _reference_states(numbers, edges):
        graph = Chem.RWMol()
        for number, count in zip(numbers, hydrogens, strict=True):
            atom = Chem.Atom(number)
            atom.SetNumExplicitHs(count)
            atom.SetNoImplicit(True)
            graph.AddAtom(atom)
        for left, right, aromatic, order in bonds:
            graph.AddBond(left, right, Chem.BondType.AROMATIC if aromatic else
                          Chem.BondType.DOUBLE if order == 2 else Chem.BondType.SINGLE)
        molecule = graph.GetMol()
        try:
            Chem.SanitizeMol(molecule)
        except (ValueError, RuntimeError):
            continue
        component = component_from_graph(molecule, dict(enumerate(labels)))
        if component is not None:
            candidates[component.identity] = component
    return next(iter(candidates.values())) if len(candidates) == 1 else None


def _hydrocarbon_component(name):
    """P-25.3.2.1.1 and .2.1 regular monocyclic hydrocarbon units."""
    from rdkit import Chem
    from .constants import ROOTS

    match = re.fullmatch(r'\[(\d+)\]annulene', name)
    if match:
        size = int(match[1])
        if size < 7:
            return None
    else:
        size = next((n for n, root in ROOTS.items() if name == 'cyclo'+root+'a'), 0)
    if not 3 <= size <= 24:
        return None
    graph = Chem.RWMol()
    for _ in range(size):
        graph.AddAtom(Chem.Atom(6))
    for i in range(size):
        order = Chem.BondType.DOUBLE if i % 2 == 0 and i < size-1 else Chem.BondType.SINGLE
        graph.AddBond(i,(i+1)%size,order)
    mol = graph.GetMol(); Chem.SanitizeMol(mol)
    return component_from_graph(mol,{i:str(i+1) for i in range(size)})


def _polyalene_component(name):
    """Build the numbered identical-ring hydrocarbon units of P-25.1.2.3."""
    from rdkit import Chem
    from .constants import MULT

    sizes = {((prefix[:-1] if prefix.endswith('a') else prefix) + 'alene'): size
             for size, prefix in MULT.items() if 5 <= size <= 8 and size != 6}
    size = sizes.get(name)
    if size is None:
        return None
    count = 2*(size-1)
    graph = Chem.RWMol()
    for _ in range(count):
        graph.AddAtom(Chem.Atom(6))
    for i in range(count):
        graph.AddBond(i, (i+1) % count,
                      Chem.BondType.DOUBLE if i % 2 == 0 else Chem.BondType.SINGLE)
    graph.AddBond(size-2, count-1, Chem.BondType.SINGLE)
    mol = graph.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except (ValueError, RuntimeError):
        return None
    labels = {i: (str(i+1) if i < size-2 else
                  str(size-2)+'a' if i == size-2 else
                  str(i) if i < count-1 else str(2*(size-2))+'a')
              for i in range(count)}
    return component_from_graph(mol, labels)


@lru_cache(maxsize=1)
def _systematic_fused_records():
    """Source-numbered polyacene/polyaphene components of P-25.1.2."""
    from .constants import MULT
    from .resource_templates import (OPSIN_PARENT_TEMPLATES, _PHENE_RETAINED_RING_RECORDS,
                                     _read_json_records)

    names = {prefix[:-1] + 'acene' for count, prefix in MULT.items()
             if count >= 4 and prefix.endswith('a')}
    names |= {prefix + 'phene' for count, prefix in MULT.items() if count >= 4}
    records = {}
    for name, smiles, labels in OPSIN_PARENT_TEMPLATES + _PHENE_RETAINED_RING_RECORDS:
        if name in names:
            records[name] = (smiles, tuple(labels))
        if name == 'acenaphthene':
            # P-25.1.2.7: the two-atom path closes the peri bay to a
            # five-membered ring. Its mancude form has a 1--2 double bond;
            # preserve the independently sourced parent atom numbering.
            from rdkit import Chem
            query = Chem.MolFromSmiles(smiles)
            by_label = {str(label): i for i, label in enumerate(labels)}
            if query is not None and {'1', '2'} <= set(by_label):
                a, b = by_label['1'], by_label['2']
                bond = query.GetBondBetweenAtoms(a, b)
                if (bond is not None and bond.GetBondTypeAsDouble() == 1
                        and query.GetAtomWithIdx(a).GetDegree() == 2
                        and query.GetAtomWithIdx(b).GetDegree() == 2
                        and any(len(ring) == 5 and {a, b} <= set(ring)
                                for ring in query.GetRingInfo().AtomRings())):
                    rw = Chem.RWMol(query)
                    rw.GetBondBetweenAtoms(a, b).SetBondType(Chem.BondType.DOUBLE)
                    for idx in (a, b):
                        rw.GetAtomWithIdx(idx).SetNumExplicitHs(0)
                        rw.GetAtomWithIdx(idx).SetNoImplicit(False)
                    mancude = rw.GetMol()
                    try:
                        Chem.SanitizeMol(mancude)
                    except (ValueError, RuntimeError):
                        continue
                    import json
                    written = Chem.MolToSmiles(mancude, canonical=False)
                    order = json.loads(mancude.GetProp('_smilesAtomOutputOrder'))
                    records['acenaphthylene'] = (written, tuple(labels[i] for i in order))
    for row in _read_json_records('opsin_polyacene_helicene_templates.json'):
        if row['name'] in names:
            labels = row['locants']
            records[row['name']] = (row['smiles'], tuple(labels[str(i)] for i in range(len(labels))))
    # The native tetraphene component was independently parsed from OPSIN;
    # it is also used by the existing P-25 horizontal-row comparison.
    records['tetraphene'] = (
        'C1=CC=CC2=CC=C3C=C4C=CC=CC4=CC3=C12',
        ('1', '2', '3', '4', '4a', '5', '6', '6a', '7', '7a',
         '8', '9', '10', '11', '11a', '12', '12a', '12b'))
    return records


@lru_cache(maxsize=64)
def _systematic_fused_component(name):
    record = _systematic_fused_records().get(name)
    if record is None:
        return None
    from rdkit import Chem
    mol = Chem.MolFromSmiles(record[0])
    if mol is None or mol.GetNumAtoms() != len(record[1]):
        return None
    return component_from_graph(mol, dict(enumerate(record[1])))


def _hw_component(name):
    """Numbered azole, azine and azepine components.

    These ordinary Hantzsch-Widman units are algorithmic OPSIN vocabulary,
    rather than plain tokens. More complicated HW electronic states delegate.
    """
    from rdkit import Chem
    from .resource_templates import _fusion_component_parent_name

    if name.endswith('o'):
        name = _fusion_component_parent_name(name)
    name = _HW_NAMES.get(name, name)
    if name == 'azepine':
        name = '[1]azepine'
    # Six-membered mixed-chalcogen HW components also have independently
    # numbered mancude forms. They contain no nitrogen and therefore do not
    # pass through the azole/azine grammar below.
    mixed = re.fullmatch(r'\[(\d+(?:,\d+)*)\](.+)ine', name)
    if mixed is not None and 'az' not in mixed.group(2):
        from itertools import product
        from .ring_state_parent import _hw_name, _reference_states
        locations = tuple(int(i) for i in mixed.group(1).split(','))
        if (len(locations) < 2 or len(locations) > 4 or min(locations) != 1
                or max(locations) > 6 or len(set(locations)) != len(locations)):
            return None
        for elements in product((8,16,34,52),repeat=len(locations)):
            numbers = [6]*6
            for loc, element in zip(locations,elements):
                numbers[loc-1] = element
            if re.sub(r'^(\d+(?:,\d+)*)-',r'[\1]',_hw_name(tuple(numbers),False)) != name:
                continue
            edges=tuple((i,(i+1)%6) for i in range(6))
            for hydrogens,_indicated,bonds in _reference_states(tuple(numbers),edges):
                graph=Chem.RWMol()
                for element,h in zip(numbers,hydrogens):
                    atom=Chem.Atom(element);atom.SetNumExplicitHs(h);atom.SetNoImplicit(True)
                    graph.AddAtom(atom)
                for a,b,aromatic,o in bonds:
                    graph.AddBond(a,b,Chem.BondType.AROMATIC if aromatic else Chem.BondType.DOUBLE if o==2 else Chem.BondType.SINGLE)
                mol=graph.GetMol()
                try:Chem.SanitizeMol(mol)
                except (ValueError,RuntimeError):continue
                component=component_from_graph(mol,{i:str(i+1) for i in range(6)})
                if component is not None:return component
        return None
    match = re.fullmatch(r'\[(\d+(?:,\d+)*)\](.*)(azole|azine|azepine)', name)
    if match is None:
        return None
    positions, prefix, ending = match.groups()
    positions = tuple(int(i) for i in positions.split(','))
    count = 1
    for multiplier, number in (('tetra',4),('tri',3),('di',2)):
        if prefix.endswith(multiplier):
            prefix, count = prefix[:-len(multiplier)], number
            break
    chalcogen = {'ox': 'o', 'oxa':'o', 'thi':'s', 'thia':'s',
                 'selen':'[se]', 'selena':'[se]', 'tellur':'[te]', 'tellura':'[te]'}.get(prefix)
    if prefix and chalcogen is None:
        return None
    if ending in {'azine', 'azepine'} and chalcogen not in {None,'o','s'}:
        return None  # This extension supplies ordinary O/S fusion units only.
    size = {'azole': 5, 'azine': 6, 'azepine': 7}[ending]
    if (len(positions) != count + bool(chalcogen) or len(set(positions)) != len(positions)
            or positions != tuple(sorted(positions)) or min(positions) != 1 or max(positions) > size):
        return None
    if size == 7 or chalcogen and size == 6:
        # P-22/P-25 ordinary six-member O/S azine components are often
        # nonaromatic despite being mancude. Build their maximum matching
        # explicitly instead of forcing lower-case aromatic atom symbols.
        from .ring_state_parent import _reference_states
        numbers = [6] * size
        for pos in positions:
            numbers[pos-1] = 7
        if chalcogen:
            numbers[positions[0]-1] = {'o':8,'s':16,'[se]':34,'[te]':52}[chalcogen]
        edges = tuple((i,(i+1)%size) for i in range(size))
        for hydrogens, _indicated, bonds in _reference_states(tuple(numbers),edges):
            graph = Chem.RWMol()
            for number, count in zip(numbers,hydrogens):
                atom=Chem.Atom(number);atom.SetNumExplicitHs(count);atom.SetNoImplicit(True)
                graph.AddAtom(atom)
            for a,b,aromatic,order in bonds:
                graph.AddBond(a,b,Chem.BondType.AROMATIC if aromatic else
                              Chem.BondType.DOUBLE if order==2 else Chem.BondType.SINGLE)
            mol=graph.GetMol()
            try:Chem.SanitizeMol(mol)
            except (ValueError,RuntimeError):continue
            component=component_from_graph(mol,{i:str(i+1) for i in range(size)})
            if component is not None:return component
        return None
    symbols = ['c'] * size
    for pos in positions:
        symbols[pos-1] = 'n'
    if chalcogen:
        symbols[positions[0]-1] = chalcogen
    elif size == 5:
        symbols[positions[0]-1] = '[nH]'
    smiles = symbols[0] + '1' + ''.join(symbols[1:]) + '1'
    mol = Chem.MolFromSmiles(smiles)
    return component_from_graph(mol,{i:str(i+1) for i in range(size)}) if mol is not None else None


@lru_cache(maxsize=8192)
def fusion_construction(name):
    name = _INDICATED.sub('', name)
    simple = _two_component_construction(name)
    if simple is not None:
        return simple
    from .fusion_networks import parse_network
    return parse_network(name,resolve_component)


def _two_component_construction(name):
    match = _FUSION.fullmatch(name)
    implicit = False
    if match is not None:
        prefix, positions, letter, base = match.groups()
    else:
        match = _HYDROCARBON_FUSION.fullmatch(name)
        if match is None:
            return None
        prefix, letter, base = match.groups()
        positions, implicit = '1,2', True
    attached, parent = resolve_component(prefix), resolve_component(base)
    numbers = tuple(positions.split(','))
    if (attached is None or parent is None or len(numbers) != 2
            or any(locant_key(value) is None for value in numbers)
            or implicit and (attached.ring_count != 1 or attached.elements != ((6,attached.atom_count),))
            or frozenset(numbers) not in attached.bonds
            or ord(letter) - ord('a') >= parent.atom_count):
        return None
    elements = Counter(dict(parent.elements)) + Counter(dict(attached.elements))
    attached_elements = dict(attached.label_elements)
    for number in numbers:
        elements[attached_elements[number]] -= 1
    return FusionConstruction(parent, attached, tuple(locant_key(n) for n in numbers), letter,
        tuple(sorted((element,count) for element,count in elements.items() if count)),
        parent.atom_count + attached.atom_count - 2, parent.ring_count + attached.ring_count,
        'benzo' in base)


def component_list_key(construction):
    if construction is None:
        return None
    from .fusion_networks import construction_components
    keys = [c.criteria for c in construction_components(construction)]
    return tuple(sorted(keys)) if all(None not in key for key in keys) else None


def component_list_criteria(left, right):
    """P-44.2.2.2.3.5, comparing most senior components first.

    A later unknown need not erase an earlier proved component distinction.
    Sorting itself must be proved; unknown comparisons are never sort keys.
    """
    from .fusion_networks import construction_components, ordered_components
    if sorted(c.identity for c in construction_components(left)) == sorted(
            c.identity for c in construction_components(right)):
        return  # The same component multiset proves a tie without sorting it.
    lists = []
    for construction in (left, right):
        components = ordered_components(construction_components(construction))
        if components is None:
            yield 'P-44.2.2.2.3.5', None, None
            return
        lists.append(components)
    for a,b in zip(*lists):
        for rule,left_value,right_value in component_criteria(a,b):
            yield 'P-44.2.2.2.3.5:' + rule, left_value, right_value
    if len(lists[0])!=len(lists[1]):
        yield 'P-44.2.2.2.3.5',None,None


def fusion_descriptor_evidence(construction):
    from .fusion_networks import FusionNetwork
    if isinstance(construction,FusionNetwork):
        # P-25 comparison owns the layered descriptor rules. A flat P-44
        # letter/number tuple cannot express primed component ownership.
        return None,None
    return (construction.letter,), (tuple(value[1] for value in construction.numbers)
        if all(value[2]=='' for value in construction.numbers) else None)


def fusion_record_frontier(records):
    """Reuse immutable resource-alias proofs while returning caller records."""
    if type(records) not in (list, tuple):
        return _fusion_record_frontier_uncached(records)
    snapshots = []
    for record in records:
        if (type(record) is not dict
                or type(record.get('name')) is not str
                or type(record.get('smiles')) is not str
                or type(record.get('locants')) is not dict
                or any(type(key) is not str or type(value) is not str
                       for key, value in record['locants'].items())):
            return _fusion_record_frontier_uncached(records)
        snapshots.append((record['name'], record['smiles'],
                          tuple(record['locants'].items())))
    return [records[i] for i in _fusion_record_frontier_indices(tuple(snapshots))]


@lru_cache(maxsize=4096)
def _fusion_record_frontier_indices(snapshots):
    # Snapshot only the fields consumed by the proof. Extra record metadata and
    # caller object identity never enter the cache or its returned values.
    records = [dict(name=name, smiles=smiles, locants=dict(locants))
               for name, smiles, locants in snapshots]
    selected = {id(record) for record in _fusion_record_frontier_uncached(records)}
    return tuple(i for i, record in enumerate(records) if id(record) in selected)


def _fusion_record_frontier_uncached(records):
    """Proved component selection before alias-length/priority budgets.

    The caller groups identical source SMILES, whose fusion names were parsed
    by OPSIN when generating the resource. No unrelated scaffold is compared.
    """
    from rdkit import Chem
    from .fusion_networks import matches_skeleton

    # Resource aliases are trusted parser output, but a missing/mismatched
    # record must not acquire component evidence from its spelling alone.
    signatures = {}
    molecules = {}
    constructions = []
    for record in records:
        smiles = str(record.get('smiles', ''))
        if smiles not in signatures:
            mol = Chem.MolFromSmiles(smiles) if smiles else None
            molecules[smiles] = mol
            ordinary = (mol is not None and len(Chem.GetMolFrags(mol)) == 1
                        and all(a.IsInRing() and not a.GetFormalCharge() and not a.GetIsotope()
                                and not a.GetNumRadicalElectrons() and not int(a.GetChiralTag())
                                for a in iter_atoms(mol)))
            signatures[smiles] = None if not ordinary else (
                mol.GetNumAtoms(), mol.GetNumBonds() - mol.GetNumAtoms() + 1,
                tuple(sorted(Counter(a.GetAtomicNum() for a in iter_atoms(mol)).items())))
        construction = fusion_construction(str(record.get('name', '')))
        if construction is not None and signatures[smiles] != (
                construction.atom_count, construction.ring_count, construction.elements):
            construction = None
        if construction is not None and not matches_skeleton(
                construction,molecules[smiles],range(molecules[smiles].GetNumAtoms())):
            construction = None
        labels = record.get('locants') or {}
        if construction is not None and (
                {str(i) for i in range(construction.atom_count)} != {str(i) for i in labels}
                or len(set(labels.values())) != construction.atom_count
                or any(locant_key(value) is None for value in labels.values())):
            # An alias without usable whole-system numbering cannot replace
            # a renderable candidate merely because its components are senior.
            construction = None
        constructions.append(construction)
    edges = [set() for _ in records]
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            if records[i].get('smiles') != records[j].get('smiles'):
                continue
            comparison = compare_constructions(constructions[i], constructions[j]).preference
            if comparison == -1:
                edges[i].add(j)
            elif comparison == 1:
                edges[j].add(i)
    visiting,visited = set(),set()
    def cyclic(i):
        if i in visiting:
            return True
        if i in visited:
            return False
        visiting.add(i)
        if any(cyclic(j) for j in edges[i]):
            return True
        visiting.remove(i); visited.add(i)
        return False
    if any(cyclic(i) for i in range(len(records))):
        return list(records)
    dominated = set().union(*edges) if edges else set()
    return [record for i, record in enumerate(records) if i not in dominated]
