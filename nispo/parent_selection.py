"""Partial, evidence-based Blue Book P-44/P-45 parent comparison.

``None`` means unknown, never zero. Comparison stops at the first unknown
applicable criterion. A later fact, rule priority, length or model score cannot
turn that missing evidence into a chemical preference. Producers retain
responsibility for valid rendering, method eligibility and suffix ownership.
"""

from collections import Counter
from dataclasses import dataclass, replace
from functools import lru_cache
import re

from .molecule_iteration import iter_atoms, iter_bonds
from .molecule_cache import mol_cache_dict, mol_cache_get, mol_cache_set
from .graph_model import graph_view, mapped_graph_components
from . import formatting as _formatting
from . import neutral_charge_components as _neutral_charge
from . import nonstandard_parent_evidence as _nonstandard_evidence
from . import parent_annotations as _annotations


_RING_ELEMENTS = (9, 17, 35, 53, 8, 16, 34, 52, 7, 15, 33, 51, 83,
                  14, 32, 50, 82, 5, 13, 31, 49, 81)
_CHAIN_ELEMENTS = (8, 16, 34, 52, 7, 15, 33, 51, 83, 14, 32, 50, 82, 5, 13, 31, 49, 81)
_CLASS_ELEMENTS = (7, 15, 33, 51, 83, 14, 32, 50, 82, 5, 13, 31, 49, 81, 8, 16, 34, 52, 6)
# P-44.2.2.2, before the detailed rules within each ring-system type.
RING_TYPE_ORDER = {name: i for i, name in enumerate((
    "spiro", "cyclic-phane", "fused", "bridged-fused", "nonfused-bridged",
    "linear-phane", "ring-assembly",
))}


_RING_SENIORITY_RULES = tuple("P-44.2.1(" + letter + ")" for letter in "abcdefg")
_P44_NUMBERING_RULES = (
    "P-44.4.1.4", "P-44.4.1.5", "P-44.4.1.6", "P-44.4.1.7",
    "P-44.4.1.8", "P-44.4.1.9", "P-44.4.1.10(all)", "P-44.4.1.10(ene)",
)
_P44_NUMBERING_FIELDS = (
    "indicated_h_locants", "replacement_locants", "replacement_element_locants",
    "fusion_locants", "suffix_locants", "attachment_locants", "unsaturation_locants",
    "double_locants",
)


class _DeferredComparisonFacts:
    # This storage is deliberately not a dataclass field. Serialization,
    # equality, hashing and dataclasses.replace still see exactly the original
    # normalized fields. A missing slot is filled on its first observation.
    __slots__ = ("_pending_facts",)

    def __getattr__(self, name):
        try:
            pending = object.__getattribute__(self, "_pending_facts")
            operation, arguments, fields = pending[name]
        except KeyError:
            # Another reader may just have completed this pure reduction.
            return object.__getattribute__(self, name)
        except AttributeError:
            raise AttributeError(name) from None
        values = operation(*arguments)
        for field_name, value in zip(fields, values):
            object.__setattr__(self, field_name, value)
            pending.pop(field_name, None)
        return object.__getattribute__(self, name)


@dataclass(frozen=True, slots=True)
class ParentFacts(_DeferredComparisonFacts):
    """Normalized evidence; all *_key values use smaller-is-senior order.

    Empty tuples prove an absent/inapplicable feature. None leaves a barrier.
    Ring-system detail keys require a type-specific producer; raw graph size
    or an OPSIN template match does not prove that those rules tie.
    """

    principal_class: tuple[int, ...] | None = None
    suffix_count: int | None = None
    kind: str = "unknown"
    elements: tuple[tuple[int, int], ...] | None = None
    ring_count: int | None = None
    ring_type: int | None = None
    ring_type_details: tuple | None = None
    ring_identity: tuple | None = None
    fusion_construction: object = None
    fusion_scope: frozenset[int] | None = None
    multiple_count: int | None = None
    double_count: int | None = None
    nonstandard_key: tuple | None = None
    indicated_h_locants: tuple | None = None
    replacement_locants: tuple | None = None
    replacement_element_locants: tuple | None = None
    fusion_locants: tuple | None = None
    suffix_locants: tuple | None = None
    attachment_locants: tuple | None = None
    unsaturation_locants: tuple | None = None
    double_locants: tuple | None = None
    isotope_key: tuple | None = None
    # P-82.5.2 numbering only: complete modified-position multiset across
    # parent atoms and isotopically modified substituent groups.
    equivalent_isotope_key: tuple | None = None
    stereo_key: tuple | None = None
    multiplied_parents: int | None = None
    prefix_count: int | None = None
    prefix_locants: tuple | None = None
    prefix_citation_locants: tuple | None = None
    substituent_nonstandard_key: tuple | None = None
    substituent_isotope_key: tuple | None = None
    alphanumerical_key: tuple | None = None
    # Only explicitly mapped cross-route producers supply this identity. It
    # distinguishes selection of a parent from equivalent numbering (P-82.5).
    numbering_scope: tuple | None = None
    stereo_identity: tuple | None = None
    stereo_numbering_key: tuple | None = None


# An audit can disable scheduling without changing comparison semantics.
_DEFER_COMPARISON_FACTS = True


def _deferred_parent_facts(groups, *, _defer=True, **values):
    """Defer pure reductions of immutable observations, never native reads.

    Groups hold only primitive tuples/strings. In particular no callback
    retains a molecule, mutable mapping, or unperformed validation. Normal
    attribute access always returns the original tuple/None contract.
    """
    if not _DEFER_COMPARISON_FACTS or not _defer:
        for operation, arguments, fields in groups:
            values.update(zip(fields, operation(*arguments)))
        return ParentFacts(**values)
    facts = ParentFacts(**values)
    pending = {}
    for group in groups:
        for name in group[2]:
            object.__delattr__(facts, name)
            pending[name] = group
    object.__setattr__(facts, '_pending_facts', pending)
    return facts


def replace_parent_facts(facts, **changes):
    """Copy normalized facts without forcing unchanged deferred reductions."""
    try:
        pending = object.__getattribute__(facts, '_pending_facts').copy()
    except AttributeError:
        return replace(facts, **changes)
    if not pending:
        return replace(facts, **changes)
    values = {name: getattr(facts, name) for name in ParentFacts.__dataclass_fields__
              if name not in pending and name not in changes}
    groups = []
    seen = set()
    for name, group in pending.items():
        if name in changes or id(group) in seen:
            continue
        seen.add(id(group))
        operation, arguments, fields = group
        remaining = tuple(field for field in fields if field not in changes)
        if remaining != fields:
            groups.append((_project_comparison_values,
                           (operation, arguments, tuple(fields.index(field) for field in remaining)),
                           remaining))
        else:
            groups.append(group)
    values.update(changes)
    return _deferred_parent_facts(groups, **values)


def _project_comparison_values(operation, arguments, indices):
    values = operation(*arguments)
    return tuple(values[index] for index in indices)


@dataclass(frozen=True, slots=True)
class ParentComparison:
    # -1: left senior, +1: right senior, 0: proved tie, None: unknown.
    preference: int | None
    rule: str
    reason: str
    criteria: tuple = ()


def _element_key(facts):
    return _element_counts(facts.elements) if facts.elements is not None else None


@lru_cache(maxsize=4096)
def _element_counts(elements):
    return Counter(dict(elements))


@lru_cache(maxsize=4096)
def ring_seniority_key(elements, ring_count):
    """P-44.2.1; presence of N precedes size, its count follows size."""
    counts = Counter(dict(elements))
    hetero = sum(n for element, n in counts.items() if element != 6)
    non_n_order = tuple(e for e in _RING_ELEMENTS if e != 7)
    senior = next((j for j, e in enumerate(non_n_order) if counts[e]), len(non_n_order))
    return (-bool(hetero), -bool(counts[7]), senior if not counts[7] else 0,
            -ring_count, -sum(counts.values()), -hetero,
            tuple(-counts[e] for e in _RING_ELEMENTS))


@lru_cache(maxsize=4096)
def _chain_seniority_key(elements):
    """P-44.3 facts shared by every comparison of the same element counts."""
    counts = _element_counts(elements)
    return (-sum(n for element, n in counts.items() if element != 6),
            -sum(counts.values()), tuple(-counts[element] for element in _CHAIN_ELEMENTS))


@lru_cache(maxsize=4096)
def _class_element_seniority(elements):
    counts = _element_counts(elements)
    return next((index for index, element in enumerate(_CLASS_ELEMENTS) if counts[element]), None)


def _criteria(left, right, *, identity_ties=True, p44_only=False):
    same_numbered_parent = (left.numbering_scope is not None
                           and left.numbering_scope == right.numbering_scope)
    yield "P-41/P-43", left.principal_class, right.principal_class
    yield "P-44.1.1", _negative(left.suffix_count), _negative(right.suffix_count)
    a, b = _element_key(left), _element_key(right)
    if a is None or b is None or left.kind not in {"chain", "ring"} or right.kind not in {"chain", "ring"}:
        yield "P-44.1", None, None
        return
    if left.kind != right.kind:
        yield "P-44.1.2", _class_element_seniority(left.elements), _class_element_seniority(right.elements)
        yield "P-44.1.2.2", -int(left.kind == "ring"), -int(right.kind == "ring")
    elif left.kind == "ring":
        if left.ring_count is None or right.ring_count is None:
            yield "P-44.2.1", None, None
            return
        yield from zip(
            _RING_SENIORITY_RULES,
            ring_seniority_key(left.elements, left.ring_count),
            ring_seniority_key(right.elements, right.ring_count),
        )
        if left.ring_count > 1 or right.ring_count > 1:
            yield "P-44.2.2.2", left.ring_type, right.ring_type
            if (left.ring_type == right.ring_type == RING_TYPE_ORDER['fused']
                    and left.fusion_construction is not None and right.fusion_construction is not None
                    and left.fusion_scope is not None and left.fusion_scope == right.fusion_scope):
                from .fusion_components import construction_criteria
                yield from construction_criteria(left.fusion_construction, right.fusion_construction)
            # Identical numbered-parent nomenclature and skeletal graphs prove
            # that intrinsic type criteria tie, including implicit descriptors.
            same_parent = identity_ties and left.ring_identity is not None and left.ring_identity == right.ring_identity
            a_details, b_details = left.ring_type_details, right.ring_type_details
            if a_details is None or b_details is None or len(a_details) != len(b_details):
                if not same_parent:
                    yield "P-44.2.2.2-details", None, None
            else:
                spiro_rules = ("P-44.2.2.2.1.1", "P-44.2.2.2.1.2",
                               "P-44.2.2.2.1.2(locants)", "P-44.2.2.2.1.3")
                section = {2: "3", 4: "5"}.get(left.ring_type)
                for index, (a_detail, b_detail) in enumerate(zip(a_details, b_details), 1):
                    if (not same_parent and left.ring_type == right.ring_type == RING_TYPE_ORDER['fused'] and index == 5
                            and left.fusion_construction is not None and right.fusion_construction is not None):
                        from .fusion_components import component_list_criteria
                        yield from component_list_criteria(left.fusion_construction, right.fusion_construction)
                        continue
                    rule = (spiro_rules[index - 1] if left.ring_type == 0 and index <= 4 else
                            f"P-44.2.2.2.{section}.{index}" if section else "P-44.2.2.2-details")
                    if same_parent and (a_detail is None or b_detail is None):
                        a_detail = b_detail = ()
                    yield rule, a_detail, b_detail
    else:
        a_chain = _chain_seniority_key(left.elements)
        b_chain = _chain_seniority_key(right.elements)
        yield "P-44.3.1", a_chain[0], b_chain[0]
        yield "P-44.3.2", a_chain[1], b_chain[1]
        yield "P-44.3.3", a_chain[2], b_chain[2]
    yield "P-44.4.1.1", _negative(left.multiple_count), _negative(right.multiple_count)
    yield "P-44.4.1.2", _negative(left.double_count), _negative(right.double_count)
    yield "P-44.4.1.3", _nonstandard_key(left.nonstandard_key), _nonstandard_key(right.nonstandard_key)
    # Observe only the next applicable criterion. Later numbering fields may
    # be deferred pure reductions and need not be forced after a decision.
    for rule, field in zip(_P44_NUMBERING_RULES, _P44_NUMBERING_FIELDS):
        yield rule, getattr(left, field), getattr(right, field)
    if not same_numbered_parent:
        yield "P-44.4.1.11", left.isotope_key, right.isotope_key
        if left.stereo_identity is not None and left.stereo_identity == right.stereo_identity:
            yield "P-44.4.1.12", (), ()
        else:
            a_stereo, b_stereo = left.stereo_key, right.stereo_key
            if left.numbering_scope is not None and right.numbering_scope is not None and (
                    a_stereo and b_stereo and a_stereo[0] != b_stereo[0]
                    or bool(a_stereo) != bool(b_stereo)):
                yield "P-44.4.1.12", None, None
            else:
                yield "P-44.4.1.12", a_stereo, b_stereo
    if p44_only:
        return
    yield "P-45.1", _negative(left.multiplied_parents), _negative(right.multiplied_parents)
    yield "P-45.2.1", _negative(left.prefix_count), _negative(right.prefix_count)
    yield "P-45.2.2", left.prefix_locants, right.prefix_locants
    yield "P-45.2.3", left.prefix_citation_locants, right.prefix_citation_locants
    yield "P-45.3", _nonstandard_key(left.substituent_nonstandard_key), _nonstandard_key(right.substituent_nonstandard_key)
    if same_numbered_parent:
        yield "P-82.5.2", left.equivalent_isotope_key, right.equivalent_isotope_key
    else:
        yield "P-45.4", left.substituent_isotope_key, right.substituent_isotope_key
    yield "P-45.5", left.alphanumerical_key, right.alphanumerical_key
    if same_numbered_parent:
        yield 'P-14.4(j)', left.stereo_numbering_key, right.stereo_numbering_key


def _negative(value):
    return -value if value is not None else None


def _nonstandard_key(value):
    """Normalize legacy empty facts against structured bonding-number keys."""
    return (1,) if value == () else value


def compare_parents(left, right, *, trace=False):
    if left is None or right is None:
        return ParentComparison(None, "unavailable", "Parent facts were not supplied.")
    evidence = []
    for rule, a, b in _criteria(left, right):
        if trace:
            evidence.append((rule, a, b))
        if a is None or b is None:
            return ParentComparison(None, rule, "An applicable earlier criterion is unproved.", tuple(evidence))
        if a != b:
            return ParentComparison(-1 if a < b else 1, rule, "First decisive Blue Book criterion.", tuple(evidence))
    return ParentComparison(0, "tie", "All supplied applicable criteria tie.", tuple(evidence))


def compare_parent_seniority(left, right):
    """Compare parent facts only through P-44, before any P-45 criterion."""
    if left is None or right is None:
        return ParentComparison(None, "unavailable", "Parent facts were not supplied.")
    for rule, a, b in _criteria(left, right, p44_only=True):
        if a is None or b is None:
            return ParentComparison(None, rule, "An applicable P-44 criterion is unproved.")
        if a != b:
            return ParentComparison(-1 if a < b else 1, rule,
                                    "First decisive P-44 parent criterion.")
    return ParentComparison(0, "P-44 tie", "All supplied applicable P-44 criteria tie.")


def parent_seniority_frontier(items, facts, *, comparisons=None):
    """Retain P-44-maximal candidates without treating unknown as equality."""
    values = [facts(item) for item in items]
    if comparisons is not None:
        # Call-local evidence for consumers checking the surviving pairs.
        # The items retain these immutable facts throughout that operation.
        comparisons.clear()
    edges = [set() for _ in items]
    for left in range(len(items)):
        for right in range(left + 1, len(items)):
            result = compare_parent_seniority(values[left], values[right])
            if comparisons is not None:
                comparisons[(id(values[left]), id(values[right]))] = result
            comparison = result.preference
            if comparison == -1:
                edges[left].add(right)
            elif comparison == 1:
                edges[right].add(left)
    visiting, visited = set(), set()

    def cyclic(index):
        if index in visiting:
            return True
        if index in visited:
            return False
        visiting.add(index)
        if any(cyclic(other) for other in edges[index]):
            return True
        visiting.remove(index)
        visited.add(index)
        return False

    if any(cyclic(index) for index in range(len(items))):
        return list(items)
    dominated = set().union(*edges) if edges else set()
    return [item for index, item in enumerate(items) if index not in dominated]


def parent_frontier(items, facts):
    """Remove proved dominated candidates without treating unknown as equality.

    A partial comparison must not be handed to a total-order sort. Removing
    incoming edges only after evaluating all pairs also makes input order
    irrelevant. A contradictory fact set conservatively retains all items.
    """
    values = [facts(item) for item in items]
    # All candidates with the same proved leading criteria have a common
    # lexicographic prefix. Prune that prefix in linear time before comparing
    # uncertain suffixes pairwise (important for branched path enumeration).
    known = [i for i, value in enumerate(values) if value is not None]
    if len(known) < 2:
        return list(items)
    if known and not any(values[i].fusion_construction is not None
                         or values[i].numbering_scope is not None for i in known):
        # Construction comparisons are scoped to the same actual skeletal
        # atoms. Self-comparison keys cannot stand in for a comparison between
        # different physical parents, even if their graphs are isomorphic.
        rows = [_criteria(values[i], values[i], identity_ties=False) for i in known]
        keys = [[] for _ in known]
        # Later criteria cannot contribute once this common proved prefix
        # ends. Keep the same global prefix without building unused tails.
        for column in zip(*rows):
            if len({entry[0] for entry in column}) != 1 or any(entry[1] is None for entry in column):
                break
            for key, entry in zip(keys, column):
                key.append(entry[1])
        if keys[0]:
            keys = [tuple(key) for key in keys]
            best = min(keys)
            keep = {i for i, key in zip(known, keys) if key == best}
            indices = [i for i in range(len(items)) if values[i] is None or i in keep]
            items, values = [items[i] for i in indices], [values[i] for i in indices]
    edges = [set() for _ in items]
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            comparison = compare_parents(values[i], values[j]).preference
            if comparison == -1:
                edges[i].add(j)
            elif comparison == 1:
                edges[j].add(i)
    visiting, visited = set(), set()

    def cyclic(i):
        if i in visiting:
            return True
        if i in visited:
            return False
        visiting.add(i)
        if any(cyclic(j) for j in edges[i]):
            return True
        visiting.remove(i)
        visited.add(i)
        return False

    if any(cyclic(i) for i in range(len(items))):
        return list(items)
    dominated = set().union(*edges) if edges else set()
    return [item for i, item in enumerate(items) if i not in dominated]


def locant_key(value):
    """Supported numerical and element locants, including N-site indices."""
    text = str(value).replace("′", "'")
    return _locant_text_key(text)



@lru_cache(maxsize=4096)
def _locant_text_key(text):
    match = re.fullmatch(r"(\d+)([a-z]*)\^(\d+)", text)
    if match:
        # P-14.3.2: primed locants of the same base precede superscripts.
        return (1, int(match[1]), match[2], 2**31, int(match[3]))
    match = re.fullmatch(r"(\d+)([a-z]*)(\'*)", text)
    if match:
        return (1, int(match[1]), match[2], len(match[3]))
    match = re.fullmatch(r"(\d+)\((\d+)\)", text)
    if match:
        return (1, int(match[1]), "", 0, int(match[2]))
    match = re.fullmatch(r"N('+)(\d+)('*)", text)
    if match:
        return (0, 'N', int(match[2]), len(match[3]), len(match[1]))
    match = re.fullmatch(r"([NOSP])(\d*)(\'*)", text)
    if match:
        return (0, match[1], int(match[2] or 0), len(match[3]))
    return None


def locant_set(values):
    # Text normalization also distinguishes equal Python values with different
    # locant spellings (for example 1, True and 1.0) before memoization.
    return _locant_series_key(tuple(str(value).replace("′", "'") for value in values))


@lru_cache(maxsize=4096)
def _locant_series_key(values):
    keys = [_locant_text_key(value) for value in values]
    return tuple(sorted(keys)) if all(key is not None for key in keys) else None


@lru_cache(maxsize=4096)
def _prefix_comparison_keys(prefixes):
    """Cache only locant/citation facts; prefix ownership remains live."""
    keys = [locant for locant, _ in prefixes]
    prefix_locs = tuple(sorted(keys)) if all(key is not None for key in keys) else None
    citation = None
    if prefix_locs is not None:
        grouped = {}
        for locant, text in prefixes:
            grouped.setdefault(text, []).append(locant)
        citation = tuple(loc for text in sorted(grouped, key=_formatting.prefix_alphabetization_key)
                         for loc in sorted(grouped[text]))
    return prefix_locs, citation


def _numbering_comparison_values(mapping_rows, hetero_rows, indicated, suffix,
                                attachment, multiple, double, replacement,
                                aromatic, kind, rings, count):
    mapping = dict(mapping_rows)
    def bond_locs(edges):
        return locant_set(min((mapping[a], mapping[b]), key=locant_key)
                          for a, b, _ in edges)
    return (
        locant_set(indicated) if indicated is not None else None,
        locant_set(mapping[i] for i, _ in hetero_rows) if replacement else (),
        tuple(locant_set(mapping[i] for i, number in hetero_rows if number == element)
              for element in _RING_ELEMENTS) if replacement else (),
        () if rings <= 1 else None,
        locant_set(suffix), locant_set(attachment),
        None if aromatic and count is None or kind == 'ring' and multiple
             else () if aromatic else bond_locs(multiple),
        None if aromatic and count is None or kind == 'ring' and multiple
             else () if aromatic else bond_locs(double),
    )


def _prefix_comparison_values(prefixes):
    return _prefix_comparison_keys(tuple((locant_key(loc), text) for loc, text in prefixes))


def _name_comparison_value(name):
    return (_formatting.prefix_alphabetization_key(name)[:3],)


def _unannotated_carbon_chain_numbering_scope(
    mol, atoms, suffix_centers, prefixes, suffix_locants, *, kind, skeleton_bonds=None,
):
    """Prove identity across numberings for the bounded neutral chain grammar.

    Atom-set equality alone is not a parent-identity proof for arbitrary ring,
    retained, hydro, or constructed parent systems. This limited signature is
    only for a fully mapped acyclic all-carbon chain with declared suffix and
    prefix ownership; it includes the actual parent bonds and suffix sites.
    """
    atoms = frozenset(atoms)
    suffix_centers = frozenset(suffix_centers)
    if (kind != "chain" or not atoms or prefixes is None
            or len(suffix_centers) != len(suffix_locants)
            or any(locant_key(value) is None for value in suffix_locants)
            or any(mol.GetAtomWithIdx(index).GetAtomicNum() != 6 for index in atoms)):
        return None
    if skeleton_bonds is None:
        skeleton_bonds = (
            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondTypeAsDouble())
            for bond in iter_bonds(mol)
            if bond.GetBeginAtomIdx() in atoms and bond.GetEndAtomIdx() in atoms
        )
    bonds = tuple(sorted((min(a, b), max(a, b), order)
                         for a, b, order in skeleton_bonds))
    if len(bonds) != len(atoms) - 1:
        return None
    degrees = Counter(index for left, right, _order in bonds for index in (left, right))
    if any(degrees[index] > 2 for index in atoms):
        return None
    return ("unannotated-neutral-carbon-chain", atoms, suffix_centers, bonds)


def _parent_skeleton_data(mol, atoms):
    """Share graph-only checks between numberings of the same physical parent.

    Store indices and primitive facts, so cached data cannot own RDKit atoms
    or bonds. Locants, prefix proofs and annotation comparisons stay outside.
    """
    cache = mol_cache_dict(mol, 'parent_skeleton_data')
    if atoms in cache:
        return cache[atoms]
    view = graph_view(mol, None)
    seen, stack = set(), [next(iter(atoms))]
    while stack:
        index = stack.pop()
        if index in seen:
            continue
        seen.add(index)
        stack.extend(other for other in view.neighbors[index]
                     if other in atoms and other not in seen)
    if seen != atoms:
        cache[atoms] = None
        return None
    for _component, attachments in mapped_graph_components(mol, atoms):
        if len(attachments) != 1:
            cache[atoms] = None
            return None
    if type(mol).__module__ != "rdkit.Chem.rdchem" or type(mol).__name__ not in {"Mol", "RWMol"}:
        # Preserve callback order and lazy field reads for custom molecules.
        bonds = [bond for bond in iter_bonds(mol)
                 if bond.GetBeginAtomIdx() in atoms and bond.GetEndAtomIdx() in atoms]
        result = (
            tuple(sorted(Counter(view.atomic_nums[i] for i in atoms).items())),
            tuple((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondTypeAsDouble())
                  for bond in bonds),
            any(bond.GetIsAromatic() for bond in bonds),
            len(bonds) - len(atoms) + 1,
            any(view.ring_atoms[i] for i in atoms),
            tuple(i for i in atoms if view.atomic_nums[i] != 6),
        )
        cache[atoms] = result
        return result
    # Only exact native RDKit objects have side-effect-free edge getters.
    # A fresh binary graph snapshot guards shared rows against edits between
    # different parent scopes. Custom molecule implementations retain the
    # original live reads. Python cache attributes are absent from this state.
    native_state = None
    try:
        native_state = mol.ToBinary()
    except (AttributeError, TypeError, ValueError, RuntimeError):
        pass
    cached_bonds = mol_cache_get(mol, 'parent_skeleton_bonds') if native_state is not None else None
    if cached_bonds is not None and cached_bonds[0] == native_state:
        bond_data = cached_bonds[1]
    else:
        # Retain native edge order, orientation and aromatic flags.
        bond_data = tuple(
            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
             bond.GetBondTypeAsDouble(), bond.GetIsAromatic())
            for bond in iter_bonds(mol))
        if native_state is not None:
            mol_cache_set(mol, 'parent_skeleton_bonds', (native_state, bond_data))
    bonds = tuple((a, b, order) for a, b, order, _ in bond_data
                  if a in atoms and b in atoms)
    result = (
        tuple(sorted(Counter(view.atomic_nums[i] for i in atoms).items())),
        bonds,
        any(aromatic for a, b, _order, aromatic in bond_data
            if a in atoms and b in atoms),
        len(bonds) - len(atoms) + 1,
        any(view.ring_atoms[i] for i in atoms),
        tuple(i for i in atoms if view.atomic_nums[i] != 6),
    )
    cache[atoms] = result
    return result


def _parent_annotation_flags(mol):
    cached = mol_cache_get(mol, 'parent_annotation_flags')
    if cached is None:
        view = graph_view(mol, None)
        charged = frozenset(i for i, charge in enumerate(view.formal_charges) if charge)
        radicals = any(a.GetNumRadicalElectrons() for a in iter_atoms(mol))
        annotated = (any(view.isotopes)
                     or any(int(a.GetChiralTag()) for a in iter_atoms(mol))
                     or any(int(b.GetStereo()) for b in iter_bonds(mol)))
        cached = mol_cache_set(mol, 'parent_annotation_flags', (charged, radicals, annotated))
    return cached


def graph_parent_facts(mol, atoms, locants, *, principal_class, suffix_centers=(),
                       suffix_locants=(), prefixes=None, kind="chain",
                       name=None, replacement=False, indicated_h=(),
                       multiplied_parents=1, attachment_locants=(), allow_annotations=False,
                       charge_role_proofs=(), nonstandard_parent_proof=None,
                       substituent_nonstandard_proof=None, prefix_branch_ownership=(),
                       resolve_branch=None):
    """Facts for a neutral, ordinary-valence parent with proved ownership.

    ``atoms`` are skeletal parent atoms, never whole-molecule coverage atoms.
    Callers explicitly declare rendered suffix centers and prefix units.
    Unknown ring-system details and aromatic numbering remain barriers.
    """
    charge_role_proofs = tuple(charge_role_proofs)
    charged_atoms, radicals, annotated = _parent_annotation_flags(mol)
    if (charged_atoms and not _neutral_charge.validate_neutral_charge_role_proofs(mol, charge_role_proofs)):
        return None
    if charge_role_proofs and not _neutral_charge.validate_neutral_charge_role_proofs(mol, charge_role_proofs):
        return None
    if not atoms:
        return None
    atoms, mapping = frozenset(atoms), dict(locants)
    view = graph_view(mol, None)
    if radicals or charged_atoms & atoms or not allow_annotations and annotated:
        return None
    if not atoms <= mapping.keys() or any(locant_key(mapping[i]) is None for i in atoms) or any(view.atomic_nums[i] not in
        {6, 7, 8, 16} for i in atoms):
        return None
    if charge_role_proofs:
        from .neutral_charge_components import neutral_charge_prefix_facts

        if any(set(proof.atoms) & atoms for proof in charge_role_proofs):
            return None
        prefix_facts = neutral_charge_prefix_facts(mol, charge_role_proofs)
        if prefix_facts is None:
            return None
        emitted_prefixes = {(str(loc), str(text)) for loc, text in (prefixes or ())}
        for motif_kind, host, motif_atoms in prefix_facts:
            if host not in atoms:
                from .neutral_charge_components import nested_neutral_prefix_owned
                if nested_neutral_prefix_owned(mol, atoms, mapping, prefixes or (),
                        (motif_kind, host, motif_atoms), prefix_branch_ownership,
                        resolve_branch=resolve_branch):
                    continue
                return None
            if (host not in atoms or host not in mapping
                    or not set(motif_atoms).isdisjoint(atoms)
                    or (str(mapping[host]), motif_kind) not in emitted_prefixes):
                return None
    # A tree or connected ring system is required; disjoint assemblies need
    # explicit multiplication/phane facts instead of E-V+1 on mixed atoms.
    # An external component attached twice is a bridge/fusion, not two
    # independently owned substituents. Its prefix count is unproved here.
    skeleton = _parent_skeleton_data(mol, atoms)
    if skeleton is None:
        return None
    elements, bonds, aromatic, rings, has_ring_atoms, hetero = skeleton
    if kind == "chain" and (rings or has_ring_atoms):
        return None
    if kind == "ring" and rings < 1:
        return None
    multiple = [b for b in bonds if b[2] in {2, 3}]
    double = [b for b in multiple if b[2] == 2]
    if aromatic:
        from .ring_parent_selection import aromatic_bond_counts
        count, double_count = aromatic_bond_counts(mol, atoms)
    else:
        count = len(multiple)
        double_count = len(double)
    if nonstandard_parent_proof is None and not any(element == 16 for element, _ in elements):
        # The cached skeletal element counts prove the sulfur scan is empty.
        nonstandard = ()
    else:
        nonstandard = _nonstandard_evidence.parent_nonstandard_bonding_key(
            mol, atoms, mapping, nonstandard_parent_proof)
    element_counts = view.elements
    if (substituent_nonstandard_proof is None
            and not element_counts["P"] and not element_counts["S"]):
        # With neither P nor S anywhere, the existing unproved-prefix path
        # proves absence without scanning every atom for each parent map.
        substituent_nonstandard = ()
    else:
        substituent_nonstandard = _nonstandard_evidence.substituent_nonstandard_bonding_key(
            mol, atoms, mapping, prefixes, substituent_nonstandard_proof,
            suffix_centers=suffix_centers)
    isotope, equivalent_isotope, stereo, branch_isotope, numbering_scope, configuration_identity, stereo_numbering_key = (), (), _unassigned_stereo_evidence(mol), (), None, None, None
    if allow_annotations:
        has_isotopes = any(view.isotopes)
        if has_isotopes:
            isotope_mapping = mapping
            if principal_class == (11, 0):
                from .amide_isotopes import amide_parent_annotation_mapping
                physical = amide_parent_annotation_mapping(mol, atoms, mapping)
                if physical is not None:
                    isotope_mapping = physical
            isotope = _annotations.isotope_seniority(mol, isotope_mapping)
            equivalent_isotope = _annotations.isotope_numbering_key(
                mol, isotope_mapping, include_substituent_groups=True)
        else:
            # All three isotope consumers otherwise return these same absent
            # keys immediately; no isotope role or ownership proof is needed.
            isotope, equivalent_isotope = (0, (), (), (), (), ()), ((), ())
        stereo = _annotations.stereo_seniority(mol, mapping)
        if has_isotopes:
            branch_isotope = _annotations.substituent_isotopes(mol, isotope_mapping)
        numbering_scope = (atoms, frozenset(suffix_centers))
        configuration_identity = _annotations.stereo_identity(mol, mapping)
        stereo_numbering_key = _annotations.stereo_numbering(mol, mapping)
    else:
        numbering_scope = _unannotated_carbon_chain_numbering_scope(
            mol, atoms, suffix_centers, prefixes, suffix_locants, kind=kind,
            skeleton_bonds=bonds,
        )
        if numbering_scope is not None:
            # The allow_annotations=False guard above excludes isotope labels
            # and assigned stereo descriptors throughout the input. Preserve
            # potential stereogenicity in `stereo_key`; only the numbering of
            # assigned descriptors is proved empty.
            stereo_numbering_key = ()
    if not _DEFER_COMPARISON_FACTS or not (replacement or prefixes is not None and len(prefixes) >= 3):
        # Do not pay for snapshots, groups, or a second constructor on cheap
        # candidates. This is the original eager normalization path.
        bond_locs = lambda edges: locant_set(min((mapping[b[0]], mapping[b[1]]),
                                                key=locant_key) for b in edges)
        prefix_locs, citation = (None, None) if prefixes is None else _prefix_comparison_keys(
            tuple((locant_key(loc), text) for loc, text in prefixes))
        return ParentFacts(
            principal_class=principal_class, suffix_count=len(suffix_centers), kind=kind,
            elements=elements, ring_count=rings, multiple_count=count,
            double_count=double_count, nonstandard_key=nonstandard,
            indicated_h_locants=locant_set(indicated_h) if indicated_h is not None else None,
            replacement_locants=locant_set(mapping[i] for i in hetero) if replacement else (),
            replacement_element_locants=tuple(locant_set(mapping[i] for i in hetero
                if view.atomic_nums[i] == e) for e in _RING_ELEMENTS) if replacement else (),
            fusion_locants=() if rings <= 1 else None,
            suffix_locants=locant_set(suffix_locants), attachment_locants=locant_set(attachment_locants),
            unsaturation_locants=None if aromatic and count is None or kind == "ring" and multiple
                                 else () if aromatic else bond_locs(multiple),
            double_locants=None if aromatic and count is None or kind == "ring" and multiple
                           else () if aromatic else bond_locs(double),
            isotope_key=isotope, equivalent_isotope_key=equivalent_isotope,
            stereo_key=stereo, multiplied_parents=multiplied_parents,
            prefix_count=len(prefixes) if prefixes is not None else None,
            prefix_locants=prefix_locs, prefix_citation_locants=citation,
            substituent_nonstandard_key=substituent_nonstandard, substituent_isotope_key=branch_isotope,
            alphanumerical_key=_formatting.prefix_alphabetization_key(name)[:3] if name is not None else None,
            numbering_scope=numbering_scope,
            stereo_identity=configuration_identity,
            stereo_numbering_key=stereo_numbering_key,
        )
    deferred = [(_numbering_comparison_values,
        (tuple(mapping.items()), tuple((i, view.atomic_nums[i]) for i in hetero),
         tuple(indicated_h) if indicated_h is not None else None,
         tuple(suffix_locants), tuple(attachment_locants), tuple(multiple), tuple(double),
         replacement, aromatic, kind, rings, count),
        ('indicated_h_locants', 'replacement_locants', 'replacement_element_locants',
         'fusion_locants', 'suffix_locants', 'attachment_locants',
         'unsaturation_locants', 'double_locants'))]
    if prefixes is not None:
        deferred.append((_prefix_comparison_values, (tuple(tuple(row) for row in prefixes),),
                         ('prefix_locants', 'prefix_citation_locants')))
    if name is not None:
        deferred.append((_name_comparison_value, (name,), ('alphanumerical_key',)))
    return _deferred_parent_facts(deferred,
        principal_class=principal_class, suffix_count=len(suffix_centers), kind=kind,
        elements=elements, ring_count=rings, multiple_count=count,
        double_count=double_count, nonstandard_key=nonstandard,
        isotope_key=isotope, equivalent_isotope_key=equivalent_isotope,
        stereo_key=stereo, multiplied_parents=multiplied_parents,
        prefix_count=len(prefixes) if prefixes is not None else None,
        substituent_nonstandard_key=substituent_nonstandard, substituent_isotope_key=branch_isotope,
        numbering_scope=numbering_scope,
        stereo_identity=configuration_identity,
        stereo_numbering_key=stereo_numbering_key,
    )


def _unassigned_stereo_evidence(mol):
    """No assigned tags does not prove that stereogenic features are absent."""
    Chem, stereo = _annotations._stereo_runtime()
    return None if stereo._potential_stereochemistry(mol, Chem) else ()


def principal_context(mol):
    """Recognize absent functions without treating an unsupported inventory as empty."""
    from .extended_inventory import selection_inventory

    inventory = selection_inventory(mol)
    if inventory.supported:
        groups = inventory.principal
        return (groups[0].seniority if groups else (40, 0)), groups
    # Ordinary neutral ring hydrides can include S even though the bounded
    # functional inventory delegates annular sulfur. No exocyclic functions or
    # nonring amines/oxygen/sulfur may be silently classified as hydrides here.
    for atom in iter_atoms(mol):
        number = atom.GetAtomicNum()
        if number in {1, 6, 9, 17, 35, 53}:
            continue
        if number not in {7, 8, 16} or not atom.IsInRing() or (
            number == 16 and atom.GetTotalValence() != 2
        ):
            return None
    return (40, 0), ()


def ring_record_facts(mol, seq, suffix_kind, raw_prefixes, name, *, parent_name=None):
    """Adapter for the monocycle renderer's explicit suffix operation."""
    context = principal_context(mol)
    if context is None:
        return None
    rank, groups = context
    covered = []
    removed = set()
    mapping = {i: str(j) for j, i in enumerate(seq, 1)}
    if groups:
        kind = {"ring_alcohol_suffix": "alcohol", "ring_carbonyl_suffix": "ketone"}.get(suffix_kind)
        if kind is None or any(g.kind != kind for g in groups):
            return None
        covered = [g.center for g in groups if g.center in mapping]
        removed = {(i, "hydroxy" if kind == "alcohol" else "oxo") for i in covered}
    elif suffix_kind:
        return None
    if len(seq) > 10 and any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in seq):
        return None
    prefixes = tuple((mapping[i], text) for i, text in raw_prefixes if (i, text) not in removed)
    if parent_name is not None:
        from .ring_parent_selection import rendered_ring_facts
        return rendered_ring_facts(mol, seq, mapping, parent_name, prefixes, name,
                                   suffix_centers=covered, suffix_locants=[mapping[i] for i in covered])
    indicated = None if any(mol.GetAtomWithIdx(i).GetIsAromatic()
                           and mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in seq) else ()
    return graph_parent_facts(
        mol, seq, mapping, principal_class=rank, suffix_centers=covered,
        suffix_locants=[mapping[i] for i in covered], prefixes=prefixes,
        kind="ring", name=name, indicated_h=indicated,
    )
