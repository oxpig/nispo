from __future__ import annotations

from functools import lru_cache

from .molecule_iteration import iter_atoms
from .molecule_cache import mol_cache_dict
from .template_access import template_record_locants, template_record_mol, template_record_name
from .template_match_utils import match_has_extra_core_bonds
from .template_store import template_store

_DEFAULT_MATCH_HAS_EXTRA_CORE_BONDS = match_has_extra_core_bonds

# Legacy resource names for these four mancude heteromonocycles are valid
# general-nomenclature aliases. P-52.2.2.2 uses explicit heteroatom locants
# in the preferred Hantzsch–Widman parent. Normalize only when the matched
# atom numbering itself proves the corresponding five-membered aromatic
# topology; this is not a substitution on a rendered whole name.
_AZOLE_RESOURCE_PARENTS = {
    "oxazole": ("O", 3, "1,3-oxazole"),
    "oxazol": ("O", 3, "1,3-oxazole"),
    "isoxazole": ("O", 2, "1,2-oxazole"),
    "isoxazol": ("O", 2, "1,2-oxazole"),
    "isooxazole": ("O", 2, "1,2-oxazole"),
    "isooxazol": ("O", 2, "1,2-oxazole"),
    "thiazole": ("S", 3, "1,3-thiazole"),
    "thiazol": ("S", 3, "1,3-thiazole"),
    "isothiazole": ("S", 2, "1,2-thiazole"),
    "isothiazol": ("S", 2, "1,2-thiazole"),
}


def _preferred_azole_resource_parent(mol, name: str, seq: tuple[int, ...]) -> str:
    if name in {"isooxazolin", "isoxazoline", "isoxazolin", "isooxazoline"} and len(seq) == 5:
        atoms = [mol.GetAtomWithIdx(idx) for idx in seq]
        if ([a.GetAtomicNum() for a in atoms] == [8, 7, 6, 6, 6]
                and not any(a.GetFormalCharge() or a.GetIsAromatic() for a in atoms)
                and [mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble()
                     for a, b in zip(seq, seq[1:] + seq[:1])] == [1, 2, 1, 1, 1]):
            return "4,5-dihydro-1,2-oxazole"
    if name in {'s-triazole', 's-triazol', 'sym-triazole', 'sym-triazol'} and len(seq) == 5:
        atoms = [mol.GetAtomWithIdx(idx) for idx in seq]
        if ([a.GetSymbol() for a in atoms] == ['N', 'N', 'C', 'N', 'C']
                and all(a.GetIsAromatic() and not a.GetFormalCharge() for a in atoms)):
            indicated = [str(i) for i, a in enumerate(atoms, 1)
                         if a.GetAtomicNum() == 7 and a.GetTotalNumHs() > 0]
            if len(indicated) == 1:
                return indicated[0] + 'H-1,2,4-triazole'
    spec = _AZOLE_RESOURCE_PARENTS.get(name)
    if spec is None or len(seq) != 5 or len(set(seq)) != 5:
        return name
    hetero, nitrogen_locant, preferred = spec
    symbols = [mol.GetAtomWithIdx(idx).GetSymbol() for idx in seq]
    expected = ["C"] * 5
    expected[0] = hetero
    expected[nitrogen_locant - 1] = "N"
    if symbols != expected or not all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in seq):
        return name
    return preferred


def resource_ring_parent_candidates(mol, ordered: tuple[int, ...], Chem) -> list[tuple[str, tuple[int, ...]]]:
    cache = mol_cache_dict(mol, "resource_ring_parent_candidates")
    # Resource queries enumerate full-molecule matches in native query order;
    # the caller's ring traversal only supplies the atom set to filter them.
    atom_set = frozenset(ordered)
    cache_key = atom_set if len(atom_set) == len(ordered) else tuple(ordered)
    cached = cache.get(cache_key)
    if cached is not None:
        return list(cached)
    candidates = _resource_ring_parent_candidates_uncached(mol, ordered, Chem)
    cache[cache_key] = tuple(candidates)
    return candidates


def _resource_ring_parent_candidates_uncached(mol, ordered: tuple[int, ...], Chem) -> list[tuple[str, tuple[int, ...]]]:
    ring_set = frozenset(ordered)
    symbol_key = tuple(sorted(mol.GetAtomWithIdx(idx).GetSymbol() for idx in ordered))
    candidates = []
    for template, atom_order in _single_ring_templates_by_symbols(Chem).get((len(ordered), symbol_key), ()):
        query = template_record_mol(template)
        matches = _resource_matches_by_atom_set(mol, query).get(ring_set, ())
        # Every orientation in this bucket has the same induced atom set;
        # extra-core topology cannot depend on its query-to-atom permutation.
        # Replaced predicates may use the orientation, so keep their original
        # per-match calls rather than assuming the default predicate's proof.
        shared_check = match_has_extra_core_bonds is _DEFAULT_MATCH_HAS_EXTRA_CORE_BONDS
        if shared_check and matches and match_has_extra_core_bonds(mol, query, matches[0]):
            continue
        for match in matches:
            if not shared_check and match_has_extra_core_bonds(mol, query, match):
                continue
            seq = _sequence_from_match(template, match, atom_order)
            if seq:
                name = _preferred_azole_resource_parent(mol, template_record_name(template), seq)
                candidates.append((name, seq))
    return candidates


def _resource_matches_by_atom_set(mol, query):
    """Share ordered native matches across rings of one naming-owned graph.

    Keep RDKit's original full-molecule search and its default match bound.
    Filtering before searching, or searching separate induced rings, could
    change which matches survive that bound. Query objects are immutable
    resource graphs, and the molecule-local lifetime follows the existing
    resource candidate cache's graph immutability contract.
    """
    cache = mol_cache_dict(mol, "resource_ring_matches_by_atom_set")
    cached = cache.get(query)
    if cached is None:
        grouped = {}
        for match in mol.GetSubstructMatches(query, uniquify=False):
            grouped.setdefault(frozenset(match), []).append(match)
        cached = {atoms: tuple(matches) for atoms, matches in grouped.items()}
        cache[query] = cached
    return cached


@lru_cache(maxsize=1)
def _single_ring_templates(Chem) -> dict[int, tuple[dict, ...]]:
    grouped: dict[int, list[dict]] = {}
    seen: set[tuple[str, str]] = set()
    for template in template_store(Chem).single_ring_templates():
        key = (template_record_name(template), Chem.MolToSmiles(template_record_mol(template), canonical=True))
        if key in seen:
            continue
        seen.add(key)
        grouped.setdefault(template_record_mol(template).GetNumAtoms(), []).append(template)
    return {size: tuple(items) for size, items in grouped.items()}


@lru_cache(maxsize=1)
def _single_ring_templates_by_symbols(Chem) -> dict[
    tuple[int, tuple[str, ...]], tuple[tuple[dict, tuple[int, ...] | None], ...]
]:
    grouped: dict[tuple[int, tuple[str, ...]], list[tuple[dict, tuple[int, ...] | None]]] = {}
    for size, templates in _single_ring_templates(Chem).items():
        for template in templates:
            mol = template_record_mol(template)
            symbol_key = tuple(sorted(atom.GetSymbol() for atom in iter_atoms(mol)))
            grouped.setdefault((size, symbol_key), []).append((template, _template_atom_order(template)))
    return {key: tuple(items) for key, items in grouped.items()}


def _template_atom_order(template: dict) -> tuple[int, ...] | None:
    """Compile static numeric locants once while indexing the resource templates.

    An empty order rejects unnumbered templates. Duplicate numeric locants
    retain the original sort below because their ties depend on matched indices.
    """
    locants = template_record_locants(template)
    values = [locants.get(idx, "") for idx in range(template_record_mol(template).GetNumAtoms())]
    if not all(value and value.isdigit() for value in values):
        return ()
    numbers = tuple(map(int, values))
    if len(set(numbers)) != len(numbers):
        return None
    return tuple(sorted(range(len(numbers)), key=numbers.__getitem__))


def _sequence_from_match(
    template: dict, match: tuple[int, ...], atom_order: tuple[int, ...] | None
) -> tuple[int, ...]:
    if atom_order is not None:
        return tuple(match[idx] for idx in atom_order)
    locant_items = []
    for template_idx, mol_idx in enumerate(match):
        locant = template_record_locants(template).get(template_idx, "")
        if not locant or not locant.isdigit():
            return ()
        locant_items.append((int(locant), mol_idx))
    locant_items.sort()
    return tuple(mol_idx for _, mol_idx in locant_items)
