from __future__ import annotations

from functools import lru_cache

from .molecule_cache import mol_cache_dict
from .scaffold_catalog import parent_scaffold_catalog
from .template_access import template_record_locants, template_record_mol, template_record_name


def resource_ring_parent_candidates(mol, ordered: tuple[int, ...], Chem) -> list[tuple[str, tuple[int, ...]]]:
    cache = mol_cache_dict(mol, "resource_ring_parent_candidates")
    cache_key = tuple(ordered)
    cached = cache.get(cache_key)
    if cached is not None:
        return list(cached)
    candidates = _resource_ring_parent_candidates_uncached(mol, ordered, Chem)
    cache[cache_key] = tuple(candidates)
    return candidates


def _resource_ring_parent_candidates_uncached(mol, ordered: tuple[int, ...], Chem) -> list[tuple[str, tuple[int, ...]]]:
    ring_set = set(ordered)
    symbol_key = tuple(sorted(mol.GetAtomWithIdx(idx).GetSymbol() for idx in ordered))
    candidates = []
    for template in _single_ring_templates_by_symbols(Chem).get((len(ordered), symbol_key), ()):
        query = template_record_mol(template)
        for match in mol.GetSubstructMatches(query, uniquify=False):
            if set(match) != ring_set:
                continue
            seq = _sequence_from_match(template, match)
            if seq:
                candidates.append((template_record_name(template), seq))
    return candidates


@lru_cache(maxsize=1)
def _single_ring_templates(Chem) -> dict[int, tuple[dict, ...]]:
    grouped: dict[int, list[dict]] = {}
    seen: set[tuple[str, str]] = set()
    for template in parent_scaffold_catalog(Chem).single_ring:
        key = (template_record_name(template), Chem.MolToSmiles(template_record_mol(template), canonical=True))
        if key in seen:
            continue
        seen.add(key)
        grouped.setdefault(template_record_mol(template).GetNumAtoms(), []).append(template)
    return {size: tuple(items) for size, items in grouped.items()}


@lru_cache(maxsize=1)
def _single_ring_templates_by_symbols(Chem) -> dict[tuple[int, tuple[str, ...]], tuple[dict, ...]]:
    grouped: dict[tuple[int, tuple[str, ...]], list[dict]] = {}
    for size, templates in _single_ring_templates(Chem).items():
        for template in templates:
            mol = template_record_mol(template)
            symbol_key = tuple(sorted(atom.GetSymbol() for atom in mol.GetAtoms()))
            grouped.setdefault((size, symbol_key), []).append(template)
    return {key: tuple(items) for key, items in grouped.items()}


def _sequence_from_match(template: dict, match: tuple[int, ...]) -> tuple[int, ...]:
    locant_items = []
    for template_idx, mol_idx in enumerate(match):
        locant = template_record_locants(template).get(template_idx, "")
        if not locant or not locant.isdigit():
            return ()
        locant_items.append((int(locant), mol_idx))
    locant_items.sort()
    return tuple(mol_idx for _, mol_idx in locant_items)
