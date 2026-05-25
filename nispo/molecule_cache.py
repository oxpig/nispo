from __future__ import annotations

from weakref import WeakKeyDictionary


_MOL_CACHES: WeakKeyDictionary[object, dict[str, object]] = WeakKeyDictionary()


def mol_cache_get(mol, key: str, default=None):
    cache = _MOL_CACHES.get(mol)
    if cache is None:
        return default
    return cache.get(key, default)


def mol_cache_set(mol, key: str, value):
    _MOL_CACHES.setdefault(mol, {})[key] = value
    return value


def mol_cache_dict(mol, key: str) -> dict:
    cache = _MOL_CACHES.setdefault(mol, {})
    value = cache.get(key)
    if value is None:
        value = {}
        cache[key] = value
    return value
