from __future__ import annotations

from weakref import WeakKeyDictionary


_MOL_CACHES: WeakKeyDictionary[object, dict[str, object]] = WeakKeyDictionary()
_MOL_CACHE_ATTR = "_nispo_python_cache"


def mol_cache_get(mol, key: str, default=None):
    cache = getattr(mol, _MOL_CACHE_ATTR, None)
    if cache is not None:
        return cache.get(key, default)
    cache = _MOL_CACHES.get(mol)
    if cache is None:
        return default
    return cache.get(key, default)


def mol_cache_set(mol, key: str, value):
    cache = getattr(mol, _MOL_CACHE_ATTR, None)
    if cache is None:
        cache = {}
        try:
            setattr(mol, _MOL_CACHE_ATTR, cache)
        except Exception:
            cache = _MOL_CACHES.setdefault(mol, {})
    cache[key] = value
    return value


def mol_cache_dict(mol, key: str) -> dict:
    cache = getattr(mol, _MOL_CACHE_ATTR, None)
    if cache is None:
        cache = {}
        try:
            setattr(mol, _MOL_CACHE_ATTR, cache)
        except Exception:
            cache = _MOL_CACHES.setdefault(mol, {})
    value = cache.get(key)
    if value is None:
        value = {}
        cache[key] = value
    return value
