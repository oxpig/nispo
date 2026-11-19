from __future__ import annotations

from weakref import WeakKeyDictionary


_MOL_CACHES: WeakKeyDictionary[object, dict[str, object]] = WeakKeyDictionary()
_MOL_CACHE_ATTR = "_nispo_python_cache"

_NAMING_STATE_PROPS = (
    "_nispo_fast_poly_bridge_route", "_nispo_fast_poly_retained_defer_bypass",
    "_nispo_allow_fragment_substituent_fallback",
    "_nispo_enable_retained_partial_hydro_fallback",
    "_nispo_fragment_substituent_depth", "_nispo_fragment_substituent_view",
    "_nispo_kekulized_aromatic_view", "_nispo_principal_prefix_mode",
)


def naming_cache_state(mol):
    """Distinguish temporary rendering modes on the same API-owned graph."""
    return tuple((prop, mol.GetProp(prop)) for prop in _NAMING_STATE_PROPS
                 if mol.HasProp(prop))


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


def exact_graph_computation(mol, Chem, namespace, parameters, compute):
    """Reuse a pure construction only for exactly the same native graph state.

    Proof validation must still rebuild on its first call. Unlike the ordinary
    naming-owned graph caches, this cache also checks all native properties,
    including atom maps, annotations and temporary rendering modes. A changed
    proof or any graph edit therefore takes the original construction path.
    Unkeyable inputs use that path too. Python caches are not part of RDKit's
    binary representation and cannot authorize a hit by themselves.
    """
    try:
        hash(parameters)
        state = mol.ToBinary(Chem.PropertyPickleOptions.AllProps)
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return compute()
    cache = mol_cache_dict(mol, namespace)
    previous = cache.get(parameters)
    if previous is not None and previous[0] == state:
        return previous[1]
    result = compute()
    # Do not skip native side effects on a later call. Only computations that
    # left the exact input state unchanged are eligible for reuse.
    try:
        unchanged = mol.ToBinary(Chem.PropertyPickleOptions.AllProps) == state
    except (AttributeError, TypeError, ValueError, RuntimeError):
        unchanged = False
    if unchanged:
        if parameters not in cache and len(cache) >= 128:
            cache.pop(next(iter(cache), None), None)
        cache[parameters] = (state, result)
    return result
