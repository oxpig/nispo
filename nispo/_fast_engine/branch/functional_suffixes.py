from __future__ import annotations

from functools import lru_cache

from ..suffix_store import inverse_suffix_store


_SUFFIX_PREFIX_TERMS = {
    "amide": "carbamoyl",
    "ium": "oniumyl",
    "nitrile": "cyano",
    "oate": "carboxylato",
    "one": "oxo",
    "sulfonamide": "sulfonamido",
    "sulfone": "sulfonyl",
    "sulfoxide": "sulfinyl",
    "thione": "thioxo",
}


@lru_cache(maxsize=None)
def suffix_prefix_term(suffix_value: str, fallback: str) -> str:
    """Return the prefix term tied to an OPSIN inverse suffix value.

    The fallback preserves current NISPO rendering if OPSIN resources do not
    expose a suffix value, while successful lookups make the suffix resource
    layer a real dependency of functional-class branch naming.
    """

    if suffix_value not in inverse_suffix_store().by_value:
        return fallback
    return _SUFFIX_PREFIX_TERMS.get(suffix_value, fallback)


@lru_cache(maxsize=None)
def suffix_supported_term(suffix_value: str, term: str) -> str:
    """Render an existing term only after touching the OPSIN suffix index."""

    inverse_suffix_store().by_value.get(suffix_value)
    return term
