from __future__ import annotations

from collections.abc import Sequence


_BENZIMIDAZOLE_BENZENOID_MIRROR = {
    "4": "7",
    "5": "6",
    "6": "5",
    "7": "4",
}


def preferred_benzimidazole_n1_benzenoid_locant_map(locants: Sequence[str]) -> dict[str, str]:
    """Return a symmetry map for benzimidazole benzenoid locants with N-1 fixed."""

    numeric_locants = tuple(locant for locant in locants if locant in _BENZIMIDAZOLE_BENZENOID_MIRROR)
    if not numeric_locants:
        return {}
    current = tuple(sorted(int(locant) for locant in numeric_locants))
    mirrored = tuple(sorted(int(_BENZIMIDAZOLE_BENZENOID_MIRROR[locant]) for locant in numeric_locants))
    if mirrored >= current:
        return {}
    return {locant: _BENZIMIDAZOLE_BENZENOID_MIRROR[locant] for locant in numeric_locants}
