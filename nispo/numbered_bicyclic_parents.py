"""Validate the numbered topology of simple P-23 bicyclic parent hydrides."""
from functools import lru_cache
import re

from .template_access import template_record_locants, template_record_mol, template_record_name


@lru_cache(maxsize=1024)
def _valid_numbered_graph(name, locants, edges):
    descriptor = re.search(r'bicyclo\[(\d+)\.(\d+)\.(\d+)\]', name)
    if descriptor is None or not name.endswith('ane'):
        return False
    a, b, c = map(int, descriptor.groups())
    n, bridgehead = a + b + c + 2, a + 2
    if not a >= b >= c >= 0 or b == 0 or set(locants) != set(range(1, n + 1)):
        return False
    paths = (
        tuple(range(1, bridgehead + 1)),
        (bridgehead, *range(bridgehead + 1, a + b + 3), 1),
        (1, *range(a + b + 3, n + 1), bridgehead),
    )
    expected = {tuple(sorted(pair)) for path in paths for pair in zip(path, path[1:])}
    return set(edges) == expected


def has_proved_bicyclic_numbering(template):
    """Recognize a complete saturated numbered P-23 parent, not an alias."""
    query = template_record_mol(template)
    mapping = template_record_locants(template)
    if (len(mapping) != query.GetNumAtoms()
            or any(not str(mapping.get(i, '')).isdigit() for i in range(query.GetNumAtoms()))
            or any(b.GetBondTypeAsDouble() != 1 for b in query.GetBonds())):
        return False
    locants = tuple(int(mapping[i]) for i in range(query.GetNumAtoms()))
    if len(set(locants)) != len(locants):
        return False
    edges = tuple(sorted(tuple(sorted((locants[b.GetBeginAtomIdx()], locants[b.GetEndAtomIdx()])))
                         for b in query.GetBonds()))
    return _valid_numbered_graph(template_record_name(template), locants, edges)
