"""Spell out a single hydroxyl hydrogen isotope for the OPSIN 2.9 oxime grammar."""
import re


_ISOTOPE = r'\((?P<mass>[23])H(?P<oxygen>,[1-9]\d*O)?\)'
_N_HYDROXY = re.compile(r'(?P<site>\bN(?:\^\(\d+\))?-(?:di\[)?)' + _ISOTOPE + r'(?=hydroxy)')
_HYDROXYIMINO = re.compile(r'(?P<site>\[)' + _ISOTOPE + r'(?=hydroxy\]imino)')


def explicit_oxime_hydrogen_count(name):
    """Return an alternative parser input only for an implicit count on an oxime OH."""
    replacement = r'\g<site>(\g<mass>H1\g<oxygen>)'
    expanded = _HYDROXYIMINO.sub(replacement, name)
    if name.endswith('imine'):
        expanded = _N_HYDROXY.sub(replacement, expanded)
    return expanded if expanded != name else None
