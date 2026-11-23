"""Enclosure for repeated ions whose names have a competing composition."""


# An ordinary multiplier can instead name a single condensed oxoanion.
# P-65.6.2/P-77 cite each ion as a component. IR-5.4.2.1 uses the enclosed
# multiplier when the component has its own modifier or is ambiguous.
_CONDENSABLE_OXOANIONS = frozenset({
    'borate', 'carbonate', 'chromate', 'phosphate', 'silicate',
    'sulfate', 'sulfite', 'thiosulfate',
})


def needs_enclosed_ion_multiplier(name: str) -> bool:
    """Resolve ambiguity in the citation of an already identified ion."""
    return (name in _CONDENSABLE_OXOANIONS
            or name.startswith(('hydrogen ', 'dihydrogen ', 'trihydrogen '))
            or name.startswith(('tetrafluoro', 'hexafluoro'))
            or name.endswith(('phosphinate', 'phosphonate')))
