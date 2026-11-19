"""Typed oxygen-isotope name operations for functional-class acid components.

These helpers operate on a mapped component name and its original atoms. They
never infer oxygen roles from a completed whole-molecule name.
"""


_RETAINED_ACID_ROOTS = {"acetic acid", "formic acid"}


def oxygen_isotope_count_descriptor(isotopes) -> str:
    """Return the P-82 oxygen-count descriptor for a typed isotope sequence."""
    isotopes = tuple(int(mass) for mass in isotopes if mass)
    if not isotopes:
        return ""
    counts = {}
    for mass in isotopes:
        counts[mass] = counts.get(mass, 0) + 1
    return ",".join(f"{mass}O{count}" for mass, count in sorted(counts.items()))


def suffix_oxygen_descriptors(name: str, isotopes) -> str:
    """Place typed oxygen counts in systematic or proven retained acid names.

    Systematic ``...oic acid`` names follow the P-82.6.4 insertion shown in
    the ester examples. The exact retained roots are supported by retained
    acid vocabulary and P-82.2.4's descriptor-before-word rule. For a retained
    root, prefixing the complete acid component also lets the functional-class
    anion conversion preserve that descriptor. Other retained or decorated
    roots are conservatively declined until their insertion site is proven.
    A string ending in ``oic acid`` (including the pre-existing benzoic-acid
    behavior) continues through the established systematic-ending operation.
    """
    descriptor = oxygen_isotope_count_descriptor(isotopes)
    if not descriptor:
        return ""
    if name.endswith("oic acid"):
        ending = "oic acid"
        return name[:-len(ending)] + f"({descriptor})" + ending
    if name in _RETAINED_ACID_ROOTS:
        return f"({descriptor}){name}"
    return ""


def organyl_oxygen_qualifier(bridge_isotope: int, carbonyl_isotope: int) -> str:
    """Return the P-82.6.4 element qualifier that disambiguates suffix O labels."""
    if bridge_isotope:
        return f"{bridge_isotope}O-"
    if carbonyl_isotope:
        return "O-"
    return ""


def apply_organyl_oxygen_qualifier(name: str, qualifier: str) -> str:
    """Place a typed O or nuclide-O qualifier before its organyl component."""
    if not qualifier:
        return name
    if not name:
        return ""
    if name.startswith('('):
        return ''  # Typed composite organyl descriptor grammar is pending.
    return qualifier + name
