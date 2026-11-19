"""Bounded sulfur suffix vocabulary and atom-site checks.

The module deliberately covers only sulfur groups whose published suffix and
local atom mapping are established by Blue Book P-63.1/P-65.3/P-66.1.1.2.
"""

SULFUR_SUFFIXES = {
    "sulfonic-acid": "sulfonic acid",
    "sulfinic-acid": "sulfinic acid",
    "sulfonate": "sulfonate",
    "sulfinate": "sulfinate",
    "sulfonamide": "sulfonamide",
    "thiol": "thiol",
}

SULFUR_RULES = {
    'sulfonic-acid': ('P-41/P-42.1', 'P-65.3.1'),
    'sulfinic-acid': ('P-41/P-42.1', 'P-65.3.1'),
    'sulfonamide': ('P-41-amide-corresponding-acid', 'P-66.1.1.2', 'P-66.1.1.3.1.1'),
    'thiol': ('P-41-hydroxy-chalcogen-seniority', 'P-63.1.5'),
}


def sulfur_suffix(kind):
    """Return the bounded suffix spelling, or an empty string if unproved."""
    return SULFUR_SUFFIXES.get(kind, "")


def sulfonamide_n_sites(mol, group):
    """Return validated non-sulfonyl neighbours of a sulfonamide N.

    The group convention is ``center=S``, ``owner=the single carbon host`` and
    ``ligand=N``. The returned roots may be ordinary carbon substituents or a
    terminal, source-shaped N-oxo group attached directly to the sulfonamide N.
    The latter is validated by the exact P-67.1.2.6.3 inventory motif and is
    mapped as nitroso/nitro by ``component_mapping``. ``None`` means that
    N-substitution is outside this bounded grammar.
    """
    if getattr(group, "kind", None) != "sulfonamide" or group.ligand < 0:
        return None
    nitrogen = mol.GetAtomWithIdx(group.ligand)
    if nitrogen.GetAtomicNum() != 7 or nitrogen.GetFormalCharge():
        return None
    heavy = []
    for bond in nitrogen.GetBonds():
        if bond.GetBondTypeAsDouble() != 1:
            return None
        other = bond.GetOtherAtom(nitrogen)
        if other.GetIdx() == group.center:
            continue
        if other.GetAtomicNum() == 6:
            heavy.append(other.GetIdx())
        elif other.GetAtomicNum() == 7:
            # P-61.5.2 permits nitro/nitroso attachment to the higher-class
            # sulfonamide nitrogen; validate the exact original N–NOx unit
            # before allowing the mapped branch renderer to see this root.
            from .branch_rules import collect_subtree
            from .nitrogen_oxo_amide_prefixes import direct_nitrogen_oxo_substituent

            root = other.GetIdx()
            atoms = collect_subtree(mol, root, nitrogen.GetIdx())
            typed = direct_nitrogen_oxo_substituent(
                mol, root, nitrogen.GetIdx(), atoms,
            )
            if typed is None or typed.atoms != frozenset(atoms):
                return None
            heavy.append(root)
        else:
            return None
    if sum(n.GetIdx() == group.center for n in nitrogen.GetNeighbors()) != 1:
        return None
    if len(heavy) > 2 or nitrogen.GetDegree() > 3:
        return None
    return tuple(heavy)


def sulfur_suffix_locants(
    *, count, locants, cyclic, path_length, prefix_count=0,
    parent_prefix_count=None, force_locants=False
):
    """Format only the locant part for bounded sulfur suffix parents.

    ``prefix_count`` records all prefixes in the composed component for older
    callers. ``parent_prefix_count`` identifies only substituents on the
    parent hydride; prefixes attached to the suffix nitrogen do not make the
    parent multisubstituted. A single isolated carbocyclic suffix has its site
    set to locant 1 by numbering and omits it. Acyclic locant 1 is omitted for
    methane, or for monosubstituted ethane only; other acyclic sites are shown.
    Mapped isotope evidence overrides these omissions.
    """
    if parent_prefix_count is None:
        parent_prefix_count = prefix_count
    if cyclic:
        return "" if not force_locants and count == 1 and parent_prefix_count == 0 and tuple(locants) == (1,) else (
            "-" + ",".join(map(str, locants)) + "-"
        )
    omit_acyclic = (
        count == 1
        and len(locants) == 1
        and locants[0] == 1
        and (path_length == 1 or (path_length == 2 and parent_prefix_count == 0))
    )
    return "" if omit_acyclic and not force_locants else (
        "-" + ",".join(map(str, locants)) + "-"
    )
