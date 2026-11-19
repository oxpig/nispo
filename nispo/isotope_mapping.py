"""Scoped isotope annotations from an explicit parent/branch atom map."""

from collections import defaultdict
import re

from .molecule_iteration import iter_atoms
from .constants import ROOTS


def locanted_isotope_prefix(mol, locants):
    labels = defaultdict(list)
    for idx, locant in locants:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetIsotope() and str(locant).isdigit():
            labels[(atom.GetSymbol(), atom.GetIsotope())].append(int(locant))
    parts = []
    for (symbol, mass), positions in sorted(labels.items()):
        positions = sorted(set(positions))
        count = str(len(positions)) if len(positions) > 1 else ""
        parts.append(f"{','.join(map(str, positions))}-{mass}{symbol}{count}")
    return '(' + ','.join(parts) + ')' if parts else ''


def apply_parent_isotopes(mol, name, locants):
    """Annotate this parent, independently of descriptors on its branches.

    The acyclic renderer supplies the original parent atom map and leaves
    parent isotope writing to this function. Its branch writers own disjoint
    atoms. A nuclide token in a branch is therefore neither evidence for nor
    a reason to suppress a descriptor on the parent (P-82.2.1).
    """
    mapping = dict(locants)
    labelled = [a for a in iter_atoms(mol)
                if a.GetIsotope() and a.GetIdx() in mapping]
    if not labelled or not all(str(mapping[a.GetIdx()]).isdigit() for a in labelled):
        return name
    # Use the same original-atom descriptor plan as mapped components. In
    # particular, a completely labelled parent still needs a descriptor;
    # P-82.6 omits its locants, not the descriptor itself. Branch atoms stay
    # outside this map even when they contain the same nuclide.
    from rdkit import Chem
    from .component_mapping import MappedComponent, _mapped_isotope_descriptor_plan
    component = MappedComponent('', tuple(mapping.items()), frozenset(mapping))
    plan = _mapped_isotope_descriptor_plan(mol, Chem, component, mapping)
    if plan is None:
        return name
    prefix = '(' + ','.join(plan[0]) + ')' if plan[0] else ''
    if not prefix:
        return name
    size = max((int(loc) for _, loc in locants if str(loc).isdigit()), default=0)
    stem = ROOTS.get(size, "")
    if not stem:
        return name
    depth = 0
    positions = []
    for index, char in enumerate(name):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif depth == 0 and name.startswith(stem, index):
            positions.append(index)
    if not positions:
        return name
    index = positions[-1]
    # Skeletal replacement prefixes are part of the parent hydride, so the
    # isotope descriptor precedes the whole replacement construction (P-82.2).
    replacement = re.search(
        r"(?:\d+(?:,\d+)*-(?:di|tri|tetra|penta|hexa)?"
        r"(?:aza|oxa|thia|phospha|sila|bora))+$", name[:index]
    )
    if replacement:
        index = replacement.start()
        before = name[:index]
        if before.endswith("-") and not before.endswith(")-"):
            before = before[:-1]
        return before + prefix + "-" + name[index:]
    return name[:index] + prefix + name[index:]


def root_carbon_isotope_is_mapped(name, mass):
    """Avoid double annotation only with explicit locant-1 isotope evidence."""
    if not name.startswith("(") or ")" not in name:
        return False
    annotation = name[1:name.index(")")]
    match = re.search(rf"(?:^|,)((?:\d+,)*\d+)-{mass}C\d*(?=,|$)", annotation)
    return bool(match and "1" in match.group(1).split(","))


def preserves_heavy_isotope_information(mol, name):
    """Return whether each specified non-hydrogen nuclide occurs in ``name``.

    This is a deliberately narrow conservation check. It does not establish
    isotope counts, atom roles, or locants; it only prevents finalization from
    silently dropping a specified heavy-atom nuclide. Hydrogen-isotope naming
    has separate aliases and is outside this check.
    """
    if not name:
        return False
    required_nuclides = {
        (atom.GetIsotope(), atom.GetSymbol())
        for atom in iter_atoms(mol)
        if atom.GetAtomicNum() > 1 and atom.GetIsotope()
    }
    return all(
        re.search(rf"(?<!\d){mass}{re.escape(symbol)}(?![a-z])", name.translate(
            str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
        )) is not None
        for mass, symbol in required_nuclides
    )


def preserves_isotope_information(mol, name):
    """Exclude a finalized name that drops an explicitly specified nuclide.

    This remains a minimum presence check, not an atom/locant/count proof.
    Typed component writers supply those stronger checks. Hydrogen isotope
    aliases emitted by the existing branch/fragment writers remain valid
    witnesses; ordinary indicated hydrogen alone is not a nuclide descriptor.
    """
    if not preserves_heavy_isotope_information(mol, name):
        return False
    hydrogen_masses = {atom.GetIsotope() for atom in iter_atoms(mol)
                       if atom.GetAtomicNum() == 1 and atom.GetIsotope()}
    if not hydrogen_masses:
        return True
    normalized = name.translate(str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹', '0123456789'))
    # Parenthesized superscript locants are structural delimiters inside an
    # isotope descriptor, e.g. (N^(1)-2H1). Flatten only that numeric notation
    # for this presence check; the emitted name and atom bindings stay intact.
    normalized = re.sub(r"\^\((\d+'*(?:\^\d+)?)\)", r"^\1", normalized)
    aliases = {1: 'proti', 2: 'deuter', 3: 'triti'}
    for mass in hydrogen_masses:
        alias = aliases.get(mass)
        if alias and alias in normalized.lower():
            continue
        # NISPO nuclide descriptors are enclosed and can include locants or
        # other nuclides. Do not mistake a bare 2H/3H parent locant for a label.
        if re.search(rf'(?<!\d)\([^()]*?(?<!\d){mass}H\d*(?![a-z])[^()]*\)', normalized) is None:
            return False
    return True


def preserves_oxygen_isotope_masses(mol, name):
    """Compatibility wrapper for the oxygen-focused conservation predicate."""
    required = {
        atom.GetIsotope()
        for atom in iter_atoms(mol)
        if atom.GetAtomicNum() == 8 and atom.GetIsotope()
    }
    if not name:
        return False
    normalized = name.translate(str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789"))
    return all(re.search(rf"(?<!\d){mass}O(?![a-z])", normalized) is not None for mass in required)
