"""P-68.4.2.2 hydride/suffix construction for bounded heterogeneous chains.

A terminal pair of identical chalcogens is kept intact as a peroxol suffix.
The preceding maximal homogeneous block is the parent hydride, also kept
intact; its complete upstream substituent is rendered on original atom IDs.
Other chain grammars, annotations and competing functions are not inferred.
"""

from .constants import MULT
from .heterochalcogen_chain_methods import _ordered_path, _organyl_component
from .molecule_iteration import iter_atoms, iter_bonds
from .naming_methods import MethodDecision


RULE = "P-68.4.2.2/P-63.4/P-63.7/P-21.2.2/P-16.7"
_HYDRIDES = {8: "oxidane", 16: "sulfane", 34: "selane", 52: "tellane"}
_SUFFIXES = {8: "peroxol", 16: "dithioperoxol", 34: "diselenoperoxol", 52: "ditelluroperoxol"}


def chalcogen_chain_suffix_decision(mol, Chem):
    """Prove one ordinary R-(chalcogen)n-H construction, or decline."""
    if (mol.GetNumHeavyAtoms() > 128 or mol.GetStereoGroups()
            or any(a.GetAtomicNum() not in {6, *_HYDRIDES}
                   or a.GetFormalCharge() or a.GetNumRadicalElectrons()
                   or a.GetIsotope() or int(a.GetChiralTag()) for a in iter_atoms(mol))
            or any(b.GetStereo() != Chem.BondStereo.STEREONONE for b in iter_bonds(mol))):
        return None
    chalcogens = {a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() in _HYDRIDES}
    carbons = {a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() == 6}
    if not carbons or len(chalcogens) < 3:
        return None
    chain = _ordered_path(mol, chalcogens)
    if len(chain) != len(chalcogens):
        return None
    if any(mol.GetAtomWithIdx(i).IsInRing()
           or mol.GetAtomWithIdx(i).GetTotalValence() != 2 for i in chain):
        return None
    if any(mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble() != 1
           for a, b in zip(chain, chain[1:])):
        return None
    links = [(i, n.GetIdx()) for i in chain
             for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetAtomicNum() == 6]
    if len(links) != 1 or links[0][0] not in {chain[0], chain[-1]}:
        return None
    attachment, carbon_root = links[0]
    if attachment == chain[-1]:
        chain = tuple(reversed(chain))
    if (mol.GetAtomWithIdx(chain[-1]).GetTotalNumHs(includeNeighbors=True) != 1
            or any(mol.GetAtomWithIdx(i).GetTotalNumHs(includeNeighbors=True) for i in chain[:-1])):
        return None
    organyl = _organyl_component(mol, carbon_root, chain[0], carbons)
    if organyl != frozenset(carbons):
        return None
    sequence = tuple(mol.GetAtomWithIdx(i).GetAtomicNum() for i in chain)
    if sequence[-1] != sequence[-2] or sequence[-3] == sequence[-1]:
        return None  # Never split a homogeneous terminal triple or longer run.
    start = len(chain) - 3
    while start and sequence[start - 1] == sequence[start]:
        start -= 1
    base = chain[start:-2]
    if not 1 <= len(base) <= 3 or start > 1:
        # More upstream chalcogens can supply a competing heterogeneous
        # parent hydride. That comparison is outside this bounded proof.
        return None
    prefix_root = chain[start - 1] if start else carbon_root
    prefix_atoms = frozenset(carbons | set(chain[:start]))
    from .component_mapping import mapped_branch, MappedComponent, decorate_component

    prefix, owned = mapped_branch(mol, Chem, prefix_root, base[0])
    if not prefix or frozenset(owned) != prefix_atoms:
        return None
    parent = (MULT[len(base)] if len(base) > 1 else "") + _HYDRIDES[sequence[start]]
    name = parent + _SUFFIXES[sequence[-1]]
    # Suffix numbering anchors the terminal end. Locants are intrinsic in
    # this two-ended construction; the map still preserves every parent atom.
    mapping = tuple((idx, str(n)) for n, idx in enumerate(reversed(base), 1))
    rendered = decorate_component(mol, Chem, MappedComponent(
        name, mapping, frozenset(base), prefixes=(("", prefix),),
        embedded_atoms=frozenset(set(chain[-2:]) | set(prefix_atoms)),
        systematic_parent=True, prefix_mode="unlocanted",
    ))
    if not rendered:
        return None
    roles = (("chalcogen-parent", frozenset(base), mapping),
             ("peroxol-suffix", frozenset(chain[-2:]), ()),
             ("upstream-substituent", prefix_atoms, ()))
    owned_ids = [i for _role, atoms, _locants in roles for i in atoms]
    heavy = frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1)
    if len(owned_ids) != len(set(owned_ids)) or frozenset(owned_ids) != heavy:
        return None
    return MethodDecision(rendered, "substitutive", frozenset({chain[-2]}), heavy,
                          RULE, mapping, component_roles=roles)
