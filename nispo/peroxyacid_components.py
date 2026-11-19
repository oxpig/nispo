"""Mapped, bounded P-65.1.4 peroxycarboxylic-acid construction.

The input group is C(=O)-O-OH (the carboxylic-acid OH replaced by OO).
This is distinct from the carbon-bound C-O-O-H peroxol handled by
``peroxol_components`` and from carbonic-acid analogues in P-65.2.
"""
from dataclasses import dataclass

from .molecule_iteration import iter_atoms
from .acid_derivative_methods import _acid_component
from .naming_methods import MethodDecision

RULES = ("P-41", "P-43.0/Table 4.3", "P-44", "P-65.1.2", "P-65.1.4.1")


@dataclass(frozen=True, slots=True)
class PeroxyAcidSite:
    carbonyl_carbon: int
    carbonyl_oxygen: int
    proximal_oxygen: int
    terminal_oxygen: int
    terminal_hydrogens: tuple[int, ...]

    @property
    def atoms(self):
        return frozenset((self.carbonyl_carbon, self.carbonyl_oxygen,
                          self.proximal_oxygen, self.terminal_oxygen,
                          *self.terminal_hydrogens))


def _single(mol, left, right):
    bond = mol.GetBondBetweenAtoms(left, right)
    return bond is not None and bond.GetBondTypeAsDouble() == 1


def recognize_peroxyacids(mol):
    """Recognize neutral unlabelled carboxylic –C(=O)–O–OH groups.

    Carbonic-acid analogues (the acyl carbon has an additional O single
    neighbour), charged/radical forms, and O-isotope-labelled sites are not
    admitted. Parent selection and final names are handled separately.
    """
    sites = []
    for carbon in iter_atoms(mol):
        c = carbon.GetIdx()
        if carbon.GetAtomicNum() != 6 or carbon.GetFormalCharge() or carbon.GetNumRadicalElectrons():
            continue
        double_oxygen = [a for a in carbon.GetNeighbors()
                         if a.GetAtomicNum() == 8 and
                         mol.GetBondBetweenAtoms(c, a.GetIdx()).GetBondTypeAsDouble() == 2]
        if len(double_oxygen) != 1:
            continue
        carbonyl = double_oxygen[0]
        if carbonyl.GetIsotope() or carbonyl.GetFormalCharge() or carbonyl.GetNumRadicalElectrons():
            continue
        single_neighbors = [a for a in carbon.GetNeighbors()
                            if _single(mol, c, a.GetIdx())]
        bridges = []
        for atom in single_neighbors:
            if atom.GetAtomicNum() != 8 or atom.GetFormalCharge() or atom.GetNumRadicalElectrons():
                continue
            if atom.GetIsotope() or atom.IsInRing() or atom.GetDegree() != 2:
                continue
            distal = [n for n in atom.GetNeighbors() if n.GetIdx() != c]
            if len(distal) != 1 or distal[0].GetAtomicNum() != 8:
                continue
            terminal = distal[0]
            if (terminal.GetIsotope() or terminal.GetFormalCharge() or terminal.GetNumRadicalElectrons()
                    or terminal.IsInRing() or terminal.GetDegree() != 1
                    or not _single(mol, atom.GetIdx(), terminal.GetIdx())):
                continue
            hydrogens = tuple(n.GetIdx() for n in terminal.GetNeighbors()
                              if n.GetAtomicNum() == 1)
            if terminal.GetTotalNumHs(includeNeighbors=True) != 1 or len(hydrogens) > 1:
                continue
            bridges.append((atom, terminal, hydrogens))
        if len(bridges) != 1:
            continue
        bridge, terminal, hydrogens = bridges[0]
        # P-65.1.4 concerns a carboxylic-acid skeletal parent. A third
        # single-bonded O at the acyl carbon is a carbonic-acid derivative
        # (P-65.2.1.2), not a carboxylic peroxy acid.
        other = [a for a in single_neighbors if a.GetIdx() != bridge.GetIdx()]
        if any(a.GetAtomicNum() == 8 for a in other):
            continue
        if any(a.GetAtomicNum() not in {1, 6} for a in other):
            continue
        sites.append(PeroxyAcidSite(c, carbonyl.GetIdx(), bridge.GetIdx(),
                                    terminal.GetIdx(), hydrogens))
    return tuple(sites)


def _systematic_peroxo_parent(acid_parent):
    """Apply the P-65.1.4.1 ending to supported mapped acid parent names."""
    if acid_parent.endswith("carboxylic acid"):
        return acid_parent[:-len("carboxylic acid")] + "carboperoxoic acid"
    if acid_parent == "benzoic acid":
        return "benzenecarboperoxoic acid"
    if acid_parent == "acetic acid":
        return "ethaneperoxoic acid"
    if acid_parent == "formic acid":
        return "methaneperoxoic acid"
    if acid_parent.endswith("oic acid"):
        stem = acid_parent[:-len("oic acid")]
        # P-15.1/P-16.7: retain terminal e before consonant-initial peroxoic.
        return stem + "eperoxoic acid"
    return ""


def peroxyacid_decision(mol, Chem):
    """Return a fully mapped simple peroxyacid suffix decision or ``None``.

    The bounded route handles one unlabelled P-65.1.4 site as the sole
    principal characteristic group. It relies on the existing capped acid
    component selector for parent choice and carbon isotope/stereo decoration.
    More than one acid site, another characteristic group, or an unsupported
    retained acid parent is left to other routes / future work.
    """
    if (mol.GetStereoGroups() or any(a.GetFormalCharge() or a.GetNumRadicalElectrons()
                                    for a in iter_atoms(mol))):
        return None
    sites = recognize_peroxyacids(mol)
    if len(sites) != 1:
        return None
    site = sites[0]
    if any(mol.GetAtomWithIdx(i).GetIsotope()
           for i in (site.proximal_oxygen, site.terminal_oxygen, *site.terminal_hydrogens)):
        return None
    part = _acid_component(mol, Chem, site.carbonyl_carbon, site.proximal_oxygen)
    if part is None or part.construction is None:
        return None
    suffix_parent = _systematic_peroxo_parent(part.construction.parent)
    if not suffix_parent:
        return None
    # A principal peroxyacid suffix is only proved here for otherwise
    # single-class molecules. P-43/P-41 prefix-mode routing is future work.
    from .functional_inventory import functional_inventory
    inventory = functional_inventory(mol)
    if inventory.principal:
        return None
    all_heavy = frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1)
    if all_heavy != part.atoms | {site.proximal_oxygen, site.terminal_oxygen}:
        return None
    from .component_mapping import MappedComponent, decorate_component
    base = part.construction
    mapped = MappedComponent(
        suffix_parent, base.atom_locants, base.atoms, base.prefixes,
        base.embedded_atoms, base.hydro_prefix, base.systematic_parent,
        base.nitrogen_site_locants, base.isotope_unlocanted_elements,
        base.prefix_mode, base.local_isotope_atoms,
        base.omit_methyl_prefix_locant, base.isotope_descriptor_parts)
    # P-65.1.4 explicitly forms the preferred names by functional replacement
    # of systematic acid names, so prefix-parent boundaries use the systematic
    # joining rule even when the capped acid helper used a retained spelling.
    mapped = MappedComponent(
        mapped.parent, mapped.atom_locants, mapped.atoms, mapped.prefixes,
        mapped.embedded_atoms, mapped.hydro_prefix, True,
        mapped.nitrogen_site_locants, mapped.isotope_unlocanted_elements,
        mapped.prefix_mode, mapped.local_isotope_atoms,
        mapped.omit_methyl_prefix_locant, mapped.isotope_descriptor_parts)
    name = decorate_component(mol, Chem, mapped)
    if not name:
        return None
    locants = dict(part.atom_locants)
    attachment = next((n.GetIdx() for n in mol.GetAtomWithIdx(site.carbonyl_carbon).GetNeighbors()
                       if n.GetIdx() in locants and n.GetAtomicNum() == 6), None)
    # For an acyclic acid the carbonyl carbon itself is a parent atom; for a
    # ring or aromatic carboxy substituent, use its mapped ring attachment.
    suffix_locant = locants.get(site.carbonyl_carbon,
                                locants.get(attachment) if attachment is not None else None)
    if suffix_locant is None:
        return None
    centers = frozenset({site.carbonyl_carbon})
    acid_role_atoms = part.atoms
    peroxy_role_atoms = frozenset({site.proximal_oxygen, site.terminal_oxygen})
    acid_role_locants = tuple(part.atom_locants)
    if (acid_role_atoms & peroxy_role_atoms
            or acid_role_atoms | peroxy_role_atoms != all_heavy
            or not {idx for idx, _ in acid_role_locants} <= acid_role_atoms):
        return None
    return MethodDecision(
        name, "substitutive", centers, all_heavy,
        "P-65.1.4.1/P-44/P-43.0", part.atom_locants,
        component_roles=(("acid_parent", acid_role_atoms, acid_role_locants),
                         ("peroxy_replacement", peroxy_role_atoms, ())),
    )
