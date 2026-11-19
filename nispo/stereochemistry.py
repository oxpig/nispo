from __future__ import annotations

import re

from .local_cip import assigned_chiral_centers
from .molecule_iteration import iter_atoms, iter_bonds
from .molecule_cache import mol_cache_dict, mol_cache_get, mol_cache_set


_STEREO_LOCANT_TEXT = r"\d+(?:[a-z])?'*"
_STEREO_ATOM_LOCANT_RE = re.compile(rf"^{_STEREO_LOCANT_TEXT}$")
_STEREO_MULTIPLIERS = {
    "di": 2, "tri": 3, "tetra": 4, "penta": 5, "hexa": 6,
    "bis": 2,
    "tris": 3,
    "tetrakis": 4,
    "pentakis": 5,
    "hexakis": 6,
}
_ENCLOSED_STEREO_MULTIPLIER_RE = re.compile(
    r"\b(di|tri|tetra|penta|hexa|bis|tris|tetrakis|pentakis|hexakis)(?=[([{])"
)


def relative_monocycle_configuration(mol, Chem, sequence) -> str:
    """Read the face relation of exactly two ligands on a saturated cycle.

    Native tetrahedral tags refer to each atom's stored neighbor order, not
    to a common ring plane. Normalize their permutation to the same oriented
    cycle before comparing faces. No conformer or guessed CIP label is used.
    """
    sequence = tuple(sequence)
    core = frozenset(sequence)
    if len(core) != len(sequence) or len(core) < 3:
        return ""
    for previous, current in zip(sequence, sequence[1:] + sequence[:1]):
        bond = mol.GetBondBetweenAtoms(previous, current)
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return ""
    substituted = []
    for position, idx in enumerate(sequence):
        atom = mol.GetAtomWithIdx(idx)
        inside = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() in core]
        outside = [n.GetIdx() for n in atom.GetNeighbors()
                   if n.GetIdx() not in core and n.GetAtomicNum() > 1]
        if len(inside) != 2:
            return ""
        if outside:
            # Identical geminal terminal groups carry no face assignment.
            # They do not prevent the other two singly substituted positions
            # from defining the cycle's cis/trans relation.
            if (atom.GetAtomicNum() == 6
                    and atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED
                    and len(outside) == 2
                    and all(mol.GetAtomWithIdx(i).GetDegree() == 1 for i in outside)):
                ligands = [mol.GetAtomWithIdx(i) for i in outside]
                signatures = [(a.GetAtomicNum(), a.GetIsotope(), a.GetFormalCharge(),
                               a.GetNumRadicalElectrons(), a.GetTotalNumHs()) for a in ligands]
                if signatures[0] == signatures[1]:
                    continue
            if (len(outside) != 1 or atom.GetAtomicNum() != 6
                    or atom.GetTotalNumHs(includeNeighbors=True) != 1
                    or atom.GetChiralTag() not in {
                        Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                        Chem.ChiralType.CHI_TETRAHEDRAL_CCW}):
                return ""
            stored = [n.GetIdx() for n in atom.GetNeighbors()]
            hydrogens = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
            if len(stored) == 3 and not hydrogens:
                stored.append(-1)  # The implicit H follows native neighbors.
                hydrogens = [-1]
            if len(stored) != 4 or len(hydrogens) != 1:
                return ""
            ordered = [sequence[position - 1], sequence[(position + 1) % len(sequence)],
                       outside[0], hydrogens[0]]
            permutation = [stored.index(i) for i in ordered]
            inversions = sum(a > b for j, a in enumerate(permutation)
                             for b in permutation[j + 1:])
            orientation = int(atom.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CW)
            substituted.append(orientation ^ (inversions % 2))
    if len(substituted) != 2:
        return ""
    return "cis" if substituted[0] == substituted[1] else "trans"


def opposite_cyclohexyl_configuration(mol, Chem, sequence):
    """Return the face relation of an isolated 1,4-disubstituted cyclohexyl."""
    if len(sequence) != 6 or not all(mol.GetAtomWithIdx(sequence[i]).GetChiralTag() for i in (0, 3)):
        return ""
    ring_info = mol.GetRingInfo()
    for position, idx in enumerate(sequence):
        atom = mol.GetAtomWithIdx(idx)
        if (atom.GetAtomicNum() != 6 or atom.GetFormalCharge() or atom.GetIsotope()
                or atom.GetNumRadicalElectrons() or ring_info.NumAtomRings(idx) != 1
                or atom.GetTotalNumHs(includeNeighbors=True) != (1 if position in (0, 3) else 2)):
            return ""
    return relative_monocycle_configuration(mol, Chem, sequence)


def apply_stereochemical_prefix(
    mol,
    Chem,
    name: str,
    *,
    atom_order: str = "idx_desc",
    invert_unlocanted_bond_stereo: bool = False,
    include_atom_stereo: bool = True,
) -> str:
    """Complete ordinary configuration after separately named branches.

    Mapped parent writers supply locanted descriptors through the companion
    function. This compatibility fallback preserves their explicit descriptors
    and emitted branch ownership before adding remaining unlocanted R/S/E/Z.
    Retained atom configuration can be left to its writer independently of
    double-bond geometry by disabling the atom fallback.
    """

    if not name:
        return name
    if not _has_assigned_stereo(mol, Chem):
        return name

    retained_atoms = _rendered_retained_aminoacyl_stereo(mol, name)
    owned_junctions = _rendered_ylidene_junctions(mol, name)
    owned_branch_bonds = _rendered_mapped_branch_bonds(mol, name)
    bond_descriptors = ([d for a,b,d in _indexed_bond_stereochemistry(mol, Chem)
                         if frozenset((a,b)) not in owned_junctions
                         and frozenset((a,b)) not in owned_branch_bonds]
                        if owned_junctions or owned_branch_bonds
                        else _bond_stereochemistry(mol, Chem))
    if invert_unlocanted_bond_stereo:
        bond_descriptors = _invert_bond_stereo_descriptors(bond_descriptors)
    locanted_bond_descriptors = _locanted_bond_stereo_descriptors_in_name(name)
    locanted_bond_descriptors += _leading_unlocanted_stereo_descriptors(name, "EZ")
    if owned_junctions:
        locanted_bond_descriptors = _remove_represented_atom_descriptors(
            locanted_bond_descriptors, [v[1] for v in owned_junctions.values()])
    if owned_branch_bonds:
        locanted_bond_descriptors = _remove_represented_atom_descriptors(
            locanted_bond_descriptors, list(owned_branch_bonds.values()))
    if locanted_bond_descriptors:
        bond_descriptors = _remove_represented_atom_descriptors(bond_descriptors, locanted_bond_descriptors)
    if not include_atom_stereo:
        return (_prepend_stereochemical_descriptors(name, bond_descriptors)
                if bond_descriptors else name)
    locanted_atom_descriptors = _locanted_atom_stereo_descriptors_in_name(name)
    relative_descriptor_count = len(re.findall(r"\((?:cis|trans)\)-", name))
    relative_descriptor_groups = _internal_relative_stereo_descriptor_groups(mol, Chem, relative_descriptor_count)
    relative_atom_descriptors = [
        descriptor for group in relative_descriptor_groups for descriptor in group
    ]
    internal_relative_descriptor_count = _internal_relative_stereo_descriptor_count(
        relative_descriptor_count,
        represented_relative_group_count=len(relative_descriptor_groups),
        represented_relative_descriptors=relative_atom_descriptors,
    )
    represented_atom_descriptors = (
        locanted_atom_descriptors
        + _internal_atom_stereo_descriptors_in_name(name)
        + _leading_unlocanted_stereo_descriptors(name, "RS")
        + relative_atom_descriptors
    )
    descriptors = []
    if include_atom_stereo and not _name_has_complete_atom_stereo(
        mol,
        Chem,
        represented_atom_descriptors,
        internal_relative_descriptor_count=internal_relative_descriptor_count,
        retained_atom_indices=frozenset(retained_atoms),
    ):
        descriptors = [descriptor for idx, descriptor in assigned_atom_stereochemistry(
            mol, Chem, atom_order=atom_order) if idx not in retained_atoms]
        if represented_atom_descriptors:
            descriptors = _remove_represented_atom_descriptors(descriptors, represented_atom_descriptors)
    descriptors.extend(bond_descriptors)
    if not descriptors:
        return name
    return _prepend_stereochemical_descriptors(name, descriptors)


def apply_locanted_stereochemical_prefix(
    mol,
    Chem,
    name: str,
    locants: tuple[tuple[int, str], ...],
    *,
    atom_order: str = "idx_desc",
    include_unlocanted: bool = True,
    invert_unlocanted_bond_stereo: bool = False,
    invert_locanted_bond_stereo: bool = False,
    reverse_descriptor_sequence: bool = False,
    reverse_atom_descriptor_sequence: bool = False,
) -> str:
    """Prepend a locanted stereo prefix when a candidate provides atom locants.

    Mapped parent stereocenters are locanted. Unmapped centers/bonds keep the
    existing unlocanted fallback descriptors, after removing descriptors already
    represented inside the rendered name.
    """

    if not name:
        return name
    if not locants or not _has_assigned_stereo(mol, Chem):
        return name
    locant_by_atom = _stereo_locant_map(locants)
    if not locant_by_atom:
        return name
    relative_descriptor, relative_atoms = _mapped_relative_monocycle_stereo(
        mol, Chem, locant_by_atom, name)

    atom_pieces, unlocanted_atom_descriptors = _locanted_atom_stereo_pieces(
        mol,
        Chem,
        locant_by_atom,
        name,
        atom_order=atom_order,
    )
    if relative_atoms:
        relative_texts = {locant_by_atom[idx] + descriptor
                          for idx, descriptor in assigned_atom_stereochemistry(
                              mol, Chem, include_pseudo=True) if idx in relative_atoms}
        atom_pieces = [piece for piece in atom_pieces if piece[1] not in relative_texts]
    bond_pieces, unlocanted_bond_descriptors = _locanted_bond_stereo_pieces(
        _indexed_bond_stereochemistry(mol, Chem),
        locant_by_atom,
        name,
        mol=mol,
        Chem=Chem,
    )
    if invert_unlocanted_bond_stereo:
        unlocanted_bond_descriptors = _invert_bond_stereo_descriptors(unlocanted_bond_descriptors)
    if not atom_pieces and not bond_pieces and not relative_descriptor:
        return name
    locanted_atoms = [piece for _sort_key, piece in sorted(atom_pieces)]
    locanted_bonds = [piece for _sort_key, piece in sorted(bond_pieces)]
    if invert_locanted_bond_stereo:
        locanted_bonds = _invert_locanted_bond_stereo_texts(locanted_bonds)
    if reverse_atom_descriptor_sequence:
        locanted_atoms = list(reversed(locanted_atoms))
    descriptors = locanted_atoms + locanted_bonds
    if include_unlocanted:
        descriptors += unlocanted_atom_descriptors + unlocanted_bond_descriptors
    if reverse_descriptor_sequence:
        descriptors = list(reversed(descriptors))
    if descriptors:
        name = _prepend_stereochemical_descriptors(name, descriptors)
    return f"({relative_descriptor})-{name}" if relative_descriptor else name


def _mapped_relative_monocycle_stereo(mol, Chem, locant_by_atom, name):
    """A two-site saturated cycle can own its native relative configuration.

    Both assigned sites must be mapped pseudoasymmetric carbon atoms of one
    complete, isolated parent cycle. The cyclic neighbor permutation supplies
    their face relation; no CIP-label inversion or generated atom order is
    used to choose cis/trans.
    """
    from .relative_cycle_evidence import mapped_native_relative_cycle
    native = mapped_native_relative_cycle(mol, Chem, locant_by_atom, name)
    if native is not None:
        return native
    if re.match(r"^\((?:cis|trans)\)-", name):
        return "", frozenset()
    modern = dict(assigned_atom_stereochemistry(mol, Chem, include_pseudo=True))
    mapped = {idx for idx, descriptor in modern.items()
              if idx in locant_by_atom and descriptor in {"r", "s"}}
    if len(mapped) != 2:
        return "", frozenset()
    existing = _parent_stereo_texts_in_name(name, "RSrs")
    if any(locant_by_atom[idx] + modern[idx] in existing for idx in mapped):
        return "", frozenset()
    rings = tuple(mol.GetRingInfo().AtomRings())
    for sequence in rings:
        core = frozenset(sequence)
        if (not mapped <= core or not core <= locant_by_atom.keys()
                or any(core & frozenset(other) for other in rings if other != sequence)
                or any(mol.GetAtomWithIdx(idx).GetAtomicNum() != 6 for idx in core)
                or {idx for idx in modern if idx in core} != mapped):
            continue
        descriptor = relative_monocycle_configuration(mol, Chem, sequence)
        if descriptor:
            return descriptor, frozenset(mapped)
    return "", frozenset()


def _has_assigned_stereo(mol, Chem) -> bool:
    cached = mol_cache_get(mol, "has_assigned_stereo")
    if cached is not None:
        return cached
    for atom in iter_atoms(mol):
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            return mol_cache_set(mol, "has_assigned_stereo", True)
    result = _has_assigned_bond_stereo(mol, Chem)
    return mol_cache_set(mol, "has_assigned_stereo", result)


def _has_assigned_bond_stereo(mol, Chem) -> bool:
    cached = mol_cache_get(mol, "has_assigned_bond_stereo")
    if cached is not None:
        return cached
    for bond in iter_bonds(mol):
        if bond.GetStereo() != Chem.BondStereo.STEREONONE:
            return mol_cache_set(mol, "has_assigned_bond_stereo", True)
    return mol_cache_set(mol, "has_assigned_bond_stereo", False)


def _potential_stereochemistry(mol, Chem):
    """Reuse the stereo census across parent maps of one naming-owned graph.

    Like the assigned-stereo caches, this belongs to the molecule, never a
    global collection. Naming copies caller-owned molecules; modified
    recognition graphs receive their own caches.
    """
    cached = mol_cache_get(mol, "potential_stereochemistry")
    if cached is None:
        potential = Chem.FindPotentialStereo(mol)
        try:
            from rdkit.Chem.rdmolops import _vectN5RDKit9Chirality10StereoInfoE as native_vector
        except ImportError:
            native_vector = None
        # Empty native Boost vectors otherwise terminate tuple conversion by
        # translating an indexed-access exception. Perception still runs;
        # custom containers and nonempty native results keep normal iteration.
        observed = () if type(potential) is native_vector and not len(potential) else tuple(potential)
        cached = mol_cache_set(
            mol, "potential_stereochemistry", observed)
    return cached


def _has_leading_stereochemical_prefix(name: str) -> bool:
    return bool(_leading_stereochemical_descriptors(name)[0])


def _leading_stereochemical_descriptors(name: str) -> tuple[list[str], str]:
    """Read descriptors belonging to this name, excluding named branches.

    A branch may use the same numerical locant as the selected parent. Its
    descriptor never represents an atom of that parent merely because the
    rendered text happens to coincide.
    """
    descriptor = rf"(?:{_STEREO_LOCANT_TEXT})?[RSrsZEξΞ]"
    match = re.match(rf"^\((?P<descriptors>{descriptor}(?:,{descriptor})*)\)-", name)
    if match is None:
        return [], name
    return match.group("descriptors").split(","), name[match.end():]


def _leading_unlocanted_stereo_descriptors(name: str, kinds: str) -> list[str]:
    return [text for text in _leading_stereochemical_descriptors(name)[0]
            if len(text) == 1 and text in kinds]


def _parent_stereo_texts_in_name(name: str, kinds: str) -> set[str]:
    return {text for text in _leading_stereochemical_descriptors(name)[0]
            if len(text) > 1 and text[-1] in kinds}


def _prepend_stereochemical_descriptors(name: str, descriptors: list[str]) -> str:
    existing, body = _leading_stereochemical_descriptors(name)
    if not existing:
        return f"({','.join(descriptors)})-{body}"
    combined = existing + descriptors
    # Locanted units can be written in parent order. Preserve the sequence of
    # the compatibility fallback's unlocanted units, whose order has meaning.
    combined.sort(key=lambda text: _stereo_locant_sort_key(text[:-1])
                  if len(text) > 1 else (9999, 9999, 0, ""))
    return f"({','.join(combined)})-{body}"


def apply_unknown_bicyclic_pseudo_stereo(mol, Chem, name, locants, parent_atoms):
    """Describe one proved unspecified pseudoasymmetric bicyclic bridge atom.

    P-91.2.1.2.1 uses locanted lowercase xi for an unknown configuration
    that would otherwise be described by r or s.  The two provisional CIP
    calculations below are evidence for the *kind* of unit, not assignments
    to the input graph or claims about a sample's composition.  This bounded
    writer accepts only a completely mapped, saturated bicyclo[n.1.0]
    parent, and requires both labelled bridgeheads to be already rendered in
    the parent prefix.  Unsupported or ambiguous evidence leaves the name
    untouched.

    https://iupac.qmul.ac.uk/BlueBook/P9.html#P91.2.1.2.1
    """
    if not name or not locants or not parent_atoms:
        return name
    try:
        parent = frozenset(parent_atoms)
        if (len(parent) < 6 or len(parent) > 10
                or any(not isinstance(i, int) or i < 0 or i >= mol.GetNumAtoms()
                       for i in parent)
                or len(locants) != len(parent)
                or any(not isinstance(item, tuple) or len(item) != 2 for item in locants)):
            return name
        mapping = _stereo_locant_map(locants)
        # Plain numeric locants are supported by OPSIN's xi handler.  Keep
        # partial, repeated, lettered and primed maps outside this scope.
        if (set(mapping) != parent or len(mapping) != len(locants)
                or len(set(mapping.values())) != len(parent)
                or any(not re.fullmatch(r"[1-9][0-9]?", value)
                       for value in mapping.values())):
            return name
        if (len(Chem.GetMolFrags(mol)) != 1 or mol.GetStereoGroups()
                or any(atom.GetIsotope() or atom.GetFormalCharge()
                       or atom.GetNumRadicalElectrons() for atom in iter_atoms(mol))):
            return name

        # Exactly the two bridgeheads have degree three in the parent.  They
        # are joined directly and by a one-atom bridge.  This establishes the
        # [n.1.0] topology without relying on a name or on SSSR ring choice.
        neighbors = {idx: {n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                           if n.GetIdx() in parent} for idx in parent}
        heads = {idx for idx, adjacent in neighbors.items() if len(adjacent) == 3}
        if (len(heads) != 2 or any(len(v) != (3 if idx in heads else 2)
                                   for idx, v in neighbors.items())):
            return name
        head_a, head_b = tuple(heads)
        if head_b not in neighbors[head_a]:
            return name
        if any(mol.GetBondBetweenAtoms(i, j).GetBondType() != Chem.BondType.SINGLE
               for i in parent for j in neighbors[i] if i < j):
            return name
        seen = set()
        todo = [head_a]
        while todo:
            idx = todo.pop()
            if idx not in seen:
                seen.add(idx)
                todo.extend(neighbors[idx] - seen)
        if seen != parent:
            return name

        features = _potential_stereochemistry(mol, Chem)
        if not features or any(feature.type != Chem.StereoType.Atom_Tetrahedral
                               for feature in features):
            return name
        unknown = [feature for feature in features
                   if feature.specified != Chem.StereoSpecified.Specified]
        if (len(unknown) != 1
                or unknown[0].specified != Chem.StereoSpecified.Unspecified):
            return name
        idx = unknown[0].centeredOn
        if idx not in parent or neighbors[idx] != heads:
            return name
        atom = mol.GetAtomWithIdx(idx)
        if (atom.GetAtomicNum() != 6 or atom.GetDegree() != 3
                or atom.GetTotalNumHs(includeNeighbors=True) != 1
                or atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
                or any(n.GetAtomicNum() == 1 for n in atom.GetNeighbors())):
            return name
        if any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6
               or mol.GetAtomWithIdx(i).GetChiralTag() not in {
                   Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                   Chem.ChiralType.CHI_TETRAHEDRAL_CCW} for i in heads):
            return name

        source_labels = dict(assigned_atom_stereochemistry(mol, Chem, include_pseudo=True))
        specified = {feature.centeredOn for feature in features if feature is not unknown[0]}
        if set(source_labels) != specified or not heads <= specified:
            return name
        existing, _body = _leading_stereochemical_descriptors(name)
        expected = {mapping[i] + descriptor for i, descriptor in source_labels.items()
                    if i in parent}
        # Parent ownership is explicit.  Do not infer a locant by looking at a
        # similarly numbered descriptor in a separately named substituent.
        if len(existing) != len(expected) or set(existing) != expected:
            return name

        from rdkit.Chem import rdCIPLabeler
        descriptors = set()
        feature_keys = {(feature.type, feature.centeredOn) for feature in features}
        for tag in (Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                    Chem.ChiralType.CHI_TETRAHEDRAL_CCW):
            probe = Chem.Mol(mol)
            for item in (*probe.GetAtoms(), *probe.GetBonds()):
                if item.HasProp('_CIPCode'):
                    item.ClearProp('_CIPCode')
            for item in probe.GetAtoms():
                item.SetAtomMapNum(0)
            probe.GetAtomWithIdx(idx).SetChiralTag(tag)
            rdCIPLabeler.AssignCIPLabels(probe, maxRecursiveIterations=1_000_000)
            labels = {a.GetIdx(): a.GetProp('_CIPCode') for a in probe.GetAtoms()
                      if a.HasProp('_CIPCode')}
            if (any(labels.get(i) != descriptor for i, descriptor in source_labels.items())
                    or set(labels) != specified | {idx}):
                return name
            descriptors.add(labels[idx])
            probe_features = tuple(Chem.FindPotentialStereo(probe))
            if ({(feature.type, feature.centeredOn) for feature in probe_features} != feature_keys
                    or any(feature.specified != Chem.StereoSpecified.Specified
                           for feature in probe_features)):
                return name
        if descriptors != {'r', 's'}:
            return name
        return _prepend_stereochemical_descriptors(name, [mapping[idx] + 'ξ'])
    except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError):
        # Unknown or unsupported inputs are not evidence for a descriptor.
        return name


def _locanted_atom_stereo_descriptors_in_name(name: str) -> list[str]:
    descriptors = [
        descriptor[-1]
        for descriptor in re.findall(rf"(?<![A-Za-z]){_STEREO_LOCANT_TEXT}[RS](?![a-z])", name)
    ]
    if descriptors:
        descriptors.extend(_extra_multiplied_locanted_stereo_descriptors(name, "RS"))
    return descriptors


def _internal_atom_stereo_descriptors_in_name(name: str) -> list[str]:
    matches = list(re.finditer(
            r"(?<![A-Za-z])\((?P<descriptor>[RS])\)-[A-Za-z0-9'\[\]\(\),.-]*"
            r"(?:sulfinyl|sulfinimidoyl|sulfonimidoyl|sulfimidoyl)",
            name,
        ))
    # A descriptor inside an enclosing substituent belongs to that component,
    # including a single-carbon methine whose name has no numerical locant.
    matches.extend(re.finditer(r"(?<=[([{])\((?P<descriptor>[RS])\)-", name))
    return [descriptor for _, descriptor in sorted(
        {m.start(): m.group('descriptor') for m in matches}.items())]


def _internal_relative_stereo_descriptor_count(
    descriptor_count: int,
    *,
    represented_relative_group_count: int,
    represented_relative_descriptors: list[str],
) -> int:
    if descriptor_count <= 0:
        return 0
    if not represented_relative_descriptors:
        return 2 * descriptor_count
    return 2 * max(0, descriptor_count - represented_relative_group_count)


def _internal_relative_stereo_descriptor_groups(mol, Chem, descriptor_count: int) -> list[list[str]]:
    if descriptor_count <= 0:
        return []
    assigned = dict(assigned_atom_stereochemistry(mol, Chem))
    descriptor_groups = []
    for raw_ring in mol.GetRingInfo().AtomRings():
        tagged = _simple_relative_ring_branch_tagged_atoms(mol, Chem, frozenset(raw_ring))
        if tagged:
            descriptors = [assigned[atom_idx] for atom_idx in sorted(tagged) if atom_idx in assigned]
            if descriptors:
                descriptor_groups.append(descriptors)
    descriptor_groups.sort(key=lambda values: (-len(values), values))
    return descriptor_groups[:descriptor_count]


def _simple_relative_ring_branch_tagged_atoms(mol, Chem, ring: frozenset[int]) -> frozenset[int]:
    tagged = [
        atom
        for atom in iter_atoms(mol)
        if atom.GetIdx() in ring and atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
    ]
    if len(tagged) != 2:
        return frozenset()
    if any(mol.GetAtomWithIdx(atom_idx).GetSymbol() != "C" for atom_idx in ring):
        return frozenset()
    if any(atom.GetSymbol() != "C" or atom.GetIsAromatic() for atom in tagged):
        return frozenset()
    if any(_ring_external_heavy_degree(mol, atom.GetIdx(), ring) != 1 for atom in tagged):
        return frozenset()
    for bond in iter_bonds(mol):
        if (
            bond.GetBeginAtomIdx() in ring
            and bond.GetEndAtomIdx() in ring
            and bond.GetBondType() != Chem.BondType.SINGLE
        ):
            return frozenset()
    return frozenset(atom.GetIdx() for atom in tagged)


def _ring_external_heavy_degree(mol, atom_idx: int, ring_atoms: frozenset[int]) -> int:
    return sum(
        1
        for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
        if nbr.GetIdx() not in ring_atoms and nbr.GetAtomicNum() > 1
    )


def _locanted_bond_stereo_descriptors_in_name(name: str) -> list[str]:
    descriptors = [
        descriptor[-1]
        for descriptor in re.findall(rf"(?<![A-Za-z]){_STEREO_LOCANT_TEXT}[EZ](?![a-z])", name)
    ]
    if descriptors:
        descriptors.extend(_extra_multiplied_locanted_stereo_descriptors(name, "EZ"))
    # A single-carbon component has no numerical alkene locant. Its embedded
    # (E)/(Z) still represents the bond and must not be repeated globally.
    descriptors.extend(re.findall(r"(?<=[([{])\(([EZ])\)-", name))
    return descriptors


def _extra_multiplied_locanted_stereo_descriptors(name: str, kinds: str) -> list[str]:
    # A multiplier repeats the entire component, including descriptors deep
    # inside its brackets. Parentheses, square brackets and braces have the
    # same ownership semantics; nested multipliers multiply their counts.
    # Counting only a leading bis((1R)-...) misses bis{[(3R)-...]carbonyl}
    # and can invent another global descriptor for an already named center.
    # Locate each multiplier once. Most names have none; otherwise a single
    # lookup replaces a regex search over every growing name[:pos] slice.
    multipliers = {match.end(): _STEREO_MULTIPLIERS[match[1]]
                   for match in _ENCLOSED_STEREO_MULTIPLIER_RE.finditer(name)}
    if not multipliers:
        return []
    pairs = {')': '(', ']': '[', '}': '{'}
    stack, repeated = [], []
    for pos, char in enumerate(name):
        if char in '([{':
            stack.append((char, pos, multipliers.get(pos, 1)))
        elif char in pairs:
            if not stack or stack[-1][0] != pairs[char]:
                return []
            _, start, count = stack.pop()
            if count > 1:
                repeated.append((start, pos, count))
    if stack or not repeated:
        return []
    extras = []
    for match in re.finditer(rf'(?<![A-Za-z]){_STEREO_LOCANT_TEXT}[{kinds}](?![a-z])', name):
        count = 1
        for start, end, multiplier in repeated:
            if start < match.start() < end:
                count *= multiplier
        extras.extend([match[0][-1]] * (count - 1))
    return extras


def _locanted_atom_stereo_texts_in_name(name: str) -> set[str]:
    return set(re.findall(rf"(?<![A-Za-z]){_STEREO_LOCANT_TEXT}[RS](?![a-z])", name))


def _locanted_bond_stereo_texts_in_name(name: str) -> set[str]:
    return set(re.findall(rf"(?<![A-Za-z]){_STEREO_LOCANT_TEXT}[EZ](?![a-z])", name))


def _name_has_complete_atom_stereo(
    mol,
    Chem,
    represented_descriptors: list[str],
    *,
    internal_relative_descriptor_count: int = 0,
    retained_atom_indices: frozenset[int] = frozenset(),
) -> bool:
    represented_count = len(represented_descriptors) + internal_relative_descriptor_count
    if represented_count <= 0:
        return False
    chiral_count = sum(idx not in retained_atom_indices
                       for idx, _ in assigned_atom_stereochemistry(mol, Chem))
    return represented_count >= chiral_count


def _remove_represented_atom_descriptors(descriptors: list[str], represented_descriptors: list[str]) -> list[str]:
    remaining = list(descriptors)
    for represented in represented_descriptors:
        try:
            remaining.remove(represented)
        except ValueError:
            continue
    return remaining


def _invert_bond_stereo_descriptors(descriptors: list[str]) -> list[str]:
    return [{"E": "Z", "Z": "E"}.get(descriptor, descriptor) for descriptor in descriptors]


def _invert_locanted_bond_stereo_texts(descriptors: list[str]) -> list[str]:
    return [descriptor[:-1] + {"E": "Z", "Z": "E"}.get(descriptor[-1:], descriptor[-1:]) for descriptor in descriptors]


def _atom_stereochemistry(mol, Chem, *, atom_order: str) -> list[str]:
    return [
        descriptor
        for _idx, descriptor in assigned_atom_stereochemistry(
            mol,
            Chem,
            atom_order=atom_order,
        )
    ]


def assigned_atom_stereochemistry(mol, Chem, *, atom_order: str = "idx_desc", include_pseudo: bool = False) -> list[tuple[int, str]]:
    # Mapped renderers opt into complete modern CIP assignment. Legacy callers
    # retain their bounded R/S contract until their atom ownership is proved.
    mode = "modern" if include_pseudo else "raw"
    cache_key = f"assigned_atom_stereochemistry_{mode}_{atom_order}"
    cached = mol_cache_get(mol, cache_key)
    if cached is not None:
        return cached
    raw_key = f"assigned_atom_stereochemistry_{mode}"
    raw_cached = mol_cache_get(mol, raw_key)
    if raw_cached is None:
        raw_cached = (_assigned_atom_stereochemistry_modern(mol, Chem) if include_pseudo
                      else _assigned_atom_stereochemistry_raw(mol, Chem))
        mol_cache_set(mol, raw_key, raw_cached)
    reverse = atom_order != "idx"
    return mol_cache_set(mol, cache_key, sorted(raw_cached, reverse=reverse))


def _assigned_atom_stereochemistry_modern(mol, Chem) -> list[tuple[int, str]]:
    """P-92/P-93: label the whole original graph, including enantiomorphic arms.

    Never compute these labels on capped fragments or overwrite the accurate
    labels with the legacy assigner. The original molecule remains untouched.
    """
    if not any(int(a.GetChiralTag()) for a in iter_atoms(mol)):
        return []
    from rdkit.Chem import rdCIPLabeler
    graph = Chem.Mol(mol)
    for atom in iter_atoms(graph):
        atom.SetAtomMapNum(0)
    for obj in list(iter_atoms(graph)) + list(iter_bonds(graph)):
        if obj.HasProp("_CIPCode"):
            obj.ClearProp("_CIPCode")
    try:
        rdCIPLabeler.AssignCIPLabels(graph, maxRecursiveIterations=1_000_000)
    except (ValueError, RuntimeError):
        return []
    return [(a.GetIdx(), a.GetProp("_CIPCode")) for a in iter_atoms(graph)
            if a.HasProp("_CIPCode") and a.GetProp("_CIPCode") in {"R", "S", "r", "s"}]


def _assigned_atom_stereochemistry_raw(mol, Chem) -> list[tuple[int, str]]:
    has_chiral_tag = any(
        atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        for atom in iter_atoms(mol)
    )
    if not has_chiral_tag:
        return []
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    centers = assigned_chiral_centers(mol, Chem)
    assigned = [
        (idx, descriptor)
        for idx, descriptor in centers
        if descriptor in {"R", "S"}
    ]
    if not assigned:
        assigned = _atom_cip_stereochemistry_from_props(mol)
    return assigned


def _atom_cip_stereochemistry_from_props(mol) -> list[tuple[int, str]]:
    assigned = []
    for atom in iter_atoms(mol):
        if not atom.HasProp("_CIPCode"):
            continue
        descriptor = atom.GetProp("_CIPCode")
        if descriptor in {"R", "S"}:
            assigned.append((atom.GetIdx(), descriptor))
    return assigned


def _stereo_locant_map(locants: tuple[tuple[int, str], ...]) -> dict[int, str]:
    mapped: dict[int, str] = {}
    for item in locants:
        if not isinstance(item, tuple) or len(item) != 2:
            continue
        atom_idx, locant = item
        if not isinstance(atom_idx, int):
            continue
        text = str(locant)
        if not text or not _STEREO_ATOM_LOCANT_RE.match(text):
            continue
        mapped[atom_idx] = text
    return mapped


def _locanted_atom_stereo_pieces(
    mol,
    Chem,
    locant_by_atom: dict[int, str],
    name: str,
    *,
    atom_order: str,
) -> tuple[list[tuple[tuple, str]], list[str]]:
    existing = _parent_stereo_texts_in_name(name, "RSrs")
    retained_atoms = _rendered_retained_aminoacyl_stereo(mol, name)
    pieces = []
    unlocanted = []
    modern = dict(assigned_atom_stereochemistry(mol, Chem, atom_order=atom_order, include_pseudo=True))
    assigned = dict(assigned_atom_stereochemistry(mol, Chem, atom_order=atom_order))
    assigned.update((idx, descriptor) for idx, descriptor in modern.items() if idx in locant_by_atom)
    for atom_idx, descriptor in sorted(assigned.items(), reverse=atom_order != "idx"):
        if atom_idx in retained_atoms:
            # A retained acyl word owns this original alpha atom. A generic
            # chain map from an alternative spelling cannot locate that atom
            # after the chain has been compacted into the retained component.
            continue
        locant = locant_by_atom.get(atom_idx)
        if not locant:
            unlocanted.append(descriptor)
            continue
        text = f"{locant}{descriptor}"
        if text in existing:
            continue
        pieces.append((_stereo_locant_sort_key(locant), text))
    if unlocanted:
        represented = _locanted_atom_stereo_descriptors_in_name(name) + _internal_atom_stereo_descriptors_in_name(name)
        represented = _remove_represented_atom_descriptors(represented, [
            text[-1] for text in existing if text[:-1] in locant_by_atom.values()])
        represented += _leading_unlocanted_stereo_descriptors(name, "RS")
        unlocanted = _remove_represented_atom_descriptors(unlocanted, represented)
    return pieces, unlocanted


def _indexed_bond_stereochemistry(mol, Chem) -> list[tuple[int, int, str]]:
    cached = mol_cache_get(mol, "indexed_bond_stereochemistry")
    if cached is not None:
        return cached
    records: list[tuple[int, int, str]] = []
    for bond in iter_bonds(mol):
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        stereo = bond.GetStereo()
        if stereo in {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOTRANS}:
            descriptor = "E"
        elif stereo in {Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS}:
            descriptor = "Z"
        else:
            continue
        descriptor = terminal_haloalkene_descriptor(mol, Chem, bond, descriptor)
        records.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), descriptor))
    return mol_cache_set(mol, "indexed_bond_stereochemistry", records)


def assigned_bond_stereochemistry(mol, Chem):
    """P-92/P-93 whole-graph CIP E/Z, retaining original atom identities.

    Native bond stereo enums use RDKit's older priority assignment. They can
    disagree with modern CIP in conjugated/fused systems. New mapped
    constructions use the labels assigned on an unchanged whole-graph copy.
    """
    cached = mol_cache_get(mol, 'assigned_bond_stereochemistry_modern')
    if cached is not None:
        return cached
    from rdkit.Chem import rdCIPLabeler
    graph = Chem.Mol(mol)
    for atom in iter_atoms(graph):
        atom.SetAtomMapNum(0)
    for item in (*iter_atoms(graph), *iter_bonds(graph)):
        if item.HasProp('_CIPCode'):
            item.ClearProp('_CIPCode')
    try:
        rdCIPLabeler.AssignCIPLabels(graph, maxRecursiveIterations=1_000_000)
    except (ValueError, RuntimeError):
        return mol_cache_set(mol, 'assigned_bond_stereochemistry_modern', ())
    rows = tuple((b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetProp('_CIPCode'))
                 for b in iter_bonds(graph) if b.HasProp('_CIPCode')
                 and b.GetProp('_CIPCode') in {'E', 'Z'})
    return mol_cache_set(mol, 'assigned_bond_stereochemistry_modern', rows)


def terminal_haloalkene_descriptor(mol, Chem, bond, native):
    """Refresh the priority label for a configured C=C(H)X unit."""
    ends = (bond.GetBeginAtom(), bond.GetEndAtom())
    if any(atom.GetAtomicNum() != 6 for atom in ends):
        return native
    if not any(atom.GetTotalNumHs() == 1 and any(
            neighbor.GetAtomicNum() in {9, 17, 35, 53} for neighbor in atom.GetNeighbors())
            for atom in ends):
        return native
    start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    return next((label for a, b, label in assigned_bond_stereochemistry(mol, Chem)
                 if a == start and b == end), native)


def _rendered_ylidene_junctions(mol, name):
    """Identify an exact emitted branch's original stereo bond, once only."""
    return {frozenset((root, host)): value
            for (root, host, prefix), value in mol_cache_get(
                mol, 'rendered_ylidene_junction_stereo', {}).items()
            if prefix in name}


def _rendered_mapped_branch_bonds(mol, name):
    """Original geometric units owned by exact emitted mapped components."""
    by_prefix = {}
    for (a, b, prefix, component), descriptor in mol_cache_get(
            mol, "rendered_mapped_branch_bond_stereo", {}).items():
        by_prefix.setdefault(prefix, {}).setdefault(component, {})[frozenset((a, b))] = descriptor
    represented = {}
    for prefix, components in by_prefix.items():
        count = 0
        for match in re.finditer(re.escape(prefix), name):
            multiplier = re.search(
                r"\b(bis|tris|tetrakis|pentakis|hexakis)[([{]+$", name[:match.start()])
            count += _STEREO_MULTIPLIERS[multiplier.group(1)] if multiplier else 1
        if count >= len(components):
            for records in components.values():
                represented.update(records)
    return represented


def apply_mapped_branch_bond_stereochemistry(mol, Chem, name, locants, *, external_parent=None,
                                           attachment_idx=None):
    """Bind branch E/Z to original mapped ring atoms before composition.

    Internal bonds use their lower parent locant. A nonring exocyclic C=C or
    C=N uses its mapped carbon locant. The caller's external attachment bond
    remains owned by its junction writer. Configuration is computed on the
    unchanged whole graph, and the emitted component records exact ownership.
    """
    if not name or not _has_assigned_bond_stereo(mol, Chem):
        return name
    mapping = _stereo_locant_map(tuple(locants))
    if not mapping:
        return name
    existing = _parent_stereo_texts_in_name(name, "EZ")
    already_owned = _rendered_mapped_branch_bonds(mol, name)
    already_owned.update(_rendered_ylidene_junctions(mol, name))
    pieces, records, local_bonds = [], [], set()
    for a, b, descriptor in assigned_bond_stereochemistry(mol, Chem):
        if (frozenset((a, b)) in already_owned or external_parent in {a, b}
                or not {a, b} & mapping.keys()):
            continue
        if {a, b} <= mapping.keys():
            locant = min((mapping[a], mapping[b]), key=_stereo_locant_sort_key)
        else:
            anchor = a if a in mapping else b
            other = b if anchor == a else a
            bond = mol.GetBondBetweenAtoms(a, b)
            if (mol.GetAtomWithIdx(anchor).GetAtomicNum() != 6
                    or mol.GetAtomWithIdx(other).GetAtomicNum() not in {6, 7}
                    or bond.IsInRing()):
                continue
            locant = mapping[anchor]
        text = locant + descriptor
        local_bonds.add(text)
        if text not in existing:
            pieces.append((_stereo_locant_sort_key(locant), text))
        records.append((a, b, descriptor))
    if pieces:
        name = _prepend_stereochemical_descriptors(name, [text for _, text in sorted(pieces)])
    ownership = mol_cache_dict(mol, "rendered_mapped_branch_bond_stereo")
    component = frozenset(mapping)
    for a, b, descriptor in records:
        ownership[(a, b, name, component)] = descriptor
    if attachment_idx in mapping and external_parent is not None:
        junction = mol.GetBondBetweenAtoms(attachment_idx, external_parent)
        leading = _leading_stereochemical_descriptors(name)[0]
        if leading and junction is not None and junction.GetStereo():
            local_atoms = {mapping[i] + descriptor
                           for i, descriptor in assigned_atom_stereochemistry(mol, Chem)
                           if i in mapping}
            if set(leading) <= local_atoms | local_bonds:
                owners = mol_cache_dict(mol, "mapped_ring_branch_stereo_owner")
                owners.setdefault((attachment_idx, external_parent, name), set()).add(
                    tuple(sorted(mapping.items())))
    return name


def apply_ring_branch_junction_descriptor(mol, attachment_idx, parent_idx,
                                          source, suffix, locant, descriptor):
    """Merge only a selected branch block proved by its numbered producer."""
    key = (attachment_idx, parent_idx, source)
    scopes = mol_cache_get(mol, "mapped_ring_branch_stereo_owner", {}).get(key, ())
    if len(scopes) != 1:
        return f"({locant}{descriptor})-{suffix}"
    mapping = dict(next(iter(scopes)))
    component = frozenset(mapping)
    if locant != mapping.get(attachment_idx):
        return f"({locant}{descriptor})-{suffix}"
    name = _prepend_stereochemical_descriptors(suffix, [locant + descriptor])
    ownership = mol_cache_dict(mol, "rendered_mapped_branch_bond_stereo")
    for (a, b, prefix, scope), label in tuple(ownership.items()):
        if prefix == source and scope == component:
            ownership[(a, b, name, scope)] = label
    return name


def _rendered_retained_aminoacyl_stereo(mol, name, *, atom_scope=None):
    """Read alpha configurations owned by exact emitted retained acyl words.

    The branch writer records original atoms only after proving its graph and
    alpha configuration. Candidate exploration may record several alternatives:
    require enough occurrences of the emitted word to own every registered
    atom, rather than allowing one alias to account for two separate groups.
    """
    by_prefix = {}
    for (idx, prefix), descriptor in mol_cache_get(
            mol, "rendered_retained_aminoacyl_stereo", {}).items():
        if atom_scope is not None and idx not in atom_scope:
            continue
        by_prefix.setdefault(prefix, {})[idx] = descriptor
    represented = {}
    multipliers = {"": 1, "mono": 1, "di": 2, "tri": 3, "tetra": 4,
                   "penta": 5, "hexa": 6}
    for prefix, atoms in by_prefix.items():
        pattern = re.compile(r"(?<![A-Za-z])(?P<mult>mono|di|tri|tetra|penta|hexa)?"
                             + re.escape(prefix))
        count = 0
        for match in pattern.finditer(name):
            multiplier = multipliers[match.group("mult") or ""]
            bracket_multiplier = re.search(
                r"\b(bis|tris|tetrakis|pentakis|hexakis)[([{]+$", name[:match.start()])
            if bracket_multiplier is not None:
                multiplier *= _STEREO_MULTIPLIERS[bracket_multiplier.group(1)]
            count += multiplier
        if count >= len(atoms):
            represented.update(atoms)
    return represented


def retained_branch_stereo_is_complete(mol, Chem, name, atom_scope):
    """Prove every assigned atom in one branch is owned by emitted acyl words.

    Scope the source-atom ledger before counting words: two equal acyl groups
    elsewhere in a molecule do not require two occurrences inside one branch.
    The caller supplies the branch's connected original atom set.
    """
    if not mol_cache_get(mol, "rendered_retained_aminoacyl_stereo"):
        return False
    assigned = {idx: descriptor for idx, descriptor in assigned_atom_stereochemistry(
        mol, Chem, include_pseudo=True) if idx in atom_scope}
    represented = _rendered_retained_aminoacyl_stereo(mol, name, atom_scope=atom_scope)
    return bool(assigned) and all(represented.get(idx) == descriptor
                                  for idx, descriptor in assigned.items())


def transfer_ylidene_junction_stereo(source, target, Chem, name):
    """Carry emitted branch ownership across an index-preserving graph view.

    A Kekule view changes aromatic representation, not original bond IDs.
    Revalidate native junction geometry and whole-graph CIP assignment on the
    target before allowing exact branch text into a target candidate. Retained
    aminoacyl alpha ownership uses the same index-preserving view boundary.
    """
    retained = _rendered_retained_aminoacyl_stereo(source, name)
    if retained:
        assigned = dict(assigned_atom_stereochemistry(target, Chem, include_pseudo=True))
        if any(assigned.get(idx) != descriptor for idx, descriptor in retained.items()):
            return False
        transferred = mol_cache_dict(target, "rendered_retained_aminoacyl_stereo")
        transferred.update((key, value) for key, value in mol_cache_get(
            source, "rendered_retained_aminoacyl_stereo", {}).items() if key[0] in retained)
    branch_bonds = [(key, value) for key, value in mol_cache_get(
        source, "rendered_mapped_branch_bond_stereo", {}).items() if key[2] in name]
    if branch_bonds:
        assigned = {frozenset((a, b)): descriptor
                    for a, b, descriptor in assigned_bond_stereochemistry(target, Chem)}
        if any(assigned.get(frozenset(key[:2])) != descriptor
               for key, descriptor in branch_bonds):
            return False
        mol_cache_dict(target, "rendered_mapped_branch_bond_stereo").update(branch_bonds)
    entries = [(key, value) for key, value in mol_cache_get(
        source, 'rendered_ylidene_junction_stereo', {}).items() if key[2] in name]
    if not entries:
        return True
    modern = {frozenset((a,b)): d for a,b,d in assigned_bond_stereochemistry(target, Chem)}
    raw = {frozenset((a,b)): d for a,b,d in _indexed_bond_stereochemistry(target, Chem)}
    if any(raw.get(frozenset(key[:2])) != value[0]
           or modern.get(frozenset(key[:2])) != value[1]
           for key, value in entries):
        return False
    mol_cache_dict(target, 'rendered_ylidene_junction_stereo').update(entries)
    return True


def _locanted_bond_stereo_pieces(
    records: list[tuple[int, int, str]],
    locant_by_atom: dict[int, str],
    name: str,
    *, mol=None, Chem=None,
) -> tuple[list[tuple[tuple, str]], list[str]]:
    if mol is not None and Chem is None:
        from rdkit import Chem
    existing = _parent_stereo_texts_in_name(name, "EZ")
    owned_junctions = _rendered_ylidene_junctions(mol, name) if mol is not None else {}
    owned_branch_bonds = _rendered_mapped_branch_bonds(mol, name) if mol is not None else {}
    existing -= {f'{value[2]}{value[1]}' for value in owned_junctions.values()}
    pieces = []
    unlocanted = []
    for begin_idx, end_idx, descriptor in records:
        if (frozenset((begin_idx, end_idx)) in owned_junctions
                or frozenset((begin_idx, end_idx)) in owned_branch_bonds):
            continue
        begin_locant = locant_by_atom.get(begin_idx)
        end_locant = locant_by_atom.get(end_idx)
        if not begin_locant or not end_locant:
            # P-93.4.2.1/P-93.5.1.4.2.1(1)(a): a side-chain double bond
            # belongs to its carbon parent at the attachment locant. An unlocanted
            # E/Z can instead select a different C=N in the same name.
            if mol is not None and bool(begin_locant) != bool(end_locant):
                anchor = begin_idx if begin_locant else end_idx
                other = end_idx if begin_locant else begin_idx
                atom = mol.GetAtomWithIdx(anchor)
                bond = mol.GetBondBetweenAtoms(begin_idx, end_idx)
                if (atom.GetAtomicNum() == 6
                        and mol.GetAtomWithIdx(other).GetAtomicNum() in {6, 7}
                        and bond is not None and not bond.IsInRing()):
                    modern = next((d for a,b,d in assigned_bond_stereochemistry(mol, Chem)
                                   if {a,b} == {begin_idx,end_idx}), '')
                    if modern:
                        locant = begin_locant or end_locant
                        pieces.append((_stereo_locant_sort_key(locant), f"{locant}{modern}"))
                        continue
            unlocanted.append(descriptor)
            continue
        locant = min(begin_locant, end_locant, key=_stereo_locant_sort_key)
        text = f"{locant}{descriptor}"
        if text in existing:
            continue
        pieces.append((_stereo_locant_sort_key(locant), text))
    if unlocanted:
        represented = _remove_represented_atom_descriptors(
            _locanted_bond_stereo_descriptors_in_name(name),
            [value[1] for value in owned_junctions.values()] + list(owned_branch_bonds.values()))
        represented = _remove_represented_atom_descriptors(represented, [
            text[-1] for text in existing if text[:-1] in locant_by_atom.values()])
        represented += _leading_unlocanted_stereo_descriptors(name, "EZ")
        unlocanted = _remove_represented_atom_descriptors(unlocanted, represented)
    return pieces, unlocanted


def _stereo_locant_sort_key(locant: str) -> tuple[int, int, str]:
    match = re.match(r"^(\d+)([a-z]?)(\'*)$", locant)
    if not match:
        return (9999, 9999, locant.count("'"), locant)
    number, letter, prime = match.groups()
    letter_rank = ord(letter) - ord("a") + 1 if letter else 0
    return (int(number), letter_rank, len(prime), locant)


def _bond_stereochemistry(mol, Chem) -> list[str]:
    cached = mol_cache_get(mol, "bond_stereochemistry")
    if cached is not None:
        return cached
    descriptors = [label for _, _, label in _indexed_bond_stereochemistry(mol, Chem)]
    return mol_cache_set(mol, "bond_stereochemistry", descriptors)


def retained_stereo_is_covered(mol, template, match):
    """A retained alias must not introduce unprovided tetrahedral stereo.

    Explicit descriptors can specify a modified configuration of a retained
    parent (P-101.6). An unspecified input center cannot inherit the template's
    configuration. Cache only the template's immutable stereocentre indices.
    """
    key = "retained_template_stereo_indices"
    indices = mol_cache_get(template, key)
    if indices is None:
        indices = tuple(a.GetIdx() for a in iter_atoms(template) if int(a.GetChiralTag()))
        mol_cache_set(template, key, indices)
    return all(int(mol.GetAtomWithIdx(match[i]).GetChiralTag()) for i in indices)
