"""Source-proved omission of printed locants, preserving original numbering.

An available atom locant is not itself an instruction to print it. Producers
can attach a component-local decision to both hydride and annotation writing.
Absence of a proof preserves the existing explicit-locant behavior.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from .molecule_iteration import iter_atoms, iter_bonds
from .molecule_cache import mol_cache_get, mol_cache_set


_PLACEMENT_PROOFS = ContextVar("nispo_substitution_placement_proofs", default=None)
_PLACEMENT_PROOF_CACHE_LIMIT = 128


@contextmanager
def placement_proof_scope():
    """Share identical placement proofs across candidates for one input."""
    if _PLACEMENT_PROOFS.get() is not None:
        yield
        return
    token = _PLACEMENT_PROOFS.set({})
    try:
        yield
    finally:
        _PLACEMENT_PROOFS.reset(token)


# Primitive snapshots must not hide Python changes to native getter semantics.
# Verify descriptor kinds as well as identities, so a patch already installed
# when this module loads cannot become a trusted native baseline.
try:
    from rdkit.Chem import rdchem as _rdchem
except ImportError:
    _PRIMITIVE_NATIVE_CLASSES = ()
    _PRIMITIVE_NATIVE_METHODS = None
else:
    _PRIMITIVE_NATIVE_CLASSES = (_rdchem.Mol, _rdchem.RWMol)
    _native_function_type = type(_rdchem.GetPeriodicTable)
    _primitive_methods = []
    for _cls, _names in (
        (_rdchem.Mol, ("GetAtomWithIdx", "GetNumAtoms", "GetBondWithIdx", "GetNumBonds", "ToBinary")),
        (_rdchem.RWMol, ("GetAtomWithIdx", "GetNumAtoms", "GetBondWithIdx", "GetNumBonds", "ToBinary")),
        (_rdchem.Atom, ("GetAtomicNum", "GetIsotope", "GetFormalCharge",
                       "GetNumRadicalElectrons", "GetChiralTag", "HasQuery", "GetNeighbors", "GetIdx")),
        (_rdchem.Bond, ("GetBeginAtomIdx", "GetEndAtomIdx", "GetStereo", "HasQuery")),
    ):
        for _name in _names:
            _descriptor = getattr(_cls, _name, None)
            _primitive_methods.append((_cls, _name, _descriptor))
    _PRIMITIVE_NATIVE_METHODS = (
        tuple(_primitive_methods)
        if (_native_function_type.__module__ == "Boost.Python"
            and _native_function_type.__name__ == "function"
            and all(type(method) is _native_function_type and method.__name__ == name
                    and method.__module__ == ("Mol" if cls in _PRIMITIVE_NATIVE_CLASSES else cls.__name__)
                    for cls, name, method in _primitive_methods)
            and all(cls.__getattribute__ is object.__getattribute__
                    and not hasattr(cls, "__getattr__")
                    and "__dict__" not in vars(cls)
                    for cls in (_rdchem.Mol, _rdchem.RWMol, _rdchem.Atom, _rdchem.Bond)))
        else None
    )
    _PRIMITIVE_INTERCEPTION_CLASSES = (_rdchem.Mol, _rdchem.RWMol, _rdchem.Atom, _rdchem.Bond)
    _PRIMITIVE_MISSING = object()
    _PRIMITIVE_CLASS_NAMESPACES = tuple(
        (cls, tuple((name, vars(cls).get(name, _PRIMITIVE_MISSING))
                    for owner, name, method in _primitive_methods if owner is cls))
        for cls in _PRIMITIVE_INTERCEPTION_CLASSES
    )


def _primitive_native_getters_unchanged():
    if _PRIMITIVE_NATIVE_METHODS is None:
        return False
    missing = _PRIMITIVE_MISSING
    for cls, methods in _PRIMITIVE_CLASS_NAMESPACES:
        namespace = vars(cls)
        if (cls.__getattribute__ is not object.__getattribute__
                or hasattr(cls, "__getattr__") or "__dict__" in namespace):
            return False
        get = namespace.get
        for name, method in methods:
            if get(name, missing) is not method:
                return False
    return True


@dataclass(frozen=True, slots=True)
class ComponentLocantPresentation:
    """P-14.3.4.2(d)/P-93.5.1.4.1 for one bare monounsaturated carbocycle.

    Atom bindings remain complete. ``bond_atoms`` identifies the physical
    multiple bond whose numerical locant is omitted from the hydride and E/Z
    descriptor; it does not erase the internal locant used by evidence.
    """

    atom_locants: tuple[tuple[int, str], ...]
    bond_atoms: frozenset[int]
    bond_order: int
    rule: str = "P-14.3.4.2(d)/P-93.5.1.4.1"

    def stereo_bond_locant(self, a, b, locant):
        return "" if frozenset((a, b)) == self.bond_atoms else locant


def monounsaturated_carbocycle_presentation(mol, atom_locants):
    """Prove the unmodified monocyclic component directly on its input graph.

    Substitution, isotope labels, charges, radicals, aromaticity, additional
    unsaturation, atom/axial stereo and unfamiliar bond stereo retain their
    established path. Ordinary E/Z alone needs no numerical locant here.
    The complete bare heavy-atom graph bounds this first omission rule.
    """
    from rdkit.Chem.rdchem import BondStereo

    ordinary_stereo = {BondStereo.STEREOE, BondStereo.STEREOZ,
                       BondStereo.STEREOCIS, BondStereo.STEREOTRANS}
    bindings = tuple(atom_locants)
    mapping = dict(bindings)
    size = len(bindings)
    if (size < 3 or len(mapping) != size
            or set(mapping.values()) != {str(i) for i in range(1, size + 1)}
            or mol.GetStereoGroups()):
        return None
    core = frozenset(mapping)
    heavy = frozenset(atom.GetIdx() for atom in iter_atoms(mol)
                      if atom.GetAtomicNum() > 1)
    if core != heavy:
        return None
    for atom in iter_atoms(mol):
        if (atom.GetIsotope() or atom.GetFormalCharge()
                or atom.GetNumRadicalElectrons() or int(atom.GetChiralTag())
                or atom.GetIsAromatic() or atom.HasQuery()):
            return None
        if atom.GetIdx() in core:
            if (atom.GetAtomicNum() != 6 or atom.GetTotalValence() != 4
                    or sum(n.GetIdx() in core for n in atom.GetNeighbors()) != 2):
                return None
        elif (atom.GetAtomicNum() != 1 or atom.GetDegree() != 1
              or atom.GetNeighbors()[0].GetIdx() not in core):
            return None
    edges, multiple = 0, []
    for bond in iter_bonds(mol):
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if bond.HasQuery() or bond.GetIsAromatic():
            return None
        order = bond.GetBondTypeAsDouble()
        stereo = bond.GetStereo()
        if {a, b} <= core:
            edges += 1
            gap = abs(int(mapping[a]) - int(mapping[b]))
            if gap not in {1, size - 1} or order not in {1, 2, 3}:
                return None
            if order != 1:
                multiple.append((frozenset((a, b)), int(order)))
                # ANY, atropisomeric and future stereo states are not proof.
                if stereo and (order != 2 or size < 8 or stereo not in ordinary_stereo):
                    return None
            elif stereo:
                return None
        elif order != 1 or stereo:
            return None
    if edges != size or len(multiple) != 1:
        return None
    atoms, order = multiple[0]
    if {mapping[idx] for idx in atoms} != {"1", "2"}:
        return None
    return ComponentLocantPresentation(tuple(sorted(bindings)), atoms, order)


def valid_component_locant_presentation(mol, component):
    """Revalidate both graph scope and the actual hydride writing boundary."""
    from .constants import ROOTS

    proof = component.locant_presentation
    if not isinstance(proof, ComponentLocantPresentation):
        return False
    if (component.prefixes or component.embedded_atoms or component.hydro_prefix
            or component.local_isotope_atoms or component.isotope_descriptor_parts
            or component.isotope_labelling_kind or component.isotope_role_evidence is not None
            or component.amide_isotope_evidence is not None
            or component.parent_annotation_proof is not None
            or component.nonstereogenic_atom_evidence is not None
            or component.assembly_stereo_evidence is not None
            or set(dict(proof.atom_locants)) != component.atoms
            or proof != monounsaturated_carbocycle_presentation(mol, component.atom_locants)):
        return False
    root = ROOTS.get(len(proof.atom_locants))
    ending = "ene" if proof.bond_order == 2 else "yne"
    return root is not None and component.parent == "cyclo" + root + ending


@dataclass(frozen=True, slots=True)
class SubstitutionSite:
    """One producer-owned branch and its original parent attachment."""

    host: int
    root: int
    atoms: frozenset[int]
    text: str


@dataclass(frozen=True, slots=True)
class SubstitutionLocantPresentation:
    atom_locants: tuple[tuple[int, str], ...]
    substitutions: tuple[SubstitutionSite, ...]
    rule: str


def _branch_isotopes_without_positions(mol, sub):
    """Conservative P-82.6.1.1/.2/.3 boundary, without reading branch text."""
    masses = {}
    for idx in sub.atoms:
        atom = mol.GetAtomWithIdx(idx)
        number, mass = atom.GetAtomicNum(), atom.GetIsotope()
        if (number == 1 and mass or any(n.GetAtomicNum() == 1 and n.GetIsotope()
                                       for n in atom.GetNeighbors())):
            return False
        masses.setdefault(number, set()).add(mass)
    return all(len(values) == 1 for values in masses.values())


def _two_atom_substitution_scope(mol, bindings, substitutions):
    """P-14.3.4.2(b), with P-82.6.1.1 annotation boundaries.

    The unmodified parent has two identical, connected skeletal atoms and
    exactly one external substituent. Its attachment can consume more than
    one H (for example diazo). A branch's own charge separation or assigned
    configuration cannot distinguish the two otherwise equivalent ends.
    Numeric isotope positions keep the conservative explicit-locant path.
    Full ownership and normal valence are checked by the restoration proof.
    """
    mapping = dict(bindings)
    if (len(bindings) != 2 or len(mapping) != 2
            or set(mapping.values()) != {'1', '2'} or len(substitutions) != 1):
        return False
    sub = substitutions[0]
    if not isinstance(sub, SubstitutionSite) or sub.host not in mapping:
        return False
    atoms = [mol.GetAtomWithIdx(i) for i in mapping]
    if (atoms[0].GetAtomicNum() != atoms[1].GetAtomicNum()
            or any(a.GetAtomicNum() <= 1 or a.GetIsAromatic() or a.IsInRing()
                   or a.GetFormalCharge() or a.GetIsotope() or a.GetNumRadicalElectrons()
                   or int(a.GetChiralTag()) or a.HasQuery()
                   or any(n.GetAtomicNum() == 1 and n.GetIsotope() for n in a.GetNeighbors())
                   for a in atoms)):
        return False
    bond = mol.GetBondBetweenAtoms(*mapping)
    if (bond is None or bond.HasQuery() or bond.GetIsAromatic() or int(bond.GetStereo())
            or bond.GetBondTypeAsDouble() not in {1, 2, 3}):
        return False
    attachment = mol.GetBondBetweenAtoms(sub.host, sub.root)
    if attachment is None or int(attachment.GetStereo()):
        return False
    # A unique N/halogen isotope needs no numeric locant. An incompletely
    # modified set of identical branch elements may need one, so it cannot
    # authorize dropping outer chemical locants under P-82.6.1.1.
    return _branch_isotopes_without_positions(mol, sub)


def _monocyclic_substitution_scope(mol, bindings, substitutions):
    """P-14.3.4.2(c) for one uniform saturated or aromatic simple cycle.

    This linear, at-most-64-atom topology check does not enumerate symmetry
    or substitution placements. One owned branch may consume multiple H;
    branch charge and configuration do not distinguish the restored ring.
    Annular unsaturation locants, partial isotope labels and nonuniform
    elements retain the explicit path. Restoration checks actual valence,
    ownership and the complete branch boundary before authorizing omission.
    """
    if not 3 <= len(bindings) <= 64 or len(substitutions) != 1:
        return False
    mapping = dict(bindings)
    if (len(mapping) != len(bindings)
            or set(mapping.values()) != {str(i) for i in range(1, len(mapping) + 1)}):
        return False
    sub = substitutions[0]
    if not isinstance(sub, SubstitutionSite) or sub.host not in mapping:
        return False
    numbers, masses, adjacent, bonds = set(), set(), {}, []
    for idx in mapping:
        atom = mol.GetAtomWithIdx(idx)
        if (atom.GetAtomicNum() <= 1 or atom.GetFormalCharge()
                or atom.GetNumRadicalElectrons() or int(atom.GetChiralTag()) or atom.HasQuery()
                or any(n.GetAtomicNum() == 1 and n.GetIsotope() for n in atom.GetNeighbors())):
            return False
        numbers.add(atom.GetAtomicNum())
        masses.add(atom.GetIsotope())
        adjacent[idx] = tuple(n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() in mapping)
        if len(adjacent[idx]) != 2:
            return False
        for other in adjacent[idx]:
            if idx < other:
                bond = mol.GetBondBetweenAtoms(idx, other)
                if bond.HasQuery() or int(bond.GetStereo()):
                    return False
                bonds.append(bond)
    if (len(numbers) != 1 or len(masses) != 1 or len(bonds) != len(mapping)
            or not (all(b.GetBondTypeAsDouble() == 1 for b in bonds)
                    or all(b.GetIsAromatic() for b in bonds))):
        return False
    reached, stack = set(), [sub.host]
    while stack:
        idx = stack.pop()
        if idx not in reached:
            reached.add(idx)
            stack.extend(adjacent[idx])
    if len(reached) != len(mapping):
        return False  # A collection of disjoint cycles is not a monocycle.
    attachment = mol.GetBondBetweenAtoms(sub.host, sub.root)
    if (attachment is None or attachment.HasQuery() or int(attachment.GetStereo())
            or attachment.GetBondTypeAsDouble() not in {1, 2, 3}):
        return False
    return _branch_isotopes_without_positions(mol, sub)


def _restored_parent(mol, Chem, bindings, substitutions):
    """Cut proved branches and restore exactly their bond valences as H."""
    mapping = dict(bindings)
    core = frozenset(mapping)
    if not core or len(core) != len(bindings) or len(core) > 64 or not substitutions:
        return None
    # P-14.3.4.2(a-c) have one kind of parent substitution position.
    # Their proved branch states do not change that positional comparison.
    # Other omission proofs retain their conservative annotation scope.
    # No branch-internal locant is removed.
    mononuclear = len(core) == 1 and set(mapping.values()) == {'1'}
    two_atom = not mononuclear and _two_atom_substitution_scope(mol, bindings, substitutions)
    monocyclic = not (mononuclear or two_atom) and _monocyclic_substitution_scope(mol, bindings, substitutions)
    independent_branch = mononuclear or two_atom or monocyclic
    owned, cuts = set(), {}
    for sub in substitutions:
        if (not isinstance(sub, SubstitutionSite) or sub.host not in core
                or not sub.atoms or sub.root not in sub.atoms
                or sub.atoms & (core | owned) or not sub.text):
            return None
        boundary = [(i, n.GetIdx()) for i in sub.atoms
                    for n in mol.GetAtomWithIdx(i).GetNeighbors()
                    if n.GetIdx() not in sub.atoms and n.GetAtomicNum() > 1]
        if boundary != [(sub.root, sub.host)]:
            return None
        bond = mol.GetBondBetweenAtoms(sub.host, sub.root)
        order = bond.GetBondTypeAsDouble()
        if order not in {1, 2, 3}:
            return None
        cuts[sub.host] = cuts.get(sub.host, 0) + int(order)
        owned.update(sub.atoms)
    scope = core | owned
    valences = {6:4, 7:3, 8:2, 13:3, 14:4, 16:2, 32:4, 34:2, 50:4, 52:2, 82:4}
    if independent_branch:
        valences.update({5:3, 15:3, 31:3, 33:3, 49:3, 51:3, 81:3, 83:3})
    for idx in scope:
        atom = mol.GetAtomWithIdx(idx)
        if (atom.GetNumRadicalElectrons() or atom.HasQuery()
                or atom.GetFormalCharge() and (idx in core or not independent_branch)
                or not independent_branch and (atom.GetIsotope() or int(atom.GetChiralTag())
                    or any(n.GetAtomicNum() == 1 and n.GetIsotope() for n in atom.GetNeighbors()))):
            return None
        if idx in core and (atom.GetAtomicNum() not in valences
                or atom.GetTotalValence() != valences[atom.GetAtomicNum()]
                or any(n.GetAtomicNum() > 1 and n.GetIdx() not in scope for n in atom.GetNeighbors())):
            return None
    if any(b.HasQuery() or (int(b.GetStereo()) and not independent_branch) for b in iter_bonds(mol)
           if b.GetBeginAtomIdx() in scope or b.GetEndAtomIdx() in scope):
        return None
    parent, positions = Chem.RWMol(), {}
    for idx in sorted(core):
        source = mol.GetAtomWithIdx(idx)
        atom = Chem.Atom(source)
        atom.SetAtomMapNum(0)
        if mononuclear:
            # The privately restored hydride has no tetrahedral substitution
            # pattern. Original configuration stays on the original graph.
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        atom.SetNoImplicit(True)
        removed_h = sum(n.GetAtomicNum() == 1 and n.GetIdx() in owned
                        for n in source.GetNeighbors())
        atom.SetNumExplicitHs(source.GetTotalNumHs(includeNeighbors=True) - removed_h + cuts.get(idx, 0))
        positions[idx] = parent.AddAtom(atom)
    for bond in iter_bonds(mol):
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in core and b in core:
            parent.AddBond(positions[a], positions[b], bond.GetBondType())
    parent = parent.GetMol()
    try:
        Chem.SanitizeMol(parent)
    except (RuntimeError, ValueError):
        return None
    # P-14.3.4.5 excludes chalcogen H and formyl H from substitutable H.
    slots = {}
    for idx, pos in positions.items():
        atom = parent.GetAtomWithIdx(pos)
        hydrogen = atom.GetTotalNumHs()
        formyl = atom.GetAtomicNum() == 6 and hydrogen and any(
            bond.GetBondTypeAsDouble() == 2 and bond.GetOtherAtom(atom).GetAtomicNum() == 8
            for bond in atom.GetBonds())
        # The chalcogen/formyl H exception belongs to .5's complete
        # substitution rule, not the local parent rules in .2(a-c).
        if hydrogen and (independent_branch
                         or (atom.GetAtomicNum() not in {8,16,34,52} and not formyl)):
            slots[idx] = hydrogen
    if not slots or any(sub.host not in slots for sub in substitutions):
        return None
    rule = ('P-14.3.4.2(a)' if mononuclear else 'P-14.3.4.2(b)' if two_atom
            else 'P-14.3.4.2(c)' if monocyclic else '')
    return parent, positions, slots, rule


def _rooted_parent_key(parent, position, Chem):
    copy = Chem.Mol(parent)
    copy.GetAtomWithIdx(position).SetAtomMapNum(1)
    return Chem.MolToSmiles(copy, canonical=True, isomericSmiles=True)


def _placement_template(mol, Chem, parent, substitutions):
    """Prepare invariant branch atoms and bonds once for the local search."""
    graph = Chem.RWMol(parent)
    attachments = []
    bonds = tuple((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType())
                  for bond in iter_bonds(mol))
    for sub in substitutions:
        bond = mol.GetBondBetweenAtoms(sub.host, sub.root)
        branch = {}
        for idx in sorted(sub.atoms):
            source = mol.GetAtomWithIdx(idx)
            atom = Chem.Atom(source)
            atom.SetAtomMapNum(0)
            atom.SetNoImplicit(True)
            atom.SetNumExplicitHs(source.GetTotalNumHs(includeNeighbors=True))
            branch[idx] = graph.AddAtom(atom)
        for a, b, kind in bonds:
            if a in sub.atoms and b in sub.atoms:
                graph.AddBond(branch[a], branch[b], kind)
        attachments.append((branch[sub.root], bond.GetBondType(),
                            int(bond.GetBondTypeAsDouble())))
    return graph, tuple(attachments)


def _placement_key(mol, Chem, parent, positions, substitutions, hosts, template=None):
    if template is None:
        template = _placement_template(mol, Chem, parent, substitutions)
    prepared, attachments = template
    graph = Chem.RWMol(prepared)
    used = {}
    for (root, kind, order), host in zip(attachments, hosts, strict=True):
        used[host] = used.get(host, 0) + order
        graph.AddBond(positions[host], root, kind)
    for host, count in used.items():
        atom = graph.GetAtomWithIdx(positions[host])
        atom.SetNumExplicitHs(atom.GetNumExplicitHs() - count)
    graph = graph.GetMol()
    Chem.SanitizeMol(graph)
    return Chem.MolToSmiles(graph, canonical=True, isomericSmiles=True)


def _unique_substitution_placement(mol, Chem, parent, positions, slots, subs, weights):
    template = _placement_template(mol, Chem, parent, subs)
    hosts = tuple(sub.host for sub in subs)
    proofs = _PLACEMENT_PROOFS.get()
    key = None
    graph, attachments = template
    if proofs is not None and type(graph) in _PRIMITIVE_NATIVE_CLASSES:
        # The bounded search depends on the prepared graph, attachments,
        # capacities and visit order. The printed locant labels are irrelevant
        # once the parent has been restored and this proof is applicable.
        try:
            key = (id(Chem), graph.ToBinary(Chem.PropertyPickleOptions.AllProps), attachments,
                   tuple(positions.items()), tuple(slots.items()), tuple(weights), hosts)
            previous = proofs.get(key)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            key = None
        else:
            if previous is not None:
                return previous

    actual = _placement_key(mol, Chem, parent, positions, subs, hosts, template)
    chosen, used = [], {}
    visited = 0

    def unique(k):
        nonlocal visited
        visited += 1
        if visited > 2048:
            return False
        if k == len(subs):
            try:
                return _placement_key(mol, Chem, parent, positions, subs, chosen, template) == actual
            except (RuntimeError, ValueError):
                return False
        for idx, capacity in slots.items():
            if used.get(idx, 0) + weights[k] > capacity:
                continue
            chosen.append(idx)
            used[idx] = used.get(idx, 0) + weights[k]
            if not unique(k + 1):
                return False
            used[idx] -= weights[k]
            chosen.pop()
        return True

    proved = unique(0)
    if key is not None and len(proofs) < _PLACEMENT_PROOF_CACHE_LIMIT:
        proofs[key] = proved
    return proved


def substitution_locant_presentation(mol, Chem, atom_locants, substitutions):
    """Prove .2(a-c), .3, .5, .6, or bounded exhaustive .4 omission.

    The budget bounds proof work, never a chemical ranking. An incomplete
    search cannot establish uniqueness. All decisions preserve the bindings.
    """
    from .molecule_cache import exact_graph_computation

    bindings, subs = tuple(atom_locants), tuple(substitutions)
    proof_bindings = bindings
    if 1 < len(bindings) <= 64:
        mapping = dict(bindings)
        locants = tuple(map(str, range(1, len(bindings) + 1)))
        if (len(mapping) == len(bindings) and set(mapping.values()) == set(locants)
                and all(type(idx) is int for idx in mapping)):
            # With consecutive unique numeric locants, restoration and all
            # omission rules depend on the sites and graph. Renumbering can
            # change the evidence's printed bindings but cannot change its rule.
            proof_bindings = tuple(zip(sorted(mapping), locants))
    rule = exact_graph_computation(
        mol, Chem, "substitution_locant_exact_graph", (id(Chem), proof_bindings, subs),
        lambda: _substitution_locant_rule(mol, Chem, proof_bindings, subs),
    )
    return SubstitutionLocantPresentation(tuple(sorted(bindings)), subs, rule) if rule else None


def _substitution_locant_rule(mol, Chem, bindings, subs):
    restored = _restored_parent(mol, Chem, bindings, subs)
    if restored is None:
        return None
    parent, positions, slots, rule = restored
    mapping = dict(bindings)
    weights = [int(mol.GetBondBetweenAtoms(sub.host,sub.root).GetBondTypeAsDouble()) for sub in subs]
    if not rule and len(subs) == 1 and weights == [1]:
        if len({_rooted_parent_key(parent,positions[idx],Chem) for idx in slots}) == 1:
            rule = "P-14.3.4.3"
    if not rule and len({mapping[idx] for idx in slots}) == 1:
        rule = "P-14.3.4.6"
    if (not rule and len({sub.text for sub in subs}) == 1
            and all(parent.GetAtomWithIdx(positions[idx]).GetAtomicNum() not in {8,16,34,52}
                    for idx in slots)):
        usage = {idx:sum(weight for sub,weight in zip(subs,weights,strict=True) if sub.host == idx)
                 for idx in slots}
        # Identical text is insufficient: compare the rooted branch graphs.
        unnumbered = Chem.Mol(mol)
        for atom in iter_atoms(unnumbered):
            atom.SetAtomMapNum(0)
        branch_keys = set()
        for sub in subs:
            fragment = Chem.MolFragmentToSmiles(unnumbered, atomsToUse=sorted(sub.atoms),
                                               rootedAtAtom=sub.root, canonical=True)
            branch_keys.add(fragment)
        if usage == slots and len(branch_keys) == 1:
            rule = "P-14.3.4.5"
    if (not rule and len(subs) <= 8
            and _unique_substitution_placement(mol, Chem, parent, positions, slots, subs, weights)):
        rule = "P-14.3.4.4"
    return rule


def _component_primitive_graph(mol):
    """Reuse primitive graph facts only while the native graph bytes agree.

    Python/native annotations do not enter these edge, atom or bond-state
    facts. Positive proof decisions retain their all-properties state key.
    Custom graph wrappers retain their existing getter semantics.
    """
    if type(mol) not in _PRIMITIVE_NATIVE_CLASSES or not _primitive_native_getters_unchanged():
        return None
    if any(name in vars(mol) for name in (
            "GetAtomWithIdx", "GetNumAtoms", "GetBondWithIdx", "GetNumBonds", "ToBinary")):
        return None
    try:
        state = mol.ToBinary()
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return None
    previous = mol_cache_get(mol, "component_primitive_graph")
    if previous is not None and previous[0] == state:
        return previous[1]
    atoms = tuple(iter_atoms(mol))
    numbers = tuple(atom.GetAtomicNum() for atom in atoms)
    forbidden = [bool(atom.GetIsotope() or atom.GetFormalCharge()
                      or atom.GetNumRadicalElectrons() or int(atom.GetChiralTag())
                      or atom.HasQuery()) for atom in atoms]
    neighbors = [[] for _ in numbers]
    for bond in iter_bonds(mol):
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        neighbors[a].append(b)
        neighbors[b].append(a)
        if int(bond.GetStereo()) or bond.HasQuery():
            forbidden[a] = forbidden[b] = True
    neighbors = tuple(tuple(row) for row in neighbors)
    forbidden = tuple(flag or any(numbers[n] == 1 and atoms[n].GetIsotope()
                                 for n in neighbors[idx])
                      for idx, flag in enumerate(forbidden))
    facts = numbers, neighbors, forbidden
    mol_cache_set(mol, "component_primitive_graph", (state, facts))
    return facts


def component_prefix_locant_presentation(mol, Chem, component):
    """Join written prefixes to their real branch ownership before proving omission."""
    from collections import Counter
    import re

    if component.substitution_locant_presentation is not None:
        return (component.substitution_locant_presentation
                if valid_joint_substitution_presentation(mol,Chem,component) else None)
    if (not component.prefixes or component.prefix_mode != "locanted"
            or component.retained_acid_locant_evidence is not None
            or component.functionalized_carbon_prefix_proofs or component.prefix_natures is not None
            or component.prefix_citation_order is not None or component.tree_assembly_evidence is not None):
        return None
    structural_locants = any(char.isdigit() for char in component.parent)
    if structural_locants:
        # Keep intrinsic ring-element locants, such as 1,3,5-triazine, in the
        # parent name while separately proving complete prefix substitution.
        # Other numeric parent/suffix/hydrogenation constructions retain the
        # previous conservative behavior.
        if (component.hydro_prefix
                or not re.fullmatch(r'[0-9]+(?:,[0-9]+)*-[a-z]+', component.parent)
                or not component.atoms
                or any(type(idx) is not int or idx < 0 or idx >= mol.GetNumAtoms()
                       or not mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in component.atoms)):
            return None
    mapping = dict(component.atom_locants)
    facts = (_component_primitive_graph(mol)
             if type(component.atoms) in (set, frozenset, tuple, list)
             and type(component.embedded_atoms) in (set, frozenset, tuple, list)
             else None)
    if facts is not None:
        count = len(facts[0])
        if (any(type(idx) is not int or not 0 <= idx < count for idx in component.atoms)
                or any(type(idx) is not int or not 0 <= idx < count
                       for idx in component.embedded_atoms)):
            facts = None
    if facts is None:
        core = frozenset(idx for idx in component.atoms if mol.GetAtomWithIdx(idx).GetAtomicNum() > 1)
        pending = set(idx for idx in component.embedded_atoms
                      if mol.GetAtomWithIdx(idx).GetAtomicNum() > 1
                      or len(core) == 1 and mol.GetAtomWithIdx(idx).GetIsotope())
        neighbors = {idx: tuple(n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors())
                     for idx in pending}
    else:
        numbers, neighbors, forbidden = facts
        core = frozenset(idx for idx in component.atoms if numbers[idx] > 1)
        pending = set(idx for idx in component.embedded_atoms if numbers[idx] > 1
                      or len(core) == 1 and mol.GetAtomWithIdx(idx).GetIsotope())
    ownership = {(root,host):(frozenset(atoms),text)
                 for root,host,atoms,text in component.prefix_branch_ownership}
    subs = []
    primitive = {9:"fluoro",17:"chloro",35:"bromo",53:"iodo"}
    while pending:
        branch, stack = set(), [min(pending)]
        while stack:
            idx = stack.pop()
            if idx in branch:
                continue
            branch.add(idx)
            stack.extend(other for other in neighbors[idx]
                         if other in pending and other not in branch)
        pending.difference_update(branch)
        boundary = [(idx, other) for idx in branch
                    for other in neighbors[idx] if other in core]
        if len(boundary) != 1:
            return None
        root, host = boundary[0]
        proved = ownership.get((root,host))
        if proved is not None and proved[0] == branch:
            text = proved[1]
        elif len(branch) == 1 and mol.GetAtomWithIdx(root).GetAtomicNum() in primitive:
            text = primitive[mol.GetAtomWithIdx(root).GetAtomicNum()]
        else:
            return None
        subs.append(SubstitutionSite(host,root,frozenset(branch),text))
    if Counter((mapping[sub.host],sub.text) for sub in subs) != Counter(component.prefixes):
        return None
    bindings = tuple((idx, mapping[idx]) for idx in sorted(core))
    if facts is not None and (any(forbidden[idx] for idx in core)
                             or any(forbidden[idx] for sub in subs for idx in sub.atoms)):
        mononuclear = len(core) == 1 and {mapping[idx] for idx in core} == {'1'}
        if not (mononuclear or _two_atom_substitution_scope(mol, bindings, subs)
                or _monocyclic_substitution_scope(mol, bindings, subs)):
            # The restoration proof rejects these same scope-local states.
            return None
    presentation = substitution_locant_presentation(
        mol, Chem, bindings, subs)
    if structural_locants and (presentation is None or presentation.rule != 'P-14.3.4.5'):
        return None
    if presentation is not None and _formamide_n_locants_required(mol, Chem, component, presentation):
        return None
    return presentation


def _formamide_n_locants_required(mol, Chem, component, presentation):
    """Respect P-66 N-site citation while retaining fully identical substitution.

    The generic equivalence proof excludes formyl H, so a substituted
    formamide can otherwise lose all N locants even when P-66.1.1.3.1
    requires them.  P-14.3.4.5 still permits e.g. dimethylformamide when
    both substitutable H atoms have been replaced by identical groups.
    """
    if component.parent != 'formamide' or not component.nitrogen_site_locants:
        return False
    substitutions = presentation.substitutions
    if not substitutions:
        return False
    sites = {sub.host for sub in substitutions}
    if (len(sites) != 1 or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 7 for i in sites)
            or any(not str(loc).startswith('N') for loc, _ in component.prefixes)):
        return False
    nitrogen = mol.GetAtomWithIdx(next(iter(sites)))
    if (len(substitutions) == 2 and nitrogen.GetTotalNumHs(includeNeighbors=True) == 0
            and len({sub.text for sub in substitutions}) == 1):
        # Identical spelling alone does not prove identical source groups.
        graph = Chem.Mol(mol)
        for atom in graph.GetAtoms():
            atom.SetAtomMapNum(0)
        keys = {Chem.MolFragmentToSmiles(graph, atomsToUse=sorted(sub.atoms),
                                         rootedAtAtom=sub.root, canonical=True,
                                         isomericSmiles=True)
                for sub in substitutions}
        if len(keys) == 1:
            return False
    return True


def valid_joint_substitution_presentation(mol, Chem, component):
    """Revalidate a source-typed suffix joined to prefixes at one H position."""
    from collections import Counter

    proof = component.substitution_locant_presentation
    if (not isinstance(proof, SubstitutionLocantPresentation)
            or proof.rule != 'P-14.3.4.6' or not proof.substitutions
            or dict(proof.atom_locants) != dict(component.atom_locants)
            or set(dict(proof.atom_locants)) != component.atoms
            or component.hydro_prefix or any(char.isdigit() for char in component.parent)):
        return False
    primary, *prefixes = proof.substitutions
    endings = {'alcohol':'ol','thiol':'thiol','acid':'carboxylic acid',
               'amide':'carboxamide','nitrile':'carbonitrile'}
    if primary.text not in endings or not component.parent.endswith(endings[primary.text]):
        return False
    if (Counter((dict(proof.atom_locants)[sub.host],sub.text) for sub in prefixes)
            != Counter(component.prefixes)
            or frozenset().union(*(sub.atoms for sub in proof.substitutions)) != component.embedded_atoms):
        return False
    return proof == substitution_locant_presentation(mol,Chem,proof.atom_locants,proof.substitutions)


def sole_suffix_locant_presentation(mol, Chem, parent_atoms, atom_locants, groups,
                                    *, structural_locants_required, prefixes=()):
    """A producer supplies the actual single suffix role and parent boundary."""
    if structural_locants_required or len(groups) != 1:
        return None
    core = frozenset(parent_atoms)
    mapping = dict(atom_locants)
    group = groups[0]
    atoms = frozenset(idx for idx in group.atoms if idx not in core
                      and mol.GetAtomWithIdx(idx).GetAtomicNum() > 1)
    boundary = [(idx,n.GetIdx()) for idx in atoms
                for n in mol.GetAtomWithIdx(idx).GetNeighbors() if n.GetIdx() in core]
    if len(boundary) != 1:
        return None
    root,host = boundary[0]
    if mol.GetAtomWithIdx(host).GetAtomicNum() != 6:
        return None
    substitutions = [SubstitutionSite(host,root,atoms,group.kind)]
    if prefixes:
        from collections import Counter
        primitive = {9:'fluoro',17:'chloro',35:'bromo',53:'iodo'}
        for idx in sorted(core):
            for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
                other = neighbor.GetIdx()
                if other in core or other in atoms or neighbor.GetAtomicNum() == 1:
                    continue
                text = primitive.get(neighbor.GetAtomicNum())
                if not text or neighbor.GetDegree() != 1:
                    return None
                substitutions.append(SubstitutionSite(idx,other,frozenset({other}),text))
        if Counter((mapping[sub.host],sub.text) for sub in substitutions[1:]) != Counter(prefixes):
            return None
    result = substitution_locant_presentation(mol,Chem,
        tuple((idx,mapping[idx]) for idx in sorted(core)),
        substitutions)
    return result if not prefixes or result is not None and result.rule == 'P-14.3.4.6' else None
