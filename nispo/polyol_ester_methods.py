"""Mapped functional-class ester naming for bounded single polyol residues."""
import re
from collections import Counter
from dataclasses import dataclass, replace

from .constants import MULT, ROOTS
from .component_mapping import MappedComponent, decorate_component, has_annotations
from .component_annotation_evidence import (
    MappedComponentDecoration, decorate_component_with_evidence,
)
from .formatting import prefix_alphabetization_key, enclose_name_unit
from .naming_methods import MethodDecision
from .naming_result import SelectionLimitation
from .acid_anion_multiplication import AnionMultiplication, multiply_acid_anion
from .acid_anion_ordering import AcidAnionOrderAssessment
from .isotope_component_coverage import (
    IsotopeComponentRecord, IsotopeCoverageResult, component_record_from_atom_map,
    validate_isotope_component_coverage,
)


@dataclass(frozen=True, slots=True)
class PolyolEsterMethodDecision(MethodDecision):
    """Polyol ester proposal with typed per-link oxygen isotope coverage."""

    isotope_components: tuple[IsotopeComponentRecord, ...] = ()
    isotope_coverage: IsotopeCoverageResult | None = None
    equivalent_organyl_atom_locants: tuple[tuple[tuple[int, str], ...], ...] = ()
    organyl_decorations: tuple[MappedComponentDecoration, ...] = ()
    acid_decorations: tuple[MappedComponentDecoration, ...] = ()
    anion_multiplications: tuple[AnionMultiplication, ...] = ()
    citation_order_evidence: AcidAnionOrderAssessment | None = None


_STEREO_ANION = re.compile(r"^\((?:\d+[RSZErs](?:\*)?)(?:,\d+[RSZErs](?:\*)?)*\)-")
_ISOTOPE_ANION = re.compile(
    r"^\((?:\d+-)?\d+[A-Z][a-z]?(?:\d+)?"
    r"(?:,(?:\d+-)?\d+[A-Z][a-z]?(?:\d+)?)*\)"
)


def _anion_needs_enclosure(name, *, outer_locant=True):
    """Keep a citation locant outside the acid's own stereo descriptor.

    A separately cited anion word needs no extra enclosing scope solely for
    its leading descriptor. P-65's site-locanted form does: the parentheses
    participate in P-16 nesting, giving a locant followed by square brackets.
    Other complex component enclosures retain their established treatment.
    """
    stereo = _STEREO_ANION.match(name)
    if stereo:
        if outer_locant:
            return True
        name = name[stereo.end():]
    isotope = _ISOTOPE_ANION.match(name)
    remainder = name[isotope.end():] if isotope else name
    return '(' in remainder


def _anion_is_modified(part, mol):
    """Whether this mapped acid component has any isotopically modified atom."""
    return any(mol.GetAtomWithIdx(i).GetIsotope() for i in part.atoms)


def _ordinary_acid_components(mol, Chem, groups, *, allow_mapped_ring=False, allow_mapped_chain=False):
    """Return acid component names/identities after clearing isotope and stereo.

    This is an applicability key only. Rendered candidates always use the
    original annotated graph; clearing annotations here lets the caller
    identify same-constitution acid citations without sorting their labels.
    """
    from .acid_derivative_methods import _acid_component, _anion
    ordinary = Chem.Mol(mol)
    for atom in ordinary.GetAtoms():
        atom.SetIsotope(0)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for bond in ordinary.GetBonds():
        bond.SetStereo(Chem.BondStereo.STEREONONE)
        bond.SetBondDir(Chem.BondDir.NONE)
    try:
        Chem.AssignStereochemistry(ordinary, cleanIt=True, force=True)
    except (ValueError, RuntimeError):
        return None
    rows = []
    for group in groups:
        part = _acid_component(ordinary, Chem, group.center, group.ligand,
                               allow_mapped_ring=allow_mapped_ring,
                               allow_mapped_chain=allow_mapped_chain)
        if part is None:
            return None
        anion = _anion(part.name)
        if not anion:
            return None
        rows.append((part.identity, part.name, anion))
    return tuple(rows)


def _acid_anion_order_assessment(mol, parts, ordinary_parts):
    """Expose source-closed ordering separately from the isotope-order inference."""
    from .acid_anion_ordering import (
        AcidAnionCitationFacts, NuclideModification, assess_acid_anion_order,
    )
    facts = []
    for part, ordinary in zip(parts, ordinary_parts):
        locants = dict(part.atom_locants)
        profile = set()
        for idx in set(part.atoms) | {part.bridge_oxygen}:
            if idx is None:
                continue
            atom = mol.GetAtomWithIdx(idx)
            if not atom.GetIsotope():
                continue
            if idx == part.carbonyl_oxygen:
                role = 'carbonyl-oxygen'
            elif idx == part.bridge_oxygen:
                role = 'ester-bridge-oxygen'
            else:
                role = locants.get(idx)
                if role is None:
                    profile = None
                    break
            profile.add(NuclideModification(atom.GetSymbol(), atom.GetIsotope(), role))
        facts.append(AcidAnionCitationFacts(
            ordinary[0], ordinary[2], part.identity,
            frozenset(profile) if profile is not None else None))
    return assess_acid_anion_order(facts)


def _multiply_acid_component(mol, part, anion, count, decoration):
    """P-16 multiplication from chemical construction and separate annotations."""
    return multiply_acid_anion(
        anion, count, construction=part.construction,
        isotope_modified=any(mol.GetAtomWithIdx(idx).GetIsotope()
                             for idx in set(part.atoms) | {part.bridge_oxygen}
                             if idx is not None),
        stereochemically_modified=bool(decoration.stereo_atoms or decoration.stereo_bonds))


def _typed_acid_parent_decoration(mol, Chem, part, profile, component_id):
    """Prove an acid parent's C/stereo text separately from its O2 suffix.

    The acid derivative producer owns the O-role suffix and its isotope
    primary claims. Here the same mapped parent writer is run with only its
    typed carbonyl-O suffix label cleared. Carbon/isotope and stereo claims
    must agree with a separately evaluated original-graph stereo plan, and
    the typed oxygen suffix operation must reproduce the acid component text.
    """
    component = part.construction
    if component is None:
        return None
    base_mol = Chem.Mol(mol)
    carbonyl = profile.carbonyl_oxygen
    if base_mol.GetAtomWithIdx(carbonyl).GetIsotope():
        base_mol.GetAtomWithIdx(carbonyl).SetIsotope(0)
    base_component = replace(
        component,
        local_isotope_atoms=frozenset(),
        isotope_role_evidence=None,
        isotope_role_evidence_mode="",
    )
    decoration = decorate_component_with_evidence(
        base_mol, Chem, base_component, component_id=component_id)
    if decoration is None:
        return None
    locants = dict(base_component.atom_locants)
    from .component_mapping import _mapped_stereochemistry_plan
    original_plan = _mapped_stereochemistry_plan(mol, Chem, base_component, locants)
    base_plan = _mapped_stereochemistry_plan(base_mol, Chem, base_component, locants)
    if original_plan is None or base_plan is None:
        return None
    original_atoms = tuple(sorted(original_plan[1]))
    original_bonds = tuple(sorted(original_plan[2]))
    base_atoms = tuple(sorted(base_plan[1]))
    base_bonds = tuple(sorted(base_plan[2]))
    if (original_atoms != base_atoms or original_bonds != base_bonds
            or tuple(sorted((row.atom_id, row.component_locant, row.descriptor)
                            for row in decoration.stereo_atoms)) != original_atoms
            or tuple(sorted((row.atom_ids[0], row.atom_ids[1], row.component_locant,
                             row.descriptor) for row in decoration.stereo_bonds)) != original_bonds):
        return None
    suffix = part.suffix_oxygen_isotopes
    from .oxygen_isotope_suffixes import suffix_oxygen_descriptors
    composed = suffix_oxygen_descriptors(decoration.text, suffix) if suffix else decoration.text
    if not composed or composed != part.name:
        return None
    from .component_annotation_evidence import (
        MappedStereoAtomEvidence, MappedStereoBondEvidence,
    )
    return replace(
        decoration,
        stereo_atoms=tuple(MappedStereoAtomEvidence(*row)
                           for row in original_plan[1]),
        stereo_bonds=tuple(MappedStereoBondEvidence((a,b),locant,descriptor)
                           for a,b,locant,descriptor in original_plan[2]),
    )


def _anion_order_key(name, locants=(), *, part=None, mol=None, ordinary_name=None):
    """Ordinary citation order with an explicitly assessed isotope inference.

    Alphabetical seniority comes from the component's ordinary chemical
    letters and italic terms. Isotope and stereochemical descriptors are not
    part of that key. The modified-first preference in P-82.2.2.1 concerns
    substituents. Its extension to tied ordinary acid-anion words is an
    implementation inference recorded separately on the decision, with exact
    PIN expectations still pending. Then the anion's own locants and its
    polyol attachment-site locants are considered. Literal text is
    a final deterministic tie-break, not a chemical seniority criterion.
    """
    alpha = prefix_alphabetization_key(ordinary_name or name)
    modified = bool(part is not None and mol is not None and _anion_is_modified(part, mol))
    return alpha[0], alpha[1], 0 if modified else 1, alpha[2], tuple(locants), name


def _swap_two_ester_acid_assignments(mol, Chem, groups):
    """Prove that exchanging two complete ester units creates the same graph.

    The bridge oxygen is part of the ester unit for isotope-profile purposes.
    Swapping carbonyl-C/bridge-O bonds alone leaves a labelled bridge oxygen at
    its original alcohol site and would split a uniform O2 profile. Exchange
    the bridge-O/alcohol-C bonds instead, so both oxygens and the whole acyl
    component move together between alcohol sites.
    """
    if len(groups)!=2:return False
    bridges=[group.ligand for group in groups]
    if bridges[0]==bridges[1]:return False
    rw=Chem.RWMol(mol)
    hosts=[]
    for group,bridge in zip(groups,bridges):
        neighbors=[neighbor.GetIdx() for neighbor in mol.GetAtomWithIdx(bridge).GetNeighbors()
                   if neighbor.GetIdx()!=group.center and neighbor.GetAtomicNum()==6]
        if len(neighbors)!=1:return False
        hosts.append(neighbors[0])
    if hosts[0]==hosts[1]:return False
    if any(rw.GetBondBetweenAtoms(bridge,host) is None
           for bridge,host in zip(bridges,hosts)):
        return False
    rw.RemoveBond(bridges[0],hosts[0]);rw.RemoveBond(bridges[1],hosts[1])
    rw.AddBond(bridges[0],hosts[1],Chem.BondType.SINGLE)
    rw.AddBond(bridges[1],hosts[0],Chem.BondType.SINGLE)
    swapped=rw.GetMol()
    try:Chem.SanitizeMol(swapped)
    except (ValueError,RuntimeError):return False
    from .stereochemistry import assigned_atom_stereochemistry, _indexed_bond_stereochemistry
    original_stereo=(tuple(assigned_atom_stereochemistry(mol,Chem,atom_order='idx',include_pseudo=True)),
                     tuple(_indexed_bond_stereochemistry(mol,Chem)))
    swapped_stereo=(tuple(assigned_atom_stereochemistry(swapped,Chem,atom_order='idx',include_pseudo=True)),
                    tuple(_indexed_bond_stereochemistry(swapped,Chem)))
    if original_stereo!=swapped_stereo:return False
    try:
        # Caller atom-map labels identify original atoms, not chemically
        # inequivalent ester sites. Compare only chemical identity on scratch
        # copies, preserving the original-index stereo check above.
        identities = (Chem.Mol(mol), Chem.Mol(swapped))
        for identity in identities:
            for atom in identity.GetAtoms():
                atom.SetAtomMapNum(0)
        original_smiles=Chem.MolToSmiles(identities[0],canonical=True,isomericSmiles=True)
        swapped_smiles=Chem.MolToSmiles(identities[1],canonical=True,isomericSmiles=True)
    except (ValueError,RuntimeError):return False
    return original_smiles==swapped_smiles


def _tree_diameter_paths(adjacency):
    """Return every orientation of every longest path in a carbon tree."""
    leaves=sorted(i for i,neighbors in adjacency.items() if len(neighbors)<=1)
    paths=set()
    maximum=0
    for i,start in enumerate(leaves):
        for end in leaves[i+1:]:
            found=set()
            def find(current,target,previous,path):
                if current==target:
                    found.add(tuple(path));return
                for neighbor in adjacency[current]-({previous} if previous is not None else set()):
                    find(neighbor,target,current,path+[neighbor])
            find(start,end,None,[start])
            if found:
                path=next(iter(found))
                if len(path)>maximum:
                    maximum=len(path);paths=set()
                if len(path)==maximum:
                    paths.add(path);paths.add(tuple(reversed(path)))
    return tuple(sorted(paths))


def _simple_offchain_branches(mol, framework, path, attachment_atoms, adjacency):
    """Map pendant methyl/ethyl branches off a candidate principal chain."""
    path_set=set(path);remaining=set(framework)-path_set;branches=[]
    while remaining:
        seed=min(remaining);component=set();stack=[seed]
        while stack:
            idx=stack.pop()
            if idx in component:continue
            component.add(idx);remaining.discard(idx)
            stack.extend(adjacency[idx]-path_set-component)
        hosts={neighbor for idx in component for neighbor in adjacency[idx] if neighbor in path_set}
        if len(hosts)!=1 or component & set(attachment_atoms):return None
        host=next(iter(hosts))
        if len(component) not in {1,2}:return None
        if sum(len(adjacency[idx]&component) for idx in component)//2!=len(component)-1:
            return None
        if any(mol.GetAtomWithIdx(idx).GetIsotope()
               or int(mol.GetAtomWithIdx(idx).GetChiralTag()) for idx in component):
            return None
        if any(mol.GetAtomWithIdx(idx).IsInRing()
               or mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in component):
            return None
        if len(component)==2:
            degrees={idx:len(adjacency[idx]&component) for idx in component}
            if sorted(degrees.values())!=[1,1]:return None
        branches.append((host,tuple(sorted(component)),
                         'methyl' if len(component)==1 else 'ethyl'))
    return tuple(branches)


def _symmetric_aza_ester_residue(mol, Chem, groups, alcohol_atoms, parts,
                               anion, decoration, centers, atoms):
    """Join two identical acid anions through an N-centred organyl."""
    if len(groups) != 2 or any(
            atom.GetIsotope() or atom.GetFormalCharge() or atom.GetChiralTag()
            or atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()):
        return None
    from types import SimpleNamespace
    from .naming_methods import _mapped_connector_name

    oxygen = {g.ligand for g in groups}
    scope = frozenset(alcohol_atoms - oxygen)
    nitrogens = [i for i in scope if mol.GetAtomWithIdx(i).GetAtomicNum() == 7]
    if len(nitrogens) != 1 or any(
            mol.GetAtomWithIdx(i).IsInRing() or mol.GetAtomWithIdx(i).GetAtomicNum() not in {6, 7, 8}
            for i in scope):
        return None
    nitrogen = nitrogens[0]
    n_atom = mol.GetAtomWithIdx(nitrogen)
    if n_atom.GetDegree() != 3 or n_atom.GetTotalValence() != 3:
        return None
    ports = []
    for group in groups:
        ends = [a.GetIdx() for a in mol.GetAtomWithIdx(group.ligand).GetNeighbors()
                if a.GetIdx() != group.center and a.GetAtomicNum() == 6]
        if len(ends) != 1 or ends[0] not in scope:
            return None
        ports.append(ends[0])
    if ports[0] == ports[1]:
        return None
    path = Chem.GetShortestPath(mol, *ports)
    if (len(path) < 5 or len(path) % 2 == 0 or path[len(path) // 2] != nitrogen
            or not set(path) <= scope
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6
                   or mol.GetAtomWithIdx(i).GetTotalNumHs(includeNeighbors=True) != 2
                   for i in path if i != nitrogen)
            or any(mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble() != 1
                   for a, b in zip(path, path[1:]))):
        return None
    units = tuple(SimpleNamespace(root=g.ligand, connector=i)
                  for g, i in zip(groups, ports, strict=True))
    organyl = _mapped_connector_name(mol, Chem, scope, units)
    multiplication = _multiply_acid_component(mol, parts[0], anion, 2, decoration)
    if not organyl or multiplication is None:
        return None
    roles = tuple((f'acid:{j}', part.atoms, part.atom_locants)
                  for j, part in enumerate(parts, 1))
    return MethodDecision(organyl + ' ' + multiplication.text, 'functional-class',
        centers, atoms, 'P-65.6.3.2.1/P-65.6.3.3.3.1',
        component_roles=roles + (('alcohol', frozenset(alcohol_atoms), ()),))


def polyol_ester_decision(mol, Chem, groups, atoms, centers, *, allow_multiple_stereocenters=False):
    """Name esters at distinct sites of one mapped polyol component (P-65.6.3.3.3)."""
    # Runtime import avoids the acid-derivative module cycle while preserving
    # AcidComponent ownership, including isotope-role extensions.
    from .acid_derivative_methods import _acid_component, _anion
    from .polyol_ester_isotopes import (
        ester_oxygen_profile, supported_uniform_polyol_oxygen_profiles,
    )
    if len(groups)<2:return None
    allow_mapped_chain = len(groups) == 2 and not mol.GetRingInfo().NumRings()
    allow_mapped_ring = len(groups) == 2 and all(
        g.owner >= 0 and mol.GetAtomWithIdx(g.owner).IsInRing() for g in groups)
    parts=[]
    ester_oxygens=[]
    oxygen_profiles=[]
    for group in groups:
        oxygen=group.ligand
        if oxygen in ester_oxygens or mol.GetAtomWithIdx(oxygen).GetAtomicNum()!=8:
            return None
        ester_oxygens.append(oxygen)
        part=_acid_component(mol,Chem,group.center,oxygen,allow_mapped_ring=allow_mapped_ring,
                             allow_mapped_chain=allow_mapped_chain)
        if part is None:return None
        profile=ester_oxygen_profile(mol,part.carbonyl_oxygen,oxygen)
        if profile is None:return None
        oxygen_profiles.append(profile)
        parts.append(part)
    if len(parts) == 4 and mol.GetRingInfo().NumRings() and all(
            part.name == 'acetic acid' and part.identity == parts[0].identity for part in parts):
        from .acetylated_glycosides import acetylated_glycoside_decision

        decision = acetylated_glycoside_decision(mol, Chem, groups, atoms, centers, parts)
        if decision is not None:
            return decision
    has_oxygen_labels=any(profile.suffix_oxygen_isotopes for profile in oxygen_profiles)
    profile_scope=supported_uniform_polyol_oxygen_profiles(oxygen_profiles)
    oxygen_component_case=profile_scope and has_oxygen_labels
    permitted_oxygen_atoms={idx for profile in oxygen_profiles
                            for idx in (profile.carbonyl_oxygen,profile.bridge_oxygen)}
    if any(atom.GetIsotope() and atom.GetAtomicNum() not in {6, 8}
           for atom in mol.GetAtoms()):
        return None
    if any(atom.GetIsotope() and atom.GetAtomicNum()==8
           and atom.GetIdx() not in permitted_oxygen_atoms for atom in mol.GetAtoms()):
        return None
    if not allow_multiple_stereocenters and sum(bool(int(atom.GetChiralTag())) for atom in mol.GetAtoms()) > 1:
        return None
    from .stereochemistry import _indexed_bond_stereochemistry
    bond_stereochemistry = _indexed_bond_stereochemistry(mol, Chem)
    # Assigned acid-arm E/Z is written by the mapped component producer.
    # Explicit unknown markers need a different writer, not a guessed label.
    if any(bond.GetStereo() == Chem.BondStereo.STEREOANY
           or bond.GetBondDir() == Chem.BondDir.UNKNOWN
           for bond in mol.GetBonds()):
        return None
    if oxygen_component_case:
        # Each acid-derived ester unit has either both O roles natural or both
        # uniformly 17O/18O. The suffix count is local to that component; this
        # does not transfer P-82.6.4 role markers to the shared organyl.
        oxygen_atoms=[atom for atom in mol.GetAtoms() if atom.GetAtomicNum()==8]
        if (any(atom.GetIsotope() and atom.GetIdx() not in permitted_oxygen_atoms
                for atom in oxygen_atoms)
                ):
            return None
        # The local O2 descriptor is complete only for an acid-derived unit
        # whose oxygen population is exactly this ester carbonyl/bridge pair.
        # Do not treat an unlabelled ether, hydroxy, or other acid oxygen as
        # covered by the pair's uniformity.
        for part,profile in zip(parts,oxygen_profiles):
            if profile.uniform_mass is None:
                continue
            part_oxygens={idx for idx in part.atoms
                          if mol.GetAtomWithIdx(idx).GetAtomicNum()==8}
            if part_oxygens!={profile.carbonyl_oxygen}:
                return None
    elif has_oxygen_labels:
        # Partial-role, mixed-role, and unlike-mass unit grammars remain
        # pending for a shared multivalent organyl component.
        return None
    # P-82.3.2 orders nuclides by increasing mass only when they occupy the
    # same place in the name. We do not use a text tie-break or site locants to
    # order otherwise-identical ester units bearing unlike uniform O masses.
    ordinary_parts = _ordinary_acid_components(mol,Chem,groups,allow_mapped_ring=allow_mapped_ring,
                                               allow_mapped_chain=allow_mapped_chain)
    if ordinary_parts is None:return None
    base_identities={}
    for part,profile,ordinary in zip(parts,oxygen_profiles,ordinary_parts):
        if profile.uniform_mass:
            base=ordinary[0]
            base_identities.setdefault(base,set()).add(profile.uniform_mass)
    if any(len(masses)>1 for masses in base_identities.values()):
        return None
    # P-14.5 uses the ordinary component name for citation order. If two
    # components have the same ordinary name/constitution but different
    # isotope or stereo annotations, do not use their decorated spelling as
    # a surrogate ordering rule; identical annotations are combined below.
    for left_index, left in enumerate(parts):
        for right_index in range(left_index + 1, len(parts)):
            right = parts[right_index]
            if (ordinary_parts[left_index][:2] == ordinary_parts[right_index][:2]
                    and (left.identity != right.identity or left.name != right.name)):
                from .stereochemistry import (
                    assigned_atom_stereochemistry, _indexed_bond_stereochemistry,
                )
                stereo_atoms = dict(assigned_atom_stereochemistry(
                    mol,Chem,atom_order="idx",include_pseudo=True))
                stereo_bonds = _indexed_bond_stereochemistry(mol,Chem)
                left_stereo = (
                    tuple(sorted((idx, descriptor) for idx, descriptor in stereo_atoms.items()
                                 if idx in left.atoms)),
                    tuple(sorted((a,b,descriptor) for a,b,descriptor in stereo_bonds
                                 if {a,b} <= left.atoms)),
                )
                right_stereo = (
                    tuple(sorted((idx, descriptor) for idx, descriptor in stereo_atoms.items()
                                 if idx in right.atoms)),
                    tuple(sorted((a,b,descriptor) for a,b,descriptor in stereo_bonds
                                 if {a,b} <= right.atoms)),
                )
                if left_stereo != right_stereo:
                    return None
                # Retain the modified-before-unmodified convention only as
                # an inference: P-82.2.2.1 states it for substituents, and its
                # extension to acid-anion words remains unproved (audit222).
                # No order is proposed for two unlike modified versions.
                if (_anion_is_modified(left,mol)
                        == _anion_is_modified(right,mol)):
                    return None
    acid_decorations=[]
    for index,(part,profile) in enumerate(zip(parts,oxygen_profiles),1):
        decoration=_typed_acid_parent_decoration(
            mol,Chem,part,profile,f"polyol-acid-parent:{index}")
        if decoration is None:
            return None
        parent_c_labels={idx for idx in part.construction.atoms
                         if mol.GetAtomWithIdx(idx).GetAtomicNum()==6
                         and mol.GetAtomWithIdx(idx).GetIsotope()}
        claimed_c_labels={row.atom_id for row in decoration.isotope_record.isotope_atoms
                          if row.element=="C"}
        if claimed_c_labels!=parent_c_labels:
            return None
        if any(row.element not in {"C"} for row in decoration.isotope_record.isotope_atoms):
            return None
        acid_decorations.append(decoration)
    acid_decorations=tuple(acid_decorations)
    # Join the existing component-local stereo facts to whole-molecule
    # admission. Every original assigned bond must be represented once by
    # one acid construction, with its original configuration. This excludes
    # missing/duplicate claims and stereo outside the saturated polyol scope.
    expected_bonds = Counter((min(a, b), max(a, b), descriptor)
                             for a, b, descriptor in bond_stereochemistry)
    claimed_bonds = Counter(
        (min(row.atom_ids), max(row.atom_ids), row.descriptor)
        for decoration in acid_decorations for row in decoration.stereo_bonds
    )
    if claimed_bonds != expected_bonds:
        return None
    first=parts[0]
    same_acid=all(part.identity==first.identity and part.name==first.name for part in parts[1:])
    anions=[_anion(part.name) for part in parts]
    if any(not name for name in anions):return None
    # If distinct component graphs collapse to one rendered citation, their
    # difference cannot be represented safely in the mixed-anion name.
    for i,left in enumerate(parts):
        for right,anion_name in zip(parts[i+1:],anions[i+1:]):
            if (left.identity!=right.identity and left.name==right.name and
                    anions[i]==anion_name):
                return None
    acid_atoms=set()
    for part in parts:
        if acid_atoms & part.atoms:return None
        acid_atoms.update(part.atoms)
    alcohol_atoms=set(atoms)-acid_atoms
    if not set(ester_oxygens)<=alcohol_atoms or acid_atoms|alcohol_atoms!=set(atoms):
        return None
    framework=alcohol_atoms-set(ester_oxygens)
    if same_acid and not oxygen_component_case and not bond_stereochemistry and any(
            mol.GetAtomWithIdx(i).GetAtomicNum() == 7 for i in framework):
        return _symmetric_aza_ester_residue(mol, Chem, groups, alcohol_atoms, parts,
            anions[0], acid_decorations[0], centers, atoms)
    oxo = []
    for i in framework:
        atom = mol.GetAtomWithIdx(i)
        if (atom.GetAtomicNum() == 8 and atom.GetDegree() == 1
                and not atom.GetFormalCharge() and not atom.GetIsotope()
                and atom.GetBonds()[0].GetBondTypeAsDouble() == 2
                and atom.GetNeighbors()[0].GetAtomicNum() == 6
                and atom.GetNeighbors()[0].GetIdx() in framework):
            oxo.append((i, atom.GetNeighbors()[0].GetIdx()))
    if oxo:
        if not same_acid or len(oxo) > 2:
            return None
        framework -= {i for i, _ in oxo}
    if not framework or any(mol.GetAtomWithIdx(i).GetAtomicNum()!=6 for i in framework):
        return None
    if (not oxygen_component_case and
            any(has_annotations(mol,{i}) for i in set(ester_oxygens))):
        return None
    # Every ester oxygen is terminal on the alcoholic side and has exactly one
    # attachment to the carbon framework.
    attachment_atoms=[]
    for oxygen in ester_oxygens:
        neighbors=[n.GetIdx() for n in mol.GetAtomWithIdx(oxygen).GetNeighbors()
                   if n.GetAtomicNum()>1 and n.GetIdx()!=next(
                       g.center for g in groups if g.ligand==oxygen)]
        if len(neighbors)!=1 or neighbors[0] not in framework:return None
        if sum(n.GetIdx() in framework for n in mol.GetAtomWithIdx(oxygen).GetNeighbors())!=1:
            return None
        if mol.GetBondBetweenAtoms(oxygen,neighbors[0]).GetBondTypeAsDouble()!=1:return None
        attachment_atoms.append(neighbors[0])
    # The supported residue is a simple saturated carbon chain/monocycle or a
    # six-carbon benzene ring. The aromatic option is restricted to the
    # source-backed divalent phenylene organyl below.
    adjacency={i:set() for i in framework}
    framework_bonds=[]
    for i in framework:
        for neighbor in mol.GetAtomWithIdx(i).GetNeighbors():
            j=neighbor.GetIdx()
            if j in framework:
                bond=mol.GetBondBetweenAtoms(i,j)
                if i<j:framework_bonds.append(bond)
                adjacency[i].add(j)
    seen=set();stack=[next(iter(framework))]
    while stack:
        i=stack.pop()
        if i in seen:continue
        seen.add(i);stack.extend(adjacency[i]-seen)
    if seen!=framework:return None
    edge_count=sum(map(len,adjacency.values()))//2
    cyclic=edge_count==len(framework) and all(len(v)==2 for v in adjacency.values())
    linear=edge_count==len(framework)-1 and all(len(v)<=2 for v in adjacency.values())
    carbon_tree=edge_count==len(framework)-1
    aromatic_ring=(cyclic and len(framework)==6 and
                   all(mol.GetAtomWithIdx(i).GetAtomicNum()==6 and mol.GetAtomWithIdx(i).GetIsAromatic()
                       for i in framework) and
                   len(framework_bonds)==6 and all(bond.GetIsAromatic() for bond in framework_bonds))
    saturated=(all(bond.GetBondTypeAsDouble()==1 for bond in framework_bonds) and
               not any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in framework))
    branched_tree=carbon_tree and not linear and saturated
    if not (linear and saturated or branched_tree or cyclic and (aromatic_ring or saturated)):return None
    if oxo and (not linear or not saturated
                or any(len(adjacency[host]) != 2 or host in attachment_atoms for _, host in oxo)):
        return None
    # Each framework carbon must accommodate its mapped bonds and implicit or
    # explicit hydrogen count in ordinary tetravalence.
    for carbon in framework:
        if mol.GetAtomWithIdx(carbon).GetTotalValence()>4:return None
    if cyclic and any(not mol.GetAtomWithIdx(i).IsInRing() for i in framework):return None
    if (linear or branched_tree) and any(mol.GetAtomWithIdx(i).IsInRing() for i in framework):return None
    if len(framework) not in ROOTS:return None
    if aromatic_ring and len(groups)!=2:return None

    if cyclic:
        paths=[]
        # Every ring atom can be the numbered first atom. Both traversal
        # directions are included before comparing attachment, isotope, and
        # original-graph stereo locants, so SMILES atom order cannot select a
        # cyclic orientation.
        for start in sorted(framework):
            for nxt in sorted(adjacency[start]):
                path=[start,nxt]
                while len(path)<len(framework):
                    options=adjacency[path[-1]]-{path[-2]}
                    next_atom=next(iter(options-{start}),None)
                    if next_atom is None:break
                    path.append(next_atom)
                if len(path)==len(framework):paths.append(tuple(path))
    elif branched_tree:
        paths=list(_tree_diameter_paths(adjacency))
        if not paths:return None
        max_length=max(len(path) for path in paths)
        paths=[path for path in paths if len(path)==max_length]
        # P-46 compares principal-chain length before free-valence locants.
        # Do not choose a shorter chain to make an off-chain ester site fit.
        if any(not set(attachment_atoms)<=set(path) for path in paths):return None
    else:
        if len(framework)==1:
            paths=[(next(iter(framework)),)]
        else:
            ends=sorted(i for i in framework if len(adjacency[i])==1)
            if len(ends)!=2:return None
            paths=[]
            for start in ends:
                path=[start]
                while len(path)<len(framework):
                    options=adjacency[path[-1]]-set(path)
                    if not options:break
                    path.append(min(options))
                if len(path)==len(framework):paths.append(tuple(path))
    candidates=[]
    from .parent_annotations import isotope_seniority, stereo_numbering
    for path in paths:
        mapping={idx:str(pos) for pos,idx in enumerate(path,1)}
        attach_locs=tuple(sorted(int(mapping[i]) for i in attachment_atoms))
        branch_records=(_simple_offchain_branches(mol,framework,path,
                         attachment_atoms,adjacency) if branched_tree else ())
        # The paths here are all globally longest parent-chain candidates. If
        # even one such candidate has an off-chain component outside the
        # proved branch grammar, do not silently select a renderable sibling.
        if branch_records is None:return None
        branch_locs=tuple(sorted(int(mapping[host]) for host,_,_ in branch_records))
        branch_count=len(branch_records)
        by_prefix={}
        for host,branch_atoms,prefix in branch_records:
            by_prefix.setdefault(prefix,[]).append(int(mapping[host]))
        branch_alpha=tuple((prefix_alphabetization_key(prefix),tuple(sorted(locs)))
                           for prefix,locs in sorted(by_prefix.items(),
                               key=lambda item:prefix_alphabetization_key(item[0])))
        # If both traversal directions give the same attachment locant set,
        # P-14 alphabetical order assigns the lower locants to the first anion.
        species_parts={}
        for part in parts:
            species_parts.setdefault((part.identity,part.name,_anion(part.name)),part)
        species_keys=sorted(species_parts,
                            key=lambda item:_anion_order_key(
                                item[2],part=species_parts[item],mol=mol,
                                ordinary_name=ordinary_parts[parts.index(species_parts[item])][2]))
        alpha_site_locs=tuple(tuple(sorted(int(mapping[site]) for part,site in zip(parts,attachment_atoms)
                                           if (part.identity,part.name,_anion(part.name))==key))
                              for key in species_keys)
        root=('cyclo' if cyclic else '')+ROOTS[len(path)]+('ane')
        valence=MULT.get(len(groups))
        if valence is None:return None
        if aromatic_ring:
            if len(attach_locs)!=2 or len(set(attach_locs))!=2:return None
            organyl=f"{','.join(map(str,sorted(attach_locs)))}-phenylene"
        elif len(path)==1:
            # P-29.6.1 retains methylene for the divalent CH2 residue, used
            # explicitly in the mixed-acid ester example at P-65.6.3.3.3.2.
            if len(groups)!=2:return None
            organyl='methylene'
        else:
            organyl=f"{root}-{','.join(map(str,attach_locs))}-{valence}yl"
        if branched_tree:
            # P-29.3.2.2 uses systematic locanted multivalent alkane names;
            # P-46 selects the parent chain before its free-valence and branch
            # locants. Branch atoms are mapped to distinct internal labels so
            # original atom ownership remains complete without assigning them
            # false skeletal locants.
            prefix_rows=tuple((mapping[host],prefix)
                              for host,_,prefix in branch_records)
            parent=f"{root}-{','.join(map(str,attach_locs))}-{valence}yl"
            # The pendant methyl/ethyl atoms are owned by their embedded
            # branch prefixes. They do not receive invented skeletal locants:
            # only principal-chain atoms have parent locants.
            embedded=frozenset(idx for _host,branch_atoms,_prefix in branch_records
                               for idx in branch_atoms)
            component=MappedComponent(parent,tuple(mapping.items()),
                frozenset(path),prefixes=prefix_rows,embedded_atoms=embedded,
                systematic_parent=True)
            # For the fixed atom set, P-46 chain selection is: chain length
            # (already globally maximal), lowest free-valence locants, most
            # substituents, lowest substituent locants, then low locants for
            # substituents in citation order. Acid-arm alphabetical placement
            # applies only after those parent-chain criteria tie.
        else:
            component=MappedComponent(organyl,tuple(mapping.items()),frozenset(framework),
                prefixes=tuple((mapping[host], 'oxo') for _, host in oxo),
                embedded_atoms=frozenset(i for i, _ in oxo), systematic_parent=True)
        # P-82.5.2 first minimizes all modified positions together, then gives
        # the heavier nuclide the lower locant. Sorting (mass, locant) would
        # incorrectly give the lighter isotope priority on a symmetric frame.
        isotope_key=isotope_seniority(mol,mapping)
        stereo_key=stereo_numbering(mol,mapping)
        if isotope_key is None or stereo_key is None:return None
        # Attachment locants are selected first; isotope and original-graph CIP
        # locants only break chemically equivalent numbering orientations.
        if branched_tree:
            parent_chain_key=(attach_locs,-branch_count,branch_locs,branch_alpha,
                              alpha_site_locs,isotope_key,stereo_key)
            organyl_locants=tuple(mapping.items())
        else:
            oxo_locants = tuple(sorted(int(mapping[host]) for _, host in oxo))
            parent_chain_key=(attach_locs,alpha_site_locs,oxo_locants,isotope_key,stereo_key)
            organyl_locants=tuple((idx,mapping[idx]) for idx in path)
        # Text is retained as a render fact only. It is never used to choose
        # between chemically tied parent mappings.
        legacy_text=decorate_component(mol,Chem,component)
        decoration=decorate_component_with_evidence(
            mol,Chem,component,component_id="polyol-organyl")
        if decoration is not None and decoration.evidence_scope!=component.atoms:
            decoration=None
        organyl_name=(legacy_text if decoration is not None
                      and decoration.text==legacy_text else "")
        carbon_isotope_ids={atom.GetIdx() for atom in mol.GetAtoms()
                            if atom.GetAtomicNum()==6 and atom.GetIsotope()}
        acid_parent_atoms=set().union(*(
            set(part.construction.atoms) for part in parts
            if part.construction is not None))
        eligible_annotated_atoms=set(component.atoms)|acid_parent_atoms
        if decoration is not None:
            if not carbon_isotope_ids<=eligible_annotated_atoms or (not allow_multiple_stereocenters and len(decoration.stereo_atoms)>1):
                decoration=None
                organyl_name=""
            if any(int(atom.GetChiralTag()) and atom.GetIdx() not in component.atoms
                   and atom.GetIdx() not in acid_parent_atoms
                   for atom in mol.GetAtoms()):
                decoration=None
                organyl_name=""
            if any(not ({a,b}<=component.atoms or {a,b}<=acid_parent_atoms)
                   for a,b,_descriptor in _indexed_bond_stereochemistry(mol,Chem)):
                decoration=None
                organyl_name=""
        candidates.append((parent_chain_key,organyl_name,organyl_locants,decoration))
    if not candidates:return None
    best_key=min(row[0] for row in candidates)
    best_rows=[row for row in candidates if row[0]==best_key]
    if any(not row[1] or row[3] is None for row in best_rows):return None
    if len({row[1] for row in best_rows})!=1:return None
    # Keep every equivalent original-atom numbering. A representative is
    # used below only to serialize citations after tied maps have agreed.
    equivalent_organyl_atom_locants=tuple(sorted({
        tuple(sorted((int(atom),str(locant)) for atom,locant in row[2]))
        for row in best_rows
    }))
    _,organyl_name,organyl_locants,_representative_decoration=best_rows[0]
    organyl_decorations=tuple(row[3] for row in best_rows)
    chosen_mapping=dict(organyl_locants)
    anion_multiplications=[]
    if same_acid:
        multiplication=_multiply_acid_component(
            mol,first,anions[0],len(groups),acid_decorations[0])
        if multiplication is None:return None
        anion_multiplications.append(multiplication)
        acid_citation=multiplication.text
    else:
        # P-65.6.3.3.3.2: sort complete anion citations and place the site
        # locants before each citation. Only identical mapped acid components
        # may be combined; isotopically/configurationally distinct components
        # remain independent groups even if their underlying skeletons match.
        species={}
        for index,(part,anion_name,site) in enumerate(zip(parts,anions,attachment_atoms)):
            key=(part.identity,part.name)
            species.setdefault(key,{'anion':anion_name,'part':part,
                                    'ordinary':ordinary_parts[index],
                                    'sites':[]})['sites'].append(int(chosen_mapping[site]))
        for item in species.values():
            item['sites'].sort()
        ordered=sorted(species.values(),key=lambda item:_anion_order_key(
            item['anion'],item['sites'],part=item['part'],mol=mol,
            ordinary_name=item['ordinary'][2]))
        citations=[]
        for item in ordered:
            site_locs=sorted(item['sites'])
            citation=item['anion']
            count=len(site_locs)
            if count>1:
                index=next(i for i,part in enumerate(parts) if part is item['part'])
                multiplication=_multiply_acid_component(
                    mol,item['part'],citation,count,acid_decorations[index])
                if multiplication is None:return None
                anion_multiplications.append(multiplication)
                citation=multiplication.text
            # P-65.6.3.3.3.2 says locants are added when required. If every
            # ester site is geminal at one numbered carbon, the multivalent
            # organyl already specifies the shared site (as in methylene).
            omit_mixed_site_locs=(not same_acid and len(groups)==2 and
                _swap_two_ester_acid_assignments(mol,Chem,groups))
            site_prefix=(','.join(map(str,site_locs))+'-'
                         if len(set(attach_locs))>1 and not omit_mixed_site_locs else '')
            if count==1 and _anion_needs_enclosure(
                    citation, outer_locant=bool(site_prefix)):
                citation=enclose_name_unit(citation)
            citations.append(site_prefix+citation)
        acid_citation=' '.join(citations)
    roles=tuple((f'acid:{j+1}',part.atoms,part.atom_locants)
                for j,part in enumerate(parts))
    roles+=(('alcohol',frozenset(alcohol_atoms),organyl_locants),)
    mapped=tuple(loc for part in parts for loc in part.atom_locants)+organyl_locants
    rule='P-65.6.3.2.1/P-65.6.3.3.3.1' if same_acid else 'P-65.6.3.2.2/P-65.6.3.3.3.2'
    isotope_components=[row.isotope_record for row in organyl_decorations[:1]]
    for unit_index,(part,anion,profile,acid_decoration) in enumerate(
            zip(parts,anions,oxygen_profiles,acid_decorations),1):
        if (part.carbonyl_oxygen!=profile.carbonyl_oxygen
                or part.bridge_oxygen!=profile.bridge_oxygen
                or tuple(part.suffix_oxygen_isotopes)!=tuple(profile.suffix_oxygen_isotopes)):
            return None
        scope_ids=frozenset(set(part.atoms)|{profile.bridge_oxygen})
        locant_by_atom=dict(part.atom_locants)
        if not set(part.construction.atoms)<=set(locant_by_atom):return None
        all_locants=tuple((idx,locant_by_atom.get(idx)) for idx in sorted(part.atoms))
        all_locants+=((profile.bridge_oxygen,None),)
        carbon_claims={row.atom_id:row for row in acid_decoration.isotope_record.isotope_atoms
                       if row.element=="C"}
        expected_carbons={idx for idx in part.construction.atoms
                          if mol.GetAtomWithIdx(idx).GetAtomicNum()==6
                          and mol.GetAtomWithIdx(idx).GetIsotope()}
        if set(carbon_claims)!=expected_carbons:return None
        if any(row.element!="C" for row in acid_decoration.isotope_record.isotope_atoms):
            return None
        oxygen_claims={idx for idx in (profile.carbonyl_oxygen,profile.bridge_oxygen)
                       if mol.GetAtomWithIdx(idx).GetIsotope()}
        if any(mol.GetAtomWithIdx(idx).GetIsotope()
               and mol.GetAtomWithIdx(idx).GetAtomicNum()==8
               and idx not in oxygen_claims for idx in scope_ids):
            return None
        isotope_ids=set(carbon_claims)|oxygen_claims
        if any(mol.GetAtomWithIdx(idx).GetIsotope() for idx in scope_ids
               if idx not in isotope_ids):
            return None
        if isotope_ids:
            printed={idx:row.printed_locant for idx,row in carbon_claims.items()}
            printed.update({idx:None for idx in oxygen_claims})
            try:
                isotope_components.append(component_record_from_atom_map(
                    mol,
                    component_id=f"polyol-acid-unit:{unit_index}",
                    component_text=anion,
                    atom_ids=scope_ids,
                    atom_locants=all_locants,
                    descriptor_atom_ids=tuple(sorted(isotope_ids)),
                    printed_locants=printed,
                ))
            except (TypeError,ValueError,IndexError,KeyError):
                return None
    has_isotope_labels=any(atom.GetIsotope() for atom in mol.GetAtoms())
    isotope_coverage=(validate_isotope_component_coverage(mol,isotope_components)
                      if has_isotope_labels else None)
    if has_isotope_labels and (isotope_coverage is None or not isotope_coverage.complete):
        return None
    citation_order_evidence = _acid_anion_order_assessment(mol, parts, ordinary_parts)
    preference_issues = (() if citation_order_evidence.status == 'source_closed' else (
        SelectionLimitation('acid-derivative-method',
                            '/'.join(citation_order_evidence.rules),
                            citation_order_evidence.reason),))
    return PolyolEsterMethodDecision(organyl_name+' '+acid_citation,'functional-class',centers,atoms,
        rule,mapped,component_roles=roles,
        isotope_components=tuple(isotope_components),isotope_coverage=isotope_coverage,
        equivalent_organyl_atom_locants=equivalent_organyl_atom_locants,
        organyl_decorations=organyl_decorations,
        acid_decorations=acid_decorations,
        anion_multiplications=tuple(anion_multiplications),
        citation_order_evidence=citation_order_evidence,
        preference_issues=preference_issues)
