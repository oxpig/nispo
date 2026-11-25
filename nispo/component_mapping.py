"""Input-atom bindings for component-local isotope and stereo assembly.

A component owns explicit atoms/locants and embeds independently decorated
branches. No annotation is inferred from a count or substring in the name.
Unsupported decorated branches decline the route instead of guessing order.
"""

from collections import defaultdict
from dataclasses import dataclass
import re

from .molecule_iteration import iter_atoms, iter_bonds
from .constants import ROOTS
from .graph_model import graph_view
from .formatting import complete_parent_name, format_prefixes, join_prefix_parent, enclose_name_unit, prefix_alphabetization_key
from .molecule_cache import mol_cache_dict, mol_cache_get, mol_cache_set, naming_cache_state
from .stereochemistry import (
    assigned_atom_stereochemistry,
    _indexed_bond_stereochemistry,
    _has_assigned_stereo,
)


@dataclass(frozen=True, slots=True)
class MappedComponent:
    parent: str
    atom_locants: tuple[tuple[int, str], ...]
    atoms: frozenset[int]
    prefixes: tuple[tuple[str, str], ...] = ()
    embedded_atoms: frozenset[int] = frozenset()
    # P-82.2.3: hydro operations precede the isotope descriptor of the
    # underlying hydride. New typed producers keep that boundary explicit.
    hydro_prefix: str = ""
    systematic_parent: bool = False
    nitrogen_site_locants: bool = False
    isotope_unlocanted_elements: tuple = ()
    prefix_mode: str = 'locanted'
    # The complete name already contains these typed characteristic units.
    # They still own original atoms, but their local isotope descriptors must
    # not be repeated on the skeletal parent (P-82.2.1).
    local_isotope_atoms: frozenset = frozenset()
    omit_methyl_prefix_locant: bool = False
    # Typed local isotope tokens that already own original atoms elsewhere in
    # the component map (for example terminal O/H suffix atoms). Entries are
    # (element symbol, mass number, component locant[, omit_single_count]).
    isotope_descriptor_parts: tuple = ()
    # Optional caller-proved nature for every already-rendered locanted
    # prefix. Entries are (component text, "simple" | "compound").
    # The opt-in formatter declines incomplete or contradictory metadata.
    prefix_natures: tuple[tuple[str, str], ...] | None = None
    # Optional source-proved citation order for the distinct locanted prefix
    # units. This is an exact permutation of rendered names, not a name-derived
    # priority. Non-locanted/legacy prefix modes must decline it.
    prefix_citation_order: tuple[str, ...] | None = None
    # Typed acid-local isotope roles can bind primary descriptor atoms and
    # secondary hydrogen-host references to the original graph. The validator
    # checks these facts but does not infer tokens from component text.
    isotope_role_evidence: object | None = None
    isotope_role_evidence_mode: str = ""
    # Proof that P-14.3.4.6 permits omission of prefix locants on a retained
    # acetic-acid/acetate component. The final writer revalidates the graph.
    retained_acid_locant_evidence: object | None = None
    # Source-typed non-lambda characteristic prefixes. Producers attach these
    # only at the mapped branch decision, after matching the source inventory
    # role to the actual emitted prefix text and original atom IDs.
    characteristic_prefix_roles: tuple = ()
    # Complete graph proof for the bounded functionalized carbon-chain branch
    # writer. Additive and empty for all existing component producers.
    functionalized_carbon_prefix_proofs: tuple = ()
    # Exact ownership returned by the top-level mapped-branch calls used for
    # locanted prefixes. Rows are (root, host, original atoms, emitted text).
    # This is additive evidence for consumers that must join graph roles to
    # the real writer; it is not reconstructed from the finished name.
    prefix_branch_ownership: tuple = ()
    # Typed ordinary amide isotope units retain a separate insertion boundary.
    amide_isotope_evidence: object | None = None
    # Source-validated complete stereo feature census for a bounded mapped
    # parent. Only this proof can authorize an explicit unknown Xi token.
    parent_annotation_proof: object | None = None
    # Component writers can supply proved compound pendant units while
    # characteristic prefixes keep their own established multiplication.
    compound_prefix_names: tuple[str, ...] = ()
    # P-83 is an explicit substance interpretation, never inferred from a
    # fixed-isotope molecule. The labelled-substance API validates this mode.
    isotope_labelling_kind: str = ""
    isotope_labelling_subset: bool = False
    # P-93.5.1.1 distinguishes native relative-ring flags from actual
    # stereogenic atoms. Only a complete bounded whole-graph classification
    # may omit a descriptor at a flagged nonstereogenic atom.
    nonstereogenic_atom_evidence: object | None = None
    # P-93: a complete original-graph assembly census owns cross-component
    # compound junction locants. Ordinary components retain their old path.
    assembly_stereo_evidence: object | None = None
    # P-82 joins require an original characteristic partition, ordinary
    # chemical-numbering frontier and complete actual descriptor ownership.
    assembly_isotope_evidence: object | None = None
    # P-28.6/P-29.3.5 original tree, complete path frontier and recursive
    # branch nature/ownership, revalidated before the source formatter runs.
    tree_assembly_evidence: object | None = None
    # A source-proved printed-locant decision shared with the hydride writer.
    # Original atom/bond locants remain available to numbering and evidence.
    locant_presentation: object | None = None
    # A joint suffix/prefix omission proof for the one-position .6 case.
    substitution_locant_presentation: object | None = None
    # Original characteristic-group atoms included implicitly in the named
    # parent, but owned outside this descriptor's map. Their H population
    # must still prevent a false P-82.6.1.3 completeness claim.
    isotope_hydrogen_context_atoms: frozenset[int] = frozenset()
    # An external C=N suffix has its own bond, distinct from the skeletal
    # atom carrying the suffix. Keep its written E/Z locant on that bond.
    suffix_stereo_locants: tuple[tuple[int, int, str], ...] = ()


def _valid_isotope_role_evidence(mol, component):
    evidence = component.isotope_role_evidence
    if evidence is None:
        return not component.isotope_role_evidence_mode
    from .acid_oxygen_isotope_methods import AcidOxygenIsotopeRecord

    if not isinstance(evidence, AcidOxygenIsotopeRecord):
        return False
    if (not isinstance(evidence.parts, tuple)
            or not isinstance(evidence.part_atom_ids, tuple)
            or not isinstance(evidence.owned_atom_ids, frozenset)
            or len(evidence.parts) != len(evidence.part_atom_ids)
            or not evidence.parts or not evidence.valid):
        return False
    if evidence.status not in {"source-rule", "source-rule-generalized-nuclide"}:
        return False
    if (not isinstance(component.local_isotope_atoms, frozenset)
            or component.local_isotope_atoms != evidence.owned_atom_ids
            or len(evidence.part_atom_ids) != len(set(evidence.part_atom_ids))):
        return False
    if component.isotope_role_evidence_mode == "component-parts":
        if component.isotope_descriptor_parts != evidence.parts:
            return False
    elif component.isotope_role_evidence_mode == "typed-suffix":
        # The systematic free-acid OH-hydrogen token is already emitted at
        # its typed suffix insertion point, rather than the parent descriptor
        # builder. Keep this narrow role form separate from component tokens.
        if (component.isotope_descriptor_parts
                or len(evidence.parts) != 1
                or evidence.parts[0][0] != "H"
                or evidence.parts[0][2:] != ("", True)
                or evidence.status != "source-rule-generalized-nuclide"):
            return False
    else:
        return False
    if not evidence.parts or len(evidence.parts) != len(evidence.part_atom_ids):
        return False
    for (symbol, mass, locant, omit_count), atom_id in zip(
            evidence.parts, evidence.part_atom_ids):
        if (not isinstance(atom_id, int) or isinstance(atom_id, bool)
                or atom_id < 0 or atom_id >= mol.GetNumAtoms()
                or symbol not in {"H", "O"}
                or not isinstance(mass, int) or isinstance(mass, bool) or mass <= 0
                or not isinstance(locant, str)
                or not isinstance(omit_count, bool)):
            return False
        atom = mol.GetAtomWithIdx(atom_id)
        if atom.GetSymbol() != symbol or atom.GetIsotope() != mass:
            return False
    h_atom_ids = {atom_id for (symbol, _, _, _), atom_id
                  in zip(evidence.parts, evidence.part_atom_ids) if symbol == "H"}
    references = evidence.hydrogen_host_references
    if (not isinstance(references, tuple)
            or len(references) != len(h_atom_ids)):
        return False
    referenced_hydrogens = set()
    for row in references:
        if not isinstance(row, tuple) or len(row) != 3:
            return False
        hydrogen_id, oxygen_id, role = row
        if (not isinstance(hydrogen_id, int) or isinstance(hydrogen_id, bool)
                or not isinstance(oxygen_id, int) or isinstance(oxygen_id, bool)
                or not 0 <= hydrogen_id < mol.GetNumAtoms()
                or not 0 <= oxygen_id < mol.GetNumAtoms()
                or role != "hydroxyl_oxygen" or hydrogen_id not in h_atom_ids
                or hydrogen_id in referenced_hydrogens):
            return False
        hydrogen = mol.GetAtomWithIdx(hydrogen_id)
        oxygen = mol.GetAtomWithIdx(oxygen_id)
        if (hydrogen.GetAtomicNum() != 1 or oxygen.GetAtomicNum() != 8
                or mol.GetBondBetweenAtoms(hydrogen_id, oxygen_id) is None):
            return False
        primary_row = evidence.parts[evidence.part_atom_ids.index(hydrogen_id)]
        if component.isotope_role_evidence_mode == "component-parts":
            expected_locator = f"{oxygen.GetIsotope()}O" if oxygen.GetIsotope() else "O"
            if primary_row[2] != expected_locator:
                return False
        referenced_hydrogens.add(hydrogen_id)
    return referenced_hydrogens == h_atom_ids


def nitrogen_locant(loc):
    """Keep N-primes and parent-occurrence primes on different symbols."""
    from .locant_order import nitrogen_site_display
    return nitrogen_site_display(loc)


def chemical_inventory(mol):
    """Stable isotope mass does not change functional-class recognition."""
    from .functional_inventory import functional_inventory
    from rdkit import Chem

    if not any(a.GetIsotope() for a in iter_atoms(mol)):
        return functional_inventory(mol)
    cached = mol_cache_get(mol, "mapped_chemical_inventory")
    if cached is not None:
        return cached
    copy = Chem.Mol(mol)
    for atom in copy.GetAtoms():
        atom.SetIsotope(0)
    result = functional_inventory(copy)
    return mol_cache_set(mol, "mapped_chemical_inventory", result)


def _locant_key(loc):
    site = re.fullmatch(r"([CNO])('*)(\d+)('*)", loc)
    if site:
        return (999, site[1], int(site[3]), len(site[4]), len(site[2]))
    m = re.fullmatch(r"(\d+)(.*)", loc)
    return (int(m[1]), m[2]) if m else (999, loc)


def hydrogen_bindings(mol, mapping):
    """Hydrogen isotope locants belong to their explicitly mapped host atom."""
    result = dict(mapping)
    view = graph_view(mol, None)
    for idx, locant in mapping.items():
        for neighbor in view.neighbors[idx]:
            if view.atomic_nums[neighbor] == 1 and view.isotopes[neighbor]:
                result[neighbor] = locant
    return result


def _hydrogen_citation_positions(mol, component, locants, entries):
    """P-82.2.5: cite one characteristic N site once, retaining the H count.

    Repeated numerical locants still locate individual isotope-H atoms. The
    characteristic-site operation is proved from original H--N bonds and an
    unambiguous conventional N locant, never from a descriptor substring.
    Different characteristic N hosts in one nuclide group retain their full
    locant series: suppressing those repetitions could lose the distribution
    of isotope H between sites. Synthetic descriptor parts are left alone.
    """
    positions = [locant for _atom_id, locant, _omit in entries]
    host_rows = []
    for atom_id, locant, _omit in entries:
        if not re.fullmatch(r"N(?:'+)?(?:\d+(?:'+)?)?", locant):
            # A further nonnumerical site would need its own distribution
            # proof before the single-N operation could be combined with it.
            if locant and not re.fullmatch(r"\d+[^,]*", locant):
                return positions
            continue
        if atom_id is None:
            return positions
        atom = mol.GetAtomWithIdx(atom_id)
        neighbors = tuple(atom.GetNeighbors())
        if (atom.GetAtomicNum() != 1 or atom.GetFormalCharge()
                or atom.GetNumRadicalElectrons() or len(neighbors) != 1):
            return positions
        host = neighbors[0]
        host_id = host.GetIdx()
        if (host.GetAtomicNum() != 7 or host_id not in component.atoms
                or locants.get(host_id) != locant
                or mol.GetBondBetweenAtoms(atom_id, host_id).GetBondTypeAsDouble() != 1):
            return positions
        # A conventional site must identify exactly one original nitrogen.
        if sum(mol.GetAtomWithIdx(idx).GetAtomicNum() == 7 and value == locant
               for idx, value in locants.items()) != 1:
            return positions
        host_rows.append((atom_id, host_id, locant))
    if not host_rows or len({host for _atom, host, _locant in host_rows}) != 1:
        return positions
    isotope_hydrogens = {atom for atom, _host, _locant in host_rows}
    result, cited_host = [], False
    for atom_id, locant, _omit in entries:
        if atom_id in isotope_hydrogens:
            if cited_host:
                continue
            cited_host = True
        result.append(locant)
    return result


def _complete_hydrogen_isotope_scope(mol, component, locants, entries, mass,
                                     skeletal_atoms):
    """Prove P-82.6.1.3 on the actual modified component, retaining H counts.

    Every hydrogen position must be occupied by this nuclide, and every
    descriptor atom must be one of those original terminal hydrogens. Merely
    having one isotope H at every heavy-atom position is insufficient. Implicit
    H, mixed masses, detached branches and synthetic parts cannot prove full
    occupancy. A source-proved assembly skeleton can retain its own scope.
    """
    scope = component.atoms if skeletal_atoms is None else skeletal_atoms
    context = component.isotope_hydrogen_context_atoms
    if (not isinstance(context, frozenset)
            or not context <= component.atoms | component.embedded_atoms
            or any(not isinstance(idx, int) or isinstance(idx, bool)
                   or not 0 <= idx < mol.GetNumAtoms()
                   or mol.GetAtomWithIdx(idx).GetAtomicNum() <= 1 for idx in context)):
        return False
    scope = scope | context
    hosts = {idx for idx in scope if mol.GetAtomWithIdx(idx).GetAtomicNum() > 1}
    actual = set()
    for atom_id, locant, _omit in entries:
        if atom_id is None or atom_id in actual:
            return False
        atom = mol.GetAtomWithIdx(atom_id)
        neighbors = tuple(atom.GetNeighbors())
        if (atom.GetAtomicNum() != 1 or atom.GetIsotope() != mass
                or atom.GetFormalCharge() or atom.GetNumRadicalElectrons()
                or len(neighbors) != 1 or neighbors[0].GetIdx() not in hosts
                or locants.get(neighbors[0].GetIdx()) != locant
                or mol.GetBondBetweenAtoms(atom_id, neighbors[0].GetIdx()).GetBondTypeAsDouble() != 1):
            return False
        actual.add(atom_id)
    expected = set()
    for host_id in hosts:
        host = mol.GetAtomWithIdx(host_id)
        hydrogens = {n.GetIdx() for n in host.GetNeighbors()
                     if n.GetAtomicNum() == 1 and n.GetIsotope() == mass}
        if host.GetTotalNumHs(includeNeighbors=True) != len(hydrogens):
            return False
        expected.update(hydrogens)
    return bool(actual) and actual == expected


def _mapped_isotope_descriptor_plan(mol, Chem, component, locants, *, skeletal_atoms=None,
                                    sole_hydrogen_atoms=()):
    """Build descriptor text and original-atom facts in one typed pass.

    Ordinary isotope-bearing atoms in ``component.atoms`` retain their source
    IDs beside the exact grouping/locant decisions used by the renderer.
    ``isotope_descriptor_parts`` remain renderable by the legacy writer, but
    have no primary atom claim unless a higher-level role record supplies it.
    A producer can identify explicit H atoms that are the sole hydrogen on
    their host. Their single substitution needs no count under P-82.2.1.
    """
    if skeletal_atoms is None and component.assembly_isotope_evidence is not None:
        from .assembly_isotopes import AssemblyIsotopeEvidence
        if not isinstance(component.assembly_isotope_evidence, AssemblyIsotopeEvidence):
            return None
        skeletal_atoms = component.assembly_isotope_evidence.context.parent_atoms
    if skeletal_atoms is not None and (not isinstance(skeletal_atoms, frozenset)
            or not skeletal_atoms <= component.atoms or not skeletal_atoms
            or any(mol.GetAtomWithIdx(i).GetAtomicNum() <= 1 for i in skeletal_atoms)):
        return None
    labelling = component.isotope_labelling_kind
    if labelling not in {'','specific','selective','nonselective','deficient','general','uniform'}:
        return None
    view = graph_view(mol, None)
    # Most numberings contain no isotope operation. The original graph view
    # proves this without rebuilding empty descriptor dictionaries or
    # acquiring the periodic table for every component map.
    if (not component.isotope_descriptor_parts
            and not any(view.isotopes[i] for i in component.atoms)):
        return (), ()
    labels = defaultdict(list)
    omit_single_count = set()
    for idx in component.atoms:
        if view.isotopes[idx] and idx not in component.local_isotope_atoms:
            labels[(view.symbols[idx], view.isotopes[idx])].append(
                (idx, locants[idx], False))
    for entry in component.isotope_descriptor_parts:
        if len(entry) not in {3, 4}:
            return None
        symbol, mass, locant = entry[:3]
        omit_count = bool(entry[3]) if len(entry) == 4 else False
        if (not isinstance(symbol, str) or not symbol or not isinstance(mass, int)
                or mass <= 0 or not isinstance(locant, str)
                or len(entry) == 4 and not isinstance(entry[3], bool)):
            return None
        labels[(symbol, mass)].append((None, locant, omit_count))
        if omit_count:
            omit_single_count.add((symbol, mass))

    rendered, primary_claims = [], []
    periodic = Chem.GetPeriodicTable()
    for (symbol, mass), entries in sorted(labels.items()):
        positions = [locant for _atom_id, locant, _omit in entries]
        if "" in positions and any(positions):
            return None
        citation_positions = (_hydrogen_citation_positions(mol, component, locants, entries)
                              if symbol == "H" else positions)
        ordered = sorted(citation_positions, key=_locant_key)
        if component.nitrogen_site_locants:
            ordered = [nitrogen_locant(loc) for loc in ordered]
        locator = ",".join(ordered) + "-" if any(ordered) else ""
        number = periodic.GetAtomicNumber(symbol)
        if (component.isotope_unlocanted_elements
                and number in component.isotope_unlocanted_elements):
            locator = ""
        if number != 1:
            population = tuple(view.isotopes[i] for i, value in enumerate(view.atomic_nums)
                               if value == number)
            if len(population) == 1 or population and all(value == mass for value in population):
                locator = ""
        carbon_scope = component.atoms if skeletal_atoms is None else skeletal_atoms
        if symbol == "C" and (skeletal_atoms is None or all(
                atom_id in skeletal_atoms for atom_id, _loc, _omit in entries)) and all(
                view.isotopes[i] == mass for i in carbon_scope
                if view.atomic_nums[i] == 6):
            locator = ""
        if symbol == "H":
            possible = {locants[i] for i in component.atoms
                        if mol.GetAtomWithIdx(i).GetAtomicNum() > 1
                        and mol.GetAtomWithIdx(i).GetTotalNumHs(includeNeighbors=True)}
            if len(possible) == 1 and set(positions) == possible:
                locator = ""
            if not component.prefixes and _complete_hydrogen_isotope_scope(
                    mol, component, locants, entries, mass, skeletal_atoms):
                # P-82.6.2 retains the locants when other groups occupy sites
                # of the numbered parent. Fully covered unsubstituted scopes
                # can use P-82.6.1.3.
                locator = ""
        count = len(positions) > 1 or symbol == "H"
        if (symbol == "H" and len(positions) == 1
                and ((symbol, mass) in omit_single_count or entries[0][0] in sole_hydrogen_atoms)):
            count = False
        if labelling:
            # Source P-83.1/.2 counts/locants describe a substance, rather
            # than the artificial maximum-occupancy graph used as a map.
            available=(sum(mol.GetAtomWithIdx(i).GetTotalNumHs(includeNeighbors=True)
                           for i in component.atoms if view.atomic_nums[i]>1)
                       if symbol=='H' else
                       sum(view.atomic_nums[i]==number for i in component.atoms))
            unique=sorted(set(citation_positions),key=_locant_key)
            if component.nitrogen_site_locants:
                unique=[nitrogen_locant(loc) for loc in unique]
            if labelling=='specific':
                count=available>1
            elif labelling=='selective':
                possible=({locants[i] for i in component.atoms
                           if view.atomic_nums[i]>1 and mol.GetAtomWithIdx(i).GetTotalNumHs(includeNeighbors=True)}
                          if symbol=='H' else {locants[i] for i in component.atoms if view.atomic_nums[i]==number})
                locator=(','.join(unique)+'-' if any(unique) and len(possible)>1 else '')
                count=False
            else:
                locator='';count=False
                if labelling=='general':
                    locator='G-'
                elif labelling=='uniform':
                    locator='U-'+(','.join(unique)+'-' if component.isotope_labelling_subset and any(unique) else '')
                elif labelling=='deficient':
                    locator='def'
        rendered.append(f"{locator}{mass}{symbol}" + (str(len(positions)) if count else ""))
        for atom_id, source_locant, _omit in entries:
            if atom_id is not None:
                printed_locant = None
                if locator:
                    printed_locant = (nitrogen_locant(source_locant)
                                      if component.nitrogen_site_locants else source_locant)
                primary_claims.append((atom_id, symbol, mass, locants[atom_id], printed_locant))
    return tuple(rendered), tuple(primary_claims)


def _mapped_stereochemistry_plan(mol, Chem, component, locants,
                                  parent_annotation_proof=None):
    """Return the exact stereo affixes and their original mapped units."""
    presentation = component.locant_presentation
    if presentation is not None:
        from .locant_presentation import valid_component_locant_presentation
        if not valid_component_locant_presentation(mol, component):
            return None
    if component.assembly_stereo_evidence is not None:
        from .assembly_stereochemistry import assembly_stereo_plan
        if parent_annotation_proof is not None or component.nonstereogenic_atom_evidence is not None:
            return None
        return assembly_stereo_plan(mol, Chem, component, locants)
    # Unassigned potential centres do not emit descriptors here. Explicit Xi
    # evidence still takes the complete proof path below, even without tags.
    if parent_annotation_proof is None and component.nonstereogenic_atom_evidence is None and not _has_assigned_stereo(mol, Chem):
        return (), (), ()
    descriptor_graph = mol
    if parent_annotation_proof is not None:
        descriptor_graph = Chem.Mol(mol)
        for atom in descriptor_graph.GetAtoms():
            atom.SetAtomMapNum(0)
    assigned = dict(assigned_atom_stereochemistry(
        descriptor_graph, Chem, atom_order="idx", include_pseudo=True))
    relative_cycle = None
    if (parent_annotation_proof is None and component.nonstereogenic_atom_evidence is None
            and component.prefix_mode == 'locanted'
            and (any(idx in locants and value in {'r', 's'} for idx, value in assigned.items())
                 or (component.parent == 'cyclohexyl' and len(locants) == 6
                     and sum(idx in locants for idx in assigned) == 2))):
        from .relative_cycle_evidence import mapped_component_relative_cycle

        relative_cycle = mapped_component_relative_cycle(descriptor_graph, Chem, locants)
    missing=frozenset(i for i in component.atoms if int(mol.GetAtomWithIdx(i).GetChiralTag()) and i not in assigned)
    if component.nonstereogenic_atom_evidence is not None:
        from .cyclic_polyol_stereo import validate_cyclic_polyol_nonstereogenic_evidence
        if (parent_annotation_proof is not None or not validate_cyclic_polyol_nonstereogenic_evidence(
                descriptor_graph,component.nonstereogenic_atom_evidence,component.atoms,assigned,missing)):return None
    elif missing and (relative_cycle is None or not missing <= relative_cycle.tagged_atoms):
        return None
    stereo_text, atom_facts, bond_facts = [], [], []
    for idx, descriptor in assigned.items():
        if idx not in component.atoms:
            continue
        locant = locants[idx]
        if not locant:
            return None
        if relative_cycle is None or idx not in relative_cycle.tagged_atoms:
            stereo_text.append((_locant_key(locant), f"{locant}{descriptor}"))
        atom_facts.append((idx, locant, descriptor))
    for a, b, descriptor in _indexed_bond_stereochemistry(mol, Chem):
        if not {a, b} & component.atoms:
            continue
        if not {a, b} <= component.atoms:
            branch = next((row for row in component.prefix_branch_ownership
                           if {row[0], row[1]} == {a, b} and row[1] in component.atoms), None)
            if branch is None:
                return None
            root, host, owned, text = branch
            parent_label = mol_cache_get(mol, 'exocyclic_parent_bond_stereo', {}).get(
                (root, host, text, owned))
            branch_claims = mol_cache_get(mol, 'rendered_mapped_branch_bond_stereo', {})
            if parent_label:
                locant, descriptor = locants[host], parent_label
            elif branch_claims.get((root, host, text, owned | {host})) == descriptor:
                continue
            else:
                return None
        else:
            locant = min((locants[a], locants[b]), key=_locant_key)
        if not locant:
            return None
        printed = (presentation.stereo_bond_locant(a, b, locant)
                   if presentation is not None else locant)
        for center, nitrogen, written in component.suffix_stereo_locants:
            if {a, b} == {center, nitrogen}:
                printed = written
                break
        stereo_text.append((_locant_key(locant), f"{printed}{descriptor}"))
        bond_facts.append((a, b, locant, descriptor))
    if parent_annotation_proof is not None:
        proof = parent_annotation_proof
        parent_map = dict(proof.parent_locants)
        if (not proof.parent_atoms <= component.atoms
                or any(locants.get(atom_id) != parent_map.get(atom_id)
                       for atom_id in proof.parent_atoms)):
            return None
        proof_atoms = set()
        for feature in proof.stereo_features:
            idx = feature.atom_id
            if idx in proof_atoms or idx not in component.atoms:
                return None
            proof_atoms.add(idx)
            locant = parent_map.get(idx)
            if not locant or locants.get(idx) != locant:
                return None
            if feature.state == "assigned":
                if assigned.get(idx) != feature.descriptor or feature.descriptor not in {"R", "S"}:
                    return None
                # The ordinary path above already inserted this exact
                # descriptor. Do not duplicate it through proof validation.
                continue
            if feature.state != "unknown_ordinary" or feature.descriptor != "Ξ":
                return None
            if (mol.GetAtomWithIdx(idx).GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
                    or idx in assigned):
                return None
            stereo_text.append((_locant_key(locant), f"{locant}Ξ"))
        if proof_atoms != {feature.atom_id for feature in proof.stereo_features}:
            return None
    if relative_cycle is not None:
        stereo_text.append(((0, ''), relative_cycle.descriptor))
    return tuple(stereo_text), tuple(atom_facts), tuple(bond_facts)


def _format_prefixes_with_functionalized_compounds(prefixes, compound_proofs, *,
                                                   citation_order=None,
                                                   hierarchical_enclosures=False):
    """Use proof-typed nature for mapped functionalized carbon branch units.

    Other prefix groups continue through the legacy formatter unchanged.
    """
    from .formatting import COMPLEX_MULT, enclose_name_unit

    if any(getattr(proof, "prefix_nature", None) not in {"simple", "compound"}
           for proof in compound_proofs):
        return None
    proof_names = {proof.text for proof in compound_proofs}
    compound_names = {proof.text for proof in compound_proofs
                      if proof.prefix_nature == "compound"}
    if compound_names & {proof.text for proof in compound_proofs
                         if proof.prefix_nature == "simple"}:
        return None
    if not proof_names:
        return None
    grouped = defaultdict(list)
    for locant, name in prefixes:
        grouped[name].append(locant)
    names = (list(citation_order) if citation_order is not None else
             sorted(grouped, key=prefix_alphabetization_key))
    if set(names) != set(grouped) or len(names) != len(set(names)):
        return None
    if not proof_names <= set(grouped):
        return None
    parts = []
    for name in names:
        locants = sorted(grouped[name], key=_locant_key)
        locator = f"{','.join(locants)}-"
        if name in compound_names:
            multiplier = (COMPLEX_MULT.get(len(locants), "") if len(locants) > 1 else "")
            if len(locants) > 1 and not multiplier:
                return None
            parts.append(locator + multiplier + enclose_name_unit(name))
        else:
            legacy_text = format_prefixes(
                [(loc, name) for loc in locants],
                hierarchical_enclosures=hierarchical_enclosures,
            )
            if not legacy_text:
                return None
            parts.append(legacy_text)
    return "-".join(parts) if parts else None


def _mapped_repeated_simple_prefix_text(mol, component, writing_prefixes):
    """Carry proved branch nature to P-16 for repeated locanted simple units.

    This opt-in path requires a complete one-to-one ownership record for
    every written prefix. If any graph, locant or component grammar is not
    proved, the ordinary writer remains responsible for the whole sequence.
    """
    if (component.prefix_mode != 'locanted' or len(writing_prefixes) < 2
            or len(component.prefix_branch_ownership) != len(writing_prefixes)):
        return None
    from collections import Counter
    from .substituent_multiplication import simple_component_has_internal_numeric_locants

    counts = Counter(text for _locant, text in writing_prefixes)
    repeated = {text for text, count in counts.items() if count > 1
                and simple_component_has_internal_numeric_locants(text)}
    if not repeated:
        return None
    locants = dict(component.atom_locants)
    owned_so_far = set()
    rows = []
    from .organyl_prefix_components import organyl_prefix_nature
    for record in component.prefix_branch_ownership:
        if not isinstance(record, (tuple, list)) or len(record) != 4:
            return None
        root, host, atoms, text = record
        if (type(root) is not int or type(host) is not int
                or host < 0 or host >= mol.GetNumAtoms()
                or not isinstance(atoms, frozenset) or not isinstance(text, str)
                or root not in atoms or host not in component.atoms
                or not atoms <= component.embedded_atoms or atoms & owned_so_far
                or any(type(idx) is not int or idx < 0 or idx >= mol.GetNumAtoms() for idx in atoms)):
            return None
        boundary = [(idx, neighbor.GetIdx()) for idx in atoms
                    for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors()
                    if neighbor.GetIdx() not in atoms and neighbor.GetAtomicNum() > 1]
        bond = mol.GetBondBetweenAtoms(root, host)
        if boundary != [(root, host)] or bond is None or bond.GetBondTypeAsDouble() != 1:
            return None
        nature = organyl_prefix_nature(mol, root, atoms, text)
        if nature is None:
            return None
        locant = locants.get(host)
        if not isinstance(locant, str):
            return None
        if component.nitrogen_site_locants:
            locant = nitrogen_locant(locant)
        rows.append((locant, text, nature))
        owned_so_far.update(atoms)
    if Counter((locant, text) for locant, text, _nature in rows) != Counter(writing_prefixes):
        return None
    if not any(text in repeated and nature == 'simple' for _loc, text, nature in rows):
        return None
    from .typed_prefixes import typed_prefix_text
    return typed_prefix_text(rows, citation_order=component.prefix_citation_order)


def _mapped_carboxyalkoxy_prefix_text(mol, Chem, component, omit_locants):
    """Cite source-owned substituted alkoxy units as compound prefixes.

    This operates on the original directed cut and atom partition before
    whole-name assembly. The separately proved P-14 omission decision controls
    whether the parent locant is printed; it never changes the inner unit.
    """
    if (component.prefix_mode != 'locanted' or component.nitrogen_site_locants
            or component.prefix_natures is not None or component.compound_prefix_names
            or component.functionalized_carbon_prefix_proofs
            or not component.prefixes
            or len(component.prefixes) != len(component.prefix_branch_ownership)
            or (omit_locants and len(component.prefixes) != 1)
            or not component.parent.endswith(('oic acid', 'carboxylic acid'))):
        return None
    if not any(isinstance(row, (tuple, list)) and len(row) == 4
               and isinstance(row[0], int) and 0 <= row[0] < mol.GetNumAtoms()
               and mol.GetAtomWithIdx(row[0]).GetAtomicNum() == 8
               for row in component.prefix_branch_ownership):
        return None
    from collections import Counter
    from .functionalized_carbon_prefixes import _scope
    from .branch.carboxyalkyl import carboxyalkoxy_prefix_proof

    mapping = dict(component.atom_locants)
    rows, proofs, owned = [], [], set()
    for record in component.prefix_branch_ownership:
        if not isinstance(record, (tuple, list)) or len(record) != 4:
            return None
        root, host, atoms, text = record
        if (not isinstance(atoms, frozenset) or not atoms <= component.embedded_atoms
                or atoms & owned or host not in component.atoms or mapping.get(host) is None
                or _scope(mol, root, host) != atoms):
            return None
        actual_text, actual_atoms = mapped_branch(mol, Chem, root, host)
        if actual_text != text or actual_atoms != atoms:
            return None
        owned.update(atoms)
        rows.append((mapping[host], text))
        proof = carboxyalkoxy_prefix_proof(mol, Chem, root, host, atoms, component.atoms)
        if proof is not None and proof.text == text and proof.prefix_nature == 'compound':
            proofs.append(proof)
    if (owned != set(component.embedded_atoms) or Counter(rows) != Counter(component.prefixes)
            or not proofs):
        return None
    # A repeated text can be grouped only if every occurrence has a proof.
    proved_names = Counter(proof.text for proof in proofs)
    written_names = Counter(text for _locant, text in rows)
    if any(written_names[name] != count for name, count in proved_names.items()):
        return None
    if omit_locants:
        return enclose_name_unit(proofs[0].text)
    return _format_prefixes_with_functionalized_compounds(
        rows, proofs, hierarchical_enclosures=component.systematic_parent)


def decorate_component(mol, Chem, component):
    locants = dict(component.atom_locants)
    if (
        set(locants) != set(component.atoms)
        or component.atoms & component.embedded_atoms
        or not component.local_isotope_atoms <= component.atoms
    ):
        return ""
    if component.substitution_locant_presentation is not None:
        from .locant_presentation import valid_joint_substitution_presentation
        if not valid_joint_substitution_presentation(mol, Chem, component):
            return ""
    if component.parent_annotation_proof is not None:
        from .extended_inventory import selection_inventory
        from .parent_annotation_proof import validate_monocycle_parent_annotation_proof

        proof = component.parent_annotation_proof
        inventory = selection_inventory(Chem.Mol(mol))
        proof_ids = set(proof.principal_identity)
        principal_groups = tuple(
            group for group in inventory.groups
            if (group.kind, group.center, tuple(group.atoms), group.owner, group.ligand)
            in proof_ids
        ) if inventory.supported else ()
        if (not inventory.supported
                or len(principal_groups) != len(proof_ids)
                or not validate_monocycle_parent_annotation_proof(
                    mol, Chem, proof, parent_locants=tuple(
                        (idx, locants[idx]) for idx in proof.parent_atoms if idx in locants
                    ), principal_groups=principal_groups,
                )):
            return ""
    if component.functionalized_carbon_prefix_proofs:
        # This producer closes explicit outer locants. Other writing contexts
        # need their own source proof before they can consume this evidence.
        if component.prefix_mode != "locanted" or component.omit_methyl_prefix_locant:
            return ""
        from collections import Counter
        from .functionalized_carbon_prefixes import (
            FunctionalizedCarbonPrefixProof,
            validate_functionalized_carbon_prefix_proof,
        )
        rendered_rows = Counter(component.prefixes)
        seen_scope = set()
        for proof in component.functionalized_carbon_prefix_proofs:
            if (not isinstance(proof, FunctionalizedCarbonPrefixProof)
                    or not validate_functionalized_carbon_prefix_proof(mol, Chem, proof)
                    or proof.scope & seen_scope
                    or not proof.scope <= component.embedded_atoms
                    or proof.parent_host not in component.atoms
                    or rendered_rows.get((locants.get(proof.parent_host), proof.text), 0) <= 0):
                return ""
            seen_scope.update(proof.scope)
            rendered_rows[(locants[proof.parent_host], proof.text)] -= 1
    if not _valid_isotope_role_evidence(mol, component):
        return ""
    if component.amide_isotope_evidence is not None:
        from .amide_isotopes import validate_amide_isotope_evidence
        if not validate_amide_isotope_evidence(mol, component):
            return ''
    if component.assembly_isotope_evidence is not None:
        from .assembly_isotopes import validate_assembly_isotope_evidence
        if not validate_assembly_isotope_evidence(mol, Chem, component):
            return ''
    if component.tree_assembly_evidence is not None:
        from .tree_ring_assemblies import validate_tree_assembly_evidence
        if not validate_tree_assembly_evidence(mol,Chem,component):
            return ''
    isotope_plan = _mapped_isotope_descriptor_plan(mol, Chem, component, locants)
    if isotope_plan is None:
        return ""
    parts, _primary_claims = isotope_plan
    parent = component.parent
    if parts:
        opening,closing=('[',']') if component.isotope_labelling_kind else ('(',')')
        parent = (
            opening + ",".join(parts) + closing + ("-" if parent[0].isdigit() else "") + parent
        )
    # Join chemical prefixes using the undecorated parent's grammatical role;
    # inserting an isotope descriptor does not create another prefix boundary.
    writing_prefixes=list(component.prefixes)
    if component.nitrogen_site_locants:
        writing_prefixes=[(nitrogen_locant(loc),text) for loc,text in writing_prefixes]
    retained_acid_omission = False
    if component.retained_acid_locant_evidence is not None:
        from .retained_acid_rules import validate_retained_acid_locant_evidence

        evidence = component.retained_acid_locant_evidence
        retained_acid_omission = (
            component.parent in {"acetic acid", "acetate"}
            and evidence.parent_atoms <= component.atoms
            and validate_retained_acid_locant_evidence(
                mol, Chem, evidence, component.atom_locants, component.prefixes
            )
            and dict(component.atom_locants).get(evidence.alpha_carbon)
                == evidence.alpha_locant
            and dict(component.atom_locants).get(evidence.carbonyl_atom) is not None
        )
        if not retained_acid_omission:
            return ""
    from .locant_presentation import component_prefix_locant_presentation
    source_omission = component_prefix_locant_presentation(mol, Chem, component)
    omit_prefix_locants = (component.omit_methyl_prefix_locant or retained_acid_omission
                          or source_omission is not None)
    if component.functionalized_carbon_prefix_proofs and omit_prefix_locants:
        return ""
    retained_prefix_text = None
    if retained_acid_omission:
        # P-16.5.1.3.2 is a specific one-position retained-parent sequence.
        # Rebuild each branch from its original mapped atoms before writing it;
        # do not pass the omitted locants through the ordinary legacy joiner.
        if (component.prefix_mode != "locanted"
                or component.omit_methyl_prefix_locant
                or component.prefix_citation_order is not None):
            return ""
        from .retained_acid_rules import retained_acid_prefix_sequence
        sequence = retained_acid_prefix_sequence(
            mol, Chem, component.retained_acid_locant_evidence,
            component.atom_locants, component.prefixes,
        )
        if sequence is None:
            return ""
        retained_prefix_text = sequence.text
    if (component.prefix_citation_order is not None
            and (component.prefix_mode != 'locanted' or omit_prefix_locants)):
        return ""
    if component.prefix_citation_order is not None:
        order = component.prefix_citation_order
        names = {text for _loc, text in writing_prefixes}
        if (not isinstance(order, (tuple, list))
                or any(not isinstance(name, str) or not name for name in order)
                or len(order) != len(set(order)) or set(order) != names):
            return ""
    carboxyalkoxy_prefix_text = None
    if (retained_prefix_text is None and component.tree_assembly_evidence is None
            and not component.prefix_citation_order
            and (not omit_prefix_locants or source_omission is not None)):
        carboxyalkoxy_prefix_text = _mapped_carboxyalkoxy_prefix_text(
            mol, Chem, component, omit_prefix_locants)
    if component.tree_assembly_evidence is not None:
        from .tree_ring_assemblies import tree_assembly_prefix_text
        prefix_text = tree_assembly_prefix_text(component,writing_prefixes)
        if prefix_text is None:return ''
    elif retained_prefix_text is not None:
        prefix_text = retained_prefix_text
    elif carboxyalkoxy_prefix_text is not None:
        prefix_text = carboxyalkoxy_prefix_text
    elif component.compound_prefix_names:
        from types import SimpleNamespace
        if (component.prefix_mode!='locanted' or omit_prefix_locants
                or component.prefix_natures is not None
                or component.functionalized_carbon_prefix_proofs
                or len(set(component.compound_prefix_names))!=len(component.compound_prefix_names)
                or not set(component.compound_prefix_names)<={text for _,text in writing_prefixes}):
            return ''
        prefix_text = _format_prefixes_with_functionalized_compounds(
            writing_prefixes, tuple(SimpleNamespace(text=text,prefix_nature='compound')
                                    for text in component.compound_prefix_names),
            citation_order=component.prefix_citation_order,
            hierarchical_enclosures=component.systematic_parent,
        )
        if prefix_text is None:return ''
    elif (component.functionalized_carbon_prefix_proofs
          and component.prefix_mode == "locanted" and not omit_prefix_locants
          and component.prefix_natures is None):
        prefix_text = _format_prefixes_with_functionalized_compounds(
            writing_prefixes,
            component.functionalized_carbon_prefix_proofs,
            citation_order=component.prefix_citation_order,
            hierarchical_enclosures=(component.nitrogen_site_locants
                                     or component.systematic_parent),
        )
        if prefix_text is None:
            return ""
    elif component.prefix_natures is not None:
        # Job 141 is intentionally opt-in and bounded to explicit locanted
        # prefixes. Do not silently route typed rows through legacy omission,
        # amino, or other grammar paths that erase their declared locants.
        if component.prefix_mode not in {'locanted', 'unlocanted', 'amino'} or omit_prefix_locants:
            return ""
        nature_by_component = {}
        for entry in component.prefix_natures:
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                return ""
            text, nature = entry
            if (not isinstance(text, str) or not text or not isinstance(nature, str)
                    or nature not in {"simple", "compound"}
                    or text in nature_by_component and nature_by_component[text] != nature):
                return ""
            nature_by_component[text] = nature
        if set(nature_by_component) != {text for _, text in writing_prefixes}:
            return ""
        if any(nature_by_component.get(proof.text) != proof.prefix_nature
               for proof in component.functionalized_carbon_prefix_proofs):
            return ""
        if component.prefix_mode in {'unlocanted', 'amino'}:
            # P-16/P-82: keep source-typed mononuclear prefixes separate
            # from their parent so its isotope descriptor follows them.
            heavy_core = [i for i in component.atoms if mol.GetAtomWithIdx(i).GetAtomicNum() > 1]
            if len(heavy_core) != 1 or any(loc for loc, _ in writing_prefixes):
                return ''
            from collections import Counter
            from .prefix_sequences import PrefixUnit, mononuclear_prefix_sequence
            counts = Counter(text for _, text in writing_prefixes)
            sequence = mononuclear_prefix_sequence(
                PrefixUnit(text, nature_by_component[text], count) for text, count in counts.items())
            prefix_text = sequence.text if sequence is not None else None
        else:
            from .typed_prefixes import typed_prefix_text
            prefix_text = typed_prefix_text([
                (loc, text, nature_by_component[text]) for loc, text in writing_prefixes
            ], citation_order=component.prefix_citation_order)
        if prefix_text is None:
            return ""
    else:
        prefix_text = (None if omit_prefix_locants else
                       _mapped_repeated_simple_prefix_text(mol, component, writing_prefixes))
        if prefix_text is None:
            prefix_text = format_prefixes(
                    writing_prefixes,
                    omit_locants=omit_prefix_locants,
                    hierarchical_enclosures=component.nitrogen_site_locants or component.systematic_parent,
                    citation_order=component.prefix_citation_order,
                    # Each mapped prefix is an independent attachment even
                    # when a one-carbon parent omits all numerical locants.
                    separate_unlocanted=omit_prefix_locants,
                )
    if component.prefix_mode in {'unlocanted', 'mononuclear'} and component.prefix_natures is None:
        # P-14.3.4.4: a child unit may need its own locants while its
        # attachment to the enclosing parent is unique by symmetry/valence.
        if component.prefix_mode == 'mononuclear':
            heavy_core = [i for i in component.atoms if mol.GetAtomWithIdx(i).GetAtomicNum() > 1]
            if len(heavy_core) != 1 or any(loc for loc, _ in writing_prefixes):
                return ''
        prefix_text = format_prefixes(writing_prefixes, omit_locants=True,
                                      hierarchical_enclosures=True,
                                      separate_unlocanted=True)
    elif component.prefix_mode=='amino' and component.prefix_natures is None:
        units=[text for loc,text in component.prefixes]
        if len(set(units))<=1:
            prefix_text=format_prefixes([('',text) for text in units],omit_locants=True,hierarchical_enclosures=True)
        else:
            units.sort(key=prefix_alphabetization_key)
            prefix_text=format_prefixes([('',units[0])],omit_locants=True,hierarchical_enclosures=True)+''.join(enclose_name_unit(t) for t in units[1:])
    undecorated = component.parent
    if component.hydro_prefix:
        join = '-' if component.parent[0].isdigit() else ''
        undecorated = component.hydro_prefix+join+component.parent
        parent = component.hydro_prefix+(('' if parts else join))+parent
    if component.systematic_parent:
        # The producer has already typed the parent; internal suffix locants
        # do not create a hyphen before its initial letters (P-16).
        # P-82.2.1 inserts an isotope descriptor directly before its modified
        # part. A following locanted parent keeps the hyphen AFTER that
        # descriptor; the old unmodified initial digit must not insert a
        # second hyphen before it. A leading hydro locant still needs its own
        # join, so inspect the actual assembled component's first character.
        name = prefix_text+('-' if prefix_text and parent[0].isdigit() else '')+parent
    else:
        base = join_prefix_parent(prefix_text, undecorated)
        if not base.endswith(undecorated):
            return ""
        name = base[: -len(undecorated)] + parent
    stereo_plan = _mapped_stereochemistry_plan(
        mol, Chem, component, locants, component.parent_annotation_proof,
    )
    if stereo_plan is None:
        return ""
    stereo, _stereo_atoms, _stereo_bonds = stereo_plan
    if stereo:
        relative = stereo[-1][1] if stereo[-1][1] in {'cis', 'trans'} else ''
        if relative:
            stereo = stereo[:-1]
        # P-16.5.4.1.3 counts parentheses inside a compound stereo locant.
        opening, closing = ('[', ']') if component.assembly_stereo_evidence is not None and any(
            '(' in text for _, text in stereo) else ('(', ')')
        if stereo:
            name = opening + ",".join(text for _, text in sorted(stereo)) + closing + "-" + name
        if relative:
            name = f"({relative})-{name}"
    return name


def has_annotations(mol, atoms):
    from rdkit import Chem

    atoms = set(atoms)
    for i in atoms:
        atom = mol.GetAtomWithIdx(i)
        if atom.GetIsotope() or int(atom.GetChiralTag()):
            return True
    return any(a in atoms or b in atoms
               for a, b, _ in _indexed_bond_stereochemistry(mol, Chem))


def mapped_branch(mol, Chem, root, blocked):
    # Parent orientations share the same original directed bridge. Preserve
    # temporary fallback modes and recursion guards in the cache identity.
    key = (id(Chem), root, blocked, naming_cache_state(mol),
           frozenset(mol_cache_get(mol, "branch_prefix_in_progress", ())))
    cache = mol_cache_dict(mol, "mapped_branch")
    if key not in cache:
        text, atoms = _mapped_branch(mol, Chem, root, blocked)
        if text in {'carbamyl', 'methylcarbamyl', 'dimethylcarbamyl'}:
            from .carboxamide_prefixes import preferred_carbamoyl_prefix
            preferred = preferred_carbamoyl_prefix(mol, root, blocked, text, atoms)
            if preferred is not None:
                text = preferred
        cache[key] = text, atoms
    return cache[key]


def mapped_general_branch(mol, Chem, root, blocked):
    """Bind an existing general prefix to its complete original branch."""
    key = (id(Chem), root, blocked, naming_cache_state(mol),
           frozenset(mol_cache_get(mol, "branch_prefix_in_progress", ())))
    cache = mol_cache_dict(mol, "mapped_general_branch")
    if key not in cache:
        cache[key] = _mapped_general_branch(mol, Chem, root, blocked)
    return cache[key]


def _mapped_general_branch(mol, Chem, root, blocked):
    from .branch_rules import collect_subtree

    cut = mol.GetBondBetweenAtoms(root, blocked)
    if cut is None:
        return '', frozenset()
    atoms = frozenset(collect_subtree(mol, root, blocked))
    if (root not in atoms or blocked in atoms
            or {(i, n.GetIdx()) for i in atoms for n in mol.GetAtomWithIdx(i).GetNeighbors()
                if n.GetAtomicNum() > 1 and n.GetIdx() not in atoms} != {(root, blocked)}):
        return '', frozenset()
    order = cut.GetBondTypeAsDouble()
    if (order == 2 and mol.GetAtomWithIdx(root).GetAtomicNum() == 7
            and any(a.GetAtomicNum() == 7 for a in mol.GetAtomWithIdx(root).GetNeighbors())):
        from .hydrazone_methods import hydrazinylidene_prefix
        result = hydrazinylidene_prefix(mol, Chem, root, blocked, atoms)
        if result[0]:
            return result
    if cut.GetStereo():
        from .oxime_components import oxime_imino_prefix

        return oxime_imino_prefix(mol, Chem, root, blocked, atoms)
    azides = mol_cache_get(mol, 'nested_azido_sites')
    if azides and not atoms.isdisjoint(azides):
        from .nested_azide_methods import compose_azido_branch

        return compose_azido_branch(mol, Chem, root, blocked, atoms, azides)
    atom = mol.GetAtomWithIdx(root)
    text, owned = mapped_branch(mol, Chem, root, blocked)
    if text and owned == atoms:
        if "-one-" in text and mol.GetAtomWithIdx(root).IsInRing():
            rooted = _complete_rooted_ring(mol, Chem, root, atoms)
            if rooted and "oxo" in rooted and "-one-" not in rooted:
                return rooted, atoms
        return text, atoms
    if order == 1 and atom.GetAtomicNum() == 7 and not atom.IsInRing() and atom.GetDegree() == 2:
        from .sulfonylguanidine import mapped_sulfonylguanidino_prefix

        guanidino = mapped_sulfonylguanidino_prefix(mol, Chem, root, blocked, atoms)
        if guanidino is not None:
            return guanidino
        text = _mapped_secondary_amino_branch(mol, Chem, root, blocked, atoms)
        if text:
            return text, atoms
    if order == 2 and atom.GetAtomicNum() == 6 and atom.IsInRing():
        text = _complete_rooted_ring(mol, Chem, root, atoms)
        if text:
            return text, atoms
    if order != 1 or any(
            mol.GetAtomWithIdx(i).GetIsotope() or mol.GetAtomWithIdx(i).GetNumRadicalElectrons()
            for i in atoms):
        return '', frozenset()
    if atom.GetAtomicNum() == 6 and not atom.IsInRing() and atom.GetDegree() == 2 and (
            atom.GetTotalNumHs() == 2 and any(mol.GetAtomWithIdx(i).GetFormalCharge() for i in atoms)):
        from .neutral_charge_components import neutral_charge_role_proofs

        proofs = neutral_charge_role_proofs(mol)
        if any(proof.kind == 'azido' and set(proof.atoms) <= atoms for proof in proofs):
            from .formatting import enclose_name_unit

            child = next(a for a in atom.GetNeighbors() if a.GetIdx() != blocked)
            child_idx = child.GetIdx()
            if mol.GetBondBetweenAtoms(root, child_idx).GetBondTypeAsDouble() == 1:
                if child.IsInRing():
                    name = _complete_rooted_ring(mol, Chem, child_idx, atoms - {root})
                    if name:
                        return enclose_name_unit(name) + 'methyl', atoms
                elif child.GetAtomicNum() == 7:
                    name, owned = mapped_branch(mol, Chem, child_idx, root)
                    if name == 'azido' and owned | {root} == atoms:
                        return 'azidomethyl', atoms
                    name = _mapped_secondary_amino_branch(mol, Chem, child_idx, root, atoms - {root})
                    if name:
                        return enclose_name_unit(name) + 'methyl', atoms
    if (atom.GetAtomicNum() == 8 and not atom.IsInRing() and not atom.GetFormalCharge()
            and atom.GetDegree() == 2):
        child = next(a for a in atom.GetNeighbors() if a.GetIdx() != blocked)
        child_idx = child.GetIdx()
        if (child.GetAtomicNum() == 6 and not child.IsInRing() and child.GetDegree() == 2
                and child.GetTotalNumHs() == 2 and any(
                    mol.GetAtomWithIdx(i).GetFormalCharge() for i in atoms)
                and mol.GetBondBetweenAtoms(root, child_idx).GetBondTypeAsDouble() == 1):
            text, owned = mapped_general_branch(mol, Chem, child_idx, root)
            if text and owned | {root} == atoms:
                from .formatting import alkyl_to_alkoxy

                return alkyl_to_alkoxy(text), atoms
    if (atom.GetAtomicNum() == 8 and atom.GetDegree() == 2 and not atom.GetFormalCharge()
            and not atom.IsInRing()):
        child = next(n for n in atom.GetNeighbors() if n.GetIdx() != blocked)
        if (child.GetAtomicNum() == 6 and child.IsInRing()
                and mol.GetBondBetweenAtoms(root, child.GetIdx()).GetBondTypeAsDouble() == 1):
            from .formatting import alkyl_to_alkoxy

            child_idx = child.GetIdx()
            child_scope = atoms - {root}
            text = _complete_rooted_ring(mol, Chem, child_idx, child_scope)
            if text:
                owned = child_scope
                _record_branch_ring_atoms(mol, Chem, child_idx, root, text, owned,
                    (i for i in owned if mol.GetAtomWithIdx(i).IsInRing()))
            else:
                text, owned = mapped_general_branch(mol, Chem, child_idx, root)
            if text and owned == atoms - {root}:
                oxy = alkyl_to_alkoxy(text)
                _record_branch_ring_atoms(mol, Chem, root, blocked, oxy, atoms,
                    mapped_branch_ring_atoms(mol, Chem, child_idx, root, text, owned))
                return oxy, atoms
    if (atom.GetAtomicNum() == 6 and atom.IsInRing() and atom.GetIsAromatic()
            and mol.GetRingInfo().NumRings() >= 4
            and (mol.GetAtomWithIdx(blocked).GetAtomicNum() == 7
                 or any(a.GetAtomicNum() == 6 and a.GetDegree() == 4 and a.IsInRing()
                        for a in (mol.GetAtomWithIdx(i) for i in atoms)))):
        text = _complete_rooted_ring(mol, Chem, root, atoms)
        if text:
            return text, atoms
    if len(atoms) >= 8 and atom.IsInRing() and mol.GetRingInfo().NumRings() >= 3:
        from .spiro_component_rules import mapped_component_spiro_yl

        text, owned = mapped_component_spiro_yl(
            mol, Chem, root, blocked, atoms, allow_replacement=True, general_branches=True)
        if text and owned == atoms:
            return text, atoms
    if len(atoms) >= 18 and atom.IsInRing() and mol.GetRingInfo().NumRings() >= 5:
        from .polycyclic_rules import mapped_azido_polycycle_yl

        text = mapped_azido_polycycle_yl(mol, Chem, root, blocked, atoms)
        if text:
            return text, atoms
    from .parent_rules import _simple_or_branch_prefix

    text = _simple_or_branch_prefix(mol, root, blocked)
    if text and _general_branch_stereo_is_cited(mol, Chem, atoms, text):
        return text, atoms
    if mol_cache_get(mol, 'numbered_polycycle_recovery'):
        if atom.IsInRing():
            text = _complete_rooted_ring(mol, Chem, root, atoms)
            if text:
                return text, atoms
        elif atom.GetAtomicNum() in {6, 8}:
            return _mapped_branch_recursive(mol, Chem, root, blocked, atoms, general_branches=True)
    if (atom.GetAtomicNum() == 6 and atom.IsInRing() and atom.GetDegree() == 4
            and mol.GetAtomWithIdx(blocked).GetAtomicNum() == 7):
        from .mapped_polycycle_recovery import symmetric_bridged_amide_arm

        text = symmetric_bridged_amide_arm(mol, Chem, root, blocked, atoms)
        if text:
            _record_branch_ring_atoms(mol, Chem, root, blocked, text, atoms,
                                      (i for i in atoms if mol.GetAtomWithIdx(i).IsInRing()))
            return text, atoms
    return '', frozenset()


def _mapped_secondary_amino_branch(mol, Chem, root, blocked, atoms):
    """Bind a complete ring or acyl substituent through an exocyclic NH bridge."""
    atom = mol.GetAtomWithIdx(root)
    if (atom.IsInRing() or atom.GetFormalCharge() or atom.GetIsotope()
            or atom.GetDegree() != 2 or atom.GetTotalNumHs() != 1):
        return ""
    child = next(n for n in atom.GetNeighbors() if n.GetIdx() != blocked)
    if (child.GetAtomicNum() != 6
            or mol.GetBondBetweenAtoms(root, child.GetIdx()).GetBondTypeAsDouble() != 1):
        return ""
    scope = atoms - {root}
    if not child.IsInRing():
        from .branch.preferred_acyl import azidoalkanoyl_prefix, ring_carbonyl_prefix

        acyl = azidoalkanoyl_prefix(mol, Chem, child.GetIdx(), root, scope)
        if acyl is not None:
            return enclose_name_unit(acyl[0]) + 'amino'
        if any(mol.GetAtomWithIdx(i).GetFormalCharge() for i in scope):
            name = ring_carbonyl_prefix(mol, child.GetIdx(), root, allow_azido=True)
            if name:
                return enclose_name_unit(name) + 'amino'
        return ''
    name = _complete_rooted_ring(mol, Chem, child.GetIdx(), scope)
    if name.endswith('phenyl') and name.count('phenyl') == 2:
        from .ring_assembly_methods import rooted_biphenyl_from_prefix

        name = rooted_biphenyl_from_prefix(mol, Chem, child.GetIdx(), root, name)
    if not name and len(scope) >= 16:
        from .extended_inventory import selection_inventory

        groups = selection_inventory(mol).principal
        if groups and all(g.kind == "acid" and g.center not in atoms for g in groups):
            from .polycyclic_rules import mapped_fused_polycycle_yl

            name = mapped_fused_polycycle_yl(mol, Chem, child.GetIdx(), root, scope)
    return enclose_name_unit(name) + "amino" if name else ""


def _complete_rooted_ring(mol, Chem, root, scope):
    from .mapped_ring_parents import records, ring_systems

    core = next((ring for ring in ring_systems(mol) if root in ring), None)
    if core is None or not core <= scope:
        return ""
    candidates = records(mol, Chem, scope=scope, attachment=root,
                         core_filter=core, general_branches=True)
    if len(candidates) == 1:
        row = candidates[0]
        component = row.construction
        if (row.hydrogen_budget_proved and row.annotation_coverage
                and component is not None and component.atoms | component.embedded_atoms == scope):
            return row.name
    if core in (mol_cache_get(mol, 'numbered_polycycle_recovery') or ()):
        from .mapped_polycycle_recovery import numbered_polycycle_component

        result = numbered_polycycle_component(mol, Chem, core, scope, attachment=root)
        if result is not None:
            return result[0]
    return ""


def _general_branch_stereo_is_cited(mol, Chem, atoms, text):
    """Check that the compatibility writer cited every assigned stereo unit."""
    import re
    from collections import Counter
    from .relative_cycle_evidence import native_relative_cycle_evidence
    from .stereochemistry import (
        _internal_atom_stereo_descriptors_in_name,
        _locanted_atom_stereo_descriptors_in_name,
        _locanted_bond_stereo_descriptors_in_name,
    )

    tagged = {i for i in atoms if int(mol.GetAtomWithIdx(i).GetChiralTag())}
    relative = Counter(re.findall(r'\((cis|trans)\)-', text))
    if relative:
        proved = Counter()
        for ring in mol.GetRingInfo().AtomRings():
            if set(ring) <= atoms and tagged.intersection(ring):
                proof = native_relative_cycle_evidence(mol, Chem, ring)
                if proof is not None:
                    proved[proof.descriptor.strip('()-')] += 1
                    tagged.difference_update(proof.tagged_atoms)
        if relative != proved:
            return False
    assigned = (dict(assigned_atom_stereochemistry(mol, Chem, include_pseudo=True))
                if tagged else {})
    expected = Counter(assigned.get(i) for i in tagged)
    cited = Counter(_locanted_atom_stereo_descriptors_in_name(text)
                    + _internal_atom_stereo_descriptors_in_name('(' + text + ')'))
    bond_expected = Counter(value for a, b, value in _indexed_bond_stereochemistry(mol, Chem)
                            if a in atoms and b in atoms)
    return expected == cited and bond_expected == Counter(
        _locanted_bond_stereo_descriptors_in_name('(' + text + ')'))


def mapped_branch_ring_atoms(mol, Chem, root, blocked, text, atoms):
    """Ring ownership actually supplied by a mapped branch construction.

    Compatibility text alone supplies no witness. Carbon/O branch writers
    propagate their children's witnesses only after completing their own
    atom partition, preserving the public two-item mapped_branch contract.
    """
    key = (id(Chem), root, blocked, naming_cache_state(mol), text, frozenset(atoms))
    return mol_cache_dict(mol, 'mapped_branch_ring_ownership').get(key, frozenset())


def _record_branch_ring_atoms(mol, Chem, root, blocked, text, atoms, ring_atoms):
    if text:
        key = (id(Chem), root, blocked, naming_cache_state(mol), text, frozenset(atoms))
        mol_cache_dict(mol, 'mapped_branch_ring_ownership')[key] = frozenset(ring_atoms)



def _equivalent_prefix_enclosures(left, right):
    """Compare prefix nesting without conflating ring descriptors and brackets.

    A mapped writer and the compatibility writer can use different nested
    enclosing-mark levels for the same branch.  Ring/fusion descriptor marks
    immediately following an alphanumeric token remain significant.
    """
    if len(left) != len(right):
        return False
    normalized = str.maketrans({'[': '(', '{': '(', ']': ')', '}': ')' })
    if left.translate(normalized) != right.translate(normalized):
        return False
    stack = []
    for i, (a, b) in enumerate(zip(left, right, strict=True)):
        if a in '([{':
            protected = i > 0 and left[i - 1].isalnum()
            stack.append(protected)
            if protected and a != b:
                return False
        elif a in ')]}':
            if not stack:
                return False
            protected = stack.pop()
            if protected and a != b:
                return False
    return not stack


def _mapped_branch(mol, Chem, root, blocked, *, mapped_cyclic_only=False):
    from .branch_rules import collect_subtree
    from .parent_rules import _simple_or_branch_prefix

    atoms = frozenset(collect_subtree(mol, root, blocked))
    root_atom = mol.GetAtomWithIdx(root)
    if not mapped_cyclic_only and root_atom.GetAtomicNum() == 8 and len(atoms) >= 5:
        from .carboxamide_prefixes import mapped_carbamoylmethoxy

        carbamoylmethoxy = mapped_carbamoylmethoxy(mol, Chem, root, blocked, atoms)
        if carbamoylmethoxy is not None:
            return carbamoylmethoxy
    if (len(atoms) in {2, 4} and mol.GetAtomWithIdx(blocked).GetAtomicNum() == 6
            and (root_atom.GetAtomicNum() == 7 and root_atom.GetFormalCharge() == 1
                 or root_atom.GetAtomicNum() == 8 and root_atom.GetDegree() == 2)):
        from .neutral_charge_components import _recognize_at_carbon

        proof = _recognize_at_carbon(mol, blocked, root)
        if proof is not None and proof.kind in {"isocyano", "nitrooxy"} and frozenset(proof.atoms) == atoms:
            return proof.kind, atoms
    if not mapped_cyclic_only and root_atom.GetAtomicNum() in {7,8,16,34,52}:
        from .rules45_97_chains_locants import branch_prefix as hetero_branch_prefix
        preferred=hetero_branch_prefix(mol,Chem,root,blocked)
        if preferred is not None:
            text,owned=preferred
            if text and frozenset(owned)==atoms and root in owned and blocked not in owned:
                return text,atoms
    if not mapped_cyclic_only and root_atom.GetAtomicNum() in {7,16}:
        from .rules45_97_amide_nitrogen import branch_prefix as nitrogen_branch_prefix
        preferred=nitrogen_branch_prefix(mol,Chem,root,blocked)
        if preferred is not None:
            text,owned=preferred
            if text and frozenset(owned)==atoms and root in owned and blocked not in owned:
                return text,atoms
    if not mapped_cyclic_only and root_atom.GetAtomicNum()==15:
        from .rules45_97_acid_halides import branch_prefix
        preferred=branch_prefix(mol,Chem,root,blocked)
        if preferred is not None:
            text,owned=preferred
            if text and frozenset(owned)==atoms and root in owned and blocked not in owned:
                return text,atoms
    if root_atom.GetAtomicNum() == 6:
        from .branch.carbon import short_substitutive_carbon_prefix
        short = short_substitutive_carbon_prefix(mol, root, blocked, atoms)
        if short is not None:
            return short
        if (root_atom.GetHybridization() == Chem.HybridizationType.SP2 and not root_atom.IsInRing()
                and any(
                b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(root_atom).GetAtomicNum() == 7
                for b in root_atom.GetBonds())):
            from .oxime_components import oxime_site

            if oxime_site(mol, root) is not None:
                name, owned = _mapped_branch_recursive(mol, Chem, root, blocked, atoms)
                if name and owned == atoms:
                    return name, owned
        if len(atoms) == 4 and sorted(mol.GetAtomWithIdx(i).GetAtomicNum() for i in atoms) == [6, 7, 7, 8]:
            from .hydrazide_components import unsubstituted_hydrazide_prefix

            hydrazide = unsubstituted_hydrazide_prefix(mol, root, blocked, atoms)
            if hydrazide is not None and hydrazide[0]:
                return hydrazide
        if (not mapped_cyclic_only and 6 <= len(atoms) <= 15 and not root_atom.IsInRing()
                and mol.GetAtomWithIdx(blocked).GetAtomicNum() == 7
                and any(mol.GetAtomWithIdx(i).GetFormalCharge() for i in atoms)
                and any(b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(root_atom).GetAtomicNum() == 8
                        for b in root_atom.GetBonds())):
            from .branch.preferred_acyl import azidoalkanoyl_prefix

            acyl = azidoalkanoyl_prefix(mol, Chem, root, blocked, atoms)
            if acyl is not None:
                return acyl
        if mol.GetAtomWithIdx(blocked).GetAtomicNum() in {6, 7} and mol.GetAtomWithIdx(blocked).IsInRing():
            from .branch.subordinate_amide_chain import subordinate_amide_chain_prefix

            amide_chain = subordinate_amide_chain_prefix(mol, Chem, root, blocked)
            if amide_chain is not None and amide_chain[1] == atoms:
                return amide_chain
        from .urea_prefixes import carbamoylamino_alkyl_prefix
        urea = carbamoylamino_alkyl_prefix(mol, Chem, root, blocked, atoms)
        if urea is not None:
            return urea
        if any(n.GetAtomicNum() == 7 and n.GetIdx() != blocked
               and mol.GetBondBetweenAtoms(root, n.GetIdx()).GetBondTypeAsDouble() == 2
               for n in root_atom.GetNeighbors()):
            from .branch.carbamimidoyl import carbamimidoyl_prefix

            amidine = carbamimidoyl_prefix(mol, Chem, root, blocked)
            if amidine:
                return amidine, atoms
    if root_atom.GetAtomicNum() == 7:
        if any(n.GetAtomicNum() == 6 and n.GetIdx() != blocked
               and any(b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(n).GetAtomicNum() == 7
                       for b in n.GetBonds()) for n in root_atom.GetNeighbors()):
            from .nitrogen_acid_derivatives import amidino_prefix

            amidino = amidino_prefix(mol, Chem, root, blocked, atoms)
            if amidino is not None:
                return amidino
        from .hydrazonamide_prefixes import hydrazonamido_prefix
        hydrazonamide = hydrazonamido_prefix(mol, Chem, root, blocked, atoms)
        if hydrazonamide is not None:
            return hydrazonamide
    if (root_atom.GetAtomicNum() == 7 and mol.GetAtomWithIdx(blocked).GetAtomicNum() == 6
            and any(graph_view(mol, None).isotopes)
            and any(b.GetBondTypeAsDouble() == 2 and b.GetOtherAtomIdx(blocked) != root
                    and b.GetOtherAtom(mol.GetAtomWithIdx(blocked)).GetAtomicNum() == 8
                    for b in mol.GetAtomWithIdx(blocked).GetBonds())):
        from .amide_isotopes import mapped_amide_nitrogen_prefix
        amide_nitrogen = mapped_amide_nitrogen_prefix(mol, Chem, root, blocked, atoms)
        if amide_nitrogen is not None:
            return amide_nitrogen
    if root_atom.GetAtomicNum() == 6 and any(graph_view(mol, None).isotopes) and any(
            b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(root_atom).GetAtomicNum() == 8
            for b in root_atom.GetBonds()):
        from .amide_isotopes import mapped_amide_prefix
        amide = mapped_amide_prefix(mol, Chem, root, blocked, atoms)
        if amide is not None:
            return amide
    if root_atom.GetAtomicNum() in {9, 17, 35, 53} and (
            root_atom.GetFormalCharge() or root_atom.GetTotalValence() != 1
            or root_atom.GetTotalNumHs(includeNeighbors=True)):
        # P-21.1.2.1/P-68.5.1: an ordinary halogen prefix cannot own a
        # nonstandard hydride's bonding number or its attached hydrogens.
        # Leave the complete hydride to its established parent writer.
        return '', frozenset()
    # This typed producer constructs one-free-valence carbon prefixes. A
    # multiple cut needs ylidene/ylidyne grammar (P-29); naming its capped
    # carbon path as an alkyl group would silently lose the original bond.
    if (mol.GetAtomWithIdx(root).GetAtomicNum()==6 and
            mol.GetBondBetweenAtoms(root,blocked).GetBondTypeAsDouble()!=1):
        if mol.GetAtomWithIdx(blocked).GetAtomicNum() == 7 and not root_atom.IsInRing():
            from .branch.carbon import _terminal_locanted_acyclic_substituent_chain_prefix

            text = _terminal_locanted_acyclic_substituent_chain_prefix(
                mol, root, blocked, attachment_order=2)
            if text:
                return text, atoms
        if has_annotations(mol,atoms):return '',frozenset()
        if mol.GetAtomWithIdx(root).IsInRing():
            from .mapped_ring_parents import rooted_prefix
            text, owned = rooted_prefix(mol, Chem, root, blocked, atoms)
            if text and owned == atoms:
                _record_branch_ring_atoms(mol, Chem, root, blocked, text, owned,
                    (i for i in owned if mol.GetAtomWithIdx(i).IsInRing()))
                return text, owned
        name=_simple_or_branch_prefix(mol,root,blocked)
        return (name,atoms) if name else ('',frozenset())
    if mol.GetAtomWithIdx(root).GetAtomicNum() in {8, 16}:
        from .peroxide_components import peroxide_bridge, mapped_peroxide_prefix
        # A recognized bridge is one typed additive unit. If annotations or
        # distal nomenclature fall outside the helper's proof, do not let the
        # generic O/S path split it into unrelated single-atom prefixes.
        if peroxide_bridge(mol, root, blocked) is not None:
            return mapped_peroxide_prefix(mol, Chem, root, blocked, atoms)
        # An unannotated ether/thioether can otherwise take the legacy branch
        # shortcut before visiting its acid-bearing carbon child. Reuse the
        # same source-owned child proof for either annotated or plain inputs.
        linker = mol.GetAtomWithIdx(root)
        children = [a.GetIdx() for a in linker.GetNeighbors()
                    if a.GetIdx() != blocked and a.GetAtomicNum() > 1]
        if (not linker.IsInRing() and not linker.GetIsotope()
                and len(children) == 1 and mol.GetAtomWithIdx(children[0]).GetAtomicNum() == 6
                and sum(mol.GetAtomWithIdx(i).GetAtomicNum() == 8 for i in atoms) >= 2
                and all(b.GetBondTypeAsDouble() == 1 for b in linker.GetBonds())):
            from .branch.carboxyalkyl import carboxyalkyl_prefix_proof
            child_atoms = atoms - {root}
            proof = carboxyalkyl_prefix_proof(mol, Chem, children[0], root, child_atoms)
            if proof is not None and proof.atoms | {root} == atoms:
                from .formatting import alkyl_to_alkoxy
                name = (alkyl_to_alkoxy(proof.name) if linker.GetAtomicNum() == 8
                        else '(' + proof.name + ')sulfanyl')
                return name, atoms
    if mol.GetAtomWithIdx(root).IsInRing():
        from .mapped_ring_parents import ring_systems, rooted_benzo_imidazothiazole, rooted_prefix
        if root_atom.GetAtomicNum() == 6 and int(root_atom.GetChiralTag()) and len(atoms) >= 17:
            from .natural_products import mapped_steroid_yl
            steroid, owned = mapped_steroid_yl(mol, Chem, root, blocked, atoms)
            if steroid:
                return steroid, owned
        systems = ring_systems(mol)
        if len(atoms) >= 12 and mol.GetRingInfo().NumRings() >= 3:
            core = next((system for system in systems if root in system), frozenset())
            fused = rooted_benzo_imidazothiazole(mol, Chem, root, atoms, core)
            if fused:
                _record_branch_ring_atoms(mol, Chem, root, blocked, fused, atoms,
                    (i for i in atoms if mol.GetAtomWithIdx(i).IsInRing()))
                return fused, atoms
        # The assembly builder's first admission requires at least two
        # complete cyclic systems in this scope. Reuse the identical physical
        # partition before entering its exact native-state proof machinery.
        if sum(core <= atoms for core in systems) >= 2:
            from .tree_ring_assemblies import ring_assembly_prefix_outcome
            assembly = ring_assembly_prefix_outcome(mol,Chem,root,blocked,atoms)
            if assembly.status!='outside-scope':
                if assembly.status!='complete':return '',frozenset()
                record=assembly.decision
                ring_atoms=frozenset().union(*record.tree.systems)
                _record_branch_ring_atoms(mol,Chem,root,blocked,record.name,atoms,ring_atoms)
                return record.name,atoms
            if len(atoms) >= 12 and root_atom.GetAtomicNum() == 6 and root_atom.GetIsAromatic():
                from .ring_assembly_methods import rooted_biphenyl

                biphenyl = rooted_biphenyl(mol, Chem, root, blocked, atoms, systems)
                if biphenyl:
                    return biphenyl, atoms
        text,owned = rooted_prefix(mol,Chem,root,blocked,atoms)
        if text:
            _record_branch_ring_atoms(mol, Chem, root, blocked, text, owned,
                (i for i in owned if mol.GetAtomWithIdx(i).IsInRing()))
            return text,owned
        if len(atoms) >= 8 and not root_atom.GetIsAromatic() and mol.GetRingInfo().NumRings() >= 3:
            from .spiro_component_rules import mapped_component_spiro_yl
            spiro, owned = mapped_component_spiro_yl(mol, Chem, root, blocked, atoms)
            if spiro:
                return spiro, owned
    if mol.GetAtomWithIdx(root).GetAtomicNum() in {8, 16, 34, 52}:
        from .peroxol_components import (chalcogen_peroxol_prefix,
                                         peroxol_site_type, recognize_peroxols)
        # P-63.4.2 has distinct order-specific prefix terms; dispatch the
        # complete mapped unit before generic sulfur/alkoxy fallback.
        is_chalcogen_peroxol = any(
            group.owner == blocked and group.center == root
            and peroxol_site_type(mol, group) not in {None, "OOH"}
            for group in recognize_peroxols(mol)
        )
        if is_chalcogen_peroxol:
            return chalcogen_peroxol_prefix(mol, Chem, root, blocked, atoms)
    if mol.GetAtomWithIdx(root).GetAtomicNum() == 7:
        # A neutral azo bond under a separately selected ketone must own its
        # source E/Z and its entire distal mapped branch as one prefix.
        from .branch.diazenyl_components import mapped_diazenyl_prefix
        diazenyl = mapped_diazenyl_prefix(mol, Chem, root, blocked, atoms)
        if diazenyl is not None:
            return diazenyl
        # The amide-derived prefix is local to this original-graph branch;
        # its producer requires a separately proved senior characteristic
        # group and complete atom ownership before changing the construction.
        from .branch.amido_components import mapped_amido_prefix, mapped_amido_ring_atoms
        amido = mapped_amido_prefix(mol, Chem, root, blocked, atoms)
        if amido is not None:
            # Propagate pendant rings only when their complete mapped
            # constructions have also proved the original atom ownership.
            from .mapped_ring_parents import ring_systems
            acyl = [neighbor for neighbor in root_atom.GetNeighbors()
                    if neighbor.GetIdx() != blocked and neighbor.GetAtomicNum() == 6
                    and any(bond.GetBondTypeAsDouble() == 2
                            and bond.GetOtherAtom(neighbor).GetAtomicNum() == 8
                            for bond in neighbor.GetBonds())]
            ring_cores = [core for core in ring_systems(mol) if core <= amido[1]
                          and len(acyl) == 1 and any(
                              mol.GetBondBetweenAtoms(acyl[0].GetIdx(), idx) is not None
                              for idx in core)]
            nested_rings = mapped_amido_ring_atoms(
                mol, Chem, root, blocked, amido[1], amido[0])
            if nested_rings is not None:
                _record_branch_ring_atoms(mol, Chem, root, blocked, amido[0], amido[1],
                                          nested_rings)
            elif len(ring_cores) == 1:
                _record_branch_ring_atoms(mol, Chem, root, blocked, amido[0], amido[1],
                                          ring_cores[0])
            return amido
        from .carbamate_methods import mapped_carbamate_amino_prefix
        carbamate = mapped_carbamate_amino_prefix(mol, Chem, root, blocked, atoms)
        if carbamate is not None:
            return carbamate
        from .branch.sulfonyl_components import mapped_sulfonamido_prefix
        sulfonamido = mapped_sulfonamido_prefix(mol, Chem, root, blocked, atoms)
        if sulfonamido is not None:
            return sulfonamido
        from .nitrogen_oxo_amide_prefixes import (
            direct_nitrogen_oxo_substituent,
            has_unresolved_direct_nitrogen_oxo_substituent,
            has_unsupported_nitrogen_oxo_chain, nitrogen_oxo_amide_prefix,
        )
        direct = direct_nitrogen_oxo_substituent(mol, root, blocked, atoms)
        if direct is not None:
            return direct.name, direct.atoms
        if has_unresolved_direct_nitrogen_oxo_substituent(mol, root, blocked):
            return "", frozenset()
        typed = nitrogen_oxo_amide_prefix(mol, root, blocked, atoms)
        if typed is not None:
            return typed.name, typed.atoms
        if has_unsupported_nitrogen_oxo_chain(mol, root, atoms):
            return "", frozenset()
    if mol.GetAtomWithIdx(root).GetAtomicNum()==16:
        from .branch.sulfonyl_components import mapped_sulfonyl_prefix
        typed=mapped_sulfonyl_prefix(mol,Chem,root,blocked,atoms)
        if typed is not None:return typed
        from .sulfur_prefixes import characteristic_sulfur_prefix
        typed=characteristic_sulfur_prefix(mol,Chem,root,blocked,atoms)
        if typed is not None:return typed
    if mol.GetAtomWithIdx(root).GetAtomicNum()==7 and any(
        b.GetBondTypeAsDouble()==2 and b.GetOtherAtom(mol.GetAtomWithIdx(root)).GetAtomicNum()==6
        for b in mol.GetAtomWithIdx(root).GetBonds()):
        from .oxime_components import oxime_site, oxime_imino_prefix
        oxime = oxime_site(mol, blocked)
        if oxime is not None and oxime.nitrogen == root:
            return oxime_imino_prefix(mol, Chem, root, blocked, atoms)
        from .imine_components import imino_prefix
        return imino_prefix(mol,Chem,root,blocked,atoms)
    if mol.GetAtomWithIdx(root).GetAtomicNum()==8 and any(
        n.GetAtomicNum()==8 for n in mol.GetAtomWithIdx(root).GetNeighbors()):
        from .peroxol_components import hydroperoxy_prefix
        return hydroperoxy_prefix(mol,Chem,root,blocked,atoms)
    if root_atom.GetAtomicNum() == 8:
        if any(a.GetAtomicNum() == 7 and a.GetIdx() != blocked for a in root_atom.GetNeighbors()):
            from .oxime_components import oxime_ether_oxy_prefix

            oxime_ether = oxime_ether_oxy_prefix(mol, Chem, root, blocked, atoms)
            if oxime_ether is not None:
                return oxime_ether
        from .mapped_aryloxy_prefix import mapped_aryloxy_prefix
        aryloxy = mapped_aryloxy_prefix(mol, Chem, root, blocked, atoms)
        if aryloxy is not None:
            text, owned, rings = aryloxy
            _record_branch_ring_atoms(mol, Chem, root, blocked, text, owned, rings)
            return text, owned
    if mol.GetAtomWithIdx(root).GetAtomicNum()==7:
        from .amine_components import ordinary_amine,amino_prefix
        if ordinary_amine(mol,root):
            return amino_prefix(mol,Chem,root,blocked,atoms)
    # Aminoalkyl branches need the same original-atom traversal even without
    # labels. The compatibility renderer cannot establish complex N ownership.
    from .amine_components import ordinary_amine
    amino_tree=any(ordinary_amine(mol,i) for i in atoms if mol.GetAtomWithIdx(i).GetAtomicNum()==7)
    # Sulfur characteristic groups nested in a carbon branch retain the same
    # local sulfo/sulfino/sulfamoyl/sulfanyl construction as direct prefixes.
    sulfur_tree=any(mol.GetAtomWithIdx(i).GetAtomicNum()==16
        and not mol.GetAtomWithIdx(i).IsInRing() and
        (mol.GetAtomWithIdx(i).GetTotalNumHs(includeNeighbors=True)==1 or
         any(n.GetAtomicNum()==8 and mol.GetBondBetweenAtoms(i,n.GetIdx()).GetBondTypeAsDouble()==2
             for n in mol.GetAtomWithIdx(i).GetNeighbors())) for i in atoms)
    cyclic_branch = any(mol.GetAtomWithIdx(i).IsInRing() for i in atoms)
    annotated = None
    if mol.GetAtomWithIdx(root).GetAtomicNum() == 6:
        atom_numbers = frozenset(mol.GetAtomWithIdx(i).GetAtomicNum() for i in atoms)
        if not atom_numbers <= {1, 6}:
            # A co-principal acid wholly inside this one-cut carbon branch is
            # cited as carboxy when an outside acid owns the parent suffix.
            # The bounded writer proves its complete rooted carbon path,
            # functional-site identity and original-graph annotation map.
            if (8 in atom_numbers and mol.GetAtomWithIdx(blocked).GetAtomicNum() in {8, 16}
                    and sum(mol.GetAtomWithIdx(i).GetAtomicNum() == 8 for i in atoms) >= 2):
                from .branch.carboxyalkyl import carboxyalkyl_prefix_proof
                carboxyalkyl = carboxyalkyl_prefix_proof(mol, Chem, root, blocked, atoms)
                if carboxyalkyl is not None:
                    return carboxyalkyl.name, carboxyalkyl.atoms
            # The shared P-29/P-46 role writer cannot own an amide carbonyl:
            # its oxo role requires every non-O neighbor to be carbon, and
            # no other role owns a double-bonded oxygen. Skip that guaranteed
            # miss before it rebuilds a whole-graph signature and cut scope.
            # The existing carbamoyl/amide path still proves the full branch.
            neighbors = tuple(root_atom.GetNeighbors())
            carbonyl = any(
                n.GetAtomicNum() == 8
                and mol.GetBondBetweenAtoms(root, n.GetIdx()).GetBondTypeAsDouble() == 2
                for n in neighbors
            )
            amide_carbonyl = carbonyl and any(
                n.GetAtomicNum() == 7
                and mol.GetBondBetweenAtoms(root, n.GetIdx()).GetBondTypeAsDouble() == 1
                for n in neighbors
            )
            if amide_carbonyl:
                if mol.GetAtomWithIdx(blocked).GetAtomicNum() == 7:
                    from .branch.preferred_acyl import n_attached_chain_acyl_prefix

                    acyl = n_attached_chain_acyl_prefix(mol, Chem, root, blocked, scope=atoms)
                    if acyl:
                        return acyl, atoms
                annotated = has_annotations(mol, atoms)
                if annotated or mol.GetAtomWithIdx(blocked).IsInRing():
                    amide_nitrogen = next((
                        n.GetIdx() for n in neighbors if n.GetIdx() != blocked and n.GetAtomicNum() == 7
                        and mol.GetBondBetweenAtoms(root, n.GetIdx()).GetBondTypeAsDouble() == 1
                    ), None)
                    if amide_nitrogen is not None:
                        from .carboxamide_prefixes import mapped_n_substituted_carbamoyl
                        carbamoyl = mapped_n_substituted_carbamoyl(
                            mol, Chem, root, blocked, amide_nitrogen, scope=atoms)
                        if carbamoyl is not None:
                            return carbamoyl
            else:
                if carbonyl and mol.GetAtomWithIdx(blocked).IsInRing():
                    from .carboxamide_prefixes import mapped_ring_oxamoyl

                    oxamoyl = mapped_ring_oxamoyl(mol, Chem, root, blocked, scope=atoms)
                    if oxamoyl is not None:
                        return oxamoyl
                # One shared writer owns the other functionalized carbon
                # branches, including mapped R/S, E/Z and isotope decoration.
                from .functionalized_carbon_prefixes import functionalized_carbon_prefix
                from .extended_inventory import selection_inventory
                inventory = selection_inventory(mol)
                proof, status = functionalized_carbon_prefix(
                    mol, Chem, root, blocked, tuple(inventory.principal))
                if status=="complete" and proof.scope==atoms:
                    return proof.text, proof.scope
        if annotated is None:
            annotated = has_annotations(mol, atoms)
        if annotated:
            from .annotated_substituent_selection import select_annotated_saturated_prefix
            typed = select_annotated_saturated_prefix(mol, Chem, root, blocked, atoms)
            if typed is not None:
                return typed
        # Both saturated writers rebuild their one-cut scope before admitting
        # it. Their necessary element/ring conditions are already known from
        # this exact subtree; do not reconstruct a scope that must decline.
        if not cyclic_branch:
            if atom_numbers == {6}:
                from .substituent_selection import mapped_saturated_carbon_prefix
                typed = mapped_saturated_carbon_prefix(mol, Chem, root, blocked, atoms)
                if typed is not None:
                    return typed
            elif atom_numbers <= {6, 9, 17, 35, 53}:
                from .substituent_selection import mapped_saturated_halo_carbon_prefix
                typed = mapped_saturated_halo_carbon_prefix(mol, Chem, root, blocked, atoms)
                if typed is not None:
                    return typed
        # Job 152 owns bounded systematic P-32 unsaturated carbon branches.
        # Admit its result only when it accounts for the complete original
        # subtree; a partial mapped result must never be mixed with the
        # compatibility renderer below.
        if not cyclic_branch and atom_numbers == {6} and any(
            bond.GetBondTypeAsDouble() in {2.0, 3.0}
            and bond.GetBeginAtomIdx() in atoms
            and bond.GetEndAtomIdx() in atoms
            for bond in iter_bonds(mol)
        ):
            from .unsaturated_substituent_methods import unsaturated_carbon_substituent
            unsaturated = unsaturated_carbon_substituent(mol, Chem, root, blocked)
            if unsaturated is not None and unsaturated.atoms == atoms:
                return unsaturated.text, atoms
    if annotated is None:
        annotated = has_annotations(mol, atoms)
    if not annotated and not amino_tree and not sulfur_tree:
        # Compatibility spelling carries no ring-atom ownership. A ring on
        # a carbon/O/S branch must reach the recursive mapped construction
        # below, even when no isotope or stereo annotation triggers it.
        if not cyclic_branch:
            name = _simple_or_branch_prefix(mol, root, blocked)
            return (name, atoms) if name else ("", frozenset())
        if not mapped_cyclic_only:
            # The original admissions above have already declined this exact
            # component. Keep their outcome while the compatibility renderer
            # runs, then continue at the recursive writer rather than repeating
            # every scope construction. Unkeyable or changed native state uses
            # the original re-entry, including all its admissions.
            try:
                state = mol.ToBinary(Chem.PropertyPickleOptions.AllProps)
            except (AttributeError, TypeError, ValueError, RuntimeError):
                state = None
            guards = frozenset(mol_cache_get(mol, "branch_prefix_in_progress", ()))
            general = _simple_or_branch_prefix(mol, root, blocked)
            try:
                unchanged = (state is not None
                             and mol.ToBinary(Chem.PropertyPickleOptions.AllProps) == state
                             and frozenset(mol_cache_get(mol, "branch_prefix_in_progress", ())) == guards)
            except (AttributeError, TypeError, ValueError, RuntimeError):
                unchanged = False
            if unchanged:
                # The local carboxamide writer may already have proved this
                # exact C/O/N+ligand unit while rendering general. Reuse only
                # its identical native graph, spelling and complete atom set.
                carbamoyl = mol_cache_get(mol, 'mapped_carbamoyl_prefixes', {}).get((id(Chem), root, blocked))
                if carbamoyl is not None and carbamoyl == (state, general, atoms):
                    return general, atoms
                name, owned = _mapped_branch_recursive(mol, Chem, root, blocked, atoms)
            else:
                name, owned = _mapped_branch(mol, Chem, root, blocked, mapped_cyclic_only=True)
            if name and (not general or name == general):
                return name, owned
            if name and owned == atoms and _equivalent_prefix_enclosures(name, general):
                # Keep the compatibility writer's enclosing levels, but
                # carry over the ring witness because every non-enclosure
                # character and the complete original branch are identical.
                rings = mapped_branch_ring_atoms(mol, Chem, root, blocked, name, owned)
                _record_branch_ring_atoms(mol, Chem, root, blocked, general, owned, rings)
                return general, owned
            # A failed mapped writer does not invalidate an existing general
            # spelling. A different mapped component cannot lend that word
            # its ring ownership: its P-57 preference/grammar join is still
            # unproved. The metadata key includes the exact emitted text.
            return (general, atoms) if general else ("", frozenset())
    return _mapped_branch_recursive(mol, Chem, root, blocked, atoms)


def _mapped_branch_recursive(mol, Chem, root, blocked, atoms, *, general_branches=False):
    """Continue a fully admitted mapped branch with its original subtree."""
    atom = mol.GetAtomWithIdx(root)
    resolve_branch = mapped_general_branch if general_branches else mapped_branch
    if (atom.GetAtomicNum() == 8 and
            mol.GetBondBetweenAtoms(root, blocked).GetBondTypeAsDouble() == 1 and
            atom.GetTotalNumHs(includeNeighbors=True) == 1 and
            all(i == root or mol.GetAtomWithIdx(i).GetAtomicNum() == 1 for i in atoms)):
        # P-82.6.2 cites a labelled subordinate hydroxy unit as (2H)hydroxy.
        # Its single O-H position takes no H1 count; skeletal D elsewhere
        # still uses the ordinary count and locant rules.
        h_atoms = tuple(i for i in atoms if i != root)
        parts = tuple(('H', mol.GetAtomWithIdx(i).GetIsotope(), '', True)
                      for i in h_atoms if mol.GetAtomWithIdx(i).GetIsotope())
        text = decorate_component(mol, Chem, MappedComponent(
            'hydroxy', tuple((i, '') for i in atoms), atoms,
            local_isotope_atoms=frozenset(i for i in h_atoms
                                         if mol.GetAtomWithIdx(i).GetIsotope()),
            isotope_descriptor_parts=parts,
        ))
        return (text, atoms) if text else ('', frozenset())
    # Terminal characteristic groups own their attached isotope H atoms.
    if atom.GetAtomicNum() in {7, 8, 9, 17, 35, 53} and all(
        i == root or mol.GetAtomWithIdx(i).GetAtomicNum() == 1 for i in atoms
    ):
        order = mol.GetBondBetweenAtoms(root, blocked).GetBondTypeAsDouble()
        if atom.GetFormalCharge() and not (
            atom.GetAtomicNum() == 7
            and atom.GetFormalCharge() == 1
            and atom.GetTotalNumHs(includeNeighbors=True) == 3
        ):
            return "", frozenset()
        if order != 1 and not (
            order == 2 and atom.GetAtomicNum() == 8 and not atom.GetFormalCharge()
        ):
            return "", frozenset()
        name = {
            7: "amino",
            8: "hydroxy",
            9: "fluoro",
            17: "chloro",
            35: "bromo",
            53: "iodo",
        }[atom.GetAtomicNum()]
        if order == 2:
            name = "oxo"
        if atom.GetFormalCharge() == 1:
            name = "azaniumyl"
        attached_h = [i for i in atoms if i != root]
        if attached_h and atom.GetFormalCharge():
            return "", frozenset()
        if attached_h:
            hydrogen_terms = {1: "protio", 2: "deuterio", 3: "tritio"}
            if atom.GetAtomicNum() not in {7, 8} or any(
                mol.GetAtomWithIdx(i).GetIsotope() not in hydrogen_terms
                for i in attached_h
            ):
                return "", frozenset()
            # An attached isotope H is an explicit substituent on the terminal
            # heteroatom. It is not an unlocanted isotope on a yet-unbuilt OH.
            terms = [
                ("", hydrogen_terms[mol.GetAtomWithIdx(i).GetIsotope()])
                for i in attached_h
            ]
            name = format_prefixes(terms, omit_locants=True) + (
                "oxy" if atom.GetAtomicNum() == 8 else "amino"
            )
            component = MappedComponent(
                name,
                ((root, ""),),
                frozenset({root}),
                embedded_atoms=frozenset(attached_h),
            )
            return decorate_component(mol, Chem, component), atoms
        mapping = hydrogen_bindings(mol, {root: ""})
        value = decorate_component(
            mol, Chem, MappedComponent(name, tuple(mapping.items()), frozenset(mapping))
        )
        return value, atoms
    if atom.GetAtomicNum() in {8,16} and not atom.IsInRing() and not atom.GetIsotope():
        children=[n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx()!=blocked and n.GetAtomicNum()>1]
        if len(children)==1 and all(b.GetBondTypeAsDouble()==1 for b in atom.GetBonds()):
            if atom.GetAtomicNum() == 8:
                from .branch.contracted_alkoxy import mapped_contracted_alkoxy
                contracted = mapped_contracted_alkoxy(mol, Chem, root, blocked)
                if contracted is not None and contracted[1] == atoms:
                    return contracted
            text,owned=resolve_branch(mol,Chem,children[0],root)
            if text and owned|{root}==atoms:
                from .formatting import alkyl_to_alkoxy
                name = alkyl_to_alkoxy(text) if atom.GetAtomicNum()==8 else '('+text+')sulfanyl'
                _record_branch_ring_atoms(mol, Chem, root, blocked, name, atoms,
                    mapped_branch_ring_atoms(mol, Chem, children[0], root, text, owned))
                return name,atoms
    if atom.GetAtomicNum() != 6 or atom.IsInRing():
        return "", frozenset()
    # P-66.5.1.1.4/P-82: the C and N of an annotated cyano prefix remain one
    # indivisible component. Neither atom is an amino substituent on a carbon
    # parent, and its isotope descriptor belongs to this prefix's local scope.
    nitrile_n = [n.GetIdx() for n in atom.GetNeighbors()
                 if n.GetIdx() != blocked and n.GetAtomicNum() == 7
                 and mol.GetBondBetweenAtoms(root, n.GetIdx()).GetBondTypeAsDouble() == 3]
    if len(nitrile_n) == 1 and atoms == {root, nitrile_n[0]}:
        value = decorate_component(mol, Chem, MappedComponent(
            'cyano', ((root, ''), (nitrile_n[0], '')), atoms))
        return value, atoms if value else frozenset()
    # P-35.2/P-66.5: the C and N of a subordinate nitrile remain one cyano
    # branch. Absorbing its carbon into this compatibility chain would let
    # the internally rooted writer treat a triple-bonded N as an amino arm.
    from .extended_inventory import selection_inventory
    cyano_centers = {group.center for group in selection_inventory(mol).groups
                     if group.kind == 'nitrile'}
    carbon,stack=set(),[root]
    while stack:
        i=stack.pop()
        if i in carbon:continue
        if mol.GetAtomWithIdx(i).IsInRing():return '',frozenset()
        carbon.add(i)
        stack.extend(n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors()
            if n.GetIdx() in atoms and n.GetAtomicNum()==6 and not n.IsInRing()
            and n.GetIdx() not in carbon and n.GetIdx() not in cyano_centers)
    path = [root]
    while True:
        next_atoms = [
            n.GetIdx()
            for n in mol.GetAtomWithIdx(path[-1]).GetNeighbors()
            if n.GetIdx() in carbon and n.GetIdx() not in path
        ]
        if not next_atoms:
            break
        if len(next_atoms) != 1:
            return _internally_rooted_carbon_prefix(
                mol, Chem, root, blocked, atoms, carbon, general_branches=general_branches)
        path.append(next_atoms[0])
    if set(path) != carbon or len(path) not in ROOTS:
        return "", frozenset()
    prefixes = []
    embedded = set()
    proved_ring_atoms = set()
    prefix_ownership = []
    for pos, idx in enumerate(path, 1):
        for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
            n = neighbor.GetIdx()
            if n == blocked or n in carbon or neighbor.GetAtomicNum() == 1:
                continue
            if (
                n not in atoms
            ):
                return "", frozenset()
            order=mol.GetBondBetweenAtoms(idx,n).GetBondTypeAsDouble()
            if order != 1 and not (order == 2 and neighbor.GetAtomicNum() in {7, 8}):
                return "", frozenset()
            prefix, owned = resolve_branch(mol, Chem, n, idx)
            if not prefix or (general_branches and owned & (carbon | embedded)):
                return "", frozenset()
            prefixes.append((str(pos), prefix))
            embedded.update(owned)
            if order == 2:
                prefix_ownership.append((n, idx, frozenset(owned), prefix))
            proved_ring_atoms.update(mapped_branch_ring_atoms(mol, Chem, n, idx, prefix, owned))
    from .branch_rules import parent_chain_name

    parent = complete_parent_name(parent_chain_name(mol, path))
    if parent.endswith("ane"):
        parent = ROOTS[len(path)] + "yl"
    else:
        parent = parent.removesuffix("e") + "-1-yl"
    mapping = hydrogen_bindings(mol, {idx: str(pos) for pos, idx in enumerate(path, 1)})
    component = MappedComponent(
        parent,
        tuple(mapping.items()),
        frozenset(mapping),
        tuple(prefixes),
        frozenset(embedded),
        systematic_parent=True,
        omit_methyl_prefix_locant=len(path)==1,
        prefix_branch_ownership=tuple(prefix_ownership),
    )
    if set(mapping)|embedded!=set(atoms):return '',frozenset()
    # P-57.1.2 retains exactly unsubstituted benzyl; P-57.4 consequently
    # prefers benzyloxy in the enclosing O branch. Retain the complete
    # mapped phenyl child's ring ownership rather than recovering this word
    # from compatibility spelling. Labels/substitution use their own map.
    heavy = {i for i in atoms if mol.GetAtomWithIdx(i).GetAtomicNum() > 1}
    retained_benzyl = (
        len(path) == 1 and prefixes == [('1', 'phenyl')]
        and len(proved_ring_atoms) == 6
        and heavy == proved_ring_atoms | {root}
        and atom.GetTotalNumHs(includeNeighbors=True) == 2
        and not has_annotations(mol, atoms)
        and all(not mol.GetAtomWithIdx(i).GetFormalCharge()
                and not mol.GetAtomWithIdx(i).GetNumRadicalElectrons() for i in atoms)
        and all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6
                and mol.GetAtomWithIdx(i).GetIsAromatic() for i in proved_ring_atoms)
    )
    name = 'benzyl' if retained_benzyl else decorate_component(mol, Chem, component)
    _record_branch_ring_atoms(mol, Chem, root, blocked, name, atoms, proved_ring_atoms)
    return name, atoms


def _internally_rooted_carbon_prefix(mol,Chem,root,blocked,atoms,carbon,*,allow_branched=False,
                                     general_branches=False):
    """P-29/P-46 mapped carbon prefixes whose root need not be an end atom.

    Callers with a bounded carbon tree can admit its maximal paths and
    separately map each pendant arm.
    Free valence and chemical numbering precede isotope/configuration choices.
    Original CIP is retained.
    """
    from .multifunctional_parent import _paths
    from .branch_rules import parent_chain_name
    from .parent_annotations import isotope_seniority,substituent_isotopes,stereo_numbering
    from .parent_selection import locant_key
    resolve_branch = mapped_general_branch if general_branches else mapped_branch
    if mol.GetBondBetweenAtoms(root,blocked).GetBondTypeAsDouble()!=1:
        return '',frozenset()
    options=[]
    if not allow_branched and any(
            sum(n.GetIdx() in carbon for n in mol.GetAtomWithIdx(i).GetNeighbors()) > 2 for i in carbon):
        return '',frozenset()
    paths=[path for path in _paths(mol,carbon)
           if root in path and (allow_branched or set(path)==carbon)]
    if not paths:return '',frozenset()
    longest=max(map(len,paths))
    for path in paths:
        if len(path)!=longest:continue
        for seq in (path,tuple(reversed(path))):
            mapping={idx:str(j) for j,idx in enumerate(seq,1)};prefixes=[];embedded=set();failed=False
            proved_ring_atoms = set()
            prefix_ownership = []
            for idx in seq:
                for n in mol.GetAtomWithIdx(idx).GetNeighbors():
                    other=n.GetIdx()
                    if other not in atoms or other in mapping or n.GetAtomicNum()==1:continue
                    order = mol.GetBondBetweenAtoms(idx, other).GetBondTypeAsDouble()
                    if order != 1 and not (order == 2 and n.GetAtomicNum() in {7, 8}):
                        failed = True
                        break
                    text,owned=resolve_branch(mol,Chem,other,idx)
                    if not text or owned&(mapping.keys()|embedded):failed=True;break
                    prefixes.append((mapping[idx],text));embedded.update(owned)
                    if order == 2:
                        prefix_ownership.append((other, idx, frozenset(owned), text))
                    proved_ring_atoms.update(mapped_branch_ring_atoms(mol, Chem, other, idx, text, owned))
                if failed:break
            bindings=hydrogen_bindings(mol,mapping)
            if failed or set(bindings)|embedded!=set(atoms):continue
            parent=complete_parent_name(parent_chain_name(mol,seq));loc=mapping[root]
            orders=[mol.GetBondBetweenAtoms(a,b).GetBondTypeAsDouble() for a,b in zip(seq,seq[1:])]
            if parent.endswith('ane') and loc=='1':parent=parent[:-3]+'yl'
            elif parent.endswith('e'):parent=parent[:-1]+'-'+loc+'-yl'
            else:continue
            name=decorate_component(mol,Chem,MappedComponent(parent,tuple(bindings.items()),frozenset(bindings),
                tuple(prefixes),frozenset(embedded),systematic_parent=True,
                prefix_branch_ownership=tuple(prefix_ownership)))
            iso=isotope_seniority(mol,bindings),substituent_isotopes(mol,bindings)
            stereo=stereo_numbering(mol,bindings)
            if not name or any(v is None for v in iso) or stereo is None:continue
            key=(-len(seq),-sum(o > 1 for o in orders),-orders.count(2),int(loc),
                 tuple(j for j,o in enumerate(orders,1) if o>1),
                 tuple(j for j,o in enumerate(orders,1) if o==2),
                 -len(prefixes) if allow_branched else 0,
                 tuple(sorted(locant_key(p) for p,t in prefixes)),
                 tuple(locant_key(p) for p,t in sorted(prefixes,key=lambda x:(prefix_alphabetization_key(x[1]),locant_key(x[0])))),iso,stereo)
            options.append((key,name,frozenset(proved_ring_atoms)))
    if not options:return '',frozenset()
    key=min(r[0] for r in options);finalists=[row for row in options if row[0]==key]
    names={name for _key,name,_rings in finalists}
    if len(names)!=1:return '',frozenset()
    name=names.pop()
    proved=frozenset.intersection(*(row[2] for row in finalists))
    _record_branch_ring_atoms(mol,Chem,root,blocked,name,atoms,proved)
    return name,frozenset(atoms)


def fully_covered(mol, owned, embedded):
    covered = set(owned) | set(embedded)
    atom_count = mol.GetNumAtoms()
    if mol.GetNumHeavyAtoms() == atom_count:
        # Every atom is required, independently of annotations. Native heavy
        # count supplies that fact without materializing each Python atom.
        return covered == set(range(atom_count))
    return covered == {
        a.GetIdx()
        for a in iter_atoms(mol)
        if a.GetAtomicNum() > 1 or a.GetIsotope() or int(a.GetChiralTag())
    }
