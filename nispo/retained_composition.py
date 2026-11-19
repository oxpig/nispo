"""Compose exact retained ring components through their graph attachment points.

The vocabulary is ordinary OPSIN nomenclature metadata, not molecule answers.
Only complete neutral components and unbranched carbon connectors are handled;
other chemistry remains with the general rule families.
"""


from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache

from .molecule_iteration import iter_atoms, iter_bonds
from .constants import MULT, ROOTS
from .formatting import format_prefixes
from .graph_model import graph_view
from .opsin_resource_utils import iter_opsin_token_records, primary_label
from .parent_selection import parent_frontier
from .ring_parent_selection import rendered_ring_facts
from .molecule_cache import mol_cache_dict


# Full standalone forms and their substituent stems are distinct grammatical roles.
_PARENTS = {
    "pyridin": "pyridine",
    "pyrimidin": "pyrimidine",
    "pyrazin": "pyrazine",
    "pyridazin": "pyridazine",
    "quinolin": "quinoline",
    "isoquinolin": "isoquinoline",
    "quinoxalin": "quinoxaline",
    "quinazolin": "quinazoline",
    "indol": "1H-indole",
    "naphthalen": "naphthalene",
    "anthracen": "anthracene",
    "phenanthren": "phenanthrene",
    "phenoxathiin": "phenoxathiine",
    "carbazol": "9H-carbazole",
    "benzofuran": "benzofuran",
    "benzofur": "benzofuran",
    "benzothiophen": "benzothiophene",
    "benzothiofuran": "benzothiophene",
    "benzimidazol": "1H-benzimidazole",
    "benzothiazol": "benzothiazole",
    "benzoxazol": "benzoxazole",
    "piperidin": "piperidine",
    "piperazin": "piperazine",
    "morpholin": "morpholine",
    "pyrrolidin": "pyrrolidine",
}


@dataclass(frozen=True)
class _RingVocabulary:
    stem: str
    parent: str
    mol: object
    labels: tuple[str, ...]
    radical_stem: str = ""
    ring_rank: tuple = ()


@dataclass(frozen=True)
class _Component:
    parent: str
    radical: str
    attachment_locant: str
    atom_indices: frozenset[int]
    # Rank by graph composition, not rendered-name spelling or length.
    parent_rank: tuple
    numbering_rank: tuple
    parent_base: str
    parent_prefixes: tuple[tuple[str, str], ...]
    hydroxy_count: int
    ketone_count: int
    parent_selection: object = None


def _signature(mol, atoms):
    return tuple(
        sorted(Counter(mol.GetAtomWithIdx(i).GetAtomicNum() for i in atoms).items())
    )


def _ring_seniority(query):
    from .parent_selection import ring_seniority_key

    elements = tuple(sorted(Counter(a.GetAtomicNum() for a in iter_atoms(query)
                                    if a.IsInRing()).items()))
    return ring_seniority_key(elements, query.GetRingInfo().NumRings())


@lru_cache(maxsize=8)
def _vocabulary(Chem):
    groups = defaultdict(list)
    seen = set()
    for record in iter_opsin_token_records(
        ("arylGroups.xml", "simpleCyclicGroups.xml")
    ):
        token = record.token
        if set(token.attrib) - {"value", "labels", "fusedRingNumbering"}:
            continue  # Token operations need separate inverse rules.
        aliases = (token.text or "").split("|")
        stem = next((alias for alias in aliases if alias in _PARENTS), None)
        if stem is None or stem in seen:
            continue
        query = Chem.MolFromSmiles(token.attrib.get("value", ""))
        labels = tuple(
            primary_label(label) for label in token.attrib.get("labels", "").split("/")
        )
        if (
            query is None
            or len(labels) != query.GetNumAtoms()
            or not all(a.IsInRing() for a in iter_atoms(query))
        ):
            continue
        groups[_signature(query, range(query.GetNumAtoms()))].append(
            _RingVocabulary(
                stem, _PARENTS[stem], query, labels, ring_rank=_ring_seniority(query)
            )
        )
        if stem == "anthracen":
            # An anthrone is the same retained skeleton with a carbonyl and
            # indicated hydrogen. Build the graph from the resource numbering.
            anthrone = Chem.RWMol(query)
            carbonyl_idx, saturated_idx = labels.index("10"), labels.index("9")
            for idx in (carbonyl_idx, saturated_idx):
                anthrone.GetAtomWithIdx(idx).SetIsAromatic(False)
                for bond in anthrone.GetAtomWithIdx(idx).GetBonds():
                    bond.SetBondType(Chem.BondType.SINGLE)
                    bond.SetIsAromatic(False)
            oxygen_idx = anthrone.AddAtom(Chem.Atom(8))
            anthrone.AddBond(carbonyl_idx, oxygen_idx, Chem.BondType.DOUBLE)
            core = anthrone.GetMol()
            Chem.SanitizeMol(core)
            groups[_signature(query, range(query.GetNumAtoms()))].append(
                _RingVocabulary(
                    "anthrone",
                    "9H-anthracen-10-one",
                    core,
                    labels + ("",),
                    "10-oxo-9H-anthracen",
                    _ring_seniority(core),
                )
            )
        seen.add(stem)
    benzene = Chem.MolFromSmiles("c1ccccc1")
    groups[_signature(benzene, range(6))].append(
        _RingVocabulary(
            "benzen",
            "benzene",
            benzene,
            tuple(map(str, range(1, 7))),
            ring_rank=_ring_seniority(benzene),
        )
    )
    return {key: tuple(value) for key, value in groups.items()}


def _safe_neutral_graph(ctx):
    cached = ctx.cache.topology_flags.get("retained_composition_supported")
    if cached is not None:
        return cached
    supported = all(
        not a.GetFormalCharge()
        and not a.GetIsotope()
        and not a.GetNumRadicalElectrons()
        and int(a.GetChiralTag()) == 0
        and a.GetAtomicNum() > 1
        for a in iter_atoms(ctx.mol)
    ) and all(int(b.GetStereo()) == 0 for b in iter_bonds(ctx.mol))
    ctx.cache.topology_flags["retained_composition_supported"] = supported
    return supported


def sole_characteristic_amide(ctx):
    """An ordinary acyclic amide is senior to a ring named using an oxo prefix."""
    if ctx.mol.GetRingInfo().NumRings() == 0 or not _safe_neutral_graph(ctx):
        return False
    profile = ctx.profile
    return (
        profile.amide_acyl_motif_count == 1
        and profile.carbonyl_count == 1
        and all(
            a.GetAtomicNum() in {6, 7, 8, 9, 17, 35, 53} for a in iter_atoms(ctx.mol)
        )
    )


def _locant_key(locant):
    number = "".join(char for char in locant if char.isdigit())
    return (int(number) if number else 999, locant)


def _component(
    ctx, atoms, attachment, connector, *, principal_group=None, attachment_prefix=None
):
    mol, Chem = ctx.mol, ctx.Chem
    selected = []
    for entry in _vocabulary(Chem).get(_signature(mol, atoms), ()):
        query = entry.mol
        for match in ctx.substruct_matches(query):
            core_atoms = frozenset(match)
            ring_atoms = frozenset(
                match[a.GetIdx()] for a in iter_atoms(query) if a.IsInRing()
            )
            if ring_atoms != atoms:
                continue
            # RDKit molecule substructure matching does not enforce all hydrogen
            # states. Account for substitution explicitly so an N-H tautomer
            # cannot inherit another tautomer's numbering and indicated hydrogen.
            if any(
                mol.GetAtomWithIdx(idx).GetTotalNumHs()
                + sum(
                    mol.GetBondBetweenAtoms(
                        idx, neighbor.GetIdx()
                    ).GetBondTypeAsDouble()
                    for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors()
                    if neighbor.GetIdx() not in core_atoms
                )
                != query.GetAtomWithIdx(query_idx).GetTotalNumHs()
                for query_idx, idx in enumerate(match)
            ):
                continue
            # A non-induced substructure match must not conceal additional core bonds.
            if (
                sum(
                    b.GetBeginAtomIdx() in core_atoms
                    and b.GetEndAtomIdx() in core_atoms
                    for b in iter_bonds(mol)
                )
                != query.GetNumBonds()
            ):
                continue
            locants = dict(zip(match, entry.labels, strict=True))
            if attachment is not None:
                query_atom = query.GetAtomWithIdx(match.index(attachment))
                if query_atom.GetAtomicNum() != 6 or query_atom.GetTotalNumHs() == 0:
                    continue
            prefixes = []
            owned = set(core_atoms)
            valid = True
            for idx in match:
                for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
                    nidx = neighbor.GetIdx()
                    if nidx in core_atoms or (idx == attachment and nidx == connector):
                        continue
                    bond = mol.GetBondBetweenAtoms(idx, nidx)
                    if (
                        bond.GetBondType() != Chem.BondType.SINGLE
                        or neighbor.GetDegree() != 1
                    ):
                        valid = False
                        break
                    prefix = {9: "fluoro", 17: "chloro", 35: "bromo", 53: "iodo"}.get(
                        neighbor.GetAtomicNum()
                    )
                    if neighbor.GetAtomicNum() == 6 and neighbor.GetTotalNumHs() == 3:
                        prefix = "methyl"
                    elif neighbor.GetAtomicNum() == 8 and neighbor.GetTotalNumHs() == 1:
                        prefix = "hydroxy"
                    if prefix is None:
                        valid = False
                        break
                    prefixes.append((locants[idx], prefix))
                    owned.add(nidx)
                if not valid:
                    break
            if not valid:
                continue
            hydroxy_locants = sorted(
                (locant for locant, prefix in prefixes if prefix == "hydroxy"),
                key=_locant_key,
            )
            ketone_count = int(entry.stem == "anthrone")
            parent_base = entry.parent
            parent_prefixes = prefixes
            if principal_group == "hydroxy" and hydroxy_locants:
                # P-44.1.1: express the principal alcohol group as a suffix.
                parent_prefixes = [
                    (loc, prefix) for loc, prefix in prefixes if prefix != "hydroxy"
                ]
                if parent_base == "benzene" and len(hydroxy_locants) == 1:
                    parent_base = "phenol"
                else:
                    count = len(hydroxy_locants)
                    stem = (
                        parent_base[:-1]
                        if count == 1 and parent_base.endswith("e")
                        else parent_base
                    )
                    multiplier = "" if count == 1 else MULT[count]
                    parent_base = (
                        stem + "-" + ",".join(hydroxy_locants) + "-" + multiplier + "ol"
                    )
            prefix_text = format_prefixes(parent_prefixes)
            join = "-" if entry.parent[0].isdigit() else ""
            parent = prefix_text + join + parent_base if prefix_text else parent_base
            locant = locants[attachment] if attachment is not None else ""
            radical_stem = entry.radical_stem or (
                entry.parent[:-1] if entry.parent.endswith("e") else entry.parent
            )
            # Indicated hydrogen belongs to the parent vocabulary and is retained.
            radical = (
                "phenyl" if entry.stem == "benzen" else f"{radical_stem}-{locant}-yl"
            )
            radical_prefix_text = format_prefixes(prefixes)
            if radical_prefix_text:
                radical = radical_prefix_text + join + radical
            rank = (
                _locant_key(locant),
                tuple(sorted(_locant_key(x) for x, _ in prefixes)),
            )
            if attachment_prefix is not None:
                # Parent numbering includes all prefix attachments, after suffix
                # locants. Substituent numbering instead starts at attachment.
                numbered_prefixes = parent_prefixes + [(locant, attachment_prefix)]
                rank = (
                    tuple(map(_locant_key, hydroxy_locants))
                    if principal_group == "hydroxy"
                    else (),
                    tuple(sorted(_locant_key(x) for x, _ in numbered_prefixes)),
                    tuple(
                        (_name, _locant_key(x))
                        for x, _name in sorted(
                            numbered_prefixes, key=lambda pair: pair[1]
                        )
                    ),
                )
            evidence_prefixes = (parent_prefixes + [(locant, attachment_prefix)]
                                 if attachment_prefix is not None else prefixes)
            evidence_name = (format_prefixes(evidence_prefixes) + join + parent_base
                             if attachment_prefix is not None else radical if attachment is not None else parent)
            selected.append(
                _Component(
                    parent,
                    radical,
                    locant,
                    frozenset(owned),
                    entry.ring_rank,
                    rank,
                    parent_base,
                    tuple(parent_prefixes),
                    len(hydroxy_locants),
                    ketone_count,
                    rendered_ring_facts(mol, core_atoms, locants, entry.parent, evidence_prefixes, evidence_name,
                        suffix_centers=[idx for idx in ring_atoms if locants[idx] in hydroxy_locants]
                            if principal_group == "hydroxy" else (),
                        suffix_locants=hydroxy_locants if principal_group == "hydroxy" else (),
                        attachment_locants=[locant] if attachment is not None and attachment_prefix is None else ()),
                )
            )
    selected = parent_frontier(selected, lambda component: component.parent_selection)
    return (
        min(selected, key=lambda component: component.numbering_rank)
        if selected
        else None
    )


class RetainedComponentCompositionRule:
    family = "retained-component-composition"

    def candidates(self, ctx):
        from .rule_engine import NameCandidate

        if not _safe_neutral_graph(ctx):
            return
        view = graph_view(ctx.mol, ctx.Chem)
        systems = view.ring_systems
        name = ""
        parent_facts = None
        if len(systems) == 1 and len(systems[0]) == view.atom_count:
            component = _component(ctx, systems[0], None, None)
            if component is not None:
                name = component.parent
                parent_facts = component.parent_selection
        elif len(systems) == 2 and any(len(system) > 6 for system in systems):
            name = self._connected_components(ctx, systems, view)
            parent_facts = mol_cache_dict(ctx.mol, "composition_parent_selection").get(name)
        if name:
            yield NameCandidate(
                name,
                self.family,
                13,
                parent_atoms=frozenset(range(view.atom_count)),
                source_family=self.family,
                render_rank=1,
                estimated_length=len(name),
                parent_selection=parent_facts,
            )

    def _connected_components(self, ctx, systems, view):
        mol, Chem = ctx.mol, ctx.Chem
        left_atoms, right_atoms = systems
        connections = []
        for i in left_atoms:
            for j in view.neighbors[i]:
                if j in right_atoms:
                    connections.append((i, j))
        if connections:
            if len(connections) != 1:
                return ""
            i, j = connections[0]
            if mol.GetBondBetweenAtoms(i, j).GetBondType() != Chem.BondType.SINGLE:
                return ""
            left = _component(ctx, left_atoms, i, j)
            right = _component(ctx, right_atoms, j, i)
            if (
                not left
                or not right
                or left.atom_indices | right.atom_indices
                != frozenset(range(view.atom_count))
            ):
                return ""
            return self._ring_parent(ctx, systems, ((i, j), (j, i)), left, right, 0)

        # Distinguish the connector from carbon-containing peripheral groups.
        # Only a non-ring component touching both ring systems can be the chain.
        remaining = set(range(view.atom_count)) - left_atoms - right_atoms
        connectors = []
        while remaining:
            component = set()
            stack = [next(iter(remaining))]
            while stack:
                idx = stack.pop()
                if idx not in remaining:
                    continue
                remaining.remove(idx)
                component.add(idx)
                stack.extend(
                    neighbor
                    for neighbor in view.neighbors[idx]
                    if neighbor in remaining
                )
            if all(
                any(
                    neighbor in system
                    for idx in component
                    for neighbor in view.neighbors[idx]
                )
                for system in systems
            ):
                connectors.append(component)
        if len(connectors) != 1:
            return ""
        carbon_atoms = {idx for idx in connectors[0] if view.atomic_nums[idx] == 6}
        if not carbon_atoms or len(carbon_atoms) not in ROOTS:
            return ""
        endpoints = []
        for system in systems:
            links = [
                (i, j) for i in system for j in view.neighbors[i] if j in carbon_atoms
            ]
            if len(links) != 1:
                return ""
            endpoints.append(links[0])
        (i, start), (j, end) = endpoints
        path = (
            (start,) if start == end else tuple(Chem.GetShortestPath(mol, start, end))
        )
        if set(path) != carbon_atoms:
            return ""
        oxygen_atoms = set()
        ketone_locants = []
        for pos, k in enumerate(path, 1):
            allowed = {path[pos - 2]} if pos > 1 else {i}
            allowed.add(path[pos] if pos < len(path) else j)
            for neighbor in view.neighbors[k]:
                bond = mol.GetBondBetweenAtoms(k, neighbor)
                if neighbor in allowed:
                    if bond.GetBondType() != Chem.BondType.SINGLE:
                        return ""
                elif (
                    view.atomic_nums[neighbor] == 8
                    and len(view.neighbors[neighbor]) == 1
                    and bond.GetBondType() == Chem.BondType.DOUBLE
                ):
                    oxygen_atoms.add(neighbor)
                    ketone_locants.append(pos)
                else:
                    return ""
        left = _component(ctx, left_atoms, i, start)
        right = _component(ctx, right_atoms, j, end)
        if not left or not right:
            return ""
        if (
            left.atom_indices | right.atom_indices | carbon_atoms | oxygen_atoms
            != frozenset(range(view.atom_count))
        ):
            return ""
        # P-44.1.1 precedes P-52.2.8. A ketone chain is selected only when
        # it carries more principal groups than either individual ring system.
        # Otherwise a plain linker is a substituent on the senior ring.
        ring_ketones = max(left.ketone_count, right.ketone_count)
        if not ketone_locants:
            return self._ring_parent(ctx, systems, endpoints, left, right, len(path))
        if len(ketone_locants) <= ring_ketones:
            # A carbonyl-bearing linker used as a ring substituent requires
            # further inverse rules; leave that chemistry to the general engine.
            return ""
        reversed_locants = sorted(len(path) + 1 - locant for locant in ketone_locants)
        if reversed_locants < ketone_locants:
            ketone_locants = reversed_locants
            left, right = right, left
        prefixes = [("1", left.radical), (str(len(path)), right.radical)]
        parent = ROOTS[len(path)] + "ane"
        if ketone_locants:
            count = len(ketone_locants)
            if count > 2:
                return ""
            suffix = "one" if count == 1 else "dione"
            parent = (
                ROOTS[len(path)]
                + ("an" if count == 1 else "ane")
                + "-"
                + ",".join(map(str, ketone_locants))
                + "-"
                + suffix
            )
            if len(path) == 1:
                parent = "methanone"
        return format_prefixes(prefixes) + parent

    def _ring_parent(self, ctx, systems, endpoints, left, right, chain_length):
        principal_group = (
            "ketone"
            if left.ketone_count or right.ketone_count
            else "hydroxy"
            if left.hydroxy_count or right.hydroxy_count
            else None
        )

        def seniority(component):
            count = (
                component.ketone_count
                if principal_group == "ketone"
                else component.hydroxy_count
                if principal_group == "hydroxy"
                else 0
            )
            return (
                -count,
                component.parent_rank,
                component.numbering_rank,
                component.parent,
            )

        alternatives = []
        for parent_index, original in enumerate((left, right)):
            substituent = right if parent_index == 0 else left
            prefix = substituent.radical
            if chain_length == 1:
                prefix = f"({prefix})methyl"
            elif chain_length > 1:
                prefix = f"{chain_length}-({prefix}){ROOTS[chain_length]}yl"
            attachment, connector = endpoints[parent_index]
            parent = _component(ctx, systems[parent_index], attachment, connector,
                principal_group=principal_group, attachment_prefix=prefix)
            if parent is None:
                continue
            prefixes = list(parent.parent_prefixes) + [(parent.attachment_locant, prefix)]
            join = "-" if parent.parent_base[0].isdigit() else ""
            name = format_prefixes(prefixes) + join + parent.parent_base
            alternatives.append((seniority(original), name, parent.parent_selection))
        alternatives = parent_frontier(alternatives, lambda row: row[2])
        if not alternatives:
            return ""
        _, name, facts = min(alternatives, key=lambda row: row[:2])
        mol_cache_dict(ctx.mol, "composition_parent_selection")[name] = facts
        return name
