"""Graph-aware retained fused parent names and indicated-hydrogen numbering."""

from functools import lru_cache

from .molecule_cache import mol_cache_get, mol_cache_set
from .parent_join_presentation import normalize_fusion_component_spelling
from .molecule_iteration import iter_atoms
from .molecule_iteration import iter_atoms


# Explicit P-25 retained-hydride vocabulary. This deliberately does not adopt
# every OPSIN alias or every legacy parent-name normalization as a PIN.
RETAINED_FUSED_HYDRIDES = frozenset({
    'naphthalene', 'indene', 'indole', 'isoindole', 'indazole',
    'benzimidazole', 'quinoline', 'isoquinoline', 'quinazoline', 'quinoxaline',
    'cinnoline', 'phthalazine', 'purine', 'pteridine', 'benzofuran',
    'acridine', 'carbazole', 'xanthene', 'anthracene',
    'isobenzofuran', 'benzothiophene', '1-benzopyran', '2-benzopyran',
    '1,5-naphthyridine', '1,6-naphthyridine', '1,7-naphthyridine',
    '1,8-naphthyridine', '2,6-naphthyridine', '2,7-naphthyridine',
    # P-25.2.1 Table 2.8 retains these complete fused parent families.
    'phenazine', 'phenanthridine', 'phenoxazine', 'phenothiazine', 'perimidine', 'quinolizine',
    'indolizine', 'pyrrolizine', 'thioxanthene',
    '1,7-phenanthroline', '1,8-phenanthroline', '1,9-phenanthroline',
    '1,10-phenanthroline', '2,7-phenanthroline', '2,8-phenanthroline',
    '2,9-phenanthroline', '3,7-phenanthroline', '3,8-phenanthroline',
    '4,7-phenanthroline',
})
_RETAINED_FUSED_ALIASES = {
    'chinolin': 'quinoline', 'quinolin': 'quinoline',
    'isochinolin': 'isoquinoline', 'isoquinolin': 'isoquinoline',
    'indol': 'indole', 'indazol': 'indazole', 'benzimidazol': 'benzimidazole',
    'cinnolin': 'cinnoline', 'phthalazin': 'phthalazine', 'purin': 'purine',
    'coumaron': 'benzofuran',
    # P-25.1.2.4 and P-25.2.1 Tables 2.8/2.9 give complete parent names.
    # These exact OPSIN parser aliases are not complete hydride spellings.
    'triphenylen': 'triphenylene',
    'perimidin': 'perimidine', 'arsanthridin': 'arsanthridine',
    # P-25.2.1 Table 2.9 and P-25.2.2.2/.3; normalize only these exact
    # complete-parent aliases, never a stem already joined to a suffix.
    'acridin': 'acridine', 'carbazol': 'carbazole', 'xanthen': 'xanthene',
    'phenanthridin': 'phenanthridine',
    'acridarsin': 'acridarsine', 'acridophosphin': 'acridophosphine',
    'arsindol': 'arsindole', 'phosphindol': 'phosphindole',
    'arsindolizin': 'arsindolizine', 'phosphindolizin': 'phosphindolizine',
    'isoarsindol': 'isoarsindole', 'isophosphindol': 'isophosphindole',
    'isoarsinolin': 'isoarsinoline', 'isophosphinolin': 'isophosphinoline',
    'phosphanthridin': 'phosphanthridine',
    'arsinolin': 'arsinoline', 'phosphinolin': 'phosphinoline',
    'arsinolizin': 'arsinolizine', 'phosphinolizin': 'phosphinolizine',
    'oxanthren': 'oxanthrene', 'thianthren': 'thianthrene',
    'selenanthren': 'selenanthrene', 'telluranthren': 'telluranthrene',
    'phosphanthren': 'phosphanthrene', 'arsanthren': 'arsanthrene',
    'boranthren': 'boranthrene', 'silanthren': 'silanthrene',
    'phenoselenazin': 'phenoselenazine', 'phenotellurazin': 'phenotellurazine',
    'phenoxaselenin': 'phenoxaselenine', 'phenoxatellurin': 'phenoxatellurine',
    'phenoxathiin': 'phenoxathiine', 'phenoxaphosphin': 'phenoxaphosphine',
    'phenoxarsin': 'phenoxarsine', 'phenothiarsin': 'phenothiarsine',
    'phenoxantimonin': 'phenoxantimonine',
    'phenarsazin': 'phenarsazine', 'phenophosphazin': 'phenophosphazine',
    'phenoxastibinin': 'phenoxastibinine',
    'selenoxanthen': 'selenoxanthene', 'telluroxanthen': 'telluroxanthene',
    'selenochromen': 'selenochromene', 'tellurochromen': 'tellurochromene',
    'isoselenochromen': 'isoselenochromene',
    'isotellurochromen': 'isotellurochromene',
    # P-25.1.1/Table 2.7: complete retained polycyclic hydrocarbon words.
    'acenaphthylen': 'acenaphthylene',
    'aceanthrylen': 'aceanthrylene', 'acephenanthrylen': 'acephenanthrylene',
    'cyclopenta[a]phenanthren': 'cyclopenta[a]phenanthrene',
}

# P-25.2.1, Table 2.8 assigns special retained numbering to these fused
# parent hydrides.  These records describe *parent* frames, not complete
# substituted molecules.  The numbered template is admitted only when its
# original atom labels, elements and catacondensed graph prove the frame.
# The ordered sequences are the source's perimeter labels; the second set
# is the pair of annular fusion edges, not external substituent bonds.
_SPECIAL_RETAINED_FUSED_FRAMES = {
    'phenanthrene': {
        'perimeter': ('1', '2', '3', '4', '4a', '4b', '5', '6',
                      '7', '8', '8a', '9', '10', '10a'),
        'fusion_edges': (('4a', '10a'), ('4b', '8a')),
        'hetero_locants': (), 'hetero_element': 6,
        'face_sizes': (6, 6, 6),
    },
    'fluorene': {
        'perimeter': ('1', '2', '3', '4', '4a', '4b', '5', '6',
                      '7', '8', '8a', '9', '9a'),
        'fusion_edges': (('4a', '9a'), ('4b', '8a')),
        'hetero_locants': (), 'hetero_element': 6,
        'face_sizes': (5, 6, 6),
    },
    'anthracene': {
        'perimeter': ('1', '2', '3', '4', '4a', '10', '10a',
                      '5', '6', '7', '8', '8a', '9', '9a'),
        'fusion_edges': (('4a', '9a'), ('10a', '8a')),
        'hetero_locants': (), 'hetero_element': 6,
        'face_sizes': (6, 6, 6),
    },
    'acridine': {
        'perimeter': ('1', '2', '3', '4', '4a', '10', '10a',
                      '5', '6', '7', '8', '8a', '9', '9a'),
        'fusion_edges': (('4a', '9a'), ('10a', '8a')),
        'hetero_locant': '10',
        'hetero_element': 7,
        'face_sizes': (6, 6, 6),
    },
    'carbazole': {
        'perimeter': ('1', '2', '3', '4', '4a', '4b',
                      '5', '6', '7', '8', '8a', '9', '9a'),
        'fusion_edges': (('4a', '9a'), ('4b', '8a')),
        'hetero_locant': '9',
        'hetero_element': 7,
        'face_sizes': (5, 6, 6),
    },
    'xanthene': {
        'perimeter': ('1', '2', '3', '4', '4a', '10', '10a',
                      '5', '6', '7', '8', '8a', '9', '9a'),
        'fusion_edges': (('4a', '9a'), ('10a', '8a')),
        'hetero_locant': '10',
        'hetero_element': 8,
        'face_sizes': (6, 6, 6),
    },
}

# The angular three-ring frame in Table 2.8 has a fixed numbering distinct
# from the generic peripheral numbering.  The phenanthroline isomers share
# that frame but have nitrogen at their individually specified positions.
_ANGULAR_FRAME = {
    'perimeter': ('1', '2', '3', '4', '4a', '5', '6', '6a',
                  '7', '8', '9', '10', '10a', '10b'),
    'fusion_edges': (('10a', '6a'), ('10b', '4a')),
    'hetero_element': 7,
    'face_sizes': (6, 6, 6),
}
_SPECIAL_RETAINED_FUSED_FRAMES['phenanthridine'] = {
    **_ANGULAR_FRAME, 'hetero_locants': ('5',),
}
for _parent in RETAINED_FUSED_HYDRIDES:
    if _parent.endswith('-phenanthroline'):
        _SPECIAL_RETAINED_FUSED_FRAMES[_parent] = {
            **_ANGULAR_FRAME, 'hetero_locants': tuple(_parent.split('-')[0].split(',')),
        }
_SPECIAL_RETAINED_FUSED_FRAMES['thioxanthene'] = {
    **_SPECIAL_RETAINED_FUSED_FRAMES['xanthene'], 'hetero_element': 16,
}
for _parent, _chalcogen in (('phenoxazine', 8), ('phenothiazine', 16)):
    _SPECIAL_RETAINED_FUSED_FRAMES[_parent] = {
        'perimeter': ('1', '2', '3', '4', '4a', '5', '5a',
                      '6', '7', '8', '9', '9a', '10', '10a'),
        'fusion_edges': (('4a', '10a'), ('5a', '9a')),
        'hetero_elements': {'5': _chalcogen, '10': 7},
        'face_sizes': (6, 6, 6),
    }


def has_special_retained_fused_frame(name):
    return name in _SPECIAL_RETAINED_FUSED_FRAMES


def special_retained_fused_locants(name, query, locants):
    """Validate, then preserve, a sourced retained parent's special numbering.

    In particular acridine's peripheral numeric locants do not run 1..10
    around the boundary. Reconstructing its labels from that assumption would
    change the fixed parent numbering. Ring faces and each labelled perimeter
    and fusion edge are checked on the parent graph before using the source
    labels; every placement on an input molecule is still matched separately.
    """
    specification = _SPECIAL_RETAINED_FUSED_FRAMES.get(name)
    if specification is None:
        return None
    from .catacondensed_numbering import peripheral_numbering_frontier

    labels = tuple(specification['perimeter'])
    if (query.GetNumAtoms() != len(labels)
            or set(locants) != set(range(query.GetNumAtoms()))
            or set(locants.values()) != set(labels)):
        return None
    at = {locants[i]: i for i in locants}
    for label, idx in at.items():
        atom = query.GetAtomWithIdx(idx)
        hetero_locants = specification.get('hetero_locants', (specification.get('hetero_locant'),))
        expected = (specification['hetero_elements'].get(label, 6)
                    if 'hetero_elements' in specification else
                    specification['hetero_element'] if label in hetero_locants else 6)
        if atom.GetAtomicNum() != expected:
            return None
    frontier = peripheral_numbering_frontier(query)
    if (not frontier.admitted
            or tuple(sorted(map(len, frontier.ring_faces))) != specification['face_sizes']):
        return None
    perimeter = frozenset(frozenset((at[a], at[b]))
                          for a, b in zip(labels, labels[1:] + labels[:1]))
    fusion = frozenset(frozenset((at[a], at[b]))
                       for a, b in specification['fusion_edges'])
    if perimeter != frontier.perimeter_edges or fusion != frontier.fusion_edges:
        return None
    return dict(locants)


def special_retained_fused_perimeter(name, edges, size):
    """Prove the all-atom perimeter of an already matched retained frame.

    Build an annotation-free topology solely for the perimeter proof. The
    original input's bond orders, hydrogen state, stereo and substituent atom
    ownership are handled later by the original-graph renderer.
    """
    specification = _SPECIAL_RETAINED_FUSED_FRAMES.get(name)
    if specification is None or size != len(specification['perimeter']):
        return None
    from rdkit import Chem
    from .catacondensed_numbering import peripheral_numbering_frontier

    graph = Chem.RWMol()
    for _ in range(size):
        graph.AddAtom(Chem.Atom(6))
    for left, right in edges:
        graph.AddBond(left, right, Chem.BondType.SINGLE)
    topology = graph.GetMol()
    try:
        Chem.SanitizeMol(topology)
    except (ValueError, RuntimeError):
        return None
    frontier = peripheral_numbering_frontier(topology)
    if (not frontier.admitted
            or tuple(sorted(map(len, frontier.ring_faces))) != specification['face_sizes']):
        return None
    return frozenset(frontier.perimeter_cycle)

# P-25.2.2.4: the position of the heteroatom is explicit in preferred
# benzoheterocycle names.  The resource's unlocanted aliases carry the same
# atom numbering, so only the hydride vocabulary changes here.
_PREFERRED_BENZO_HETEROCYCLE_HYDRIDES = {
    'benzofuran': '1-benzofuran',
    'isobenzofuran': '2-benzofuran',
    'benzothiophene': '1-benzothiophene',
}


def retained_fused_vocabulary(name):
    """Normalize only reviewed whole hydride aliases, leaving maps intact."""
    return normalize_fusion_component_spelling(_RETAINED_FUSED_ALIASES.get(name, name))


def _proved_quinoline_alias(name, template):
    """Check the resource's numbered parent graph before normalizing an alias.

    P-25.2.1 retains quinoline and isoquinoline. Some OPSIN resources use the
    German e-ending spellings, so ensure that the whole template has exactly
    the corresponding elements and labelled bonds before changing vocabulary.
    This proof does not inspect or modify substituents on the original graph.
    """
    if name not in {'chinoline', 'isochinoline'}:
        return name
    preferred = {'chinoline': 'quinoline', 'isochinoline': 'isoquinoline'}[name]
    from .fusion_components import resolve_component

    component = resolve_component(preferred)
    query = template.get('mol')
    locants = template.get('locants') or {}
    if (component is None or query is None
            or set(locants) != set(range(query.GetNumAtoms()))
            or len(set(locants.values())) != query.GetNumAtoms()):
        return name
    labels = {str(locants[i]): query.GetAtomWithIdx(i).GetAtomicNum() for i in locants}
    bonds = frozenset(frozenset((str(locants[bond.GetBeginAtomIdx()]),
                                str(locants[bond.GetEndAtomIdx()])))
                      for bond in query.GetBonds())
    if labels != dict(component.label_elements) or bonds != component.bonds:
        return name
    return preferred


# Keep detachable groups separate from their hydride so that all intrinsic
# and external prefixes participate in the same ordering and multiplication.
_SUBSTITUENT_FUNCTIONAL_COMPONENTS = {
    "chromone": ("4H-chromene", (("4", "oxo", "C"),)),
    "chroman-4-one": ("chromane", (("4", "oxo", "C"),)),
    "coumarin": ("2H-1-benzopyran", (("2", "oxo", "C"),)),
    "isoindolin-1-one": ("isoindoline", (("1", "oxo", "C"),)),
}


def preferred_fused_substituent_prefix_records(template):
    return _SUBSTITUENT_FUNCTIONAL_COMPONENTS.get(str(template["name"]), ("", ()))[1]


def _mancude_nh_parent(mol, template, match, name, locants):
    """Make the actual aromatic NH state explicit without guessing tautomers."""
    import re

    # Functional parents such as guanine already encode their amino/oxo and
    # H-state conventions. This rule is for mancude ring hydrides themselves.
    if any(not atom.IsInRing() for atom in template["mol"].GetAtoms()):
        return name, locants
    nh_atoms = mol_cache_get(mol, "preferred_fused_aromatic_nh_atoms")
    if nh_atoms is None:
        nh_atoms = mol_cache_set(mol, "preferred_fused_aromatic_nh_atoms", frozenset(
            a.GetIdx() for a in iter_atoms(mol)
            if a.GetAtomicNum() == 7 and a.GetIsAromatic() and a.GetTotalNumHs() > 0
        ))
    hydrogen = [locants.get(j, "") for j, i in enumerate(match) if i in nh_atoms]
    if not hydrogen or not all(loc.isdigit() for loc in hydrogen):
        return name, locants
    if re.search(r"\d+[a-z]?H", name) or not all(
        mol.GetAtomWithIdx(i).GetIsAromatic()
        and not mol.GetAtomWithIdx(i).GetFormalCharge()
        and not mol.GetAtomWithIdx(i).GetIsotope()
        for i in match
        if mol.GetAtomWithIdx(i).IsInRing()
    ):
        return name, locants
    core = set(match)
    if any(b.GetBondTypeAsDouble() == 2 and b.GetOtherAtomIdx(i) not in core
           for i in match for b in mol.GetAtomWithIdx(i).GetBonds()):
        return name, locants
    positions = ",".join(f"{loc}H" for loc in sorted(hydrogen, key=int))
    return f"{positions}-{name}", locants


def _indazole_substituent_hydride(mol, match, locants):
    """Find the original aromatic indazole H-state after capping substitutions.

    P-25.2.1/Table 2.8 and P-25.7.1.3.1 use indicated hydrogen for the
    retained indazole parent. In particular, a single external bond replacing
    the pyrrolic N-H does not remove the indicated hydrogen from that parent.
    Check the fixed labelled ring topology and original N valences before using
    that operation; do not infer a tautomer from a name or alter the graph.
    """
    labels = ("1", "2", "3", "3a", "4", "5", "6", "7", "7a")
    if (len(match) != len(labels) or len(set(match)) != len(labels)
            or set(locants) != set(range(len(labels)))
            or set(locants.values()) != set(labels)):
        return ""
    at = {locants[j]: match[j] for j in range(len(labels))}
    core = set(match)
    edges = {
        frozenset(edge) for edge in (
            ("1", "2"), ("2", "3"), ("3", "3a"), ("3a", "7a"),
            ("7a", "1"), ("3a", "4"), ("4", "5"), ("5", "6"),
            ("6", "7"), ("7", "7a"),
        )
    }
    actual = set()
    for label, idx in at.items():
        atom = mol.GetAtomWithIdx(idx)
        if (atom.GetAtomicNum() != (7 if label in {"1", "2"} else 6)
                or not atom.GetIsAromatic() or atom.GetFormalCharge()
                or atom.GetIsotope() or atom.GetNumRadicalElectrons()):
            return ""
        for bond in atom.GetBonds():
            other = bond.GetOtherAtomIdx(idx)
            if other in core:
                if not bond.GetIsAromatic():
                    return ""
                actual.add(frozenset((label, locants[match.index(other)])))
            elif (bond.GetBondTypeAsDouble() != 1
                  or mol.GetAtomWithIdx(other).GetAtomicNum() == 1
                  and mol.GetAtomWithIdx(other).GetIsotope()):
                return ""
    if actual != edges:
        return ""
    budgets = []
    for label in ("1", "2"):
        atom = mol.GetAtomWithIdx(at[label])
        outside = sum(n.GetIdx() not in core and n.GetAtomicNum() != 1
                      for n in atom.GetNeighbors())
        budgets.append(atom.GetTotalNumHs(includeNeighbors=True) + outside)
    if budgets not in ([1, 0], [0, 1]):
        return ""
    return f"{budgets.index(1) + 1}H-indazole"


@lru_cache(maxsize=256)
def _template_symmetries(query):
    return query.GetSubstructMatches(query, uniquify=False, useChirality=True, maxMatches=128)


def preferred_quinone_numbering(mol, template, match, prefix_records):
    """Choose an equivalent fixed-parent orientation after the suffix ties."""
    from .formatting import prefix_alphabetization_key

    locants = template["locants"]
    if any(int(mol.GetAtomWithIdx(i).GetChiralTag()) for i in match):
        return locants
    query = template["mol"]
    oxo = [a.GetIdx() for a in query.GetAtoms() if a.GetAtomicNum() == 6
           and any(n.GetAtomicNum() == 8 and query.GetBondBetweenAtoms(
               a.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 2 for n in a.GetNeighbors())]
    if not oxo or not all(locants.get(i, "").isdigit() for i in oxo):
        return locants
    suffix_locs = sorted(int(locants[i]) for i in oxo)
    best, best_key = locants, None
    ordered = sorted(prefix_records, key=lambda item: prefix_alphabetization_key(item[1]))
    for symmetry in _template_symmetries(query):
        mapped = {i: locants[symmetry[i]] for i in locants if symmetry[i] in locants}
        if len(mapped) != len(locants) or sorted(int(mapped[i]) for i in oxo) != suffix_locs:
            continue
        translation = {locants[i]: mapped[i] for i in locants}
        if not all(translation.get(loc, "").isdigit() for loc, *_ in prefix_records):
            continue
        key = (tuple(sorted(int(translation[loc]) for loc, *_ in prefix_records)),
               tuple(int(translation[loc]) for loc, *_ in ordered))
        if best_key is None or key < best_key:
            best, best_key = mapped, key
    return best


@lru_cache(maxsize=1)
def _retained_aromatic_frames():
    from rdkit import Chem

    return (
        ('quinoline', Chem.MolFromSmiles('c1ccc2ncccc2c1'),
         ('6', '7', '8', '8a', '1', '2', '3', '4', '4a', '5')),
        ('1H-indazole', Chem.MolFromSmiles('c1ccc2[nH]ncc2c1'),
         ('5', '6', '7', '7a', '1', '2', '3', '3a', '4')),
    )


@lru_cache(maxsize=1)
def _phthalide_frame():
    from rdkit import Chem

    return Chem.MolFromSmiles('O=C1OCc2ccccc21')


def _proved_phthalide_parent_locants(mol, match):
    query = _phthalide_frame()
    core = frozenset(match)
    hits = [row for row in mol.GetSubstructMatches(query, uniquify=False)
            if frozenset(row) == core]
    if len(set(hits)) != 1:
        return None
    # Parent ring numbering starts on the carbonyl rather than its oxygen.
    labels = ('', '1', '2', '3', '3a', '4', '5', '6', '7', '7a')
    source_labels = dict(zip(hits[0], labels))
    return {i: source_labels[idx] for i, idx in enumerate(match) if source_labels[idx]}


def exact_retained_aromatic_frame(mol, core):
    """P-25.2.1 retained names with their proved original atom numbering."""
    from .molecule_cache import mol_cache_dict

    core = frozenset(core)
    if (len(core) not in {9, 10}
            or any(not mol.GetAtomWithIdx(i).GetIsAromatic()
                   or mol.GetAtomWithIdx(i).GetFormalCharge()
                   or mol.GetAtomWithIdx(i).GetIsotope() for i in core)):
        return None
    cache = mol_cache_dict(mol, 'exact_retained_aromatic_frame')
    if core in cache:
        return cache[core]
    for name, query, labels in _retained_aromatic_frames():
        if query.GetNumAtoms() != len(core):
            continue
        accepted = [match for match in mol.GetSubstructMatches(query, uniquify=False)
                    if frozenset(match) == core]
        if len(set(accepted)) == 1:
            locants = dict(enumerate(labels))
            if name == '1H-indazole' and _indazole_substituent_hydride(mol, accepted[0], locants) != name:
                continue
            template = {'name': name, 'mol': query, 'locants': locants,
                        'preserve_indicated_hydrogen': name == '1H-indazole'}
            result = (template, accepted[0])
            cache[core] = result
            return result
    cache[core] = None
    return None


def _configured_flavanone_frame(mol, template, match):
    """Keep the published flavan-4-one frame and its phenyl automorphism."""
    from .principal_carbonyl import _ring_carbonyl_comparison_sites, configured_methine_ring_junction
    groups = _ring_carbonyl_comparison_sites(mol)
    locants, query = template['locants'], template['mol']
    if (len(groups) != 1 or groups[0].kind != 'ketone'
            or groups[0].center not in match
            or not configured_methine_ring_junction(mol, match)
            or any(mol.GetAtomWithIdx(i).GetIsotope() or mol.GetAtomWithIdx(i).GetChiralTag()
                   or mol.GetAtomWithIdx(i).GetFormalCharge() for i in match)):
        return None
    expected = ('1', '2', '3', '4', '', '4a', '5', '6', '7', '8', '8a',
                "1'", "2'", "3'", "4'", "5'", "6'")
    if (len(match) != 17 or set(locants.values()) != set(expected)
            or locants.get(match.index(groups[0].center)) != '4'):
        return None
    from rdkit import Chem
    reference = Chem.MolFromSmiles('O1C(CC(=O)c2ccccc21)c1ccccc1')
    def numbered_edges(graph, labels):
        return frozenset((frozenset((labels[b.GetBeginAtomIdx()], labels[b.GetEndAtomIdx()])),
                          b.GetBondTypeAsDouble()) for b in graph.GetBonds())
    if ({locants[a.GetIdx()]: a.GetAtomicNum() for a in query.GetAtoms()}
            != {expected[a.GetIdx()]: a.GetAtomicNum() for a in reference.GetAtoms()}
            or numbered_edges(query, locants) != numbered_edges(reference, expected)):
        return None
    # Only the pendant phenyl orientation is free. The numbered heterocycle
    # and carbonyl remain fixed. Multiple phenyl substituents need the normal
    # complete prefix comparison, so retain their original general frame.
    core = set(match)
    hosts = [j for j, idx in enumerate(match) if str(locants[j]).endswith("'")
             and any(n.GetIdx() not in core for n in mol.GetAtomWithIdx(idx).GetNeighbors())]
    if len(hosts) != 1:
        return dict(locants)
    options = []
    for symmetry in _template_symmetries(query):
        mapped = {j: locants[symmetry[j]] for j in locants}
        if all(mapped[j] == locants[j] for j in locants if not str(locants[j]).endswith("'")):
            options.append((int(mapped[hosts[0]].rstrip("'")), mapped))
    return min(options, key=lambda row: row[0])[1] if options else None


def preferred_fused_name_locants(
    mol, template, match, *, substituent=False, attachment_idx=None, parent_idx=None
):
    name = _proved_quinoline_alias(str(template["name"]), template)
    locants = template["locants"]
    if not substituent and name in {'flavanon', 'flavanone'}:
        corrected = _configured_flavanone_frame(mol, template, match)
        if corrected is not None:
            return 'flavan-4-one', corrected
    if not substituent and name in {'phthalid', 'phthalide'}:
        from .principal_carbonyl import configured_methine_ring_junction

        if configured_methine_ring_junction(mol, match):
            mapped = _proved_phthalide_parent_locants(mol, match)
            if mapped is not None:
                return '2-benzofuran-1(3H)-one', mapped
    if len(match) == 10 and name != 'quinoline':
        from .principal_carbonyl import configured_methine_ring_junction

        if configured_methine_ring_junction(mol, frozenset().union(*mol.GetRingInfo().AtomRings())):
            retained = exact_retained_aromatic_frame(mol, match)
            if retained is not None:
                retained_template, retained_match = retained
                preferred_locants = {idx: retained_template['locants'][pos]
                                     for pos, idx in enumerate(retained_match)}
                return retained_template['name'], {i: preferred_locants[idx] for i, idx in enumerate(match)}

    if substituent and name in {"indazole", "indazol"}:
        hydride = _indazole_substituent_hydride(mol, match, locants)
        if hydride:
            return hydride, locants
    if substituent and name in {"1H-benzotriazole", "2H-benzotriazole"}:
        if _benzotriazole_numbering_is_proved(mol, match, locants):
            return name.replace("benzotriazole", "1,2,3-benzotriazole"), locants
    if substituent and name in _PREFERRED_BENZO_HETEROCYCLE_HYDRIDES:
        return _PREFERRED_BENZO_HETEROCYCLE_HYDRIDES[name], locants
    if substituent and name in {"fluoren", "fluorene"}:
        # P-25.1.1/Table 2.7: the retained preferred hydride is
        # 9H-fluorene. The indicated H identifies the hydride's saturated
        # position even if a subsequent substitution replaces one of its Hs.
        ninth = [match[i] for i, loc in locants.items() if loc == "9"]
        if len(ninth) == 1:
            atom = mol.GetAtomWithIdx(ninth[0])
            core = set(match)
            if (atom.GetAtomicNum() == 6 and not atom.GetFormalCharge()
                    and not atom.GetIsotope() and not atom.GetIsAromatic()
                    and not any(n.GetAtomicNum() == 1 and n.GetIsotope()
                                for n in atom.GetNeighbors())
                    and all(b.GetBondTypeAsDouble() == 1 for b in atom.GetBonds())
                    and sum(n.GetIdx() in core for n in atom.GetNeighbors()) == 2):
                return "9H-fluorene", locants
    if substituent and name in {"tetralin", "tetraline"}:
        return _preferred_tetrahydronaphthalenyl(template, match, attachment_idx)
    if substituent and name in _SUBSTITUENT_FUNCTIONAL_COMPONENTS:
        return _SUBSTITUENT_FUNCTIONAL_COMPONENTS[name][0], locants
    if (not substituent and name in _SUBSTITUENT_FUNCTIONAL_COMPONENTS
            and all(not mol.GetAtomWithIdx(i).GetFormalCharge()
                    and not mol.GetAtomWithIdx(i).GetIsotope() for i in match)):
        hydride, prefixes = _SUBSTITUENT_FUNCTIONAL_COMPONENTS[name]
        if len(prefixes) == 1 and prefixes[0][1] == "oxo":
            return f"{hydride.removesuffix('e')}-{prefixes[0][0]}-one", locants
    if substituent and template.get("source_file") == "naturalProducts.xml":
        from .parent_rules import _NATURAL_PRODUCT_TOPOLOGY_STEMS
        if name in _NATURAL_PRODUCT_TOPOLOGY_STEMS:
            return name + "an", locants
    if not substituent and "quinon" in name:
        from .principal_carbonyl import principal_carbonyl_sites
        from .formatting import multiplied_suffix

        hydride = next((parent for stem, parent in (
            ("acenaphtho", "acenaphthylene"), ("naphtho", "naphthalene"),
            ("anthra", "anthracene"),
        ) if name.startswith(stem)), "")
        if hydride and principal_carbonyl_sites(mol)[0] == "one":
            query = template["mol"]
            oxo = [str(locants.get(a.GetIdx(), "")) for a in query.GetAtoms()
                   if a.GetAtomicNum() == 6 and any(n.GetAtomicNum() == 8
                   and query.GetBondBetweenAtoms(a.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 2
                   for n in a.GetNeighbors())]
            if 2 <= len(oxo) <= 4 and all(loc.isdigit() for loc in oxo):
                positions = ",".join(sorted(oxo, key=int))
                return f"{hydride}-{positions}-{multiplied_suffix(len(oxo), 'one')}", locants
    if name not in {
        "indol",
        "indole",
        "1H-indole",
        "indolizin",
        "indolizine",
        "benzimidazole",
        "1H-benzimidazole",
        "oxindol",
        "oxindole",
        "purin",
        "purine",
    }:
        return _mancude_nh_parent(mol, template, match, name, locants) if substituent else (name, locants)
    if any(
        mol.GetAtomWithIdx(i).GetFormalCharge() or mol.GetAtomWithIdx(i).GetIsotope()
        for i in match
    ):
        return name, locants
    if substituent and name in {"purin", "purine"}:
        bond = mol.GetBondBetweenAtoms(attachment_idx, parent_idx) if attachment_idx is not None and parent_idx is not None else None
        core = set(match)
        reverse = {i: j for j, i in enumerate(match)}
        free_loc = locants.get(reverse.get(attachment_idx))
        if bond is not None and bond.GetBondTypeAsDouble() == 2 and free_loc and mol.GetAtomWithIdx(attachment_idx).GetAtomicNum() == 6:
            h_locs = [locants.get(j) for j, i in enumerate(match) if mol.GetAtomWithIdx(i).GetAtomicNum() == 7 and mol.GetAtomWithIdx(i).GetTotalNumHs() > 0]
            substituted_n = [locants.get(j) for j, i in enumerate(match) if mol.GetAtomWithIdx(i).GetAtomicNum() == 7 and any(n.GetIdx() not in core and n.GetAtomicNum() > 1 for n in mol.GetAtomWithIdx(i).GetNeighbors())]
            hydro_locs = set(h_locs + substituted_n)
            if len(h_locs) == len(substituted_n) == 1 and len(hydro_locs) == 2 and all(x and x.isdigit() for x in hydro_locs):
                hydro = ",".join(sorted(hydro_locs, key=int))
                return f"{hydro}-dihydro-{free_loc}H-purin", locants
        return name, locants
    if name in {"indol", "indole", "1H-indole"}:
        return "1H-indole", locants
    if name in {"indolizin", "indolizine"}:
        return "indolizine", locants
    if substituent and name in {"oxindol", "oxindole"}:
        # The free valence is introduced before ordinary substituent prefixes.
        # A 3-ylidene group uses 3H to accommodate that operation; benzene-ring
        # attachments retain the lower indicated-hydrogen locant of indole.
        reverse = {atom_idx: j for j, atom_idx in enumerate(match)}
        attachment = locants.get(reverse.get(attachment_idx))
        bond = (
            mol.GetBondBetweenAtoms(attachment_idx, parent_idx)
            if attachment_idx is not None and parent_idx is not None
            else None
        )
        if (
            attachment in {"1", "4", "5", "6", "7"}
            and bond is not None
            and bond.GetBondTypeAsDouble() == 1
        ):
            return "2-oxo-2,3-dihydro-1H-indol", locants
        return "2-oxo-1,2-dihydro-3H-indol", locants
    if not substituent and name in {"oxindol", "oxindole"}:
        from .principal_carbonyl import principal_carbonyl_sites

        suffix, sites = principal_carbonyl_sites(mol)
        if suffix == "one" and len(sites) == 1:
            return "1,3-dihydro-2H-indol-2-one", locants
    if name not in {"benzimidazole", "1H-benzimidazole"}:
        return name, locants
    core = set(match)
    if any(
        b.GetBondTypeAsDouble() == 2 and b.GetOtherAtomIdx(i) not in core
        for i in match
        for b in mol.GetAtomWithIdx(i).GetBonds()
    ):
        return name, locants
    # The indicated H belongs to the hydride before substitution. An ordinary
    # external single bond on a neutral pyrrolic N replaces that H; it must
    # retain the same numbering as an actual N-H. This conservation is needed
    # for every suffix class, not only the senior-acid route.
    hydrogen = []
    for j, i in enumerate(match):
        atom = mol.GetAtomWithIdx(i)
        if atom.GetAtomicNum() != 7:
            continue
        if not atom.GetIsAromatic():
            return name, locants
        external = [b for b in atom.GetBonds() if b.GetOtherAtomIdx(i) not in core
                    and b.GetOtherAtom(atom).GetAtomicNum() > 1]
        if any(b.GetBondTypeAsDouble() != 1 for b in external):
            return name, locants
        budget = atom.GetTotalNumHs() + len(external)
        if budget not in {0, 1}:
            return name, locants
        if budget:
            hydrogen.append(locants.get(j))
    if hydrogen not in [["1"], ["3"]]:
        return name, locants
    if hydrogen == ["3"]:
        mirror = {
            "1": "3",
            "3": "1",
            "3a": "7a",
            "7a": "3a",
            "4": "7",
            "7": "4",
            "5": "6",
            "6": "5",
        }
        locants = {i: mirror.get(loc, loc) for i, loc in locants.items()}
    return "1H-1,3-benzimidazole", locants


def _benzotriazole_numbering_is_proved(mol, match, locants):
    """P-25.2.2.4 heteroatom locants on a proved benzotriazole core."""
    labels = ("1", "2", "3", "3a", "4", "5", "6", "7", "7a")
    if (len(match) != len(labels) or len(set(match)) != len(labels)
            or set(locants) != set(range(len(labels)))
            or set(locants.values()) != set(labels)):
        return False
    at = {locants[i]: match[i] for i in range(len(labels))}
    for label, idx in at.items():
        atom = mol.GetAtomWithIdx(idx)
        if (atom.GetAtomicNum() != (7 if label in {"1", "2", "3"} else 6)
                or not atom.GetIsAromatic() or atom.GetFormalCharge()
                or atom.GetIsotope()
                or any(n.GetAtomicNum() == 1 and n.GetIsotope()
                       for n in atom.GetNeighbors())):
            return False
    edges = (("1", "2"), ("2", "3"), ("3", "3a"), ("3a", "7a"),
             ("7a", "1"), ("3a", "4"), ("4", "5"), ("5", "6"),
             ("6", "7"), ("7", "7a"))
    core = set(match)
    return (all(mol.GetBondBetweenAtoms(at[a], at[b]) is not None for a, b in edges)
            and sum(1 for idx in match for nbr in mol.GetAtomWithIdx(idx).GetNeighbors()
                    if nbr.GetIdx() in core) == 2 * len(edges))


def hydrogenation_parent_templates(templates):
    """Numbered mancude hydrides, excluding functional/stereo/partial aliases."""
    import re
    selected = []
    for template in templates:
        name = str(template["name"])
        query = template["mol"]
        atoms = tuple(iter_atoms(query))
        if (not name.endswith("e") or re.search(r"hydro|cyclo\[|spiro\[|\dH-", name)
                or template.get("stereochemical")
                or query.GetRingInfo().NumRings() < 2
                or any(not a.IsInRing() or a.GetAtomicNum() != 6
                       or a.GetFormalCharge() or a.GetIsotope() for a in atoms)):
            continue
        # A mancude ring hydride has at most one saturated indicated-H atom.
        saturated = [a for a in atoms if not a.GetIsAromatic()
                     and all(b.GetBondTypeAsDouble() == 1 for b in a.GetBonds())]
        if len(saturated) > 1:
            continue
        if not all(template["locants"].get(a.GetIdx()) for a in atoms):
            continue
        # Numeric positions fix the traversal; imported junction letters may
        # have been assigned in its opposite direction. Hydro operations and
        # saturated bridgehead substituents must use the same proved map.
        from .ring_state_parent import _peripheral_junction_locants
        mapped = _peripheral_junction_locants(query, template['locants'])
        selected.append({**template, 'locants': mapped} if mapped is not None else template)
    return tuple(selected)


def hydrogenated_fused_parent_name(mol, template, match):
    """Derive the full hydro state from atom valences in a numbered hydride.

    External single/double bonds are capped with the corresponding number of
    H atoms before comparing with the reference hydride. Thus suffix carbonyls
    and substituents do not hide hydrogenation at bridgeheads. Residual aromatic
    rings must be preserved; shifted nonaromatic double bonds are unsupported.
    No combinatorial locant guesses or implied stereochemistry are introduced.
    """
    import re
    from .constants import MULT

    query = template["mol"]
    core = set(match)
    if any(mol.GetAtomWithIdx(i).GetFormalCharge()
           or mol.GetAtomWithIdx(i).GetNumRadicalElectrons() for i in match):
        return ""
    hydro = []
    for j, i in enumerate(match):
        atom = mol.GetAtomWithIdx(i)
        ref = query.GetAtomWithIdx(j)
        external = sum(b.GetBondTypeAsDouble() for b in atom.GetBonds()
                       if b.GetOtherAtomIdx(i) not in core)
        delta = atom.GetTotalNumHs() + external - ref.GetTotalNumHs()
        if delta not in {0, 1, 2}:
            return ""
        hydro.extend([template["locants"][j]] * int(delta))
    if not hydro or len(hydro) % 2 or len(hydro) not in MULT:
        return ""
    for ref in query.GetBonds():
        bond = mol.GetBondBetweenAtoms(match[ref.GetBeginAtomIdx()], match[ref.GetEndAtomIdx()])
        if bond is None:
            return ""
        if bond.GetIsAromatic():
            if not ref.GetIsAromatic():
                return ""
        elif bond.GetBondTypeAsDouble() > 1:
            if ref.GetIsAromatic() or ref.GetBondTypeAsDouble() != bond.GetBondTypeAsDouble():
                return ""
    def key(loc):
        return int(re.match(r"\d+", loc)[0]), loc
    positions = ",".join(sorted(hydro, key=key)) + "-"
    if all(not mol.GetBondBetweenAtoms(match[b.GetBeginAtomIdx()], match[b.GetEndAtomIdx()]).GetIsAromatic()
           and mol.GetBondBetweenAtoms(match[b.GetBeginAtomIdx()], match[b.GetEndAtomIdx()]).GetBondTypeAsDouble() == 1
           for b in query.GetBonds()):
        positions = ""
    parent = template["name"]
    indicated = [template["locants"][a.GetIdx()] for a in query.GetAtoms()
                 if not a.GetIsAromatic() and a.GetTotalNumHs() == 2
                 and all(b.GetBondTypeAsDouble() == 1 for b in a.GetBonds())]
    if indicated:
        parent = ",".join(f"{loc}H" for loc in sorted(indicated, key=key)) + "-" + parent
    # Preserve existing internal hydride tokens used by suffix recognition.
    # The newly mapped numeric benzofuran parent needs its own separator.
    join = "-" if indicated or parent in {'1-benzofuran', '2-benzofuran'} else ""
    return f"{positions}{MULT[len(hydro)]}hydro{join}{parent}"


def _preferred_tetrahydronaphthalenyl(template, match, attachment_idx):
    """Keep tetralin atom identities while numbering its free valence first."""
    locants = template["locants"]
    reverse = {i: j for j, i in enumerate(match)}
    attachment = locants.get(reverse.get(attachment_idx), "")
    if not attachment.isdigit():
        return template["name"], locants
    mirror = {str(i): str(5-i if i <= 4 else 13-i) for i in range(1, 9)}
    mirror.update({"4a": "8a", "8a": "4a"})
    turn = {str(i): str((i + 3) % 8 + 1) for i in range(1, 9)}
    turn.update({"4a": "8a", "8a": "4a"})
    identity = {loc: loc for loc in locants.values()}
    mappings = [identity, mirror, turn, {loc: mirror[turn[loc]] for loc in identity}]
    def key(mapping):
        hydro = sorted(int(mapping[str(i)]) for i in range(1, 5))
        return int(mapping[attachment]), hydro
    mapping = min(mappings, key=key)
    hydro = ",".join(map(str, key(mapping)[1]))
    return f"{hydro}-tetrahydronaphthalen", {i: mapping[loc] for i, loc in locants.items()}
