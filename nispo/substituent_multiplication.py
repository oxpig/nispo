"""Source-typed multiplication of bounded substituent components.

The caller supplies the component's simple/compound status from its mapped
construction. Simple forms use a narrow proof grammar; caller-proved compound
units receive safe-syntax validation and source-based enclosure.
"""

from dataclasses import dataclass
from functools import lru_cache
import re

from .constants import MULT, ROOTS, ROOT_LENGTHS
from .formatting import COMPLEX_MULT, enclose_name_unit


RULE = "P-16.3.2/P-16.3.3/P-16.3.4/P-16.3.5/P-16.5/P-82.2.1"


def terminally_substituted_carbon_component(mol, atoms):
    """Prove substitution on one connected carbon organyl framework.

    Terminal heteroatom pendants (halo, hydroxy, amino, oxo, etc.) are
    substituents on the carbon hydride, not skeletal atoms of a simple
    heterocyclic parent. This is sufficient P-16.3.2(c) compound evidence;
    false leaves other component families to their existing producers.
    """
    atoms = frozenset(atoms)
    carbons = {i for i in atoms if mol.GetAtomWithIdx(i).GetAtomicNum() == 6}
    pendants = {i for i in atoms if mol.GetAtomWithIdx(i).GetAtomicNum() not in {1, 6}}
    if not carbons or not pendants:
        return False
    connected, stack = set(), [next(iter(carbons))]
    while stack:
        i = stack.pop()
        if i in connected:
            continue
        connected.add(i)
        stack.extend(a.GetIdx() for a in mol.GetAtomWithIdx(i).GetNeighbors()
                     if a.GetIdx() in carbons and a.GetIdx() not in connected)
    if connected != carbons:
        return False
    return all(not mol.GetAtomWithIdx(i).IsInRing()
               and len([a for a in mol.GetAtomWithIdx(i).GetNeighbors()
                        if a.GetAtomicNum() > 1]) == 1
               and any(a.GetIdx() in carbons for a in mol.GetAtomWithIdx(i).GetNeighbors())
               for i in pendants)
_ROOT_NAMES = frozenset(ROOTS.values())
_MULTIPLICATIVE_HEADS = frozenset(
    word for count, word in MULT.items() if count > 1
)
# P-16.3.6(a): a direct numerical prefix on these mononuclear units would
# instead denote one polynuclear chain, e.g. disulfanyl versus bis(sulfanyl).
_CHAIN_AMBIGUOUS_PREFIXES = frozenset({
    "azanyl", "oxidanyl", "sulfanyl", "selanyl", "tellanyl",
    "phosphanyl", "arsanyl", "stibanyl", "bismuthanyl",
})
_RETAINED_SIMPLE_OXY_PREFIXES = frozenset({"methoxy", "ethoxy", "propoxy", "butoxy", "phenoxy", "tert-butoxy"})
_UNSATURATION_ENDINGS = {
    1: {"ene": "en", "yne": "yn"},
    2: {"ene": "dien", "yne": "diyn"},
    3: {"ene": "trien", "yne": "triyn"},
    4: {"ene": "tetraen", "yne": "tetrayn"},
    5: {"ene": "pentaen", "yne": "pentayn"},
    6: {"ene": "hexaen", "yne": "hexayn"},
}


@dataclass(frozen=True, slots=True)
class SubstituentMultiplication:
    text: str
    component: str
    count: int
    nature: str
    multiplier: str
    enclosed: bool
    rule: str = RULE


def _alkyl_form(component):
    """Prove a plain acyclic alkyl prefix from the configured parent roots."""
    # Parsing one stem avoids compiling one expression per possible root on
    # every miss. The larger vocabulary otherwise thrashes re's cache, even
    # for ordinary components. Keep the exact original spelling/locant gate.
    match = re.fullmatch(r'([a-z]+)(?:an-([0-9]+)-)?yl', component)
    if match is None or match[1] not in _ROOT_NAMES:
        return None
    root, locant = match[1], match[2]
    if locant is None:
        return root, False
    carbon_count = ROOT_LENGTHS.get(root)
    if (carbon_count is not None and ROOTS.get(carbon_count) == root
            and 1 <= int(locant) <= carbon_count):
        return root, True
    return None


def _unsaturated_alkyl_form(component):
    """Validate a simple monovalent unsaturated acyclic prefix.

    This is a grammar check for one substituent attachment, not a chemical
    name lookup. The caller remains responsible for proving that the mapped
    source component is an unsubstituted carbon chain with one free valence.
    """
    if component in {"ethenyl", "ethynyl"}:
        return (2, (), (), 1, False)

    # P-32.1 method (1): for ordinary single ene/yne endings, all free
    # valence and unsaturation locants are explicit, including locant 1.
    for carbon_count, root in ROOTS.items():
        if carbon_count < 3:
            continue
        root = str(root)
        tail = component[len(root):] if component.startswith(root) else None
        if tail is None:
            continue
        # one or more locants in a single unsaturation type; the `a` is
        # inserted before locants only for multiplicative ene/yne endings.
        match = re.fullmatch(
            r"(?P<a>a)?-(?P<locants>[1-9][0-9]*(?:,[1-9][0-9]*)*)-"
            r"(?P<ending>en|yn|dien|diyn|trien|triyn|tetraen|tetrayn|"
            r"pentaen|pentayn|hexaen|hexayn)-(?P<free>[1-9][0-9]*)-yl",
            tail,
        )
        if match:
            ending = match.group("ending")
            kind = "ene" if ending.endswith("en") or ending.endswith("ien") else "yne"
            count_by_ending = {
                "en": 1, "yn": 1, "dien": 2, "diyn": 2,
                "trien": 3, "triyn": 3, "tetraen": 4,
                "tetrayn": 4, "pentaen": 5, "pentayn": 5,
                "hexaen": 6, "hexayn": 6,
            }
            n_bonds = count_by_ending[ending]
            locants = tuple(int(item) for item in match.group("locants").split(","))
            free = int(match.group("free"))
            if (len(locants) == n_bonds and tuple(sorted(set(locants))) == locants
                    and (match.group("a") is None) == (n_bonds == 1)
                    and _valid_unsaturated_sites(
                        carbon_count,
                        locants if kind == "ene" else (),
                        locants if kind == "yne" else (),
                        free,
                    )
                    and _canonical_single_unsaturation(root, kind, locants, free) == component):
                return (carbon_count, locants if kind == "ene" else (),
                        locants if kind == "yne" else (), free, True)

        # A mixed ene/yne component. This bounded form admits one site of each
        # kind, with the source-required en-before-yn order.
        mixed = re.fullmatch(
            r"-(?P<elocs>[1-9][0-9]*(?:,[1-9][0-9]*)*)-en-"
            r"(?P<ylocs>[1-9][0-9]*(?:,[1-9][0-9]*)*)-yn-"
            r"(?P<free>[1-9][0-9]*)-yl",
            tail,
        )
        if mixed:
            ene = tuple(int(item) for item in mixed.group("elocs").split(","))
            yne = tuple(int(item) for item in mixed.group("ylocs").split(","))
            free = int(mixed.group("free"))
            if (len(ene) == 1 and len(yne) == 1
                    and tuple(sorted(set(ene))) == ene
                    and tuple(sorted(set(yne))) == yne
                    and not set(ene).intersection(yne)
                    and _valid_unsaturated_sites(carbon_count, ene, yne, free)
                    and _canonical_mixed_unsaturation(root, ene, yne, free) == component):
                return (carbon_count, ene, yne, free, True)
    return None


def _valid_unsaturated_sites(carbon_count, ene_locs, yne_locs, free_locant):
    """Check chain-edge ranges and one available monovalent free valence."""
    if not 1 <= free_locant <= carbon_count:
        return False
    sites = tuple(ene_locs) + tuple(yne_locs)
    if (len(set(sites)) != len(sites)
            or any(not 1 <= site < carbon_count for site in sites)):
        return False
    valence = [0] * (carbon_count + 1)
    for site in range(1, carbon_count):
        order = 3 if site in yne_locs else 2 if site in ene_locs else 1
        valence[site] += order
        valence[site + 1] += order
    return all(value <= 4 for value in valence[1:]) and valence[free_locant] <= 3


def _locant_text(locants):
    return ",".join(str(value) for value in locants)


def _canonical_single_unsaturation(root, kind, locants, free):
    suffix = "en" if kind == "ene" else "yn"
    if len(locants) == 1:
        stem = f"{root}-{locants[0]}-{suffix}"
    else:
        # P-31.1.1.2 inserts `a` before locants for multiplied endings.
        ending = _UNSATURATION_ENDINGS[len(locants)][kind]
        stem = f"{root}a-{_locant_text(locants)}-{ending}"
    return f"{stem}-{free}-yl"


def _canonical_mixed_unsaturation(root, ene_locs, yne_locs, free):
    return (f"{root}-{_locant_text(ene_locs)}-en-"
            f"{_locant_text(yne_locs)}-yn-{free}-yl")


@lru_cache(maxsize=1)
def _retained_hydrocarbon_ring_stems():
    """Use the existing P-25 hydride vocabulary, not arbitrary word endings."""
    from .retained_hydrocarbon_parents import _vocabulary

    result = {}
    for record in _vocabulary().values():
        parent = record['name']
        if re.fullmatch(r'[a-z]+e', parent):
            result[parent[:-1]] = frozenset(record['original_locants'].values())
    return result


def _retained_hydrocarbon_ring_form(component):
    """Check a source-named, unmodified retained ring's free-valence locant.

    This validates a spelling only. A caller must independently establish
    that the original mapped component is an unsubstituted ring hydride.
    """
    match = re.fullmatch(r"([a-z]+)-([1-9][0-9]*[a-z]?)-yl", component)
    if match is None:
        return False
    return match.group(2) in _retained_hydrocarbon_ring_stems().get(match.group(1), ())


@lru_cache(maxsize=4096)
def _known_heteroaryl_ring_form(component):
    """Validate native numbered heteroaryl vocabulary, not graph nature.

    The caller separately proves the complete unmodified aromatic component.
    Known FusionComponent parents supply the actual permissible locant set.
    """
    match=re.fullmatch(r'(.+)-([1-9][0-9]*[a-z]?)-yl',component)
    if match is None:
        return False
    from .fusion_components import resolve_component
    stem=re.sub(r'^(\d+(?:,\d+)*)-',r'[\1]',match[1])
    for parent in (stem,stem+'e'):
        record=resolve_component(parent)
        if (record is not None and any(element!=6 for element,count in record.elements)
                and match[2] in dict(record.label_elements)):
            return True
    return False


def _is_supported_component(component, nature):
    """Check simple forms or validate caller-proved compound text."""
    if nature == "simple":
        if (component in {"phenyl", "tert-butyl"} | _CHAIN_AMBIGUOUS_PREFIXES | _RETAINED_SIMPLE_OXY_PREFIXES
                or _alkyl_form(component)
                or _unsaturated_alkyl_form(component)
                or _retained_hydrocarbon_ring_form(component)
                or _known_heteroaryl_ring_form(component)
                or any(component == 'cyclo' + root + 'yl'
                       for count, root in ROOTS.items() if count >= 3)):
            return True
        # P-82.2.1 and its printed repeated-methyl example establish a simple
        # isotopically modified methyl prefix; other isotope grammars are
        # deliberately left to their owning component producer.
        return bool(re.fullmatch(r"\((?:13|14)C\)methyl", component))
    if nature == "compound":
        return _wellformed_rendered_component(component)
    return False


def simple_component_has_internal_numeric_locants(component):
    """Whether the bounded simple-name grammar proves internal numeric locants.

    This supports P-16.5.1.2's separate locant-separation rule. It does not
    change the component's simple/compound nature or the multiplication rules.
    Unknown component vocabularies are deliberately not inferred from hyphens.
    """
    if not isinstance(component, str) or not _is_supported_component(component, "simple"):
        return False
    alkyl = _alkyl_form(component)
    if alkyl is not None and "an-" in component:
        return True
    if _retained_hydrocarbon_ring_form(component) or _known_heteroaryl_ring_form(component):
        return True
    unsaturated = _unsaturated_alkyl_form(component)
    return bool(unsaturated and unsaturated[4])


def _wellformed_rendered_component(component):
    """Validate display syntax without reparsing caller-proved chemistry."""
    if (not component or component != component.strip() or any(c.isspace() for c in component)
            # The shared enclosing helper can generate braces for a new
            # wrapper, but it has no next level for wrapping a unit that
            # already contains braces.
            or "{" in component or "}" in component
            or not re.fullmatch(r"[A-Za-z0-9,.'’′^+\-\[\]()]+", component)
            or not any(c.isalpha() for c in component)):
        return False
    pairs = {"(": ")", "[": "]"}
    closing = {value: key for key, value in pairs.items()}
    stack = []
    for char in component:
        if char in pairs:
            stack.append(char)
        elif char in closing:
            if not stack or stack.pop() != closing[char]:
                return False
    return not stack


def _needs_parentheses(component, nature):
    if nature == "compound":
        return True
    if component in {"tert-butyl", "tert-butoxy"}:
        return False
    if _retained_hydrocarbon_ring_form(component) or _known_heteroaryl_ring_form(component):
        return True
    if _alkyl_form(component) and "an-" in component:
        return True
    unsaturated = _unsaturated_alkyl_form(component)
    if unsaturated:
        # The two unlocanted, two-carbon names are simple prefixes without
        # the P-16.3.4(b) locant trigger. Other P-32 acyclic forms are locanted.
        return unsaturated[4]
    if component.startswith("(") or "[" in component or "]" in component:
        return True
    alkyl = _alkyl_form(component)
    if not alkyl:
        return False
    root = alkyl[0]
    # P-16.3.4(c) explicitly identifies dodecyl and includes names whose
    # spelling begins with a full numerical prefix. Clause (d) separately
    # identifies decyl. Do not strip a final ``a`` from every multiplier:
    # that would incorrectly classify hexyl/heptyl/octyl/nonyl as ambiguous.
    if any(root.startswith(head) for head in _MULTIPLICATIVE_HEADS) or root in {"dec", "dodec"}:
        return True
    # These adjacent roots are directly established as ordinary simple
    # prefixes by the focused source audit. A different root that begins
    # only with the vowel-elided spelling of a numerical prefix is left
    # unresolved rather than assigned an enclosure by analogy.
    if root in {"hex", "hept", "oct", "non"}:
        return False
    if any(root.startswith(head[:-1]) and not root.startswith(head)
           for head in _MULTIPLICATIVE_HEADS if head.endswith("a")):
        return None
    return False


def multiply_substituent(component, count, *, nature):
    """Render a mapped component's P-16 multiplier, or return ``None``.

    ``nature`` must be ``"simple"`` or ``"compound"`` and comes from the
    caller's component mapping. In particular, ``compound`` is an explicit
    caller proof that the already-rendered unit is substituted/complex under
    P-16.3.2(c); its internals are not reclassified here. Such text is checked
    only for safe balanced enclosure syntax. ``simple`` is accepted only for
    bounded plain alkyl, phenyl, retained tert-butyl, source-illustrated 13C/14C
    methyl, unannotated monovalent acyclic unsaturated carbon-prefix forms, and
    locanted retained hydrocarbon ring prefixes in the P-25 parent vocabulary.
    The caller must prove the mapped graph is simple; this text grammar does
    not detect branches or annotations in the source structure. Counts of one
    preserve the unit; larger counts use P-16's numerical prefix and enclosure
    rules.
    """
    if (not isinstance(component, str) or not component or isinstance(count, bool)
            or not isinstance(count, int) or count < 1
            or not _is_supported_component(component, nature)):
        return None
    if count == 1:
        return SubstituentMultiplication(component, component, 1, nature, "", False)
    if nature == "simple" and component in _CHAIN_AMBIGUOUS_PREFIXES:
        multiplier = COMPLEX_MULT.get(count)
        if multiplier is None:
            return None
        return SubstituentMultiplication(multiplier + enclose_name_unit(component),
            component, count, nature, multiplier, True)
    if nature == "compound":
        multiplier = COMPLEX_MULT.get(count)
        if multiplier is None:
            return None
        return SubstituentMultiplication(
            multiplier + enclose_name_unit(component), component, count, nature,
            multiplier, True)
    multiplier = MULT.get(count)
    if multiplier is None:
        return None
    if component in {"tert-butyl", "tert-butoxy"}:
        # P-16.2.4.1(d) separates this retained italic prefix from the Roman
        # multiplier with a hyphen.
        text = multiplier + "-" + component
        enclosed = False
    else:
        enclosure = _needs_parentheses(component, nature)
        if enclosure is None:
            return None
        if enclosure:
            text = multiplier + enclose_name_unit(component)
            enclosed = True
        else:
            text = multiplier + component
            enclosed = False
    return SubstituentMultiplication(text, component, count, nature, multiplier, enclosed)


substituent_multiplication = multiply_substituent
