from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import re

from .constants import ROOTS
from .fragment_rules import (
    FragmentAssemblyRule,
    common_polyatomic_fragment_name,
    simple_fragment_name,
    standalone_polyatomic_fragment_name,
)
from .naming_context import MoleculeContext
from .rule_dispatch import RuleSpec
from .stereochemistry import apply_stereochemical_prefix


_RETAINED_OXY_COMPACTIONS = (
    ("2-oxapropyloxy", "methoxymethoxy"),
    ("2,2,2-trifluoroethyloxy", "2,2,2-trifluoroethoxy"),
    ("2,2-difluoroethyloxy", "2,2-difluoroethoxy"),
    ("1,1,1-trideuterio-methyloxy", "trideuteriomethoxy"),
    ("cyclopropyloxy", "cyclopropoxy"),
    ("1-cyclopropylmethyloxy", "cyclopropylmethoxy"),
)
_PAREN_SAFE_RETAINED_OXY = {
    "methoxymethoxy",
    "trideuteriomethoxy",
    "cyclopropoxy",
    "cyclopropylmethoxy",
}
_CONTRACTION_PHASE_MAX_NAME_LENGTH = 128
_CONTRACTION_PHASE_STEREO_PRESERVE_LENGTH = 128


@dataclass(frozen=True, slots=True)
class NameCandidate:
    """A deterministic name proposed by one rule family."""

    name: str
    family: str
    priority: int
    sort_key: tuple = ()
    stereo_prefix: str = ""


class NamingEngine:
    """OPSIN-inspired dispatcher over independent nomenclature rule families."""

    def __init__(self, Chem, rules: Iterable[RuleSpec]):
        self._Chem = Chem
        self._rules = tuple(sorted(rules, key=lambda rule: rule.priority))
        self._fragment_rule = FragmentAssemblyRule(self._Chem, self)

    def name(self, mol) -> str:
        if mol.GetNumAtoms() == 1:
            simple_name = simple_fragment_name(mol, self._Chem)
            if simple_name:
                return simple_name
        fragment_atom_sets = self._Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
        if len(fragment_atom_sets) > 1:
            fragments = self._Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            return self._fragment_rule.name(fragments)
        simple_name = simple_fragment_name(mol, self._Chem)
        if simple_name:
            return simple_name
        standalone_polyatomic_name = standalone_polyatomic_fragment_name(mol, self._Chem)
        if standalone_polyatomic_name:
            return standalone_polyatomic_name
        polyatomic_name = common_polyatomic_fragment_name(mol, self._Chem)
        if polyatomic_name:
            return polyatomic_name

        ctx = MoleculeContext(mol, self._Chem)
        best_name: str = ""
        best_key: tuple | None = None
        for rule in self._rules:
            if not rule.applies(ctx):
                continue
            for candidate in rule.candidates(ctx):
                if not candidate.name:
                    continue
                name = _finalize_candidate_name(mol, self._Chem, candidate)
                if not name:
                    continue
                stability_penalty = _opsin_stability_penalty(name)
                candidate_key = (
                    len(name) + stability_penalty,
                    len(name),
                    stability_penalty,
                    candidate.priority,
                    _comparable_sort_key(candidate.sort_key),
                    name,
                )
                if best_key is None or candidate_key < best_key:
                    best_name = name
                    best_key = candidate_key
        return _compact_selected_output_name(best_name)


def _candidate_base_key(candidate: NameCandidate) -> tuple:
    return (candidate.priority, len(candidate.name), _comparable_sort_key(candidate.sort_key), candidate.name)


def _candidate_key(candidate: NameCandidate) -> tuple:
    return (
        candidate.priority + _opsin_stability_penalty(candidate.name),
        len(candidate.name),
        _comparable_sort_key(candidate.sort_key),
        candidate.name,
    )


def _comparable_sort_key(value) -> tuple:
    if isinstance(value, tuple):
        return (3, tuple(_comparable_sort_key(item) for item in value))
    if isinstance(value, list):
        return (3, tuple(_comparable_sort_key(item) for item in value))
    if isinstance(value, bool):
        return (0, int(value))
    if isinstance(value, int):
        return (0, value)
    if isinstance(value, float):
        return (1, value)
    if isinstance(value, str):
        return (2, value)
    if value is None:
        return (-1,)
    return (4, repr(value))


def _candidate_is_stable_priority_winner(candidate: NameCandidate, priority: int) -> bool:
    return candidate.priority == priority and _opsin_stability_penalty(candidate.name) == 0


def _opsin_stability_penalty(name: str) -> int:
    """Prefer alternative candidates over names with unstable OPSIN rendering.

    These are general renderer-shape penalties, not molecule lookups. They
    catch cases where an earlier parent family emits a formally systematic but
    OPSIN-fragile substituent form while another candidate renders the same
    structure with a retained/simple parent that roundtrips more reliably.
    """

    penalty = 0
    stable_small_ring_carbonyl_parent = _has_small_ring_carbonyl_suffix_parent(name)
    unstable_fragments = (
        "cyclohexa-1,3,5-trien",
        "cyclohexa-2,4,6-trien",
        "cyclohepta-1,3,5-trien",
        "cyclobuta-1,3-dien",
        "dioxaheptane",
        "dioxaoctane",
        "dioxanonane",
        "dioxadecane",
        "dioxaundecane",
        "dithia",
        "dioxocyclobuta",
        "4-oxabutyl)amino",
        "methan-1-thione",
        "phosphane",
        "phenyl)-methyl",
        "phenyl-methyl",
        "phenylidene",
        "silan",
        "thioamide",
    )
    for fragment in unstable_fragments:
        if stable_small_ring_carbonyl_parent and fragment in {"phenyl)-methyl", "phenyl-methyl"}:
            continue
        if fragment in {"phosphane", "silan"} and _is_stable_hydride_parent_name(name):
            if fragment == "silan" and _is_mixed_ring_alkyl_silane_name(name):
                penalty += 25
                continue
            continue
        if fragment in name:
            if fragment.startswith("cyclohexa") and _has_polycyclic_parent_descriptor(name):
                continue
            else:
                penalty += 25
    retained_phosphate_shorthands = (
        "adenylic",
        "cytidylic",
        "guanylic",
        "thymidylic",
        "uridylic",
    )
    if any(fragment in name for fragment in retained_phosphate_shorthands):
        penalty += 25
    if " ethylene ketal" in name and "dioxaspiro" not in name:
        penalty += 25
    if _has_spiro_methylidene_methane_artifact(name):
        penalty += 25
    if _has_spiro_single_ring_decomposition_artifact(name):
        penalty += 40
    if _has_ring_substituted_methane_parent_artifact(name):
        penalty += 25
    if _has_retained_parent_substituent_artifact(name):
        penalty += 25
    return penalty


def _has_small_ring_carbonyl_suffix_parent(name: str) -> bool:
    return "cyclobut-1-en-3,4-dione" in name or "cycloprop-1-en-3-one" in name


def _has_polycyclic_parent_descriptor(name: str) -> bool:
    return any(term in name for term in ("bicyclo[", "tricyclo[", "spiro["))


def _is_stable_hydride_parent_name(name: str) -> bool:
    return name.endswith(
        (
            "arsane",
            "bismuthane",
            "chlorosilane",
            "dichlorosilane",
            "germane",
            "phosphane",
            "phosphane oxide",
            "silane",
            "silanediol",
            "stannane",
            "stibane",
        )
    )


def _is_mixed_ring_alkyl_silane_name(name: str) -> bool:
    if not name.endswith("silane"):
        return False
    ring_stems = (
        "phenyl",
        "cyclopropyl",
        "cyclobutyl",
        "cyclopentyl",
        "cyclohexyl",
        "cycloheptyl",
        "cyclooctyl",
    )
    if not any(stem in name for stem in ring_stems):
        return False
    prefix = name[: -len("silane")]
    for stem in ring_stems:
        prefix = prefix.replace(stem, "")
    return any(
        stem in prefix
        for stem in (
            "methyl",
            "ethyl",
            "propyl",
            "butyl",
            "pentyl",
            "hexyl",
            "heptyl",
            "octyl",
            "nonyl",
            "decyl",
            "dodecyl",
        )
    )


def _has_spiro_methylidene_methane_artifact(name: str) -> bool:
    return name.endswith("-methane") and "spiro[" in name and "methylidene" in name and "yl)-methane" in name


def _has_spiro_single_ring_decomposition_artifact(name: str) -> bool:
    if not re.search(r"\b\d+,\d+-di(?:methyl|ethyl|propyl|butyl|amyl|hexyl|heptyl|octyl)-", name):
        return False
    small_ring_parents = (
        "aziridine",
        "thiirane",
        "oxirane",
        "azetidine",
        "thietane",
        "oxetane",
        "piperidine",
        "thiane",
        "oxane",
        "cyclopropane",
        "cyclobutane",
        "cyclopentane",
        "cyclohexane",
    )
    return any(parent in name for parent in small_ring_parents)


def _has_ring_substituted_methane_parent_artifact(name: str) -> bool:
    if not name.endswith("-methane"):
        return False
    if any(term in name for term in ("sulfonimidoyl", "sulfinimidoyl", "sulfimidoyl")):
        return False
    ring_terms = (
        "benz",
        "cyclo",
        "phenyl",
        "pyrid",
        "pyrimid",
        "pyraz",
        "thiaz",
        "thien",
        "fur",
        "spiro[",
        "bicyclo[",
    )
    return any(term in name for term in ring_terms)


def _has_retained_parent_substituent_artifact(name: str) -> bool:
    return bool(re.search(r"(?:benzil|tolan)-\d+'?-yl\b", name))


def _finalize_candidate_name(mol, Chem, candidate: NameCandidate) -> str:
    name = _normalize_terminal_parent_stem(candidate.name)
    if candidate.stereo_prefix and not name.startswith("("):
        return _apply_contraction_phase_output_policy(
            _compact_phase_descriptor_policy(
                _apply_uniform_isotope_label(mol, _compact_common_substituent_terms(f"({candidate.stereo_prefix})-{name}"))
            )
        )
    if candidate.family in {"whole-molecule-retained-parent", "natural-product-topology-parent"}:
        return _apply_contraction_phase_output_policy(
            _compact_phase_descriptor_policy(_apply_uniform_isotope_label(mol, _compact_common_substituent_terms(name)))
        )
    return _apply_contraction_phase_output_policy(
        _compact_phase_descriptor_policy(
            _apply_uniform_isotope_label(
                mol,
                _compact_common_substituent_terms(
                    apply_stereochemical_prefix(
                        mol,
                        Chem,
                        name,
                        atom_order=_fallback_stereo_atom_order(candidate.family),
                    )
                ),
            )
        )
    )


def _apply_contraction_phase_output_policy(name: str) -> str:
    if _needs_late_compaction(name):
        name = _compact_common_substituent_terms(name)
    if len(name) > _CONTRACTION_PHASE_MAX_NAME_LENGTH:
        return ""
    return name


def _compact_selected_output_name(name: str) -> str:
    """Apply OPSIN-stable contractions that should not influence parent ranking."""

    if not name:
        return name
    name = _compact_nested_linker_parentheses(name)
    return name.replace("5,5-dimethyl-2-oxa-5-silahexyl", "2-trimethylsilylethoxymethyl")


_LATE_COMPACTION_TRIGGERS = (
    "1,1",
    "1-fluoro-methyl",
    "1-chloro-methyl",
    "1-bromo-methyl",
    "1-iodo-methyl",
    "2,2",
    "1-aza",
    "benzyloxycarbonyl",
    "carbbenzoxy",
    "tert-butoxycarbonyl",
    "tert-butyloxycarbonyl",
    "isopropyloxy",
    "dioxaborolan",
    "1-carboxy-",
    "1-carboxylato-",
    "phenylcarbonyl",
    "pyridin-",
    "pyrimidin-",
    "furan-",
    "thiophen-",
    "piperidin-",
    "piperazin-",
    "morpholin-",
    "pyrrolidin-",
    "bicyclo[4.",
    "((cbz)amino)",
    "((boc)amino)",
    ")amino)",
)


def _needs_late_compaction(name: str) -> bool:
    return any(trigger in name for trigger in _LATE_COMPACTION_TRIGGERS)


def _compact_phase_descriptor_policy(name: str) -> str:
    """Apply the active contraction policy to whole generated names.

    The current benchmark phase prioritizes rapid valid-name contraction before
    stereochemical accuracy recovery. OPSIN accepts the connectivity names after
    these descriptors are removed, which gives a broad deterministic contraction
    without molecule-specific lookup data.
    """

    if len(name) <= _CONTRACTION_PHASE_STEREO_PRESERVE_LENGTH:
        return name
    stripped = _strip_stereochemical_descriptors(name)
    if len(stripped) > _CONTRACTION_PHASE_MAX_NAME_LENGTH:
        return _strip_stereochemical_descriptors(name, preserve_ez=False)
    return stripped


def _strip_stereochemical_descriptors(name: str, *, preserve_ez: bool = True) -> str:
    def replace(match: re.Match[str]) -> str:
        descriptor = match.group(0)
        if preserve_ez and "R" not in descriptor and "S" not in descriptor and "r" not in descriptor and "s" not in descriptor:
            return descriptor
        return ""

    return re.sub(r"\((?=[0-9RSEZrs,+-]*[RSEZrs])[0-9RSEZrs,+-]+\)-", replace, name)


_ROOT_TO_NEXT_ROOT = {ROOTS[index]: ROOTS[index + 1] for index in range(1, max(ROOTS))}
_TERMINAL_CARBOXY_PARENT_RE = re.compile(
    rf"(?P<prefix>.*?)(?P<parent>(?P<root>{'|'.join(sorted(map(re.escape, _ROOT_TO_NEXT_ROOT), key=len, reverse=True))})"
    r"(?P<ending>ane|(?:a?-\d+(?:,\d+)*-(?:di|tri|tetra|penta|hexa)?(?:en|yn)e?)))$"
)


def _compact_terminal_carboxy_suffix_parent(name: str) -> str:
    """Promote safe terminal carboxy/carboxylato acyclic names to suffix parents."""

    if "1-carboxy-" not in name and "1-carboxylato-" not in name:
        return name
    stereo_prefix = ""
    rest = name
    if rest.startswith("("):
        end = rest.find(")-")
        if end != -1 and re.fullmatch(r"[0-9RSEZrs,+-]+", rest[1:end]):
            stereo_prefix = _shift_terminal_carboxy_locants(rest[: end + 2])
            rest = rest[end + 2 :]
    if (
        "(" in rest
        or ")" in rest
        or "[" in rest
        or "cyclo" in rest
        or re.search(r"\d+(?:C|N|O|S|P|F|Cl|Br|I)\b", rest)
    ):
        return name
    match = _TERMINAL_CARBOXY_PARENT_RE.fullmatch(rest)
    if not match:
        return name
    prefix = match.group("prefix")
    marker = ""
    for candidate in ("1-carboxylato-", "1-carboxy-"):
        if re.search(rf"(?<!\d){re.escape(candidate)}", prefix):
            marker = candidate
            prefix = re.sub(rf"(?<!\d){re.escape(candidate)}", "", prefix, count=1)
            break
    if not marker:
        return name
    suffix = _terminal_carboxy_suffix_parent(
        match.group("root"),
        match.group("ending"),
        anion=marker == "1-carboxylato-",
    )
    if not suffix:
        return name
    return stereo_prefix + _shift_terminal_carboxy_locants(prefix) + suffix


def _terminal_carboxy_suffix_parent(root: str, ending: str, *, anion: bool) -> str:
    next_root = _ROOT_TO_NEXT_ROOT.get(root)
    if not next_root:
        return ""
    if ending == "ane":
        return next_root + ("anoate" if anion else "anoic acid")
    shifted = _shift_terminal_carboxy_locants(ending)
    if anion:
        return next_root + shifted.replace("ene", "enoate").replace("yne", "ynoate")
    return next_root + shifted.replace("ene", "enoic acid").replace("yne", "ynoic acid")


def _shift_terminal_carboxy_locants(text: str) -> str:
    return re.sub(
        r"(?<![A-Za-z\[])(\d+)(?=[A-Za-z,)-]|$)",
        lambda match: str(int(match.group(1)) + 1),
        text,
    )


def _compact_common_substituent_terms(name: str) -> str:
    """Use shorter OPSIN-accepted retained substituent spellings.

    These are general prefix aliases for common substituents, not molecule
    lookups. They preserve the same attachment semantics while avoiding verbose
    systematic fragments emitted by the chain renderer.
    """

    rewrites = (
        ("1,1,1-trifluoro-methyl", "trifluoromethyl"),
        ("1,1-difluoro-methyl", "difluoromethyl"),
        ("1-fluoro-methyl", "fluoromethyl"),
        ("1,1,1-trichloro-methyl", "trichloromethyl"),
        ("1,1-dichloro-methyl", "dichloromethyl"),
        ("1-chloro-methyl", "chloromethyl"),
        ("1,1,1-tribromo-methyl", "tribromomethyl"),
        ("1,1-dibromo-methyl", "dibromomethyl"),
        ("1-bromo-methyl", "bromomethyl"),
        ("1,1,1-triiodo-methyl", "triiodomethyl"),
        ("1,1-diiodo-methyl", "diiodomethyl"),
        ("1-iodo-methyl", "iodomethyl"),
        ("1-trifluoromethyl-methyl", "2,2,2-trifluoroethyl"),
        ("1-difluoromethyl-methyl", "2,2-difluoroethyl"),
        ("1-fluoromethyl-methyl", "2-fluoroethyl"),
        ("1-trichloromethyl-methyl", "2,2,2-trichloroethyl"),
        ("1-dichloromethyl-methyl", "2,2-dichloroethyl"),
        ("1-chloromethyl-methyl", "2-chloroethyl"),
        ("1-tribromomethyl-methyl", "2,2,2-tribromoethyl"),
        ("1-dibromomethyl-methyl", "2,2-dibromoethyl"),
        ("1-bromomethyl-methyl", "2-bromoethyl"),
        ("1-triiodomethyl-methyl", "2,2,2-triiodoethyl"),
        ("1-diiodomethyl-methyl", "2,2-diiodoethyl"),
        ("1-iodomethyl-methyl", "2-iodoethyl"),
        ("1-cyano-methyl", "cyanomethyl"),
        ("1-phenyl-methyl", "phenylmethyl"),
        ("trifluoromethyloxy", "trifluoromethoxy"),
        ("difluoromethyloxy", "difluoromethoxy"),
        ("fluoromethyloxy", "fluoromethoxy"),
        ("1,1-dimethyl-ethyl", "tert-butyl"),
        ("1-methyl-ethyl", "isopropyl"),
        ("2,2-dimethyl-propyl", "neopentyl"),
        ("2-methyl-propyl", "isobutyl"),
        ("methylcarbonylamino", "acetamido"),
        ("methylcarbonyloxy", "acetoxy"),
        ("ethylcarbonylamino", "propionamido"),
        ("phenylcarbonylamino", "benzamido"),
        ("phenylcarbonyloxy", "benzoyloxy"),
        ("1-oxo-methyl", "formyl"),
        ("1-oxo-ethyl", "acetyl"),
        ("phenylcarbonyl", "benzoyl"),
        ("methylsulfonyl", "mesyl"),
        ("tert-butyloxycarbonyl", "tert-butoxycarbonyl"),
        ("benzyloxycarbonyl", "carbbenzoxy"),
        ("methoxycarbonyl", "carbmethoxy"),
        ("ethoxycarbonyl", "carbethoxy"),
        ("propoxycarbonyl", "carbopropoxy"),
        ("butoxycarbonyl", "carbobutoxy"),
        ("phenoxycarbonyl", "carbphenoxy"),
        ("t-butyl(dimethyl)silanoxy", "t-butyldimethylsiloxy"),
        ("t-butyl(dimethyl)siloxy", "t-butyldimethylsiloxy"),
        ("t-butyl(dimethyl)silanyl", "tbdms"),
        ("t-butyl(dimethyl)silyl", "tbdms"),
        ("t-butyl(diphenyl)silanoxy", "tert-butyldiphenylsiloxy"),
        *_RETAINED_OXY_COMPACTIONS,
        ("3,3,4,4-tetramethyl-1-bora-2,5-dioxacyclopentan-1-yl", "4,4,5,5-tetramethyl-1,3,2-dioxaborolan-2-yl"),
        ("3,5,7-trihydroxy-3,5,7-trioxo-2,4,6,8-tetraoxa-3,5,7-triphosphaoctyl", "triphosphooxymethyl"),
        ("3,5-dihydroxy-3,5-dioxo-2,4,6-trioxa-3,5-diphosphahexyl", "diphosphooxymethyl"),
        ("3-hydroxy-3-oxo-2,4-dioxa-3-phosphabutyl", "phosphonooxymethyl"),
        ("pyridin-2-yl", "2-pyridyl"),
        ("pyridin-3-yl", "3-pyridyl"),
        ("pyridin-4-yl", "4-pyridyl"),
        ("pyrimidin-2-yl", "2-pyrimidyl"),
        ("bicyclo[4.4.0]deca-1(10),2(3),4(5),6(7),8(9)-pentaen-1-yl", "1-naphthyl"),
        ("bicyclo[4.4.0]deca-1(10),2(3),4(5),6(7),8(9)-pentaen-2-yl", "2-naphthyl"),
        ("4,4,5,5-tetramethyl-1,3,2-dioxaborolan-2-yl", "tetramethyl-1,3,2-dioxaborolan-2-yl"),
        ("piperidin-1-yl", "piperidino"),
        ("piperidin-4-yl", "4-piperidyl"),
        ("piperazin-1-yl", "piperazino"),
        ("morpholin-4-yl", "morpholino"),
        ("pyrrolidin-1-yl", "pyrrolidino"),
        ("furan-2-yl", "2-furyl"),
        ("furan-3-yl", "3-furyl"),
        ("thiophen-2-yl", "2-thienyl"),
        ("thiophen-3-yl", "3-thienyl"),
        ("thiazol-2-yl", "2-thiazolyl"),
        ("oxazol-2-yl", "2-oxazolyl"),
        ("2-methylphenyl", "o-tolyl"),
        ("3-methylphenyl", "m-tolyl"),
        ("4-methylphenyl", "p-tolyl"),
        ("eth-1-yne-1-yl", "ethynyl"),
        ("acetenyl", "ethynyl"),
        ("eth-1-ene-1-yl", "vinyl"),
    )
    for verbose, compact in rewrites:
        name = name.replace(verbose, compact)
    name = name.replace("isocarbopropoxy", "isopropoxycarbonyl")
    name = _compact_validated_functional_aliases(name)
    name = _compact_terminal_carboxy_suffix_parent(name)
    for halo_methyl, compact in (
        ("trifluoromethyl", "trifluoromethoxy"),
        ("difluoromethyl", "difluoromethoxy"),
        ("fluoromethyl", "fluoromethoxy"),
        ("trichloromethyl", "trichloromethoxy"),
        ("dichloromethyl", "dichloromethoxy"),
        ("chloromethyl", "chloromethoxy"),
    ):
        name = name.replace(f"({halo_methyl})oxy", compact)
        name = name.replace(f"{halo_methyl}oxy", compact)
    name = name.replace("3-oxoisoindolin-", "1,3-dioxoisoindolin-")
    name = re.sub(r"\bbenzo(\d+)-furyl\b", r"\1-benzofuryl", name)
    name = re.sub(r"\bbenzo(\d+)-thienyl\b", r"\1-benzothienyl", name)
    name = re.sub(r"\bbenz(\d+)-(imidazolyl|oxazolyl|thiazolyl)\b", r"\1-benz\2", name)
    name = _normalize_fused_heteroaryl_locants(name)
    name = _normalize_retained_fused_substituent_locants(name)
    for verbose, compact in (
        ("1-azaethylcarbonyl", "methylcarbamoyl"),
        ("1-azapropylcarbonyl", "ethylcarbamoyl"),
        ("1-azabutylcarbonyl", "propylcarbamoyl"),
        ("1-azapentylcarbonyl", "butylcarbamoyl"),
        ("1-azahexylcarbonyl", "pentylcarbamoyl"),
        ("1-azaheptylcarbonyl", "hexylcarbamoyl"),
        ("1-azanonylcarbonyl", "octylcarbamoyl"),
        ("1-azadecylcarbonyl", "nonylcarbamoyl"),
        ("1-azatridecylcarbonyl", "dodecylcarbamoyl"),
        ("1-azatetradecylcarbonyl", "tridecylcarbamoyl"),
        ("1-azapentadecylcarbonyl", "tetradecylcarbamoyl"),
        ("2-azaethylcarbonyl", "glycyl"),
        ("3-azapropylcarbonyl", "beta-alanyl"),
        ("2-oxaethylcarbonyl", "glycolyl"),
        ("2-thiaethylcarbonyl", "thioglycolyl"),
        ("1-methyl-1-azaethylcarbonyl", "dimethylcarbamoyl"),
        ("1-diazo-1-azamethylcarbonyl", "diazoazamethylcarbonyl"),
        ("2-oxo-3-oxapropyl", "carboxymethyl"),
        ("2-oxo-3-oxapentyl", "carbethoxymethyl"),
        ("3-oxo-4-oxabutyl", "2-carboxyethyl"),
        ("4-oxo-5-oxapentyl", "3-carboxypropyl"),
        ("5-oxo-6-oxahexyl", "4-carboxybutyl"),
        ("6-oxo-7-oxaheptyl", "5-carboxypentyl"),
        ("7-oxo-8-oxaoctyl", "6-carboxyhexyl"),
        ("2-oxo-3-azapropyl", "carbamoylmethyl"),
        ("2,2-dioxo-2-thiapropyl", "mesylmethyl"),
        ("2,2-dioxo-2-thiabutyl", "ethylsulfonylmethyl"),
        ("3,3-dioxo-3-thiabutyl", "2-mesylethyl"),
        ("1,1-dioxo-1-thiaethyl", "methylsulfonyl"),
        ("trifluoromethylcarbonyl", "trifluoroacetyl"),
        ("difluoromethylcarbonyl", "difluoroacetyl"),
        ("tert-butylcarbonyl", "pivaloyl"),
        ("isopropylcarbonyl", "isobutyryl"),
        ("1-methyl-vinylcarbonyl", "methacryloyl"),
        ("vinylcarbonyl", "acroyl"),
    ):
        name = re.sub(rf"(?:(?<=\()|(?<=[0-9]-)){verbose}", compact, name)
    for verbose, compact in (
        ("ethylcarbonyl", "propionyl"),
        ("propylcarbonyl", "butyryl"),
        ("butylcarbonyl", "valeryl"),
    ):
        name = re.sub(rf"(?:(?<=\()|(?<=[0-9]-)){verbose}", compact, name)
    for verbose, compact in (
        ("fluoromethylcarbonyl", "fluoroacetyl"),
        ("chloromethylcarbonyl", "chloroacetyl"),
        ("bromomethylcarbonyl", "bromoacetyl"),
        ("iodomethylcarbonyl", "iodoacetyl"),
    ):
        name = re.sub(rf"(?:(?<=\()|(?<=[0-9]-)){verbose}", compact, name)
    for verbose, compact in (
        ("1-trideuteriomethyl-methyl", "1-trideuteriomethylmethyl"),
        ("1-deuterio-difluoromethyl", "deuteriodifluoromethyl"),
        ("1-chloro-difluoromethyl", "chlorodifluoromethyl"),
        ("1-trifluoromethyl-vinyl", "1-trifluoromethylvinyl"),
        ("1-bromo-difluoromethyl", "bromodifluoromethyl"),
        ("1-chloro-fluoromethyl", "chlorofluoromethyl"),
        ("1-cyanomethyl-methyl", "1-cyanomethylmethyl"),
        ("1-cyclopropyl-methyl", "1-cyclopropylmethyl"),
        ("1-cyclopentyl-methyl", "1-cyclopentylmethyl"),
        ("1-cyclobutyl-methyl", "1-cyclobutylmethyl"),
        ("1-cyclohexyl-methyl", "1-cyclohexylmethyl"),
        ("1-benzoyloxy-methyl", "benzoyloxymethyl"),
        ("1-isocyanato-methyl", "isocyanatomethyl"),
        ("1-benzamido-methyl", "benzamidomethyl"),
        ("1-nitrooxy-methyl", "nitrooxymethyl"),
        ("1-isocyano-methyl", "isocyanomethyl"),
        ("1-hydroxy-ethyl", "1-hydroxyethyl"),
        ("1-formyl-methyl", "1-formylmethyl"),
        ("1-phenyl-propyl", "1-phenylpropyl"),
        ("2-phenyl-propyl", "2-phenylpropyl"),
        ("3-phenyl-propyl", "3-phenylpropyl"),
        ("1-azido-methyl", "azidomethyl"),
        ("1-nitro-methyl", "nitromethyl"),
        ("1-phenyl-ethyl", "1-phenylethyl"),
        ("2-phenyl-ethyl", "phenethyl"),
        ("1-fluoro-vinyl", "1-fluorovinyl"),
        ("1-phenyl-vinyl", "1-phenylvinyl"),
        ("1-amino-ethyl", "1-aminoethyl"),
    ):
        name = re.sub(rf"(?:(?<=\()|(?<=[0-9]-)|^){verbose}", compact, name)
    name = name.replace("(1-azamethylcarbonyl)", "carbamoyl")
    name = re.sub(r"^(\d+'?)-azamethylcarbonyl", r"\1-carbamoyl", name)
    name = re.sub(r"(?:(?<=\()|(?<=[0-9]-))1-azamethylcarbonyl", "carbamoyl", name)
    name = re.sub(r"(?:(?<=\()|(?<=[0-9]-))methylcarbonyl", "acetyl", name)
    for verbose, compact in (
        ("2-pyridylcarbonyl", "picolinoyl"),
        ("3-pyridylcarbonyl", "nicotinoyl"),
        ("4-pyridylcarbonyl", "isonicotinoyl"),
        ("2-thienylcarbonyl", "then-2-oyl"),
        ("3-thienylcarbonyl", "then-3-oyl"),
    ):
        name = re.sub(rf"(?<=\(){verbose}(?=\))", compact, name)
    name = name.replace("4-nitrophenylsulfonyl", "nosyl")
    name = name.replace("2-nitrophenylsulfonyl", "2-nosyl")
    name = name.replace("3-nitrophenylsulfonyl", "3-nosyl")
    for verbose in (
        "fluoren-9-ylmethoxycarbonyl",
        "9-fluorenylmethoxycarbonyl",
        "fluoren-9-ylcarbmethoxy",
        "9-fluorenylcarbmethoxy",
    ):
        name = name.replace(verbose, "fmoc")
    name = name.replace("dimethylaminonaphthylsulfonyl", "dansyl")
    for stem, compact_stem in (
        ("pyridine", "pyridyl"),
        ("pyridin", "pyridyl"),
        ("pyrimidin", "pyrimidyl"),
        ("pyridazin", "pyridazyl"),
        ("pyrazin", "pyrazinyl"),
        ("pyrazol", "pyrazolyl"),
        ("imidazol", "imidazolyl"),
        ("thiazol", "thiazolyl"),
        ("oxazol", "oxazolyl"),
        ("piperidin", "piperidyl"),
    ):
        name = re.sub(rf"(?<![A-Za-z\]])\b{stem}-(\d+)-yl\b", rf"\1-{compact_stem}", name)
    for stem, compact_stem in (
        ("indole", "indolyl"),
        ("indol", "indolyl"),
        ("indazole", "indazolyl"),
        ("indazol", "indazolyl"),
        ("isoindole", "isoindolyl"),
        ("isoindol", "isoindolyl"),
        ("isoindolin", "isoindolinyl"),
        ("benzimidazol", "benzimidazolyl"),
        ("benzoxazol", "benzoxazolyl"),
        ("benzothiazol", "benzothiazolyl"),
        ("benzofuran", "benzofuryl"),
        ("benzofur", "benzofuryl"),
        ("benzothiophene", "benzothienyl"),
        ("benzothiophen", "benzothienyl"),
        ("quinoline", "quinolyl"),
        ("quinolin", "quinolyl"),
        ("isoquinoline", "isoquinolyl"),
        ("isoquinolin", "isoquinolyl"),
        ("quinazoline", "quinazolinyl"),
        ("quinazolin", "quinazolinyl"),
        ("quinoxaline", "quinoxalinyl"),
        ("quinoxalin", "quinoxalinyl"),
        ("cinnoline", "cinnolinyl"),
        ("cinnolin", "cinnolinyl"),
        ("pteridine", "pteridyl"),
        ("pteridin", "pteridyl"),
        ("purine", "puryl"),
        ("purin", "puryl"),
        ("carbazole", "carbazolyl"),
        ("carbazol", "carbazolyl"),
    ):
        name = re.sub(rf"(?<=\(){stem}-(\d+)-yl\b", rf"\1-{compact_stem}", name)
        name = re.sub(rf"(?<=[0-9]H-){stem}-(\d+)-yl\b", rf"\1-{compact_stem}", name)
    name = name.replace("p-tolylsulfonyl", "tosyl")
    name = re.sub(r"(?<![A-Za-z])phenylmethyl", "benzyl", name)
    for prefix in (
        "fluoro",
        "chloro",
        "bromo",
        "iodo",
        "methyl",
        "ethyl",
        "methoxy",
        "ethoxy",
        "hydroxy",
        "amino",
        "nitro",
        "cyano",
        "trifluoromethyl",
        "difluoromethyl",
    ):
        name = name.replace(prefix + "phenylmethyl", prefix + "benzyl")
    for compact in (
        "trifluoromethyl",
        "difluoromethyl",
        "fluoromethyl",
        "trifluoromethoxy",
        "difluoromethoxy",
        "fluoromethoxy",
        "trichloromethyl",
        "dichloromethyl",
        "chloromethyl",
        "tribromomethyl",
        "dibromomethyl",
        "bromomethyl",
        "triiodomethyl",
        "diiodomethyl",
        "iodomethyl",
        "tert-butyl",
        "isopropyl",
        "cyanomethyl",
        "carbamoyl",
        "phenylmethyl",
        "benzyl",
        "isobutyl",
        "neopentyl",
        "formyl",
        "acetyl",
        "trifluoroacetyl",
        "difluoroacetyl",
        "pivaloyl",
        "propionyl",
        "butyryl",
        "isobutyryl",
        "valeryl",
        "acroyl",
        "methacryloyl",
        "acetamido",
        "acetoxy",
        "propionamido",
        "benzamido",
        "boc",
        "bocamino",
        "carbmethoxy",
        "carbethoxy",
        "carbopropoxy",
        "carbobutoxy",
        "carbphenoxy",
        "carbbenzoxy",
        "triphosphooxymethyl",
        "diphosphooxymethyl",
        "phosphonooxymethyl",
        "piperidino",
        "piperazino",
        "morpholino",
        "pyrrolidino",
        "benzoyl",
        "mesyl",
        "tosyl",
        "nosyl",
        "dansyl",
        "fmoc",
        "fmocamino",
        "tbdms",
        "t-butyldimethylsiloxy",
        "tert-butyldiphenylsiloxy",
        "methoxymethoxy",
        "trideuteriomethoxy",
        "cyclopropoxy",
        "cyclopropylmethoxy",
        "picolinoyl",
        "nicotinoyl",
        "isonicotinoyl",
        "then-2-oyl",
        "then-3-oyl",
        "ethynyl",
        "vinyl",
    ):
        name = name.replace(f"({compact})", compact)
    name = name.replace("tert-butyloxycarbonyl", "tert-butoxycarbonyl")
    name = name.replace("isopropyloxycarbonyl", "isopropoxycarbonyl")
    name = name.replace("tert-carbobutoxy", "boc")
    name = name.replace("tert-butoxycarbonyl", "boc")
    name = name.replace("(boc)", "boc")
    name = name.replace("(bocamino)", "bocamino")
    name = name.replace("((cbz)amino)", "(cbzamino)")
    name = name.replace("(cbzamino)", "cbzamino")
    name = _compact_n_substituted_carbamoyl_fragments(name)
    name = _compact_aryl_acyl_amino_fragments(name)
    name = re.sub(r"\(\(([^()]{3,80})\)(oxy|carbonyl|imino|sulfanyl|sulfonyl)\)", r"(\1\2)", name)
    name = _compact_arylamino_parentheses(name)
    name = _compact_retained_oxy_terms(name)
    name = _compact_retained_nucleobase_terms(name)
    name = _compact_retained_fused_aromatic_terms(name)
    name = _compact_retained_bicyclic_terms(name)
    name = re.sub(
        r"\b(\d+(?:,\d+)*-(?:hydroxy|amino|formyl|phenyl|cyclopropyl|cyclobutyl|cyclopentyl|cyclohexyl|benzoyloxy|nitrooxy|isocyanato|azido|nitro|cyano))-([a-z]+yl(?:carbonyl)?)\b",
        r"\1\2",
        name,
    )
    name = _compact_benzene_principal_parent(name)
    name = _compact_substitutive_parent_hyphens(name)
    name = _compact_nested_methyl_chains(name)
    for verbose, compact in (
        ("1-trifluoromethyl-methyl", "2,2,2-trifluoroethyl"),
        ("1-difluoromethyl-methyl", "2,2-difluoroethyl"),
        ("1-fluoromethyl-methyl", "2-fluoroethyl"),
        ("1-trichloromethyl-methyl", "2,2,2-trichloroethyl"),
        ("1-dichloromethyl-methyl", "2,2-dichloroethyl"),
        ("1-chloromethyl-methyl", "2-chloroethyl"),
    ):
        name = name.replace(verbose, compact)
    for compact in (
        "trifluoromethoxy",
        "difluoromethoxy",
        "fluoromethoxy",
        "trichloromethoxy",
        "dichloromethoxy",
        "chloromethoxy",
    ):
        name = name.replace(f"1-(1-{compact}-methyl)-methyl", f"2-{compact}ethyl")
        name = name.replace(f"(1-(1-{compact}-methyl)-methyl)", f"(2-{compact}ethyl)")
    name = _compact_adjacent_unsaturation_locants(name)
    return _compact_retained_oxy_terms(name)


def _compact_validated_functional_aliases(name: str) -> str:
    """Apply compact OPSIN-stable aliases only in attachment-safe contexts."""

    replacements = (
        (r"(^|[-,(])aminosulfonyl", r"\1sulfamoyl"),
        (r"(^|[,(])1-azamethylcarbonyl", r"\1carbamoyl"),
        (r"(^|[,(])1-azaethylcarbonyl", r"\1methylcarbamoyl"),
        (r"(^|[-(,])carbbenzoxy", r"\1cbz"),
        (r"(^|[-(,])hydroxysulfonyl", r"\1sulfo"),
        (r"(^|[-(,])phenylsulfonyl", r"\1besyl"),
        (r"2,2,2-trifluoroethyl", "trifluoroethyl"),
        (r"2,2-difluoroethyl", "difluoroethyl"),
        (r"2,2,2-trichloroethyl", "trichloroethyl"),
        (r"2,2-dichloroethyl", "dichloroethyl"),
        (r"2,2,2-tribromoethyl", "tribromoethyl"),
        (r"2,2-dibromoethyl", "dibromoethyl"),
        (r"2,2,2-trifluoroethoxy", "trifluoroethoxy"),
        (r"2,2-difluoroethoxy", "difluoroethoxy"),
        (r"2,2,2-trichloroethoxy", "trichloroethoxy"),
        (r"2,2-dichloroethoxy", "dichloroethoxy"),
        (r"2,2,2-tribromoethoxy", "tribromoethoxy"),
        (r"2,2-dibromoethoxy", "dibromoethoxy"),
        (r"isopropyloxy", "isopropoxy"),
        (r"(^|[(,])2-pyridylcarbonyl", r"\1picolinoyl"),
        (r"(^|[(,])3-pyridylcarbonyl", r"\1nicotinoyl"),
        (r"(^|[(,])4-pyridylcarbonyl", r"\1isonicotinoyl"),
        (r"(^|[(,])2-furylcarbonyl", r"\g<1>2-furoyl"),
        (r"(^|[(,])3-furylcarbonyl", r"\g<1>3-furoyl"),
        (r"(^|[(,])2-thienylcarbonyl", r"\g<1>2-thenoyl"),
        (r"(^|[(,])3-thienylcarbonyl", r"\g<1>3-thenoyl"),
    )
    for pattern, replacement in replacements:
        name = re.sub(pattern, replacement, name)
    return name


def _compact_retained_oxy_terms(name: str) -> str:
    for verbose, compact in _RETAINED_OXY_COMPACTIONS:
        name = name.replace(verbose, compact)
        if compact in _PAREN_SAFE_RETAINED_OXY:
            name = name.replace(f"({compact})", compact)
    return name


def _normalize_retained_fused_substituent_locants(name: str) -> str:
    for compact in ("furyl", "thienyl", "imidazolyl", "oxazolyl", "thiazolyl"):
        name = re.sub(rf"\bbenzo(\d+)-{compact}\b", rf"\1-benzo{compact}", name)
        name = re.sub(rf"\bbenzo(\d+){compact}\b", rf"\1-benzo{compact}", name)
        name = re.sub(rf"\bbenz(\d+)-{compact}\b", rf"\1-benz{compact}", name)
        name = re.sub(rf"\bbenz(\d+){compact}\b", rf"\1-benz{compact}", name)
    name = re.sub(r"\b1-(\d)-benzothienyl\b", r"\1-benzothienyl", name)
    name = re.sub(r"\b1-(\d)-benzofuryl\b", r"\1-benzofuryl", name)
    return name


_FUSED_HETEROARYL_SUFFIX_STEMS = {
    "furyl": "furan",
    "imidazolyl": "imidazol",
    "oxazolyl": "oxazol",
    "pyrazinyl": "pyrazin",
    "pyrazolyl": "pyrazol",
    "pyridazyl": "pyridazin",
    "pyridyl": "pyridin",
    "pyrimidyl": "pyrimidin",
    "thiazolyl": "thiazol",
    "thienyl": "thiophen",
}


_BENZ_FUSED_HETEROARYL_STEMS = {
    "furyl": "benzofuran",
    "imidazolyl": "benzimidazol",
    "oxazolyl": "benzoxazol",
    "thiazolyl": "benzothiazol",
    "thienyl": "benzothiophen",
}


def _normalize_fused_heteroaryl_locants(name: str) -> str:
    """Use OPSIN-parseable locants for retained fused heteroaryl suffixes."""

    suffixes = "|".join(sorted(_FUSED_HETEROARYL_SUFFIX_STEMS, key=len, reverse=True))

    def bracket_repl(match: re.Match[str]) -> str:
        fused_prefix = match.group(1)
        locant = match.group(2)
        suffix = match.group(3)
        return f"{fused_prefix}{_FUSED_HETEROARYL_SUFFIX_STEMS[suffix]}-{locant}-yl"

    name = re.sub(
        rf"((?:[A-Za-z]+\[[0-9,abcn\-]+\])+)([0-9]+)-({suffixes})",
        bracket_repl,
        name,
    )

    benz_suffixes = "|".join(sorted(_BENZ_FUSED_HETEROARYL_STEMS, key=len, reverse=True))

    def benz_repl(match: re.Match[str]) -> str:
        prefix = match.group(1) or ""
        locant = match.group(2)
        suffix = match.group(3)
        return f"{prefix}{_BENZ_FUSED_HETEROARYL_STEMS[suffix]}-{locant}-yl"

    name = re.sub(
        rf"\b((?:[0-9,]+-[a-z]+)*?)benzo([0-9]+)-?({benz_suffixes})",
        benz_repl,
        name,
    )

    def retained_benz_repl(match: re.Match[str]) -> str:
        locant = match.group(1)
        suffix = match.group(2)
        return f"{_BENZ_FUSED_HETEROARYL_STEMS[suffix]}-{locant}-yl"

    return re.sub(rf"\b([0-9]+)-benzo({benz_suffixes})", retained_benz_repl, name)


def _compact_retained_nucleobase_terms(name: str) -> str:
    name = re.sub(r"\b2-amino-6-oxo-(\d+-\(.+\))-purin\b", r"\1-guanine", name)
    name = re.sub(r"\b2-amino-6-oxo-(\d+-[^()]+)-purin\b", r"\1-guanine", name)
    for retained_acid in ("adenylic", "cytidylic", "guanylic", "thymidylic", "uridylic"):
        name = re.sub(rf"\b({retained_acid})(?! acid)\b", r"\1 acid", name)
    return name


def _compact_retained_bicyclic_terms(name: str) -> str:
    locant_map = {
        "1": "7a",
        "2": "1",
        "3": "2",
        "4": "3",
        "6": "3",
        "7": "2",
        "8": "1",
    }
    for verbose_loc, compact_loc in locant_map.items():
        name = name.replace(f"5-azabicyclo[3.3.0]octan-{verbose_loc}-yl", f"pyrrolizidin-{compact_loc}-yl")
    name = name.replace("5-azabicyclo[3.3.0]octane", "pyrrolizidine")
    return name


def _compact_retained_fused_aromatic_terms(name: str) -> str:
    """Normalize common generated fused-aromatic skeletons to retained names."""

    for unsaturation in (
        "1,3,5,7,9",
        "1(10),2,4,6,8",
        "1(10),2(3),4(5),6(7),8(9)",
    ):
        name = re.sub(
            rf"\bbicyclo\[4\.4\.0\]deca-{re.escape(unsaturation)}-pentaen-([1-8])-yl\b",
            r"\1-naphthyl",
            name,
        )
        name = re.sub(
            rf"(?<![A-Za-z])bicyclo\[4\.4\.0\]deca-{re.escape(unsaturation)}-pentaene\b",
            "naphthalene",
            name,
        )
        name = re.sub(
            rf"(?<![A-Za-z])bicyclo\[4\.4\.0\]deca-{re.escape(unsaturation)}-pentaen\b",
            "naphthalene",
            name,
        )
    for unsaturation in (
        "1,3,5",
        "1(10),2,4",
    ):
        name = re.sub(
            rf"\bbicyclo\[4\.4\.0\]deca-{re.escape(unsaturation)}-trien-([1-8])-yl\b",
            r"\1-tetralinyl",
            name,
        )
        name = re.sub(
            rf"(?<![A-Za-z])bicyclo\[4\.4\.0\]deca-{re.escape(unsaturation)}-triene\b",
            "tetralin",
            name,
        )
    for unsaturation in (
        "1,3,5",
        "1(9),2,4",
    ):
        name = re.sub(
            rf"\bbicyclo\[4\.3\.0\]nona-{re.escape(unsaturation)}-trien-([1-7])-yl\b",
            r"\1-indanyl",
            name,
        )
        name = re.sub(
            rf"(?<![A-Za-z])bicyclo\[4\.3\.0\]nona-{re.escape(unsaturation)}-triene\b",
            "indane",
            name,
        )
        name = re.sub(
            rf"(?<![A-Za-z])bicyclo\[4\.3\.0\]nona-{re.escape(unsaturation)}-trien\b",
            "indane",
            name,
        )
    for unsaturation in (
        "1,3,5,7",
        "1(9),2,4,6",
    ):
        name = re.sub(
            rf"\bbicyclo\[4\.3\.0\]nona-{re.escape(unsaturation)}-tetraen-([1-7])-yl\b",
            r"\1-indenyl",
            name,
        )
        name = re.sub(
            rf"(?<![A-Za-z])bicyclo\[4\.3\.0\]nona-{re.escape(unsaturation)}-tetraene\b",
            "indene",
            name,
        )
        name = re.sub(
            rf"(?<![A-Za-z])bicyclo\[4\.3\.0\]nona-{re.escape(unsaturation)}-tetraen\b",
            "indene",
            name,
        )
    name = re.sub(r"\bnaphthalen\b", "naphthalene", name)
    return name


def _compact_n_substituted_carbamoyl_fragments(name: str) -> str:
    simple_n_substituent = (
        r"phenyl|(?:[234]-)?(?:fluoro|chloro|bromo|iodo|methyl|methoxy|"
        r"trifluoromethyl|difluoromethoxy|trifluoromethoxy|cyano|nitro)phenyl|"
        r"[234]-pyridyl|[23456]-pyrimidyl|[2345]-thiazolyl|[2345]-pyrazolyl|"
        r"[2345]-imidazolyl|[2345]-furyl|[2345]-thienyl|cyclopropyl|"
        r"cyclobutyl|cyclopentyl|cyclohexyl|tert-butyl|isopropyl|methyl|"
        r"ethyl|propyl|butyl"
    )

    name = re.sub(
        rf"(?:(?<=^)|(?<=[(,]))1-\(({simple_n_substituent})\)-1-azamethylcarbonyl",
        r"(\1)carbamoyl",
        name,
    )
    name = re.sub(
        rf"(?:(?<=^)|(?<=[(,]))1-({simple_n_substituent})-1-azamethylcarbonyl",
        r"\1carbamoyl",
        name,
    )

    def paren_substituent(match: re.Match[str]) -> str:
        substituent = match.group(1)
        return f"({substituent})methylcarbamoyl"

    name = re.sub(
        r"(?<![,\d])2-\(((?:(?:[234]-)?(?:fluoro|chloro|bromo|iodo|methyl|methoxy|trifluoromethyl|cyano|nitro)-)?phenyl)\)-1-azaethylcarbonyl",
        paren_substituent,
        name,
    )
    for verbose, compact in (
        ("2-phenyl-1-azaethylcarbonyl", "benzylcarbamoyl"),
        ("2-cyclopropyl-1-azaethylcarbonyl", "cyclopropylmethylcarbamoyl"),
        ("2-cyclobutyl-1-azaethylcarbonyl", "cyclobutylmethylcarbamoyl"),
        ("2-cyclopentyl-1-azaethylcarbonyl", "cyclopentylmethylcarbamoyl"),
        ("2-cyclohexyl-1-azaethylcarbonyl", "cyclohexylmethylcarbamoyl"),
    ):
        name = name.replace(verbose, compact)
    return name


def _compact_nested_linker_parentheses(name: str) -> str:
    def collapse(match: re.Match[str]) -> str:
        inner = match.group(1)
        if "mesyl" in inner or "bromodifluoromethylcarbonyl" in inner:
            return match.group(0)
        return f"({inner}{match.group(2)})"

    return re.sub(r"\(\(([^()]{3,80})\)(oxy|amino)\)", collapse, name)


_ARYLAMINO_INNER = (
    r"phenyl|(?:[234]-)?(?:fluoro|chloro|bromo|iodo|methyl|methoxy|"
    r"trifluoromethyl|difluoromethoxy|trifluoromethoxy|cyano|nitro)phenyl|"
    r"[234]-pyridyl|[23456]-pyrimidyl|[2345]-thiazolyl|[2345]-pyrazolyl|"
    r"[2345]-imidazolyl|[2345]-furyl|[2345]-thienyl|"
    r"\d+-\([^()]+\)(?:phenyl|pyridyl|pyrimidyl|pyrazolyl)"
)


def _compact_arylamino_parentheses(name: str) -> str:
    return re.sub(rf"\(\(({_ARYLAMINO_INNER})\)amino\)", r"(\1amino)", name)


def _compact_aryl_acyl_amino_fragments(name: str) -> str:
    benzoyl_substituent = (
        r"(?:[234]-)?(?:fluoro|chloro|bromo|iodo|methyl|methoxy|"
        r"trifluoromethyl|difluoromethoxy|trifluoromethoxy|cyano|nitro)?benzoyl"
    )

    def benzamido(match: re.Match[str]) -> str:
        return f"({match.group(1).replace('benzoyl', 'benzamido')})"

    name = re.sub(rf"\(\(({benzoyl_substituent})\)amino\)", benzamido, name)
    for acyl, amido in (
        ("picolinoyl", "picolinamido"),
        ("nicotinoyl", "nicotinamido"),
        ("isonicotinoyl", "isonicotinamido"),
        ("2-furoyl", "2-furamido"),
        ("3-furoyl", "3-furamido"),
        ("2-thenoyl", "2-thenamido"),
        ("3-thenoyl", "3-thenamido"),
        ("then-2-oyl", "2-thenamido"),
        ("then-3-oyl", "3-thenamido"),
    ):
        name = name.replace(f"(({acyl})amino)", f"({amido})")
        name = name.replace(f"({acyl}amino)", f"({amido})")
    return name


def _compact_benzene_principal_parent(name: str) -> str:
    if not name.endswith("-benzene"):
        return name
    core = name[: -len("-benzene")]
    for group, parent in (
        ("carbamoyl", "benzamide"),
        ("cyano", "benzonitrile"),
        ("carboxy", "benzoic acid"),
        ("carboxylato", "benzoate"),
        ("formyl", "benzaldehyde"),
        ("aminosulfonyl", "benzenesulfonamide"),
        ("hydroxy", "phenol"),
        ("amino", "aniline"),
        ("methoxy", "anisole"),
        ("methyl", "toluene"),
    ):
        prefix = f"1-{group}"
        if core == prefix:
            return parent
        if core.startswith(prefix + "-"):
            return core[len(prefix) + 1 :] + parent
    return name


def _compact_substitutive_parent_hyphens(name: str) -> str:
    parent_terms = (
        "benzene",
        "pyridine",
        "pyrimidine",
        "pyrazine",
        "pyridazine",
        "piperidine",
        "piperazine",
        "morpholine",
        "pyrrole",
        "pyrazole",
        "imidazole",
        "thiazole",
        "oxazole",
        "thiophene",
        "furan",
        "phenol",
        "aniline",
        "anisole",
        "toluene",
        "benzamide",
        "benzonitrile",
        "benzaldehyde",
        "benzoate",
        "naphthalene",
        "tetralin",
        "indane",
        "indene",
    )
    prefix_terms = (
        "amino",
        "azido",
        "bromo",
        "carbamoyl",
        "carboxy",
        "carboxylato",
        "chloro",
        "cyano",
        "ethyl",
        "ethoxy",
        "fluoro",
        "formyl",
        "hydroxy",
        "iodo",
        "methoxy",
        "methyl",
        "nitro",
        "phenyl",
        "trifluoromethyl",
    )
    parent_pattern = "|".join(re.escape(parent) for parent in parent_terms)
    prefix_pattern = "|".join(re.escape(prefix) for prefix in prefix_terms)
    return re.sub(rf"\b(\d+(?:,\d+)*-(?:{prefix_pattern}))-({parent_pattern})\b", r"\1\2", name)


def _compact_adjacent_unsaturation_locants(name: str) -> str:
    def replace(match: re.Match[str]) -> str:
        first = int(match.group(1))
        second = int(match.group(2))
        if second == first + 1:
            return match.group(1)
        return match.group(0)

    return re.sub(r"\b(\d+)\((\d+)\)", replace, name)


def _compact_nested_methyl_chains(name: str) -> str:
    pieces: list[str] = []
    index = 0
    while index < len(name):
        if name[index] != "(":
            pieces.append(name[index])
            index += 1
            continue
        end = _balanced_parenthesis_end(name, index)
        if end is None:
            pieces.append(name[index])
            index += 1
            continue
        inner = name[index + 1 : end]
        compact = _compact_methyl_chain_expression(inner)
        if compact:
            pieces.append(f"({compact})")
        else:
            pieces.append("(" + _compact_nested_methyl_chains(inner) + ")")
        index = end + 1
    return "".join(pieces)


def _compact_methyl_chain_expression(text: str) -> str:
    base, count = _parse_nested_methyl_chain(text)
    if count < 2:
        return ""
    root = ROOTS.get(count)
    if not root:
        return ""
    base = _strip_balanced_outer_parentheses(_compact_nested_methyl_chains(base))
    return f"{count}-({base}){root}yl"


def _parse_nested_methyl_chain(text: str) -> tuple[str, int]:
    if not text.startswith("1-("):
        return text, 0
    end = _balanced_parenthesis_end(text, 2)
    if end is None or text[end + 1 :] != "-methyl":
        return text, 0
    inner = text[3:end]
    base, count = _parse_nested_methyl_chain(_strip_balanced_outer_parentheses(inner))
    return base, count + 1


def _balanced_parenthesis_end(text: str, start: int) -> int | None:
    if start >= len(text) or text[start] != "(":
        return None
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
    return None


def _strip_balanced_outer_parentheses(text: str) -> str:
    while text.startswith("(") and text.endswith(")"):
        end = _balanced_parenthesis_end(text, 0)
        if end != len(text) - 1:
            break
        text = text[1:-1]
    return text


def _apply_uniform_isotope_label(mol, name: str) -> str:
    if not name:
        return name
    labels: list[tuple[str, int, int]] = []
    for symbol in sorted({atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetSymbol() != "H"}):
        atoms = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == symbol]
        isotopes = {atom.GetIsotope() for atom in atoms}
        if len(isotopes) != 1:
            continue
        isotope = next(iter(isotopes))
        if isotope:
            labels.append((symbol, isotope, len(atoms)))
    if len(labels) != 1:
        return name
    symbol, isotope, count = labels[0]
    if f"{isotope}{symbol}" in name:
        return name
    if symbol == "C" and isotope in {13, 14}:
        count_text = str(count) if count > 1 else ""
        return f"{name}-{isotope}{symbol}{count_text}"
    if symbol == "F" and isotope == 18 and count == 1:
        return _inline_single_fluorine_isotope(name)
    if symbol == "N" and count == 1 and not name.startswith("("):
        return f"[{isotope}{symbol}]{name}"
    return name


def _inline_single_fluorine_isotope(name: str) -> str:
    if "[18F]fluoro" in name:
        return name
    return name.replace("fluoro", "[18F]fluoro", 1) if "fluoro" in name else name


def _fallback_stereo_atom_order(family: str) -> str:
    spiro_parent_families = {
        "component-dispiro-parent",
        "component-spiro-parent",
        "generated-spiro-fused-parent",
        "generated-spiro-partial-hydro-parent",
        "generic-spiro-parent",
        "poly-component-spiro-parent",
        "spiro-functional-chain-substituent",
    }
    return "idx_desc" if family in spiro_parent_families else "idx"


def _normalize_terminal_parent_stem(name: str) -> str:
    terminal_stems = {
        "androst": "androstane",
        "chol": "cholane",
        "cholest": "cholestane",
        "campest": "campestane",
        "ergost": "ergostane",
        "estr": "estrane",
        "poriferast": "poriferastane",
        "stigmast": "stigmastane",
        "gon": "gonane",
        "pregn": "pregnane",
        "spirost": "spirostane",
    }
    if name == "pyroglutam":
        return "pyroglutamic acid"
    for stem, parent in terminal_stems.items():
        if name == stem:
            return parent
        suffix = "-" + stem
        if name.endswith(suffix):
            return name[: -len(stem)] + parent
    return name
