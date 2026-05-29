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
                candidate_key = (
                    len(name),
                    _opsin_stability_penalty(name),
                    candidate.priority,
                    _comparable_sort_key(candidate.sort_key),
                    name,
                )
                if best_key is None or candidate_key < best_key:
                    best_name = name
                    best_key = candidate_key
        return best_name


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
        return _apply_uniform_isotope_label(mol, _compact_common_substituent_terms(f"({candidate.stereo_prefix})-{name}"))
    if candidate.family in {"whole-molecule-retained-parent", "natural-product-topology-parent"}:
        return _apply_uniform_isotope_label(mol, _compact_common_substituent_terms(name))
    return _apply_uniform_isotope_label(
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
        ("eth-1-ene-1-yl", "vinyl"),
    )
    for verbose, compact in rewrites:
        name = name.replace(verbose, compact)
    name = re.sub(r"\bbenzo(\d+)-furyl\b", r"\1-benzofuryl", name)
    name = re.sub(r"\bbenzo(\d+)-thienyl\b", r"\1-benzothienyl", name)
    name = re.sub(r"\bbenz(\d+)-(imidazolyl|oxazolyl|thiazolyl)\b", r"\1-benz\2", name)
    for verbose, compact in (
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
    name = _compact_benzene_principal_parent(name)
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
    name = _compact_adjacent_unsaturation_locants(name)
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
        "poriferast": "poriferastane",
        "stigmast": "stigmastane",
        "gon": "gonane",
        "pregn": "pregnane",
    }
    for stem, parent in terminal_stems.items():
        if name == stem:
            return parent
        suffix = "-" + stem
        if name.endswith(suffix):
            return name[: -len(stem)] + parent
    return name
