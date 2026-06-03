from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
import re
import time

from .candidate_facts import (
    CandidateSafety,
    ChargeEvidence,
    HydrogenEvidence,
    PrefixCandidate,
    StereoEvidence,
    SuffixCandidate,
)
from .fragment_rules import (
    FragmentAssemblyRule,
    common_polyatomic_fragment_name,
    simple_fragment_name,
    standalone_polyatomic_fragment_name,
)
from .naming_context import MoleculeContext
from .molecule_cache import mol_cache_get, mol_cache_set
from .rule_dispatch import RuleSpec
from .stereochemistry import apply_stereochemical_prefix
from .telemetry import CandidateTelemetry


_DEFAULT_MAX_NAME_LENGTH = 180
_STEREO_EMPTY_FALLBACK_EVALUATION_LENGTH = 320
_STEREO_EMPTY_FALLBACK_MAX_LENGTH = 320
_NONSTEREO_EMPTY_FALLBACK_EVALUATION_LENGTH = 320
_NONSTEREO_EMPTY_FALLBACK_MAX_LENGTH = 320


@dataclass(frozen=True, slots=True)
class NameCandidate:
    """A deterministic name proposed by one rule family."""

    name: str
    family: str
    priority: int
    sort_key: tuple = ()
    stereo_prefix: str = ""
    parent_atoms: frozenset[int] = field(default_factory=frozenset)
    locants: tuple[tuple[int, str], ...] = ()
    template_id: int | None = None
    template_name: str = ""
    source_family: str = ""
    source_file: str = ""
    ring_system_atoms: frozenset[int] = field(default_factory=frozenset)
    prefixes: tuple[PrefixCandidate, ...] = ()
    suffixes: tuple[SuffixCandidate, ...] = ()
    charge: ChargeEvidence | None = None
    hydrogen: HydrogenEvidence | None = None
    stereo: StereoEvidence | None = None
    safety: CandidateSafety | None = None
    estimated_length: int | None = None
    render_cost: int = 1
    render_rank: int | None = None
    rejection_reason: str = ""


@dataclass(frozen=True, slots=True)
class _PreparedCandidate:
    name: str
    candidate: NameCandidate
    producer_family: str
    stability_penalty: int
    selection_bonus: int


class NamingEngine:
    """OPSIN-inspired dispatcher over independent nomenclature rule families."""

    def __init__(self, Chem, rules: Iterable[RuleSpec]):
        self._Chem = Chem
        self._rules = tuple(sorted(rules, key=lambda rule: rule.priority))
        self._fragment_rule = FragmentAssemblyRule(self._Chem, self)

    def name(self, mol, *, telemetry: CandidateTelemetry | None = None) -> str:
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

        ctx = MoleculeContext(mol, self._Chem, telemetry=telemetry)
        best_name = _select_best_prepared_candidate(
            _prepare_candidates(mol, self._Chem, ctx, self._rules),
            telemetry=telemetry,
        )

        if best_name:
            return best_name

        fallback_evaluation_length, fallback_max_length = _empty_fallback_limits(mol, self._Chem)
        fallback_name = _select_best_prepared_candidate(
            _prepare_candidates(
                mol,
                self._Chem,
                ctx,
                self._rules,
                max_length=fallback_evaluation_length,
                output_max_length=fallback_max_length,
            ),
            telemetry=telemetry,
        )
        return fallback_name


def _prepare_candidates(
    mol,
    Chem,
    ctx: MoleculeContext,
    rules: Iterable[RuleSpec],
    *,
    max_length: int = _DEFAULT_MAX_NAME_LENGTH,
    output_max_length: int | None = None,
) -> list[_PreparedCandidate]:
    prepared: list[_PreparedCandidate] = []
    for rule in rules:
        if not rule.applies(ctx):
            continue
        if ctx.telemetry is None:
            for candidate in rule.candidates(ctx):
                if not candidate.name:
                    continue
                name = _finalize_candidate_name(mol, Chem, candidate, max_length=max_length)
                if not name:
                    continue
                if output_max_length is not None and len(name) > output_max_length:
                    continue
                prepared.append(
                    _PreparedCandidate(
                        name=name,
                        candidate=candidate,
                        producer_family=rule.family,
                        stability_penalty=_candidate_selection_stability_penalty(name, candidate),
                        selection_bonus=_candidate_selection_bonus(name, candidate),
                    )
                )
            continue
        ctx.telemetry.rule_applies[rule.family] += 1
        started = time.perf_counter()
        previous_rule_family = ctx.telemetry.active_rule_family
        ctx.telemetry.active_rule_family = rule.family
        try:
            for candidate in rule.candidates(ctx):
                ctx.telemetry.candidate_count[rule.family] += 1
                _record_candidate_fact_telemetry(ctx.telemetry, rule.family, candidate)
                if not candidate.name:
                    ctx.telemetry.rejected_candidate_count[rule.family] += 1
                    continue
                name = _finalize_candidate_name(mol, Chem, candidate, max_length=max_length)
                if not name:
                    ctx.telemetry.rejected_candidate_count[rule.family] += 1
                    continue
                if output_max_length is not None and len(name) > output_max_length:
                    ctx.telemetry.rejected_candidate_count[rule.family] += 1
                    continue
                ctx.telemetry.rendered_candidate_count[rule.family] += 1
                prepared.append(
                    _PreparedCandidate(
                        name=name,
                        candidate=candidate,
                        producer_family=rule.family,
                        stability_penalty=_candidate_selection_stability_penalty(name, candidate),
                        selection_bonus=_candidate_selection_bonus(name, candidate),
                    )
                )
        finally:
            ctx.telemetry.active_rule_family = previous_rule_family
            ctx.telemetry.rule_seconds[rule.family] += time.perf_counter() - started
    return prepared


def _select_best_prepared_candidate(
    prepared: list[_PreparedCandidate],
    *,
    telemetry: CandidateTelemetry | None = None,
) -> str:
    best_name = ""
    best_key: tuple | None = None
    for item in prepared:
        candidate = item.candidate
        stability_penalty = item.stability_penalty + _candidate_selection_contextual_penalty(item, prepared)
        candidate_key = (
            len(item.name) + stability_penalty - item.selection_bonus,
            len(item.name),
            stability_penalty,
            candidate.priority,
            _comparable_sort_key(candidate.sort_key),
            item.name,
        )
        if best_key is None or candidate_key < best_key:
            best_name = item.name
            best_key = candidate_key
    if best_name and telemetry is not None:
        for item in prepared:
            if item.name == best_name:
                telemetry.selected_family[item.candidate.family] += 1
                telemetry.selected_rule_family[item.producer_family] += 1
                if _candidate_has_structured_facts(item.candidate):
                    telemetry.structured_selected_count[item.candidate.family] += 1
                if item.candidate.template_id is not None:
                    telemetry.selected_with_template_id += 1
                if item.candidate.parent_atoms:
                    telemetry.selected_with_parent_atoms += 1
                if item.candidate.render_rank is not None:
                    telemetry.selected_render_rank_count[item.producer_family] += 1
                    telemetry.selected_render_rank_total[item.producer_family] += item.candidate.render_rank
                    telemetry.selected_render_rank_bucket[
                        f"{item.producer_family}:{_render_rank_bucket(item.candidate.render_rank)}"
                    ] += 1
                telemetry.selected_safety_reasons.update(candidate_safety_reasons(item.candidate))
                break
    return best_name


def _candidate_has_structured_facts(candidate: NameCandidate) -> bool:
    return bool(
        candidate.parent_atoms
        or candidate.locants
        or candidate.template_id is not None
        or candidate.template_name
        or candidate.source_family
        or candidate.source_file
        or candidate.ring_system_atoms
        or candidate.prefixes
        or candidate.suffixes
        or candidate.charge is not None
        or candidate.hydrogen is not None
        or candidate.stereo is not None
        or candidate.safety is not None
        or candidate.estimated_length is not None
        or candidate.render_cost != 1
        or candidate.render_rank is not None
        or candidate.rejection_reason
    )


def _render_rank_bucket(rank: int) -> str:
    if rank <= 4:
        return str(rank)
    if rank <= 8:
        return "5-8"
    if rank <= 16:
        return "9-16"
    if rank <= 32:
        return "17-32"
    return "33+"


def _record_candidate_fact_telemetry(
    telemetry: CandidateTelemetry,
    rule_family: str,
    candidate: NameCandidate,
) -> None:
    if _candidate_has_structured_facts(candidate):
        telemetry.structured_candidate_count[rule_family] += 1
    else:
        telemetry.unstructured_candidate_count[rule_family] += 1


def _empty_fallback_limits(mol, Chem) -> tuple[int, int]:
    if _has_assigned_stereo(mol, Chem):
        return (
            _STEREO_EMPTY_FALLBACK_EVALUATION_LENGTH,
            _STEREO_EMPTY_FALLBACK_MAX_LENGTH,
        )
    return (
        _NONSTEREO_EMPTY_FALLBACK_EVALUATION_LENGTH,
        _NONSTEREO_EMPTY_FALLBACK_MAX_LENGTH,
    )


def _has_assigned_stereo(mol, Chem) -> bool:
    cached = mol_cache_get(mol, "has_assigned_stereo")
    if cached is not None:
        return cached
    for atom in mol.GetAtoms():
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            return mol_cache_set(mol, "has_assigned_stereo", True)
    for bond in mol.GetBonds():
        if bond.GetStereo() != Chem.BondStereo.STEREONONE:
            return mol_cache_set(mol, "has_assigned_stereo", True)
    return mol_cache_set(mol, "has_assigned_stereo", False)


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


def candidate_parent_atoms(candidate: NameCandidate) -> frozenset[int]:
    return candidate.parent_atoms


def candidate_source_key(candidate: NameCandidate) -> tuple[str, int | None]:
    return (candidate.source_family or candidate.family, candidate.template_id)


def candidate_safety_reasons(candidate: NameCandidate) -> tuple[str, ...]:
    return candidate.safety.reasons if candidate.safety is not None else ()


def candidate_estimated_length(candidate: NameCandidate) -> int:
    if candidate.estimated_length is not None:
        return candidate.estimated_length
    return len(candidate.name)


def candidate_has_complete_graph(candidate: NameCandidate) -> bool:
    if candidate.safety is not None:
        return candidate.safety.graph_complete
    return bool(candidate.parent_atoms or candidate.template_id is not None)


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
    if _has_fragile_bicyclic_carbamate_lactam_name(name):
        penalty += 35
    if _has_fragile_imido_oxycarbonyl_alias(name):
        penalty += 35
    if "2-methyl-2-((" in name and "carbamyloxy)propan" in name:
        penalty += 80
    if "alpha-fmocamino-alpha-" in name:
        penalty += 80
    if "pyrrolidin" in name and "carboxy" in name and any(term in name for term in ("glycyl", "alanyl", "arginyl", "aspartyl", "glutamyl")):
        penalty += 220
    if "aminosulfonamido" in name and "N-(" in name:
        penalty += 80
    penalty += _unstable_aza_polycycle_alias_penalty(name)
    if _has_unstable_benzimidazolyl_orientation_artifact(name):
        penalty += 18
    if _has_unstable_benz_a_anthracene_substituent_artifact(name):
        penalty += 90
    if _has_unstable_benzoanthracene_candidate(name):
        penalty += 160
    return penalty


def _has_unstable_benz_a_anthracene_substituent_artifact(name: str) -> bool:
    marker = "benz[a]anthracene-"
    marker_index = name.find(marker)
    if marker_index < 0 or "-yl" not in name[marker_index:]:
        return False
    before = name[:marker_index]
    return before not in {"", "-"} and not before.endswith("(")


def _has_unstable_benzoanthracene_candidate(name: str) -> bool:
    if "benzoanthracene" not in name:
        return False
    if name.startswith("("):
        return True
    if "6a-" in name or ",6a" in name:
        return True
    if "12b-" in name or ",12b" in name:
        return True
    if "acetyl-6-oxabenzoanthracene" in name:
        return True
    if "dioxo" in name and "triaza" in name:
        return True
    if "pentaaza" in name:
        return True
    return False


def _candidate_selection_stability_penalty(name: str, candidate: NameCandidate) -> int:
    penalty = _opsin_stability_penalty(name)
    if candidate.family == "skeletal-replacement-fused-parent":
        penalty += _skeletal_replacement_fused_parent_penalty(name)
    penalty += _unstable_component_spiro_alias_penalty(name, candidate)
    penalty += _unlocanted_multicenter_stereo_penalty(name, candidate)
    return penalty


_CONTEXTUAL_GRAPH_PARENT_FAMILIES = {
    "component-spiro-parent",
    "fused-retained-parent",
    "generated-fused-parent",
    "generated-hydro-fused-parent",
    "generated-partial-hydro-fused-parent",
    "generated-spiro-fused-parent",
    "generated-spiro-partial-hydro-parent",
    "generic-bicyclic-parent",
    "generic-spiro-parent",
    "hydro-fused-retained-parent",
    "kekulized-generic-polycyclic-parent",
    "single-ring-parent",
}


def _candidate_selection_contextual_penalty(item: _PreparedCandidate, prepared: list[_PreparedCandidate]) -> int:
    """Defer orientation-fragile aliases only when alternatives exist."""

    name = item.name
    unstable_benzimidazolyl = _has_unstable_benzimidazolyl_orientation_artifact(name)
    unstable_replacement = _has_unstable_replacement_polycycle_alias(name)
    generic_polycycle = _has_contextual_generic_polycycle_fallback(name, item.candidate)
    fragile_sulfamoyl_benzene = _has_fragile_single_ring_sulfamoyl_benzene(name, item.candidate)
    fragile_fused_substituent_benzene = _has_fragile_single_ring_fused_substituent_benzene(name, item.candidate)
    missing_relative_stereo = _has_missing_relative_stereo_ring_candidate(name, item.candidate)
    fragile_bridge_h_ketone = _has_fragile_bridge_indicated_h_ketone_suffix(name)
    aza_alias_penalty = _unstable_aza_polycycle_alias_penalty(name)
    spiro_penalty = _unstable_component_spiro_alias_penalty(name, item.candidate)
    if not (
        unstable_benzimidazolyl
        or unstable_replacement
        or generic_polycycle
        or fragile_sulfamoyl_benzene
        or fragile_fused_substituent_benzene
        or missing_relative_stereo
        or fragile_bridge_h_ketone
        or aza_alias_penalty
        or spiro_penalty
    ):
        return 0

    penalty = 0
    has_graph_parent = _has_contextual_graph_parent_alternative(item, prepared)
    if has_graph_parent and unstable_benzimidazolyl:
        penalty += 24
    if has_graph_parent and unstable_replacement:
        penalty += 32
    if generic_polycycle and _has_contextual_named_graph_parent_alternative(item, prepared, max_extra=45, max_length=155):
        penalty += 28
    if fragile_sulfamoyl_benzene and _has_contextual_acyclic_parent_alternative(
        item, prepared, max_extra=25, max_length=170
    ):
        penalty += 36
    if fragile_fused_substituent_benzene and _has_contextual_named_graph_parent_alternative(
        item, prepared, max_extra=35, max_length=140
    ):
        penalty += 36
    if fragile_bridge_h_ketone and _has_contextual_oxo_bridge_parent_alternative(item, prepared):
        penalty += 80
    if missing_relative_stereo and _has_contextual_relative_stereo_alternative(item, prepared, max_extra=35):
        penalty += 36
    if aza_alias_penalty and _has_contextual_graph_parent_alternative(item, prepared, max_extra=90, max_length=180):
        penalty += 24
    if spiro_penalty and _has_contextual_graph_parent_alternative(item, prepared, max_extra=35):
        penalty += 12
    return penalty


def _has_contextual_graph_parent_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
    *,
    max_extra: int = 70,
    max_length: int = 125,
) -> bool:
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        if other.candidate.family not in _CONTEXTUAL_GRAPH_PARENT_FAMILIES:
            continue
        if len(other.name) > max_length or len(other.name) - len(item.name) > max_extra:
            continue
        if other.stability_penalty >= item.stability_penalty + 60:
            continue
        if _has_unstable_replacement_polycycle_alias(other.name) and not _has_generic_polycyclic_parent_descriptor(
            other.name
        ):
            continue
        return True
    return False


def _has_contextual_named_graph_parent_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
    *,
    max_extra: int = 70,
    max_length: int = 125,
) -> bool:
    named_parent_families = {
        "component-spiro-parent",
        "fused-retained-parent",
        "generated-fused-parent",
        "generated-hydro-fused-parent",
        "generated-partial-hydro-fused-parent",
        "generated-spiro-fused-parent",
        "generated-spiro-partial-hydro-parent",
        "hydro-fused-retained-parent",
    }
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        if other.candidate.family not in named_parent_families:
            continue
        if len(other.name) > max_length or len(other.name) - len(item.name) > max_extra:
            continue
        if other.stability_penalty >= item.stability_penalty + 60:
            continue
        return True
    return False


def _has_contextual_acyclic_parent_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
    *,
    max_extra: int = 70,
    max_length: int = 170,
) -> bool:
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        if other.candidate.family != "acyclic-skeleton-with-ring-substituents":
            continue
        if len(other.name) > max_length or len(other.name) - len(item.name) > max_extra:
            continue
        if other.stability_penalty >= item.stability_penalty + 60:
            continue
        return True
    return False


_BRIDGE_PARENT_PATTERN = r"(?:bicyclo|tricyclo|tetracyclo|pentacyclo|hexacyclo|heptacyclo|spiro|dispiro)\["


def _has_fragile_bridge_indicated_h_ketone_suffix(name: str) -> bool:
    if not re.search(r"-\d+[a-z]?\(\d+[a-z]?H\)-one\b", name):
        return False
    return bool(re.search(rf"(?:^|[-(])[^()\s]*?{_BRIDGE_PARENT_PATTERN}[^\]]+\][^)]*-", name))


def _has_contextual_oxo_bridge_parent_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
    *,
    max_extra: int = 30,
    max_length: int = 140,
) -> bool:
    graph_oxo_families = {
        "functional-generic-bicyclic-parent",
        "generic-bicyclic-parent",
        "generic-spiro-parent",
        "kekulized-generic-polycyclic-parent",
    }
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        if other.candidate.family not in graph_oxo_families:
            continue
        if len(other.name) > max_length or len(other.name) - len(item.name) > max_extra:
            continue
        if other.stability_penalty >= item.stability_penalty + 60:
            continue
        if "-oxo-" not in other.name and not re.search(r"(?:^|-)\d+[a-z]?-oxo-", other.name):
            continue
        if not re.search(_BRIDGE_PARENT_PATTERN, other.name):
            continue
        if _has_fragile_bridge_indicated_h_ketone_suffix(other.name):
            continue
        return True
    return False


def _has_fragile_single_ring_fused_substituent_benzene(name: str, candidate: NameCandidate) -> bool:
    if candidate.family != "single-ring-parent" or not name.endswith("benzene"):
        return False
    if not any(fragment in name for fragment in ("benzodioxan-", "benzofuran-", "benzothiophen-", "carbazol-")):
        return False
    return "benzyl" in name or "phenyloxy" in name or "benzonitrile" in name


def _has_contextual_relative_stereo_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
    *,
    max_extra: int = 35,
    max_length: int = 160,
) -> bool:
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        if not _has_relative_stereo_descriptor(other.name):
            continue
        if len(other.name) > max_length or len(other.name) - len(item.name) > max_extra:
            continue
        if other.stability_penalty >= item.stability_penalty + 60:
            continue
        return True
    return False


def _unlocanted_multicenter_stereo_penalty(name: str, candidate: NameCandidate) -> int:
    if not _has_unlocanted_multicenter_stereo_prefix(name):
        return 0
    if _locanted_atom_stereo_descriptor_count(name):
        return 0
    if candidate.stereo_prefix and _locanted_atom_stereo_descriptor_count(candidate.stereo_prefix):
        return 0
    if candidate.family in {
        "acyclic-parent",
        "acyclic-skeleton-with-ring-substituents",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "fused-retained-parent",
        "functional-acyclic-skeleton-with-ring-substituents",
        "generated-fused-parent",
        "generated-hydro-fused-parent",
        "generated-partial-hydro-fused-parent",
        "hydro-fused-retained-parent",
        "kekulized-generic-polycyclic-parent",
        "natural-product-topology-parent",
        "single-ring-parent",
    }:
        return 18
    return 10


def _has_missing_relative_stereo_ring_candidate(name: str, candidate: NameCandidate) -> bool:
    if candidate.family != "single-ring-parent" or _has_relative_stereo_descriptor(name):
        return False
    return "cyclobutane" in name and any(term in name for term in ("carbamyl", "carbonylamino", "oxycarbonylamino"))


def _has_relative_stereo_descriptor(name: str) -> bool:
    return "(trans)" in name or "(cis)" in name


def _skeletal_replacement_fused_parent_penalty(name: str) -> int:
    """Defer fragile hetero-fused replacement names to retained scaffolds.

    Replacement names such as ``triazaindeno[...]benzene`` are concise, but
    OPSIN frequently parses their fusion orientation poorly or rejects them
    outright.  When a generated/retained fused template describes the same core,
    it is usually a more stable roundtrip parent even if it is longer.
    """

    if not re.search(
        r"(?:(?:di|tri|tetra|penta)?aza|oxa|thia|phospha|bora)[a-z0-9,\-]*indeno\[[^\]]+\]benzene",
        name,
    ):
        return 0
    return 40


def _has_contextual_generic_polycycle_fallback(name: str, candidate: NameCandidate) -> bool:
    if candidate.family not in {
        "functional-generic-bicyclic-parent",
        "kekulized-generic-polycyclic-parent",
        "skeletal-replacement-fused-parent",
    }:
        return False
    if not _has_generic_polycyclic_parent_descriptor(name):
        return False
    return not _has_natural_product_topology_stem(name)


def _has_natural_product_topology_stem(name: str) -> bool:
    return bool(
        re.search(
            r"(?:gon|cholest|chol|androst|pregn|ergost|campest|stigmast|poriferast|cedran|guaian)(?:an|en|yl|e)?\b",
            name,
        )
    )


def _has_fragile_single_ring_sulfamoyl_benzene(name: str, candidate: NameCandidate) -> bool:
    if candidate.family != "single-ring-parent":
        return False
    if ")sulfamoyl)-benzene" not in name:
        return False
    return bool(re.search(r"(?:aza|diaza|oxa|thia|imino|dioxo).*\)sulfamoyl\)-benzene", name))


def _unstable_aza_polycycle_alias_penalty(name: str) -> int:
    """Defer aza-polycycle aliases with unstable OPSIN orientation.

    These retained-looking skeletal replacement aliases are concise, but in
    fused SureChEMBL scaffolds OPSIN often chooses a different atom numbering
    than the graph actually has. Generated fused or generic polycyclic
    candidates are longer but preserve the graph more reliably.
    """

    if "spiro[" in name or "bicyclo[" in name or "tricyclo[" in name:
        return 0
    if re.search(r"(?:^|-)\d+(?:,\d+)*-dioxo-.*azaacenaphthen\b", name):
        return 0
    if re.search(
        r"(?:^|-)(?:\d+[a-z]?,?)*(?:di|tri|tetra|penta)?aza"
        r"(?:acenaphthen|aceanthren|anthracen|fluoren|phenanthren|as-indacen|s-indacen|azuleno)",
        name,
    ):
        return 35
    return 0


def _has_unstable_replacement_polycycle_alias(name: str) -> bool:
    if "spiro[" in name or "bicyclo[" in name or "tricyclo[" in name:
        return False
    if re.search(r"(?:^|-)\d+(?:,\d+)*-dioxo-.*azaacenaphthen\b", name):
        return False
    return bool(
        re.search(
            r"(?:^|-)(?:di|tri|tetra|penta)?(?:aza|oxa|thia)"
            r"(?:acenaphthen|aceanthren|anthracen|fluoren|phenanthren|as-indacen|s-indacen|azuleno)",
            name,
        )
    )


def _has_unstable_benzimidazolyl_orientation_artifact(name: str) -> bool:
    if "benzimidazolium" in name:
        return False
    return bool(
        re.search(r"\b(?:\d+[a-z]?H-)?\d+-benzimidazolyl\b", name)
        or re.search(r"\b\d+(?:-[a-z]+)?1H-\d-benzimidazolyl\b", name)
    )


def _unstable_component_spiro_alias_penalty(name: str, candidate: NameCandidate) -> int:
    if candidate.family not in {
        "component-spiro-parent",
        "poly-component-spiro-parent",
        "component-dispiro-parent",
    }:
        return 0
    if "spiro[" not in name:
        return 0
    if re.search(r"(?:^|-|')(?:aza|oxa|thia|diaza|diox|triaza).*spiro\[", name) and re.search(
        r"\b(?:indane|inden|tetralin|decalin|fluoren|xanthen)\b",
        name,
    ):
        return 14
    return 0


def _candidate_selection_bonus(name: str, candidate: NameCandidate) -> int:
    bonus = _locanted_multicenter_stereo_bonus(name, candidate)
    if candidate.family not in {
        "fused-retained-parent",
        "generated-fused-parent",
        "generated-hydro-fused-parent",
        "generated-partial-hydro-fused-parent",
        "generated-spiro-partial-hydro-parent",
    }:
        return bonus
    if candidate.family == "generated-partial-hydro-fused-parent" and _has_stable_tetrahydro_imidazopyridine_parent(name):
        bonus += 12
    if candidate.family in {
        "generated-partial-hydro-fused-parent",
        "generated-spiro-partial-hydro-parent",
    } and _has_complete_hydro_generated_fused_parent(name):
        bonus += 10
    if candidate.family == "generated-fused-parent" and _has_oriented_generated_benzimidazole_parent(name):
        bonus += 22
    if candidate.family == "generated-hydro-fused-parent" and _has_complete_hydro_generated_fused_parent(name):
        bonus += 8
    if _has_indicated_h_fused_functional_suffix_parent(name):
        bonus += 12
    if _has_fused_indicated_h_ketone_suffix(name):
        bonus += 6
    if _has_fused_functional_suffix_parent(name):
        bonus += 6
    if _has_retained_indicated_h_parent(name):
        bonus += 6
    if _has_indicated_h_generated_fused_parent(name):
        bonus += 6
    return bonus


def _locanted_multicenter_stereo_bonus(name: str, candidate: NameCandidate) -> int:
    descriptor_count = _locanted_atom_stereo_descriptor_count(name)
    if descriptor_count < 2:
        return 0
    if candidate.family in {
        "acyclic-skeleton-with-ring-substituents",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "component-spiro-parent",
        "fused-retained-parent",
        "functional-acyclic-skeleton-with-ring-substituents",
        "generated-fused-parent",
        "generated-hydro-fused-parent",
        "generated-partial-hydro-fused-parent",
        "kekulized-generic-polycyclic-parent",
        "natural-product-topology-parent",
        "single-ring-parent",
    }:
        return min(24, 8 + 2 * descriptor_count)
    return min(12, 4 + descriptor_count)


def _has_unlocanted_multicenter_stereo_prefix(name: str) -> bool:
    match = re.match(r"^\(([^)]*)\)-", name)
    if not match:
        return False
    descriptors = match.group(1).split(",")
    atom_descriptors = [descriptor for descriptor in descriptors if descriptor in {"R", "S", "r", "s"}]
    if len(atom_descriptors) < 2:
        return False
    return not any(re.fullmatch(r"\d+'?[RSrs]", descriptor) for descriptor in descriptors)


def _locanted_atom_stereo_descriptor_count(text: str) -> int:
    return len(re.findall(r"(?<![A-Za-z])\d+'?[RSrs](?![a-z])", text))


def _has_stable_tetrahydro_imidazopyridine_parent(name: str) -> bool:
    return bool(re.search(r"4,5,6,7-tetrahydro-imidazo\[4,5-[cd]\]pyridine\b", name))


def _has_complete_hydro_generated_fused_parent(name: str) -> bool:
    if not re.search(r"(?:tetra|hexa|octa)hydro-", name):
        return False
    return bool(
        re.search(
            r"(?:pyrazolo|pyrrolo|imidazo|pyrido|pyrimido|quinazolino|quino|indolo|purino)"
            r"\[[^\]]+\](?:pyridin|pyrimidin|pyrazin|pyrazol|pyrrol|indol|quinolin)|"
            r"(?:naphthyridine|quinazoline|quinoline|quinoxaline|pteridine|purine)",
            name,
        )
    )


def _has_oriented_generated_benzimidazole_parent(name: str) -> bool:
    if "benzo[1,2-d]imidazol" not in name:
        return False
    return _has_top_level_generated_benzimidazole_orientation_locant(name)


def _has_top_level_generated_benzimidazole_orientation_locant(name: str) -> bool:
    parent_index = name.find("benzo[1,2-d]imidazol")
    if parent_index < 0:
        return False
    prefix = name[:parent_index]
    orientation_pattern = re.compile(
        r"(?:1|3)(?:,\d+)*-(?:di|tri)?"
        r"(?:methyl|ethyl|propyl|isopropyl|butyl|t-butyl|cyclo|phenyl|benzyl|"
        r"methoxy|ethoxy|piperidin|piperidino|piperazin|piperazino|pyrrolidin|"
        r"azetidin|morpholin|\()"
    )
    paren_depth = 0
    bracket_depth = 0
    for index, char in enumerate(prefix):
        if char == "(":
            paren_depth += 1
        elif char == ")" and paren_depth:
            paren_depth -= 1
        elif char == "[":
            bracket_depth += 1
        elif char == "]" and bracket_depth:
            bracket_depth -= 1
        if paren_depth or bracket_depth:
            continue
        if index and prefix[index - 1] != "-":
            continue
        if orientation_pattern.match(prefix, index):
            return True
    return False


def _has_retained_indicated_h_parent(name: str) -> bool:
    return bool(re.search(r"(?:^|-)\d+H-(?:indazol|purin)e?", name))


def _has_indicated_h_generated_fused_parent(name: str) -> bool:
    return bool(re.search(r"(?:^|-)\d+H-(?:pyrazolo|pyrrolo|imidazo)\[[^\]]+\](?:pyridin|pyrimidin)", name))


def _has_fused_indicated_h_ketone_suffix(name: str) -> bool:
    if "acenaphthen" in name:
        return False
    return bool(
        re.search(r"\[[^\]]+\].*-\d+\(\d+[a-z]?H\)-one\b", name)
        or re.search(r"\b(?:quinazolin|pteridin|triazafluoren)-\d+\(\d+[a-z]?H\)-one\b", name)
    )


def _has_fused_functional_suffix_parent(name: str) -> bool:
    return bool(
        re.search(
            r"(?:\[[^\]]+\]|benzimidazol|benzotriazol|indazol|quinazolin|pyrrolo|pyrazolo|imidazo).*-\d+[a-z]?-(?:carboxamide|carbonitrile|carboxylic acid|carboxylate)\b",
            name,
        )
    )


def _has_indicated_h_fused_functional_suffix_parent(name: str) -> bool:
    return bool(
        re.search(
            r"(?:^|-)\d+[a-z]?H-.*-\d+[a-z]?-(?:carboxamide|carbonitrile|carboxylic acid|carboxylate)\b",
            name,
        )
    )


def _has_small_ring_carbonyl_suffix_parent(name: str) -> bool:
    return "cyclobut-1-en-3,4-dione" in name or "cycloprop-1-en-3-one" in name


def _has_polycyclic_parent_descriptor(name: str) -> bool:
    return any(term in name for term in ("bicyclo[", "tricyclo[", "spiro["))


def _has_generic_polycyclic_parent_descriptor(name: str) -> bool:
    return any(
        term in name
        for term in (
            "bicyclo[",
            "tricyclo[",
            "tetracyclo[",
            "pentacyclo[",
            "spiro[",
        )
    )


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
        "azetidin",
        "thietane",
        "oxetane",
        "azolidine",
        "thiolane",
        "oxolane",
        "piperidine",
        "piperidin",
        "thiane",
        "oxane",
        "oxan",
        "cyclopropane",
        "cyclopropan",
        "cyclobutane",
        "cyclobutan",
        "cyclopentane",
        "cyclopentan",
        "cyclohexane",
        "cyclohexan",
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


def _has_fragile_bicyclic_carbamate_lactam_name(name: str) -> bool:
    if "bicyclo[" not in name or "oxycarbonyl" not in name:
        return False
    if "oxycarbonyl)amino" not in name or not ("oxo" in name or "one" in name):
        return False
    return True


def _has_fragile_imido_oxycarbonyl_alias(name: str) -> bool:
    return bool(re.search(r"\b\d+,\d+-dioxopyrrolidino\)?oxycarbonyl\b", name))


def _finalize_candidate_name(
    mol,
    Chem,
    candidate: NameCandidate,
    *,
    max_length: int = _DEFAULT_MAX_NAME_LENGTH,
) -> str:
    """Apply whole-molecule decorations without shortening the generated name.

    Retained groups, suffix parents, and shorter branch spellings must already
    have been produced by the upstream rule that created the candidate. This
    step only adds stereo/isotope information and enforces the length cap.
    """

    name = candidate.name
    if candidate.stereo_prefix and not name.startswith("("):
        name = f"({candidate.stereo_prefix})-{name}"
    if candidate.family in {
        "whole-molecule-retained-parent",
        "natural-product-topology-parent",
        "exact-biochemical-retained-parent",
    }:
        name = _apply_uniform_isotope_label(mol, name)
        return name if len(name) <= max_length else ""
    skip_fallback_stereo = candidate.family == "retained-acid-ester-parent" and _has_internal_locanted_stereo(name)
    if (
        not candidate.stereo_prefix
        and not _has_retained_carbohydrate_substituent(name)
        and not skip_fallback_stereo
        and _has_assigned_stereo(mol, Chem)
    ):
        name = apply_stereochemical_prefix(
            mol,
            Chem,
            name,
            atom_order=_fallback_stereo_atom_order(candidate.family, name),
        )
    name = _apply_uniform_isotope_label(mol, name)
    return name if len(name) <= max_length else ""


def _has_retained_carbohydrate_substituent(name: str) -> bool:
    return any(
        fragment in name
        for fragment in (
            "glucopyranosyl",
            "galactopyranosyl",
            "mannopyranosyl",
            "fucopyranosyl",
            "ribofuranosyl",
            "fructofuranosyl",
            "apiofuranosyl",
            "glucosaminyl",
            "galactosaminyl",
            "mannosaminyl",
            "muramyl",
            "neuraminyl",
        )
    )


def _has_internal_locanted_stereo(name: str) -> bool:
    return bool(re.search(r"\(\d+[RSZE](?:,\d+[RSZE])*\)-", name))


def _apply_uniform_isotope_label(mol, name: str) -> str:
    if not name:
        return name
    if not _mol_has_isotope(mol):
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


def _mol_has_isotope(mol) -> bool:
    cached = mol_cache_get(mol, "has_isotope")
    if cached is not None:
        return cached
    return mol_cache_set(mol, "has_isotope", any(atom.GetIsotope() for atom in mol.GetAtoms()))


def _fallback_stereo_atom_order(family: str, name: str = "") -> str:
    if any(fragment in name for fragment in ("pseudouridin", "uridin", "spirostane")):
        return "idx"
    return "idx_desc"
