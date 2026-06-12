from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
import re
import time

from .candidate_facts import (
    CandidateSafety,
    ChargeEvidence,
    HydrogenEvidence,
    OPSIN_FRAGILE_EXOCYCLIC_CHARGED_RING_SUFFIX,
    OPSIN_FRAGILE_GROUP14_HETERO_CARBON_LIGAND,
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
from .stereochemistry import (
    apply_locanted_stereochemical_prefix,
    apply_stereochemical_prefix,
    assigned_atom_stereochemistry,
)
from .telemetry import CandidateTelemetry


_DEFAULT_MAX_NAME_LENGTH = 260
_STEREO_EMPTY_FALLBACK_EVALUATION_LENGTH = 320
_STEREO_EMPTY_FALLBACK_MAX_LENGTH = 320
_NONSTEREO_EMPTY_FALLBACK_EVALUATION_LENGTH = 320
_NONSTEREO_EMPTY_FALLBACK_MAX_LENGTH = 320
_FRAGMENT_SUBSTITUENT_FALLBACK_PROP = "_nispo_allow_fragment_substituent_fallback"
_RETAINED_PARTIAL_HYDRO_FALLBACK_PROP = "_nispo_enable_retained_partial_hydro_fallback"


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
    stereo_locants: tuple[tuple[int, str], ...] = ()
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
    # `name` is the selection surface. `output_name` allows a renderer-only
    # improvement to be returned without changing parent-family selection.
    name: str
    candidate: NameCandidate
    producer_family: str
    stability_penalty: int
    selection_bonus: int
    output_name: str = ""


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
                return _finalize_direct_name(mol, self._Chem, simple_name)
        fragment_atom_sets = self._Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
        if len(fragment_atom_sets) > 1:
            fragments = self._Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            return _finalize_direct_name(mol, self._Chem, self._fragment_rule.name(fragments))
        simple_name = simple_fragment_name(mol, self._Chem)
        if simple_name:
            return _finalize_direct_name(mol, self._Chem, simple_name)
        standalone_polyatomic_name = standalone_polyatomic_fragment_name(mol, self._Chem)
        if standalone_polyatomic_name:
            return _finalize_direct_name(mol, self._Chem, standalone_polyatomic_name)
        polyatomic_name = common_polyatomic_fragment_name(mol, self._Chem)
        if polyatomic_name:
            return _finalize_direct_name(mol, self._Chem, polyatomic_name)

        ctx = MoleculeContext(mol, self._Chem, telemetry=telemetry)
        best_name = _select_best_prepared_candidate(
            _prepare_candidates(mol, self._Chem, ctx, self._rules),
            telemetry=telemetry,
        )

        if best_name:
            return best_name

        fallback_evaluation_length, fallback_max_length = _empty_fallback_limits(mol, self._Chem)
        had_fragment_fallback_prop = mol.HasProp(_FRAGMENT_SUBSTITUENT_FALLBACK_PROP)
        previous_fragment_fallback_value = (
            mol.GetProp(_FRAGMENT_SUBSTITUENT_FALLBACK_PROP)
            if had_fragment_fallback_prop
            else ""
        )
        had_partial_hydro_fallback_prop = mol.HasProp(_RETAINED_PARTIAL_HYDRO_FALLBACK_PROP)
        previous_partial_hydro_fallback_value = (
            mol.GetProp(_RETAINED_PARTIAL_HYDRO_FALLBACK_PROP)
            if had_partial_hydro_fallback_prop
            else ""
        )
        mol.SetProp(_FRAGMENT_SUBSTITUENT_FALLBACK_PROP, "1")
        mol.SetProp(_RETAINED_PARTIAL_HYDRO_FALLBACK_PROP, "1")
        try:
            fallback_ctx = MoleculeContext(mol, self._Chem, telemetry=telemetry)
            fallback_name = _select_best_prepared_candidate(
                _prepare_candidates(
                    mol,
                    self._Chem,
                    fallback_ctx,
                    self._rules,
                    max_length=fallback_evaluation_length,
                    output_max_length=fallback_max_length,
                ),
                telemetry=telemetry,
            )
            if not fallback_name:
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
        finally:
            if had_fragment_fallback_prop:
                mol.SetProp(_FRAGMENT_SUBSTITUENT_FALLBACK_PROP, previous_fragment_fallback_value)
            else:
                mol.ClearProp(_FRAGMENT_SUBSTITUENT_FALLBACK_PROP)
            if had_partial_hydro_fallback_prop:
                mol.SetProp(_RETAINED_PARTIAL_HYDRO_FALLBACK_PROP, previous_partial_hydro_fallback_value)
            else:
                mol.ClearProp(_RETAINED_PARTIAL_HYDRO_FALLBACK_PROP)
        return fallback_name


def _finalize_direct_name(mol, Chem, name: str, *, max_length: int = _DEFAULT_MAX_NAME_LENGTH) -> str:
    if not name:
        return ""
    if _has_assigned_stereo(mol, Chem):
        name = apply_stereochemical_prefix(mol, Chem, name)
    name = _apply_uniform_isotope_label(mol, name)
    return name if len(name) <= max_length else ""


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
                output_name = _finalize_candidate_name(mol, Chem, candidate, max_length=max_length)
                if not output_name:
                    continue
                if output_max_length is not None and len(output_name) > output_max_length:
                    continue
                selection_name = _candidate_selection_surface_name(
                    mol,
                    Chem,
                    candidate,
                    output_name,
                    max_length=max_length,
                )
                prepared.append(
                    _PreparedCandidate(
                        name=selection_name,
                        candidate=candidate,
                        producer_family=rule.family,
                        stability_penalty=_candidate_selection_stability_penalty(selection_name, candidate),
                        selection_bonus=_candidate_selection_bonus(selection_name, candidate),
                        output_name=output_name,
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
                output_name = _finalize_candidate_name(mol, Chem, candidate, max_length=max_length)
                if not output_name:
                    ctx.telemetry.rejected_candidate_count[rule.family] += 1
                    continue
                if output_max_length is not None and len(output_name) > output_max_length:
                    ctx.telemetry.rejected_candidate_count[rule.family] += 1
                    continue
                selection_name = _candidate_selection_surface_name(
                    mol,
                    Chem,
                    candidate,
                    output_name,
                    max_length=max_length,
                )
                ctx.telemetry.rendered_candidate_count[rule.family] += 1
                prepared.append(
                    _PreparedCandidate(
                        name=selection_name,
                        candidate=candidate,
                        producer_family=rule.family,
                        stability_penalty=_candidate_selection_stability_penalty(selection_name, candidate),
                        selection_bonus=_candidate_selection_bonus(selection_name, candidate),
                        output_name=output_name,
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
        stability_penalty = (
            item.stability_penalty
            + _candidate_selection_contextual_penalty(item, prepared)
            + _candidate_selection_broad_graph_parent_penalty(item, prepared)
        )
        selection_bonus = (
            item.selection_bonus
            + _candidate_selection_graph_parent_bonus(candidate)
            + _candidate_selection_contextual_bonus(item, prepared)
        )
        candidate_key = (
            stability_penalty,
            candidate.priority,
            _candidate_selection_render_rank_key(candidate),
            len(item.name) - selection_bonus,
            len(item.name),
            _comparable_sort_key(candidate.sort_key),
            item.name,
        )
        if best_key is None or candidate_key < best_key:
            best_name = _prepared_output_name(item)
            best_key = candidate_key
    if best_name and telemetry is not None:
        for item in prepared:
            if _prepared_output_name(item) == best_name:
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


def _prepared_output_name(item: _PreparedCandidate) -> str:
    return item.output_name or item.name


def _candidate_selection_surface_name(
    mol,
    Chem,
    candidate: NameCandidate,
    output_name: str,
    *,
    max_length: int,
) -> str:
    if (
        candidate.family not in {"single-ring-parent", "positive-single-ring-parent"}
        or not candidate.stereo_locants
    ):
        return output_name
    selection_candidate = replace(candidate, stereo_locants=())
    return _finalize_candidate_name(mol, Chem, selection_candidate, max_length=max_length) or output_name


def _candidate_has_structured_facts(candidate: NameCandidate) -> bool:
    return bool(
        candidate.parent_atoms
        or candidate.locants
        or candidate.stereo_locants
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
    if _has_fragile_ethanediylidene_alias(name):
        penalty += 45
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
    if candidate.family == "kekulized-generic-polycyclic-parent":
        penalty += _kekulized_polycyclic_late_render_rank_penalty(name, candidate)
    penalty += _unstable_component_spiro_alias_penalty(name, candidate)
    penalty += _unlocanted_multicenter_stereo_penalty(name, candidate)
    return penalty


def _kekulized_polycyclic_late_render_rank_penalty(name: str, candidate: NameCandidate) -> int:
    if candidate.render_rank is None:
        return 0
    if "tetraaza" in name:
        return 0
    return max(0, candidate.render_rank - 1) * 4


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

_CONTEXTUAL_SYSTEMATIC_ALTERNATIVE_FAMILIES = {
    "acyclic-parent",
    "acyclic-skeleton-with-ring-substituents",
    "chain-dominant-acyclic-skeleton-with-ring-substituents",
    "functional-acyclic-skeleton-with-ring-substituents",
    "functional-generic-bicyclic-parent",
    "generic-bicyclic-parent",
    "kekulized-generic-polycyclic-parent",
    "positive-single-ring-parent",
    "preferred-acyclic-skeleton-with-ring-substituents",
    "single-ring-parent",
}


_GRAPH_PARENT_STABILITY_BONUS_BY_FAMILY = {
    "acyclic-skeleton-with-ring-substituents": 19,
    "chain-dominant-acyclic-skeleton-with-ring-substituents": 19,
    "component-dispiro-parent": 21,
    "component-spiro-parent": 21,
    "functional-generic-bicyclic-parent": 24,
    "generated-fused-parent": 144,
    "generated-hydro-fused-parent": 24,
    "generated-partial-hydro-fused-parent": 21,
    "generic-bicyclic-parent": 24,
    "generic-spiro-parent": 24,
    "kekulized-generic-polycyclic-parent": 30,
    "poly-component-spiro-parent": 21,
    "preferred-acyclic-skeleton-with-ring-substituents": 19,
}

_FUSED_PARENT_ATOM_BONUS_FAMILIES = {
    "fused-retained-parent",
    "generated-fused-parent",
    "generated-hydro-fused-parent",
    "generated-partial-hydro-fused-parent",
    "hydro-fused-retained-parent",
    "retained-topology-replacement-fused-parent",
}

_RENDER_RANK_BEFORE_LENGTH_FAMILIES = {
    "component-spiro-parent",
    "functional-generic-bicyclic-parent",
    "fused-retained-parent",
    "generated-fused-parent",
    "generic-bicyclic-parent",
    "kekulized-generic-polycyclic-parent",
    "retained-topology-replacement-fused-parent",
}

_ACYCLIC_SKELETON_PARENT_FAMILIES = {
    "acyclic-skeleton-with-ring-substituents",
    "chain-dominant-acyclic-skeleton-with-ring-substituents",
    "functional-acyclic-skeleton-with-ring-substituents",
    "imino-functional-acyclic-skeleton-with-ring-substituents",
    "preferred-acyclic-skeleton-with-ring-substituents",
}

_ACYCLIC_RING_DECOMPOSITION_PARENT_FAMILIES = _ACYCLIC_SKELETON_PARENT_FAMILIES | {
    "charged-ring-acyclic-skeleton-with-ring-substituents",
}

_ACYCLIC_INTEGRATED_RING_PARENT_ALTERNATIVE_FAMILIES = {
    "component-spiro-parent",
    "fused-retained-parent",
    "functional-generic-bicyclic-parent",
    "generated-fused-parent",
    "generated-hydro-fused-parent",
    "generated-partial-hydro-fused-parent",
    "generic-bicyclic-parent",
    "generic-spiro-parent",
    "hydro-fused-retained-parent",
    "kekulized-generic-polycyclic-parent",
    "retained-topology-replacement-fused-parent",
    "skeletal-replacement-fused-parent",
    "single-ring-parent",
}

_HYDRO_FUSED_SYSTEMATIC_ALTERNATIVE_FAMILIES = {
    "fused-retained-parent",
    "generated-fused-parent",
    "kekulized-generic-polycyclic-parent",
    "retained-topology-replacement-fused-parent",
}

_COMPARABLE_COMPLETE_FUSED_PARENT_ALTERNATIVE_FAMILIES = {
    "fused-retained-parent",
    "generated-fused-parent",
    "generated-hydro-fused-parent",
    "generated-oxido-fused-parent",
    "generated-partial-hydro-fused-parent",
    "hydro-fused-retained-parent",
    "kekulized-generic-polycyclic-parent",
    "oxido-fused-retained-parent",
    "retained-topology-replacement-fused-parent",
    "skeletal-replacement-fused-parent",
}

_LARGE_FUSED_PARENT_WITH_SINGLE_RING_SUFFIX_FAMILIES = {
    "fused-retained-parent",
    "generated-fused-parent",
}

_SINGLE_RING_SUFFIX_PARENT_ALTERNATIVE_FAMILIES = {
    "positive-single-ring-parent",
    "single-ring-parent",
}

_BROAD_GRAPH_PARENT_ALTERNATIVE_FAMILIES = (
    _CONTEXTUAL_GRAPH_PARENT_FAMILIES
    | _CONTEXTUAL_SYSTEMATIC_ALTERNATIVE_FAMILIES
    | {
        "component-dispiro-parent",
        "functional-generic-bicyclic-parent",
        "poly-component-spiro-parent",
        "retained-topology-replacement-fused-parent",
        "skeletal-replacement-fused-parent",
    }
)

_BROAD_GRAPH_PARENT_FRAGILE_WINNER_FAMILIES = {
    "acyclic-parent",
    "fused-retained-parent",
    "generated-fused-parent",
    "generated-hydro-fused-parent",
    "generated-partial-hydro-fused-parent",
    "retained-topology-replacement-fused-parent",
    "skeletal-replacement-fused-parent",
    "single-ring-parent",
    "whole-molecule-retained-parent",
}

_RETAINED_LACTONE_TEMPLATE_NAMES = frozenset({"coumarin", "coumarine"})
_RETAINED_LACTONE_TOPOLOGY_BONUS = 160
_RETAINED_CYCLIC_ANHYDRIDE_TEMPLATE_NAMES = frozenset({"isatoic anhydride"})
_RETAINED_CYCLIC_ANHYDRIDE_TOPOLOGY_BONUS = 80
_FRAGILE_RETAINED_TOPOLOGY_REPLACEMENT_SELECTED_TEMPLATE_TERMS = (
    "benzothiazep",
    "pyrido[",
)
_BIOCHEM_HETEROFUSED_RETAINED_STEMS = (
    "adenin",
    "guanin",
    "hypoxanthin",
    "mercaptopurin",
    "pteridin",
    "pterin",
    "purin",
    "xanthin",
)
_BIOCHEM_HETEROFUSED_TEMPLATE_TERMS = (
    "adenin",
    "guanin",
    "hypoxanthin",
    "imidazo",
    "pteridin",
    "pterin",
    "purin",
    "pyrimido",
    "pyrimidino",
    "xanthin",
)
_BIOCHEM_RETAINED_TEMPLATE_TERMS = (
    "adenin",
    "guanin",
    "hypoxanthin",
    "pteridin",
    "pterin",
    "purin",
    "xanthin",
)
_BIOCHEM_GENERATED_FUSION_TEMPLATE_TERMS = (
    "imidazo",
    "pyrimido",
    "pyrimidino",
)
_EXOCYCLIC_FUSED_TEMPLATE_TERMS = (
    "azaindan",
    "benzo[1,2-b]pyrrole",
    "indane",
    "indan",
    "indolin",
    "oxindol",
)
_EXOCYCLIC_FUSED_PARENT_FAMILIES = {
    "fused-retained-parent",
    "generated-fused-parent",
    "generated-partial-hydro-fused-parent",
    "hydro-fused-retained-parent",
    "retained-topology-replacement-fused-parent",
}

_GENERATED_FUSED_SAME_ATOMS_RETAINED_TEMPLATE_BLOCKLIST = frozenset({"1H-benzimidazole"})
_HIGH_CONFIDENCE_FUSED_TEMPLATE_PARENT_NAMES = frozenset(
    {
        "1,4-benzodioxane",
        "1H-benzotriazole",
        "fluoren",
        "fluorene",
        "imidazo[4,5-a]benzene",
        "tetralin",
    }
)
_VERY_HIGH_CONFIDENCE_RETAINED_FUSED_TEMPLATE_PARENT_NAMES = frozenset(
    {
        "1,4-benzodioxane",
        "1H-benzotriazole",
        "tetralin",
    }
)
_HIGH_CONFIDENCE_FUSED_TEMPLATE_PARENT_FAMILIES = frozenset(
    {
        "fused-retained-parent",
        "generated-fused-parent",
        "generated-hydro-fused-parent",
        "generated-partial-hydro-fused-parent",
        "hydro-fused-retained-parent",
        "oxido-fused-retained-parent",
        "retained-topology-replacement-fused-parent",
        "skeletal-replacement-fused-parent",
    }
)
_HIGH_CONFIDENCE_FUSED_TEMPLATE_PARENT_BONUS = 960
_VERY_HIGH_CONFIDENCE_RETAINED_FUSED_TEMPLATE_PARENT_BONUS = 1320
_HIGH_CONFIDENCE_IMIDAZO45A_FUSED_TEMPLATE_PARENT_BONUS = 640
_HIGH_CONFIDENCE_FUSED_TEMPLATE_COMPETITOR_STABILITY_MARGIN = 220
_TETRALIN_TO_BENZODIOXANE_TEMPLATE_PAIR_BONUS = 220
_IMIDAZO45A_TEMPLATELESS_COMPETITOR_FAMILIES = frozenset(
    {
        "acyclic-skeleton-with-ring-substituents",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "charged-ring-acyclic-skeleton-with-ring-substituents",
        "generic-bicyclic-parent",
        "generic-spiro-parent",
        "retained-acid-ester-parent",
        "spiro-functional-chain-substituent",
    }
)
_IMIDAZO45A_LARGE_ACYCLIC_COMPETITOR_FAMILIES = frozenset(
    {
        "acyclic-skeleton-with-ring-substituents",
        "charged-ring-acyclic-skeleton-with-ring-substituents",
    }
)
_LOW_CONFIDENCE_FUSED_TEMPLATE_TO_TEMPLATELESS_FAMILIES = {
    "benzo[1,2-b]furan": frozenset({"single-ring-parent"}),
    "imidazo[1,2-b]pyridazine": frozenset(
        {
            "acyclic-skeleton-with-ring-substituents",
            "single-ring-parent",
        }
    ),
    "pyrimido[4,5-a]benzene": frozenset({"acyclic-skeleton-with-ring-substituents"}),
    "pyrrolo[2,1-f]as-triazine": frozenset({"single-ring-parent"}),
}
_LOW_CONFIDENCE_FUSED_TEMPLATE_PARENT_FAMILIES = frozenset(
    {
        "fused-retained-parent",
        "generated-fused-parent",
        "generated-hydro-fused-parent",
        "generated-partial-hydro-fused-parent",
    }
)
_LOW_CONFIDENCE_FUSED_TEMPLATE_PARENT_PENALTY = 180
_NAMED_TEMPLATE_PAIR_PARENT_BONUS = 260


def _candidate_selection_graph_parent_bonus(candidate: NameCandidate) -> int:
    bonus = _GRAPH_PARENT_STABILITY_BONUS_BY_FAMILY.get(candidate.family, 0)
    bonus += _candidate_selection_parent_atom_bonus(candidate)
    bonus += _candidate_selection_kekulized_early_rank_bonus(candidate)
    return bonus


def _candidate_selection_parent_atom_bonus(candidate: NameCandidate) -> int:
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return 0
    if candidate.family == "natural-product-topology-parent":
        return 0
    if (
        candidate.family == "fused-retained-parent"
        and candidate.source_family == "static-retained"
        and _static_retained_fused_parent_atom_bonus_blocked(candidate)
    ):
        return 0
    parent_count = len(candidate.parent_atoms)
    if candidate.family in _FUSED_PARENT_ATOM_BONUS_FAMILIES:
        return min(120, max(0, parent_count - 6) * 8)
    return min(80, max(0, parent_count - 6) * 4)


def _candidate_selection_kekulized_early_rank_bonus(candidate: NameCandidate) -> int:
    if candidate.family != "kekulized-generic-polycyclic-parent":
        return 0
    if candidate.render_rank is None:
        return 0
    if "tetraaza" in candidate.name:
        return 0
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return 0
    return 2 * max(0, 40 - (candidate.render_rank - 1) * 5)


def _candidate_selection_render_rank_key(candidate: NameCandidate) -> int:
    if candidate.family not in _RENDER_RANK_BEFORE_LENGTH_FAMILIES:
        return 999
    if candidate.render_rank is None:
        return 999
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return 999
    return candidate.render_rank


def _static_retained_fused_parent_atom_bonus_blocked(candidate: NameCandidate) -> bool:
    template_name = (candidate.template_name or "").lower()
    if template_name == "pterin":
        return True
    return "one" in template_name


def _candidate_selection_broad_graph_parent_penalty(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> int:
    if item.candidate.family in _ACYCLIC_SKELETON_PARENT_FAMILIES:
        penalty = 0
        if _has_contextual_single_ring_parent_candidate_alternative(item, prepared):
            penalty = max(penalty, 55)
        if _has_contextual_terminal_ring_methan_parent_alternative(item, prepared):
            penalty = max(penalty, 142)
        if _has_contextual_integrated_ring_parent_alternative(item, prepared):
            penalty = max(penalty, 142)
        return penalty
    if item.candidate.family not in _BROAD_GRAPH_PARENT_FRAGILE_WINNER_FAMILIES:
        return 0
    if item.candidate.family in {"single-ring-parent", "positive-single-ring-parent"}:
        if not _has_contextual_non_acyclic_broad_graph_parent_alternative(item, prepared):
            return 0
    if not _has_contextual_broad_graph_parent_alternative(item, prepared):
        return 0
    return 70


def _has_contextual_single_ring_parent_candidate_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        if other.candidate.family not in {"single-ring-parent", "positive-single-ring-parent"}:
            continue
        if not other.candidate.parent_atoms or other.candidate.render_rank is None:
            continue
        if len(other.name) > 170 or len(other.name) - len(item.name) > 45:
            continue
        if other.stability_penalty >= item.stability_penalty + 60:
            continue
        if not candidate_has_complete_graph(other.candidate):
            continue
        return True
    return False


def _has_contextual_non_acyclic_broad_graph_parent_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        if other.candidate.family in _ACYCLIC_SKELETON_PARENT_FAMILIES:
            continue
        if item.candidate.family in {
            "single-ring-parent",
            "positive-single-ring-parent",
        } and other.candidate.family in {
            "single-ring-parent",
            "positive-single-ring-parent",
        }:
            continue
        if other.candidate.family not in _BROAD_GRAPH_PARENT_ALTERNATIVE_FAMILIES:
            continue
        if len(other.name) > 240 or len(other.name) - len(item.name) > 140:
            continue
        if other.stability_penalty >= item.stability_penalty + 60:
            continue
        return True
    return False


def _has_contextual_integrated_ring_parent_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    if not _has_acyclic_ring_substituent_decomposition_signal(item.name):
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        if other.candidate.family not in _ACYCLIC_INTEGRATED_RING_PARENT_ALTERNATIVE_FAMILIES:
            continue
        if len(other.name) > 260 or len(other.name) - len(item.name) > 125:
            continue
        if other.stability_penalty >= item.stability_penalty + 80:
            continue
        if other.candidate.family in {
            "functional-generic-bicyclic-parent",
            "generic-bicyclic-parent",
            "generic-spiro-parent",
            "kekulized-generic-polycyclic-parent",
        } and not re.search(_BRIDGE_PARENT_PATTERN, other.name):
            continue
        if other.candidate.family == "single-ring-parent" and not _has_single_ring_parent_reintegration_signal(
            item.name,
            other.name,
        ):
            continue
        if not _has_integrated_ring_parent_signal(other.name):
            continue
        return True
    return False


def _has_contextual_terminal_ring_methan_parent_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    if not _has_terminal_ring_methan_decomposition_signal(item.name):
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        if other.candidate.family not in _ACYCLIC_INTEGRATED_RING_PARENT_ALTERNATIVE_FAMILIES:
            continue
        if len(other.name) > 260 or len(other.name) - len(item.name) > 125:
            continue
        if other.stability_penalty >= item.stability_penalty + 80:
            continue
        if not _has_integrated_ring_parent_signal(other.name):
            continue
        return True
    return False


def _has_terminal_ring_methan_decomposition_signal(name: str) -> bool:
    if not name.endswith("-methan"):
        return False
    if not any(term in name for term in ("benz", "cyclo", "phenyl", "pyrid", "pyrim", "pyraz", "imidaz", "triaz")):
        return False
    if "pyrazol-1-yl)-methan" in name:
        return False
    if re.search(r"oxo-2-oxabicyclo\[4\.4\.0\].*yl\)-methan\b", name):
        return False
    return bool(re.search(r"(?:yl|ino|oxy)\)-methan\b|yl\)oxy\)-methan\b", name))


def _has_acyclic_ring_substituent_decomposition_signal(name: str) -> bool:
    if _has_stable_acyclic_ring_substituent_exception(name):
        return False
    if _has_unstable_benzimidazolyl_orientation_artifact(name):
        return True
    if not any(
        term in name
        for term in (
            "benzimidazol",
            "benzotriazol",
            "benzoxazol",
            "benzothiazol",
            "coumarin",
            "coumaron",
            "oxanaphthalin",
            "azacoumarin",
            "indazol",
            "indolin",
            "isoindolin",
            "oxindol",
            "pterin",
            "pteridin",
            "purin",
            "quinazolin",
            "quinoxalin",
            "naphthyridin",
            "pyrazolo[",
            "imidazo[",
            "pyrido[",
            "pyrimido[",
            "pyrrolo[",
            "thieno[",
        )
    ):
        return False
    return any(
        token in name
        for token in (
            "-yl",
            "yl)",
            "ylcarbonyl",
            "ylmethyl",
            "ylsulfonyl",
            "ylsulfanyl",
            "yloxy",
            "carbamoyl",
            "carbamyl",
            "carboxamido",
            "carbonylamino",
            "sulfonamido",
        )
    )


def _has_stable_acyclic_ring_substituent_exception(name: str) -> bool:
    if any(term in name for term in ("glutarimide", "2,6-dioxo-3-piperidyl", "oxoisoindolin")):
        return True
    return bool(re.search(r"pyrido\[[^\]]+\]pyrrole", name))


def _has_single_ring_parent_reintegration_signal(acyclic_name: str, ring_name: str) -> bool:
    if not any(term in acyclic_name for term in ("carbamoyl", "carbamyl", "carboxamido", "carbonylamino")):
        return False
    return bool(re.search(r"(?:benzene|pyridin|pyrimidin|pyrazin|pyridazin|pyrazol|imidazol|triazol)", ring_name))


def _has_integrated_ring_parent_signal(name: str) -> bool:
    if re.search(_BRIDGE_PARENT_PATTERN, name) or re.search(r"\[[^\]]+\]", name):
        return True
    return any(
        term in name
        for term in (
            "benzimidazol",
            "benzotriazol",
            "benzoxazol",
            "benzothiazol",
            "coumarin",
            "coumaron",
            "oxanaphthalin",
            "azacoumarin",
            "indazol",
            "indolin",
            "isoindolin",
            "oxindol",
            "pterin",
            "pteridin",
            "purin",
            "quinazolin",
            "quinoxalin",
            "naphthyridin",
        )
    )


def _has_contextual_broad_graph_parent_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        if other.candidate.family not in _BROAD_GRAPH_PARENT_ALTERNATIVE_FAMILIES:
            continue
        if len(other.name) > 240 or len(other.name) - len(item.name) > 140:
            continue
        if other.stability_penalty >= item.stability_penalty + 60:
            continue
        return True
    return False


def _has_contextual_retained_lactone_topology_candidate(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    return _has_contextual_retained_topology_candidate_for_templates(
        item,
        prepared,
        template_names=_RETAINED_LACTONE_TEMPLATE_NAMES,
    )


def _has_contextual_retained_cyclic_anhydride_topology_candidate(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    return _has_contextual_retained_topology_candidate_for_templates(
        item,
        prepared,
        template_names=_RETAINED_CYCLIC_ANHYDRIDE_TEMPLATE_NAMES,
    )


def _has_contextual_fused_parent_alternative_for_retained_topology_replacement(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family != "retained-topology-replacement-fused-parent":
        return False
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return False
    if candidate.source_family != "opsin-retained-ring-cml":
        return False
    if candidate.suffixes:
        return False
    if candidate.render_rank is not None and candidate.render_rank > 1:
        return False
    candidate_parent_count = len(candidate.parent_atoms)
    if candidate_parent_count < 10:
        return False
    if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "fused-retained-parent":
            continue
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        other_template_name = (other_candidate.template_name or "").lower()
        if any(
            term in other_template_name
            for term in _FRAGILE_RETAINED_TOPOLOGY_REPLACEMENT_SELECTED_TEMPLATE_TERMS
        ):
            continue
        if len(candidate.parent_atoms & other_candidate.parent_atoms) < min(candidate_parent_count, 8):
            continue
        if len(other_candidate.parent_atoms) < candidate_parent_count:
            continue
        if len(item.name) - len(other.name) > 160:
            continue
        if item.stability_penalty >= other.stability_penalty + 90:
            continue
        return True
    return False


def _has_contextual_retained_topology_candidate_for_templates(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
    *,
    template_names: frozenset[str],
) -> bool:
    candidate = item.candidate
    if candidate.family != "retained-topology-replacement-fused-parent":
        return False
    if not candidate_has_complete_graph(candidate):
        return False
    if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
        return False
    if candidate.render_rank is not None and candidate.render_rank > 2:
        return False
    candidate_parent_count = len(candidate.parent_atoms)
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family not in {"fused-retained-parent", "hydro-fused-retained-parent"}:
            continue
        if other_candidate.template_name not in template_names:
            continue
        if not candidate_has_complete_graph(other_candidate):
            continue
        other_parent_count = len(other_candidate.parent_atoms)
        if candidate_parent_count and other_parent_count and abs(candidate_parent_count - other_parent_count) > 2:
            continue
        if item.stability_penalty >= other.stability_penalty + 100:
            continue
        if len(item.name) - len(other.name) > 140:
            continue
        return True
    return False


def _has_contextual_biochem_heterofused_parent_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family not in (
        _ACYCLIC_SKELETON_PARENT_FAMILIES
        | {
            "retained-acid-ester-parent",
            "retained-amide-parent",
            "single-ring-parent",
        }
    ):
        return False
    if _is_complete_retained_senior_parent(candidate):
        return False
    if not any(stem in item.name for stem in _BIOCHEM_HETEROFUSED_RETAINED_STEMS):
        return False
    if candidate.family in _ACYCLIC_SKELETON_PARENT_FAMILIES | {"single-ring-parent"}:
        if "8-oxoguanin" not in item.name:
            return False
    if candidate.family in {"retained-acid-ester-parent", "retained-amide-parent"}:
        if "pterin-" not in item.name:
            return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family not in {
            "fused-retained-parent",
            "generated-fused-parent",
            "generated-hydro-fused-parent",
            "generated-partial-hydro-fused-parent",
            "hydro-fused-retained-parent",
            "retained-topology-replacement-fused-parent",
        }:
            continue
        if not candidate_has_complete_graph(other_candidate):
            continue
        if other_candidate.parent_atoms and len(other_candidate.parent_atoms) < 8:
            continue
        template_name = other_candidate.template_name or ""
        if not any(term in template_name for term in _BIOCHEM_HETEROFUSED_TEMPLATE_TERMS):
            continue
        if len(other.name) > 220 or len(other.name) - len(item.name) > 130:
            continue
        if other.stability_penalty >= item.stability_penalty + 80:
            continue
        return True
    return False


def _has_contextual_exocyclic_fused_single_ring_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family not in _EXOCYCLIC_FUSED_PARENT_FAMILIES:
        return False
    if not candidate_has_complete_graph(candidate):
        return False
    if not (item.name.startswith("(E)-") or item.name.startswith("(Z)-") or candidate.stereo_locants):
        return False
    if _has_stable_bis_indolinylidene_parent(item.name):
        return False
    template_name = candidate.template_name or ""
    if not any(term in template_name for term in _EXOCYCLIC_FUSED_TEMPLATE_TERMS):
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family not in {
            "acyclic-skeleton-with-ring-substituents",
            "functional-acyclic-skeleton-with-ring-substituents",
            "preferred-acyclic-skeleton-with-ring-substituents",
            "single-ring-parent",
        }:
            continue
        if not candidate_has_complete_graph(other_candidate):
            continue
        if other_candidate.family == "single-ring-parent":
            if not other_candidate.parent_atoms or len(other_candidate.parent_atoms) > 6:
                continue
            if other_candidate.render_rank is not None and other_candidate.render_rank > 3:
                continue
        if len(other.name) > 150 or len(other.name) - len(item.name) > 55:
            continue
        if other.stability_penalty >= item.stability_penalty + 70:
            continue
        return True
    return False


def _has_stable_bis_indolinylidene_parent(name: str) -> bool:
    return bool(
        re.search(r"indolin-\d+-ylidene\)oxindol\b", name)
        or re.search(r"indolin-\d+-ylidene\)indolin", name)
    )


def _has_contextual_biochem_fused_sibling_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family not in {
        "fused-retained-parent",
        "generated-fused-parent",
        "generated-hydro-fused-parent",
        "generated-partial-hydro-fused-parent",
        "hydro-fused-retained-parent",
        "retained-topology-replacement-fused-parent",
    }:
        return False
    if not candidate_has_complete_graph(candidate):
        return False
    template_name = candidate.template_name or ""
    if any(term in template_name for term in _BIOCHEM_HETEROFUSED_TEMPLATE_TERMS):
        return False
    if not any(stem in item.name for stem in _BIOCHEM_HETEROFUSED_RETAINED_STEMS):
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family not in {
            "fused-retained-parent",
            "generated-fused-parent",
            "generated-hydro-fused-parent",
            "generated-partial-hydro-fused-parent",
            "hydro-fused-retained-parent",
            "retained-topology-replacement-fused-parent",
        }:
            continue
        if not candidate_has_complete_graph(other_candidate):
            continue
        other_template_name = other_candidate.template_name or ""
        if not any(term in other_template_name for term in _BIOCHEM_HETEROFUSED_TEMPLATE_TERMS):
            continue
        if other_candidate.parent_atoms and len(other_candidate.parent_atoms) < 8:
            continue
        if len(other.name) > 220 or len(other.name) - len(item.name) > 130:
            continue
        if other.stability_penalty >= item.stability_penalty + 80:
            continue
        return True
    return False


def _has_contextual_generated_fused_same_atoms_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family not in {
        "fused-retained-parent",
        "hydro-fused-retained-parent",
        "oxido-fused-retained-parent",
        "retained-topology-replacement-fused-parent",
    }:
        return False
    if candidate.template_name in _GENERATED_FUSED_SAME_ATOMS_RETAINED_TEMPLATE_BLOCKLIST:
        return False
    parent_atoms = candidate.parent_atoms
    ring_atoms = candidate.ring_system_atoms
    if not parent_atoms and not ring_atoms:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "generated-fused-parent":
            continue
        if not candidate_has_complete_graph(other_candidate):
            continue
        if other_candidate.render_rank is None or other_candidate.render_rank > 3:
            continue
        same_parent = bool(parent_atoms and other_candidate.parent_atoms == parent_atoms)
        same_ring = bool(ring_atoms and other_candidate.ring_system_atoms == ring_atoms)
        if not same_parent and not same_ring:
            continue
        if len(other.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(other.name) - len(item.name) > 90:
            continue
        if other.stability_penalty >= item.stability_penalty + 80:
            continue
        return True
    return False


def _has_contextual_chain_dominant_ring_substituent_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family != "kekulized-generic-polycyclic-parent":
        return False
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "chain-dominant-acyclic-skeleton-with-ring-substituents":
            continue
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        if len(other.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(other.name) - len(item.name) > 90:
            continue
        if other.stability_penalty >= item.stability_penalty + 80:
            continue
        return True
    return False


def _has_contextual_kekulized_polycycle_parent_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family != "chain-dominant-acyclic-skeleton-with-ring-substituents":
        return False
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "kekulized-generic-polycyclic-parent":
            continue
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(item.name) - len(other.name) > 90:
            continue
        if item.stability_penalty >= other.stability_penalty + 80:
            continue
        return True
    return False


def _is_large_structured_kekulized_polycycle(candidate: NameCandidate) -> bool:
    if candidate.family != "kekulized-generic-polycyclic-parent":
        return False
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return False
    parent_count = len(candidate.parent_atoms)
    if parent_count < 14 or parent_count > 24:
        return False
    if candidate.render_rank is None or candidate.render_rank > 4:
        return False
    if candidate.suffixes:
        return False
    if len(candidate.stereo_locants) < parent_count:
        return False
    return True


def _has_contextual_large_structured_kekulized_polycycle_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family != "chain-dominant-acyclic-skeleton-with-ring-substituents":
        return False
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return False
    parent_count = len(candidate.parent_atoms)
    if parent_count < 8 or parent_count > 12:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        if not _is_large_structured_kekulized_polycycle(other.candidate):
            continue
        if len(other.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(other.name) - len(item.name) > 45:
            continue
        if other.stability_penalty > item.stability_penalty + 30:
            continue
        return True
    return False


def _has_contextual_chain_parent_alternative_for_large_kekulized_polycycle(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if not _is_large_structured_kekulized_polycycle(candidate):
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "chain-dominant-acyclic-skeleton-with-ring-substituents":
            continue
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        parent_count = len(other_candidate.parent_atoms)
        if parent_count < 8 or parent_count > 12:
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(item.name) - len(other.name) > 45:
            continue
        if other.stability_penalty > item.stability_penalty + 30:
            continue
        return True
    return False


def _has_contextual_retained_biochem_fused_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family not in {
        "generated-fused-parent",
        "generated-hydro-fused-parent",
        "generated-partial-hydro-fused-parent",
        "retained-topology-replacement-fused-parent",
    }:
        return False
    if not candidate_has_complete_graph(candidate):
        return False
    template_name = candidate.template_name or ""
    if not any(term in template_name for term in _BIOCHEM_GENERATED_FUSION_TEMPLATE_TERMS):
        return False
    if "dioxo" not in item.name:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family not in {
            "fused-retained-parent",
            "hydro-fused-retained-parent",
        }:
            continue
        if not candidate_has_complete_graph(other_candidate):
            continue
        other_template_name = other_candidate.template_name or ""
        if not any(term in other_template_name for term in _BIOCHEM_RETAINED_TEMPLATE_TERMS):
            continue
        if not any(term in other_template_name for term in {"guanin", "hypoxanthin"}):
            continue
        if other_candidate.parent_atoms and len(other_candidate.parent_atoms) < 8:
            continue
        if len(other.name) > 220 or len(other.name) - len(item.name) > 130:
            continue
        if other.stability_penalty >= item.stability_penalty + 80:
            continue
        return True
    return False


def _candidate_selection_contextual_penalty(item: _PreparedCandidate, prepared: list[_PreparedCandidate]) -> int:
    """Defer orientation-fragile aliases only when alternatives exist."""

    name = item.name
    retained_lactone_topology = _has_contextual_retained_lactone_topology_candidate(item, prepared)
    retained_cyclic_anhydride_topology = _has_contextual_retained_cyclic_anhydride_topology_candidate(
        item,
        prepared,
    )
    retained_topology_replacement_parent = (
        _has_contextual_fused_parent_alternative_for_retained_topology_replacement(item, prepared)
    )
    unstable_benzimidazolyl = _has_unstable_benzimidazolyl_orientation_artifact(name)
    unstable_replacement = _has_unstable_replacement_polycycle_alias(name)
    generic_polycycle = _has_contextual_generic_polycycle_fallback(name, item.candidate)
    fragile_sulfamoyl_benzene = _has_fragile_single_ring_sulfamoyl_benzene(name, item.candidate)
    fragile_fused_substituent_benzene = _has_fragile_single_ring_fused_substituent_benzene(name, item.candidate)
    fragile_fused_pyrazole_substituent = _has_fragile_fused_pyrazole_substituent_parent(name, item.candidate)
    fragile_fused_pyrazole_carbonyl = _has_fragile_fused_pyrazole_carbonyl_parent(name, item.candidate)
    fragile_retained_spiro_alias = _has_fragile_retained_spiro_alias(name, item.candidate)
    fragile_dibenzofuran_alias = _has_fragile_dibenzofuran_alias(name, item.candidate)
    fragile_isoindolinone_alias = _has_fragile_isoindolinone_alias(name)
    fragile_indolinone_alias = _has_fragile_indolinone_alias(name, item.candidate)
    fragile_azaindanone_alias = _has_fragile_azaindanone_alias(name, item.candidate)
    fragile_oxaanthracen_alias = _has_fragile_oxaanthracen_alias(name, item.candidate)
    fragile_pterin_substituent = _has_fragile_pterin_substituent_parent(name, item.candidate)
    fragile_pterin_chain = _has_fragile_pterin_substituent_chain_parent(name, item.candidate)
    fragile_bis_bicyclic_diazene = _has_fragile_bis_bicyclic_diazene_chain_parent(name, item.candidate)
    fragile_retained_amide = _has_fragile_retained_unsaturated_amide_parent(name, item.candidate)
    fragile_neutral_guanidine = _has_fragile_neutral_guanidine_parent(name, item.candidate)
    fragile_retained_stub = _has_fragile_whole_molecule_retained_stub(name, item.candidate)
    fragile_anthranil_alias = _has_fragile_anthranil_ketone_alias(name, item.candidate)
    fragile_retained_like_polycycle = _has_fragile_retained_like_generic_polycycle_alias(name, item.candidate)
    fragile_pentaazaind_alias = _has_fragile_pentaazaind_replacement_alias(name, item.candidate)
    fragile_diazenyl_pyridone = _has_fragile_diazenyl_pyridone_parent(name, item.candidate)
    fragile_imino_purine = _has_fragile_imino_purine_parent(name, item.candidate)
    fragile_imino_purine_alias = _has_fragile_imino_purine_alias(name, item.candidate)
    fragile_acyclic_oxoindolin = _has_fragile_acyclic_oxoindolin_substituent_parent(name, item.candidate)
    fragile_oxo_hydro_fused = _has_fragile_oxo_hydro_fused_alias(name, item.candidate)
    fragile_hydro_fused_bridge_alias = _has_fragile_hydro_fused_bridge_alias(name, item.candidate)
    fragile_imidazothiophene = _has_fragile_imidazothiophene_bridge_alias(name, item.candidate)
    fragile_ergosta_side_chain = _has_fragile_ergosta_terminal_side_chain_parent(name, item.candidate)
    missing_relative_stereo = _has_missing_relative_stereo_ring_candidate(name, item.candidate)
    locanted_stereo_miss = (
        not _is_complete_retained_senior_parent(item.candidate)
        and _has_unlocanted_atom_stereo_prefix(name)
        and _has_contextual_locanted_stereo_alternative(
            item,
            prepared,
            max_extra=80,
            max_length=220,
        )
    )
    fragile_bridge_h_ketone = _has_fragile_bridge_indicated_h_ketone_suffix(name)
    fragile_short_fused_alias = _has_fragile_short_fused_alias(name, item.candidate)
    fragile_decomposed_bis_bridge = _has_fragile_decomposed_bis_bridge_parent(name, item.candidate)
    fragile_component_fused_alias = _has_fragile_component_fused_alias(name, item.candidate)
    fragile_short_fusion_orientation = _has_fragile_short_fusion_orientation_alias(name, item.candidate)
    fragile_purine_tautomer_alias = _has_fragile_purine_tautomer_alias(name, item.candidate)
    fragile_terminal_azabicyclo_benzimidazole = _has_fragile_terminal_azabicyclo_benzimidazole_parent(
        name,
        item.candidate,
    )
    fragile_group14_hetero_ligand = _has_fragile_group14_hetero_carbon_ligand(item.candidate)
    generic_spiro_parent = _has_fragile_generic_spiro_parent(name, item.candidate)
    generated_spiro_functional_chain = _has_contextual_generated_spiro_functional_chain_alternative(item, prepared)
    complete_spiro_functional_chain = _has_contextual_complete_spiro_functional_chain_alternative(item, prepared)
    single_ring_lactam_parent = _has_contextual_single_ring_lactam_parent_alternative(item, prepared)
    fused_retained_parent_for_generic_spiro = (
        _has_contextual_fused_retained_parent_alternative_for_generic_spiro(item, prepared)
    )
    natural_topology_generic_polycycle = _has_contextual_complete_kekulized_polycycle_alternative(item, prepared)
    neutral_amine_shortcut = _has_contextual_neutral_amine_systematic_alternative(item, prepared)
    incomplete_shortcut_graph_parent = _has_contextual_complete_parent_alternative_for_incomplete_shortcut(
        item,
        prepared,
    )
    complete_fused_retained_amide_parent_bonus = _contextual_incomplete_retained_amide_alternative_bonus_for_complete_fused_parent(
        item,
        prepared,
    )
    high_confidence_fused_template_parent_bonus = _contextual_high_confidence_fused_template_parent_bonus(
        item,
        prepared,
    )
    named_template_pair_parent_bonus = _contextual_named_template_pair_parent_bonus(
        item,
        prepared,
    )
    low_confidence_fused_template_parent_penalty = _contextual_low_confidence_fused_template_parent_penalty(
        item,
        prepared,
    )
    broad_incomplete_shortcut_graph_parent = (
        _has_contextual_broad_complete_parent_alternative_for_incomplete_shortcut(item, prepared)
    )
    hydro_fused_systematic_parent = _has_contextual_hydro_fused_systematic_parent_alternative(item, prepared)
    lower_external_locant_polycycle = _has_contextual_lower_external_locant_polycycle_alternative(item, prepared)
    small_acyclic_ring_decomposition = _has_contextual_small_acyclic_complete_ring_parent_alternative(
        item,
        prepared,
    )
    comparable_complete_fused_parent = _has_contextual_comparable_complete_fused_parent_alternative(
        item,
        prepared,
    )
    large_fused_single_ring_suffix_parent = _has_contextual_single_ring_suffix_parent_alternative(
        item,
        prepared,
    )
    incomplete_senior_hydride_shortcut = _has_contextual_complete_parent_alternative_for_incomplete_senior_hydride_shortcut(
        item,
        prepared,
    )
    benzotriazole_retained_anchor = _has_contextual_benzotriazole_retained_anchor_alternative(item, prepared)
    retained_shortcut = _has_contextual_systematic_alternative_for_retained_shortcut(item, prepared)
    retained_ester_shortcut = _has_contextual_retained_acid_ester_alternative(item, prepared)
    acyclic_carbalkoxy_ester = _has_contextual_acyclic_carbalkoxy_retained_ester_alternative(item, prepared)
    retained_aryl_urea = _has_contextual_aryl_urea_alternative(item, prepared)
    biochem_heterofused_parent = _has_contextual_biochem_heterofused_parent_alternative(item, prepared)
    exocyclic_fused_single_ring = _has_contextual_exocyclic_fused_single_ring_alternative(item, prepared)
    biochem_fused_sibling = _has_contextual_biochem_fused_sibling_alternative(item, prepared)
    generated_fused_same_atoms = _has_contextual_generated_fused_same_atoms_alternative(item, prepared)
    chain_dominant_ring_substituent = _has_contextual_chain_dominant_ring_substituent_alternative(item, prepared)
    kekulized_polycycle_parent = _has_contextual_kekulized_polycycle_parent_alternative(item, prepared)
    chain_candidate_has_large_kekulized_alternative = (
        _has_contextual_large_structured_kekulized_polycycle_alternative(item, prepared)
    )
    large_kekulized_candidate_has_chain_alternative = _has_contextual_chain_parent_alternative_for_large_kekulized_polycycle(
        item,
        prepared,
    )
    retained_biochem_fused = _has_contextual_retained_biochem_fused_alternative(item, prepared)
    small_late_single_ring_parent = _has_contextual_early_six_member_single_ring_parent_alternative(item, prepared)
    aza_alias_penalty = _unstable_aza_polycycle_alias_penalty(name)
    spiro_penalty = _unstable_component_spiro_alias_penalty(name, item.candidate)
    if not (
        retained_lactone_topology
        or retained_cyclic_anhydride_topology
        or retained_topology_replacement_parent
        or unstable_benzimidazolyl
        or unstable_replacement
        or generic_polycycle
        or fragile_sulfamoyl_benzene
        or fragile_fused_substituent_benzene
        or fragile_fused_pyrazole_substituent
        or fragile_fused_pyrazole_carbonyl
        or fragile_retained_spiro_alias
        or fragile_dibenzofuran_alias
        or fragile_isoindolinone_alias
        or fragile_indolinone_alias
        or fragile_azaindanone_alias
        or fragile_oxaanthracen_alias
        or fragile_pterin_substituent
        or fragile_pterin_chain
        or fragile_bis_bicyclic_diazene
        or fragile_retained_amide
        or fragile_neutral_guanidine
        or fragile_retained_stub
        or fragile_anthranil_alias
        or fragile_retained_like_polycycle
        or fragile_pentaazaind_alias
        or fragile_diazenyl_pyridone
        or fragile_imino_purine
        or fragile_imino_purine_alias
        or fragile_acyclic_oxoindolin
        or fragile_oxo_hydro_fused
        or fragile_hydro_fused_bridge_alias
        or fragile_imidazothiophene
        or fragile_ergosta_side_chain
        or missing_relative_stereo
        or locanted_stereo_miss
        or fragile_bridge_h_ketone
        or fragile_short_fused_alias
        or fragile_decomposed_bis_bridge
        or fragile_component_fused_alias
        or fragile_short_fusion_orientation
        or fragile_purine_tautomer_alias
        or fragile_terminal_azabicyclo_benzimidazole
        or fragile_group14_hetero_ligand
        or generic_spiro_parent
        or generated_spiro_functional_chain
        or complete_spiro_functional_chain
        or single_ring_lactam_parent
        or fused_retained_parent_for_generic_spiro
        or natural_topology_generic_polycycle
        or neutral_amine_shortcut
        or incomplete_shortcut_graph_parent
        or complete_fused_retained_amide_parent_bonus
        or high_confidence_fused_template_parent_bonus
        or named_template_pair_parent_bonus
        or low_confidence_fused_template_parent_penalty
        or broad_incomplete_shortcut_graph_parent
        or hydro_fused_systematic_parent
        or lower_external_locant_polycycle
        or small_acyclic_ring_decomposition
        or comparable_complete_fused_parent
        or large_fused_single_ring_suffix_parent
        or incomplete_senior_hydride_shortcut
        or benzotriazole_retained_anchor
        or retained_shortcut
        or retained_ester_shortcut
        or acyclic_carbalkoxy_ester
        or retained_aryl_urea
        or biochem_heterofused_parent
        or exocyclic_fused_single_ring
        or biochem_fused_sibling
        or generated_fused_same_atoms
        or chain_dominant_ring_substituent
        or kekulized_polycycle_parent
        or chain_candidate_has_large_kekulized_alternative
        or large_kekulized_candidate_has_chain_alternative
        or retained_biochem_fused
        or small_late_single_ring_parent
        or aza_alias_penalty
        or spiro_penalty
    ):
        return 0

    penalty = 0
    if retained_lactone_topology:
        penalty -= _RETAINED_LACTONE_TOPOLOGY_BONUS
    if retained_cyclic_anhydride_topology:
        penalty -= _RETAINED_CYCLIC_ANHYDRIDE_TOPOLOGY_BONUS
    if retained_topology_replacement_parent:
        penalty -= 140
    has_graph_parent = _has_contextual_graph_parent_alternative(item, prepared)
    if has_graph_parent and unstable_benzimidazolyl:
        penalty += 85
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
    if fragile_fused_pyrazole_substituent and _has_contextual_named_graph_parent_alternative(
        item, prepared, max_extra=20, max_length=90
    ):
        penalty += 14
    if fragile_fused_pyrazole_carbonyl and _has_contextual_generic_bicyclic_alternative(
        item, prepared, max_extra=25, max_length=75
    ):
        penalty += 24
    if fragile_retained_spiro_alias and _has_contextual_component_spiro_alternative(
        item, prepared, max_extra=55, max_length=130
    ):
        penalty += 60
    if fragile_dibenzofuran_alias and _has_contextual_oxafluoren_alternative(item, prepared):
        penalty += 40
    if fragile_isoindolinone_alias and _has_contextual_isoindolinone_alternative(item, prepared):
        penalty += 90
    if fragile_indolinone_alias and _has_contextual_named_graph_parent_alternative(
        item, prepared, max_extra=35, max_length=90
    ):
        penalty += 60
    if fragile_azaindanone_alias and _has_contextual_named_graph_parent_alternative(
        item, prepared, max_extra=45, max_length=90
    ):
        penalty += 70
    if fragile_oxaanthracen_alias and _has_contextual_oxo_bridge_parent_alternative(
        item, prepared, max_extra=80, max_length=160
    ):
        penalty += 80
    if fragile_pterin_substituent and _has_contextual_named_graph_parent_alternative(
        item, prepared, max_extra=30, max_length=100
    ):
        penalty += 70
    if fragile_pterin_chain and _has_contextual_pterin_parent_alternative(
        item, prepared, max_extra=45, max_length=150
    ):
        penalty += 190
    if fragile_bis_bicyclic_diazene and _has_contextual_generic_bridge_parent_alternative(
        item, prepared, max_extra=90, max_length=190
    ):
        penalty += 70
    if fragile_retained_amide and _has_contextual_acyclic_parent_alternative(
        item, prepared, max_extra=20, max_length=90
    ):
        penalty += 16
    if fragile_neutral_guanidine and _has_contextual_ium_parent_alternative(item, prepared):
        penalty += 10
    if fragile_retained_stub and _has_contextual_systematic_parent_alternative(
        item, prepared, max_extra=90, max_length=150
    ):
        penalty += 90
    if fragile_anthranil_alias and _has_contextual_named_graph_parent_alternative(
        item, prepared, max_extra=20, max_length=70
    ):
        penalty += 40
    if fragile_retained_like_polycycle and _has_contextual_named_graph_parent_alternative(
        item, prepared, max_extra=40, max_length=90
    ):
        penalty += 60
    if fragile_retained_like_polycycle and _has_contextual_generic_bridge_parent_alternative(
        item, prepared, max_extra=115, max_length=180
    ):
        penalty += 110
    if fragile_pentaazaind_alias and _has_contextual_generic_bridge_parent_alternative(
        item, prepared, max_extra=40, max_length=90
    ):
        penalty += 45
    if fragile_diazenyl_pyridone and _has_contextual_acyclic_parent_alternative(
        item, prepared, max_extra=10, max_length=70
    ):
        penalty += 20
    if fragile_imino_purine and _has_contextual_named_graph_parent_alternative(
        item, prepared, max_extra=70, max_length=130
    ):
        penalty += 70
    if fragile_imino_purine_alias and _has_contextual_named_graph_parent_alternative(
        item, prepared, max_extra=35, max_length=115
    ):
        penalty += 32
    if fragile_acyclic_oxoindolin and _has_contextual_named_graph_parent_alternative(
        item, prepared, max_extra=45, max_length=90
    ):
        penalty += 35
    if fragile_oxo_hydro_fused and _has_contextual_generic_bridge_parent_alternative(
        item, prepared, max_extra=70, max_length=125
    ):
        penalty += 95
    if fragile_hydro_fused_bridge_alias and _has_contextual_generic_bridge_parent_alternative(
        item, prepared, max_extra=8, max_length=55
    ):
        penalty += 10
    if fragile_imidazothiophene and _has_contextual_generic_bridge_parent_alternative(
        item, prepared, max_extra=10, max_length=170
    ):
        penalty += 20
    if fragile_ergosta_side_chain and _has_contextual_natural_topology_alternative(
        item, prepared, stems=("cholest", "cholesta"), max_extra=25, max_length=90
    ):
        penalty += 20
    if fragile_bridge_h_ketone and _has_contextual_oxo_bridge_parent_alternative(item, prepared):
        penalty += 80
    if fragile_short_fused_alias and _has_contextual_generic_bridge_parent_alternative(
        item, prepared, max_extra=80, max_length=140
    ):
        penalty += 95
    if fragile_short_fused_alias and _has_contextual_systematic_parent_alternative(
        item, prepared, max_extra=45, max_length=110
    ):
        penalty += 60
    if fragile_decomposed_bis_bridge and _has_contextual_generic_bridge_parent_alternative(
        item, prepared, max_extra=75, max_length=180
    ):
        penalty += 80
    if fragile_component_fused_alias and _has_contextual_bridge_descriptor_alternative(
        item, prepared, max_extra=150, max_length=260
    ):
        penalty += 150
    if fragile_short_fusion_orientation and _has_contextual_generated_exact_fusion_alternative(
        item, prepared, max_extra=12, max_length=90
    ):
        penalty += 36
    if fragile_purine_tautomer_alias and _has_contextual_generated_exact_fusion_alternative(
        item, prepared, max_extra=35, max_length=120
    ):
        penalty += 48
    if fragile_terminal_azabicyclo_benzimidazole and _has_contextual_integrated_benzimidazole_carboxamide_alternative(
        item,
        prepared,
    ):
        penalty += 125
    if fragile_group14_hetero_ligand and _has_contextual_group14_systematic_alternative(item, prepared):
        penalty += 140
    if generic_spiro_parent and _has_contextual_spiro_functional_chain_alternative(item, prepared):
        penalty += 125
    if generic_spiro_parent and complete_spiro_functional_chain:
        penalty += 160
    if generated_spiro_functional_chain:
        penalty += 280
    if single_ring_lactam_parent:
        penalty += 330
    if fused_retained_parent_for_generic_spiro:
        penalty += 95
    if natural_topology_generic_polycycle:
        penalty += 90
    if neutral_amine_shortcut:
        penalty += 140
    if incomplete_shortcut_graph_parent:
        penalty += 140
    if complete_fused_retained_amide_parent_bonus:
        penalty -= complete_fused_retained_amide_parent_bonus
    if high_confidence_fused_template_parent_bonus:
        penalty -= high_confidence_fused_template_parent_bonus
    if named_template_pair_parent_bonus:
        penalty -= named_template_pair_parent_bonus
    if low_confidence_fused_template_parent_penalty:
        penalty += low_confidence_fused_template_parent_penalty
    if broad_incomplete_shortcut_graph_parent:
        penalty += 125
    if hydro_fused_systematic_parent:
        penalty += 80
    if lower_external_locant_polycycle:
        penalty += 18
    if small_acyclic_ring_decomposition:
        penalty += 90
    if comparable_complete_fused_parent:
        penalty += 150
    if large_fused_single_ring_suffix_parent:
        penalty += 180
    if incomplete_senior_hydride_shortcut:
        penalty += 150
    if benzotriazole_retained_anchor:
        penalty += 45
    if missing_relative_stereo and _has_contextual_relative_stereo_alternative(item, prepared, max_extra=35):
        penalty += 36
    if retained_shortcut:
        penalty += 140
    if retained_ester_shortcut:
        penalty += 160
    if acyclic_carbalkoxy_ester:
        penalty += 180
    if retained_aryl_urea:
        penalty += 90
    if aza_alias_penalty and _has_contextual_graph_parent_alternative(item, prepared, max_extra=90, max_length=180):
        penalty += 24
    if spiro_penalty and _has_contextual_graph_parent_alternative(item, prepared, max_extra=35):
        penalty += 12
    if (
        locanted_stereo_miss
        and complete_fused_retained_amide_parent_bonus < 600
        and not _allow_generated_fused_unlocanted_stereo_candidate(item, prepared)
        and not _allow_high_confidence_imidazo45a_unlocanted_stereo_candidate(item, prepared)
        and not _allow_high_confidence_retained_template_unlocanted_stereo_candidate(item, prepared)
    ):
        penalty = max(penalty, 220)
    if biochem_heterofused_parent:
        penalty += 120
    if exocyclic_fused_single_ring:
        penalty += 95
    if biochem_fused_sibling:
        penalty += 90
    if generated_fused_same_atoms:
        penalty += 140
    if chain_dominant_ring_substituent:
        penalty += 150
    if kekulized_polycycle_parent:
        penalty -= 260
    if chain_candidate_has_large_kekulized_alternative:
        penalty += 300
    if large_kekulized_candidate_has_chain_alternative:
        penalty -= 370
    if retained_biochem_fused:
        penalty += 80
    if small_late_single_ring_parent:
        penalty += 72
    return penalty


def _candidate_selection_contextual_bonus(item: _PreparedCandidate, prepared: list[_PreparedCandidate]) -> int:
    bonus = 0
    if _has_contextual_larger_single_ring_parent_alternative(item, prepared):
        bonus += 40
    return bonus


_LOCANTED_STEREO_GRAPH_RING_PARENT_FAMILIES = {
    "functional-generic-bicyclic-parent",
    "fused-retained-parent",
    "generic-bicyclic-parent",
    "hydro-fused-retained-parent",
    "preferred-generic-bicyclic-parent",
    "retained-topology-replacement-fused-parent",
    "skeletal-replacement-fused-parent",
}


def _allow_generated_fused_unlocanted_stereo_candidate(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family != "generated-fused-parent":
        return False
    if not candidate.parent_atoms:
        return False
    if candidate.render_rank is None or candidate.render_rank > 4:
        return False
    if not _has_contextual_locanted_stereo_family_alternative(
        item,
        prepared,
        _LOCANTED_STEREO_GRAPH_RING_PARENT_FAMILIES,
        max_extra=80,
        max_length=220,
    ):
        return False
    return not _has_contextual_locanted_stereo_family_alternative(
        item,
        prepared,
        _ACYCLIC_SKELETON_PARENT_FAMILIES,
        max_extra=100,
        max_length=260,
    )


def _allow_high_confidence_imidazo45a_unlocanted_stereo_candidate(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family != "generated-fused-parent":
        return False
    if candidate.template_name != "imidazo[4,5-a]benzene":
        return False
    if candidate.render_rank is None or candidate.render_rank > 2:
        return False
    if len(candidate.parent_atoms) < 9:
        return False
    return bool(_contextual_high_confidence_fused_template_parent_bonus(item, prepared))


def _allow_high_confidence_retained_template_unlocanted_stereo_candidate(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family not in _HIGH_CONFIDENCE_FUSED_TEMPLATE_PARENT_FAMILIES:
        return False
    if candidate.template_name not in _VERY_HIGH_CONFIDENCE_RETAINED_FUSED_TEMPLATE_PARENT_NAMES:
        return False
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return False
    if len(candidate.parent_atoms) < 9:
        return False
    if candidate.render_rank is not None and candidate.render_rank > 2:
        return False
    return bool(_contextual_high_confidence_fused_template_parent_bonus(item, prepared))


def _has_contextual_locanted_stereo_family_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
    families: set[str],
    *,
    max_extra: int,
    max_length: int,
) -> bool:
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        if other.candidate.family not in families:
            continue
        if _locanted_atom_stereo_descriptor_count(other.name) == 0:
            continue
        if len(other.name) > max_length or len(other.name) - len(item.name) > max_extra:
            continue
        if other.stability_penalty >= item.stability_penalty + 60:
            continue
        return True
    return False


def _has_contextual_systematic_alternative_for_retained_shortcut(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    if item.candidate.family not in {"amino-acid-retained-parent", "whole-molecule-retained-parent"}:
        return False
    if _is_complete_retained_senior_parent(item.candidate):
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        if other.candidate.family not in _CONTEXTUAL_SYSTEMATIC_ALTERNATIVE_FAMILIES:
            continue
        if len(other.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if other.stability_penalty >= item.stability_penalty + 60:
            continue
        return True
    return False


def _has_contextual_neutral_amine_systematic_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family != "neutral-amine-parent":
        return False
    if candidate_has_complete_graph(candidate):
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "acyclic-parent":
            continue
        if not candidate_has_complete_graph(other_candidate):
            continue
        if len(other.name) < len(item.name):
            continue
        if len(other.name) > 160 or len(other.name) - len(item.name) > 80:
            continue
        if other.stability_penalty >= item.stability_penalty + 60:
            continue
        return True
    return False


def _has_contextual_complete_kekulized_polycycle_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family != "natural-product-topology-parent":
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "kekulized-generic-polycyclic-parent":
            continue
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        if len(other.name) > _DEFAULT_MAX_NAME_LENGTH or len(other.name) - len(item.name) > 150:
            continue
        if other.stability_penalty >= item.stability_penalty + 80:
            continue
        return True
    return False


_INCOMPLETE_SHORTCUT_GRAPH_PARENT_FAMILIES = frozenset(
    {
        "component-spiro-parent",
        "glutarimide-retained-parent",
        "retained-acid-ester-parent",
    }
)

_BROAD_INCOMPLETE_SHORTCUT_GRAPH_PARENT_FAMILIES = frozenset(
    {
        "component-spiro-parent",
        "glutarimide-retained-parent",
        "retained-acid-ester-parent",
        "retained-amide-parent",
    }
)

_COMPLETE_SHORTCUT_GRAPH_PARENT_ALTERNATIVE_FAMILIES = frozenset(
    {
        "acyclic-parent",
        "acyclic-skeleton-with-ring-substituents",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "functional-acyclic-skeleton-with-ring-substituents",
        "functional-generic-bicyclic-parent",
        "fused-retained-parent",
        "generated-fused-parent",
        "generic-bicyclic-parent",
        "kekulized-generic-polycyclic-parent",
        "preferred-acyclic-skeleton-with-ring-substituents",
        "retained-topology-replacement-fused-parent",
    }
)

_BROAD_COMPLETE_SHORTCUT_GRAPH_PARENT_ALTERNATIVE_FAMILIES = frozenset(
    {
        "acyclic-parent",
        "acyclic-skeleton-with-ring-substituents",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "functional-acyclic-skeleton-with-ring-substituents",
        "functional-generic-bicyclic-parent",
        "fused-retained-parent",
        "generated-fused-parent",
        "generic-bicyclic-parent",
        "kekulized-generic-polycyclic-parent",
        "preferred-acyclic-skeleton-with-ring-substituents",
        "retained-topology-replacement-fused-parent",
        "single-ring-parent",
    }
)

_RETAINED_AMIDE_COMPLETE_FUSED_ALTERNATIVE_FAMILIES = frozenset(
    {
        "fused-retained-parent",
        "generated-fused-parent",
    }
)
_RETAINED_AMIDE_GENERIC_ACYL_HIGH_BONUS_TEMPLATE_BLOCKLIST = frozenset(
    {
        "1,4-benzodioxane",
        "fluoren",
    }
)
_RETAINED_AMIDE_GENERIC_ACYL_HIGH_BONUS_CONTEXT_TEMPLATE_BLOCKLIST = frozenset(
    {
        "1,4-benzodioxane",
    }
)


def _has_contextual_complete_parent_alternative_for_incomplete_shortcut(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family not in _INCOMPLETE_SHORTCUT_GRAPH_PARENT_FAMILIES:
        return False
    if candidate_has_complete_graph(candidate):
        return False
    if (
        candidate.family == "retained-acid-ester-parent"
        and candidate.source_family.endswith(":stable-retained")
    ):
        return False
    if candidate.family == "retained-acid-ester-parent" and _has_complete_non_acyclic_parent_alternative(
        item,
        prepared,
    ):
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family not in _COMPLETE_SHORTCUT_GRAPH_PARENT_ALTERNATIVE_FAMILIES:
            continue
        if candidate.family == "retained-acid-ester-parent" and other_candidate.family not in (
            _ACYCLIC_SKELETON_PARENT_FAMILIES | {"acyclic-parent"}
        ):
            continue
        if not candidate_has_complete_graph(other_candidate) or len(other_candidate.parent_atoms) < 8:
            continue
        if candidate.family == "glutarimide-retained-parent" and len(other.name) + 20 < len(item.name):
            continue
        if (
            candidate.family == "glutarimide-retained-parent"
            and _candidate_selection_broad_graph_parent_penalty(other, prepared) >= 70
        ):
            continue
        if len(other.name) > 220 or len(other.name) - len(item.name) > 90:
            continue
        if other.stability_penalty >= item.stability_penalty + 70:
            continue
        return True
    return False


def _contextual_incomplete_retained_amide_alternative_bonus_for_complete_fused_parent(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> int:
    candidate = item.candidate
    if candidate.family not in _RETAINED_AMIDE_COMPLETE_FUSED_ALTERNATIVE_FAMILIES:
        return 0
    if candidate.template_name in (
        _RETAINED_LACTONE_TEMPLATE_NAMES | _RETAINED_CYCLIC_ANHYDRIDE_TEMPLATE_NAMES
    ):
        return 0
    if not candidate_has_complete_graph(candidate) or len(candidate.parent_atoms) < 8:
        return 0
    if candidate.suffixes:
        return 0
    if candidate.render_rank is not None and candidate.render_rank > 3:
        return 0
    template_name = candidate.template_name or ""
    bonus = 0
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "retained-amide-parent":
            continue
        if other_candidate.source_family in {"ring-carboxamide:selenium", "ring-carboxamide:tellurium"}:
            continue
        if candidate_has_complete_graph(other_candidate):
            continue
        if len(item.name) > 260 or len(item.name) - len(other.name) > 135:
            continue
        if item.stability_penalty >= other.stability_penalty + 90:
            continue
        bonus = max(bonus, 220)
        if other_candidate.source_family == "generic-acyl-amide":
            if template_name in _RETAINED_AMIDE_GENERIC_ACYL_HIGH_BONUS_TEMPLATE_BLOCKLIST:
                continue
            if _has_contextual_complete_template_parent_alternative(
                item,
                prepared,
                _RETAINED_AMIDE_GENERIC_ACYL_HIGH_BONUS_CONTEXT_TEMPLATE_BLOCKLIST,
            ):
                continue
            bonus = max(bonus, 1100)
        elif (
            other_candidate.source_family == "ring-carboxamide"
            and template_name in {"1,3-benzodioxole", "indan"}
        ):
            bonus = max(bonus, 400)
    return bonus


def _contextual_high_confidence_fused_template_parent_bonus(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> int:
    candidate = item.candidate
    if candidate.family not in _HIGH_CONFIDENCE_FUSED_TEMPLATE_PARENT_FAMILIES:
        return 0
    if candidate.template_name not in _HIGH_CONFIDENCE_FUSED_TEMPLATE_PARENT_NAMES:
        return 0
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return 0
    if candidate.render_rank is not None and candidate.render_rank > 8:
        return 0
    if len(item.name) > 280:
        return 0
    if (
        candidate.family == "retained-topology-replacement-fused-parent"
        and candidate.template_name == "tetralin"
        and (
            _has_contextual_same_template_static_retained_parent(item, prepared)
            or _has_contextual_large_acyclic_parent_for_retained_topology_template(item, prepared)
        )
    ):
        return 0
    competitor_families = None
    if candidate.template_name == "imidazo[4,5-a]benzene":
        competitor_families = _IMIDAZO45A_TEMPLATELESS_COMPETITOR_FAMILIES
    if not _has_contextual_template_less_parent_competitor_for_fused_template(
        item,
        prepared,
        allowed_families=competitor_families,
    ):
        return 0
    if candidate.template_name == "imidazo[4,5-a]benzene":
        return _HIGH_CONFIDENCE_IMIDAZO45A_FUSED_TEMPLATE_PARENT_BONUS
    if candidate.template_name in _VERY_HIGH_CONFIDENCE_RETAINED_FUSED_TEMPLATE_PARENT_NAMES:
        return _VERY_HIGH_CONFIDENCE_RETAINED_FUSED_TEMPLATE_PARENT_BONUS
    return _HIGH_CONFIDENCE_FUSED_TEMPLATE_PARENT_BONUS


def _has_contextual_template_less_parent_competitor_for_fused_template(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
    *,
    allowed_families: frozenset[str] | None = None,
) -> bool:
    parent_count = len(item.candidate.parent_atoms)
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.template_name:
            continue
        if allowed_families is not None and other_candidate.family not in allowed_families:
            continue
        if item.candidate.template_name == "imidazo[4,5-a]benzene" and not _is_imidazo45a_template_less_competitor(
            other_candidate
        ):
            continue
        if other_candidate.family in _HIGH_CONFIDENCE_FUSED_TEMPLATE_PARENT_FAMILIES and candidate_has_complete_graph(
            other_candidate
        ):
            continue
        if (
            other.stability_penalty
            > item.stability_penalty + _HIGH_CONFIDENCE_FUSED_TEMPLATE_COMPETITOR_STABILITY_MARGIN
        ):
            continue
        if parent_count and len(other_candidate.parent_atoms) > parent_count + 4:
            continue
        if len(item.name) - len(other.name) > 150:
            continue
        return True
    return False


def _is_imidazo45a_template_less_competitor(candidate: NameCandidate) -> bool:
    if candidate.family in _IMIDAZO45A_LARGE_ACYCLIC_COMPETITOR_FAMILIES:
        return len(candidate.parent_atoms) >= 9
    if candidate.family == "chain-dominant-acyclic-skeleton-with-ring-substituents":
        return len(candidate.parent_atoms) >= 7
    return candidate.family in {
        "generic-bicyclic-parent",
        "generic-spiro-parent",
        "retained-acid-ester-parent",
        "spiro-functional-chain-substituent",
    }


def _has_contextual_same_template_static_retained_parent(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "fused-retained-parent":
            continue
        if other_candidate.source_family != "static-retained":
            continue
        if other_candidate.template_name != candidate.template_name:
            continue
        if not candidate_has_complete_graph(other_candidate):
            continue
        if other.stability_penalty > item.stability_penalty + 80:
            continue
        return True
    return False


def _has_contextual_large_acyclic_parent_for_retained_topology_template(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return False
    parent_count = len(candidate.parent_atoms)
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family not in _ACYCLIC_RING_DECOMPOSITION_PARENT_FAMILIES:
            continue
        if not other_candidate.parent_atoms or not candidate_has_complete_graph(other_candidate):
            continue
        if (
            other_candidate.family != "charged-ring-acyclic-skeleton-with-ring-substituents"
            and len(other_candidate.parent_atoms) < parent_count + 5
        ):
            continue
        if other.stability_penalty > item.stability_penalty + 120:
            continue
        return True
    return False


def _has_contextual_complete_named_fused_template_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
    template_names: frozenset[str],
) -> bool:
    parent_count = len(item.candidate.parent_atoms)
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family not in _HIGH_CONFIDENCE_FUSED_TEMPLATE_PARENT_FAMILIES:
            continue
        if other_candidate.template_name not in template_names:
            continue
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        if parent_count and len(other_candidate.parent_atoms) + 1 < parent_count:
            continue
        if other.stability_penalty > item.stability_penalty:
            continue
        return True
    return False


def _contextual_named_template_pair_parent_bonus(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> int:
    candidate = item.candidate
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return 0
    if candidate.render_rank is not None and candidate.render_rank > 8:
        return 0
    if len(item.name) > 280:
        return 0
    if (
        candidate.family == "skeletal-replacement-fused-parent"
        and candidate.template_name in {"fluoren", "fluorene"}
        and _has_contextual_complete_named_fused_template_alternative(
            item,
            prepared,
            frozenset({"dibenzofuran"}),
        )
    ):
        return _NAMED_TEMPLATE_PAIR_PARENT_BONUS
    if (
        candidate.family == "generated-hydro-fused-parent"
        and candidate.template_name in {"pyrido[4,3-d]imidazole", "pyridino[4,3-d]imidazole"}
        and _has_contextual_complete_named_fused_template_alternative(
            item,
            prepared,
            frozenset({"benzo[1,2-d]pyrazole"}),
        )
    ):
        return _NAMED_TEMPLATE_PAIR_PARENT_BONUS
    if (
        candidate.family == "retained-topology-replacement-fused-parent"
        and candidate.source_family == "opsin-retained-ring-cml"
        and candidate.template_name == "tetralin"
        and _has_contextual_complete_named_fused_template_alternative(
            item,
            prepared,
            frozenset({"1,4-benzodioxane"}),
        )
    ):
        return _TETRALIN_TO_BENZODIOXANE_TEMPLATE_PAIR_BONUS
    return 0


def _contextual_low_confidence_fused_template_parent_penalty(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> int:
    candidate = item.candidate
    alternative_families = _LOW_CONFIDENCE_FUSED_TEMPLATE_TO_TEMPLATELESS_FAMILIES.get(
        candidate.template_name
    )
    if not alternative_families:
        return 0
    if candidate.family not in _LOW_CONFIDENCE_FUSED_TEMPLATE_PARENT_FAMILIES:
        return 0
    if candidate.render_rank is not None and candidate.render_rank > 8:
        return 0
    if not _has_contextual_template_less_family_alternative_for_low_confidence_template(
        item,
        prepared,
        alternative_families,
    ):
        return 0
    return _LOW_CONFIDENCE_FUSED_TEMPLATE_PARENT_PENALTY


def _has_contextual_template_less_family_alternative_for_low_confidence_template(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
    alternative_families: frozenset[str],
) -> bool:
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.template_name:
            continue
        if other_candidate.family not in alternative_families:
            continue
        if other.stability_penalty > item.stability_penalty + 140:
            continue
        if len(other.name) > 280:
            continue
        if len(other.name) - len(item.name) > 160:
            continue
        return True
    return False


def _has_contextual_complete_template_parent_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
    template_names: frozenset[str],
) -> bool:
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family not in _RETAINED_AMIDE_COMPLETE_FUSED_ALTERNATIVE_FAMILIES:
            continue
        if other_candidate.template_name not in template_names:
            continue
        if not candidate_has_complete_graph(other_candidate) or len(other_candidate.parent_atoms) < 8:
            continue
        if other.stability_penalty >= item.stability_penalty + 90:
            continue
        return True
    return False


def _has_contextual_broad_complete_parent_alternative_for_incomplete_shortcut(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family not in _BROAD_INCOMPLETE_SHORTCUT_GRAPH_PARENT_FAMILIES:
        return False
    if candidate.source_family in {"ring-carboxamide:selenium", "ring-carboxamide:tellurium"}:
        return False
    if candidate_has_complete_graph(candidate):
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family not in _BROAD_COMPLETE_SHORTCUT_GRAPH_PARENT_ALTERNATIVE_FAMILIES:
            continue
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        if candidate.family == "retained-amide-parent":
            if other_candidate.family == "single-ring-parent" and not other_candidate.suffixes:
                continue
            if (
                other_candidate.family in (_ACYCLIC_SKELETON_PARENT_FAMILIES | {"acyclic-parent"})
                and len(other_candidate.parent_atoms) < 7
            ):
                continue
        if len(other.name) > 260 or len(other.name) - len(item.name) > 135:
            continue
        if other.stability_penalty >= item.stability_penalty + 90:
            continue
        return True
    return False


def _has_contextual_hydro_fused_systematic_parent_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family not in {"hydro-fused-retained-parent", "generated-hydro-fused-parent"}:
        return False
    parent_count = len(candidate.parent_atoms)
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family not in _HYDRO_FUSED_SYSTEMATIC_ALTERNATIVE_FAMILIES:
            continue
        if not other_candidate.parent_atoms or not candidate_has_complete_graph(other_candidate):
            continue
        if parent_count and len(other_candidate.parent_atoms) + 2 < parent_count:
            continue
        if other_candidate.render_rank is not None and other_candidate.render_rank > 5:
            continue
        if len(other.name) > _DEFAULT_MAX_NAME_LENGTH or len(other.name) - len(item.name) > 130:
            continue
        if other.stability_penalty >= item.stability_penalty + 90:
            continue
        return True
    return False


def _has_contextual_small_acyclic_complete_ring_parent_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family not in _ACYCLIC_RING_DECOMPOSITION_PARENT_FAMILIES:
        return False
    if len(candidate.parent_atoms) > 3:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family not in _ACYCLIC_INTEGRATED_RING_PARENT_ALTERNATIVE_FAMILIES:
            continue
        if not other_candidate.parent_atoms or not candidate_has_complete_graph(other_candidate):
            continue
        if other_candidate.render_rank is not None and other_candidate.render_rank > 4:
            continue
        if len(other.name) > _DEFAULT_MAX_NAME_LENGTH or len(other.name) - len(item.name) > 130:
            continue
        if other.stability_penalty >= item.stability_penalty + 80:
            continue
        return True
    return False


def _has_contextual_comparable_complete_fused_parent_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family not in _ACYCLIC_SKELETON_PARENT_FAMILIES:
        return False
    if not candidate.parent_atoms:
        return False
    parent_count = len(candidate.parent_atoms)
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family not in _COMPARABLE_COMPLETE_FUSED_PARENT_ALTERNATIVE_FAMILIES:
            continue
        if not other_candidate.parent_atoms or not candidate_has_complete_graph(other_candidate):
            continue
        if len(other_candidate.parent_atoms) + 2 < parent_count:
            continue
        return True
    return False


def _has_contextual_single_ring_suffix_parent_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family not in _LARGE_FUSED_PARENT_WITH_SINGLE_RING_SUFFIX_FAMILIES:
        return False
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return False
    if len(candidate.parent_atoms) < 8:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family not in _SINGLE_RING_SUFFIX_PARENT_ALTERNATIVE_FAMILIES:
            continue
        if not other_candidate.parent_atoms or not candidate_has_complete_graph(other_candidate):
            continue
        if len(other_candidate.parent_atoms) > 6:
            continue
        if not other_candidate.suffixes:
            continue
        return True
    return False


def _has_complete_non_acyclic_parent_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family in _ACYCLIC_SKELETON_PARENT_FAMILIES | {"acyclic-parent"}:
            continue
        if other_candidate.family not in _COMPLETE_SHORTCUT_GRAPH_PARENT_ALTERNATIVE_FAMILIES:
            continue
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        if len(other.name) > 220 or len(other.name) - len(item.name) > 90:
            continue
        if other.stability_penalty >= item.stability_penalty + 70:
            continue
        return True
    return False


_INCOMPLETE_SENIOR_HYDRIDE_SHORTCUT_FAMILIES = frozenset(
    {
        "azanium-parent",
        "guanidine-retained-parent",
        "pnictogen-hydride-parent",
    }
)

_COMPLETE_SENIOR_HYDRIDE_ALTERNATIVE_FAMILIES = frozenset(
    {
        "acyclic-parent",
        "acyclic-skeleton-with-ring-substituents",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "functional-acyclic-skeleton-with-ring-substituents",
        "phosphorus-acid-parent",
        "positive-single-ring-parent",
        "preferred-acyclic-skeleton-with-ring-substituents",
        "single-ring-parent",
    }
)


def _has_contextual_complete_parent_alternative_for_incomplete_senior_hydride_shortcut(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family not in _INCOMPLETE_SENIOR_HYDRIDE_SHORTCUT_FAMILIES:
        return False
    if candidate_has_complete_graph(candidate):
        return False
    if _has_same_priority_incomplete_parent_blocker(item, prepared):
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family not in _COMPLETE_SENIOR_HYDRIDE_ALTERNATIVE_FAMILIES:
            continue
        if not candidate_has_complete_graph(other_candidate) or len(other_candidate.parent_atoms) < 5:
            continue
        if candidate.family == "azanium-parent" and other_candidate.family == "phosphorus-acid-parent":
            continue
        if candidate.family == "pnictogen-hydride-parent" and other_candidate.family in {
            "positive-single-ring-parent",
            "single-ring-parent",
        }:
            continue
        if candidate.family == "guanidine-retained-parent" and other_candidate.family not in {
            "acyclic-parent",
            "functional-acyclic-skeleton-with-ring-substituents",
            "phosphorus-acid-parent",
        }:
            continue
        if len(other.name) < len(item.name):
            continue
        if len(other.name) > 220 or len(other.name) - len(item.name) > 110:
            continue
        if other.stability_penalty >= item.stability_penalty + 80:
            continue
        return True
    return False


def _has_same_priority_incomplete_parent_blocker(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family == item.candidate.family:
            continue
        if candidate_has_complete_graph(other_candidate):
            continue
        if other_candidate.priority > item.candidate.priority:
            continue
        if other.stability_penalty > item.stability_penalty + 20:
            continue
        return True
    return False


def _has_contextual_lower_external_locant_polycycle_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family != "kekulized-generic-polycyclic-parent":
        return False
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return False
    current_locants = _candidate_external_numeric_locants(candidate)
    if not current_locants:
        return False
    if len(current_locants) > 2:
        return False
    current_rank = candidate.render_rank or 0
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != candidate.family:
            continue
        if other_candidate.parent_atoms != candidate.parent_atoms:
            continue
        if not candidate_has_complete_graph(other_candidate):
            continue
        other_locants = _candidate_external_numeric_locants(other_candidate)
        if not other_locants or other_locants >= current_locants:
            continue
        other_rank = other_candidate.render_rank or 0
        if current_rank and other_rank and other_rank > current_rank + 8:
            continue
        if len(other.name) > _DEFAULT_MAX_NAME_LENGTH or len(other.name) - len(item.name) > 35:
            continue
        if other.stability_penalty > item.stability_penalty + 36:
            continue
        return True
    return False


def _candidate_external_numeric_locants(candidate: NameCandidate) -> tuple[int, ...]:
    locants = []
    for _atom_idx, locant in candidate.locants:
        if not locant.isdigit():
            return ()
        locants.append(int(locant))
    return tuple(locants)


def _has_contextual_larger_single_ring_parent_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family != "single-ring-parent":
        return False
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return False
    current_parent_count = len(candidate.parent_atoms)
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "single-ring-parent":
            continue
        if not other_candidate.parent_atoms or not candidate_has_complete_graph(other_candidate):
            continue
        if len(other_candidate.parent_atoms) <= current_parent_count:
            continue
        if len(other.name) > 180 or len(other.name) - len(item.name) > 80:
            continue
        if other.stability_penalty > item.stability_penalty + 30:
            continue
        return True
    return False


def _has_contextual_early_six_member_single_ring_parent_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family != "single-ring-parent":
        return False
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return False
    if len(candidate.parent_atoms) > 4:
        return False
    current_rank = candidate.render_rank or 0
    if current_rank <= 2:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "single-ring-parent":
            continue
        if not other_candidate.parent_atoms or not candidate_has_complete_graph(other_candidate):
            continue
        if len(other_candidate.parent_atoms) != 6:
            continue
        other_rank = other_candidate.render_rank or 0
        if other_rank < 1 or other_rank > min(2, current_rank - 1):
            continue
        if len(other.name) > 180 or len(other.name) - len(item.name) > 45:
            continue
        if other.stability_penalty > item.stability_penalty + 30:
            continue
        return True
    return False


def _has_contextual_benzotriazole_retained_anchor_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family != "fused-retained-parent":
        return False
    template_name = (candidate.template_name or "").lower()
    if not template_name or "benzotriazol" in template_name:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "fused-retained-parent":
            continue
        other_template_name = (other_candidate.template_name or "").lower()
        if "benzotriazol" not in other_template_name:
            continue
        if not candidate_has_complete_graph(other_candidate):
            continue
        if len(other.name) > len(item.name) + 8:
            continue
        if other.stability_penalty > item.stability_penalty + 20:
            continue
        return True
    return False


_COMPLETE_RETAINED_SENIOR_PARENT_FAMILIES = frozenset(
    {
        "dithiophosphate-retained-parent",
        "phosphorothioic-triamide-retained-parent",
        "thiooxo-ester-retained-parent",
    }
)


def _is_complete_retained_senior_parent(candidate: NameCandidate) -> bool:
    return (
        candidate.family in _COMPLETE_RETAINED_SENIOR_PARENT_FAMILIES
        and bool(candidate.parent_atoms)
        and candidate_has_complete_graph(candidate)
    )


def _has_contextual_retained_acid_ester_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    if item.candidate.family not in {"retained-acid-ester-parent", "single-ring-parent"}:
        return False
    if "yloxy" not in item.name or any(term in item.name for term in ("oate", "ate", " acid")):
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        if other.candidate.family != "retained-acid-ester-parent":
            continue
        if "oate" not in other.name:
            continue
        if len(other.name) > 170 or len(other.name) - len(item.name) > 100:
            continue
        if other.stability_penalty >= item.stability_penalty + 60:
            continue
        return True
    return False


def _has_contextual_acyclic_carbalkoxy_retained_ester_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    if item.candidate.family not in {
        "acyclic-skeleton-with-ring-substituents",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "functional-acyclic-skeleton-with-ring-substituents",
        "imino-functional-acyclic-skeleton-with-ring-substituents",
        "preferred-acyclic-skeleton-with-ring-substituents",
    }:
        return False
    if any(term in item.name for term in ("oate", " acid", "amide")):
        return False
    if not re.search(r"(?:carbo[a-z]+oxy|[a-z]+oxycarbonyl)", item.name):
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        if other.candidate.family != "retained-acid-ester-parent":
            continue
        if not any(term in other.name for term in ("oate", "acrylate", "methacrylate")):
            continue
        if len(other.name) > 170 or len(other.name) - len(item.name) > 110:
            continue
        if other.stability_penalty >= item.stability_penalty + 80:
            continue
        return True
    return False


def _has_contextual_aryl_urea_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    if item.candidate.family != "urea-retained-parent":
        return False
    if not re.search(r"\bN,N'-di.*(?:phenyl|tolyl)urea\b", item.name):
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        if other.candidate.family != "single-ring-parent":
            continue
        if "carbamyl)amino" not in other.name:
            continue
        if len(other.name) > 100 or len(other.name) - len(item.name) > 70:
            continue
        if other.stability_penalty >= item.stability_penalty + 60:
            continue
        return True
    return False


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


def _has_contextual_generic_bicyclic_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
    *,
    max_extra: int = 70,
    max_length: int = 140,
) -> bool:
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        if other.candidate.family not in {"functional-generic-bicyclic-parent", "generic-bicyclic-parent"}:
            continue
        if len(other.name) > max_length or len(other.name) - len(item.name) > max_extra:
            continue
        if other.stability_penalty >= item.stability_penalty + 60:
            continue
        if "bicyclo[" not in other.name:
            continue
        return True
    return False


def _has_contextual_generic_bridge_parent_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
    *,
    max_extra: int = 90,
    max_length: int = 180,
) -> bool:
    bridge_families = {
        "functional-generic-bicyclic-parent",
        "generic-bicyclic-parent",
        "generic-spiro-parent",
        "kekulized-generic-polycyclic-parent",
    }
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        if other.candidate.family not in bridge_families:
            continue
        if len(other.name) > max_length or len(other.name) - len(item.name) > max_extra:
            continue
        if other.stability_penalty >= item.stability_penalty + 60:
            continue
        if not re.search(_BRIDGE_PARENT_PATTERN, other.name):
            continue
        return True
    return False


def _has_contextual_bridge_descriptor_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
    *,
    max_extra: int = 120,
    max_length: int = 220,
) -> bool:
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        if other.candidate.family not in _BROAD_GRAPH_PARENT_ALTERNATIVE_FAMILIES:
            continue
        if len(other.name) > max_length or len(other.name) - len(item.name) > max_extra:
            continue
        if other.stability_penalty >= item.stability_penalty + 80:
            continue
        if not re.search(_BRIDGE_PARENT_PATTERN, other.name):
            continue
        return True
    return False


def _has_contextual_generated_exact_fusion_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
    *,
    max_extra: int = 12,
    max_length: int = 90,
) -> bool:
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        if other.candidate.family not in {"generated-exact-parent", "generated-fused-parent"}:
            continue
        if len(other.name) > max_length or len(other.name) - len(item.name) > max_extra:
            continue
        if other.stability_penalty >= item.stability_penalty + 50:
            continue
        if not re.search(r"\[[^\]]+\]", other.name):
            continue
        return True
    return False


def _has_contextual_component_spiro_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
    *,
    max_extra: int = 70,
    max_length: int = 140,
) -> bool:
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        if other.candidate.family not in {
            "component-spiro-parent",
            "poly-component-spiro-parent",
            "component-dispiro-parent",
        }:
            continue
        if len(other.name) > max_length or len(other.name) - len(item.name) > max_extra:
            continue
        if other.stability_penalty >= item.stability_penalty + 60:
            continue
        if "spiro[" not in other.name:
            continue
        return True
    return False


def _has_contextual_pterin_parent_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
    *,
    max_extra: int = 45,
    max_length: int = 150,
) -> bool:
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        if other.candidate.family not in {
            "fused-retained-parent",
            "generated-fused-parent",
            "generated-hydro-fused-parent",
            "generated-partial-hydro-fused-parent",
            "retained-topology-replacement-fused-parent",
        }:
            continue
        if len(other.name) > max_length or len(other.name) - len(item.name) > max_extra:
            continue
        if other.stability_penalty >= item.stability_penalty + 60:
            continue
        template_name = other.candidate.template_name or ""
        if "pterin" in template_name or "pteridin" in template_name:
            return True
        if re.search(r"(?:^|-|[)])(?:pterin|pteridin)\b", other.name):
            return True
    return False


def _has_contextual_ium_parent_alternative(item: _PreparedCandidate, prepared: list[_PreparedCandidate]) -> bool:
    ium_name = item.name + "ium"
    for other in prepared:
        if other is item:
            continue
        if other.candidate.family != item.candidate.family:
            continue
        if other.name == ium_name:
            return True
    return False


def _has_contextual_natural_topology_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
    *,
    stems: tuple[str, ...],
    max_extra: int = 25,
    max_length: int = 90,
) -> bool:
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        if other.candidate.family != "natural-product-topology-parent":
            continue
        if len(other.name) > max_length or len(other.name) - len(item.name) > max_extra:
            continue
        if other.stability_penalty >= item.stability_penalty + 60:
            continue
        if any(stem in other.name for stem in stems):
            return True
    return False


def _has_contextual_systematic_parent_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
    *,
    max_extra: int = 90,
    max_length: int = 150,
) -> bool:
    systematic_families = _CONTEXTUAL_GRAPH_PARENT_FAMILIES | {
        "acyclic-parent",
        "acyclic-skeleton-with-ring-substituents",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "functional-acyclic-skeleton-with-ring-substituents",
        "functional-generic-bicyclic-parent",
        "preferred-acyclic-skeleton-with-ring-substituents",
    }
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        if other.candidate.family not in systematic_families:
            continue
        if len(other.name) > max_length or len(other.name) - len(item.name) > max_extra:
            continue
        if other.stability_penalty >= item.stability_penalty + 60:
            continue
        return True
    return False


def _has_contextual_group14_systematic_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    if item.candidate.family != "group14-hydride-parent":
        return False
    systematic_families = {
        "acyclic-parent",
        "acyclic-skeleton-with-ring-substituents",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "kekulized-generic-polycyclic-parent",
        "preferred-acyclic-skeleton-with-ring-substituents",
        "single-ring-parent",
        "spiro-functional-chain-substituent",
    }
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        if other.candidate.family not in systematic_families:
            continue
        if len(other.name) > _DEFAULT_MAX_NAME_LENGTH or len(other.name) - len(item.name) > 180:
            continue
        if other.stability_penalty >= item.stability_penalty + 80:
            continue
        return True
    return False


def _has_contextual_oxafluoren_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
    *,
    max_extra: int = 12,
    max_length: int = 90,
) -> bool:
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        if other.candidate.family != "skeletal-replacement-fused-parent":
            continue
        if "oxafluoren" not in other.name:
            continue
        if len(other.name) > max_length or len(other.name) - len(item.name) > max_extra:
            continue
        if other.stability_penalty >= item.stability_penalty + 60:
            continue
        return True
    return False


def _has_contextual_isoindolinone_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
    *,
    max_extra: int = 8,
    max_length: int = 150,
) -> bool:
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_name = other.name
        if "(2H)-one" in other_name:
            continue
        if "isoindolin-1-one" not in other_name and "1-oxoisoindolin" not in other_name:
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


_FUSED_PYRAZOLE_PATTERN = re.compile(
    r"(?:pyrido|pyrimido|quino|triazolo|s-triazolo)\[[^\]]+\](?:pyrazol|indazol)|"
    r"(?:pyrazolo|indazolo)\[[^\]]+\](?:pyridin|pyrimidin|quinolin)"
)


def _has_fragile_fused_pyrazole_substituent_parent(name: str, candidate: NameCandidate) -> bool:
    if candidate.family not in {
        "acyclic-skeleton-with-ring-substituents",
        "functional-acyclic-skeleton-with-ring-substituents",
        "retained-acid-ester-parent",
        "retained-amide-parent",
        "single-ring-parent",
    }:
        return False
    if not _FUSED_PYRAZOLE_PATTERN.search(name):
        return False
    return "-yl)" in name or "-yl" in name


def _has_fragile_fused_pyrazole_carbonyl_parent(name: str, candidate: NameCandidate) -> bool:
    if candidate.family not in {"generated-fused-parent", "generated-partial-hydro-fused-parent"}:
        return False
    if not _FUSED_PYRAZOLE_PATTERN.search(name):
        return False
    return bool(re.search(r"(?:^|-)\d+-oxo-\d+,\d+-dihydro-", name) or re.search(r"-\d+\(\dH\)-one\b", name))


def _has_fragile_retained_spiro_alias(name: str, candidate: NameCandidate) -> bool:
    if candidate.family not in {
        "fused-retained-parent",
        "generated-spiro-fused-parent",
        "generated-spiro-partial-hydro-parent",
        "skeletal-replacement-fused-parent",
    }:
        return False
    if "spiro[" not in name:
        return False
    if "sila" in name:
        return False
    if re.search(r"spiro\[.*ind-\d+-en", name):
        return True
    return bool(
        re.search(
            r"(?:aza|oxa|thia|diaza|diox|triaza).*spiro\[.*"
            r"(?:benzodioxane|indane|inden|tetralin|decalin|naphthaline|fluoren|xanthen)",
            name,
        )
    )


def _has_fragile_generic_spiro_parent(name: str, candidate: NameCandidate) -> bool:
    return candidate.family == "generic-spiro-parent" and "spiro[" in name


def _has_contextual_generated_spiro_functional_chain_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family not in {
        "generated-fused-parent",
        "generated-spiro-fused-parent",
        "generated-partial-hydro-fused-parent",
        "generated-spiro-partial-hydro-parent",
    }:
        return False
    if not (candidate.template_name or "").startswith("spiro["):
        return False
    return _has_contextual_complete_spiro_functional_chain_alternative(item, prepared)


def _has_contextual_complete_spiro_functional_chain_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family not in {
        "generated-fused-parent",
        "generated-spiro-fused-parent",
        "generated-partial-hydro-fused-parent",
        "generated-spiro-partial-hydro-parent",
        "generic-spiro-parent",
    }:
        return False
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return False
    if candidate.render_rank is not None and candidate.render_rank > 8:
        return False
    parent_count = len(candidate.parent_atoms)
    if parent_count > 12:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "spiro-functional-chain-substituent":
            continue
        if other_candidate.source_family == "spiro-functional-chain:noncarbon-group14":
            continue
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        if len(other_candidate.parent_atoms) < max(7, parent_count):
            continue
        if len(other.name) > 260 or len(other.name) - len(item.name) > 160:
            continue
        if other.stability_penalty >= item.stability_penalty + 80:
            continue
        return True
    return False


def _has_contextual_spiro_functional_chain_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        if other.candidate.family != "spiro-functional-chain-substituent":
            continue
        if "spiro[" not in other.name:
            continue
        if len(other.name) > 260 or len(other.name) - len(item.name) > 160:
            continue
        if other.stability_penalty >= item.stability_penalty + 80:
            continue
        return True
    return False


def _has_contextual_single_ring_lactam_parent_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family not in {
        "acyclic-skeleton-with-ring-substituents",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "generated-spiro-fused-parent",
        "generated-spiro-partial-hydro-parent",
        "generic-spiro-parent",
        "spiro-functional-chain-substituent",
    }:
        return False
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return False
    if len(candidate.parent_atoms) > 14:
        return False
    if candidate.family in {
        "acyclic-skeleton-with-ring-substituents",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
    } and not _has_sibling_spiro_functional_chain_parent(item, prepared):
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "single-ring-parent":
            continue
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        if len(other_candidate.parent_atoms) not in {5, 6}:
            continue
        ring_oxo_suffixes = [suffix for suffix in other_candidate.suffixes if suffix.kind == "ring_oxo_suffix"]
        if not ring_oxo_suffixes:
            continue
        if any(OPSIN_FRAGILE_EXOCYCLIC_CHARGED_RING_SUFFIX in suffix.safety.reasons for suffix in ring_oxo_suffixes):
            continue
        if other_candidate.render_rank is not None and other_candidate.render_rank > 2:
            continue
        if len(other.name) > 180 or len(other.name) - len(item.name) > 65:
            continue
        if other.stability_penalty >= item.stability_penalty + 120:
            continue
        return True
    return False


def _has_contextual_fused_retained_parent_alternative_for_generic_spiro(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family != "generic-spiro-parent":
        return False
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return False
    if candidate.charge is not None and candidate.charge.formal_charge != 0:
        return False
    if candidate.render_rank is not None and candidate.render_rank > 4:
        return False
    parent_count = len(candidate.parent_atoms)
    if parent_count < 7 or parent_count > 8:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "fused-retained-parent":
            continue
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        if other_candidate.template_id is None:
            continue
        if other_candidate.render_rank != 1:
            continue
        if len(other_candidate.parent_atoms) < max(8, parent_count):
            continue
        if len(other.name) > 180 or len(other.name) - len(item.name) > 50:
            continue
        if other.stability_penalty > item.stability_penalty + 80:
            continue
        return True
    return False


def _has_sibling_spiro_functional_chain_parent(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if not candidate.parent_atoms:
        return False
    for other in prepared:
        if other is item:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "spiro-functional-chain-substituent":
            continue
        if not candidate_has_complete_graph(other_candidate):
            continue
        if other_candidate.parent_atoms == candidate.parent_atoms:
            return True
    return False


def _has_fragile_dibenzofuran_alias(name: str, candidate: NameCandidate) -> bool:
    return "dibenzofuran" in name and "oxafluoren" not in name


def _has_fragile_isoindolinone_alias(name: str) -> bool:
    return bool(re.search(r"isoindolin-\d+\(2H\)-one", name))


def _has_fragile_indolinone_alias(name: str, candidate: NameCandidate) -> bool:
    if candidate.family not in {"fused-retained-parent", "hydro-fused-retained-parent"}:
        return False
    return bool(re.search(r"indolin-\d+\(\dH\)-one\b", name))


def _has_fragile_azaindanone_alias(name: str, candidate: NameCandidate) -> bool:
    if candidate.family != "retained-topology-replacement-fused-parent":
        return False
    return bool(
        re.search(r"\b\d+-azaindan-\d+\(\dH\)-one\b", name)
        or re.search(r"\b(?:\d+(?:,\d+)*-)?(?:di|tri)azaindane?-\d+\(\dH\)-one\b", name)
    )


def _has_fragile_oxaanthracen_alias(name: str, candidate: NameCandidate) -> bool:
    if candidate.family != "skeletal-replacement-fused-parent":
        return False
    return bool(re.search(r"(?:^|-)\d+-oxo-\d+-oxaanthracen(?:e)?\b", name))


def _has_fragile_pterin_substituent_parent(name: str, candidate: NameCandidate) -> bool:
    if candidate.family != "single-ring-parent":
        return False
    if "pterin-" not in name and "pteridin-" not in name:
        return False
    return "benz" in name or "phenyl" in name


def _has_fragile_pterin_substituent_chain_parent(name: str, candidate: NameCandidate) -> bool:
    if candidate.family not in {
        "acyclic-skeleton-with-ring-substituents",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "functional-acyclic-skeleton-with-ring-substituents",
        "preferred-acyclic-skeleton-with-ring-substituents",
        "retained-acid-ester-parent",
    }:
        return False
    if "pterin-" not in name and "pteridin-" not in name:
        return False
    return "benzoyl" in name or "benzene" in name or "benzoate" in name


def _has_fragile_bis_bicyclic_diazene_chain_parent(name: str, candidate: NameCandidate) -> bool:
    if candidate.family not in {
        "acyclic-skeleton-with-ring-substituents",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "functional-acyclic-skeleton-with-ring-substituents",
    }:
        return False
    if "bis(" not in name or "bicyclo[" not in name:
        return False
    return bool(re.search(r"\bdiaza\d*-decen\b|\bdiaza\d*-ethen\b|\bthianonan\b|\bthianonyl\b", name))


def _has_fragile_retained_unsaturated_amide_parent(name: str, candidate: NameCandidate) -> bool:
    if candidate.family != "retained-amide-parent":
        return False
    if not re.match(r"^\([EZ]\)-N-[^-]+-", name):
        return False
    return bool(re.search(r"\(\d+[EZ]\)-\d+-[a-z]+enamide\b", name))


def _has_fragile_neutral_guanidine_parent(name: str, candidate: NameCandidate) -> bool:
    if candidate.family != "guanidine-retained-parent":
        return False
    return name.endswith("guanidin")


def _has_fragile_anthranil_ketone_alias(name: str, candidate: NameCandidate) -> bool:
    if candidate.family not in {
        "fused-retained-parent",
        "generated-fused-parent",
        "hydro-fused-retained-parent",
        "retained-topology-replacement-fused-parent",
    }:
        return False
    return bool(re.search(r"anthranil-\d+\(\dH\)-one\b", name))


def _has_fragile_retained_like_generic_polycycle_alias(name: str, candidate: NameCandidate) -> bool:
    if candidate.family not in {
        "kekulized-generic-polycyclic-parent",
        "retained-topology-replacement-fused-parent",
        "skeletal-replacement-fused-parent",
    }:
        return False
    if _has_generic_polycyclic_parent_descriptor(name):
        return False
    if _has_natural_product_topology_stem(name):
        return False
    return bool(
        re.search(
            r"(?:di|tri|tetra|penta)?(?:aza|oxa|thia).*(?:"
            r"acephenanthren|acephenanthrylen|aceanthren|acenaphthen|acenaphtho|azachroman|chroman|anthracen|oxaanthracen|xanthen"
            r")",
            name,
        )
    )


def _has_fragile_pentaazaind_replacement_alias(name: str, candidate: NameCandidate) -> bool:
    if candidate.family != "retained-topology-replacement-fused-parent":
        return False
    return "pentaazaind" in name and "-one" in name


def _has_fragile_diazenyl_pyridone_parent(name: str, candidate: NameCandidate) -> bool:
    if candidate.family != "single-ring-parent":
        return False
    return "diazenyl" in name and "pyridin-2-on" in name


def _has_fragile_imino_purine_parent(name: str, candidate: NameCandidate) -> bool:
    if candidate.family != "imino-functional-acyclic-skeleton-with-ring-substituents":
        return False
    return "imino" in name and "purin" in name


def _has_fragile_imino_purine_alias(name: str, candidate: NameCandidate) -> bool:
    if candidate.family not in {
        "acyclic-skeleton-with-ring-substituents",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "fused-retained-parent",
        "single-ring-parent",
    }:
        return False
    return bool(re.search(r"(?:imino.*purin|iminopurin)", name))


def _has_fragile_acyclic_oxoindolin_substituent_parent(name: str, candidate: NameCandidate) -> bool:
    if candidate.family not in {
        "acyclic-skeleton-with-ring-substituents",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "functional-acyclic-skeleton-with-ring-substituents",
    }:
        return False
    return "oxoindolin-2-yl" in name


def _has_fragile_oxo_hydro_fused_alias(name: str, candidate: NameCandidate) -> bool:
    if candidate.family not in {
        "generated-partial-hydro-fused-parent",
        "hydro-fused-retained-parent",
    }:
        return False
    if "oxo-" not in name or "dihydro-" not in name:
        return False
    if "pyrrolo[" in name and any(parent in name for parent in ("pyridin", "pyrimidin", "benzene")):
        return True
    return bool(re.search(r"\b(?:acridin|phenanthridin|carbazol|xanthen)\b", name))


def _has_fragile_hydro_fused_bridge_alias(name: str, candidate: NameCandidate) -> bool:
    if candidate.family not in {
        "generated-partial-hydro-fused-parent",
        "hydro-fused-retained-parent",
    }:
        return False
    if "dihydro-" not in name:
        return False
    return bool(re.search(r"(?:imidazo|pyrimido|pyrido)\[[^\]]+\]pyrimidin", name))


def _has_fragile_imidazothiophene_bridge_alias(name: str, candidate: NameCandidate) -> bool:
    if candidate.family != "generated-partial-hydro-fused-parent":
        return False
    if "imidazo[" not in name or "thiophene" not in name:
        return False
    return "thianonyl" in name or "diazabicyclo[" in name


def _has_fragile_ergosta_terminal_side_chain_parent(name: str, candidate: NameCandidate) -> bool:
    if candidate.family != "natural-product-topology-parent":
        return False
    if "ergosta-" not in name:
        return False
    return bool(re.search(r"(?:22,)?24,25-(?:di|tri|tetra)?ene|22,24,25-triene", name))


_SHORT_FUSED_ALIAS_PATTERN = re.compile(
    r"(?:"
    r"purin|adenin|indazol|anthranil|chroman|flavan|coumarin|coumaron|"
    r"cepham|trypt(?:yl|amin)|oxanaphthalin|azacoumarin|benzothiazin|"
    r"azaind(?:an|en)|oxindol|indolin|tetralin|tetracene|naphthalin"
    r")"
)


def _has_fragile_short_fused_alias(name: str, candidate: NameCandidate) -> bool:
    if candidate.family not in {
        "fused-retained-parent",
        "generated-fused-parent",
        "generated-hydro-fused-parent",
        "generated-partial-hydro-fused-parent",
        "hydro-fused-retained-parent",
        "retained-topology-replacement-fused-parent",
        "skeletal-replacement-fused-parent",
    }:
        return False
    if _has_generic_polycyclic_parent_descriptor(name):
        return False
    parent = re.sub(r"^\([^)]*\)-", "", name)
    if len(parent) > 42:
        return False
    if "[" in parent and not re.search(r"(?:coumaron|purino|imidazo|pyrrolo|thieno)", parent):
        return False
    return bool(_SHORT_FUSED_ALIAS_PATTERN.search(parent))


def _has_fragile_component_fused_alias(name: str, candidate: NameCandidate) -> bool:
    if candidate.family not in {
        "fused-retained-parent",
        "generated-fused-parent",
        "generated-hydro-fused-parent",
        "generated-partial-hydro-fused-parent",
        "retained-topology-replacement-fused-parent",
        "skeletal-replacement-fused-parent",
    }:
        return False
    if _has_generic_polycyclic_parent_descriptor(name):
        return False
    return bool(
        re.search(
            r"(?:(?:\d+[a-z]?[-,])?(?:aza|oxa|thia|diaza|triaza|tetraaza|pentaaza|diox|dioxa|trioxa|dithia|thioxa)-?)+"
            r"[a-z]*?(?:azuleno|indeno)\[[^\]]+\]benzene",
            name,
        )
        or re.search(
            r"(?:aza|oxa|thia|diaza|triaza|tetraaza|pentaaza|diox|dioxa|trioxa|dithia|thioxa)"
            r"[a-z]*?(?:as-indacen|s-indacen|acenaphthen|aceanthren|phenalen)",
            name,
        )
    )


def _has_fragile_short_fusion_orientation_alias(name: str, candidate: NameCandidate) -> bool:
    if candidate.family not in {"generated-exact-parent", "generated-fused-parent"}:
        return False
    return bool(
        re.search(r"\b(?:quino|isoquino|cinnolino|phthalazino)\[[^\]]+\](?:pyrazole|imidazole|pyrrole)\b", name)
        or re.search(r"\b(?:pyridazino|pyrimido|pyrido)\[[^\]]+\](?:indol|indazol)\b", name)
    )


def _has_fragile_purine_tautomer_alias(name: str, candidate: NameCandidate) -> bool:
    if candidate.family not in {
        "exact-retained-parent",
        "fused-retained-parent",
        "whole-molecule-retained-parent",
        "generated-fused-parent",
        "generated-hydro-fused-parent",
        "generated-partial-hydro-fused-parent",
    }:
        return False
    return bool(re.search(r"\b(?:purin|adenin|guanin|xanthin|hypoxanthin)(?:-\d+-yl)?\b", name))


def _has_fragile_terminal_azabicyclo_benzimidazole_parent(name: str, candidate: NameCandidate) -> bool:
    if candidate.family in {
        "fused-retained-parent",
        "generic-bicyclic-parent",
        "retained-topology-replacement-fused-parent",
    }:
        return bool(
            re.search(
                r"benzimidazol[^)]*carbonyl\)-(?:(?:\d-azabicyclo\[2\.2\.1\]heptane)|(?:\d-azanor(?:bornane|camphane)))$",
                name,
            )
        )
    if candidate.family == "skeletal-replacement-fused-parent":
        return (
            "diazaacenaphth" in name
            and "benzimidazol" in name
            and "azabicyclo[2.2.1]heptane-2-ylcarbonyl" in name
        )
    return False


def _has_fragile_group14_hetero_carbon_ligand(candidate: NameCandidate) -> bool:
    if candidate.family != "group14-hydride-parent":
        return False
    return OPSIN_FRAGILE_GROUP14_HETERO_CARBON_LIGAND in candidate_safety_reasons(candidate)


def _has_contextual_integrated_benzimidazole_carboxamide_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        if other.candidate.family not in {
            "fused-retained-parent",
            "generated-fused-parent",
            "generated-hydro-fused-parent",
            "generated-partial-hydro-fused-parent",
            "single-ring-parent",
        }:
            continue
        if len(other.name) > 230 or len(other.name) - len(item.name) > 45:
            continue
        if other.stability_penalty >= item.stability_penalty + 120:
            continue
        if "azabicyclo[2.2.1]heptane-2-ylcarbonyl" not in other.name:
            continue
        if not re.search(r"(?:benzimidazol|imidazo\[[^\]]+\]benzene)", other.name):
            continue
        if other.name.endswith("azabicyclo[2.2.1]heptane"):
            continue
        return True
    return False


def _has_fragile_decomposed_bis_bridge_parent(name: str, candidate: NameCandidate) -> bool:
    if candidate.family not in {
        "acyclic-skeleton-with-ring-substituents",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "functional-acyclic-skeleton-with-ring-substituents",
    }:
        return False
    if "bis(" not in name or "bicyclo[" not in name:
        return False
    return bool(re.search(r"bis\(\([^)]*(?:diaza|oxa|thia)[^)]*bicyclo\[[^\]]+\][^)]*\)", name))


def _has_fragile_whole_molecule_retained_stub(name: str, candidate: NameCandidate) -> bool:
    if candidate.family not in {
        "whole-molecule-retained-parent",
        "whole-molecule-retained-parent-with-stereo",
    }:
        return False
    if candidate.source_file not in {
        "aminoAcids.xml",
        "carbohydrates.xml",
        "naturalProducts.xml",
    }:
        return False
    parent = re.sub(r"^\([^)]*\)-", "", name)
    if len(parent) > 24:
        return False
    if candidate.source_file == "carbohydrates.xml":
        return not parent.endswith(("ose", "ulose"))
    return not parent.endswith(
        (
            "ane",
            "ene",
            "en",
            "one",
            "ol",
            "al",
            "acid",
            "ium",
            "ide",
            "ate",
            "ose",
            "in",
            "ine",
        )
    )


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


def _has_contextual_locanted_stereo_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
    *,
    max_extra: int = 35,
    max_length: int = 140,
) -> bool:
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        if _locanted_atom_stereo_descriptor_count(other.name) == 0:
            continue
        if len(other.name) > max_length or len(other.name) - len(item.name) > max_extra:
            continue
        if other.stability_penalty >= item.stability_penalty + 60:
            continue
        if other.candidate.family not in _CONTEXTUAL_SYSTEMATIC_ALTERNATIVE_FAMILIES | _CONTEXTUAL_GRAPH_PARENT_FAMILIES:
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
    if _has_stable_oxafluorene_alias(name):
        return 0
    if re.search(r"(?:^|-)\d+(?:,\d+)*-dioxo-.*azaacenaphthen\b", name):
        return 0
    if re.search(
        r"(?:^|-)(?:\d+[a-z]?,?)*(?:di|tri|tetra|penta)?aza"
        r"(?:acenaphthen|aceanthren|anthracen|fluoren|phenanthren|as-indacen|s-indacen|azuleno)",
        name,
    ):
        return 35
    if re.search(
        r"(?:^|-)(?:\d+[a-z]?(?:,\d+[a-z]?)*-)?(?:di|tri|tetra|penta)?(?:aza|oxa|thia)"
        r"(?:acenaphthen|aceanthren|anthracen|fluoren|phenanthren|as-indacen|s-indacen|azuleno)",
        name,
    ):
        return 28
    return 0


def _has_unstable_replacement_polycycle_alias(name: str) -> bool:
    if "spiro[" in name or "bicyclo[" in name or "tricyclo[" in name:
        return False
    if _has_stable_oxafluorene_alias(name):
        return False
    if re.search(r"(?:^|-)\d+(?:,\d+)*-dioxo-.*azaacenaphthen\b", name):
        return False
    return bool(
        re.search(
            r"(?:^|-)(?:di|tri|tetra|penta)?(?:aza|oxa|thia)"
            r"(?:acenaphthen|aceanthren|anthracen|fluoren|phenanthren|as-indacen|s-indacen|azuleno)",
            name,
        )
        or re.search(
            r"(?:^|-)(?:\d+[a-z]?(?:,\d+[a-z]?)*-)(?:di|tri|tetra|penta)?(?:aza|oxa|thia)"
            r"(?:acenaphthen|aceanthren|anthracen|fluoren|phenanthren|as-indacen|s-indacen|azuleno)",
            name,
        )
    )


def _has_stable_oxafluorene_alias(name: str) -> bool:
    return bool(re.search(r"(?:^|-)(?:\d+[a-z]?-)?oxafluoren\b", name))


def _has_unstable_benzimidazolyl_orientation_artifact(name: str) -> bool:
    if "benzimidazolium" in name:
        return False
    return bool(
        re.search(r"\b(?:\d+[a-z]?H-)?\d+-benzimidazolyl\b", name)
        or re.search(r"\b\d+(?:-[a-z]+)?1H-\d-benzimidazolyl\b", name)
        or re.search(r"\b(?:\d+[a-z]?H-)?benzimidazol-\d+-yl\b", name)
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
    bonus += _structured_stereo_parent_bonus(candidate)
    if candidate.family in {"single-ring-parent", "positive-single-ring-parent"}:
        bonus += _charged_single_ring_parent_bonus(candidate)
        bonus += _single_ring_suffix_parent_bonus(candidate)
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
    if candidate.family == "generated-hydro-fused-parent" and _has_stable_tetrahydro_pyridoimidazole_parent(name):
        bonus += 56
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


def _charged_single_ring_parent_bonus(candidate: NameCandidate) -> int:
    evidence = candidate.charge
    if evidence is None or not evidence.charged_atoms:
        return 0
    if not candidate_has_complete_graph(candidate):
        return 0
    return min(56, 36 + 4 * len(evidence.charged_atoms))


def _single_ring_suffix_parent_bonus(candidate: NameCandidate) -> int:
    if not candidate.suffixes or not candidate_has_complete_graph(candidate):
        return 0
    chalcogen_suffixes = [
        suffix
        for suffix in candidate.suffixes
        if suffix.kind in {"ring_oxo_suffix", "ring_thione_suffix"}
    ]
    if chalcogen_suffixes:
        if any(
            "compact_hetero_ring_chalcogen_suffix_parent" in suffix.source
            for suffix in chalcogen_suffixes
        ):
            return 64
        return 24
    kinds = {suffix.kind for suffix in candidate.suffixes}
    if "ring_functional_suffix" in kinds:
        return 18
    if kinds.intersection({"ring_imino_suffix", "ring_carbonyl_suffix"}):
        return 12
    return 0


def _structured_stereo_parent_bonus(candidate: NameCandidate) -> int:
    if not candidate.stereo_locants or not candidate.parent_atoms:
        return 0
    if not candidate_has_complete_graph(candidate):
        return 0
    return min(80, len(candidate.stereo_locants) * 10)


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


def _has_unlocanted_atom_stereo_prefix(name: str) -> bool:
    match = re.match(r"^\(([^)]*)\)-", name)
    if not match:
        return False
    descriptors = match.group(1).split(",")
    if not any(descriptor in {"R", "S", "r", "s"} for descriptor in descriptors):
        return False
    return not any(re.fullmatch(r"\d+'?[RSrs]", descriptor) for descriptor in descriptors)


def _locanted_atom_stereo_descriptor_count(text: str) -> int:
    return len(re.findall(r"(?<![A-Za-z])\d+'?[RSrs](?![a-z])", text))


def _has_stable_tetrahydro_imidazopyridine_parent(name: str) -> bool:
    return bool(re.search(r"4,5,6,7-tetrahydro-imidazo\[4,5-[cd]\]pyridine\b", name))


def _has_stable_tetrahydro_pyridoimidazole_parent(name: str) -> bool:
    return bool(re.search(r"4,5,6,7-tetrahydro-pyrido\[[^\]]+\]imidazol\b", name))


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


def _has_fragile_ethanediylidene_alias(name: str) -> bool:
    return "ethanediylidene" in name and any(
        fragment in name
        for fragment in (
            "az",
            "oxa",
            "thia",
            "dioxo",
            "bicyclo[",
        )
    )


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
    structured_stereo_applied = False
    if candidate.stereo_prefix and not name.startswith("("):
        name = f"({candidate.stereo_prefix})-{name}"
        structured_stereo_applied = True
    elif candidate.stereo_locants:
        locanted_name = apply_locanted_stereochemical_prefix(
            mol,
            Chem,
            name,
            candidate.stereo_locants,
            atom_order=_fallback_stereo_atom_order(
                candidate.family,
                name,
                stereo_locant_count=_candidate_assigned_stereo_locant_count(mol, Chem, candidate),
            ),
            include_unlocanted=not _has_retained_carbohydrate_substituent(name),
        )
        if locanted_name != name:
            name = locanted_name
            structured_stereo_applied = True
    if candidate.family in {
        "whole-molecule-retained-parent",
        "natural-product-topology-parent",
        "exact-biochemical-retained-parent",
    }:
        name = _apply_uniform_isotope_label(mol, name)
        return name if len(name) <= max_length else ""
    skip_fallback_stereo = candidate.family in {
        "retained-acid-ester-parent",
        "retained-amide-parent",
        "carbamate-retained-parent",
    } and _has_internal_locanted_stereo(name)
    if (
        not candidate.stereo_prefix
        and not structured_stereo_applied
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


def _fallback_stereo_atom_order(family: str, name: str = "", *, stereo_locant_count: int = 0) -> str:
    if family in {
        "fused-retained-parent",
        "hydro-fused-retained-parent",
        "retained-topology-replacement-fused-parent",
    } and stereo_locant_count <= 1 and any(
        fragment in name for fragment in ("tetralin", "tetrahydroquinolin")
    ):
        return "idx_desc"
    if any(fragment in name for fragment in ("pseudouridin", "uridin", "spirostane")):
        return "idx"
    if ("pterin" in name or "pteridin" in name) and "tetrahydropterin" not in name:
        return "idx_desc"
    return "idx"


def _candidate_assigned_stereo_locant_count(mol, Chem, candidate: NameCandidate) -> int:
    if not candidate.stereo_locants:
        return 0
    locanted_atoms = {atom_idx for atom_idx, _locant in candidate.stereo_locants}
    return sum(
        1
        for atom_idx, _descriptor in assigned_atom_stereochemistry(mol, Chem)
        if atom_idx in locanted_atoms
    )
