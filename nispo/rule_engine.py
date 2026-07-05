from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from functools import lru_cache
import re
import time

from rdkit.Chem import rdMolDescriptors

from .candidate_facts import (
    CandidateSafety,
    ChargeEvidence,
    HydrogenEvidence,
    OPSIN_FRAGILE_CHARGED_REPLACEMENT_UNSATURATION,
    OPSIN_FRAGILE_EXOCYCLIC_CHARGED_RING_SUFFIX,
    OPSIN_FRAGILE_GENERATED_SPIRO_UNSATURATION,
    OPSIN_FRAGILE_GROUP14_HETERO_CARBON_LIGAND,
    OPSIN_FRAGILE_OXYIMINO_CARBONYL_PARENT,
    OPSIN_FRAGILE_POLYCYCLIC_BRIDGE_TOPOLOGY,
    OPSIN_FRAGILE_POLYCYCLIC_BRIDGE_UNSATURATION,
    PrefixCandidate,
    StereoEvidence,
    SuffixCandidate,
)
from .candidate_fact_ranker_model import score as _candidate_fact_ranker_score
from .candidate_fact_switch_gate_model import score as _candidate_fact_switch_gate_score
from .engine_flags import (
    FAST_POLY_BRIDGE_ROUTE_PROP,
    FAST_POLY_RETAINED_DEFER_BYPASS_PROP,
    FAST_POLY_RETAINED_DEFER_CACHE_KEY,
    FAST_POLY_RETAINED_DEFER_FRAGILE_TEMPLATE_KEYS,
    FAST_POLY_RETAINED_TOPOLOGY_DEFER_TEMPLATE_NAMES,
)
from .fragment_rules import (
    FragmentAssemblyRule,
    common_polyatomic_fragment_name,
    simple_fragment_name,
    standalone_polyatomic_fragment_name,
)
from .naming_context import MoleculeContext
from .molecule_cache import mol_cache_dict, mol_cache_get, mol_cache_set
from .route_plan import RoutePlan
from .rule_dispatch import RuleSpec, compile_rules_for_context
from .stereochemistry import (
    apply_locanted_stereochemical_prefix,
    apply_stereochemical_prefix,
    assigned_atom_stereochemistry,
)
from .telemetry import CandidateTelemetry


_DEFAULT_MAX_NAME_LENGTH = 260
_CANDIDATE_FACT_RANKER_MIN_MARGIN = 0.0
_CANDIDATE_FACT_SWITCH_GATE_THRESHOLD = 0.6
_STEREO_EMPTY_FALLBACK_EVALUATION_LENGTH = 320
_STEREO_EMPTY_FALLBACK_MAX_LENGTH = 320
_NONSTEREO_EMPTY_FALLBACK_EVALUATION_LENGTH = 320
_NONSTEREO_EMPTY_FALLBACK_MAX_LENGTH = 320
_SENIOR_CANDIDATE_EARLY_STOP_PRIORITY = 60
_FRAGMENT_SUBSTITUENT_FALLBACK_PROP = "_nispo_allow_fragment_substituent_fallback"
_RETAINED_PARTIAL_HYDRO_FALLBACK_PROP = "_nispo_enable_retained_partial_hydro_fallback"
_EMPTY_FALLBACK_SKIPPED_PRIMARY_ROUTES = frozenset(
    {
        "acyclic_senior_functional",
        "acyclic_simple",
        "multi_monocycle",
        "simple_monocycle",
    }
)
_ACYCLIC_ISOTHIOCYANATE_IMINO_THIONE_SOURCE = "acyclic-isothiocyanate-imino-thione"
_SINGLE_RING_RETAINED_CARBOHYDRATE_SUBSTITUENT_SOURCE = "single-ring:retained-carbohydrate-substituent"
_FAST_MONOCYCLE_SYSTEM_RULE_FAMILIES = frozenset(
    {
        "carbodiimide-retained-parent",
        "dithiazole-retained-parent",
        "oxazinone-retained-parent",
        "oxirane-dione-retained-parent",
        "single-ring-parent",
    }
)
_UNSATURATED_ACYCLIC_ALTERNATIVE_FAMILIES = frozenset(
    {
        "acyclic-skeleton-with-ring-substituents",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "functional-acyclic-skeleton-with-ring-substituents",
        "imino-functional-acyclic-skeleton-with-ring-substituents",
        "preferred-acyclic-skeleton-with-ring-substituents",
        "sulfur-oxo-acyclic-skeleton-with-ring-substituents",
    }
)
_FAST_MONOCYCLE_UNSATURATED_ACYCLIC_RULE_FAMILIES = _UNSATURATED_ACYCLIC_ALTERNATIVE_FAMILIES
_FAST_COMPONENT_SPIRO_ROUTE_RULE_FAMILIES = frozenset({"component-spiro-parent"})
_FAST_COMPONENT_SPIRO_SKELETAL_REPLACEMENT_RULE_FAMILIES = frozenset(
    {"skeletal-replacement-fused-parent"}
)
_FAST_POLY_BRIDGE_ROUTE_RULE_FAMILIES = frozenset(
    {
        "kekulized-generic-polycyclic-parent",
    }
)
_FAST_POLY_BRIDGE_RETAINED_HYDRO_RULE_FAMILIES = frozenset(
    {
        "generated-hydro-fused-parent",
        "fused-retained-parent",
        "hydro-fused-retained-parent",
    }
)
_HEAVY_POLYCYCLE_GENERATED_FUSED_RESCUE_RULE_FAMILIES = frozenset(
    {
        "generated-fused-parent",
        "high-signal-generated-topology-fused-parent",
        "skeletal-replacement-fused-parent",
    }
)
_HEAVY_POLYCYCLE_GENERATED_FUSED_MAX_LENGTH = 200
_TEMPORARY_COMPLEXITY_RANKED_RESCUE_FAMILIES = frozenset(
    {
        "kekulized-generic-polycyclic-parent",
    }
)
_TEMPORARY_COMPLEXITY_BOUNDED_RESCUE_FAMILIES = frozenset(
    {
        "acyclic-skeleton-with-ring-substituents",
        "generic-bicyclic-parent",
        "kekulized-generic-polycyclic-parent",
        "single-ring-parent",
    }
)
_FAST_POLY_PRE_ROUTE_SENIOR_FAMILIES = frozenset(
    {
        "azanium-parent",
        "carbodiimide-retained-parent",
        "charged-retained-parent",
        "exact-retained-n-oxide-parent",
        "lambda-sulfanone-parent",
        "nitrile-oxide-retained-parent",
        "oxo-carboxylate-retained-parent",
        "phosphate-ester-retained-parent",
        "phosphite-ester-parent",
        "phosphorane-parent",
        "phosphorus-acid-parent",
        "small-inorganic-retained-parent",
        "sulfonamide-parent",
        "sulfur-oxo-acyclic-skeleton-with-ring-substituents",
        "sydnone-retained-parent",
        "thiooxo-ester-retained-parent",
    }
)
_FAST_POLY_LOW_HETERO_SENIOR_FAMILIES = _FAST_POLY_PRE_ROUTE_SENIOR_FAMILIES | frozenset(
    {
        "acyl-diphosphate-retained-parent",
        "acyl-hydrazide-retained-parent",
        "acyclic-skeleton-with-ring-substituents",
        "amino-acid-retained-parent",
        "aminophosphane-parent",
        "azidobenzylidene-cyclohexanone-retained-parent",
        "azinobis-benzothiazoline-retained-parent",
        "beta-dicarbonyl-parent",
        "biphenyl-diphosphonite-parent",
        "bis-diethylaminobenzylidene-cyclopentanone-retained-parent",
        "bis-phosphino-amine-retained-parent",
        "bis-sulfonium-sulfide-retained-parent",
        "boronic-pinacol-ester-parent",
        "carbamate-retained-parent",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "charged-iminium-retained-parent",
        "charged-organophosphorus-oxo-parent",
        "charged-ring-acyclic-skeleton-with-ring-substituents",
        "conjugated-carbonyl-acyclic-skeleton-with-ring-substituents",
        "cyclic-n-carbamate-carboxylate-retained-parent",
        "cyclic-nitroimine-parent",
        "cyclopropane-triylidene-nitrile-retained-parent",
        "dithiocarbamate-retained-parent",
        "dithiocarbonate-retained-parent",
        "dithiophosphate-retained-parent",
        "dithiazole-retained-parent",
        "formamidine-retained-parent",
        "functional-acyclic-skeleton-with-ring-substituents",
        "glutarimide-retained-parent",
        "group14-hydride-parent",
        "guanidine-retained-parent",
        "hexamethylenediamide-retained-parent",
        "hydrazine-parent",
        "imino-functional-acyclic-skeleton-with-ring-substituents",
        "iodane-parent",
        "iodine-oxo-parent",
        "iodonium-parent",
        "methylenebis-dithiocarbamate-retained-parent",
        "neutral-amine-parent",
        "organoborane-parent",
        "organic-peroxide-retained-parent",
        "organophosphorus-oxide-parent",
        "oxazinone-retained-parent",
        "oxido-nitroso-azanium-parent",
        "oxime-sulfonate-nitrile-parent",
        "oxirane-dione-retained-parent",
        "oxo-sulfanium-parent",
        "phosphanylium-imino-parent",
        "phosphanylium-oxide-parent",
        "phosphanylium-ylidene-parent",
        "phosphanium-parent",
        "phosphorothioic-triamide-retained-parent",
        "phosphorylcholine-retained-parent",
        "piperazine-dione-ylidene-retained-parent",
        "pnictogen-hydride-parent",
        "preferred-acyclic-skeleton-with-ring-substituents",
        "retained-acid-ester-parent",
        "retained-amide-parent",
        "sulfanium-parent",
        "sulfinate-retained-parent",
        "sulfonamide-retained-parent",
        "sulfonohydrazide-retained-parent",
        "sulfonimidamide-parent",
        "sulfur-oxo-retained-parent",
        "terminal-carboxylate-parent",
        "thiadiazole-dithio-retained-parent",
        "urea-retained-parent",
    }
)
_FAST_MULTI_MONOCYCLE_SENIOR_RULE_FAMILIES = frozenset(
    {
        "aminophosphane-parent",
        "biphenyl-diphosphonite-parent",
        "bis-phosphino-amine-retained-parent",
        "charged-organophosphorus-oxo-parent",
        "organophosphorus-oxide-parent",
        "phosphanium-parent",
        "phosphite-ester-parent",
        "phosphorane-parent",
        "phosphorus-acid-parent",
        "sulfinate-retained-parent",
        "sulfonamide-parent",
    }
)
_FAST_POLY_ROUTE_MISS_FALLBACK_RULE_FAMILIES = frozenset(
    {
        "acyclic-skeleton-with-ring-substituents",
        "carbodiimide-retained-parent",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "charged-ring-acyclic-skeleton-with-ring-substituents",
        "dithiazole-retained-parent",
        "exact-retained-n-oxide-parent",
        "exact-retained-parent",
        "fused-retained-parent",
        "generated-fused-parent",
        "generic-spiro-parent",
        "hydro-fused-retained-parent",
        "kekulized-generic-polycyclic-parent",
        "lambda-sulfanone-parent",
        "natural-product-topology-parent",
        "oxazinone-retained-parent",
        "oxirane-dione-retained-parent",
        "phosphate-ester-retained-parent",
        "phosphanylium-imino-parent",
        "preferred-acyclic-skeleton-with-ring-substituents",
        "retained-ring-scaffold-parent",
        "retained-topology-replacement-fused-parent",
        "single-ring-parent",
        "skeletal-replacement-fused-parent",
        "sulfonimidamide-parent",
        "sulfur-oxo-acyclic-skeleton-with-ring-substituents",
        "sydnone-urea-amidate-retained-parent",
    }
)
_BRIDGEHEAD_POLYCYCLE_SELECTOR_RULE_FAMILIES = _FAST_POLY_ROUTE_MISS_FALLBACK_RULE_FAMILIES | frozenset(
    {
        "functional-generic-bicyclic-parent",
        "generic-bicyclic-parent",
        "preferred-generic-bicyclic-parent",
    }
)
_RANKED_POLYCYCLE_SELECTOR_RULE_FAMILIES = (
    _BRIDGEHEAD_POLYCYCLE_SELECTOR_RULE_FAMILIES
    - frozenset({"preferred-acyclic-skeleton-with-ring-substituents"})
) | frozenset(
    {
        "generated-hydro-fused-parent",
        "high-signal-generated-topology-fused-parent",
    }
)
_SPIRO_SELECTOR_RULE_FAMILIES = frozenset(
    {
        "acyclic-skeleton-with-ring-substituents",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "fused-retained-parent",
        "generated-fused-parent",
        "generated-partial-hydro-fused-parent",
        "generated-spiro-fused-parent",
        "generated-spiro-partial-hydro-parent",
        "generic-bicyclic-parent",
        "high-signal-generated-topology-fused-parent",
        "hydro-fused-retained-parent",
        "kekulized-generic-polycyclic-parent",
        "lambda-sulfanone-spiro-substituent",
        "neutral-amine-parent",
        "preferred-acyclic-skeleton-with-ring-substituents",
        "retained-topology-replacement-fused-parent",
        "single-ring-parent",
        "sulfonamide-parent",
        "sulfonimidamide-parent",
        "sulfur-oxo-acyclic-skeleton-with-ring-substituents",
    }
)
_FAST_POLY_RETAINED_DEFER_BRIDGE_RETRY_FAMILIES = frozenset(
    {
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "generic-spiro-parent",
        "hydro-fused-retained-parent",
        "retained-ring-scaffold-parent",
        "retained-topology-replacement-fused-parent",
        "skeletal-replacement-fused-parent",
        "sulfur-oxo-acyclic-skeleton-with-ring-substituents",
    }
)
_FAST_POLY_RETAINED_DEFER_LATE_ACYCLIC_FAMILIES = frozenset(
    {
        "acyclic-skeleton-with-ring-substituents",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "charged-ring-acyclic-skeleton-with-ring-substituents",
        "preferred-acyclic-skeleton-with-ring-substituents",
        "sulfur-oxo-acyclic-skeleton-with-ring-substituents",
    }
)
_EXPLICIT_ISOTOPE_HYDROGEN_RETAINED_RULE_FAMILIES = frozenset(
    {
        "whole-molecule-retained-parent",
        "exact-retained-parent",
        "exact-functional-retained-parent",
        "exact-biochemical-retained-parent",
        "retained-ring-scaffold-parent",
        "fused-retained-parent",
        "hydro-fused-retained-parent",
    }
)
_FAST_MONOCYCLE_ALLOWED_ATOMIC_NUMBERS = frozenset({6, 7, 8, 9, 16, 17, 35, 53})
_FAST_MULTI_MONOCYCLE_SENIOR_BLOCKED_FAMILIES = frozenset(
    {
        "acyclic-skeleton-with-ring-substituents",
        "beta-dicarbonyl-parent",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "charged-ring-acyclic-skeleton-with-ring-substituents",
        "functional-acyclic-skeleton-with-ring-substituents",
        "imino-functional-acyclic-skeleton-with-ring-substituents",
        "preferred-acyclic-skeleton-with-ring-substituents",
        "retained-amide-parent",
        "sulfur-oxo-acyclic-skeleton-with-ring-substituents",
    }
)
_NONHALOGEN_HETERO_EXCLUDED_SYMBOLS = frozenset({"C", "H", "F", "Cl", "Br", "I", "At"})


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
    parent_element_counts: tuple[tuple[str, int], ...] = ()
    parent_aromatic_atom_count: int = 0
    parent_hetero_atom_count: int = 0
    parent_terminal_hetero_substituent_count: int = 0
    parent_terminal_heterocarbon_substituent_count: int = 0
    parent_terminal_carbonyl_substituent_count: int = 0
    parent_terminal_chalcogen_double_bond_count: int = 0
    parent_external_unsaturated_bond_count: int = 0
    parent_internal_unsaturated_bond_count: int = 0
    parent_substituted_hetero_atom_count: int = 0
    parent_external_substituted_hetero_atom_count: int = 0
    parent_external_nitrogen_substituent_count: int = 0
    parent_external_acyl_nitrogen_substituent_count: int = 0
    parent_external_oxygen_substituent_count: int = 0
    parent_external_isotope_hydrogen_count: int = 0
    molecule_isotope_atom_count: int = 0
    parent_isotope_atom_count: int = 0
    candidate_isotope_atom_count: int = 0
    parent_external_relative_stereo_branch_count: int = 0
    parent_indicated_hydrogen_locants: tuple[str, ...] = ()
    molecule_formal_charge_atom_count: int = 0
    molecule_abs_formal_charge_sum: int = 0
    parent_formal_charge_atom_count: int = 0
    parent_abs_formal_charge_sum: int = 0
    template_formal_charge_atom_count: int = 0
    template_abs_formal_charge_sum: int = 0
    prefixes: tuple[PrefixCandidate, ...] = ()
    suffixes: tuple[SuffixCandidate, ...] = ()
    charge: ChargeEvidence | None = None
    hydrogen: HydrogenEvidence | None = None
    stereo: StereoEvidence | None = None
    safety: CandidateSafety | None = None
    estimated_length: int | None = None
    render_cost: int = 1
    render_rank: int | None = None
    external_bond_stereo_parent_fragile: bool = False
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
        self._fast_monocycle_system_rules = tuple(
            rule for rule in self._rules if rule.family in _FAST_MONOCYCLE_SYSTEM_RULE_FAMILIES
        )
        self._fast_monocycle_unsaturated_acyclic_rules = tuple(
            rule for rule in self._rules if rule.family in _FAST_MONOCYCLE_UNSATURATED_ACYCLIC_RULE_FAMILIES
        )
        self._fast_component_spiro_route_rules = tuple(
            rule for rule in self._rules if rule.family in _FAST_COMPONENT_SPIRO_ROUTE_RULE_FAMILIES
        )
        self._fast_component_spiro_skeletal_replacement_rules = tuple(
            rule
            for rule in self._rules
            if rule.family in _FAST_COMPONENT_SPIRO_SKELETAL_REPLACEMENT_RULE_FAMILIES
        )
        self._fast_poly_bridge_route_rules = tuple(
            rule for rule in self._rules if rule.family in _FAST_POLY_BRIDGE_ROUTE_RULE_FAMILIES
        )
        self._fast_poly_bridge_retained_hydro_rules = tuple(
            rule for rule in self._rules if rule.family in _FAST_POLY_BRIDGE_RETAINED_HYDRO_RULE_FAMILIES
        )
        self._heavy_polycycle_generated_fused_rescue_rules = tuple(
            rule
            for rule in self._rules
            if rule.family in _HEAVY_POLYCYCLE_GENERATED_FUSED_RESCUE_RULE_FAMILIES
        )
        self._fast_poly_pre_route_senior_rules = tuple(
            rule for rule in self._rules if rule.family in _FAST_POLY_PRE_ROUTE_SENIOR_FAMILIES
        )
        self._fast_poly_low_hetero_senior_rules = tuple(
            rule for rule in self._rules if rule.family in _FAST_POLY_LOW_HETERO_SENIOR_FAMILIES
        )
        self._fast_multi_monocycle_senior_rules = tuple(
            rule for rule in self._rules if rule.family in _FAST_MULTI_MONOCYCLE_SENIOR_RULE_FAMILIES
        )
        self._explicit_isotope_hydrogen_retained_rules = tuple(
            rule for rule in self._rules if rule.family in _EXPLICIT_ISOTOPE_HYDROGEN_RETAINED_RULE_FAMILIES
        )
        self._fast_poly_route_miss_fallback_rules = tuple(
            rule for rule in self._rules if rule.family in _FAST_POLY_ROUTE_MISS_FALLBACK_RULE_FAMILIES
        )
        self._ranked_polycycle_selector_rules = tuple(
            rule for rule in self._rules if rule.family in _RANKED_POLYCYCLE_SELECTOR_RULE_FAMILIES
        )
        self._temporary_complexity_ranked_rescue_rules = tuple(
            rule for rule in self._rules if rule.family in _TEMPORARY_COMPLEXITY_RANKED_RESCUE_FAMILIES
        )
        self._temporary_complexity_bounded_rescue_rules = tuple(
            rule for rule in self._rules if rule.family in _TEMPORARY_COMPLEXITY_BOUNDED_RESCUE_FAMILIES
        )
        self._spiro_selector_rules = tuple(
            rule for rule in self._rules if rule.family in _SPIRO_SELECTOR_RULE_FAMILIES
        )
        self._fast_poly_route_miss_retained_defer_ring_rules = tuple(
            rule
            for rule in self._fast_poly_route_miss_fallback_rules
            if rule.family not in _FAST_POLY_BRIDGE_ROUTE_RULE_FAMILIES
            and rule.family not in _FAST_POLY_RETAINED_DEFER_LATE_ACYCLIC_FAMILIES
        )
        self._fast_poly_route_miss_retained_defer_acyclic_rules = tuple(
            rule
            for rule in self._fast_poly_route_miss_fallback_rules
            if rule.family in _FAST_POLY_RETAINED_DEFER_LATE_ACYCLIC_FAMILIES
        )
        self._fragment_rule = FragmentAssemblyRule(self._Chem, self)

    def name(self, mol, *, telemetry: CandidateTelemetry | None = None) -> str:
        atom_count = mol.GetNumAtoms()
        if atom_count == 1:
            simple_name = simple_fragment_name(mol, self._Chem)
            if simple_name:
                _record_direct_route(telemetry, "single_atom")
                return _finalize_direct_name(mol, self._Chem, simple_name)
        fragment_atom_sets = self._Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
        if len(fragment_atom_sets) > 1:
            fragments = self._Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            _record_direct_route(telemetry, "fragment_assembly")
            return _finalize_direct_name(mol, self._Chem, self._fragment_rule.name(fragments))
        if 2 <= atom_count <= 24:
            standalone_polyatomic_name = standalone_polyatomic_fragment_name(mol, self._Chem)
            if standalone_polyatomic_name:
                _record_direct_route(telemetry, "standalone_polyatomic")
                return _finalize_direct_name(mol, self._Chem, standalone_polyatomic_name)
        if atom_count >= 2:
            polyatomic_name = common_polyatomic_fragment_name(mol, self._Chem)
            if polyatomic_name:
                _record_direct_route(telemetry, "common_polyatomic")
                return _finalize_direct_name(mol, self._Chem, polyatomic_name)
        temporary_complexity_bailout = _temporary_complexity_bailout_name(mol)
        if temporary_complexity_bailout and not _temporary_complexity_fast_route_rescue_allowed(mol):
            ctx: MoleculeContext | None = None
            if _temporary_heavy_polycycle_generated_fused_rescue_allowed(mol):
                ctx = MoleculeContext(mol, self._Chem, telemetry=telemetry)
                heavy_polycycle_name = _fast_heavy_polycycle_generated_fused_name(
                    mol,
                    self._Chem,
                    ctx,
                    self._heavy_polycycle_generated_fused_rescue_rules,
                    telemetry=telemetry,
                )
                if heavy_polycycle_name:
                    return heavy_polycycle_name
            if _temporary_complexity_ranked_polycycle_rescue_allowed(mol):
                if ctx is None:
                    ctx = MoleculeContext(mol, self._Chem, telemetry=telemetry)
                ranked_rescue_name = _temporary_complexity_ranked_polycycle_rescue_name(
                    mol,
                    self._Chem,
                    ctx,
                    self._temporary_complexity_ranked_rescue_rules,
                    telemetry=telemetry,
                )
                if ranked_rescue_name:
                    return ranked_rescue_name
            if _temporary_complexity_bounded_rescue_allowed(mol):
                if ctx is None:
                    ctx = MoleculeContext(mol, self._Chem, telemetry=telemetry)
                bounded_rescue_name = _temporary_complexity_bounded_rescue_name(
                    mol,
                    self._Chem,
                    ctx,
                    self._temporary_complexity_bounded_rescue_rules,
                    telemetry=telemetry,
                )
                if bounded_rescue_name:
                    return bounded_rescue_name
            _record_direct_route(telemetry, "temporary_complexity_bailout")
            return ""

        ctx = MoleculeContext(mol, self._Chem, telemetry=telemetry)
        if telemetry is not None:
            _record_route_plan(telemetry, ctx.route_plan)
        use_bridgehead_polycycle_selector = _should_use_bridgehead_polycycle_selector(mol, ctx)
        use_ranked_polycycle_selector = use_bridgehead_polycycle_selector or _should_use_nonspiro_polycycle_selector(
            mol,
            ctx,
        )
        if use_ranked_polycycle_selector:
            ranked_polycycle_name = _select_route_miss_prepared_candidate(
                _prepare_candidates(mol, self._Chem, ctx, self._ranked_polycycle_selector_rules),
                telemetry=telemetry,
            )
            if ranked_polycycle_name:
                return ranked_polycycle_name
        isotope_hydrogen_retained_name = ""
        if not use_bridgehead_polycycle_selector:
            multi_monocycle_senior_name = _fast_multi_monocycle_senior_name(
                mol,
                self._Chem,
                ctx,
                self._fast_multi_monocycle_senior_rules,
                telemetry=telemetry,
            )
            if multi_monocycle_senior_name:
                return multi_monocycle_senior_name
            fast_monocycle_name = _fast_monocycle_system_name(
                mol,
                self._Chem,
                ctx,
                self._fast_monocycle_system_rules,
                unsaturated_acyclic_rules=self._fast_monocycle_unsaturated_acyclic_rules,
                telemetry=telemetry,
            )
            if fast_monocycle_name:
                return fast_monocycle_name
            pre_route_senior_name = _fast_poly_pre_route_senior_name(
                mol,
                self._Chem,
                ctx,
                self._fast_poly_pre_route_senior_rules,
                telemetry=telemetry,
            )
            if pre_route_senior_name:
                return pre_route_senior_name
            low_hetero_senior_name = _fast_poly_low_hetero_senior_name(
                mol,
                self._Chem,
                ctx,
                self._fast_poly_low_hetero_senior_rules,
                telemetry=telemetry,
            )
            if low_hetero_senior_name:
                return low_hetero_senior_name
            isotope_hydrogen_retained_name = _explicit_isotope_hydrogen_retained_name(
                mol,
                self._Chem,
                ctx,
                self._explicit_isotope_hydrogen_retained_rules,
                telemetry=telemetry,
            )
        if isotope_hydrogen_retained_name:
            return isotope_hydrogen_retained_name
        if _should_use_spiro_selector(mol, ctx):
            spiro_selector_name = _select_stable_spiro_prepared_candidate(
                mol,
                ctx,
                _prepare_candidates(mol, self._Chem, ctx, self._spiro_selector_rules),
                telemetry=telemetry,
            )
            if spiro_selector_name:
                return spiro_selector_name
        routed_polycyclic_name = ""
        if (
            not use_bridgehead_polycycle_selector
            and mol.GetRingInfo().NumRings() >= 2
            and _largest_ring_system_cycle_rank(ctx) >= 2
        ):
            routed_polycyclic_name = _fast_polycyclic_route_name(
                mol,
                self._Chem,
                ctx,
                component_spiro_rules=self._fast_component_spiro_route_rules,
                component_spiro_skeletal_replacement_rules=self._fast_component_spiro_skeletal_replacement_rules,
                bridge_rules=self._fast_poly_bridge_route_rules,
                bridge_retained_hydro_rules=self._fast_poly_bridge_retained_hydro_rules,
                telemetry=telemetry,
            )
        if routed_polycyclic_name:
            return routed_polycyclic_name
        if not use_bridgehead_polycycle_selector and mol.GetRingInfo().NumRings() >= 3:
            if mol_cache_get(mol, FAST_POLY_RETAINED_DEFER_CACHE_KEY):
                route_miss_name = _retained_deferred_poly_route_miss_name(
                    mol,
                    self._Chem,
                    ctx,
                    route_miss_ring_rules=self._fast_poly_route_miss_retained_defer_ring_rules,
                    route_miss_acyclic_rules=self._fast_poly_route_miss_retained_defer_acyclic_rules,
                    bridge_rules=self._fast_poly_bridge_route_rules,
                    telemetry=telemetry,
                )
            elif mol.GetRingInfo().NumRings() <= 3 and _largest_ring_system_cycle_rank(ctx) <= 1:
                route_miss_name = _select_route_miss_prepared_candidate(
                    _prepare_candidates(mol, self._Chem, ctx, self._fast_poly_route_miss_fallback_rules),
                    telemetry=telemetry,
                )
            else:
                route_miss_name = _select_route_miss_prepared_candidate(
                    _prepare_candidates(
                        mol,
                        self._Chem,
                        ctx,
                        self._fast_poly_route_miss_fallback_rules,
                    ),
                    telemetry=telemetry,
                )
            if route_miss_name:
                retained_hydro_name = _fast_retained_hydro_override_name(
                    mol,
                    self._Chem,
                    ctx,
                    self._fast_poly_bridge_retained_hydro_rules,
                    telemetry=telemetry,
                )
                if retained_hydro_name:
                    return retained_hydro_name
                return route_miss_name
        if temporary_complexity_bailout:
            if _temporary_complexity_bounded_rescue_allowed(mol):
                bounded_rescue_name = _temporary_complexity_bounded_rescue_name(
                    mol,
                    self._Chem,
                    ctx,
                    self._temporary_complexity_bounded_rescue_rules,
                    telemetry=telemetry,
                )
                if bounded_rescue_name:
                    return bounded_rescue_name
            _record_direct_route(telemetry, "temporary_complexity_fast_route_miss")
            return ""
        best_name = _select_best_prepared_candidate(
            _prepare_candidates(mol, self._Chem, ctx, self._rules),
            telemetry=telemetry,
        )

        if best_name:
            return best_name
        if _should_skip_complex_empty_fallback(mol, ctx):
            if telemetry is not None:
                telemetry.legacy_fallback_skipped_count[ctx.route_plan.primary_route] += 1
            return ""

        fallback_evaluation_length, fallback_max_length = _empty_fallback_limits(mol, self._Chem)
        if telemetry is not None:
            telemetry.legacy_fallback_attempt_count[ctx.route_plan.primary_route] += 1
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
            fallback_ctx = MoleculeContext(
                mol,
                self._Chem,
                telemetry=telemetry,
                cache=ctx.cache.shared_for_fallback(),
            )
            fallback_name = _fallback_first_valid_or_selected_candidate_name(
                mol,
                self._Chem,
                fallback_ctx,
                self._rules,
                telemetry=telemetry,
                max_length=fallback_evaluation_length,
                output_max_length=fallback_max_length,
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
        if telemetry is not None:
            if fallback_name:
                telemetry.legacy_fallback_success_count[ctx.route_plan.primary_route] += 1
            else:
                telemetry.legacy_fallback_empty_count[ctx.route_plan.primary_route] += 1
        return fallback_name


def _fast_monocycle_system_name(
    mol,
    Chem,
    ctx: MoleculeContext,
    rules: Iterable[RuleSpec],
    *,
    unsaturated_acyclic_rules: Iterable[RuleSpec] = (),
    telemetry: CandidateTelemetry | None = None,
) -> str:
    if not _can_fast_stage_monocycle_system(mol):
        return ""
    prepared = _prepare_candidates(mol, Chem, ctx, rules)
    selected = _select_best_prepared_candidate_item(prepared)
    if selected is None:
        return ""
    if unsaturated_acyclic_rules and _single_ring_choice_can_consider_unsaturated_acyclic(selected):
        extended_unsaturated_acyclic_context = _single_ring_choice_uses_extended_unsaturated_acyclic_context(selected)
        alternatives = _prepare_candidates(mol, Chem, ctx, unsaturated_acyclic_rules)
        if alternatives:
            keyed_items = [
                (_prepared_candidate_selection_key(item, prepared + alternatives), item)
                for item in [selected, *alternatives]
            ]
            alternative = _unsaturated_acyclic_alternative_for_single_ring_choice(keyed_items, selected)
            if alternative is not None:
                selected = alternative
            elif extended_unsaturated_acyclic_context:
                return ""
        elif extended_unsaturated_acyclic_context:
            return ""
    return _record_and_return_prepared_candidate(selected, telemetry=telemetry)


def _fast_multi_monocycle_senior_name(
    mol,
    Chem,
    ctx: MoleculeContext,
    rules: Iterable[RuleSpec],
    *,
    telemetry: CandidateTelemetry | None = None,
) -> str:
    if not _can_fast_stage_multi_monocycle_senior(ctx):
        return ""
    prepared = [
        item
        for item in _prepare_candidates(mol, Chem, ctx, rules)
        if item.candidate.family not in _FAST_MULTI_MONOCYCLE_SENIOR_BLOCKED_FAMILIES
    ]
    return _select_best_prepared_candidate(
        prepared,
        telemetry=telemetry,
    )


def _fast_poly_pre_route_senior_name(
    mol,
    Chem,
    ctx: MoleculeContext,
    rules: Iterable[RuleSpec],
    *,
    telemetry: CandidateTelemetry | None = None,
) -> str:
    if not _can_fast_stage_poly_pre_route_senior(ctx):
        return ""
    prepared = _prepare_candidates(mol, Chem, ctx, rules)
    selected = _select_best_prepared_candidate_item(prepared)
    if _should_defer_fast_senior_fragile_external_unsaturated_acyclic(selected):
        return ""
    return _record_and_return_prepared_candidate(selected, telemetry=telemetry)


def _fast_poly_low_hetero_senior_name(
    mol,
    Chem,
    ctx: MoleculeContext,
    rules: Iterable[RuleSpec],
    *,
    telemetry: CandidateTelemetry | None = None,
) -> str:
    if not _can_fast_stage_low_hetero_poly_senior(ctx):
        return ""
    prepared = _prepare_candidates(mol, Chem, ctx, rules)
    selected = _select_best_prepared_candidate_item(prepared)
    if selected is not None and _hydrazine_parent_can_consider_systematic_alternative(selected):
        return ""
    if _should_defer_fast_senior_fragile_external_unsaturated_acyclic(selected):
        return ""
    return _record_and_return_prepared_candidate(selected, telemetry=telemetry)


def _should_defer_fast_senior_fragile_external_unsaturated_acyclic(
    selected: _PreparedCandidate | None,
) -> bool:
    if selected is None:
        return False
    candidate = selected.candidate
    return (
        candidate.family == "sulfur-oxo-acyclic-skeleton-with-ring-substituents"
        and candidate.parent_atoms
        and candidate.parent_external_unsaturated_bond_count >= 2
    )


def _explicit_isotope_hydrogen_retained_name(
    mol,
    Chem,
    ctx: MoleculeContext,
    rules: Iterable[RuleSpec],
    *,
    telemetry: CandidateTelemetry | None = None,
) -> str:
    if not _mol_has_explicit_isotope_hydrogen(mol):
        return ""
    prepared = _prepare_candidates(mol, Chem, ctx, rules)
    if not prepared:
        return ""
    selected = _select_best_prepared_candidate_item(prepared)
    if selected is None or not _candidate_name_mentions_isotope(selected.name):
        return ""
    return _record_and_return_prepared_candidate(selected, telemetry=telemetry)


def _fast_heavy_polycycle_generated_fused_name(
    mol,
    Chem,
    ctx: MoleculeContext,
    rules: Iterable[RuleSpec],
    *,
    telemetry: CandidateTelemetry | None = None,
) -> str:
    return _first_valid_candidate_name(
        mol,
        Chem,
        ctx,
        rules,
        max_length=_HEAVY_POLYCYCLE_GENERATED_FUSED_MAX_LENGTH,
        output_max_length=_HEAVY_POLYCYCLE_GENERATED_FUSED_MAX_LENGTH,
        telemetry=telemetry,
    )


def _temporary_complexity_ranked_polycycle_rescue_name(
    mol,
    Chem,
    ctx: MoleculeContext,
    rules: Iterable[RuleSpec],
    *,
    telemetry: CandidateTelemetry | None = None,
) -> str:
    prepared = [
        item
        for item in _prepare_candidates(mol, Chem, ctx, rules)
        if _temporary_complexity_ranked_polycycle_rescue_is_safe(item)
    ]
    selected = _select_best_prepared_candidate_item(prepared)
    if selected is None:
        return ""
    return _record_and_return_prepared_candidate(selected, telemetry=telemetry)


def _temporary_complexity_ranked_polycycle_rescue_is_safe(
    selected: _PreparedCandidate | None,
) -> bool:
    if selected is None:
        return False
    candidate = selected.candidate
    if candidate.family not in _TEMPORARY_COMPLEXITY_RANKED_RESCUE_FAMILIES:
        return False
    if candidate_safety_reasons(candidate):
        return False
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return False
    return True


def _temporary_complexity_bounded_rescue_name(
    mol,
    Chem,
    ctx: MoleculeContext,
    rules: Iterable[RuleSpec],
    *,
    telemetry: CandidateTelemetry | None = None,
) -> str:
    prepared = [
        item
        for item in _prepare_candidates(mol, Chem, ctx, rules)
        if _temporary_complexity_bounded_rescue_is_safe(item)
    ]
    selected = _select_best_prepared_candidate_item(prepared)
    if selected is None:
        return ""
    return _record_and_return_prepared_candidate(selected, telemetry=telemetry)


def _temporary_complexity_bounded_rescue_is_safe(selected: _PreparedCandidate | None) -> bool:
    if selected is None:
        return False
    candidate = selected.candidate
    if candidate.family not in _TEMPORARY_COMPLEXITY_BOUNDED_RESCUE_FAMILIES:
        return False
    if candidate_safety_reasons(candidate):
        return False
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return False
    return True


def _fast_polycyclic_route_name(
    mol,
    Chem,
    ctx: MoleculeContext,
    *,
    component_spiro_rules: Iterable[RuleSpec],
    component_spiro_skeletal_replacement_rules: Iterable[RuleSpec],
    bridge_rules: Iterable[RuleSpec],
    bridge_retained_hydro_rules: Iterable[RuleSpec],
    telemetry: CandidateTelemetry | None = None,
) -> str:
    ring_count = mol.GetRingInfo().NumRings()
    if ring_count < 2:
        return ""
    if ctx.has_component_spiro_topology:
        component_prepared = _prepare_candidates(mol, Chem, ctx, component_spiro_rules)
        selected = _select_best_prepared_candidate_item(component_prepared)
        if selected is not None:
            if _component_spiro_route_needs_skeletal_replacement_alternative(selected):
                replacement_prepared = _prepare_candidates(
                    mol,
                    Chem,
                    ctx,
                    component_spiro_skeletal_replacement_rules,
                )
                if replacement_prepared:
                    selected = (
                        _select_best_prepared_candidate_item(component_prepared + replacement_prepared)
                        or selected
                    )
            return _record_and_return_prepared_candidate(selected, telemetry=telemetry)
    if ring_count < 3:
        return ""
    had_route_prop = mol.HasProp(FAST_POLY_BRIDGE_ROUTE_PROP)
    previous_route_prop = mol.GetProp(FAST_POLY_BRIDGE_ROUTE_PROP) if had_route_prop else ""
    mol.SetProp(FAST_POLY_BRIDGE_ROUTE_PROP, "1")
    try:
        bridge_selected = _first_valid_prepared_candidate(
            mol,
            Chem,
            ctx,
            bridge_rules,
        )
        if bridge_selected is None:
            return ""
        if had_route_prop:
            mol.SetProp(FAST_POLY_BRIDGE_ROUTE_PROP, previous_route_prop)
        else:
            mol.ClearProp(FAST_POLY_BRIDGE_ROUTE_PROP)
        retained_hydro_name = _fast_retained_hydro_override_name(
            mol,
            Chem,
            ctx,
            bridge_retained_hydro_rules,
            telemetry=telemetry,
        )
        mol.SetProp(FAST_POLY_BRIDGE_ROUTE_PROP, "1")
        if retained_hydro_name:
            return retained_hydro_name
        return _record_and_return_prepared_candidate(bridge_selected, telemetry=telemetry)
    finally:
        if had_route_prop:
            mol.SetProp(FAST_POLY_BRIDGE_ROUTE_PROP, previous_route_prop)
        else:
            mol.ClearProp(FAST_POLY_BRIDGE_ROUTE_PROP)


def _component_spiro_route_needs_skeletal_replacement_alternative(
    selected: _PreparedCandidate,
) -> bool:
    candidate = selected.candidate
    if candidate.family != "component-spiro-parent":
        return False
    if candidate.charge is not None:
        return False
    if candidate_safety_reasons(candidate):
        return False
    if candidate.parent_atoms and candidate_has_complete_graph(candidate):
        if candidate.render_rank is not None and candidate.render_rank >= 300:
            return True
        if (
            candidate.parent_abs_formal_charge_sum
            or candidate.parent_formal_charge_atom_count
            or candidate.molecule_abs_formal_charge_sum
            or candidate.molecule_formal_charge_atom_count
        ):
            return False
        if candidate.render_rank != 0:
            return False
        if len(candidate.parent_atoms) not in {13, 14}:
            return False
        return candidate.parent_element_counts in {
            (("C", 12), ("N", 1)),
            (("C", 13), ("N", 1)),
            (("C", 12), ("O", 1)),
            (("C", 13), ("O", 1)),
            (("C", 11), ("N", 1), ("O", 1)),
            (("C", 10), ("N", 2), ("O", 1)),
            (("C", 11), ("N", 2)),
            (("C", 12), ("N", 2)),
            (("C", 12), ("N", 1), ("S", 1)),
        }
    return True


def _fast_retained_hydro_override_name(
    mol,
    Chem,
    ctx: MoleculeContext,
    rules: Iterable[RuleSpec],
    *,
    telemetry: CandidateTelemetry | None = None,
) -> str:
    prepared = _prepare_candidates(mol, Chem, ctx, rules)
    if not prepared:
        return ""
    keyed_items = [(_prepared_candidate_selection_key(item, prepared), item) for item in prepared]
    for _key, item in sorted(keyed_items, key=lambda keyed: keyed[0]):
        if item.candidate.family != "fused-retained-parent":
            continue
        selected = _generated_hydro_fused_alternative_for_retained_fused_choice(
            keyed_items,
            item,
        ) or _hydro_retained_alternative_for_retained_fused_choice(keyed_items, item)
        if selected is not None:
            return _record_and_return_prepared_candidate(selected, telemetry=telemetry)
    return ""


def _retained_deferred_poly_route_miss_name(
    mol,
    Chem,
    ctx: MoleculeContext,
    *,
    route_miss_ring_rules: Iterable[RuleSpec],
    route_miss_acyclic_rules: Iterable[RuleSpec],
    bridge_rules: Iterable[RuleSpec],
    telemetry: CandidateTelemetry | None = None,
) -> str:
    # Acyclic route-miss candidates are expensive on retained/polycyclic misses.
    # Try graph-first ring/polycyclic candidates first and only add acyclic
    # context when the selected ring candidate already points at a bridge retry.
    prepared = _prepare_candidates(mol, Chem, ctx, route_miss_ring_rules)
    selected = _select_best_prepared_candidate_item(prepared)
    if selected is not None and _retained_deferred_selected_should_bridge_retry(selected):
        prepared_with_acyclic_context = prepared + _prepare_candidates(mol, Chem, ctx, route_miss_acyclic_rules)
        selected = _select_best_prepared_candidate_item(prepared_with_acyclic_context)
    if selected is None:
        bridge_name = _fast_poly_bridge_name_with_retained_defer_bypass(
            mol,
            Chem,
            ctx,
            bridge_rules,
            telemetry=telemetry,
        )
        if bridge_name:
            return bridge_name
        return ""
    if _retained_deferred_selected_should_bridge_retry(selected):
        bridge_name = _fast_poly_bridge_name_with_retained_defer_bypass(
            mol,
            Chem,
            ctx,
            bridge_rules,
            telemetry=telemetry,
        )
        if bridge_name:
            return bridge_name
    if _should_defer_fast_senior_fragile_external_unsaturated_acyclic(selected):
        return ""
    return _record_and_return_prepared_candidate(selected, telemetry=telemetry)


def _retained_deferred_selected_should_bridge_retry(selected: _PreparedCandidate) -> bool:
    if _is_stable_retained_ring_scaffold_selection(selected):
        return False
    if _is_stable_retained_topology_selection(selected):
        return False
    if _is_stable_skeletal_replacement_selection(selected):
        return False
    return (
        selected.candidate.family in _FAST_POLY_RETAINED_DEFER_BRIDGE_RETRY_FAMILIES
        or selected.producer_family in _FAST_POLY_RETAINED_DEFER_BRIDGE_RETRY_FAMILIES
        or _retained_deferred_selected_template_is_fragile(selected)
    )


def _is_stable_retained_ring_scaffold_selection(selected: _PreparedCandidate) -> bool:
    candidate = selected.candidate
    if selected.producer_family != "retained-ring-scaffold-parent":
        return False
    if candidate.source_family != "opsin-retained-ring-cml":
        return False
    if candidate.template_name not in _STABLE_RETAINED_RING_SCAFFOLD_TEMPLATE_NAMES and candidate.stereo is not None:
        return False
    if not candidate.template_name or _retained_deferred_selected_template_is_fragile(selected):
        return False
    return not candidate_safety_reasons(candidate)


def _is_stable_retained_topology_selection(selected: _PreparedCandidate) -> bool:
    candidate = selected.candidate
    if selected.producer_family != "retained-topology-replacement-fused-parent":
        return False
    if candidate.family != "retained-topology-replacement-fused-parent":
        return False
    if candidate.source_family != "opsin-retained-ring-cml":
        return False
    if candidate.template_name not in FAST_POLY_RETAINED_TOPOLOGY_DEFER_TEMPLATE_NAMES:
        return False
    if candidate.render_rank is not None and candidate.render_rank > 2:
        return False
    if candidate.stereo is not None:
        return False
    if _retained_deferred_selected_template_is_fragile(selected):
        return False
    if not candidate_has_complete_graph(candidate):
        return False
    if candidate.prefixes or candidate.suffixes or candidate.charge is not None:
        return False
    return not candidate_safety_reasons(candidate)


def _is_stable_skeletal_replacement_selection(selected: _PreparedCandidate) -> bool:
    candidate = selected.candidate
    if selected.producer_family != "skeletal-replacement-fused-parent":
        return False
    if candidate.family != "skeletal-replacement-fused-parent":
        return False
    if candidate.template_name not in {"fluoren", "fluorene"}:
        return False
    if candidate.source_family not in {"arylGroups:ring:ring", "opsin-retained-ring-cml"}:
        return False
    if candidate.parent_element_counts != (("C", 12), ("O", 1)):
        return False
    if candidate.stereo is not None:
        return False
    return not candidate_safety_reasons(candidate)


_STABLE_RETAINED_RING_SCAFFOLD_TEMPLATE_NAMES = frozenset(
    {
        "pentacene",
        "hexacene",
        "heptacene",
        "octacene",
        "nonacene",
        "decacene",
        "undecacene",
        "dodecacene",
        "pentaphene",
        "hexaphene",
        "heptaphene",
        "octaphene",
        "nonaphene",
        "decaphene",
        "undecaphene",
        "dodecaphene",
    }
)


def _retained_deferred_selected_template_is_fragile(selected: _PreparedCandidate) -> bool:
    candidate = selected.candidate
    if not candidate.template_name:
        return False
    return (
        candidate.source_family,
        candidate.template_name,
    ) in FAST_POLY_RETAINED_DEFER_FRAGILE_TEMPLATE_KEYS


def _fast_poly_bridge_name_with_retained_defer_bypass(
    mol,
    Chem,
    ctx: MoleculeContext,
    bridge_rules: Iterable[RuleSpec],
    *,
    telemetry: CandidateTelemetry | None = None,
) -> str:
    had_route_prop = mol.HasProp(FAST_POLY_BRIDGE_ROUTE_PROP)
    previous_route_prop = mol.GetProp(FAST_POLY_BRIDGE_ROUTE_PROP) if had_route_prop else ""
    had_bypass_prop = mol.HasProp(FAST_POLY_RETAINED_DEFER_BYPASS_PROP)
    previous_bypass_prop = mol.GetProp(FAST_POLY_RETAINED_DEFER_BYPASS_PROP) if had_bypass_prop else ""
    mol.SetProp(FAST_POLY_BRIDGE_ROUTE_PROP, "1")
    mol.SetProp(FAST_POLY_RETAINED_DEFER_BYPASS_PROP, "1")
    try:
        return _first_valid_candidate_name(
            mol,
            Chem,
            ctx,
            bridge_rules,
            telemetry=telemetry,
        )
    finally:
        if had_route_prop:
            mol.SetProp(FAST_POLY_BRIDGE_ROUTE_PROP, previous_route_prop)
        else:
            mol.ClearProp(FAST_POLY_BRIDGE_ROUTE_PROP)
        if had_bypass_prop:
            mol.SetProp(FAST_POLY_RETAINED_DEFER_BYPASS_PROP, previous_bypass_prop)
        else:
            mol.ClearProp(FAST_POLY_RETAINED_DEFER_BYPASS_PROP)


def _should_skip_complex_empty_fallback(mol, ctx: MoleculeContext) -> bool:
    """Return whether an empty molecule should bypass the final fallback replay."""

    return (
        ctx.route_plan.primary_route in _EMPTY_FALLBACK_SKIPPED_PRIMARY_ROUTES
        and not _has_empty_fallback_high_signal(ctx)
    )


def _temporary_complexity_bailout_name(mol) -> bool:
    """Temporarily skip expensive weak complex classes during speed-first work."""

    heavy_atoms = mol.GetNumHeavyAtoms()
    ring_count = mol.GetRingInfo().NumRings()
    if heavy_atoms < 35 or ring_count < 2:
        return False
    has_isotope = False
    has_charge = False
    has_phosphorus = False
    nonring_heavy_atoms = 0
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num <= 1:
            continue
        if not atom.IsInRing():
            nonring_heavy_atoms += 1
        if atom.GetIsotope():
            has_isotope = True
        if atom.GetFormalCharge():
            has_charge = True
        if atomic_num == 15:
            has_phosphorus = True
    return (
        has_isotope
        or (has_charge and ring_count >= 2)
        or (has_phosphorus and ring_count >= 1)
        or (heavy_atoms >= 40 and ring_count >= 8)
        or (heavy_atoms >= 45 and nonring_heavy_atoms >= 18 and ring_count >= 3)
    )


def _temporary_complexity_fast_route_rescue_allowed(mol) -> bool:
    """Allow bounded bailout classes to try staged routes before returning empty."""

    heavy_atoms = mol.GetNumHeavyAtoms()
    ring_count = mol.GetRingInfo().NumRings()
    if heavy_atoms >= 40 and ring_count >= 8:
        return False
    has_charge = False
    has_phosphorus = False
    has_isotope = False
    hetero_heavy_atoms = 0
    chiral_atoms = 0
    nonring_heavy_atoms = 0
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num <= 1:
            continue
        if int(atom.GetChiralTag()) != 0:
            chiral_atoms += 1
        if atomic_num != 6:
            hetero_heavy_atoms += 1
        if not atom.IsInRing():
            nonring_heavy_atoms += 1
        if atom.GetFormalCharge():
            has_charge = True
        if atomic_num == 15:
            has_phosphorus = True
        if atom.GetIsotope():
            has_isotope = True
    if has_charge or has_phosphorus or has_isotope:
        return True
    if heavy_atoms >= 45 and nonring_heavy_atoms >= 18 and ring_count >= 3:
        return (
            heavy_atoms < 50
            and 4 <= ring_count <= 5
            and nonring_heavy_atoms <= 24
            and 10 <= hetero_heavy_atoms <= 12
            and chiral_atoms <= 4
        )
    return True


def _temporary_complexity_bounded_rescue_allowed(mol) -> bool:
    """Allow a narrow full-selector rescue for high-yield large side-chain polycycles."""

    heavy_atoms = mol.GetNumHeavyAtoms()
    if heavy_atoms < 45 or heavy_atoms >= 50:
        return False
    ring_count = mol.GetRingInfo().NumRings()
    largest_ring_system = _largest_ring_system_ring_count(mol)
    if (ring_count, largest_ring_system) not in {(4, 1), (5, 2)}:
        return False

    nonring_heavy_atoms = 0
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num <= 1:
            continue
        if atom.GetIsotope() or atom.GetFormalCharge() or atomic_num == 15:
            return False
        if not atom.IsInRing():
            nonring_heavy_atoms += 1
    if ring_count == 4 and largest_ring_system == 1:
        return (
            (heavy_atoms == 46 and 22 <= nonring_heavy_atoms <= 26)
            or (heavy_atoms == 47 and 24 <= nonring_heavy_atoms <= 27)
            or (heavy_atoms == 49 and nonring_heavy_atoms == 28)
        )
    return (
        (heavy_atoms == 45 and nonring_heavy_atoms in {19, 20, 21, 25})
        or (heavy_atoms == 46 and nonring_heavy_atoms in {19, 21, 22})
        or (heavy_atoms == 48 and nonring_heavy_atoms in {24, 28})
        or (heavy_atoms == 49 and nonring_heavy_atoms == 25)
    )


def _temporary_heavy_polycycle_generated_fused_rescue_allowed(mol) -> bool:
    if mol.GetNumHeavyAtoms() < 40 or mol.GetRingInfo().NumRings() < 8:
        return False
    chiral_atoms = 0
    hetero_heavy_atoms = 0
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num <= 1:
            continue
        if int(atom.GetChiralTag()) != 0:
            chiral_atoms += 1
        if atomic_num != 6:
            hetero_heavy_atoms += 1
    return chiral_atoms == 3 and 9 <= hetero_heavy_atoms <= 12


def _temporary_complexity_ranked_polycycle_rescue_allowed(mol) -> bool:
    """Gate ranked rescue to high-yield large fused/bridged polycycles."""

    heavy_atoms = mol.GetNumHeavyAtoms()
    ring_count = mol.GetRingInfo().NumRings()
    if heavy_atoms < 40:
        return False
    largest_ring_system = _largest_ring_system_ring_count(mol)
    spiro_atoms = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    bridgehead_atoms = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    if spiro_atoms != 0 or bridgehead_atoms < 2:
        return False
    hetero_heavy_atoms = 0
    chiral_atoms = 0
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num <= 1:
            continue
        if atom.GetIsotope():
            return False
        if int(atom.GetChiralTag()) != 0:
            chiral_atoms += 1
        if atomic_num != 6:
            hetero_heavy_atoms += 1
    if (
        bridgehead_atoms >= 4
        and ring_count >= 4
        and largest_ring_system >= 3
        and hetero_heavy_atoms <= 17
        and chiral_atoms <= 6
    ):
        return True
    if (
        bridgehead_atoms >= 2
        and 6 <= ring_count <= 8
        and largest_ring_system >= 5
        and hetero_heavy_atoms <= 16
        and chiral_atoms <= 6
    ):
        return True
    if ring_count < 5 or ring_count > 9:
        return False
    if largest_ring_system < 5:
        return False
    if chiral_atoms > 5:
        return False
    return hetero_heavy_atoms <= 12 or (largest_ring_system >= 8 and hetero_heavy_atoms <= 14)


def _has_empty_fallback_high_signal(ctx: MoleculeContext) -> bool:
    return bool(
        ctx.profile.has_functional_skeleton_signal
        or ctx.profile.has_imino_skeleton_signal
        or ctx.facts.has_sulfur
        or ctx.facts.has_phosphorus
        or ctx.facts.has_formal_charge
    )


def _is_complex_polycycle_fallback_bounded(mol, ctx: MoleculeContext) -> bool:
    """Return true for complex polycycles where empty fallback is too expensive."""

    if mol.GetNumHeavyAtoms() < 40:
        return False
    if mol.GetRingInfo().NumRings() < 6:
        return False
    if ctx.features.get("max_ring_system_ring_count", 0) < 3:
        return False
    return True


def _record_direct_route(telemetry: CandidateTelemetry | None, route: str) -> None:
    if telemetry is None:
        return
    telemetry.primary_route_count[route] += 1
    telemetry.route_plan_count[route] += 1


def _record_route_plan(telemetry: CandidateTelemetry | None, route_plan: RoutePlan) -> None:
    if telemetry is None:
        return
    telemetry.primary_route_count[route_plan.primary_route] += 1
    telemetry.route_plan_count[route_plan.telemetry_key] += 1
    if route_plan.uses_fused_resources:
        telemetry.route_uses_fused_resources[route_plan.primary_route] += 1
    if route_plan.uses_polycyclic_search:
        telemetry.route_uses_polycyclic_search[route_plan.primary_route] += 1


def _first_valid_candidate_name(
    mol,
    Chem,
    ctx: MoleculeContext,
    rules: Iterable[RuleSpec],
    *,
    telemetry: CandidateTelemetry | None = None,
    max_length: int = _DEFAULT_MAX_NAME_LENGTH,
    output_max_length: int | None = None,
) -> str:
    rules = compile_rules_for_context(ctx, rules, telemetry=telemetry)
    for rule in rules:
        if telemetry is not None:
            telemetry.rule_evaluations[rule.family] += 1
        if not rule.applies(ctx):
            if telemetry is not None:
                telemetry.rule_gate_skips[rule.family] += 1
            continue
        if telemetry is None:
            for candidate in rule.candidates(ctx):
                if not candidate.name:
                    continue
                output_name = _finalize_candidate_name(mol, Chem, candidate, max_length=max_length)
                if output_name and (output_max_length is None or len(output_name) <= output_max_length):
                    return output_name
            continue

        telemetry.rule_applies[rule.family] += 1
        started = time.perf_counter()
        previous_rule_family = telemetry.active_rule_family
        telemetry.active_rule_family = rule.family
        try:
            for candidate in rule.candidates(ctx):
                telemetry.candidate_count[rule.family] += 1
                _record_candidate_fact_telemetry(telemetry, rule.family, candidate)
                if not candidate.name:
                    telemetry.rejected_candidate_count[rule.family] += 1
                    continue
                output_name = _finalize_candidate_name(mol, Chem, candidate, max_length=max_length)
                if not output_name:
                    telemetry.rejected_candidate_count[rule.family] += 1
                    continue
                if output_max_length is not None and len(output_name) > output_max_length:
                    telemetry.rejected_candidate_count[rule.family] += 1
                    continue
                telemetry.rendered_candidate_count[rule.family] += 1
                telemetry.selected_family[candidate.family] += 1
                telemetry.selected_rule_family[rule.family] += 1
                if _candidate_has_structured_facts(candidate):
                    telemetry.structured_selected_count[candidate.family] += 1
                if candidate.template_id is not None:
                    telemetry.selected_with_template_id += 1
                if candidate.parent_atoms:
                    telemetry.selected_with_parent_atoms += 1
                if candidate.render_rank is not None:
                    telemetry.selected_render_rank_count[rule.family] += 1
                    telemetry.selected_render_rank_total[rule.family] += candidate.render_rank
                    telemetry.selected_render_rank_bucket[
                        f"{rule.family}:{_render_rank_bucket(candidate.render_rank)}"
                    ] += 1
                telemetry.selected_safety_reasons.update(candidate_safety_reasons(candidate))
                return output_name
        finally:
            telemetry.active_rule_family = previous_rule_family
            telemetry.rule_seconds[rule.family] += time.perf_counter() - started
    return ""


def _first_valid_prepared_candidate(
    mol,
    Chem,
    ctx: MoleculeContext,
    rules: Iterable[RuleSpec],
    *,
    max_length: int = _DEFAULT_MAX_NAME_LENGTH,
    output_max_length: int | None = None,
) -> _PreparedCandidate | None:
    rules = rules if isinstance(rules, tuple) else tuple(rules)
    rule_candidate_cache_state = _rule_candidate_cache_state(mol) if ctx.telemetry is None else ()
    if ctx.telemetry is None:
        prepared_cache_key = (
            id(rules),
            max_length,
            output_max_length,
            rule_candidate_cache_state,
        )
        cached_prepared = ctx.cache.prepared_candidates.get(prepared_cache_key)
        if cached_prepared is not None:
            return cached_prepared[0] if cached_prepared else None
    rules = compile_rules_for_context(ctx, rules)
    for rule in rules:
        if not rule.applies(ctx):
            continue
        candidates = (
            _cached_rule_candidates(ctx, rule, rule_candidate_cache_state)
            if ctx.telemetry is None
            else rule.candidates(ctx)
        )
        for candidate in candidates:
            item = _prepared_candidate_item(
                mol,
                Chem,
                ctx,
                candidate,
                producer_family=rule.family,
                max_length=max_length,
                output_max_length=output_max_length,
                use_cache=True,
            )
            if item is not None:
                return item
    return None


def _fallback_first_valid_or_selected_candidate_name(
    mol,
    Chem,
    ctx: MoleculeContext,
    rules: Iterable[RuleSpec],
    *,
    telemetry: CandidateTelemetry | None = None,
    max_length: int = _DEFAULT_MAX_NAME_LENGTH,
    output_max_length: int | None = None,
) -> str:
    first_item = _first_valid_prepared_candidate(
        mol,
        Chem,
        ctx,
        rules,
        max_length=max_length,
        output_max_length=output_max_length,
    )
    if first_item is None:
        return ""
    if not _fallback_first_valid_candidate_needs_selection(first_item):
        return _record_and_return_prepared_candidate(first_item, telemetry=telemetry)
    selected_item = _fallback_retained_topology_alternative_for_first_valid(
        first_item,
        _prepare_candidates(
            mol,
            Chem,
            ctx,
            rules,
            max_length=max_length,
            output_max_length=output_max_length,
        ),
    )
    return _record_and_return_prepared_candidate(selected_item or first_item, telemetry=telemetry)


def _fallback_retained_topology_alternative_for_first_valid(
    first_item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> _PreparedCandidate | None:
    if not prepared:
        return None
    selected = next(
        (
            item
            for item in prepared
            if item.name == first_item.name
            and item.candidate.family == first_item.candidate.family
            and item.candidate.source_family == first_item.candidate.source_family
            and item.candidate.parent_atoms == first_item.candidate.parent_atoms
        ),
        prepared[0],
    )
    keyed_items = [(_prepared_candidate_selection_key(item, prepared), item) for item in prepared]
    selected = _retained_topology_alternative_for_kekulized_generic_choice(keyed_items, selected)
    if selected is None:
        return None
    return _kekulized_polycyclic_alternative_for_retained_topology_choice(
        keyed_items,
        selected,
    ) or selected


def _fallback_first_valid_candidate_needs_selection(item: _PreparedCandidate) -> bool:
    candidate = item.candidate
    if candidate.family != "kekulized-generic-polycyclic-parent":
        return False
    if candidate.source_family != "kekulized":
        return False
    if candidate.template_name or candidate.source_file:
        return False
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return False
    if candidate.charge is not None or candidate.suffixes or candidate.prefixes:
        return False
    if candidate_safety_reasons(candidate):
        return False
    return True


def _largest_ring_system_cycle_rank(ctx: MoleculeContext) -> int:
    return max((topology.cycle_rank for topology in ctx.ring_topologies), default=0)


def _should_use_bridgehead_polycycle_selector(mol, ctx: MoleculeContext) -> bool:
    if mol.GetRingInfo().NumRings() < 2:
        return False
    if not ctx.facts.has_polycyclic_ring_system:
        return False
    return rdMolDescriptors.CalcNumBridgeheadAtoms(mol) > 0


def _should_use_nonspiro_polycycle_selector(mol, ctx: MoleculeContext) -> bool:
    if mol.GetRingInfo().NumRings() < 3:
        return False
    if ctx.has_component_spiro_topology:
        return False
    if not ctx.facts.has_polycyclic_ring_system:
        return False
    return _largest_ring_system_cycle_rank(ctx) >= 2


def _should_use_spiro_selector(mol, ctx: MoleculeContext) -> bool:
    return rdMolDescriptors.CalcNumSpiroAtoms(mol) > 0


def _can_fast_stage_poly_pre_route_senior(ctx: MoleculeContext) -> bool:
    ring_count = ctx.mol.GetRingInfo().NumRings()
    if ring_count < 3:
        return False
    if ctx.has_component_spiro_topology:
        return False
    if _largest_ring_system_cycle_rank(ctx) != 2:
        return False
    profile = ctx.profile
    return (
        profile.has_functional_skeleton_signal
        or profile.has_imino_skeleton_signal
        or profile.has_neutral_phosphorus
        or profile.has_sulfur
        or profile.has_formal_charge
    )


def _can_fast_stage_multi_monocycle_senior(ctx: MoleculeContext) -> bool:
    ring_count = ctx.mol.GetRingInfo().NumRings()
    if ring_count < 3 or ring_count > 5:
        return False
    if ctx.has_component_spiro_topology:
        return False
    nonhalogen_hetero_count = _context_nonhalogen_hetero_atom_count(ctx)
    if nonhalogen_hetero_count > 6:
        return False
    elements = ctx.features["elements"]
    if not (elements.get("S", 0) or elements.get("P", 0)):
        return False
    if _largest_ring_system_cycle_rank(ctx) > 1:
        return False
    profile = ctx.profile
    return (
        profile.has_functional_skeleton_signal
        or profile.has_imino_skeleton_signal
        or profile.has_neutral_phosphorus
        or profile.has_sulfur
        or profile.has_formal_charge
    )


def _can_fast_stage_low_hetero_poly_senior(ctx: MoleculeContext) -> bool:
    ring_count = ctx.mol.GetRingInfo().NumRings()
    if ring_count < 3 or ring_count > 5:
        return False
    if ctx.has_component_spiro_topology:
        return False
    if _largest_ring_system_cycle_rank(ctx) != 2:
        return False
    if _context_nonhalogen_hetero_atom_count(ctx) > 5:
        return False
    profile = ctx.profile
    return (
        profile.has_functional_skeleton_signal
        or profile.has_imino_skeleton_signal
        or profile.has_neutral_phosphorus
        or profile.has_sulfur
        or profile.has_formal_charge
    )


def _context_nonhalogen_hetero_atom_count(ctx: MoleculeContext) -> int:
    return sum(
        count
        for symbol, count in ctx.features["elements"].items()
        if symbol not in _NONHALOGEN_HETERO_EXCLUDED_SYMBOLS
    )


def _can_fast_stage_monocycle_system(mol) -> bool:
    rings = tuple(frozenset(ring) for ring in mol.GetRingInfo().AtomRings())
    if not rings:
        return False
    if _has_fast_monocycle_heterocarbonyl_defer_shape(mol, rings):
        return False
    for index, ring in enumerate(rings):
        if any(ring & other for other in rings[index + 1 :]):
            return False
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in _FAST_MONOCYCLE_ALLOWED_ATOMIC_NUMBERS:
            return False
        if atom.GetFormalCharge() != 0:
            return False
        if int(atom.GetChiralTag()) != 0:
            return False
        if not atom.IsInRing() and atom.GetSymbol() in {"I", "S"}:
            return False
    for bond in mol.GetBonds():
        if str(bond.GetStereo()) != "STEREONONE":
            return False
        order = bond.GetBondTypeAsDouble()
        if order <= 1.0:
            continue
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        if begin.IsInRing() and end.IsInRing():
            continue
        if order != 2.0 or {begin.GetSymbol(), end.GetSymbol()} != {"C", "O"}:
            return False
    return True


def _has_fast_monocycle_heterocarbonyl_defer_shape(mol, rings: tuple[frozenset[int], ...]) -> bool:
    for ring in rings:
        symbols = [mol.GetAtomWithIdx(idx).GetSymbol() for idx in ring]
        counts = {symbol: symbols.count(symbol) for symbol in set(symbols)}
        if (
            len(ring) == 5
            and counts.get("C") == 2
            and counts.get("N") == 1
            and counts.get("O") == 1
            and counts.get("S") == 1
            and _ring_has_exocyclic_carbonyl(mol, ring)
        ):
            return True
        if {"N", "O", "S"}.issubset(counts) and _ring_has_terminal_carboxamide_or_ester(mol, ring):
            return True
    return False


def _ring_has_exocyclic_carbonyl(mol, ring: frozenset[int]) -> bool:
    for atom_idx in ring:
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetSymbol() != "C":
            continue
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in ring or nbr.GetSymbol() != "O":
                continue
            bond = mol.GetBondBetweenAtoms(atom_idx, nidx)
            if bond is not None and bond.GetBondTypeAsDouble() == 2.0:
                return True
    return False


def _ring_has_terminal_carboxamide_or_ester(mol, ring: frozenset[int]) -> bool:
    for atom_idx in ring:
        for carbon in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
            carbon_idx = carbon.GetIdx()
            if carbon_idx in ring or carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0:
                continue
            bond_to_ring = mol.GetBondBetweenAtoms(atom_idx, carbon_idx)
            if bond_to_ring is None or bond_to_ring.GetBondTypeAsDouble() != 1.0:
                continue
            has_double_oxygen = False
            has_single_hetero = False
            has_other_heavy = False
            for nbr in carbon.GetNeighbors():
                nidx = nbr.GetIdx()
                if nidx == atom_idx:
                    continue
                bond = mol.GetBondBetweenAtoms(carbon_idx, nidx)
                if nbr.GetSymbol() == "O" and bond is not None and bond.GetBondTypeAsDouble() == 2.0:
                    has_double_oxygen = True
                elif nbr.GetSymbol() in {"N", "O"} and bond is not None and bond.GetBondTypeAsDouble() == 1.0:
                    has_single_hetero = True
                elif nbr.GetAtomicNum() > 1:
                    has_other_heavy = True
            if has_double_oxygen and has_single_hetero and not has_other_heavy:
                return True
    return False


def _largest_ring_system_ring_count(mol) -> int:
    rings = tuple(frozenset(ring) for ring in mol.GetRingInfo().AtomRings())
    if not rings:
        return 0
    remaining = set(range(len(rings)))
    largest = 0
    while remaining:
        start = remaining.pop()
        stack = [start]
        atoms = set(rings[start])
        count = 1
        while stack:
            _ = stack.pop()
            linked = [idx for idx in remaining if rings[idx] & atoms]
            for idx in linked:
                remaining.remove(idx)
                stack.append(idx)
                atoms.update(rings[idx])
                count += 1
        if count > largest:
            largest = count
    return largest


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
    rules = rules if isinstance(rules, tuple) else tuple(rules)
    rule_candidate_cache_state = _rule_candidate_cache_state(mol) if ctx.telemetry is None else ()
    prepared_cache_key = None
    if ctx.telemetry is None:
        prepared_cache_key = (
            id(rules),
            max_length,
            output_max_length,
            rule_candidate_cache_state,
        )
        cached_prepared = ctx.cache.prepared_candidates.get(prepared_cache_key)
        if cached_prepared is not None:
            return list(cached_prepared)
    rules = compile_rules_for_context(ctx, rules, telemetry=ctx.telemetry)
    for rule in rules:
        if prepared and rule.priority > _SENIOR_CANDIDATE_EARLY_STOP_PRIORITY:
            break
        if ctx.telemetry is not None:
            ctx.telemetry.rule_evaluations[rule.family] += 1
        if not rule.applies(ctx):
            if ctx.telemetry is not None:
                ctx.telemetry.rule_gate_skips[rule.family] += 1
            continue
        if ctx.telemetry is None:
            for candidate in _cached_rule_candidates(ctx, rule, rule_candidate_cache_state):
                item = _prepared_candidate_item(
                    mol,
                    Chem,
                    ctx,
                    candidate,
                    producer_family=rule.family,
                    max_length=max_length,
                    output_max_length=output_max_length,
                    use_cache=bool(ctx.cache.prepared_candidate_items),
                )
                if item is not None:
                    prepared.append(item)
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
                candidate = _with_molecule_stereo_evidence(mol, Chem, candidate)
                candidate = _with_molecule_graph_fragility_evidence(mol, Chem, candidate)
                candidate = _with_parent_element_counts(mol, candidate)
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
    if prepared_cache_key is not None:
        ctx.cache.prepared_candidates[prepared_cache_key] = tuple(prepared)
    return prepared


def _prepared_candidate_item(
    mol,
    Chem,
    ctx: MoleculeContext,
    candidate: NameCandidate,
    *,
    producer_family: str,
    max_length: int,
    output_max_length: int | None,
    use_cache: bool = False,
) -> _PreparedCandidate | None:
    """Finalize and enrich a candidate once per context/rule family."""

    if not candidate.name:
        return None
    cache_key = None
    if use_cache and ctx.telemetry is None:
        cache_key = (id(candidate), producer_family, max_length, output_max_length)
        if cache_key in ctx.cache.prepared_candidate_items:
            return ctx.cache.prepared_candidate_items[cache_key]
    output_name = _finalize_candidate_name(mol, Chem, candidate, max_length=max_length)
    if not output_name or (output_max_length is not None and len(output_name) > output_max_length):
        if cache_key is not None:
            ctx.cache.prepared_candidate_items[cache_key] = None
        return None
    prepared_candidate = _with_molecule_stereo_evidence(mol, Chem, candidate)
    prepared_candidate = _with_molecule_graph_fragility_evidence(mol, Chem, prepared_candidate)
    prepared_candidate = _with_parent_element_counts(mol, prepared_candidate)
    selection_name = _candidate_selection_surface_name(
        mol,
        Chem,
        prepared_candidate,
        output_name,
        max_length=max_length,
    )
    item = _PreparedCandidate(
        name=selection_name,
        candidate=prepared_candidate,
        producer_family=producer_family,
        stability_penalty=_candidate_selection_stability_penalty(selection_name, prepared_candidate),
        selection_bonus=_candidate_selection_bonus(selection_name, prepared_candidate),
        output_name=output_name,
    )
    if cache_key is not None:
        ctx.cache.prepared_candidate_items[cache_key] = item
    return item


def _cached_rule_candidates(
    ctx: MoleculeContext,
    rule: RuleSpec,
    cache_state: tuple[tuple[bool, str], ...],
) -> tuple[NameCandidate, ...]:
    cache = getattr(ctx.cache, "rule_candidates", None)
    if cache is None:
        return tuple(rule.candidates(ctx))
    cache_key = (id(rule), cache_state)
    cached = cache.get(cache_key)
    if cached is None:
        cached = tuple(rule.candidates(ctx))
        cache[cache_key] = cached
    return cached


def _rule_candidate_cache_state(mol) -> tuple[tuple[bool, str], ...]:
    props = (
        FAST_POLY_BRIDGE_ROUTE_PROP,
        FAST_POLY_RETAINED_DEFER_BYPASS_PROP,
        _FRAGMENT_SUBSTITUENT_FALLBACK_PROP,
        _RETAINED_PARTIAL_HYDRO_FALLBACK_PROP,
        "_nispo_fragment_substituent_depth",
        "_nispo_fragment_substituent_view",
        "_nispo_kekulized_aromatic_view",
    )
    return tuple(
        (mol.HasProp(prop), mol.GetProp(prop) if mol.HasProp(prop) else "")
        for prop in props
    )


def _select_best_prepared_candidate(
    prepared: list[_PreparedCandidate],
    *,
    telemetry: CandidateTelemetry | None = None,
) -> str:
    best_item = _select_best_prepared_candidate_item(prepared)
    return _record_and_return_prepared_candidate(best_item, telemetry=telemetry)


def _select_stable_spiro_prepared_candidate(
    mol,
    ctx: MoleculeContext,
    prepared: list[_PreparedCandidate],
    *,
    telemetry: CandidateTelemetry | None = None,
) -> str:
    best_item = _select_best_prepared_candidate_item(prepared)
    if best_item is None or not _stable_spiro_selector_item(mol, ctx, best_item):
        return ""
    return _record_and_return_prepared_candidate(best_item, telemetry=telemetry)


def _select_route_miss_prepared_candidate(
    prepared: list[_PreparedCandidate],
    *,
    telemetry: CandidateTelemetry | None = None,
) -> str:
    selected = _select_best_prepared_candidate_item(prepared)
    if _should_defer_fast_senior_fragile_external_unsaturated_acyclic(selected):
        return ""
    return _record_and_return_prepared_candidate(selected, telemetry=telemetry)


def _stable_spiro_selector_item(mol, ctx: MoleculeContext, item: _PreparedCandidate) -> bool:
    candidate = item.candidate
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return False
    if candidate_safety_reasons(candidate):
        return False
    family = candidate.family
    if family == "kekulized-generic-polycyclic-parent":
        return mol.GetRingInfo().NumRings() >= 3 and candidate.render_rank == 1
    if family in {
        "hydro-fused-retained-parent",
        "sulfur-oxo-acyclic-skeleton-with-ring-substituents",
        "sulfonimidamide-parent",
        "lambda-sulfanone-spiro-substituent",
        "sulfonamide-parent",
        "neutral-amine-parent",
    }:
        return True
    if family in {"generated-spiro-fused-parent", "generated-fused-parent"}:
        return (
            mol.GetRingInfo().NumRings() >= 3
            and candidate.parent_terminal_hetero_substituent_count > 0
            and candidate.render_rank is not None
            and candidate.render_rank <= 3
        )
    return False


def _select_best_prepared_candidate_item(prepared: list[_PreparedCandidate]) -> _PreparedCandidate | None:
    if not prepared:
        return None
    if len(prepared) == 1:
        return prepared[0]
    best_item: _PreparedCandidate | None = None
    best_key: tuple | None = None
    keyed_items: list[tuple[tuple, _PreparedCandidate]] = []
    for item in prepared:
        candidate_key = _prepared_candidate_selection_key(item, prepared)
        keyed_items.append((candidate_key, item))
        if best_key is None or candidate_key < best_key:
            best_item = item
            best_key = candidate_key
    return _candidate_fact_ranked_item(keyed_items, best_item)


def _prepared_candidate_selection_key(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> tuple:
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
    return (
        stability_penalty,
        candidate.priority,
        _candidate_selection_render_rank_key(candidate),
        len(item.name) - selection_bonus,
        len(item.name),
        _comparable_sort_key(candidate.sort_key),
        item.name,
    )


def _record_and_return_prepared_candidate(
    best_item: _PreparedCandidate | None,
    *,
    telemetry: CandidateTelemetry | None = None,
) -> str:
    best_name = _prepared_output_name(best_item) if best_item is not None else ""
    if best_name and telemetry is not None:
        telemetry.selected_family[best_item.candidate.family] += 1
        telemetry.selected_rule_family[best_item.producer_family] += 1
        if _candidate_has_structured_facts(best_item.candidate):
            telemetry.structured_selected_count[best_item.candidate.family] += 1
        if best_item.candidate.template_id is not None:
            telemetry.selected_with_template_id += 1
        if best_item.candidate.parent_atoms:
            telemetry.selected_with_parent_atoms += 1
        if best_item.candidate.render_rank is not None:
            telemetry.selected_render_rank_count[best_item.producer_family] += 1
            telemetry.selected_render_rank_total[best_item.producer_family] += best_item.candidate.render_rank
            telemetry.selected_render_rank_bucket[
                f"{best_item.producer_family}:{_render_rank_bucket(best_item.candidate.render_rank)}"
            ] += 1
        telemetry.selected_safety_reasons.update(candidate_safety_reasons(best_item.candidate))
    return best_name


def _candidate_fact_ranked_item(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    current: _PreparedCandidate | None,
) -> _PreparedCandidate | None:
    if current is None or len(keyed_items) < 2:
        return current
    selected_item = _hydro_retained_alternative_for_retained_fused_choice(keyed_items, current)
    if selected_item is not None:
        return selected_item
    selected_item = _high_confidence_generated_hydro_pyrido45d_alternative(keyed_items, current)
    if selected_item is not None:
        return selected_item
    selected_item = _high_confidence_single_ring_suffix_generated_fused_template_alternative(
        keyed_items,
        current,
    )
    if selected_item is not None:
        return selected_item
    selected_item = _high_confidence_generated_imidazo45a_alternative(
        keyed_items,
        current,
    )
    if selected_item is not None:
        return selected_item
    selected_item = _high_confidence_pyrazolopyridine_hydro_alternative(
        keyed_items,
        current,
    )
    if selected_item is not None:
        return selected_item
    selected_item = _high_confidence_hetero_generated_fused_alternative(
        keyed_items,
        current,
    )
    if selected_item is not None:
        return selected_item
    selected_item = _high_confidence_acyclic_parent_alternative_for_beta_dicarbonyl_choice(
        keyed_items,
        current,
    )
    if selected_item is not None:
        return selected_item
    selected_item = _acyclic_skeleton_alternative_for_stable_single_ring_choice(
        keyed_items,
        current,
    )
    if selected_item is not None:
        return selected_item
    selected_item = _kekulized_polycyclic_alternative_for_retained_topology_choice(
        keyed_items,
        current,
    )
    if selected_item is not None:
        return selected_item
    selected_item = _unsaturated_acyclic_alternative_for_single_ring_choice(keyed_items, current)
    if selected_item is not None:
        return selected_item
    selected_item = _generated_spiro_skeletal_replacement_alternative_for_component_spiro_choice(
        keyed_items,
        current,
    )
    if selected_item is not None:
        return selected_item
    selected_item = _fused_parent_alternative_for_selected_choice(keyed_items, current)
    if selected_item is not None:
        return selected_item
    selected_item = _systematic_alternative_for_hydrazine_parent_choice(keyed_items, current)
    if selected_item is not None:
        return selected_item
    selected_item = _systematic_alternative_for_dithiocarbonate_choice(keyed_items, current)
    if selected_item is not None:
        return selected_item
    selected_item = _safe_family_alternative_for_selected_choice(keyed_items, current)
    if selected_item is not None:
        return selected_item
    if _deterministic_key_winner_is_high_confidence(current):
        selected_item = (
            _complete_azanium_alternative_for_charged_parent_choice(
                keyed_items,
                current,
            )
            or current
        )
        selected_item = (
            _acyclic_parent_alternative_for_charged_retained_choice(
                keyed_items,
                selected_item,
            )
            or selected_item
        )
        selected_item = (
            _retained_charged_fused_alternative_for_generated_choice(
                keyed_items,
                selected_item,
            )
            or selected_item
        )
        selected_item = (
            _nitrogen_retained_fused_alternative_for_single_ring_choice(
                keyed_items,
                selected_item,
            )
            or selected_item
        )
        selected_item = (
            _kekulized_polycyclic_alternative_for_single_ring_choice(
                keyed_items,
                selected_item,
            )
            or selected_item
        )
        selected_item = (
            _isotope_preserving_alternative_for_selected_choice(
                keyed_items,
                selected_item,
            )
            or selected_item
        )
        selected_item = _stereo_coverage_alternative_for_selected_choice(keyed_items, selected_item) or selected_item
        selected_item = (
            _kekulized_polycyclic_alternative_for_retained_topology_choice(
                keyed_items,
                selected_item,
            )
            or selected_item
        )
        selected_item = (
            _retained_amide_alternative_for_acyclic_choice(
                keyed_items,
                selected_item,
            )
            or selected_item
        )
        selected_item = (
            _high_confidence_generated_fused_template_alternative(
                keyed_items,
                selected_item,
                template_names=_HIGH_CONFIDENCE_FAST_PATH_GENERATED_FUSED_TEMPLATES,
            )
            or selected_item
        )
        selected_item = (
            _high_confidence_single_ring_generated_fused_template_alternative(
                keyed_items,
                selected_item,
            )
            or selected_item
        )
        selected_item = (
            _generated_hydro_fused_alternative_for_retained_fused_choice(
                keyed_items,
                selected_item,
                include_partial_hydro=True,
                reject_isotopes=True,
                blocked_template_names=_HIGH_CONFIDENCE_RETAINED_FUSED_HYDRO_BLOCKED_TEMPLATES,
            )
            or selected_item
        )
        selected_item = (
            _terminal_azole_chalcogen_suffix_alternative_for_high_confidence_generated_choice(
                keyed_items,
                selected_item,
            )
            or selected_item
        )
        return selected_item
    metrics = _candidate_fact_ranker_row_metrics([item for _key, item in keyed_items])
    try:
        current_facts = _candidate_fact_ranker_facts(current, metrics)
        current_score = _candidate_fact_ranker_score(current_facts)
        best_score = current_score
        current_key = next(key for key, item in keyed_items if item is current)
        best_key = current_key
        best_item = current
        best_facts = current_facts
        for candidate_key, item in keyed_items:
            facts = _candidate_fact_ranker_facts(item, metrics)
            score = _candidate_fact_ranker_score(facts)
            if score > best_score or (score == best_score and candidate_key < best_key):
                best_score = score
                best_key = candidate_key
                best_item = item
                best_facts = facts
    except Exception:
        return current
    selected_item = current
    if (
        best_item is not current
        and best_score > current_score + _CANDIDATE_FACT_RANKER_MIN_MARGIN
        and _candidate_fact_ranked_switch_is_safe(
            current,
            best_item,
            current_facts,
            best_facts,
            current_key,
            best_key,
        )
        and _candidate_fact_switch_gate_accepts(
            current_facts,
            best_facts,
            current_score,
            best_score,
        )
    ):
        selected_item = best_item
    selected_item = (
        _isotope_preserving_alternative_for_selected_choice(keyed_items, selected_item)
        or selected_item
    )
    selected_item = (
        _high_confidence_retained_benzimidazole_alternative(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _high_confidence_single_ring_generated_fused_template_alternative(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _high_confidence_generated_imidazo45a_alternative(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _high_confidence_generated_hydro_pyrido45d_alternative(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _single_ring_alternative_for_low_confidence_generated_partial_hydro_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _acyclic_skeleton_alternative_for_generated_partial_hydro_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _acyclic_skeleton_alternative_for_functionalized_generated_fused_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _adamantane_retained_parent_alternative_for_templateless_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _benzimidazole_retained_parent_alternative_for_templateless_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _generated_hetero_fused_alternative_for_tetralin_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _chromane_topology_replacement_alternative_for_nonchromane_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _templateless_alternative_for_static_chromanone_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _acyclic_skeleton_alternative_for_opsin_retained_fused_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _functional_bicyclic_alternative_for_acyclic_skeleton_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _functional_bicyclic_alternative_for_retained_fused_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _acyclic_skeleton_alternative_for_high_rank_hetero_single_ring_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _acyclic_senior_heterocarbon_alternative_for_single_ring_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _kekulized_polycyclic_alternative_for_single_ring_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _kekulized_polycyclic_alternative_for_oxido_retained_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _generated_purine_fused_alternative_for_single_ring_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _kekulized_polycyclic_alternative_for_low_confidence_generated_fused_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _retained_topology_alternative_for_kekulized_generic_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _kekulized_polycyclic_alternative_for_retained_topology_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _acyclic_skeleton_alternative_for_low_confidence_retained_fused_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _hydro_retained_alternative_for_skeletal_replacement_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _hydro_retained_alternative_for_retained_fused_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _generated_fused_alternative_for_hydro_retained_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _high_confidence_generated_fused_template_alternative(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _hydro_retained_benzimidazole_alternative_for_generated_imidazo45a_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _generated_hydro_fused_alternative_for_aromatic_generated_fused_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _generated_hydro_fused_alternative_for_retained_fused_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _generated_hydro_fused_alternative_for_late_single_ring_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _functionalized_generated_fused_alternative_for_generated_fused_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _external_acyl_generated_fused_single_ring_alternative_for_generated_fused_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _exact_high_impact_retained_lactone_alternative_for_generated_topology_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _retained_fused_alternative_for_neutral_amine_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _graph_parent_alternative_for_neutral_amine_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _acyclic_parent_alternative_for_unsaturated_neutral_amine_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _retained_amide_alternative_for_spiro_carboxamide_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _generated_spiro_skeletal_replacement_alternative_for_component_spiro_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _single_ring_alternative_for_zero_parent_retained_functional_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _single_ring_alternative_for_small_ring_phosphorus_acid_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _complete_senior_functional_parent_alternative_for_shortcut_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _acyclic_skeleton_alternative_for_ring_carboxamide_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _high_confidence_generated_fused_template_alternative(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _acyclic_skeleton_alternative_for_low_confidence_retained_fused_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _explicit_charged_parent_alternative_for_neutral_charged_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _complete_azanium_alternative_for_charged_parent_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _acyclic_parent_alternative_for_charged_retained_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _retained_charged_fused_alternative_for_generated_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _charged_retained_fused_alternative_for_replacement_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _charged_large_acyclic_single_ring_alternative_for_acyclic_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _kekulized_polycyclic_alternative_for_chain_dominant_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _chain_dominant_acyclic_alternative_for_compact_kekulized_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _terminal_azole_chalcogen_suffix_alternative_for_non_suffix_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _heterocarbonyl_aromatic_single_ring_alternative_for_single_ring_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _acyclic_skeleton_alternative_for_oxathiazolone_single_ring_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _stable_purin_6_amine_alternative_for_generated_or_adenin_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _stable_purin_thioxo_alternative_for_generated_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _retained_quinoxalin_dicarbonyl_alternative_for_templateless_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _nitrogen_retained_fused_alternative_for_single_ring_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _generated_benzopyrazole_alternative_for_templateless_indazole_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _earlier_bicyclic_orientation_for_late_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _clean_bicyclic_alternative_for_decomposition_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _alternate_bicyclic_orientation_for_rank1_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _azachroman_topology_replacement_alternative_for_generated_fused_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _aromatic_single_ring_alternative_for_alkene_stereo_acyclic_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _functional_acyclic_alternative_for_external_stereo_carbocyclic_single_ring_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _internal_bond_stereo_parent_alternative_for_fragile_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = _steroid_retained_parent_alternative(
        keyed_items,
        selected_item,
    ) or selected_item
    selected_item = (
        _sulfur_oxo_acyclic_alternative_for_generated_fused_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _stable_base_key_parent_alternative_for_post_ranker_choice(
            keyed_items,
            current,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _spiro25_retained_topology_alternative_for_decomposed_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _stereo_coverage_alternative_for_selected_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _kekulized_polycyclic_alternative_for_retained_topology_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _isotope_preserving_alternative_for_selected_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _retained_amide_alternative_for_acyclic_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    selected_item = (
        _retained_fused_alternative_for_stable_single_ring_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )
    return (
        _hydrogenated_sulfur_oxide_single_ring_alternative_for_non_hydrogen_choice(
            keyed_items,
            selected_item,
        )
        or selected_item
    )


_HIGH_CONFIDENCE_DETERMINISTIC_KEY_FAMILIES = frozenset(
    {
        "exact-biochemical-retained-parent",
        "exact-functional-retained-parent",
        "exact-retained-parent",
        "fused-retained-parent",
        "generated-fused-parent",
        "generated-hydro-fused-parent",
        "generated-partial-hydro-fused-parent",
        "whole-molecule-retained-parent",
    }
)


def _single_ring_choice_can_consider_unsaturated_acyclic(item: _PreparedCandidate) -> bool:
    if not _single_ring_choice_has_unsaturated_acyclic_base_context(item):
        return False
    if _single_ring_choice_matches_narrow_unsaturated_acyclic_context(item):
        return True
    return _single_ring_choice_matches_extended_unsaturated_acyclic_context(item)


def _single_ring_choice_has_unsaturated_acyclic_base_context(item: _PreparedCandidate) -> bool:
    candidate = item.candidate
    if candidate.family != "single-ring-parent":
        return False
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return False
    if candidate.parent_internal_unsaturated_bond_count:
        return False
    if candidate.charge is not None or candidate.suffixes:
        return False
    if candidate_safety_reasons(candidate):
        return False
    return True


def _single_ring_choice_matches_narrow_unsaturated_acyclic_context(item: _PreparedCandidate) -> bool:
    candidate = item.candidate
    if candidate.render_rank is not None and candidate.render_rank > 2:
        return False
    if len(item.name) > 90:
        return False
    if candidate.parent_hetero_atom_count:
        return False
    if candidate.parent_external_oxygen_substituent_count:
        return False
    if candidate.parent_external_isotope_hydrogen_count:
        return False
    return True


def _single_ring_choice_matches_extended_unsaturated_acyclic_context(item: _PreparedCandidate) -> bool:
    candidate = item.candidate
    if candidate.render_rank is not None and candidate.render_rank > 3:
        return False
    return len(item.name) <= 125


def _single_ring_choice_uses_extended_unsaturated_acyclic_context(item: _PreparedCandidate) -> bool:
    return _single_ring_choice_can_consider_unsaturated_acyclic(
        item
    ) and not _single_ring_choice_matches_narrow_unsaturated_acyclic_context(item)


def _unsaturated_acyclic_alternative_for_single_ring_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    if not _single_ring_choice_can_consider_unsaturated_acyclic(selected):
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None
    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family not in _UNSATURATED_ACYCLIC_ALTERNATIVE_FAMILIES:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        parent_count = len(candidate.parent_atoms)
        if parent_count < 4 or parent_count > 22:
            continue
        if candidate.parent_internal_unsaturated_bond_count < 1:
            continue
        if candidate.charge is not None or candidate.suffixes:
            continue
        if candidate_safety_reasons(candidate):
            continue
        if len(item.name) > 175 or len(item.name) - len(selected.name) > 140:
            continue
        if item.stability_penalty > selected.stability_penalty + 280:
            continue
        if candidate_key[0] > selected_key[0] + 340:
            continue
        if _guarded_unsaturated_acyclic_alternative_blocks(selected, item):
            continue
        alternatives.append(
            (
                (
                    candidate_key[0],
                    -candidate.parent_internal_unsaturated_bond_count,
                    -len(candidate.parent_atoms),
                    len(item.name) - item.selection_bonus,
                    len(item.name),
                    candidate_key,
                ),
                item,
            )
        )
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _guarded_unsaturated_acyclic_alternative_blocks(
    selected: _PreparedCandidate,
    item: _PreparedCandidate,
) -> bool:
    selected_candidate = selected.candidate
    candidate = item.candidate
    selected_parent_count = len(selected_candidate.parent_atoms)
    parent_count = len(candidate.parent_atoms)
    length_delta = len(item.name) - len(selected.name)
    if (
        selected_candidate.parent_hetero_atom_count
        and parent_count < selected_parent_count
        and selected_candidate.parent_external_nitrogen_substituent_count == 0
    ):
        return True
    selected_is_aromatic_carbocycle = (
        selected_parent_count > 0
        and selected_candidate.parent_hetero_atom_count == 0
        and selected_candidate.parent_aromatic_atom_count == selected_parent_count
    )
    if not selected_is_aromatic_carbocycle:
        return False
    parent_elements = {element for element, _count in candidate.parent_element_counts}
    if "C" not in parent_elements and parent_elements & {"F", "Cl", "Br", "I"}:
        return True
    if (
        selected_candidate.parent_external_oxygen_substituent_count
        and length_delta > 15
        and candidate.parent_internal_unsaturated_bond_count <= 2
        and parent_count <= 12
    ):
        return True
    if (
        length_delta <= 0
        and candidate.parent_internal_unsaturated_bond_count == 1
        and parent_count <= selected_parent_count
        and _candidate_stereo_atom_descriptor_count(candidate) >= 3
        and _candidate_stereo_atom_descriptor_count(selected_candidate) >= 3
    ):
        return True
    if (
        len(selected.name) > 110
        and length_delta > -5
        and candidate.parent_internal_unsaturated_bond_count <= 3
        and parent_count >= 18
    ):
        return True
    if (
        length_delta > 20
        and candidate.parent_internal_unsaturated_bond_count <= 2
        and parent_count >= 14
        and {"N", "O", "S"} <= parent_elements
        and selected_candidate.parent_terminal_hetero_substituent_count == 0
    ):
        return True
    return False


def _candidate_stereo_atom_descriptor_count(candidate: NameCandidate) -> int:
    stereo = candidate.stereo
    if stereo is None:
        return 0
    return len(stereo.atom_descriptors)


def _deterministic_key_winner_is_high_confidence(item: _PreparedCandidate) -> bool:
    """Preserve strong graph-complete key winners from learned post-rank switches."""

    candidate = item.candidate
    if item.stability_penalty:
        return False
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return False
    if candidate_safety_reasons(candidate):
        return False
    if candidate.family not in _HIGH_CONFIDENCE_DETERMINISTIC_KEY_FAMILIES:
        return True
    if candidate.family == "whole-molecule-retained-parent":
        return len(candidate.parent_atoms) >= 6
    if candidate.family == "fused-retained-parent":
        if candidate.source_family == "static-retained":
            return False
        if candidate.render_rank is not None and candidate.render_rank > 4:
            return False
        return len(candidate.parent_atoms) >= 9
    if candidate.family in {
        "generated-fused-parent",
        "generated-hydro-fused-parent",
        "generated-partial-hydro-fused-parent",
    }:
        if candidate.render_rank is None or candidate.render_rank > 3:
            return False
        return len(candidate.parent_atoms) >= 8
    return True


_ADAMANTANE_TEMPLATELESS_PARENT_COMPETITOR_FAMILIES = frozenset(
    {
        "acyclic-skeleton-with-ring-substituents",
        "carbamate-retained-parent",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "kekulized-generic-polycyclic-parent",
        "single-ring-parent",
    }
)
_ADAMANTANE_RETAINED_PARENT_FAMILIES = frozenset(
    {
        "fused-retained-parent",
        "retained-topology-replacement-fused-parent",
    }
)
_ADAMANTANE_RETAINED_SOURCE_FAMILIES = frozenset(
    {
        "opsin-retained-ring-cml",
        "static-retained",
    }
)


def _adamantane_retained_parent_alternative_for_templateless_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.template_name:
        return None
    if selected_candidate.family not in _ADAMANTANE_TEMPLATELESS_PARENT_COMPETITOR_FAMILIES:
        return None
    if selected_candidate.charge is not None and selected_candidate.charge.charged_atoms:
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None
    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family not in _ADAMANTANE_RETAINED_PARENT_FAMILIES:
            continue
        if candidate.template_name != "adamantane":
            continue
        if candidate.source_family not in _ADAMANTANE_RETAINED_SOURCE_FAMILIES:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        if len(candidate.parent_atoms) != 10:
            continue
        if candidate.parent_external_unsaturated_bond_count:
            continue
        if candidate.charge is not None and candidate.charge.charged_atoms:
            continue
        if candidate.suffixes:
            continue
        render_rank = candidate.render_rank
        if render_rank is None or render_rank > 6:
            continue
        if len(item.name) > 280:
            continue
        if len(item.name) - len(selected.name) > 130:
            continue
        if candidate_key[0] > selected_key[0] + 220:
            continue
        alternatives.append((candidate_key, item))
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


_BENZIMIDAZOLE_TEMPLATELESS_PARENT_COMPETITOR_FAMILIES = frozenset(
    {
        "acyclic-skeleton-with-ring-substituents",
        "carbamate-retained-parent",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "generic-bicyclic-parent",
        "single-ring-parent",
        "spiro-functional-chain-substituent",
        "sulfonamide-retained-parent",
    }
)
_HIGH_CONFIDENCE_RETAINED_BENZIMIDAZOLE_COMPETITOR_FAMILIES = frozenset(
    {
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "charged-retained-parent",
        "fused-retained-parent",
        "generated-fused-parent",
        "high-signal-generated-topology-fused-parent",
        "hydro-fused-retained-parent",
        "neutral-amine-parent",
        "partial-hydro-fused-retained-parent",
        "preferred-acyclic-skeleton-with-ring-substituents",
        "retained-amide-parent",
        "single-ring-parent",
        "sulfur-oxo-acyclic-skeleton-with-ring-substituents",
        "urea-retained-parent",
    }
)


def _benzimidazole_retained_parent_alternative_for_templateless_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.template_name:
        return None
    if selected_candidate.family not in _BENZIMIDAZOLE_TEMPLATELESS_PARENT_COMPETITOR_FAMILIES:
        return None
    if selected_candidate.charge is not None and selected_candidate.charge.charged_atoms:
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None
    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "fused-retained-parent":
            continue
        if candidate.template_name != "benzimidazole":
            continue
        if candidate.source_family != "static-retained":
            continue
        if candidate.source_file != "high_impact_fused_parent_templates.json":
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        if len(candidate.parent_atoms) != 9:
            continue
        if candidate.parent_element_counts != (("C", 7), ("N", 2)):
            continue
        if candidate.parent_substituted_hetero_atom_count:
            continue
        if candidate.parent_external_unsaturated_bond_count:
            continue
        if candidate.parent_external_nitrogen_substituent_count:
            continue
        if candidate.charge is not None and candidate.charge.charged_atoms:
            continue
        if candidate.suffixes:
            continue
        render_rank = candidate.render_rank
        if render_rank is None or render_rank > 2:
            continue
        if len(item.name) > 260:
            continue
        if len(item.name) - len(selected.name) > 80:
            continue
        if candidate_key[0] > selected_key[0] + 220:
            continue
        alternatives.append((candidate_key, item))
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _high_confidence_retained_benzimidazole_alternative(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family not in _HIGH_CONFIDENCE_RETAINED_BENZIMIDAZOLE_COMPETITOR_FAMILIES:
        return None
    if selected_candidate.source_file == "naturalProducts.xml":
        return None
    if _candidate_has_isotope_evidence(selected_candidate):
        return None
    if selected_candidate.charge is not None and selected_candidate.charge.charged_atoms:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None
    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if not _is_high_confidence_retained_benzimidazole_candidate(candidate):
            continue
        if (
            selected_candidate.family == "fused-retained-parent"
            and selected_candidate.template_name == "benzimidazole"
            and selected_candidate.render_rank is not None
            and candidate.render_rank is not None
            and candidate.render_rank >= selected_candidate.render_rank
        ):
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(item.name) - len(selected.name) > 40:
            continue
        if item.stability_penalty > selected.stability_penalty + 260:
            continue
        if candidate_key[0] > selected_key[0] + 260:
            continue
        render_rank = candidate.render_rank or 0
        alternatives.append(
            (
                (
                    candidate_key[0],
                    render_rank,
                    len(item.name),
                    candidate_key,
                ),
                item,
            )
        )
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _is_high_confidence_retained_benzimidazole_candidate(candidate: NameCandidate) -> bool:
    if candidate.family != "fused-retained-parent":
        return False
    if candidate.template_name != "benzimidazole":
        return False
    if candidate.source_family != "static-retained":
        return False
    if candidate.source_file != "high_impact_fused_parent_templates.json":
        return False
    if candidate.priority != 18:
        return False
    if candidate.render_rank is None or candidate.render_rank > 6:
        return False
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return False
    if len(candidate.parent_atoms) != 9:
        return False
    if candidate.parent_element_counts != (("C", 7), ("N", 2)):
        return False
    if candidate.parent_substituted_hetero_atom_count:
        return False
    if candidate.parent_external_unsaturated_bond_count:
        return False
    if _candidate_has_isotope_evidence(candidate):
        return False
    if candidate.suffixes:
        return False
    if candidate.charge is not None and candidate.charge.charged_atoms:
        return False
    if candidate_safety_reasons(candidate):
        return False
    return True


_TETRALIN_GENERATED_HETERO_FUSED_PARENT_COUNTS = (("C", 6), ("N", 3))


def _generated_hetero_fused_alternative_for_tetralin_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family != "fused-retained-parent":
        return None
    if selected_candidate.template_name != "tetralin":
        return None
    if selected_candidate.parent_element_counts != (("C", 10),):
        return None
    if not selected_candidate.stereo_locants:
        return None
    if selected_candidate.charge is not None or selected_candidate.suffixes:
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    alternatives: list[tuple[tuple[int, tuple], _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "generated-fused-parent":
            continue
        if candidate.source_family != "generated-fused":
            continue
        if candidate.source_file != "generated_fused_template_index.json":
            continue
        if candidate.parent_element_counts != _TETRALIN_GENERATED_HETERO_FUSED_PARENT_COUNTS:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        if len(candidate.parent_atoms) != 9:
            continue
        if candidate.parent_hetero_atom_count < 3:
            continue
        if candidate.molecule_abs_formal_charge_sum:
            continue
        if candidate.charge is not None or candidate.suffixes:
            continue
        if candidate_safety_reasons(candidate):
            continue
        render_rank = candidate.render_rank
        if render_rank is None or render_rank > 4:
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(item.name) - len(selected.name) > 80:
            continue
        alternatives.append(((render_rank, candidate_key), item))
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _functionalized_generated_fused_alternative_for_generated_fused_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    """Prefer a same-scale generated fused core with graph functionalization evidence."""

    selected_candidate = selected.candidate
    if selected_candidate.family != "generated-fused-parent":
        return None
    if selected_candidate.source_family != "generated-fused":
        return None
    if selected_candidate.source_file != "generated_fused_template_index.json":
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    selected_parent_count = len(selected_candidate.parent_atoms)
    if selected_parent_count < 8 or selected_parent_count > 14:
        return None
    if selected_candidate.parent_aromatic_atom_count != selected_parent_count:
        return None
    if selected_candidate.charge is not None or selected_candidate.suffixes:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    selected_rank = selected_candidate.render_rank
    if selected_rank is None or selected_rank < 1 or selected_rank > 4:
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None

    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "generated-fused-parent":
            continue
        if candidate.source_family != "generated-fused":
            continue
        if candidate.source_file != "generated_fused_template_index.json":
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        candidate_parent_count = len(candidate.parent_atoms)
        if candidate_parent_count < selected_parent_count:
            continue
        if candidate_parent_count > selected_parent_count + 6:
            continue
        if candidate.parent_aromatic_atom_count != selected_parent_count:
            if candidate_parent_count == selected_parent_count:
                continue
            if candidate.parent_aromatic_atom_count != candidate_parent_count:
                continue
        same_scale_external_n = (
            candidate_parent_count == selected_parent_count
            and selected_candidate.parent_hetero_atom_count >= 2
            and candidate.parent_hetero_atom_count >= selected_candidate.parent_hetero_atom_count
            and not selected_candidate.parent_external_oxygen_substituent_count
            and not candidate.parent_external_oxygen_substituent_count
            and candidate.parent_external_nitrogen_substituent_count
            > selected_candidate.parent_external_nitrogen_substituent_count
            and candidate.parent_terminal_hetero_substituent_count
            >= max(1, selected_candidate.parent_terminal_hetero_substituent_count)
        )
        larger_terminal_functional_core = (
            candidate_parent_count > selected_parent_count
            and candidate.parent_hetero_atom_count >= max(2, selected_candidate.parent_hetero_atom_count)
            and candidate.parent_terminal_hetero_substituent_count
            >= max(3, selected_candidate.parent_terminal_hetero_substituent_count + 2)
        )
        if not (same_scale_external_n or larger_terminal_functional_core):
            continue
        if candidate.charge is not None or candidate.suffixes:
            continue
        if candidate_safety_reasons(candidate):
            continue
        render_rank = candidate.render_rank
        rank_window = selected_rank if same_scale_external_n else selected_rank + 4
        if render_rank is None or render_rank < 1 or render_rank > rank_window:
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(item.name) - len(selected.name) > 90:
            continue
        if item.stability_penalty > selected.stability_penalty + 240:
            continue
        if candidate_key[0] > selected_key[0] + 240:
            continue
        alternatives.append(
            (
                (
                    candidate_key[0],
                    render_rank,
                    -candidate.parent_external_nitrogen_substituent_count,
                    -candidate.parent_terminal_hetero_substituent_count,
                    candidate_key,
                ),
                item,
            )
        )
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _external_acyl_generated_fused_single_ring_alternative_for_generated_fused_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family != "generated-fused-parent":
        return None
    if selected_candidate.source_family != "generated-fused":
        return None
    if selected_candidate.source_file != "generated_fused_template_index.json":
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    selected_parent_count = len(selected_candidate.parent_atoms)
    if selected_parent_count < 9 or selected_parent_count > 14:
        return None
    if selected_candidate.parent_aromatic_atom_count != selected_parent_count:
        return None
    if selected_candidate.parent_terminal_heterocarbon_substituent_count < 1:
        return None
    if selected_candidate.parent_terminal_hetero_substituent_count != 1:
        return None
    if selected_candidate.parent_external_nitrogen_substituent_count != 1:
        return None
    if selected_candidate.parent_external_oxygen_substituent_count:
        return None
    if selected_candidate.charge is not None or selected_candidate.suffixes:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    selected_rank = selected_candidate.render_rank
    if selected_rank is None or selected_rank < 1 or selected_rank > 2:
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None

    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "single-ring-parent":
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        if len(candidate.parent_atoms) != 6:
            continue
        if candidate.parent_aromatic_atom_count != 6:
            continue
        if candidate.parent_hetero_atom_count > 1:
            continue
        if candidate.parent_terminal_hetero_substituent_count:
            continue
        if candidate.parent_external_nitrogen_substituent_count:
            continue
        if candidate.parent_external_oxygen_substituent_count:
            continue
        if candidate.charge is not None or candidate.suffixes:
            continue
        if candidate_safety_reasons(candidate):
            continue
        render_rank = candidate.render_rank
        if render_rank is None or render_rank < 1 or render_rank > 2:
            continue
        if len(item.name) > 125:
            continue
        if len(item.name) - len(selected.name) > 40:
            continue
        if item.stability_penalty > selected.stability_penalty + 120:
            continue
        if candidate_key[0] > selected_key[0] + 160:
            continue
        alternatives.append((candidate_key, item))
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _exact_high_impact_retained_lactone_alternative_for_generated_topology_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family not in {
        "generated-fused-parent",
        "high-signal-generated-topology-fused-parent",
        "retained-topology-replacement-fused-parent",
    }:
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    if selected_candidate.charge is not None or candidate_safety_reasons(selected_candidate):
        return None

    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if not _is_exact_high_impact_retained_lactone_parent(candidate):
            continue
        if candidate.parent_atoms != selected_candidate.parent_atoms:
            continue
        if candidate.priority > selected_candidate.priority:
            continue
        if candidate.charge is not None:
            continue
        if item.stability_penalty > selected.stability_penalty + 80:
            continue
        alternatives.append((candidate_key, item))
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _retained_fused_alternative_for_neutral_amine_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family != "neutral-amine-parent":
        return None
    if selected_candidate.molecule_abs_formal_charge_sum:
        return None
    if selected_candidate.charge is not None or selected_candidate.suffixes:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    selected_parent_count = len(selected_candidate.parent_atoms)
    if selected_parent_count < 20:
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None

    ranked_items = sorted(keyed_items, key=lambda keyed: keyed[0])
    rank_by_id = {
        id(item): rank for rank, (_key, item) in enumerate(ranked_items, start=1)
    }
    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "fused-retained-parent":
            continue
        if candidate.molecule_abs_formal_charge_sum:
            continue
        if candidate.charge is not None or candidate.suffixes:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        candidate_parent_count = len(candidate.parent_atoms)
        if candidate_parent_count < 8 or candidate_parent_count > 14:
            continue
        if (
            candidate.source_file == "arylGroups.xml"
            and candidate.parent_external_unsaturated_bond_count
            and candidate.parent_element_counts != (("C", candidate_parent_count),)
        ):
            continue
        if selected_parent_count - candidate_parent_count < 12:
            continue
        if candidate_safety_reasons(candidate):
            continue
        render_rank = candidate.render_rank
        if render_rank is None or render_rank < 1 or render_rank > 9:
            continue
        if candidate.source_file != "arylGroups.xml" and render_rank > 2:
            continue
        if (
            candidate.source_file == "arylGroups.xml"
            and render_rank > 2
            and candidate.parent_element_counts != (("C", candidate_parent_count),)
        ):
            continue
        candidate_rank = rank_by_id.get(id(item))
        if candidate_rank is None or candidate_rank > 16:
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(item.name) - len(selected.name) > 80:
            continue
        if item.stability_penalty > selected.stability_penalty + 240:
            continue
        if candidate_key[0] > selected_key[0] + 380:
            continue
        alternatives.append(
            (
                (
                    render_rank,
                    candidate_rank,
                    candidate_key[0],
                    len(item.name),
                    candidate_key,
                ),
                item,
            )
        )
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


_NEUTRAL_AMINE_GRAPH_PARENT_ALTERNATIVE_FAMILIES = frozenset(
    {
        "functional-generic-bicyclic-parent",
        "generated-fused-parent",
        "generic-bicyclic-parent",
        "kekulized-generic-polycyclic-parent",
    }
)


def _graph_parent_alternative_for_neutral_amine_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family != "neutral-amine-parent":
        return None
    if selected_candidate.molecule_abs_formal_charge_sum:
        return None
    if selected_candidate.charge is not None or selected_candidate.suffixes:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    selected_parent_count = len(selected_candidate.parent_atoms)
    if selected_parent_count < 18:
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None

    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family not in _NEUTRAL_AMINE_GRAPH_PARENT_ALTERNATIVE_FAMILIES:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        candidate_parent_count = len(candidate.parent_atoms)
        if candidate_parent_count < 8 or candidate_parent_count > 10:
            continue
        if selected_parent_count - candidate_parent_count < 9:
            continue
        if (
            candidate.family == "kekulized-generic-polycyclic-parent"
            and candidate.parent_external_unsaturated_bond_count
        ):
            continue
        if candidate.suffixes:
            continue
        if candidate.charge is not None and candidate.charge.formal_charge not in {0, 1}:
            continue
        if candidate_safety_reasons(candidate):
            continue
        render_rank = candidate.render_rank
        if render_rank is None or render_rank < 1 or render_rank > 2:
            continue
        if candidate.family == "generated-fused-parent":
            if candidate.source_family != "generated-fused":
                continue
            if candidate.source_file != "generated_fused_template_index.json":
                continue
            if candidate.priority != 28 or render_rank != 1:
                continue
        elif candidate.family in {"functional-generic-bicyclic-parent", "kekulized-generic-polycyclic-parent"}:
            if candidate.priority != 14:
                continue
        elif candidate.priority != 19:
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(item.name) - len(selected.name) > 120:
            continue
        if item.stability_penalty > selected.stability_penalty + 180:
            continue
        if candidate_key[0] > selected_key[0] + 260:
            continue
        alternatives.append(
            (
                (
                    candidate_key[0],
                    render_rank,
                    len(item.name),
                    candidate_key,
                ),
                item,
            )
        )
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _acyclic_parent_alternative_for_unsaturated_neutral_amine_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family != "neutral-amine-parent":
        return None
    if selected_candidate.molecule_abs_formal_charge_sum:
        return None
    if selected_candidate.charge is not None or selected_candidate.suffixes:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    if len(selected_candidate.parent_atoms) < 10:
        return None
    if selected_candidate.parent_internal_unsaturated_bond_count < 2:
        return None
    if selected_candidate.parent_external_unsaturated_bond_count:
        return None
    selected_element_counts = dict(selected_candidate.parent_element_counts)
    if selected_element_counts.get("N") != 1:
        return None
    if any(element not in {"C", "N"} for element in selected_element_counts):
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None

    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "acyclic-parent":
            continue
        if candidate.parent_atoms != selected_candidate.parent_atoms:
            continue
        if not candidate_has_complete_graph(candidate):
            continue
        if candidate.parent_element_counts != selected_candidate.parent_element_counts:
            continue
        if candidate.parent_internal_unsaturated_bond_count < selected_candidate.parent_internal_unsaturated_bond_count:
            continue
        if candidate.parent_external_unsaturated_bond_count:
            continue
        if candidate.molecule_abs_formal_charge_sum:
            continue
        if candidate.charge is not None or candidate.suffixes:
            continue
        if candidate_safety_reasons(candidate):
            continue
        if len(item.name) > 180:
            continue
        if len(item.name) - len(selected.name) > 80:
            continue
        if item.stability_penalty > selected.stability_penalty + 40:
            continue
        if candidate_key[0] > selected_key[0] + 40:
            continue
        alternatives.append(
            (
                (
                    candidate_key[0],
                    len(item.name) - item.selection_bonus,
                    len(item.name),
                    candidate_key,
                ),
                item,
            )
        )
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


_SPIRO_PYRAZOLO_AZEPINE_CARBOXAMIDE_TEMPLATE = (
    "8-oxo-spiro[6,7-dihydro-4H-pyrazolo[3,4-c]azepine-5,4'-oxane]"
)
_HYDRO_RETAINED_ALIAS_REPLACEMENT_TEMPLATES = frozenset(
    {
        "1,2,3,4-tetrahydroquinoline",
        "1,2,3,4-tetrahydroquinoxaline",
    }
)


def _retained_amide_alternative_for_spiro_carboxamide_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family != "fused-retained-parent":
        return None
    if selected_candidate.template_name != _SPIRO_PYRAZOLO_AZEPINE_CARBOXAMIDE_TEMPLATE:
        return None
    if selected_candidate.source_file != "generated_spiro_scaffold_templates.json":
        return None
    if selected_candidate.render_rank is None or selected_candidate.render_rank > 4:
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    if len(selected_candidate.parent_atoms) != 16:
        return None
    if selected_candidate.charge is not None or selected_candidate.suffixes:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None

    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "retained-amide-parent":
            continue
        if candidate.source_family != "ring-carboxamide":
            continue
        if candidate.priority != 9:
            continue
        if candidate.charge is not None or candidate.suffixes:
            continue
        if candidate_safety_reasons(candidate):
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(item.name) - len(selected.name) > 110:
            continue
        if item.stability_penalty > selected.stability_penalty + 260:
            continue
        if candidate_key[0] > selected_key[0] + 360:
            continue
        alternatives.append(
            (
                (
                    candidate_key[0],
                    len(item.name),
                    candidate_key,
                ),
                item,
            )
        )
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _generated_spiro_skeletal_replacement_alternative_for_component_spiro_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family != "component-spiro-parent":
        return None
    selected_has_complete_parent = bool(
        selected_candidate.parent_atoms and candidate_has_complete_graph(selected_candidate)
    )
    if selected_has_complete_parent:
        selected_render_rank = selected_candidate.render_rank
        if selected_render_rank is None or selected_render_rank < 300:
            return None
    if selected_candidate.charge is not None or selected_candidate.suffixes:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None

    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "skeletal-replacement-fused-parent":
            continue
        if candidate.source_family != "generated-fused":
            continue
        if candidate.source_file != "generated_fused_template_index.json":
            continue
        if not (candidate.template_name or "").startswith("spiro["):
            continue
        if candidate.render_rank is None or candidate.render_rank > 2:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        parent_count = len(candidate.parent_atoms)
        if parent_count < 10 or parent_count > 24:
            continue
        if _is_low_confidence_generated_spiro_skeletal_replacement(candidate):
            continue
        if candidate.suffixes or candidate.charge is not None:
            continue
        if candidate_safety_reasons(candidate):
            continue
        if selected_has_complete_parent:
            if not _generated_spiro_skeletal_replacement_matches_component_spiro_facts(
                selected_candidate,
                candidate,
            ):
                continue
            if len(item.name) > len(selected.name) + 5:
                continue
        if len(item.name) > 220 or len(item.name) - len(selected.name) > 120:
            continue
        if item.stability_penalty > selected.stability_penalty + 190:
            continue
        if candidate_key[0] > selected_key[0] + 220:
            continue
        alternatives.append(
            (
                (
                    candidate_key[0],
                    candidate.render_rank,
                    len(item.name) - item.selection_bonus,
                    len(item.name),
                    candidate_key,
                ),
                item,
            )
        )
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _generated_spiro_skeletal_replacement_matches_component_spiro_facts(
    selected: NameCandidate,
    candidate: NameCandidate,
) -> bool:
    parent_count = len(candidate.parent_atoms)
    if len(candidate.parent_atoms) != len(selected.parent_atoms):
        return False
    element_counts = dict(candidate.parent_element_counts)
    if (
        parent_count >= 17
        and element_counts.get("N", 0) >= 5
        and "s-indacen" in (candidate.template_name or "")
    ):
        return False
    if candidate.parent_element_counts != selected.parent_element_counts:
        return False
    if candidate.parent_aromatic_atom_count != selected.parent_aromatic_atom_count:
        return False
    if candidate.parent_hetero_atom_count != selected.parent_hetero_atom_count:
        return False
    if candidate.parent_internal_unsaturated_bond_count != selected.parent_internal_unsaturated_bond_count:
        return False
    if candidate.parent_external_unsaturated_bond_count != selected.parent_external_unsaturated_bond_count:
        return False
    if candidate.parent_terminal_hetero_substituent_count != selected.parent_terminal_hetero_substituent_count:
        return False
    if candidate.parent_external_nitrogen_substituent_count != selected.parent_external_nitrogen_substituent_count:
        return False
    if candidate.parent_external_oxygen_substituent_count != selected.parent_external_oxygen_substituent_count:
        return False
    if len(candidate.stereo_locants) < len(selected.stereo_locants):
        return False
    return True


def _candidate_has_isotope_evidence(candidate: NameCandidate) -> bool:
    return bool(
        candidate.molecule_isotope_atom_count
        or candidate.parent_isotope_atom_count
        or candidate.candidate_isotope_atom_count
        or candidate.parent_external_isotope_hydrogen_count
    )


def _is_clean_generated_fused_template_candidate(candidate: NameCandidate) -> bool:
    if candidate.family != "generated-fused-parent":
        return False
    if candidate.source_family != "generated-fused":
        return False
    if candidate.source_file != "generated_fused_template_index.json":
        return False
    if candidate.priority != 28:
        return False
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return False
    if candidate.suffixes or candidate.charge is not None:
        return False
    if candidate_safety_reasons(candidate):
        return False
    if _candidate_has_isotope_evidence(candidate):
        return False
    return True


def _fused_parent_alternative_for_selected_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    family = selected.candidate.family
    if family == "single-ring-parent":
        return _generated_fused_alternative_for_single_ring_choice(keyed_items, selected)
    if family == "generated-fused-parent":
        return _generated_fused_alternative_for_generated_fused_choice(keyed_items, selected)
    return None


def _generated_fused_alternative_for_single_ring_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family != "single-ring-parent":
        return None
    if selected_candidate.template_name or selected_candidate.source_family or selected_candidate.source_file:
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    if len(selected_candidate.parent_atoms) not in {5, 6}:
        return None
    if selected_candidate.render_rank not in {1, 2, 3}:
        return None
    if selected_candidate.parent_external_oxygen_substituent_count:
        return None
    if selected_candidate.suffixes or selected_candidate.charge is not None:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    if _candidate_has_isotope_evidence(selected_candidate):
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None

    selected_signal = (
        selected_candidate.parent_terminal_hetero_substituent_count
        + selected_candidate.parent_external_nitrogen_substituent_count
    )
    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if not _is_clean_generated_fused_template_candidate(candidate):
            continue
        render_rank = candidate.render_rank
        if render_rank is None or render_rank > 3:
            continue
        parent_count = len(candidate.parent_atoms)
        if parent_count < 9 or parent_count > 13:
            continue
        if candidate.parent_aromatic_atom_count != parent_count:
            continue
        if candidate.parent_hetero_atom_count < 2:
            continue
        if candidate.parent_hetero_atom_count < selected_candidate.parent_hetero_atom_count:
            continue
        candidate_signal = (
            candidate.parent_terminal_hetero_substituent_count
            + candidate.parent_external_nitrogen_substituent_count
        )
        if selected_candidate.parent_hetero_atom_count == 0:
            if candidate_signal < 2:
                continue
        elif candidate_signal < min(2, selected_signal):
            continue
        if len(item.name) > 160 or len(item.name) - len(selected.name) > 45:
            continue
        if item.stability_penalty > selected.stability_penalty + 180:
            continue
        if candidate_key[0] > selected_key[0] + 260:
            continue
        alternatives.append(
            (
                (
                    candidate_key[0],
                    render_rank,
                    len(item.name),
                    candidate_key,
                ),
                item,
            )
        )
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _generated_fused_alternative_for_generated_fused_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if not _is_clean_generated_fused_template_candidate(selected_candidate):
        return None
    if selected_candidate.render_rank != 1:
        return None
    selected_parent_count = len(selected_candidate.parent_atoms)
    selected_is_internalized_candidate = (
        9 <= selected_parent_count <= 10
        and selected_candidate.parent_aromatic_atom_count == selected_parent_count
    )
    selected_is_spiro_hydro_candidate = (
        (selected_candidate.template_name or "").startswith("spiro[")
        and selected_parent_count == 14
        and len(selected_candidate.ring_system_atoms) == 14
        and selected_candidate.parent_aromatic_atom_count == 6
        and selected_candidate.parent_hetero_atom_count == 0
    )
    if not selected_is_internalized_candidate and not selected_is_spiro_hydro_candidate:
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None

    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if selected_is_internalized_candidate and _is_clean_generated_fused_template_candidate(candidate):
            render_rank = candidate.render_rank
            if render_rank is None:
                continue
            candidate_parent_count = len(candidate.parent_atoms)
            if candidate.parent_aromatic_atom_count != candidate_parent_count:
                continue
            terminal_internalized = (
                render_rank <= 3
                and candidate_parent_count in {selected_parent_count, selected_parent_count - 1}
                and candidate_parent_count >= 8
                and selected_candidate.parent_terminal_hetero_substituent_count >= 1
                and candidate.parent_terminal_hetero_substituent_count
                < selected_candidate.parent_terminal_hetero_substituent_count
                and candidate.parent_hetero_atom_count >= selected_candidate.parent_hetero_atom_count
            )
            hetero_enriched_same_parent = (
                render_rank <= 2
                and candidate_parent_count == selected_parent_count
                and candidate.parent_hetero_atom_count >= selected_candidate.parent_hetero_atom_count + 2
                and candidate.parent_terminal_hetero_substituent_count
                >= selected_candidate.parent_terminal_hetero_substituent_count
                and candidate.parent_external_nitrogen_substituent_count
                <= selected_candidate.parent_external_nitrogen_substituent_count + 1
            )
            if not (terminal_internalized or hetero_enriched_same_parent):
                continue
            if len(item.name) > 170 or len(item.name) - len(selected.name) > 35:
                continue
            if item.stability_penalty > selected.stability_penalty + 160:
                continue
            if candidate_key[0] > selected_key[0] + 260:
                continue
            alternatives.append(
                (
                    (
                        candidate_key[0],
                        render_rank,
                        len(item.name),
                        candidate_key,
                    ),
                    item,
                )
            )
            continue
        if selected_is_spiro_hydro_candidate:
            if candidate.family != "hydro-fused-retained-parent":
                continue
            if candidate.source_family != "opsin-retained-ring-cml":
                continue
            if candidate.source_file != "opsin_retained_mono_fused_ring_smiles.csv":
                continue
            if candidate.priority != 18:
                continue
            if candidate.render_rank is not None:
                continue
            if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
                continue
            if len(candidate.parent_atoms) != 10:
                continue
            if candidate.suffixes or candidate.charge is not None:
                continue
            if candidate_safety_reasons(candidate):
                continue
            if _candidate_has_isotope_evidence(candidate):
                continue
            if len(item.name) > 180 or len(item.name) - len(selected.name) > 8:
                continue
            if item.stability_penalty > selected.stability_penalty + 180:
                continue
            if candidate_key[0] > selected_key[0] + 260:
                continue
            alternatives.append(
                (
                    (
                        candidate_key[0],
                        len(item.name),
                        candidate_key,
                    ),
                    item,
                )
            )
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


_HYDRAZINE_SYSTEMATIC_ALTERNATIVE_FAMILIES = frozenset(
    {
        "acyclic-parent",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "fused-retained-parent",
        "generated-fused-parent",
        "preferred-acyclic-skeleton-with-ring-substituents",
        "single-ring-parent",
    }
)
_DITHIOCARBONATE_SYSTEMATIC_ALTERNATIVE_FAMILIES = frozenset(
    {
        "acyclic-parent",
        "acyclic-skeleton-with-ring-substituents",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "single-ring-parent",
    }
)


def _hydrazine_parent_can_consider_systematic_alternative(item: _PreparedCandidate) -> bool:
    candidate = item.candidate
    if candidate.family != "hydrazine-parent":
        return False
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return False
    if len(candidate.parent_atoms) < 6:
        return False
    if candidate.charge is not None or candidate.suffixes:
        return False
    return not candidate_safety_reasons(candidate)


def _systematic_alternative_for_hydrazine_parent_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    if not _hydrazine_parent_can_consider_systematic_alternative(selected):
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None
    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family not in _HYDRAZINE_SYSTEMATIC_ALTERNATIVE_FAMILIES:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        if not _candidate_has_hydrazine_like_nitrogen_evidence(candidate):
            continue
        if candidate.charge is not None or candidate.suffixes:
            continue
        if candidate_safety_reasons(candidate):
            continue
        if len(item.name) > 180 or len(item.name) - len(selected.name) > 80:
            continue
        if item.stability_penalty > selected.stability_penalty + 160:
            continue
        if candidate_key[0] > selected_key[0] + 180:
            continue
        alternatives.append(
            (
                (
                    candidate_key[0],
                    -len(candidate.parent_atoms),
                    len(item.name) - item.selection_bonus,
                    len(item.name),
                    candidate_key,
                ),
                item,
            )
        )
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _candidate_has_hydrazine_like_nitrogen_evidence(candidate: NameCandidate) -> bool:
    element_counts = dict(candidate.parent_element_counts)
    if element_counts.get("N", 0) >= 2:
        return True
    return candidate.parent_external_nitrogen_substituent_count >= 1


def _systematic_alternative_for_dithiocarbonate_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family != "dithiocarbonate-retained-parent":
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    if selected_candidate.charge is not None or selected_candidate.suffixes:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None

    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family not in _DITHIOCARBONATE_SYSTEMATIC_ALTERNATIVE_FAMILIES:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        if candidate.charge is not None or candidate.suffixes:
            continue
        if candidate_safety_reasons(candidate):
            continue
        if len(item.name) > 160:
            continue
        alternatives.append(
            (
                (
                    len(item.name) - item.selection_bonus,
                    len(item.name),
                    candidate_key,
                ),
                item,
            )
        )
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _safe_family_alternative_for_selected_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family == "component-spiro-parent":
        return _strict_skeletal_replacement_alternative_for_component_spiro_choice(
            keyed_items,
            selected,
        )
    if selected_candidate.family == "retained-topology-replacement-fused-parent":
        return _generic_bicyclic_alternative_for_retained_topology_choice(keyed_items, selected)
    if selected_candidate.family == "generated-fused-parent":
        return _hydro_retained_alternative_for_generated_fused_choice(keyed_items, selected)
    if selected_candidate.family == "beta-dicarbonyl-parent":
        return _acyclic_parent_alternative_for_beta_dicarbonyl_choice(keyed_items, selected)
    return None


def _strict_skeletal_replacement_alternative_for_component_spiro_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if not _is_safe_complete_suffixless_parent_candidate(selected_candidate):
        return None
    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "skeletal-replacement-fused-parent":
            continue
        if not _is_safe_complete_suffixless_parent_candidate(candidate):
            continue
        if len(candidate.parent_atoms) != len(selected_candidate.parent_atoms):
            continue
        if candidate.parent_element_counts != selected_candidate.parent_element_counts:
            continue
        if candidate.parent_aromatic_atom_count != selected_candidate.parent_aromatic_atom_count:
            continue
        if candidate.parent_hetero_atom_count != selected_candidate.parent_hetero_atom_count:
            continue
        if len(item.name) - len(selected.name) < -25:
            continue
        alternatives.append(((_candidate_fact_alternative_key(item, candidate_key)), item))
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _generic_bicyclic_alternative_for_retained_topology_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    if not _is_safe_complete_suffixless_parent_candidate(selected.candidate):
        return None
    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "generic-bicyclic-parent":
            continue
        if not _is_safe_complete_suffixless_parent_candidate(candidate):
            continue
        if len(item.name) > 160:
            continue
        alternatives.append(((_candidate_fact_alternative_key(item, candidate_key)), item))
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _hydro_retained_alternative_for_generated_fused_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if not _is_safe_complete_suffixless_parent_candidate(selected_candidate):
        return None
    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "hydro-fused-retained-parent":
            continue
        if not _is_safe_complete_suffixless_parent_candidate(candidate):
            continue
        if len(candidate.parent_atoms) > len(selected_candidate.parent_atoms):
            continue
        if candidate.parent_element_counts != selected_candidate.parent_element_counts:
            continue
        alternatives.append(((_candidate_fact_alternative_key(item, candidate_key)), item))
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _generated_fused_alternative_for_hydro_retained_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family != "hydro-fused-retained-parent":
        return None
    if not _is_safe_complete_suffixless_parent_candidate(selected_candidate):
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None
    selected_parent_count = len(selected_candidate.parent_atoms)
    if selected_parent_count < 9 or selected_parent_count > 12:
        return None

    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "generated-fused-parent":
            continue
        if candidate.source_family != "generated-fused":
            continue
        if candidate.source_file != "generated_fused_template_index.json":
            continue
        if candidate.render_rank != 1:
            continue
        if not _is_safe_complete_suffixless_parent_candidate(candidate):
            continue
        candidate_parent_count = len(candidate.parent_atoms)
        if candidate_parent_count != selected_parent_count - 1:
            continue
        if candidate.parent_aromatic_atom_count != candidate_parent_count:
            continue
        if candidate.parent_hetero_atom_count < selected_candidate.parent_hetero_atom_count:
            continue
        selected_element_counts = dict(selected_candidate.parent_element_counts)
        adjusted_selected_counts = dict(selected_element_counts)
        adjusted_selected_counts["C"] = adjusted_selected_counts.get("C", 0) - 1
        if adjusted_selected_counts.get("C", 0) <= 0:
            continue
        if dict(candidate.parent_element_counts) != adjusted_selected_counts:
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(item.name) - len(selected.name) > 60:
            continue
        if candidate_key > selected_key:
            continue
        alternatives.append(((_candidate_fact_alternative_key(item, candidate_key)), item))
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _acyclic_parent_alternative_for_beta_dicarbonyl_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    if not _is_safe_complete_suffixless_parent_candidate(selected.candidate):
        return None
    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "acyclic-parent":
            continue
        if not _is_safe_complete_suffixless_parent_candidate(candidate):
            continue
        alternatives.append(((_candidate_fact_alternative_key(item, candidate_key)), item))
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _is_safe_complete_suffixless_parent_candidate(candidate: NameCandidate) -> bool:
    return bool(
        candidate.parent_atoms
        and candidate_has_complete_graph(candidate)
        and candidate.charge is None
        and not candidate.suffixes
        and not candidate_safety_reasons(candidate)
    )


def _candidate_fact_alternative_key(
    item: _PreparedCandidate,
    candidate_key: tuple,
) -> tuple:
    return (
        len(item.name) - item.selection_bonus,
        len(item.name),
        candidate_key,
    )


def _is_low_confidence_generated_spiro_skeletal_replacement(candidate: NameCandidate) -> bool:
    if candidate.family != "skeletal-replacement-fused-parent":
        return False
    if candidate.source_family != "generated-fused":
        return False
    if candidate.source_file != "generated_fused_template_index.json":
        return False
    if len(candidate.parent_atoms) != 11:
        return False
    if candidate.parent_element_counts != (("C", 8), ("N", 2), ("O", 1)):
        return False
    if candidate.parent_external_unsaturated_bond_count:
        return False
    return candidate.parent_internal_unsaturated_bond_count == 0


_ZERO_PARENT_RETAINED_FUNCTIONAL_SINGLE_RING_FAMILIES = frozenset(
    {
        "beta-dicarbonyl-parent",
        "carbamate-retained-parent",
        "organophosphorus-oxide-parent",
        "sulfonamide-retained-parent",
    }
)


def _single_ring_alternative_for_zero_parent_retained_functional_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family not in _ZERO_PARENT_RETAINED_FUNCTIONAL_SINGLE_RING_FAMILIES:
        return None
    if selected_candidate.priority != 9:
        return None
    if selected_candidate.parent_atoms or selected_candidate.locants:
        return None
    if candidate_has_complete_graph(selected_candidate):
        return None
    if selected_candidate.charge is not None or selected_candidate.prefixes or selected_candidate.suffixes:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None

    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "single-ring-parent":
            continue
        if (
            selected_candidate.family == "sulfonamide-retained-parent"
            and candidate.parent_hetero_atom_count <= 0
            and candidate.parent_external_nitrogen_substituent_count <= 0
        ):
            continue
        if candidate.priority not in {30, 61}:
            continue
        if candidate.render_rank is None or candidate.render_rank > 3:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        candidate_parent_atom_count = len(candidate.parent_atoms)
        if candidate_parent_atom_count < 4 or candidate_parent_atom_count > 12:
            continue
        if len(candidate.locants) != candidate_parent_atom_count:
            continue
        if candidate.charge is not None or candidate.prefixes:
            continue
        if len(candidate.suffixes) > 1:
            continue
        if candidate_safety_reasons(candidate):
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(item.name) - len(selected.name) > 80:
            continue
        if item.stability_penalty > selected.stability_penalty + 240:
            continue
        if candidate_key[0] > selected_key[0] + 260:
            continue
        sulfonamide_hetero_parent_rank = (
            0
            if (
                selected_candidate.family == "sulfonamide-retained-parent"
                and candidate.parent_hetero_atom_count > 0
            )
            else 1
        )
        alternatives.append(
            (
                (
                    candidate_key[0],
                    candidate.priority,
                    sulfonamide_hetero_parent_rank,
                    _candidate_selection_render_rank_key(candidate),
                    len(item.name) - item.selection_bonus,
                    len(item.name),
                    candidate_key,
                ),
                item,
            )
        )
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _single_ring_alternative_for_small_ring_phosphorus_acid_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family != "phosphorus-acid-parent":
        return None
    if selected_candidate.priority != 9:
        return None
    if selected_candidate.template_name or selected_candidate.source_family or selected_candidate.source_file:
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    if len(selected_candidate.parent_atoms) > 20:
        return None
    if _candidate_parent_element_count(selected_candidate, "P") != 1:
        return None
    if selected_candidate.charge is not None or selected_candidate.prefixes or selected_candidate.suffixes:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None

    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "single-ring-parent":
            continue
        if candidate.priority != 30:
            continue
        if candidate.render_rank is None or candidate.render_rank > 3:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        if len(candidate.parent_atoms) != 3 or len(candidate.locants) != 3:
            continue
        if candidate.charge is not None or candidate.prefixes or candidate.suffixes:
            continue
        if candidate_safety_reasons(candidate):
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(item.name) - len(selected.name) > 120:
            continue
        if item.stability_penalty > selected.stability_penalty + 160:
            continue
        if candidate_key[0] > selected_key[0] + 260:
            continue
        alternatives.append(
            (
                (
                    candidate_key[0],
                    _candidate_selection_render_rank_key(candidate),
                    len(item.name) - item.selection_bonus,
                    len(item.name),
                    candidate_key,
                ),
                item,
            )
        )
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


_SENIOR_FUNCTIONAL_PARENT_PROMOTION_FAMILIES = frozenset(
    {
        "phosphite-ester-parent",
    }
)


def _complete_senior_functional_parent_alternative_for_shortcut_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family not in _RANKER_SENIOR_FUNCTIONAL_SHORTCUT_FAMILIES:
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    if selected_candidate.suffixes or selected_candidate.charge is not None:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None
    selected_parent_count = len(selected_candidate.parent_atoms)

    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family not in _SENIOR_FUNCTIONAL_PARENT_PROMOTION_FAMILIES:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        if candidate.priority >= selected_candidate.priority:
            continue
        if candidate.suffixes or candidate.prefixes:
            continue
        if candidate.charge is not None and candidate.charge.formal_charge != 0:
            continue
        if candidate_safety_reasons(candidate):
            continue
        candidate_parent_count = len(candidate.parent_atoms)
        if candidate_parent_count < 12 or candidate_parent_count < selected_parent_count + 8:
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(item.name) - len(selected.name) > 90:
            continue
        if item.stability_penalty > selected.stability_penalty + 140:
            continue
        if candidate_key[0] > selected_key[0] + 160:
            continue
        alternatives.append(
            (
                (
                    candidate.priority,
                    candidate_key[0],
                    len(item.name) - item.selection_bonus,
                    len(item.name),
                    candidate_key,
                ),
                item,
            )
        )
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _acyclic_skeleton_alternative_for_ring_carboxamide_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family != "retained-amide-parent":
        return None
    if selected_candidate.source_family != "ring-carboxamide":
        return None
    if selected_candidate.charge is not None or selected_candidate.suffixes:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None

    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "acyclic-skeleton-with-ring-substituents":
            continue
        if candidate.template_name or candidate.source_family or candidate.source_file:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        if candidate.parent_hetero_atom_count or candidate.parent_aromatic_atom_count:
            continue
        if candidate.suffixes or candidate.charge is not None:
            continue
        if candidate_safety_reasons(candidate):
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(item.name) - len(selected.name) > 80:
            continue
        if item.stability_penalty > selected.stability_penalty + 220:
            continue
        if candidate_key[0] > selected_key[0] + 220:
            continue
        alternatives.append(
            (
                (
                    candidate_key[0],
                    len(item.name) - item.selection_bonus,
                    len(item.name),
                    candidate_key,
                ),
                item,
            )
        )
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


_CHARGED_GENERATED_FUSED_COMPETITOR_FAMILIES = frozenset(
    {
        "acyclic-skeleton-with-ring-substituents",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "generic-bicyclic-parent",
        "neutral-amine-parent",
        "preferred-acyclic-skeleton-with-ring-substituents",
        "retained-amide-parent",
        "retained-topology-replacement-fused-parent",
    }
)
_CHARGED_PYRIDINIUM_ACYL_COMPETITOR_FAMILIES = frozenset(
    {
        "acyclic-skeleton-with-ring-substituents",
        "charged-ring-acyclic-skeleton-with-ring-substituents",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "fused-retained-parent",
        "generated-fused-parent",
        "preferred-acyclic-skeleton-with-ring-substituents",
    }
)
_CHARGED_RETAINED_FUSED_COMPETITOR_FAMILIES = frozenset(
    {
        "acyclic-skeleton-with-ring-substituents",
        "charged-ring-acyclic-skeleton-with-ring-substituents",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "functional-acyclic-skeleton-with-ring-substituents",
        "generated-fused-parent",
        "preferred-acyclic-skeleton-with-ring-substituents",
        "skeletal-replacement-fused-parent",
        "single-ring-parent",
    }
)
_COMPLETE_AZANIUM_COMPETITOR_FAMILIES = frozenset(
    {
        "acyclic-parent",
        "acyclic-skeleton-with-ring-substituents",
        "beta-dicarbonyl-parent",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "fused-retained-parent",
        "generated-fused-parent",
        "kekulized-generic-polycyclic-parent",
        "neutral-amine-parent",
        "phosphate-ester-retained-parent",
        "phosphorus-acid-parent",
        "preferred-acyclic-skeleton-with-ring-substituents",
        "single-ring-parent",
    }
)


def _explicit_charged_parent_alternative_for_neutral_charged_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.molecule_abs_formal_charge_sum < 1:
        return None
    if _explicit_candidate_formal_charge(selected_candidate):
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None

    ranked_items = sorted(keyed_items, key=lambda keyed: keyed[0])
    rank_by_id = {id(item): rank for rank, (_key, item) in enumerate(ranked_items, start=1)}
    has_competitive_charged_ring_acyclic = (
        _has_competitive_charged_ring_acyclic_alternative(
            keyed_items,
            selected,
            selected_key,
        )
        if selected_candidate.family == "neutral-amine-parent"
        else False
    )
    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        candidate_charge = _explicit_candidate_formal_charge(candidate)
        if not candidate_charge:
            continue
        if candidate.molecule_abs_formal_charge_sum != selected_candidate.molecule_abs_formal_charge_sum:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        if candidate_safety_reasons(candidate):
            continue
        candidate_rank = rank_by_id.get(id(item))
        if candidate_rank is None:
            continue
        case_priority = _explicit_charged_parent_alternative_priority(
            selected_candidate,
            candidate,
            candidate_charge,
            candidate_rank,
            has_competitive_charged_ring_acyclic,
        )
        if case_priority is None:
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(item.name) - len(selected.name) > 160:
            continue
        if item.stability_penalty > selected.stability_penalty + 240:
            continue
        if candidate_key[0] > selected_key[0] + 280:
            continue
        alternatives.append(
            (
                (
                    case_priority,
                    candidate_rank,
                    candidate_key[0],
                    len(item.name),
                    candidate_key,
                ),
                item,
            )
        )
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _complete_azanium_alternative_for_charged_parent_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family == "azanium-parent":
        return None
    if selected_candidate.family not in _COMPLETE_AZANIUM_COMPETITOR_FAMILIES:
        return None
    if selected_candidate.molecule_abs_formal_charge_sum < 1:
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None

    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "azanium-parent":
            continue
        if candidate.molecule_abs_formal_charge_sum != selected_candidate.molecule_abs_formal_charge_sum:
            continue
        if candidate.parent_abs_formal_charge_sum != candidate.molecule_abs_formal_charge_sum:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        if selected_candidate.parent_atoms and len(candidate.parent_atoms) < len(selected_candidate.parent_atoms):
            continue
        if candidate_safety_reasons(candidate):
            continue
        if candidate.suffixes or candidate.charge is not None:
            continue
        if len(item.name) > 180:
            continue
        if len(item.name) - len(selected.name) > 140:
            continue
        if item.stability_penalty > selected.stability_penalty + 180:
            continue
        if candidate_key[0] > selected_key[0] + 180:
            continue
        if (
            selected_candidate.family == "acyclic-parent"
            and candidate_key[0] > selected_key[0] + 120
            and len(item.name) > len(selected.name) + 20
        ):
            continue
        if (
            selected_candidate.family == "acyclic-parent"
            and not selected_candidate.ring_system_atoms
            and _candidate_parent_element_count(selected_candidate, "N") == 1
            and _candidate_parent_element_count(selected_candidate, "O") >= 4
            and _candidate_parent_element_count(selected_candidate, "P") == 0
            and len(selected_candidate.parent_atoms) >= 12
        ):
            continue
        alternatives.append(
            (
                (
                    candidate_key[0],
                    len(item.name) - item.selection_bonus,
                    len(item.name),
                    candidate_key,
                ),
                item,
            )
        )
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _acyclic_parent_alternative_for_charged_retained_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family != "charged-retained-parent":
        return None
    if selected_candidate.molecule_abs_formal_charge_sum < 1:
        return None
    if selected_candidate.ring_system_atoms or selected_candidate.parent_aromatic_atom_count:
        return None
    if _candidate_parent_element_count(selected_candidate, "S") < 1:
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    if selected_candidate.suffixes or selected_candidate.charge is not None:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None

    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    selected_atoms = frozenset(selected_candidate.parent_atoms)
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "acyclic-parent":
            continue
        if candidate.molecule_abs_formal_charge_sum != selected_candidate.molecule_abs_formal_charge_sum:
            continue
        if not candidate.parent_atoms or frozenset(candidate.parent_atoms) != selected_atoms:
            continue
        if not candidate_has_complete_graph(candidate):
            continue
        if candidate.parent_abs_formal_charge_sum != candidate.molecule_abs_formal_charge_sum:
            continue
        if candidate.ring_system_atoms or candidate.parent_aromatic_atom_count:
            continue
        if _candidate_parent_element_count(candidate, "S") < 1:
            continue
        if candidate.suffixes or candidate.charge is not None:
            continue
        if candidate_safety_reasons(candidate):
            continue
        if len(item.name) > 180 or len(item.name) - len(selected.name) > 40:
            continue
        if item.stability_penalty > selected.stability_penalty + 120:
            continue
        if candidate_key[0] > selected_key[0] + 160:
            continue
        alternatives.append(
            (
                (
                    candidate_key[0],
                    len(item.name) - item.selection_bonus,
                    len(item.name),
                    candidate_key,
                ),
                item,
            )
        )
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _retained_charged_fused_alternative_for_generated_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family != "generated-fused-parent":
        return None
    if selected_candidate.source_family != "generated-fused":
        return None
    selected_charge = _explicit_candidate_formal_charge(selected_candidate)
    if not selected_charge:
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    if len(selected_candidate.parent_atoms) < 8:
        return None
    if selected_candidate.parent_hetero_atom_count > 2:
        return None
    if selected_candidate.suffixes or candidate_safety_reasons(selected_candidate):
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None

    selected_atoms = frozenset(selected_candidate.parent_atoms)
    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "fused-retained-parent":
            continue
        if _explicit_candidate_formal_charge(candidate) != selected_charge:
            continue
        if not candidate.parent_atoms or frozenset(candidate.parent_atoms) != selected_atoms:
            continue
        if not candidate_has_complete_graph(candidate):
            continue
        if candidate.suffixes or candidate_safety_reasons(candidate):
            continue
        render_rank = candidate.render_rank
        if render_rank is None or render_rank > 4:
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(item.name) - len(selected.name) > 80:
            continue
        if item.stability_penalty > selected.stability_penalty + 200:
            continue
        if candidate_key[0] > selected_key[0] + 260:
            continue
        alternatives.append(
            (
                (
                    candidate_key[0],
                    render_rank,
                    len(item.name) - item.selection_bonus,
                    len(item.name),
                    candidate_key,
                ),
                item,
            )
        )
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _charged_retained_fused_alternative_for_replacement_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family not in _CHARGED_RETAINED_FUSED_COMPETITOR_FAMILIES:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None

    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "fused-retained-parent":
            continue
        if not _explicit_candidate_formal_charge(candidate):
            continue
        if _tautomer_sensitive_charged_retained_fused_template(candidate):
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        if len(candidate.parent_atoms) < 8:
            continue
        if candidate_safety_reasons(candidate):
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(item.name) - len(selected.name) > 140:
            continue
        if item.stability_penalty > selected.stability_penalty + 180:
            continue
        if candidate_key[0] > selected_key[0] + 220:
            continue
        alternatives.append(
            (
                (
                    candidate_key[0],
                    candidate.priority,
                    len(item.name),
                    candidate_key,
                ),
                item,
            )
        )
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _tautomer_sensitive_charged_retained_fused_template(candidate: NameCandidate) -> bool:
    template_name = candidate.template_name or ""
    if "ium" in template_name:
        return True
    if "mercaptopurin" in template_name or "hypoxanthin" in template_name:
        return True
    return any(
        fragment in template_name
        for fragment in (
            "imidazo[4,5-b]pyridine",
            "imidazo[4,5-c]pyridine",
            "pyrido[4,5-d]imidazole",
            "pyrido[5,6-d]imidazole",
        )
    )


def _explicit_candidate_formal_charge(candidate: NameCandidate) -> int:
    return candidate.charge.formal_charge if candidate.charge is not None else 0


def _explicit_charged_parent_alternative_priority(
    selected_candidate: NameCandidate,
    candidate: NameCandidate,
    candidate_charge: int,
    candidate_rank: int,
    has_competitive_charged_ring_acyclic: bool,
) -> int | None:
    if (
        selected_candidate.family in _CHARGED_PYRIDINIUM_ACYL_COMPETITOR_FAMILIES
        and _is_charged_pyridinium_acyl_single_ring_candidate(
            candidate,
            candidate_charge,
            candidate_rank,
        )
    ):
        return 0
    if (
        selected_candidate.family in _CHARGED_GENERATED_FUSED_COMPETITOR_FAMILIES
        and candidate.family == "generated-fused-parent"
        and candidate.source_family == "generated-fused"
        and candidate.source_file == "generated_fused_template_index.json"
        and candidate_rank <= 4
        and len(candidate.parent_atoms) >= 8
        and candidate.parent_aromatic_atom_count >= 8
        and candidate.parent_aromatic_atom_count == len(candidate.parent_atoms)
        and candidate.parent_hetero_atom_count >= 2
        and (
            candidate.parent_external_nitrogen_substituent_count
            + candidate.parent_external_oxygen_substituent_count
        )
        >= 1
        and not candidate.suffixes
    ):
        return 1
    if (
        selected_candidate.family == "neutral-amine-parent"
        and not has_competitive_charged_ring_acyclic
        and candidate.family in {"fused-retained-parent", "generated-fused-parent"}
        and candidate_rank <= 3
        and len(candidate.parent_atoms) >= 9
        and not candidate.suffixes
    ):
        return 2
    if (
        selected_candidate.family == "single-ring-parent"
        and selected_candidate.suffixes
        and candidate.family == "fused-retained-parent"
        and candidate_rank <= 8
        and len(candidate.parent_atoms) >= 9
        and not candidate.suffixes
    ):
        return 3
    if (
        selected_candidate.family == "single-ring-parent"
        and candidate.family == "single-ring-parent"
        and candidate_charge < 0
        and candidate_rank <= 4
        and (candidate.suffixes or candidate.parent_hetero_atom_count >= 2)
    ):
        return 4
    return None


def _has_competitive_charged_ring_acyclic_alternative(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
    selected_key: tuple,
) -> bool:
    selected_candidate = selected.candidate
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "charged-ring-acyclic-skeleton-with-ring-substituents":
            continue
        if _explicit_candidate_formal_charge(candidate):
            continue
        if candidate.molecule_abs_formal_charge_sum != selected_candidate.molecule_abs_formal_charge_sum:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        if candidate_safety_reasons(candidate):
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if item.stability_penalty > selected.stability_penalty + 80:
            continue
        if candidate_key[0] > selected_key[0] + 80:
            continue
        return True
    return False


def _is_charged_pyridinium_acyl_single_ring_candidate(
    candidate: NameCandidate,
    candidate_charge: int,
    candidate_rank: int,
) -> bool:
    if candidate.family not in {"single-ring-parent", "positive-single-ring-parent"}:
        return False
    if candidate_charge <= 0:
        return False
    if candidate_rank > 3:
        return False
    if len(candidate.parent_atoms) != 6:
        return False
    if candidate.parent_aromatic_atom_count != 6:
        return False
    if candidate.parent_hetero_atom_count != 1:
        return False
    if candidate.parent_external_nitrogen_substituent_count != 1:
        return False
    if candidate.parent_external_oxygen_substituent_count:
        return False
    if candidate.suffixes:
        return False
    return True


def _charged_large_acyclic_single_ring_alternative_for_acyclic_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family not in {
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "preferred-acyclic-skeleton-with-ring-substituents",
    }:
        return None
    if selected_candidate.molecule_abs_formal_charge_sum < 1:
        return None
    if selected_candidate.parent_abs_formal_charge_sum < 1:
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    selected_parent_count = len(selected_candidate.parent_atoms)
    if selected_parent_count < 20:
        return None
    selected_stereo = selected_candidate.stereo
    if selected_stereo is None or not selected_stereo.bond_descriptors:
        return None
    if any(descriptor != "STEREOCIS" for _atom_idx, descriptor in selected_stereo.bond_descriptors):
        return None
    if selected_candidate.charge is not None or selected_candidate.suffixes:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None

    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "single-ring-parent":
            continue
        if candidate.molecule_abs_formal_charge_sum != selected_candidate.molecule_abs_formal_charge_sum:
            continue
        if candidate.charge is not None or candidate.suffixes:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        candidate_parent_count = len(candidate.parent_atoms)
        if candidate_parent_count < 5 or candidate_parent_count > 6:
            continue
        if candidate.priority != 61:
            continue
        if candidate.parent_aromatic_atom_count != candidate_parent_count:
            continue
        if candidate.parent_terminal_hetero_substituent_count < 3:
            continue
        if candidate.parent_external_nitrogen_substituent_count:
            continue
        if candidate.parent_external_oxygen_substituent_count:
            continue
        if selected_parent_count - candidate_parent_count < 14:
            continue
        render_rank = candidate.render_rank
        if render_rank != 1:
            continue
        if candidate_safety_reasons(candidate):
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(item.name) - len(selected.name) > 80:
            continue
        if item.stability_penalty > selected.stability_penalty + 120:
            continue
        if candidate_key[0] > selected_key[0] + 180:
            continue
        alternatives.append((candidate_key, item))
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


_CHROMANE_TOPOLOGY_REPLACEMENT_COMPETITOR_FAMILIES = frozenset(
    {
        "acyclic-skeleton-with-ring-substituents",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "functional-generic-bicyclic-parent",
        "fused-retained-parent",
        "generic-bicyclic-parent",
        "generic-spiro-parent",
        "single-ring-parent",
        "spiro-functional-chain-substituent",
        "sulfonamide-retained-parent",
    }
)
_CHROMANE_PARENT_ELEMENT_COUNTS = frozenset(
    {
        (("C", 9), ("O", 1)),
        (("C", 8), ("N", 1), ("O", 1)),
        (("C", 7), ("N", 2), ("O", 1)),
    }
)


def _chromane_topology_replacement_alternative_for_nonchromane_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.template_name == "chromane":
        return None
    if selected_candidate.family not in _CHROMANE_TOPOLOGY_REPLACEMENT_COMPETITOR_FAMILIES:
        return None
    if selected_candidate.charge is not None or selected_candidate.suffixes:
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None
    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "retained-topology-replacement-fused-parent":
            continue
        if candidate.template_name != "chromane":
            continue
        if candidate.source_family != "opsin-retained-ring-cml":
            continue
        if candidate.source_file != "opsin_retained_mono_fused_ring_smiles.csv":
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        if len(candidate.parent_atoms) != 10:
            continue
        if candidate.parent_element_counts not in _CHROMANE_PARENT_ELEMENT_COUNTS:
            continue
        if candidate.parent_substituted_hetero_atom_count:
            continue
        if candidate.parent_external_unsaturated_bond_count:
            continue
        if candidate.parent_external_nitrogen_substituent_count:
            continue
        if candidate.parent_external_oxygen_substituent_count:
            continue
        if candidate.charge is not None or candidate.suffixes:
            continue
        render_rank = candidate.render_rank
        if render_rank is None or render_rank > 2:
            continue
        if len(item.name) > 260:
            continue
        if len(item.name) - len(selected.name) > 80:
            continue
        if candidate_key[0] > selected_key[0] + 120:
            continue
        alternatives.append((candidate_key, item))
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


_AZACHROMAN_TOPOLOGY_REPLACEMENT_TEMPLATE_NAMES = frozenset({"chroman", "chromen"})
_AZACHROMAN_PARENT_ELEMENT_COUNTS = frozenset(
    {
        (("C", 8), ("N", 1), ("O", 1)),
        (("C", 7), ("N", 2), ("O", 1)),
    }
)


def _azachroman_topology_replacement_alternative_for_generated_fused_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family != "generated-fused-parent":
        return None
    if selected_candidate.source_family != "generated-fused":
        return None
    if selected_candidate.source_file != "generated_fused_template_index.json":
        return None
    if not candidate_has_complete_graph(selected_candidate) or not selected_candidate.parent_atoms:
        return None
    selected_parent_count = len(selected_candidate.parent_atoms)
    if selected_parent_count < 9 or selected_parent_count > 14:
        return None
    selected_render_rank = selected_candidate.render_rank
    if selected_render_rank is None or selected_render_rank < 1 or selected_render_rank > 3:
        return None
    if selected_candidate.priority != 28:
        return None
    if selected_candidate.charge is not None or selected_candidate.suffixes:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    if selected_candidate.parent_external_unsaturated_bond_count:
        return None
    if selected_candidate.parent_terminal_chalcogen_double_bond_count:
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None

    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "retained-topology-replacement-fused-parent":
            continue
        if candidate.template_name not in _AZACHROMAN_TOPOLOGY_REPLACEMENT_TEMPLATE_NAMES:
            continue
        if candidate.source_family != "opsin-retained-ring-cml":
            continue
        if candidate.source_file != "opsin_retained_mono_fused_ring_smiles.csv":
            continue
        if candidate.priority != 28:
            continue
        if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
            continue
        if len(candidate.parent_atoms) != 10:
            continue
        if candidate.parent_element_counts not in _AZACHROMAN_PARENT_ELEMENT_COUNTS:
            continue
        render_rank = candidate.render_rank
        if render_rank is None or render_rank < 1 or render_rank > 2:
            continue
        if candidate.charge is not None or candidate.suffixes:
            continue
        if candidate_safety_reasons(candidate):
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(item.name) - len(selected.name) > 80:
            continue
        if item.stability_penalty > selected.stability_penalty + 120:
            continue
        if candidate_key[0] > selected_key[0] + 180:
            continue
        alternatives.append(
            (
                (
                    render_rank,
                    candidate_key[0],
                    len(item.name),
                    candidate_key,
                ),
                item,
            )
        )
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


_STATIC_CHROMANONE_TEMPLATELESS_ALTERNATIVE_FAMILIES = frozenset(
    {
        "acyclic-skeleton-with-ring-substituents",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "functional-acyclic-skeleton-with-ring-substituents",
        "preferred-acyclic-skeleton-with-ring-substituents",
        "single-ring-parent",
    }
)


def _templateless_alternative_for_static_chromanone_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family != "fused-retained-parent":
        return None
    if selected_candidate.template_name != "chroman-4-one":
        return None
    if selected_candidate.source_family != "static-retained":
        return None
    if selected_candidate.source_file:
        return None
    if selected_candidate.render_rank not in {None, 1}:
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    if len(selected_candidate.parent_atoms) != 11:
        return None
    if selected_candidate.charge is not None or selected_candidate.suffixes:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None
    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family not in _STATIC_CHROMANONE_TEMPLATELESS_ALTERNATIVE_FAMILIES:
            continue
        if candidate.template_name or candidate.source_family or candidate.source_file:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        if candidate.charge is not None or candidate.suffixes:
            continue
        if candidate_safety_reasons(candidate):
            continue
        if candidate.parent_abs_formal_charge_sum:
            continue
        if candidate.family == "single-ring-parent":
            if len(candidate.parent_atoms) != 6 or candidate.parent_aromatic_atom_count != 6:
                continue
        elif len(candidate.parent_atoms) < 7:
            continue
        if len(item.name) > 240:
            continue
        if len(item.name) - len(selected.name) > 120:
            continue
        if candidate_key[0] > selected_key[0] + 220:
            continue
        alternatives.append((candidate_key, item))
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _acyclic_skeleton_alternative_for_opsin_retained_fused_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family != "fused-retained-parent":
        return None
    if selected_candidate.source_family != "opsin-retained-ring-cml":
        return None
    if selected_candidate.source_file != "opsin_retained_mono_fused_ring_smiles.csv":
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    if selected_candidate.charge is not None or selected_candidate.suffixes:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    if (
        selected_candidate.parent_element_counts == (("C", 10), ("O", 1))
        and not selected_candidate.parent_external_unsaturated_bond_count
    ):
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None
    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "acyclic-skeleton-with-ring-substituents":
            continue
        if candidate.template_name or candidate.source_family or candidate.source_file:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        if len(candidate.parent_atoms) < 5:
            continue
        if candidate.charge is not None or candidate.suffixes:
            continue
        if candidate_safety_reasons(candidate):
            continue
        if candidate.parent_abs_formal_charge_sum > 1:
            continue
        if len(item.name) > 220:
            continue
        if len(item.name) - len(selected.name) > 140:
            continue
        if candidate_key[0] > selected_key[0] + 220:
            continue
        alternatives.append((candidate_key, item))
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _acyclic_skeleton_alternative_for_high_rank_hetero_single_ring_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family != "single-ring-parent":
        return None
    if selected_candidate.priority != 61:
        return None
    selected_render_rank = selected_candidate.render_rank
    if selected_render_rank is None or selected_render_rank < 2 or selected_render_rank > 3:
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    if selected_candidate.parent_hetero_atom_count <= 0:
        return None
    if selected_candidate.parent_external_nitrogen_substituent_count:
        return None
    if selected_candidate.prefixes or selected_candidate.suffixes:
        return None
    if selected_candidate.charge is not None and selected_candidate.charge.charged_atoms:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None
    if _has_clean_compact_topology_alternative_for_high_rank_single_ring_choice(
        keyed_items,
        selected,
        selected_key,
    ):
        return None
    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family not in _RANKER_ACYCLIC_SKELETON_WITH_RING_SUBSTITUENT_FAMILIES:
            continue
        if candidate.template_name or candidate.source_family or candidate.source_file:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        candidate_parent_atom_count = len(candidate.parent_atoms)
        if candidate_parent_atom_count < 5 or candidate_parent_atom_count > 15:
            continue
        if candidate.parent_hetero_atom_count:
            continue
        if candidate.parent_external_unsaturated_bond_count and candidate.priority >= 60:
            continue
        if candidate.parent_external_unsaturated_bond_count >= 2 and selected_render_rank != 3:
            continue
        if (
            selected_render_rank == 3
            and selected_candidate.parent_external_unsaturated_bond_count
            and candidate.parent_external_unsaturated_bond_count >= 2
        ):
            continue
        if (
            selected_render_rank == 2
            and candidate.priority < 60
            and not (
                (
                    selected_candidate.parent_hetero_atom_count >= 2
                    and selected_candidate.parent_aromatic_atom_count == 0
                )
                or selected_candidate.parent_terminal_hetero_substituent_count >= 2
            )
        ):
            continue
        if (
            selected_render_rank == 2
            and selected_candidate.parent_aromatic_atom_count >= 5
            and candidate_parent_atom_count >= 15
        ):
            continue
        if candidate.prefixes or candidate.suffixes:
            continue
        if candidate.charge is not None and candidate.charge.charged_atoms:
            continue
        if candidate_safety_reasons(candidate):
            continue
        if len(item.name) > 220:
            continue
        if len(item.name) - len(selected.name) > 100:
            continue
        if candidate_key[0] > selected_key[0] + 260:
            continue
        alternatives.append((candidate_key, item))
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _acyclic_skeleton_alternative_for_stable_single_ring_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family != "single-ring-parent":
        return None
    if selected_candidate.template_name or selected_candidate.source_family or selected_candidate.source_file:
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    if len(selected_candidate.parent_atoms) != 6:
        return None
    selected_render_rank = selected_candidate.render_rank
    if selected_render_rank is None or selected_render_rank < 1 or selected_render_rank > 3:
        return None
    if selected_candidate.suffixes:
        return None
    if selected_candidate.molecule_abs_formal_charge_sum or selected_candidate.parent_abs_formal_charge_sum:
        return None
    if selected_candidate.charge is not None and selected_candidate.charge.charged_atoms:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    if _candidate_has_isotope_evidence(selected_candidate):
        return None
    selected_functionalized_aryl = (
        selected_render_rank == 2
        and selected_candidate.parent_aromatic_atom_count == 6
        and (
            selected_candidate.parent_terminal_hetero_substituent_count
            + selected_candidate.parent_external_nitrogen_substituent_count
            + selected_candidate.parent_external_oxygen_substituent_count
        )
        >= 1
    )
    if selected_functionalized_aryl:
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None

    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "acyclic-skeleton-with-ring-substituents":
            continue
        if candidate.template_name or candidate.source_family or candidate.source_file:
            continue
        if candidate.priority != 60 or candidate.render_rank is not None:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        candidate_parent_count = len(candidate.parent_atoms)
        candidate_sulfur_count = _candidate_parent_element_count(candidate, "S")
        candidate_halogen_count = (
            _candidate_parent_element_count(candidate, "F")
            + _candidate_parent_element_count(candidate, "Cl")
            + _candidate_parent_element_count(candidate, "Br")
            + _candidate_parent_element_count(candidate, "I")
        )
        candidate_stereo_atom_count = _candidate_stereo_atom_descriptor_count(candidate)
        if candidate_stereo_atom_count >= 6:
            continue
        if (
            candidate_stereo_atom_count >= 3
            and candidate_parent_count >= 25
            and selected_candidate.parent_hetero_atom_count
        ):
            continue
        if (
            selected_candidate.parent_external_oxygen_substituent_count
            and candidate.parent_external_unsaturated_bond_count >= 2
        ):
            continue
        if (
            selected_candidate.parent_external_nitrogen_substituent_count
            and candidate_sulfur_count
            and _candidate_parent_element_count(candidate, "O") >= 5
            and _candidate_parent_element_count(candidate, "N") >= 2
            and candidate.parent_external_unsaturated_bond_count >= 3
        ):
            continue
        if candidate_sulfur_count and candidate_halogen_count == 1:
            continue
        if (
            candidate_sulfur_count
            and selected_candidate.parent_aromatic_atom_count == 6
            and selected_candidate.parent_hetero_atom_count
            and candidate_parent_count >= 4
        ):
            continue
        if (
            selected_candidate.parent_aromatic_atom_count == 6
            and candidate_parent_count >= 18
            and _candidate_parent_element_count(candidate, "N") >= 3
            and _candidate_parent_element_count(candidate, "O") <= 4
        ):
            continue
        if selected_candidate.parent_aromatic_atom_count == 6 and candidate_parent_count > 18:
            continue
        if candidate.parent_hetero_atom_count:
            continue
        if candidate.molecule_abs_formal_charge_sum or candidate.parent_abs_formal_charge_sum:
            continue
        if candidate.suffixes:
            continue
        if candidate.charge is not None and candidate.charge.charged_atoms:
            continue
        if candidate_safety_reasons(candidate):
            continue
        if _candidate_has_isotope_evidence(candidate):
            continue
        if len(item.name) > 130:
            continue
        if len(item.name) - len(selected.name) > 120:
            continue
        if item.stability_penalty > selected.stability_penalty + 220:
            continue
        if candidate_key[0] > selected_key[0] + 260:
            continue
        alternatives.append(
            (
                (
                    candidate_key[0],
                    len(item.name) - item.selection_bonus,
                    len(item.name),
                    candidate_key,
                ),
                item,
            )
        )
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _retained_fused_alternative_for_stable_single_ring_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family != "single-ring-parent":
        return None
    if selected_candidate.template_name or selected_candidate.source_family or selected_candidate.source_file:
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    if len(selected_candidate.parent_atoms) not in {5, 6}:
        return None
    if selected_candidate.priority != 61:
        return None
    if selected_candidate.charge is not None:
        return None
    if selected_candidate.molecule_abs_formal_charge_sum:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    selected_render_rank = selected_candidate.render_rank
    if selected_render_rank is not None and selected_render_rank > 3:
        return None

    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "fused-retained-parent":
            continue
        if not candidate.template_name:
            continue
        if candidate.charge is not None:
            continue
        if candidate.molecule_abs_formal_charge_sum:
            continue
        if candidate.candidate_isotope_atom_count != candidate.molecule_isotope_atom_count:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        parent_elements = dict(candidate.parent_element_counts)
        if frozenset(parent_elements.items()) not in {
            frozenset({"C": 7, "N": 2}.items()),
            frozenset({"C": 6, "N": 3}.items()),
        }:
            continue
        parent_delta = len(candidate.parent_atoms) - len(selected_candidate.parent_atoms)
        if parent_delta < 3:
            continue
        if len(candidate.locants) > len(selected_candidate.locants):
            continue
        if candidate.parent_abs_formal_charge_sum or candidate.template_abs_formal_charge_sum:
            continue
        if candidate_safety_reasons(candidate):
            continue
        render_rank = candidate.render_rank
        if render_rank is not None and selected_render_rank is not None and render_rank > selected_render_rank + 2:
            continue
        if render_rank is not None and render_rank > 4:
            continue
        if len(item.name) > 140:
            continue
        if len(item.name) - len(selected.name) > 15:
            continue
        if item.stability_penalty > selected.stability_penalty + 220:
            continue
        alternatives.append(((len(item.name) - item.selection_bonus, render_rank or 999, candidate_key), item))
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _has_clean_compact_topology_alternative_for_high_rank_single_ring_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
    selected_key: tuple,
) -> bool:
    selected_parent_count = len(selected.candidate.parent_atoms)
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family not in {"generic-bicyclic-parent", "kekulized-generic-polycyclic-parent"}:
            continue
        if candidate.family == "generic-bicyclic-parent" and candidate.source_family != "bicyclic":
            continue
        if candidate.family == "kekulized-generic-polycyclic-parent" and candidate.source_family != "kekulized":
            continue
        if candidate.template_name or candidate.source_file:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        parent_count = len(candidate.parent_atoms)
        if parent_count < 8 or parent_count > 10:
            continue
        if parent_count < selected_parent_count + 3:
            continue
        if candidate.parent_hetero_atom_count:
            continue
        if len(candidate.locants) < 3:
            continue
        if candidate.render_rank != 1:
            continue
        if candidate.prefixes or candidate.suffixes:
            continue
        if candidate.charge is not None and candidate.charge.charged_atoms:
            continue
        if candidate_safety_reasons(candidate):
            continue
        if candidate_key[0] > selected_key[0] + 220:
            continue
        return True
    return False


def _acyclic_senior_heterocarbon_alternative_for_single_ring_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family != "single-ring-parent":
        return None
    if selected_candidate.source_family or selected_candidate.source_file or selected_candidate.template_name:
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    if selected_candidate.suffixes or selected_candidate.charge is not None:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    if selected_candidate.parent_terminal_heterocarbon_substituent_count <= 0:
        return None
    if (
        selected_candidate.parent_terminal_carbonyl_substituent_count <= 0
        and selected_candidate.parent_external_unsaturated_bond_count <= 0
    ):
        return None
    if (
        selected_candidate.parent_aromatic_atom_count
        and selected_candidate.parent_aromatic_atom_count != len(selected_candidate.parent_atoms)
    ):
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None

    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family not in _RANKER_ACYCLIC_SKELETON_WITH_RING_SUBSTITUENT_FAMILIES:
            continue
        if candidate.template_name or candidate.source_family or candidate.source_file:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        if not (2 <= len(candidate.parent_atoms) <= 6):
            continue
        if candidate.parent_hetero_atom_count:
            continue
        if candidate.parent_aromatic_atom_count:
            continue
        if candidate.suffixes or candidate.charge is not None:
            continue
        if candidate_safety_reasons(candidate):
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(item.name) - len(selected.name) > 110:
            continue
        if item.stability_penalty > selected.stability_penalty + 90:
            continue
        if candidate_key[0] > selected_key[0] + 220:
            continue
        alternatives.append(
            (
                (
                    candidate.parent_external_unsaturated_bond_count,
                    candidate.priority,
                    len(item.name) - item.selection_bonus,
                    len(item.name),
                    candidate_key,
                ),
                item,
            )
        )
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _functional_bicyclic_alternative_for_acyclic_skeleton_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family != "acyclic-skeleton-with-ring-substituents":
        return None
    if selected_candidate.priority != 60:
        return None
    if selected_candidate.template_name or selected_candidate.source_family or selected_candidate.source_file:
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    selected_parent_atom_count = len(selected_candidate.parent_atoms)
    if selected_parent_atom_count > 22:
        return None
    if selected_candidate.parent_hetero_atom_count:
        return None
    if selected_candidate.prefixes or selected_candidate.suffixes:
        return None
    if selected_candidate.charge is not None and selected_candidate.charge.charged_atoms:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None
    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "functional-generic-bicyclic-parent":
            continue
        if candidate.priority != 14:
            continue
        if candidate.source_family != "bicyclic":
            continue
        if candidate.template_name or candidate.source_file:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        candidate_parent_atom_count = len(candidate.parent_atoms)
        if candidate_parent_atom_count < 6 or candidate_parent_atom_count > 9:
            continue
        if selected_parent_atom_count - candidate_parent_atom_count > 15:
            continue
        if candidate_parent_atom_count != 7 and len(candidate.locants) < 3:
            continue
        if candidate.parent_hetero_atom_count:
            continue
        if len(candidate.locants) < 2:
            continue
        render_rank = candidate.render_rank
        if render_rank is None or render_rank > 2:
            continue
        if candidate.prefixes or candidate.suffixes:
            continue
        if candidate.charge is not None and candidate.charge.charged_atoms:
            continue
        if candidate_safety_reasons(candidate):
            continue
        if len(item.name) > 260:
            continue
        if len(item.name) - len(selected.name) > 120:
            continue
        if candidate_key[0] > selected_key[0] + 220:
            continue
        alternatives.append((candidate_key, item))
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _functional_bicyclic_alternative_for_retained_fused_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family != "fused-retained-parent":
        return None
    if selected_candidate.priority != 18:
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    selected_parent_atom_count = len(selected_candidate.parent_atoms)
    if selected_parent_atom_count < 6 or selected_parent_atom_count > 15:
        return None
    if selected_candidate.parent_hetero_atom_count:
        return None
    if selected_candidate.parent_aromatic_atom_count:
        return None
    if selected_candidate.locants:
        return None
    if selected_candidate.prefixes or selected_candidate.suffixes:
        return None
    if selected_candidate.charge is not None and selected_candidate.charge.charged_atoms:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "functional-generic-bicyclic-parent":
            continue
        if candidate.priority != 14:
            continue
        if candidate.source_family != "bicyclic":
            continue
        if candidate.template_name or candidate.source_file:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        candidate_parent_atom_count = len(candidate.parent_atoms)
        if candidate_parent_atom_count < 6 or candidate_parent_atom_count > 11:
            continue
        if selected_parent_atom_count - candidate_parent_atom_count > 6:
            continue
        if candidate_parent_atom_count - selected_parent_atom_count > 3:
            continue
        if candidate.parent_hetero_atom_count:
            continue
        if len(candidate.locants) < 2:
            continue
        render_rank = candidate.render_rank
        if render_rank is None or render_rank > 2:
            continue
        if candidate.prefixes or candidate.suffixes:
            continue
        if candidate.charge is not None and candidate.charge.charged_atoms:
            continue
        if candidate_safety_reasons(candidate):
            continue
        if len(item.name) > 280:
            continue
        if len(item.name) - len(selected.name) > 160:
            continue
        if candidate_key[0] > 220:
            continue
        alternatives.append((candidate_key, item))
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


_KEKULIZED_SINGLE_RING_PARENT_PROFILE_COUNTS = frozenset(
    {
        (("C", 6), ("N", 1)),
        (("C", 6), ("N", 3)),
        (("C", 14),),
    }
)

_LOW_CONFIDENCE_GENERATED_FUSED_TO_KEKULIZED_FAMILIES = frozenset(
    {
        "generated-hydro-fused-parent",
        "generated-partial-hydro-fused-parent",
        "skeletal-replacement-fused-parent",
    }
)
_GENERATED_HYDRO_PYRIDO_IMIDAZOLE_PARENT_COUNTS = (("C", 6), ("N", 3))
_GENERATED_HYDRO_PYRIDO_IMIDAZOLE_SELECTED_PARENT_COUNTS = frozenset(
    {
        (("C", 6), ("N", 3)),
        (("C", 7), ("N", 1), ("S", 1)),
        (("C", 7), ("N", 2)),
        (("C", 8), ("N", 1)),
    }
)
_HIGH_CONFIDENCE_GENERATED_HYDRO_PYRIDO45D_SELECTED_FAMILIES = frozenset(
    {
        "fused-retained-parent",
        "generated-fused-parent",
        "hydro-fused-retained-parent",
        "retained-topology-replacement-fused-parent",
        "single-ring-parent",
    }
)


def _kekulized_polycyclic_alternative_for_single_ring_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family != "single-ring-parent":
        return None
    if selected_candidate.template_name or selected_candidate.source_family or selected_candidate.source_file:
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    if selected_candidate.charge is not None or selected_candidate.suffixes:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None
    selected_parent_count = len(selected_candidate.parent_atoms)
    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "kekulized-generic-polycyclic-parent":
            continue
        if candidate.source_family != "kekulized":
            continue
        if candidate.template_name or candidate.source_file:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        if len(candidate.parent_atoms) < selected_parent_count + 1:
            continue
        if len(candidate.parent_atoms) > 15:
            continue
        if not _kekulized_single_ring_parent_profile_is_supported(candidate, item.name):
            continue
        if len(candidate.parent_atoms) <= 7 and len(candidate.locants) < 3:
            continue
        if candidate.charge is not None or candidate.suffixes:
            continue
        if candidate_safety_reasons(candidate):
            continue
        if candidate.render_rank != 1:
            continue
        if len(item.name) > 220:
            continue
        if len(item.name) - len(selected.name) > 140:
            continue
        if candidate_key[0] > selected_key[0] + 340:
            continue
        alternatives.append((candidate_key, item))
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _kekulized_single_ring_parent_profile_is_supported(
    candidate: NameCandidate,
    name: str,
) -> bool:
    if candidate.parent_element_counts in _KEKULIZED_SINGLE_RING_PARENT_PROFILE_COUNTS:
        return True
    if candidate.parent_element_counts == (("C", 7), ("N", 2)):
        return len(name) <= 110
    return False


def _kekulized_polycyclic_alternative_for_oxido_retained_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family != "oxido-fused-retained-parent":
        return None
    if selected_candidate.suffixes:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    selected_parent_count = len(selected_candidate.parent_atoms)
    if selected_parent_count < 8 or selected_parent_count > 11:
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None
    ranked_items = sorted(keyed_items, key=lambda keyed: keyed[0])
    rank_by_id = {id(item): rank for rank, (_key, item) in enumerate(ranked_items, start=1)}

    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "kekulized-generic-polycyclic-parent":
            continue
        if candidate.source_family != "kekulized":
            continue
        if candidate.template_name or candidate.source_file:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        candidate_parent_count = len(candidate.parent_atoms)
        if candidate_parent_count < selected_parent_count - 1:
            continue
        if candidate_parent_count > selected_parent_count + 1:
            continue
        if candidate.render_rank != 1:
            continue
        if candidate.suffixes:
            continue
        if _explicit_candidate_formal_charge(candidate) <= 0:
            continue
        if candidate_safety_reasons(candidate):
            continue
        candidate_rank = rank_by_id.get(id(item))
        if candidate_rank is None or candidate_rank > 6:
            continue
        if len(item.name) > 220:
            continue
        if len(item.name) - len(selected.name) > 130:
            continue
        alternatives.append(
            (
                (
                    candidate_rank,
                    candidate_key[0],
                    len(item.name),
                    candidate_key,
                ),
                item,
            )
        )
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _generated_purine_fused_alternative_for_single_ring_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family != "single-ring-parent":
        return None
    if selected_candidate.template_name or selected_candidate.source_family or selected_candidate.source_file:
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    selected_parent_atom_count = len(selected_candidate.parent_atoms)
    if selected_parent_atom_count not in {5, 6}:
        return None
    if selected_parent_atom_count == 5 and selected_candidate.parent_terminal_hetero_substituent_count < 2:
        return None
    if selected_candidate.charge is not None:
        return None
    if selected_candidate.molecule_abs_formal_charge_sum:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    if selected_candidate.render_rank is not None and selected_candidate.render_rank > 2:
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None
    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "generated-fused-parent":
            continue
        if candidate.template_name not in _GENERATED_PURINE_PARENT_TEMPLATES:
            continue
        if candidate.source_family != "generated-fused":
            continue
        if candidate.source_file != "generated_fused_template_index.json":
            continue
        if candidate.render_rank is None or candidate.render_rank > 2:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        if len(candidate.parent_atoms) != 9:
            continue
        if candidate.parent_aromatic_atom_count != 9 or candidate.parent_hetero_atom_count != 4:
            continue
        if candidate.parent_external_nitrogen_substituent_count > 1:
            continue
        if candidate.parent_external_oxygen_substituent_count:
            continue
        if candidate.molecule_abs_formal_charge_sum:
            continue
        if candidate.charge is not None or candidate.suffixes or candidate.prefixes:
            continue
        if candidate_safety_reasons(candidate):
            continue
        if len(item.name) > 280:
            continue
        if len(item.name) - len(selected.name) > 160:
            continue
        if candidate_key[0] > selected_key[0] + 220:
            continue
        alternatives.append((candidate_key, item))
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


_BENZOPYRAZOLE_TEMPLATELESS_COMPETITOR_FAMILIES = frozenset(
    {
        "acyclic-skeleton-with-ring-substituents",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "generic-spiro-parent",
        "phosphorus-acid-parent",
        "single-ring-parent",
        "sulfonamide-retained-parent",
        "sulfonohydrazide-retained-parent",
    }
)
_BENZOPYRAZOLE_TEMPLATELESS_SOURCE_FAMILIES = frozenset({"", "spiro"})
_BENZOPYRAZOLE_GRAPHLESS_COMPETITOR_FAMILIES = frozenset(
    {
        "phosphorus-acid-parent",
        "sulfonamide-retained-parent",
        "sulfonohydrazide-retained-parent",
    }
)


def _generated_benzopyrazole_alternative_for_templateless_indazole_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.template_name or selected_candidate.source_file:
        return None
    if selected_candidate.source_family not in _BENZOPYRAZOLE_TEMPLATELESS_SOURCE_FAMILIES:
        return None
    if selected_candidate.family not in _BENZOPYRAZOLE_TEMPLATELESS_COMPETITOR_FAMILIES:
        return None
    if selected_candidate.charge is not None:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    if selected_candidate.parent_atoms:
        if not candidate_has_complete_graph(selected_candidate):
            return None
        if len(selected_candidate.parent_atoms) > 11:
            return None
        if selected_candidate.parent_aromatic_atom_count:
            return None
    elif selected_candidate.family not in _BENZOPYRAZOLE_GRAPHLESS_COMPETITOR_FAMILIES:
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None
    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "generated-fused-parent":
            continue
        if candidate.source_family != "generated-fused":
            continue
        if candidate.source_file != "generated_fused_template_index.json":
            continue
        if candidate.template_name != "benzo[1,2-c]pyrazole":
            continue
        if candidate.parent_element_counts != (("C", 7), ("N", 2)):
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        if len(candidate.parent_atoms) != 9:
            continue
        if candidate.parent_aromatic_atom_count != 9 or candidate.parent_hetero_atom_count != 2:
            continue
        if candidate.charge is not None:
            continue
        if candidate.parent_abs_formal_charge_sum or candidate.molecule_abs_formal_charge_sum:
            continue
        if candidate.prefixes or candidate.suffixes:
            continue
        if candidate_safety_reasons(candidate):
            continue
        render_rank = candidate.render_rank
        if render_rank is None or render_rank > 2:
            continue
        if len(item.name) > 320:
            continue
        if len(item.name) - len(selected.name) > 180:
            continue
        if candidate_key[0] > selected_key[0] + 260:
            continue
        alternatives.append((candidate_key, item))
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _kekulized_polycyclic_alternative_for_low_confidence_generated_fused_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family not in _LOW_CONFIDENCE_GENERATED_FUSED_TO_KEKULIZED_FAMILIES:
        return None
    if selected_candidate.source_family != "generated-fused":
        return None
    if selected_candidate.source_file != "generated_fused_template_index.json":
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    selected_parent_atom_count = len(selected_candidate.parent_atoms)
    if selected_parent_atom_count < 9 or selected_parent_atom_count > 14:
        return None
    if selected_candidate.charge is not None or selected_candidate.suffixes or selected_candidate.prefixes:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    if selected_candidate.render_rank is not None and selected_candidate.render_rank > 2:
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None
    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "kekulized-generic-polycyclic-parent":
            continue
        if candidate.source_family != "kekulized":
            continue
        if candidate.template_name or candidate.source_file:
            continue
        if candidate.render_rank != 1:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        if len(candidate.parent_atoms) != selected_parent_atom_count:
            continue
        if candidate.charge is not None or candidate.suffixes or candidate.prefixes:
            continue
        if candidate_safety_reasons(candidate):
            continue
        if len(item.name) > 280:
            continue
        if len(item.name) - len(selected.name) > 140:
            continue
        if candidate_key[0] > selected_key[0] + 220:
            continue
        alternatives.append((candidate_key, item))
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


_RETAINED_TOPOLOGY_REPLACEMENT_SOURCE_FAMILIES = frozenset(
    {
        "opsin-retained-ring-cml",
        "static-retained",
        "static-exact-retained",
    }
)
_KEKULIZED_POLYCYCLIC_RETAINED_TOPOLOGY_TEMPLATES = frozenset(
    {
        "benzthiepane",
        "indan",
        "tetralin",
    }
)


def _kekulized_polycyclic_alternative_for_retained_topology_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family != "retained-topology-replacement-fused-parent":
        return None
    if selected_candidate.template_name not in _KEKULIZED_POLYCYCLIC_RETAINED_TOPOLOGY_TEMPLATES:
        return None
    if selected_candidate.source_family not in _RETAINED_TOPOLOGY_REPLACEMENT_SOURCE_FAMILIES:
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    selected_parent_count = len(selected_candidate.parent_atoms)
    if selected_parent_count < 9 or selected_parent_count > 11:
        return None
    if selected_candidate.render_rank is not None and selected_candidate.render_rank > 2:
        return None
    if selected_candidate.suffixes or selected_candidate.prefixes:
        return None
    if selected_candidate.charge is not None:
        return None
    if _candidate_has_isotope_evidence(selected_candidate):
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None

    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "kekulized-generic-polycyclic-parent":
            continue
        if candidate.source_family != "kekulized":
            continue
        if candidate.template_name or candidate.source_file:
            continue
        if candidate.render_rank != 1:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        if len(candidate.parent_atoms) != selected_parent_count:
            continue
        if candidate.parent_element_counts != selected_candidate.parent_element_counts:
            continue
        if candidate.suffixes or candidate.prefixes:
            continue
        if candidate.charge is not None:
            continue
        if _candidate_has_isotope_evidence(candidate):
            continue
        if candidate_safety_reasons(candidate):
            continue
        if len(candidate.locants) < 1:
            continue
        if len(item.name) > 180:
            continue
        if len(item.name) - len(selected.name) > 100:
            continue
        if item.stability_penalty > selected.stability_penalty + 160:
            continue
        alternatives.append(
            (
                (
                    candidate_key[0],
                    len(item.name),
                    candidate_key,
                ),
                item,
            )
        )
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _retained_topology_alternative_for_kekulized_generic_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family != "kekulized-generic-polycyclic-parent":
        return None
    if selected_candidate.source_family != "kekulized":
        return None
    if selected_candidate.template_name or selected_candidate.source_file:
        return None
    if selected_candidate.render_rank is None or selected_candidate.render_rank > 4:
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    if selected_candidate.charge is not None or selected_candidate.suffixes or selected_candidate.prefixes:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    selected_parent_count = len(selected_candidate.parent_atoms)
    if selected_parent_count < 8:
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None

    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "retained-topology-replacement-fused-parent":
            continue
        if candidate.source_family not in _RETAINED_TOPOLOGY_REPLACEMENT_SOURCE_FAMILIES:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        if candidate.parent_atoms != selected_candidate.parent_atoms:
            continue
        if candidate.parent_element_counts != selected_candidate.parent_element_counts:
            continue
        if candidate.parent_hetero_atom_count != selected_candidate.parent_hetero_atom_count:
            continue
        if candidate.parent_aromatic_atom_count > selected_candidate.parent_aromatic_atom_count:
            continue
        if candidate.charge is not None or candidate.suffixes or candidate.prefixes:
            continue
        if candidate_safety_reasons(candidate):
            continue
        if candidate.render_rank is None or candidate.render_rank > 2:
            continue
        if len(item.name) >= len(selected.name):
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if candidate_key[0] > selected_key[0] + 220:
            continue
        alternatives.append((candidate_key, item))
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: (len(keyed[1].name), keyed[0]))[1]


def _acyclic_skeleton_alternative_for_low_confidence_retained_fused_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    allowed_families = _LOW_CONFIDENCE_FUSED_TEMPLATE_TO_TEMPLATELESS_FAMILIES.get(
        selected_candidate.template_name
    )
    if not allowed_families or "acyclic-skeleton-with-ring-substituents" not in allowed_families:
        return None
    if selected_candidate.template_name != "imidazo[1,2-b]pyridazine":
        return None
    if selected_candidate.family == "fused-retained-parent":
        if selected_candidate.priority != 18:
            return None
    elif selected_candidate.family == "generated-fused-parent":
        if selected_candidate.source_family != "generated-fused":
            return None
        if selected_candidate.source_file != "generated_fused_template_index.json":
            return None
        if selected_candidate.priority != 28:
            return None
        if not _has_same_name_retained_fused_template_sibling(
            keyed_items,
            selected,
            selected_candidate.template_name,
        ):
            return None
    else:
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    if len(selected_candidate.parent_atoms) != 9:
        return None
    if selected_candidate.render_rank is not None and selected_candidate.render_rank > 5:
        return None
    if selected_candidate.suffixes or selected_candidate.charge is not None:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None

    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "acyclic-skeleton-with-ring-substituents":
            continue
        if candidate.template_name or candidate.source_family or candidate.source_file:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        if candidate.parent_hetero_atom_count or candidate.parent_aromatic_atom_count:
            continue
        if candidate.suffixes or candidate.charge is not None:
            continue
        if candidate_safety_reasons(candidate):
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(item.name) - len(selected.name) > 80:
            continue
        if item.stability_penalty > selected.stability_penalty + 260:
            continue
        if candidate_key[0] > selected_key[0] + 260:
            continue
        alternatives.append(
            (
                (
                    candidate_key[0],
                    len(item.name) - item.selection_bonus,
                    len(item.name),
                    candidate_key,
                ),
                item,
            )
        )
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _has_same_name_retained_fused_template_sibling(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
    template_name: str | None,
) -> bool:
    if not template_name:
        return False
    for _candidate_key, item in keyed_items:
        if item is selected or item.name != selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "fused-retained-parent":
            continue
        if candidate.template_name != template_name:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        if candidate.suffixes or candidate.charge is not None:
            continue
        if candidate_safety_reasons(candidate):
            continue
        return True
    return False


def _hydro_retained_alternative_for_skeletal_replacement_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family != "skeletal-replacement-fused-parent":
        return None
    if selected_candidate.source_family != "arylGroups:ring:ring":
        return None
    if selected_candidate.source_file != "arylGroups.xml":
        return None
    if selected_candidate.priority != 28:
        return None
    if selected_candidate.render_rank is None or selected_candidate.render_rank > 2:
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    selected_parent_count = len(selected_candidate.parent_atoms)
    if selected_parent_count != 13:
        return None
    if selected_candidate.parent_aromatic_atom_count or selected_candidate.parent_hetero_atom_count:
        return None
    if selected_candidate.parent_abs_formal_charge_sum or selected_candidate.molecule_abs_formal_charge_sum:
        return None
    if selected_candidate.prefixes or selected_candidate.suffixes:
        return None
    if selected_candidate.charge is not None and selected_candidate.charge.charged_atoms:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None

    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "hydro-fused-retained-parent":
            continue
        if candidate.source_family != "opsin-retained-ring-cml":
            continue
        if candidate.source_file != "opsin_retained_mono_fused_ring_smiles.csv":
            continue
        if candidate.priority != 18:
            continue
        if candidate.render_rank is not None:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        if len(candidate.parent_atoms) != selected_parent_count:
            continue
        if candidate.parent_aromatic_atom_count or candidate.parent_hetero_atom_count:
            continue
        if candidate.parent_abs_formal_charge_sum or candidate.molecule_abs_formal_charge_sum:
            continue
        if candidate.prefixes or candidate.suffixes:
            continue
        if candidate.charge is not None and candidate.charge.charged_atoms:
            continue
        if candidate_safety_reasons(candidate):
            continue
        if len(item.name) > 280:
            continue
        if len(item.name) - len(selected.name) > 80:
            continue
        if candidate_key[0] > 120:
            continue
        alternatives.append((candidate_key, item))
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


_GENERATED_FUSED_TEMPLATE_ALTERNATIVE_SPECS = {
    "imidazo[4,5-a]benzene": {
        "parent_atom_count": 9,
        "parent_hetero_atom_count": 2,
        "allowed_selected_families": frozenset(
            {
                "fused-retained-parent",
                "generated-fused-parent",
                "generated-hydro-fused-parent",
                "high-signal-generated-topology-fused-parent",
                "retained-topology-replacement-fused-parent",
            }
        ),
    },
    "imidazo[1,2-b]pyridazine": {
        "parent_atom_count": 9,
        "parent_hetero_atom_count": 3,
        "allowed_selected_families": frozenset(
            {
                "acyclic-skeleton-with-ring-substituents",
                "fused-retained-parent",
                "generated-fused-parent",
                "retained-topology-replacement-fused-parent",
                "single-ring-parent",
            }
        ),
    },
    "imidazo[4,5-c]pyridine": {
        "parent_atom_count": 9,
        "parent_hetero_atom_count": 3,
        "allowed_selected_families": frozenset(
            {
                "acyclic-skeleton-with-ring-substituents",
                "chain-dominant-acyclic-skeleton-with-ring-substituents",
                "single-ring-parent",
            }
        ),
    },
    "imidazo[4,5-d]pyrimidine": {
        "parent_atom_count": 9,
        "parent_hetero_atom_count": 4,
        "allowed_selected_families": frozenset(
            {
                "acyclic-skeleton-with-ring-substituents",
                "fused-retained-parent",
            }
        ),
    },
    "imidazo[4,5-b]pyridine": {
        "parent_atom_count": 9,
        "parent_hetero_atom_count": 3,
        "allowed_selected_families": frozenset(
            {
                "component-spiro-parent",
                "single-ring-parent",
            }
        ),
    },
    "benzo[1,2-c]pyridazine": {
        "parent_atom_count": 10,
        "parent_hetero_atom_count": 2,
        "allowed_selected_families": frozenset(
            {
                "fused-retained-parent",
                "single-ring-parent",
            }
        ),
    },
    "pyrido[3,4-d]pyrazole": {
        "parent_atom_count": 9,
        "parent_hetero_atom_count": 3,
        "allowed_selected_families": frozenset(
            {
                "acyclic-skeleton-with-ring-substituents",
                "single-ring-parent",
            }
        ),
    },
    "pyrimido[4,5-b]pyrrole": {
        "parent_atom_count": 9,
        "parent_hetero_atom_count": 3,
        "allowed_selected_families": frozenset(
            {
                "single-ring-parent",
                "sulfur-oxo-acyclic-skeleton-with-ring-substituents",
            }
        ),
    },
    "pyrimido[5,4-d]pyrazole": {
        "parent_atom_count": 9,
        "parent_hetero_atom_count": 4,
        "allowed_selected_families": frozenset(
            {
                "single-ring-parent",
                "sulfur-oxo-acyclic-skeleton-with-ring-substituents",
            }
        ),
    },
    "benzo[1,2-c]pyrazole": {
        "parent_atom_count": 9,
        "parent_hetero_atom_count": 2,
        "allowed_selected_families": frozenset(
            {
                "single-ring-parent",
                "sulfur-oxo-acyclic-skeleton-with-ring-substituents",
            }
        ),
    },
}
_HIGH_CONFIDENCE_FAST_PATH_GENERATED_FUSED_TEMPLATES = frozenset(
    {
        "benzo[1,2-c]pyrazole",
        "benzo[1,2-c]pyridazine",
        "imidazo[4,5-b]pyridine",
        "imidazo[4,5-c]pyridine",
        "pyrido[3,4-d]pyrazole",
        "pyrimido[4,5-b]pyrrole",
        "pyrimido[5,4-d]pyrazole",
    }
)
_GENERATED_FUSED_TEMPLATE_ALTERNATIVE_SELECTED_FAMILIES = frozenset(
    family
    for spec in _GENERATED_FUSED_TEMPLATE_ALTERNATIVE_SPECS.values()
    for family in spec["allowed_selected_families"]
)
_HIGH_CONFIDENCE_FAST_PATH_GENERATED_FUSED_SELECTED_FAMILIES = frozenset(
    family
    for template_name in _HIGH_CONFIDENCE_FAST_PATH_GENERATED_FUSED_TEMPLATES
    for family in _GENERATED_FUSED_TEMPLATE_ALTERNATIVE_SPECS[template_name][
        "allowed_selected_families"
    ]
)
_HIGH_CONFIDENCE_SINGLE_RING_GENERATED_FUSED_TEMPLATE_SPECS = {
    "benzo[1,2-c]isoxazole": {
        "parent_atom_count": 9,
        "parent_hetero_atom_count": 2,
    },
    "imidazo[4,5-a]benzene": {
        "parent_atom_count": 9,
        "parent_hetero_atom_count": 2,
    },
    "imidazo[4,5-c]quinoline": {
        "parent_atom_count": 13,
        "parent_hetero_atom_count": 3,
    },
    "imidazo[4,5-d]pyrimidine": {
        "parent_atom_count": 9,
        "parent_hetero_atom_count": 4,
    },
    "isoquino[4,3-d]pyrazole": {
        "parent_atom_count": 13,
        "parent_hetero_atom_count": 3,
    },
    "pyrido[1,2-b]pyrazole": {
        "parent_atom_count": 9,
        "parent_hetero_atom_count": 2,
    },
    "pyrido[3,4-b]pyrrole": {
        "parent_atom_count": 9,
        "parent_hetero_atom_count": 2,
    },
    "pyrido[4,3-b]pyrrole": {
        "parent_atom_count": 9,
        "parent_hetero_atom_count": 2,
    },
    "pyrido[4,3-f]indazole": {
        "parent_atom_count": 13,
        "parent_hetero_atom_count": 3,
    },
    "pyrimido[4,5-b]pyridine": {
        "parent_atom_count": 10,
        "parent_hetero_atom_count": 3,
    },
    "pyrimido[4,5-e]indazole": {
        "parent_atom_count": 13,
        "parent_hetero_atom_count": 4,
    },
}
_HIGH_CONFIDENCE_SINGLE_RING_SUFFIX_GENERATED_FUSED_TEMPLATE_SPECS = {
    "pyrido[3,4-b]pyrrole": {
        "parent_atom_count": 9,
        "parent_hetero_atom_count": 2,
    },
}
_HIGH_CONFIDENCE_GENERATED_IMIDAZO45A_SELECTED_FAMILIES = frozenset(
    {
        "acyclic-skeleton-with-ring-substituents",
        "charged-retained-parent",
        "fused-retained-parent",
        "generated-fused-parent",
        "generated-partial-hydro-fused-parent",
        "glutarimide-retained-parent",
        "neutral-amine-parent",
        "preferred-acyclic-skeleton-with-ring-substituents",
        "retained-acid-ester-parent",
        "single-ring-parent",
        "sulfonamide-retained-parent",
        "sulfur-oxo-acyclic-skeleton-with-ring-substituents",
        "urea-retained-parent",
    }
)
_HIGH_CONFIDENCE_PYRAZOLOPYRIDINE_HYDRO_SELECTED_FAMILIES = frozenset(
    {
        "acyclic-skeleton-with-ring-substituents",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "functional-acyclic-skeleton-with-ring-substituents",
        "generated-fused-parent",
        "generated-hydro-fused-parent",
        "generated-partial-hydro-fused-parent",
        "generated-spiro-fused-parent",
        "preferred-acyclic-skeleton-with-ring-substituents",
        "single-ring-parent",
    }
)
_HIGH_CONFIDENCE_HETERO_GENERATED_FUSED_SELECTED_FAMILIES = frozenset(
    {
        "acyclic-skeleton-with-ring-substituents",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "functional-acyclic-skeleton-with-ring-substituents",
        "fused-retained-parent",
        "generated-fused-parent",
        "generated-partial-hydro-fused-parent",
        "neutral-amine-parent",
        "preferred-acyclic-skeleton-with-ring-substituents",
        "single-ring-parent",
        "sulfur-oxo-acyclic-skeleton-with-ring-substituents",
    }
)
_HIGH_CONFIDENCE_HETERO_GENERATED_FUSED_TEMPLATE_SPECS = {
    "imidazo[4,5-c]quinoline": {
        "parent_atom_count": 13,
        "parent_element_counts": (("C", 10), ("N", 3)),
        "max_render_rank": 1,
    },
    "imidazolo[4,5-c]quinoline": {
        "parent_atom_count": 13,
        "parent_element_counts": (("C", 10), ("N", 3)),
        "max_render_rank": 2,
    },
    "isoquino[4,3-d]pyrazole": {
        "parent_atom_count": 13,
        "parent_element_counts": (("C", 10), ("N", 3)),
        "max_render_rank": 1,
    },
    "pyrido[4,3-f]indazole": {
        "parent_atom_count": 13,
        "parent_element_counts": (("C", 10), ("N", 3)),
        "max_render_rank": 1,
    },
    "pyrimido[4,5-b]pyridine": {
        "parent_atom_count": 10,
        "parent_element_counts": (("C", 7), ("N", 3)),
        "max_render_rank": 1,
    },
}


def _high_confidence_generated_fused_template_alternative(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
    *,
    template_names: frozenset[str] | None = None,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.source_file == "naturalProducts.xml":
        return None
    if _is_guarded_retained_benzocyclobutene_parent(selected_candidate):
        return None
    allowed_selected_families = (
        _GENERATED_FUSED_TEMPLATE_ALTERNATIVE_SELECTED_FAMILIES
        if template_names is None
        else _HIGH_CONFIDENCE_FAST_PATH_GENERATED_FUSED_SELECTED_FAMILIES
    )
    if selected_candidate.family not in allowed_selected_families:
        return None
    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "generated-fused-parent":
            continue
        if template_names is not None and candidate.template_name not in template_names:
            continue
        spec = _GENERATED_FUSED_TEMPLATE_ALTERNATIVE_SPECS.get(candidate.template_name)
        if spec is None:
            continue
        if selected_candidate.family not in spec["allowed_selected_families"]:
            continue
        if selected_candidate.family == "generated-fused-parent" and (
            selected_candidate.template_name == candidate.template_name
        ):
            continue
        if (
            selected_candidate.molecule_isotope_atom_count
            and selected_candidate.candidate_isotope_atom_count
            >= selected_candidate.molecule_isotope_atom_count
            and candidate.candidate_isotope_atom_count
            < selected_candidate.molecule_isotope_atom_count
        ):
            continue
        if selected_candidate.source_file == "naturalProducts.xml":
            continue
        if not _is_high_confidence_generated_fused_template_alternative(candidate, spec):
            continue
        alternatives.append((candidate_key, item))
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _high_confidence_generated_imidazo45a_alternative(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family not in _HIGH_CONFIDENCE_GENERATED_IMIDAZO45A_SELECTED_FAMILIES:
        return None
    if selected_candidate.source_file == "naturalProducts.xml":
        return None
    if (
        selected_candidate.family == "fused-retained-parent"
        and selected_candidate.template_name in {"1,3-benzoxazine", "benzimidazole"}
    ):
        return None
    if (
        selected_candidate.family == "generated-fused-parent"
        and selected_candidate.template_name == "imidazo[4,5-a]benzene"
    ):
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None
    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        max_render_rank = 3 if selected_candidate.family == "generated-fused-parent" else 1
        if not _is_high_confidence_generated_imidazo45a_candidate(
            candidate,
            max_render_rank=max_render_rank,
        ):
            continue
        if (
            selected_candidate.molecule_isotope_atom_count
            and candidate.candidate_isotope_atom_count
            < selected_candidate.molecule_isotope_atom_count
        ):
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(item.name) - len(selected.name) > 90:
            continue
        if item.stability_penalty > selected.stability_penalty + 180:
            continue
        if candidate_key[0] > selected_key[0] + 240:
            continue
        alternatives.append(
            (
                (
                    candidate_key[0],
                    len(item.name),
                    candidate_key,
                ),
                item,
            )
        )
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _high_confidence_pyrazolopyridine_hydro_alternative(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family not in _HIGH_CONFIDENCE_PYRAZOLOPYRIDINE_HYDRO_SELECTED_FAMILIES:
        return None
    if selected_candidate.source_file == "naturalProducts.xml":
        return None
    if (
        selected_candidate.molecule_isotope_atom_count
        or selected_candidate.candidate_isotope_atom_count
    ):
        return None
    if selected_candidate.charge is not None and selected_candidate.charge.charged_atoms:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None

    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family not in {
            "generated-hydro-fused-parent",
            "generated-partial-hydro-fused-parent",
        }:
            continue
        if candidate.template_name != "pyrazolo[3,4-d]pyridine":
            continue
        if candidate.source_family != "generated-fused":
            continue
        if candidate.source_file != "generated_fused_template_index.json":
            continue
        if candidate.priority != 28:
            continue
        if candidate.family == "generated-partial-hydro-fused-parent":
            if candidate.render_rank != 1:
                continue
        elif candidate.render_rank is not None:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        if len(candidate.parent_atoms) != 9:
            continue
        if candidate.parent_element_counts != (("C", 6), ("N", 3)):
            continue
        if candidate.parent_aromatic_atom_count or candidate.parent_hetero_atom_count:
            continue
        if candidate.parent_abs_formal_charge_sum or candidate.molecule_abs_formal_charge_sum:
            continue
        if _candidate_has_isotope_evidence(candidate):
            continue
        if candidate.prefixes or candidate.suffixes:
            continue
        if candidate.charge is not None and candidate.charge.charged_atoms:
            continue
        if candidate_safety_reasons(candidate):
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(item.name) - len(selected.name) > 140:
            continue
        if item.stability_penalty > selected.stability_penalty + 220:
            continue
        if candidate_key[0] > selected_key[0] + 260:
            continue
        alternatives.append(
            (
                (
                    candidate_key[0],
                    len(item.name),
                    candidate_key,
                ),
                item,
            )
        )
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _high_confidence_hetero_generated_fused_alternative(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family not in _HIGH_CONFIDENCE_HETERO_GENERATED_FUSED_SELECTED_FAMILIES:
        return None
    if selected_candidate.source_file == "naturalProducts.xml":
        return None
    if selected_candidate.charge is not None and selected_candidate.charge.charged_atoms:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None

    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        spec = _HIGH_CONFIDENCE_HETERO_GENERATED_FUSED_TEMPLATE_SPECS.get(
            candidate.template_name
        )
        if spec is None:
            continue
        if candidate.family != "generated-fused-parent":
            continue
        if candidate.source_family != "generated-fused":
            continue
        if candidate.source_file != "generated_fused_template_index.json":
            continue
        if candidate.priority != 28:
            continue
        max_render_rank = int(spec["max_render_rank"])
        if candidate.render_rank is None or candidate.render_rank > max_render_rank:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        if len(candidate.parent_atoms) != int(spec["parent_atom_count"]):
            continue
        if candidate.parent_element_counts != spec["parent_element_counts"]:
            continue
        if candidate.parent_hetero_atom_count != 3:
            continue
        if candidate.parent_abs_formal_charge_sum or candidate.molecule_abs_formal_charge_sum:
            continue
        if (
            selected_candidate.molecule_isotope_atom_count
            and candidate.candidate_isotope_atom_count
            < selected_candidate.molecule_isotope_atom_count
        ):
            continue
        if candidate.prefixes or candidate.suffixes:
            continue
        if candidate.charge is not None and candidate.charge.charged_atoms:
            continue
        if candidate_safety_reasons(candidate):
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(item.name) - len(selected.name) > 170:
            continue
        if item.stability_penalty > selected.stability_penalty + 220:
            continue
        if candidate_key[0] > selected_key[0] + 280:
            continue
        alternatives.append(
            (
                (
                    candidate_key[0],
                    candidate.render_rank,
                    len(item.name),
                    candidate_key,
                ),
                item,
            )
        )
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _high_confidence_acyclic_parent_alternative_for_beta_dicarbonyl_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family != "beta-dicarbonyl-parent":
        return None
    if selected_candidate.charge is not None or selected_candidate.suffixes:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None

    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "acyclic-parent":
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        candidate_element_counts = dict(candidate.parent_element_counts)
        if candidate_element_counts.get("O", 0) < 2:
            continue
        if candidate.parent_abs_formal_charge_sum or candidate.molecule_abs_formal_charge_sum:
            continue
        if (
            selected_candidate.molecule_isotope_atom_count
            and candidate.candidate_isotope_atom_count
            < selected_candidate.molecule_isotope_atom_count
        ):
            continue
        if candidate.prefixes or candidate.suffixes:
            continue
        if candidate.charge is not None and candidate.charge.charged_atoms:
            continue
        if candidate_safety_reasons(candidate):
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(item.name) - len(selected.name) > 80:
            continue
        if item.stability_penalty > selected.stability_penalty + 180:
            continue
        if candidate_key[0] > selected_key[0] + 120:
            continue
        alternatives.append(
            (
                (
                    candidate_key[0],
                    len(item.name),
                    candidate_key,
                ),
                item,
            )
        )
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _high_confidence_single_ring_generated_fused_template_alternative(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family != "single-ring-parent":
        return None
    if selected_candidate.template_name or selected_candidate.source_file:
        return None
    if selected_candidate.source_family:
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    selected_parent_count = len(selected_candidate.parent_atoms)
    if selected_parent_count < 3 or selected_parent_count > 6:
        return None
    if selected_candidate.molecule_isotope_atom_count or selected_candidate.candidate_isotope_atom_count:
        return None
    if selected_candidate.molecule_abs_formal_charge_sum:
        return None
    if selected_candidate.charge is not None or selected_candidate.suffixes:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    selected_rank = selected_candidate.render_rank
    if selected_rank is not None and selected_rank > 3:
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None

    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        spec = _HIGH_CONFIDENCE_SINGLE_RING_GENERATED_FUSED_TEMPLATE_SPECS.get(
            candidate.template_name
        )
        if spec is None:
            continue
        if not _is_high_confidence_generated_fused_template_alternative(candidate, spec):
            continue
        if candidate.molecule_isotope_atom_count or candidate.candidate_isotope_atom_count:
            continue
        if len(item.name) > 280:
            continue
        if len(item.name) - len(selected.name) > 170:
            continue
        if candidate_key[0] > selected_key[0] + 260:
            continue
        alternatives.append(
            (
                (
                    candidate_key[0],
                    len(item.name),
                    candidate_key,
                ),
                item,
            )
        )
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _high_confidence_single_ring_suffix_generated_fused_template_alternative(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family != "single-ring-parent":
        return None
    if selected_candidate.template_name or selected_candidate.source_file:
        return None
    if selected_candidate.source_family:
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    selected_parent_count = len(selected_candidate.parent_atoms)
    if selected_parent_count < 5 or selected_parent_count > 6:
        return None
    if not selected_candidate.suffixes:
        return None
    if selected_candidate.render_rank is not None and selected_candidate.render_rank > 3:
        return None
    if selected_candidate.molecule_isotope_atom_count or selected_candidate.candidate_isotope_atom_count:
        return None
    if selected_candidate.charge is not None and selected_candidate.charge.charged_atoms:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None
    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        spec = _HIGH_CONFIDENCE_SINGLE_RING_SUFFIX_GENERATED_FUSED_TEMPLATE_SPECS.get(
            candidate.template_name
        )
        if spec is None:
            continue
        if not _is_high_confidence_generated_fused_template_alternative(candidate, spec):
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(item.name) - len(selected.name) > 80:
            continue
        if item.stability_penalty > selected.stability_penalty + 220:
            continue
        if candidate_key[0] > selected_key[0] + 260:
            continue
        alternatives.append(
            (
                (
                    candidate_key[0],
                    len(item.name),
                    candidate_key,
                ),
                item,
            )
        )
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _is_high_confidence_generated_fused_template_alternative(
    candidate: NameCandidate,
    spec: dict[str, object],
) -> bool:
    parent_atom_count = int(spec["parent_atom_count"])
    parent_hetero_atom_count = int(spec["parent_hetero_atom_count"])
    if candidate.family != "generated-fused-parent":
        return False
    if candidate.source_family != "generated-fused":
        return False
    if candidate.source_file != "generated_fused_template_index.json":
        return False
    if candidate.priority != 28:
        return False
    if candidate.render_rank != 1:
        return False
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return False
    if len(candidate.parent_atoms) != parent_atom_count:
        return False
    if len(candidate.ring_system_atoms) != parent_atom_count:
        return False
    if candidate.parent_hetero_atom_count != parent_hetero_atom_count:
        return False
    if candidate.parent_abs_formal_charge_sum or candidate.molecule_abs_formal_charge_sum:
        return False
    if candidate.prefixes or candidate.suffixes:
        return False
    if candidate.charge is not None and candidate.charge.charged_atoms:
        return False
    if candidate_safety_reasons(candidate):
        return False
    return True


def _is_high_confidence_generated_imidazo45a_candidate(
    candidate: NameCandidate,
    *,
    max_render_rank: int = 1,
) -> bool:
    if candidate.family != "generated-fused-parent":
        return False
    if candidate.template_name != "imidazo[4,5-a]benzene":
        return False
    if candidate.source_family != "generated-fused":
        return False
    if candidate.source_file != "generated_fused_template_index.json":
        return False
    if candidate.priority != 28:
        return False
    if candidate.render_rank is None or candidate.render_rank > max_render_rank:
        return False
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return False
    if len(candidate.parent_atoms) != 9:
        return False
    if len(candidate.ring_system_atoms) != 9:
        return False
    if candidate.parent_element_counts != (("C", 7), ("N", 2)):
        return False
    if candidate.parent_aromatic_atom_count != 9:
        return False
    if candidate.parent_hetero_atom_count != 2:
        return False
    if candidate.parent_abs_formal_charge_sum or candidate.molecule_abs_formal_charge_sum:
        return False
    if _candidate_has_isotope_evidence(candidate):
        return False
    if candidate.prefixes or candidate.suffixes:
        return False
    if candidate.charge is not None and candidate.charge.charged_atoms:
        return False
    if candidate_safety_reasons(candidate):
        return False
    return True


_STEROID_RETAINED_PARENT_TEMPLATES = frozenset({"estr", "gon"})
_STEROID_TOPOLOGY_PARENT_TEMPLATES = frozenset({"pregn", "chol", "cholest"})
_STEROID_PARENT_COMPETITOR_FAMILIES = frozenset(
    {
        "acyclic-skeleton-with-ring-substituents",
        "charged-retained-parent",
        "fused-retained-parent",
        "natural-product-topology-parent",
    }
)


def _steroid_retained_parent_alternative(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if not _is_steroid_retained_parent_alternative(candidate):
            continue
        if not _selected_allows_steroid_retained_parent_alternative(
            selected_candidate,
            candidate,
        ):
            continue
        if candidate_key[0] > 130:
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        alternatives.append((candidate_key, item))
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _selected_allows_steroid_retained_parent_alternative(
    selected_candidate: NameCandidate,
    candidate: NameCandidate,
) -> bool:
    if selected_candidate.family not in _STEROID_PARENT_COMPETITOR_FAMILIES:
        return False
    if selected_candidate.source_file == "naturalProducts.xml":
        if selected_candidate.source_family != "naturalProducts:ring:biochemical":
            return False
        if selected_candidate.family != "natural-product-topology-parent":
            return False
        if selected_candidate.template_name not in _STEROID_TOPOLOGY_PARENT_TEMPLATES:
            return False
        if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
            return False
        if len(selected_candidate.parent_atoms) < 21:
            return False
        return candidate.render_rank is not None and candidate.render_rank <= 5
    if selected_candidate.family == "fused-retained-parent":
        if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
            return False
        if len(selected_candidate.parent_atoms) > 9:
            return False
    return candidate.render_rank is not None and candidate.render_rank <= 7


def _is_steroid_retained_parent_alternative(candidate: NameCandidate) -> bool:
    if candidate.family != "fused-retained-parent":
        return False
    if candidate.template_name not in _STEROID_RETAINED_PARENT_TEMPLATES:
        return False
    if candidate.source_family != "naturalProducts:ring:biochemical":
        return False
    if candidate.source_file != "naturalProducts.xml":
        return False
    if candidate.priority != 18:
        return False
    if candidate.render_rank is None or candidate.render_rank > 7:
        return False
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return False
    if len(candidate.parent_atoms) not in {17, 18}:
        return False
    if candidate.parent_aromatic_atom_count or candidate.parent_hetero_atom_count:
        return False
    if candidate.parent_abs_formal_charge_sum or candidate.template_abs_formal_charge_sum:
        return False
    if candidate.prefixes or candidate.suffixes:
        return False
    if candidate.charge is not None and candidate.charge.charged_atoms:
        return False
    if candidate_safety_reasons(candidate):
        return False
    return True


def _generated_hydro_fused_alternative_for_aromatic_generated_fused_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family != "generated-fused-parent":
        return None
    if selected_candidate.source_family != "generated-fused":
        return None
    if selected_candidate.source_file != "generated_fused_template_index.json":
        return None
    if selected_candidate.priority != 28 or selected_candidate.render_rank != 1:
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    selected_parent_atom_count = len(selected_candidate.parent_atoms)
    if selected_parent_atom_count != 9:
        return None
    if selected_candidate.parent_element_counts not in _GENERATED_HYDRO_PYRIDO_IMIDAZOLE_SELECTED_PARENT_COUNTS:
        return None
    if selected_candidate.parent_aromatic_atom_count != selected_parent_atom_count:
        return None
    if selected_candidate.parent_hetero_atom_count < 1 or selected_candidate.parent_hetero_atom_count > 3:
        return None
    if selected_candidate.parent_external_oxygen_substituent_count:
        return None
    if selected_candidate.prefixes or selected_candidate.suffixes:
        return None
    if selected_candidate.charge is not None and selected_candidate.charge.charged_atoms:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None
    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "generated-hydro-fused-parent":
            continue
        if candidate.source_family != "generated-fused":
            continue
        if candidate.source_file != "generated_fused_template_index.json":
            continue
        if candidate.template_name != "pyrido[4,5-d]imidazole":
            continue
        if candidate.priority != 28 or candidate.render_rank is not None:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        candidate_parent_atom_count = len(candidate.parent_atoms)
        if candidate_parent_atom_count != 9:
            continue
        if candidate.parent_element_counts != _GENERATED_HYDRO_PYRIDO_IMIDAZOLE_PARENT_COUNTS:
            continue
        if candidate_parent_atom_count < selected_parent_atom_count:
            continue
        if candidate.parent_aromatic_atom_count:
            continue
        if candidate.parent_hetero_atom_count:
            continue
        if candidate.prefixes or candidate.suffixes:
            continue
        if candidate.charge is not None and candidate.charge.charged_atoms:
            continue
        if candidate_safety_reasons(candidate):
            continue
        if len(item.name) > 280:
            continue
        if len(item.name) - len(selected.name) > 160:
            continue
        if candidate_key[0] > selected_key[0] + 220:
            continue
        alternatives.append((candidate_key, item))
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _hydro_retained_benzimidazole_alternative_for_generated_imidazo45a_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family != "generated-fused-parent":
        return None
    if selected_candidate.template_name != "imidazo[4,5-a]benzene":
        return None
    if selected_candidate.source_family != "generated-fused":
        return None
    if selected_candidate.source_file != "generated_fused_template_index.json":
        return None
    if selected_candidate.priority != 28 or selected_candidate.render_rank != 1:
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    if len(selected_candidate.parent_atoms) != 9:
        return None
    if selected_candidate.parent_element_counts != (("C", 7), ("N", 2)):
        return None
    if selected_candidate.prefixes or selected_candidate.suffixes:
        return None
    if selected_candidate.charge is not None:
        return None
    if _candidate_has_isotope_evidence(selected_candidate):
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None

    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "hydro-fused-retained-parent":
            continue
        if candidate.template_name != "benzimidazole":
            continue
        if candidate.source_family != "static-retained":
            continue
        if candidate.source_file != "high_impact_fused_parent_templates.json":
            continue
        if candidate.priority != 18:
            continue
        if candidate.render_rank is not None:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        if len(candidate.parent_atoms) != 9:
            continue
        if candidate.parent_element_counts != selected_candidate.parent_element_counts:
            continue
        if candidate.prefixes or candidate.suffixes:
            continue
        if candidate.charge is not None:
            continue
        if _candidate_has_isotope_evidence(candidate):
            continue
        if candidate_safety_reasons(candidate):
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(item.name) - len(selected.name) > 30:
            continue
        if item.stability_penalty > selected.stability_penalty + 80:
            continue
        if candidate_key[0] > selected_key[0] + 140:
            continue
        alternatives.append(
            (
                (
                    candidate_key[0],
                    len(item.name),
                    candidate_key,
                ),
                item,
            )
        )
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _high_confidence_generated_hydro_pyrido45d_alternative(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family not in _HIGH_CONFIDENCE_GENERATED_HYDRO_PYRIDO45D_SELECTED_FAMILIES:
        return None
    if selected_candidate.source_file == "naturalProducts.xml":
        return None
    if _candidate_has_isotope_evidence(selected_candidate):
        return None
    if selected_candidate.charge is not None and selected_candidate.charge.charged_atoms:
        return None
    if selected_candidate.suffixes:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None
    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if not _is_high_confidence_generated_hydro_pyrido45d_candidate(candidate):
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(item.name) - len(selected.name) > 80:
            continue
        if item.stability_penalty > selected.stability_penalty + 220:
            continue
        if candidate_key[0] > selected_key[0] + 260:
            continue
        alternatives.append(
            (
                (
                    candidate_key[0],
                    len(item.name),
                    candidate_key,
                ),
                item,
            )
        )
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _is_high_confidence_generated_hydro_pyrido45d_candidate(candidate: NameCandidate) -> bool:
    if candidate.family != "generated-hydro-fused-parent":
        return False
    if candidate.template_name != "pyrido[4,5-d]imidazole":
        return False
    if candidate.source_family != "generated-fused":
        return False
    if candidate.source_file != "generated_fused_template_index.json":
        return False
    if candidate.priority != 28:
        return False
    if candidate.render_rank is not None:
        return False
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return False
    if len(candidate.parent_atoms) != 9:
        return False
    if candidate.parent_element_counts != _GENERATED_HYDRO_PYRIDO_IMIDAZOLE_PARENT_COUNTS:
        return False
    if candidate.parent_aromatic_atom_count:
        return False
    if candidate.parent_hetero_atom_count:
        return False
    if _candidate_has_isotope_evidence(candidate):
        return False
    if candidate.prefixes or candidate.suffixes:
        return False
    if candidate.charge is not None and candidate.charge.charged_atoms:
        return False
    if candidate_safety_reasons(candidate):
        return False
    return True


def _chain_dominant_acyclic_alternative_for_compact_kekulized_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family != "kekulized-generic-polycyclic-parent":
        return None
    if selected_candidate.source_family != "kekulized":
        return None
    if selected_candidate.template_name or selected_candidate.source_file:
        return None
    if selected_candidate.render_rank != 1:
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    selected_parent_atom_count = len(selected_candidate.parent_atoms)
    if selected_parent_atom_count < 8 or selected_parent_atom_count > 12:
        return None
    selected_locant_count = len(selected_candidate.locants)
    if selected_locant_count < 2:
        return None
    if selected_candidate.parent_hetero_atom_count:
        return None
    if selected_candidate.charge is not None or selected_candidate.suffixes or selected_candidate.prefixes:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None
    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "chain-dominant-acyclic-skeleton-with-ring-substituents":
            continue
        if candidate.template_name or candidate.source_family or candidate.source_file:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        candidate_parent_atom_count = len(candidate.parent_atoms)
        if not (
            (
                selected_parent_atom_count == 9
                and selected_locant_count == 3
                and candidate_parent_atom_count in {5, 6}
            )
            or (
                candidate_parent_atom_count >= 6
                and candidate_parent_atom_count >= selected_parent_atom_count - 3
            )
        ):
            continue
        if selected_parent_atom_count == 10 and candidate_parent_atom_count > selected_parent_atom_count:
            continue
        if candidate.parent_hetero_atom_count:
            continue
        if candidate.charge is not None or candidate.suffixes or candidate.prefixes:
            continue
        if candidate_safety_reasons(candidate):
            continue
        if len(item.name) > 260:
            continue
        if abs(len(item.name) - len(selected.name)) > 12:
            continue
        if candidate_key[0] > selected_key[0] + 220:
            continue
        alternatives.append((candidate_key, item))
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _kekulized_polycyclic_alternative_for_chain_dominant_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family != "chain-dominant-acyclic-skeleton-with-ring-substituents":
        return None
    if selected_candidate.template_name or selected_candidate.source_family or selected_candidate.source_file:
        return None
    if selected_candidate.priority != 19 or selected_candidate.render_rank is not None:
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    if selected_candidate.parent_aromatic_atom_count or selected_candidate.parent_hetero_atom_count:
        return None
    if selected_candidate.charge is not None or selected_candidate.suffixes or selected_candidate.prefixes:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None

    ranked_items = sorted(keyed_items, key=lambda keyed: keyed[0])
    rank_by_id = {id(item): rank for rank, (_key, item) in enumerate(ranked_items, start=1)}
    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "kekulized-generic-polycyclic-parent":
            continue
        if candidate.source_family != "kekulized":
            continue
        if candidate.template_name or candidate.source_file:
            continue
        if candidate.priority not in {14, 28} or candidate.render_rank != 1:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        candidate_parent_atom_count = len(candidate.parent_atoms)
        if candidate_parent_atom_count < 11:
            continue
        candidate_locant_count = len(candidate.locants)
        if candidate_locant_count < 2 or candidate_locant_count > 7:
            continue
        if candidate.parent_aromatic_atom_count or candidate.parent_hetero_atom_count:
            continue
        if candidate.charge is not None or candidate.suffixes or candidate.prefixes:
            continue
        if candidate_safety_reasons(candidate):
            continue
        candidate_rank = rank_by_id.get(id(item))
        if candidate_rank is None or candidate_rank > 2:
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(item.name) - len(selected.name) > 90:
            continue
        if item.stability_penalty > selected.stability_penalty + 220:
            continue
        if candidate_key[0] > selected_key[0] + 260:
            continue
        alternatives.append(
            (
                (
                    candidate_rank,
                    candidate_key[0],
                    candidate_locant_count,
                    len(item.name),
                    candidate_key,
                ),
                item,
            )
        )
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


_BICYCLIC_RANK2_ORIENTATION_MIN_LOCANTS = {
    7: 3,
    8: 3,
    9: 3,
    10: 2,
    11: 3,
}


def _is_clean_templateless_bicyclic_parent(candidate: NameCandidate) -> bool:
    if candidate.family not in {"generic-bicyclic-parent", "functional-generic-bicyclic-parent"}:
        return False
    if candidate.source_family != "bicyclic":
        return False
    if candidate.template_name or candidate.source_file:
        return False
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return False
    if candidate.charge is not None or candidate.suffixes or candidate.prefixes:
        return False
    return not candidate_safety_reasons(candidate)


def _bicyclic_orientation_min_locants(parent_count: int) -> int | None:
    return _BICYCLIC_RANK2_ORIENTATION_MIN_LOCANTS.get(parent_count)


def _bicyclic_orientation_min_locants_for_candidate(candidate: NameCandidate) -> int | None:
    parent_count = len(candidate.parent_atoms)
    min_locants = _bicyclic_orientation_min_locants(parent_count)
    if (
        parent_count == 8
        and min_locants is not None
        and candidate.parent_internal_unsaturated_bond_count
    ):
        return min(min_locants, 2)
    return min_locants


def _alternate_bicyclic_orientation_for_rank1_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if not _is_clean_templateless_bicyclic_parent(selected_candidate):
        return None
    if selected_candidate.render_rank != 1:
        return None
    selected_parent_count = len(selected_candidate.parent_atoms)
    min_locants = _bicyclic_orientation_min_locants_for_candidate(selected_candidate)
    if min_locants is None:
        return None
    if selected_parent_count < 10 and selected_candidate.family != "generic-bicyclic-parent":
        return None
    if len(selected_candidate.locants) < min_locants:
        return None
    if selected_parent_count < 10 and len(selected_candidate.locants) > 5:
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None
    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != selected_candidate.family:
            continue
        if not _is_clean_templateless_bicyclic_parent(candidate):
            continue
        if candidate.render_rank != 2:
            continue
        if len(candidate.parent_atoms) != selected_parent_count:
            continue
        if candidate.parent_hetero_atom_count != selected_candidate.parent_hetero_atom_count:
            continue
        if candidate.parent_internal_unsaturated_bond_count != selected_candidate.parent_internal_unsaturated_bond_count:
            continue
        if len(candidate.locants) < min_locants:
            continue
        if selected_parent_count < 10 and len(candidate.locants) > 5:
            continue
        if len(item.name) > 280:
            continue
        if abs(len(item.name) - len(selected.name)) > 4:
            continue
        if candidate_key[0] != selected_key[0]:
            continue
        if candidate_key[0] > selected_key[0] + 220:
            continue
        alternatives.append((candidate_key, item))
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _earlier_bicyclic_orientation_for_late_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if not _is_clean_templateless_bicyclic_parent(selected_candidate):
        return None
    selected_render_rank = selected_candidate.render_rank
    if selected_render_rank is None or selected_render_rank <= 3:
        return None
    selected_parent_count = len(selected_candidate.parent_atoms)
    if selected_parent_count < 7 or selected_parent_count > 10:
        return None
    min_locants = _bicyclic_orientation_min_locants_for_candidate(selected_candidate)
    if min_locants is None or len(selected_candidate.locants) < min_locants:
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None

    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if not _is_clean_templateless_bicyclic_parent(candidate):
            continue
        if len(candidate.parent_atoms) != selected_parent_count:
            continue
        if candidate.parent_hetero_atom_count != selected_candidate.parent_hetero_atom_count:
            continue
        if candidate.parent_internal_unsaturated_bond_count != selected_candidate.parent_internal_unsaturated_bond_count:
            continue
        candidate_render_rank = candidate.render_rank
        if candidate_render_rank is None or candidate_render_rank > 3:
            continue
        if candidate_render_rank >= selected_render_rank:
            continue
        if len(candidate.locants) < min_locants:
            continue
        if len(item.name) > 280:
            continue
        if len(item.name) - len(selected.name) > 20:
            continue
        if candidate_key[0] > selected_key[0] + 80:
            continue
        alternatives.append(
            (
                (
                    candidate_key[0],
                    0 if candidate.family == selected_candidate.family else 1,
                    candidate_render_rank,
                    candidate_key,
                ),
                item,
            )
        )
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


_BICYCLIC_DECOMPOSITION_PARENT_FAMILIES = frozenset(
    {
        "acyclic-skeleton-with-ring-substituents",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "functional-acyclic-skeleton-with-ring-substituents",
        "preferred-acyclic-skeleton-with-ring-substituents",
        "generated-fused-parent",
        "high-signal-generated-topology-fused-parent",
    }
)


def _clean_bicyclic_alternative_for_decomposition_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family not in _BICYCLIC_DECOMPOSITION_PARENT_FAMILIES:
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    if selected_candidate.charge is not None or selected_candidate.suffixes or selected_candidate.prefixes:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    selected_parent_count = len(selected_candidate.parent_atoms)
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None

    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if not _is_clean_templateless_bicyclic_parent(candidate):
            continue
        candidate_parent_count = len(candidate.parent_atoms)
        if candidate_parent_count < 8 or candidate_parent_count > 10:
            continue
        min_locants = _bicyclic_orientation_min_locants_for_candidate(candidate)
        if min_locants is None or len(candidate.locants) < min_locants:
            continue
        candidate_render_rank = candidate.render_rank
        if candidate_render_rank is None or candidate_render_rank > 2:
            continue
        if selected_candidate.family in _ACYCLIC_SKELETON_PARENT_FAMILIES:
            if selected_parent_count - candidate_parent_count < 1:
                continue
        elif abs(selected_parent_count - candidate_parent_count) > 2:
            continue
        if len(item.name) > 220:
            continue
        if len(item.name) - len(selected.name) > 100:
            continue
        if candidate_key[0] > selected_key[0] + 220:
            continue
        alternatives.append(
            (
                (
                    candidate_key[0],
                    candidate_render_rank,
                    candidate_key,
                ),
                item,
            )
        )
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


_HIGH_CONFIDENCE_RETAINED_FUSED_HYDRO_BLOCKED_TEMPLATES = frozenset(
    {
        "benzo[1,2-c]pyrazole",
    }
)


def _generated_hydro_fused_alternative_for_retained_fused_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
    *,
    include_partial_hydro: bool = False,
    reject_isotopes: bool = False,
    blocked_template_names: frozenset[str] = frozenset(),
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family != "fused-retained-parent":
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    selected_parent_count = len(selected_candidate.parent_atoms)
    if selected_parent_count < 9 or selected_parent_count > 14:
        return None
    if selected_candidate.charge is not None or selected_candidate.suffixes:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    selected_rank = selected_candidate.render_rank
    if selected_rank is not None and selected_rank > 3:
        return None
    if reject_isotopes and (
        selected_candidate.molecule_isotope_atom_count
        or selected_candidate.candidate_isotope_atom_count
    ):
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None

    allowed_families = {"generated-hydro-fused-parent"}
    if include_partial_hydro:
        allowed_families.add("generated-partial-hydro-fused-parent")
    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family not in allowed_families:
            continue
        if candidate.source_family != "generated-fused":
            continue
        if candidate.source_file != "generated_fused_template_index.json":
            continue
        if candidate.template_name in blocked_template_names:
            continue
        if candidate.priority != 28:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        if reject_isotopes and (
            candidate.molecule_isotope_atom_count
            or candidate.candidate_isotope_atom_count
        ):
            continue
        candidate_parent_count = len(candidate.parent_atoms)
        if candidate_parent_count < 8 or candidate_parent_count > 14:
            continue
        if candidate.family == "generated-partial-hydro-fused-parent":
            render_rank = candidate.render_rank
            if render_rank is None or render_rank > 3:
                continue
            if candidate_parent_count < selected_parent_count - 1:
                continue
            if candidate_parent_count > selected_parent_count + 2:
                continue
        elif abs(candidate_parent_count - selected_parent_count) > 3:
            continue
        if candidate.parent_external_unsaturated_bond_count < 1:
            continue
        if candidate.charge is not None or candidate.suffixes:
            continue
        if candidate_safety_reasons(candidate):
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(item.name) - len(selected.name) > 150:
            continue
        if item.stability_penalty > selected.stability_penalty + 180:
            continue
        if candidate_key[0] > selected_key[0] + 260:
            continue
        alternatives.append(
            (
                (
                    candidate_key[0],
                    len(item.name),
                    candidate_parent_count,
                    candidate_key,
                ),
                item,
            )
        )
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _hydro_retained_alternative_for_retained_fused_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family != "fused-retained-parent":
        return None
    if selected_candidate.template_name not in _HYDRO_RETAINED_ALIAS_REPLACEMENT_TEMPLATES:
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    selected_parent_count = len(selected_candidate.parent_atoms)
    if selected_parent_count < 9 or selected_parent_count > 14:
        return None
    if not selected_candidate.stereo_locants:
        return None
    if selected_candidate.parent_external_unsaturated_bond_count < 1:
        return None
    if selected_candidate.charge is not None or selected_candidate.suffixes:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    selected_rank = selected_candidate.render_rank
    if selected_rank is not None and selected_rank > 3:
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None

    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "hydro-fused-retained-parent":
            continue
        if candidate.source_family != "opsin-retained-ring-cml":
            continue
        if candidate.source_file != "opsin_retained_mono_fused_ring_smiles.csv":
            continue
        if candidate.parent_atoms != selected_candidate.parent_atoms:
            continue
        if not candidate_has_complete_graph(candidate):
            continue
        if set(candidate.stereo_locants) != set(selected_candidate.stereo_locants):
            continue
        if candidate.parent_element_counts != selected_candidate.parent_element_counts:
            continue
        if candidate.parent_external_unsaturated_bond_count != selected_candidate.parent_external_unsaturated_bond_count:
            continue
        if candidate.charge is not None or candidate.suffixes:
            continue
        if candidate_safety_reasons(candidate):
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(item.name) - len(selected.name) > 120:
            continue
        if item.stability_penalty > selected.stability_penalty + 120:
            continue
        if candidate_key[0] > selected_key[0] + 180:
            continue
        alternatives.append((candidate_key, item))
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


_LATE_SINGLE_RING_GENERATED_HYDRO_TEMPLATES = frozenset(
    {
        "pyrano[3,4-d]imidazole",
        "pyrido[4,5-d]imidazole",
        "pyrimido[4,5-b]pyrrole",
    }
)


def _generated_hydro_fused_alternative_for_late_single_ring_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family != "single-ring-parent":
        return None
    if selected_candidate.priority != 61:
        return None
    if selected_candidate.render_rank not in {1, 2}:
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    if len(selected_candidate.parent_atoms) != 6:
        return None
    if len(selected_candidate.locants) < 6:
        return None
    if selected_candidate.charge is not None or selected_candidate.suffixes:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None

    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "generated-hydro-fused-parent":
            continue
        if candidate.source_family != "generated-fused":
            continue
        if candidate.source_file != "generated_fused_template_index.json":
            continue
        if candidate.template_name not in _LATE_SINGLE_RING_GENERATED_HYDRO_TEMPLATES:
            continue
        if candidate.priority != 28 or candidate.render_rank is not None:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        if len(candidate.parent_atoms) != 9:
            continue
        if candidate.charge is not None or candidate.suffixes:
            continue
        if candidate_safety_reasons(candidate):
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(item.name) - len(selected.name) > 160:
            continue
        if item.stability_penalty > selected.stability_penalty + 180:
            continue
        if candidate_key[0] > selected_key[0] + 260:
            continue
        alternatives.append(
            (
                (
                    candidate_key[0],
                    len(item.name),
                    candidate_key,
                ),
                item,
            )
        )
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _single_ring_alternative_for_low_confidence_generated_partial_hydro_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family != "generated-partial-hydro-fused-parent":
        return None
    if selected_candidate.source_family != "generated-fused":
        return None
    if selected_candidate.source_file != "generated_fused_template_index.json":
        return None
    if not selected_candidate.template_name:
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    if selected_candidate.charge is not None or selected_candidate.suffixes:
        return None
    if len(selected_candidate.parent_atoms) < 8:
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None
    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "single-ring-parent":
            continue
        if candidate.template_name:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        if candidate.charge is not None or candidate.suffixes:
            continue
        if len(candidate.parent_atoms) < 4:
            continue
        if len(item.name) > 280:
            continue
        if len(item.name) - len(selected.name) > 170:
            continue
        if candidate_key[0] > selected_key[0]:
            continue
        alternatives.append((candidate_key, item))
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _acyclic_skeleton_alternative_for_generated_partial_hydro_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family != "generated-partial-hydro-fused-parent":
        return None
    if selected_candidate.source_family != "generated-fused":
        return None
    if selected_candidate.source_file != "generated_fused_template_index.json":
        return None
    if not selected_candidate.template_name:
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    selected_parent_count = len(selected_candidate.parent_atoms)
    if selected_parent_count < 8 or selected_parent_count > 10:
        return None
    if selected_candidate.parent_hetero_atom_count:
        return None
    if selected_candidate.parent_aromatic_atom_count:
        return None
    if selected_candidate.prefixes or selected_candidate.suffixes:
        return None
    if selected_candidate.charge is not None and selected_candidate.charge.charged_atoms:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None

    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "acyclic-skeleton-with-ring-substituents":
            continue
        if candidate.template_name or candidate.source_family or candidate.source_file:
            continue
        if candidate.priority != 60 or candidate.render_rank is not None:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        candidate_parent_count = len(candidate.parent_atoms)
        if candidate_parent_count < 4 or candidate_parent_count > 14:
            continue
        if candidate.parent_hetero_atom_count:
            continue
        if candidate.parent_aromatic_atom_count:
            continue
        if candidate.locants:
            continue
        if candidate.prefixes or candidate.suffixes:
            continue
        if candidate.charge is not None or candidate_safety_reasons(candidate):
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(item.name) - len(selected.name) > 180:
            continue
        if item.stability_penalty > selected.stability_penalty + 220:
            continue
        if candidate_key[0] > selected_key[0] + 360:
            continue
        alternatives.append(
            (
                (
                    candidate_key[0],
                    len(item.name),
                    candidate_parent_count,
                    candidate_key,
                ),
                item,
            )
        )
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


_FUNCTIONALIZED_GENERATED_FUSED_ACYCLIC_TEMPLATES = frozenset(
    {
        "benzo[1,2-d]oxazole",
        "benzo[1,2-d]thiazole",
        "imidazo[1,2-a]pyridine",
        "imidazo[4,5-a]benzene",
        "pyrido[2,3-b]pyrrole",
        "pyrrolo[2,3-b]pyridine",
    }
)


def _acyclic_skeleton_alternative_for_functionalized_generated_fused_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family != "generated-fused-parent":
        return None
    if selected_candidate.source_family != "generated-fused":
        return None
    if selected_candidate.source_file != "generated_fused_template_index.json":
        return None
    if selected_candidate.template_name not in _FUNCTIONALIZED_GENERATED_FUSED_ACYCLIC_TEMPLATES:
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    if len(selected_candidate.parent_atoms) != 9:
        return None
    if selected_candidate.parent_aromatic_atom_count != 9 or selected_candidate.parent_hetero_atom_count < 2:
        return None
    if selected_candidate.prefixes or selected_candidate.suffixes:
        return None
    if selected_candidate.charge is not None and selected_candidate.charge.charged_atoms:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None
    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "acyclic-skeleton-with-ring-substituents":
            continue
        if candidate.template_name or candidate.source_family or candidate.source_file:
            continue
        if candidate.priority != 60 or candidate.render_rank is not None:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        candidate_parent_count = len(candidate.parent_atoms)
        if candidate_parent_count < 10 or candidate_parent_count > 14:
            continue
        if candidate.parent_aromatic_atom_count or candidate.parent_hetero_atom_count:
            continue
        if candidate.parent_external_nitrogen_substituent_count or candidate.parent_external_oxygen_substituent_count:
            continue
        if candidate.prefixes or candidate.suffixes:
            continue
        if candidate.charge is not None or candidate_safety_reasons(candidate):
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(item.name) - len(selected.name) > 160:
            continue
        if item.stability_penalty > selected.stability_penalty + 160:
            continue
        if candidate_key[0] > selected_key[0] + 260:
            continue
        alternatives.append(
            (
                (
                    candidate_key[0],
                    candidate_parent_count,
                    len(item.name),
                    candidate_key,
                ),
                item,
            )
        )
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


_RANKER_PROTECTED_SENIOR_FUNCTIONAL_FAMILIES = frozenset(
    {
        "spiro-functional-chain-substituent",
    }
)

_RANKER_SUFFIXLESS_GRAPH_PARENT_FAMILIES = frozenset(
    {
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
        "retained-topology-replacement-fused-parent",
        "skeletal-replacement-fused-parent",
    }
)

_RANKER_ACYCLIC_SKELETON_WITH_RING_SUBSTITUENT_FAMILIES = frozenset(
    {
        "acyclic-skeleton-with-ring-substituents",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "functional-acyclic-skeleton-with-ring-substituents",
        "preferred-acyclic-skeleton-with-ring-substituents",
    }
)

_RANKER_COMPACT_FUSED_SHORTCUT_FAMILIES = frozenset(
    {
        "fused-retained-parent",
        "generated-fused-parent",
        "generated-hydro-fused-parent",
        "generated-partial-hydro-fused-parent",
        "hydro-fused-retained-parent",
        "retained-topology-replacement-fused-parent",
        "skeletal-replacement-fused-parent",
    }
)

_RANKER_SENIOR_FUNCTIONAL_COMPLETE_PARENT_FAMILIES = frozenset(
    {
        "dithiazole-retained-parent",
        "phosphorus-acid-parent",
        "small-inorganic-retained-parent",
        "sulfonamide-parent",
    }
)

_RANKER_SENIOR_FUNCTIONAL_SHORTCUT_FAMILIES = (
    _RANKER_ACYCLIC_SKELETON_WITH_RING_SUBSTITUENT_FAMILIES
    | _RANKER_SUFFIXLESS_GRAPH_PARENT_FAMILIES
    | frozenset(
        {
            "neutral-amine-parent",
            "single-ring-parent",
        }
    )
)


_STEREO_COVERAGE_PARENT_FAMILIES = (
    _RANKER_ACYCLIC_SKELETON_WITH_RING_SUBSTITUENT_FAMILIES
    | _RANKER_SUFFIXLESS_GRAPH_PARENT_FAMILIES
    | frozenset(
        {
            "acyclic-parent",
            "component-dispiro-parent",
            "functional-generic-bicyclic-parent",
            "natural-product-topology-parent",
            "poly-component-spiro-parent",
            "preferred-acyclic-parent",
            "retained-amide-parent",
            "single-ring-parent",
            "spiro-functional-chain-substituent",
        }
    )
)


_STEREO_COVERAGE_SELECTED_FAMILIES = frozenset(
    {
        "acyclic-parent",
        "acyclic-skeleton-with-ring-substituents",
        "beta-dicarbonyl-parent",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "functional-acyclic-skeleton-with-ring-substituents",
        "hydrazine-parent",
        "neutral-amine-parent",
        "phosphate-ester-retained-parent",
        "preferred-acyclic-skeleton-with-ring-substituents",
        "retained-acid-ester-parent",
        "retained-amide-parent",
        "retained-ester-parent",
        "sulfonamide-retained-parent",
        "terminal-carboxylate-parent",
    }
)


def _stereo_coverage_alternative_for_selected_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family not in _STEREO_COVERAGE_SELECTED_FAMILIES:
        return None
    if _is_high_confidence_amino_acid_retained_parent(selected_candidate):
        return None
    if selected_candidate.source_family == "acyclic-carbohydrate-retained":
        return None
    if selected_candidate.stereo is None or selected_candidate.stereo.bond_descriptors:
        return None
    stereo_total, selected_covered = _candidate_atom_stereo_coverage(selected_candidate)
    if stereo_total <= 0 or selected_covered >= stereo_total:
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None
    selected_name = _prepared_output_name(selected)
    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family not in _STEREO_COVERAGE_PARENT_FAMILIES:
            continue
        if candidate.stereo_prefix:
            continue
        if candidate.stereo is None or candidate.stereo.bond_descriptors:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        if candidate_safety_reasons(candidate):
            continue
        if candidate.charge is not None or candidate.suffixes:
            continue
        total, covered = _candidate_atom_stereo_coverage(candidate)
        if total != stereo_total or covered < stereo_total:
            continue
        if covered <= selected_covered:
            continue
        if len(candidate.parent_atoms) < min(len(selected_candidate.parent_atoms), 6):
            continue
        proposed_name = _prepared_output_name(item)
        if len(proposed_name) > 220 or len(proposed_name) - len(selected_name) > 80:
            continue
        if item.stability_penalty > selected.stability_penalty + 90:
            continue
        if candidate_key[0] > selected_key[0] + 220:
            continue
        alternatives.append(((-covered, len(proposed_name), candidate_key), item))
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _candidate_atom_stereo_coverage(candidate: NameCandidate) -> tuple[int, int]:
    stereo = candidate.stereo
    if stereo is None or not stereo.atom_descriptors:
        return 0, 0
    locanted_atoms = {atom_idx for atom_idx, _locant in candidate.stereo_locants}
    total = len(stereo.atom_descriptors)
    covered = sum(1 for atom_idx, _descriptor in stereo.atom_descriptors if atom_idx in locanted_atoms)
    return total, covered


def _candidate_fact_ranked_switch_is_safe(
    current: _PreparedCandidate,
    proposed: _PreparedCandidate,
    current_facts: dict[str, object],
    proposed_facts: dict[str, object],
    current_key: tuple,
    proposed_key: tuple,
) -> bool:
    """Keep learned switches from discarding senior functional coverage.

    The fact ranker is intentionally allowed to prefer graph parents when a
    correct candidate exists, but a suffixless retained/fused/spiro parent can
    hide senior functional structure that the deterministic key has already
    selected.  This guard uses only candidate facts: lower priority, complete
    graph coverage, candidate family, suffix/charge evidence, and parent size.
    """

    current_candidate = current.candidate
    proposed_candidate = proposed.candidate
    if _should_defer_fast_senior_fragile_external_unsaturated_acyclic(proposed):
        return False
    if (
        current_candidate.source_family == "acyclic-carbohydrate-retained"
        and proposed_candidate.family == "acyclic-parent"
    ):
        return False
    if not _ranker_preserves_high_signal_topology_parent(current, proposed):
        return False
    if not _ranker_preserves_generated_purine_parent(current, proposed):
        return False
    if not _ranker_preserves_biochemical_retained_parent_over_shortcut(
        current,
        proposed,
        current_key,
        proposed_key,
    ):
        return False
    if not _ranker_preserves_retained_benzimidazole_parent(current, proposed):
        return False
    if not _ranker_preserves_early_retained_benzimidazole_parent(current, proposed):
        return False
    if not _ranker_preserves_rank1_retained_fused_parent(current, proposed):
        return False
    if not _ranker_preserves_systematic_candidate_over_penalized_retained_shortcut(
        current,
        proposed,
        current_key,
        proposed_key,
    ):
        return False
    if not _ranker_preserves_generated_fused_parent_over_penalized_retained_shortcut(
        current,
        proposed,
        current_key,
        proposed_key,
    ):
        return False
    if not _ranker_preserves_single_ring_parent_over_external_nitrogen_generated_fused(
        current,
        proposed,
        current_key,
        proposed_key,
    ):
        return False
    if not _ranker_preserves_rank1_hetero_generated_fused_parent_over_acyclic_skeleton(current, proposed):
        return False
    if not _ranker_preserves_substituent_ring_parent_over_compact_chalcogen_suffix(
        current,
        proposed,
        current_key,
        proposed_key,
    ):
        return False
    if not _ranker_preserves_charged_ring_skeleton_parent_over_single_ring_shortcut(
        current,
        proposed,
        current_key,
        proposed_key,
    ):
        return False
    if not _ranker_preserves_charged_single_ring_parent_over_neutral_shortcut(
        current,
        proposed,
        current_key,
        proposed_key,
    ):
        return False
    if not _ranker_preserves_single_ring_parent_over_low_confidence_generated_partial_hydro(
        current,
        proposed,
        current_key,
        proposed_key,
    ):
        return False
    if not _ranker_preserves_large_fused_parent_over_single_ring_decomposition(current, proposed):
        return False
    if not _ranker_preserves_earlier_aromatic_single_ring_parent(current, proposed):
        return False
    if not _ranker_preserves_equivalent_single_ring_parent_choice(current, proposed, current_key, proposed_key):
        return False
    if not _ranker_preserves_oxindol_parent_over_indolin_tautomer(current, proposed):
        return False
    if not _ranker_preserves_preferred_acyclic_parent_over_whole_neutral_amine(current, proposed):
        return False
    if not _ranker_preserves_chain_dominant_acyclic_parent_over_functional_bicyclic(current, proposed):
        return False
    if not _ranker_preserves_compact_bicyclic_parent(current, proposed):
        return False
    if not _ranker_preserves_generic_bicyclic_parent_over_generated_fused_shortcut(
        current,
        proposed,
        current_key,
        proposed_key,
    ):
        return False
    if not _ranker_preserves_complete_single_ring_parent_over_acyclic_skeleton(current, proposed):
        return False
    if not _ranker_preserves_relative_stereo_parent_over_carbocycle_switch(
        current,
        proposed,
        current_key,
        proposed_key,
    ):
        return False
    if not _ranker_preserves_complete_senior_functional_parent(current, proposed):
        return False
    if not _ranker_preserves_acyclic_parent_for_generated_fused(current, proposed):
        return False
    if not _ranker_preserves_acyclic_parent_for_compact_fused_shortcut(current, proposed):
        return False
    if not _ranker_preserves_early_kekulized_polycycle_parent(current, proposed):
        return False
    if not _ranker_preserves_small_generated_fused_orientation(
        current,
        proposed,
        current_facts,
        proposed_facts,
    ):
        return False
    if not _ranker_preserves_exocyclic_imino_generated_fused_orientation(current, proposed):
        return False
    if not _ranker_preserves_terminal_hetero_generated_fused_parent(current, proposed):
        return False
    if not _ranker_preserves_generated_fused_same_parent_orientation(current, proposed):
        return False
    if not _ranker_promotes_only_first_generated_indazole_h_orientation(proposed):
        return False
    if not _ranker_promotes_only_first_generated_charged_imidazopyridine_orientation(proposed):
        return False
    if not _ranker_preserves_generated_indazole_h_template_order(current, proposed):
        return False
    if not _ranker_preserves_generated_imidazopyridine_parent(current, proposed, current_key, proposed_key):
        return False
    if not _ranker_preserves_generated_hydro_pyridino_imidazole_parent(
        current,
        proposed,
        current_key,
        proposed_key,
    ):
        return False
    if not _ranker_preserves_retained_azabicyclo320_heptane_parent(
        current,
        proposed,
        current_key,
        proposed_key,
    ):
        return False
    if not _ranker_preserves_retained_benzocyclobutene_parent(
        current,
        proposed,
        current_key,
        proposed_key,
    ):
        return False
    if not _ranker_preserves_generated_hydro_pyrido45d_imidazole_parent(
        current,
        proposed,
        current_key,
        proposed_key,
    ):
        return False
    if not _ranker_preserves_clean_aromatic_parent_over_terminal_carbonyl_switch(current, proposed):
        return False
    if not _ranker_preserves_small_inorganic_retained_parent(current, proposed):
        return False
    if not _ranker_preserves_exact_retained_n_oxide_parent(current, proposed):
        return False
    if not _ranker_preserves_complete_carbodiimide_parent(current, proposed):
        return False
    if current_candidate.family == "phosphite-ester-parent" and proposed_candidate.family != current_candidate.family:
        return False
    if current_candidate.family not in _RANKER_PROTECTED_SENIOR_FUNCTIONAL_FAMILIES:
        return True
    if proposed_candidate.family not in _RANKER_SUFFIXLESS_GRAPH_PARENT_FAMILIES:
        return True
    if not candidate_has_complete_graph(current_candidate):
        return True
    if proposed_candidate.suffixes or proposed_candidate.charge is not None:
        return True
    if current_candidate.priority >= proposed_candidate.priority:
        return True
    if len(proposed_candidate.parent_atoms) <= len(current_candidate.parent_atoms):
        return True
    return False


def _ranker_preserves_small_inorganic_retained_parent(
    current: _PreparedCandidate,
    proposed: _PreparedCandidate,
) -> bool:
    if current.candidate.family != "small-inorganic-retained-parent":
        return True
    return proposed.candidate.family not in _RANKER_SUFFIXLESS_GRAPH_PARENT_FAMILIES


def _ranker_preserves_exact_retained_n_oxide_parent(
    current: _PreparedCandidate,
    proposed: _PreparedCandidate,
) -> bool:
    current_candidate = current.candidate
    proposed_candidate = proposed.candidate
    if current_candidate.family != "exact-retained-n-oxide-parent":
        return True
    if proposed_candidate.family not in {
        "fused-retained-parent",
        "generated-oxido-fused-parent",
        "oxido-fused-retained-parent",
    }:
        return True
    if current_candidate.priority >= proposed_candidate.priority:
        return True
    if not candidate_has_complete_graph(current_candidate):
        return True
    if current_candidate.charge is None or "oxide" not in current_candidate.charge.rendered_terms:
        return True
    if len(proposed_candidate.parent_atoms) >= len(current_candidate.parent_atoms):
        return True
    return False


def _ranker_preserves_complete_carbodiimide_parent(
    current: _PreparedCandidate,
    proposed: _PreparedCandidate,
) -> bool:
    current_candidate = current.candidate
    if current_candidate.family != "carbodiimide-retained-parent":
        return True
    if not current_candidate.parent_atoms or not candidate_has_complete_graph(current_candidate):
        return True
    return proposed.candidate.family == current_candidate.family


def _ranker_preserves_clean_aromatic_parent_over_terminal_carbonyl_switch(
    current: _PreparedCandidate,
    proposed: _PreparedCandidate,
) -> bool:
    current_candidate = current.candidate
    proposed_candidate = proposed.candidate
    if not _is_clean_aromatic_carbocycle_single_ring_candidate(current_candidate):
        return True
    if proposed_candidate.family != current_candidate.family:
        return True
    if not candidate_has_complete_graph(proposed_candidate) or not proposed_candidate.parent_atoms:
        return True
    if len(proposed_candidate.parent_atoms) != len(current_candidate.parent_atoms):
        return True
    if proposed_candidate.parent_aromatic_atom_count != current_candidate.parent_aromatic_atom_count:
        return True
    if proposed_candidate.parent_hetero_atom_count != 0:
        return True
    if proposed_candidate.parent_terminal_carbonyl_substituent_count <= 0:
        return True
    if proposed_candidate.parent_terminal_heterocarbon_substituent_count <= 0:
        return True
    if len(proposed.name) <= len(current.name) + 10:
        return True
    if proposed_candidate.charge is not None or proposed_candidate.suffixes:
        return True
    if candidate_safety_reasons(proposed_candidate):
        return True
    return False


def _ranker_preserves_early_retained_benzimidazole_parent(
    current: _PreparedCandidate,
    proposed: _PreparedCandidate,
) -> bool:
    current_candidate = current.candidate
    proposed_candidate = proposed.candidate
    if current_candidate.family != "fused-retained-parent":
        return True
    if current_candidate.template_name != "benzimidazole":
        return True
    if proposed_candidate.family != "generated-fused-parent":
        return True
    if not candidate_has_complete_graph(current_candidate):
        return True
    if not candidate_has_complete_graph(proposed_candidate):
        return True
    current_render_rank = current_candidate.render_rank
    proposed_render_rank = proposed_candidate.render_rank
    if current_render_rank is None or current_render_rank > 2:
        return True
    if proposed_render_rank is None or proposed_render_rank <= current_render_rank:
        return True
    if current_candidate.charge is not None or proposed_candidate.charge is not None:
        return True
    if current_candidate.suffixes or proposed_candidate.suffixes:
        return True
    if len(current_candidate.parent_atoms) != 9 or len(proposed_candidate.parent_atoms) != 9:
        return True
    if proposed_candidate.parent_hetero_atom_count < 3:
        return True
    if (
        proposed_candidate.parent_terminal_hetero_substituent_count < 1
        and proposed_candidate.parent_substituted_hetero_atom_count < 1
    ):
        return True
    return False


def _ranker_preserves_generated_imidazopyridine_parent(
    current: _PreparedCandidate,
    proposed: _PreparedCandidate,
    current_key: tuple,
    proposed_key: tuple,
) -> bool:
    current_candidate = current.candidate
    proposed_candidate = proposed.candidate
    if not _is_guarded_generated_imidazopyridine_parent(current_candidate):
        return True
    if proposed_candidate.family == current_candidate.family and (
        proposed_candidate.template_name in _GENERATED_IMIDAZOPYRIDINE_PARENT_BONUS_TEMPLATES
    ):
        return True
    current_penalty = int(current_key[0]) if current_key else current.stability_penalty
    proposed_penalty = int(proposed_key[0]) if proposed_key else proposed.stability_penalty
    if current_penalty >= proposed_penalty:
        return True
    return False


def _is_guarded_generated_imidazopyridine_parent(candidate: NameCandidate) -> bool:
    if candidate.family != "generated-fused-parent":
        return False
    if candidate.template_name not in _GENERATED_IMIDAZOPYRIDINE_PARENT_BONUS_TEMPLATES:
        return False
    if candidate.source_family != "generated-fused":
        return False
    if candidate.source_file != "generated_fused_template_index.json":
        return False
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return False
    if len(candidate.parent_atoms) != 9:
        return False
    if candidate.parent_hetero_atom_count != 3:
        return False
    if candidate.parent_external_nitrogen_substituent_count > 1:
        return False
    if candidate.parent_external_oxygen_substituent_count:
        return False
    if candidate.suffixes or candidate.charge is not None:
        return False
    if candidate_safety_reasons(candidate):
        return False
    return candidate.render_rank is not None and candidate.render_rank <= 2


def _ranker_preserves_generated_hydro_pyridino_imidazole_parent(
    current: _PreparedCandidate,
    proposed: _PreparedCandidate,
    current_key: tuple,
    proposed_key: tuple,
) -> bool:
    current_candidate = current.candidate
    proposed_candidate = proposed.candidate
    if not _is_guarded_generated_hydro_pyridino_imidazole_parent(current_candidate):
        return True
    if (
        proposed_candidate.family == current_candidate.family
        and proposed_candidate.template_name == current_candidate.template_name
    ):
        return True
    current_penalty = int(current_key[0]) if current_key else current.stability_penalty
    proposed_penalty = int(proposed_key[0]) if proposed_key else proposed.stability_penalty
    if current_penalty >= proposed_penalty:
        return True
    return False


def _is_guarded_generated_hydro_pyridino_imidazole_parent(candidate: NameCandidate) -> bool:
    if candidate.family != "generated-hydro-fused-parent":
        return False
    if candidate.template_name != "pyridino[4,3-d]imidazole":
        return False
    if candidate.source_family != "generated-fused":
        return False
    if candidate.source_file != "generated_fused_template_index.json":
        return False
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return False
    if len(candidate.parent_atoms) != 9:
        return False
    if candidate.suffixes or candidate.charge is not None:
        return False
    if candidate_safety_reasons(candidate):
        return False
    return True


def _ranker_preserves_retained_azabicyclo320_heptane_parent(
    current: _PreparedCandidate,
    proposed: _PreparedCandidate,
    current_key: tuple,
    proposed_key: tuple,
) -> bool:
    current_candidate = current.candidate
    proposed_candidate = proposed.candidate
    if not _is_guarded_retained_azabicyclo320_heptane_parent(current_candidate):
        return True
    if (
        proposed_candidate.family == current_candidate.family
        and proposed_candidate.template_name == current_candidate.template_name
    ):
        return True
    current_penalty = int(current_key[0]) if current_key else current.stability_penalty
    proposed_penalty = int(proposed_key[0]) if proposed_key else proposed.stability_penalty
    if current_penalty >= proposed_penalty:
        return True
    return False


def _is_guarded_retained_azabicyclo320_heptane_parent(candidate: NameCandidate) -> bool:
    if candidate.family != "retained-topology-replacement-fused-parent":
        return False
    if candidate.template_name != "1-azabicyclo[3.2.0]heptane":
        return False
    if candidate.source_family != "static-retained":
        return False
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return False
    if len(candidate.parent_atoms) != 7:
        return False
    if candidate.parent_aromatic_atom_count:
        return False
    if candidate.suffixes or candidate.charge is not None:
        return False
    if candidate_safety_reasons(candidate):
        return False
    render_rank = candidate.render_rank or 0
    return 1 <= render_rank <= 2


def _ranker_preserves_retained_benzocyclobutene_parent(
    current: _PreparedCandidate,
    proposed: _PreparedCandidate,
    current_key: tuple,
    proposed_key: tuple,
) -> bool:
    current_candidate = current.candidate
    proposed_candidate = proposed.candidate
    if not _is_guarded_retained_benzocyclobutene_parent(current_candidate):
        return True
    if (
        proposed_candidate.family == current_candidate.family
        and proposed_candidate.template_name == current_candidate.template_name
    ):
        return True
    current_penalty = int(current_key[0]) if current_key else current.stability_penalty
    proposed_penalty = int(proposed_key[0]) if proposed_key else proposed.stability_penalty
    if current_penalty >= proposed_penalty:
        return True
    return False


def _is_guarded_retained_benzocyclobutene_parent(candidate: NameCandidate) -> bool:
    if candidate.family != "fused-retained-parent":
        return False
    if candidate.template_name != "benzocyclobutene":
        return False
    if candidate.source_family != "simpleGroups:simpleGroup:simpleGroup":
        return False
    if candidate.source_file != "simpleGroups.xml":
        return False
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return False
    if len(candidate.parent_atoms) != 8:
        return False
    if candidate.suffixes or candidate.charge is not None:
        return False
    if candidate_safety_reasons(candidate):
        return False
    render_rank = candidate.render_rank or 0
    return 1 <= render_rank <= 4


def _ranker_preserves_generated_hydro_pyrido45d_imidazole_parent(
    current: _PreparedCandidate,
    proposed: _PreparedCandidate,
    current_key: tuple,
    proposed_key: tuple,
) -> bool:
    current_candidate = current.candidate
    proposed_candidate = proposed.candidate
    if not _is_guarded_generated_hydro_pyrido45d_imidazole_parent(current_candidate):
        return True
    if (
        proposed_candidate.family == current_candidate.family
        and proposed_candidate.template_name == current_candidate.template_name
    ):
        return True
    current_penalty = int(current_key[0]) if current_key else current.stability_penalty
    proposed_penalty = int(proposed_key[0]) if proposed_key else proposed.stability_penalty
    if current_penalty >= proposed_penalty:
        return True
    return False


def _is_guarded_generated_hydro_pyrido45d_imidazole_parent(candidate: NameCandidate) -> bool:
    if candidate.family != "generated-hydro-fused-parent":
        return False
    if candidate.template_name != "pyrido[4,5-d]imidazole":
        return False
    if candidate.source_family != "generated-fused":
        return False
    if candidate.source_file != "generated_fused_template_index.json":
        return False
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return False
    if len(candidate.parent_atoms) != 9:
        return False
    if candidate.suffixes or candidate.charge is not None:
        return False
    if candidate_safety_reasons(candidate):
        return False
    return True


_GENERATED_INDAZOLE_H_ORIENTATION_TEMPLATES = frozenset(
    {
        "benzo[1,2-c]pyrazole",
        "benzo[1,2-d]pyrazole",
    }
)


def _ranker_preserves_generated_indazole_h_template_order(
    current: _PreparedCandidate,
    proposed: _PreparedCandidate,
) -> bool:
    current_candidate = current.candidate
    proposed_candidate = proposed.candidate
    if current_candidate.family != "generated-fused-parent":
        return True
    if proposed_candidate.family != "generated-fused-parent":
        return True
    if current_candidate.template_name not in _GENERATED_INDAZOLE_H_ORIENTATION_TEMPLATES:
        return True
    if proposed_candidate.template_name not in _GENERATED_INDAZOLE_H_ORIENTATION_TEMPLATES:
        return True
    if current_candidate.template_name == proposed_candidate.template_name:
        return True
    if current_candidate.source_family != "generated-fused":
        return True
    if proposed_candidate.source_family != "generated-fused":
        return True
    if current_candidate.source_file != "generated_fused_template_index.json":
        return True
    if proposed_candidate.source_file != "generated_fused_template_index.json":
        return True
    if not candidate_has_complete_graph(current_candidate):
        return True
    if not candidate_has_complete_graph(proposed_candidate):
        return True
    current_parent_atoms = frozenset(current_candidate.parent_atoms)
    if not current_parent_atoms:
        return True
    if current_parent_atoms != frozenset(proposed_candidate.parent_atoms):
        return True
    return False


def _ranker_promotes_only_first_generated_indazole_h_orientation(proposed: _PreparedCandidate) -> bool:
    candidate = proposed.candidate
    if candidate.family != "generated-fused-parent":
        return True
    if candidate.template_name not in _GENERATED_INDAZOLE_H_ORIENTATION_TEMPLATES:
        return True
    if candidate.source_family != "generated-fused":
        return True
    if candidate.source_file != "generated_fused_template_index.json":
        return True
    if not candidate_has_complete_graph(candidate):
        return True
    return candidate.render_rank == 1


def _ranker_promotes_only_first_generated_charged_imidazopyridine_orientation(
    proposed: _PreparedCandidate,
) -> bool:
    candidate = proposed.candidate
    if candidate.family != "generated-fused-parent":
        return True
    if candidate.source_family != "generated-fused":
        return True
    if candidate.source_file != "generated_fused_template_index.json":
        return True
    if not candidate_has_complete_graph(candidate):
        return True
    template_name = candidate.template_name or ""
    if re.match(r"^imidazo\[[^\]]+\]pyridine$", template_name):
        if candidate.charge is not None and candidate.charge.formal_charge > 0:
            return candidate.render_rank == 1
        return True
    return True


def _ranker_preserves_rank1_retained_fused_parent(
    current: _PreparedCandidate,
    proposed: _PreparedCandidate,
) -> bool:
    current_candidate = current.candidate
    proposed_candidate = proposed.candidate
    if current_candidate.family != "fused-retained-parent":
        return True
    if proposed_candidate.family != "generated-fused-parent":
        return True
    if not candidate_has_complete_graph(current_candidate):
        return True
    if not candidate_has_complete_graph(proposed_candidate):
        return True
    current_render_rank = current_candidate.render_rank
    proposed_render_rank = proposed_candidate.render_rank
    if current_render_rank != 1:
        return True
    if proposed_render_rank is None or proposed_render_rank <= current_render_rank:
        return True
    if current_candidate.charge is not None or proposed_candidate.charge is not None:
        return True
    if current_candidate.suffixes or proposed_candidate.suffixes:
        return True
    current_parent_count = len(current_candidate.parent_atoms)
    proposed_parent_count = len(proposed_candidate.parent_atoms)
    if current_parent_count < 9:
        return True
    if proposed_parent_count > current_parent_count:
        return True
    if proposed_candidate.parent_hetero_atom_count < 2:
        return True
    if current.stability_penalty > proposed.stability_penalty + 80:
        return True
    return False


def _ranker_preserves_systematic_candidate_over_penalized_retained_shortcut(
    current: _PreparedCandidate,
    proposed: _PreparedCandidate,
    current_key: tuple,
    proposed_key: tuple,
) -> bool:
    current_candidate = current.candidate
    proposed_candidate = proposed.candidate
    if current_candidate.family not in _CONTEXTUAL_SYSTEMATIC_ALTERNATIVE_FAMILIES:
        return True
    if proposed_candidate.family not in {"amino-acid-retained-parent", "whole-molecule-retained-parent"}:
        return True
    if not candidate_has_complete_graph(current_candidate):
        return True
    if not candidate_has_complete_graph(proposed_candidate):
        return True
    if proposed_candidate.suffixes or proposed_candidate.charge is not None:
        return True
    if current_candidate.charge is not None:
        return True
    if proposed_candidate.priority >= current_candidate.priority:
        return True
    current_penalty = int(current_key[0]) if current_key else current.stability_penalty
    proposed_penalty = int(proposed_key[0]) if proposed_key else proposed.stability_penalty
    if proposed_penalty < current_penalty + 100:
        return True
    return False


def _ranker_preserves_generated_fused_parent_over_penalized_retained_shortcut(
    current: _PreparedCandidate,
    proposed: _PreparedCandidate,
    current_key: tuple,
    proposed_key: tuple,
) -> bool:
    current_candidate = current.candidate
    proposed_candidate = proposed.candidate
    if current_candidate.family != "generated-fused-parent":
        return True
    if proposed_candidate.family not in {
        "fused-retained-parent",
        "hydro-fused-retained-parent",
        "oxido-fused-retained-parent",
        "retained-topology-replacement-fused-parent",
    }:
        return True
    if not candidate_has_complete_graph(current_candidate):
        return True
    if not candidate_has_complete_graph(proposed_candidate):
        return True
    if current_candidate.charge is not None or proposed_candidate.charge is not None:
        return True
    current_parent_count = len(current_candidate.parent_atoms)
    proposed_parent_count = len(proposed_candidate.parent_atoms)
    if current_parent_count < 8:
        return True
    if proposed_parent_count < current_parent_count - 1 or proposed_parent_count > current_parent_count + 2:
        return True
    current_render_rank = current_candidate.render_rank
    if current_render_rank is None or current_render_rank > 3:
        return True
    if _is_unanchored_indicated_h_generated_purine_candidate(current_candidate) and _is_matching_retained_purin_candidate(
        proposed_candidate,
        current_candidate,
    ):
        return True
    current_penalty = int(current_key[0]) if current_key else current.stability_penalty
    proposed_penalty = int(proposed_key[0]) if proposed_key else proposed.stability_penalty
    if proposed_penalty < current_penalty + 180:
        return True
    if len(current.name) > 260 or len(current.name) - len(proposed.name) > 150:
        return True
    return False


def _ranker_preserves_single_ring_parent_over_external_nitrogen_generated_fused(
    current: _PreparedCandidate,
    proposed: _PreparedCandidate,
    current_key: tuple,
    proposed_key: tuple,
) -> bool:
    current_candidate = current.candidate
    proposed_candidate = proposed.candidate
    if current_candidate.family != "single-ring-parent":
        return True
    if not _is_external_nitrogen_generated_fused_shortcut_candidate(proposed_candidate):
        return True
    if not _is_clean_aromatic_single_ring_parent_candidate(current_candidate):
        return True
    current_penalty = int(current_key[0]) if current_key else current.stability_penalty
    proposed_penalty = int(proposed_key[0]) if proposed_key else proposed.stability_penalty
    if proposed_penalty <= current_penalty:
        return True
    if len(current.name) > _DEFAULT_MAX_NAME_LENGTH:
        return True
    if len(current.name) - len(proposed.name) > 90:
        return True
    return False


def _ranker_preserves_rank1_hetero_generated_fused_parent_over_acyclic_skeleton(
    current: _PreparedCandidate,
    proposed: _PreparedCandidate,
) -> bool:
    current_candidate = current.candidate
    proposed_candidate = proposed.candidate
    if current_candidate.family != "generated-fused-parent":
        return True
    if proposed_candidate.family not in _RANKER_ACYCLIC_SKELETON_WITH_RING_SUBSTITUENT_FAMILIES:
        return True
    if current_candidate.source_family != "generated-fused":
        return True
    if not current_candidate.parent_atoms or not candidate_has_complete_graph(current_candidate):
        return True
    if not proposed_candidate.parent_atoms or not candidate_has_complete_graph(proposed_candidate):
        return True
    if current_candidate.render_rank != 1:
        return True
    if len(current_candidate.parent_atoms) < 9:
        return True
    if len(proposed_candidate.parent_atoms) > len(current_candidate.parent_atoms) + 12:
        return True
    if current_candidate.suffixes or proposed_candidate.suffixes:
        return True
    if proposed_candidate.charge is not None:
        return True
    if candidate_safety_reasons(current_candidate) or candidate_safety_reasons(proposed_candidate):
        return True
    current_charge = current_candidate.charge.formal_charge if current_candidate.charge is not None else 0
    if current_charge == 0 and current_candidate.parent_hetero_atom_count != 3:
        return True
    return False


def _ranker_preserves_substituent_ring_parent_over_compact_chalcogen_suffix(
    current: _PreparedCandidate,
    proposed: _PreparedCandidate,
    current_key: tuple,
    proposed_key: tuple,
) -> bool:
    current_candidate = current.candidate
    proposed_candidate = proposed.candidate
    if current_candidate.family != "single-ring-parent" or proposed_candidate.family != "single-ring-parent":
        return True
    if not _is_compact_osn_chalcogen_suffix_parent(proposed_candidate):
        return True
    if current_candidate.suffixes or current_candidate.charge is not None:
        return True
    if not current_candidate.parent_atoms or not candidate_has_complete_graph(current_candidate):
        return True
    if current_candidate.parent_atoms & proposed_candidate.parent_atoms:
        return True
    current_parent_count = len(current_candidate.parent_atoms)
    if current_parent_count < 3 or current_parent_count > 8:
        return True
    if (
        current_candidate.parent_hetero_atom_count > 1
        and current_candidate.parent_aromatic_atom_count != current_parent_count
    ):
        return True
    current_penalty = int(current_key[0]) if current_key else current.stability_penalty
    proposed_penalty = int(proposed_key[0]) if proposed_key else proposed.stability_penalty
    if proposed_penalty < current_penalty + 80:
        return True
    if len(current.name) > 200 or len(current.name) - len(proposed.name) > 120:
        return True
    return False


def _ranker_preserves_charged_ring_skeleton_parent_over_single_ring_shortcut(
    current: _PreparedCandidate,
    proposed: _PreparedCandidate,
    current_key: tuple,
    proposed_key: tuple,
) -> bool:
    current_candidate = current.candidate
    proposed_candidate = proposed.candidate
    if current_candidate.family != "charged-ring-acyclic-skeleton-with-ring-substituents":
        return True
    if proposed_candidate.family not in {"single-ring-parent", "positive-single-ring-parent"}:
        return True
    if not _is_simple_charged_aromatic_single_ring_parent(proposed_candidate):
        return True
    if not current_candidate.parent_atoms or not candidate_has_complete_graph(current_candidate):
        return True
    current_penalty = int(current_key[0]) if current_key else current.stability_penalty
    proposed_penalty = int(proposed_key[0]) if proposed_key else proposed.stability_penalty
    if proposed_penalty < current_penalty + 80:
        return True
    if len(current.name) > 180 or len(current.name) - len(proposed.name) > 90:
        return True
    return False


def _ranker_preserves_charged_single_ring_parent_over_neutral_shortcut(
    current: _PreparedCandidate,
    proposed: _PreparedCandidate,
    current_key: tuple,
    proposed_key: tuple,
) -> bool:
    current_candidate = current.candidate
    proposed_candidate = proposed.candidate
    if current_candidate.family not in {"single-ring-parent", "positive-single-ring-parent"}:
        return True
    if current_candidate.charge is None or not current_candidate.charge.charged_atoms:
        return True
    if not current_candidate.parent_atoms or not candidate_has_complete_graph(current_candidate):
        return True
    if candidate_safety_reasons(current_candidate):
        return True
    if _is_exact_retained_charge_distribution_shortcut(proposed_candidate):
        if not proposed_candidate.parent_atoms or not candidate_has_complete_graph(proposed_candidate):
            return True
        if not current_candidate.parent_atoms.issubset(proposed_candidate.parent_atoms):
            return True
        if len(current_candidate.parent_atoms) < 5 or len(current_candidate.parent_atoms) > 6:
            return True
        if len(current.name) > 180 or len(current.name) - len(proposed.name) > 90:
            return True
        return proposed_key <= current_key
    if proposed_candidate.family in _CHARGED_SINGLE_RING_PARENT_COMPETITOR_FAMILIES:
        if proposed_candidate.charge is not None and proposed_candidate.charge.charged_atoms:
            return True
        if not proposed_candidate.parent_atoms or not candidate_has_complete_graph(proposed_candidate):
            return True
        if len(current.name) > 220 or len(current.name) - len(proposed.name) > 130:
            return True
        return proposed_key <= current_key
    if (
        proposed_candidate.family == "charged-retained-parent"
        and _is_six_membered_ring_dication_single_ring_parent(current_candidate)
    ):
        if not _is_charged_retained_parent_candidate(proposed_candidate):
            return True
        if len(current.name) > 180 or len(current.name) - len(proposed.name) > 90:
            return True
        return proposed_key <= current_key
    if proposed_candidate.family not in {"single-ring-parent", "positive-single-ring-parent"}:
        return True
    if proposed_candidate.charge is not None:
        return True
    if not proposed_candidate.parent_atoms or not candidate_has_complete_graph(proposed_candidate):
        return True
    if len(current_candidate.parent_atoms) != len(proposed_candidate.parent_atoms):
        return True
    if len(current.name) > 180 or len(current.name) - len(proposed.name) > 90:
        return True
    return proposed_key <= current_key


def _is_exact_retained_charge_distribution_shortcut(candidate: NameCandidate) -> bool:
    if candidate.family != "exact-functional-retained-parent":
        return False
    if candidate.charge is not None or candidate.suffixes:
        return False
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return False
    return candidate.parent_abs_formal_charge_sum > candidate.template_abs_formal_charge_sum


def _ranker_preserves_single_ring_parent_over_low_confidence_generated_partial_hydro(
    current: _PreparedCandidate,
    proposed: _PreparedCandidate,
    current_key: tuple,
    proposed_key: tuple,
) -> bool:
    current_candidate = current.candidate
    proposed_candidate = proposed.candidate
    if current_candidate.family != "single-ring-parent":
        return True
    if proposed_candidate.family != "generated-partial-hydro-fused-parent":
        return True
    if proposed_candidate.source_family != "generated-fused":
        return True
    if proposed_candidate.source_file != "generated_fused_template_index.json":
        return True
    if not proposed_candidate.template_name:
        return True
    if current_candidate.template_name:
        return True
    if not current_candidate.parent_atoms or not candidate_has_complete_graph(current_candidate):
        return True
    if not proposed_candidate.parent_atoms or not candidate_has_complete_graph(proposed_candidate):
        return True
    if current_candidate.charge is not None or proposed_candidate.charge is not None:
        return True
    if current_candidate.suffixes or proposed_candidate.suffixes:
        return True
    if len(current_candidate.parent_atoms) < 4:
        return True
    if len(proposed_candidate.parent_atoms) < 8:
        return True
    if len(proposed_candidate.parent_atoms) < len(current_candidate.parent_atoms):
        return True
    proposed_render_rank = proposed_candidate.render_rank
    if proposed_render_rank is None or proposed_render_rank > 4:
        return True
    if len(current.name) > 280:
        return True
    if len(current.name) - len(proposed.name) > 170:
        return True
    if proposed_key <= current_key:
        return True
    return False


def _ranker_preserves_large_fused_parent_over_single_ring_decomposition(
    current: _PreparedCandidate,
    proposed: _PreparedCandidate,
) -> bool:
    current_candidate = current.candidate
    proposed_candidate = proposed.candidate
    if current_candidate.family not in {
        "fused-retained-parent",
        "generated-fused-parent",
        "generated-hydro-fused-parent",
        "generated-partial-hydro-fused-parent",
        "hydro-fused-retained-parent",
        "kekulized-generic-polycyclic-parent",
        "retained-topology-replacement-fused-parent",
        "skeletal-replacement-fused-parent",
    }:
        return True
    if proposed_candidate.family not in {"single-ring-parent", "positive-single-ring-parent"}:
        return True
    if not candidate_has_complete_graph(current_candidate):
        return True
    if not candidate_has_complete_graph(proposed_candidate):
        return True
    if current_candidate.charge is not None or proposed_candidate.charge is not None:
        return True
    if current_candidate.suffixes or proposed_candidate.suffixes:
        return True
    current_parent_count = len(current_candidate.parent_atoms)
    proposed_parent_count = len(proposed_candidate.parent_atoms)
    if current_parent_count < 9:
        return True
    if proposed_parent_count > 6:
        return True
    if current_parent_count < proposed_parent_count + 5:
        return True
    current_render_rank = current_candidate.render_rank
    if current_render_rank is None or current_render_rank > 3:
        return True
    if len(current.name) > 180:
        return True
    if current.stability_penalty > proposed.stability_penalty + 80:
        return True
    return False


def _ranker_preserves_small_generated_fused_orientation(
    current: _PreparedCandidate,
    proposed: _PreparedCandidate,
    current_facts: dict[str, object],
    proposed_facts: dict[str, object],
) -> bool:
    current_candidate = current.candidate
    proposed_candidate = proposed.candidate
    if current_candidate.family != "generated-fused-parent":
        return True
    if proposed_candidate.family != current_candidate.family:
        return True
    if current_candidate.source_family != "generated-fused":
        return True
    if proposed_candidate.source_family != current_candidate.source_family:
        return True
    if current_candidate.source_file != "generated_fused_template_index.json":
        return True
    if proposed_candidate.source_file != current_candidate.source_file:
        return True
    if not candidate_has_complete_graph(current_candidate):
        return True
    if not candidate_has_complete_graph(proposed_candidate):
        return True
    current_parent_atom_count = len(current_candidate.parent_atoms)
    if current_parent_atom_count == 0 or current_parent_atom_count != len(proposed_candidate.parent_atoms):
        return True
    if current_parent_atom_count > 9:
        return True
    if current_candidate.parent_hetero_atom_count > 3:
        return True
    if proposed_candidate.parent_hetero_atom_count > 3:
        return True
    if current_candidate.priority != proposed_candidate.priority:
        return True
    current_render_rank = current_candidate.render_rank
    proposed_render_rank = proposed_candidate.render_rank
    if current_render_rank is None or proposed_render_rank is None:
        return True
    if proposed_render_rank <= current_render_rank:
        return True
    if current_candidate.suffixes or proposed_candidate.suffixes:
        return True
    if current_candidate.charge is not None or proposed_candidate.charge is not None:
        return True
    if candidate_safety_reasons(current_candidate) or candidate_safety_reasons(proposed_candidate):
        return True
    if float(current_facts.get("candidate_count_in_row", 99.0)) > 3.0:
        return True
    if float(proposed_facts.get("candidate_count_in_row", 99.0)) > 3.0:
        return True
    if float(current_facts.get("parent_atom_delta_min", -1.0)) != 0.0:
        return True
    if float(proposed_facts.get("parent_atom_delta_min", -1.0)) != 0.0:
        return True
    return False


def _ranker_preserves_exocyclic_imino_generated_fused_orientation(
    current: _PreparedCandidate,
    proposed: _PreparedCandidate,
) -> bool:
    current_candidate = current.candidate
    proposed_candidate = proposed.candidate
    if current_candidate.family != "generated-fused-parent":
        return True
    if proposed_candidate.family != current_candidate.family:
        return True
    if current_candidate.source_family != "generated-fused":
        return True
    if proposed_candidate.source_family != current_candidate.source_family:
        return True
    if current_candidate.source_file != "generated_fused_template_index.json":
        return True
    if proposed_candidate.source_file != current_candidate.source_file:
        return True
    if not candidate_has_complete_graph(current_candidate):
        return True
    if not candidate_has_complete_graph(proposed_candidate):
        return True
    if current_candidate.parent_atoms != proposed_candidate.parent_atoms:
        return True
    if len(current_candidate.parent_atoms) != 9:
        return True
    if current_candidate.parent_hetero_atom_count != 4 or proposed_candidate.parent_hetero_atom_count != 4:
        return True
    if (
        current_candidate.parent_external_nitrogen_substituent_count != 1
        or proposed_candidate.parent_external_nitrogen_substituent_count != 1
    ):
        return True
    if current_candidate.parent_terminal_hetero_substituent_count:
        return True
    if proposed_candidate.parent_terminal_hetero_substituent_count:
        return True
    if current_candidate.parent_external_oxygen_substituent_count:
        return True
    if proposed_candidate.parent_external_oxygen_substituent_count:
        return True
    if current_candidate.suffixes or proposed_candidate.suffixes:
        return True
    if current_candidate.charge is not None or proposed_candidate.charge is not None:
        return True
    if candidate_safety_reasons(current_candidate) or candidate_safety_reasons(proposed_candidate):
        return True
    current_render_rank = current_candidate.render_rank
    proposed_render_rank = proposed_candidate.render_rank
    if current_render_rank is None or proposed_render_rank is None:
        return True
    if current_render_rank >= proposed_render_rank:
        return True
    if not _is_exocyclic_imino_preferred_generated_fused_template(current_candidate.template_name):
        return True
    if not _is_exocyclic_imino_disfavored_generated_fused_template(proposed_candidate.template_name):
        return True
    return False


def _ranker_preserves_terminal_hetero_generated_fused_parent(
    current: _PreparedCandidate,
    proposed: _PreparedCandidate,
) -> bool:
    current_candidate = current.candidate
    proposed_candidate = proposed.candidate
    if current_candidate.family != "generated-fused-parent":
        return True
    if proposed_candidate.family != current_candidate.family:
        return True
    if current_candidate.source_family != "generated-fused":
        return True
    if proposed_candidate.source_family != current_candidate.source_family:
        return True
    if current_candidate.source_file != "generated_fused_template_index.json":
        return True
    if proposed_candidate.source_file != current_candidate.source_file:
        return True
    if not current_candidate.parent_atoms or not candidate_has_complete_graph(current_candidate):
        return True
    if not proposed_candidate.parent_atoms or not candidate_has_complete_graph(proposed_candidate):
        return True
    current_parent_count = len(current_candidate.parent_atoms)
    if current_parent_count < 8 or current_parent_count != len(proposed_candidate.parent_atoms):
        return True
    if not current_candidate.ring_system_atoms or not proposed_candidate.ring_system_atoms:
        return True
    if len(current_candidate.ring_system_atoms) != current_parent_count:
        return True
    if len(proposed_candidate.ring_system_atoms) != current_parent_count:
        return True
    if current_candidate.parent_aromatic_atom_count != current_parent_count:
        return True
    if proposed_candidate.parent_aromatic_atom_count != current_parent_count:
        return True
    if current_candidate.parent_hetero_atom_count < 2:
        return True
    if proposed_candidate.parent_hetero_atom_count != current_candidate.parent_hetero_atom_count:
        return True
    current_rank = current_candidate.render_rank or 0
    proposed_rank = proposed_candidate.render_rank or 0
    if current_rank < 1 or proposed_rank <= current_rank:
        return True
    if current_candidate.parent_terminal_hetero_substituent_count <= proposed_candidate.parent_terminal_hetero_substituent_count:
        return True
    if current_candidate.suffixes or proposed_candidate.suffixes:
        return True
    if current_candidate.charge is not None or proposed_candidate.charge is not None:
        return True
    if candidate_safety_reasons(current_candidate) or candidate_safety_reasons(proposed_candidate):
        return True
    return False


def _ranker_preserves_generated_fused_same_parent_orientation(
    current: _PreparedCandidate,
    proposed: _PreparedCandidate,
) -> bool:
    current_candidate = current.candidate
    proposed_candidate = proposed.candidate
    if current_candidate.family != "generated-fused-parent":
        return True
    if proposed_candidate.family != current_candidate.family:
        return True
    if current_candidate.source_family != "generated-fused":
        return True
    if proposed_candidate.source_family != current_candidate.source_family:
        return True
    if current_candidate.source_file != "generated_fused_template_index.json":
        return True
    if proposed_candidate.source_file != current_candidate.source_file:
        return True
    if not current_candidate.parent_atoms or not candidate_has_complete_graph(current_candidate):
        return True
    if not proposed_candidate.parent_atoms or not candidate_has_complete_graph(proposed_candidate):
        return True
    if current_candidate.parent_atoms != proposed_candidate.parent_atoms:
        return True
    if len(current_candidate.parent_atoms) < 9:
        return True
    if current_candidate.suffixes or proposed_candidate.suffixes:
        return True
    if current_candidate.charge is not None or proposed_candidate.charge is not None:
        return True
    if candidate_safety_reasons(current_candidate) or candidate_safety_reasons(proposed_candidate):
        return True
    return False


def _is_exocyclic_imino_preferred_generated_fused_template(template_name: str) -> bool:
    return template_name == "imidazo[4,5-d]pyrimidine" or re.match(
        r"^pyrazolo\[[^\]]+\]pyrimidine$",
        template_name,
    )


def _is_exocyclic_imino_disfavored_generated_fused_template(template_name: str) -> bool:
    return bool(re.match(r"^(?:pyrimido|pyrimidino)\[[^\]]+\](?:imidazole|pyrazole)$", template_name))


def _ranker_preserves_early_kekulized_polycycle_parent(
    current: _PreparedCandidate,
    proposed: _PreparedCandidate,
) -> bool:
    current_candidate = current.candidate
    proposed_candidate = proposed.candidate
    if current_candidate.family != "kekulized-generic-polycyclic-parent":
        return True
    if proposed_candidate.family != current_candidate.family:
        return True
    if not current_candidate.parent_atoms or not candidate_has_complete_graph(current_candidate):
        return True
    if not proposed_candidate.parent_atoms or not candidate_has_complete_graph(proposed_candidate):
        return True
    current_rank = current_candidate.render_rank or 0
    proposed_rank = proposed_candidate.render_rank or 0
    if (
        current_candidate.parent_atoms == proposed_candidate.parent_atoms
        and current_rank >= 1
        and proposed_rank > current_rank
        and not candidate_safety_reasons(current_candidate)
        and not candidate_safety_reasons(proposed_candidate)
        and not current_candidate.suffixes
        and not proposed_candidate.suffixes
        and current_candidate.charge is None
        and proposed_candidate.charge is None
    ):
        return False
    current_parent_count = len(current_candidate.parent_atoms)
    if current_parent_count < 18 or current_parent_count > 19:
        return True
    if len(proposed_candidate.parent_atoms) != current_parent_count:
        return True
    if current_candidate.parent_aromatic_atom_count or proposed_candidate.parent_aromatic_atom_count:
        return True
    if current_candidate.parent_hetero_atom_count or proposed_candidate.parent_hetero_atom_count:
        return True
    if current_rank < 1 or current_rank > 3:
        return True
    if proposed_rank < current_rank + 4:
        return True
    if current_candidate.suffixes or proposed_candidate.suffixes:
        return True
    if current_candidate.charge is not None or proposed_candidate.charge is not None:
        return True
    if current.stability_penalty > proposed.stability_penalty + 80:
        return True
    return False


def _ranker_preserves_acyclic_parent_for_generated_fused(
    current: _PreparedCandidate,
    proposed: _PreparedCandidate,
) -> bool:
    current_candidate = current.candidate
    proposed_candidate = proposed.candidate
    if current_candidate.family not in _RANKER_ACYCLIC_SKELETON_WITH_RING_SUBSTITUENT_FAMILIES:
        return True
    if proposed_candidate.family not in {"generated-fused-parent", "generated-hydro-fused-parent"}:
        return True
    if not current_candidate.parent_atoms or not candidate_has_complete_graph(current_candidate):
        return True
    if not proposed_candidate.parent_atoms or not candidate_has_complete_graph(proposed_candidate):
        return True
    proposed_parent_count = len(proposed_candidate.parent_atoms)
    if proposed_candidate.family == "generated-hydro-fused-parent":
        if proposed_candidate.template_name not in _RANKER_ACYCLIC_GENERATED_HYDRO_FUSED_TEMPLATES:
            return True
        if proposed_parent_count != 10:
            return True
        if len(current_candidate.parent_atoms) < proposed_parent_count + 3:
            return True
        if current_candidate.suffixes or proposed_candidate.suffixes:
            return True
        if current_candidate.charge is not None or proposed_candidate.charge is not None:
            return True
        if current.stability_penalty > proposed.stability_penalty + 80:
            return True
        if len(current.name) > 180 or len(current.name) - len(proposed.name) > 40:
            return True
        return False
    if proposed_parent_count != 9:
        return True
    if proposed_candidate.parent_aromatic_atom_count != proposed_parent_count:
        return True
    if proposed_candidate.parent_hetero_atom_count != 2:
        return True
    if proposed_candidate.parent_external_oxygen_substituent_count:
        return True
    if proposed_candidate.parent_external_nitrogen_substituent_count:
        return True
    proposed_rank = proposed_candidate.render_rank or 0
    if proposed_rank < 2 or proposed_rank > 4:
        return True
    if len(current_candidate.parent_atoms) < proposed_parent_count + 3:
        return True
    if current_candidate.suffixes or proposed_candidate.suffixes:
        return True
    if current_candidate.charge is not None or proposed_candidate.charge is not None:
        return True
    if current.stability_penalty > proposed.stability_penalty + 80:
        return True
    return False


def _ranker_preserves_acyclic_parent_for_compact_fused_shortcut(
    current: _PreparedCandidate,
    proposed: _PreparedCandidate,
) -> bool:
    current_candidate = current.candidate
    proposed_candidate = proposed.candidate
    if current_candidate.family not in _RANKER_ACYCLIC_SKELETON_WITH_RING_SUBSTITUENT_FAMILIES:
        return True
    if proposed_candidate.family not in _RANKER_COMPACT_FUSED_SHORTCUT_FAMILIES:
        return True
    if not current_candidate.parent_atoms or not candidate_has_complete_graph(current_candidate):
        return True
    if not proposed_candidate.parent_atoms or not candidate_has_complete_graph(proposed_candidate):
        return True
    proposed_parent_count = len(proposed_candidate.parent_atoms)
    if proposed_parent_count != 10:
        return True
    if len(current_candidate.parent_atoms) < proposed_parent_count + 3:
        return True
    if current_candidate.suffixes or proposed_candidate.suffixes:
        return True
    if current_candidate.charge is not None or proposed_candidate.charge is not None:
        return True
    if candidate_safety_reasons(current_candidate) or candidate_safety_reasons(proposed_candidate):
        return True
    return False


def _ranker_preserves_complete_single_ring_parent_over_acyclic_skeleton(
    current: _PreparedCandidate,
    proposed: _PreparedCandidate,
) -> bool:
    current_candidate = current.candidate
    proposed_candidate = proposed.candidate
    if current_candidate.family != "single-ring-parent":
        return True
    if proposed_candidate.family not in _RANKER_ACYCLIC_SKELETON_WITH_RING_SUBSTITUENT_FAMILIES:
        return True
    if not current_candidate.parent_atoms or not candidate_has_complete_graph(current_candidate):
        return True
    if not proposed_candidate.parent_atoms or not candidate_has_complete_graph(proposed_candidate):
        return True
    current_parent_count = len(current_candidate.parent_atoms)
    proposed_parent_count = len(proposed_candidate.parent_atoms)
    if (
        current_parent_count == 6
        and proposed_parent_count >= 12
        and current_candidate.parent_aromatic_atom_count == current_parent_count
        and current_candidate.parent_hetero_atom_count == 0
        and len(current_candidate.locants) == current_parent_count
        and current_candidate.render_rank is not None
        and current_candidate.render_rank <= 2
        and not current_candidate.suffixes
        and current_candidate.charge is None
        and proposed_candidate.parent_aromatic_atom_count == 0
        and proposed_candidate.parent_hetero_atom_count == 0
        and not proposed_candidate.locants
        and not proposed_candidate.suffixes
        and proposed_candidate.charge is None
        and not candidate_safety_reasons(current_candidate)
        and not candidate_safety_reasons(proposed_candidate)
        and len(current.name) - len(proposed.name) <= 90
    ):
        return False
    if proposed_parent_count > current_parent_count + 5:
        return True
    if current_candidate.suffixes and proposed_parent_count >= current_parent_count + 5:
        return True
    if current_candidate.render_rank is not None and current_candidate.render_rank > 3:
        return True
    if proposed_candidate.suffixes or proposed_candidate.charge is not None:
        return True
    if candidate_safety_reasons(current_candidate) or candidate_safety_reasons(proposed_candidate):
        return True
    return False


def _ranker_preserves_relative_stereo_parent_over_carbocycle_switch(
    current: _PreparedCandidate,
    proposed: _PreparedCandidate,
    current_key: tuple,
    proposed_key: tuple,
) -> bool:
    current_candidate = current.candidate
    proposed_candidate = proposed.candidate
    current_penalty = int(current_key[0]) if current_key else current.stability_penalty
    proposed_penalty = int(proposed_key[0]) if proposed_key else proposed.stability_penalty
    if proposed_penalty < current_penalty + 60:
        return True
    if proposed_candidate.family != "single-ring-parent":
        return True
    if not proposed_candidate.parent_atoms or not candidate_has_complete_graph(proposed_candidate):
        return True
    if proposed_candidate.parent_aromatic_atom_count or proposed_candidate.parent_hetero_atom_count:
        return True
    if not proposed_candidate.stereo or not proposed_candidate.stereo.atom_descriptors:
        return True
    parent_atoms = set(proposed_candidate.parent_atoms)
    parent_stereo_count = sum(
        1 for atom_idx, _descriptor in proposed_candidate.stereo.atom_descriptors if atom_idx in parent_atoms
    )
    if parent_stereo_count < 2:
        return True
    if current_candidate.family in _ACYCLIC_SKELETON_PARENT_FAMILIES:
        return False
    if (
        current_candidate.family == "single-ring-parent"
        and current_candidate.parent_atoms
        and current_candidate.parent_aromatic_atom_count == len(current_candidate.parent_atoms)
    ):
        return False
    return True


def _ranker_preserves_complete_senior_functional_parent(
    current: _PreparedCandidate,
    proposed: _PreparedCandidate,
) -> bool:
    current_candidate = current.candidate
    proposed_candidate = proposed.candidate
    if current_candidate.family not in _RANKER_SENIOR_FUNCTIONAL_COMPLETE_PARENT_FAMILIES:
        return True
    if proposed_candidate.family not in _RANKER_SENIOR_FUNCTIONAL_SHORTCUT_FAMILIES:
        return True
    if not candidate_has_complete_graph(current_candidate):
        return True
    if not candidate_has_complete_graph(proposed_candidate):
        return True
    if candidate_safety_reasons(current_candidate) or candidate_safety_reasons(proposed_candidate):
        return True
    if proposed_candidate.suffixes or proposed_candidate.charge is not None:
        return True
    return False


def _ranker_preserves_compact_bicyclic_parent(
    current: _PreparedCandidate,
    proposed: _PreparedCandidate,
) -> bool:
    current_candidate = current.candidate
    proposed_candidate = proposed.candidate
    if current_candidate.family not in {
        "functional-generic-bicyclic-parent",
        "generic-bicyclic-parent",
        "preferred-generic-bicyclic-parent",
    }:
        return True
    if proposed_candidate.family not in {
        "acyclic-skeleton-with-ring-substituents",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "functional-acyclic-skeleton-with-ring-substituents",
        "preferred-acyclic-skeleton-with-ring-substituents",
    }:
        return True
    if not current_candidate.parent_atoms or not candidate_has_complete_graph(current_candidate):
        return True
    if not proposed_candidate.parent_atoms or not candidate_has_complete_graph(proposed_candidate):
        return True
    current_parent_count = len(current_candidate.parent_atoms)
    proposed_parent_count = len(proposed_candidate.parent_atoms)
    if current_parent_count < 6 or current_parent_count > 9:
        return True
    compact_saturated_bicyclic_parent = current_parent_count == 6 and current_candidate.parent_aromatic_atom_count == 0
    minimum_locant_count = 2 if compact_saturated_bicyclic_parent else 3
    if len(current_candidate.locants) < minimum_locant_count:
        return True
    if proposed_parent_count < current_parent_count + 5:
        return True
    current_rank = current_candidate.render_rank or 0
    if current_rank < 1 or current_rank > 3:
        return True
    if current_candidate.suffixes or proposed_candidate.suffixes:
        return True
    if current_candidate.charge is not None or proposed_candidate.charge is not None:
        return True
    if current.stability_penalty > proposed.stability_penalty + 80:
        return True
    return False


def _ranker_preserves_generic_bicyclic_parent_over_generated_fused_shortcut(
    current: _PreparedCandidate,
    proposed: _PreparedCandidate,
    current_key: tuple,
    proposed_key: tuple,
) -> bool:
    current_candidate = current.candidate
    proposed_candidate = proposed.candidate
    if current_candidate.family != "generic-bicyclic-parent":
        return True
    if proposed_candidate.family != "generated-fused-parent":
        return True
    if current_candidate.source_family != "bicyclic":
        return True
    if current_candidate.template_name or current_candidate.source_file:
        return True
    if proposed_candidate.source_family != "generated-fused":
        return True
    if proposed_candidate.source_file != "generated_fused_template_index.json":
        return True
    if not current_candidate.parent_atoms or not candidate_has_complete_graph(current_candidate):
        return True
    if not proposed_candidate.parent_atoms or not candidate_has_complete_graph(proposed_candidate):
        return True
    if candidate_safety_reasons(current_candidate) or candidate_safety_reasons(proposed_candidate):
        return True
    current_parent_count = len(current_candidate.parent_atoms)
    proposed_parent_count = len(proposed_candidate.parent_atoms)
    if current_parent_count < 8 or current_parent_count > 10:
        return True
    if len(current_candidate.locants) < 3:
        return True
    if current_candidate.render_rank != 1:
        return True
    if current_candidate.suffixes or current_candidate.charge is not None:
        return True
    if proposed_candidate.suffixes or proposed_candidate.charge is not None:
        return True
    if proposed_candidate.render_rank != 1:
        return True
    if proposed_parent_count < current_parent_count or proposed_parent_count > current_parent_count + 2:
        return True
    if proposed_candidate.parent_aromatic_atom_count != proposed_parent_count:
        return True
    if proposed_candidate.parent_hetero_atom_count > 2:
        return True
    if proposed_candidate.parent_external_oxygen_substituent_count:
        return True
    if proposed_candidate.parent_external_nitrogen_substituent_count > 1:
        return True
    current_penalty = int(current_key[0]) if current_key else current.stability_penalty
    proposed_penalty = int(proposed_key[0]) if proposed_key else proposed.stability_penalty
    if proposed_penalty < current_penalty + 40:
        return True
    return False


def _ranker_preserves_earlier_aromatic_single_ring_parent(
    current: _PreparedCandidate,
    proposed: _PreparedCandidate,
) -> bool:
    current_candidate = current.candidate
    proposed_candidate = proposed.candidate
    if current_candidate.family != "single-ring-parent":
        return True
    if proposed_candidate.family != current_candidate.family:
        return True
    if not current_candidate.parent_atoms or not candidate_has_complete_graph(current_candidate):
        return True
    if not proposed_candidate.parent_atoms or not candidate_has_complete_graph(proposed_candidate):
        return True
    current_parent_count = len(current_candidate.parent_atoms)
    proposed_parent_count = len(proposed_candidate.parent_atoms)
    if current_parent_count < 6:
        return True
    if current_candidate.parent_aromatic_atom_count != current_parent_count:
        return True
    if proposed_candidate.parent_aromatic_atom_count:
        return True
    if proposed_candidate.parent_hetero_atom_count:
        return True
    if proposed_candidate.parent_terminal_hetero_substituent_count:
        return True
    current_rank = current_candidate.render_rank or 0
    proposed_rank = proposed_candidate.render_rank or 0
    if current_rank < 1 or proposed_rank <= current_rank:
        return True
    if proposed_parent_count == 5:
        return True
    if proposed_parent_count >= 6 and current_parent_count < proposed_parent_count:
        return True
    if proposed_parent_count < 5 and current_parent_count < 7:
        return True
    if current.stability_penalty > proposed.stability_penalty + 50:
        return True
    return False


def _ranker_preserves_equivalent_single_ring_parent_choice(
    current: _PreparedCandidate,
    proposed: _PreparedCandidate,
    current_key: tuple,
    proposed_key: tuple,
) -> bool:
    current_candidate = current.candidate
    proposed_candidate = proposed.candidate
    if current_candidate.family != "single-ring-parent":
        return True
    if proposed_candidate.family != current_candidate.family:
        return True
    if proposed_key <= current_key:
        return True
    if not current_candidate.parent_atoms or not candidate_has_complete_graph(current_candidate):
        return True
    if not proposed_candidate.parent_atoms or not candidate_has_complete_graph(proposed_candidate):
        return True
    if current_candidate.charge is not None or proposed_candidate.charge is not None:
        return True
    if current_candidate.suffixes or proposed_candidate.suffixes:
        return True
    if current_candidate.prefixes or proposed_candidate.prefixes:
        return True
    if candidate_safety_reasons(current_candidate) or candidate_safety_reasons(proposed_candidate):
        return True
    current_parent_count = len(current_candidate.parent_atoms)
    proposed_parent_count = len(proposed_candidate.parent_atoms)
    if current_parent_count != proposed_parent_count:
        return True
    if current_parent_count < 5 or current_parent_count > 7:
        return True
    if current_candidate.parent_aromatic_atom_count != proposed_candidate.parent_aromatic_atom_count:
        return True
    if current_candidate.parent_hetero_atom_count != proposed_candidate.parent_hetero_atom_count:
        return True
    if (
        current_candidate.parent_terminal_hetero_substituent_count
        != proposed_candidate.parent_terminal_hetero_substituent_count
    ):
        return True
    if (
        current_candidate.parent_external_nitrogen_substituent_count
        != proposed_candidate.parent_external_nitrogen_substituent_count
    ):
        return True
    if (
        current_candidate.parent_external_oxygen_substituent_count
        != proposed_candidate.parent_external_oxygen_substituent_count
    ):
        return True
    current_render_rank = current_candidate.render_rank or 0
    proposed_render_rank = proposed_candidate.render_rank or 0
    if current_render_rank and proposed_render_rank and proposed_render_rank < current_render_rank:
        return True
    if len(current.name) > _DEFAULT_MAX_NAME_LENGTH:
        return True
    if len(current.name) - len(proposed.name) > 120:
        return True
    return False


def _ranker_preserves_oxindol_parent_over_indolin_tautomer(
    current: _PreparedCandidate,
    proposed: _PreparedCandidate,
) -> bool:
    current_candidate = current.candidate
    proposed_candidate = proposed.candidate
    if current_candidate.family != "fused-retained-parent" or proposed_candidate.family != current_candidate.family:
        return True
    if current_candidate.template_name != "oxindol" or proposed_candidate.template_name != "indolin":
        return True
    if current_candidate.source_family != "arylGroups:ring:ring" or proposed_candidate.source_family != "arylGroups:ring:ring":
        return True
    if current_candidate.priority != 18 or proposed_candidate.priority != 18:
        return True
    if current_candidate.render_rank != 1 or proposed_candidate.render_rank != 2:
        return True
    if not current_candidate.parent_atoms or not candidate_has_complete_graph(current_candidate):
        return True
    if not proposed_candidate.parent_atoms or not candidate_has_complete_graph(proposed_candidate):
        return True
    if len(current_candidate.parent_atoms) != 10 or len(proposed_candidate.parent_atoms) != 9:
        return True
    if current_candidate.charge is not None or proposed_candidate.charge is not None:
        return True
    if current_candidate.suffixes or proposed_candidate.suffixes:
        return True
    if current_candidate.prefixes or proposed_candidate.prefixes:
        return True
    if candidate_safety_reasons(current_candidate) or candidate_safety_reasons(proposed_candidate):
        return True
    if len(current.name) > _DEFAULT_MAX_NAME_LENGTH:
        return True
    return False


def _ranker_preserves_preferred_acyclic_parent_over_whole_neutral_amine(
    current: _PreparedCandidate,
    proposed: _PreparedCandidate,
) -> bool:
    current_candidate = current.candidate
    proposed_candidate = proposed.candidate
    if current_candidate.family != "preferred-acyclic-skeleton-with-ring-substituents":
        return True
    if proposed_candidate.family != "neutral-amine-parent":
        return True
    if current_candidate.priority != 14 or proposed_candidate.priority != 9:
        return True
    if not current_candidate.parent_atoms or not candidate_has_complete_graph(current_candidate):
        return True
    if not proposed_candidate.parent_atoms or not candidate_has_complete_graph(proposed_candidate):
        return True
    current_parent_count = len(current_candidate.parent_atoms)
    proposed_parent_count = len(proposed_candidate.parent_atoms)
    if current_parent_count < 12:
        return True
    if proposed_parent_count < current_parent_count + 8:
        return True
    if current_candidate.charge is not None or proposed_candidate.charge is not None:
        return True
    if current_candidate.suffixes or proposed_candidate.suffixes:
        return True
    if current_candidate.prefixes or proposed_candidate.prefixes:
        return True
    if candidate_safety_reasons(current_candidate) or candidate_safety_reasons(proposed_candidate):
        return True
    if len(current.name) > _DEFAULT_MAX_NAME_LENGTH:
        return True
    return False


def _ranker_preserves_chain_dominant_acyclic_parent_over_functional_bicyclic(
    current: _PreparedCandidate,
    proposed: _PreparedCandidate,
) -> bool:
    current_candidate = current.candidate
    proposed_candidate = proposed.candidate
    if current_candidate.family != "chain-dominant-acyclic-skeleton-with-ring-substituents":
        return True
    if proposed_candidate.family not in {
        "functional-generic-bicyclic-parent",
        "generic-bicyclic-parent",
    }:
        return True
    if current_candidate.priority != 19:
        return True
    if proposed_candidate.family == "functional-generic-bicyclic-parent":
        if proposed_candidate.priority != 14 or proposed_candidate.source_family != "bicyclic":
            return True
    elif proposed_candidate.priority != 19:
        return True
    proposed_render_rank = proposed_candidate.render_rank or 0
    if proposed_render_rank != 1 and (
        proposed_render_rank > 8
        or proposed.selection_bonus < current.selection_bonus + 20
    ):
        return True
    if current_candidate.render_rank is not None:
        return True
    if not current_candidate.parent_atoms or not candidate_has_complete_graph(current_candidate):
        return True
    if not proposed_candidate.parent_atoms or not candidate_has_complete_graph(proposed_candidate):
        return True
    current_parent_count = len(current_candidate.parent_atoms)
    proposed_parent_count = len(proposed_candidate.parent_atoms)
    if current_parent_count < 7:
        return True
    if proposed_parent_count > 9:
        return True
    if proposed_parent_count > current_parent_count + 1:
        return True
    if current_candidate.charge is not None or proposed_candidate.charge is not None:
        return True
    if current_candidate.suffixes or proposed_candidate.suffixes:
        return True
    if current_candidate.prefixes or proposed_candidate.prefixes:
        return True
    if current_candidate.parent_aromatic_atom_count or proposed_candidate.parent_aromatic_atom_count:
        return True
    if current_candidate.parent_hetero_atom_count or proposed_candidate.parent_hetero_atom_count:
        return True
    if candidate_safety_reasons(current_candidate) or candidate_safety_reasons(proposed_candidate):
        return True
    if len(current.name) > _DEFAULT_MAX_NAME_LENGTH:
        return True
    return False


def _ranker_preserves_high_signal_topology_parent(
    current: _PreparedCandidate,
    proposed: _PreparedCandidate,
) -> bool:
    current_candidate = current.candidate
    proposed_candidate = proposed.candidate
    if current_candidate.family != "high-signal-generated-topology-fused-parent":
        return True
    if proposed_candidate.family == current_candidate.family:
        return True
    if proposed_candidate.family not in {
        "fused-retained-parent",
        "generated-partial-hydro-fused-parent",
        "retained-topology-replacement-fused-parent",
    }:
        return True
    if not candidate_has_complete_graph(current_candidate):
        return True
    if current_candidate.render_rank not in {None, 1}:
        return True
    if current_candidate.charge is not None or current_candidate.suffixes:
        return True
    if proposed_candidate.charge is not None or proposed_candidate.suffixes:
        return True
    if not proposed_candidate.parent_atoms:
        return True
    if len(proposed_candidate.parent_atoms) > len(current_candidate.parent_atoms) + 2:
        return True
    return False


def _ranker_preserves_generated_purine_parent(
    current: _PreparedCandidate,
    proposed: _PreparedCandidate,
) -> bool:
    current_candidate = current.candidate
    proposed_candidate = proposed.candidate
    if _is_stable_retained_purin_6_amine_candidate(current_candidate):
        if proposed_candidate.family != "generated-fused-parent":
            return True
        if proposed_candidate.template_name not in _GENERATED_PURINE_PARENT_TEMPLATES:
            return True
        if not candidate_has_complete_graph(proposed_candidate):
            return True
        if proposed_candidate.parent_atoms != current_candidate.parent_atoms:
            return True
        if proposed_candidate.charge is not None or proposed_candidate.suffixes:
            return True
        return False
    if current_candidate.family != "generated-fused-parent":
        return True
    if current_candidate.template_name not in _GENERATED_PURINE_PARENT_TEMPLATES:
        return True
    if proposed_candidate.family != "fused-retained-parent":
        return True
    if proposed_candidate.template_name != "purin":
        return True
    if not candidate_has_complete_graph(current_candidate):
        return True
    if not candidate_has_complete_graph(proposed_candidate):
        return True
    if len(current_candidate.parent_atoms) != 9:
        return True
    if proposed_candidate.parent_atoms != current_candidate.parent_atoms:
        return True
    if current_candidate.parent_substituted_hetero_atom_count:
        return True
    if proposed_candidate.parent_substituted_hetero_atom_count:
        return True
    if _generated_purine_candidate_has_terminal_hetero_substituent(current_candidate):
        return True
    if current_candidate.render_rank is None or current_candidate.render_rank > 2:
        return True
    if current_candidate.charge is not None or proposed_candidate.charge is not None:
        return True
    if current_candidate.suffixes or proposed_candidate.suffixes:
        return True
    return False


_STABLE_RETAINED_PURIN_6_AMINE_SOURCE_FAMILY = "fused-retained:purin-6-amine"
_STABLE_RETAINED_PURIN_THIOXO_SOURCE_FAMILY = "fused-retained:purin-thioxo"


def _is_stable_retained_purin_6_amine_candidate(candidate: NameCandidate) -> bool:
    return (
        candidate.family == "fused-retained-parent"
        and candidate.template_name == "purin"
        and candidate.source_family == _STABLE_RETAINED_PURIN_6_AMINE_SOURCE_FAMILY
        and candidate_has_complete_graph(candidate)
        and bool(candidate.parent_atoms)
        and len(candidate.parent_atoms) == 9
        and candidate.charge is None
        and not candidate.suffixes
    )


def _is_stable_retained_purin_thioxo_candidate(candidate: NameCandidate) -> bool:
    return (
        candidate.family == "fused-retained-parent"
        and candidate.template_name == "purin"
        and candidate.source_family == _STABLE_RETAINED_PURIN_THIOXO_SOURCE_FAMILY
        and candidate_has_complete_graph(candidate)
        and bool(candidate.parent_atoms)
        and len(candidate.parent_atoms) == 9
        and candidate.charge is None
        and not candidate.suffixes
    )


def _stable_purin_6_amine_alternative_for_generated_or_adenin_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if not _can_replace_with_stable_purin_6_amine(selected_candidate):
        return None
    if not candidate_has_complete_graph(selected_candidate) or not selected_candidate.parent_atoms:
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None
    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if not _is_stable_retained_purin_6_amine_candidate(candidate):
            continue
        if not _stable_purin_6_amine_matches_selected_parent(candidate, selected_candidate):
            continue
        render_rank = candidate.render_rank
        if render_rank is None or render_rank > 3:
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(item.name) - len(selected.name) > 55:
            continue
        alternatives.append(((render_rank, candidate_key), item))
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _can_replace_with_stable_purin_6_amine(candidate: NameCandidate) -> bool:
    if candidate.family == "generated-fused-parent":
        return candidate.template_name in _GENERATED_PURINE_PARENT_TEMPLATES
    if candidate.family == "neutral-amine-parent":
        return True
    return candidate.family == "fused-retained-parent" and candidate.template_name == "adenin"


def _stable_purin_6_amine_matches_selected_parent(
    candidate: NameCandidate,
    selected_candidate: NameCandidate,
) -> bool:
    if selected_candidate.family == "generated-fused-parent":
        return candidate.parent_atoms == selected_candidate.parent_atoms
    if selected_candidate.template_name != "adenin":
        if selected_candidate.family != "neutral-amine-parent":
            return False
        if not candidate.parent_atoms.issubset(selected_candidate.parent_atoms):
            return False
        return 1 <= len(selected_candidate.parent_atoms) - len(candidate.parent_atoms) <= 8
    if not candidate.parent_atoms.issubset(selected_candidate.parent_atoms):
        return False
    return len(selected_candidate.parent_atoms) - len(candidate.parent_atoms) == 1


def _stable_purin_thioxo_alternative_for_generated_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family != "generated-fused-parent":
        return None
    if selected_candidate.template_name not in _GENERATED_PURINE_PARENT_TEMPLATES:
        return None
    if not candidate_has_complete_graph(selected_candidate) or not selected_candidate.parent_atoms:
        return None
    if len(selected_candidate.parent_atoms) != 9:
        return None
    if selected_candidate.parent_terminal_chalcogen_double_bond_count < 2:
        return None
    if selected_candidate.charge is not None or selected_candidate.suffixes:
        return None
    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if not _is_stable_retained_purin_thioxo_candidate(candidate):
            continue
        if candidate.parent_atoms != selected_candidate.parent_atoms:
            continue
        render_rank = candidate.render_rank
        if render_rank is None or render_rank > 3:
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(item.name) - len(selected.name) > 70:
            continue
        if item.stability_penalty > selected.stability_penalty + 160:
            continue
        alternatives.append(((render_rank, candidate_key), item))
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _heterocarbonyl_aromatic_single_ring_alternative_for_single_ring_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family != "single-ring-parent":
        return None
    if (
        selected_candidate.source_family == _SINGLE_RING_RETAINED_CARBOHYDRATE_SUBSTITUENT_SOURCE
        or _is_carbohydrate_like_single_ring_candidate(selected_candidate)
    ):
        return None
    if not candidate_has_complete_graph(selected_candidate) or not selected_candidate.parent_atoms:
        return None
    if selected_candidate.charge is not None or selected_candidate.suffixes:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None
    selected_carbonyl = selected_candidate.parent_terminal_carbonyl_substituent_count
    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "single-ring-parent":
            continue
        if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
            continue
        if len(candidate.parent_atoms) != 6:
            continue
        if candidate.parent_aromatic_atom_count != 6 or candidate.parent_hetero_atom_count:
            continue
        if (
            candidate.parent_terminal_hetero_substituent_count
            or candidate.parent_external_nitrogen_substituent_count
            or candidate.parent_external_oxygen_substituent_count
        ):
            continue
        if candidate.parent_terminal_carbonyl_substituent_count <= selected_carbonyl:
            continue
        if _terminal_carbonyl_switch_displaces_preferred_single_ring_parent(
            selected_candidate,
            selected.name,
            item.name,
        ):
            continue
        if candidate.charge is not None or candidate.suffixes:
            continue
        if candidate_safety_reasons(candidate):
            continue
        render_rank = candidate.render_rank
        if render_rank is None or render_rank > 4:
            continue
        selected_rank = selected_candidate.render_rank
        if selected_rank is not None and render_rank > selected_rank + 3:
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if (
            selected_carbonyl == 0
            and _is_clean_aromatic_carbocycle_single_ring_candidate(selected_candidate)
            and len(item.name) - len(selected.name) > 10
        ):
            continue
        if len(item.name) - len(selected.name) > 90:
            continue
        if item.stability_penalty > selected.stability_penalty + 60:
            continue
        alternatives.append(((-candidate.parent_terminal_carbonyl_substituent_count, candidate_key), item))
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _terminal_carbonyl_switch_displaces_preferred_single_ring_parent(
    selected_candidate: NameCandidate,
    selected_name: str,
    proposed_name: str,
) -> bool:
    if selected_candidate.parent_hetero_atom_count >= 1:
        return True
    return (
        selected_candidate.parent_terminal_hetero_substituent_count >= 2
        and len(proposed_name) > len(selected_name)
    )


def _is_clean_aromatic_carbocycle_single_ring_candidate(candidate: NameCandidate) -> bool:
    return (
        candidate.family == "single-ring-parent"
        and bool(candidate.parent_atoms)
        and candidate_has_complete_graph(candidate)
        and len(candidate.parent_atoms) == 6
        and candidate.parent_aromatic_atom_count == 6
        and candidate.parent_hetero_atom_count == 0
        and candidate.parent_terminal_hetero_substituent_count == 0
        and candidate.parent_terminal_heterocarbon_substituent_count == 0
        and candidate.parent_terminal_carbonyl_substituent_count == 0
        and candidate.parent_external_nitrogen_substituent_count == 0
        and candidate.parent_external_oxygen_substituent_count == 0
        and candidate.charge is None
        and not candidate.suffixes
        and not candidate_safety_reasons(candidate)
    )


def _is_carbohydrate_like_single_ring_candidate(candidate: NameCandidate) -> bool:
    return (
        candidate.family == "single-ring-parent"
        and bool(candidate.parent_atoms)
        and candidate_has_complete_graph(candidate)
        and len(candidate.parent_atoms) in {5, 6}
        and candidate.parent_aromatic_atom_count == 0
        and candidate.parent_hetero_atom_count == 1
        and candidate.parent_external_oxygen_substituent_count >= 1
        and len(candidate.stereo_locants) >= 3
    )


def _acyclic_skeleton_alternative_for_oxathiazolone_single_ring_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if not _is_oxathiazolone_single_ring_candidate(selected_candidate):
        return None
    if selected_candidate.charge is not None:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family not in _RANKER_ACYCLIC_SKELETON_WITH_RING_SUBSTITUENT_FAMILIES:
            continue
        if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
            continue
        if len(candidate.parent_atoms) < 3 or len(candidate.parent_atoms) > 6:
            continue
        if candidate.parent_hetero_atom_count:
            continue
        if candidate.parent_external_unsaturated_bond_count > selected_candidate.parent_external_unsaturated_bond_count:
            continue
        if candidate.charge is not None or candidate.suffixes:
            continue
        if candidate_safety_reasons(candidate):
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(item.name) - len(selected.name) > 80:
            continue
        if item.stability_penalty > selected.stability_penalty + 80:
            continue
        alternatives.append(((candidate.priority, len(item.name), candidate_key), item))
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _is_oxathiazolone_single_ring_candidate(candidate: NameCandidate) -> bool:
    if candidate.family != "single-ring-parent":
        return False
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return False
    if len(candidate.parent_atoms) != 5:
        return False
    element_counts = dict(candidate.parent_element_counts)
    return (
        element_counts.get("C") == 2
        and element_counts.get("N") == 1
        and element_counts.get("O") == 1
        and element_counts.get("S") == 1
        and bool(candidate.suffixes)
        and candidate.parent_terminal_hetero_substituent_count >= 1
    )


def _nitrogen_retained_fused_alternative_for_single_ring_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family != "single-ring-parent":
        return None
    if selected_candidate.template_name or selected_candidate.source_family or selected_candidate.source_file:
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    if len(selected_candidate.parent_atoms) not in {5, 6}:
        return None
    if selected_candidate.charge is not None:
        return None
    if selected_candidate.molecule_abs_formal_charge_sum:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    selected_render_rank = selected_candidate.render_rank
    if selected_render_rank is not None and selected_render_rank > 3:
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None

    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "fused-retained-parent":
            continue
        if not candidate.template_name:
            continue
        if candidate.source_family != "static-retained":
            continue
        if candidate.charge is not None or candidate.molecule_abs_formal_charge_sum:
            continue
        if candidate.candidate_isotope_atom_count != candidate.molecule_isotope_atom_count:
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        parent_elements = dict(candidate.parent_element_counts)
        if parent_elements.get("N", 0) < 2:
            continue
        if any(symbol not in {"C", "N"} for symbol in parent_elements):
            continue
        parent_delta = len(candidate.parent_atoms) - len(selected_candidate.parent_atoms)
        if parent_delta < 3:
            continue
        if len(candidate.locants) > len(selected_candidate.locants):
            continue
        if candidate.parent_abs_formal_charge_sum or candidate.template_abs_formal_charge_sum:
            continue
        if candidate_safety_reasons(candidate):
            continue
        render_rank = candidate.render_rank
        if render_rank is not None and selected_render_rank is not None and render_rank > selected_render_rank + 2:
            continue
        if render_rank is not None and render_rank > 4:
            continue
        if len(item.name) > 115:
            continue
        if len(item.name) - len(selected.name) > 15:
            continue
        if item.stability_penalty > selected.stability_penalty + 220:
            continue
        if candidate_key[0] > selected_key[0] + 260:
            continue
        alternatives.append(((len(item.name) - item.selection_bonus, render_rank or 999, candidate_key), item))
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _generated_purine_candidate_has_terminal_hetero_substituent(candidate: NameCandidate) -> bool:
    return (
        candidate.family == "generated-fused-parent"
        and candidate.template_name in _GENERATED_PURINE_PARENT_TEMPLATES
        and candidate.parent_terminal_hetero_substituent_count > 0
    )


def _is_unanchored_indicated_h_generated_purine_candidate(candidate: NameCandidate) -> bool:
    return (
        _generated_purine_candidate_has_terminal_hetero_substituent(candidate)
        and candidate.parent_terminal_chalcogen_double_bond_count > 0
        and bool(candidate.parent_indicated_hydrogen_locants)
        and "1" not in candidate.parent_indicated_hydrogen_locants
        and "3" in candidate.parent_indicated_hydrogen_locants
    )


def _is_matching_retained_purin_candidate(candidate: NameCandidate, generated_candidate: NameCandidate) -> bool:
    return (
        candidate.family == "fused-retained-parent"
        and candidate.template_name == "purin"
        and candidate_has_complete_graph(candidate)
        and candidate_has_complete_graph(generated_candidate)
        and bool(candidate.parent_atoms)
        and candidate.parent_atoms == generated_candidate.parent_atoms
        and len(candidate.parent_atoms) == 9
        and not _has_acylated_external_nitrogen_monochalcogen_purine_shape(generated_candidate)
        and candidate.charge is None
        and generated_candidate.charge is None
        and not candidate.suffixes
        and not generated_candidate.suffixes
    )


def _has_acylated_external_nitrogen_monochalcogen_purine_shape(candidate: NameCandidate) -> bool:
    return (
        candidate.parent_external_acyl_nitrogen_substituent_count > 0
        and candidate.parent_terminal_chalcogen_double_bond_count == 1
    )


def _ranker_preserves_biochemical_retained_parent_over_shortcut(
    current: _PreparedCandidate,
    proposed: _PreparedCandidate,
    current_key: tuple,
    proposed_key: tuple,
) -> bool:
    current_candidate = current.candidate
    proposed_candidate = proposed.candidate
    if not _is_guarded_biochemical_retained_parent(current_candidate):
        return True
    if (
        current_candidate.template_name == "pterin"
        and proposed_candidate.family == "fused-retained-parent"
        and proposed_candidate.template_name == "pteridin"
    ):
        return True
    if proposed_candidate.family == "fused-retained-parent":
        if proposed_candidate.template_name not in _BIOCHEMICAL_RETAINED_FUSED_PARENT_TEMPLATES:
            return True
        if not candidate_has_complete_graph(proposed_candidate) or not proposed_candidate.parent_atoms:
            return True
        current_oxygen_count = _candidate_parent_element_count(current_candidate, "O")
        proposed_oxygen_count = _candidate_parent_element_count(proposed_candidate, "O")
        if (
            len(proposed_candidate.parent_atoms) >= len(current_candidate.parent_atoms)
            and proposed_oxygen_count >= current_oxygen_count
        ):
            return True
        return proposed_key <= current_key
    if proposed_candidate.family in _RANKER_ACYCLIC_SKELETON_WITH_RING_SUBSTITUENT_FAMILIES | {
        "acyclic-parent",
        "generated-fused-parent",
        "single-ring-parent",
    }:
        return proposed_key <= current_key
    return True


def _is_guarded_biochemical_retained_parent(candidate: NameCandidate) -> bool:
    if candidate.family != "fused-retained-parent":
        return False
    if candidate.template_name not in _BIOCHEMICAL_RETAINED_FUSED_PARENT_TEMPLATES:
        return False
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return False
    if candidate.render_rank is not None and candidate.render_rank > 2:
        return False
    if candidate.charge is not None or candidate.suffixes:
        return False
    return True


def _ranker_preserves_retained_benzimidazole_parent(
    current: _PreparedCandidate,
    proposed: _PreparedCandidate,
) -> bool:
    current_candidate = current.candidate
    proposed_candidate = proposed.candidate
    if current_candidate.family != "fused-retained-parent":
        return True
    if current_candidate.template_name != "benzimidazole":
        return True
    if proposed_candidate.family != "generated-fused-parent":
        return True
    if proposed_candidate.template_name != "imidazo[4,5-a]benzene":
        return True
    if not candidate_has_complete_graph(current_candidate):
        return True
    if not candidate_has_complete_graph(proposed_candidate):
        return True
    if current_candidate.parent_element_counts != (("C", 7), ("N", 2)):
        return True
    if len(current_candidate.parent_atoms) != 9:
        return True
    if proposed_candidate.parent_atoms != current_candidate.parent_atoms:
        return True
    if current_candidate.parent_substituted_hetero_atom_count:
        return True
    if proposed_candidate.parent_substituted_hetero_atom_count:
        return True
    current_render_rank = current_candidate.render_rank
    proposed_render_rank = proposed_candidate.render_rank
    if current_render_rank is None or proposed_render_rank is None:
        return True
    if current_render_rank > proposed_render_rank:
        return True
    if current_candidate.priority > proposed_candidate.priority:
        return True
    if current_candidate.charge is not None or proposed_candidate.charge is not None:
        return True
    if current_candidate.suffixes or proposed_candidate.suffixes:
        return True
    return False


_CANDIDATE_FACT_SWITCH_NUMERIC_KEYS = (
    "priority",
    "render_rank",
    "parent_atom_count",
    "ring_system_atom_count",
    "parent_aromatic_atom_count",
    "parent_hetero_atom_count",
    "parent_terminal_hetero_substituent_count",
    "parent_external_nitrogen_substituent_count",
    "parent_external_oxygen_substituent_count",
    "locant_count",
    "prefix_count",
    "suffix_count",
    "parent_atom_delta_max",
    "parent_atom_delta_min",
    "locant_delta_max",
    "priority_delta_min",
    "candidate_count_in_row",
)


def _candidate_fact_switch_gate_accepts(
    current_facts: dict[str, object],
    proposed_facts: dict[str, object],
    current_score: float,
    proposed_score: float,
) -> bool:
    features = _candidate_fact_switch_features(
        current_facts,
        proposed_facts,
        current_score,
        proposed_score,
    )
    return _candidate_fact_switch_gate_score(features) >= _CANDIDATE_FACT_SWITCH_GATE_THRESHOLD


def _candidate_fact_switch_features(
    current_facts: dict[str, object],
    proposed_facts: dict[str, object],
    current_score: float,
    proposed_score: float,
) -> dict[str, float]:
    current_family = str(current_facts["family"])
    proposed_family = str(proposed_facts["family"])
    features: dict[str, float] = {
        f"old_family={current_family}": 1.0,
        f"new_family={proposed_family}": 1.0,
        f"pair={current_family}=>{proposed_family}": 1.0,
        "same_family": float(current_family == proposed_family),
        f"old_template={current_facts.get('template_name', '')}": 1.0,
        f"new_template={proposed_facts.get('template_name', '')}": 1.0,
        f"template_pair={current_facts.get('template_name', '')}=>{proposed_facts.get('template_name', '')}": 1.0,
        "same_template": float(current_facts.get("template_name", "") == proposed_facts.get("template_name", "")),
        f"old_complete={current_facts['complete_graph']}": 1.0,
        f"new_complete={proposed_facts['complete_graph']}": 1.0,
        f"old_suffix={current_facts['suffix_kinds']}": 1.0,
        f"new_suffix={proposed_facts['suffix_kinds']}": 1.0,
        f"old_charge={current_facts['charge']}": 1.0,
        f"new_charge={proposed_facts['charge']}": 1.0,
        f"old_safety={current_facts['safety_reasons']}": 1.0,
        f"new_safety={proposed_facts['safety_reasons']}": 1.0,
        "old_score": float(current_score),
        "new_score": float(proposed_score),
        "score_margin": float(proposed_score - current_score),
    }
    for key in _CANDIDATE_FACT_SWITCH_NUMERIC_KEYS:
        current_value = float(current_facts.get(key, -1.0))
        proposed_value = float(proposed_facts.get(key, -1.0))
        features[f"old_{key}"] = current_value
        features[f"new_{key}"] = proposed_value
        features[f"delta_{key}"] = proposed_value - current_value
    return features


def _candidate_fact_ranker_row_metrics(items: list[_PreparedCandidate]) -> dict[str, float]:
    parent_counts = [float(len(item.candidate.parent_atoms)) for item in items]
    locant_counts = [float(len(item.candidate.locants)) for item in items]
    priorities = [float(item.candidate.priority) for item in items]
    return {
        "candidate_count_in_row": float(len(items)),
        "max_parent_atom_count": max(parent_counts, default=-1.0),
        "min_parent_atom_count": min(parent_counts, default=-1.0),
        "max_locant_count": max(locant_counts, default=-1.0),
        "min_priority": min(priorities, default=-1.0),
    }


def _candidate_fact_ranker_uses_legacy_generated_fused_graph_facts(candidate: NameCandidate) -> bool:
    return (
        candidate.family == "generated-fused-parent"
        and candidate.source_file == "generated_fused_template_index.json"
    )


def _candidate_fact_ranker_ring_system_atom_count(candidate: NameCandidate) -> float:
    # Keep the generated-fused ranker input stable; explicit safety gates use
    # the populated graph counts directly.
    if _candidate_fact_ranker_uses_legacy_generated_fused_graph_facts(candidate):
        return 0.0
    return float(len(candidate.ring_system_atoms))


def _candidate_fact_ranker_parent_aromatic_atom_count(candidate: NameCandidate) -> float:
    if _candidate_fact_ranker_uses_legacy_generated_fused_graph_facts(candidate):
        return 0.0
    return float(candidate.parent_aromatic_atom_count)


def _candidate_fact_ranker_parent_hetero_atom_count(candidate: NameCandidate) -> float:
    if _candidate_fact_ranker_uses_legacy_generated_fused_graph_facts(candidate):
        return 0.0
    return float(candidate.parent_hetero_atom_count)


def _candidate_fact_ranker_parent_terminal_hetero_substituent_count(candidate: NameCandidate) -> float:
    if _candidate_fact_ranker_uses_legacy_generated_fused_graph_facts(candidate):
        return 0.0
    return float(candidate.parent_terminal_hetero_substituent_count)


def _candidate_fact_ranker_parent_external_nitrogen_substituent_count(candidate: NameCandidate) -> float:
    if _candidate_fact_ranker_uses_legacy_generated_fused_graph_facts(candidate):
        return 0.0
    return float(candidate.parent_external_nitrogen_substituent_count)


def _candidate_fact_ranker_parent_external_oxygen_substituent_count(candidate: NameCandidate) -> float:
    if _candidate_fact_ranker_uses_legacy_generated_fused_graph_facts(candidate):
        return 0.0
    return float(candidate.parent_external_oxygen_substituent_count)


def _candidate_fact_ranker_facts(
    item: _PreparedCandidate,
    metrics: dict[str, float],
) -> dict[str, object]:
    candidate = item.candidate
    parent_atom_count = float(len(candidate.parent_atoms))
    locant_count = float(len(candidate.locants))
    priority = float(candidate.priority)
    return {
        "family": candidate.family,
        "producer_family": item.producer_family,
        "template_name": candidate.template_name,
        "complete_graph": "1" if candidate_has_complete_graph(candidate) else "0",
        "suffix_kinds": ",".join(sorted({suffix.kind for suffix in candidate.suffixes})),
        "charge": candidate.charge.formal_charge if candidate.charge is not None else 0,
        "safety_reasons": ",".join(candidate_safety_reasons(candidate)),
        "priority": priority,
        "render_rank": float(candidate.render_rank) if candidate.render_rank is not None else -1.0,
        "parent_atom_count": parent_atom_count,
        "ring_system_atom_count": _candidate_fact_ranker_ring_system_atom_count(candidate),
        "parent_aromatic_atom_count": _candidate_fact_ranker_parent_aromatic_atom_count(candidate),
        "parent_hetero_atom_count": _candidate_fact_ranker_parent_hetero_atom_count(candidate),
        "parent_terminal_hetero_substituent_count": _candidate_fact_ranker_parent_terminal_hetero_substituent_count(
            candidate,
        ),
        "parent_external_nitrogen_substituent_count": _candidate_fact_ranker_parent_external_nitrogen_substituent_count(
            candidate,
        ),
        "parent_external_oxygen_substituent_count": _candidate_fact_ranker_parent_external_oxygen_substituent_count(
            candidate,
        ),
        "locant_count": locant_count,
        "prefix_count": float(len(candidate.prefixes)),
        "suffix_count": float(len(candidate.suffixes)),
        "parent_atom_delta_max": parent_atom_count - metrics["max_parent_atom_count"],
        "parent_atom_delta_min": parent_atom_count - metrics["min_parent_atom_count"],
        "locant_delta_max": locant_count - metrics["max_locant_count"],
        "priority_delta_min": priority - metrics["min_priority"],
        "candidate_count_in_row": metrics["candidate_count_in_row"],
    }


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
        or candidate.parent_element_counts
        or candidate.parent_external_relative_stereo_branch_count
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


def _with_molecule_stereo_evidence(mol, Chem, candidate: NameCandidate) -> NameCandidate:
    if candidate.stereo is not None:
        return candidate
    if not _has_assigned_stereo(mol, Chem):
        return candidate
    stereo = _molecule_stereo_evidence(mol, Chem)
    if stereo is None:
        return candidate
    return replace(candidate, stereo=stereo)


def _molecule_stereo_evidence(mol, Chem) -> StereoEvidence | None:
    cached = mol_cache_get(mol, "molecule_stereo_evidence")
    if cached is not None:
        return cached or None
    atom_descriptors = tuple(
        (atom.GetIdx(), str(atom.GetChiralTag()))
        for atom in mol.GetAtoms()
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
    )
    bond_descriptors = tuple(
        (bond.GetIdx(), str(bond.GetStereo()))
        for bond in mol.GetBonds()
        if bond.GetStereo() != Chem.BondStereo.STEREONONE
    )
    stereo = (
        StereoEvidence(
            atom_descriptors=atom_descriptors,
            bond_descriptors=bond_descriptors,
        )
        if atom_descriptors or bond_descriptors
        else None
    )
    mol_cache_set(mol, "molecule_stereo_evidence", stereo or False)
    return stereo


def _parent_atom_evidence(mol, parent_atoms: frozenset[int]) -> tuple[
    tuple[tuple[str, int], ...],
    int,
    int,
    int,
    int,
    int,
]:
    cache = mol_cache_dict(mol, "parent_atom_evidence")
    cached = cache.get(parent_atoms)
    if cached is not None:
        return cached
    counts: dict[str, int] = {}
    formal_charge_atom_count = 0
    abs_formal_charge_sum = 0
    external_unsaturated_bond_count = 0
    internal_unsaturated_bond_count = 0
    external_isotope_hydrogen_count = 0
    for atom_idx in parent_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        symbol = atom.GetSymbol()
        counts[symbol] = counts.get(symbol, 0) + 1
        formal_charge = atom.GetFormalCharge()
        if formal_charge:
            formal_charge_atom_count += 1
            abs_formal_charge_sum += abs(formal_charge)
        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx in parent_atoms:
                if atom_idx < neighbor_idx:
                    bond = mol.GetBondBetweenAtoms(atom_idx, neighbor_idx)
                    if (
                        bond is not None
                        and not bond.GetIsAromatic()
                        and bond.GetBondTypeAsDouble() > 1.1
                    ):
                        internal_unsaturated_bond_count += 1
                continue
            if neighbor.GetAtomicNum() == 1 and neighbor.GetIsotope() in {2, 3}:
                external_isotope_hydrogen_count += 1
            bond = mol.GetBondBetweenAtoms(atom_idx, neighbor_idx)
            if bond is not None and bond.GetBondTypeAsDouble() > 1.1:
                external_unsaturated_bond_count += 1
    evidence = (
        tuple(sorted(counts.items())),
        formal_charge_atom_count,
        abs_formal_charge_sum,
        external_unsaturated_bond_count,
        internal_unsaturated_bond_count,
        external_isotope_hydrogen_count,
    )
    cache[parent_atoms] = evidence
    return evidence


def _molecule_isotope_atom_indices(mol) -> frozenset[int]:
    cached = mol_cache_get(mol, "molecule_isotope_atom_indices")
    if cached is not None:
        return cached
    isotope_atoms = frozenset(atom.GetIdx() for atom in mol.GetAtoms() if atom.GetIsotope())
    return mol_cache_set(mol, "molecule_isotope_atom_indices", isotope_atoms)


def _candidate_isotope_atom_indices(mol, candidate: NameCandidate) -> frozenset[int]:
    molecule_isotope_atoms = _molecule_isotope_atom_indices(mol)
    if not molecule_isotope_atoms:
        return frozenset()
    covered_atoms: set[int] = set(candidate.parent_atoms)
    for prefix in candidate.prefixes:
        covered_atoms.update(prefix.atoms)
    for suffix in candidate.suffixes:
        covered_atoms.update(suffix.atoms)
    for atom_idx in tuple(covered_atoms):
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetAtomicNum() == 1:
            continue
        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx in molecule_isotope_atoms and neighbor.GetAtomicNum() == 1:
                covered_atoms.add(neighbor_idx)
    return frozenset(idx for idx in covered_atoms if idx in molecule_isotope_atoms)


def _with_molecule_graph_fragility_evidence(mol, Chem, candidate: NameCandidate) -> NameCandidate:
    updated = candidate
    if _candidate_has_external_bond_stereo_parent(mol, Chem, candidate):
        updated = replace(updated, external_bond_stereo_parent_fragile=True)
    fragility_reasons: list[str] = []
    if _candidate_has_external_oxyimino_carbonyl_parent(mol, Chem, candidate):
        fragility_reasons.append(OPSIN_FRAGILE_OXYIMINO_CARBONYL_PARENT)
    if _candidate_has_parent_oxyazo_carbonyl(mol, Chem, candidate):
        fragility_reasons.append(OPSIN_FRAGILE_OXYIMINO_CARBONYL_PARENT)
    if not fragility_reasons:
        return updated
    safety = updated.safety or CandidateSafety()
    reasons = safety.opsin_fragility_reasons
    new_reasons = tuple(reason for reason in fragility_reasons if reason not in reasons)
    if not new_reasons:
        return updated
    return replace(
        updated,
        safety=replace(
            safety,
            opsin_fragility_reasons=(*reasons, *new_reasons),
        ),
    )


def _candidate_has_external_bond_stereo_parent(mol, Chem, candidate: NameCandidate) -> bool:
    if candidate.family not in _EXTERNAL_BOND_STEREO_FRAGILITY_PARENT_FAMILIES:
        return False
    stereo = candidate.stereo
    if stereo is None or not stereo.bond_descriptors:
        return False
    parent_atoms = set(candidate.parent_atoms)
    locanted_atoms = {atom_idx for atom_idx, _locant in candidate.stereo_locants}
    if not parent_atoms and locanted_atoms:
        return False
    for bond_idx, _descriptor in stereo.bond_descriptors:
        bond = mol.GetBondWithIdx(bond_idx)
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        if begin_idx in parent_atoms and end_idx in parent_atoms:
            continue
        if begin_idx in locanted_atoms and end_idx in locanted_atoms:
            continue
        return True
    return False


def _with_parent_element_counts(mol, candidate: NameCandidate) -> NameCandidate:
    molecule_isotope_atom_count = len(_molecule_isotope_atom_indices(mol))
    candidate_isotope_atom_count = (
        len(_candidate_isotope_atom_indices(mol, candidate)) if molecule_isotope_atom_count else 0
    )
    if not candidate.parent_atoms:
        if (
            candidate.molecule_isotope_atom_count == molecule_isotope_atom_count
            and candidate.parent_isotope_atom_count == 0
            and candidate.candidate_isotope_atom_count == candidate_isotope_atom_count
        ):
            return candidate
        return replace(
            candidate,
            molecule_isotope_atom_count=molecule_isotope_atom_count,
            parent_isotope_atom_count=0,
            candidate_isotope_atom_count=candidate_isotope_atom_count,
        )
    parent_atoms = frozenset(candidate.parent_atoms)
    (
        parent_element_counts,
        formal_charge_atom_count,
        abs_formal_charge_sum,
        external_unsaturated_bond_count,
        internal_unsaturated_bond_count,
        external_isotope_hydrogen_count,
    ) = _parent_atom_evidence(mol, parent_atoms)
    parent_isotope_atom_count = len(_molecule_isotope_atom_indices(mol).intersection(parent_atoms))
    molecule_formal_charge_atom_count, molecule_abs_formal_charge_sum = _molecule_charge_counts(mol)
    if (
        candidate.parent_element_counts == parent_element_counts
        and candidate.molecule_formal_charge_atom_count == molecule_formal_charge_atom_count
        and candidate.molecule_abs_formal_charge_sum == molecule_abs_formal_charge_sum
        and candidate.parent_formal_charge_atom_count == formal_charge_atom_count
        and candidate.parent_abs_formal_charge_sum == abs_formal_charge_sum
        and candidate.parent_external_unsaturated_bond_count == external_unsaturated_bond_count
        and candidate.parent_internal_unsaturated_bond_count == internal_unsaturated_bond_count
        and candidate.parent_external_isotope_hydrogen_count == external_isotope_hydrogen_count
        and candidate.molecule_isotope_atom_count == molecule_isotope_atom_count
        and candidate.parent_isotope_atom_count == parent_isotope_atom_count
        and candidate.candidate_isotope_atom_count == candidate_isotope_atom_count
    ):
        return candidate
    return replace(
        candidate,
        parent_element_counts=parent_element_counts,
        molecule_formal_charge_atom_count=molecule_formal_charge_atom_count,
        molecule_abs_formal_charge_sum=molecule_abs_formal_charge_sum,
        parent_formal_charge_atom_count=formal_charge_atom_count,
        parent_abs_formal_charge_sum=abs_formal_charge_sum,
        parent_external_unsaturated_bond_count=external_unsaturated_bond_count,
        parent_internal_unsaturated_bond_count=internal_unsaturated_bond_count,
        parent_external_isotope_hydrogen_count=external_isotope_hydrogen_count,
        molecule_isotope_atom_count=molecule_isotope_atom_count,
        parent_isotope_atom_count=parent_isotope_atom_count,
        candidate_isotope_atom_count=candidate_isotope_atom_count,
    )


def _molecule_charge_counts(mol) -> tuple[int, int]:
    cached = mol_cache_get(mol, "molecule_charge_counts")
    if cached is not None:
        return cached
    formal_charge_atom_count = 0
    abs_formal_charge_sum = 0
    for atom in mol.GetAtoms():
        formal_charge = atom.GetFormalCharge()
        if formal_charge:
            formal_charge_atom_count += 1
            abs_formal_charge_sum += abs(formal_charge)
    return mol_cache_set(
        mol,
        "molecule_charge_counts",
        (formal_charge_atom_count, abs_formal_charge_sum),
    )


def _candidate_has_external_oxyimino_carbonyl_parent(mol, Chem, candidate: NameCandidate) -> bool:
    if not candidate.parent_atoms:
        return False
    parent_atoms = set(candidate.parent_atoms)
    for parent_idx in parent_atoms:
        parent = mol.GetAtomWithIdx(parent_idx)
        for carbonyl in parent.GetNeighbors():
            carbonyl_idx = carbonyl.GetIdx()
            if carbonyl_idx in parent_atoms:
                continue
            bond = mol.GetBondBetweenAtoms(parent_idx, carbonyl_idx)
            if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
                continue
            if _is_external_oxyimino_carbonyl(mol, Chem, carbonyl_idx, blocked_idx=parent_idx):
                return True
    return False


def _candidate_has_parent_oxyazo_carbonyl(mol, Chem, candidate: NameCandidate) -> bool:
    if candidate.family not in _UNSATURATED_ACYCLIC_ALTERNATIVE_FAMILIES:
        return False
    if not candidate.parent_atoms:
        return False
    parent_atoms = set(candidate.parent_atoms)
    for carbonyl_idx in parent_atoms:
        carbonyl = mol.GetAtomWithIdx(carbonyl_idx)
        if carbonyl.GetSymbol() != "C" or carbonyl.GetFormalCharge() != 0 or carbonyl.IsInRing():
            continue
        double_oxygen_count = 0
        single_oxygen_idx = None
        for nbr in carbonyl.GetNeighbors():
            nidx = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(carbonyl_idx, nidx)
            if bond is None:
                continue
            if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
                double_oxygen_count += 1
            elif nidx in parent_atoms and nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.SINGLE:
                single_oxygen_idx = nidx if single_oxygen_idx is None else -1
        if double_oxygen_count != 1 or single_oxygen_idx is None or single_oxygen_idx < 0:
            continue
        oxygen = mol.GetAtomWithIdx(single_oxygen_idx)
        if oxygen.GetFormalCharge() != 0 or oxygen.IsInRing() or oxygen.GetIsotope():
            continue
        for nitrogen in oxygen.GetNeighbors():
            nitrogen_idx = nitrogen.GetIdx()
            if nitrogen_idx == carbonyl_idx or nitrogen_idx not in parent_atoms:
                continue
            if nitrogen.GetSymbol() != "N" or nitrogen.GetFormalCharge() != 0 or nitrogen.IsInRing() or nitrogen.GetIsotope():
                continue
            bond = mol.GetBondBetweenAtoms(single_oxygen_idx, nitrogen_idx)
            if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
                continue
            for distal in nitrogen.GetNeighbors():
                distal_idx = distal.GetIdx()
                if distal_idx == single_oxygen_idx or distal_idx not in parent_atoms:
                    continue
                distal_bond = mol.GetBondBetweenAtoms(nitrogen_idx, distal_idx)
                if (
                    distal.GetSymbol() == "N"
                    and distal.GetFormalCharge() == 0
                    and not distal.IsInRing()
                    and not distal.GetIsotope()
                    and distal_bond is not None
                    and distal_bond.GetBondType() == Chem.BondType.DOUBLE
                ):
                    return True
    return False


def _is_external_oxyimino_carbonyl(mol, Chem, carbonyl_idx: int, *, blocked_idx: int) -> bool:
    carbonyl = mol.GetAtomWithIdx(carbonyl_idx)
    if carbonyl.GetSymbol() != "C" or carbonyl.GetFormalCharge() != 0 or carbonyl.IsInRing():
        return False
    double_oxygen_count = 0
    single_oxygen_idx = None
    for nbr in carbonyl.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == blocked_idx:
            continue
        bond = mol.GetBondBetweenAtoms(carbonyl_idx, nidx)
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_oxygen_count += 1
        elif nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.SINGLE:
            if single_oxygen_idx is not None:
                return False
            single_oxygen_idx = nidx
        elif nbr.GetAtomicNum() > 1:
            return False
    if double_oxygen_count != 1 or single_oxygen_idx is None:
        return False
    oxygen = mol.GetAtomWithIdx(single_oxygen_idx)
    if oxygen.GetFormalCharge() != 0 or oxygen.IsInRing() or oxygen.GetIsotope():
        return False
    attached = [
        nbr.GetIdx()
        for nbr in oxygen.GetNeighbors()
        if nbr.GetIdx() != carbonyl_idx and nbr.GetAtomicNum() > 1
    ]
    if len(attached) != 1:
        return False
    nitrogen_idx = attached[0]
    nitrogen = mol.GetAtomWithIdx(nitrogen_idx)
    if nitrogen.GetSymbol() != "N" or nitrogen.GetFormalCharge() != 0 or nitrogen.IsInRing() or nitrogen.GetIsotope():
        return False
    if mol.GetBondBetweenAtoms(single_oxygen_idx, nitrogen_idx).GetBondType() != Chem.BondType.SINGLE:
        return False
    imine_carbons = []
    for nbr in nitrogen.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == single_oxygen_idx:
            continue
        bond = mol.GetBondBetweenAtoms(nitrogen_idx, nidx)
        if nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.DOUBLE:
            imine_carbons.append(nidx)
        elif nbr.GetAtomicNum() > 1:
            return False
    if len(imine_carbons) != 1:
        return False
    imine_carbon = mol.GetAtomWithIdx(imine_carbons[0])
    return imine_carbon.GetFormalCharge() == 0 and not imine_carbon.IsInRing()


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
    try:
        return _comparable_sort_key_cached(value)
    except TypeError:
        return _comparable_sort_key_uncached(value)


@lru_cache(maxsize=65536)
def _comparable_sort_key_cached(value) -> tuple:
    return _comparable_sort_key_uncached(value)


def _comparable_sort_key_uncached(value) -> tuple:
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


def _candidate_parent_element_count(candidate: NameCandidate, symbol: str) -> int:
    return dict(candidate.parent_element_counts).get(symbol, 0)


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
    if candidate.source_family == _ACYCLIC_ISOTHIOCYANATE_IMINO_THIONE_SOURCE:
        penalty = 0
    else:
        penalty = _opsin_stability_penalty(name)
    if candidate.family == "skeletal-replacement-fused-parent":
        penalty += _skeletal_replacement_fused_parent_penalty(name)
    if candidate.family == "kekulized-generic-polycyclic-parent":
        penalty += _kekulized_polycyclic_late_render_rank_penalty(name, candidate)
    penalty += _kekulized_polycyclic_bridge_unsaturation_penalty(candidate)
    if OPSIN_FRAGILE_OXYIMINO_CARBONYL_PARENT in candidate_safety_reasons(candidate):
        penalty += 80
    if OPSIN_FRAGILE_CHARGED_REPLACEMENT_UNSATURATION in candidate_safety_reasons(candidate):
        penalty += 40
    if OPSIN_FRAGILE_GENERATED_SPIRO_UNSATURATION in candidate_safety_reasons(candidate):
        penalty += 120
    penalty += _unsupported_charge_rendering_penalty(candidate)
    penalty += _unstable_component_spiro_alias_penalty(name, candidate)
    penalty += _unlocanted_multicenter_stereo_penalty(name, candidate)
    return penalty


def _unsupported_charge_rendering_penalty(candidate: NameCandidate) -> int:
    safety = candidate.safety
    if safety is None or safety.charge_supported:
        return 0
    charge = candidate.charge
    if charge is None or charge.rendered_terms:
        return 0
    return 40


def _kekulized_polycyclic_late_render_rank_penalty(name: str, candidate: NameCandidate) -> int:
    if candidate.render_rank is None:
        return 0
    if "tetraaza" in name:
        return 0
    return max(0, candidate.render_rank - 1) * 4


def _kekulized_polycyclic_bridge_unsaturation_penalty(candidate: NameCandidate) -> int:
    safety_reasons = candidate_safety_reasons(candidate)
    if (
        OPSIN_FRAGILE_POLYCYCLIC_BRIDGE_UNSATURATION not in safety_reasons
        and OPSIN_FRAGILE_POLYCYCLIC_BRIDGE_TOPOLOGY not in safety_reasons
    ):
        return 0
    return 180 * max(1, candidate.render_cost - 1)


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
_CHARGED_FUSED_PARENT_ALTERNATIVE_FAMILIES = _FUSED_PARENT_ATOM_BONUS_FAMILIES | {
    "generated-oxido-fused-parent",
    "oxido-fused-retained-parent",
    "skeletal-replacement-fused-parent",
}
_CHARGED_FUSED_PARENT_FALLBACK_PENALTY = 150

_RENDER_RANK_BEFORE_LENGTH_FAMILIES = {
    "component-dispiro-parent",
    "component-spiro-parent",
    "functional-generic-bicyclic-parent",
    "fused-retained-parent",
    "generated-fused-parent",
    "generic-bicyclic-parent",
    "kekulized-generic-polycyclic-parent",
    "poly-component-spiro-parent",
    "retained-topology-replacement-fused-parent",
}

_ACYCLIC_SKELETON_PARENT_FAMILIES = {
    "acyclic-skeleton-with-ring-substituents",
    "chain-dominant-acyclic-skeleton-with-ring-substituents",
    "functional-acyclic-skeleton-with-ring-substituents",
    "imino-functional-acyclic-skeleton-with-ring-substituents",
    "preferred-acyclic-skeleton-with-ring-substituents",
}

_RETAINED_AZABICYCLO320_HEPTANE_ALTERNATIVE_FAMILIES = _ACYCLIC_SKELETON_PARENT_FAMILIES | {
    "carbamate-retained-parent",
    "functional-generic-bicyclic-parent",
    "generic-bicyclic-parent",
    "kekulized-generic-polycyclic-parent",
    "preferred-generic-bicyclic-parent",
    "single-ring-parent",
}

_RETAINED_AZABICYCLO320_HEPTANE_SAME_PARENT_FAMILIES = frozenset(
    {
        "functional-generic-bicyclic-parent",
        "generic-bicyclic-parent",
        "kekulized-generic-polycyclic-parent",
        "preferred-generic-bicyclic-parent",
    }
)

_GENERATED_HYDRO_PYRIDO45D_IMIDAZOLE_BROAD_PARENT_FAMILIES = _ACYCLIC_SKELETON_PARENT_FAMILIES | {
    "generic-bicyclic-parent",
    "retained-amide-parent",
}

_GENERATED_HYDRO_PYRIDO45D_IMIDAZOLE_SIBLING_FAMILIES = frozenset(
    {
        "generated-hydro-fused-parent",
        "generated-partial-hydro-fused-parent",
        "hydro-fused-retained-parent",
    }
)

_GENERATED_HYDRO_PYRIDO45D_IMIDAZOLE_SIBLING_TEMPLATES = frozenset(
    {
        "imidazo[4,5-c]pyridine",
        "imidazo[4,5-d]pyridine",
    }
)

_RETAINED_BENZOCYCLOBUTENE_ALTERNATIVE_FAMILIES = _ACYCLIC_SKELETON_PARENT_FAMILIES | {
    "fused-retained-parent",
    "functional-generic-bicyclic-parent",
    "generated-fused-parent",
    "generated-spiro-fused-parent",
    "generic-bicyclic-parent",
    "generic-spiro-parent",
    "preferred-generic-bicyclic-parent",
    "retained-amide-parent",
    "single-ring-parent",
    "skeletal-replacement-fused-parent",
    "sulfonimidamide-parent",
}

_ACYCLIC_RING_DECOMPOSITION_PARENT_FAMILIES = _ACYCLIC_SKELETON_PARENT_FAMILIES | {
    "charged-ring-acyclic-skeleton-with-ring-substituents",
}
_CHARGED_SINGLE_RING_PARENT_COMPETITOR_FAMILIES = _ACYCLIC_SKELETON_PARENT_FAMILIES | {
    "beta-dicarbonyl-parent",
    "biphenyl-parent",
    "neutral-amine-parent",
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


def _is_exact_high_impact_retained_lactone_parent(candidate: NameCandidate) -> bool:
    if candidate.family != "fused-retained-parent":
        return False
    if candidate.source_family != "static-retained":
        return False
    if candidate.source_file != "high_impact_fused_parent_templates.json":
        return False
    if candidate.template_name not in _RETAINED_LACTONE_TEMPLATE_NAMES:
        return False
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return False
    return not candidate_safety_reasons(candidate)


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

_EXTERNAL_BOND_STEREO_PARENT_ALTERNATIVE_FAMILIES = {
    "carbamate-retained-parent": {
        "acyclic-parent",
        "acyclic-skeleton-with-ring-substituents",
        "single-ring-parent",
    },
    "functional-generic-bicyclic-parent": {"imino-functional-acyclic-skeleton-with-ring-substituents"},
    "fused-retained-parent": {"retained-amide-parent", "single-ring-parent"},
    "generated-fused-parent": {
        "preferred-acyclic-skeleton-with-ring-substituents",
        "single-ring-parent",
    },
    "guanidine-retained-parent": {"acyclic-parent"},
    "phosphorus-acid-parent": {"acyclic-parent"},
    "retained-topology-replacement-fused-parent": {
        "glutarimide-retained-parent",
        "imino-functional-acyclic-skeleton-with-ring-substituents",
        "single-ring-parent",
    },
    "skeletal-replacement-fused-parent": {
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "fused-retained-parent",
        "imino-functional-acyclic-skeleton-with-ring-substituents",
        "single-ring-parent",
    },
    "sulfonamide-retained-parent": {"imino-functional-acyclic-skeleton-with-ring-substituents"},
    "sulfonohydrazide-retained-parent": {"acyclic-skeleton-with-ring-substituents"},
}
_GENERATED_FUSED_EXTERNAL_STEREO_ACYCLIC_TEMPLATE_CAPABLE = frozenset(
    {
        "benzo[1,2-d]pyrazole",
        "benzo[1,2-d]thiazole",
        "imidazo[4,5-a]benzene",
        "pyrimido[4,5-c]pyridine",
        "thieno[2,3-a]benzene",
    }
)
_EXTERNAL_BOND_STEREO_FRAGILITY_PARENT_FAMILIES = frozenset(
    _EXTERNAL_BOND_STEREO_PARENT_ALTERNATIVE_FAMILIES
)
_ALKENE_STEREO_ACYCLIC_RING_PARENT_FAMILIES = frozenset(
    {
        "acyclic-skeleton-with-ring-substituents",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "functional-acyclic-skeleton-with-ring-substituents",
        "preferred-acyclic-skeleton-with-ring-substituents",
    }
)

_GENERATED_FUSED_SAME_ATOMS_RETAINED_TEMPLATE_BLOCKLIST = frozenset({"1H-benzimidazole"})
_PYRIDOPYRROLE_ACYCLIC_DECOMPOSITION_TEMPLATES = frozenset(
    {
        "pyrido[2,3-b]pyrrole",
        "pyrido[3,2-d]pyrrole",
    }
)
_PYRIDOPYRROLE_ACYCLIC_DECOMPOSITION_BONUS = 230
_PYRIDOPYRROLE_ACYCLIC_DECOMPOSITION_MAX_ALTERNATIVE_STABILITY = 292
_BENZIMIDAZOLE_GENERATED_IMIDAZO45A_REPLACEMENT_PENALTY = 100
_BENZIMIDAZOLE_GENERATED_IMIDAZO45A_REPLACEMENT_MIN_LENGTH = 130
_BENZIMIDAZOLE_GENERATED_IMIDAZO45A_REPLACEMENT_MAX_LENGTH = 170
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
_HIGH_CONFIDENCE_IMIDAZO45A_FUSED_TEMPLATE_PARENT_BONUS = 0
_HIGH_CONFIDENCE_FUSED_TEMPLATE_COMPETITOR_STABILITY_MARGIN = 220
_IMIDAZO45A_GENERATED_ORIENTATION_BONUS_BY_RENDER_RANK = {}
_IMIDAZO45A_GENERATED_RANK2_PARENT_BONUS = 0
_BENZOTRIAZOLE_OVER_THIENO23A_PARENT_BONUS = 280
_QUINO34D_PYRAZOLE_SINGLE_RING_PARENT_BONUS = 220
_PYRIDO23D_PYRAZOLE_PARENT_BONUS = 260
_TETRALIN_TO_BENZODIOXANE_TEMPLATE_PAIR_BONUS = 220
_RETAINED_FUSED_TEMPLATE_PAIR_PARENT_BONUS = 430
_RETAINED_FUSED_TEMPLATE_PAIR_COMPETITORS = {
    "1H-benzotriazole": frozenset({"thieno[2,3-a]benzene"}),
    "2H-benzotriazole": frozenset({"thieno[2,3-a]benzene"}),
}
_ACYCLIC_SKELETON_RING_SYSTEM_PARENT_BONUS = 0
_ACYCLIC_SKELETON_RING_SYSTEM_BONUS_FAMILIES = frozenset(
    {
        "acyclic-skeleton-with-ring-substituents",
        "preferred-acyclic-skeleton-with-ring-substituents",
    }
)
_ACYCLIC_SKELETON_RING_SYSTEM_COMPETITOR_FAMILIES = frozenset(
    {
        "generated-fused-parent",
        "generated-hydro-fused-parent",
        "generated-partial-hydro-fused-parent",
        "positive-single-ring-parent",
        "single-ring-parent",
    }
)
_ACYCLIC_SKELETON_PROTECTED_PARENT_FAMILIES = frozenset(
    {
        "carbodiimide-retained-parent",
        "exact-retained-parent",
        "fused-retained-parent",
        "iodane-parent",
        "natural-product-topology-parent",
        "phosphorus-acid-parent",
        "retained-topology-replacement-fused-parent",
        "sydnone-retained-parent",
        "thiooxo-ester-retained-parent",
        "whole-molecule-retained-parent",
    }
)
_KEKULIZED_POLYCYCLE_CONTEXTUAL_PARENT_BONUS = 130
_KEKULIZED_POLYCYCLE_CONTEXTUAL_COMPETITOR_FAMILIES = frozenset(
    {
        "kekulized-generic-polycyclic-parent",
        "oxido-fused-retained-parent",
        "preferred-acyclic-skeleton-with-ring-substituents",
    }
)
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
_LOW_CONFIDENCE_RETAINED_TOPOLOGY_TEMPLATE_TO_TEMPLATELESS_FAMILIES = {
    "acenaphthen": _ACYCLIC_SKELETON_PARENT_FAMILIES,
    "anthranil": _ACYCLIC_SKELETON_PARENT_FAMILIES,
    "benzoxepane": _ACYCLIC_SKELETON_PARENT_FAMILIES,
    "coumaron": _ACYCLIC_SKELETON_PARENT_FAMILIES,
    "coumarone": _ACYCLIC_SKELETON_PARENT_FAMILIES,
    "fluoren": _ACYCLIC_SKELETON_PARENT_FAMILIES,
    "indane": _ACYCLIC_SKELETON_PARENT_FAMILIES,
    "naphthalin": _ACYCLIC_SKELETON_PARENT_FAMILIES,
    "phenanthren": _ACYCLIC_SKELETON_PARENT_FAMILIES,
    "tetralin": _ACYCLIC_SKELETON_PARENT_FAMILIES,
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
_LOW_CONFIDENCE_RETAINED_TOPOLOGY_PARENT_PENALTY = 430
_GENERATED_PARTIAL_HYDRO_TEMPLATELESS_PARENT_FAMILIES = (
    _ACYCLIC_SKELETON_PARENT_FAMILIES
    | {
        "functional-generic-bicyclic-parent",
        "generic-bicyclic-parent",
        "generic-spiro-parent",
        "kekulized-generic-polycyclic-parent",
        "single-ring-parent",
    }
)
_GENERATED_PARTIAL_HYDRO_TEMPLATELESS_PARENT_PENALTY = 180
_LOW_CONFIDENCE_INDAN_TOPOLOGY_TEMPLATELESS_PARENT_PENALTY = 180
_PARTIAL_HYDRO_DEMOTION_HYDRO_QUINOLINE_TEMPLATES = frozenset(
    {
        "1,2,3,4-tetrahydroquinoline",
    }
)
_RANKER_ACYCLIC_GENERATED_HYDRO_FUSED_TEMPLATES = frozenset(
    {
        "pyrido[2,3-a]benzene",
        "pyrido[3,2-b]benzene",
    }
)
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
    if candidate.family == "single-ring-parent" and candidate.parent_external_relative_stereo_branch_count:
        pass
    elif candidate.family not in _RENDER_RANK_BEFORE_LENGTH_FAMILIES:
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
    if item.candidate.source_family == "acyclic-carbohydrate-retained":
        return 0
    if item.candidate.source_family == _ACYCLIC_ISOTHIOCYANATE_IMINO_THIONE_SOURCE:
        return 0
    if _is_exact_high_impact_retained_lactone_parent(item.candidate):
        return 0
    charged_fused_parent_penalty = _contextual_charged_fused_parent_fallback_penalty(item, prepared)
    if item.candidate.family in _ACYCLIC_SKELETON_PARENT_FAMILIES:
        penalty = 0
        if _has_contextual_single_ring_parent_candidate_alternative(item, prepared):
            penalty = max(penalty, 55)
        if _has_contextual_terminal_ring_methan_parent_alternative(item, prepared):
            penalty = max(penalty, 142)
        if _has_contextual_integrated_ring_parent_alternative(item, prepared):
            penalty = max(penalty, 142)
        penalty = max(penalty, charged_fused_parent_penalty)
        return penalty
    if charged_fused_parent_penalty and item.candidate.family in {
        "single-ring-parent",
        "positive-single-ring-parent",
    }:
        return charged_fused_parent_penalty
    if item.candidate.family not in _BROAD_GRAPH_PARENT_FRAGILE_WINNER_FAMILIES:
        return 0
    if item.candidate.family in {"single-ring-parent", "positive-single-ring-parent"}:
        if not _has_contextual_non_acyclic_broad_graph_parent_alternative(item, prepared):
            return 0
    if not _has_contextual_broad_graph_parent_alternative(item, prepared):
        return 0
    return max(70, charged_fused_parent_penalty)


def _contextual_charged_fused_parent_fallback_penalty(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> int:
    if item.candidate.family not in (
        _ACYCLIC_RING_DECOMPOSITION_PARENT_FAMILIES
        | {"single-ring-parent", "positive-single-ring-parent"}
    ):
        return 0
    item_charged_atoms = _candidate_charged_atom_set(item.candidate)
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family not in _CHARGED_FUSED_PARENT_ALTERNATIVE_FAMILIES:
            continue
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        other_charged_atoms = _candidate_charged_atom_set(other_candidate)
        if not other_charged_atoms or not other_charged_atoms - item_charged_atoms:
            continue
        if len(other.name) > 260 or len(other.name) - len(item.name) > 180:
            continue
        if other.stability_penalty > item.stability_penalty + 160:
            continue
        return _CHARGED_FUSED_PARENT_FALLBACK_PENALTY
    return 0


def _candidate_charged_atom_set(candidate: NameCandidate) -> set[int]:
    if candidate.charge is None:
        return set()
    return set(candidate.charge.charged_atoms)


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


def _contextual_pyridopyrrole_acyclic_decomposition_bonus(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> int:
    candidate = item.candidate
    if candidate.family != "acyclic-skeleton-with-ring-substituents":
        return 0
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return 0
    if len(candidate.parent_atoms) < 7 or len(candidate.parent_atoms) > 22:
        return 0
    if candidate.parent_hetero_atom_count:
        return 0
    if candidate.suffixes or candidate.charge is not None:
        return 0
    if candidate_safety_reasons(candidate):
        return 0
    alternative_stability = item.stability_penalty + _candidate_selection_broad_graph_parent_penalty(
        item,
        prepared,
    )
    if alternative_stability > _PYRIDOPYRROLE_ACYCLIC_DECOMPOSITION_MAX_ALTERNATIVE_STABILITY:
        return 0
    if not _has_contextual_generated_pyridopyrrole_parent_competitor(item, prepared):
        return 0
    return _PYRIDOPYRROLE_ACYCLIC_DECOMPOSITION_BONUS


def _has_contextual_generated_pyridopyrrole_parent_competitor(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "generated-fused-parent":
            continue
        if other_candidate.template_name not in _PYRIDOPYRROLE_ACYCLIC_DECOMPOSITION_TEMPLATES:
            continue
        if other_candidate.source_family != "generated-fused":
            continue
        if other_candidate.source_file != "generated_fused_template_index.json":
            continue
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        if len(other_candidate.parent_atoms) != 9:
            continue
        if other_candidate.parent_hetero_atom_count != 2:
            continue
        if other_candidate.parent_external_oxygen_substituent_count:
            continue
        if (
            other_candidate.parent_external_nitrogen_substituent_count
            and other_candidate.parent_substituted_hetero_atom_count
            and not other_candidate.parent_terminal_hetero_substituent_count
        ):
            continue
        if other_candidate.suffixes or other_candidate.charge is not None:
            continue
        if candidate_safety_reasons(other_candidate):
            continue
        if other_candidate.render_rank is None or other_candidate.render_rank > 2:
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


def _has_contextual_exact_high_impact_retained_lactone_parent(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    candidate_parent_count = len(candidate.parent_atoms)
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if not _is_exact_high_impact_retained_lactone_parent(other_candidate):
            continue
        if candidate_parent_count and len(candidate.parent_atoms & other_candidate.parent_atoms) < min(
            candidate_parent_count,
            len(other_candidate.parent_atoms),
        ):
            continue
        if item.stability_penalty >= other.stability_penalty + 100:
            continue
        return True
    return False


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


def _aromatic_single_ring_alternative_for_alkene_stereo_acyclic_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    candidate = selected.candidate
    if candidate.family not in _ALKENE_STEREO_ACYCLIC_RING_PARENT_FAMILIES:
        return None
    if candidate.stereo is None or not candidate.stereo.bond_descriptors:
        return None
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return None
    if candidate.charge is not None or candidate.suffixes:
        return None
    if candidate.molecule_abs_formal_charge_sum:
        return None
    if candidate_safety_reasons(candidate):
        return None
    if len(candidate.parent_atoms) < 9:
        return None

    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None

    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, other in keyed_items:
        if other is selected or other.name == selected.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "single-ring-parent":
            continue
        if other_candidate.stereo is None or not other_candidate.stereo.bond_descriptors:
            continue
        if other_candidate.charge is not None or other_candidate.suffixes:
            continue
        if other_candidate.molecule_abs_formal_charge_sum:
            continue
        if candidate_safety_reasons(other_candidate):
            continue
        if not other_candidate.parent_atoms or not candidate_has_complete_graph(other_candidate):
            continue
        other_parent_count = len(other_candidate.parent_atoms)
        if other_parent_count not in {5, 6}:
            continue
        if other_candidate.parent_aromatic_atom_count != other_parent_count:
            continue
        if other_candidate.parent_hetero_atom_count > 2:
            continue
        if other_candidate.priority == 61:
            if other_candidate.parent_terminal_hetero_substituent_count:
                continue
            if other_candidate.parent_external_oxygen_substituent_count:
                continue
            if other_candidate.parent_hetero_atom_count and other_parent_count != 5:
                continue
            if other_candidate.parent_hetero_atom_count > 1:
                continue
        elif other_candidate.priority == 30:
            if other_candidate.parent_hetero_atom_count and other_parent_count != 5:
                continue
            if other_candidate.parent_external_nitrogen_substituent_count:
                continue
            if candidate_key[0] < 200 and len(candidate.parent_atoms) < 13:
                continue
            if (
                other_candidate.render_rank is not None
                and other_candidate.render_rank > 1
                and other.selection_bonus <= 0
            ):
                continue
            if other_candidate.parent_terminal_hetero_substituent_count and candidate_key[0] < 200:
                continue
            if (
                other_candidate.parent_external_oxygen_substituent_count
                and candidate_key[0] < 200
                and other_candidate.render_rank != 1
            ):
                continue
        else:
            continue
        if other_candidate.external_bond_stereo_parent_fragile:
            continue
        if other_candidate.render_rank is not None and other_candidate.render_rank > 3:
            continue
        if len(other.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(other.name) - len(selected.name) > 130:
            continue
        if candidate_key[0] > selected_key[0] + 260:
            continue
        if other.stability_penalty > selected.stability_penalty + 120:
            continue
        if other_candidate.priority > candidate.priority + 80:
            continue
        alternatives.append(
            (
                (
                    candidate_key[0],
                    _candidate_selection_render_rank_key(other_candidate),
                    len(other.name) - other.selection_bonus,
                    len(other.name),
                    candidate_key,
                ),
                other,
            )
        )
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _functional_acyclic_alternative_for_external_stereo_carbocyclic_single_ring_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    candidate = selected.candidate
    if candidate.family != "single-ring-parent":
        return None
    if candidate.priority != 30 or (candidate.render_rank or 0) <= 1:
        return None
    if candidate.stereo is None or not candidate.stereo.bond_descriptors:
        return None
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return None
    parent_count = len(candidate.parent_atoms)
    if parent_count != 6 or candidate.parent_aromatic_atom_count != parent_count:
        return None
    if candidate.parent_hetero_atom_count or candidate.parent_terminal_hetero_substituent_count:
        return None
    if candidate.charge is not None or candidate.suffixes or candidate.prefixes:
        return None
    if candidate.molecule_abs_formal_charge_sum or candidate_safety_reasons(candidate):
        return None

    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None

    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, other in keyed_items:
        if other is selected or other.name == selected.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "functional-acyclic-skeleton-with-ring-substituents":
            continue
        if other_candidate.priority >= candidate.priority:
            continue
        if other_candidate.stereo is None or not other_candidate.stereo.bond_descriptors:
            continue
        if not other_candidate.parent_atoms or not candidate_has_complete_graph(other_candidate):
            continue
        if len(other_candidate.parent_atoms) < parent_count + 4:
            continue
        if other_candidate.charge is not None or other_candidate.suffixes or other_candidate.prefixes:
            continue
        if other_candidate.molecule_abs_formal_charge_sum or candidate_safety_reasons(other_candidate):
            continue
        if len(other.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(other.name) - len(selected.name) > 45:
            continue
        if candidate_key[0] > selected_key[0] + 90:
            continue
        if other.stability_penalty > selected.stability_penalty + 90:
            continue
        alternatives.append(
            (
                (
                    candidate_key[0],
                    other_candidate.priority,
                    len(other.name) - other.selection_bonus,
                    len(other.name),
                    candidate_key,
                ),
                other,
            )
        )
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _internal_bond_stereo_parent_alternative_for_fragile_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    candidate = selected.candidate
    allowed_families = _EXTERNAL_BOND_STEREO_PARENT_ALTERNATIVE_FAMILIES.get(candidate.family)
    if not allowed_families:
        return None
    if not candidate.external_bond_stereo_parent_fragile:
        return None
    if (
        candidate.family == "generated-fused-parent"
        and not _generated_fused_external_stereo_acyclic_switch_supported(candidate)
    ):
        return None
    if candidate.stereo is None or not candidate.stereo.bond_descriptors:
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None
    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, other in keyed_items:
        if other is selected or other.name == selected.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family not in allowed_families:
            continue
        if other_candidate.stereo is None or not other_candidate.stereo.bond_descriptors:
            continue
        if other_candidate.external_bond_stereo_parent_fragile:
            continue
        if (
            other_candidate.family == "single-ring-parent"
            and _external_bond_stereo_single_ring_switch_blocked(candidate, other_candidate)
        ):
            continue
        if other_candidate.family == "single-ring-parent" and other_candidate.suffixes:
            continue
        if not other_candidate.parent_atoms or not candidate_has_complete_graph(other_candidate):
            continue
        if len(other.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(other.name) - len(selected.name) > 130:
            continue
        if candidate_key[0] > selected_key[0] + 260:
            continue
        if other_candidate.priority > candidate.priority + 70:
            continue
        alternatives.append((candidate_key, other))
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _external_bond_stereo_single_ring_switch_blocked(
    selected_candidate: NameCandidate,
    proposed_candidate: NameCandidate,
) -> bool:
    if _stable_fused_parent_blocks_unsubstituted_diaza_single_ring_switch(
        selected_candidate,
        proposed_candidate,
    ):
        return True
    if selected_candidate.family == "fused-retained-parent":
        if selected_candidate.template_name in {"benzocyclobutene", "inden", "naphthalen"}:
            return True
        if selected_candidate.template_name == "indan" and (selected_candidate.render_rank or 0) > 3:
            return True
    if (
        selected_candidate.family == "generated-fused-parent"
        and selected_candidate.template_name == "benzo[1,2-d]pyrazole"
        and (proposed_candidate.render_rank or 0) > 1
    ):
        return True
    if (
        selected_candidate.family == "retained-topology-replacement-fused-parent"
        and selected_candidate.template_name == "indan"
        and selected_candidate.molecule_abs_formal_charge_sum
    ):
        return True
    if (
        selected_candidate.family == "retained-topology-replacement-fused-parent"
        and selected_candidate.template_name == "1-azabicyclo[3.2.0]heptane"
    ):
        return True
    return False


def _stable_fused_parent_blocks_unsubstituted_diaza_single_ring_switch(
    selected_candidate: NameCandidate,
    proposed_candidate: NameCandidate,
) -> bool:
    if selected_candidate.family != "fused-retained-parent":
        return False
    if selected_candidate.template_name not in {"indolin", "oxindol"}:
        return False
    if selected_candidate.source_family != "arylGroups:ring:ring":
        return False
    if selected_candidate.priority != 18 or selected_candidate.render_rank != 1:
        return False
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return False
    if len(selected_candidate.parent_atoms) not in {9, 10}:
        return False
    if selected_candidate.parent_hetero_atom_count:
        return False
    if selected_candidate.charge is not None or selected_candidate.suffixes or selected_candidate.prefixes:
        return False
    if candidate_safety_reasons(selected_candidate):
        return False
    if proposed_candidate.family != "single-ring-parent":
        return False
    if proposed_candidate.priority != 61 or proposed_candidate.render_rank != 1:
        return False
    if not proposed_candidate.parent_atoms or not candidate_has_complete_graph(proposed_candidate):
        return False
    if len(proposed_candidate.parent_atoms) != 6:
        return False
    if proposed_candidate.parent_hetero_atom_count < 2:
        return False
    if proposed_candidate.parent_external_substituted_hetero_atom_count >= proposed_candidate.parent_hetero_atom_count:
        return False
    if proposed_candidate.charge is not None or proposed_candidate.suffixes or proposed_candidate.prefixes:
        return False
    if candidate_safety_reasons(proposed_candidate):
        return False
    return True


def _generated_fused_external_stereo_acyclic_switch_supported(candidate: NameCandidate) -> bool:
    if candidate.source_family != "generated-fused":
        return False
    if candidate.source_file != "generated_fused_template_index.json":
        return False
    if candidate.template_name not in _GENERATED_FUSED_EXTERNAL_STEREO_ACYCLIC_TEMPLATE_CAPABLE:
        return False
    if candidate.render_rank != 1:
        return False
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return False
    if len(candidate.parent_atoms) not in {9, 10}:
        return False
    if candidate.parent_aromatic_atom_count != len(candidate.parent_atoms):
        return False
    if candidate.suffixes:
        return False
    if candidate.molecule_abs_formal_charge_sum:
        return False
    return True


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
        if _is_matching_retained_purin_candidate(candidate, other_candidate) and _is_unanchored_indicated_h_generated_purine_candidate(
            other_candidate,
        ):
            continue
        if len(other.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(other.name) - len(item.name) > 90:
            continue
        if other.stability_penalty >= item.stability_penalty + 80:
            continue
        return True
    return False


def _has_contextual_generated_heterofused_parent_alternative_for_retained_fragment(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family not in {"fused-retained-parent", "hydro-fused-retained-parent"}:
        return False
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return False
    if candidate.ring_system_atoms:
        return False
    if candidate.suffixes or candidate.charge is not None:
        return False
    if candidate.render_rank is not None and candidate.render_rank > 2:
        return False
    parent_count = len(candidate.parent_atoms)
    if parent_count < 8 or parent_count > 12:
        return False
    min_generated_parent_count = parent_count - 1
    if candidate.source_family == "arylGroups:ring:biochemical":
        min_generated_parent_count = parent_count - 2
    if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "generated-fused-parent":
            continue
        if other_candidate.source_family != "generated-fused":
            continue
        if other_candidate.source_file != "generated_fused_template_index.json":
            continue
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        if other_candidate.parent_atoms == candidate.parent_atoms:
            continue
        other_parent_count = len(other_candidate.parent_atoms)
        if other_parent_count < min_generated_parent_count or other_parent_count > parent_count + 1:
            continue
        if not other_candidate.ring_system_atoms:
            continue
        if len(other_candidate.ring_system_atoms) != other_parent_count:
            continue
        if other_candidate.parent_aromatic_atom_count != other_parent_count:
            continue
        if other_candidate.parent_hetero_atom_count < 2:
            continue
        if (
            other_candidate.parent_terminal_hetero_substituent_count
            <= candidate.parent_terminal_hetero_substituent_count
        ):
            continue
        if other_candidate.render_rank is None or other_candidate.render_rank > 4:
            continue
        if other_candidate.suffixes or other_candidate.charge is not None:
            continue
        if candidate_safety_reasons(other_candidate):
            continue
        if len(other.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(other.name) - len(item.name) > 45:
            continue
        if other.stability_penalty > item.stability_penalty + 80:
            continue
        return True
    return False


def _has_contextual_low_rank_generated_fused_parent_alternative(
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
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return False
    parent_count = len(candidate.parent_atoms)
    if parent_count < 6:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "generated-fused-parent":
            continue
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        other_parent_count = len(other_candidate.parent_atoms)
        if other_parent_count < max(6, parent_count - 1):
            continue
        if other_parent_count > parent_count + 1:
            continue
        same_ring = bool(
            candidate.ring_system_atoms
            and other_candidate.ring_system_atoms == candidate.ring_system_atoms
        )
        parent_overlap = len(candidate.parent_atoms & other_candidate.parent_atoms)
        if not same_ring and parent_overlap < min(parent_count, other_parent_count, 8):
            continue
        if other_candidate.render_rank is None or other_candidate.render_rank > 3:
            continue
        if _is_matching_retained_purin_candidate(candidate, other_candidate) and _is_unanchored_indicated_h_generated_purine_candidate(
            other_candidate,
        ):
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


def _has_contextual_retained_quinuclidin_parent_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family == "fused-retained-parent" and candidate.template_name == "quinuclidin":
        return False
    if candidate.family not in {
        "acyclic-skeleton-with-ring-substituents",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "charged-ring-acyclic-skeleton-with-ring-substituents",
        "functional-acyclic-skeleton-with-ring-substituents",
        "functional-generic-bicyclic-parent",
        "generic-bicyclic-parent",
        "preferred-generic-bicyclic-parent",
    }:
        return False
    if not candidate_has_complete_graph(candidate):
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "fused-retained-parent":
            continue
        if other_candidate.template_name != "quinuclidin":
            continue
        if not candidate_has_complete_graph(other_candidate):
            continue
        if candidate_safety_reasons(other_candidate):
            continue
        if len(other_candidate.parent_atoms) != 8:
            continue
        render_rank = other_candidate.render_rank or 0
        if render_rank < 1 or render_rank > 4:
            continue
        if len(other.name) > 180 or len(other.name) - len(item.name) > 80:
            continue
        if other.stability_penalty >= item.stability_penalty + 120:
            continue
        return True
    return False


def _has_contextual_preferred_compact_bicyclic_parent_alternative_for_acyclic_decomposition(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family not in _ACYCLIC_SKELETON_PARENT_FAMILIES:
        return False
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return False
    if len(candidate.parent_atoms) < 10:
        return False
    if candidate.suffixes or candidate.charge is not None:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "preferred-generic-bicyclic-parent":
            continue
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        if len(other_candidate.parent_atoms) != 6:
            continue
        if other_candidate.render_rank is None or other_candidate.render_rank > 2:
            continue
        if other_candidate.suffixes or other_candidate.charge is not None:
            continue
        if len(other.name) > _DEFAULT_MAX_NAME_LENGTH or len(other.name) - len(item.name) > 170:
            continue
        if other.stability_penalty >= item.stability_penalty + 100:
            continue
        return True
    return False


def _has_contextual_generated_hydro_pyridino_imidazole_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.template_name != "imidazo[4,5-c]pyridine":
        return False
    if candidate.family not in {
        "generated-hydro-fused-parent",
        "generated-partial-hydro-fused-parent",
        "hydro-fused-retained-parent",
    }:
        return False
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return False
    if len(candidate.parent_atoms) != 9:
        return False
    if candidate.suffixes or candidate.charge is not None:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if not _is_guarded_generated_hydro_pyridino_imidazole_parent(other_candidate):
            continue
        if other_candidate.parent_atoms != candidate.parent_atoms:
            continue
        if len(other.name) > _DEFAULT_MAX_NAME_LENGTH or len(other.name) - len(item.name) > 60:
            continue
        if other.stability_penalty >= item.stability_penalty + 120:
            continue
        return True
    return False


def _has_contextual_retained_azabicyclo320_heptane_parent_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if _is_guarded_retained_azabicyclo320_heptane_parent(candidate):
        return False
    if candidate.family not in _RETAINED_AZABICYCLO320_HEPTANE_ALTERNATIVE_FAMILIES:
        return False
    if candidate.suffixes or candidate.charge is not None:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if not _is_guarded_retained_azabicyclo320_heptane_parent(other_candidate):
            continue
        if candidate.parent_atoms:
            candidate_parent_atoms = set(candidate.parent_atoms)
            other_parent_atoms = set(other_candidate.parent_atoms)
            if candidate.family in _RETAINED_AZABICYCLO320_HEPTANE_SAME_PARENT_FAMILIES:
                if candidate_parent_atoms != other_parent_atoms:
                    continue
            elif len(candidate_parent_atoms & other_parent_atoms) < 5:
                continue
        if len(other.name) > _DEFAULT_MAX_NAME_LENGTH or len(other.name) - len(item.name) > 170:
            continue
        if other.stability_penalty >= item.stability_penalty + 160:
            continue
        return True
    return False


def _has_contextual_retained_benzocyclobutene_parent_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if _is_guarded_retained_benzocyclobutene_parent(candidate):
        return False
    if candidate.family not in _RETAINED_BENZOCYCLOBUTENE_ALTERNATIVE_FAMILIES:
        return False
    if candidate.charge is not None:
        return False
    if candidate.suffixes and candidate.family != "single-ring-parent":
        return False
    if candidate_safety_reasons(candidate):
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if not _is_guarded_retained_benzocyclobutene_parent(other_candidate):
            continue
        if len(other.name) > _DEFAULT_MAX_NAME_LENGTH or len(other.name) - len(item.name) > 120:
            continue
        if other.stability_penalty >= item.stability_penalty + 220:
            continue
        return True
    return False


def _has_contextual_generated_hydro_pyrido45d_imidazole_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    broad_shortcut_family = candidate.family in _GENERATED_HYDRO_PYRIDO45D_IMIDAZOLE_BROAD_PARENT_FAMILIES
    compact_carbocyclic_fused_parent = (
        candidate.family == "fused-retained-parent"
        and not candidate.source_family
        and not candidate.source_file
        and candidate_has_complete_graph(candidate)
        and bool(candidate.parent_atoms)
        and len(candidate.parent_atoms) <= 9
        and candidate.parent_aromatic_atom_count == 0
        and candidate.parent_hetero_atom_count == 0
        and not candidate.suffixes
        and candidate.charge is None
        and not candidate_safety_reasons(candidate)
    )
    if candidate.family == "kekulized-generic-polycyclic-parent":
        if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
            return False
        if len(candidate.parent_atoms) != 9:
            return False
    elif not broad_shortcut_family and not compact_carbocyclic_fused_parent:
        if candidate.template_name not in _GENERATED_HYDRO_PYRIDO45D_IMIDAZOLE_SIBLING_TEMPLATES:
            return False
        if candidate.family not in _GENERATED_HYDRO_PYRIDO45D_IMIDAZOLE_SIBLING_FAMILIES:
            return False
    if (
        not broad_shortcut_family
        and not compact_carbocyclic_fused_parent
        and (not candidate_has_complete_graph(candidate) or not candidate.parent_atoms)
    ):
        return False
    if not broad_shortcut_family and not compact_carbocyclic_fused_parent and len(candidate.parent_atoms) != 9:
        return False
    if (
        not broad_shortcut_family
        and not compact_carbocyclic_fused_parent
        and (candidate.suffixes or candidate.charge is not None)
    ):
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if not _is_guarded_generated_hydro_pyrido45d_imidazole_parent(other_candidate):
            continue
        if (
            not broad_shortcut_family
            and not compact_carbocyclic_fused_parent
            and set(other_candidate.parent_atoms) != set(candidate.parent_atoms)
        ):
            continue
        if len(other.name) > _DEFAULT_MAX_NAME_LENGTH or len(other.name) - len(item.name) > 80:
            continue
        if other.stability_penalty >= item.stability_penalty + 120:
            continue
        return True
    return False


def _has_contextual_complete_nitrile_oxide_parent_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family == "nitrile-oxide-retained-parent":
        return False
    if not candidate_has_complete_graph(candidate):
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "nitrile-oxide-retained-parent":
            continue
        if not candidate_has_complete_graph(other_candidate):
            continue
        if len(other.name) - len(item.name) > 100:
            continue
        return True
    return False


def _has_contextual_retained_carbohydrate_substituent_single_ring_sibling(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family != "single-ring-parent":
        return False
    if candidate.source_family == _SINGLE_RING_RETAINED_CARBOHYDRATE_SUBSTITUENT_SOURCE:
        return False
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return False
    if candidate.parent_aromatic_atom_count or candidate.parent_hetero_atom_count != 1:
        return False
    if candidate.parent_external_oxygen_substituent_count < 1:
        return False
    if len(candidate.parent_atoms) not in {5, 6} or len(candidate.stereo_locants) < 2:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "single-ring-parent":
            continue
        if other_candidate.source_family != _SINGLE_RING_RETAINED_CARBOHYDRATE_SUBSTITUENT_SOURCE:
            continue
        if not other_candidate.parent_atoms or not candidate_has_complete_graph(other_candidate):
            continue
        if other_candidate.parent_atoms == candidate.parent_atoms:
            continue
        if other_candidate.parent_aromatic_atom_count or other_candidate.parent_hetero_atom_count != 1:
            continue
        if other_candidate.parent_external_oxygen_substituent_count < 1:
            continue
        if len(other_candidate.parent_atoms) not in {5, 6} or len(other_candidate.stereo_locants) < 2:
            continue
        if item.stability_penalty > other.stability_penalty + 100:
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH or len(item.name) - len(other.name) > 180:
            continue
        return True
    return False


def _has_contextual_higher_isotope_coverage_parent_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.molecule_isotope_atom_count <= 0:
        return False
    candidate_mentions_isotope = _candidate_name_mentions_isotope(item.name)
    if not candidate.parent_atoms and candidate_mentions_isotope:
        return False
    if candidate.parent_atoms and candidate_has_complete_graph(candidate):
        pass
    elif not _is_graph_incomplete_isotope_shortcut(candidate):
        return False
    if candidate.candidate_isotope_atom_count != 0 and candidate_mentions_isotope:
        return False
    if len(item.name) > 80:
        return False
    if (
        candidate.family == "single-ring-parent"
        and candidate.parent_hetero_atom_count >= 2
        and candidate.parent_aromatic_atom_count >= 5
    ):
        return False
    if candidate.family in {
        "high-signal-generated-topology-fused-parent",
        "retained-topology-replacement-fused-parent",
    }:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if not other_candidate.parent_atoms or not candidate_has_complete_graph(other_candidate):
            continue
        if candidate_safety_reasons(other_candidate):
            continue
        if other_candidate.candidate_isotope_atom_count < candidate.molecule_isotope_atom_count:
            continue
        if not _candidate_name_mentions_isotope(other.name):
            continue
        if other.stability_penalty > item.stability_penalty + 320:
            continue
        if len(other.name) > 220 or len(other.name) - len(item.name) > 140:
            continue
        if other_candidate.render_rank is not None and other_candidate.render_rank > 5:
            continue
        return True
    return False


def _isotope_preserving_alternative_for_selected_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    candidate = selected.candidate
    if candidate.molecule_isotope_atom_count <= 0:
        return None
    selected_mentions_isotope = _candidate_name_mentions_isotope(selected.name)
    selected_has_complete_coverage = (
        candidate.candidate_isotope_atom_count >= candidate.molecule_isotope_atom_count
    )
    if selected_mentions_isotope and selected_has_complete_coverage:
        return None
    alternatives: list[tuple[int, tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        other_candidate = item.candidate
        if other_candidate.molecule_isotope_atom_count != candidate.molecule_isotope_atom_count:
            continue
        if not _candidate_name_mentions_isotope(item.name):
            continue
        if selected_mentions_isotope:
            if other_candidate.family not in {
                "exact-biochemical-retained-parent",
                "whole-molecule-retained-parent",
            }:
                continue
            if other_candidate.source_family == "opsin-exact-unlocanted":
                continue
            if other_candidate.molecule_isotope_atom_count > 10:
                continue
            if other_candidate.candidate_isotope_atom_count < candidate.molecule_isotope_atom_count:
                continue
            if not other_candidate.parent_atoms or not candidate_has_complete_graph(other_candidate):
                continue
        if candidate_safety_reasons(other_candidate):
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(item.name) - len(selected.name) > 140:
            continue
        if item.stability_penalty > selected.stability_penalty + 360:
            continue
        alternatives.append((len(item.name), candidate_key, item))
    if not alternatives:
        return None
    return min(alternatives, key=lambda entry: (entry[0], entry[1]))[2]


_ACYCLIC_RING_SUBSTITUENT_PARENT_FAMILIES = frozenset(
    {
        "acyclic-skeleton-with-ring-substituents",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "functional-acyclic-skeleton-with-ring-substituents",
        "imino-functional-acyclic-skeleton-with-ring-substituents",
        "preferred-acyclic-skeleton-with-ring-substituents",
        "sulfur-oxo-acyclic-skeleton-with-ring-substituents",
    }
)


def _retained_amide_alternative_for_acyclic_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family not in _ACYCLIC_RING_SUBSTITUENT_PARENT_FAMILIES:
        return None
    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "retained-amide-parent":
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        if candidate_safety_reasons(candidate):
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(item.name) - len(selected.name) > 80:
            continue
        if item.stability_penalty > selected.stability_penalty + 260:
            continue
        alternatives.append((candidate_key, item))
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _candidate_name_mentions_isotope(name: str) -> bool:
    return bool(
        re.search(
            r"(?:^|[^A-Za-z0-9])(?:2H|3H|13C|14C|15N|18O|32P|33P|34S|35S|36Cl|37Cl|125I|131I)",
            name,
        )
        or "deuterio" in name
        or "tritio" in name
    )


def _is_graph_incomplete_isotope_shortcut(candidate: NameCandidate) -> bool:
    if candidate.parent_atoms:
        return False
    if candidate_has_complete_graph(candidate):
        return False
    if candidate.prefixes or candidate.suffixes or candidate.charge is not None:
        return False
    return candidate.family in {
        "acyclic-parent",
        "amino-acid-retained-parent",
        "carbamate-retained-parent",
        "exact-functional-retained-parent",
        "glutarimide-retained-parent",
        "neutral-amine-parent",
        "oxo-carboxylate-retained-parent",
        "phosphate-ester-retained-parent",
        "phosphite-ester-parent",
        "retained-acid-ester-parent",
        "retained-amide-parent",
        "retained-ester-parent",
        "sulfonamide-retained-parent",
        "terminal-carboxylate-parent",
    }


def _candidate_selection_contextual_penalty(item: _PreparedCandidate, prepared: list[_PreparedCandidate]) -> int:
    """Defer orientation-fragile aliases only when alternatives exist."""

    if item.candidate.source_family == "acyclic-carbohydrate-retained":
        return 0
    if _is_exact_high_impact_retained_lactone_parent(item.candidate):
        return 0
    if _is_high_confidence_amino_acid_retained_parent(item.candidate):
        return 0
    name = item.name
    retained_lactone_topology = _has_contextual_retained_lactone_topology_candidate(item, prepared)
    exact_high_impact_lactone_parent = _has_contextual_exact_high_impact_retained_lactone_parent(
        item,
        prepared,
    )
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
    retained_purine_chalcogen_tautomer_parent = (
        _has_contextual_purin_parent_alternative_for_chalcogen_retained_purine_alias(item, prepared)
    )
    retained_purine_generated_parent = _has_contextual_generated_purine_parent_alternative(item, prepared)
    retained_purine_unanchored_generated_parent = (
        _has_contextual_unanchored_generated_purine_parent_alternative_for_retained_purin(item, prepared)
    )
    substituted_pterin_with_pteridin_parent = (
        _has_contextual_pteridin_parent_alternative_for_substituted_pterin(item, prepared)
    )
    retained_imidazopyridine_benzazole_parent = _has_contextual_generated_benzazole_parent_alternative(
        item,
        prepared,
    )
    retained_imidazopyridine_with_first_indazole_h_parent = (
        _has_contextual_first_generated_indazole_h_orientation_alternative(item, prepared)
    )
    incomplete_organophosphorus_with_complete_purine_parent = (
        _has_contextual_complete_purine_parent_alternative_for_incomplete_organophosphorus(item, prepared)
    )
    purine_parent_with_acyclic_phosphorus_parent = (
        _has_contextual_acyclic_phosphorus_parent_alternative_for_purine_parent(item, prepared)
    )
    preferred_generated_fused_template = _has_contextual_preferred_generated_fused_template_alternative(
        item,
        prepared,
    )
    fragile_terminal_azabicyclo_benzimidazole = _has_fragile_terminal_azabicyclo_benzimidazole_parent(
        name,
        item.candidate,
    )
    fragile_group14_hetero_ligand = _has_fragile_group14_hetero_carbon_ligand(item.candidate)
    generic_spiro_parent = _has_fragile_generic_spiro_parent(name, item.candidate)
    generated_spiro_functional_chain = _has_contextual_generated_spiro_functional_chain_alternative(item, prepared)
    complete_spiro_functional_chain = _has_contextual_complete_spiro_functional_chain_alternative(item, prepared)
    compact_bicyclic_parent_for_spiro_chain = _has_contextual_compact_bicyclic_parent_alternative_for_spiro_chain(
        item,
        prepared,
    )
    compact_functional_bicyclic_parent_for_chain_decomposition = (
        _has_contextual_compact_functional_bicyclic_parent_alternative_for_chain_decomposition(item, prepared)
    )
    single_ring_lactam_parent = _has_contextual_single_ring_lactam_parent_alternative(item, prepared)
    fused_retained_parent_for_generic_spiro = (
        _has_contextual_fused_retained_parent_alternative_for_generic_spiro(item, prepared)
    )
    natural_topology_generic_polycycle = _has_contextual_complete_kekulized_polycycle_alternative(item, prepared)
    neutral_amine_shortcut = _has_contextual_neutral_amine_systematic_alternative(item, prepared)
    neutral_amine_with_charged_iminium_parent = (
        _has_contextual_charged_iminium_parent_alternative_for_neutral_amine(item, prepared)
    )
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
    imidazo45a_generated_orientation_bonus = _contextual_imidazo45a_generated_orientation_bonus(
        item,
        prepared,
    )
    imidazopyridine_generated_fused_parent_bonus = _contextual_generated_imidazopyridine_parent_bonus(
        item,
        prepared,
    )
    azole_template_parent_bonus = _contextual_azole_template_parent_bonus(item, prepared)
    retained_fused_template_pair_parent_bonus = _contextual_retained_fused_template_pair_parent_bonus(
        item,
        prepared,
    )
    retained_benzimidazole_parent_bonus = _contextual_retained_benzimidazole_parent_bonus(item, prepared)
    retained_benzimidazole_with_low_complexity_generated_imidazo45a = (
        _has_contextual_low_complexity_generated_imidazo45a_alternative_for_retained_benzimidazole(item, prepared)
    )
    pyrazolopyridine_with_retained_benzimidazole = (
        _has_contextual_retained_benzimidazole_parent_alternative_for_pyrazolopyridine(item, prepared)
    )
    retained_benzoannulene_with_generated_imidazopyridine = (
        _has_contextual_generated_imidazopyridine_alternative_for_retained_benzoannulene(item, prepared)
    )
    acyclic_skeleton_ring_system_parent_bonus = _contextual_acyclic_skeleton_ring_system_parent_bonus(
        item,
        prepared,
    )
    pyridopyrrole_acyclic_decomposition_bonus = _contextual_pyridopyrrole_acyclic_decomposition_bonus(
        item,
        prepared,
    )
    kekulized_polycycle_contextual_parent_bonus = _contextual_kekulized_polycycle_parent_bonus(
        item,
        prepared,
    )
    high_impact_exact_retained_parent = _has_contextual_high_impact_exact_retained_parent_alternative(
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
    low_confidence_retained_topology_parent_penalty = (
        _contextual_low_confidence_retained_topology_parent_penalty(item, prepared)
    )
    generated_partial_hydro_templateless_parent = (
        _has_contextual_templateless_parent_alternative_for_generated_partial_hydro(item, prepared)
    )
    retained_topology_generated_partial_hydro_parent = (
        _has_contextual_generated_partial_hydro_parent_alternative_for_retained_topology(item, prepared)
    )
    low_confidence_indan_topology_parent = _has_contextual_low_confidence_indan_topology_parent(
        item,
        prepared,
    )
    broad_incomplete_shortcut_graph_parent = (
        _has_contextual_broad_complete_parent_alternative_for_incomplete_shortcut(item, prepared)
    )
    strict_incomplete_shortcut_graph_parent = (
        _has_contextual_strict_complete_parent_alternative_for_incomplete_shortcut(item, prepared)
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
    compact_chalcogen_suffix_substituent_ring_parent = (
        _has_contextual_substituent_ring_parent_alternative_for_compact_chalcogen_suffix_parent(item, prepared)
    )
    charged_single_ring_parent = _has_contextual_charged_single_ring_parent_alternative(item, prepared)
    charged_ring_skeleton_parent = _has_contextual_charged_ring_skeleton_parent_alternative(item, prepared)
    incomplete_senior_hydride_shortcut = _has_contextual_complete_parent_alternative_for_incomplete_senior_hydride_shortcut(
        item,
        prepared,
    )
    benzotriazole_retained_anchor = _has_contextual_benzotriazole_retained_anchor_alternative(item, prepared)
    retained_shortcut = _has_contextual_systematic_alternative_for_retained_shortcut(item, prepared)
    retained_ester_shortcut = _has_contextual_retained_acid_ester_alternative(item, prepared)
    cyclic_n_carbamate_carboxylate = _has_contextual_complete_cyclic_n_carbamate_alternative(item, prepared)
    acyclic_carbalkoxy_ester = _has_contextual_acyclic_carbalkoxy_retained_ester_alternative(item, prepared)
    complete_urea_parent = _has_contextual_complete_urea_parent_alternative(item, prepared)
    retained_aryl_urea = _has_contextual_aryl_urea_alternative(item, prepared)
    complete_sulfonimidamide_parent = _has_contextual_complete_sulfonimidamide_parent_alternative(item, prepared)
    biochem_heterofused_parent = _has_contextual_biochem_heterofused_parent_alternative(item, prepared)
    exocyclic_fused_single_ring = _has_contextual_exocyclic_fused_single_ring_alternative(item, prepared)
    external_dinitrogen_generated_fused = _has_contextual_external_dinitrogen_generated_fused_alternative(
        item,
        prepared,
    )
    external_nitrogen_generated_fused_single_ring = (
        _has_contextual_external_nitrogen_generated_fused_single_ring_alternative(item, prepared)
    )
    biochem_fused_sibling = _has_contextual_biochem_fused_sibling_alternative(item, prepared)
    generated_fused_low_rank_parent = _has_contextual_low_rank_generated_fused_parent_alternative(
        item,
        prepared,
    )
    generated_fused_same_atoms = _has_contextual_generated_fused_same_atoms_alternative(item, prepared)
    generated_heterofused_parent_for_retained_fragment = (
        _has_contextual_generated_heterofused_parent_alternative_for_retained_fragment(item, prepared)
    )
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
    retained_quinuclidin_parent = _has_contextual_retained_quinuclidin_parent_alternative(item, prepared)
    preferred_compact_bicyclic_parent = (
        _has_contextual_preferred_compact_bicyclic_parent_alternative_for_acyclic_decomposition(item, prepared)
    )
    generated_hydro_pyridino_imidazole_parent = _has_contextual_generated_hydro_pyridino_imidazole_alternative(
        item,
        prepared,
    )
    retained_azabicyclo320_heptane_parent = _has_contextual_retained_azabicyclo320_heptane_parent_alternative(
        item,
        prepared,
    )
    retained_benzocyclobutene_parent = _has_contextual_retained_benzocyclobutene_parent_alternative(
        item,
        prepared,
    )
    generated_hydro_pyrido45d_imidazole_parent = _has_contextual_generated_hydro_pyrido45d_imidazole_alternative(
        item,
        prepared,
    )
    complete_nitrile_oxide_parent = _has_contextual_complete_nitrile_oxide_parent_alternative(item, prepared)
    small_late_single_ring_parent = _has_contextual_early_six_member_single_ring_parent_alternative(item, prepared)
    aromatic_single_ring_parent = _has_contextual_aromatic_single_ring_parent_alternative(item, prepared)
    earlier_aromatic_single_ring_parent = _has_contextual_earlier_aromatic_single_ring_parent_alternative(
        item,
        prepared,
    )
    earlier_external_hetero_single_ring_parent = (
        _has_contextual_earlier_external_hetero_single_ring_parent_alternative(item, prepared)
    )
    single_ring_stereo_prefix_parent = _has_contextual_stereo_prefix_free_ring_alternative(item, prepared)
    external_stereo_acyclic_parent = _has_contextual_external_stereo_acyclic_parent_alternative(item, prepared)
    retained_carbohydrate_substituent_expanded_single_ring = (
        _has_contextual_retained_carbohydrate_substituent_single_ring_sibling(item, prepared)
    )
    higher_isotope_coverage_parent = _has_contextual_higher_isotope_coverage_parent_alternative(item, prepared)
    aza_alias_penalty = _unstable_aza_polycycle_alias_penalty(name)
    spiro_penalty = _unstable_component_spiro_alias_penalty(name, item.candidate)
    if _is_contextual_senior_hetero_single_ring_parent(item, prepared):
        return 0
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
        or retained_purine_chalcogen_tautomer_parent
        or retained_purine_generated_parent
        or retained_purine_unanchored_generated_parent
        or substituted_pterin_with_pteridin_parent
        or retained_imidazopyridine_benzazole_parent
        or retained_imidazopyridine_with_first_indazole_h_parent
        or incomplete_organophosphorus_with_complete_purine_parent
        or purine_parent_with_acyclic_phosphorus_parent
        or preferred_generated_fused_template
        or fragile_terminal_azabicyclo_benzimidazole
        or fragile_group14_hetero_ligand
        or generic_spiro_parent
        or generated_spiro_functional_chain
        or complete_spiro_functional_chain
        or compact_bicyclic_parent_for_spiro_chain
        or compact_functional_bicyclic_parent_for_chain_decomposition
        or single_ring_lactam_parent
        or fused_retained_parent_for_generic_spiro
        or natural_topology_generic_polycycle
        or neutral_amine_shortcut
        or neutral_amine_with_charged_iminium_parent
        or incomplete_shortcut_graph_parent
        or complete_fused_retained_amide_parent_bonus
        or high_confidence_fused_template_parent_bonus
        or imidazo45a_generated_orientation_bonus
        or imidazopyridine_generated_fused_parent_bonus
        or azole_template_parent_bonus
        or retained_fused_template_pair_parent_bonus
        or retained_benzimidazole_parent_bonus
        or retained_benzimidazole_with_low_complexity_generated_imidazo45a
        or pyrazolopyridine_with_retained_benzimidazole
        or retained_benzoannulene_with_generated_imidazopyridine
        or acyclic_skeleton_ring_system_parent_bonus
        or pyridopyrrole_acyclic_decomposition_bonus
        or kekulized_polycycle_contextual_parent_bonus
        or high_impact_exact_retained_parent
        or named_template_pair_parent_bonus
        or low_confidence_fused_template_parent_penalty
        or low_confidence_retained_topology_parent_penalty
        or generated_partial_hydro_templateless_parent
        or retained_topology_generated_partial_hydro_parent
        or low_confidence_indan_topology_parent
        or broad_incomplete_shortcut_graph_parent
        or strict_incomplete_shortcut_graph_parent
        or hydro_fused_systematic_parent
        or lower_external_locant_polycycle
        or small_acyclic_ring_decomposition
        or comparable_complete_fused_parent
        or large_fused_single_ring_suffix_parent
        or compact_chalcogen_suffix_substituent_ring_parent
        or charged_single_ring_parent
        or charged_ring_skeleton_parent
        or incomplete_senior_hydride_shortcut
        or benzotriazole_retained_anchor
        or retained_shortcut
        or retained_ester_shortcut
        or cyclic_n_carbamate_carboxylate
        or acyclic_carbalkoxy_ester
        or complete_urea_parent
        or retained_aryl_urea
        or complete_sulfonimidamide_parent
        or biochem_heterofused_parent
        or exocyclic_fused_single_ring
        or external_dinitrogen_generated_fused
        or external_nitrogen_generated_fused_single_ring
        or biochem_fused_sibling
        or generated_fused_low_rank_parent
        or generated_fused_same_atoms
        or generated_heterofused_parent_for_retained_fragment
        or chain_dominant_ring_substituent
        or kekulized_polycycle_parent
        or chain_candidate_has_large_kekulized_alternative
        or large_kekulized_candidate_has_chain_alternative
        or retained_biochem_fused
        or retained_quinuclidin_parent
        or preferred_compact_bicyclic_parent
        or generated_hydro_pyridino_imidazole_parent
        or retained_azabicyclo320_heptane_parent
        or retained_benzocyclobutene_parent
        or generated_hydro_pyrido45d_imidazole_parent
        or complete_nitrile_oxide_parent
        or small_late_single_ring_parent
        or aromatic_single_ring_parent
        or earlier_aromatic_single_ring_parent
        or earlier_external_hetero_single_ring_parent
        or single_ring_stereo_prefix_parent
        or external_stereo_acyclic_parent
        or retained_carbohydrate_substituent_expanded_single_ring
        or higher_isotope_coverage_parent
        or aza_alias_penalty
        or spiro_penalty
    ):
        return 0

    penalty = 0
    if retained_lactone_topology and not exact_high_impact_lactone_parent:
        penalty -= _RETAINED_LACTONE_TOPOLOGY_BONUS
    if retained_cyclic_anhydride_topology:
        penalty -= _RETAINED_CYCLIC_ANHYDRIDE_TOPOLOGY_BONUS
    if retained_topology_replacement_parent:
        penalty -= 140
        penalty += 120
    if complete_nitrile_oxide_parent:
        penalty += 180
    has_graph_parent = _has_contextual_graph_parent_alternative(item, prepared)
    if high_impact_exact_retained_parent:
        penalty += 260
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
    if retained_purine_chalcogen_tautomer_parent:
        penalty += 560
    if retained_purine_generated_parent:
        penalty += 140
    if retained_purine_unanchored_generated_parent:
        penalty -= 120
    if substituted_pterin_with_pteridin_parent:
        penalty += 80
    if retained_imidazopyridine_benzazole_parent:
        penalty += 45
    if retained_imidazopyridine_with_first_indazole_h_parent:
        penalty += 180
    if incomplete_organophosphorus_with_complete_purine_parent:
        penalty += 90
    if purine_parent_with_acyclic_phosphorus_parent:
        penalty += 160
    if preferred_generated_fused_template:
        penalty += 90
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
    if compact_bicyclic_parent_for_spiro_chain:
        penalty += 110
    if compact_functional_bicyclic_parent_for_chain_decomposition:
        penalty += 260
    if single_ring_lactam_parent:
        penalty += 330
    if fused_retained_parent_for_generic_spiro:
        penalty += 95
    if natural_topology_generic_polycycle:
        penalty += 90
    if neutral_amine_shortcut:
        penalty += 140
    if neutral_amine_with_charged_iminium_parent:
        penalty += 140
    if incomplete_shortcut_graph_parent:
        penalty += 140
    if complete_fused_retained_amide_parent_bonus:
        penalty -= complete_fused_retained_amide_parent_bonus
    if high_confidence_fused_template_parent_bonus:
        penalty -= high_confidence_fused_template_parent_bonus
    if imidazo45a_generated_orientation_bonus:
        penalty -= imidazo45a_generated_orientation_bonus
    if imidazopyridine_generated_fused_parent_bonus:
        penalty -= imidazopyridine_generated_fused_parent_bonus
    if azole_template_parent_bonus:
        penalty -= azole_template_parent_bonus
    if retained_fused_template_pair_parent_bonus:
        penalty -= retained_fused_template_pair_parent_bonus
    if retained_benzimidazole_parent_bonus:
        penalty -= retained_benzimidazole_parent_bonus
    if retained_benzimidazole_with_low_complexity_generated_imidazo45a:
        penalty += _BENZIMIDAZOLE_GENERATED_IMIDAZO45A_REPLACEMENT_PENALTY
    if pyrazolopyridine_with_retained_benzimidazole:
        penalty += 300
    if retained_benzoannulene_with_generated_imidazopyridine:
        penalty += 40
    if acyclic_skeleton_ring_system_parent_bonus:
        penalty -= acyclic_skeleton_ring_system_parent_bonus
    if pyridopyrrole_acyclic_decomposition_bonus:
        penalty -= pyridopyrrole_acyclic_decomposition_bonus
    if kekulized_polycycle_contextual_parent_bonus:
        penalty -= kekulized_polycycle_contextual_parent_bonus
    if named_template_pair_parent_bonus:
        penalty -= named_template_pair_parent_bonus
    if low_confidence_fused_template_parent_penalty:
        penalty += low_confidence_fused_template_parent_penalty
    if low_confidence_retained_topology_parent_penalty:
        penalty += low_confidence_retained_topology_parent_penalty
    if generated_partial_hydro_templateless_parent:
        penalty += _GENERATED_PARTIAL_HYDRO_TEMPLATELESS_PARENT_PENALTY
    if retained_topology_generated_partial_hydro_parent:
        penalty += 140
    if low_confidence_indan_topology_parent:
        penalty += _LOW_CONFIDENCE_INDAN_TOPOLOGY_TEMPLATELESS_PARENT_PENALTY
    if broad_incomplete_shortcut_graph_parent:
        penalty += 125
    if strict_incomplete_shortcut_graph_parent:
        penalty += 140
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
    if compact_chalcogen_suffix_substituent_ring_parent:
        penalty += 128
    if charged_single_ring_parent:
        penalty += 128
    if charged_ring_skeleton_parent:
        penalty += 115
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
    if cyclic_n_carbamate_carboxylate:
        penalty += 1600
    if acyclic_carbalkoxy_ester:
        penalty += 180
    if complete_urea_parent:
        penalty += 1500
    if retained_aryl_urea:
        penalty += 90
    if complete_sulfonimidamide_parent:
        penalty += 520
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
    if external_dinitrogen_generated_fused:
        penalty += 150
    if external_nitrogen_generated_fused_single_ring:
        penalty += 120
    if biochem_fused_sibling:
        penalty += 90
    if generated_fused_low_rank_parent:
        penalty += 120
    if generated_fused_same_atoms:
        penalty += 140
    if generated_heterofused_parent_for_retained_fragment:
        penalty += 180
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
    if retained_quinuclidin_parent:
        penalty += 180
    if preferred_compact_bicyclic_parent:
        penalty += 180
    if generated_hydro_pyridino_imidazole_parent:
        penalty += 100
    if retained_azabicyclo320_heptane_parent:
        penalty += 150
    if retained_benzocyclobutene_parent:
        if item.candidate.family == "sulfonimidamide-parent":
            penalty += 660
        elif item.candidate.family in {
            "functional-generic-bicyclic-parent",
            "generic-bicyclic-parent",
            "preferred-generic-bicyclic-parent",
        }:
            penalty += 340
        else:
            penalty += 260
    if generated_hydro_pyrido45d_imidazole_parent:
        penalty += 300
    if small_late_single_ring_parent:
        penalty += 72
    if aromatic_single_ring_parent:
        penalty += 72
    if earlier_aromatic_single_ring_parent:
        penalty += 128
    if earlier_external_hetero_single_ring_parent:
        penalty += 90
    if single_ring_stereo_prefix_parent:
        penalty += 150
        if _has_one_prefix_carbocycle_parent_with_external_stereo(item.candidate):
            penalty += 80
    if external_stereo_acyclic_parent:
        penalty += 80
    if retained_carbohydrate_substituent_expanded_single_ring:
        penalty -= 240
    if higher_isotope_coverage_parent:
        penalty += 220
    return penalty


def _candidate_selection_contextual_bonus(item: _PreparedCandidate, prepared: list[_PreparedCandidate]) -> int:
    bonus = 0
    if _has_contextual_larger_single_ring_parent_alternative(item, prepared):
        bonus += 40
    if _has_contextual_retained_carbohydrate_substituent_single_ring_sibling(item, prepared):
        bonus += 90
    if _is_contextual_senior_hetero_single_ring_parent(item, prepared):
        bonus += 24
    if _is_contextual_terminal_heterocarbon_pyrazolopyrazine_parent(item, prepared):
        bonus += 12
    if _has_contextual_preferred_generated_exact_orientation(item, prepared):
        bonus += 24
    return bonus


def _is_high_confidence_amino_acid_retained_parent(candidate: NameCandidate) -> bool:
    if candidate.family != "amino-acid-retained-parent":
        return False
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return False
    if candidate_safety_reasons(candidate):
        return False
    return len(candidate.parent_atoms) >= 5


def _has_contextual_preferred_generated_exact_orientation(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family != "generated-exact-parent":
        return False
    if candidate.source_family != "generated-fused":
        return False
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return False
    if not candidate.sort_key or not isinstance(candidate.sort_key[0], tuple):
        return False
    preference = candidate.sort_key[0]
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "generated-exact-parent":
            continue
        if other_candidate.source_family != "generated-fused":
            continue
        if other_candidate.parent_atoms != candidate.parent_atoms:
            continue
        if not other_candidate.sort_key or not isinstance(other_candidate.sort_key[0], tuple):
            continue
        if preference >= other_candidate.sort_key[0]:
            continue
        if item.stability_penalty > other.stability_penalty + 40:
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH or len(item.name) - len(other.name) > 6:
            continue
        return True
    return False


def _has_contextual_stereo_prefix_free_ring_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family != "single-ring-parent":
        return False
    if not candidate.stereo_prefix:
        return False
    stereo_parts = [part for part in candidate.stereo_prefix.split(",") if part.endswith(("R", "S", "r", "s"))]
    if len(stereo_parts) != 1:
        return False
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return False
    if candidate.parent_aromatic_atom_count or candidate.parent_hetero_atom_count:
        return False
    parent_count = len(candidate.parent_atoms)
    if parent_count < 4 or parent_count > 8:
        return False
    current_rank = candidate.render_rank or 0
    alternative_families = _ACYCLIC_SKELETON_PARENT_FAMILIES | {"single-ring-parent"}
    if _has_one_prefix_carbocycle_parent_with_external_stereo(candidate):
        alternative_families = alternative_families | _LOCANTED_STEREO_GRAPH_RING_PARENT_FAMILIES
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family not in alternative_families:
            continue
        if other_candidate.stereo_prefix:
            continue
        if not other_candidate.parent_atoms or not candidate_has_complete_graph(other_candidate):
            continue
        if len(other_candidate.parent_atoms) < parent_count:
            continue
        other_rank = other_candidate.render_rank or 0
        if (
            other_candidate.family == "single-ring-parent"
            and current_rank
            and other_rank
            and other_rank > current_rank + 3
        ):
            continue
        if len(other.name) > 220 or len(other.name) - len(item.name) > 95:
            continue
        if other.stability_penalty >= item.stability_penalty + 70:
            continue
        return True
    return False


def _has_one_prefix_carbocycle_parent_with_external_stereo(candidate: NameCandidate) -> bool:
    if candidate.family != "single-ring-parent":
        return False
    if not candidate.stereo_prefix or "," in candidate.stereo_prefix:
        return False
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return False
    if candidate.parent_aromatic_atom_count or candidate.parent_hetero_atom_count:
        return False
    if not candidate.stereo or not candidate.stereo.atom_descriptors:
        return False
    if _is_compact_external_nitrogen_unsaturated_single_ring_parent(candidate):
        return False
    parent_atoms = set(candidate.parent_atoms)
    parent_stereo_count = sum(
        1 for atom_idx, _descriptor in candidate.stereo.atom_descriptors if atom_idx in parent_atoms
    )
    external_stereo_count = len(candidate.stereo.atom_descriptors) - parent_stereo_count
    return parent_stereo_count > 0 and external_stereo_count > 0


def _has_contextual_external_stereo_acyclic_parent_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family != "single-ring-parent":
        return False
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return False
    if candidate.parent_aromatic_atom_count != len(candidate.parent_atoms):
        return False
    if candidate.stereo_prefix or candidate.stereo_locants:
        return False
    if not candidate.stereo or not candidate.stereo.atom_descriptors:
        return False
    if _is_compact_external_nitrogen_unsaturated_single_ring_parent(candidate):
        return False
    parent_atoms = set(candidate.parent_atoms)
    if any(atom_idx in parent_atoms for atom_idx, _descriptor in candidate.stereo.atom_descriptors):
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family not in _ACYCLIC_SKELETON_PARENT_FAMILIES:
            continue
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        if len(other_candidate.stereo_locants) < 3:
            continue
        if len(other.name) > 180 or len(other.name) - len(item.name) > 80:
            continue
        if other.stability_penalty >= item.stability_penalty + 80:
            continue
        return True
    return False


def _is_compact_external_nitrogen_unsaturated_single_ring_parent(candidate: NameCandidate) -> bool:
    if candidate.family != "single-ring-parent":
        return False
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return False
    return (
        len(candidate.parent_atoms) <= 3
        and candidate.parent_external_unsaturated_bond_count > 0
        and candidate.parent_external_nitrogen_substituent_count >= 2
        and (candidate.render_rank or 0) <= 2
    )


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
    if candidate.render_rank is None or candidate.render_rank > 4:
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


def _has_contextual_charged_iminium_parent_alternative_for_neutral_amine(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family != "neutral-amine-parent":
        return False
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "charged-iminium-retained-parent":
            continue
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        if other_candidate.parent_atoms != candidate.parent_atoms:
            continue
        if other.stability_penalty > item.stability_penalty + 40:
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
    if _contextual_low_confidence_retained_topology_parent_penalty(item, prepared):
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


def _contextual_imidazo45a_generated_orientation_bonus(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> int:
    candidate = item.candidate
    if candidate.family != "generated-fused-parent":
        return 0
    if candidate.template_name != "imidazo[4,5-a]benzene":
        return 0
    if candidate.source_family != "generated-fused":
        return 0
    if candidate.source_file != "generated_fused_template_index.json":
        return 0
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return 0
    if len(candidate.parent_atoms) != 9:
        return 0
    if candidate.render_rank not in _IMIDAZO45A_GENERATED_ORIENTATION_BONUS_BY_RENDER_RANK:
        return 0
    if candidate.render_rank == 2 and _candidate_has_stereo_evidence(candidate):
        return 0
    if len(item.name) > 280:
        return 0
    if not _has_lower_rank_same_imidazo45a_generated_orientation(item, prepared):
        return 0
    return _IMIDAZO45A_GENERATED_ORIENTATION_BONUS_BY_RENDER_RANK[candidate.render_rank]


def _contextual_azole_template_parent_bonus(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> int:
    candidate = item.candidate
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return 0
    if len(item.name) > 280:
        return 0
    family = candidate.family
    template_name = candidate.template_name or ""
    if not (
        (family == "generated-fused-parent" and template_name in {"imidazo[4,5-a]benzene", "pyrido[2,3-d]pyrazole"})
        or (family == "fused-retained-parent" and template_name in {"1H-benzotriazole", "2H-benzotriazole"})
        or family == "single-ring-parent"
    ):
        return 0
    bonus = 0
    if _has_contextual_imidazo45a_rank2_parent_alternative(item, prepared):
        bonus += _IMIDAZO45A_GENERATED_RANK2_PARENT_BONUS
    if _has_contextual_benzotriazole_parent_for_thieno23a(item, prepared):
        bonus += _BENZOTRIAZOLE_OVER_THIENO23A_PARENT_BONUS
    if _has_contextual_single_ring_parent_for_quino34d_pyrazole(item, prepared):
        bonus += _QUINO34D_PYRAZOLE_SINGLE_RING_PARENT_BONUS
    if _has_contextual_pyrido23d_pyrazole_parent(item, prepared):
        bonus += _PYRIDO23D_PYRAZOLE_PARENT_BONUS
    return bonus


def _has_contextual_imidazo45a_rank2_parent_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family != "generated-fused-parent":
        return False
    if candidate.template_name != "imidazo[4,5-a]benzene":
        return False
    if candidate.source_family != "generated-fused":
        return False
    if candidate.source_file != "generated_fused_template_index.json":
        return False
    if candidate.render_rank != 2:
        return False
    if len(candidate.parent_atoms) != 9:
        return False
    return _has_lower_rank_same_imidazo45a_generated_orientation(item, prepared)


def _has_contextual_benzotriazole_parent_for_thieno23a(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family != "fused-retained-parent":
        return False
    if candidate.template_name not in {"1H-benzotriazole", "2H-benzotriazole"}:
        return False
    if candidate.render_rank is not None and candidate.render_rank > 3:
        return False
    if len(candidate.parent_atoms) < 9:
        return False
    if len(item.name) > 220:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "generated-fused-parent":
            continue
        if other_candidate.template_name != "thieno[2,3-a]benzene":
            continue
        if not other_candidate.parent_atoms or not candidate_has_complete_graph(other_candidate):
            continue
        if other.stability_penalty > item.stability_penalty + 20:
            continue
        return True
    return False


def _contextual_retained_fused_template_pair_parent_bonus(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> int:
    candidate = item.candidate
    if candidate.family != "fused-retained-parent":
        return 0
    template_name = candidate.template_name or ""
    competitor_templates = _RETAINED_FUSED_TEMPLATE_PAIR_COMPETITORS.get(template_name)
    if not competitor_templates:
        return 0
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return 0
    if candidate.render_rank is not None and candidate.render_rank > 4:
        return 0
    if len(item.name) > 260:
        return 0
    if template_name == "1H-benzimidazole" and _has_larger_retained_parent_competitor(item, prepared):
        return 0
    if template_name == "1H-benzimidazole" and candidate.parent_substituted_hetero_atom_count:
        return 0
    parent_count = len(candidate.parent_atoms)
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "generated-fused-parent":
            continue
        if other_candidate.template_name not in competitor_templates:
            continue
        if template_name == "1H-benzimidazole" and other_candidate.parent_substituted_hetero_atom_count:
            continue
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        other_parent_count = len(other_candidate.parent_atoms)
        if other_parent_count + 1 < parent_count or other_parent_count > parent_count + 1:
            continue
        if len(other.name) > 280:
            continue
        if other.stability_penalty > item.stability_penalty + 80:
            continue
        return _RETAINED_FUSED_TEMPLATE_PAIR_PARENT_BONUS
    return 0


def _contextual_retained_benzimidazole_parent_bonus(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> int:
    candidate = item.candidate
    if candidate.family != "fused-retained-parent":
        return 0
    if candidate.template_name != "benzimidazole":
        return 0
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return 0
    if len(candidate.parent_atoms) != 9:
        return 0
    if candidate.parent_element_counts != (("C", 7), ("N", 2)):
        return 0
    if candidate.parent_substituted_hetero_atom_count:
        return 0
    if candidate.render_rank is not None and candidate.render_rank > 2:
        return 0
    if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
        return 0
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "generated-fused-parent":
            continue
        if other_candidate.template_name != "imidazo[4,5-a]benzene":
            continue
        if other_candidate.parent_atoms != candidate.parent_atoms:
            continue
        if not candidate_has_complete_graph(other_candidate):
            continue
        if other_candidate.render_rank is None or other_candidate.render_rank > 2:
            continue
        if len(other.name) > 280:
            continue
        if len(item.name) - len(other.name) > 20:
            continue
        if item.stability_penalty > other.stability_penalty + 260:
            continue
        return 320
    if _has_contextual_pyrazolopyridine_parent_competitor_for_retained_benzimidazole(item, prepared):
        return 360
    return 0


def _has_contextual_low_complexity_generated_imidazo45a_alternative_for_retained_benzimidazole(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family != "fused-retained-parent":
        return False
    if candidate.template_name != "benzimidazole":
        return False
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return False
    if len(candidate.parent_atoms) != 9:
        return False
    if candidate.parent_element_counts != (("C", 7), ("N", 2)):
        return False
    if candidate.parent_substituted_hetero_atom_count:
        return False
    if candidate.render_rank is not None and candidate.render_rank > 2:
        return False
    if candidate.suffixes or candidate.charge is not None:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "generated-fused-parent":
            continue
        if other_candidate.template_name != "imidazo[4,5-a]benzene":
            continue
        if other_candidate.source_family != "generated-fused":
            continue
        if other_candidate.source_file != "generated_fused_template_index.json":
            continue
        if other_candidate.parent_atoms != candidate.parent_atoms:
            continue
        if not candidate_has_complete_graph(other_candidate):
            continue
        if other_candidate.render_rank != 1:
            continue
        if other_candidate.parent_hetero_atom_count != 2:
            continue
        if other_candidate.parent_terminal_hetero_substituent_count:
            continue
        if other_candidate.parent_external_nitrogen_substituent_count:
            continue
        if other_candidate.parent_external_oxygen_substituent_count:
            continue
        if other_candidate.suffixes or other_candidate.charge is not None:
            continue
        if candidate_safety_reasons(other_candidate):
            continue
        if not (
            _BENZIMIDAZOLE_GENERATED_IMIDAZO45A_REPLACEMENT_MIN_LENGTH
            < len(other.name)
            <= _BENZIMIDAZOLE_GENERATED_IMIDAZO45A_REPLACEMENT_MAX_LENGTH
        ):
            continue
        if other.stability_penalty > item.stability_penalty + 80:
            continue
        return True
    return False


_BENZIMIDAZOLE_COMPETING_PYRAZOLOPYRIDINE_TEMPLATES = frozenset(
    {
        "pyrazolo[3,4-e]pyridine",
        "pyrazolo[4,3-b]pyridine",
        "pyridino[2,3-c]pyrazole",
        "pyridino[3,2-c]pyrazole",
        "pyrido[2,3-d]pyrazole",
        "pyrido[3,2-c]pyrazole",
    }
)


def _has_contextual_pyrazolopyridine_parent_competitor_for_retained_benzimidazole(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family != "fused-retained-parent":
        return False
    if candidate.template_name != "benzimidazole":
        return False
    if candidate.parent_element_counts != (("C", 7), ("N", 2)):
        return False
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return False
    if len(candidate.parent_atoms) != 9:
        return False
    if candidate.parent_substituted_hetero_atom_count:
        return False
    if candidate.render_rank is None or candidate.render_rank > 4:
        return False
    if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family not in {"fused-retained-parent", "generated-fused-parent"}:
            continue
        if other_candidate.template_name not in _BENZIMIDAZOLE_COMPETING_PYRAZOLOPYRIDINE_TEMPLATES:
            continue
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        if len(other_candidate.parent_atoms) != 9:
            continue
        if other_candidate.parent_substituted_hetero_atom_count == 0:
            continue
        if other_candidate.render_rank is None or other_candidate.render_rank > 6:
            continue
        if len(other.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(item.name) - len(other.name) > 90:
            continue
        if item.stability_penalty > other.stability_penalty + 260:
            continue
        return True
    return False


def _has_contextual_retained_benzimidazole_parent_alternative_for_pyrazolopyridine(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family not in {"fused-retained-parent", "generated-fused-parent"}:
        return False
    if candidate.template_name not in _BENZIMIDAZOLE_COMPETING_PYRAZOLOPYRIDINE_TEMPLATES:
        return False
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return False
    if len(candidate.parent_atoms) != 9:
        return False
    if candidate.parent_substituted_hetero_atom_count == 0:
        return False
    if candidate.render_rank is None or candidate.render_rank > 6:
        return False
    if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "fused-retained-parent":
            continue
        if other_candidate.template_name != "benzimidazole":
            continue
        if other_candidate.parent_element_counts != (("C", 7), ("N", 2)):
            continue
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        if len(other_candidate.parent_atoms) != 9:
            continue
        if other_candidate.parent_substituted_hetero_atom_count:
            continue
        if other_candidate.render_rank is None or other_candidate.render_rank > 4:
            continue
        if len(other.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(other.name) - len(item.name) > 80:
            continue
        if other.stability_penalty > item.stability_penalty + 260:
            continue
        return True
    return False


_GENERATED_IMIDAZOPYRIDINE_ALTERNATIVE_TEMPLATES = frozenset(
    {
        "imidazo[4,5-b]pyridine",
        "imidazo[4,5-e]pyridine",
        "pyridino[3,2-d]imidazole",
        "pyrido[5,6-d]imidazole",
    }
)


_GENERATED_IMIDAZOPYRIDINE_PARENT_BONUS_TEMPLATES = frozenset(
    {
        "imidazo[4,5-b]pyridine",
        "imidazo[4,5-e]pyridine",
    }
)

_GENERATED_IMIDAZOPYRIDINE_PARENT_BONUS = 220


def _contextual_generated_imidazopyridine_parent_bonus(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> int:
    candidate = item.candidate
    if candidate.family != "generated-fused-parent":
        return 0
    if candidate.template_name not in _GENERATED_IMIDAZOPYRIDINE_PARENT_BONUS_TEMPLATES:
        return 0
    if candidate.source_family != "generated-fused":
        return 0
    if candidate.source_file != "generated_fused_template_index.json":
        return 0
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return 0
    if len(candidate.parent_atoms) != 9:
        return 0
    if candidate.parent_hetero_atom_count != 3:
        return 0
    if candidate.parent_external_nitrogen_substituent_count > 1:
        return 0
    if candidate.parent_external_oxygen_substituent_count:
        return 0
    if candidate.suffixes or candidate.charge is not None:
        return 0
    if candidate_safety_reasons(candidate):
        return 0
    if candidate.render_rank is None or candidate.render_rank > 2:
        return 0
    if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
        return 0
    if _has_contextual_chain_dominant_acyclic_parent_alternative(item, prepared):
        return 0
    if _has_contextual_complete_senior_sulfonamide_parent_alternative(item, prepared):
        return 0
    if _has_contextual_ring_oxo_suffix_parent_alternative(item, prepared):
        return 0
    if not _has_contextual_imidazopyridine_parent_competitor(item, prepared):
        return 0
    return _GENERATED_IMIDAZOPYRIDINE_PARENT_BONUS


def _has_contextual_imidazopyridine_parent_competitor(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if (
            other_candidate.family == candidate.family
            and other_candidate.template_name in _GENERATED_IMIDAZOPYRIDINE_PARENT_BONUS_TEMPLATES
        ):
            continue
        if len(other.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if other.stability_penalty > item.stability_penalty + 220:
            continue
        if len(other.name) - len(item.name) > 120:
            continue
        return True
    return False


def _has_contextual_chain_dominant_acyclic_parent_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
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
        if len(other.name) - len(item.name) > 140:
            continue
        if other.stability_penalty > item.stability_penalty + 120:
            continue
        return True
    return False


def _has_contextual_complete_senior_sulfonamide_parent_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    item_parent_count = len(item.candidate.parent_atoms)
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family not in {"sulfonamide-parent", "sulfonamide-retained-parent"}:
            continue
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        if len(other_candidate.parent_atoms) <= item_parent_count:
            continue
        if len(other.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(other.name) - len(item.name) > 80:
            continue
        if other.stability_penalty > item.stability_penalty + 120:
            continue
        if candidate_safety_reasons(other_candidate):
            continue
        return True
    return False


def _has_contextual_ring_oxo_suffix_parent_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if not other_candidate.suffixes:
            continue
        if not any(suffix.kind == "ring_oxo_suffix" for suffix in other_candidate.suffixes):
            continue
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        if len(other.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(other.name) - len(item.name) > 80:
            continue
        if other.stability_penalty > item.stability_penalty + 120:
            continue
        if candidate_safety_reasons(other_candidate):
            continue
        return True
    return False


def _has_contextual_generated_imidazopyridine_alternative_for_retained_benzoannulene(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family != "fused-retained-parent":
        return False
    if candidate.template_name != "6,7,8,9-tetrahydro-5H-benzo[7]annulene":
        return False
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return False
    if len(candidate.parent_atoms) != 11:
        return False
    if candidate.parent_element_counts != (("C", 11),):
        return False
    if candidate.render_rank is not None and candidate.render_rank > 2:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "generated-fused-parent":
            continue
        if other_candidate.template_name not in _GENERATED_IMIDAZOPYRIDINE_ALTERNATIVE_TEMPLATES:
            continue
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        if len(other_candidate.parent_atoms) != 9:
            continue
        if other_candidate.parent_element_counts != (("C", 6), ("N", 3)):
            continue
        if other_candidate.render_rank is None or other_candidate.render_rank > 4:
            continue
        if len(other.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(other.name) - len(item.name) > 40:
            continue
        if other.stability_penalty > item.stability_penalty + 80:
            continue
        return True
    return False


def _has_larger_retained_parent_competitor(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if not candidate.parent_atoms:
        return False
    parent_count = len(candidate.parent_atoms)
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family not in {
            "fused-retained-parent",
            "retained-topology-replacement-fused-parent",
        }:
            continue
        if other_candidate.template_name == candidate.template_name:
            continue
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        if len(other_candidate.parent_atoms) <= parent_count + 1:
            continue
        if other.stability_penalty > item.stability_penalty + 100:
            continue
        return True
    return False


def _contextual_acyclic_skeleton_ring_system_parent_bonus(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> int:
    candidate = item.candidate
    if candidate.family not in _ACYCLIC_SKELETON_RING_SYSTEM_BONUS_FAMILIES:
        return 0
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return 0
    parent_count = len(candidate.parent_atoms)
    if parent_count < 8:
        return 0
    if len(item.name) > 220:
        return 0
    if not _has_smaller_ring_system_parent_competitor(item, prepared):
        return 0
    if _has_protected_acyclic_skeleton_parent_competitor(item, prepared):
        return 0
    return _ACYCLIC_SKELETON_RING_SYSTEM_PARENT_BONUS


def _has_smaller_ring_system_parent_competitor(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    parent_count = len(item.candidate.parent_atoms)
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family not in _ACYCLIC_SKELETON_RING_SYSTEM_COMPETITOR_FAMILIES:
            continue
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        if parent_count < len(other_candidate.parent_atoms) + 2:
            continue
        if other.stability_penalty > item.stability_penalty + 240:
            continue
        return True
    return False


def _has_protected_acyclic_skeleton_parent_competitor(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family not in _ACYCLIC_SKELETON_PROTECTED_PARENT_FAMILIES:
            continue
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        if other.stability_penalty > item.stability_penalty + 80:
            continue
        return True
    return False


def _contextual_kekulized_polycycle_parent_bonus(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> int:
    candidate = item.candidate
    if candidate.family != "kekulized-generic-polycyclic-parent":
        return 0
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return 0
    parent_count = len(candidate.parent_atoms)
    if parent_count < 8 or parent_count > 36:
        return 0
    if candidate.render_rank is None or candidate.render_rank > 5:
        return 0
    if candidate.suffixes:
        return 0
    if len(item.name) > 260:
        return 0
    for other in prepared:
        if other is item:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "kekulized-generic-polycyclic-parent":
            continue
        if other_candidate.render_rank is None or candidate.render_rank is None:
            continue
        if other_candidate.render_rank >= candidate.render_rank:
            continue
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        if frozenset(other_candidate.parent_atoms) != frozenset(candidate.parent_atoms):
            continue
        if other.stability_penalty > item.stability_penalty:
            continue
        return 0
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family not in _KEKULIZED_POLYCYCLE_CONTEXTUAL_COMPETITOR_FAMILIES:
            continue
        if other_candidate.family == "kekulized-generic-polycyclic-parent":
            if other_candidate.render_rank is None or other_candidate.render_rank <= candidate.render_rank:
                continue
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        if other.stability_penalty > item.stability_penalty + 160:
            continue
        return _KEKULIZED_POLYCYCLE_CONTEXTUAL_PARENT_BONUS
    return 0


def _has_contextual_single_ring_parent_for_quino34d_pyrazole(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family != "single-ring-parent":
        return False
    if len(item.name) > 190:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "generated-fused-parent":
            continue
        if other_candidate.template_name != "quino[3,4-d]pyrazole":
            continue
        if not other_candidate.parent_atoms or not candidate_has_complete_graph(other_candidate):
            continue
        if other.stability_penalty > item.stability_penalty + 260:
            continue
        return True
    return False


def _has_contextual_pyrido23d_pyrazole_parent(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family != "generated-fused-parent":
        return False
    if candidate.template_name != "pyrido[2,3-d]pyrazole":
        return False
    if candidate.render_rank is not None and candidate.render_rank > 3:
        return False
    if len(candidate.parent_atoms) < 9:
        return False
    competitor_templates = {
        "1H-benzimidazole",
        "imidazo[4,5-a]benzene",
        "imidazo[4,5-d]pyridine",
    }
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.template_name not in competitor_templates:
            continue
        if other_candidate.family not in {"fused-retained-parent", "generated-fused-parent"}:
            continue
        if not other_candidate.parent_atoms or not candidate_has_complete_graph(other_candidate):
            continue
        if other.stability_penalty > item.stability_penalty + 260:
            continue
        return True
    return False


def _candidate_has_stereo_evidence(candidate: NameCandidate) -> bool:
    return bool(
        candidate.stereo is not None
        and (candidate.stereo.atom_descriptors or candidate.stereo.bond_descriptors)
    )


def _has_lower_rank_same_imidazo45a_generated_orientation(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    current_rank = candidate.render_rank or 0
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != candidate.family:
            continue
        if other_candidate.template_name != candidate.template_name:
            continue
        if other_candidate.source_family != candidate.source_family:
            continue
        if other_candidate.source_file != candidate.source_file:
            continue
        if other_candidate.parent_atoms != candidate.parent_atoms:
            continue
        if not candidate_has_complete_graph(other_candidate):
            continue
        other_rank = other_candidate.render_rank or 0
        if other_rank < 1 or other_rank >= current_rank:
            continue
        if other.stability_penalty > item.stability_penalty + 80:
            continue
        return True
    return False


def _has_contextual_high_impact_exact_retained_parent_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    if _is_high_impact_exact_retained_parent_candidate(item.candidate):
        return False
    if item.candidate.family == "whole-molecule-retained-parent":
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        if not _is_high_impact_exact_retained_parent_candidate(other.candidate):
            continue
        if other.stability_penalty > item.stability_penalty + 80:
            continue
        if len(other.name) - len(item.name) > 260:
            continue
        return True
    return False


def _is_high_impact_exact_retained_parent_candidate(candidate: NameCandidate) -> bool:
    if candidate.family not in {"exact-retained-parent", "exact-functional-retained-parent"}:
        return False
    if candidate.source_family != "static-retained":
        return False
    if candidate.source_file != "high_impact_fused_parent_templates.json":
        return False
    if candidate.template_id is None:
        return False
    return bool(candidate.parent_atoms and candidate_has_complete_graph(candidate))


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
        if _has_contextual_exact_opsin_dibenzofuran_parent(item, prepared):
            return 0
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
        candidate.family == "fused-retained-parent"
        and candidate.template_name == "6,7-dihydro-5H-cyclopenta[b]pyridine"
        and _has_contextual_complete_named_fused_template_alternative(
            item,
            prepared,
            frozenset({"5H-cyclopenta[b]pyridine"}),
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


def _has_contextual_exact_opsin_dibenzofuran_parent(
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
        if not _is_exact_opsin_dibenzofuran_parent(other_candidate):
            continue
        if other_candidate.parent_atoms != candidate.parent_atoms:
            continue
        if other.stability_penalty > item.stability_penalty + 80:
            continue
        return True
    return False


def _is_exact_opsin_dibenzofuran_parent(candidate: NameCandidate) -> bool:
    if candidate.template_name != "dibenzofuran":
        return False
    if candidate.source_family != "opsin-retained-ring-cml":
        return False
    if candidate.source_file != "opsin-derived-heteroarene-templates":
        return False
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return False
    return not candidate_safety_reasons(candidate)


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


def _contextual_low_confidence_retained_topology_parent_penalty(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> int:
    candidate = item.candidate
    if candidate.family != "retained-topology-replacement-fused-parent":
        return 0
    alternative_families = _LOW_CONFIDENCE_RETAINED_TOPOLOGY_TEMPLATE_TO_TEMPLATELESS_FAMILIES.get(
        candidate.template_name
    )
    if not alternative_families:
        return 0
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return 0
    if candidate.render_rank is not None and candidate.render_rank > 2:
        return 0
    if not _has_contextual_template_less_family_alternative_for_low_confidence_template(
        item,
        prepared,
        alternative_families,
    ):
        return 0
    return _LOW_CONFIDENCE_RETAINED_TOPOLOGY_PARENT_PENALTY


def _has_contextual_templateless_parent_alternative_for_generated_partial_hydro(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family != "generated-partial-hydro-fused-parent":
        return False
    if candidate.source_family != "generated-fused":
        return False
    if candidate.source_file != "generated_fused_template_index.json":
        return False
    if not candidate.template_name:
        return False
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return False
    if candidate.render_rank is not None and candidate.render_rank > 4:
        return False
    if candidate.suffixes or candidate.charge is not None:
        return False
    parent_count = len(candidate.parent_atoms)
    if parent_count < 8:
        return False
    return _has_contextual_low_broad_templateless_parent_alternative(
        item,
        prepared,
        max_length=280,
        max_extra=170,
        max_stability_extra=240,
    )


def _has_contextual_generated_partial_hydro_parent_alternative_for_retained_topology(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family != "retained-topology-replacement-fused-parent":
        return False
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return False
    if candidate.render_rank is not None and candidate.render_rank > 2:
        return False
    if candidate.suffixes or candidate.charge is not None or candidate_safety_reasons(candidate):
        return False
    parent_count = len(candidate.parent_atoms)
    if parent_count < 8:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "generated-partial-hydro-fused-parent":
            continue
        if other_candidate.source_family != "generated-fused":
            continue
        if other_candidate.source_file != "generated_fused_template_index.json":
            continue
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        if other_candidate.render_rank is None or other_candidate.render_rank > 3:
            continue
        if other_candidate.suffixes or other_candidate.charge is not None or candidate_safety_reasons(other_candidate):
            continue
        if len(other.name) > 160 or len(other.name) - len(item.name) > 90:
            continue
        if other.stability_penalty > item.stability_penalty + 80:
            continue
        if other_candidate.parent_atoms == candidate.parent_atoms:
            return True
        if (
            len(other_candidate.parent_atoms) == parent_count
            and other_candidate.parent_element_counts == candidate.parent_element_counts
        ):
            return True
    return False


def _has_contextual_low_confidence_indan_topology_parent(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family != "retained-topology-replacement-fused-parent":
        return False
    if candidate.template_name != "indan":
        return False
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return False
    if candidate.render_rank is not None and candidate.render_rank > 1:
        return False
    if candidate.suffixes or candidate.charge is not None:
        return False
    if len(candidate.parent_atoms) < 8:
        return False
    return _has_contextual_low_broad_templateless_parent_alternative(
        item,
        prepared,
        max_length=280,
        max_extra=170,
        max_stability_extra=240,
    )


def _has_contextual_low_broad_templateless_parent_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
    *,
    max_length: int,
    max_extra: int,
    max_stability_extra: int,
) -> bool:
    if _has_contextual_low_broad_templateless_parent_demotion_blocker(item, prepared):
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.template_name:
            continue
        if other_candidate.family not in _GENERATED_PARTIAL_HYDRO_TEMPLATELESS_PARENT_FAMILIES:
            continue
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        if len(other_candidate.parent_atoms) < 4:
            continue
        if other_candidate.charge is not None:
            continue
        if other_candidate.suffixes:
            continue
        if other_candidate.render_rank is not None and other_candidate.render_rank > 4:
            continue
        if _candidate_selection_broad_graph_parent_penalty(other, prepared) >= 100:
            continue
        if other.stability_penalty > item.stability_penalty + max_stability_extra:
            continue
        if len(other.name) > max_length:
            continue
        if len(other.name) - len(item.name) > max_extra:
            continue
        return True
    return False


def _has_contextual_low_broad_templateless_parent_demotion_blocker(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family not in {
        "generated-partial-hydro-fused-parent",
        "retained-topology-replacement-fused-parent",
    }:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        if other_candidate.charge is not None or other_candidate.suffixes:
            continue
        if (
            other_candidate.family == "skeletal-replacement-fused-parent"
            and other_candidate.template_name in _HIGH_CONFIDENCE_FUSED_TEMPLATE_PARENT_NAMES
        ):
            if other_candidate.render_rank is None or other_candidate.render_rank > 2:
                continue
            if len(other.name) > len(item.name) + 10:
                continue
            if other.stability_penalty > item.stability_penalty + 60:
                continue
            return True
        if (
            other_candidate.family == "fused-retained-parent"
            and other_candidate.template_name in _PARTIAL_HYDRO_DEMOTION_HYDRO_QUINOLINE_TEMPLATES
        ):
            if other_candidate.render_rank is not None and other_candidate.render_rank > 1:
                continue
            if len(other.name) > len(item.name) + 20:
                continue
            if other.stability_penalty > item.stability_penalty + 60:
                continue
            return True
    return False


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


def _has_contextual_strict_complete_parent_alternative_for_incomplete_shortcut(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family not in _BROAD_INCOMPLETE_SHORTCUT_GRAPH_PARENT_FAMILIES:
        return False
    if candidate_has_complete_graph(candidate):
        return False
    if candidate.source_family in {"ring-carboxamide:selenium", "ring-carboxamide:tellurium"}:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family not in _BROAD_COMPLETE_SHORTCUT_GRAPH_PARENT_ALTERNATIVE_FAMILIES:
            continue
        if other_candidate.family in {"single-ring-parent", "positive-single-ring-parent"}:
            continue
        if (
            candidate.family == "retained-amide-parent"
            and other_candidate.family not in (_ACYCLIC_SKELETON_PARENT_FAMILIES | {"acyclic-parent"})
        ):
            continue
        if (
            candidate.family == "retained-amide-parent"
            and _candidate_selection_broad_graph_parent_penalty(other, prepared) >= 100
        ):
            continue
        if not candidate_has_complete_graph(other_candidate) or len(other_candidate.parent_atoms) < 7:
            continue
        if len(other.name) > 220 or len(other.name) - len(item.name) > 90:
            continue
        if other.stability_penalty >= item.stability_penalty + 70:
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


def _has_contextual_substituent_ring_parent_alternative_for_compact_chalcogen_suffix_parent(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if not _is_compact_osn_chalcogen_suffix_parent(candidate):
        return False
    current_parent_atoms = candidate.parent_atoms
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "single-ring-parent":
            continue
        if not other_candidate.parent_atoms or not candidate_has_complete_graph(other_candidate):
            continue
        if current_parent_atoms & other_candidate.parent_atoms:
            continue
        if other_candidate.suffixes or other_candidate.charge is not None:
            continue
        other_parent_count = len(other_candidate.parent_atoms)
        if other_parent_count < 3 or other_parent_count > 8:
            continue
        if (
            other_candidate.parent_hetero_atom_count > 1
            and other_candidate.parent_aromatic_atom_count != other_parent_count
        ):
            continue
        if len(other.name) > 200 or len(other.name) - len(item.name) > 110:
            continue
        if other.stability_penalty > item.stability_penalty + 80:
            continue
        return True
    return False


def _is_compact_osn_chalcogen_suffix_parent(candidate: NameCandidate) -> bool:
    if candidate.family != "single-ring-parent":
        return False
    if candidate.charge is not None or not candidate.suffixes:
        return False
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return False
    if len(candidate.parent_atoms) > 5:
        return False
    chalcogen_suffixes = [
        suffix
        for suffix in candidate.suffixes
        if suffix.kind in {"ring_oxo_suffix", "ring_thione_suffix"}
    ]
    if not chalcogen_suffixes:
        return False
    if not any("compact_hetero_ring_chalcogen_suffix_parent" in suffix.source for suffix in chalcogen_suffixes):
        return False
    if (
        _candidate_parent_element_count(candidate, "O") < 1
        or _candidate_parent_element_count(candidate, "S") < 1
        or _candidate_parent_element_count(candidate, "N") < 1
    ):
        return False
    return candidate.parent_terminal_hetero_substituent_count >= 1


def _has_contextual_charged_ring_skeleton_parent_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if not _is_simple_charged_aromatic_single_ring_parent(candidate):
        return False
    if len(item.name) > 70:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "charged-ring-acyclic-skeleton-with-ring-substituents":
            continue
        if not other_candidate.parent_atoms or not candidate_has_complete_graph(other_candidate):
            continue
        if len(other.name) > 180 or len(other.name) - len(item.name) > 80:
            continue
        if other.stability_penalty > item.stability_penalty + 120:
            continue
        return True
    return False


def _has_contextual_charged_single_ring_parent_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    charge_distribution_shortcut = _is_exact_retained_charge_distribution_shortcut(candidate)
    charge_erasing_competitor = _is_charge_erasing_single_ring_competitor_candidate(candidate)
    charged_retained_parent = _is_charged_retained_parent_candidate(candidate)
    if not charge_distribution_shortcut and not charge_erasing_competitor and not charged_retained_parent:
        if candidate.family not in {"single-ring-parent", "positive-single-ring-parent"}:
            return False
        if candidate.charge is not None or candidate.suffixes:
            return False
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            return False
        current_parent_count = len(candidate.parent_atoms)
        if current_parent_count < 3 or current_parent_count > 8:
            return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if not _is_charge_located_single_ring_parent_candidate(other_candidate):
            continue
        evidence = other_candidate.charge
        if evidence is None:
            continue
        if evidence.formal_charge < 0 and len(evidence.charged_atoms) > 1:
            continue
        if charge_distribution_shortcut:
            if len(other_candidate.parent_atoms) < 5 or len(other_candidate.parent_atoms) > 6:
                continue
            if not other_candidate.parent_atoms.issubset(candidate.parent_atoms):
                continue
        elif charge_erasing_competitor:
            if len(other.name) > 220 or len(other.name) - len(item.name) > 130:
                continue
        elif charged_retained_parent:
            if not _is_six_membered_ring_dication_single_ring_parent(other_candidate):
                continue
            if len(other.name) > 180 or len(other.name) - len(item.name) > 90:
                continue
        else:
            if len(other_candidate.parent_atoms) != current_parent_count:
                continue
        if len(other.name) > 180 or len(other.name) - len(item.name) > 80:
            if not charge_erasing_competitor and not charged_retained_parent:
                continue
        if other.stability_penalty > item.stability_penalty + 80:
            if (
                not (charge_erasing_competitor or charged_retained_parent)
                or other.stability_penalty > item.stability_penalty + 180
            ):
                continue
        return True
    return False


def _is_charge_located_single_ring_parent_candidate(candidate: NameCandidate) -> bool:
    if candidate.family not in {"single-ring-parent", "positive-single-ring-parent"}:
        return False
    evidence = candidate.charge
    if evidence is None or not evidence.charged_atoms:
        return False
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return False
    parent_count = len(candidate.parent_atoms)
    if parent_count < 4 or parent_count > 6:
        return False
    if candidate_safety_reasons(candidate):
        return False
    return True


def _is_charge_erasing_single_ring_competitor_candidate(candidate: NameCandidate) -> bool:
    if candidate.family not in _CHARGED_SINGLE_RING_PARENT_COMPETITOR_FAMILIES:
        return False
    if candidate.charge is not None and candidate.charge.charged_atoms:
        return False
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return False
    if candidate_safety_reasons(candidate):
        return False
    return True


def _is_charged_retained_parent_candidate(candidate: NameCandidate) -> bool:
    if candidate.family != "charged-retained-parent":
        return False
    if candidate.charge is not None and candidate.charge.charged_atoms:
        return False
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return False
    if candidate_safety_reasons(candidate):
        return False
    return True


def _is_six_membered_ring_dication_single_ring_parent(candidate: NameCandidate) -> bool:
    if not _is_charge_located_single_ring_parent_candidate(candidate):
        return False
    evidence = candidate.charge
    if evidence is None:
        return False
    if evidence.formal_charge != 2 or len(evidence.charged_atoms) != 2:
        return False
    if len(candidate.parent_atoms) != 6:
        return False
    if candidate.parent_element_counts != (("C", 4), ("N", 2)):
        return False
    if candidate.parent_aromatic_atom_count:
        return False
    if candidate.suffixes:
        return False
    return True


def _is_simple_charged_aromatic_single_ring_parent(candidate: NameCandidate) -> bool:
    if candidate.family not in {"single-ring-parent", "positive-single-ring-parent"}:
        return False
    if candidate.charge is None or not candidate.charge.charged_atoms:
        return False
    if candidate.suffixes:
        return False
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return False
    parent_count = len(candidate.parent_atoms)
    if parent_count < 5 or parent_count > 6:
        return False
    if candidate.parent_aromatic_atom_count != parent_count:
        return False
    if (
        candidate.parent_terminal_hetero_substituent_count
        or candidate.parent_external_nitrogen_substituent_count
        or candidate.parent_external_oxygen_substituent_count
    ):
        return False
    return True


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


def _is_contextual_senior_hetero_single_ring_parent(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family != "single-ring-parent":
        return False
    if candidate.suffixes or candidate.charge is not None:
        return False
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return False
    if candidate.safety is not None and candidate.safety.reasons:
        return False
    parent_count = len(candidate.parent_atoms)
    if parent_count < 5 or parent_count > 8:
        return False
    if len(item.name) > 120:
        return False
    if candidate.parent_hetero_atom_count <= 0:
        return False
    if (
        candidate.parent_aromatic_atom_count != parent_count
        and candidate.parent_hetero_atom_count == 1
        and _candidate_parent_element_count(candidate, "O") == 1
    ):
        return False
    if candidate.parent_aromatic_atom_count != parent_count and _has_equal_or_senior_hetero_single_ring_alternative(
        item,
        prepared,
    ):
        return False
    current_rank = candidate.render_rank or 0
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "single-ring-parent":
            continue
        if other_candidate.suffixes or other_candidate.charge is not None:
            continue
        if not other_candidate.parent_atoms or not candidate_has_complete_graph(other_candidate):
            continue
        if other_candidate.safety is not None and other_candidate.safety.reasons:
            continue
        if len(other_candidate.parent_atoms) != parent_count:
            continue
        if other_candidate.parent_hetero_atom_count >= candidate.parent_hetero_atom_count:
            continue
        other_rank = other_candidate.render_rank or 0
        if current_rank and other_rank and current_rank > other_rank + 4:
            continue
        if len(item.name) > 180 or len(item.name) - len(other.name) > 45:
            continue
        if item.stability_penalty > other.stability_penalty + 120:
            continue
        return True
    return False


def _has_equal_or_senior_hetero_single_ring_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "single-ring-parent":
            continue
        if not other_candidate.parent_atoms or not candidate_has_complete_graph(other_candidate):
            continue
        if other_candidate.safety is not None and other_candidate.safety.reasons:
            continue
        if other_candidate.parent_hetero_atom_count < candidate.parent_hetero_atom_count:
            continue
        other_parent_count = len(other_candidate.parent_atoms)
        if (
            other_candidate.parent_aromatic_atom_count != other_parent_count
            and not other_candidate.parent_terminal_hetero_substituent_count
        ):
            continue
        if len(other.name) > 180 or len(other.name) - len(item.name) > 45:
            continue
        if other.stability_penalty > item.stability_penalty + 80:
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
    current_parent_count = len(candidate.parent_atoms)
    if current_parent_count > 4:
        return False
    if current_parent_count <= 3 and candidate.parent_terminal_hetero_substituent_count:
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
        other_parent_count = len(other_candidate.parent_atoms)
        if other_parent_count != 6:
            continue
        if (
            candidate.parent_terminal_hetero_substituent_count
            or candidate.parent_external_nitrogen_substituent_count
        ) and (
            other_candidate.parent_aromatic_atom_count != other_parent_count
            and not other_candidate.parent_hetero_atom_count
        ):
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


def _has_contextual_aromatic_single_ring_parent_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family != "single-ring-parent":
        return False
    if candidate.charge is not None:
        return False
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return False
    if candidate.parent_aromatic_atom_count:
        return False
    if candidate.parent_hetero_atom_count:
        return False
    if candidate.parent_terminal_hetero_substituent_count:
        return False
    if candidate.parent_external_nitrogen_substituent_count:
        return False
    current_parent_count = len(candidate.parent_atoms)
    if current_parent_count >= 5 and not candidate.parent_external_oxygen_substituent_count:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "single-ring-parent":
            continue
        if not other_candidate.parent_atoms or not candidate_has_complete_graph(other_candidate):
            continue
        other_parent_count = len(other_candidate.parent_atoms)
        if other_parent_count < min(6, current_parent_count):
            continue
        if other_candidate.parent_aromatic_atom_count != other_parent_count:
            continue
        if len(other.name) > 180 or len(other.name) - len(item.name) > 55:
            continue
        if other.stability_penalty > item.stability_penalty + 40:
            continue
        return True
    return False


def _has_contextual_earlier_aromatic_single_ring_parent_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family != "single-ring-parent":
        return False
    if candidate.charge is not None:
        return False
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return False
    if candidate.parent_aromatic_atom_count:
        return False
    if candidate.parent_hetero_atom_count:
        return False
    if candidate.parent_terminal_hetero_substituent_count:
        return False
    current_rank = candidate.render_rank or 0
    if current_rank <= 1:
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
        other_parent_count = len(other_candidate.parent_atoms)
        if other_parent_count < 6:
            continue
        if other_candidate.parent_aromatic_atom_count != other_parent_count:
            continue
        if current_parent_count == 5:
            continue
        if current_parent_count >= 6 and other_parent_count < current_parent_count:
            continue
        if current_parent_count < 5 and other_parent_count < 7:
            continue
        other_rank = other_candidate.render_rank or 0
        if other_rank < 1 or other_rank >= current_rank:
            continue
        if other.stability_penalty > item.stability_penalty + 50:
            continue
        return True
    return False


def _has_contextual_earlier_external_hetero_single_ring_parent_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family != "single-ring-parent":
        return False
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return False
    if candidate.suffixes or candidate.charge is not None:
        return False
    if candidate.parent_terminal_hetero_substituent_count:
        return False
    if (
        candidate.parent_external_nitrogen_substituent_count
        or candidate.parent_external_oxygen_substituent_count
    ):
        return False
    current_rank = candidate.render_rank or 0
    if current_rank <= 1:
        return False
    parent_count = len(candidate.parent_atoms)
    if parent_count < 5 or parent_count > 6:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "single-ring-parent":
            continue
        if not other_candidate.parent_atoms or not candidate_has_complete_graph(other_candidate):
            continue
        if other_candidate.suffixes or other_candidate.charge is not None:
            continue
        if len(other_candidate.parent_atoms) != parent_count:
            continue
        if not (
            other_candidate.parent_external_nitrogen_substituent_count
            or other_candidate.parent_external_oxygen_substituent_count
        ):
            continue
        other_rank = other_candidate.render_rank or 0
        if other_rank < 1 or other_rank >= current_rank:
            continue
        if len(other.name) > 260 or len(other.name) - len(item.name) > 100:
            continue
        if other.stability_penalty > item.stability_penalty + 50:
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
        "cyclic-n-carbamate-carboxylate-retained-parent",
        "dithiophosphate-retained-parent",
        "phosphate-ester-retained-parent",
        "phosphorothioic-triamide-retained-parent",
        "thiooxo-ester-retained-parent",
        "urea-retained-parent",
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


_CYCLIC_N_CARBAMATE_FALLBACK_FAMILIES = frozenset(
    {
        "acyclic-skeleton-with-ring-substituents",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "functional-acyclic-skeleton-with-ring-substituents",
        "fused-retained-parent",
        "generic-bicyclic-parent",
        "kekulized-generic-polycyclic-parent",
        "retained-topology-replacement-fused-parent",
        "single-ring-parent",
    }
)


def _has_contextual_complete_cyclic_n_carbamate_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    if item.candidate.family not in _CYCLIC_N_CARBAMATE_FALLBACK_FAMILIES:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        if other.candidate.family != "cyclic-n-carbamate-carboxylate-retained-parent":
            continue
        if not candidate_has_complete_graph(other.candidate) or not other.candidate.parent_atoms:
            continue
        if len(other.name) > 220:
            continue
        if other.stability_penalty >= item.stability_penalty + 80:
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


_UREA_FALLBACK_FAMILIES = frozenset(
    {
        "acyclic-skeleton-with-ring-substituents",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "fused-retained-parent",
        "functional-acyclic-skeleton-with-ring-substituents",
        "generated-fused-parent",
        "generated-hydro-fused-parent",
        "generated-partial-hydro-fused-parent",
        "hydro-fused-retained-parent",
        "preferred-acyclic-skeleton-with-ring-substituents",
        "retained-topology-replacement-fused-parent",
        "single-ring-parent",
    }
)

_SULFONIMIDAMIDE_FALLBACK_FAMILIES = _UREA_FALLBACK_FAMILIES | frozenset(
    {
        "acyclic-parent",
        "functional-generic-bicyclic-parent",
        "generic-bicyclic-parent",
        "kekulized-generic-polycyclic-parent",
        "skeletal-replacement-fused-parent",
    }
)


def _has_contextual_complete_sulfonimidamide_parent_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    if item.candidate.family not in _SULFONIMIDAMIDE_FALLBACK_FAMILIES:
        return False
    if not candidate_has_complete_graph(item.candidate) or not item.candidate.parent_atoms:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "sulfonimidamide-parent":
            continue
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        if len(other.name) > 240 or len(other.name) - len(item.name) > 130:
            continue
        if other.stability_penalty > item.stability_penalty + 360:
            continue
        return True
    return False


def _has_contextual_complete_urea_parent_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    if item.candidate.family not in _UREA_FALLBACK_FAMILIES:
        return False
    if "carbamoyl" not in item.name or "amino" not in item.name:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "urea-retained-parent":
            continue
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        if len(other.name) > 160 or len(other.name) - len(item.name) > 80:
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


@lru_cache(maxsize=32768)
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


def _has_contextual_compact_bicyclic_parent_alternative_for_spiro_chain(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family != "spiro-functional-chain-substituent":
        return False
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return False
    if len(candidate.parent_atoms) < 14:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "functional-generic-bicyclic-parent":
            continue
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        parent_count = len(other_candidate.parent_atoms)
        if parent_count < 5 or parent_count > 10:
            continue
        if other_candidate.render_rank is not None and other_candidate.render_rank > 4:
            continue
        if len(other.name) > 260 or len(other.name) - len(item.name) > 80:
            continue
        if other.stability_penalty >= item.stability_penalty + 100:
            continue
        return True
    return False


def _has_contextual_compact_functional_bicyclic_parent_alternative_for_chain_decomposition(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family != "chain-dominant-acyclic-skeleton-with-ring-substituents":
        return False
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return False
    if len(candidate.parent_atoms) > 18:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "functional-generic-bicyclic-parent":
            continue
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        if len(other_candidate.parent_atoms) != 9:
            continue
        if other_candidate.render_rank is None or other_candidate.render_rank > 2:
            continue
        if len(other.name) > _DEFAULT_MAX_NAME_LENGTH or len(other.name) - len(item.name) > 160:
            continue
        if other.stability_penalty >= item.stability_penalty + 100:
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
    if _is_exact_opsin_dibenzofuran_parent(candidate):
        return False
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


_GENERATED_PURINE_PARENT_TEMPLATES = frozenset(
    {
        "imidazo[4,5-d]pyrimidine",
        "imidazo[4,5-e]pyrimidine",
        "pyrimido[5,6-d]imidazole",
        "pyrimidino[5,4-d]imidazole",
    }
)

_BIOCHEMICAL_RETAINED_FUSED_PARENT_TEMPLATES = frozenset(
    {
        "adenin",
        "guanin",
        "hypoxanthin",
        "pteridin",
        "pterin",
        "purin",
        "uric",
        "xanthin",
    }
)

_PURINE_LIKE_RETAINED_PARENT_TEMPLATES = frozenset(
    {
        "adenin",
        "guanin",
        "hypoxanthin",
        "mercaptopurin",
        "purin",
        "xanthin",
    }
)

_CHALCOGEN_RETAINED_PURINE_ALIAS_TEMPLATES = frozenset(
    {
        "guanin",
        "hypoxanthin",
        "isoguanin",
        "xanthin",
    }
)


def _has_contextual_purin_parent_alternative_for_chalcogen_retained_purine_alias(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family != "fused-retained-parent":
        return False
    if candidate.template_name not in _CHALCOGEN_RETAINED_PURINE_ALIAS_TEMPLATES:
        return False
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return False
    if len(candidate.parent_atoms) <= 9 or len(candidate.parent_atoms) > 12:
        return False
    if candidate.render_rank is not None and candidate.render_rank > 2:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "fused-retained-parent":
            continue
        if other_candidate.template_name != "purin":
            continue
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        if len(other_candidate.parent_atoms) != 9:
            continue
        if not other_candidate.parent_atoms.issubset(candidate.parent_atoms):
            continue
        if other_candidate.render_rank is None or other_candidate.render_rank > 4:
            continue
        if other_candidate.charge is not None:
            continue
        if len(other.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(other.name) - len(item.name) > 140:
            continue
        if other.stability_penalty > item.stability_penalty + 560:
            continue
        return True
    return False


def _has_contextual_pteridin_parent_alternative_for_substituted_pterin(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family != "fused-retained-parent":
        return False
    if candidate.template_name != "pterin":
        return False
    if candidate.source_family != "static-retained":
        return False
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return False
    if len(candidate.parent_atoms) != 12:
        return False
    if candidate.parent_substituted_hetero_atom_count < 1:
        return False
    if candidate.charge is not None or candidate.suffixes:
        return False
    if _candidate_parent_element_count(candidate, "N") != 5:
        return False
    if _candidate_parent_element_count(candidate, "O") != 1:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "fused-retained-parent":
            continue
        if other_candidate.template_name != "pteridin":
            continue
        if other_candidate.source_family != "opsin-retained-ring-cml":
            continue
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        if len(other_candidate.parent_atoms) != 10:
            continue
        if other_candidate.parent_substituted_hetero_atom_count != candidate.parent_substituted_hetero_atom_count:
            continue
        if other_candidate.charge is not None or other_candidate.suffixes:
            continue
        if other_candidate.render_rank is None or other_candidate.render_rank > 3:
            continue
        if _candidate_parent_element_count(other_candidate, "N") != 4:
            continue
        if _candidate_parent_element_count(other_candidate, "O"):
            continue
        if len(other.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(other.name) - len(item.name) > 80:
            continue
        if other.stability_penalty > item.stability_penalty + 80:
            continue
        return True
    return False


def _has_contextual_external_dinitrogen_generated_fused_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family != "generated-fused-parent":
        return False
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return False
    parent_count = len(candidate.parent_atoms)
    if parent_count < 8 or parent_count > 13:
        return False
    if candidate.parent_external_nitrogen_substituent_count < 2:
        return False
    if candidate.parent_external_oxygen_substituent_count:
        return False
    if candidate.suffixes or candidate.charge is not None:
        return False
    if candidate.safety is not None and candidate.safety.reasons:
        return False
    if candidate.render_rank is not None and candidate.render_rank > 5:
        return False
    alternative_families = _ACYCLIC_SKELETON_PARENT_FAMILIES | {"single-ring-parent"}
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family not in alternative_families:
            continue
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        if other_candidate.suffixes and other_candidate.family != "single-ring-parent":
            continue
        if other_candidate.charge is not None:
            continue
        if len(other.name) > _DEFAULT_MAX_NAME_LENGTH or len(other.name) - len(item.name) > 130:
            continue
        if other.stability_penalty >= item.stability_penalty + 90:
            continue
        return True
    return False


def _has_contextual_external_nitrogen_generated_fused_single_ring_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if not _is_external_nitrogen_generated_fused_shortcut_candidate(candidate):
        return False
    alternative_names: set[str] = set()
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if not _is_clean_aromatic_single_ring_parent_candidate(other_candidate):
            continue
        if len(other.name) > 80:
            continue
        if len(other.name) - len(item.name) > 90:
            continue
        if other.stability_penalty > item.stability_penalty + 90:
            continue
        alternative_names.add(other.name)
        if len(alternative_names) > 1:
            return False
    return len(alternative_names) == 1


def _is_external_nitrogen_generated_fused_shortcut_candidate(candidate: NameCandidate) -> bool:
    if candidate.family != "generated-fused-parent":
        return False
    if candidate.source_family != "generated-fused":
        return False
    if candidate.source_file != "generated_fused_template_index.json":
        return False
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return False
    parent_count = len(candidate.parent_atoms)
    if parent_count < 9 or parent_count > 14:
        return False
    if candidate.parent_aromatic_atom_count != parent_count:
        return False
    if candidate.parent_hetero_atom_count > 2:
        return False
    if candidate.parent_external_nitrogen_substituent_count != 1:
        return False
    if candidate.parent_terminal_hetero_substituent_count > 1:
        return False
    if candidate.suffixes or candidate.charge is not None:
        return False
    if candidate.safety is not None and candidate.safety.reasons:
        return False
    if candidate.render_rank is None or candidate.render_rank > 4:
        return False
    return True


def _is_clean_aromatic_single_ring_parent_candidate(candidate: NameCandidate) -> bool:
    if candidate.family != "single-ring-parent":
        return False
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return False
    if len(candidate.parent_atoms) != 6:
        return False
    if candidate.parent_aromatic_atom_count != 6:
        return False
    if candidate.parent_hetero_atom_count > 2:
        return False
    if candidate.suffixes or candidate.charge is not None:
        return False
    if candidate.safety is not None and candidate.safety.reasons:
        return False
    if candidate.render_rank is not None and candidate.render_rank > 3:
        return False
    return True


_PURINE_ACYCLIC_PHOSPHORUS_PARENT_FAMILIES = frozenset(
    {
        "acyclic-skeleton-with-ring-substituents",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "functional-acyclic-skeleton-with-ring-substituents",
        "preferred-acyclic-skeleton-with-ring-substituents",
    }
)


def _has_contextual_generated_purine_parent_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family != "fused-retained-parent":
        return False
    if candidate.template_name != "purin":
        return False
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return False
    if len(candidate.parent_atoms) != 9:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "generated-fused-parent":
            continue
        if other_candidate.template_name not in _GENERATED_PURINE_PARENT_TEMPLATES:
            continue
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        if len(other_candidate.parent_atoms) != len(candidate.parent_atoms):
            continue
        if other_candidate.render_rank is None or other_candidate.render_rank > 2:
            continue
        if _generated_purine_candidate_has_terminal_hetero_substituent(other_candidate):
            continue
        if len(other.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(other.name) - len(item.name) > 180:
            continue
        if other.stability_penalty >= item.stability_penalty + 120:
            continue
        return True
    return False


def _has_contextual_unanchored_generated_purine_parent_alternative_for_retained_purin(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family != "fused-retained-parent":
        return False
    if candidate.template_name != "purin":
        return False
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return False
    if len(candidate.parent_atoms) != 9:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if not _is_unanchored_indicated_h_generated_purine_candidate(other_candidate):
            continue
        if not _is_matching_retained_purin_candidate(candidate, other_candidate):
            continue
        if other_candidate.render_rank is None or other_candidate.render_rank > 4:
            continue
        if len(other.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(other.name) - len(item.name) > 180:
            continue
        if other.stability_penalty >= item.stability_penalty + 120:
            continue
        return True
    return False


def _has_contextual_acyclic_phosphorus_parent_alternative_for_purine_parent(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family == "generated-fused-parent":
        if candidate.template_name not in _GENERATED_PURINE_PARENT_TEMPLATES:
            return False
    elif candidate.family == "fused-retained-parent":
        if candidate.template_name not in _PURINE_LIKE_RETAINED_PARENT_TEMPLATES:
            return False
    elif candidate.family == "single-ring-parent":
        if not _has_contextual_purine_like_fused_parent_candidate(item, prepared):
            return False
    else:
        return False
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return False
    parent_count = len(candidate.parent_atoms)
    if candidate.family == "single-ring-parent":
        if parent_count > 6:
            return False
    elif parent_count < 9 or parent_count > 11:
        return False
    if _candidate_parent_element_count(candidate, "P"):
        return False
    if candidate.render_rank is not None and candidate.render_rank > 4:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family not in _PURINE_ACYCLIC_PHOSPHORUS_PARENT_FAMILIES:
            continue
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        other_parent_count = len(other_candidate.parent_atoms)
        if other_parent_count < 12 or other_parent_count > 13 or other_parent_count < parent_count + 2:
            continue
        if _candidate_parent_element_count(other_candidate, "P") != 1:
            continue
        if _candidate_parent_element_count(other_candidate, "S"):
            continue
        if len(other.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(other.name) - len(item.name) > 90:
            continue
        if other.stability_penalty > item.stability_penalty + 220:
            continue
        return True
    return False


def _has_contextual_purine_like_fused_parent_candidate(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        candidate = other.candidate
        if candidate.family == "generated-fused-parent":
            if candidate.template_name not in _GENERATED_PURINE_PARENT_TEMPLATES:
                continue
        elif candidate.family == "fused-retained-parent":
            if candidate.template_name not in _PURINE_LIKE_RETAINED_PARENT_TEMPLATES:
                continue
        else:
            continue
        if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
            continue
        if len(candidate.parent_atoms) < 9 or len(candidate.parent_atoms) > 11:
            continue
        if _candidate_parent_element_count(candidate, "P"):
            continue
        return True
    return False


def _has_contextual_first_generated_indazole_h_orientation_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family != "fused-retained-parent":
        return False
    if candidate.template_name != "imidazo[1,2-a]pyridine":
        return False
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return False
    if len(candidate.parent_atoms) != 9:
        return False
    if candidate.suffixes or candidate.charge is not None:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "generated-fused-parent":
            continue
        if other_candidate.template_name not in _GENERATED_INDAZOLE_H_ORIENTATION_TEMPLATES:
            continue
        if other_candidate.source_family != "generated-fused":
            continue
        if other_candidate.source_file != "generated_fused_template_index.json":
            continue
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        if len(other_candidate.parent_atoms) != 9:
            continue
        if other_candidate.render_rank != 1:
            continue
        if other_candidate.suffixes or other_candidate.charge is not None:
            continue
        if len(other.name) > 180 or len(other.name) - len(item.name) > 70:
            continue
        if other.stability_penalty > item.stability_penalty + 90:
            continue
        return True
    return False


_GENERATED_BENZAZOLE_PARENT_TEMPLATES = frozenset(
    {
        "benzo[1,2-d]oxazole",
        "benzo[1,2-d]thiazole",
        "imidazo[4,5-a]benzene",
    }
)


def _has_contextual_generated_benzazole_parent_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family != "fused-retained-parent":
        return False
    if candidate.template_name != "6,7-dihydro-1H-imidazo[4,5-c]pyridine":
        return False
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return False
    if len(candidate.parent_atoms) != 9:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "generated-fused-parent":
            continue
        if other_candidate.template_name not in _GENERATED_BENZAZOLE_PARENT_TEMPLATES:
            continue
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        if len(other_candidate.parent_atoms) != len(candidate.parent_atoms):
            continue
        if other_candidate.render_rank is None or other_candidate.render_rank > 2:
            continue
        if len(other.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(other.name) - len(item.name) > 90:
            continue
        if other.stability_penalty > item.stability_penalty + 80:
            continue
        return True
    return False


_PREFERRED_GENERATED_FUSED_TEMPLATE_ALTERNATIVES = {
    "imidazo[4,5-c]quinoline": frozenset({"quino[4,3-d]imidazole"}),
    "naphtho[8,7-d]imidazole": frozenset({"naphtho[7,8-d]imidazole"}),
    "pyrazolo[3,4-b]pyrazine": frozenset({"pyrazino[2,3-c]pyrazole"}),
    "pyrazolo[3,4-c]quinoline": frozenset({"quino[4,3-d]pyrazole"}),
    "pyrazolo[3,4-c]isoquinoline": frozenset({"isoquino[4,3-d]pyrazole"}),
    "pyrazolo[3,4-c]pyridazine": frozenset({"pyridazino[3,4-c]pyrazole"}),
    "pyrazolo[3,4-d]pyridazine": frozenset({"pyridazino[4,5-c]pyrazole"}),
    "pyrazolo[4,3-b]pyrazine": frozenset({"pyrazino[2,3-c]pyrazole"}),
    "pyrazolo[4,3-c]quinoline": frozenset({"quino[3,4-d]pyrazole"}),
    "pyrazolo[4,3-d]pyridazine": frozenset({"pyridazino[4,5-c]pyrazole"}),
    "pyrazolo[4,3-e]pyridazine": frozenset({"pyridazino[3,4-c]pyrazole"}),
    "pyridazino[6,5-d]imidazole": frozenset({"pyridazino[5,6-d]imidazole"}),
    "thieno[5,4-d]imidazole": frozenset({"thieno[4,5-d]imidazole"}),
}


_PYRAZOLO_PYRAZINE_GENERATED_TEMPLATES = frozenset(
    {
        "pyrazolo[3,4-b]pyrazine",
        "pyrazolo[4,3-b]pyrazine",
    }
)


_TERMINAL_AZOLE_CHALCOGEN_SUFFIX_KINDS = frozenset(
    {
        "ring_oxo_suffix",
        "ring_thione_suffix",
    }
)


_PROTECTED_SENIOR_RETAINED_PARENT_FAMILIES = frozenset(
    {
        "carbamate-retained-parent",
        "charged-urea-amidate-retained-parent",
        "dithiazole-retained-parent",
        "formamidine-retained-parent",
        "organophosphorus-oxide-parent",
        "phosphate-ester-retained-parent",
        "retained-amide-parent",
        "sulfonamide-parent",
        "sulfonamide-retained-parent",
    }
)


_PROTECTED_TERMINAL_AZOLE_RETAINED_FUSED_PARENT_FAMILIES = frozenset(
    {
        "fused-retained-parent",
        "retained-topology-replacement-fused-parent",
    }
)


_PROTECTED_TERMINAL_AZOLE_RETAINED_FUSED_TEMPLATE_NAMES = frozenset(
    {
        "benzo[d][1,3]oxazin-4-one",
        "isatoic anhydride",
    }
)


_QUINOXALIN_DICARBONYL_TEMPLATELESS_COMPETITOR_FAMILIES = frozenset(
    {
        "acyclic-skeleton-with-ring-substituents",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "preferred-acyclic-skeleton-with-ring-substituents",
        "single-ring-parent",
    }
)


def _has_contextual_preferred_generated_fused_template_alternative(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family != "generated-fused-parent":
        return False
    preferred_templates = _PREFERRED_GENERATED_FUSED_TEMPLATE_ALTERNATIVES.get(candidate.template_name or "")
    if not preferred_templates:
        return False
    if candidate.template_name in _PYRAZOLO_PYRAZINE_GENERATED_TEMPLATES and (
        candidate.parent_external_nitrogen_substituent_count
        or candidate.parent_terminal_heterocarbon_substituent_count
    ):
        return False
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return False
    if candidate.render_rank is None:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "generated-fused-parent":
            continue
        if other_candidate.template_name not in preferred_templates:
            continue
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        if len(other_candidate.parent_atoms) != len(candidate.parent_atoms):
            continue
        if other_candidate.render_rank is None or other_candidate.render_rank > max(4, candidate.render_rank):
            continue
        if len(other.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(other.name) - len(item.name) > 140:
            continue
        if other.stability_penalty > item.stability_penalty + 90:
            continue
        return True
    return False


def _has_contextual_terminal_azole_chalcogen_suffix_parent(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family not in {"single-ring-parent", "positive-single-ring-parent"}:
        return False
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return False
    if len(candidate.parent_atoms) != 5:
        return False
    if dict(candidate.parent_element_counts).get("S", 0):
        return False
    if candidate.parent_aromatic_atom_count != 5:
        return False
    if candidate.parent_hetero_atom_count < 3:
        return False
    if candidate.parent_terminal_hetero_substituent_count != 1:
        return False
    if candidate.parent_external_nitrogen_substituent_count:
        return False
    if candidate.charge is not None:
        return False
    if len(candidate.suffixes) != 1:
        return False
    if candidate.suffixes[0].kind not in _TERMINAL_AZOLE_CHALCOGEN_SUFFIX_KINDS:
        return False
    render_rank = candidate.render_rank or 0
    if render_rank < 1 or render_rank > 3:
        return False
    if candidate_safety_reasons(candidate):
        return False
    if len(item.name) > 260:
        return False
    has_competing_parent = False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family in _PROTECTED_SENIOR_RETAINED_PARENT_FAMILIES:
            if (
                candidate_has_complete_graph(other_candidate)
                and not candidate_safety_reasons(other_candidate)
                and len(other.name) <= len(item.name) + 100
            ):
                return False
        if other_candidate.family in _PROTECTED_TERMINAL_AZOLE_RETAINED_FUSED_PARENT_FAMILIES:
            if (
                other_candidate.template_name in _PROTECTED_TERMINAL_AZOLE_RETAINED_FUSED_TEMPLATE_NAMES
                and candidate_has_complete_graph(other_candidate)
                and not candidate_safety_reasons(other_candidate)
                and len(other.name) <= len(item.name) + 160
                and other.stability_penalty <= item.stability_penalty + 500
            ):
                return False
        if other_candidate.suffixes:
            continue
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        if other_candidate.family in _PROTECTED_SENIOR_RETAINED_PARENT_FAMILIES:
            continue
        if len(other.name) > len(item.name) + 120:
            continue
        if other.stability_penalty > item.stability_penalty + 320:
            continue
        has_competing_parent = True
    return has_competing_parent


def _terminal_azole_chalcogen_suffix_alternative_for_non_suffix_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.suffixes:
        return None
    if selected_candidate.family in _PROTECTED_SENIOR_RETAINED_PARENT_FAMILIES:
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None
    prepared = [item for _key, item in keyed_items]
    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        if not _has_contextual_terminal_azole_chalcogen_suffix_parent(item, prepared):
            continue
        if len(item.name) - len(selected.name) > 90:
            continue
        if candidate_key[0] > selected_key[0] + 340:
            continue
        alternatives.append((candidate_key, item))
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _terminal_azole_chalcogen_suffix_alternative_for_high_confidence_generated_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    if selected.candidate.family != "generated-fused-parent":
        return None
    alternative = _terminal_azole_chalcogen_suffix_alternative_for_non_suffix_choice(
        keyed_items,
        selected,
    )
    if alternative is None:
        return None
    if alternative.candidate.parent_external_substituted_hetero_atom_count:
        return None
    return alternative


def _hydrogenated_sulfur_oxide_single_ring_alternative_for_non_hydrogen_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.hydrogen is not None and selected_candidate.hydrogen.saturated_locs:
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None
    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if not _is_hydrogenated_sulfur_oxide_single_ring_suffix_parent(candidate):
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(item.name) - len(selected.name) > 80:
            continue
        if candidate_key[0] > selected_key[0] + 120:
            continue
        alternatives.append((candidate_key, item))
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _is_hydrogenated_sulfur_oxide_single_ring_suffix_parent(candidate: NameCandidate) -> bool:
    if candidate.family not in {"single-ring-parent", "positive-single-ring-parent"}:
        return False
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return False
    if candidate.charge is not None and candidate.charge.charged_atoms:
        return False
    if candidate.hydrogen is None or not candidate.hydrogen.saturated_locs:
        return False
    if candidate_safety_reasons(candidate):
        return False
    if not any(suffix.kind == "ring_sulfur_oxide_suffix" for suffix in candidate.suffixes):
        return False
    render_rank = candidate.render_rank or 0
    return 1 <= render_rank <= 4


_POST_RANKER_BASE_RESTORE_COMPETITOR_FAMILIES = frozenset(
    {
        "acyclic-skeleton-with-ring-substituents",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "generated-fused-parent",
        "generated-partial-hydro-fused-parent",
        "generic-bicyclic-parent",
        "high-signal-generated-topology-fused-parent",
        "neutral-amine-parent",
        "poly-component-spiro-parent",
        "retained-topology-replacement-fused-parent",
        "single-ring-parent",
        "skeletal-replacement-fused-parent",
    }
)

_STABLE_BASE_RESTORE_ACYCLIC_PARENT_FAMILIES = frozenset(
    {
        "acyclic-skeleton-with-ring-substituents",
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "functional-acyclic-skeleton-with-ring-substituents",
        "preferred-acyclic-skeleton-with-ring-substituents",
        "sulfur-oxo-acyclic-skeleton-with-ring-substituents",
    }
)

_POST_RANKER_BASE_RESTORE_ACYCLIC_SHORTCUT_FAMILIES = frozenset(
    {
        "fused-retained-parent",
        "functional-generic-bicyclic-parent",
        "generated-fused-parent",
        "generated-hydro-fused-parent",
        "generic-bicyclic-parent",
        "kekulized-generic-polycyclic-parent",
        "retained-topology-replacement-fused-parent",
        "single-ring-parent",
        "skeletal-replacement-fused-parent",
    }
)


def _stable_base_key_parent_alternative_for_post_ranker_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    base: _PreparedCandidate,
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    if base is selected or base.name == selected.name:
        return None
    base_candidate = base.candidate
    selected_candidate = selected.candidate
    if not _is_stable_base_key_parent_candidate(base_candidate):
        return None
    if not _is_post_ranker_base_restore_competitor(selected_candidate, base_candidate):
        return None
    if _post_ranker_base_restore_has_unsafe_context(
        keyed_items,
        base_candidate,
        selected_candidate,
    ):
        return None
    base_key = next((key for key, item in keyed_items if item is base), None)
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if base_key is None or selected_key is None:
        return None
    if selected_key < base_key:
        return None
    if len(base.name) > _DEFAULT_MAX_NAME_LENGTH:
        return None
    if len(base.name) - len(selected.name) > 145:
        return None
    if base.stability_penalty > selected.stability_penalty + 220:
        return None
    return base


def _sulfur_oxo_acyclic_alternative_for_generated_fused_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if not (
        (
            selected_candidate.family == "generated-fused-parent"
            and selected_candidate.source_family == "generated-fused"
            and selected_candidate.source_file == "generated_fused_template_index.json"
            and selected_candidate.render_rank is not None
            and selected_candidate.render_rank <= 3
        )
        or (
            selected_candidate.family == "fused-retained-parent"
            and selected_candidate.source_family == "static-retained"
            and selected_candidate.source_file == "high_impact_fused_parent_templates.json"
            and selected_candidate.render_rank == 1
        )
    ):
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    selected_parent_count = len(selected_candidate.parent_atoms)
    if selected_parent_count != 9:
        return None
    if (
        selected_candidate.family == "generated-fused-parent"
        and selected_candidate.parent_aromatic_atom_count != selected_parent_count
    ):
        return None
    if selected_candidate.charge is not None or selected_candidate.suffixes or selected_candidate.prefixes:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None

    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if not _is_sulfur_oxo_acyclic_base_restore_candidate(candidate):
            continue
        if not _is_provenanced_sulfur_oxo_acyclic_base_restore_candidate(candidate):
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(item.name) - len(selected.name) > 145:
            continue
        if item.stability_penalty > selected.stability_penalty + 220:
            continue
        if candidate_key[0] > selected_key[0] + 80:
            continue
        alternatives.append((candidate_key, item))
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _spiro25_retained_topology_alternative_for_decomposed_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family not in {
        "chain-dominant-acyclic-skeleton-with-ring-substituents",
        "spiro-functional-chain-substituent",
    }:
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    if selected_candidate.charge is not None or selected_candidate.suffixes:
        return None
    if candidate_safety_reasons(selected_candidate):
        return None
    selected_parent_count = len(selected_candidate.parent_atoms)
    if selected_candidate.family == "spiro-functional-chain-substituent":
        if selected_parent_count < 10 or selected_parent_count > 16:
            return None
    else:
        if selected_parent_count < 20 or len(selected.name) < 150:
            return None
        if selected_candidate.molecule_abs_formal_charge_sum:
            return None
        if selected_candidate.parent_external_unsaturated_bond_count:
            return None
        if selected_candidate.parent_internal_unsaturated_bond_count < 6:
            return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None

    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if not _is_spiro25_retained_topology_candidate(candidate):
            continue
        if selected_candidate.family == "spiro-functional-chain-substituent":
            if candidate.render_rank is None or candidate.render_rank > 12:
                continue
            if len(item.name) > len(selected.name):
                continue
            if candidate_key[0] > selected_key[0] + 360:
                continue
        else:
            if candidate.render_rank is None or candidate.render_rank > 4:
                continue
            if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
                continue
            if len(item.name) - len(selected.name) > 20:
                continue
            if candidate_key[0] > selected_key[0] + 340:
                continue
        if item.stability_penalty > selected.stability_penalty + 360:
            continue
        alternatives.append((candidate_key, item))
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _is_spiro25_retained_topology_candidate(candidate: NameCandidate) -> bool:
    if candidate.family != "retained-topology-replacement-fused-parent":
        return False
    if candidate.template_name != "spiro[2.5]octane":
        return False
    if candidate.source_family != "static-retained" or candidate.source_file:
        return False
    if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
        return False
    if len(candidate.parent_atoms) != 8:
        return False
    if candidate.parent_aromatic_atom_count:
        return False
    if candidate.charge is not None or candidate.suffixes or candidate.prefixes:
        return False
    if candidate_safety_reasons(candidate):
        return False
    return True


def _is_stable_base_key_parent_candidate(candidate: NameCandidate) -> bool:
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return False
    if candidate_safety_reasons(candidate):
        return False
    if candidate.charge is not None and candidate.charge.formal_charge not in {0, 1}:
        return False
    if candidate.suffixes and candidate.family not in {"single-ring-parent", "phosphorane-parent"}:
        return False
    parent_count = len(candidate.parent_atoms)
    render_rank = candidate.render_rank
    if candidate.family == "chain-dominant-acyclic-skeleton-with-ring-substituents":
        return parent_count >= 17 and candidate.priority <= 19
    if candidate.family in _STABLE_BASE_RESTORE_ACYCLIC_PARENT_FAMILIES:
        return _is_sulfur_oxo_acyclic_base_restore_candidate(candidate)
    if candidate.family == "fused-retained-parent":
        if render_rank is None or render_rank > 1:
            return False
        if candidate.source_family in {
            "opsin-retained-ring-cml",
            "simpleCyclicGroups:ring:simpleGroup",
        }:
            return parent_count == 12
        if not candidate.source_family and not candidate.source_file:
            return parent_count == 9
        return (
            candidate.source_family == "static-retained"
            and candidate.source_file == "high_impact_fused_parent_templates.json"
            and parent_count == 11
        )
    if candidate.family == "generated-fused-parent":
        return (
            candidate.source_family == "generated-fused"
            and candidate.source_file == "generated_fused_template_index.json"
            and render_rank == 1
            and parent_count in {9, 13, 14}
        )
    if candidate.family == "high-signal-generated-topology-fused-parent":
        return (
            candidate.source_family == "generated-fused"
            and candidate.source_file == "generated_fused_template_index.json"
            and render_rank == 1
            and parent_count == 9
        )
    if candidate.family == "generated-partial-hydro-fused-parent":
        return (
            candidate.source_family == "generated-fused"
            and candidate.source_file == "generated_fused_template_index.json"
            and render_rank == 1
            and parent_count >= 12
        )
    if candidate.family == "kekulized-generic-polycyclic-parent":
        return candidate.source_family == "kekulized" and render_rank == 1 and parent_count >= 17
    if candidate.family == "charged-ring-acyclic-skeleton-with-ring-substituents":
        return parent_count >= 26 and candidate.charge is not None and candidate.charge.charged_atoms
    if candidate.family == "phosphorane-parent":
        return candidate.source_family.startswith("phosphorane") and parent_count >= 20
    if candidate.family == "component-dispiro-parent":
        return parent_count >= 18
    return False


def _post_ranker_base_restore_has_unsafe_context(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    base: NameCandidate,
    selected: NameCandidate,
) -> bool:
    if _blocks_kekulized_base_restore_for_retained_topology(base, selected):
        return True
    if _is_sulfur_oxo_acyclic_base_restore_candidate(base):
        if selected.family not in _POST_RANKER_BASE_RESTORE_ACYCLIC_SHORTCUT_FAMILIES:
            return True
        if not selected.parent_atoms or not candidate_has_complete_graph(selected):
            return True
        if selected.suffixes or selected.charge is not None:
            return True
        if candidate_safety_reasons(selected):
            return True
        base_parent_count = len(base.parent_atoms)
        selected_parent_count = len(selected.parent_atoms)
        if _is_provenanced_sulfur_oxo_acyclic_base_restore_candidate(base):
            return not (
                selected_parent_count == 9
                and (
                    (
                        selected.family == "generated-fused-parent"
                        and selected.source_family == "generated-fused"
                        and selected.source_file == "generated_fused_template_index.json"
                        and selected.render_rank is not None
                        and selected.render_rank <= 3
                    )
                    or (
                        selected.family == "fused-retained-parent"
                        and selected.source_family == "static-retained"
                        and selected.source_file == "high_impact_fused_parent_templates.json"
                        and selected.render_rank == 1
                    )
                )
            )
        if selected.family not in {"fused-retained-parent", "generated-fused-parent"}:
            return True
        return not (selected_parent_count == 9 and 15 <= base_parent_count <= 16)
    if (
        base.family == "high-signal-generated-topology-fused-parent"
        and selected.family != "generated-fused-parent"
    ):
        return True
    if base.family == "generated-fused-parent":
        if (
            selected.family == "acyclic-skeleton-with-ring-substituents"
            and len(selected.parent_atoms) > len(base.parent_atoms)
        ):
            return True
        if selected.family == "single-ring-parent":
            return any(
                item.candidate.family == "retained-amide-parent" for _, item in keyed_items
            )
    return False


def _blocks_kekulized_base_restore_for_retained_topology(
    base: NameCandidate,
    selected: NameCandidate,
) -> bool:
    if base.family != "kekulized-generic-polycyclic-parent":
        return False
    if base.source_family != "kekulized" or base.template_name or base.source_file:
        return False
    if selected.family != "retained-topology-replacement-fused-parent":
        return False
    if selected.source_family not in _RETAINED_TOPOLOGY_REPLACEMENT_SOURCE_FAMILIES:
        return False
    if selected.template_name == "":
        return False
    if not base.parent_atoms or not selected.parent_atoms:
        return False
    if base.parent_atoms != selected.parent_atoms:
        return False
    if not candidate_has_complete_graph(base) or not candidate_has_complete_graph(selected):
        return False
    if base.parent_element_counts != selected.parent_element_counts:
        return False
    if base.parent_hetero_atom_count != selected.parent_hetero_atom_count:
        return False
    if selected.charge is not None or selected.suffixes or selected.prefixes:
        return False
    if candidate_safety_reasons(selected):
        return False
    return True


def _is_post_ranker_base_restore_competitor(
    selected: NameCandidate,
    base: NameCandidate,
) -> bool:
    if _is_sulfur_oxo_acyclic_base_restore_candidate(base):
        return selected.family in _POST_RANKER_BASE_RESTORE_ACYCLIC_SHORTCUT_FAMILIES
    if selected.family in _POST_RANKER_BASE_RESTORE_COMPETITOR_FAMILIES:
        return True
    if selected.family != "fused-retained-parent":
        return False
    if base.family != "fused-retained-parent":
        return False
    if selected.source_family in {
        "opsin-retained-ring-cml",
        "simpleCyclicGroups:ring:simpleGroup",
        "static-retained",
    }:
        return False
    selected_rank = selected.render_rank or 0
    base_rank = base.render_rank or 0
    return bool(selected_rank and base_rank and selected_rank > base_rank)


def _is_sulfur_oxo_acyclic_base_restore_candidate(candidate: NameCandidate) -> bool:
    if candidate.family not in _STABLE_BASE_RESTORE_ACYCLIC_PARENT_FAMILIES:
        return False
    if candidate.template_name or candidate.source_file:
        return False
    if candidate.source_family and candidate.source_family != "sulfur-oxo-acyclic-skeleton":
        return False
    if candidate.parent_hetero_atom_count or candidate.parent_aromatic_atom_count:
        return False
    if candidate.prefixes or candidate.suffixes or candidate.charge is not None:
        return False
    parent_count = len(candidate.parent_atoms)
    if _is_provenanced_sulfur_oxo_acyclic_base_restore_candidate(candidate):
        pass
    elif parent_count < 15 or parent_count > 16:
        return False
    element_counts = dict(candidate.parent_element_counts)
    min_oxygen_count = (
        3 if _is_provenanced_sulfur_oxo_acyclic_base_restore_candidate(candidate) else 4
    )
    return (
        element_counts.get("S", 0) >= 1
        and element_counts.get("O", 0) >= min_oxygen_count
        and element_counts.get("N", 0) >= 2
    )


def _is_provenanced_sulfur_oxo_acyclic_base_restore_candidate(candidate: NameCandidate) -> bool:
    return (
        candidate.family == "sulfur-oxo-acyclic-skeleton-with-ring-substituents"
        and candidate.source_family == "sulfur-oxo-acyclic-skeleton"
        and bool(candidate.parent_atoms)
        and len(candidate.parent_atoms) == 13
    )


def _retained_quinoxalin_dicarbonyl_alternative_for_templateless_choice(
    keyed_items: list[tuple[tuple, _PreparedCandidate]],
    selected: _PreparedCandidate,
) -> _PreparedCandidate | None:
    selected_candidate = selected.candidate
    if selected_candidate.family not in _QUINOXALIN_DICARBONYL_TEMPLATELESS_COMPETITOR_FAMILIES:
        return None
    if selected_candidate.template_name:
        return None
    if selected_candidate.charge is not None and selected_candidate.charge.charged_atoms:
        return None
    if not selected_candidate.parent_atoms or not candidate_has_complete_graph(selected_candidate):
        return None
    selected_key = next((key for key, item in keyed_items if item is selected), None)
    if selected_key is None:
        return None
    alternatives: list[tuple[tuple, _PreparedCandidate]] = []
    for candidate_key, item in keyed_items:
        if item is selected or item.name == selected.name:
            continue
        candidate = item.candidate
        if candidate.family != "fused-retained-parent":
            continue
        if candidate.template_name != "quinoxalin":
            continue
        if candidate.source_family != "opsin-retained-ring-cml":
            continue
        if not candidate.parent_atoms or not candidate_has_complete_graph(candidate):
            continue
        if len(candidate.parent_atoms) != 10:
            continue
        element_counts = dict(candidate.parent_element_counts)
        if element_counts.get("N", 0) != 2 or element_counts.get("C", 0) != 8:
            continue
        if candidate.parent_external_unsaturated_bond_count < 2:
            continue
        if candidate.charge is not None and candidate.charge.charged_atoms:
            continue
        if candidate.suffixes:
            continue
        if candidate_safety_reasons(candidate):
            continue
        render_rank = candidate.render_rank
        if render_rank is None or render_rank > 3:
            continue
        if len(item.name) > 260:
            continue
        if len(item.name) - len(selected.name) > 120:
            continue
        if candidate_key[0] > selected_key[0] + 260:
            continue
        alternatives.append((candidate_key, item))
    if not alternatives:
        return None
    return min(alternatives, key=lambda keyed: keyed[0])[1]


def _is_contextual_terminal_heterocarbon_pyrazolopyrazine_parent(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family != "generated-fused-parent":
        return False
    if candidate.template_name not in _PYRAZOLO_PYRAZINE_GENERATED_TEMPLATES:
        return False
    if candidate.parent_terminal_heterocarbon_substituent_count < 1:
        return False
    if candidate.parent_external_nitrogen_substituent_count:
        return False
    if not candidate_has_complete_graph(candidate) or not candidate.parent_atoms:
        return False
    if candidate.charge is not None or candidate.suffixes:
        return False
    if candidate_safety_reasons(candidate):
        return False
    current_rank = candidate.render_rank or 0
    if current_rank < 1 or current_rank > 3:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "generated-fused-parent":
            continue
        if other_candidate.template_name != "pyrazino[2,3-c]pyrazole":
            continue
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        if len(other_candidate.parent_atoms) != len(candidate.parent_atoms):
            continue
        if other_candidate.parent_terminal_heterocarbon_substituent_count != (
            candidate.parent_terminal_heterocarbon_substituent_count
        ):
            continue
        if other_candidate.parent_external_nitrogen_substituent_count:
            continue
        other_rank = other_candidate.render_rank or 0
        if other_rank < 1 or other_rank > 4:
            continue
        if len(item.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(item.name) - len(other.name) > 80:
            continue
        if item.stability_penalty > other.stability_penalty + 90:
            continue
        return True
    return False


def _has_contextual_complete_purine_parent_alternative_for_incomplete_organophosphorus(
    item: _PreparedCandidate,
    prepared: list[_PreparedCandidate],
) -> bool:
    candidate = item.candidate
    if candidate.family != "organophosphorus-oxide-parent":
        return False
    if candidate.parent_atoms:
        return False
    for other in prepared:
        if other is item or other.name == item.name:
            continue
        other_candidate = other.candidate
        if other_candidate.family != "generated-fused-parent":
            continue
        if other_candidate.template_name not in _GENERATED_PURINE_PARENT_TEMPLATES:
            continue
        if not candidate_has_complete_graph(other_candidate) or not other_candidate.parent_atoms:
            continue
        if len(other_candidate.parent_atoms) != 9:
            continue
        if other_candidate.render_rank is None or other_candidate.render_rank > 2:
            continue
        if len(other.name) > _DEFAULT_MAX_NAME_LENGTH:
            continue
        if len(other.name) - len(item.name) > 80:
            continue
        if other.stability_penalty > item.stability_penalty + 90:
            continue
        return True
    return False


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


@lru_cache(maxsize=32768)
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


@lru_cache(maxsize=32768)
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


@lru_cache(maxsize=32768)
def _has_stable_oxafluorene_alias(name: str) -> bool:
    return bool(re.search(r"(?:^|-)(?:\d+[a-z]?-)?oxafluoren\b", name))


@lru_cache(maxsize=32768)
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
    if candidate.source_family == "acyclic-carbohydrate-retained":
        bonus += 16
    if candidate.family in {"single-ring-parent", "positive-single-ring-parent"}:
        bonus += _charged_single_ring_parent_bonus(candidate)
        bonus += _single_ring_suffix_parent_bonus(candidate)
    elif candidate.family in {
        "fused-retained-parent",
        "generated-fused-parent",
        "generated-hydro-fused-parent",
        "generated-partial-hydro-fused-parent",
        "generated-spiro-partial-hydro-parent",
        "high-signal-generated-topology-fused-parent",
        "hydro-fused-retained-parent",
        "partial-hydro-fused-retained-parent",
        "retained-topology-replacement-fused-parent",
        "skeletal-replacement-fused-parent",
    }:
        bonus += _fused_functional_suffix_parent_bonus(candidate)
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
    if (
        "ring_sulfur_oxide_suffix" in kinds
        and candidate.hydrogen is not None
        and candidate.hydrogen.saturated_locs
    ):
        return 48
    if "ring_functional_suffix" in kinds:
        if any("fused-parent:functional-suffix" in suffix.source for suffix in candidate.suffixes):
            return 40
        return 18
    if kinds.intersection({"ring_imino_suffix", "ring_carbonyl_suffix"}):
        return 12
    return 0


def _fused_functional_suffix_parent_bonus(candidate: NameCandidate) -> int:
    if not candidate.suffixes or not candidate_has_complete_graph(candidate):
        return 0
    if any(
        suffix.kind == "ring_functional_suffix"
        and "fused-parent:functional-suffix" in suffix.source
        for suffix in candidate.suffixes
    ):
        return 40
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


@lru_cache(maxsize=32768)
def _has_retained_indicated_h_parent(name: str) -> bool:
    return bool(re.search(r"(?:^|-)\d+H-(?:indazol|purin)e?", name))


@lru_cache(maxsize=32768)
def _has_indicated_h_generated_fused_parent(name: str) -> bool:
    return bool(re.search(r"(?:^|-)\d+H-(?:pyrazolo|pyrrolo|imidazo)\[[^\]]+\](?:pyridin|pyrimidin)", name))


@lru_cache(maxsize=32768)
def _has_fused_indicated_h_ketone_suffix(name: str) -> bool:
    if "acenaphthen" in name:
        return False
    return bool(
        re.search(r"\[[^\]]+\].*-\d+\(\d+[a-z]?H\)-one\b", name)
        or re.search(r"\b(?:quinazolin|pteridin|triazafluoren)-\d+\(\d+[a-z]?H\)-one\b", name)
    )


@lru_cache(maxsize=32768)
def _has_fused_functional_suffix_parent(name: str) -> bool:
    return bool(
        re.search(
            r"(?:\[[^\]]+\]|benzimidazol|benzotriazol|indazol|quinazolin|pyrrolo|pyrazolo|imidazo).*-\d+[a-z]?-(?:carboxamide|carbonitrile|carboxylic acid|carboxylate)\b",
            name,
        )
    )


@lru_cache(maxsize=32768)
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


@lru_cache(maxsize=32768)
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


@lru_cache(maxsize=32768)
def _has_retained_parent_substituent_artifact(name: str) -> bool:
    return bool(re.search(r"(?:benzil|tolan)-\d+'?-yl\b", name))


def _has_fragile_bicyclic_carbamate_lactam_name(name: str) -> bool:
    if "bicyclo[" not in name or "oxycarbonyl" not in name:
        return False
    if "oxycarbonyl)amino" not in name or not ("oxo" in name or "one" in name):
        return False
    return True


@lru_cache(maxsize=32768)
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
    if candidate.source_family == "acyclic-carbohydrate-retained":
        name = _apply_uniform_isotope_label(mol, name)
        return name if len(name) <= max_length else ""
    structured_stereo_applied = False
    if candidate.stereo_prefix and not name.startswith("("):
        name = f"({candidate.stereo_prefix})-{name}"
        structured_stereo_applied = True
    elif candidate.stereo_locants:
        invert_unlocanted_bond_stereo = _candidate_needs_exocyclic_bond_stereo_inversion(
            mol,
            Chem,
            candidate,
        )
        reverse_descriptor_sequence = _candidate_prefers_reversed_mixed_stereo_prefix(
            mol,
            Chem,
            candidate,
        )
        reverse_atom_descriptor_sequence = False
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
            include_unlocanted=not (
                _has_retained_carbohydrate_substituent(name)
                or _has_relative_stereo_descriptor(name)
            ),
            invert_unlocanted_bond_stereo=invert_unlocanted_bond_stereo,
            invert_locanted_bond_stereo=_candidate_needs_endocyclic_bond_stereo_inversion(
                mol,
                Chem,
                candidate,
            ),
            reverse_descriptor_sequence=reverse_descriptor_sequence,
            reverse_atom_descriptor_sequence=reverse_atom_descriptor_sequence,
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
        "phosphite-ester-parent",
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
            invert_unlocanted_bond_stereo=_candidate_needs_exocyclic_bond_stereo_inversion(
                mol,
                Chem,
                candidate,
            ),
        )
    name = _apply_uniform_isotope_label(mol, name)
    return name if len(name) <= max_length else ""


def _candidate_needs_exocyclic_bond_stereo_inversion(mol, Chem, candidate: NameCandidate) -> bool:
    return (
        _candidate_needs_indene_exocyclic_carbon_stereo_inversion(mol, Chem, candidate)
        or _candidate_needs_carbofused_arylidene_substituent_stereo_inversion(mol, Chem, candidate)
    )


def _candidate_needs_endocyclic_bond_stereo_inversion(mol, Chem, candidate: NameCandidate) -> bool:
    if candidate.family != "single-ring-parent":
        return False
    if not candidate.parent_atoms or not candidate.stereo_locants:
        return False
    stereo_bonds = _assigned_double_stereo_bonds(mol, Chem)
    if len(stereo_bonds) != 1:
        return False
    bond = stereo_bonds[0]
    if not bond.IsInRing():
        return False
    begin_idx = bond.GetBeginAtomIdx()
    end_idx = bond.GetEndAtomIdx()
    parent_atoms = set(candidate.parent_atoms)
    if begin_idx not in parent_atoms or end_idx not in parent_atoms:
        return False
    begin_atom = mol.GetAtomWithIdx(begin_idx)
    end_atom = mol.GetAtomWithIdx(end_idx)
    if begin_atom.GetAtomicNum() != 6 or end_atom.GetAtomicNum() != 6:
        return False
    if begin_atom.GetIsAromatic() or end_atom.GetIsAromatic():
        return False
    if candidate.suffixes or candidate.charge is not None:
        return False
    locant_by_atom = {atom_idx: locant for atom_idx, locant in candidate.stereo_locants}
    if "1" not in {locant_by_atom.get(begin_idx), locant_by_atom.get(end_idx)}:
        return False
    ring_atoms = [
        ring
        for ring in mol.GetRingInfo().AtomRings()
        if begin_idx in ring and end_idx in ring
    ]
    return any(
        len(ring) == 8
        and all(
            mol.GetAtomWithIdx(atom_idx).GetAtomicNum() == 6
            and not mol.GetAtomWithIdx(atom_idx).GetIsAromatic()
            for atom_idx in ring
        )
        for ring in ring_atoms
    )


def _candidate_needs_indene_exocyclic_carbon_stereo_inversion(mol, Chem, candidate: NameCandidate) -> bool:
    if (
        candidate.family != "fused-retained-parent"
        or candidate.template_name != "inden"
        or not candidate.parent_atoms
        or not candidate.stereo_locants
    ):
        return False
    parent_atoms = set(candidate.parent_atoms)
    locant_by_atom = dict(candidate.stereo_locants)
    external_numeric_locants: set[str] = set()
    exocyclic_stereo_bonds = 0
    for parent_idx in parent_atoms:
        parent_atom = mol.GetAtomWithIdx(parent_idx)
        parent_locant = locant_by_atom.get(parent_idx, "")
        for neighbor in parent_atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx in parent_atoms or neighbor.GetAtomicNum() <= 1:
                continue
            if parent_locant.isdigit():
                external_numeric_locants.add(parent_locant)
            bond = mol.GetBondBetweenAtoms(parent_idx, neighbor_idx)
            if (
                parent_locant == "1"
                and neighbor.GetSymbol() == "C"
                and not neighbor.IsInRing()
                and bond is not None
                and bond.GetBondType() == Chem.BondType.DOUBLE
                and bond.GetStereo() != Chem.BondStereo.STEREONONE
            ):
                exocyclic_stereo_bonds += 1
    return (
        exocyclic_stereo_bonds == 1
        and {"1", "2", "3"}.issubset(external_numeric_locants)
        and "7" not in external_numeric_locants
    )


def _candidate_needs_carbofused_arylidene_substituent_stereo_inversion(mol, Chem, candidate: NameCandidate) -> bool:
    """Invert OPSIN's ylidene-substituent E/Z projection for arylidene indenes.

    Direct indene parents already roundtrip with RDKit's descriptor.  The
    inversion is only needed when the carbofused five-member endpoint is outside
    the selected parent and is rendered through a substituent/branch path.
    """

    if not candidate.parent_atoms:
        return False
    stereo_bonds = _assigned_double_stereo_bonds(mol, Chem)
    if len(stereo_bonds) != 1:
        return False
    bond = stereo_bonds[0]
    if bond.IsInRing():
        return False
    begin_idx = bond.GetBeginAtomIdx()
    end_idx = bond.GetEndAtomIdx()
    begin_atom = mol.GetAtomWithIdx(begin_idx)
    end_atom = mol.GetAtomWithIdx(end_idx)
    if begin_atom.GetAtomicNum() != 6 or end_atom.GetAtomicNum() != 6:
        return False
    if begin_atom.GetIsAromatic() or end_atom.GetIsAromatic():
        return False
    begin_in_ring = begin_atom.IsInRing()
    end_in_ring = end_atom.IsInRing()
    if begin_in_ring == end_in_ring:
        return False
    ring_idx, outer_idx = (begin_idx, end_idx) if begin_in_ring else (end_idx, begin_idx)
    parent_atoms = set(candidate.parent_atoms)
    if ring_idx in parent_atoms:
        return False
    ring_substituent_neighbors = [
        nbr.GetIdx()
        for nbr in mol.GetAtomWithIdx(ring_idx).GetNeighbors()
        if nbr.GetIdx() != outer_idx and nbr.GetAtomicNum() > 1
    ]
    if any(nidx in parent_atoms for nidx in ring_substituent_neighbors):
        return False
    return (
        _atom_in_carbofused_five_ring_with_aromatic_fusion(mol, ring_idx)
        and _atom_has_aromatic_neighbor_except(mol, ring_idx, outer_idx)
        and _atom_has_aromatic_neighbor_except(mol, outer_idx, ring_idx)
    )


def _assigned_double_stereo_bonds(mol, Chem):
    cached = mol_cache_get(mol, "assigned_double_stereo_bonds")
    if cached is not None:
        return cached
    return mol_cache_set(mol, "assigned_double_stereo_bonds", tuple(
        bond
        for bond in mol.GetBonds()
        if bond.GetBondType() == Chem.BondType.DOUBLE
        and bond.GetStereo()
        in {
            Chem.BondStereo.STEREOE,
            Chem.BondStereo.STEREOTRANS,
            Chem.BondStereo.STEREOZ,
            Chem.BondStereo.STEREOCIS,
        }
    ))


def _atom_has_aromatic_neighbor_except(mol, atom_idx: int, blocked_idx: int) -> bool:
    return any(
        nbr.GetIdx() != blocked_idx and nbr.GetAtomicNum() > 1 and nbr.GetIsAromatic()
        for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
    )


def _atom_in_carbofused_five_ring_with_aromatic_fusion(mol, atom_idx: int) -> bool:
    for ring in mol.GetRingInfo().AtomRings():
        if atom_idx not in ring or len(ring) != 5:
            continue
        if any(mol.GetAtomWithIdx(idx).GetAtomicNum() != 6 for idx in ring):
            continue
        if _ring_has_exocyclic_carbonyl(mol, ring):
            continue
        aromatic_count = sum(1 for idx in ring if mol.GetAtomWithIdx(idx).GetIsAromatic())
        if aromatic_count == 2:
            return True
    return False


def _ring_has_exocyclic_carbonyl(mol, ring: tuple[int, ...]) -> bool:
    ring_atoms = set(ring)
    for atom_idx in ring_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetAtomicNum() != 6:
            continue
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() in ring_atoms or nbr.GetAtomicNum() != 8:
                continue
            bond = mol.GetBondBetweenAtoms(atom_idx, nbr.GetIdx())
            if bond is not None and int(bond.GetBondTypeAsDouble()) == 2:
                return True
    return False


_MIXED_STEREO_COMPLEX_PARENT_FAMILIES = frozenset(
    {
        "fused-retained-parent",
        "generated-fused-parent",
        "generated-hydro-fused-parent",
        "generated-partial-hydro-fused-parent",
        "hydro-fused-retained-parent",
        "kekulized-generic-polycyclic-parent",
        "retained-topology-replacement-fused-parent",
    }
)


def _candidate_prefers_reversed_mixed_stereo_prefix(mol, Chem, candidate: NameCandidate) -> bool:
    if candidate.family not in _MIXED_STEREO_COMPLEX_PARENT_FAMILIES:
        return False
    if not candidate.stereo_locants or not candidate.parent_atoms:
        return False
    if not candidate_has_complete_graph(candidate):
        return False
    if len(candidate.parent_atoms) < 8 or mol.GetRingInfo().NumRings() < 2:
        return False
    locanted_atoms = {atom_idx for atom_idx, _locant in candidate.stereo_locants}
    assigned_atoms = {atom_idx for atom_idx, _descriptor in assigned_atom_stereochemistry(mol, Chem)}
    if not assigned_atoms or assigned_atoms.issubset(locanted_atoms):
        return False
    if len(assigned_atoms & locanted_atoms) < 4:
        return False
    external_assigned_atoms = assigned_atoms - locanted_atoms
    if len(external_assigned_atoms) < 2:
        return False
    if not any(atom_idx not in candidate.parent_atoms for atom_idx in external_assigned_atoms):
        return False
    if _has_any_assigned_bond_stereo(mol, Chem) and not _has_external_assigned_bond_stereo(mol, Chem, candidate):
        return False
    return True


def _has_any_assigned_bond_stereo(mol, Chem) -> bool:
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE and bond.GetStereo() != Chem.BondStereo.STEREONONE:
            return True
    return False


def _has_external_assigned_bond_stereo(mol, Chem, candidate: NameCandidate) -> bool:
    parent_atoms = set(candidate.parent_atoms)
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        if bond.GetStereo() == Chem.BondStereo.STEREONONE:
            continue
        if bond.GetBeginAtomIdx() not in parent_atoms or bond.GetEndAtomIdx() not in parent_atoms:
            return True
    return False


def _has_retained_carbohydrate_substituent(name: str) -> bool:
    return any(
        fragment in name
        for fragment in (
            "arabinosyl",
            "fructosyl",
            "fucosyl",
            "galactosyl",
            "glucopyranosyl",
            "glucosyl",
            "galactopyranosyl",
            "lactosyl",
            "maltosyl",
            "mannosyl",
            "mannopyranosyl",
            "fucopyranosyl",
            "ribofuranosyl",
            "ribosyl",
            "xylosyl",
            "apiosyl",
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
    labels = _heavy_isotope_label_groups(mol)
    if len(labels) != 1:
        multi_label_name = _multi_heavy_isotope_prefix_name(name, labels)
        if multi_label_name:
            return multi_label_name
        return name
    symbol, isotope, count, total_count = labels[0]
    if f"{isotope}{symbol}" in name:
        return name
    if symbol == "C" and isotope in {13, 14}:
        if count != total_count:
            count_text = str(count) if count > 1 else ""
            if _prefers_prefix_carbon_isotope_label(name):
                return f"[{isotope}{symbol}{count_text}]{name}"
            return f"{name}-{isotope}{symbol}{count_text}"
        count_text = str(count) if count > 1 else ""
        return f"{name}-{isotope}{symbol}{count_text}"
    if symbol == "C":
        count_text = str(count) if count > 1 else ""
        if _prefers_prefix_carbon_isotope_label(name):
            return f"[{isotope}{symbol}{count_text}]{name}"
        return f"{name}-{isotope}{symbol}{count_text}"
    if symbol == "F" and isotope == 18 and count == 1:
        return _inline_single_fluorine_isotope(name)
    if symbol == "N" and count == 1 and not name.startswith("("):
        return f"[{isotope}{symbol}]{name}"
    if symbol in {"B", "N", "O", "P", "S", "Se"} and not name.startswith("("):
        count_text = str(count) if count > 1 else ""
        return f"[{isotope}{symbol}{count_text}]{name}"
    return name


def _heavy_isotope_label_groups(mol) -> list[tuple[str, int, int, int]]:
    totals: dict[str, int] = {}
    labeled: dict[tuple[str, int], int] = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        symbol = atom.GetSymbol()
        totals[symbol] = totals.get(symbol, 0) + 1
        isotope = atom.GetIsotope()
        if isotope:
            key = (symbol, isotope)
            labeled[key] = labeled.get(key, 0) + 1
    return [
        (symbol, isotope, count, totals.get(symbol, count))
        for (symbol, isotope), count in sorted(labeled.items())
    ]


def _prefers_prefix_carbon_isotope_label(name: str) -> bool:
    if name.startswith("("):
        return False
    if re.match(r"^\d", name):
        return False
    if len(name) > 16:
        return False
    return not any(token in name for token in (",", "(", ")", "[", "]"))


def _multi_heavy_isotope_prefix_name(name: str, labels: list[tuple[str, int, int, int]]) -> str:
    if not labels or len(labels) > 3 or name.startswith("("):
        return ""
    if any(f"{isotope}{symbol}" in name for symbol, isotope, _count, _total in labels):
        return ""
    if len(name) > 24 and name not in {"urea", "hydrazine"}:
        return ""
    label_text = ",".join(
        f"{isotope}{symbol}{count if count > 1 else ''}"
        for symbol, isotope, count, _total in labels
    )
    return f"[{label_text}]{name}"


def _inline_single_fluorine_isotope(name: str) -> str:
    if "[18F]fluoro" in name:
        return name
    return name.replace("fluoro", "[18F]fluoro", 1) if "fluoro" in name else name


def _mol_has_isotope(mol) -> bool:
    cached = mol_cache_get(mol, "has_isotope")
    if cached is not None:
        return cached
    return mol_cache_set(mol, "has_isotope", any(atom.GetIsotope() for atom in mol.GetAtoms()))


def _mol_has_explicit_isotope_hydrogen(mol) -> bool:
    cached = mol_cache_get(mol, "has_explicit_isotope_hydrogen")
    if cached is not None:
        return cached
    return mol_cache_set(
        mol,
        "has_explicit_isotope_hydrogen",
        any(atom.GetAtomicNum() == 1 and atom.GetIsotope() in {2, 3} for atom in mol.GetAtoms()),
    )


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
