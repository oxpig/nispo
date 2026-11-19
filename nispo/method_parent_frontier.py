"""Status-bearing comparison of supplied naming-method parent proposals.

This module does not discover eligible parents. Callers supply proposals whose
method eligibility and mapped suffix/prefix coverage have already been established.
Only ``ParentFacts`` comparisons can establish preference; candidate order,
name length and enumeration keys are deliberately ignored.
"""

from dataclasses import dataclass
from enum import Enum

from .parent_selection import compare_parents
from .selection_evidence import CandidateCompleteness, proposal_identity


class MethodParentStatus(str, Enum):
    SELECTED = "selected"
    TIE = "tie"
    UNRESOLVED = "unresolved"
    OUT_OF_SCOPE = "out-of-scope"


@dataclass(frozen=True, slots=True)
class MethodParentPairEvidence:
    left_index: int
    right_index: int
    preference: int | None
    rule: str
    reason: str


@dataclass(frozen=True, slots=True)
class MethodParentFrontier:
    status: MethodParentStatus
    candidates: tuple = ()
    survivor_indices: tuple[int, ...] = ()
    pairwise: tuple[MethodParentPairEvidence, ...] = ()
    reason: str = ""

    @property
    def selected(self):
        """One usable proposal when the preferred name is fully established."""
        if self.status == MethodParentStatus.SELECTED and len(self.candidates) == 1:
            return self.candidates[0]
        if self.status == MethodParentStatus.TIE and self.candidates:
            names = {getattr(candidate, "name", None) for candidate in self.candidates}
            if len(names) == 1:
                return self.candidates[0]
        return None


def select_method_parent(candidates, facts, *, physical_parent_identity=None,
                         proposal_set_complete=False, completeness=None, record_pairwise=True):
    """Select among already-eligible proposals using pairwise source facts.

    Proved dominations are removed first. A comparison cycle is a barrier.
    Any unknown comparison among the remaining candidates is unresolved. A proved
    tie is reportable only when all survivors have the same rendered preferred
    name. A supplied physical-parent identity adds evidence to a same-spelling
    tie; it never licenses a choice between different spellings. The caller must
    certify proposal-pool completeness before any supplied pool can be selected;
    an incomplete pool remains unresolved even when one proposal was rendered.
    Callers that need only the selection may omit the pairwise evidence records.
    """
    candidates = tuple(candidates)
    if not candidates:
        return MethodParentFrontier(MethodParentStatus.OUT_OF_SCOPE,
                                    reason="no-eligible-parent-proposals")

    parent_facts = tuple(facts(candidate) for candidate in candidates)
    if (not record_pairwise and proposal_set_complete and completeness is None
            and len(candidates) > 1 and all(value is None for value in parent_facts)):
        return MethodParentFrontier(
            MethodParentStatus.UNRESOLVED, candidates, tuple(range(len(candidates))),
            reason="surviving-parent-comparison-is-unknown",
        )
    evidence = []
    edges = [set() for _ in candidates]
    unknown = set()
    tied = set()
    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            comparison = compare_parents(parent_facts[left], parent_facts[right])
            if record_pairwise:
                evidence.append(MethodParentPairEvidence(
                    left, right, comparison.preference, comparison.rule, comparison.reason,
                ))
            if comparison.preference == -1:
                edges[left].add(right)
            elif comparison.preference == 1:
                edges[right].add(left)
            elif comparison.preference is None:
                unknown.add((left, right))
            else:
                tied.add((left, right))

    # Contradictory source facts must not become a tie after pruning.
    visiting = set()
    visited = set()

    def has_cycle(index):
        if index in visiting:
            return True
        if index in visited:
            return False
        visiting.add(index)
        if any(has_cycle(other) for other in edges[index]):
            return True
        visiting.remove(index)
        visited.add(index)
        return False

    if any(has_cycle(index) for index in range(len(candidates))):
        return MethodParentFrontier(
            MethodParentStatus.UNRESOLVED, candidates,
            tuple(range(len(candidates))), tuple(evidence),
            "pairwise-parent-preferences-contain-a-cycle",
        )

    # An edge points from a senior proposal to a proposal it proves inferior.
    dominated = {right for outgoing in edges for right in outgoing}
    survivors = tuple(index for index in range(len(candidates)) if index not in dominated)
    if completeness is not None:
        proposal_set_complete = (isinstance(completeness, CandidateCompleteness)
                                 and completeness.complete
                                 and set(completeness.proposal_keys)
                                 == {proposal_identity(candidate) for candidate in candidates})
    if not proposal_set_complete:
        return MethodParentFrontier(
            MethodParentStatus.UNRESOLVED,
            tuple(candidates[index] for index in survivors), survivors,
            tuple(evidence), "eligible-proposal-pool-completeness-is-unproved",
        )
    if len(survivors) == 1:
        selected_index = survivors[0]
        self_comparison = compare_parents(parent_facts[selected_index], parent_facts[selected_index])
        if self_comparison.preference is None:
            return MethodParentFrontier(
                MethodParentStatus.UNRESOLVED, (candidates[selected_index],), survivors,
                tuple(evidence), 'selected-parent-has-unproved-applicable-facts:' + self_comparison.rule,
            )
        # A unique maximum must be proved senior to every other proposal.
        incomplete = [index for index in range(len(candidates))
                      if index != selected_index
                      and selected_index not in edges[index]
                      and index not in edges[selected_index]]
        if incomplete:
            return MethodParentFrontier(
                MethodParentStatus.UNRESOLVED, candidates, survivors,
                tuple(evidence), "unique-survivor-has-unproved-pairwise-comparisons",
            )
        return MethodParentFrontier(
            MethodParentStatus.SELECTED, (candidates[selected_index],), survivors,
            tuple(evidence), "one-proposal-is-proved-senior",
        )

    survivor_set = set(survivors)
    if any(left in survivor_set and right in survivor_set for left, right in unknown):
        return MethodParentFrontier(
            MethodParentStatus.UNRESOLVED,
            tuple(candidates[index] for index in survivors), survivors,
            tuple(evidence), "surviving-parent-comparison-is-unknown",
        )

    # If there is no unknown survivor pair, all maximal proposals must be tied
    # pairwise. P-45.5 and earlier rules are represented by compare_parents.
    survivor_pairs = {
        (left, right) for left in survivors for right in survivors if left < right
    }
    if survivor_pairs and not survivor_pairs.issubset(tied):
        return MethodParentFrontier(
            MethodParentStatus.UNRESOLVED,
            tuple(candidates[index] for index in survivors), survivors,
            tuple(evidence), "surviving-parent-preferences-are-inconsistent",
        )

    survivor_candidates = tuple(candidates[index] for index in survivors)
    if any(compare_parents(parent_facts[index], parent_facts[index]).preference is None
           for index in survivors):
        return MethodParentFrontier(
            MethodParentStatus.UNRESOLVED, survivor_candidates, survivors,
            tuple(evidence), 'tied-parents-have-unproved-applicable-facts',
        )
    names = {getattr(candidate, "name", None) for candidate in survivor_candidates}
    if None in names or len(names) != 1:
        return MethodParentFrontier(
            MethodParentStatus.UNRESOLVED, survivor_candidates, survivors,
            tuple(evidence), "proved-parent-tie-has-distinct-rendered-names",
        )
    if len(survivors) > 1 and physical_parent_identity is not None:
        identities = tuple(physical_parent_identity(candidate)
                           for candidate in survivor_candidates)
        # Identity does not choose between distinct tied parents. It only
        # licenses retaining multiple renderings of one physical parent.
        if all(identity is not None for identity in identities) and len(set(identities)) == 1:
            reason = "proved-same-name-tie-and-equivalent-physical-parent"
        else:
            reason = "proved-same-name-parent-tie"
    else:
        reason = "proved-same-name-parent-tie"
    return MethodParentFrontier(
        MethodParentStatus.TIE, survivor_candidates, survivors,
        tuple(evidence), reason,
    )
