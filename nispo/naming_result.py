"""Public delivery status, separate from the spelling of a generated name.

``resolved`` means the applicable implemented selection completed. It is not
an independent certification that the result is a PIN. ``not-assessed`` marks
legacy routes without a complete preferred-name comparison.
"""

from dataclasses import dataclass, field
from .selection_evidence import PreferredNameJustification


@dataclass(frozen=True, slots=True)
class SelectionLimitation:
    stage: str
    rule: str
    reason: str


@dataclass(frozen=True, slots=True)
class NamingResult:
    name: str
    generation_status: str
    preferred_status: str
    unresolved: tuple[SelectionLimitation, ...] = ()
    route: str = ""
    justification: PreferredNameJustification | None = None


def _selection_issues(decision):
    """Do not hide pending attached evidence behind a completed writer."""
    issues = list(decision.preference_issues)
    issues.extend(getattr(decision.prepared, 'preference_issues', ()))
    justification = (getattr(decision, 'justification', None)
                     or getattr(decision.comparison, 'justification', None))
    if justification is not None:
        issues.extend(SelectionLimitation(step.stage, step.rule, step.reason)
                      for step in justification.steps if step.status == 'pending')
    pending = {'pending', 'blocked', 'conflict', 'barrier', 'unresolved',
               'deferred', 'preselected-parent'}
    for field_name in ('comparison', 'source_class_constraint', 'deferred_parent_resolution'):
        evidence = getattr(decision, field_name, None)
        status = getattr(evidence, 'status', getattr(evidence, 'disposition', ''))
        if getattr(status, 'value', status) not in pending:
            continue
        issue = SelectionLimitation(
            decision.stage, getattr(evidence, 'rule', '') or decision.direct_route or decision.stage,
            getattr(evidence, 'reason', '') or field_name + ' remains unresolved')
        if issue not in issues:
            issues.append(issue)
    if not decision.output_name and decision.prepared is None and not issues:
        evidence = decision.comparison
        issues.append(SelectionLimitation(
            decision.stage, getattr(evidence, 'rule', '') or decision.direct_route or decision.stage,
            getattr(evidence, 'reason', '') or decision.requirement.kind))
    return tuple(issues)


@dataclass(slots=True)
class _SelectionState:
    preferred_status: str = "not-assessed"
    unresolved: list[SelectionLimitation] = field(default_factory=list)
    route: str = ""
    justification: PreferredNameJustification | None = None

    def record(self, decision):
        self.route = decision.direct_route or decision.stage
        self.justification = (getattr(decision, 'justification', None)
                              or getattr(decision.comparison, 'justification', None))
        for issue in _selection_issues(decision):
            if issue not in self.unresolved:
                self.unresolved.append(issue)
        if self.unresolved:
            self.preferred_status = "unresolved"
        elif self.justification is not None and self.justification.complete:
            self.preferred_status = "resolved"
        else:
            self.preferred_status = "not-assessed"

    def result(self, name):
        justification = self.justification
        issues = list(self.unresolved)
        if justification is not None:
            from dataclasses import replace
            from .selection_evidence import SelectionStep
            if justification.complete:
                from .prefix_presentation import preferred_prefix_presentation
                if preferred_prefix_presentation(justification.name) != name:
                    issues.append(SelectionLimitation('delivery', 'P-44/P-45',
                        'The delivered name differs from the construction bound to the preferred-selection evidence.'))
            if justification.complete and issues:
                justification = replace(justification, steps=(*justification.steps,
                    *(SelectionStep(issue.stage, issue.rule, 'pending', issue.reason)
                      for issue in issues)))
            justification = replace(justification, name=name)
        return NamingResult(
            name, "available" if name else "unavailable",
            ('unresolved' if issues else self.preferred_status) if name or issues else "not-assessed",
            tuple(issues), self.route, justification,
        )


@dataclass(slots=True)
class _MappedSelectionState(_SelectionState):
    """Opt-in original-atom evidence tied to the actually delivered name.

    An ordinary result does not acquire a guessed map when a legacy route
    supplies none. The consumer must additionally prove its descriptor scope.
    """
    mapped_outputs: list = field(default_factory=list)

    def record(self, decision):
        super(_MappedSelectionState,self).record(decision)
        prepared=decision.prepared
        name=prepared.name if prepared is not None else decision.output_name
        mapping=(getattr(prepared.candidate,'stereo_locants',())
                 if prepared is not None else decision.atom_locants)
        if name and mapping:
            self.mapped_outputs.append((name,tuple(mapping)))

    def mapping_for(self,name):
        from .prefix_presentation import preferred_prefix_presentation
        maps={mapping for output,mapping in self.mapped_outputs
              if preferred_prefix_presentation(output)==name}
        return next(iter(maps)) if len(maps)==1 else ()
