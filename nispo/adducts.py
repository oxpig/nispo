"""Explicit P-14.8 component proportions and adduct/solvate nomenclature.

An adduct is an explicit input operation. Disconnected SMILES keep their
existing molecular/salt meaning. Component preferred-name limitations and
unknown class/order comparisons remain visible in the result.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd, lcm
from typing import Iterable

from .input_stereochemistry import (
    unsupported_relative_stereochemistry,
    unsupported_relative_stereochemistry_result,
)
from .naming_result import NamingResult, SelectionLimitation


@dataclass(frozen=True, slots=True)
class AdductComponent:
    """One molecular component and its positive exact relative amount.

    Use an integer, Fraction or rational string (for example ``"3/2"``).
    Floating-point amounts are not exact composition statements.
    """
    smiles: str
    amount: int | str | Fraction = 1


@dataclass(frozen=True, slots=True)
class NamedAdductComponent:
    canonical_smiles: str
    name: str
    proportion: int
    organic: bool
    class_rank: int | None
    result: NamingResult


@dataclass(frozen=True, slots=True)
class AdductNamingResult(NamingResult):
    components: tuple[NamedAdductComponent, ...] = ()
    rules: tuple[str, ...] = ("P-14.8.1", "P-14.8.2")


def _invalid(reason):
    return AdductNamingResult(
        name="", generation_status="invalid-input", preferred_status="not-assessed",
        unresolved=(SelectionLimitation("adduct-input", "P-14.8", reason),),
        route="explicit-adduct")


def _class_facts(mol):
    """Return proved P-41 class facts; do not infer a class from name text."""
    from .api import Chem
    from .adduct_priority import component_priority
    priority=component_priority(mol,Chem)
    return priority.class_rank,priority.functional_facts


def adduct_to_iupac_result(components: Iterable[AdductComponent]) -> AdductNamingResult:
    """Name an explicitly specified adduct, preserving exact proportions.

    Organic P-41 class order, organic-before-inorganic and water-last order
    are implemented. Unclosed same-class or multiple-inorganic comparisons
    are reported as unresolved. P-14.8.2 assigns no PIN to mixed adducts.
    """
    from .api import Chem, mol_to_iupac_result
    if Chem is None:
        return AdductNamingResult(name="", generation_status="unavailable",
                                 preferred_status="not-assessed", route="explicit-adduct")
    try:
        supplied = tuple(components)
    except (TypeError, ValueError):
        return _invalid("Components must be an iterable of AdductComponent objects.")
    if len(supplied) < 2 or any(not isinstance(c, AdductComponent) for c in supplied):
        return _invalid("An adduct requires at least two explicit components.")
    species = {}
    for component in supplied:
        if (not isinstance(component.smiles, str) or not component.smiles.strip()
                or isinstance(component.amount, bool)
                or not isinstance(component.amount, (int, str, Fraction))):
            return _invalid("Each component needs a structure and a positive exact rational amount.")
        try:
            amount = Fraction(component.amount)
            mol = Chem.MolFromSmiles(component.smiles)
        except (ValueError, TypeError, ZeroDivisionError, OverflowError):
            return _invalid("A component structure or exact rational amount is invalid.")
        if amount <= 0 or mol is None or not mol.GetNumAtoms():
            return _invalid("A component structure or positive amount is invalid.")
        # Plain canonical SMILES omits relative/enhanced stereo semantics.
        # Validate every supplied component before its identity can be merged.
        if unsupported_relative_stereochemistry(mol, Chem):
            result = unsupported_relative_stereochemistry_result()
            return AdductNamingResult(
                name="", generation_status=result.generation_status,
                preferred_status=result.preferred_status,
                unresolved=tuple(SelectionLimitation(
                    "adduct-component/" + issue.stage, issue.rule, issue.reason)
                    for issue in result.unresolved),
                route="explicit-adduct")
        fragments=Chem.GetMolFrags(mol,asMols=True)
        if len(fragments)>1:
            charges=[sum(a.GetFormalCharge() for a in part.GetAtoms()) for part in fragments]
            if not all(charges) or sum(charges):
                return _invalid("A molecular component must be connected or one balanced ionic salt.")
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        identity = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        if identity in species:
            species[identity]['amount'] += amount
        else:
            species[identity] = dict(mol=mol, amount=amount)
    if len(species) < 2:
        return _invalid("The composition has fewer than two distinct molecular species.")
    denominator = lcm(*(row['amount'].denominator for row in species.values()))
    counts = {identity:int(row['amount']*denominator) for identity,row in species.items()}
    divisor = gcd(*counts.values())
    issues = []
    material = []
    for identity,row in species.items():
        mol = row['mol']
        result = mol_to_iupac_result(mol)
        if not result.name:
            return AdductNamingResult(
                name="", generation_status="unavailable", preferred_status="not-assessed",
                unresolved=(SelectionLimitation("adduct-component", "P-14.8", "A component has no supported molecular name."),),
                route="explicit-adduct")
        for issue in result.unresolved:
            inherited = SelectionLimitation("adduct-component/"+issue.stage, issue.rule, issue.reason)
            if inherited not in issues:issues.append(inherited)
        organic = any(a.GetAtomicNum()==6 for a in mol.GetAtoms())
        from .adduct_priority import component_priority,ComponentPriority
        priority=component_priority(mol,Chem) if organic else ComponentPriority()
        rank=priority.class_rank
        component = NamedAdductComponent(identity,result.name,counts[identity]//divisor,
                                        organic,rank,result)
        material.append((component,priority,identity=='O'))

    def chemical_order(left,right):
        a,af,aw = left
        b,bf,bw = right
        if a.organic != b.organic:return -1 if a.organic else 1
        if not a.organic:
            if aw != bw:return 1 if aw else -1
            return None
        if a.class_rank is None or b.class_rank is None:return None
        if a.class_rank != b.class_rank:return -1 if a.class_rank < b.class_rank else 1
        if af.functional_facts is not None and bf.functional_facts is not None:
            from .group_priority import compare_functional_group_priority
            comparison=compare_functional_group_priority(af.functional_facts,bf.functional_facts)
            if comparison.status=='senior':return -1
            if comparison.status=='junior':return 1
            if comparison.status!='tie':return None
        elif a.class_rank<21:return None
        if af.parent_facts is not None and bf.parent_facts is not None:
            from .parent_selection import compare_parent_seniority
            comparison=compare_parent_seniority(af.parent_facts,bf.parent_facts)
            if comparison.preference in {-1,1}:return comparison.preference
        return None

    # Inspect every pair before sorting. A sorting algorithm need not compare
    # all pairs, so it cannot by itself establish complete ordering evidence.
    comparisons = {}
    for i,left in enumerate(material):
        for right in material[i+1:]:
            order=chemical_order(left,right)
            a,b=left[0].canonical_smiles,right[0].canonical_smiles
            comparisons[a,b]=order
            comparisons[b,a]=None if order is None else -order
            if order is None:
                issue=SelectionLimitation("adduct-order", "P-14.8/P-41/P-44" if left[0].organic
                    else "P-14.8.2/IR-5.5", "The component pair lacks a completed source-order comparison.")
                if issue not in issues:issues.append(issue)
    # Topological sorting respects every proved comparison even when other
    # pairs remain unknown. A fallback pairwise total comparator could form
    # cycles by mixing an arbitrary identity tie-break with source order.
    remaining={row[0].canonical_smiles:row for row in material}
    incoming={identity:set() for identity in remaining}
    for (a,b),order in comparisons.items():
        if order==-1:incoming[b].add(a)
    ordered=[]
    while remaining:
        ready=sorted(identity for identity in remaining if not incoming[identity]&remaining.keys())
        if not ready:
            issue=SelectionLimitation("adduct-order","P-14.8", "The source component comparisons form a cycle.")
            return AdductNamingResult(name='', generation_status='unavailable',
                preferred_status='unresolved', unresolved=tuple(issues+[issue]), route='explicit-adduct')
        identity=ready[0]
        ordered.append(remaining.pop(identity))
    named=tuple(row[0] for row in ordered)
    name='—'.join(c.name for c in named)+' ('+'/'.join(str(c.proportion) for c in named)+')'
    mixed=any(not c.organic for c in named)
    preferred=('unresolved' if issues else 'resolved' if not mixed and
               all(c.result.preferred_status=='resolved' for c in named) else 'not-assessed')
    return AdductNamingResult(name=name, generation_status='available',
        preferred_status=preferred, unresolved=tuple(issues), route='explicit-adduct',
        components=named)


def adduct_to_iupac(components: Iterable[AdductComponent]) -> str:
    """String counterpart of :func:`adduct_to_iupac_result`."""
    return adduct_to_iupac_result(components).name
