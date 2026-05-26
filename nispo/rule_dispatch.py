from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass


Gate = Callable[[object], bool]
CandidateProducer = Callable[[object], Iterable[object]]


@dataclass(frozen=True, slots=True)
class RuleSpec:
    """Explicit rule registration record used by the naming engine."""

    family: str
    priority: int
    gates: tuple[Gate, ...]
    producer: CandidateProducer

    def applies(self, ctx) -> bool:
        for gate in self.gates:
            if not gate(ctx):
                return False
        return True

    def candidates(self, ctx) -> Iterable[object]:
        return self.producer(ctx)


def acyclic(ctx) -> bool:
    return ctx.mol.GetRingInfo().NumRings() == 0


def min_ring_count(count: int) -> Gate:
    return lambda ctx: ctx.mol.GetRingInfo().NumRings() >= count


def min_ring_system_ring_count(count: int) -> Gate:
    return lambda ctx: ctx.features["max_ring_system_ring_count"] >= count


def exact_ring_system(ctx) -> bool:
    return bool(ctx.features["all_atoms_in_one_ring_system"])


def nonring_positive_sulfur(ctx) -> bool:
    return ctx.profile.has_nonring_positive_sulfur


def nonring_positive_nitrogen(ctx) -> bool:
    return ctx.profile.has_nonring_positive_nitrogen


def nonring_positive_phosphorus(ctx) -> bool:
    return ctx.profile.has_nonring_positive_phosphorus


def nonring_neutral_phosphorus(ctx) -> bool:
    return ctx.profile.has_nonring_neutral_phosphorus


def charged_ring_atom(ctx) -> bool:
    return ctx.profile.has_charged_ring_atom


def no_charged_ring_atom(ctx) -> bool:
    return not ctx.profile.has_charged_ring_atom


def imino_skeleton_signal(ctx) -> bool:
    return ctx.profile.has_imino_skeleton_signal


def functional_skeleton_signal(ctx) -> bool:
    return ctx.profile.has_functional_skeleton_signal


def acyclic_preference_signal(ctx) -> bool:
    return ctx.profile.may_prefer_acyclic_skeleton


def kekulized_polycycle_signal(ctx) -> bool:
    return ctx.profile.has_kekulized_polycycle_signal


def generic_bicyclic_topology(ctx) -> bool:
    return ctx.has_generic_bicyclic_topology


def generic_polycyclic_topology(ctx) -> bool:
    return ctx.has_generic_polycyclic_topology


def spiro_topology(ctx) -> bool:
    return ctx.has_spiro_topology


def no_exocyclic_charged_fused_parent(ctx) -> bool:
    return not _has_charged_fused_thioxo_imino_system(ctx)


def _has_charged_fused_thioxo_imino_system(ctx) -> bool:
    mol = ctx.mol
    Chem = ctx.Chem
    if mol.GetRingInfo().NumRings() < 2 or not ctx.profile.has_charged_ring_atom:
        return False
    has_thioxo = False
    has_terminal_imino = False
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "C" or not atom.IsInRing():
            continue
        cidx = atom.GetIdx()
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nbr.IsInRing():
                continue
            bond = mol.GetBondBetweenAtoms(cidx, nidx)
            if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
                continue
            if nbr.GetSymbol() == "S":
                has_thioxo = True
            elif nbr.GetSymbol() == "C":
                terminal_n = [
                    nnbr.GetIdx()
                    for nnbr in nbr.GetNeighbors()
                    if nnbr.GetIdx() != cidx
                    and nnbr.GetSymbol() == "N"
                    and mol.GetBondBetweenAtoms(nidx, nnbr.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
                ]
                if len(terminal_n) == 1:
                    has_terminal_imino = True
    return has_thioxo and has_terminal_imino
