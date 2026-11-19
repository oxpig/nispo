from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ..molecule_iteration import iter_bonds
from ..molecule_cache import mol_cache_dict


@dataclass(frozen=True, slots=True)
class BranchGraphFacts:
    root_symbol: str
    parent_bond_order: str
    atom_count: int
    atom_count_truncated: bool
    ring_atom_count: int
    hetero_counts: tuple[tuple[str, int], ...]
    charged_atom_count: int
    unsaturated_bond_count: int
    branch_point_count: int

    @property
    def hetero_key(self) -> str:
        return ",".join(f"{symbol}{count}" for symbol, count in self.hetero_counts) or "none"


def branch_graph_facts(mol, idx: int, parent_idx: int) -> BranchGraphFacts:
    cache = mol_cache_dict(mol, "branch_graph_fact_cache")
    key = (idx, parent_idx)
    cached = cache.get(key)
    if cached is not None:
        return cached
    facts = _branch_graph_facts_uncached(mol, idx, parent_idx)
    cache[key] = facts
    return facts


def branch_miss_shape_key(mol, idx: int, parent_idx: int) -> str:
    cache = mol_cache_dict(mol, "branch_render_miss_key_cache")
    key = (idx, parent_idx)
    cached = cache.get(key)
    if cached is not None:
        return cached
    facts = branch_graph_facts(mol, idx, parent_idx)
    value = (
        "simple_external_prefix:"
        f"root={facts.root_symbol};"
        f"bond={facts.parent_bond_order};"
        f"atoms={_count_bucket(facts.atom_count, facts.atom_count_truncated)};"
        f"ring={_count_bucket(facts.ring_atom_count, False)};"
        f"hetero={facts.hetero_key};"
        f"charge={_count_bucket(facts.charged_atom_count, False)};"
        f"unsat={_count_bucket(facts.unsaturated_bond_count, False)};"
        f"branch={_count_bucket(facts.branch_point_count, False)}"
    )
    cache[key] = value
    return value


def _branch_graph_facts_uncached(mol, idx: int, parent_idx: int) -> BranchGraphFacts:
    atom = mol.GetAtomWithIdx(idx)
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    atoms, truncated = _bounded_branch_atoms(mol, idx, parent_idx)
    symbols = Counter(mol.GetAtomWithIdx(atom_idx).GetSymbol() for atom_idx in atoms)
    hetero_counts = tuple(
        (symbol, count)
        for symbol, count in sorted(symbols.items())
        if symbol not in {"C", "H"}
    )
    charged = sum(1 for atom_idx in atoms if mol.GetAtomWithIdx(atom_idx).GetFormalCharge() != 0)
    ring_atoms = sum(1 for atom_idx in atoms if mol.GetAtomWithIdx(atom_idx).IsInRing())
    unsaturated = sum(
        1
        for bond in iter_bonds(mol)
        if bond.GetBeginAtomIdx() in atoms
        and bond.GetEndAtomIdx() in atoms
        and bond.GetBondTypeAsDouble() > 1.0
    )
    branch_points = sum(
        1
        for atom_idx in atoms
        if sum(1 for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors() if nbr.GetAtomicNum() > 1) > 2
    )
    return BranchGraphFacts(
        root_symbol=atom.GetSymbol(),
        parent_bond_order=_bond_order_bucket(parent_bond.GetBondTypeAsDouble() if parent_bond is not None else 0.0),
        atom_count=len(atoms),
        atom_count_truncated=truncated,
        ring_atom_count=ring_atoms,
        hetero_counts=hetero_counts,
        charged_atom_count=charged,
        unsaturated_bond_count=unsaturated,
        branch_point_count=branch_points,
    )


def _bounded_branch_atoms(mol, idx: int, parent_idx: int, *, limit: int = 32) -> tuple[frozenset[int], bool]:
    seen = {parent_idx}
    stack = [idx]
    atoms: set[int] = set()
    truncated = False
    while stack:
        atom_idx = stack.pop()
        if atom_idx in seen:
            continue
        seen.add(atom_idx)
        atoms.add(atom_idx)
        if len(atoms) >= limit:
            truncated = True
            break
        for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx not in seen and nbr.GetAtomicNum() > 1:
                stack.append(nbr_idx)
    return frozenset(atoms), truncated


def _bond_order_bucket(order: float) -> str:
    if order == 1.0:
        return "1"
    if order == 2.0:
        return "2"
    if order == 3.0:
        return "3"
    if order == 1.5:
        return "arom"
    return "other"


def _count_bucket(count: int, truncated: bool) -> str:
    if truncated:
        return "33+"
    if count <= 2:
        return str(count)
    if count <= 4:
        return "3-4"
    if count <= 8:
        return "5-8"
    if count <= 16:
        return "9-16"
    return "17-32"
