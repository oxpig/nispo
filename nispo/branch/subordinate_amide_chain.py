"""Bounded original-graph P-66.1.1.4.1.1 amide carbon-chain prefix."""
from __future__ import annotations

from types import SimpleNamespace

from ..constants import ROOTS


_MAX_CHAIN_LENGTH = max(ROOTS)


def subordinate_amide_chain_prefix(mol, Chem, root, blocked):
    """Return a fully owned ring-linked linear alkyl branch ending at an amide.

    The attachment is at a ring atom or at a sulfur linking to a ring.
    The existing mapped writer owns the nitrogen branch and its substituents.
    """
    from .functional_groups import collect_subtree
    from ..component_mapping import (
        _format_prefixes_with_functionalized_compounds, has_annotations,
        mapped_branch,
    )
    from ..extended_inventory import selection_inventory
    from ..group_priority import compare_functional_group_priority

    parent = mol.GetAtomWithIdx(blocked)
    carbon = mol.GetAtomWithIdx(root)
    cut = mol.GetBondBetweenAtoms(root, blocked)
    if (parent.GetFormalCharge() or parent.GetNumRadicalElectrons()
            or parent.GetIsotope() or int(parent.GetChiralTag())
            or cut is None or cut.GetBondTypeAsDouble() != 1):
        return None
    if parent.GetAtomicNum() == 16 and not parent.IsInRing() and parent.GetTotalValence() == 2:
        links = [b for b in parent.GetBonds()
                 if b.GetOtherAtomIdx(blocked) != root and b.GetOtherAtom(parent).GetAtomicNum() > 1]
        if (len(links) != 1 or not links[0].GetOtherAtom(parent).IsInRing()
                or links[0].GetBondTypeAsDouble() != 1):
            return None
    elif parent.GetAtomicNum() not in {6, 7} or not parent.IsInRing():
        return None
    chain = []
    previous = blocked
    while (carbon.GetAtomicNum() == 6 and not carbon.IsInRing()
           and not carbon.GetFormalCharge() and carbon.GetTotalNumHs(includeNeighbors=True) == 2
           and len(chain) < _MAX_CHAIN_LENGTH):
        current = carbon.GetIdx()
        children = [n for n in carbon.GetNeighbors()
                    if n.GetIdx() != previous and n.GetAtomicNum() > 1]
        if (len(children) != 1 or children[0].GetAtomicNum() != 6
                or mol.GetBondBetweenAtoms(current, children[0].GetIdx()).GetBondTypeAsDouble() != 1):
            return None
        chain.append(current)
        previous, carbon = current, children[0]
    if not chain or len(chain) + 1 not in ROOTS:
        return None
    carbonyl = carbon
    cidx = carbonyl.GetIdx()
    if carbonyl.IsInRing() or carbonyl.GetFormalCharge():
        return None
    others = [n for n in carbonyl.GetNeighbors()
              if n.GetIdx() != previous and n.GetAtomicNum() > 1]
    oxygens = [n for n in others if n.GetAtomicNum() == 8
               and mol.GetBondBetweenAtoms(cidx, n.GetIdx()).GetBondTypeAsDouble() == 2]
    nitrogens = [n for n in others if n.GetAtomicNum() == 7
                 and mol.GetBondBetweenAtoms(cidx, n.GetIdx()).GetBondTypeAsDouble() == 1]
    if len(others) != 2 or len(oxygens) != 1 or len(nitrogens) != 1:
        return None
    oxo, nitrogen = oxygens[0], nitrogens[0]
    nidx, oidx = nitrogen.GetIdx(), oxo.GetIdx()
    if (oxo.GetFormalCharge() or oxo.GetDegree() != 1
            or nitrogen.GetFormalCharge() or nitrogen.GetTotalValence() != 3):
        return None
    scope = frozenset(collect_subtree(mol, root, blocked))
    nscope = frozenset(collect_subtree(mol, nidx, cidx))
    carbon_side = set(chain) | {cidx, oidx}
    if (scope != nscope | carbon_side or carbon_side & nscope
            or has_annotations(mol, scope)
            or any(mol.GetAtomWithIdx(i).GetFormalCharge()
                   or mol.GetAtomWithIdx(i).GetNumRadicalElectrons()
                   or any(n.GetAtomicNum() == 1 for n in mol.GetAtomWithIdx(i).GetNeighbors())
                   for i in scope)):
        return None

    # Heterogroups can be single heterounits, so counting heteroatoms may
    # overestimate. That is safe for this bounded admission: four *possible*
    # heterounits make us decline rather than displace a replacement name.
    acyclic = {i for i in scope | {blocked} if not mol.GetAtomWithIdx(i).IsInRing()}
    for start in acyclic:
        stack = [(start, -1, 0)]
        while stack:
            current, previous, count = stack.pop()
            atom = mol.GetAtomWithIdx(current)
            count += int(atom.GetAtomicNum() != 6)
            if count >= 4:
                return None
            for neighbor in atom.GetNeighbors():
                ni = neighbor.GetIdx()
                if ni in acyclic and ni != previous:
                    stack.append((ni, current, count))

    inventory = selection_inventory(mol)
    if (not inventory.supported or len(inventory.groups) != len(inventory.source_group_facts)
            or not inventory.principal):
        return None
    targets = [(g, fact) for g, fact in zip(inventory.groups, inventory.source_group_facts)
               if g.kind == ("pseudoketone" if nitrogen.IsInRing() else "amide") and g.center == cidx
               and {cidx, oidx, nidx} <= set(g.atoms)]
    if len(targets) != 1:
        return None
    _, amide_fact = targets[0]
    fact_by_id = {id(g): fact for g, fact in zip(inventory.groups, inventory.source_group_facts)}
    if any(g.kind not in {"acid", "ester", "carbamate"} or set(g.atoms) & scope
           or compare_functional_group_priority(fact_by_id.get(id(g)), amide_fact).status != "senior"
           for g in inventory.principal):
        return None

    amino, owned = mapped_branch(mol, Chem, nidx, cidx)
    if not amino or owned != nscope:
        return None
    locant = str(len(chain) + 1)
    rows = [(locant, amino), (locant, "oxo")]
    # A substituted amino unit is compound even when it contains no
    # punctuation. Its graph, rather than a spelling heuristic, proves this.
    typed = (SimpleNamespace(text=amino, prefix_nature="simple" if amino == "amino" else "compound"),)
    prefix = _format_prefixes_with_functionalized_compounds(
        rows, typed, hierarchical_enclosures=True)
    if not prefix:
        return None
    return prefix + ROOTS[len(chain) + 1] + "yl", scope
