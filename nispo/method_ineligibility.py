"""Graph upper bounds that exclude P-51 multiplication before enumeration."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MultiplicationIneligibility:
    rule: str
    reason: str
    maximum_senior_copies: int
    principal_identity: tuple = ()
    cyclic_parent_atoms: frozenset[int] = frozenset()


def cyclic_alcohol_tree_bound(mol,Chem,groups):
    """One senior-alcohol cycle with pendant trees cannot multiply.

    Removing a connected central linker leaves at most one cycle-containing
    arc with senior alcohol centres; detached pendant pieces have none.
    Source inventory identity and class selection are independent gates.
    """
    if (mol is None or len(Chem.GetMolFrags(mol))!=1 or mol.GetRingInfo().NumRings()!=1
            or any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in mol.GetAtoms())):return None
    from .extended_inventory import selection_inventory
    inventory=selection_inventory(mol)
    def identity(entries):
        return tuple(sorted(((g.kind,g.center,tuple(sorted(g.atoms)),g.owner,g.ligand) for g in entries),key=repr))
    actual=identity(inventory.principal)
    if (not inventory.supported or not groups or any(g.kind!='alcohol' for g in groups)
            or identity(groups)!=actual):return None
    frontier=getattr(inventory,'source_frontier',None)
    if frontier is None or frontier.status not in {'selected','tie'} or frontier.unknown_pairs or frontier.cycle:return None
    ring=tuple(mol.GetRingInfo().AtomRings()[0]);core=frozenset(ring)
    if (any(mol.GetAtomWithIdx(i).GetAtomicNum()!=6 for i in core)
            or any(mol.GetBondBetweenAtoms(a,b).GetBondType()!=Chem.BondType.SINGLE for a,b in zip(ring,ring[1:]+ring[:1]))
            or not {g.center for g in groups}<=core):return None
    remaining={a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum()>1 and a.GetIdx() not in core}
    while remaining:
        component={remaining.pop()};stack=list(component)
        while stack:
            current=stack.pop();onward={n.GetIdx() for n in mol.GetAtomWithIdx(current).GetNeighbors()}&remaining
            remaining-=onward;component|=onward;stack.extend(onward)
        links={n.GetIdx() for i in component for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in core}
        if len(links)!=1:return None
    return MultiplicationIneligibility('P-15.3.1/P-51.3.1/P-51.3.2.3/P-63.1.2',
        'one-senior-alcohol-carbon-cycle-with-pendant-trees-has-at-most-one-independent-senior-unit-around-a-connected-linker',
        1,actual,core)


def ordinary_multiplication_ineligibility(mol,Chem,groups):
    """Prove a closed ordinary scope without interpreting enumerator failure.

    The argument groups must equal the complete source-selected principal
    inventory. Unknown class comparisons and any unowned characteristic
    group keep the generic multiplicative obligation open.
    """
    if (mol is None or len(Chem.GetMolFrags(mol))!=1
            or any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in mol.GetAtoms())):
        return None
    from .extended_inventory import selection_inventory
    inventory=selection_inventory(mol)
    if not inventory.supported:
        return None
    def identity(entries):
        return tuple(sorted(((g.kind,g.center,tuple(sorted(g.atoms)),g.owner,g.ligand)
                            for g in entries),key=repr))
    actual=identity(inventory.principal)
    if actual!=identity(groups):
        return None
    centres=frozenset(g.center for g in groups)
    if groups and len(centres)==1:
        frontier=getattr(inventory,'source_frontier',None)
        if (frontier is None or frontier.status not in {'selected','tie'}
                or frontier.unknown_pairs or frontier.cycle):
            return None
        return MultiplicationIneligibility('P-51.3.1(3)/P-51.3.2.3/P-44.1.1',
            'one-distinct-principal-centre-cannot-belong-to-two-disjoint-identical-senior-parent-units',
            1,actual)
    if (groups and all(g.kind=='alcohol' for g in inventory.groups)
            and identity(inventory.groups)==actual and mol.GetRingInfo().NumRings()==1
            and all(b.GetBondType()==Chem.BondType.SINGLE for b in mol.GetBonds())):
        core=frozenset(mol.GetRingInfo().AtomRings()[0])
        carbons=frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum()==6)
        if (core==carbons and centres<=core
                and all(sum(n.GetIdx() in core for n in mol.GetAtomWithIdx(i).GetNeighbors())==2 for i in core)
                and all(a.GetAtomicNum()==1 or a.GetIdx() in core
                    or a.GetAtomicNum()==8 and a.GetTotalNumHs(includeNeighbors=True)==1
                    and sum(n.GetAtomicNum()>1 for n in a.GetNeighbors())==1
                    and any(n.GetIdx() in core for n in a.GetNeighbors()) for a in mol.GetAtoms())):
            # A connected central linking group removes one connected arc
            # from this carbon cycle; the complement has at most one
            # carbon-containing component. Every senior alcohol parent
            # contains its carbon/OH centre, so at most one independent
            # senior unit can remain attached through that linking group.
            return MultiplicationIneligibility('P-15.3.1/P-51.3.1/P-51.3.2.3/P-63.1.2',
                'a-single-carbon-cycle-with-only-terminal-OH-has-at-most-one-independent-senior-unit-around-a-connected-linker',
                1,actual,core)
    cycle_bound=cyclic_alcohol_tree_bound(mol,Chem,groups)
    if cycle_bound is not None:return cycle_bound
    if groups or inventory.groups:
        return None
    # P-51.3.1 excludes alkanes as repeated parent structures. In this
    # acyclic saturated scope every eligible nonalkane parent needs the
    # sole heteroatom, so two disjoint eligible copies are impossible.
    # P-63.2.4.1/.5 prescribe substitutive names for these ethers/sulfides.
    hetero=tuple(a for a in mol.GetAtoms() if a.GetAtomicNum() not in {1,6})
    if (len(hetero)==1 and hetero[0].GetAtomicNum() in {8,16}
            and hetero[0].GetDegree()==2
            and all(a.GetAtomicNum()==6 for a in hetero[0].GetNeighbors())
            and mol.GetRingInfo().NumRings()==0
            and all(b.GetBondType()==Chem.BondType.SINGLE for b in mol.GetBonds())):
        return MultiplicationIneligibility('P-51.3.1/P-63.2.4.1/P-63.2.5',
            'one-heteroatom-in-an-acyclic-saturated-ether-allows-at-most-one-eligible-nonalkane-parent',
            1)
    from .mapped_ring_parents import ring_systems
    systems=ring_systems(mol)
    if len(systems)!=1:
        return None
    core=systems[0]
    # Only one cyclic parent occurrence remains possible. The pendants are
    # saturated carbon hydrides (alkanes), excluded by P-51.3.1; halo prefixes
    # cannot become a second organic parent. Unsaturated/heteroatom arms do
    # not inherit this proof.
    if any(a.GetIdx() not in core and a.GetAtomicNum() not in {1,6,9,17,35,53}
           for a in mol.GetAtoms()):
        return None
    if any(not {b.GetBeginAtomIdx(),b.GetEndAtomIdx()}<=core
           and b.GetBondType()!=Chem.BondType.SINGLE for b in mol.GetBonds()):
        return None
    return MultiplicationIneligibility('P-51.3.1/P-51.3.2.3',
        'one-ring-system-with-only-alkane-or-halo-pendants-has-at-most-one-eligible-cyclic-parent',
        1,(),core)
