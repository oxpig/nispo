"""Graph-backed simple/compound facts for bounded mapped carbon organyls.

Callers supply the original mapped component atom set and its independently
rendered component name. This helper classifies P-16 nature from topology; it
does not generate or rank names.
"""
from __future__ import annotations


_HALOGENS = frozenset({9, 17, 35, 53})


def organyl_prefix_nature(mol, root: int, atoms, rendered_name: str) -> str | None:
    """Return ``simple`` / ``compound`` for a proved ordinary carbon organyl.

    Acyclic unbranched carbon parents are simple even when attachment is at an
    internal carbon (for example, a locanted propan-2-yl unit). The retained
    tert-butyl prefix and unsubstituted phenyl are also source-classified as
    simple. Other connected carbon trees and halogen-substituted organyls are
    compound. Unsubstituted monocyclic groups and fully aromatic fused carbon
    hydrides are simple; substituted rings and aryl-ring assemblies are
    compound. Unsupported atoms and incomplete graph claims decline.
    """
    component = frozenset(atoms)
    if not component or root not in component:
        return None
    atoms_by_id = {idx: mol.GetAtomWithIdx(idx) for idx in component}
    if (any(atom.GetAtomicNum()!=6 for atom in atoms_by_id.values())
            and all(atom.IsInRing() and atom.GetIsAromatic() for atom in atoms_by_id.values())):
        boundary=[(i,n.GetIdx()) for i,atom in atoms_by_id.items() for n in atom.GetNeighbors()
                  if n.GetAtomicNum()>1 and n.GetIdx() not in component]
        if len(boundary)==1 and boundary[0][0]==root:
            from .rules45_97_chains_locants import simple_aryl_branch
            if simple_aryl_branch(mol,root,boundary[0][1],component):
                return 'simple'
    if any(atom.GetAtomicNum() not in {6, *_HALOGENS} for atom in atoms_by_id.values()):
        return None
    carbons = {idx for idx, atom in atoms_by_id.items() if atom.GetAtomicNum() == 6}
    halogens = component - carbons
    if not carbons or root not in carbons:
        return None

    # A plain phenyl unit is one unsubstituted aromatic carbon ring.
    if (rendered_name == "phenyl" and len(component) == 6 and len(carbons) == 6
            and all(atoms_by_id[idx].GetIsAromatic() and atoms_by_id[idx].IsInRing()
                    for idx in carbons)):
        edges = [bond for bond in mol.GetBonds()
                 if bond.GetBeginAtomIdx() in carbons and bond.GetEndAtomIdx() in carbons]
        if (len(edges) == 6 and all(bond.GetIsAromatic() for bond in edges)
                and all(sum(1 for bond in atoms_by_id[idx].GetBonds()
                            if bond.GetOtherAtom(atoms_by_id[idx]).GetIdx() in carbons) == 2
                        for idx in carbons)):
            return "simple"

    adjacency = {
        idx: {neighbor.GetIdx() for neighbor in atoms_by_id[idx].GetNeighbors()
              if neighbor.GetIdx() in carbons}
        for idx in carbons
    }
    reached, todo = set(), [root]
    while todo:
        current = todo.pop()
        if current in reached:
            continue
        reached.add(current)
        todo.extend(adjacency[current] - reached)
    if reached != carbons:
        return None
    carbon_edges = sum(len(neighbors) for neighbors in adjacency.values()) // 2
    if carbon_edges < len(carbons) - 1:
        return None

    for idx in halogens:
        atom = atoms_by_id[idx]
        bonds = tuple(atom.GetBonds())
        if (len(bonds) != 1 or bonds[0].GetBondTypeAsDouble() != 1
                or bonds[0].GetOtherAtom(atom).GetIdx() not in carbons):
            return None

    if carbon_edges >= len(carbons):
        if (not halogens and carbon_edges == len(carbons)
                and all(len(neighbors) == 2 for neighbors in adjacency.values())):
            return "simple"
        # P-16.3.3(b)/P-16.3.4(a): an unsubstituted fused hydrocarbon
        # hydride is one simple substituent, even though its free-valence
        # locant requires enclosure. Every owned atom must be an unmodified
        # aromatic ring carbon and every internal bond aromatic; a side chain,
        # attached phenyl ring, or annotated source is not this proof.
        if (not halogens and len(adjacency[root]) == 2
                and all(atom.GetIsAromatic() and atom.IsInRing()
                        and not atom.GetFormalCharge() and not atom.GetIsotope()
                        and not atom.GetNumRadicalElectrons() and not int(atom.GetChiralTag())
                        for atom in atoms_by_id.values())
                and all(bond.GetIsAromatic() for bond in mol.GetBonds()
                        if bond.GetBeginAtomIdx() in component and bond.GetEndAtomIdx() in component)):
            boundary = [(idx, neighbor.GetIdx()) for idx, atom in atoms_by_id.items()
                        for neighbor in atom.GetNeighbors()
                        if neighbor.GetIdx() not in component and neighbor.GetAtomicNum() > 1]
            if len(boundary) == 1 and boundary[0][0] == root:
                return "simple"
        return "compound"

    if not halogens and all(len(neighbors) <= 2 for neighbors in adjacency.values()):
        return "simple"

    # P-16 explicitly treats the source-retained tert-butyl prefix as simple.
    if (not halogens and rendered_name == "tert-butyl" and len(carbons) == 4
            and len(adjacency[root]) == 3
            and all(len(adjacency[idx]) == 1 for idx in carbons - {root})):
        return "simple"
    return "compound"


def multiplied_organyl_prefix(mol, root: int, atoms, name: str, count: int) -> str | None:
    """Multiply a mapped carbon-rooted unit without collapsing its modifiers.

    In addition to the ordinary carbon/halogen proof above, the functional
    class callers can supply one complete monovalent carbon component with
    neutral acyclic N/O/S modifications. Unsubstituted heterocyclic hydrides
    and charged or isotopically modified functional components keep their
    own grammar. No component boundaries are inferred from spelling.
    """
    from .component_vocabulary import SUBSTITUENT_VOCABULARY
    from .substituent_multiplication import multiply_substituent

    component = frozenset(atoms)
    if root not in component or mol.GetAtomWithIdx(root).GetAtomicNum() != 6:
        return None
    nature = organyl_prefix_nature(mol, root, component, name)
    if nature is None:
        heavy = frozenset(i for i in component if mol.GetAtomWithIdx(i).GetAtomicNum() > 1)
        hetero = [mol.GetAtomWithIdx(i) for i in heavy
                  if mol.GetAtomWithIdx(i).GetAtomicNum() not in {6, *_HALOGENS}]
        if (not hetero or any(a.GetAtomicNum() not in {7, 8, 16} or a.IsInRing()
                             for a in hetero)
                or any(mol.GetAtomWithIdx(i).GetFormalCharge()
                       or mol.GetAtomWithIdx(i).GetIsotope()
                       or mol.GetAtomWithIdx(i).GetNumRadicalElectrons() for i in component)):
            return None
        boundary = [(i, b) for i in heavy for b in mol.GetAtomWithIdx(i).GetBonds()
                    if b.GetOtherAtomIdx(i) not in component
                    and b.GetOtherAtom(mol.GetAtomWithIdx(i)).GetAtomicNum() > 1]
        if len(boundary) != 1 or boundary[0][0] != root or boundary[0][1].GetBondTypeAsDouble() != 1:
            return None
        reached, todo = set(), [root]
        while todo:
            i = todo.pop()
            if i in reached:
                continue
            reached.add(i)
            todo.extend(n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors()
                        if n.GetIdx() in heavy and n.GetIdx() not in reached)
        if reached != set(heavy):
            return None
        nature = 'compound'
    retained = SUBSTITUENT_VOCABULARY.get(name)
    if nature == 'compound' and retained is not None and retained.source_status == 'preferred-retained':
        return None
    operation = multiply_substituent(name, count, nature=nature)
    return operation.text if operation is not None else None
