"""Substitution by two equivalent rooted arms on a small heterohydride."""

from functools import lru_cache

from .branch_rules import collect_subtree
from .component_mapping import mapped_general_branch
from .formatting import format_prefixes
from .naming_methods import MethodDecision
from .naming_result import SelectionLimitation


@lru_cache(maxsize=2)
def _connectors(Chem):
    return tuple((Chem.MolFromSmarts(query), word) for query, word in (
        ("[C,c]-[S;!R;+0;D2]-[S;!R;+0;D2]-[C,c]", "disulfane"),
        ("[C,c]-[N;!R;+0;D2]=[N;!R;+0;D2]-[C,c]", "diazene"),
        ("[C,c]-[N;!R;+0;H1;D2]-[C,c]", "amine"),
    ))


def equivalent_hydride_arms(mol, Chem, *, parents):
    """Require two closed branches and an original-graph arm exchange."""
    if mol.GetNumHeavyAtoms() > 80 or mol.GetStereoGroups():
        return None
    for query, parent in _connectors(Chem):
        if parent not in parents:
            continue
        matches = mol.GetSubstructMatches(query, maxMatches=2)
        if len(matches) != 1:
            continue
        match = matches[0]
        root1, root2 = match[0], match[-1]
        core = frozenset(match[1:-1])
        left, right = match[1], match[-2]
        if any(mol.GetAtomWithIdx(i).GetIsotope() or mol.GetAtomWithIdx(i).GetChiralTag()
               for i in core):
            continue
        if parent == "diazene" and mol.GetBondBetweenAtoms(left, right).GetStereo():
            continue
        arm1 = frozenset(collect_subtree(mol, root1, left))
        arm2 = frozenset(collect_subtree(mol, root2, right))
        heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
        if (len(arm1) != len(arm2) or arm1 & arm2 or (arm1 | arm2) & core
                or arm1 | arm2 | core != heavy):
            continue
        # Check the whole isomeric graph with labelled attachment sites.
        # This retains geometric and tetrahedral annotations on both arms.
        graph = Chem.Mol(mol)
        for atom in graph.GetAtoms():
            atom.SetAtomMapNum(0)
        graph.GetAtomWithIdx(root1).SetAtomMapNum(1)
        first = Chem.MolToSmiles(graph, isomericSmiles=True)
        graph.GetAtomWithIdx(root1).SetAtomMapNum(0)
        graph.GetAtomWithIdx(root2).SetAtomMapNum(1)
        if first != Chem.MolToSmiles(graph, isomericSmiles=True):
            continue
        from .extended_inventory import selection_inventory

        inventory = selection_inventory(mol)
        centers = frozenset(group.center for group in inventory.principal)
        if not inventory.supported or (centers and (
                parent != "amine" or centers != core
                or any(group.kind != "amine" for group in inventory.principal))):
            continue
        text1, owned1 = mapped_general_branch(mol, Chem, root1, left)
        text2, owned2 = mapped_general_branch(mol, Chem, root2, right)
        if not text1 or text1 != text2 or owned1 != arm1 or owned2 != arm2:
            continue
        if parent == "diazene":
            prefix = format_prefixes([("1", text1), ("2", text1)], hierarchical_enclosures=True)
        else:
            prefix = format_prefixes([("", text1), ("", text1)], omit_locants=True,
                                     hierarchical_enclosures=True)
        rule = "P-16.3/P-29/P-51"
        return MethodDecision(prefix + parent, "substitutive", centers, heavy, rule,
                              component_roles=(("parent", core, ()), ("arm", arm1, ()), ("arm", arm2, ())),
                              preference_issues=(SelectionLimitation(
                                  "parent-and-method", rule,
                                  "The two substituents exchange on the original isomeric graph."),))
    return None
