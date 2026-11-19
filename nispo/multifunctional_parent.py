"""Complete characteristic-group coverage on bounded acyclic carbon parents.

P-44 maximizes suffix groups before parent length and numbering. The route
renders repeated acids, ordinary esters, primary amides and nitriles only when
all principal sites can be explicitly represented. Unsupported connections,
N-substituted polyamides and differing ester owners retain existing renderers.
"""
from itertools import combinations

from .acyclic.rules import (
    AcyclicNameRecord, _acyclic_graph_view, _shortest_path_in_allowed_graph,
    _path_substituent_prefixes, _path_stereo_locants,
)
from .branch_rules import collect_subtree, parent_chain_name
from .constants import MULT
from .formatting import complete_parent_name, format_prefixes, join_prefix_parent
from .functional_inventory import functional_inventory


def multifunctional_parent_record(mol):
    empty = AcyclicNameRecord("")
    inventory = functional_inventory(mol)
    groups = inventory.principal
    if not groups or len(inventory.groups) < 2:
        return empty
    kind = groups[0].kind
    if kind not in {"acid", "ester", "amide", "nitrile"} or any(g.kind != kind for g in groups):
        return empty
    if len(groups) == 1 and kind != "nitrile":
        return empty  # Preserve the established mono-functional renderers.
    if any(mol.GetAtomWithIdx(g.center).IsInRing() for g in groups):
        return empty
    if kind == "nitrile" and any(g.owner < 0 or mol.GetAtomWithIdx(g.owner).IsInRing() for g in groups):
        # A directly ring-attached CN needs the ring-carbonitrile parent and
        # its hydrogen-state map, not a one-carbon acyclic nitrile parent.
        return empty
    if kind == "amide" and any(
        mol.GetAtomWithIdx(g.ligand).GetTotalNumHs(includeNeighbors=True) != 2
        or mol.GetAtomWithIdx(g.ligand).GetDegree() != 1 for g in groups
    ):
        return empty
    # Off-parent stereo must not be flattened by a new functional-class name.
    # Existing stereo-aware specialised routes remain authoritative here.
    if any(int(a.GetChiralTag()) for a in mol.GetAtoms()) or any(int(b.GetStereo()) for b in mol.GetBonds()):
        return empty
    owners, skipped = _functional_atoms_and_owners(mol, groups, kind)
    if owners is None:
        return empty
    centers = {g.center for g in groups}
    carbon_atoms = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6 and not a.IsInRing()}
    if len(carbon_atoms) > 40:
        return empty
    records = []
    for external in (False, True):
        if external and len(groups) <= 2:
            continue  # Terminal -dioic/-diamide grammar accommodates both.
        allowed = carbon_atoms - skipped - (centers if external else set())
        required = [g.owner if external else g.center for g in groups]
        if not set(required) <= allowed:
            continue
        for path in _paths(mol, allowed):
            if not set(required) <= set(path):
                continue
            if not external and any(path.index(i) not in {0, len(path)-1} for i in required):
                continue
            for oriented in (path, tuple(reversed(path))):
                locants = tuple(sorted(oriented.index(i)+1 for i in required))
                prefixes = _path_substituent_prefixes(mol, oriented, skip_atoms=skipped | (centers if external else set()), skip_plain_hydrogen=True)
                parent = complete_parent_name(parent_chain_name(mol, oriented))
                if prefixes is None or not parent:
                    continue
                if any(any(char.isspace() for char in prefix) for _, prefix in prefixes):
                    # A functional-class parent containing spaces cannot be
                    # inserted as an ordinary substituent without conversion.
                    continue
                ending = _suffix_parent(parent, kind, len(groups), locants, external, len(path))
                if not ending:
                    continue
                name = join_prefix_parent(format_prefixes(prefixes), ending)
                if kind == "ester":
                    owner = owners[0]
                    multiplier = MULT.get(len(groups), "") if len(groups) > 1 else ""
                    if not multiplier and len(groups) > 1:
                        continue
                    # Only simple, identical owners are admitted here. Compound
                    # and unlike ester owners need their own distributive grammar.
                    name = multiplier + owner + " " + name
                unsaturated = tuple(i for i, (a,b) in enumerate(zip(oriented, oriented[1:]),1)
                                    if mol.GetBondBetweenAtoms(a,b).GetBondTypeAsDouble() != 1)
                prefix_locs = tuple(sorted(int(loc) for loc,_ in prefixes))
                key = (-len(groups), -len(path), locants, unsaturated, prefix_locs, name)
                records.append((key, name, oriented))
    if not records:
        return empty
    _, name, path = min(records)
    return AcyclicNameRecord(name, _path_stereo_locants(path))


def _functional_atoms_and_owners(mol, groups, kind):
    skipped, owners = set(), []
    all_centers = {g.center for g in groups}
    for group in groups:
        skipped.update(i for i in group.atoms if i != group.center)
        if kind == "ester":
            from .parent_rules import _stable_ester_owner_name
            name = _stable_ester_owner_name(mol, group.ligand, group.center)
            if not name or not name.isalpha() or name not in {"methyl", "ethyl", "propyl", "butyl", "pentyl", "hexyl"}:
                return None, set()
            atoms = set(collect_subtree(mol, group.ligand, group.center))
            if atoms & all_centers:
                return None, set()
            owners.append(name)
            skipped.update(atoms)
    if owners and len(set(owners)) != 1:
        return None, set()
    return owners, skipped


def _paths(mol, allowed):
    view = _acyclic_graph_view(mol)
    ends = sorted(i for i in allowed if sum(n in allowed for n in view.neighbors[i]) <= 1)
    result = {(i,) for i in allowed}
    for a,b in combinations(ends,2):
        path = _shortest_path_in_allowed_graph(view,a,b,allowed)
        if path:
            result.add(path)
    return sorted(result)


def _suffix_parent(parent, kind, count, locants, external, size):
    if external:
        suffix = {"acid":"carboxylic acid", "ester":"carboxylate", "amide":"carboxamide", "nitrile":"carbonitrile"}[kind]
        multi = MULT.get(count)
        if not multi:
            return ""
        positions = "" if size == 1 else "-" + ",".join(map(str,locants)) + "-"
        return parent + positions + multi + suffix
    if count == 1 and kind == "nitrile":
        # P-66.5.1.2.1 retains these saturated parents as PINs, including
        # substitution. Isotopic decoration changes neither the ordinary
        # parent vocabulary nor its atom numbering (P-82.5.1).
        return {"methane": "formonitrile", "ethane": "acetonitrile"}.get(
            parent, parent + "nitrile")
    if count != 2:
        return ""
    # Oxalic acid and oxamide are retained PIN functional parents (P-65.1,
    # P-66.1), including oxalate ester derivatives.
    if parent == "ethane":
        retained = {"acid": "oxalic acid", "ester": "oxalate", "amide": "oxamide",
                    "nitrile": "oxalonitrile"}
        if kind in retained:
            return retained[kind]
    return parent + {"acid":"dioic acid", "ester":"dioate", "amide":"diamide", "nitrile":"dinitrile"}[kind]
