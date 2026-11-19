"""Principal oxygen suffixes and parent selection for neutral carbon skeletons.

P-41 and P-44 put characteristic-group coverage before chain length and
suffix locants before substituent locants. Restrict this route to graph classes
whose complete groups can be rendered by the existing branch renderer.
"""


from collections import Counter
from itertools import combinations
import re

from .molecule_iteration import iter_atoms, iter_bonds
from .acyclic.rules import (
    AcyclicNameRecord,
    _acyclic_graph_view,
    _complete_path_stereo_prefix,
    _path_stereo_locants,
    _path_substituent_prefixes,
    _shortest_path_in_allowed_graph,
)
from .branch_rules import parent_chain_name
from .constants import MULT, ROOTS
from .parent_selection import graph_parent_facts, parent_frontier
from .formatting import (
    complete_parent_name,
    format_prefixes,
    join_prefix_parent,
    multiplied_suffix,
    prefix_alphabetization_key,
)


def _single_site_prefixes(prefixes):
    """Omit one-site locants without joining distinct substituent groups."""
    counts = Counter(prefix for _, prefix in prefixes)
    pieces = []
    for index, name in enumerate(sorted(counts, key=prefix_alphabetization_key)):
        count = counts[name]
        piece = format_prefixes([("", name)] * count, omit_locants=True)
        if index and re.fullmatch(r"[A-Za-z]+", name):
            multiplier = MULT.get(count, "") if count > 1 else ""
            piece = f"{multiplier}({name})"
        pieces.append(piece)
    return "".join(pieces)


def principal_oxygen_record(mol) -> AcyclicNameRecord:
    from .functional_inventory import functional_inventory
    inventory = functional_inventory(mol)
    if inventory.supported and inventory.principal and inventory.principal[0].kind not in {
        "acid", "aldehyde", "ketone", "pseudoketone", "lactam", "lactone", "alcohol"
    }:
        return AcyclicNameRecord("")
    atoms = tuple(iter_atoms(mol))
    ordinary_amines = (inventory.supported and not mol.GetRingInfo().NumRings()
        and all(g.kind in {'acid','aldehyde','ketone','alcohol','amine'} for g in inventory.groups)
        and all(not a.IsInRing() for a in atoms if a.GetAtomicNum()==7))
    ring_nitrogen = any(a.GetAtomicNum() == 7 for a in atoms) and all(a.IsInRing() and not int(a.GetChiralTag()) for a in atoms if a.GetAtomicNum() == 7)
    if not any(a.GetAtomicNum() == 8 for a in atoms):
        return AcyclicNameRecord("")
    if any(
        a.GetAtomicNum() not in ({1, 6, 7, 8, 9, 17, 35, 53} if ring_nitrogen or ordinary_amines else {1, 6, 8, 9, 17, 35, 53})
        or a.GetFormalCharge()
        or a.GetIsotope()
        or a.GetNumRadicalElectrons()
        for a in atoms
    ):
        return AcyclicNameRecord("")
    acids, ketones, aldehydes, alcohols = [], [], [], []
    for carbon in (a for a in atoms if a.GetAtomicNum() == 6):
        double_o, single_o = [], []
        for neighbor in carbon.GetNeighbors():
            if neighbor.GetAtomicNum() == 8:
                order = mol.GetBondBetweenAtoms(
                    carbon.GetIdx(), neighbor.GetIdx()
                ).GetBondTypeAsDouble()
                (double_o if order == 2 else single_o).append(neighbor)
        if double_o:
            if len(double_o) != 1:
                return AcyclicNameRecord("")
            if single_o:
                if (
                    len(single_o) != 1
                    or single_o[0].GetTotalNumHs(includeNeighbors=True) != 1
                ):
                    return AcyclicNameRecord("")  # Esters/anhydrides are senior.
                acids.append(carbon.GetIdx())
            elif carbon.GetTotalNumHs(includeNeighbors=True):
                aldehydes.append(carbon.GetIdx())
            elif (sum(a.GetAtomicNum() == 6 for a in carbon.GetNeighbors()) == 2
                  or ring_nitrogen and sum(a.GetAtomicNum() == 6 for a in carbon.GetNeighbors()) == 1
                  and any(a.GetAtomicNum() == 7 and a.IsInRing() for a in carbon.GetNeighbors())):
                ketones.append(carbon.GetIdx())
            else:
                return AcyclicNameRecord("")
        elif any(o.GetTotalNumHs(includeNeighbors=True) == 1 for o in single_o):
            alcohols.extend(
                carbon.GetIdx()
                for o in single_o
                if o.GetTotalNumHs(includeNeighbors=True) == 1
            )
    if ring_nitrogen and not (acids or aldehydes or ketones):
        return AcyclicNameRecord("")  # Preserve specialized ring-alcohol routes.
    suffix_atoms = acids or aldehydes or ketones or alcohols
    suffix = "oic acid" if acids else "al" if aldehydes else "one" if ketones else "ol"
    if not suffix_atoms or any(mol.GetAtomWithIdx(i).IsInRing() for i in suffix_atoms):
        return AcyclicNameRecord("")
    if any(
        a.GetAtomicNum() == 8
        and a.GetTotalNumHs(includeNeighbors=True) == 1
        and (a.GetDegree() != 1 or a.GetNeighbors()[0].GetAtomicNum() != 6)
        for a in atoms
    ):
        return AcyclicNameRecord("")
    allowed = {a.GetIdx() for a in atoms if a.GetAtomicNum() == 6 and not a.IsInRing()}
    if len(allowed) > 40:
        return AcyclicNameRecord("")
    view = _acyclic_graph_view(mol)
    endpoints = sorted(
        i for i in allowed if sum(j in allowed for j in view.neighbors[i]) <= 1
    )
    paths = {(i,) for i in allowed}
    for a, b in combinations(endpoints, 2):
        path = _shortest_path_in_allowed_graph(view, a, b, allowed)
        if path:
            paths.add(path)
    # Coverage precedes length; number the suffix before unsaturation/prefixes.
    ranked = []
    # Annotation ownership is invariant across the enumerated parent paths.
    # Observe native stereo once, then test only the annotated atom sets.
    from rdkit import Chem
    native_stereo = type(mol) is Chem.Mol
    chiral_atoms = (frozenset(a.GetIdx() for a in atoms if int(a.GetChiralTag()))
                    if native_stereo and paths and not ring_nitrogen else frozenset())
    stereo_bond_atoms = (tuple(frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
                              for b in iter_bonds(mol) if int(b.GetStereo()))
                         if native_stereo and paths else ())
    for path in paths:
        # A single carboxy carbon attached to a ring belongs to a ring
        # carboxylic-acid parent (including retained benzoic acid), P-65.1.
        if (acids or aldehydes) and len(path) == 1 and mol.GetRingInfo().NumRings():
            continue
        # Off-parent stereochemistry needs the specialized ring/branch route.
        if native_stereo:
            path_atoms = frozenset(path)
            if chiral_atoms - path_atoms or any(pair - path_atoms for pair in stereo_bond_atoms):
                continue
        else:
            if not ring_nitrogen and any(int(a.GetChiralTag()) and a.GetIdx() not in path for a in atoms):
                continue
            if any(
                int(b.GetStereo())
                and not {b.GetBeginAtomIdx(), b.GetEndAtomIdx()} <= set(path)
                for b in iter_bonds(mol)
            ):
                continue
        for oriented in (path, tuple(reversed(path))):
            locs = tuple(
                sorted(oriented.index(i) + 1 for i in suffix_atoms if i in oriented)
            )
            if not locs or (
                (acids or aldehydes)
                and (
                    len(locs) != len(suffix_atoms)
                    or any(locant not in {1, len(path)} for locant in locs)
                )
            ):
                continue
            unsat = tuple(
                i
                for i, (a, b) in enumerate(zip(oriented, oriented[1:]), 1)
                if view.bond_orders.get(tuple(sorted((a, b))), 1) != 1
            )
            ranked.append(((-len(locs), -len(path), locs, unsat), oriented, locs))
    ranked.sort(key=lambda r: r[0])
    principal_rank = (7, 0) if acids else (15, 0) if aldehydes else (16, 0) if ketones else (17, 0)
    structural = [(row, graph_parent_facts(
        mol, row[1], _path_stereo_locants(row[1]), principal_class=principal_rank,
        suffix_centers=[i for i in suffix_atoms if i in row[1]], suffix_locants=row[2],
    )) for row in ranked]
    proved = bool(structural) and all(facts is not None for _, facts in structural)
    if proved:
        ranked = [row for row, _ in parent_frontier(structural, lambda item: item[1])]
    best_key, records = None, []
    for key, path, locs in ranked:
        if not proved and best_key is not None and key != best_key:
            break
        prefixes = _path_substituent_prefixes(mol, path, skip_plain_hydrogen=True)
        parent = complete_parent_name(parent_chain_name(mol, path))
        if prefixes is None or not parent:
            continue
        if ring_nitrogen:
            from rdkit import Chem
            from .stereochemistry import assigned_atom_stereochemistry
            assigned = assigned_atom_stereochemistry(mol, Chem, atom_order="idx")
            outside = Counter(d for i, d in assigned if i not in path)
            rendered = Counter(re.findall(r"\d+[a-z]?([RSrs])(?=[,)])", ";".join(prefix for _, prefix in prefixes)))
            if outside != rendered:
                continue
        removed = {
            (str(locant), "hydroxy" if suffix == "ol" else "oxo") for locant in locs
        }
        if acids:
            removed.update((str(locant), "hydroxy") for locant in locs)
        remaining = [item for item in prefixes if item not in removed]
        if acids:
            if len(locs) == 1:
                core = parent.removesuffix("e") + "oic acid"
                if len(path) == 1 and not remaining:
                    core = "formic acid"
                elif len(path) == 2 and parent == "ethane":
                    core = "acetic acid"  # P-12 retained functional parent.
            elif len(locs) == 2:
                core = (
                    "oxalic acid"
                    if len(path) == 2 and not remaining
                    else parent + "dioic acid"
                )
            else:
                continue
        elif suffix == "al":
            core = (
                parent.removesuffix("e") + "al" if len(locs) == 1 else parent + "dial"
            )
            if len(locs) == 1:
                core = {"methane": "formaldehyde", "ethane": "acetaldehyde"}.get(parent, core)
        elif len(locs) == 1:
            core = parent.removesuffix("e") + f"-{locs[0]}-{suffix}"
            if suffix == "ol" and (len(path) == 1 or len(path) == 2 and not remaining):
                core = parent.removesuffix("e") + "ol"
        else:
            multiplier = MULT.get(len(locs))
            if not multiplier:
                continue
            core = (
                parent
                + f"-{','.join(map(str, locs))}-{multiplied_suffix(len(locs), suffix)}"
            )
        if len(path) == 1 and suffix in {"one", "ol"} and len(locs) == 1:
            core = parent.removesuffix("e") + suffix
            prefix_text = _single_site_prefixes(remaining)
        elif core == "acetaldehyde":
            prefix_text = _single_site_prefixes(remaining)
        else:
            prefix_text = format_prefixes(remaining)
        name = join_prefix_parent(prefix_text, core)
        stereo = _complete_path_stereo_prefix(mol, path)
        if ring_nitrogen and assigned:
            stereo = ",".join(f"{path.index(i)+1}{d}" for i, d in assigned if i in path)
        if stereo:
            name = f"({stereo})-{name}"
        prefix_locs = tuple(sorted(int(locant) for locant, _ in remaining))
        facts = graph_parent_facts(
            mol, path, _path_stereo_locants(path), principal_class=principal_rank,
            suffix_centers=[i for i in suffix_atoms if i in path], suffix_locants=locs,
            prefixes=remaining, name=name,
        )
        records.append((prefix_locs, name, path, facts))
        best_key = key
    if not records:
        return AcyclicNameRecord("")
    records = parent_frontier(records, lambda record: record[3])
    _, name, path, facts = min(records, key=lambda record: record[:3])
    return AcyclicNameRecord(name, _path_stereo_locants(path), facts)


def monocyclic_polyol_record(mol, Chem):
    """P-41/44/45: all principal OH groups on one saturated carbon ring.

    Return the chosen name, actual parent atoms, locants and suffix centers.
    Other annotations/classes remain on their existing mapped routes.
    """
    from .functional_inventory import functional_inventory
    from .component_mapping import MappedComponent,decorate_component,mapped_branch
    from .parent_annotations import isotope_numbering_key,stereo_numbering
    inventory=functional_inventory(mol)
    groups=inventory.principal
    if not groups or any(g.kind!='alcohol' for g in groups):return None
    if any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in iter_atoms(mol)):return None
    if any(int(b.GetStereo()) for b in iter_bonds(mol)):return None
    rings=mol.GetRingInfo().AtomRings()
    if len(rings)!=1:return None
    ring=tuple(rings[0]);centers={g.center for g in groups}
    if not centers<=set(ring) or len(ring) not in ROOTS:return None
    if any(mol.GetAtomWithIdx(i).GetAtomicNum()!=6 for i in ring):return None
    if any(mol.GetBondBetweenAtoms(a,b).GetBondTypeAsDouble()!=1 for a,b in zip(ring,ring[1:]+ring[:1])):return None
    annotated=any(a.GetIsotope() or int(a.GetChiralTag()) for a in iter_atoms(mol))
    nonstereogenic=None
    if annotated:
        from .method_ineligibility import ordinary_multiplication_ineligibility
        proof=ordinary_multiplication_ineligibility(mol,Chem,groups)
        if (proof is None or not proof.reason.startswith('a-single-carbon-cycle-with-only-terminal-OH')
                or any(a.GetIsotope() and (a.GetAtomicNum()!=6 or a.GetIdx() not in ring) for a in iter_atoms(mol))
                or any(a.GetChiralTag() not in {Chem.ChiralType.CHI_UNSPECIFIED,Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                    Chem.ChiralType.CHI_TETRAHEDRAL_CCW} for a in iter_atoms(mol))):return None
        from .cyclic_polyol_stereo import cyclic_polyol_nonstereogenic_evidence
        nonstereogenic=cyclic_polyol_nonstereogenic_evidence(mol,Chem)
    suffix_atoms={i for g in groups for i in g.atoms if i!=g.center}
    results=[]
    for direction in (ring,tuple(reversed(ring))):
        for start in range(len(ring)):
            path=direction[start:]+direction[:start]
            mapping={i:str(j) for j,i in enumerate(path,1)}
            embedded=set(suffix_atoms);prefixes=[]
            for idx in path:
                for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
                    root=neighbor.GetIdx()
                    if root in mapping or root in suffix_atoms or neighbor.GetAtomicNum()<=1:continue
                    text,owned=mapped_branch(mol,Chem,root,idx)
                    if not text or owned&(set(mapping)|embedded):break
                    embedded.update(owned);prefixes.append((mapping[idx],text))
                else:continue
                break
            else:
                if set(mapping)|embedded!={a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum()>1}:continue
                locants=tuple(sorted(int(mapping[g.center]) for g in groups))
                parent='cyclo'+ROOTS[len(path)]+'ane'
                suffix=multiplied_suffix(len(groups),'ol')
                stem=parent[:-1] if suffix[0] in 'aeiouy' else parent
                name=stem+'-'+','.join(map(str,locants))+'-'+suffix
                isotope=isotope_numbering_key(mol,mapping,include_substituent_groups=True) if annotated else ()
                stereo=stereo_numbering(mol,mapping) if annotated else ()
                if isotope is None or stereo is None:continue
                name=decorate_component(mol,Chem,MappedComponent(name,tuple(mapping.items()),
                    frozenset(mapping),tuple(prefixes),frozenset(embedded),systematic_parent=True,
                    nonstereogenic_atom_evidence=nonstereogenic))
                if name:
                    key=(locants,tuple(sorted(int(loc) for loc,_ in prefixes)),
                         tuple((prefix_alphabetization_key(text),int(loc)) for loc,text in prefixes),isotope,stereo,name)
                    results.append((key,name,path,tuple(mapping.items()),tuple(sorted(centers))))
    if not results:return None
    _,name,parent,mapping,centers=min(results)
    return name,parent,mapping,centers
