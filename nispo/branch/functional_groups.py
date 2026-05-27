from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

try:
    from rdkit import Chem, RDLogger
except Exception:  # pragma: no cover - benchmark environment provides RDKit.
    Chem = None
else:
    RDLogger.DisableLog("rdApp.*")

from ..constants import HALO, MULT
from ..formatting import alkyl_to_alkoxy as _alkyl_to_alkoxy
from ..formatting import format_prefixes as _format_prefixes
from ..molecule_cache import mol_cache_dict
from .functional_suffixes import suffix_prefix_term, suffix_supported_term

_HALOSULFUR_OXO_PREFIXES = {
    2: {"F": "fluorosulfonyl", "Cl": "chlorosulfonyl", "Br": "bromosulfonyl", "I": "iodosulfonyl"},
    1: {"F": "fluorosulfinyl", "Cl": "chlorosulfinyl", "Br": "bromosulfinyl", "I": "iodosulfinyl"},
}
_SULFUR_OXO_SUBSTITUENT_SUFFIX = {2: "sulfonyl", 1: "sulfinyl", 0: "sulfanyl"}
_GROUP14_SUBSTITUENT_SUFFIX = {"Si": "silyl", "Ge": "germyl", "Sn": "stannyl", "Pb": "plumbyl"}


def _branch_rules():
    from .. import branch_rules

    return branch_rules


def _oxygen_prefix(mol, idx: int, parent_idx: int) -> str:
    return _branch_rules()._oxygen_prefix(mol, idx, parent_idx)


def _nitrogen_prefix(mol, idx: int, parent_idx: int) -> str:
    return _branch_rules()._nitrogen_prefix(mol, idx, parent_idx)


def _ring_substituent_prefix(mol, attachment_idx: int, parent_idx: int) -> str:
    return _branch_rules()._ring_substituent_prefix(mol, attachment_idx, parent_idx)


def branch_prefix(mol, idx: int, parent_idx: int) -> str:
    return _branch_rules().branch_prefix(mol, idx, parent_idx)


def can_be_in_parent_skeleton(mol, idx: int) -> bool:
    return _branch_rules().can_be_in_parent_skeleton(mol, idx)


def parent_chain_name(mol, path: Sequence[int], *, as_substituent: bool = False) -> str:
    return _branch_rules().parent_chain_name(mol, path, as_substituent=as_substituent)


def replacement_prefixes(mol, path: Sequence[int]) -> list[tuple[str, str]]:
    return _branch_rules().replacement_prefixes(mol, path)

def _sulfur_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if atom.GetFormalCharge() < 0:
        heavy = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() != parent_idx]
        if not heavy and parent_bond is not None and parent_bond.GetBondType() == Chem.BondType.SINGLE:
            return "sulfido"
        return ""
    if atom.GetFormalCharge() > 0:
        heavy = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() != parent_idx]
        if not heavy and parent_bond is not None and parent_bond.GetBondType() == Chem.BondType.SINGLE:
            return suffix_supported_term("ium", "sulfaniumyl")
        return ""
    if atom.GetFormalCharge() != 0:
        return ""
    if parent_bond is not None and parent_bond.GetBondType() == Chem.BondType.DOUBLE:
        heavy = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() != parent_idx]
        if not heavy:
            return suffix_prefix_term("thione", "thioxo")
    halogens = [
        n.GetSymbol()
        for n in atom.GetNeighbors()
        if n.GetIdx() != parent_idx and n.GetSymbol() in HALO
    ]
    if len(halogens) == 5 and all(symbol == "F" for symbol in halogens):
        return "pentafluorosulfanyl"
    non_parent = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() != parent_idx]
    if len(halogens) == 3 and all(symbol == "F" for symbol in halogens) and len(non_parent) == 3:
        return "trifluorosulfanyl"
    double_oxygens = {
        n.GetIdx()
        for n in atom.GetNeighbors()
        if n.GetSymbol() == "O"
        and mol.GetBondBetweenAtoms(idx, n.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
    }
    double_o = len(double_oxygens)
    if _is_sulfonate_sulfur(mol, idx):
        return "sulfonato"
    sulfonimidoyl = _sulfonimidoyl_prefix(mol, idx, parent_idx, double_o)
    if sulfonimidoyl:
        return sulfonimidoyl
    halosulfur = _halosulfur_prefix(double_o, halogens, len(non_parent))
    if halosulfur:
        return halosulfur
    heavy = [
        n.GetIdx()
        for n in atom.GetNeighbors()
        if n.GetIdx() != parent_idx and n.GetIdx() not in double_oxygens and not _is_terminal_carbonyl_oxygen(n)
    ]
    if double_o == 2 and not heavy:
        return suffix_prefix_term("sulfone", "sulfonyl")
    if double_o == 2 and len(heavy) == 1 and _is_nitrile_carbon(mol, heavy[0]):
        return "cyanosulfonyl"
    if double_o == 0:
        sulfimidoyl = _sulfimidoyl_prefix(mol, idx, parent_idx)
        if sulfimidoyl:
            return sulfimidoyl
    if not heavy:
        return "sulfanyl"
    if len(heavy) == 1:
        if mol.GetAtomWithIdx(heavy[0]).IsInRing():
            ring = _ring_substituent_prefix(mol, heavy[0], idx)
            if not ring:
                return ""
            suffix = _sulfur_oxo_substituent_suffix(double_o)
            return ring + suffix if suffix else ""
        if mol.GetAtomWithIdx(heavy[0]).GetSymbol() == "N" and double_o == 2:
            amino = _nitrogen_prefix(mol, heavy[0], idx)
            if not amino:
                return ""
            return amino + "sulfonyl"
        if mol.GetAtomWithIdx(heavy[0]).GetSymbol() == "O" and double_o == 2:
            oxy = _oxygen_prefix(mol, heavy[0], idx)
            if not oxy:
                return ""
            return oxy + "sulfonyl"
        if mol.GetAtomWithIdx(parent_idx).IsInRing() and mol.GetAtomWithIdx(heavy[0]).GetSymbol() == "S" and double_o == 0:
            sulfur = _sulfur_prefix(mol, heavy[0], idx)
            if not sulfur:
                return ""
            if sulfur.endswith("sulfanyl"):
                return sulfur[: -len("sulfanyl")] + "disulfanyl"
            return sulfur + "sulfanyl"
        alk = substituent_from_root(mol, heavy[0], idx)
        if not alk:
            return ""
        suffix = _sulfur_oxo_substituent_suffix(double_o)
        return alk + suffix if suffix else ""
    return ""


def _sulfimidoyl_prefix(mol, idx: int, parent_idx: int) -> str:
    imino = []
    single_groups = []
    for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if nbr.GetSymbol() == "N" and bond.GetBondType() == Chem.BondType.DOUBLE:
            imino.append(nidx)
        elif bond.GetBondType() == Chem.BondType.SINGLE:
            single_groups.append(nidx)
    if len(imino) != 1 or len(single_groups) != 1:
        return ""
    s_group = substituent_from_root(mol, single_groups[0], idx)
    if not s_group:
        return ""
    n_group = _imino_n_substituent_name(mol, imino[0], idx)
    if n_group is None:
        return ""
    if not n_group:
        return f"{s_group}sulfinimidoyl"
    return f"N-({n_group})-S-{s_group}sulfinimidoyl"


def _metal_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    symbol = atom.GetSymbol()
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    heavy = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1
    ]
    if symbol in {"Li", "Na", "K"} and not heavy:
        return {"Li": "lithio", "Na": "sodio", "K": "potassio"}[symbol]
    if symbol == "Mg" and len(heavy) == 1:
        ligand = mol.GetAtomWithIdx(heavy[0])
        if ligand.GetSymbol() in HALO:
            return HALO[ligand.GetSymbol()] + "magnesio"
    return ""


def _phosphorus_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "P" or atom.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    double_oxygen_count = 0
    hydroxy_count = 0
    oxido_count = 0
    other_heavy = []
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_oxygen_count += 1
        elif (
            nbr.GetSymbol() == "O"
            and bond.GetBondType() == Chem.BondType.SINGLE
            and len(nbr.GetNeighbors()) == 1
        ):
            if nbr.GetFormalCharge() == 0 and nbr.GetTotalNumHs() == 1:
                hydroxy_count += 1
            elif nbr.GetFormalCharge() < 0:
                oxido_count += 1
            else:
                return ""
        elif nbr.GetAtomicNum() > 1:
            if nbr.GetSymbol() not in {"C", "O", "F", "Cl", "Br", "I"}:
                return ""
            other_heavy.append(nidx)
    if double_oxygen_count == 0 and hydroxy_count == 0 and oxido_count == 0:
        name = _substituted_hydride_prefix(mol, other_heavy, idx, "phosphanyl")
        return "phosphino" if name == "phosphanyl" else name
    if double_oxygen_count != 1:
        return ""
    if not other_heavy:
        if hydroxy_count == 2 and oxido_count == 0:
            return "phosphono"
        if hydroxy_count == 0 and oxido_count == 2:
            return "phosphonato"
        return ""
    if hydroxy_count or oxido_count:
        return ""
    ligand_names = []
    for nidx in other_heavy:
        ligand = branch_prefix(mol, nidx, idx)
        if not ligand:
            return ""
        ligand_names.append(ligand)
    if len(ligand_names) < 2:
        return ""
    if mol.GetAtomWithIdx(parent_idx).GetSymbol() == "N":
        return ""
    if len(set(ligand_names)) > 1 and any(_is_simple_phosphoryl_alkyl_or_alkoxy(name) for name in ligand_names):
        return ""
    ligand_text = _phosphoryl_ligand_text(ligand_names)
    return ligand_text + "phosphoryl" if ligand_text else ""


def _group14_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    suffix = _GROUP14_SUBSTITUENT_SUFFIX.get(atom.GetSymbol())
    if not suffix or atom.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    heavy = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1
    ]
    return _substituted_hydride_prefix(mol, heavy, idx, suffix)


def _selenium_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "Se" or atom.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None:
        return ""
    if parent_bond.GetBondType() == Chem.BondType.DOUBLE:
        heavy = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetIdx() != parent_idx]
        return "selenoxo" if not heavy else ""
    if parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    heavy = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx and nbr.GetAtomicNum() > 1
    ]
    if not heavy:
        return "selanyl"
    ligand_text = _substituent_ligand_text(mol, heavy, idx)
    return ligand_text + "selanyl" if ligand_text else ""


def _substituted_hydride_prefix(mol, ligand_indices: Sequence[int], center_idx: int, suffix: str) -> str:
    ligand_text = _substituent_ligand_text(mol, ligand_indices, center_idx)
    if ligand_indices and not ligand_text:
        return ""
    return ligand_text + suffix


def _substituent_ligand_text(mol, ligand_indices: Sequence[int], center_idx: int) -> str:
    ligands = []
    for nidx in ligand_indices:
        ligand = branch_prefix(mol, nidx, center_idx)
        if not ligand:
            return ""
        ligands.append(ligand)
    return _ligand_text(ligands)


def _ligand_text(ligands: Sequence[str]) -> str:
    counts = Counter(ligands)
    parts = []
    for ligand in sorted(counts, key=_phosphoryl_ligand_sort_key):
        count = counts[ligand]
        unit = _phosphoryl_ligand_unit(ligand)
        if count == 1:
            parts.append(unit)
        elif _needs_complex_phosphoryl_ligand(ligand):
            parts.append(f"{MULT.get(count, '')}({unit})")
        else:
            parts.append(MULT.get(count, "") + ligand)
    return "".join(parts)


def _phosphoryl_ligand_text(ligands: Sequence[str]) -> str:
    return _ligand_text(ligands)


def _phosphoryl_ligand_unit(ligand: str) -> str:
    return f"({ligand})" if _needs_complex_phosphoryl_ligand(ligand) else ligand


def _needs_complex_phosphoryl_ligand(ligand: str) -> bool:
    return "-" in ligand or ligand[:1].isdigit() or ligand.startswith("(")


def _phosphoryl_ligand_sort_key(ligand: str) -> tuple[int, str]:
    simple_alkyl = {
        "methyl",
        "ethyl",
        "propyl",
        "butyl",
        "pentyl",
        "hexyl",
        "heptyl",
        "octyl",
        "nonyl",
        "decyl",
    }
    if ligand in simple_alkyl:
        return (0, ligand)
    if _needs_complex_phosphoryl_ligand(ligand):
        return (2, ligand)
    return (1, ligand)


def _is_simple_phosphoryl_alkyl_or_alkoxy(ligand: str) -> bool:
    simple = {
        "methyl",
        "ethyl",
        "propyl",
        "butyl",
        "pentyl",
        "hexyl",
        "heptyl",
        "octyl",
        "nonyl",
        "decyl",
        "methoxy",
        "ethoxy",
        "propoxy",
        "butoxy",
        "pentoxy",
        "hexoxy",
        "heptoxy",
        "octoxy",
        "nonoxy",
        "decoxy",
    }
    return ligand in simple


def _halosulfur_prefix(double_o: int, halogens: Sequence[str], non_parent_count: int) -> str:
    if len(halogens) != 1:
        return ""
    if double_o == 0 and non_parent_count == 1:
        return HALO[halogens[0]] + "sulfanyl"
    return _HALOSULFUR_OXO_PREFIXES.get(double_o, {}).get(halogens[0], "")


def _sulfur_oxo_substituent_suffix(double_o: int) -> str:
    if double_o == 2:
        return suffix_prefix_term("sulfone", "sulfonyl")
    if double_o == 1:
        return suffix_prefix_term("sulfoxide", "sulfinyl")
    return _SULFUR_OXO_SUBSTITUENT_SUFFIX.get(double_o, "sulfanyl")


def _sulfonimidoyl_prefix(mol, idx: int, parent_idx: int, double_o: int) -> str:
    if double_o != 1:
        return ""
    imino = []
    single_groups = []
    for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx or nbr.GetSymbol() == "O":
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if nbr.GetSymbol() == "N" and bond.GetBondType() == Chem.BondType.DOUBLE:
            imino.append(nidx)
        elif bond.GetBondType() == Chem.BondType.SINGLE:
            single_groups.append(nidx)
    if len(imino) != 1 or len(single_groups) != 1:
        return ""

    single_atom = mol.GetAtomWithIdx(single_groups[0])
    if single_atom.GetSymbol() in HALO:
        s_group = HALO[single_atom.GetSymbol()]
    else:
        s_group = substituent_from_root(mol, single_groups[0], idx)
    if not s_group:
        return ""
    n_group = _imino_n_substituent_name(mol, imino[0], idx)
    if n_group is None:
        return ""
    if not n_group:
        return f"{s_group}sulfonimidoyl"
    return f"N-({n_group})-S-{s_group}sulfonimidoyl"


def _imino_n_substituent_name(mol, nidx: int, sulfur_idx: int) -> str | None:
    heavy = [n.GetIdx() for n in mol.GetAtomWithIdx(nidx).GetNeighbors() if n.GetIdx() != sulfur_idx]
    if not heavy:
        return ""
    if len(heavy) != 1:
        return None
    attached = heavy[0]
    carbamoyl = _carbamoyl_substituent_name(mol, attached, nidx)
    if carbamoyl:
        return carbamoyl
    name = substituent_from_root(mol, attached, nidx)
    if not name:
        return None
    return name


def _carbamoyl_substituent_name(mol, cidx: int, nidx: int) -> str:
    atom = mol.GetAtomWithIdx(cidx)
    if atom.GetSymbol() != "C":
        return ""
    has_double_o = False
    amine = None
    for nbr in atom.GetNeighbors():
        nbr_idx = nbr.GetIdx()
        if nbr_idx == nidx:
            continue
        bond = mol.GetBondBetweenAtoms(cidx, nbr_idx)
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            has_double_o = True
        elif nbr.GetSymbol() == "N" and bond.GetBondType() == Chem.BondType.SINGLE:
            amine = nbr_idx
        else:
            return ""
    if not has_double_o or amine is None:
        return ""
    groups = []
    for nbr in mol.GetAtomWithIdx(amine).GetNeighbors():
        nbr_idx = nbr.GetIdx()
        if nbr_idx == cidx:
            continue
        name = substituent_from_root(mol, nbr_idx, amine)
        if not name:
            return ""
        groups.append(name)
    if not groups:
        return suffix_prefix_term("amide", "carbamoyl")
    counts = Counter(groups)
    parts = []
    for name in sorted(counts):
        count = counts[name]
        if count > 1:
            if "-" in name or name.startswith("("):
                parts.append(f"{MULT.get(count, '')}({name})")
            else:
                parts.append(MULT.get(count, "") + name)
        else:
            parts.append(name)
    return "".join(parts) + suffix_prefix_term("amide", "carbamoyl")


def substituent_from_root(mol, root_idx: int, blocked_idx: int) -> str:
    cache = mol_cache_dict(mol, "substituent_from_root_cache")
    key = (root_idx, blocked_idx)
    cached = cache.get(key)
    if cached is not None:
        return cached
    atoms = collect_subtree(mol, root_idx, blocked_idx)
    paths = rooted_paths(mol, root_idx, atoms)
    if not paths:
        cache[key] = ""
        return ""
    best = ""
    best_key = None
    for path in paths:
        name = _name_subtree_path(mol, path, atoms, blocked_idx)
        if not name:
            continue
        key = (-len(path), substituent_locants(mol, path, atoms, blocked_idx), name)
        if best_key is None or key < best_key:
            best_key = key
            best = name
    cache[(root_idx, blocked_idx)] = best
    return best


def _name_subtree_path(mol, path: Sequence[int], atoms: set[int], blocked_idx: int) -> str:
    path_set = set(path)
    prefixes = []
    for pos, idx in enumerate(path, start=1):
        atom = mol.GetAtomWithIdx(idx)
        if not can_be_in_parent_skeleton(mol, idx):
            return ""
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx == blocked_idx or nidx in path_set:
                continue
            if nbr.GetSymbol() == "H" and nbr.GetIsotope() == 0:
                continue
            if nidx not in atoms:
                continue
            prefix = branch_prefix(mol, nidx, idx)
            if not prefix:
                return ""
            prefixes.append((str(pos), prefix))
    parent = parent_chain_name(mol, path, as_substituent=True)
    if not parent:
        return ""
    repl = replacement_prefixes(mol, path)
    prefix_text = _format_prefixes(prefixes)
    pieces = []
    if prefix_text:
        pieces.append(prefix_text)
    if repl:
        pieces.append(repl + parent)
    else:
        pieces.append(parent)
    return "-".join(pieces)


def substituent_locants(mol, path: Sequence[int], atoms: set[int], blocked_idx: int) -> list[int]:
    path_set = set(path)
    locs = []
    for pos, idx in enumerate(path, start=1):
        if mol.GetAtomWithIdx(idx).GetSymbol() != "C":
            locs.append(pos)
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nbr.GetSymbol() == "H" and nbr.GetIsotope() == 0:
                continue
            if nidx != blocked_idx and nidx in atoms and nidx not in path_set:
                locs.append(pos)
    return locs


def rooted_paths(mol, root_idx: int, atoms: set[int]) -> list[tuple[int, ...]]:
    cache = mol_cache_dict(mol, "rooted_paths_cache")
    cache_key = (root_idx, tuple(sorted(atoms)))
    cached = cache.get(cache_key)
    if cached is not None:
        return list(cached)
    if mol.GetAtomWithIdx(root_idx).IsInRing():
        path_atoms = set(atoms)
    else:
        path_atoms = {idx for idx in atoms if not mol.GetAtomWithIdx(idx).IsInRing()}
        path_atoms.add(root_idx)
    leaves = [
        idx
        for idx in path_atoms
        if idx != root_idx
        and sum(1 for n in mol.GetAtomWithIdx(idx).GetNeighbors() if n.GetIdx() in path_atoms) <= 1
        and can_be_in_parent_skeleton(mol, idx)
    ]
    if not leaves:
        leaves = [root_idx]
    paths = []
    best_len = 0
    for leaf in leaves:
        if leaf == root_idx:
            path = (root_idx,)
        else:
            path = Chem.rdmolops.GetShortestPath(mol, root_idx, leaf)
        if any(idx not in path_atoms or not can_be_in_parent_skeleton(mol, idx) for idx in path):
            continue
        if len(path) > best_len:
            best_len = len(path)
            paths.clear()
        if len(path) == best_len:
            paths.append(tuple(path))
    cache[cache_key] = tuple(paths)
    return paths


def collect_subtree(mol, start_idx: int, blocked_idx: int) -> set[int]:
    cache = mol_cache_dict(mol, "collect_subtree_cache")
    key = (start_idx, blocked_idx)
    cached = cache.get(key)
    if cached is not None:
        return set(cached)
    seen = {blocked_idx}
    stack = [start_idx]
    atoms = set()
    while stack:
        idx = stack.pop()
        if idx in seen:
            continue
        seen.add(idx)
        atoms.add(idx)
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nbr.GetSymbol() == "H" and nbr.GetIsotope() == 0:
                continue
            if nidx not in seen:
                stack.append(nidx)
    cache[key] = frozenset(atoms)
    return atoms


def _is_terminal_carbonyl_oxygen(atom) -> bool:
    if atom.GetSymbol() != "O":
        return False
    if atom.GetDegree() != 1:
        return False
    bond = atom.GetBonds()[0]
    return bond.GetBondType() == Chem.BondType.DOUBLE


def is_carbonyl_carbon(mol, idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C":
        return False
    return any(
        nbr.GetSymbol() == "O"
        and mol.GetBondBetweenAtoms(idx, nbr.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
        for nbr in atom.GetNeighbors()
    )


def _is_terminal_ketene_alpha_carbon(mol, idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0:
        return False
    single_neighbors = 0
    has_ketene_carbonyl = False
    for nbr in atom.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(idx, nbr.GetIdx())
        if bond.GetBondType() == Chem.BondType.SINGLE:
            single_neighbors += 1
            continue
        if bond.GetBondType() != Chem.BondType.DOUBLE or nbr.GetSymbol() != "C":
            return False
        has_double_o = any(
            onbr.GetSymbol() == "O"
            and mol.GetBondBetweenAtoms(nbr.GetIdx(), onbr.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
            for onbr in nbr.GetNeighbors()
            if onbr.GetIdx() != idx
        )
        if has_double_o:
            has_ketene_carbonyl = True
        else:
            return False
    return has_ketene_carbonyl and single_neighbors == 1


def _is_carboxylate_carbon(mol, idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C":
        return False
    has_double_o = False
    has_single_ominus = False
    for nbr in atom.GetNeighbors():
        if nbr.GetSymbol() != "O":
            continue
        bond = mol.GetBondBetweenAtoms(idx, nbr.GetIdx())
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            has_double_o = True
        elif bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetFormalCharge() < 0:
            has_single_ominus = True
    return has_double_o and has_single_ominus


def _acyl_halide_prefix(mol, idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C":
        return ""
    has_double_o = any(
        nbr.GetSymbol() == "O"
        and mol.GetBondBetweenAtoms(idx, nbr.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
        for nbr in atom.GetNeighbors()
    )
    if not has_double_o:
        return ""
    halogens = [
        nbr.GetSymbol()
        for nbr in atom.GetNeighbors()
        if nbr.GetSymbol() in HALO
        and mol.GetBondBetweenAtoms(idx, nbr.GetIdx()).GetBondType() == Chem.BondType.SINGLE
    ]
    if len(halogens) != 1:
        return ""
    return {
        "F": "fluorocarbonyl",
        "Cl": "chlorocarbonyl",
        "Br": "bromocarbonyl",
        "I": "iodocarbonyl",
    }.get(halogens[0], "")


def _carbonyl_substituent_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    has_double_o = False
    heavy = []
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx:
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            has_double_o = True
            continue
        heavy.append(nidx)
    if not has_double_o or len(heavy) != 1:
        return ""
    distal = heavy[0]
    hydroxycarbamoyl = _hydroxycarbamoyl_prefix(mol, distal, idx)
    if hydroxycarbamoyl:
        return hydroxycarbamoyl
    alkoxycarbonyl = _alkoxycarbonyl_prefix(mol, distal, idx)
    if alkoxycarbonyl:
        return alkoxycarbonyl
    if mol.GetAtomWithIdx(distal).IsInRing():
        name = _ring_substituent_prefix(mol, distal, idx)
    else:
        name = substituent_from_root(mol, distal, idx)
    if not name:
        return ""
    return name + "carbonyl"


def _hydroxycarbamoyl_prefix(mol, nitrogen_idx: int, carbonyl_idx: int) -> str:
    nitrogen = mol.GetAtomWithIdx(nitrogen_idx)
    if nitrogen.GetSymbol() != "N" or nitrogen.GetFormalCharge() != 0:
        return ""
    bond = mol.GetBondBetweenAtoms(nitrogen_idx, carbonyl_idx)
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    hydroxy = []
    other = []
    for nbr in nitrogen.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == carbonyl_idx:
            continue
        n_bond = mol.GetBondBetweenAtoms(nitrogen_idx, nidx)
        if nbr.GetSymbol() == "O" and nbr.GetFormalCharge() == 0 and n_bond.GetBondType() == Chem.BondType.SINGLE:
            terminal = [nnbr for nnbr in nbr.GetNeighbors() if nnbr.GetIdx() != nitrogen_idx and nnbr.GetAtomicNum() > 1]
            if terminal:
                other.append(nidx)
            else:
                hydroxy.append(nidx)
        else:
            other.append(nidx)
    if len(hydroxy) == 1 and not other:
        return "(hydroxyamino)carbonyl"
    return ""


def _alkoxycarbonyl_prefix(mol, oxygen_idx: int, carbonyl_idx: int) -> str:
    oxygen = mol.GetAtomWithIdx(oxygen_idx)
    if oxygen.GetSymbol() != "O" or oxygen.GetFormalCharge() != 0:
        return ""
    bond = mol.GetBondBetweenAtoms(oxygen_idx, carbonyl_idx)
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    heavy = [nbr.GetIdx() for nbr in oxygen.GetNeighbors() if nbr.GetIdx() != carbonyl_idx]
    if not heavy:
        return "carboxy"
    if len(heavy) != 1:
        return ""
    attached = heavy[0]
    if mol.GetAtomWithIdx(attached).IsInRing():
        ring = _ring_substituent_prefix(mol, attached, oxygen_idx)
        if not ring:
            return ""
        oxy = "phenoxy" if ring == "phenyl" else f"({ring})oxy"
    else:
        alk = substituent_from_root(mol, attached, oxygen_idx)
        if not alk:
            return ""
        oxy = _alkyl_to_alkoxy(alk)
    return oxy + "carbonyl"


def _is_isocyano_carbon(mol, idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() >= 0:
        return False
    return any(
        nbr.GetSymbol() == "N"
        and nbr.GetFormalCharge() > 0
        and mol.GetBondBetweenAtoms(idx, nbr.GetIdx()).GetBondType() == Chem.BondType.TRIPLE
        for nbr in atom.GetNeighbors()
    )


def _is_nitrile_carbon(mol, idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0:
        return False
    return any(
        nbr.GetSymbol() == "N"
        and nbr.GetFormalCharge() == 0
        and mol.GetBondBetweenAtoms(idx, nbr.GetIdx()).GetBondType() == Chem.BondType.TRIPLE
        for nbr in atom.GetNeighbors()
    )


def _is_nitrile_nitrogen(mol, idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0:
        return False
    return any(
        nbr.GetSymbol() == "C"
        and nbr.GetFormalCharge() == 0
        and mol.GetBondBetweenAtoms(idx, nbr.GetIdx()).GetBondType() == Chem.BondType.TRIPLE
        for nbr in atom.GetNeighbors()
    )


def _is_isocyano_nitrogen(mol, idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() <= 0:
        return False
    return any(
        nbr.GetSymbol() == "C"
        and nbr.GetFormalCharge() < 0
        and mol.GetBondBetweenAtoms(idx, nbr.GetIdx()).GetBondType() == Chem.BondType.TRIPLE
        for nbr in atom.GetNeighbors()
    )


def _is_sulfonate_sulfur(mol, idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "S":
        return False
    double_o = 0
    single_ominus = 0
    for nbr in atom.GetNeighbors():
        if nbr.GetSymbol() != "O":
            continue
        bond = mol.GetBondBetweenAtoms(idx, nbr.GetIdx())
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            double_o += 1
        elif bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetFormalCharge() < 0:
            single_ominus += 1
    return double_o >= 2 and single_ominus >= 1


def _is_carbonyl_oxygen_idx(mol, idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    return _is_terminal_carbonyl_oxygen(atom)


def _is_nitro_nitrogen(mol, idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N":
        return False
    oxy = [n for n in atom.GetNeighbors() if n.GetSymbol() == "O"]
    return atom.GetFormalCharge() > 0 and len(oxy) >= 2


def _is_azido_attachment(mol, idx: int, parent_idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N":
        return False
    for mid in atom.GetNeighbors():
        if mid.GetIdx() == parent_idx or mid.GetSymbol() != "N" or mid.GetFormalCharge() <= 0:
            continue
        for terminal in mid.GetNeighbors():
            if terminal.GetIdx() != idx and terminal.GetSymbol() == "N" and terminal.GetFormalCharge() < 0:
                return True
    return False


def _is_diazo_attachment(mol, idx: int, parent_idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() <= 0:
        return False
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.DOUBLE:
        return False
    return any(
        nbr.GetIdx() != parent_idx
        and nbr.GetSymbol() == "N"
        and nbr.GetFormalCharge() < 0
        and mol.GetBondBetweenAtoms(idx, nbr.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
        for nbr in atom.GetNeighbors()
    )


def _is_diazo_nitrogen(mol, idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() <= 0:
        return False
    has_terminal_n = False
    has_carbon = False
    for nbr in atom.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(idx, nbr.GetIdx())
        if nbr.GetSymbol() == "N" and nbr.GetFormalCharge() < 0 and bond.GetBondType() == Chem.BondType.DOUBLE:
            has_terminal_n = True
        elif nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.DOUBLE:
            has_carbon = True
    return has_terminal_n and has_carbon


def _is_diazonio_attachment(mol, idx: int, parent_idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() <= 0:
        return False
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return False
    return any(
        nbr.GetIdx() != parent_idx
        and nbr.GetSymbol() == "N"
        and mol.GetBondBetweenAtoms(idx, nbr.GetIdx()).GetBondType() == Chem.BondType.TRIPLE
        for nbr in atom.GetNeighbors()
    )


def _is_nitroso_attachment(mol, idx: int, parent_idx: int) -> bool:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N":
        return False
    return any(
        nbr.GetIdx() != parent_idx
        and nbr.GetSymbol() == "O"
        and mol.GetBondBetweenAtoms(idx, nbr.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
        for nbr in atom.GetNeighbors()
    )
