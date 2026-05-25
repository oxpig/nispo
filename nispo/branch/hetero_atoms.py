from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

try:
    from rdkit import Chem, RDLogger
except Exception:  # pragma: no cover - benchmark environment provides RDKit.
    Chem = None
else:
    RDLogger.DisableLog("rdApp.*")

from ..constants import HALO, MULT, ROOTS
from ..formatting import alkyl_to_alkoxy as _alkyl_to_alkoxy
from .functional_groups import (
    _carbonyl_substituent_prefix,
    _is_azido_attachment,
    _is_diazo_attachment,
    _is_diazonio_attachment,
    _is_isocyano_nitrogen,
    _is_nitrile_carbon,
    _is_nitro_nitrogen,
    _is_nitroso_attachment,
    _sulfur_prefix,
    substituent_from_root,
)
from .ring_substituents import _ring_substituent_prefix


def _branch_rules():
    from .. import branch_rules

    return branch_rules


def _alkylidene_prefix(mol, carbon_idx: int, parent_idx: int) -> str:
    return _branch_rules()._alkylidene_prefix(mol, carbon_idx, parent_idx)


def _alkynyl_substituent_prefix(mol, idx: int, parent_idx: int) -> str:
    return _branch_rules()._alkynyl_substituent_prefix(mol, idx, parent_idx)


def _enal_substituent_prefix(mol, idx: int, parent_idx: int) -> str:
    return _branch_rules()._enal_substituent_prefix(mol, idx, parent_idx)


def _enone_substituent_prefix(mol, idx: int, parent_idx: int) -> str:
    return _branch_rules()._enone_substituent_prefix(mol, idx, parent_idx)


def _haloethenyl_prefix(mol, idx: int, parent_idx: int) -> str:
    return _branch_rules()._haloethenyl_prefix(mol, idx, parent_idx)


def _ketenyl_prefix(mol, idx: int, parent_idx: int) -> str:
    return _branch_rules()._ketenyl_prefix(mol, idx, parent_idx)


def _linear_carbon_chain(mol, start_idx: int, blocked_idx: int, max_len: int) -> tuple[int, ...]:
    return _branch_rules()._linear_carbon_chain(mol, start_idx, blocked_idx, max_len)


def _nitro_branch_predicate(mol, idx: int, parent_idx: int) -> bool:
    return _branch_rules()._nitro_branch_predicate(mol, idx, parent_idx)

def _oxygen_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    parent = mol.GetAtomWithIdx(parent_idx)
    heavy = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() != parent_idx]
    if atom.GetFormalCharge() < 0 and parent.GetSymbol() == "N" and parent.GetFormalCharge() > 0:
        if bond is not None and bond.GetBondType() == Chem.BondType.SINGLE:
            return "oxido"
    if atom.GetFormalCharge() < 0 and not heavy:
        if bond is not None and bond.GetBondType() == Chem.BondType.SINGLE:
            return "oxido"
    if atom.GetFormalCharge() > 0 and not heavy:
        if bond is not None and bond.GetBondType() == Chem.BondType.SINGLE:
            return "oxidaniumyl"
    if atom.GetFormalCharge() > 0 and len(heavy) == 1:
        ylidene_oxidanium = _ylidene_oxidaniumyl_prefix(mol, heavy[0], idx)
        if ylidene_oxidanium:
            return ylidene_oxidanium
    if bond.GetBondType() == Chem.BondType.DOUBLE:
        return "oxo"
    if atom.GetFormalCharge() != 0:
        return ""
    if not heavy:
        return "hydroxy"
    if len(heavy) == 1:
        if _is_nitro_nitrogen(mol, heavy[0]):
            return "nitrooxy"
        if mol.GetAtomWithIdx(heavy[0]).GetSymbol() in HALO:
            return HALO[mol.GetAtomWithIdx(heavy[0]).GetSymbol()] + "oxy"
        carbonyl = _carbonyl_substituent_prefix(mol, heavy[0], idx)
        if carbonyl:
            return carbonyl + "oxy"
        if mol.GetAtomWithIdx(heavy[0]).IsInRing():
            ring = _ring_substituent_prefix(mol, heavy[0], idx)
            if not ring:
                return ""
            if ring == "phenyl":
                return "phenoxy"
            return f"({ring})oxy"
        alk = substituent_from_root(mol, heavy[0], idx)
        return _alkyl_to_alkoxy(alk)
    return ""


def _ylidene_oxidaniumyl_prefix(mol, carbon_idx: int, oxygen_idx: int) -> str:
    carbon = mol.GetAtomWithIdx(carbon_idx)
    if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0:
        return ""
    bond = mol.GetBondBetweenAtoms(carbon_idx, oxygen_idx)
    if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
        return ""
    heavy = [nbr.GetIdx() for nbr in carbon.GetNeighbors() if nbr.GetIdx() != oxygen_idx]
    if not heavy:
        return "methylideneoxidaniumyl"
    if len(heavy) != 1:
        return ""
    chain = _linear_carbon_chain(mol, heavy[0], carbon_idx, max_len=9)
    if not chain:
        return ""
    root = ROOTS.get(len(chain) + 1)
    return root + "ylideneoxidaniumyl" if root else ""


def _nitrogen_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if _is_diazo_attachment(mol, idx, parent_idx):
        return "diazo"
    if parent_bond is not None and parent_bond.GetBondType() == Chem.BondType.DOUBLE:
        return _imino_prefix(mol, idx, parent_idx)
    cumulene = _isocyanato_prefix(mol, idx, parent_idx)
    if cumulene:
        return cumulene
    if _is_isocyano_nitrogen(mol, idx):
        return "isocyano"
    if _is_diazonio_attachment(mol, idx, parent_idx):
        return "diazonio"
    azoxy = _azoxy_prefix(mol, idx, parent_idx)
    if azoxy:
        return azoxy
    diazenyl = _diazenyl_prefix(mol, idx, parent_idx)
    if diazenyl:
        return diazenyl
    hydrazone = _hydrazone_prefix(mol, idx, parent_idx)
    if hydrazone:
        return hydrazone
    imino_amino = _imino_amino_prefix(mol, idx, parent_idx)
    if imino_amino:
        return imino_amino
    amidino_amino = _amidino_amino_prefix(mol, idx, parent_idx)
    if amidino_amino:
        return amidino_amino
    if _is_azido_attachment(mol, idx, parent_idx):
        return "azido"
    if _nitro_branch_predicate(mol, idx, parent_idx):
        return "nitro"
    if _is_nitroso_attachment(mol, idx, parent_idx):
        return "nitroso"
    if atom.GetFormalCharge() > 0:
        heavy = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() != parent_idx]
        if not heavy:
            return "azaniumyl"
        names = []
        for nidx in heavy:
            if mol.GetAtomWithIdx(nidx).IsInRing():
                name = _ring_substituent_prefix(mol, nidx, idx)
            else:
                name = substituent_from_root(mol, nidx, idx)
            if not name:
                return ""
            names.append(name)
        counts = Counter(names)
        parts = []
        for name in sorted(counts):
            count = counts[name]
            if count > 1:
                parts.append(MULT.get(count, "") + name)
            else:
                parts.append(name)
        return "".join(parts) + "azaniumyl"
    if atom.GetFormalCharge() < 0:
        azanidyl = _azanidyl_prefix(mol, idx, parent_idx)
        if azanidyl:
            return azanidyl
    if atom.GetFormalCharge() != 0:
        return ""
    heavy = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() != parent_idx]
    if heavy and all(mol.GetAtomWithIdx(nidx).GetSymbol() in HALO for nidx in heavy):
        names = [HALO[mol.GetAtomWithIdx(nidx).GetSymbol()] for nidx in heavy]
        counts = Counter(names)
        return "".join((MULT.get(count, "") if count > 1 else "") + name for name, count in sorted(counts.items())) + "amino"
    if not heavy:
        return "amino"
    names = []
    for nidx in heavy:
        name = _nitrogen_ligand_prefix(mol, nidx, idx)
        if not name:
            return ""
        names.append(name)
    return _amino_prefix_from_ligands(names)


def _nitrogen_ligand_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    symbol = atom.GetSymbol()
    if symbol in HALO:
        return HALO[symbol]
    if atom.IsInRing():
        return _ring_substituent_prefix(mol, idx, parent_idx)
    if symbol == "N":
        return _nitrogen_prefix(mol, idx, parent_idx)
    if symbol == "S":
        return _sulfur_prefix(mol, idx, parent_idx)
    return (
        "cyano"
        if _is_nitrile_carbon(mol, idx)
        else _functional_carbon_substituent_name(mol, idx, parent_idx)
        or _carbonyl_substituent_prefix(mol, idx, parent_idx)
        or substituent_from_root(mol, idx, parent_idx)
    )


def _amino_prefix_from_ligands(names: Sequence[str]) -> str:
    if len(names) == 1:
        if "-" in names[0] or names[0].startswith("("):
            return f"({names[0]})amino"
        return names[0] + "amino"
    counts = Counter(names)
    if all(count == 1 for count in counts.values()) and all(
        "-" not in name and not name.startswith("(") and ")" not in name for name in counts
    ):
        ordered = sorted(counts)
        return ordered[0] + "".join(f"({name})" for name in ordered[1:]) + "amino"
    if all(count == 1 for count in counts.values()):
        parts = []
        for name in sorted(counts):
            if "-" in name or name.startswith("(") or ")" in name:
                parts.append(f"N-({name})")
            else:
                parts.append(f"N-{name}")
        return "-".join(parts) + "amino"
    parts = []
    for name in sorted(counts):
        count = counts[name]
        if count > 1 and name.endswith(("sulfonyl", "sulfinyl")):
            multi = {2: "bis", 3: "tris", 4: "tetrakis"}.get(count, MULT.get(count, "") + "kis")
            parts.append(f"{multi}({name})")
        elif count > 1 and ("-" in name or name.startswith("(")):
            multi = {2: "bis", 3: "tris", 4: "tetrakis"}.get(count, MULT.get(count, "") + "kis")
            parts.append(f"{multi}({name})")
        elif count == 1 and (name.endswith("carbonyl") or name.endswith("sulfonyl")) and len(names) > 1:
            parts.append(f"({name})")
        elif "-" in name or name.startswith("("):
            parts.append(f"({name})")
        else:
            parts.append((MULT.get(count, "") if count > 1 else "") + name)
    return "".join(parts) + "amino"


def _azanidyl_prefix(mol, idx: int, parent_idx: int) -> str:
    parent = mol.GetAtomWithIdx(parent_idx)
    if parent.GetFormalCharge() <= 0:
        return ""
    heavy = [n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors() if n.GetIdx() != parent_idx]
    if not heavy:
        return "azanidyl"
    if len(heavy) != 1:
        return ""
    substituent_idx = heavy[0]
    substituent = mol.GetAtomWithIdx(substituent_idx)
    if substituent.IsInRing():
        name = _ring_substituent_prefix(mol, substituent_idx, idx)
    elif substituent.GetSymbol() == "S":
        name = _sulfur_prefix(mol, substituent_idx, idx)
    elif substituent.GetSymbol() == "C":
        name = (
            "cyano"
            if _is_nitrile_carbon(mol, substituent_idx)
            else _functional_carbon_substituent_name(mol, substituent_idx, idx)
            or _carbonyl_substituent_prefix(mol, substituent_idx, idx)
            or substituent_from_root(mol, substituent_idx, idx)
        )
    elif substituent.GetSymbol() == "N":
        name = _nitrogen_prefix(mol, substituent_idx, idx)
    else:
        return ""
    if not name:
        return ""
    if "-" in name or name.startswith("("):
        return f"({name})azanidyl"
    return name + "azanidyl"


def _functional_carbon_substituent_name(mol, idx: int, parent_idx: int) -> str:
    if mol.GetAtomWithIdx(idx).GetSymbol() != "C":
        return ""
    return (
        _ketenyl_prefix(mol, idx, parent_idx)
        or _enone_substituent_prefix(mol, idx, parent_idx)
        or _enal_substituent_prefix(mol, idx, parent_idx)
        or _haloethenyl_prefix(mol, idx, parent_idx)
        or _alkynyl_substituent_prefix(mol, idx, parent_idx)
    )


def _imino_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0:
        return ""
    heavy = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() != parent_idx]
    if not heavy:
        return "imino"
    if len(heavy) != 1:
        return ""
    attached = heavy[0]
    if mol.GetAtomWithIdx(attached).IsInRing():
        name = _ring_substituent_prefix(mol, attached, idx)
    else:
        name = "cyano" if _is_nitrile_carbon(mol, attached) else substituent_from_root(mol, attached, idx)
    if not name:
        return ""
    if "-" in name or name.startswith("("):
        return f"({name})imino"
    return name + "imino"


def _isocyanato_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    for nbr in atom.GetNeighbors():
        cidx = nbr.GetIdx()
        if cidx == parent_idx or nbr.GetSymbol() != "C":
            continue
        n_c_bond = mol.GetBondBetweenAtoms(idx, cidx)
        if n_c_bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        terminal = [
            other
            for other in nbr.GetNeighbors()
            if other.GetIdx() != idx
            and mol.GetBondBetweenAtoms(cidx, other.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
        ]
        if len(terminal) != 1:
            continue
        symbol = terminal[0].GetSymbol()
        if symbol == "O":
            return "isocyanato"
        if symbol == "S":
            return "isothiocyanato"
    return ""


def _diazenyl_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0:
        return ""
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx or nbr.GetSymbol() != "N" or nbr.GetFormalCharge() != 0:
            continue
        bond = mol.GetBondBetweenAtoms(idx, nidx)
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        distal = [n.GetIdx() for n in nbr.GetNeighbors() if n.GetIdx() != idx]
        if not distal:
            return "diazenyl"
        if len(distal) != 1:
            continue
        didx = distal[0]
        if mol.GetAtomWithIdx(didx).IsInRing():
            name = _ring_substituent_prefix(mol, didx, nidx)
        else:
            name = substituent_from_root(mol, didx, nidx)
        if name:
            return name + "diazenyl"
    return ""


def _azoxy_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == parent_idx or nbr.GetSymbol() != "N" or nbr.GetFormalCharge() <= 0:
            continue
        n_bond = mol.GetBondBetweenAtoms(idx, nidx)
        if n_bond is None or n_bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        oxido = []
        organic = []
        for distal in nbr.GetNeighbors():
            didx = distal.GetIdx()
            if didx == idx:
                continue
            bond = mol.GetBondBetweenAtoms(nidx, didx)
            if (
                distal.GetSymbol() == "O"
                and distal.GetFormalCharge() < 0
                and bond is not None
                and bond.GetBondType() == Chem.BondType.SINGLE
            ):
                oxido.append(didx)
            elif bond is not None and bond.GetBondType() == Chem.BondType.SINGLE:
                organic.append(didx)
        if len(oxido) != 1 or len(organic) != 1:
            continue
        didx = organic[0]
        organic_atom = mol.GetAtomWithIdx(didx)
        if organic_atom.IsInRing():
            name = _ring_substituent_prefix(mol, didx, nidx)
            if name and not (name.startswith("(") or "-" in name):
                return f"({name}azoxy)"
        else:
            name = substituent_from_root(mol, didx, nidx)
        if not name:
            continue
        if "-" in name or name.startswith("("):
            return f"({name})azoxy"
        return name + "azoxy"
    return ""


def _hydrazone_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    heavy = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetIdx() != parent_idx]
    if len(heavy) != 1:
        return ""
    distal_n_idx = heavy[0]
    distal_n = mol.GetAtomWithIdx(distal_n_idx)
    if distal_n.GetSymbol() != "N" or distal_n.GetFormalCharge() != 0:
        return ""
    n_bond = mol.GetBondBetweenAtoms(idx, distal_n_idx)
    if n_bond is None or n_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    imine_carbons = [
        nbr.GetIdx()
        for nbr in distal_n.GetNeighbors()
        if nbr.GetIdx() != idx
        and nbr.GetSymbol() == "C"
        and mol.GetBondBetweenAtoms(distal_n_idx, nbr.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
    ]
    if len(imine_carbons) != 1:
        return ""
    ylidene = _imine_carbon_ylidene(mol, imine_carbons[0], distal_n_idx)
    return ylidene + "hydrazinyl" if ylidene else ""


def _imino_amino_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    imine_carbons = [
        nbr.GetIdx()
        for nbr in atom.GetNeighbors()
        if nbr.GetIdx() != parent_idx
        and nbr.GetSymbol() == "C"
        and mol.GetBondBetweenAtoms(idx, nbr.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
    ]
    if len(imine_carbons) != 1:
        return ""
    ylidene = _imine_carbon_ylidene(mol, imine_carbons[0], idx)
    return ylidene + "amino" if ylidene else ""


def _amidino_amino_prefix(mol, idx: int, parent_idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetSymbol() != "N" or atom.GetFormalCharge() != 0:
        return ""
    parent_bond = mol.GetBondBetweenAtoms(idx, parent_idx)
    if parent_bond is None or parent_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    heavy = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetIdx() != parent_idx]
    if len(heavy) != 1:
        return ""
    carbon_idx = heavy[0]
    carbon = mol.GetAtomWithIdx(carbon_idx)
    if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() != 0:
        return ""
    n_c_bond = mol.GetBondBetweenAtoms(idx, carbon_idx)
    if n_c_bond is None or n_c_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    terminal_imino = []
    other_heavy = []
    for nbr in carbon.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == idx:
            continue
        bond = mol.GetBondBetweenAtoms(carbon_idx, nidx)
        if (
            nbr.GetSymbol() == "N"
            and nbr.GetFormalCharge() == 0
            and bond is not None
            and bond.GetBondType() == Chem.BondType.DOUBLE
            and not [nnbr for nnbr in nbr.GetNeighbors() if nnbr.GetIdx() != carbon_idx and nnbr.GetAtomicNum() > 1]
        ):
            terminal_imino.append(nidx)
        else:
            other_heavy.append(nidx)
    if len(terminal_imino) != 1 or other_heavy:
        return ""
    return "iminomethylamino"


def _imine_carbon_ylidene(mol, carbon_idx: int, nitrogen_idx: int) -> str:
    atom = mol.GetAtomWithIdx(carbon_idx)
    if atom.IsInRing():
        rings = [ring for ring in mol.GetRingInfo().AtomRings() if carbon_idx in ring]
        if len(rings) != 1:
            return ""
        ring = rings[0]
        if not all(mol.GetAtomWithIdx(idx).GetSymbol() == "C" for idx in ring):
            return ""
        ring_set = set(ring)
        for a, b in zip(ring, ring[1:] + ring[:1]):
            bond = mol.GetBondBetweenAtoms(a, b)
            if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
                return ""
        for ring_idx in ring:
            for nbr in mol.GetAtomWithIdx(ring_idx).GetNeighbors():
                nidx = nbr.GetIdx()
                if nidx not in ring_set and nidx != nitrogen_idx:
                    return ""
        root = ROOTS.get(len(ring))
        return f"cyclo{root}ylidene" if root else ""
    substituents = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetIdx() != nitrogen_idx]
    if len(substituents) == 1:
        sub_idx = substituents[0]
        sub_atom = mol.GetAtomWithIdx(sub_idx)
        sub_bond = mol.GetBondBetweenAtoms(carbon_idx, sub_idx)
        if (
            sub_atom.GetSymbol() == "N"
            and sub_atom.GetFormalCharge() == 0
            and sub_bond is not None
            and sub_bond.GetBondType() == Chem.BondType.SINGLE
            and not [nbr for nbr in sub_atom.GetNeighbors() if nbr.GetIdx() != carbon_idx and nbr.GetAtomicNum() > 1]
        ):
            return "aminomethylidene"
        if (
            sub_atom.GetSymbol() == "N"
            and sub_atom.GetFormalCharge() == 0
            and sub_bond is not None
            and sub_bond.GetBondType() == Chem.BondType.SINGLE
        ):
            amino = _nitrogen_prefix(mol, sub_idx, carbon_idx)
            if amino and amino.endswith("amino") and "-" not in amino and "(" not in amino and ")" not in amino:
                return amino + "methylidene"
        if (
            sub_atom.GetSymbol() == "N"
            and sub_atom.GetFormalCharge() == 0
            and sub_bond is not None
            and sub_bond.GetBondType() == Chem.BondType.DOUBLE
            and not [nbr for nbr in sub_atom.GetNeighbors() if nbr.GetIdx() != carbon_idx and nbr.GetAtomicNum() > 1]
        ):
            return "iminomethylidene"
        if mol.GetAtomWithIdx(sub_idx).IsInRing():
            name = _ring_substituent_prefix(mol, sub_idx, carbon_idx)
            if name and name.endswith("yl"):
                return name + "methylidene"
            return ""
        name = substituent_from_root(mol, sub_idx, carbon_idx)
        if name and "-" not in name and not name.startswith("("):
            return name + "methylidene"
        return ""
    if len(substituents) == 2:
        names = []
        for sub_idx in substituents:
            if mol.GetAtomWithIdx(sub_idx).IsInRing():
                name = _ring_substituent_prefix(mol, sub_idx, carbon_idx)
            else:
                name = substituent_from_root(mol, sub_idx, carbon_idx)
            if not name or "-" in name or name.startswith("("):
                return ""
            names.append(name)
        counts = Counter(names)
        parts = []
        for name in sorted(counts):
            count = counts[name]
            parts.append((MULT.get(count, "") if count > 1 else "") + name)
        return "".join(parts) + "methylidene"
    return _alkylidene_prefix(mol, carbon_idx, nitrogen_idx)
