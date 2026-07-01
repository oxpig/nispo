from __future__ import annotations

import re
from functools import lru_cache

from .constants import ROOTS
from .molecule_cache import mol_cache_get, mol_cache_set
from .name_parts import NameParts, Suffix
from .suffix_store import inverse_suffix_store


@lru_cache(maxsize=None)
def _suffix_value(value: str) -> str:
    specs = inverse_suffix_store().by_value.get(value)
    return specs[0].value if specs else value


class FragmentAssemblyRule:
    """First-class naming rule for disconnected fragments, salts, and adducts."""

    def __init__(self, Chem, engine):
        self.Chem = Chem
        self.engine = engine

    def name(self, fragments: tuple[object, ...]) -> str:
        named_fragments: list[str] = []
        cations: list[str] = []
        neutral_halide_suffixes: list[str] = []
        neutral_halide_names: list[str] = []
        anions: list[str] = []

        for fragment in fragments:
            halide_anion = single_atom_halide_anion_name(fragment)
            if halide_anion is not None:
                anions.append(halide_anion)
                continue
            neutral_halide = neutral_halide_name(fragment)
            if neutral_halide is not None:
                symbol, _charge = single_atom_halide(fragment)
                neutral_halide_suffixes.append(neutral_halide)
                neutral_halide_names.append(f"hydrogen {HALIDE_ANIONS[symbol]}")
                continue
            common_anion = common_single_atom_anion_name(fragment)
            if common_anion is not None:
                anions.append(common_anion)
                continue
            cation = single_atom_cation_name(fragment, self.Chem)
            if cation is not None:
                cations.append(cation)
                continue
            polyatomic = common_polyatomic_fragment_name(fragment, self.Chem)
            if polyatomic:
                anions.append(polyatomic)
                continue
            simple_name = simple_fragment_name(fragment, self.Chem)
            if simple_name:
                named_fragments.append(simple_name)
                continue
            name = self.engine.name(fragment)
            if not name:
                return ""
            named_fragments.append(name)

        if len(named_fragments) == 1 and neutral_halide_suffixes and not anions and not cations:
            return _salt_name(named_fragments[0], neutral_halide_suffixes)
        if len(named_fragments) == 1 and anions and not neutral_halide_suffixes and not cations:
            return _salt_name(named_fragments[0], anions)
        if len(cations) == 1 and anions and not named_fragments and not neutral_halide_suffixes:
            return _salt_name(cations[0], anions)

        standalone = cations + named_fragments + anions + neutral_halide_names
        if standalone:
            return " and ".join(sorted(standalone, key=lambda item: (len(item), item)))
        return ""


HALIDE_STEMS = {"F": "fluor", "Cl": "chlor", "Br": "brom", "I": "iod"}
HALOGEN_ELEMENT_NAMES = {"F": "fluorine", "Cl": "chlorine", "Br": "bromine", "I": "iodine"}
HALOGEN_ELECTROPOSITIVE_ORDER = {"F": 0, "Cl": 1, "Br": 2, "I": 3}

NEUTRAL_HALIDE_SALTS = {
    symbol: "hydro" + stem + _suffix_value("ide")
    for symbol, stem in HALIDE_STEMS.items()
}

HALIDE_ANIONS = {symbol: stem + _suffix_value("ide") for symbol, stem in HALIDE_STEMS.items()}

COMMON_MONATOMIC_CATIONS = {
    "Li": "lithium",
    "Na": "sodium",
    "K": "potassium",
    "Rb": "rubidium",
    "Cs": "cesium",
    "Mg": "magnesium",
    "Ca": "calcium",
    "Sr": "strontium",
    "Ba": "barium",
    "Al": "aluminium",
    "Zn": "zinc",
    "Ag": "silver",
}

COMMON_SINGLE_ATOM_CATIONS = {
    ("N", 1, 4): "ammonium",
    ("O", 1, 3): "hydronium",
    ("P", 1, 4): "phosphonium",
    ("S", 1, 3): "sulfonium",
}

NEUTRAL_HYDRIDES = {
    ("B", 3): "borane",
    ("C", 4): "methane",
    ("N", 3): "ammonia",
    ("O", 2): "water",
    ("P", 3): "phosphane",
    ("S", 2): "sulfane",
    ("Si", 4): "silane",
}

MULTIPLIERS = {
    2: "di",
    3: "tri",
    4: "tetra",
    5: "penta",
    6: "hexa",
}

COMPLEX_MULTIPLIERS = {
    2: "bis",
    3: "tris",
    4: "tetrakis",
    5: "pentakis",
    6: "hexakis",
}

SMALL_RETAINED_TEMPLATES = (
    ("azapyrone", "O=c1nccco1"),
    ("azepinone", "O=c1cccccn1"),
    ("azasydnone", "O=c1n[nH+][n-]o1"),
    ("benzoylium", "O=[C+]c1ccccc1"),
    ("diazirinone", "O=c1nn1"),
    ("epoxyurea", "O=c1[nH]o[nH]1"),
    ("furoxanone", "O=C1C=NO[NH+]1[O-]"),
    ("4-oxazinone", "O=c1cconc1"),
    ("oxalimide", "O=c1[nH]c1=O"),
    ("oxoiodolium", "O=[I+]1C=CC=C1"),
    ("selenazolium", "c1c[se]c[nH+]1"),
    ("thiapyrone", "O=c1ccccs1"),
)

PERFLUORO_ANIONS = {
    ("B", 4, -1): "tetrafluoroborate",
    ("P", 6, -1): "hexafluorophosphate",
    ("As", 6, -1): "hexafluoroarsenate",
    ("Sb", 6, -1): "hexafluoroantimonate",
}

OXOANIONS = {
    ("C", 2, -1, 0): "formate",
    ("C", 2, 0, 1): "formic acid",
    ("C", 3, -2, 0): "carbonate",
    ("C", 3, -1, 1): "hydrogen carbonate",
    ("N", 2, -1, 0): "nitrite",
    ("N", 3, -1, 0): "nitrate",
    ("N", 3, -2, 0): "azonate",
    ("N", 4, -3, 0): "orthonitrate",
    ("P", 4, -3, 0): "phosphate",
    ("P", 4, -2, 1): "hydrogen phosphate",
    ("P", 4, -1, 2): "dihydrogen phosphate",
    ("P", 3, -3, 0): "phosphite",
    ("P", 2, -1, 0): "hypophosphite",
    ("S", 3, -2, 0): "sulfite",
    ("S", 3, -1, 1): "hydrogen sulfite",
    ("S", 4, -2, 0): "sulfate",
    ("S", 4, -1, 1): "hydrogen sulfate",
    ("F", 1, -1, 0): "hypofluorite",
    ("Cl", 1, 0, 1): "hypochlorous acid",
    ("Cl", 1, -1, 0): "hypochlorite",
    ("Cl", 2, 0, 1): "chlorous acid",
    ("Cl", 2, -1, 0): "chlorite",
    ("Cl", 3, 0, 1): "chloric acid",
    ("Cl", 3, -1, 0): "chlorate",
    ("Cl", 4, 0, 1): "perchloric acid",
    ("Cl", 4, -1, 0): "perchlorate",
    ("Br", 1, 0, 1): "hypobromous acid",
    ("Br", 1, -1, 0): "hypobromite",
    ("Br", 2, 0, 1): "bromous acid",
    ("Br", 2, -1, 0): "bromite",
    ("Br", 3, 0, 1): "bromic acid",
    ("Br", 3, -1, 0): "bromate",
    ("Br", 4, 0, 1): "perbromic acid",
    ("Br", 4, -1, 0): "perbromate",
    ("I", 1, 0, 1): "hypoiodous acid",
    ("I", 1, -1, 0): "hypoiodite",
    ("I", 2, 0, 1): "iodous acid",
    ("I", 2, -1, 0): "iodite",
    ("I", 3, 0, 1): "iodic acid",
    ("I", 3, -1, 0): "iodate",
    ("I", 4, 0, 1): "periodic acid",
    ("I", 4, -1, 0): "periodate",
    ("I", 6, 0, 5): "orthoperiodic acid",
    ("I", 6, -5, 0): "orthoperiodate",
}


def single_atom_halide(mol) -> tuple[str, int] | None:
    if mol.GetNumAtoms() != 1:
        return None
    atom = mol.GetAtomWithIdx(0)
    symbol = atom.GetSymbol()
    if symbol not in HALIDE_ANIONS:
        return None
    return symbol, atom.GetFormalCharge()


def single_atom_halide_anion_name(mol) -> str | None:
    halide = single_atom_halide(mol)
    if halide is None:
        return None
    symbol, charge = halide
    return HALIDE_ANIONS[symbol] if charge < 0 else None


def neutral_halide_name(mol) -> str | None:
    halide = single_atom_halide(mol)
    if halide is None:
        return None
    symbol, charge = halide
    return NEUTRAL_HALIDE_SALTS[symbol] if charge == 0 else None


def common_single_atom_anion_name(mol) -> str | None:
    if mol.GetNumAtoms() != 1:
        return None
    atom = mol.GetAtomWithIdx(0)
    hydride = _hydride_anion_name(atom)
    if hydride:
        return hydride
    if atom.GetSymbol() == "B" and atom.GetFormalCharge() == -1 and atom.GetTotalNumHs() == 4:
        return "borohydride"
    if atom.GetSymbol() == "O" and atom.GetFormalCharge() < 0 and atom.GetTotalNumHs() > 0:
        return "hydroxide"
    return None


def single_atom_cation_name(mol, Chem) -> str | None:
    if mol.GetNumAtoms() != 1:
        return None
    atom = mol.GetAtomWithIdx(0)
    if atom.GetFormalCharge() <= 0:
        return None
    symbol = atom.GetSymbol()
    common = COMMON_SINGLE_ATOM_CATIONS.get((symbol, atom.GetFormalCharge(), atom.GetTotalNumHs()))
    if common:
        return common
    if symbol in COMMON_MONATOMIC_CATIONS:
        return COMMON_MONATOMIC_CATIONS[symbol]
    element_name = Chem.GetPeriodicTable().GetElementName(atom.GetAtomicNum()).lower()
    return NameParts(parent=element_name, charge=f"({atom.GetFormalCharge()}+)").render()


def single_atom_anion_name(mol, Chem) -> str | None:
    if mol.GetNumAtoms() != 1:
        return None
    atom = mol.GetAtomWithIdx(0)
    if atom.GetFormalCharge() >= 0:
        return None
    symbol = atom.GetSymbol()
    if symbol in HALIDE_ANIONS:
        return HALIDE_ANIONS[symbol]
    element_name = Chem.GetPeriodicTable().GetElementName(atom.GetAtomicNum()).lower()
    return NameParts(parent=element_name, charge=f"({abs(atom.GetFormalCharge())}-)").render()


def simple_fragment_name(mol, Chem) -> str:
    if mol.GetNumAtoms() != 1:
        return ""
    atom = mol.GetAtomWithIdx(0)
    symbol = atom.GetSymbol()
    charge = atom.GetFormalCharge()
    common_anion = common_single_atom_anion_name(mol)
    if common_anion:
        return common_anion
    anion = single_atom_anion_name(mol, Chem)
    if anion:
        return anion
    neutral_halide = neutral_halide_name(mol)
    if neutral_halide:
        return f"hydrogen {HALIDE_ANIONS[symbol]}"
    cation = single_atom_cation_name(mol, Chem)
    if cation:
        return cation
    neutral_hydride = NEUTRAL_HYDRIDES.get((symbol, atom.GetTotalNumHs()))
    if neutral_hydride and charge == 0:
        return neutral_hydride

    element_name = Chem.GetPeriodicTable().GetElementName(atom.GetAtomicNum()).lower()
    if not element_name:
        return ""
    if charge == 0:
        return element_name
    sign = "+" if charge > 0 else "-"
    return NameParts(parent=element_name, charge=f"({abs(charge)}{sign})").render()


def common_polyatomic_fragment_name(mol, Chem) -> str:
    """Name common standalone counterions by graph features, not row lookups."""

    if mol.GetNumAtoms() < 2:
        return ""
    return (
        _small_standalone_polyatomic_name(mol, Chem)
        or _substituted_borohydride_name(mol, Chem)
        or _organotrifluoroborate_name(mol, Chem)
        or _tetraorganoborate_name(mol, Chem)
        or _azanide_name(mol, Chem)
        or _perfluoro_counterion_name(mol, Chem)
        or _trifluoroacetate_name(mol, Chem)
        or _oxoanion_name(mol, Chem)
        or _diphosphite_name(mol, Chem)
        or _phosphorus_imido_chalcogen_oxoanion_name(mol, Chem)
        or _phosphorus_hydrido_oxoanion_name(mol, Chem)
        or _phosphorus_hydrido_thiooxoanion_name(mol, Chem)
        or _simple_phosphorus_chalcogen_anion_name(mol, Chem)
        or _simple_nitrogen_sulfur_anion_name(mol, Chem)
        or _simple_thiosulfur_oxoanion_name(mol, Chem)
        or _chalcogen_thiooxoanion_name(mol, Chem)
        or _phosphorus_thiooxoanion_name(mol, Chem)
        or _phosphorus_selenooxoanion_name(mol, Chem)
        or _organophosphorus_oxoanion_name(mol, Chem)
        or _pseudohalide_name(mol, Chem)
        or _bis_trifluoromethanesulfonyl_azanide_name(mol, Chem)
        or _trifluoromethanesulfonate_name(mol, Chem)
    )


def standalone_polyatomic_fragment_name(mol, Chem) -> str:
    """Name retained neutral/cationic simple fragments outside salt assembly."""

    if mol.GetNumAtoms() < 2:
        return ""
    return (
        _simple_phosphorus_chalcogen_neutral_name(mol, Chem)
        or _simple_nitrogen_sulfur_neutral_name(mol, Chem)
        or _small_retained_template_name(mol, Chem)
        or _isodiazene_name(mol, Chem)
        or _azanol_name(mol, Chem)
        or _nitronic_acid_or_ester_name(mol, Chem)
        or _phosphorus_oxo_cation_name(mol, Chem)
        or _sulfur_oxo_cation_name(mol, Chem)
        or _peracetate_name(mol, Chem)
        or _thiauronium_name(mol, Chem)
        or _isothiouronium_name(mol, Chem)
        or _isouronium_name(mol, Chem)
        or _selenocysteamine_name(mol, Chem)
    )


_SMALL_RETAINED_TEMPLATE_CACHE = {}


def _small_retained_template_name(mol, Chem) -> str:
    template_index = _small_retained_template_index(Chem)
    if not template_index:
        return ""
    key = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    return template_index.get(key, "")


def _small_retained_template_index(Chem) -> dict[str, str]:
    key = id(Chem)
    cached = _SMALL_RETAINED_TEMPLATE_CACHE.get(key)
    if cached is not None:
        return cached
    index = {}
    for name, smiles in SMALL_RETAINED_TEMPLATES:
        template = Chem.MolFromSmiles(smiles)
        if template is None:
            continue
        index[Chem.MolToSmiles(template, canonical=True, isomericSmiles=True)] = name
    _SMALL_RETAINED_TEMPLATE_CACHE[key] = index
    return index


def _small_standalone_polyatomic_name(mol, Chem) -> str:
    return (
        _oxidonitrogen_anion_name(mol, Chem)
        or _cyanonitrate_name(mol, Chem)
        or _cyanoiodate_name(mol, Chem)
        or _hypothiocyanite_name(mol, Chem)
        or _alkanesulfenate_name(mol, Chem)
        or _phosphacycloalkanolate_name(mol, Chem)
        or _diazenide_name(mol, Chem)
        or _diselenide_name(mol, Chem)
        or _halogen_oxide_name(mol, Chem)
        or _interhalogen_name(mol, Chem)
        or _haloazanide_name(mol, Chem)
        or _acetonide_name(mol, Chem)
        or _nitronite_name(mol, Chem)
        or _peroxycarbonate_name(mol, Chem)
        or _peroxyoxalate_name(mol, Chem)
    )


def _simple_phosphorus_chalcogen_neutral_name(mol, Chem) -> str:
    return (
        _phosphaethene_yne_name(mol, Chem)
        or _alkanethiosulfonic_acid_name(mol, Chem)
        or _alkyl_phosphorous_acid_name(mol, Chem)
        or _alkyl_phosphinic_halide_name(mol, Chem)
        or _halo_alkenyl_phosphinic_acid_name(mol, Chem)
        or _phosphorous_acid_dialkylamide_name(mol, Chem)
    )


def _simple_phosphorus_chalcogen_anion_name(mol, Chem) -> str:
    return (
        _alkanesulfonimidate_name(mol, Chem)
        or _alkyl_trithiophosphonate_name(mol, Chem)
        or _alkyl_phosphonochloridoate_name(mol, Chem)
        or _dialkoxy_oxo_phosphanuide_name(mol, Chem)
    )


def _simple_nitrogen_sulfur_neutral_name(mol, Chem) -> str:
    return _alkyl_thiosulfoximine_name(mol, Chem)


def _simple_nitrogen_sulfur_anion_name(mol, Chem) -> str:
    return (
        _isothiocyanoamide_name(mol, Chem)
        or _dialkylhydrazinide_name(mol, Chem)
        or _n_alkyl_nitrosoamide_name(mol, Chem)
        or _dialkyl_thiolate_name(mol, Chem)
    )


def _phosphaethene_yne_name(mol, Chem) -> str:
    if _element_counts(mol) != {"C": 1, "P": 1} or _total_charge(mol) != 0:
        return ""
    carbon = next(atom for atom in mol.GetAtoms() if atom.GetSymbol() == "C")
    phosphorus = next(atom for atom in mol.GetAtoms() if atom.GetSymbol() == "P")
    bond = mol.GetBondBetweenAtoms(carbon.GetIdx(), phosphorus.GetIdx())
    if bond is None:
        return ""
    if carbon.GetFormalCharge() or phosphorus.GetFormalCharge():
        return ""
    if bond.GetBondType() == Chem.BondType.DOUBLE and carbon.GetTotalNumHs() == 2 and phosphorus.GetTotalNumHs() == 1:
        return "phosphaethene"
    if bond.GetBondType() == Chem.BondType.TRIPLE and carbon.GetTotalNumHs() == 1 and phosphorus.GetTotalNumHs() == 0:
        return "phosphaethyne"
    return ""


def _alkanethiosulfonic_acid_name(mol, Chem) -> str:
    if _total_charge(mol) != 0:
        return ""
    sulfurs = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == "S" and not atom.IsInRing()]
    if len(sulfurs) != 2:
        return ""
    for sulfur in sulfurs:
        sidx = sulfur.GetIdx()
        carbon_idx = terminal_sulfur_idx = None
        double_oxygen = hydroxy = 0
        covered = {sidx}
        for nbr in sulfur.GetNeighbors():
            nidx = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(sidx, nidx)
            if bond is None:
                return ""
            if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE and nbr.GetFormalCharge() == 0:
                double_oxygen += 1
                covered.add(nidx)
            elif (
                nbr.GetSymbol() == "O"
                and bond.GetBondType() == Chem.BondType.SINGLE
                and nbr.GetFormalCharge() == 0
                and nbr.GetTotalNumHs() == 1
            ):
                hydroxy += 1
                covered.add(nidx)
            elif nbr.GetSymbol() == "S" and bond.GetBondType() == Chem.BondType.DOUBLE and nbr.GetFormalCharge() == 0:
                terminal_sulfur_idx = nidx
                covered.add(nidx)
            elif nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE:
                carbon_idx = nidx
            else:
                return ""
        if double_oxygen != 1 or hydroxy != 1 or terminal_sulfur_idx is None or carbon_idx is None:
            continue
        chain = _linear_fragment_carbon_chain(mol, Chem, carbon_idx, sidx)
        if not chain:
            continue
        chain_atoms, amino_locants = chain
        covered.update(chain_atoms)
        for locant, nitrogen_idx in amino_locants:
            covered.add(nitrogen_idx)
        if covered != set(range(mol.GetNumAtoms())):
            continue
        root = ROOTS.get(len(chain_atoms), "")
        if not root:
            continue
        prefixes = _fragment_locant_prefix_text("amino", [locant for locant, _idx in amino_locants])
        return f"{prefixes}{root}anethiosulfonic acid"
    return ""


def _alkanesulfonimidate_name(mol, Chem) -> str:
    if _total_charge(mol) != -1:
        return ""
    sulfur_atoms = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == "S" and atom.GetFormalCharge() == 0]
    if len(sulfur_atoms) != 1:
        return ""
    sulfur = sulfur_atoms[0]
    sidx = sulfur.GetIdx()
    carbon_idx = imino_idx = oxido_idx = None
    double_oxygen = 0
    covered = {sidx}
    for nbr in sulfur.GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(sidx, nidx)
        if bond is None:
            return ""
        if nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE:
            carbon_idx = nidx
        elif nbr.GetSymbol() == "N" and bond.GetBondType() == Chem.BondType.DOUBLE and nbr.GetFormalCharge() == 0:
            imino_idx = nidx
            covered.add(nidx)
        elif nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE and nbr.GetFormalCharge() == 0:
            double_oxygen += 1
            covered.add(nidx)
        elif nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetFormalCharge() == -1:
            oxido_idx = nidx
            covered.add(nidx)
        else:
            return ""
    if carbon_idx is None or imino_idx is None or oxido_idx is None or double_oxygen != 1:
        return ""
    chain = _linear_fragment_carbon_chain(mol, Chem, carbon_idx, sidx)
    if not chain:
        return ""
    chain_atoms, amino_locants = chain
    if amino_locants:
        return ""
    covered.update(chain_atoms)
    if covered != set(range(mol.GetNumAtoms())):
        return ""
    root = ROOTS.get(len(chain_atoms), "")
    return f"{root}anesulfonimidate" if root else ""


def _alkyl_phosphorous_acid_name(mol, Chem) -> str:
    if _total_charge(mol) != 0:
        return ""
    phosphorus = _single_phosphorus_atom(mol)
    if phosphorus is None or phosphorus.GetFormalCharge() != 0 or phosphorus.GetTotalNumHs() != 1:
        return ""
    pidx = phosphorus.GetIdx()
    carbon_idx = None
    hydroxy = 0
    covered = {pidx}
    for nbr in phosphorus.GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(pidx, nidx)
        if bond is None:
            return ""
        if nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE:
            carbon_idx = nidx
        elif (
            nbr.GetSymbol() == "O"
            and bond.GetBondType() == Chem.BondType.SINGLE
            and nbr.GetFormalCharge() == 0
            and nbr.GetTotalNumHs() == 1
        ):
            hydroxy += 1
            covered.add(nidx)
        else:
            return ""
    if carbon_idx is None or hydroxy != 3:
        return ""
    owner, owner_atoms = _simple_fragment_owner_name(mol, Chem, carbon_idx, pidx)
    if not owner:
        return ""
    covered.update(owner_atoms)
    return f"{owner}phosphorous acid" if covered == set(range(mol.GetNumAtoms())) else ""


def _alkyl_phosphinic_halide_name(mol, Chem) -> str:
    if _total_charge(mol) != 0:
        return ""
    phosphorus = _single_phosphorus_atom(mol)
    if phosphorus is None or phosphorus.GetFormalCharge() != 0 or phosphorus.GetTotalNumHs() != 1:
        return ""
    pidx = phosphorus.GetIdx()
    carbon_idx = halogen = None
    double_oxygen = 0
    covered = {pidx}
    for nbr in phosphorus.GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(pidx, nidx)
        if bond is None:
            return ""
        if nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE:
            carbon_idx = nidx
        elif nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE and nbr.GetFormalCharge() == 0:
            double_oxygen += 1
            covered.add(nidx)
        elif nbr.GetSymbol() in HALIDE_ANIONS and bond.GetBondType() == Chem.BondType.SINGLE:
            halogen = nbr.GetSymbol()
            covered.add(nidx)
        else:
            return ""
    if carbon_idx is None or halogen is None or double_oxygen != 1:
        return ""
    owner, owner_atoms = _simple_fragment_owner_name(mol, Chem, carbon_idx, pidx)
    if not owner:
        return ""
    covered.update(owner_atoms)
    return f"{owner}phosphinic {HALIDE_ANIONS[halogen]}" if covered == set(range(mol.GetNumAtoms())) else ""


def _halo_alkenyl_phosphinic_acid_name(mol, Chem) -> str:
    if _total_charge(mol) != 0:
        return ""
    phosphorus = _single_phosphorus_atom(mol)
    if phosphorus is None or phosphorus.GetFormalCharge() != 0 or phosphorus.GetTotalNumHs() != 0:
        return ""
    pidx = phosphorus.GetIdx()
    carbon_idx = halogen = None
    double_oxygen = hydroxy = 0
    covered = {pidx}
    for nbr in phosphorus.GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(pidx, nidx)
        if bond is None:
            return ""
        if nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE:
            carbon_idx = nidx
        elif nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE and nbr.GetFormalCharge() == 0:
            double_oxygen += 1
            covered.add(nidx)
        elif (
            nbr.GetSymbol() == "O"
            and bond.GetBondType() == Chem.BondType.SINGLE
            and nbr.GetFormalCharge() == 0
            and nbr.GetTotalNumHs() == 1
        ):
            hydroxy += 1
            covered.add(nidx)
        elif nbr.GetSymbol() in HALIDE_ANIONS and bond.GetBondType() == Chem.BondType.SINGLE:
            halogen = nbr.GetSymbol()
            covered.add(nidx)
        else:
            return ""
    if carbon_idx is None or halogen is None or double_oxygen != 1 or hydroxy != 1:
        return ""
    owner, owner_atoms = _simple_fragment_owner_name(mol, Chem, carbon_idx, pidx)
    if owner == "vinyl":
        owner = "ethenyl"
    if not owner:
        return ""
    covered.update(owner_atoms)
    return f"{HALIDE_STEMS[halogen]}o({owner})phosphinic acid" if covered == set(range(mol.GetNumAtoms())) else ""


def _alkyl_trithiophosphonate_name(mol, Chem) -> str:
    if _total_charge(mol) != -2:
        return ""
    phosphorus = _single_phosphorus_atom(mol)
    if phosphorus is None or phosphorus.GetFormalCharge() != 0:
        return ""
    pidx = phosphorus.GetIdx()
    carbon_idx = None
    double_sulfur = anionic_sulfur = 0
    covered = {pidx}
    for nbr in phosphorus.GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(pidx, nidx)
        if bond is None:
            return ""
        if nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE:
            carbon_idx = nidx
        elif nbr.GetSymbol() == "S" and bond.GetBondType() == Chem.BondType.DOUBLE and nbr.GetFormalCharge() == 0:
            double_sulfur += 1
            covered.add(nidx)
        elif nbr.GetSymbol() == "S" and bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetFormalCharge() == -1:
            anionic_sulfur += 1
            covered.add(nidx)
        else:
            return ""
    if carbon_idx is None or double_sulfur != 1 or anionic_sulfur != 2:
        return ""
    owner, owner_atoms = _simple_fragment_owner_name(mol, Chem, carbon_idx, pidx)
    if not owner:
        return ""
    covered.update(owner_atoms)
    return f"{owner}trithiophosphonate" if covered == set(range(mol.GetNumAtoms())) else ""


def _alkyl_phosphonochloridoate_name(mol, Chem) -> str:
    if _total_charge(mol) != -1:
        return ""
    phosphorus = _single_phosphorus_atom(mol)
    if phosphorus is None or phosphorus.GetFormalCharge() != 0:
        return ""
    pidx = phosphorus.GetIdx()
    carbon_idx = None
    double_oxygen = oxido = chloride = 0
    covered = {pidx}
    for nbr in phosphorus.GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(pidx, nidx)
        if bond is None:
            return ""
        if nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE:
            carbon_idx = nidx
        elif nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE and nbr.GetFormalCharge() == 0:
            double_oxygen += 1
            covered.add(nidx)
        elif nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetFormalCharge() == -1:
            oxido += 1
            covered.add(nidx)
        elif nbr.GetSymbol() == "Cl" and bond.GetBondType() == Chem.BondType.SINGLE:
            chloride += 1
            covered.add(nidx)
        else:
            return ""
    if carbon_idx is None or double_oxygen != 1 or oxido != 1 or chloride != 1:
        return ""
    owner, owner_atoms = _simple_fragment_owner_name(mol, Chem, carbon_idx, pidx)
    if not owner:
        return ""
    covered.update(owner_atoms)
    return f"{owner}phosphonochloridoate" if covered == set(range(mol.GetNumAtoms())) else ""


def _dialkoxy_oxo_phosphanuide_name(mol, Chem) -> str:
    if _total_charge(mol) != -1:
        return ""
    phosphorus = _single_phosphorus_atom(mol)
    if phosphorus is None or phosphorus.GetFormalCharge() != -1:
        return ""
    pidx = phosphorus.GetIdx()
    double_oxygen = 0
    alkoxy_names = []
    covered = {pidx}
    for nbr in phosphorus.GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(pidx, nidx)
        if bond is None:
            return ""
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE and nbr.GetFormalCharge() == 0:
            double_oxygen += 1
            covered.add(nidx)
        elif nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetFormalCharge() == 0:
            owner_atoms = [atom.GetIdx() for atom in nbr.GetNeighbors() if atom.GetIdx() != pidx and atom.GetSymbol() == "C"]
            if len(owner_atoms) != 1:
                return ""
            owner, atoms = _simple_fragment_owner_name(mol, Chem, owner_atoms[0], nidx)
            alkoxy = _alkoxy_fragment_name(owner)
            if not alkoxy:
                return ""
            alkoxy_names.append(alkoxy)
            covered.add(nidx)
            covered.update(atoms)
        else:
            return ""
    if double_oxygen != 1 or len(alkoxy_names) != 2 or covered != set(range(mol.GetNumAtoms())):
        return ""
    if alkoxy_names[0] != alkoxy_names[1]:
        return ""
    return f"di{alkoxy_names[0]}(oxo)phosphanuide"


def _phosphorous_acid_dialkylamide_name(mol, Chem) -> str:
    if _total_charge(mol) != 0:
        return ""
    phosphorus = _single_phosphorus_atom(mol)
    if phosphorus is None or phosphorus.GetFormalCharge() != 0 or phosphorus.GetTotalNumHs() != 0:
        return ""
    pidx = phosphorus.GetIdx()
    nitrogen_idx = None
    hydroxy = 0
    covered = {pidx}
    for nbr in phosphorus.GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(pidx, nidx)
        if bond is None:
            return ""
        if nbr.GetSymbol() == "N" and bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetFormalCharge() == 0:
            nitrogen_idx = nidx
            covered.add(nidx)
        elif (
            nbr.GetSymbol() == "O"
            and bond.GetBondType() == Chem.BondType.SINGLE
            and nbr.GetFormalCharge() == 0
            and nbr.GetTotalNumHs() == 1
        ):
            hydroxy += 1
            covered.add(nidx)
        else:
            return ""
    if nitrogen_idx is None or hydroxy != 2:
        return ""
    nitrogen = mol.GetAtomWithIdx(nitrogen_idx)
    owner_names = []
    for nbr in nitrogen.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == pidx:
            continue
        if nbr.GetSymbol() != "C":
            return ""
        owner, atoms = _simple_fragment_owner_name(mol, Chem, nidx, nitrogen_idx)
        if not owner:
            return ""
        owner_names.append(owner)
        covered.update(atoms)
    if len(owner_names) != 2 or owner_names[0] != owner_names[1] or covered != set(range(mol.GetNumAtoms())):
        return ""
    return f"phosphorous acid di{owner_names[0]}amide"


def _isothiocyanoamide_name(mol, Chem) -> str:
    if _element_counts(mol) != {"C": 1, "N": 2, "S": 1} or _total_charge(mol) != -1:
        return ""
    carbon = next(atom for atom in mol.GetAtoms() if atom.GetSymbol() == "C")
    sulfur = next(atom for atom in mol.GetAtoms() if atom.GetSymbol() == "S")
    nitrogens = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == "N"]
    anionic_n = [atom for atom in nitrogens if atom.GetFormalCharge() == -1]
    neutral_n = [atom for atom in nitrogens if atom.GetFormalCharge() == 0]
    if len(anionic_n) != 1 or len(neutral_n) != 1:
        return ""
    anionic_n = anionic_n[0]
    neutral_n = neutral_n[0]
    if not _specific_bond(mol, Chem, carbon.GetIdx(), sulfur.GetIdx(), Chem.BondType.DOUBLE):
        return ""
    if not _specific_bond(mol, Chem, carbon.GetIdx(), neutral_n.GetIdx(), Chem.BondType.DOUBLE):
        return ""
    if not _specific_bond(mol, Chem, neutral_n.GetIdx(), anionic_n.GetIdx(), Chem.BondType.SINGLE):
        return ""
    return "isothiocyanoamide" if anionic_n.GetTotalNumHs() == 1 else ""


def _dialkylhydrazinide_name(mol, Chem) -> str:
    if _total_charge(mol) != -1:
        return ""
    nitrogens = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == "N"]
    if len(nitrogens) != 2:
        return ""
    bond = mol.GetBondBetweenAtoms(nitrogens[0].GetIdx(), nitrogens[1].GetIdx())
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    owner_names = []
    covered = {nitrogens[0].GetIdx(), nitrogens[1].GetIdx()}
    if sorted(atom.GetFormalCharge() for atom in nitrogens) != [-1, 0]:
        return ""
    for nitrogen in nitrogens:
        nidx = nitrogen.GetIdx()
        carbon_neighbors = [nbr.GetIdx() for nbr in nitrogen.GetNeighbors() if nbr.GetSymbol() == "C"]
        expected_hydrogens = 0 if nitrogen.GetFormalCharge() == -1 else 1
        if len(carbon_neighbors) != 1 or nitrogen.GetTotalNumHs() != expected_hydrogens:
            return ""
        owner, atoms = _simple_fragment_owner_name(mol, Chem, carbon_neighbors[0], nidx)
        if not owner:
            return ""
        owner_names.append(owner)
        covered.update(atoms)
    if owner_names[0] != owner_names[1] or covered != set(range(mol.GetNumAtoms())):
        return ""
    return f"1,2-di{owner_names[0]}hydrazinide"


def _n_alkyl_nitrosoamide_name(mol, Chem) -> str:
    if _total_charge(mol) != -1:
        return ""
    nitrogens = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == "N"]
    if len(nitrogens) != 2:
        return ""
    anionic = [atom for atom in nitrogens if atom.GetFormalCharge() == -1]
    neutral = [atom for atom in nitrogens if atom.GetFormalCharge() == 0]
    if len(anionic) != 1 or len(neutral) != 1:
        return ""
    anionic_n = anionic[0]
    neutral_n = neutral[0]
    if not _specific_bond(mol, Chem, anionic_n.GetIdx(), neutral_n.GetIdx(), Chem.BondType.SINGLE):
        return ""
    oxygens = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == "O"]
    if len(oxygens) != 1 or not _specific_bond(mol, Chem, neutral_n.GetIdx(), oxygens[0].GetIdx(), Chem.BondType.DOUBLE):
        return ""
    carbon_neighbors = [nbr.GetIdx() for nbr in anionic_n.GetNeighbors() if nbr.GetSymbol() == "C"]
    if len(carbon_neighbors) != 1 or anionic_n.GetTotalNumHs() != 0:
        return ""
    owner, owner_atoms = _simple_fragment_owner_name(mol, Chem, carbon_neighbors[0], anionic_n.GetIdx())
    if not owner:
        return ""
    covered = {anionic_n.GetIdx(), neutral_n.GetIdx(), oxygens[0].GetIdx(), *owner_atoms}
    return f"N-{owner}nitrosoamide" if covered == set(range(mol.GetNumAtoms())) else ""


def _dialkyl_thiolate_name(mol, Chem) -> str:
    if _total_charge(mol) != -1:
        return ""
    sulfurs = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == "S" and atom.GetFormalCharge() == -1]
    if len(sulfurs) != 1 or sulfurs[0].GetTotalNumHs() != 1:
        return ""
    sulfur = sulfurs[0]
    owner_names = []
    covered = {sulfur.GetIdx()}
    for nbr in sulfur.GetNeighbors():
        if nbr.GetSymbol() != "C" or mol.GetBondBetweenAtoms(sulfur.GetIdx(), nbr.GetIdx()).GetBondType() != Chem.BondType.SINGLE:
            return ""
        owner, atoms = _simple_fragment_owner_name(mol, Chem, nbr.GetIdx(), sulfur.GetIdx())
        if not owner:
            return ""
        owner_names.append(owner)
        covered.update(atoms)
    if len(owner_names) != 2 or covered != set(range(mol.GetNumAtoms())):
        return ""
    parent_name = {
        "methyl": "methanethiolate",
        "ethyl": "ethanethiolate",
        "propyl": "propanethiolate",
        "butyl": "butanethiolate",
    }.get(owner_names[1], "")
    if not parent_name:
        return ""
    return f"S-{owner_names[0]}-{parent_name}"


def _alkyl_thiosulfoximine_name(mol, Chem) -> str:
    if _total_charge(mol) != 0:
        return ""
    sulfurs = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == "S"]
    if len(sulfurs) != 2:
        return ""
    for sulfur in sulfurs:
        if sulfur.GetFormalCharge() != 0 or sulfur.GetTotalNumHs() != 1:
            continue
        sidx = sulfur.GetIdx()
        carbon_idx = imino_idx = thioxo_idx = None
        covered = {sidx}
        for nbr in sulfur.GetNeighbors():
            nidx = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(sidx, nidx)
            if bond is None:
                return ""
            if nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE:
                carbon_idx = nidx
            elif nbr.GetSymbol() == "N" and bond.GetBondType() == Chem.BondType.DOUBLE and nbr.GetFormalCharge() == 0:
                imino_idx = nidx
                covered.add(nidx)
            elif nbr.GetSymbol() == "S" and bond.GetBondType() == Chem.BondType.DOUBLE and nbr.GetFormalCharge() == 0:
                thioxo_idx = nidx
                covered.add(nidx)
            else:
                return ""
        if carbon_idx is None or imino_idx is None or thioxo_idx is None:
            continue
        owner, owner_atoms = _simple_fragment_owner_name(mol, Chem, carbon_idx, sidx)
        if not owner:
            return ""
        covered.update(owner_atoms)
        if covered == set(range(mol.GetNumAtoms())):
            return f"S-{owner}-thiosulfoximine"
    return ""


def _single_phosphorus_atom(mol):
    phosphorus_atoms = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == "P" and not atom.IsInRing()]
    return phosphorus_atoms[0] if len(phosphorus_atoms) == 1 else None


def _simple_fragment_owner_name(mol, Chem, atom_idx: int, parent_idx: int) -> tuple[str, set[int]]:
    from .branch_rules import branch_prefix

    atom = mol.GetAtomWithIdx(atom_idx)
    if atom.GetSymbol() != "C" or atom.IsInRing():
        return "", set()
    name = branch_prefix(mol, atom_idx, parent_idx)
    name = _strip_fragment_parentheses(name)
    name = {
        "ethen-1-yl": "vinyl",
        "ethenyl": "vinyl",
        "propan-2-yl": "isopropyl",
        "2-propyl": "isopropyl",
    }.get(name, name)
    if name not in {
        "methyl",
        "ethyl",
        "propyl",
        "isopropyl",
        "butyl",
        "isobutyl",
        "t-butyl",
        "tert-butyl",
        "vinyl",
    }:
        return "", set()
    atoms = _fragment_component_atoms(mol, atom_idx, parent_idx)
    return name, atoms


def _linear_fragment_carbon_chain(mol, Chem, start_idx: int, parent_idx: int) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    chain = []
    amino_locants = []
    previous = parent_idx
    current = start_idx
    seen = set()
    while True:
        if current in seen:
            return (), ()
        seen.add(current)
        atom = mol.GetAtomWithIdx(current)
        if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing():
            return (), ()
        chain.append(current)
        next_carbons = []
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx == previous or nbr.GetAtomicNum() <= 1:
                continue
            bond = mol.GetBondBetweenAtoms(current, nidx)
            if nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE:
                next_carbons.append(nidx)
            elif _is_amino_substituent(mol, Chem, nidx, current):
                amino_locants.append((len(chain), nidx))
            else:
                return (), ()
        if not next_carbons:
            break
        if len(next_carbons) != 1:
            return (), ()
        previous, current = current, next_carbons[0]
    if len(chain) > 10:
        return (), ()
    return tuple(chain), tuple(amino_locants)


def _is_amino_substituent(mol, Chem, nitrogen_idx: int, carbon_idx: int) -> bool:
    nitrogen = mol.GetAtomWithIdx(nitrogen_idx)
    if nitrogen.GetSymbol() != "N" or nitrogen.GetFormalCharge() != 0 or nitrogen.IsInRing():
        return False
    if nitrogen.GetTotalNumHs() != 2:
        return False
    if not _specific_bond(mol, Chem, nitrogen_idx, carbon_idx, Chem.BondType.SINGLE):
        return False
    return all(nbr.GetIdx() == carbon_idx or nbr.GetAtomicNum() <= 1 for nbr in nitrogen.GetNeighbors())


def _fragment_component_atoms(mol, start_idx: int, parent_idx: int) -> set[int]:
    stack = [start_idx]
    seen = {parent_idx}
    atoms = set()
    while stack:
        idx = stack.pop()
        if idx in seen:
            continue
        seen.add(idx)
        atoms.add(idx)
        atom = mol.GetAtomWithIdx(idx)
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx not in seen and nbr.GetAtomicNum() > 1:
                stack.append(nidx)
    return atoms


def _fragment_locant_prefix_text(kind: str, locants: list[int]) -> str:
    if not locants:
        return ""
    counts: dict[int, int] = {}
    for locant in locants:
        counts[locant] = counts.get(locant, 0) + 1
    if any(count != 1 for count in counts.values()):
        return ""
    locant_text = ",".join(str(locant) for locant in sorted(counts))
    multiplier = {1: "", 2: "di", 3: "tri"}.get(len(counts), "")
    return f"{locant_text}-{multiplier}{kind}-"


def _alkoxy_fragment_name(owner: str) -> str:
    return {
        "methyl": "methoxy",
        "ethyl": "ethoxy",
        "propyl": "propoxy",
        "isopropyl": "isopropoxy",
        "butyl": "butoxy",
        "isobutyl": "isobutoxy",
        "t-butyl": "tert-butoxy",
        "tert-butyl": "tert-butoxy",
    }.get(owner, "")


def _strip_fragment_parentheses(name: str) -> str:
    if name.startswith("(") and name.endswith(")"):
        return name[1:-1]
    return name


def _oxidonitrogen_anion_name(mol, Chem) -> str:
    if _element_counts(mol) != {"N": 1, "O": 1} or _total_charge(mol) != -1:
        return ""
    nitrogen = next(atom for atom in mol.GetAtoms() if atom.GetSymbol() == "N")
    oxygen = next(atom for atom in mol.GetAtoms() if atom.GetSymbol() == "O")
    bond = mol.GetBondBetweenAtoms(nitrogen.GetIdx(), oxygen.GetIdx())
    if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
        return ""
    return "oxidonitrogen(1-)" if nitrogen.GetFormalCharge() == -1 and oxygen.GetFormalCharge() == 0 else ""


def _cyanonitrate_name(mol, Chem) -> str:
    if _element_counts(mol) != {"C": 1, "N": 2, "O": 2} or _total_charge(mol) != 0:
        return ""
    carbon = next(atom for atom in mol.GetAtoms() if atom.GetSymbol() == "C")
    nitrile_nitrogens = []
    nitrate_nitrogens = []
    for nbr in carbon.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(carbon.GetIdx(), nbr.GetIdx())
        if nbr.GetSymbol() != "N":
            return ""
        if bond.GetBondType() == Chem.BondType.TRIPLE and nbr.GetFormalCharge() == 0:
            nitrile_nitrogens.append(nbr.GetIdx())
        elif bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetFormalCharge() == 1:
            nitrate_nitrogens.append(nbr.GetIdx())
        else:
            return ""
    if len(nitrile_nitrogens) != 1 or len(nitrate_nitrogens) != 1:
        return ""
    nitrate_n = mol.GetAtomWithIdx(nitrate_nitrogens[0])
    double_o = 0
    oxido_o = 0
    for nbr in nitrate_n.GetNeighbors():
        if nbr.GetIdx() == carbon.GetIdx():
            continue
        bond = mol.GetBondBetweenAtoms(nitrate_n.GetIdx(), nbr.GetIdx())
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE and nbr.GetFormalCharge() == 0:
            double_o += 1
        elif nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetFormalCharge() == -1:
            oxido_o += 1
        else:
            return ""
    return "cyanonitrate" if double_o == 1 and oxido_o == 1 else ""


def _cyanoiodate_name(mol, Chem) -> str:
    if _element_counts(mol) != {"C": 1, "I": 1, "N": 1, "O": 2} or _total_charge(mol) != 0:
        return ""
    carbon = next(atom for atom in mol.GetAtoms() if atom.GetSymbol() == "C")
    iodine = next(atom for atom in mol.GetAtoms() if atom.GetSymbol() == "I")
    nitrogen = next(atom for atom in mol.GetAtoms() if atom.GetSymbol() == "N")
    if carbon.GetFormalCharge() != 0 or iodine.GetFormalCharge() != 0 or nitrogen.GetFormalCharge() != 0:
        return ""
    if not _specific_bond(mol, Chem, carbon.GetIdx(), nitrogen.GetIdx(), Chem.BondType.TRIPLE):
        return ""
    if not _specific_bond(mol, Chem, carbon.GetIdx(), iodine.GetIdx(), Chem.BondType.SINGLE):
        return ""
    oxo_count = sum(
        1
        for oxygen in mol.GetAtoms()
        if oxygen.GetSymbol() == "O"
        and oxygen.GetFormalCharge() == 0
        and _specific_bond(mol, Chem, iodine.GetIdx(), oxygen.GetIdx(), Chem.BondType.DOUBLE)
    )
    return "cyanoiodate" if oxo_count == 2 else ""


def _hypothiocyanite_name(mol, Chem) -> str:
    if _element_counts(mol) != {"C": 1, "N": 1, "O": 1, "S": 1} or _total_charge(mol) != -1:
        return ""
    carbon = next(atom for atom in mol.GetAtoms() if atom.GetSymbol() == "C")
    sulfur = next(atom for atom in mol.GetAtoms() if atom.GetSymbol() == "S")
    oxygen = next(atom for atom in mol.GetAtoms() if atom.GetSymbol() == "O")
    nitrogen = next(atom for atom in mol.GetAtoms() if atom.GetSymbol() == "N")
    if carbon.GetFormalCharge() != 0 or sulfur.GetFormalCharge() != 0 or nitrogen.GetFormalCharge() != 0:
        return ""
    if oxygen.GetFormalCharge() != -1:
        return ""
    if not _specific_bond(mol, Chem, carbon.GetIdx(), nitrogen.GetIdx(), Chem.BondType.TRIPLE):
        return ""
    if not _specific_bond(mol, Chem, carbon.GetIdx(), sulfur.GetIdx(), Chem.BondType.SINGLE):
        return ""
    if not _specific_bond(mol, Chem, sulfur.GetIdx(), oxygen.GetIdx(), Chem.BondType.SINGLE):
        return ""
    return "hypothiocyanite"


def _alkanesulfenate_name(mol, Chem) -> str:
    counts = _element_counts(mol)
    if counts.get("S") != 1 or counts.get("O") != 1 or _total_charge(mol) != -1:
        return ""
    if set(counts) != {"C", "O", "S"}:
        return ""
    sulfur = next(atom for atom in mol.GetAtoms() if atom.GetSymbol() == "S")
    oxygen = next(atom for atom in mol.GetAtoms() if atom.GetSymbol() == "O")
    if sulfur.GetFormalCharge() != 0 or oxygen.GetFormalCharge() != -1:
        return ""
    if not _specific_bond(mol, Chem, sulfur.GetIdx(), oxygen.GetIdx(), Chem.BondType.SINGLE):
        return ""
    carbon_neighbors = [
        nbr.GetIdx()
        for nbr in sulfur.GetNeighbors()
        if nbr.GetSymbol() == "C" and mol.GetBondBetweenAtoms(sulfur.GetIdx(), nbr.GetIdx()).GetBondType() == Chem.BondType.SINGLE
    ]
    if len(carbon_neighbors) != 1:
        return ""
    chain_len = _terminal_linear_alkane_chain_length(mol, Chem, carbon_neighbors[0], sulfur.GetIdx())
    root = ROOTS.get(chain_len, "")
    return f"{root}anesulfenate" if root else ""


def _terminal_linear_alkane_chain_length(mol, Chem, start_idx: int, sulfur_idx: int) -> int:
    carbon_atoms = {atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == "C"}
    if not carbon_atoms:
        return 0
    path = []
    previous = sulfur_idx
    current = start_idx
    seen = set()
    while True:
        if current in seen:
            return 0
        seen.add(current)
        atom = mol.GetAtomWithIdx(current)
        if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing():
            return 0
        path.append(current)
        next_carbons = []
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx == previous or nbr.GetAtomicNum() <= 1:
                continue
            bond = mol.GetBondBetweenAtoms(current, nidx)
            if nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE:
                next_carbons.append(nidx)
            else:
                return 0
        if not next_carbons:
            break
        if len(next_carbons) != 1:
            return 0
        previous, current = current, next_carbons[0]
    return len(path) if set(path) == carbon_atoms else 0


def _phosphacycloalkanolate_name(mol, Chem) -> str:
    counts = _element_counts(mol)
    if counts.get("P") != 1 or counts.get("O") != 1 or _total_charge(mol) != -1:
        return ""
    if set(counts) != {"C", "O", "P"}:
        return ""
    phosphorus = next(atom for atom in mol.GetAtoms() if atom.GetSymbol() == "P")
    oxygen = next(atom for atom in mol.GetAtoms() if atom.GetSymbol() == "O")
    if phosphorus.GetFormalCharge() != 0 or oxygen.GetFormalCharge() != -1:
        return ""
    if not _specific_bond(mol, Chem, phosphorus.GetIdx(), oxygen.GetIdx(), Chem.BondType.SINGLE):
        return ""
    rings = [ring for ring in mol.GetRingInfo().AtomRings() if phosphorus.GetIdx() in ring]
    if len(rings) != 1:
        return ""
    ring = tuple(rings[0])
    if set(ring) | {oxygen.GetIdx()} != set(range(mol.GetNumAtoms())):
        return ""
    if any(mol.GetAtomWithIdx(idx).GetSymbol() not in {"C", "P"} for idx in ring):
        return ""
    if any(mol.GetAtomWithIdx(idx).GetFormalCharge() != 0 for idx in ring):
        return ""
    names = {5: "phospholanolate", 6: "phosphinanolate"}
    return names.get(len(ring), "")


def _diazenide_name(mol, Chem) -> str:
    if _element_counts(mol) != {"N": 2} or _total_charge(mol) != -1:
        return ""
    atoms = list(mol.GetAtoms())
    bond = mol.GetBondBetweenAtoms(atoms[0].GetIdx(), atoms[1].GetIdx())
    return "diazenide" if bond is not None and bond.GetBondType() == Chem.BondType.DOUBLE else ""


def _diselenide_name(mol, Chem) -> str:
    if _element_counts(mol) != {"Se": 2} or _total_charge(mol) != -1:
        return ""
    atoms = list(mol.GetAtoms())
    bond = mol.GetBondBetweenAtoms(atoms[0].GetIdx(), atoms[1].GetIdx())
    if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
        return ""
    return "diselenide" if sum(atom.GetTotalNumHs() for atom in atoms) == 1 else ""


def _halogen_oxide_name(mol, Chem) -> str:
    counts = _element_counts(mol)
    halogens = [symbol for symbol in HALIDE_STEMS if counts.get(symbol) == 1]
    if len(halogens) != 1 or counts.get("O") != 1 or sum(counts.values()) != 2:
        return ""
    if _total_charge(mol) != 0:
        return ""
    halogen = next(atom for atom in mol.GetAtoms() if atom.GetSymbol() == halogens[0])
    oxygen = next(atom for atom in mol.GetAtoms() if atom.GetSymbol() == "O")
    bond = mol.GetBondBetweenAtoms(halogen.GetIdx(), oxygen.GetIdx())
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    if oxygen.GetFormalCharge() != -1 or halogen.GetFormalCharge() != 1 or halogen.GetTotalNumHs() != 1:
        return ""
    if halogen.GetSymbol() not in {"Cl", "Br", "I"}:
        return ""
    return HALIDE_STEMS[halogen.GetSymbol()] + "ane oxide"


def _interhalogen_name(mol, Chem) -> str:
    if mol.GetNumAtoms() != 2 or _total_charge(mol) != 0:
        return ""
    atoms = list(mol.GetAtoms())
    if any(atom.GetSymbol() not in HALIDE_STEMS or atom.GetTotalNumHs() for atom in atoms):
        return ""
    bond = mol.GetBondBetweenAtoms(atoms[0].GetIdx(), atoms[1].GetIdx())
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    center, ligand = sorted(atoms, key=lambda atom: HALOGEN_ELECTROPOSITIVE_ORDER[atom.GetSymbol()], reverse=True)
    return f"{HALOGEN_ELEMENT_NAMES[center.GetSymbol()]} {HALIDE_ANIONS[ligand.GetSymbol()]}"


def _haloazanide_name(mol, Chem) -> str:
    atoms = list(mol.GetAtoms())
    nitrogens = [atom for atom in atoms if atom.GetSymbol() == "N"]
    if len(nitrogens) != 1 or _total_charge(mol) != -1:
        return ""
    nitrogen = nitrogens[0]
    if nitrogen.GetFormalCharge() != -1:
        return ""
    halogens = [atom for atom in atoms if atom.GetSymbol() in HALIDE_STEMS]
    if len(halogens) not in {1, 2} or len(halogens) + 1 != len(atoms):
        return ""
    if nitrogen.GetTotalNumHs() + len(halogens) != 2:
        return ""
    prefixes = []
    for halogen in halogens:
        bond = mol.GetBondBetweenAtoms(nitrogen.GetIdx(), halogen.GetIdx())
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE or halogen.GetTotalNumHs():
            return ""
        prefixes.append(HALIDE_STEMS[halogen.GetSymbol()] + "o")
    counts: dict[str, int] = {}
    for prefix in prefixes:
        counts[prefix] = counts.get(prefix, 0) + 1
    terms = []
    for prefix, count in sorted(counts.items()):
        if count == 1:
            terms.append(prefix)
        elif count == 2:
            terms.append("di" + prefix)
        else:
            return ""
    return "".join(terms) + "azanide"


def _acetonide_name(mol, Chem) -> str:
    if _element_counts(mol) != {"C": 3, "O": 1} or _total_charge(mol) != -1:
        return ""
    carbonyl_carbons = [
        atom
        for atom in mol.GetAtoms()
        if atom.GetSymbol() == "C"
        and any(
            nbr.GetSymbol() == "O"
            and mol.GetBondBetweenAtoms(atom.GetIdx(), nbr.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
            for nbr in atom.GetNeighbors()
        )
    ]
    if len(carbonyl_carbons) != 1:
        return ""
    carbonyl = carbonyl_carbons[0]
    methyl = anionic_methylene = 0
    for nbr in carbonyl.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(carbonyl.GetIdx(), nbr.GetIdx())
        if nbr.GetSymbol() != "C" or bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        if nbr.GetFormalCharge() == 0 and nbr.GetTotalNumHs() == 3:
            methyl += 1
        elif nbr.GetFormalCharge() == -1 and nbr.GetTotalNumHs() == 2:
            anionic_methylene += 1
    return "acetonide" if methyl == 1 and anionic_methylene == 1 else ""


def _nitronite_name(mol, Chem) -> str:
    if _element_counts(mol) != {"C": 1, "N": 1, "O": 1} or _total_charge(mol) != -1:
        return ""
    carbon = next(atom for atom in mol.GetAtoms() if atom.GetSymbol() == "C")
    nitrogen = next(atom for atom in mol.GetAtoms() if atom.GetSymbol() == "N")
    oxygen = next(atom for atom in mol.GetAtoms() if atom.GetSymbol() == "O")
    cn_bond = mol.GetBondBetweenAtoms(carbon.GetIdx(), nitrogen.GetIdx())
    no_bond = mol.GetBondBetweenAtoms(nitrogen.GetIdx(), oxygen.GetIdx())
    if cn_bond is None or no_bond is None:
        return ""
    if cn_bond.GetBondType() != Chem.BondType.DOUBLE or no_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    return "nitronite" if oxygen.GetFormalCharge() == -1 else ""


def _peroxycarbonate_name(mol, Chem) -> str:
    if _element_counts(mol) != {"C": 1, "O": 4} or _total_charge(mol) != -2:
        return ""
    carbon = next(atom for atom in mol.GetAtoms() if atom.GetSymbol() == "C")
    double_oxygen = terminal_oxido = peroxy_oxido = 0
    for nbr in carbon.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(carbon.GetIdx(), nbr.GetIdx())
        if nbr.GetSymbol() != "O" or bond is None:
            return ""
        if bond.GetBondType() == Chem.BondType.DOUBLE and nbr.GetFormalCharge() == 0:
            double_oxygen += 1
            continue
        if bond.GetBondType() != Chem.BondType.SINGLE:
            return ""
        heavy = [other for other in nbr.GetNeighbors() if other.GetIdx() != carbon.GetIdx() and other.GetAtomicNum() > 1]
        if not heavy and nbr.GetFormalCharge() < 0:
            terminal_oxido += 1
            continue
        if len(heavy) == 1 and heavy[0].GetSymbol() == "O" and heavy[0].GetFormalCharge() < 0:
            peroxy_bond = mol.GetBondBetweenAtoms(nbr.GetIdx(), heavy[0].GetIdx())
            if peroxy_bond is None or peroxy_bond.GetBondType() != Chem.BondType.SINGLE:
                return ""
            peroxy_oxido += 1
            continue
        return ""
    return "peroxycarbonate" if double_oxygen == 1 and terminal_oxido == 1 and peroxy_oxido == 1 else ""


def _peroxyoxalate_name(mol, Chem) -> str:
    if _element_counts(mol) != {"C": 2, "O": 5} or _total_charge(mol) != -2:
        return ""
    carbons = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == "C"]
    cc_bond = mol.GetBondBetweenAtoms(carbons[0].GetIdx(), carbons[1].GetIdx())
    if cc_bond is None or cc_bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    has_carboxylate = False
    has_peroxycarbonyl = False
    for carbon in carbons:
        double_oxygen = terminal_oxido = peroxy_oxido = 0
        for nbr in carbon.GetNeighbors():
            if nbr.GetSymbol() == "C":
                continue
            bond = mol.GetBondBetweenAtoms(carbon.GetIdx(), nbr.GetIdx())
            if nbr.GetSymbol() != "O" or bond is None:
                return ""
            if bond.GetBondType() == Chem.BondType.DOUBLE and nbr.GetFormalCharge() == 0:
                double_oxygen += 1
                continue
            if bond.GetBondType() != Chem.BondType.SINGLE:
                return ""
            heavy = [other for other in nbr.GetNeighbors() if other.GetIdx() != carbon.GetIdx() and other.GetAtomicNum() > 1]
            if not heavy and nbr.GetFormalCharge() < 0:
                terminal_oxido += 1
                continue
            if len(heavy) == 1 and heavy[0].GetSymbol() == "O" and heavy[0].GetFormalCharge() < 0:
                peroxy_bond = mol.GetBondBetweenAtoms(nbr.GetIdx(), heavy[0].GetIdx())
                if peroxy_bond is None or peroxy_bond.GetBondType() != Chem.BondType.SINGLE:
                    return ""
                peroxy_oxido += 1
                continue
            return ""
        has_carboxylate = has_carboxylate or (double_oxygen == 1 and terminal_oxido == 1 and peroxy_oxido == 0)
        has_peroxycarbonyl = has_peroxycarbonyl or (double_oxygen == 1 and terminal_oxido == 0 and peroxy_oxido == 1)
    return "peroxyoxalate" if has_carboxylate and has_peroxycarbonyl else ""


def _nitronic_acid_or_ester_name(mol, Chem) -> str:
    if _total_charge(mol) != 0:
        return ""
    nitrogens = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == "N" and atom.GetFormalCharge() > 0]
    if len(nitrogens) != 1:
        return ""
    nitrogen = nitrogens[0]
    carbon = None
    oxido = None
    neutral_oxygen = None
    for nbr in nitrogen.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(nitrogen.GetIdx(), nbr.GetIdx())
        if bond is None:
            return ""
        if nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.DOUBLE:
            carbon = nbr
        elif nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetFormalCharge() < 0:
            oxido = nbr
        elif nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetFormalCharge() == 0:
            neutral_oxygen = nbr
        else:
            return ""
    if carbon is None or oxido is None or neutral_oxygen is None:
        return ""
    carbon_heavy = [
        nbr for nbr in carbon.GetNeighbors() if nbr.GetIdx() != nitrogen.GetIdx() and nbr.GetAtomicNum() > 1
    ]
    oxygen_heavy = [
        nbr for nbr in neutral_oxygen.GetNeighbors() if nbr.GetIdx() != nitrogen.GetIdx() and nbr.GetAtomicNum() > 1
    ]
    if not carbon_heavy and not oxygen_heavy and neutral_oxygen.GetTotalNumHs() == 1:
        return "nitronic acid"
    if len(carbon_heavy) == 1 and not oxygen_heavy and neutral_oxygen.GetTotalNumHs() == 1:
        vinyl_bond = mol.GetBondBetweenAtoms(carbon.GetIdx(), carbon_heavy[0].GetIdx())
        if (
            carbon_heavy[0].GetSymbol() == "C"
            and vinyl_bond is not None
            and vinyl_bond.GetBondType() == Chem.BondType.DOUBLE
            and carbon_heavy[0].GetTotalNumHs() == 2
        ):
            return "aci-nitroethene"
    if not carbon_heavy and len(oxygen_heavy) == 1:
        if _is_terminal_ethoxy_owner(mol, Chem, oxygen_heavy[0].GetIdx(), neutral_oxygen.GetIdx()):
            return "ethyl nitronate"
    return ""


def _is_terminal_ethoxy_owner(mol, Chem, carbon_idx: int, oxygen_idx: int) -> bool:
    first = mol.GetAtomWithIdx(carbon_idx)
    if first.GetSymbol() != "C" or first.GetFormalCharge() != 0 or first.IsInRing():
        return False
    bond_to_oxygen = mol.GetBondBetweenAtoms(carbon_idx, oxygen_idx)
    if bond_to_oxygen is None or bond_to_oxygen.GetBondType() != Chem.BondType.SINGLE:
        return False
    carbon_neighbors = [
        nbr for nbr in first.GetNeighbors() if nbr.GetIdx() != oxygen_idx and nbr.GetAtomicNum() > 1
    ]
    if len(carbon_neighbors) != 1:
        return False
    terminal = carbon_neighbors[0]
    bond = mol.GetBondBetweenAtoms(first.GetIdx(), terminal.GetIdx())
    if terminal.GetSymbol() != "C" or terminal.GetFormalCharge() != 0 or bond is None:
        return False
    if bond.GetBondType() != Chem.BondType.SINGLE:
        return False
    return not [nbr for nbr in terminal.GetNeighbors() if nbr.GetIdx() != first.GetIdx() and nbr.GetAtomicNum() > 1]


def _azanol_name(mol, Chem) -> str:
    if _element_counts(mol) != {"N": 1, "O": 1} or _total_charge(mol) != 0:
        return ""
    nitrogen = next((atom for atom in mol.GetAtoms() if atom.GetSymbol() == "N"), None)
    oxygen = next((atom for atom in mol.GetAtoms() if atom.GetSymbol() == "O"), None)
    if nitrogen is None or oxygen is None:
        return ""
    bond = mol.GetBondBetweenAtoms(nitrogen.GetIdx(), oxygen.GetIdx())
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return ""
    if nitrogen.GetNumRadicalElectrons() or oxygen.GetNumRadicalElectrons():
        return ""
    return "azanol" if nitrogen.GetTotalNumHs() == 2 and oxygen.GetTotalNumHs() == 1 else ""


def _isodiazene_name(mol, Chem) -> str:
    if _element_counts(mol) != {"N": 2} or _total_charge(mol) != 0:
        return ""
    nitrogens = list(mol.GetAtoms())
    bond = mol.GetBondBetweenAtoms(nitrogens[0].GetIdx(), nitrogens[1].GetIdx())
    if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
        return ""
    return (
        "isodiazene"
        if sorted((atom.GetFormalCharge(), atom.GetTotalNumHs()) for atom in nitrogens) == [(-1, 0), (1, 2)]
        else ""
    )


def _peracetate_name(mol, Chem) -> str:
    if _element_counts(mol) != {"C": 2, "O": 3} or _total_charge(mol) != -1:
        return ""
    carbonyl_carbons = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetSymbol() == "C"
        and any(
            nbr.GetSymbol() == "O"
            and mol.GetBondBetweenAtoms(atom.GetIdx(), nbr.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
            for nbr in atom.GetNeighbors()
        )
    ]
    if len(carbonyl_carbons) != 1:
        return ""
    carbonyl = mol.GetAtomWithIdx(carbonyl_carbons[0])
    has_methyl = False
    has_anionic_peroxy = False
    for nbr in carbonyl.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(carbonyl.GetIdx(), nbr.GetIdx())
        if nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetDegree() == 1:
            has_methyl = True
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.SINGLE:
            terminal_oxygen = [
                other
                for other in nbr.GetNeighbors()
                if other.GetIdx() != carbonyl.GetIdx()
                and other.GetSymbol() == "O"
                and mol.GetBondBetweenAtoms(nbr.GetIdx(), other.GetIdx()).GetBondType() == Chem.BondType.SINGLE
                and other.GetFormalCharge() == -1
            ]
            has_anionic_peroxy = len(terminal_oxygen) == 1
    return "peracetate" if has_methyl and has_anionic_peroxy else ""


def _isothiouronium_name(mol, Chem) -> str:
    return _chalcogenouronium_name(mol, Chem, "S", "isothiouronium")


def _isouronium_name(mol, Chem) -> str:
    return _chalcogenouronium_name(mol, Chem, "O", "isouronium")


def _thiauronium_name(mol, Chem) -> str:
    if _element_counts(mol) != {"N": 2, "O": 1, "S": 1} or _total_charge(mol) != 1:
        return ""
    sulfur = next((atom for atom in mol.GetAtoms() if atom.GetSymbol() == "S"), None)
    if sulfur is None:
        return ""
    single_n = double_cation_n = single_o = 0
    for nbr in sulfur.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(sulfur.GetIdx(), nbr.GetIdx())
        if bond is None:
            return ""
        if nbr.GetSymbol() == "N" and bond.GetBondType() == Chem.BondType.SINGLE:
            single_n += 1
        elif nbr.GetSymbol() == "N" and bond.GetBondType() == Chem.BondType.DOUBLE and nbr.GetFormalCharge() > 0:
            double_cation_n += 1
        elif nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.SINGLE:
            single_o += 1
        else:
            return ""
    return "thiauronium" if single_n == 1 and double_cation_n == 1 and single_o == 1 else ""


def _chalcogenouronium_name(mol, Chem, chalcogen: str, name: str) -> str:
    if _element_counts(mol) != {"C": 1, "N": 2, chalcogen: 1} or _total_charge(mol) != 1:
        return ""
    carbon = next((atom for atom in mol.GetAtoms() if atom.GetSymbol() == "C"), None)
    if carbon is None:
        return ""
    single_n = double_n = single_chalcogen = 0
    for nbr in carbon.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(carbon.GetIdx(), nbr.GetIdx())
        if nbr.GetSymbol() == "N" and bond.GetBondType() == Chem.BondType.SINGLE:
            single_n += 1
        elif nbr.GetSymbol() == "N" and bond.GetBondType() == Chem.BondType.DOUBLE and nbr.GetFormalCharge() > 0:
            double_n += 1
        elif nbr.GetSymbol() == chalcogen and bond.GetBondType() == Chem.BondType.SINGLE:
            single_chalcogen += 1
        else:
            return ""
    return name if single_n == 1 and double_n == 1 and single_chalcogen == 1 else ""


def _selenocysteamine_name(mol, Chem) -> str:
    if _element_counts(mol) != {"C": 2, "N": 1, "Se": 1} or _total_charge(mol) != 0:
        return ""
    selenium = next((atom for atom in mol.GetAtoms() if atom.GetSymbol() == "Se"), None)
    nitrogen = next((atom for atom in mol.GetAtoms() if atom.GetSymbol() == "N"), None)
    if selenium is None or nitrogen is None or selenium.GetTotalNumHs() != 1:
        return ""
    carbon_graph = {
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetSymbol() == "C"
    }
    if len(carbon_graph) != 2:
        return ""
    carbon_neighbors = {
        atom.GetIdx(): {
            nbr.GetIdx()
            for nbr in atom.GetNeighbors()
            if nbr.GetSymbol() in {"C", "N", "Se"}
            and mol.GetBondBetweenAtoms(atom.GetIdx(), nbr.GetIdx()).GetBondType() == Chem.BondType.SINGLE
        }
        for atom in mol.GetAtoms()
        if atom.GetSymbol() == "C"
    }
    nidx = nitrogen.GetIdx()
    seidx = selenium.GetIdx()
    return (
        "selenocysteamine"
        if any(nidx in nbrs for nbrs in carbon_neighbors.values())
        and any(seidx in nbrs for nbrs in carbon_neighbors.values())
        and any(len(nbrs & carbon_graph) == 1 for nbrs in carbon_neighbors.values())
        else ""
    )


def _hydride_anion_name(atom) -> str:
    if atom.GetSymbol() != "H" or atom.GetFormalCharge() >= 0:
        return ""
    if atom.GetIsotope() == 2:
        return "deuteride"
    if atom.GetIsotope() == 3:
        return "tritide"
    return "hydride"


def _substituted_borohydride_name(mol, Chem) -> str:
    borons = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == "B" and atom.GetFormalCharge() == -1]
    if len(borons) != 1:
        return ""
    boron = borons[0]
    bidx = boron.GetIdx()
    if any(
        mol.GetBondBetweenAtoms(bidx, nbr.GetIdx()).GetBondType() != Chem.BondType.SINGLE
        for nbr in boron.GetNeighbors()
    ):
        return ""
    hydrogens = boron.GetTotalNumHs()
    neighbors = [nbr.GetIdx() for nbr in boron.GetNeighbors()]
    if hydrogens == 3 and len(neighbors) == 1 and _is_cyano_group(mol, Chem, neighbors[0], bidx):
        return "cyanoborohydride"
    if hydrogens == 1 and len(neighbors) == 3 and all(_is_acetoxy_group(mol, Chem, nidx, bidx) for nidx in neighbors):
        return "triacetoxyborohydride"
    return ""


def _is_cyano_group(mol, Chem, carbon_idx: int, boron_idx: int) -> bool:
    carbon = mol.GetAtomWithIdx(carbon_idx)
    if carbon.GetSymbol() != "C":
        return False
    nitrogen_neighbors = [
        nbr
        for nbr in carbon.GetNeighbors()
        if nbr.GetIdx() != boron_idx
        and nbr.GetSymbol() == "N"
        and mol.GetBondBetweenAtoms(carbon_idx, nbr.GetIdx()).GetBondType() == Chem.BondType.TRIPLE
    ]
    return len(nitrogen_neighbors) == 1 and carbon.GetDegree() == 2


def _is_acetoxy_group(mol, Chem, oxygen_idx: int, boron_idx: int) -> bool:
    oxygen = mol.GetAtomWithIdx(oxygen_idx)
    if oxygen.GetSymbol() != "O":
        return False
    carbon_neighbors = [
        nbr.GetIdx()
        for nbr in oxygen.GetNeighbors()
        if nbr.GetIdx() != boron_idx
        and nbr.GetSymbol() == "C"
        and mol.GetBondBetweenAtoms(oxygen_idx, nbr.GetIdx()).GetBondType() == Chem.BondType.SINGLE
    ]
    if len(carbon_neighbors) != 1:
        return False
    carbon_idx = carbon_neighbors[0]
    carbon = mol.GetAtomWithIdx(carbon_idx)
    double_oxygen_count = 0
    methyl_count = 0
    for nbr in carbon.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == oxygen_idx:
            continue
        bond = mol.GetBondBetweenAtoms(carbon_idx, nidx)
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_oxygen_count += 1
        elif nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetTotalNumHs() == 3:
            methyl_count += 1
        else:
            return False
    return double_oxygen_count == 1 and methyl_count == 1


def _organotrifluoroborate_name(mol, Chem) -> str:
    borons = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == "B" and atom.GetFormalCharge() == -1]
    if len(borons) != 1 or _total_charge(mol) != -1:
        return ""
    boron = borons[0]
    bidx = boron.GetIdx()
    carbon_neighbors = []
    fluorine_count = 0
    for nbr in boron.GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(bidx, nidx)
        if bond.GetBondType() != Chem.BondType.SINGLE:
            return ""
        if nbr.GetSymbol() == "F":
            fluorine_count += 1
        elif nbr.GetSymbol() == "C":
            carbon_neighbors.append(nidx)
        else:
            return ""
    if fluorine_count != 3 or len(carbon_neighbors) != 1:
        return ""
    from .branch_rules import branch_prefix

    prefix = branch_prefix(mol, carbon_neighbors[0], bidx)
    if not prefix:
        return ""
    if _needs_fragment_parentheses(prefix):
        prefix = f"({prefix})"
    return prefix + "trifluoroborate"


def _tetraorganoborate_name(mol, Chem) -> str:
    borons = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == "B" and atom.GetFormalCharge() == -1]
    if len(borons) != 1 or _total_charge(mol) != -1:
        return ""
    boron = borons[0]
    bidx = boron.GetIdx()
    names = []
    for nbr in boron.GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(bidx, nidx)
        if nbr.GetSymbol() != "C" or bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return ""
        from .branch_rules import branch_prefix

        prefix = branch_prefix(mol, nidx, bidx)
        if not prefix:
            return ""
        names.append(prefix)
    if len(names) != 4:
        return ""
    prefix_text = _substituted_borate_prefix_text(names)
    return prefix_text + "borate" if prefix_text else ""


def _substituted_borate_prefix_text(names: list[str]) -> str:
    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    terms = []
    for name, count in counts.items():
        is_complex = _needs_fragment_parentheses(name)
        if count > 1 and is_complex:
            multiplier = COMPLEX_MULTIPLIERS.get(count, "")
            term = f"{multiplier}({name})" if multiplier else ""
        elif count > 1:
            multiplier = MULTIPLIERS.get(count, "")
            term = multiplier + name if multiplier else ""
        elif is_complex:
            term = f"({name})"
        else:
            term = name
        if not term:
            return ""
        terms.append((0 if is_complex else 1, name, term))
    return "".join(term for _complex_order, _name, term in sorted(terms))


def _azanide_name(mol, Chem) -> str:
    nitrogens = [
        atom
        for atom in mol.GetAtoms()
        if atom.GetSymbol() == "N"
        and atom.GetFormalCharge() == -1
        and not atom.IsInRing()
    ]
    if len(nitrogens) != 1 or _total_charge(mol) != -1:
        return ""
    nitrogen = nitrogens[0]
    nidx = nitrogen.GetIdx()
    neighbors = [nbr.GetIdx() for nbr in nitrogen.GetNeighbors()]
    implicit_hydrogens = nitrogen.GetTotalNumHs()
    if len(neighbors) + implicit_hydrogens != 2 or len(neighbors) > 2:
        return ""
    carbamate_name = _carbamate_azanide_fragment_name(mol, Chem, nidx, neighbors)
    if carbamate_name:
        return carbamate_name
    acyl_name = _acyl_azanide_fragment_name(mol, Chem, nidx, neighbors)
    if acyl_name:
        return acyl_name
    names = []
    for nbr_idx in neighbors:
        bond = mol.GetBondBetweenAtoms(nidx, nbr_idx)
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return ""
        prefix = _azanide_substituent_prefix(mol, Chem, nbr_idx, nidx)
        if not prefix:
            return ""
        names.append(prefix)
    if not names:
        return "azanide"
    from .parent.rule_helpers import _substituted_onium_name

    return _substituted_onium_name(names, "azanide")


def _carbamate_azanide_fragment_name(mol, Chem, nitrogen_idx: int, neighbors: list[int]) -> str:
    carbonyl_records: list[tuple[str, set[int]]] = []
    substituent_records: list[tuple[str, set[int]]] = []
    covered = {nitrogen_idx}
    for nbr_idx in neighbors:
        carbonyl_record = _carbamate_azanide_carbonyl_record(mol, Chem, nitrogen_idx, nbr_idx)
        if carbonyl_record:
            carbonyl_records.append(carbonyl_record)
            covered.update(carbonyl_record[1])
            continue
        substituent, substituent_atoms = _simple_fragment_owner_name(mol, Chem, nbr_idx, nitrogen_idx)
        if not substituent:
            return ""
        substituent_records.append((substituent, substituent_atoms))
        covered.update(substituent_atoms)
    if len(carbonyl_records) != 1 or len(substituent_records) > 1:
        return ""
    if covered != set(range(mol.GetNumAtoms())):
        return ""
    owner = carbonyl_records[0][0]
    if not substituent_records:
        return f"{owner} azanidecarboxylate"
    substituent = substituent_records[0][0]
    return f"{owner} {substituent}azanidecarboxylate"


def _carbamate_azanide_carbonyl_record(mol, Chem, nitrogen_idx: int, carbonyl_idx: int) -> tuple[str, set[int]] | None:
    carbonyl = mol.GetAtomWithIdx(carbonyl_idx)
    if carbonyl.GetSymbol() != "C" or carbonyl.GetFormalCharge() != 0 or carbonyl.IsInRing():
        return None
    bond_to_nitrogen = mol.GetBondBetweenAtoms(carbonyl_idx, nitrogen_idx)
    if bond_to_nitrogen is None or bond_to_nitrogen.GetBondType() != Chem.BondType.SINGLE:
        return None
    double_oxygens = []
    single_oxygens = []
    for nbr in carbonyl.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == nitrogen_idx:
            continue
        bond = mol.GetBondBetweenAtoms(carbonyl_idx, nidx)
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_oxygens.append(nidx)
        elif nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.SINGLE:
            single_oxygens.append(nidx)
        elif nbr.GetAtomicNum() > 1:
            return None
    if len(double_oxygens) != 1 or len(single_oxygens) != 1:
        return None
    owner, owner_atoms = _carbamate_azanide_ester_owner_record(mol, Chem, single_oxygens[0], carbonyl_idx)
    if not owner:
        return None
    atoms = {carbonyl_idx, double_oxygens[0], single_oxygens[0], *owner_atoms}
    return owner, atoms


def _carbamate_azanide_ester_owner_record(mol, Chem, oxygen_idx: int, carbonyl_idx: int) -> tuple[str, set[int]]:
    oxygen = mol.GetAtomWithIdx(oxygen_idx)
    if oxygen.GetSymbol() != "O" or oxygen.GetFormalCharge() != 0 or oxygen.IsInRing():
        return "", set()
    bond_to_carbonyl = mol.GetBondBetweenAtoms(oxygen_idx, carbonyl_idx)
    if bond_to_carbonyl is None or bond_to_carbonyl.GetBondType() != Chem.BondType.SINGLE:
        return "", set()
    heavy = [
        nbr.GetIdx()
        for nbr in oxygen.GetNeighbors()
        if nbr.GetIdx() != carbonyl_idx and nbr.GetAtomicNum() > 1
    ]
    if len(heavy) != 1:
        return "", set()
    from .branch_rules import branch_prefix

    owner = _strip_fragment_parentheses(branch_prefix(mol, heavy[0], oxygen_idx))
    owner = {
        "2-propyl": "isopropyl",
        "propan-2-yl": "isopropyl",
        "t-butyl": "tert-butyl",
    }.get(owner, owner)
    if owner not in {
        "methyl",
        "ethyl",
        "propyl",
        "isopropyl",
        "butyl",
        "isobutyl",
        "tert-butyl",
        "benzyl",
    }:
        return "", set()
    return owner, _fragment_component_atoms(mol, heavy[0], oxygen_idx)


def _acyl_azanide_fragment_name(mol, Chem, nitrogen_idx: int, neighbors: list[int]) -> str:
    acyl_records: list[tuple[str, set[int]]] = []
    substituent_records: list[tuple[str, set[int]]] = []
    covered = {nitrogen_idx}
    for nbr_idx in neighbors:
        acyl_record = _acyl_azanide_carbonyl_record(mol, Chem, nitrogen_idx, nbr_idx)
        if acyl_record:
            acyl_records.append(acyl_record)
            covered.update(acyl_record[1])
            continue
        substituent_record = _acyl_azanide_substituent_record(mol, Chem, nbr_idx, nitrogen_idx)
        if substituent_record is None:
            return ""
        substituent_records.append(substituent_record)
        covered.update(substituent_record[1])
    if len(acyl_records) != 1 or len(substituent_records) != 1:
        return ""
    if covered != set(range(mol.GetNumAtoms())):
        return ""
    acyl = acyl_records[0][0]
    substituent = substituent_records[0][0]
    if not _safe_acyl_azanide_term(acyl) or not _safe_acyl_azanide_term(substituent):
        return ""
    return f"{substituent}({acyl})azanide"


def _acyl_azanide_carbonyl_record(mol, Chem, nitrogen_idx: int, carbonyl_idx: int) -> tuple[str, set[int]] | None:
    carbonyl = mol.GetAtomWithIdx(carbonyl_idx)
    if carbonyl.GetSymbol() != "C" or carbonyl.GetFormalCharge() != 0 or carbonyl.IsInRing():
        return None
    bond_to_nitrogen = mol.GetBondBetweenAtoms(carbonyl_idx, nitrogen_idx)
    if bond_to_nitrogen is None or bond_to_nitrogen.GetBondType() != Chem.BondType.SINGLE:
        return None
    double_oxygens = []
    terminal_oxygens = []
    acyl_carbons = []
    for nbr in carbonyl.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == nitrogen_idx:
            continue
        bond = mol.GetBondBetweenAtoms(carbonyl_idx, nidx)
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_oxygens.append(nidx)
        elif nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.SINGLE:
            heavy = [
                atom.GetIdx()
                for atom in nbr.GetNeighbors()
                if atom.GetIdx() != carbonyl_idx and atom.GetAtomicNum() > 1
            ]
            if heavy or nbr.GetFormalCharge() != 0:
                return None
            terminal_oxygens.append(nidx)
        elif nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE:
            acyl_carbons.append(nidx)
        else:
            return None
    if len(double_oxygens) != 1:
        return None
    if len(terminal_oxygens) == 1 and not acyl_carbons:
        return "carboxy", {carbonyl_idx, double_oxygens[0], terminal_oxygens[0]}
    if len(acyl_carbons) != 1 or terminal_oxygens:
        return None
    acyl_idx = acyl_carbons[0]
    from .parent_rules import _ester_acyl_parent_paths, _generic_acyl_ligand_path_name

    names = [
        _generic_acyl_ligand_path_name(mol, Chem, carbonyl_idx, path)
        for path in _ester_acyl_parent_paths(mol, Chem, carbonyl_idx, acyl_idx)
    ]
    names = [name for name in names if name]
    if not names:
        return None
    atoms = {carbonyl_idx, double_oxygens[0]}
    atoms.update(_fragment_component_atoms(mol, acyl_idx, carbonyl_idx))
    return min(names, key=lambda value: (len(value), value)), atoms


def _acyl_azanide_substituent_record(mol, Chem, atom_idx: int, nitrogen_idx: int) -> tuple[str, set[int]] | None:
    atom = mol.GetAtomWithIdx(atom_idx)
    if atom.GetSymbol() == "C":
        return _linear_alkyl_fragment_owner_record(mol, Chem, atom_idx, nitrogen_idx)
    if atom.GetSymbol() == "O":
        heavy = [
            nbr.GetIdx()
            for nbr in atom.GetNeighbors()
            if nbr.GetIdx() != nitrogen_idx and nbr.GetAtomicNum() > 1
        ]
        if not heavy and atom.GetFormalCharge() == 0:
            return "hydroxy", {atom_idx}
        return None
    if atom.GetSymbol() == "S":
        heavy = [
            nbr.GetIdx()
            for nbr in atom.GetNeighbors()
            if nbr.GetIdx() != nitrogen_idx and nbr.GetAtomicNum() > 1
        ]
        if len(heavy) != 1 or atom.GetFormalCharge() != 0:
            return None
        owner_record = _linear_alkyl_fragment_owner_record(mol, Chem, heavy[0], atom_idx)
        if owner_record is None:
            return None
        owner, owner_atoms = owner_record
        return owner + "sulfanyl", {atom_idx, *owner_atoms}
    return None


def _linear_alkyl_fragment_owner_record(mol, Chem, atom_idx: int, parent_idx: int) -> tuple[str, set[int]] | None:
    simple_name, simple_atoms = _simple_fragment_owner_name(mol, Chem, atom_idx, parent_idx)
    if simple_name:
        return {"t-butyl": "tert-butyl"}.get(simple_name, simple_name), simple_atoms
    component = _fragment_component_atoms(mol, atom_idx, parent_idx)
    if not component:
        return None
    path = [atom_idx]
    previous = parent_idx
    current = atom_idx
    seen = {atom_idx}
    while True:
        atom = mol.GetAtomWithIdx(current)
        if atom.GetSymbol() != "C" or atom.GetFormalCharge() != 0 or atom.IsInRing():
            return None
        next_carbons = []
        for nbr in atom.GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx == previous or nbr.GetAtomicNum() <= 1:
                continue
            bond = mol.GetBondBetweenAtoms(current, nidx)
            if nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE:
                next_carbons.append(nidx)
            else:
                return None
        if not next_carbons:
            break
        if len(next_carbons) != 1 or next_carbons[0] in seen:
            return None
        previous, current = current, next_carbons[0]
        path.append(current)
        seen.add(current)
    if set(path) != component:
        return None
    root = ROOTS.get(len(path), "")
    return (root + "yl", set(path)) if root else None


def _safe_acyl_azanide_term(name: str) -> bool:
    if not name or len(name) > 80:
        return False
    if " " in name or "[" in name or "]" in name or name.startswith("("):
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9,'\\-]+", name))


def _azanide_substituent_prefix(mol, Chem, atom_idx: int, nitrogen_idx: int) -> str:
    atom = mol.GetAtomWithIdx(atom_idx)
    if atom.GetSymbol() == "C":
        from .branch_rules import branch_prefix

        return branch_prefix(mol, atom_idx, nitrogen_idx)
    if atom.GetSymbol() != "Si" or atom.GetFormalCharge() != 0:
        return ""
    names = []
    for nbr in atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == nitrogen_idx:
            continue
        bond = mol.GetBondBetweenAtoms(atom_idx, nidx)
        if nbr.GetSymbol() != "C" or bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return ""
        from .branch_rules import branch_prefix

        prefix = branch_prefix(mol, nidx, atom_idx)
        if not prefix:
            return ""
        names.append(prefix)
    if len(names) + atom.GetTotalNumHs() != 3:
        return ""
    from .parent.rule_helpers import _substituted_onium_name

    return _substituted_onium_name(names, "silyl")


def _perfluoro_counterion_name(mol, Chem) -> str:
    atoms = list(mol.GetAtoms())
    centers = [atom for atom in atoms if atom.GetSymbol() != "F"]
    fluorines = [atom for atom in atoms if atom.GetSymbol() == "F"]
    total_charge = _total_charge(mol)
    if len(centers) != 1 or len(fluorines) != len(atoms) - 1:
        return ""
    center = centers[0]
    if center.GetDegree() != len(fluorines):
        return ""
    for fluorine in fluorines:
        bond = mol.GetBondBetweenAtoms(center.GetIdx(), fluorine.GetIdx())
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return ""
    return PERFLUORO_ANIONS.get((center.GetSymbol(), len(fluorines), total_charge), "")


def _trifluoroacetate_name(mol, Chem) -> str:
    if _element_counts(mol) != {"C": 2, "F": 3, "O": 2} or _total_charge(mol) != -1:
        return ""
    carbonyl_carbons = [
        atom
        for atom in mol.GetAtoms()
        if atom.GetSymbol() == "C"
        and any(
            nbr.GetSymbol() == "O"
            and mol.GetBondBetweenAtoms(atom.GetIdx(), nbr.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
            for nbr in atom.GetNeighbors()
        )
    ]
    if len(carbonyl_carbons) != 1:
        return ""
    carbonyl = carbonyl_carbons[0]
    has_anionic_oxygen = False
    trifluoromethyl_neighbor = None
    for nbr in carbonyl.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(carbonyl.GetIdx(), nbr.GetIdx())
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetFormalCharge() == -1:
            has_anionic_oxygen = True
        elif nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE:
            trifluoromethyl_neighbor = nbr
    if not has_anionic_oxygen or trifluoromethyl_neighbor is None:
        return ""
    fluorine_neighbors = [
        nbr
        for nbr in trifluoromethyl_neighbor.GetNeighbors()
        if nbr.GetSymbol() == "F"
        and mol.GetBondBetweenAtoms(trifluoromethyl_neighbor.GetIdx(), nbr.GetIdx()).GetBondType() == Chem.BondType.SINGLE
    ]
    return "trifluoroacetate" if len(fluorine_neighbors) == 3 and trifluoromethyl_neighbor.GetDegree() == 4 else ""


def _oxoanion_name(mol, Chem) -> str:
    atoms = list(mol.GetAtoms())
    centers = [atom for atom in atoms if atom.GetSymbol() != "O"]
    oxygens = [atom for atom in atoms if atom.GetSymbol() == "O"]
    if len(centers) != 1 or len(oxygens) != len(atoms) - 1:
        return ""
    center = centers[0]
    if any(oxygen.GetDegree() != 1 for oxygen in oxygens):
        return ""
    if any(mol.GetBondBetweenAtoms(center.GetIdx(), oxygen.GetIdx()) is None for oxygen in oxygens):
        return ""
    hydroxy_count = sum(oxygen.GetTotalNumHs() for oxygen in oxygens)
    return OXOANIONS.get((center.GetSymbol(), len(oxygens), _total_charge(mol), hydroxy_count), "")


def _chalcogen_thiooxoanion_name(mol, Chem) -> str:
    return (
        _terminal_chalcogen_oxoanion_name(
            mol,
            Chem,
            center_symbol="S",
            terminal_symbol="S",
            oxygen_count=3,
            double_oxygen_count=1,
            dianion_name="thiosulfate",
            hydrogen_name="hydrogen thiosulfate",
        )
        or _terminal_chalcogen_oxoanion_name(
            mol,
            Chem,
            center_symbol="S",
            terminal_symbol="S",
            oxygen_count=2,
            double_oxygen_count=0,
            dianion_name="thiosulfite",
            hydrogen_name="hydrogen thiosulfite",
        )
        or _terminal_chalcogen_oxoanion_name(
            mol,
            Chem,
            center_symbol="Se",
            terminal_symbol="S",
            oxygen_count=3,
            double_oxygen_count=1,
            dianion_name="thioselenate",
            hydrogen_name="",
        )
        or _terminal_chalcogen_oxoanion_name(
            mol,
            Chem,
            center_symbol="S",
            terminal_symbol="Se",
            oxygen_count=3,
            double_oxygen_count=1,
            dianion_name="selenosulfate",
            hydrogen_name="",
        )
    )


def _simple_thiosulfur_oxoanion_name(mol, Chem) -> str:
    counts = _element_counts(mol)
    if counts not in ({"O": 1, "S": 2}, {"O": 2, "S": 2}) or _total_charge(mol) != -1:
        return ""
    centers = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == "S" and atom.GetTotalNumHs() == 1]
    for center in centers:
        single_oxido = double_oxygen = double_sulfur = 0
        for nbr in center.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(center.GetIdx(), nbr.GetIdx())
            if bond is None:
                return ""
            if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetFormalCharge() < 0:
                single_oxido += 1
            elif nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
                double_oxygen += 1
            elif nbr.GetSymbol() == "S" and bond.GetBondType() == Chem.BondType.DOUBLE:
                double_sulfur += 1
            else:
                return ""
        if single_oxido != 1 or double_sulfur != 1:
            continue
        if counts["O"] == 1 and double_oxygen == 0:
            return "thiosulfinate"
        if counts["O"] == 2 and double_oxygen == 1:
            return "thiosulfonate"
    return ""


def _terminal_chalcogen_oxoanion_name(
    mol,
    Chem,
    *,
    center_symbol: str,
    terminal_symbol: str,
    oxygen_count: int,
    double_oxygen_count: int,
    dianion_name: str,
    hydrogen_name: str,
) -> str:
    counts = _element_counts(mol)
    expected = {"O": oxygen_count}
    expected[center_symbol] = expected.get(center_symbol, 0) + 1
    expected[terminal_symbol] = expected.get(terminal_symbol, 0) + 1
    if counts != expected:
        return ""
    centers = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == center_symbol]
    for center in centers:
        single_oxido = hydroxy = double_oxygen = double_terminal = 0
        for nbr in center.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(center.GetIdx(), nbr.GetIdx())
            if bond is None:
                return ""
            if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
                double_oxygen += 1
            elif nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.SINGLE:
                if nbr.GetFormalCharge() < 0:
                    single_oxido += 1
                elif nbr.GetFormalCharge() == 0 and nbr.GetTotalNumHs() == 1:
                    hydroxy += 1
                else:
                    return ""
            elif nbr.GetSymbol() == terminal_symbol and bond.GetBondType() == Chem.BondType.DOUBLE:
                double_terminal += 1
            else:
                return ""
        if double_terminal != 1 or double_oxygen != double_oxygen_count:
            continue
        total_charge = _total_charge(mol)
        if total_charge == -2 and single_oxido == oxygen_count - double_oxygen_count and hydroxy == 0:
            return dianion_name
        if hydrogen_name and total_charge == -1 and single_oxido == oxygen_count - double_oxygen_count - 1 and hydroxy == 1:
            return hydrogen_name
    return ""


def _sulfur_oxo_cation_name(mol, Chem) -> str:
    counts = _element_counts(mol)
    if counts not in ({"O": 1, "S": 1}, {"O": 2, "S": 1}) or _total_charge(mol) != 1:
        return ""
    sulfur = next((atom for atom in mol.GetAtoms() if atom.GetSymbol() == "S"), None)
    if sulfur is None or sulfur.GetFormalCharge() <= 0 or sulfur.GetTotalNumHs() != 1:
        return ""
    oxygen_count = counts["O"]
    double_oxygen = 0
    for nbr in sulfur.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(sulfur.GetIdx(), nbr.GetIdx())
        if nbr.GetSymbol() != "O" or bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
            return ""
        double_oxygen += 1
    if double_oxygen != oxygen_count:
        return ""
    return "oxosulfonium" if oxygen_count == 1 else "dioxosulfonium"


def _phosphorus_oxo_cation_name(mol, Chem) -> str:
    counts = _element_counts(mol)
    if counts != {"O": 1, "P": 1} or _total_charge(mol) != 1:
        return ""
    phosphorus = next((atom for atom in mol.GetAtoms() if atom.GetSymbol() == "P"), None)
    oxygen = next((atom for atom in mol.GetAtoms() if atom.GetSymbol() == "O"), None)
    if phosphorus is None or oxygen is None:
        return ""
    bond = mol.GetBondBetweenAtoms(phosphorus.GetIdx(), oxygen.GetIdx())
    if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
        return ""
    return "phosphinylium" if phosphorus.GetFormalCharge() > 0 and phosphorus.GetTotalNumHs() == 2 else ""


def _phosphorus_thiooxoanion_name(mol, Chem) -> str:
    if _total_charge(mol) != -3:
        return ""
    phosphorus_atoms = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == "P"]
    if len(phosphorus_atoms) != 1:
        return ""
    phosphorus = phosphorus_atoms[0]
    pidx = phosphorus.GetIdx()
    single_oxido = 0
    single_sulfido = 0
    double_sulfur = 0
    for nbr in phosphorus.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(pidx, nbr.GetIdx())
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetFormalCharge() == -1:
            single_oxido += 1
        elif nbr.GetSymbol() == "S" and bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetFormalCharge() == -1:
            single_sulfido += 1
        elif nbr.GetSymbol() == "S" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_sulfur += 1
        else:
            return ""
    if double_sulfur != 1:
        return ""
    if single_oxido == 3 and single_sulfido == 0:
        return "thiophosphate"
    if single_oxido == 2 and single_sulfido == 1:
        return "dithiophosphate"
    return ""


def _diphosphite_name(mol, Chem) -> str:
    counts = _element_counts(mol)
    if counts != {"O": 5, "P": 2} or _total_charge(mol) != -4:
        return ""
    bridging_oxygens = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == "O" and atom.GetDegree() == 2]
    if len(bridging_oxygens) != 1:
        return ""
    bridge = bridging_oxygens[0]
    phosphorus_neighbors = [nbr for nbr in bridge.GetNeighbors() if nbr.GetSymbol() == "P"]
    if len(phosphorus_neighbors) != 2:
        return ""
    for phosphorus in phosphorus_neighbors:
        terminal_oxido = 0
        for nbr in phosphorus.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(phosphorus.GetIdx(), nbr.GetIdx())
            if nbr.GetIdx() == bridge.GetIdx() and bond.GetBondType() == Chem.BondType.SINGLE:
                continue
            if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetFormalCharge() < 0:
                terminal_oxido += 1
            else:
                return ""
        if terminal_oxido != 2:
            return ""
    return "diphosphite"


def _phosphorus_imido_chalcogen_oxoanion_name(mol, Chem) -> str:
    counts = _element_counts(mol)
    if counts not in ({"N": 1, "O": 3, "P": 1}, {"O": 3, "P": 1, "Se": 1}):
        return ""
    if _total_charge(mol) != -3:
        return ""
    phosphorus_atoms = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == "P"]
    if len(phosphorus_atoms) != 1:
        return ""
    phosphorus = phosphorus_atoms[0]
    single_oxido = double_nitrogen = double_selenium = 0
    for nbr in phosphorus.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(phosphorus.GetIdx(), nbr.GetIdx())
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetFormalCharge() < 0:
            single_oxido += 1
        elif nbr.GetSymbol() == "N" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_nitrogen += 1
        elif nbr.GetSymbol() == "Se" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_selenium += 1
        else:
            return ""
    if single_oxido != 3:
        return ""
    if double_nitrogen == 1 and double_selenium == 0:
        return "imidophosphate"
    if double_selenium == 1 and double_nitrogen == 0:
        return "selenophosphate"
    return ""


def _phosphorus_hydrido_thiooxoanion_name(mol, Chem) -> str:
    counts = _element_counts(mol)
    if counts not in ({"O": 2, "P": 1, "S": 1}, {"O": 1, "P": 1, "S": 2}):
        return ""
    if _total_charge(mol) != -2:
        return ""
    phosphorus_atoms = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == "P" and atom.GetTotalNumHs() == 1]
    if len(phosphorus_atoms) != 1:
        return ""
    phosphorus = phosphorus_atoms[0]
    single_oxido = single_sulfido = double_sulfur = 0
    for nbr in phosphorus.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(phosphorus.GetIdx(), nbr.GetIdx())
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetFormalCharge() < 0:
            single_oxido += 1
        elif nbr.GetSymbol() == "S" and bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetFormalCharge() < 0:
            single_sulfido += 1
        elif nbr.GetSymbol() == "S" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_sulfur += 1
        else:
            return ""
    if double_sulfur != 1:
        return ""
    if single_oxido == 2 and single_sulfido == 0:
        return "thiophosphonate"
    if single_oxido == 1 and single_sulfido == 1:
        return "dithiophosphonate"
    return ""


def _phosphorus_hydrido_oxoanion_name(mol, Chem) -> str:
    if _element_counts(mol) != {"O": 2, "P": 1} or _total_charge(mol) != -2:
        return ""
    phosphorus_atoms = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == "P" and atom.GetTotalNumHs() == 1]
    if len(phosphorus_atoms) != 1:
        return ""
    phosphorus = phosphorus_atoms[0]
    if phosphorus.GetFormalCharge() != -1:
        return ""
    single_oxido = 0
    double_oxygen = 0
    for nbr in phosphorus.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(phosphorus.GetIdx(), nbr.GetIdx())
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetFormalCharge() < 0:
            single_oxido += 1
        elif nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_oxygen += 1
        else:
            return ""
    return "phosphoranidate" if single_oxido == 1 and double_oxygen == 1 else ""


def _phosphorus_selenooxoanion_name(mol, Chem) -> str:
    counts = _element_counts(mol)
    if counts != {"O": 2, "P": 1, "Se": 2} or _total_charge(mol) != -3:
        return ""
    phosphorus_atoms = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == "P"]
    if len(phosphorus_atoms) != 1:
        return ""
    phosphorus = phosphorus_atoms[0]
    single_oxido = 0
    single_selenido = 0
    double_selenium = 0
    for nbr in phosphorus.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(phosphorus.GetIdx(), nbr.GetIdx())
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetFormalCharge() < 0:
            single_oxido += 1
        elif nbr.GetSymbol() == "Se" and bond.GetBondType() == Chem.BondType.SINGLE and nbr.GetFormalCharge() < 0:
            single_selenido += 1
        elif nbr.GetSymbol() == "Se" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_selenium += 1
        else:
            return ""
    return "diselenophosphate" if single_oxido == 2 and single_selenido == 1 and double_selenium == 1 else ""


def _organophosphorus_oxoanion_name(mol, Chem) -> str:
    phosphorus_atoms = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == "P" and atom.GetFormalCharge() == 0]
    if len(phosphorus_atoms) != 1:
        return ""
    phosphorus = phosphorus_atoms[0]
    pidx = phosphorus.GetIdx()
    carbon_neighbors = []
    double_oxygen_count = 0
    hydroxy_count = 0
    oxido_count = 0
    for nbr in phosphorus.GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(pidx, nidx)
        if nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE:
            carbon_neighbors.append(nidx)
        elif nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_oxygen_count += 1
        elif nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.SINGLE and len(nbr.GetNeighbors()) == 1:
            if nbr.GetFormalCharge() < 0:
                oxido_count += 1
            elif nbr.GetFormalCharge() == 0 and nbr.GetTotalNumHs() == 1:
                hydroxy_count += 1
            else:
                return ""
        else:
            return ""
    if len(carbon_neighbors) != 1 or double_oxygen_count != 1:
        return ""
    implicit_hydrogens = phosphorus.GetTotalNumHs()
    total_charge = _total_charge(mol)
    if implicit_hydrogens == 1 and oxido_count == 1 and hydroxy_count == 0 and total_charge == -1:
        suffix = "phosphinate"
        hydrogen_prefix = ""
    elif implicit_hydrogens == 0 and oxido_count == 2 and hydroxy_count == 0 and total_charge == -2:
        suffix = "phosphonate"
        hydrogen_prefix = ""
    elif implicit_hydrogens == 0 and oxido_count == 1 and hydroxy_count == 1 and total_charge == -1:
        suffix = "phosphonate"
        hydrogen_prefix = "hydrogen "
    else:
        return ""
    from .branch_rules import branch_prefix

    prefix = branch_prefix(mol, carbon_neighbors[0], pidx)
    if not prefix:
        return ""
    if _needs_fragment_parentheses(prefix):
        prefix = f"({prefix})"
    return hydrogen_prefix + prefix + suffix


def _pseudohalide_name(mol, Chem) -> str:
    counts = _element_counts(mol)
    total_charge = _total_charge(mol)
    if total_charge != -1:
        return ""
    if counts == {"C": 1, "N": 1}:
        return "cyanide" if _has_bond(mol, Chem, "C", "N", Chem.BondType.TRIPLE) else ""
    if counts == {"N": 3}:
        return "azide" if any(atom.GetDegree() == 2 for atom in mol.GetAtoms()) else ""
    if counts == {"C": 1, "N": 1, "O": 1}:
        if _has_bond(mol, Chem, "C", "N", Chem.BondType.TRIPLE):
            if _is_fulminate_anion(mol, Chem):
                return "fulminate"
            return "cyanate"
        if _has_bond(mol, Chem, "C", "N", Chem.BondType.DOUBLE) and _has_bond(
            mol, Chem, "C", "O", Chem.BondType.DOUBLE
        ):
            return "isocyanate"
    if counts == {"C": 1, "N": 1, "S": 1}:
        if _has_bond(mol, Chem, "C", "N", Chem.BondType.TRIPLE):
            return "thiocyanate"
        if _has_bond(mol, Chem, "C", "N", Chem.BondType.DOUBLE) and _has_bond(
            mol, Chem, "C", "S", Chem.BondType.DOUBLE
        ):
            return "isothiocyanate"
    if counts == {"C": 1, "N": 1, "Se": 1}:
        if _has_bond(mol, Chem, "C", "N", Chem.BondType.TRIPLE):
            return "selenocyanate"
        if _has_bond(mol, Chem, "C", "N", Chem.BondType.DOUBLE) and _has_bond(
            mol, Chem, "C", "Se", Chem.BondType.DOUBLE
        ):
            return "isoselenocyanate"
    return ""


def _is_fulminate_anion(mol, Chem) -> bool:
    if _total_charge(mol) != -1:
        return False
    for carbon in mol.GetAtoms():
        if carbon.GetSymbol() != "C" or carbon.GetFormalCharge() >= 0:
            continue
        cidx = carbon.GetIdx()
        for nitrogen in carbon.GetNeighbors():
            nidx = nitrogen.GetIdx()
            bond = mol.GetBondBetweenAtoms(cidx, nidx)
            if (
                nitrogen.GetSymbol() != "N"
                or nitrogen.GetFormalCharge() <= 0
                or bond is None
                or bond.GetBondType() != Chem.BondType.TRIPLE
            ):
                continue
            terminal_oxygens = [
                neighbor
                for neighbor in nitrogen.GetNeighbors()
                if neighbor.GetIdx() != cidx
                and neighbor.GetSymbol() == "O"
                and neighbor.GetFormalCharge() < 0
                and len(neighbor.GetNeighbors()) == 1
            ]
            if len(terminal_oxygens) == 1:
                return True
    return False


def _trifluoromethanesulfonate_name(mol, Chem) -> str:
    counts = _element_counts(mol)
    if counts != {"C": 1, "F": 3, "O": 3, "S": 1} or _total_charge(mol) != -1:
        return ""
    carbon = next(atom for atom in mol.GetAtoms() if atom.GetSymbol() == "C")
    sulfur = next(atom for atom in mol.GetAtoms() if atom.GetSymbol() == "S")
    carbon_neighbors = {neighbor.GetSymbol() for neighbor in carbon.GetNeighbors()}
    sulfur_oxygen_count = sum(1 for neighbor in sulfur.GetNeighbors() if neighbor.GetSymbol() == "O")
    if carbon_neighbors == {"F", "S"} and sulfur_oxygen_count == 3:
        return "trifluoromethanesulfonate"
    return ""


def _bis_trifluoromethanesulfonyl_azanide_name(mol, Chem) -> str:
    counts = _element_counts(mol)
    if counts != {"C": 2, "F": 6, "N": 1, "O": 4, "S": 2} or _total_charge(mol) != -1:
        return ""
    nitrogens = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == "N" and atom.GetFormalCharge() < 0]
    if len(nitrogens) != 1:
        return ""
    nitrogen = nitrogens[0]
    sulfur_neighbors = [
        nbr
        for nbr in nitrogen.GetNeighbors()
        if nbr.GetSymbol() == "S"
        and mol.GetBondBetweenAtoms(nitrogen.GetIdx(), nbr.GetIdx()).GetBondType() == Chem.BondType.SINGLE
    ]
    if len(sulfur_neighbors) != 2:
        return ""
    if all(_is_trifluoromethanesulfonyl_group(mol, Chem, sulfur.GetIdx(), nitrogen.GetIdx()) for sulfur in sulfur_neighbors):
        return "bis(trifluoromethylsulfonyl)azanide"
    return ""


def _is_trifluoromethanesulfonyl_group(mol, Chem, sulfur_idx: int, nitrogen_idx: int) -> bool:
    sulfur = mol.GetAtomWithIdx(sulfur_idx)
    double_oxygen_count = 0
    carbon_neighbors = []
    for nbr in sulfur.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == nitrogen_idx:
            continue
        bond = mol.GetBondBetweenAtoms(sulfur_idx, nidx)
        if nbr.GetSymbol() == "O" and bond.GetBondType() == Chem.BondType.DOUBLE:
            double_oxygen_count += 1
        elif nbr.GetSymbol() == "C" and bond.GetBondType() == Chem.BondType.SINGLE:
            carbon_neighbors.append(nidx)
        else:
            return False
    if double_oxygen_count != 2 or len(carbon_neighbors) != 1:
        return False
    carbon = mol.GetAtomWithIdx(carbon_neighbors[0])
    fluorines = [
        nbr
        for nbr in carbon.GetNeighbors()
        if nbr.GetIdx() != sulfur_idx
        and nbr.GetSymbol() == "F"
        and mol.GetBondBetweenAtoms(carbon.GetIdx(), nbr.GetIdx()).GetBondType() == Chem.BondType.SINGLE
    ]
    return len(fluorines) == 3 and carbon.GetDegree() == 4


def _has_bond(mol, Chem, left_symbol: str, right_symbol: str, bond_type) -> bool:
    symbols = {left_symbol, right_symbol}
    for bond in mol.GetBonds():
        if bond.GetBondType() != bond_type:
            continue
        if {bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()} == symbols:
            return True
    return False


def _specific_bond(mol, Chem, left_idx: int, right_idx: int, bond_type) -> bool:
    bond = mol.GetBondBetweenAtoms(left_idx, right_idx)
    return bond is not None and bond.GetBondType() == bond_type


def _element_counts(mol) -> dict[str, int]:
    cached = mol_cache_get(mol, "fragment_element_counts")
    if cached is not None:
        return cached
    counts: dict[str, int] = {}
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        counts[symbol] = counts.get(symbol, 0) + 1
    return mol_cache_set(mol, "fragment_element_counts", counts)


def _total_charge(mol) -> int:
    cached = mol_cache_get(mol, "fragment_total_charge")
    if cached is not None:
        return cached
    return mol_cache_set(mol, "fragment_total_charge", sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))


def _salt_name(parent: str, fragments: list[str]) -> str:
    suffix = _multiplied_suffix(fragments)
    if not suffix:
        return ""
    return NameParts(parent=parent, suffixes=(Suffix(suffix, separator=" "),)).render()


def _multiplied_suffix(names: list[str]) -> str:
    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    parts = []
    for name in sorted(counts):
        count = counts[name]
        multiplied = _multiplied_fragment_suffix(name, count)
        if not multiplied:
            return ""
        parts.append(multiplied)
    return " ".join(parts)


def _multiplied_fragment_suffix(name: str, count: int) -> str:
    if count == 1:
        return name
    if _needs_complex_fragment_multiplier(name):
        multiplier = COMPLEX_MULTIPLIERS.get(count, "")
        return f"{multiplier}({name})" if multiplier else ""
    multiplier = MULTIPLIERS.get(count, "")
    return multiplier + name if multiplier else ""


def _needs_complex_fragment_multiplier(name: str) -> bool:
    if "(" in name or ")" in name:
        return True
    if " " in name:
        return True
    return name.startswith(("bis", "tris", "tetrakis", "tetra", "penta", "hexa")) or name.endswith(
        ("azanide", "phosphinate", "phosphonate")
    )


def _needs_fragment_parentheses(name: str) -> bool:
    return name[:1].isdigit() or "-" in name or "," in name or name.startswith("(")
