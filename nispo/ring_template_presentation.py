"""Restore distinctions that abbreviated ring catalog aliases omit."""

import re
from itertools import combinations


_NAPHTHYRIDINE_LOCANTS = {"1,5", "1,6", "1,7", "1,8", "2,6", "2,7"}
_IMIDAZONAPHTHYRIDINE = re.compile(r"imidazo\[4,5-[cf]\]naphthyridin")


def ring_template_name(name, mol, locants):
    if name in {"naphthyridin", "naphthyridine"}:
        if (mol.GetNumAtoms() == len(locants) == 10 and mol.GetNumBonds() == 11
                and sorted(map(len, mol.GetRingInfo().AtomRings())) == [6, 6]
                and set(locants.values()) == {"1", "2", "3", "4", "4a", "5", "6", "7", "8", "8a"}
                and all(a.GetIsAromatic() and a.GetAtomicNum() in {6, 7} for a in mol.GetAtoms())):
            nitrogen = ",".join(sorted(locants[a.GetIdx()] for a in mol.GetAtoms()
                                       if a.GetAtomicNum() == 7))
            if nitrogen in _NAPHTHYRIDINE_LOCANTS:
                return f"{nitrogen}-naphthyridine"
    if re.fullmatch(r"imidazo\[1,[25]-a\]naphthyridine", name):
        rings = sorted(map(frozenset, mol.GetRingInfo().AtomRings()), key=len)
        if list(map(len, rings)) == [5, 6, 6] and _has_a_fused_18_naphthyridine(mol, *rings):
            return name.removesuffix("naphthyridine") + "[1,8]naphthyridine"
    if name.startswith("spiro[") and re.search(r"-pyrrolin(?:e)?\]", name):
        for ring in mol.GetRingInfo().AtomRings():
            if (len(ring) == 5 and sorted(mol.GetAtomWithIdx(i).GetAtomicNum() for i in ring)
                    == [6, 6, 6, 6, 7]
                    and all(locants.get(i, "").endswith("'") for i in ring)
                    and all(mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble() == 1
                            for a, b in zip(ring, ring[1:] + ring[:1]))):
                return re.sub(r"-pyrrolin(?:e)?\]", "-pyrrolidine]", name)
    return name


def _has_a_fused_18_naphthyridine(mol, added, left, right, *, edge_is_carbon=False):
    core = left | right
    fusion = left & right
    if len(fusion) != 2 or len(added & core) != 2:
        return False
    n_sites = {i for i in core if mol.GetAtomWithIdx(i).GetAtomicNum() == 7}
    return (len(n_sites) == 2 and len(n_sites & (left - right)) == 1
            and len(n_sites & (right - left)) == 1
            and len(n_sites & added) == (0 if edge_is_carbon else 1)
            and sum(mol.GetAtomWithIdx(i).GetAtomicNum() == 7 for i in added) == 2
            and all(mol.GetAtomWithIdx(i).GetIsAromatic()
                    for i in (core if edge_is_carbon else core | added))
            and all(mol.GetAtomWithIdx(i).GetAtomicNum() in {6, 7} for i in core | added)
            and any(all(mol.GetBondBetweenAtoms(i, j) is not None for j in n_sites)
                    for i in fusion))


def locanted_fused_component_presentation(mol, name):
    """Cite the base component's two nitrogen positions in the selected name."""
    if "]naphthyridin" not in name:
        return name
    matches = tuple(_IMIDAZONAPHTHYRIDINE.finditer(name))
    if len(matches) != 1:
        return name
    rings = tuple(map(frozenset, mol.GetRingInfo().AtomRings()))
    systems = {added | left | right for added in rings if len(added) == 5
               for left, right in combinations((ring for ring in rings if len(ring) == 6), 2)
               if _has_a_fused_18_naphthyridine(mol, added, left, right, edge_is_carbon=True)}
    if len(systems) == 1:
        match = matches[0]
        return name[:match.start()] + "imidazo[4,5-c][1,8]naphthyridin" + name[match.end():]
    return name


def naphthyridine_hydro_prefix(mol, name, locants, reverse):
    """Return the two hydro locants for a mapped mono-oxo aza ring."""
    if not name.endswith("-naphthyridine") or name.split("-", 1)[0] not in _NAPHTHYRIDINE_LOCANTS:
        return ""
    core = reverse.keys()
    oxo_sites, hydrogenated_n = [], []
    for i in core:
        atom = mol.GetAtomWithIdx(i)
        if not atom.GetIsAromatic() or atom.GetFormalCharge():
            return ""
        outer = [b for b in atom.GetBonds() if b.GetOtherAtomIdx(i) not in core]
        if atom.GetAtomicNum() == 6:
            for bond in outer:
                if bond.GetBondTypeAsDouble() != 2:
                    continue
                ligand = bond.GetOtherAtom(atom)
                if (ligand.GetAtomicNum() not in {8, 16} or ligand.GetDegree() != 1
                        or ligand.GetFormalCharge() or ligand.GetNumRadicalElectrons()):
                    return ""
                oxo_sites.append(i)
        elif atom.GetAtomicNum() == 7 and atom.GetTotalValence() == 3:
            if atom.GetTotalNumHs() == 1 or len(outer) == 1 and outer[0].GetBondTypeAsDouble() == 1:
                hydrogenated_n.append(i)
    if len(oxo_sites) != 1 or len(hydrogenated_n) != 1:
        return ""
    sites = {*oxo_sites, *hydrogenated_n}
    if not any(sites <= set(ring) <= core for ring in mol.GetRingInfo().AtomRings()):
        return ""
    return ",".join(sorted(locants[reverse[i]] for i in sites)) + "-dihydro"


def purine_attachment_hydride(mol, match, locants, attachment):
    """Hydrogen accommodation for a 6-ylidene purine attached through N9."""
    if attachment != "9":
        return ""
    by_locant = {loc: mol.GetAtomWithIdx(match[i]) for i, loc in locants.items() if loc}
    if not {"1", "3", "6", "7", "9"} <= by_locant.keys():
        return ""
    if any(by_locant[loc].GetAtomicNum() != 7 or by_locant[loc].GetFormalCharge()
           for loc in ("1", "3", "7", "9")):
        return ""
    hydrogens = {loc for loc in ("1", "3", "7", "9") if by_locant[loc].GetTotalNumHs() == 1}
    core = set(match)
    external_doubles = [(loc, b.GetOtherAtom(atom)) for loc, atom in by_locant.items()
                        for b in atom.GetBonds() if b.GetBondTypeAsDouble() == 2
                        and b.GetOtherAtomIdx(atom.GetIdx()) not in core]
    if (hydrogens != {"3"} or len(external_doubles) != 1
            or external_doubles[0][0] != "6" or external_doubles[0][1].GetAtomicNum() != 7
            or external_doubles[0][1].GetFormalCharge()):
        return ""
    return "3,6-dihydro-9H-purin"


_FUSED_LACTAM_TEMPLATE = re.compile(
    r"(?:(?:\d+,\d+-dihydro-)?\d+H-)?"
    r"(?P<base>pyrazolo\[3,4-c\]isoquinolin|pyrazolo\[3,4-d\]pyrimidin|1,4-benzoxazin)"
    r"-(?P<oxo>\d+)(?:\(\d+H\))?-one$"
)
_FUSED_LACTAM_HYDRO_STATES = {
    "pyrazolo[3,4-c]isoquinolin": ("5", "4,5-dihydro-3H-", ("2", "3", "4"), {"3"}),
    "pyrazolo[3,4-d]pyrimidin": ("4", "4,5-dihydro-1H-", ("1", "2", "5", "7"), {"1", "5"}),
    "1,4-benzoxazin": ("3", "2,3-dihydro-4H-", ("4",), set()),
}


def fused_lactam_substituent_presentation(mol, parent, match, locants, attachment, records, stereo):
    """Cite an oxo group on the numbered fused hydride of this branch."""
    template = _FUSED_LACTAM_TEMPLATE.fullmatch(parent)
    if template is None:
        return ""
    base, carbonyl = template.group("base", "oxo")
    oxo, hydride, nitrogens, nh = _FUSED_LACTAM_HYDRO_STATES[base]
    if oxo != carbonyl:
        return ""
    by_locant = {loc: mol.GetAtomWithIdx(match[i]) for i, loc in locants.items() if loc}
    if not {oxo, *nitrogens} <= by_locant.keys():
        return ""
    if base == "1,4-benzoxazin" and attachment != "4":
        # An aryl attachment leaves the lactam hydrogen on N4. Use the
        # indicated-H hydride with the same mapped C3 carbonyl and N4.
        if not by_locant[attachment].GetIsAromatic():
            return ""
        hydride, nh = "3,4-dihydro-2H-", {"4"}
    if base == "pyrazolo[3,4-d]pyrimidin" and (
            attachment == "1" or any(loc == "1" for loc, _, _ in records)):
        nh = nh - {"1"}
    if any(by_locant[i].GetAtomicNum() != 7 or by_locant[i].GetFormalCharge()
           or by_locant[i].GetTotalNumHs() != int(i in nh) for i in nitrogens):
        return ""
    carbon = by_locant[oxo]
    if carbon.GetAtomicNum() != 6 or not any(
            b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(carbon).GetAtomicNum() == 8
            and b.GetOtherAtom(carbon).GetDegree() == 1 for b in carbon.GetBonds()):
        return ""
    if base == "pyrazolo[3,4-c]isoquinolin" and not any(loc == "4" for loc, _, _ in records):
        return ""
    from .formatting import format_prefixes

    prefixes = format_prefixes([(loc, word) for loc, word, _ in records] + [(oxo, "oxo")],
                               hierarchical_enclosures=True)
    name = f"{prefixes}-{hydride}{base}-{attachment}-yl"
    return f"({stereo})-{name}" if stereo else name
