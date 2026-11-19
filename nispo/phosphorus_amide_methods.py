"""Mapped neutral P(V) amides and O-esters of amidic phosphorus acids.

P-67.1.2.6.1 retains the acid class followed by amide when every acid OH
has been replaced by nitrogen. P-67.1.2.3/P-67.1.3.2 instead constructs an
amidate anion after the O-organyl words when ester arms remain. These are
graph operations on one ordinary P=O centre, not names looked up by molecule.
"""

from itertools import permutations

from .constants import MULT
from .molecule_iteration import iter_atoms, iter_bonds
from .naming_methods import MethodDecision
from .phosphorus_ester_methods import _carbon_component, _organyl_words, _branch_equivalence


RULE = "P-67.1.2.3/P-67.1.2.6.1/P-67.1.3.2/P-14.3.4.5/P-14.4/P-14.5/P-16.3"


def _recognize(mol, Chem):
    if (mol.GetNumHeavyAtoms() > 128 or mol.GetStereoGroups()
            or any(a.GetAtomicNum() not in {6, 7, 8, 15}
                   or a.GetFormalCharge() or a.GetNumRadicalElectrons()
                   or a.GetIsotope() or int(a.GetChiralTag()) for a in iter_atoms(mol))
            or any(b.GetStereo() != Chem.BondStereo.STEREONONE for b in iter_bonds(mol))):
        return None
    ps = [a for a in iter_atoms(mol) if a.GetAtomicNum() == 15]
    if len(ps) != 1:
        return None
    p = ps[0]
    if p.IsInRing() or p.GetTotalValence() != 5:
        return None
    phosphorus = p.GetIdx()
    double_o, p_arms, esters, nitrogens, components = [], [], [], [], []
    core = {phosphorus}

    def carbon_arm(root, attachment):
        atoms = _carbon_component(mol, root, attachment)
        if atoms is None or any(atoms & prior for prior in components):
            return None
        components.append(atoms)
        return root, atoms

    for neighbor in p.GetNeighbors():
        idx = neighbor.GetIdx()
        order = mol.GetBondBetweenAtoms(phosphorus, idx).GetBondTypeAsDouble()
        if (neighbor.GetAtomicNum() == 8 and order == 2
                and neighbor.GetDegree() == 1 and neighbor.GetTotalValence() == 2):
            double_o.append(idx)
            core.add(idx)
        elif neighbor.GetAtomicNum() == 6 and order == 1:
            arm = carbon_arm(idx, phosphorus)
            if arm is None:
                return None
            p_arms.append(arm)
        elif neighbor.GetAtomicNum() == 8 and order == 1:
            if (neighbor.IsInRing() or neighbor.GetTotalValence() != 2
                    or neighbor.GetTotalNumHs(includeNeighbors=True)):
                return None  # Acid OH and cyclic/bridging esters need their own proof.
            attached = [a for a in neighbor.GetNeighbors() if a.GetIdx() != phosphorus]
            if len(attached) != 1 or attached[0].GetAtomicNum() != 6:
                return None
            arm = carbon_arm(attached[0].GetIdx(), idx)
            if arm is None:
                return None
            esters.append((idx, *arm))
            core.add(idx)
        elif neighbor.GetAtomicNum() == 7 and order == 1:
            if (neighbor.IsInRing() or neighbor.GetTotalValence() != 3
                    or neighbor.GetIsAromatic()):
                return None
            arms = []
            for attached in neighbor.GetNeighbors():
                if attached.GetIdx() == phosphorus:
                    continue
                if (attached.GetAtomicNum() != 6
                        or mol.GetBondBetweenAtoms(idx, attached.GetIdx()).GetBondTypeAsDouble() != 1):
                    return None
                arm = carbon_arm(attached.GetIdx(), idx)
                if arm is None:
                    return None
                arms.append(arm)
            if len(arms) + neighbor.GetTotalNumHs(includeNeighbors=True) != 2:
                return None
            nitrogens.append((idx, tuple(arms)))
            core.add(idx)
        else:
            return None
    if len(double_o) != 1 or not nitrogens:
        return None
    substitutions = len(p_arms) + p.GetTotalNumHs(includeNeighbors=True)
    kinds = {0: "phosphoric", 1: "phosphonic", 2: "phosphinic"}
    if substitutions not in kinds or len(esters) + len(nitrogens) != 3 - substitutions:
        return None
    heavy = frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1)
    if frozenset(core).union(*components) != heavy:
        return None
    return dict(p=phosphorus, double_o=double_o[0], p_arms=tuple(p_arms),
                ester_oxygens=tuple(esters), nitrogens=tuple(nitrogens),
                kind=kinds[substitutions], core=frozenset(core), heavy=heavy)


def _substitution(mol, Chem, site):
    """Assign primed N locants from prefix citation, independent of atom IDs."""
    from .component_mapping import mapped_branch
    from .formatting import format_prefixes, prefix_alphabetization_key, multiply_component

    def render(arms, attachment):
        rows = []
        for root, atoms in arms:
            text, owned = mapped_branch(mol, Chem, root, attachment)
            if not text or frozenset(owned) != atoms:
                return None
            rows.append((text, atoms, _branch_equivalence(mol, Chem, atoms, root)))
        return tuple(rows)

    p_rows = render(site["p_arms"], site["p"])
    if p_rows is None:
        return None
    n_rows = []
    for nitrogen, arms in site["nitrogens"]:
        rows = render(arms, nitrogen)
        if rows is None:
            return None
        n_rows.append((nitrogen, rows))
    has_n_prefix = any(rows for _n, rows in n_rows)
    words = sorted({text for _n, rows in n_rows for text, _atoms, _sig in rows},
                   key=prefix_alphabetization_key)
    choices = []
    for arrangement in permutations(n_rows):
        key = tuple(tuple(index for index, (_n, rows) in enumerate(arrangement)
                          for text, _atoms, _signature in rows if text == word)
                    for word in words)
        nmap = {n: "N" + "'" * index for index, (n, _rows) in enumerate(arrangement)}
        prefixes = [("P" if has_n_prefix else "", text) for text, _atoms, _sig in p_rows]
        prefixes.extend((nmap[n], text) for n, rows in arrangement
                        for text, _atoms, _sig in rows)
        # In phosphoric triamide there are no P-H substitution sites. Six
        # identical groups filling all N-H sites have an intrinsic assignment.
        all_n = [row for _n, rows in arrangement for row in rows]
        intrinsic = (site["kind"] == "phosphoric" and not site["ester_oxygens"]
                     and len(all_n) == 6 and len({(row[0], row[2]) for row in all_n}) == 1)
        text = (multiply_component(all_n[0][0], 6) if intrinsic else
                format_prefixes(prefixes, omit_locants=not has_n_prefix,
                                hierarchical_enclosures=True))
        if prefixes and not text:
            return None
        choices.append((key, text, nmap))
    if not choices:
        return None
    best = min(row[0] for row in choices)
    finalists = [row for row in choices if row[0] == best]
    if len({row[1] for row in finalists}) != 1:
        return None
    return finalists[0][1:]


def phosphorus_amide_decision(mol, Chem):
    """Return a complete neutral P(V) amide/amide-ester decision, or decline."""
    site = _recognize(mol, Chem)
    if site is None:
        return None
    substitution = _substitution(mol, Chem, site)
    if substitution is None:
        return None
    prefix, nitrogen_map = substitution
    n_count = len(site["nitrogens"])
    if site["ester_oxygens"]:
        # Retained acid stems plus the productive functional replacement infix.
        stem = {"phosphoric": "phosphor", "phosphonic": "phosphon",
                "phosphinic": "phosphin"}[site["kind"]]
        infix = "amid" if n_count == 1 else "o" + MULT[n_count] + "amid"
        anion = stem + infix + "ate"
        words = _organyl_words(mol, Chem, site)
        if words is None:
            return None
        name = " ".join([*(word for _key, word, _rows in words), prefix + anion])
    else:
        multiplier = "" if n_count == 1 else MULT[n_count]
        name = prefix + site["kind"] + " " + multiplier + "amide"
    mapping = {site["p"]: "P", site["double_o"]: "O", **nitrogen_map}
    roles = [("phosphorus-amide-core", site["core"], tuple(mapping.items()))]
    roles.extend(("P-organyl", atoms, ()) for _root, atoms in site["p_arms"])
    roles.extend(("O-organyl", atoms, ()) for _o, _root, atoms in site["ester_oxygens"])
    roles.extend(("N-organyl", atoms, ()) for _n, arms in site["nitrogens"] for _root, atoms in arms)
    owned = [idx for _role, atoms, _locants in roles for idx in atoms]
    if len(owned) != len(set(owned)) or frozenset(owned) != site["heavy"]:
        return None
    return MethodDecision(name, "functional-class", frozenset({site["p"]}),
                          site["heavy"], RULE, tuple(mapping.items()), component_roles=tuple(roles))
