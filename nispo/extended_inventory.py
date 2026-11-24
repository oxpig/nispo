"""Broader selection inventory with explicit ion and S/P functional sites.

This inventory is opt-in for the complete-parent selector. Existing bounded
renderers keep their eligibility. Unknown charged or S/P motifs decline rather
than acquiring the rank of an unrelated neutral function.
"""

from .molecule_iteration import iter_atoms
from .functional_inventory import CharacteristicGroup, _inventory, functional_inventory
from .molecule_cache import mol_cache_get, mol_cache_set
from .sulfur_selection import sulfonamide_n_sites, sulfonamide_ylidene_root


def selection_inventory(mol):
    cached = mol_cache_get(mol, "selection_inventory")
    if cached is None:
        ordinary = functional_inventory(mol)
        cached = mol_cache_set(
            mol,
            "selection_inventory",
            ordinary if ordinary.supported else _inventory(mol, extended=True),
        )
    return cached


def additional_sites(mol, *, charge_role_proofs=(), subordinate_thioxo=(), subordinate_heterones=None):
    from .functional_inventory import nitrogen_oxoamide_sites
    from .neutral_charge_components import (
        _recognize_at_carbon, neutral_charge_atom_ids, validate_neutral_charge_role_proofs,
    )

    charge_role_proofs = tuple(charge_role_proofs)
    if charge_role_proofs and not validate_neutral_charge_role_proofs(mol, charge_role_proofs):
        return (), set(), set(), False

    oxo_sites = nitrogen_oxoamide_sites(mol)
    oxo_pairs = {(n1, n0) for _kind, n0, n1, _oxygens in oxo_sites}
    oxo_core_nitrogens = {n0 for _kind, n0, _n1, _oxygens in oxo_sites}
    groups, covered, used_n = [], set(neutral_charge_atom_ids(charge_role_proofs)), set()
    # The exact charge-separated N+/O- pair is part of the neutral nitramide
    # resonance representation. Cover only those two atoms so the extended
    # inventory can retain isotope annotations without opening a general
    # charged-graph path.
    for kind, n0, _n1, oxygens in oxo_sites:
        if kind == "nitramide":
            covered.add(n0)
            covered.update(o for o in oxygens if mol.GetAtomWithIdx(o).GetFormalCharge() == -1)
    from .oxoacid_ester_methods import oxoacid_ester_site

    oxoacid_esters = {atom.GetIdx(): site for atom in iter_atoms(mol)
                     if atom.GetAtomicNum() in {15, 16} and any(
                         n.GetAtomicNum() in {8, 16} and n.GetDegree() == 2
                         for n in atom.GetNeighbors())
                     if (site := oxoacid_ester_site(mol, atom)) is not None}
    ester_links = {link for site in oxoacid_esters.values() for link, _ in site.esters}
    for atom in iter_atoms(mol):
        idx, number = atom.GetIdx(), atom.GetAtomicNum()
        if (
            number == 7
            and atom.GetFormalCharge() == 1
            and (all(b.GetBondTypeAsDouble() == 1 for b in atom.GetBonds())
                 or atom.GetIsAromatic() and atom.IsInRing()
                 and sum(n.GetAtomicNum() > 1 and not mol.GetBondBetweenAtoms(
                     idx, n.GetIdx()).IsInRing() for n in atom.GetNeighbors()) == 1)
        ):
            groups.append(CharacteristicGroup("ammonium", idx, (idx,)))
            covered.add(idx)
            used_n.add(idx)
        if number not in {15, 16} or idx in ester_links or idx in subordinate_thioxo:
            continue
        if number == 16 and atom.GetDegree() == 4 and any(
                neighbor.GetAtomicNum() in {9, 17, 35, 53} for neighbor in atom.GetNeighbors()):
            from .bounded_sulfonyl_halides import sulfonyl_halide_site

            site = sulfonyl_halide_site(atom)
            if site is not None:
                first_o, second_o, halogen, owner = site
                groups.append(CharacteristicGroup("sulfonyl-halide", idx,
                    (idx, first_o, second_o, halogen), owner, halogen))
                continue
        if number == 16 and atom.GetDegree() == 1 and atom.GetBonds()[0].GetBondTypeAsDouble() == 2:
            from .hydrazide_components import thiosemicarbazone_site

            site = thiosemicarbazone_site(atom)
            if site is not None:
                atoms, _ = site
                groups.append(CharacteristicGroup("thiosemicarbazone", atoms[0], atoms, -1, atoms[2]))
                used_n.update(atoms[2:])
                continue
        if number == 16 and atom.GetDegree() == 4 and any(
                b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(atom).GetAtomicNum() == 7
                for b in atom.GetBonds()):
            from .nitrogen_acid_derivatives import sulfonimidamide_site

            imidamide = sulfonimidamide_site(atom)
            if imidamide is not None:
                groups.append(imidamide)
                used_n.update(i for i in imidamide.atoms
                              if mol.GetAtomWithIdx(i).GetAtomicNum() == 7)
                continue
        if idx in oxoacid_esters:
            ester = oxoacid_esters[idx]
            groups.append(ester.group())
            covered.update(ester.anions)
            if ester.nitrogen is not None:
                used_n.add(ester.nitrogen)
            continue
        # P-22/P-41: an annular divalent sulfur is a skeletal atom, not an
        # unrecognized sulfur characteristic group. Its original isotope
        # remains available to the mapped ring producer (P-82).
        if (number==16 and atom.IsInRing() and atom.GetTotalValence()==2
                and not atom.GetFormalCharge() and not int(atom.GetChiralTag())):
            continue
        if (number == 16 and atom.IsInRing() and atom.GetTotalValence() == 6
                and not atom.GetFormalCharge() and not atom.GetIsotope()
                and not int(atom.GetChiralTag()) and atom.GetDegree() == 4):
            # P-66.1.1.4.5: a sultam can remain a heterocyclic branch
            # under a senior carboxylic acid. The two ring bonds must lead
            # to ordinary carbon and nitrogen, with two exocyclic oxo O.
            annular = sorted(b.GetOtherAtom(atom).GetAtomicNum() for b in atom.GetBonds()
                             if b.IsInRing() and b.GetBondTypeAsDouble() == 1)
            oxo = [b.GetOtherAtom(atom) for b in atom.GetBonds()
                   if not b.IsInRing() and b.GetBondTypeAsDouble() == 2]
            if annular in ([6, 6], [6, 7], [7, 8]) and len(oxo) == 2 and all(
                    n.GetAtomicNum() == 8 and n.GetDegree() == 1 and not n.GetFormalCharge() for n in oxo):
                if annular == [6, 6]:
                    continue  # A cyclic sulfone supplies two subordinate oxo groups.
                from rdkit import Chem
                from .functional_inventory import _neutral_carboxyl_query
                if mol.HasSubstructMatch(_neutral_carboxyl_query(Chem)):
                    continue
        if (
            atom.IsInRing()
            or atom.GetFormalCharge()
            or atom.GetIsotope()
            or int(atom.GetChiralTag())
        ):
            return (), set(), set(), False
        neighbors = tuple(atom.GetNeighbors())
        if any(
            a.GetAtomicNum() != 6 and (a.GetIsotope() or int(a.GetChiralTag()))
            for a in neighbors
        ):
            return (
                (),
                set(),
                set(),
                False,
            )  # Central S/P annotations need element-specific bindings.
        double_o = [
            a.GetIdx()
            for a in neighbors
            if a.GetAtomicNum() == 8
            and not a.GetFormalCharge()
            and a.GetDegree() == 1
            and mol.GetBondBetweenAtoms(idx, a.GetIdx()).GetBondTypeAsDouble() == 2
        ]
        single_o = [
            a.GetIdx()
            for a in neighbors
            if a.GetAtomicNum() == 8
            and mol.GetBondBetweenAtoms(idx, a.GetIdx()).GetBondTypeAsDouble() == 1
        ]
        carbon = [
            a.GetIdx()
            for a in neighbors
            if a.GetAtomicNum() == 6
            and mol.GetBondBetweenAtoms(idx, a.GetIdx()).GetBondTypeAsDouble() == 1
        ]
        nitrogen = [
            a.GetIdx()
            for a in neighbors
            if a.GetAtomicNum() == 7
            and not a.GetFormalCharge()
            and mol.GetBondBetweenAtoms(idx, a.GetIdx()).GetBondTypeAsDouble() == 1
        ]
        hydroxyl = [
            i
            for i in single_o
            if mol.GetAtomWithIdx(i).GetDegree() == 1
            and (
                mol.GetAtomWithIdx(i).GetTotalNumHs() == 1
                or mol.GetAtomWithIdx(i).GetFormalCharge() == -1
            )
        ]
        anions = [i for i in hydroxyl if mol.GetAtomWithIdx(i).GetFormalCharge() == -1]
        heavy_neighbors = [a for a in neighbors if a.GetAtomicNum() > 1]
        kind = ""
        ligand = -1
        if number == 16:
            if (len(carbon) == 1 and len(neighbors) == 6
                    and all(a.GetAtomicNum() == 9 and a.GetDegree() == a.GetTotalValence() == 1
                            and not a.GetFormalCharge()
                            for a in neighbors if a.GetIdx() != carbon[0])):
                continue  # Carbon-bound pentafluorosulfanyl is cited as a prefix.
            if (len(double_o) == len(nitrogen) == 2 and len(neighbors) == 4
                    and all(mol.GetAtomWithIdx(i).GetDegree() == 1
                            and not mol.GetAtomWithIdx(i).GetFormalCharge() for i in double_o)
                    and all(mol.GetAtomWithIdx(i).GetTotalValence() == 3
                            and all(b.GetBondTypeAsDouble() in {1, 1.5}
                                    for b in mol.GetAtomWithIdx(i).GetBonds()) for i in nitrogen)
                    and (any(mol.GetAtomWithIdx(i).GetDegree() == 1
                             and mol.GetAtomWithIdx(i).GetTotalNumHs() == 2 for i in nitrogen)
                         or all(a.GetAtomicNum() == 6
                                and mol.GetBondBetweenAtoms(i, a.GetIdx()).GetBondTypeAsDouble() == 1
                                for i in nitrogen for a in mol.GetAtomWithIdx(i).GetNeighbors()
                                if a.GetIdx() != idx and a.GetAtomicNum() > 1))):
                # P-67.1.2.6.1: sulfuric diamide retains both N roles.
                # Keep both N roles out of the ordinary amine inventory.
                groups.append(CharacteristicGroup("sulfamide", idx,
                    (idx, *double_o, *nitrogen), -1))
                used_n.update(nitrogen)
                continue
            if (
                len(carbon) == 1
                and len(double_o) in {1, 2}
                and len(hydroxyl) == 1
                and len(neighbors) == len(double_o) + 2
            ):
                kind = (
                    ("sulfonate" if len(double_o) == 2 else "sulfinate")
                    if anions
                    else ("sulfonic-acid" if len(double_o) == 2 else "sulfinic-acid")
                )
                ligand = hydroxyl[0]
            elif (
                len(carbon) == 1
                and len(double_o) in {1, 2}
                and len(nitrogen) == 1
                and len(neighbors) == len(double_o) + 2
            ):
                if any(
                    b.GetBondTypeAsDouble() != 1
                    for b in mol.GetAtomWithIdx(nitrogen[0]).GetBonds()
                ) and (len(double_o) != 2 or sulfonamide_ylidene_root(mol, idx, nitrogen[0]) is None):
                    if len(double_o) == 2:
                        from rdkit import Chem
                        from .functional_inventory import _neutral_carboxyl_query
                        from .sulfonylguanidine import sulfonylguanidine

                        guanidine = sulfonylguanidine(mol, idx)
                        if guanidine is not None and mol.HasSubstructMatch(_neutral_carboxyl_query(Chem)):
                            used_n.update((guanidine.imino, *guanidine.amino))
                            continue
                    return (), set(), set(), False
                n_atom = mol.GetAtomWithIdx(nitrogen[0])
                n_nitroso = [a for a in n_atom.GetNeighbors()
                             if a.GetAtomicNum() == 7 and (nitrogen[0], a.GetIdx()) in oxo_pairs]
                if any(a.GetAtomicNum() == 7 and a.GetIdx() not in oxo_core_nitrogens
                       for a in n_atom.GetNeighbors()):
                    from .hydrazide_components import recognize_hydrazides
                    site = next((site for site in recognize_hydrazides(mol)
                                 if site.center == idx), None)
                    if site is None:
                        return (), set(), set(), False
                    groups.append(CharacteristicGroup(
                        "hydrazide", idx, site.atoms, carbon[0], nitrogen[0],
                        seniority_rank=1 if len(double_o) == 2 else 2))
                    used_n.update((site.acyl_nitrogen, site.terminal_nitrogen))
                    continue
                kind = "sulfonamide" if len(double_o) == 2 else "sulfinamide"
                ligand = nitrogen[0]
                proposed = CharacteristicGroup(
                    kind,
                    idx,
                    tuple([idx] + double_o + single_o + nitrogen),
                    carbon[0],
                    ligand,
                )
                if kind == "sulfinamide":
                    # P-66.1.1.2 recognizes the one-oxo amide separately
                    # from the two-oxo sulfonamide. The raw site recognizer
                    # proves only local class/atom roles; no branch spelling
                    # or whole-name eligibility participates in admission.
                    from rdkit import Chem
                    from .sulfinamide_components import recognize_sulfinamide_sites

                    sites = recognize_sulfinamide_sites(mol, Chem)
                    if not any(site.center == idx and site.owner == carbon[0]
                               and site.ligand == ligand
                               and frozenset(site.atoms) == frozenset(proposed.atoms)
                               for site in sites):
                        return (), set(), set(), False
                elif sulfonamide_n_sites(mol, proposed) is None:
                    # P-61.5.2 permits a nitroso prefix attached to the
                    # sulfonamide nitrogen. Preserve the parent group only
                    # when the N–N bond is the exact acyclic oxoamide core
                    # recognized above and its remaining N substituents are
                    # ordinary carbon groups within neutral N valence.
                    other = [a for a in n_atom.GetNeighbors()
                             if a.GetIdx() != idx and a.GetIdx() not in {x.GetIdx() for x in n_nitroso}]
                    if (len(n_nitroso) != 1 or n_atom.GetTotalValence() != 3
                            or any(a.GetAtomicNum() != 6 for a in other)
                            or len(other) > 1):
                        return (), set(), set(), False
                used_n.add(ligand)
            elif (
                not double_o
                and len(carbon) == 1
                and len(heavy_neighbors) == 1
                and atom.GetTotalNumHs(includeNeighbors=True) == 1
            ):
                kind = "thiol"
            elif (len(carbon) == 2 and len(double_o) == 1 and len(neighbors) == 4
                  and atom.GetTotalValence() == 6 and atom.GetTotalNumHs() == 0):
                imino = [neighbor for neighbor in neighbors
                         if neighbor.GetAtomicNum() == 7 and neighbor.GetTotalValence() == 3
                         and neighbor.GetDegree() in {1, 2}
                         and neighbor.GetTotalNumHs() == 2 - neighbor.GetDegree()
                         and not neighbor.GetFormalCharge() and not neighbor.IsInRing()
                         and all(n.GetIdx() == idx or n.GetAtomicNum() == 6
                                 and mol.GetBondBetweenAtoms(neighbor.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() == 1
                                 for n in neighbor.GetNeighbors())
                         and mol.GetBondBetweenAtoms(idx, neighbor.GetIdx()).GetBondTypeAsDouble() == 2]
                if len(imino) != 1:
                    return (), set(), set(), False
                if subordinate_heterones is None:
                    return (), set(), set(), False
                subordinate_heterones.add(idx)
                used_n.add(imino[0].GetIdx())
                # Admission is provisional: the completed inventory must have
                # a class senior to this P-64.4/P-68.4.3.6 heterone.
                continue
            elif (
                len(carbon) == 2
                and len(neighbors) == 2 + len(double_o)
                and len(double_o) <= 2
            ):
                continue  # Sulfide/sulfoxide/sulfone are not senior suffixes.
            elif (len(neighbors) == 2 and len(carbon) == 1 and atom.GetTotalValence() == 2
                  and any(a.GetAtomicNum() == 16 and not a.GetFormalCharge()
                          and not a.IsInRing() and a.GetDegree() == a.GetTotalValence() == 2
                          and any(n.GetAtomicNum() == 6 for n in a.GetNeighbors())
                          for a in neighbors)):
                continue  # The neutral C–S–S–C connector carries no senior site.
            else:
                return (), set(), set(), False
        elif (len(carbon) == len(heavy_neighbors) == 1
              and atom.GetTotalNumHs(includeNeighbors=True) == 2 and atom.GetTotalValence() == 3):
            continue  # A neutral phosphanyl substituent carries no senior suffix.
        elif len(carbon) == 3 and len(double_o) == 1 and len(neighbors) == 4:
            continue  # An ordinary R3P=O is an oxide, with no acidic OH.
        elif (
            len(double_o) == 1
            and len(hydroxyl) in {1, 2}
            and len(carbon) in {1, 2}
            and len(neighbors) == len(double_o) + len(hydroxyl) + len(carbon)
        ):
            kind = "phosphonate" if len(hydroxyl) == 2 else "phosphinate"
            if not anions:
                kind = kind.replace("ate", "ic-acid")
            ligand = hydroxyl[0]
        elif (
            len(double_o) == 1
            and len(single_o) == 3
            and len(hydroxyl) in {1, 2, 3}
            and len(neighbors) == 4
        ):
            kind = (
                "phosphate"
                if anions
                else (
                    "phosphoric-acid" if len(hydroxyl) == 3 else "phosphate-acid-ester"
                )
            )
        else:
            return (), set(), set(), False
        covered.update(anions)
        groups.append(
            CharacteristicGroup(
                kind,
                idx,
                tuple([idx] + double_o + single_o + nitrogen),
                carbon[0] if len(carbon) == 1 else -1,
                ligand,
            )
        )
    # A terminal carboxylate oxygen is handled by the carbonyl recognizer.
    for atom in iter_atoms(mol):
        if (
            atom.GetFormalCharge() != -1
            or atom.GetAtomicNum() != 8
            or atom.GetDegree() != 1
        ):
            continue
        other = atom.GetNeighbors()[0]
        if other.GetAtomicNum() == 6 and any(
            n.GetAtomicNum() == 8
            and mol.GetBondBetweenAtoms(
                other.GetIdx(), n.GetIdx()
            ).GetBondTypeAsDouble()
            == 2
            for n in other.GetNeighbors()
        ):
            covered.add(atom.GetIdx())
    if covered and not charge_role_proofs:
        # A neutral nitro group can occur alongside separately owned ionic
        # acid sites. Prove its internal charges by its original C–N cut.
        for atom in iter_atoms(mol):
            if atom.GetAtomicNum() != 7 or atom.GetFormalCharge() != 1 or atom.GetIdx() in covered:
                continue
            for host in atom.GetNeighbors():
                if host.GetAtomicNum() == 6:
                    proof = _recognize_at_carbon(mol, host.GetIdx(), atom.GetIdx())
                    if proof is not None and proof.kind == 'nitro':
                        covered.update(proof.charge_atom_ids)
    return (
        tuple(groups),
        covered,
        used_n,
        all(not a.GetFormalCharge() or a.GetIdx() in covered for a in iter_atoms(mol)),
    )
