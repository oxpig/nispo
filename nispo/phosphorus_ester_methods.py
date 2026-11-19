"""Mapped construction of bounded neutral mononuclear phosphorus acid esters.

P-67.1.3.2 places each O-bound organyl component before the anion and, for
partial esters, inserts the separate hydrogen word(s) immediately before the
anion. This producer handles simple ordinary O-esters of phosphoric,
phosphonic, and phosphinic acid, with mapped carbon-only O-organyl branches.
"""

from collections import defaultdict
from dataclasses import dataclass

from .molecule_iteration import iter_atoms, iter_bonds
from .naming_methods import MethodDecision

RULE = "P-67.1.1.1/P-67.1.1.2/P-67.1.3.2/P-65.6.3.2.1/P-65.6.3.3.2.1/P-14.5/P-13.6.2"
ISOTOPE_RULE = "P-82.2.1/P-82.2.4/P-82.6.1.2/P-82.6.4/P-83.1.2.2"


@dataclass(frozen=True, slots=True)
class PhosphorusEsterIsotopeDecision(MethodDecision):
    """Original isotope owners plus source-required secondary citations."""

    isotope_components: tuple = ()
    isotope_coverage: object | None = None
    isotope_citations: object | None = None


def _carbon_component(mol, start, blocked):
    """Return a disjoint neutral carbon-only branch, or None."""
    atoms, pending = set(), [start]
    while pending:
        idx = pending.pop()
        if idx in atoms:
            continue
        atom = mol.GetAtomWithIdx(idx)
        if (atom.GetAtomicNum() != 6 or atom.GetFormalCharge()
                or atom.GetNumRadicalElectrons() or atom.GetIsotope()
                or int(atom.GetChiralTag())):
            return None
        atoms.add(idx)
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            j = other.GetIdx()
            if j == blocked or other.GetAtomicNum() == 1:
                continue
            if other.GetAtomicNum() != 6:
                return None
            pending.append(j)
    if not atoms:
        return None
    for idx in atoms:
        atom = mol.GetAtomWithIdx(idx)
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            if other.GetAtomicNum() == 1:
                continue
            if other.GetIdx() == blocked:
                if bond.GetBondTypeAsDouble() != 1:
                    return None
            elif other.GetIdx() in atoms:
                if not (bond.GetIsAromatic() or bond.GetBondTypeAsDouble() in {1.0, 2.0}):
                    return None
            else:
                return None
    return frozenset(atoms)


def _recognize(mol, Chem):
    if mol.GetNumHeavyAtoms() > 128 or mol.GetStereoGroups():
        return None
    if any(a.GetFormalCharge() or a.GetNumRadicalElectrons()
           or int(a.GetChiralTag()) for a in iter_atoms(mol)):
        return None
    if any(b.GetStereo() != Chem.BondStereo.STEREONONE for b in iter_bonds(mol)):
        return None
    # The source-closed isotope route is the P-83.1.2.2 phosphonate pattern:
    # one labeled P-O-C bridge, with ordinary phosphoryl oxygen and no residual
    # P-OH groups. Other isotope positions/combinations remain pending.
    isotope_atoms = [a for a in iter_atoms(mol) if a.GetIsotope()]
    periodic = Chem.GetPeriodicTable()
    if any(a.GetAtomicNum() != 8 or a.GetSymbol() != "O"
           or periodic.GetMassForIsotope(8, a.GetIsotope()) <= 0
           for a in isotope_atoms):
        return None
    # This operation is deliberately limited to P/O/C/H.
    if any(a.GetAtomicNum() not in {1, 6, 8, 15} for a in iter_atoms(mol)):
        return None
    ps = [a for a in iter_atoms(mol) if a.GetAtomicNum() == 15]
    if len(ps) != 1:
        return None
    p = ps[0]
    pidx = p.GetIdx()
    if p.IsInRing():
        return None
    roots, ester_oxygens, hydroxyls, double_oxygens, p_carbon_roots = [], [], [], [], []
    for nbr in p.GetNeighbors():
        idx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(pidx, idx)
        if nbr.GetAtomicNum() == 6 and bond.GetBondTypeAsDouble() == 1:
            p_carbon_roots.append(idx)
            continue
        if nbr.GetAtomicNum() != 8 or nbr.IsInRing():
            return None
        if bond.GetBondTypeAsDouble() == 2 and nbr.GetDegree() == 1:
            double_oxygens.append(idx)
            continue
        if bond.GetBondTypeAsDouble() != 1:
            return None
        heavy = [a for a in nbr.GetNeighbors() if a.GetAtomicNum() > 1 and a.GetIdx() != pidx]
        if not heavy:
            if nbr.GetTotalNumHs(includeNeighbors=True) != 1:
                return None
            # Explicit ordinary hydrogen atoms are outside this bounded mapping.
            if any(a.GetAtomicNum() == 1 for a in nbr.GetNeighbors()):
                return None
            hydroxyls.append(idx)
        elif len(heavy) == 1 and heavy[0].GetAtomicNum() == 6:
            root = heavy[0].GetIdx()
            branch = _carbon_component(mol, root, idx)
            if branch is None:
                return None
            ester_oxygens.append((idx, root, branch))
        else:
            return None
    if len(double_oxygens) != 1 or not ester_oxygens:
        return None
    labeled = {a.GetIdx() for a in isotope_atoms}
    if len(p_carbon_roots) > 2:
        return None
    # Existing mapped P-acid construction proves only these two substituted cores.
    single_o_count = len(hydroxyls) + len(ester_oxygens)
    p_hydrogens = p.GetTotalNumHs(includeNeighbors=True)
    substitutions = len(p_carbon_roots) + p_hydrogens
    if substitutions == 0 and single_o_count == 3:
        acid_kind = "phosphoric acid"
    elif substitutions == 1 and single_o_count == 2:
        acid_kind = "phosphonic acid"
    elif substitutions == 2 and single_o_count == 1:
        acid_kind = "phosphinic acid"
    else:
        return None
    if labeled:
        bridge_ids = {row[0] for row in ester_oxygens}
        if (len(labeled) != 1 or not labeled <= bridge_ids
                or acid_kind != "phosphonic acid" or hydroxyls
                or len(ester_oxygens) != 2 or len(p_carbon_roots) != 1
                or p_hydrogens or double_oxygens[0] in labeled):
            return None
    components = [atoms for _o, _root, atoms in ester_oxygens]
    for root in p_carbon_roots:
        branch = _carbon_component(mol, root, pidx)
        if branch is None:
            return None
        components.append(branch)
    for i, atoms in enumerate(components):
        if any(atoms & other for other in components[i + 1:]):
            return None
    core = {pidx, double_oxygens[0], *hydroxyls, *p_carbon_roots}
    for branch in components[len(ester_oxygens):]:
        core.update(branch)
    covered = set(core)
    for oxygen, _root, atoms in ester_oxygens:
        covered.add(oxygen)
        covered.update(atoms)
    for root, atoms in zip(p_carbon_roots, components[len(ester_oxygens):]):
        covered.update(atoms)
    heavy = {a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1}
    if covered != heavy:
        return None
    return {
        "p": pidx, "double_o": double_oxygens[0], "hydroxyls": tuple(hydroxyls),
        "ester_oxygens": tuple(ester_oxygens), "p_roots": tuple(p_carbon_roots),
        "acid_kind": acid_kind, "core_atoms": frozenset(core),
        "p_hydrogens": p_hydrogens,
        "p_components": tuple(components[len(ester_oxygens):]),
        "isotope_atoms": tuple(sorted(labeled)),
    }


def recognize_phosphorus_ester(mol, Chem=None):
    """Return a structural site record without rendering, or ``None``."""
    if Chem is None:
        from rdkit import Chem
    return _recognize(mol, Chem)


def _acid_core_parts(mol, Chem, site):
    anion = {"phosphoric acid": "phosphate", "phosphonic acid": "phosphonate",
             "phosphinic acid": "phosphinate"}[site["acid_kind"]]
    if not site["p_roots"]:
        return "", anion
    # Work on a private copy: replace each P-O-C arm by a terminal O-H,
    # retaining direct P-C arms for the existing mapped acid producer.
    copy = Chem.RWMol(mol)
    # This private graph is used only to render the ordinary anion identity.
    # Clear source-identified O labels here rather than allowing the acid
    # writer to move them onto the artificial hydroxylated copy component.
    for isotope_atom in site["isotope_atoms"]:
        copy.GetAtomWithIdx(isotope_atom).SetIsotope(0)
    removed = set()
    for oxygen, root, atoms in site["ester_oxygens"]:
        copy.RemoveBond(oxygen, root)
        # Re-express the former ester bridge as the acid hydroxyl in the
        # temporary acid graph; its original atom and all output roles remain
        # bound to the unmodified input graph.
        capped_oxygen = copy.GetAtomWithIdx(oxygen)
        # The acid-core renderer sees only the ordinary chemical identity.
        # The original bridge isotope is retained by the O-organyl component
        # ledger below and is never copied onto the artificial OH group.
        capped_oxygen.SetAtomicNum(8)
        capped_oxygen.SetIsotope(0)
        capped_oxygen.SetNoImplicit(False)
        capped_oxygen.SetNumExplicitHs(0)
        removed.update(atoms)
    old_to_new = {idx: idx - sum(deleted < idx for deleted in removed)
                  for idx in range(mol.GetNumAtoms()) if idx not in removed}
    for idx in sorted(removed, reverse=True):
        copy.RemoveAtom(idx)
    acid = copy.GetMol()
    try:
        Chem.SanitizeMol(acid)
    except Exception:
        return None
    # Check the copy's P and every capped O against the original IDs before
    # using its acid-name construction. Nothing from this graph is returned
    # as output ownership or an annotation map.
    for original in (site["p"], *(o for o, _root, _atoms in site["ester_oxygens"])):
        new_idx = old_to_new.get(original)
        expected = 15 if original == site['p'] else 8
        if new_idx is None or acid.GetAtomWithIdx(new_idx).GetAtomicNum() != expected:
            return None
    for oxygen, _root, _atoms in site["ester_oxygens"]:
        capped = acid.GetAtomWithIdx(old_to_new[oxygen])
        p_new = old_to_new[site["p"]]
        if (acid.GetBondBetweenAtoms(p_new, capped.GetIdx()) is None
                or capped.GetDegree() != 1
                or capped.GetTotalNumHs(includeNeighbors=True) != 1):
            return None
    # A phosphoryl label belongs to the original anion component, not to the
    # acid copy used to establish its ordinary name.
    for isotope_atom in site["isotope_atoms"]:
        mapped = old_to_new.get(isotope_atom)
        if mapped is None or acid.GetAtomWithIdx(mapped).GetIsotope():
            return None
    from .phosphorus_acid_methods import phosphorus_acid_decision
    decision = phosphorus_acid_decision(acid, Chem)
    if decision is None:
        return None
    acid_ending = {"phosphoric acid": "phosphoric acid",
                   "phosphonic acid": "phosphonic acid",
                   "phosphinic acid": "phosphinic acid"}[site["acid_kind"]]
    if not decision.name.endswith(acid_ending):
        return None
    prefix = decision.name[:-len(acid_ending)]
    return prefix, anion


def _decorate_phosphorus_anion_oxygen(name, mass):
    """Insert the P-83 cited oxygen descriptor before a proved anion suffix.

    The caller preserves the independently constructed P-bound organyl
    prefix and passes only the base anion word. The isotope is owned by the
    O-organyl bridge component and cited again at the anion boundary, matching
    the P-83.1.2.2 source role pattern.
    """
    for suffix in ("phosphonate", "phosphinate", "phosphate"):
        if name.endswith(suffix):
            return name[:-len(suffix)] + f"({mass}O)" + suffix
    return ""


def _branch_equivalence(mol, Chem, atoms, root):
    copy = Chem.Mol(mol)
    for atom in copy.GetAtoms():
        atom.SetIsotope(0)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for bond in copy.GetBonds():
        bond.SetStereo(Chem.BondStereo.STEREONONE)
        bond.SetBondDir(Chem.BondDir.NONE)
    return Chem.MolFragmentToSmiles(copy, atomsToUse=sorted(atoms), rootedAtAtom=root,
                                    canonical=True, isomericSmiles=False)


def _organyl_words(mol, Chem, site, *, general=False):
    from .component_mapping import mapped_branch, mapped_general_branch
    from .formatting import prefix_alphabetization_key, multiply_component

    rows = []
    branch = mapped_general_branch if general else mapped_branch
    for oxygen, root, atoms in site["ester_oxygens"]:
        text, owned = branch(mol, Chem, root, oxygen)
        heavy_owned = {idx for idx in owned if mol.GetAtomWithIdx(idx).GetAtomicNum() > 1}
        if not text or any(char.isspace() for char in text) or heavy_owned != set(atoms):
            return None
        if general:
            from .polyacid_ester_methods import _organyl_identity

            signature = _organyl_identity(mol, Chem, atoms, oxygen, root)
            if not signature:
                return None
        else:
            signature = _branch_equivalence(mol, Chem, atoms, root)
        ordinary_text = text
        isotope = mol.GetAtomWithIdx(oxygen).GetIsotope()
        if isotope:
            from .oxygen_isotope_suffixes import apply_organyl_oxygen_qualifier
            qualifier = f"{isotope}O-"
            text = apply_organyl_oxygen_qualifier(text, qualifier)
            if not text:
                return None
        rows.append({"text": text, "ordinary_text": ordinary_text,
                     "atoms": set(atoms), "owned": set(owned) | {oxygen},
                     "oxygen": oxygen, "root": root, "signature": signature,
                     "isotope": isotope})
    # A word's structural multiplicity is supported only when both its rooted
    # graph and component text agree; don't infer identity from spelling.
    buckets = {}
    for row in rows:
        if general and any(other['ordinary_text'] == row['ordinary_text']
                           and other['signature'] != row['signature'] for other in rows):
            return None
        if row["isotope"] and any(
                other["ordinary_text"] == row["ordinary_text"] and other is not row
                for other in rows):
            # A modified/unmodified copy of the same ordinary ester word has
            # no proved P-67/P-82 ordering or multiplication operation. This
            # also guards homonymous branches with different rooted graphs.
            return None
        key = (row["signature"], row["text"])
        buckets.setdefault(key, []).append(row)
    words = []
    for (_signature, text), grouped in buckets.items():
        if len(grouped) == 1:
            word = text
        elif len(grouped) >= 2:
            from .substituent_multiplication import terminally_substituted_carbon_component
            nature = ('compound' if terminally_substituted_carbon_component(
                mol, grouped[0]['atoms']) else None)
            word = multiply_component(text, len(grouped), nature=nature)
            if not word:
                return None
        else:
            return None
        words.append((prefix_alphabetization_key(grouped[0]["ordinary_text"]), word, grouped))
    words.sort(key=lambda item: item[0])
    return words


def phosphorus_ester_decision(mol, Chem=None):
    """Return a mapped P-67.1.3.2 ester decision, or ``None``."""
    if Chem is None:
        from rdkit import Chem
    site = _recognize(mol, Chem)
    if site is None:
        return None
    core_parts = _acid_core_parts(mol, Chem, site)
    if core_parts is None:
        return None
    core_prefix, anion_unit = core_parts
    core_name = core_prefix + anion_unit
    words = _organyl_words(mol, Chem, site)
    if words is None:
        return None
    hydrogen_count = len(site["hydroxyls"])
    hydrogen = {0: "", 1: "hydrogen", 2: "dihydrogen", 3: "trihydrogen"}.get(hydrogen_count)
    if hydrogen is None:
        return None
    isotope_atoms = set(site["isotope_atoms"])
    if isotope_atoms:
        # P-83.1.2.2 names the same bridge oxygen in its O-organyl word and
        # cites it again on the fully esterified phosphonate anion. This is a
        # secondary reference to the bridge atom, never a P=O isotope claim.
        bridge_id = next(iter(isotope_atoms))
        mass = mol.GetAtomWithIdx(bridge_id).GetIsotope()
        decorated_anion = _decorate_phosphorus_anion_oxygen(anion_unit, mass)
        if not decorated_anion:
            return None
        core_name = core_prefix + decorated_anion
    isotope_records = ()
    coverage = None
    citation_ledger = None
    if isotope_atoms:
        from .isotope_component_coverage import (
            IsotopeComponentRecord, IsotopeDescriptorAtom,
        )
        from .isotope_citation_ledger import (
            IsotopeCitationReference, IsotopeCitationRequirement,
            IsotopeCitationRole, IsotopeCitationRule,
            PhosphorusBridgeOxygenCitationProof,
            validate_isotope_citation_ledger,
        )
        records = []
        bridge_id = next(iter(isotope_atoms))
        selected = next((row for _key, _word, grouped in words for row in grouped
                         if row["oxygen"] == bridge_id), None)
        if selected is None:
            return None
        atom = mol.GetAtomWithIdx(bridge_id)
        organyl_id = f"O-organyl:{bridge_id}"
        anion_id = "phosphorus-anion"
        records.append(IsotopeComponentRecord(
            component_id=organyl_id, component_text=selected["text"],
            atom_ids=frozenset({bridge_id}), atom_locants=((bridge_id, "O"),),
            isotope_atoms=(IsotopeDescriptorAtom(bridge_id, "O", atom.GetIsotope(),
                                                 "O", None),)))
        nonisotopic_core = set(site["core_atoms"])
        if bridge_id in nonisotopic_core:
            return None
        records.append(IsotopeComponentRecord(
            component_id=anion_id, component_text=core_name,
            atom_ids=frozenset(nonisotopic_core),
            atom_locants=tuple((idx, None) for idx in sorted(nonisotopic_core))))
        proof = PhosphorusBridgeOxygenCitationProof(
            phosphorus=site["p"], bridge_oxygen=bridge_id,
            organyl_carbon=selected["root"], phosphoryl_oxygen=site["double_o"])
        requirement = IsotopeCitationRequirement(
            atom_id=bridge_id, source_component_id=organyl_id,
            target_component_id=anion_id,
            rule=IsotopeCitationRule.P83_1_2_2_PHOSPHORUS_ESTER,
            source_role=IsotopeCitationRole.ORGANYL_BRIDGE_OXYGEN,
            target_role=IsotopeCitationRole.PHOSPHORUS_ANION_OXYGEN_DESCRIPTOR,
            role_proof=proof)
        reference = IsotopeCitationReference(
            atom_id=bridge_id, element="O", mass=atom.GetIsotope(),
            source_component_id=organyl_id, target_component_id=anion_id,
            rule=IsotopeCitationRule.P83_1_2_2_PHOSPHORUS_ESTER,
            source_role=IsotopeCitationRole.ORGANYL_BRIDGE_OXYGEN,
            target_role=IsotopeCitationRole.PHOSPHORUS_ANION_OXYGEN_DESCRIPTOR,
            role_proof=proof)
        citation_ledger = validate_isotope_citation_ledger(
            mol, tuple(records), (requirement,), (reference,))
        coverage = citation_ledger.primary_coverage
        if (not citation_ledger.complete
                or set(coverage.covered_atom_ids) != isotope_atoms):
            return None
        isotope_records = tuple(records)
    organyl_words = [row[1] for row in words]
    if isotope_atoms:
        # In the P-83 source construction, an ordinary O-linked organyl in a
        # phosphonate with a separately P-bound organyl is explicitly marked
        # O-; a bridge isotope replaces that locant with the nuclide-O form.
        # Keep this rule-local to the source-closed isotope operation.
        organyl_words = [
            word if any(row["isotope"] for row in grouped) else f"O-{word}"
            for _key, word, grouped in words
        ]
    name = " ".join([*organyl_words, *([hydrogen] if hydrogen else []), core_name])
    core_atoms = set(site["core_atoms"])
    role_rows = [("phosphorus-acid-anion", frozenset(core_atoms),
                  tuple((idx, "") for idx in sorted(core_atoms)))]
    for index, (_key, _word, grouped) in enumerate(words, 1):
        owned = frozenset().union(*(frozenset(row["owned"]) for row in grouped))
        role_rows.append((f"O-organyl:{index}", owned, ()))
    heavy = frozenset(a.GetIdx() for a in iter_atoms(mol) if a.GetAtomicNum() > 1)
    if frozenset(core_atoms).union(*(role[1] for role in role_rows[1:])) != heavy:
        return None
    # Ensure every role is disjoint, with the complete map covering all heavy atoms.
    role_atom_sets = [set(row[1]) for row in role_rows]
    if any(role_atom_sets[i] & role_atom_sets[j]
           for i in range(len(role_atom_sets)) for j in range(i + 1, len(role_atom_sets))):
        return None
    if set().union(*role_atom_sets) != set(heavy):
        return None
    has_isotopes = bool(isotope_atoms)
    decision_type = PhosphorusEsterIsotopeDecision if has_isotopes else MethodDecision
    kwargs = {}
    if has_isotopes:
        kwargs = {"isotope_components": isotope_records,
                  "isotope_coverage": coverage,
                  "isotope_citations": citation_ledger}
    return decision_type(name, "functional-class", frozenset({site["p"]}), heavy,
                         RULE + ("/" + ISOTOPE_RULE if has_isotopes else ""),
                         tuple((idx, "") for idx in sorted(core_atoms)),
                         component_roles=tuple(role_rows), **kwargs)


standardMethodDecision = phosphorus_ester_decision
