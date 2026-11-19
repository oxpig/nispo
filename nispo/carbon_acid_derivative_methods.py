"""Mapped, source-bounded residual-OH carbon-acid derivatives.

P-65.1.8 distinguishes permitted substitution of retained formic acid from
groups that must be named as carbonic-acid functional replacements. This
module admits only exact, neutral one-center source families and keeps their
class identity separate from the graph's superficially similar carboxylic
acid inventory records.
"""

from dataclasses import dataclass

from .molecule_iteration import iter_atoms, iter_bonds
from .naming_methods import MethodDecision


_HALIDE_NAMES = {9: "fluorid", 17: "chlorid", 35: "bromid", 53: "iodid"}
_RULE_HALO = "P-65.1.8.1/P-65.2.1.4/P-41"
_RULE_AMINO = "P-65.2.1.1/P-65.2.1.4/P-41"
_RULE_NITRO = "P-65.1.8.2/P-41"
_RULE_PSEUDOHALIDE = "P-65.1.8.1/P-65.2.1.4/P-16.7/P-41"

# The infix vocabulary is finite and source-listed. Its acid adjective is
# formed by changing terminal -o to -ic (P-16.7.1(e)); P-16.8.1 supplies an
# euphonic o between consonants. This reproduces the source-printed
# carbonocyanidic, carbonazidic, and carbonisothiocyanatidic forms.
_PSEUDOHALIDE_INFICES = {
    "cyanido": "cyanido",
    "azido": "azido",
    "isocyanido": "isocyanido",
    "isocyanatido": "isocyanatido",
    "isothiocyanatido": "isothiocyanatido",
    "isoselenocyanatido": "isoselenocyanatido",
    "isotellurocyanatido": "isotellurocyanatido",
}


def _pseudohalide_acid_name(family):
    infix = _PSEUDOHALIDE_INFICES[family]
    adjective = infix[:-1] + "ic"
    linker = "o" if adjective[0] not in "aeiou" else ""
    return "carbon" + linker + adjective + " acid"


@dataclass(frozen=True, slots=True)
class CarbonAcidDerivativeFrontier:
    """Status-bearing result for a recognized source-class construction."""

    status: str
    rule: str
    classification: str
    center: int | None = None
    residual_oxygen: int | None = None
    carbonyl_oxygen: int | None = None
    replacement_atoms: frozenset[int] = frozenset()
    roles: tuple = ()
    decision: MethodDecision | None = None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class CarbonAcidSourceClass:
    """Pure source taxonomy for an exact mononuclear carbon-acid graph.

    ``classification`` is the semantic source class: ``carboxylic-acid`` for
    retained formic-acid substitutions and ``carbonic-acid`` for carbonic-acid
    functional replacements. ``principal_kind`` uses the inventory enum.
    """

    classification: str
    principal_kind: str
    ligand_family: str
    rule: str
    center: int
    carbonyl_oxygen: int
    residual_oxygen: int
    ligand_atoms: frozenset[int]
    core_atoms: frozenset[int]
    all_heavy_atoms: frozenset[int]


def carbon_acid_source_class(mol, Chem):
    """Classify exact P-65.1.8/P-65.2.1 carbon-acid topologies.

    This API deliberately performs no inventory or naming call. Its class
    result comes from source-specific connectivity: permitted nitro and
    hydrazino formic-acid operations are carboxylic-acid class, while the
    listed carbonic-acid replacements are carbon-acid class.
    """
    if (mol is None or mol.GetNumAtoms() == 0
            or mol.GetNumHeavyAtoms() > 128
            or len(Chem.GetMolFrags(mol, asMols=False)) != 1):
        return None
    candidates = []
    for center, carbonyl_o in _carbonyl_centers(mol):
        carbon = mol.GetAtomWithIdx(center)
        if (carbon.IsInRing() or carbon.GetFormalCharge()
                or carbon.GetNumRadicalElectrons()):
            continue
        carbonyl = mol.GetAtomWithIdx(carbonyl_o)
        if (carbonyl.GetFormalCharge() or carbonyl.GetDegree() != 1
                or carbonyl.GetNumRadicalElectrons()
                or mol.GetBondBetweenAtoms(center, carbonyl_o).GetBondTypeAsDouble() != 2):
            continue
        single_o = _single_bonded_atoms(mol, center, 8)
        if len(single_o) not in {1, 2}:
            continue
        if len(single_o) == 1:
            residual_o = single_o[0]
        else:
            # P-65.2.1.2 peroxo replacement has two C–O single bonds:
            # one residual acid hydroxy group and one O–OH replacement.
            ordinary_oh = [idx for idx in single_o
                           if _heavy_degree(mol.GetAtomWithIdx(idx)) == 1
                           and mol.GetAtomWithIdx(idx).GetTotalNumHs(includeNeighbors=True) == 1]
            if len(ordinary_oh) != 1:
                continue
            residual_o = ordinary_oh[0]
        oxygen = mol.GetAtomWithIdx(residual_o)
        if (oxygen.GetFormalCharge()
                or sum(n.GetAtomicNum() > 1 for n in oxygen.GetNeighbors()) != 1
                or oxygen.GetNumRadicalElectrons()
                or oxygen.GetTotalNumHs(includeNeighbors=True) != 1
                or mol.GetBondBetweenAtoms(center, residual_o).GetBondTypeAsDouble() != 1):
            continue
        other = [n.GetIdx() for n in carbon.GetNeighbors()
                 if n.GetIdx() not in {carbonyl_o, residual_o}]
        if len(other) != 1 or len([n for n in carbon.GetNeighbors()
                                   if n.GetAtomicNum() > 1]) != 3:
            continue
        root = other[0]
        branch = _source_replacement_ligand(mol, center, root)
        if branch is None:
            continue
        ligand_family, ligand_atoms, principal_kind, classification, rule = branch
        core = frozenset({center, carbonyl_o, residual_o})
        all_heavy = frozenset(atom.GetIdx() for atom in iter_atoms(mol)
                              if atom.GetAtomicNum() > 1)
        if core | ligand_atoms != all_heavy or core & ligand_atoms:
            continue
        candidates.append(CarbonAcidSourceClass(
            classification, principal_kind, ligand_family, rule,
            center, carbonyl_o, residual_o, ligand_atoms, core, all_heavy,
        ))
    return candidates[0] if len(candidates) == 1 else None


def _source_replacement_ligand(mol, center, root):
    atom = mol.GetAtomWithIdx(root)
    number = atom.GetAtomicNum()
    charges = sum(a.GetFormalCharge() for a in iter_atoms(mol))
    if (number in _HALIDE_NAMES and _heavy_degree(atom) == 1
            and not atom.GetFormalCharge() and not atom.GetNumRadicalElectrons()):
        return ("halogen", frozenset({root}), "carbonic-acid",
                "carbonic-acid", _RULE_HALO)
    if (number == 7 and not atom.GetFormalCharge() and _heavy_degree(atom) == 1
            and not atom.GetNumRadicalElectrons()
            and atom.GetTotalNumHs(includeNeighbors=True) == 2):
        return ("amino", frozenset({root}), "carbonic-acid",
                "carbonic-acid", _RULE_AMINO)
    nitro = _nitro_branch(mol, center, root)
    if nitro is not None and charges == 0:
        return ("nitro", nitro, "acid", "carboxylic-acid",
                "P-65.1.8.2/P-41")
    if number == 6:
        neighbors = [n for n in atom.GetNeighbors() if n.GetIdx() != center]
        if (not atom.GetFormalCharge() and not atom.GetNumRadicalElectrons()
                and _heavy_degree(atom) == 2
                and len(neighbors) == 1 and neighbors[0].GetAtomicNum() == 7
                and not neighbors[0].GetFormalCharge()
                and _heavy_degree(neighbors[0]) == 1
                and mol.GetBondBetweenAtoms(root, neighbors[0].GetIdx()).GetBondTypeAsDouble() == 3):
            return ("cyanido", frozenset({root, neighbors[0].GetIdx()}),
                    "carbonic-acid", "carbonic-acid",
                    _RULE_PSEUDOHALIDE)
    if number == 7 and _heavy_degree(atom) == 2:
        carbon_neighbors = [n for n in atom.GetNeighbors()
                            if n.GetAtomicNum() == 6 and n.GetIdx() != center
                            and mol.GetBondBetweenAtoms(root, n.GetIdx()).GetBondTypeAsDouble() == 3]
        if len(carbon_neighbors) == 1:
            terminal_c = carbon_neighbors[0]
            # Neutral isocyanide valence uses the formal-charge-separated form.
            if (atom.GetFormalCharge() == 1 and not atom.GetNumRadicalElectrons()
                    and terminal_c.GetFormalCharge() == -1
                    and not terminal_c.GetNumRadicalElectrons()
                    and charges == 0 and _heavy_degree(terminal_c) == 1):
                return ("isocyanido", frozenset({root, terminal_c.GetIdx()}),
                        "carbonic-acid", "carbonic-acid",
                        _RULE_PSEUDOHALIDE)
        # The source infixes –NCO, –NCS, –NCSe, and –NCTe are admitted only
        # in the neutral linear R–N=C=X form, where R is the acid carbon and
        # X is the listed terminal chalcogen.
        if not atom.GetFormalCharge() and not atom.GetNumRadicalElectrons():
            attached = [n for n in atom.GetNeighbors() if n.GetIdx() != center]
            if (len(attached) == 1 and attached[0].GetAtomicNum() == 6
                    and mol.GetBondBetweenAtoms(center, root).GetBondTypeAsDouble() == 1
                    and mol.GetBondBetweenAtoms(root, attached[0].GetIdx()).GetBondTypeAsDouble() == 2
                    and _heavy_degree(atom) == 2 and charges == 0):
                central = attached[0]
                terminals = [n for n in central.GetNeighbors() if n.GetIdx() != root]
                if (central.GetFormalCharge() == 0 and not central.GetNumRadicalElectrons()
                        and _heavy_degree(central) == 2 and len(terminals) == 1):
                    terminal = terminals[0]
                    terminal_number = terminal.GetAtomicNum()
                    family_by_terminal = {8: "isocyanatido", 16: "isothiocyanatido",
                                          34: "isoselenocyanatido", 52: "isotellurocyanatido"}
                    if (terminal_number in family_by_terminal
                            and mol.GetBondBetweenAtoms(central.GetIdx(), terminal.GetIdx()).GetBondTypeAsDouble() == 2
                            and terminal.GetFormalCharge() == 0
                            and not terminal.GetNumRadicalElectrons()
                            and _heavy_degree(terminal) == 1
                            and terminal.GetTotalNumHs(includeNeighbors=True) == 0):
                        family = family_by_terminal[terminal_number]
                        return (family, frozenset({root, central.GetIdx(), terminal.GetIdx()}),
                                "carbonic-acid", "carbonic-acid", _RULE_PSEUDOHALIDE)
        n_neighbors = [n for n in atom.GetNeighbors()
                       if n.GetAtomicNum() == 7 and n.GetIdx() != center]
        if (len(n_neighbors) == 1 and not atom.GetFormalCharge()
                and not atom.GetNumRadicalElectrons() and charges == 0):
            terminal = n_neighbors[0]
            if (mol.GetBondBetweenAtoms(root, terminal.GetIdx()).GetBondTypeAsDouble() == 1
                    and _heavy_degree(terminal) == 1 and not terminal.GetFormalCharge()
                    and not terminal.GetNumRadicalElectrons()
                    and terminal.GetTotalNumHs(includeNeighbors=True) == 2):
                return ("hydrazino", frozenset({root, terminal.GetIdx()}),
                        "acid", "carboxylic-acid",
                        "P-65.1.8.1/P-41/P-68.3.1.2")
        n_neighbors = [n for n in atom.GetNeighbors()
                       if n.GetAtomicNum() == 7 and n.GetIdx() != center]
        if (len(n_neighbors) == 1 and not atom.GetFormalCharge()
                and not atom.GetNumRadicalElectrons() and charges == 0):
            middle = n_neighbors[0]
            terminal = [n for n in middle.GetNeighbors() if n.GetIdx() != root]
            if (middle.GetFormalCharge() == 1 and _heavy_degree(middle) == 2
                    and len(terminal) == 1 and terminal[0].GetAtomicNum() == 7
                    and terminal[0].GetFormalCharge() == -1
                    and _heavy_degree(terminal[0]) == 1
                    and mol.GetBondBetweenAtoms(root, middle.GetIdx()).GetBondTypeAsDouble() == 2
                    and mol.GetBondBetweenAtoms(middle.GetIdx(), terminal[0].GetIdx()).GetBondTypeAsDouble() == 2):
                return ("azido", frozenset({root, middle.GetIdx(), terminal[0].GetIdx()}),
                        "carbonic-acid", "carbonic-acid",
                        _RULE_PSEUDOHALIDE)
    if (number in {16, 34, 52} and not atom.GetFormalCharge()
            and not atom.GetNumRadicalElectrons() and _heavy_degree(atom) == 1):
        if atom.GetTotalNumHs(includeNeighbors=True) == 1:
            suffix = {16: "thio", 34: "seleno", 52: "telluro"}[number]
            return (suffix, frozenset({root}), "carbonic-acid",
                    "carbonic-acid", "P-65.1.8.1/P-65.2.1.2")
    if (number == 8 and not atom.GetFormalCharge()
            and not atom.GetNumRadicalElectrons() and _heavy_degree(atom) == 2):
        others = [n for n in atom.GetNeighbors() if n.GetIdx() != center]
        if (len(others) == 1 and others[0].GetAtomicNum() == 8
                and not others[0].GetFormalCharge() and _heavy_degree(others[0]) == 1
                and others[0].GetTotalNumHs(includeNeighbors=True) == 1
                and mol.GetBondBetweenAtoms(root, others[0].GetIdx()).GetBondTypeAsDouble() == 1):
            return ("peroxo", frozenset({root, others[0].GetIdx()}),
                    "carbonic-acid", "carbonic-acid",
                    "P-65.2.1.2")
    return None


def _single_bonded_atoms(mol, center, atomic_number):
    return tuple(sorted(
        atom.GetIdx() for atom in mol.GetAtomWithIdx(center).GetNeighbors()
        if atom.GetAtomicNum() == atomic_number
        and mol.GetBondBetweenAtoms(center, atom.GetIdx()).GetBondTypeAsDouble() == 1
    ))


def _heavy_degree(atom):
    return sum(neighbor.GetAtomicNum() > 1 for neighbor in atom.GetNeighbors())


def _carbonyl_centers(mol):
    result = []
    for atom in iter_atoms(mol):
        if atom.GetAtomicNum() != 6 or atom.GetFormalCharge() or atom.GetNumRadicalElectrons():
            continue
        doubles = [neighbor.GetIdx() for neighbor in atom.GetNeighbors()
                   if neighbor.GetAtomicNum() == 8
                   and mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondTypeAsDouble() == 2]
        if len(doubles) == 1:
            result.append((atom.GetIdx(), doubles[0]))
    return tuple(result)


def _nitro_branch(mol, center, nitrogen):
    """Recognize only the neutral nitro connectivity shown at P-65.1.8.2."""
    atom = mol.GetAtomWithIdx(nitrogen)
    if atom.GetAtomicNum() != 7 or atom.GetFormalCharge() != 1 or _heavy_degree(atom) != 3:
        return None
    if mol.GetBondBetweenAtoms(center, nitrogen).GetBondTypeAsDouble() != 1:
        return None
    double_o, anionic_o = [], []
    for neighbor in atom.GetNeighbors():
        idx = neighbor.GetIdx()
        if idx == center:
            continue
        order = mol.GetBondBetweenAtoms(nitrogen, idx).GetBondTypeAsDouble()
        if (neighbor.GetAtomicNum() == 8 and order == 2
                and neighbor.GetFormalCharge() == 0 and _heavy_degree(neighbor) == 1):
            double_o.append(idx)
        elif (neighbor.GetAtomicNum() == 8 and order == 1
                and neighbor.GetFormalCharge() == -1 and _heavy_degree(neighbor) == 1):
            anionic_o.append(idx)
        else:
            return None
    if len(double_o) != 1 or len(anionic_o) != 1:
        return None
    return frozenset({nitrogen, *double_o, *anionic_o})


def carbon_acid_derivative_frontier(mol, Chem):
    """Return outside, pending, or a completely mapped acid derivative.

    Complete coverage is intentionally limited to the general halogen ligand,
    primary amino ligand (retained carbamic-acid name), and the directly
    published nitro-substituted formic-acid graph. Cyanido and other
    P-65.2.1.4 pseudohalides remain pending for a future source-closed writer.
    """
    source = carbon_acid_source_class(mol, Chem)
    if source is None:
        return CarbonAcidDerivativeFrontier(
            "outside", "P-65.1.8/P-65.2.1", "",
            reason="graph is outside the exact source-class topology",
        )
    center, residual_o, carbonyl_o = (
        source.center, source.residual_oxygen, source.carbonyl_oxygen
    )
    branch_atoms = source.ligand_atoms
    core_atoms = source.core_atoms
    classification, rule = source.classification, source.rule
    kind = source.ligand_family
    roles = (
        (("retained-formic-acid", core_atoms, ()) if classification == "carboxylic-acid"
         else ("carbonic-acid-core", core_atoms, ())),
        ("replacement", branch_atoms, ()),
    )
    if classification == "carboxylic-acid" and kind == "hydrazino":
        # P-41 makes the carboxylic-acid-class name senior; this module only
        # classifies the source operation, and leaves the name to that route.
        return CarbonAcidDerivativeFrontier(
            "outside", rule, classification, center, residual_o,
            carbonyl_o, branch_atoms, roles, None,
            "source class recognized, but the senior carboxylic-acid construction belongs to another route",
        )
    supported_pseudohalides = set(_PSEUDOHALIDE_INFICES)
    if kind not in {"halogen", "amino", "nitro"} | supported_pseudohalides:
        return CarbonAcidDerivativeFrontier(
            "pending", rule, classification, center, residual_o,
            carbonyl_o, branch_atoms, roles, None,
            "source class recognized; this ligand family has no writer in this bounded producer",
        )
    if (any(atom.GetNumRadicalElectrons() for atom in iter_atoms(mol))
            or any(atom.GetAtomicNum() == 1 for atom in iter_atoms(mol))):
        return CarbonAcidDerivativeFrontier(
            "pending", rule, classification, center, residual_o,
            carbonyl_o, branch_atoms, roles, None,
            "source topology recognized; radicals or explicit hydrogen atoms are outside this writer",
        )
    annotations = any(atom.GetIsotope() or int(atom.GetChiralTag())
                      for atom in iter_atoms(mol))
    annotations |= any(int(bond.GetStereo()) or int(bond.GetBondDir())
                       for bond in iter_bonds(mol))
    annotations |= bool(mol.GetStereoGroups())
    annotations |= any(atom.GetAtomicNum() == 1 for atom in iter_atoms(mol))
    if annotations:
        return CarbonAcidDerivativeFrontier(
            "pending", rule, classification, center, residual_o,
            carbonyl_o, branch_atoms, roles, None,
            "source-class topology recognized; annotation or explicit-H decoration is outside this writer",
        )

    if kind == "nitro":
        stem = "nitroformic acid"
    elif kind == "amino":
        stem = "carbamic acid"
    elif kind in _PSEUDOHALIDE_INFICES:
        stem = _pseudohalide_acid_name(kind)
    else:
        stem = "carbono" + _HALIDE_NAMES[mol.GetAtomWithIdx(next(iter(branch_atoms))).GetAtomicNum()] + "ic acid"

    method = {
        "halogen": "functional-replacement",
        "amino": "retained-parent",
        "nitro": "substitutive",
    }.get(kind, "functional-replacement")
    decision = MethodDecision(
        name=stem,
        method=method,
        principal_centers=frozenset({center}),
        atoms=source.all_heavy_atoms,
        rule=rule,
        atom_locants=(),
        component_roles=roles,
    )
    return CarbonAcidDerivativeFrontier(
        "complete", rule, classification, center, residual_o,
        carbonyl_o, branch_atoms, roles, decision, "",
    )


def carbon_acid_derivative_decision(mol, Chem):
    """Compatibility decision-only API; callers needing barriers use frontier."""
    result = carbon_acid_derivative_frontier(mol, Chem)
    return result.decision if result.status == "complete" else None
