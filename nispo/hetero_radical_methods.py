"""Mapped P-71 monoradicals on straight saturated carbon chains.

This opt-in producer covers characteristic-group operations with direct Blue
Book support: amine-N H-loss (aminyl), hydroxy-O H-loss (oxyl), and terminal
S/Se/Te chalcogen radicals. It does not name skeletal-replacement chains or
fall back to closed-shell parent names.
"""

from __future__ import annotations

from dataclasses import dataclass

from .naming_methods import MethodDecision


RULE = "P-70.3/P-71.3.2/P-71.3.4/P-71.7/P-14.4"
_ELEMENTS = {7: "N", 8: "O", 16: "S", 34: "Se", 52: "Te"}
_ROOT_LIMIT = 64  # resource bound; the supported parent-root table is narrower


@dataclass(frozen=True, slots=True)
class HeteroRadicalRecognition:
    status: str
    reason: str
    radical_atom: int | None = None
    element: str | None = None
    carbon_path: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class HeteroRadicalProposal:
    decision: MethodDecision
    radical_atom: int
    radical_element: str
    carbon_path: tuple[int, ...]
    atom_locants: tuple[tuple[int, str], ...]
    operation: str


def recognize_hetero_radical(mol, Chem=None) -> HeteroRadicalRecognition:
    """Recognize one terminal N/O/S radical on an unbranched saturated C chain."""
    if mol is None or not mol.GetNumAtoms() or mol.GetNumHeavyAtoms() > _ROOT_LIMIT:
        return HeteroRadicalRecognition("declined", "empty or over-bound graph")
    if mol.GetRingInfo().NumRings() or mol.GetStereoGroups():
        return HeteroRadicalRecognition("declined", "cyclic or enhanced-stereo graph")
    atoms = tuple(mol.GetAtoms())
    if any(a.GetAtomicNum() == 1 for a in atoms):
        return HeteroRadicalRecognition("declined", "explicit hydrogen vertices")
    heavy = tuple(a for a in atoms if a.GetAtomicNum() > 1)
    if any(a.GetIsotope() or int(a.GetChiralTag()) for a in heavy):
        return HeteroRadicalRecognition("declined", "isotopic or atom-stereo annotation")
    if any(b.GetBondTypeAsDouble() != 1.0 or int(b.GetStereo())
           or int(b.GetBondDir()) for b in mol.GetBonds()):
        return HeteroRadicalRecognition("declined", "non-single or directed bond")
    if any(a.GetFormalCharge() for a in atoms):
        return HeteroRadicalRecognition("declined", "charged graph")
    radical_atoms = [a for a in atoms if a.GetNumRadicalElectrons()]
    if (len(radical_atoms) != 1 or radical_atoms[0].GetNumRadicalElectrons() != 1
            or radical_atoms[0].GetAtomicNum() not in _ELEMENTS):
        return HeteroRadicalRecognition("declined", "requires one N/O/S/Se/Te radical electron")
    center = radical_atoms[0]
    carbons = tuple(a for a in heavy if a.GetAtomicNum() == 6)
    if not carbons or len(heavy) != len(carbons) + 1:
        return HeteroRadicalRecognition("declined", "requires one carbon chain and one radical atom")
    from .constants import ROOTS
    if len(carbons) not in ROOTS:
        return HeteroRadicalRecognition("declined", "carbon chain root unavailable in bounded grammar")
    if any(a.GetAtomicNum() != 6 and a.GetIdx() != center.GetIdx() for a in heavy):
        return HeteroRadicalRecognition("declined", "unsupported skeletal heteroatom")
    if center.GetDegree() != 1 or center.GetNeighbors()[0].GetAtomicNum() != 6:
        return HeteroRadicalRecognition("declined", "radical atom must be terminal and C-bound")
    element = center.GetAtomicNum()
    hcount = center.GetTotalNumHs(includeNeighbors=True)
    valence = center.GetTotalValence()
    if element == 7 and (valence != 2 or hcount != 1):
        return HeteroRadicalRecognition("declined", "N radical is outside terminal aminyl valence")
    if element in {8, 16, 34, 52} and (valence != 1 or hcount != 0):
        return HeteroRadicalRecognition("declined", "O/chalcogen radical is outside terminal radical valence")
    carbon_set = {a.GetIdx() for a in carbons}
    if any(a.GetTotalValence() != 4 for a in carbons):
        return HeteroRadicalRecognition("declined", "nonordinary carbon valence")
    adjacency = {idx: tuple(n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                            if n.GetIdx() in carbon_set) for idx in carbon_set}
    if any(len(neighbors) > 2 for neighbors in adjacency.values()):
        return HeteroRadicalRecognition("declined", "branched carbon framework")
    endpoints = sorted(idx for idx, neighbors in adjacency.items() if len(neighbors) <= 1)
    if len(endpoints) != 2 and len(carbons) > 1:
        return HeteroRadicalRecognition("declined", "carbon framework is not a chain")
    start = center.GetNeighbors()[0].GetIdx()
    path, previous, current = [], None, start
    while True:
        path.append(current)
        nxt = [idx for idx in adjacency[current] if idx != previous]
        if not nxt:
            break
        if len(nxt) != 1:
            return HeteroRadicalRecognition("declined", "ambiguous carbon-chain traversal")
        previous, current = current, nxt[0]
    if len(path) != len(carbons) or not set(path) == carbon_set:
        return HeteroRadicalRecognition("declined", "disconnected or cyclic carbon framework")
    # Name support, rather than an arbitrary chemical cutoff, limits chain
    # length. The corresponding parent and alkyl form must both exist.
    from .branch_rules import parent_chain_name
    if not parent_chain_name(mol, tuple(path)) or not parent_chain_name(
            mol, tuple(path), as_substituent=True):
        return HeteroRadicalRecognition("declined", "parent or substituent root unavailable")
    # Ensure the non-radical graph is connected and has exactly the declared edges.
    if mol.GetNumBonds() != len(heavy) - 1:
        return HeteroRadicalRecognition("declined", "unexpected edge or disconnected graph")
    if Chem is not None:
        try:
            Chem.SanitizeMol(Chem.Mol(mol))
        except Exception:
            return HeteroRadicalRecognition("declined", "RDKit valence validation failed")
    return HeteroRadicalRecognition("accepted", "bounded terminal heteroatom radical",
                                    center.GetIdx(), _ELEMENTS[element], tuple(path))


def hetero_radical_proposals(mol, Chem=None) -> tuple[HeteroRadicalProposal, ...]:
    recognition = recognize_hetero_radical(mol, Chem)
    if recognition.status != "accepted":
        return ()
    from .branch_rules import parent_chain_name
    from .formatting import attach_suffix

    path = recognition.carbon_path
    chain = parent_chain_name(mol, path)
    alkyl = parent_chain_name(mol, path, as_substituent=True)
    if not chain or not alkyl:
        return ()
    locants = tuple((idx, str(position)) for position, idx in enumerate(path, 1))
    radical_atom = recognition.radical_atom
    if recognition.element == "N":
        # P-71.3.2 characteristic amine-radical suffix; a locant is needed
        # only once the chain length permits more than one amine position.
        suffix_locants = (1,) if len(path) >= 3 else ()
        name = attach_suffix(chain, "aminyl", suffix_locants)
        operation = "aminyl"
    elif recognition.element == "O":
        # P-71.3.4 method (1): alkyl + oxyl. Its retained contractions cover
        # methoxyl through butoxyl; longer chains use the systematic form.
        name = alkyl[:-2] + "oxyl" if len(path) <= 4 else alkyl + "oxyl"
        operation = "oxyl"
    else:
        # P-71.3.4 chalcogen analogue formed with the corresponding
        # preselected radical; the source gives methylselanyl.
        suffix = {"S": "sulfanyl", "Se": "selanyl", "Te": "tellanyl"}[recognition.element]
        name = alkyl + suffix
        operation = suffix
    if not name:
        return ()
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    roles = (
        ("carbon_chain", frozenset(path), locants),
        (f"{operation}_radical_atom", frozenset({radical_atom}), ()),
    )
    owned = [idx for _, role_atoms, _ in roles for idx in role_atoms]
    if len(owned) != len(set(owned)) or set(owned) != set(heavy):
        return ()
    decision = MethodDecision(
        name=name, method="substitutive", principal_centers=frozenset({radical_atom}),
        atoms=heavy, rule=RULE, atom_locants=locants, component_roles=roles,
    )
    proposal = HeteroRadicalProposal(decision, radical_atom, recognition.element,
                                     path, locants, operation)
    return (proposal,)


def hetero_radical_decision(mol, Chem):
    """Return the complete mapped name decision, or ``None`` on a boundary."""
    cyclic = _monocyclic_lactam_n_radical(mol, Chem)
    if cyclic is not None:
        return cyclic
    proposals = hetero_radical_proposals(mol, Chem)
    if not proposals or len({p.decision.name for p in proposals}) != 1:
        return None
    return proposals[0].decision


def _monocyclic_lactam_n_radical(mol, Chem):
    """P-71.2: one N free valence on a plain saturated monocyclic lactam.

    A temporary external bond fills that free valence solely to invoke the
    established, rooted -yl component constructor. Its complete atom scope
    must equal the original graph; the temporary atom never enters the name.
    """
    rings = mol.GetRingInfo().AtomRings()
    if (len(rings) != 1 or not 4 <= len(rings[0]) <= 8
            or mol.GetNumHeavyAtoms() > 24 or len(Chem.GetMolFrags(mol)) != 1
            or mol.GetStereoGroups()
            or any(a.GetAtomicNum() not in {6, 7, 8} or a.GetFormalCharge()
                   or a.GetIsotope() or a.GetChiralTag() or a.HasQuery() for a in mol.GetAtoms())
            or any(b.GetStereo() or b.HasQuery() for b in mol.GetBonds())):
        return None
    nitrogen = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
    oxygens = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 8]
    if len(nitrogen) != 1 or len(oxygens) != 1:
        return None
    center, oxygen = nitrogen[0], oxygens[0]
    if (not center.IsInRing() or center.GetNumRadicalElectrons() != 1
            or center.GetDegree() != 2 or center.GetTotalNumHs() or center.GetTotalValence() != 2
            or oxygen.GetDegree() != 1 or oxygen.GetTotalNumHs()
            or oxygen.GetBonds()[0].GetBondType() != Chem.BondType.DOUBLE
            or oxygen.GetNeighbors()[0].GetIdx() not in {a.GetIdx() for a in center.GetNeighbors()}
            or any(a.GetNumRadicalElectrons() for a in mol.GetAtoms() if a.GetIdx() != center.GetIdx())
            or any(b.GetBondType() != Chem.BondType.SINGLE for b in mol.GetBonds()
                   if oxygen.GetIdx() not in {b.GetBeginAtomIdx(), b.GetEndAtomIdx()})):
        return None
    working = Chem.RWMol(mol)
    working.GetAtomWithIdx(center.GetIdx()).SetNumRadicalElectrons(0)
    anchor = working.AddAtom(Chem.Atom(6))
    working.AddBond(center.GetIdx(), anchor, Chem.BondType.SINGLE)
    capped = working.GetMol()
    Chem.SanitizeMol(capped)
    from .component_mapping import mapped_branch
    name, owned = mapped_branch(capped, Chem, center.GetIdx(), anchor)
    atoms = frozenset(range(mol.GetNumAtoms()))
    if not name or not name.endswith('-yl') or owned != atoms:
        return None
    return MethodDecision(name=name, method='substitutive',
        principal_centers=frozenset({center.GetIdx()}), atoms=atoms,
        rule='P-71.2/P-29.2', component_roles=(('cyclic-N-radical', atoms, ()),))
