"""Compose an acyclic carbon prefix with a graph-proved urea-derived unit."""


def carbamoylamino_alkyl_prefix(mol, Chem, root, blocked, scope):
    if (mol.GetAtomWithIdx(root).GetAtomicNum() != 6
            or any(mol.GetAtomWithIdx(i).GetIsotope()
                   or mol.GetAtomWithIdx(i).GetFormalCharge()
                   or mol.GetAtomWithIdx(i).GetNumRadicalElectrons()
                   or int(mol.GetAtomWithIdx(i).GetChiralTag()) for i in scope)):
        return None
    chain, previous, current = [], blocked, root
    attachment = None
    while True:
        atom = mol.GetAtomWithIdx(current)
        if atom.IsInRing() or atom.GetAtomicNum() != 6 or current in chain:
            return None
        chain.append(current)
        others = [n for n in atom.GetNeighbors() if n.GetIdx() != previous]
        if any(mol.GetBondBetweenAtoms(current, n.GetIdx()).GetBondTypeAsDouble() != 1 for n in others):
            return None
        carbons = [n.GetIdx() for n in others if n.GetAtomicNum() == 6]
        hetero = [n for n in others if n.GetAtomicNum() != 6]
        if len(carbons) > 1 or len(hetero) > 1 or carbons and hetero:
            return None
        if hetero:
            attachment = hetero[0]
            break
        if not carbons:
            return None
        previous, current = current, carbons[0]
    if attachment.GetAtomicNum() != 7:
        return None
    ns = []
    for carbon in attachment.GetNeighbors():
        if carbon.GetAtomicNum() != 6 or carbon.GetIdx() in chain:
            continue
        oxo = [a for a in carbon.GetNeighbors() if a.GetAtomicNum() == 8 and a.GetDegree() == 1
               and mol.GetBondBetweenAtoms(carbon.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 2]
        amino = [a for a in carbon.GetNeighbors() if a.GetAtomicNum() == 7
                 and mol.GetBondBetweenAtoms(carbon.GetIdx(), a.GetIdx()).GetBondTypeAsDouble() == 1]
        if len(oxo) == 1 and len(amino) == 2 and carbon.GetDegree() == 3:
            ns.append(carbon.GetIdx())
    if len(ns) != 1:
        return None
    from .component_mapping import mapped_branch
    text, atoms = mapped_branch(mol, Chem, attachment.GetIdx(), current)
    if not text or frozenset(atoms) | frozenset(chain) != frozenset(scope) or set(atoms) & set(chain):
        return None
    from .constants import ROOTS
    from .formatting import format_prefixes
    stem = ROOTS.get(len(chain))
    if not stem:
        return None
    prefix = format_prefixes([(str(len(chain)), text)], omit_locants=len(chain) == 1,
                             hierarchical_enclosures=True)
    return prefix + stem + 'yl', frozenset(scope)
