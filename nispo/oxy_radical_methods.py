"""Mapped additive oxyl/peroxyl construction preferred by P-71.3.4.

One terminal oxygen radical owns its original oxygen chain; carbon/acyl or
amino components retain distinct atom sets. No closed-shell whole name is
rewritten into a radical name.
"""
from .acid_derivative_methods import _acid_component, _acyl
from .component_mapping import fully_covered, mapped_branch
from .formatting import alkyl_to_alkoxy, enclose_name_unit
from .group14_mononuclear_methods import _carbon_organyl
from .molecule_iteration import iter_atoms
from .naming_methods import MethodDecision
from .prefix_sequences import PrefixUnit, mononuclear_prefix_sequence


RULE = 'P-71.3.4(1)/P-63.2.2.2/P-65.1.7'


def _organyl(mol, Chem, root, host):
    group = _carbon_organyl(mol, root, host)
    if not group:
        return None
    text, owned = mapped_branch(mol, Chem, root, host)
    return (text, group) if text and frozenset(owned) == group else None


def _acyl_component(mol, Chem, root, host):
    carbon = mol.GetAtomWithIdx(root)
    oxo = [a.GetIdx() for a in carbon.GetNeighbors() if a.GetAtomicNum() == 8
           and a.GetDegree() == 1 and a.GetTotalValence() == 2
           and mol.GetBondBetweenAtoms(root, a.GetIdx()).GetBondType() == Chem.BondType.DOUBLE]
    if len(oxo) != 1 or carbon.GetTotalValence() != 4 or carbon.IsInRing():
        return None
    acid = _acid_component(mol, Chem, root, host)
    if acid:
        text = _acyl(acid.name)
        if text:
            return text, acid.atoms, acid.atom_locants, acid.substituted
    # The retained benzoic-acid component is not yet exposed by every acid
    # frontier. Prove its phenyl/halo-phenyl arm directly at this boundary.
    distal = [a.GetIdx() for a in carbon.GetNeighbors() if a.GetIdx() not in {host, *oxo}]
    if len(distal) != 1:
        return None
    ligand = mol.GetAtomWithIdx(distal[0])
    halide = {9:'fluorid',17:'chlorid',35:'bromid',53:'iodid'}.get(ligand.GetAtomicNum())
    if (halide and ligand.GetDegree() == 1 and ligand.GetTotalValence() == 1
            and mol.GetBondBetweenAtoms(root,distal[0]).GetBondType() == Chem.BondType.SINGLE):
        # P-65.2.1.5: carbonic-acid halide acyl groups have their own
        # preferred carbonohalidoyl construction, rather than haloformyl.
        return 'carbono' + halide + 'oyl', frozenset({root,*oxo,*distal}), (), False
    organyl = _organyl(mol, Chem, distal[0], root)
    if organyl is None or not mol.GetAtomWithIdx(distal[0]).GetIsAromatic():
        return None
    text, group = organyl
    if not text.endswith('phenyl'):
        return None
    return text[:-6] + 'benzoyl', group | {root, *oxo}, (), text != 'phenyl'


def oxy_radical_decision(mol, Chem):
    """Prove a neutral terminal RO•, ROO•, or R2NO• construction.

    The component grammar admits saturated halo-carbon trees, phenyl arms
    and acid-frontier acyls. Labels/stereo, longer oxygen chains and non-oxygen
    radical analogues keep their separate existing producers.
    """
    if mol.GetNumAtoms() > 128 or mol.GetStereoGroups():
        return None
    atoms = tuple(iter_atoms(mol))
    if any(a.GetAtomicNum() == 1 or a.GetFormalCharge() or a.GetIsotope()
           or int(a.GetChiralTag()) for a in atoms):
        return None
    if any(int(b.GetStereo()) for b in mol.GetBonds()):
        return None
    radicals = [a for a in atoms if a.GetNumRadicalElectrons()]
    if len(radicals) != 1:
        return None
    radical = radicals[0]
    if (radical.GetAtomicNum() != 8 or radical.GetNumRadicalElectrons() != 1
            or radical.GetDegree() != 1 or radical.GetTotalNumHs()
            or radical.GetTotalValence() != 1):
        return None
    ri = radical.GetIdx()
    root = radical.GetNeighbors()[0].GetIdx()
    if mol.GetBondBetweenAtoms(ri, root).GetBondType() != Chem.BondType.SINGLE:
        return None
    oxygen_chain, host = {ri}, ri
    if mol.GetAtomWithIdx(root).GetAtomicNum() == 8:
        inner = mol.GetAtomWithIdx(root)
        if (inner.GetDegree() != 2 or inner.GetTotalValence() != 2
                or inner.GetTotalNumHs() or inner.IsInRing()):
            return None
        host = root
        oxygen_chain.add(root)
        root = next(a.GetIdx() for a in inner.GetNeighbors() if a.GetIdx() != ri)
        if mol.GetBondBetweenAtoms(host, root).GetBondType() != Chem.BondType.SINGLE:
            return None
    ending = 'peroxyl' if len(oxygen_chain) == 2 else 'oxyl'
    atom = mol.GetAtomWithIdx(root)
    bindings = ()
    if atom.GetAtomicNum() == 6:
        acyl = _acyl_component(mol, Chem, root, host)
        if acyl:
            text, group, bindings, compound = acyl
            name = (enclose_name_unit(text) if compound else text) + ending
        else:
            organyl = _organyl(mol, Chem, root, host)
            if organyl is None:
                return None
            text, group = organyl
            # The contracted oxy forms are fully substitutable, except for
            # tert-butoxy. The shared component writer implements that grammar.
            if len(oxygen_chain) == 1:
                name = alkyl_to_alkoxy(text) + 'l'
            else:
                simple = text in {'phenyl', 'tert-butyl'} or all(
                    mol.GetAtomWithIdx(i).GetAtomicNum() == 6
                    and sum(n.GetIdx() in group for n in mol.GetAtomWithIdx(i).GetNeighbors()) <= 2
                    for i in group)
                sequence = mononuclear_prefix_sequence([
                    PrefixUnit(text, 'simple' if simple else 'compound')])
                if sequence is None:
                    return None
                name = sequence.text + ending
    elif atom.GetAtomicNum() == 7 and len(oxygen_chain) == 1:
        if atom.IsInRing() or atom.GetTotalValence() != 3:
            return None
        group, units = {root}, []
        for neighbor in atom.GetNeighbors():
            ni = neighbor.GetIdx()
            if ni == host:
                continue
            organyl = _organyl(mol, Chem, ni, root)
            if organyl is None:
                return None
            text, owned = organyl
            if owned & group:
                return None
            group.update(owned)
            simple = text in {'phenyl', 'tert-butyl'} or all(
                mol.GetAtomWithIdx(i).GetAtomicNum() == 6
                and sum(n.GetIdx() in owned for n in mol.GetAtomWithIdx(i).GetNeighbors()) <= 2
                for i in owned)
            units.append(PrefixUnit(text, 'simple' if simple else 'compound'))
        if len(units) > 2 or atom.GetTotalNumHs() != 2 - len(units):
            return None
        sequence = mononuclear_prefix_sequence(units) if units else None
        if units and sequence is None:
            return None
        name = (sequence.text if sequence else '') + 'aminoxyl'
        group = frozenset(group)
    else:
        return None
    if group & oxygen_chain or not fully_covered(mol, group, oxygen_chain):
        return None
    roles = (('additive-radical-oxygen-chain', frozenset(oxygen_chain), ()),
             ('radical-substituent-component', frozenset(group), bindings))
    return MethodDecision(name, 'additive', frozenset({ri}),
        frozenset(group) | oxygen_chain, RULE, bindings, component_roles=roles)
