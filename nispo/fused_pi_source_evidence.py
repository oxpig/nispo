"""Published evidence for fused hydrides written without a Delta descriptor.

The maximum-bond matchings of a fused hydride alone do not establish whether
those matchings denote distinct localized isomers. In particular, standard
InChI collapses both some unlocalized examples and the localized heptalene
examples in P-25.7.1.2. This table records only the precise hydride graph for
which the cited source itself supplies the unqualified name. Callers may
compare an unmodified parent graph after explicitly accounting for the bonds
to its substituents. This is not a resonance rule and does not extend the
evidence to other parent bond assignments.
"""

from functools import lru_cache


# These graphs were independently transcribed from the published drawings
# and checked against OPSIN parses of the source parent names. They document
# specific parent bond states, not all structures sharing their InChI.
_UNQUALIFIED_HYDRIDES = (
    # P-25.1.2.3's octalene drawing is written without a descriptor.
    'C1=CC=CC=CC2=CC=CC=CC=C12',
    # P-25.3.4.2.1(a), complete polyalene/azulene fusion parent.
    'c1cc2ccc3cc4cc5ccc1c5c4c23',
    # P-25.3.4.2.3.1, fusion-prefix citation example.
    'N1=CN=C2N=C3C(N=C21)=NC2=C3OC=C2',
    # P-25.3.3.1.2(e), interior-heteroatom orientation example.
    'C1=CC2=CC=C3C=CCC4=C3N2C(=C1)C=C4',
    # P-25.3.8.4, omission of peri-fusion carbon locants.
    'C1=CC2=CC=CC3=c4ccccc4=NC(=C1)N23',
    # P-25.5.2, the hydrocarbon fusion example and the carbon skeleton
    # explicitly used for the adjacent skeletal-replacement example.
    'C1=Cc2cc3cc4ccccc4cc3c3c2C1=C3',
    'C1=Cc2cc3cc4cc5ccccc5cc4cc3c3c2C1=C3',
    # P-25.4.3.2.1 and .3.3, complete bridged parent bond states.
    'C1=C2CC(=C1)C1=C2OC2=C(O1)C1=CC=C2C1',
    'C1=C2CC(=C1)c1c2c2ccc1o2',
    'C1=C2CC(=C1)c1cc3c4ccc(c3cc12)CC4',
    'c1c2cc3cc4c5ccc(o5)c4cc3c1C2',
    # P-25.4.3.2.2, complete composite-bridge source structures.
    'C1=C2COc3coc(c31)CO2',
    'C1=C2OCc3coc(c31)CO2',
    'C12=CC=C(C3=CC=4C5=CC=C(C4C=C13)OC5)CO2',
)


@lru_cache(maxsize=1)
def _identities():
    from rdkit import Chem

    return frozenset(Chem.MolToSmiles(Chem.MolFromSmiles(smiles),
                                      canonical=True, isomericSmiles=True)
                     for smiles in _UNQUALIFIED_HYDRIDES)


@lru_cache(maxsize=2048)
def published_unqualified_hydride(identity):
    """Match the documented hydrogen and bond state, ignoring isotope labels."""
    if identity in _identities():
        return True
    from rdkit import Chem

    mol = Chem.MolFromSmiles(identity)
    if mol is None or not any(atom.GetIsotope() for atom in mol.GetAtoms()):
        return False
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)
    unlabeled = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    return unlabeled in _identities()
