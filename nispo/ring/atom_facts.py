"""Primitive atom flags shared across alternative ring numberings."""

from ..molecule_cache import mol_cache_get, mol_cache_set
from ..molecule_iteration import iter_atoms


def ring_adjustment_atoms(mol) -> tuple[frozenset[int], frozenset[int]]:
    """Return atoms needing charge/lambda handling and isotope handling.

    Naming works on an owned molecule with fixed atomic states, just like the
    other per-molecule graph caches. Store indices rather than RDKit Atom
    objects so the cache does not retain wrappers for its owning molecule.
    """
    cached = mol_cache_get(mol, "ring_adjustment_atoms")
    if cached is not None:
        return cached
    charge_or_lambda, isotope = set(), set()
    for atom in iter_atoms(mol):
        idx = atom.GetIdx()
        if atom.GetIsotope():
            isotope.add(idx)
        symbol = atom.GetSymbol()
        if atom.GetFormalCharge() or (
            symbol == "N"
            and atom.GetTotalNumHs() == 0
            and (atom.GetNumRadicalElectrons() or atom.GetExplicitValence() <= 2)
        ) or (
            symbol == "S" and int(atom.GetTotalValence()) in {4, 6}
        ) or (
            symbol == "P" and int(atom.GetTotalValence()) == 5
        ):
            charge_or_lambda.add(idx)
    return mol_cache_set(
        mol, "ring_adjustment_atoms", (frozenset(charge_or_lambda), frozenset(isotope))
    )
