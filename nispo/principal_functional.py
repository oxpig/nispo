"""Select senior supported organic functions from the shared graph inventory."""

from .functional_inventory import functional_inventory
from .molecule_cache import mol_cache_get, mol_cache_set


def principal_organic_function(mol):
    cached = mol_cache_get(mol, "principal_organic_function")
    if cached is not None:
        return cached
    return mol_cache_set(mol, "principal_organic_function", _principal_organic_function(mol))


def _principal_organic_function(mol):
    inventory = functional_inventory(mol)
    groups = inventory.principal
    if len(groups) != 1:
        return ""  # Repeated sites need an explicit complete-parent renderer.
    kind = groups[0].kind
    # Newly supported class competition must not flatten stereo on another
    # component. Existing single-function stereo routes keep their prior scope.
    new_competition = extended_class_competition(inventory)
    if new_competition and (any(int(a.GetChiralTag()) for a in mol.GetAtoms())
                            or any(int(b.GetStereo()) for b in mol.GetBonds())):
        return ""
    family = {
        "ester": "retained-acid-ester-parent",
        "carbamate": "carbamate-retained-parent",
        "amide": "retained-amide-parent",
        "hydrazide": "acyl-hydrazide-retained-parent",
    }.get(kind, "")
    if kind == "amide" and any(a.GetAtomicNum() == 7 and a.GetIsAromatic() for a in mol.GetAtoms()):
        return ""  # Preserve the established fused N-tautomer prefix route.
    return family


def extended_class_competition(inventory):
    """The mixed classes previously excluded by sole-function eligibility."""
    groups = inventory.principal
    if len(groups) != 1:
        return False
    kind = groups[0].kind
    other = {g.kind for g in inventory.groups}
    return (
        kind == "ester" and "carbamate" in other
        or kind == "carbamate" and bool(other & {"amide", "hydrazide", "urea"})
        or kind == "amide" and bool(other & {"hydrazide", "urea"})
        or kind == "hydrazide" and any(g.kind in {
            "aldehyde", "ketone", "lactone", "lactam", "pseudoketone"
        } for g in inventory.groups)
    )
