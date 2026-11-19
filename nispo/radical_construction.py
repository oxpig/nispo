"""Complete current radical components after existing delivery is unavailable."""


def general_radical_construction(mol, Chem, native_hydride_numberings=None):
    from .hydrogen_loss_components import mapped_carbon_hydrogen_loss
    decision = mapped_carbon_hydrogen_loss(mol, Chem, allow_acyl=True)
    if decision is None:
        from .hetero_radical_components import hetero_hydride_radical
        decision = hetero_hydride_radical(mol, Chem)
    if decision is None:
        from .carbon_polyradical_components import carbon_tree_polyradical
        decision = carbon_tree_polyradical(mol, Chem)
    if decision is None:
        from .ionic_radical_components import localized_ionic_radical
        decision = localized_ionic_radical(mol, Chem)
    if decision is None:
        from .hetero_radical_components import hetero_hydride_radical
        decision = hetero_hydride_radical(mol, Chem, allow_nonstandard=True)
    if decision is None and native_hydride_numberings is not None:
        from .parent_hydride_radicals import parent_hydride_radicals
        decision = parent_hydride_radicals(mol, Chem, native_hydride_numberings)
    return decision
