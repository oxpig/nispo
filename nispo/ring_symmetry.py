"""Call-local sharing under proved, attachment-fixing ring automorphisms.

Every numbered record is retained. Only the repeated construction is shared;
the original atom-to-locant witnesses are projected to each orientation.
Annotated/native-property inputs and unfamiliar evidence use normal rendering.
"""
from dataclasses import fields, replace

from .molecule_iteration import iter_atoms, iter_bonds


_SHARE_RING_SYMMETRY = True


class RingRenderingSymmetry:
    def __init__(self):
        self.sources = {}
        self.graphs = {}
        self.orientations = {}
        self.plan_keys = {}
        self.fixed_atoms = {}
        self.checked_records = {}

    def _graph(self, mol):
        key = id(mol)
        if key not in self.graphs:
            if type(mol).__module__ != 'rdkit.Chem.rdchem' or type(mol).__name__ != 'Mol':
                self.graphs[key] = None
                return None
            atoms, bonds = tuple(iter_atoms(mol)), tuple(iter_bonds(mol))
            if (mol.GetStereoGroups() or any(
                    a.GetIsotope() or int(a.GetChiralTag()) or a.HasQuery()
                    or a.HasProp('_CIPCode')
                    or tuple(a.GetPropNames(includePrivate=True, includeComputed=False))
                    for a in atoms)
                    or any(int(b.GetStereo()) or int(b.GetBondDir()) or b.HasQuery()
                           or tuple(b.GetPropNames(includePrivate=True, includeComputed=False))
                           for b in bonds)):
                self.graphs[key] = None
            else:
                colors = tuple((a.GetAtomicNum(), a.GetFormalCharge(),
                    a.GetNumRadicalElectrons(), a.GetIsAromatic(),
                    a.GetNumExplicitHs(), a.GetNumImplicitHs(), a.GetNoImplicit(),
                    a.GetAtomMapNum()) for a in atoms)
                neighbors = [[] for _ in atoms]
                for bond in bonds:
                    a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    state = (int(bond.GetBondType()), bond.GetIsAromatic(), bond.GetIsConjugated())
                    neighbors[a].append((b, state))
                    neighbors[b].append((a, state))
                self.graphs[key] = colors, neighbors
        return self.graphs[key]

    def _key(self, plan):
        if id(plan) in self.plan_keys:
            return self.plan_keys[id(plan)]
        r = plan.rendering
        skeleton = r.skeleton
        # Absolute/localized skeleton evidence needs its own atom-map proof.
        # Nitrogen suffix maps and modified oxygen units keep their own writer.
        if (not _SHARE_RING_SYMMETRY or r.nmap or r.prepared.modified_oxygen_suffix
                or len(skeleton) > 8 and skeleton[8] is not None
                or r.kind in {'amide', 'amine', 'imine', 'peroxol', 'sulfonamide'}):
            self.plan_keys[id(plan)] = None
            return None
        fixed = self.fixed_atoms.get(id(r.prepared))
        if fixed is None:
            fixed = {idx for g in r.groups
                     for idx in (*g.atoms, g.center, g.owner, getattr(g, 'ligand', -1))}
            fixed.update(i for i, rows in r.prepared.branch_edges.items() if rows)
            fixed = tuple(sorted(fixed & r.core))
            self.fixed_atoms[id(r.prepared)] = fixed
        key = (id(r.ctx), id(r.prepared), r.core, r.scope,
                tuple((i, r.mapping[i]) for i in fixed),
                tuple((i, loc) for i, loc in r.mapping.items() if i not in r.core),
                skeleton[0], tuple(skeleton[2]), tuple(skeleton[3]),
                tuple(skeleton[4]), skeleton[5], skeleton[7] if len(skeleton) > 7 else None,
                r.h, r.actual, plan.state, plan.chemical_key)
        self.plan_keys[id(plan)] = key
        return key

    def _orientation(self, plan, graph):
        r = plan.rendering
        skeleton = r.skeleton
        sequence = tuple(skeleton[1])
        orientation_key = id(r.ctx), sequence, id(r.prepared)
        orientation = self.orientations.get(orientation_key)
        if orientation is None:
            colors, neighbors = graph
            positions = dict(zip(sequence, range(len(sequence))))
            # Relabel the core into numbering coordinates, leaving every
            # external atom in a disjoint, original-index namespace. Equality
            # is exactly an automorphism fixing those external atoms.
            signature = tuple((colors[i], tuple(sorted(
                (positions.get(j, len(sequence) + j), state)
                for j, state in neighbors[i]))) for i in sequence)
            orientation = signature
            self.orientations[orientation_key] = orientation
        return orientation

    def reuse(self, plan):
        key = self._key(plan)
        if key is None:
            return None
        sources = self.sources.get(key)
        if not sources:
            return None
        graph = self._graph(plan.rendering.ctx.mol)
        if graph is None:
            return None
        orientation = self._orientation(plan, graph)
        for previous, rows in sources:
            if (orientation != self._orientation(previous, graph)
                    or not self._ordinary_records(rows)):
                continue
            permutation = dict(zip(previous.rendering.skeleton[1], plan.rendering.skeleton[1]))
            def project(mapping):
                return tuple((permutation.get(i, i), loc) for i, loc in mapping)
            return tuple(replace(row, parent_atoms=tuple(plan.rendering.skeleton[1]),
                atom_locants=project(row.atom_locants),
                construction=replace(row.construction,
                                     atom_locants=project(row.construction.atom_locants)))
                for row in rows)
        return None

    def remember(self, plan, rows):
        if not rows:
            return
        key = self._key(plan)
        if key is None:
            return
        self.sources.setdefault(key, []).append((plan, tuple(rows)))

    def _ordinary_records(self, rows):
        if id(rows) in self.checked_records:
            return self.checked_records[id(rows)]
        allowed = {'parent', 'atom_locants', 'atoms', 'prefixes', 'embedded_atoms',
                   'hydro_prefix', 'systematic_parent', 'omit_methyl_prefix_locant',
                   'prefix_branch_ownership'}
        for row in rows:
            component = row.construction
            if component is None or row.unsaturation_state is not None:
                self.checked_records[id(rows)] = False
                return False
            if any(getattr(component, field.name) != field.default
                   for field in fields(component) if field.name not in allowed):
                self.checked_records[id(rows)] = False
                return False
        self.checked_records[id(rows)] = True
        return True
