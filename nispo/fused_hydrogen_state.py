"""Graph-derived functional H states for neutral fused heterocyclic carbonyls.

Capped external bonds establish the parent hydrogen budget. A carbonyl suffix
accommodates paired valence changes; the remaining changes are hydro prefixes.
The input graph is never tautomerized. Only reference hydrides are enumerated.
"""

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
import re

from .constants import MULT
from .formatting import format_prefixes, join_prefix_parent, multiplied_suffix
from .functional_inventory import functional_inventory
from .hydrogen_accommodation import annular_pi_bonds, joint_hydrogen_states, peripheral_ring_atoms


@dataclass(frozen=True, slots=True)
class FusedHydrogenRecord:
    name: str = ""
    atom_locants: tuple[tuple[int, str], ...] = ()
    carbonyl_atoms: tuple[int, ...] = ()
    mapped: bool = False


# Parent vocabulary, not molecule answers. Retained preferred fused hydrides
# supply their fixed atom numbering; other fusion families keep their routes.
_PARENT_NAMES = frozenset(
    {
        "indole",
        "isoindole",
        "indazole",
        "benzimidazole",
        "quinoline",
        "isoquinoline",
        "quinazoline",
        "quinoxaline",
        "cinnoline",
        "phthalazine",
        "purine",
        "pteridine",
        "benzofuran",
        "isobenzofuran",
        "1-benzopyran",
        "2-benzopyran",
    }
)


def _locant_key(loc):
    match = re.fullmatch(r"(\d+)(.*)", loc)
    return (int(match[1]), match[2]) if match else (999, loc)


@lru_cache(maxsize=8)
def _templates(Chem):
    from .parent_rules import _curated_fused_templates

    seen, result = set(), []
    for t in _curated_fused_templates(Chem):
        if t["name"] not in _PARENT_NAMES or t["name"] in seen:
            continue
        if not all(
            a.IsInRing() and a.GetAtomicNum() in {6, 7, 8} and not a.GetFormalCharge()
            for a in t["mol"].GetAtoms()
        ) or not all(t["locants"].get(i) for i in range(t["mol"].GetNumAtoms())):
            continue
        from .ring_state_parent import _peripheral_junction_locants
        locants = _peripheral_junction_locants(t['mol'], t['locants'])
        if locants is None:
            continue
        seen.add(t["name"])
        result.append(dict(t, locants=locants))
    return tuple(result)


@lru_cache(maxsize=128)
def _reference_states(query):
    """Enumerate neutral mancude H states by noncumulative pi-bond matching.

    Reference H can occupy peripheral C as well as N. A tautomer enumerator
    restricted to ordinary N-H shifts misses carbonyl-accommodating hydrides
    such as 4H-benzopyran and 6H-purine. Input atom order is retained throughout.
    """
    from rdkit import Chem

    atoms = tuple(query.GetAtoms())
    if any(a.GetDegree() not in {2, 3} for a in atoms):
        return ()
    default_h = {
        a.GetIdx(): int(a.GetAtomicNum() == 6 and a.GetDegree() == 2) for a in atoms
    }
    needed = {
        a.GetIdx()
        for a in atoms
        if a.GetAtomicNum() == 6 or a.GetAtomicNum() == 7 and a.GetDegree() == 2
    }
    count = sum(a.GetTotalNumHs() for a in atoms) - sum(default_h.values())
    if count not in {0, 1, 2}:
        return ()
    positions = [
        a.GetIdx() for a in atoms if a.GetDegree() == 2 and a.GetAtomicNum() in {6, 7}
    ]
    neighbors = {a.GetIdx(): {n.GetIdx() for n in a.GetNeighbors()} for a in atoms}

    def matchings(remaining, bonds=()):
        if not remaining:
            yield bonds
            return
        first = min(remaining, key=lambda i: len(neighbors[i] & remaining))
        for second in sorted(neighbors[first] & remaining):
            yield from matchings(
                remaining - {first, second}, bonds + ((first, second),)
            )

    states = []
    for indicated in combinations(positions, count):
        required = needed - set(indicated)
        if len(required) % 2:
            continue
        for number, bonds in enumerate(matchings(required)):
            if number >= 64:
                break
            state = Chem.RWMol(query)
            for atom in state.GetAtoms():
                atom.SetIsAromatic(False)
                atom.SetNumExplicitHs(
                    default_h[atom.GetIdx()] + int(atom.GetIdx() in indicated)
                )
                atom.SetNoImplicit(True)
            for bond in state.GetBonds():
                bond.SetIsAromatic(False)
                bond.SetBondType(Chem.BondType.SINGLE)
            for first, second in bonds:
                state.GetBondBetweenAtoms(first, second).SetBondType(
                    Chem.BondType.DOUBLE
                )
            result = state.GetMol()
            try:
                Chem.SanitizeMol(result)
            except (ValueError, RuntimeError):
                continue
            states.append(result)
            break  # Equivalent Kekule arrangements have the same H budget.
    return tuple(states)


def fused_hydrogen_record(ctx, *, mapped=False):
    from .parent_rules import _cached_iter_fused_template_matches
    from .cycle_utils import ring_system_extends_core

    mol, Chem = ctx.mol, ctx.Chem
    if mapped:
        from .component_mapping import chemical_inventory

        inventory = chemical_inventory(mol)
    else:
        inventory = functional_inventory(mol)
    if (
        not inventory.supported
        or not inventory.principal
        or inventory.principal[0].kind
        not in {"ketone", "lactam", "lactone", "pseudoketone"}
    ):
        return FusedHydrogenRecord()
    if not mapped and (
        any(int(a.GetChiralTag()) or a.GetIsotope() for a in mol.GetAtoms())
        or any(int(b.GetStereo()) for b in mol.GetBonds())
    ):
        return FusedHydrogenRecord()
    carbonyls = tuple(g.center for g in inventory.principal)
    if not all(mol.GetAtomWithIdx(i).IsInRing() for i in carbonyls):
        return FusedHydrogenRecord()
    candidates = []
    for template, match in _cached_iter_fused_template_matches(
        ctx,
        ctx.candidate_templates(_templates(Chem)),
        Chem,
        topology_query=True,
        require_one_ring_system=True,
        reject_extra_core_bonds=True,
    ):
        core = set(match)
        if not set(carbonyls) <= core or ring_system_extends_core(mol, core):
            continue
        for reference in _reference_states(template["mol"]):
            record = _render_state(
                mol, Chem, template, match, reference, carbonyls, mapped=mapped
            )
            if record is not None:
                key, name = record
                candidates.append(
                    (
                        key,
                        FusedHydrogenRecord(
                            name,
                            tuple(
                                (idx, template["locants"][j])
                                for j, idx in enumerate(match)
                            ),
                            carbonyls,
                            mapped,
                        ),
                    )
                )
    return (
        min(candidates, key=lambda x: x[0])[1] if candidates else FusedHydrogenRecord()
    )


def _render_state(mol, Chem, template, match, reference, carbonyls, *, mapped=False):
    from .parent_rules import _simple_or_branch_prefix

    core = set(match)
    reverse = {idx: j for j, idx in enumerate(match)}
    locants = template["locants"]
    deltas, prefixes, oxygens, indicated = {}, [], set(), []
    bindings = {idx: locants[j] for j, idx in enumerate(match)}
    embedded = set()
    for j, idx in enumerate(match):
        atom, ref = mol.GetAtomWithIdx(idx), reference.GetAtomWithIdx(j)
        external = [b for b in atom.GetBonds() if b.GetOtherAtomIdx(idx) not in core]
        capped = atom.GetTotalNumHs() + sum(b.GetBondTypeAsDouble() for b in external)
        delta = capped - ref.GetTotalNumHs()
        if delta not in {0, 1}:
            return None
        deltas[j] = int(delta)
        if (ref.GetAtomicNum() == 7 and ref.GetTotalNumHs()) or (
            not ref.GetIsAromatic()
            and ref.GetTotalNumHs() == 2
            and all(b.GetBondTypeAsDouble() == 1 for b in ref.GetBonds())
        ):
            indicated.append(locants[j])
        for bond in external:
            nbr = bond.GetOtherAtomIdx(idx)
            other = mol.GetAtomWithIdx(nbr)
            if mapped and other.GetAtomicNum() == 1 and other.GetIsotope():
                bindings[nbr] = locants[j]
                continue
            if (
                idx in carbonyls
                and other.GetAtomicNum() == 8
                and bond.GetBondTypeAsDouble() == 2
            ):
                oxygens.add(nbr)
                bindings[nbr] = "O" + locants[j]
                continue
            if bond.GetBondTypeAsDouble() != 1:
                return (
                    None  # Exocyclic imines/alkenes need separate H/stereo operations.
                )
            if mapped:
                from .component_mapping import mapped_branch

                prefix, atoms = mapped_branch(mol, Chem, nbr, idx)
                from .component_mapping import has_annotations

                if not has_annotations(mol, atoms):
                    from .principal_groups import (
                        principal_ring_groups,
                        alcohol_substituent_prefix,
                    )

                    if principal_ring_groups(mol)[0] == "lactone":
                        prefix = alcohol_substituent_prefix(mol, nbr, idx) or prefix
                embedded.update(atoms)
            else:
                prefix = _simple_or_branch_prefix(mol, nbr, idx)
            if not mapped:
                from .principal_groups import (
                    principal_ring_groups,
                    alcohol_substituent_prefix,
                )

                if principal_ring_groups(mol)[0] == "lactone":
                    prefix = alcohol_substituent_prefix(mol, nbr, idx) or prefix
            if not prefix:
                return None
            prefixes.append((locants[j], prefix))
    if len(oxygens) != len(carbonyls):
        return None
    numbers = tuple(a.GetAtomicNum() for a in reference.GetAtoms())
    edges = tuple((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in reference.GetBonds())
    perimeter = peripheral_ring_atoms(edges)
    if perimeter is None:
        return None
    hydrogens = tuple(a.GetTotalNumHs() for a in reference.GetAtoms())
    states = joint_hydrogen_states(numbers, edges, hydrogens,
        tuple(hydrogens[j]+deltas[j] for j in range(len(match))),
        tuple(2 if idx in carbonyls else 0 for idx in match), annular_pi_bonds(mol, Chem, match))
    possibilities = []
    for state in states:
        hydro = sorted((locants[j] for j in state.hydro), key=_locant_key)
        if len(hydro) % 2 or hydro and len(hydro) not in MULT:
            continue
        added = sorted((locants[j] for j in state.added), key=_locant_key)
        suffix_locs = sorted((locants[reverse[i]] for i in carbonyls), key=_locant_key)
        leading = sorted(indicated, key=_locant_key)
        hbase = (",".join(f"{loc}H" for loc in leading) + "-") if leading else ""
        parent = template["name"]
        stem = parent.removesuffix("e") if len(carbonyls) == 1 else parent
        hpart = "(" + ",".join(f"{loc}H" for loc in added) + ")" if added else ""
        suffix = multiplied_suffix(len(carbonyls), "one")
        name = hbase + stem + "-" + ",".join(suffix_locs) + hpart + "-" + suffix
        hydro_prefix = ','.join(hydro)+'-'+MULT[len(hydro)]+'hydro' if hydro else ''
        if hydro and not mapped:
            name = (
                ",".join(hydro)
                + "-"
                + MULT[len(hydro)]
                + "hydro"
                + ("-" if leading else "")
                + name
            )
        if mapped:
            from .component_mapping import (
                MappedComponent,
                decorate_component,
                fully_covered,
            )

            if not fully_covered(mol, bindings, embedded):
                continue
            name = decorate_component(
                mol,
                Chem,
                MappedComponent(
                    name,
                    tuple(bindings.items()),
                    frozenset(bindings),
                    tuple(prefixes),
                    frozenset(embedded),
                    hydro_prefix=hydro_prefix,
                    systematic_parent=True,
                ),
            )
            if not name:
                continue
        else:
            name = join_prefix_parent(format_prefixes(prefixes), name)
        accommodation = sum(locants[reverse[i]] in leading for i in carbonyls)
        # P-58.2.3.1.1: when enough indicated H is available, first use it
        # to accommodate suffix sites. With fewer indicated H atoms, the
        # lowest permissible indicated-H locants take precedence (1.3).
        accommodation_key = -accommodation if len(leading) >= len(carbonyls) else 0
        key = (
            -len(core),
            accommodation_key,
            tuple(map(_locant_key, leading)),
            tuple(map(_locant_key, suffix_locs)),
            -len(state.pairs),
            sum(j not in perimeter for j in state.added),
            tuple(map(_locant_key, added)),
            tuple(map(_locant_key, hydro)),
        )
        possibilities.append((key, name))
    if not possibilities:
        return None
    best = min(key for key, name in possibilities)
    finalists = {name for key, name in possibilities if key == best}
    return (best, next(iter(finalists))) if len(finalists) == 1 else None
