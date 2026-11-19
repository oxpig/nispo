"""P-67.1.3.2 esters of mapped mononuclear phosphorus and sulfur oxoacids."""

from dataclasses import dataclass

from .branch_rules import collect_subtree
from .component_mapping import mapped_general_branch
from .formatting import enclose_name_unit, format_prefixes
from .functional_inventory import CharacteristicGroup
from .naming_methods import MethodDecision


@dataclass(frozen=True, slots=True)
class OxoacidEster:
    center: int
    kind: str
    anion: str
    oxo: tuple[int, ...]
    hydroxyls: tuple[int, ...]
    anions: tuple[int, ...]
    esters: tuple[tuple[int, int], ...]
    organyl: tuple[int, ...]
    nitrogen: int | None = None

    def group(self):
        links = tuple(link for link, _root in self.esters)
        nitrogen = () if self.nitrogen is None else (self.nitrogen,)
        return CharacteristicGroup(self.kind, self.center,
            (self.center, *self.oxo, *self.hydroxyls, *self.anions, *links, *nitrogen),
            self.organyl[0] if len(self.organyl) == 1 else -1, links[0])


def oxoacid_ester_site(mol, atom):
    """Classify the original acid core independently of its arm vocabulary."""
    number = atom.GetAtomicNum()
    if (number not in {15, 16} or atom.IsInRing() or atom.GetFormalCharge()
            or atom.GetIsotope() or atom.GetChiralTag() or atom.GetNumRadicalElectrons()
            or atom.HasQuery()):
        return None
    center = atom.GetIdx()
    oxo, hydroxyls, anions, esters, organyl, nitrogen = [], [], [], [], [], []
    for bond in atom.GetBonds():
        neighbor = bond.GetOtherAtom(atom)
        idx, element = neighbor.GetIdx(), neighbor.GetAtomicNum()
        order = bond.GetBondTypeAsDouble()
        if element == 1:
            if number != 15 or neighbor.GetIsotope():
                return None
        elif element == 8 and order == 2 and neighbor.GetDegree() == 1 and not neighbor.GetFormalCharge():
            oxo.append(idx)
        elif element in {8, 16} and order == 1 and not neighbor.IsInRing():
            children = [n for n in neighbor.GetNeighbors()
                        if n.GetAtomicNum() > 1 and n.GetIdx() != center]
            if not children and element == 8:
                if neighbor.GetFormalCharge() == -1 and not neighbor.GetTotalNumHs(includeNeighbors=True):
                    anions.append(idx)
                elif not neighbor.GetFormalCharge() and neighbor.GetTotalNumHs(includeNeighbors=True) == 1:
                    hydroxyls.append(idx)
                else:
                    return None
            elif len(children) == 1 and not neighbor.GetFormalCharge():
                child = children[0]
                root = child.GetIdx()
                if mol.GetBondBetweenAtoms(idx, root).GetBondTypeAsDouble() != 1:
                    return None
                if child.GetAtomicNum() != 6 and not (
                    number == 16 and element == 8 and child.GetAtomicNum() == 7
                    and child.IsInRing() and not child.GetFormalCharge()
                    and child.GetDegree() == child.GetTotalValence() == 3
                    and all(b.GetBondTypeAsDouble() == 1 for b in child.GetBonds())
                    and all(n.GetIdx() == idx or n.GetAtomicNum() == 6 for n in child.GetNeighbors())
                ):
                    return None
                esters.append((idx, root))
            else:
                return None
        elif element == 6 and order == 1:
            organyl.append(idx)
        elif element == 7 and order == 1 and not neighbor.GetFormalCharge() and neighbor.GetTotalValence() == 3:
            nitrogen.append(idx)
        else:
            return None
    if not esters or len(nitrogen) > 1:
        return None
    if any(mol.GetAtomWithIdx(i).GetIsotope() for i in (*oxo, *hydroxyls, *anions, *nitrogen)):
        return None
    sites = len(hydroxyls) + len(anions) + len(esters)
    if number == 15 and len(oxo) == 1:
        p_substituents = len(organyl) + atom.GetTotalNumHs(includeNeighbors=True)
        anion = {
            (0, 0, 3): 'phosphate', (1, 0, 2): 'phosphonate', (2, 0, 1): 'phosphinate',
            (0, 1, 2): 'phosphoramidate',
        }.get((p_substituents, len(nitrogen), sites))
        kind = 'phosphate' if anions else 'phosphate-acid-ester'
    elif number == 16 and not atom.GetTotalNumHs(includeNeighbors=True):
        anion = {
            (0, 0, 2, 2): 'sulfate', (0, 1, 2, 1): 'sulfamate',
            (1, 0, 2, 1): 'sulfonate', (1, 0, 1, 1): 'sulfinate',
        }.get((len(organyl), len(nitrogen), len(oxo), sites))
        kind = 'sulfur-oxoacid-ester-anion' if anions else 'sulfur-oxoacid-ester'
    else:
        return None
    if anion is None or (any(mol.GetAtomWithIdx(root).GetAtomicNum() == 7 for _, root in esters)
                         and not (anion == 'sulfate' and len(esters) == len(hydroxyls) == 1)):
        return None
    return OxoacidEster(center, kind, anion, tuple(oxo), tuple(hydroxyls), tuple(anions),
                        tuple(esters), tuple(organyl), nitrogen[0] if nitrogen else None)


def oxoacid_ester_decision(mol, Chem, *, amidoxime_arm_only=False):
    """Join mapped organyl words to the selected noncarbon acid anion."""
    from .extended_inventory import selection_inventory
    from .naming_result import SelectionLimitation
    from .phosphorus_ester_methods import _organyl_words
    from .neutral_charge_components import neutral_charge_atom_ids

    if (mol.GetStereoGroups() or len(Chem.GetMolFrags(mol)) != 1
            or any(a.GetIsotope() or a.GetNumRadicalElectrons() or a.HasQuery()
                   for a in mol.GetAtoms())):
        return None
    inventory = selection_inventory(mol)
    if not inventory.supported:
        return None
    kinds = {'phosphate-acid-ester', 'phosphate', 'sulfur-oxoacid-ester',
             'sulfur-oxoacid-ester-anion'}
    general_phosphoramidate = False
    if len(inventory.principal) == 1 and inventory.principal[0].kind in kinds:
        group = inventory.principal[0]
    elif not amidoxime_arm_only and any(proof.kind == 'azido' for proof in inventory.charge_role_proofs):
        # A complete general phosphoramidate name can incorporate subordinate
        # ester groups in its two O-organyl components or its N-organyl arm.
        sites = [g for g in inventory.groups if g.kind == 'phosphate-acid-ester']
        if (len(sites) != 1 or not inventory.principal
                or any(g.kind != 'ester' for g in inventory.principal)):
            return None
        group, = sites
        general_phosphoramidate = True
    else:
        return None
    center = mol.GetAtomWithIdx(group.center)
    element = center.GetAtomicNum()
    site = oxoacid_ester_site(mol, center)
    if site is None or any(mol.GetAtomWithIdx(root).GetAtomicNum() == 7 for _, root in site.esters):
        return None
    if general_phosphoramidate and (site.anion != 'phosphoramidate'
            or site.anions or site.hydroxyls or len(site.esters) != 2):
        return None
    amidoxime_arm = None
    if site.anion == 'phosphonate' and len(site.organyl) == 1 and any(
            g.kind == 'amidoxime' for g in inventory.groups):
        from .branch.carbamimidoyl import amidoxime_alkyl_prefix

        root = site.organyl[0]
        scope = frozenset(collect_subtree(mol, root, site.center))
        amidoxime_arm = amidoxime_alkyl_prefix(mol, Chem, root, site.center, scope)
    if amidoxime_arm_only and amidoxime_arm is None:
        return None
    core = set(site.group().atoms) - {link for link, _ in site.esters}
    esters = [(link, root, frozenset(collect_subtree(mol, root, link)))
              for link, root in site.esters]
    # Build each rooted word on the unchanged graph. The same alcohol
    # components and P-14.5 multiplication rules serve either acid family.
    by_element = {number: [row for row in esters if mol.GetAtomWithIdx(row[0]).GetAtomicNum() == number]
                  for number in (8, 16)}
    words = []
    sulfur_count = len(by_element[16])
    for number, arms in by_element.items():
        if not arms:
            continue
        rows = _organyl_words(mol, Chem, {'ester_oxygens': arms}, general=True)
        if rows is None:
            return None
        for key, word, members in rows:
            if sulfur_count:
                locant = ','.join(['S' if number == 16 else 'O'] * len(members))
                word = locant + '-' + (word if word.isalpha() else enclose_name_unit(word))
            words.append((key, word, members))
    words.sort(key=lambda row: row[0])
    if not words:
        return None
    charged = {a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge()} - set(
        neutral_charge_atom_ids(inventory.charge_role_proofs))
    if charged and not _ester_charge_balance(mol, site, words, charged):
        return None
    roles = [('ester-organyl', frozenset().union(*(r['owned'] for r in rows)), ())
             for _key, _word, rows in words]
    anion = site.anion
    if site.nitrogen is not None:
        n_prefixes = []
        for atom in mol.GetAtomWithIdx(site.nitrogen).GetNeighbors():
            if atom.GetIdx() == site.center or atom.GetAtomicNum() == 1:
                continue
            if mol.GetBondBetweenAtoms(site.nitrogen, atom.GetIdx()).GetBondTypeAsDouble() != 1:
                return None
            text, owned = mapped_general_branch(mol, Chem, atom.GetIdx(), site.nitrogen)
            if not text or owned & core:
                return None
            core.update(owned)
            n_prefixes.append(('N', text))
        anion = format_prefixes(n_prefixes, omit_locants=len(n_prefixes) == 1,
                                hierarchical_enclosures=True, separate_unlocanted=True) + anion
    elif site.organyl:
        if element == 16:
            from .sulfur_ester_methods import mapped_sulfur_ester_anion

            rendered = mapped_sulfur_ester_anion(mol, Chem, site)
            if rendered is None:
                return None
            anion, owned = rendered
            core.update(owned)
        elif amidoxime_arm is not None:
            text, owned = amidoxime_arm
            anion = enclose_name_unit(text) + anion
            core.update(owned)
        else:
            from .phosphorus_ester_methods import _acid_core_parts

            site_parts = {'p': site.center, 'p_roots': site.organyl, 'isotope_atoms': (),
                          'ester_oxygens': esters, 'acid_kind': {
                              'phosphonate': 'phosphonic acid', 'phosphinate': 'phosphinic acid',
                          }[site.anion]}
            parts = _acid_core_parts(mol, Chem, site_parts)
            if parts is None:
                return None
            anion = ''.join(parts)
            core.update(i for root in site.organyl for i in collect_subtree(mol, root, site.center))
    roles.append(('oxoacid', frozenset(core), ()))
    heavy = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    if sum(len(atoms) for _, atoms, _ in roles) != len(heavy) or (
            frozenset().union(*(atoms for _, atoms, _ in roles)) != heavy):
        return None
    hydrogen = {0: (), 1: ('hydrogen',), 2: ('dihydrogen',)}.get(len(site.hydroxyls))
    if hydrogen is None:
        return None
    if sulfur_count:
        from .constants import MULT

        if site.nitrogen is not None:
            return None  # Concurrent amido/thio replacement needs its own infix order.
        base = {
            'phosphate': 'phosphoro', 'phosphonate': 'phosphono', 'phosphinate': 'phosphino',
            'sulfate': 'sulfuro', 'sulfonate': 'sulfono', 'sulfinate': 'sulfino',
        }[site.anion]
        anion = anion.removesuffix(site.anion) + base + MULT.get(sulfur_count, '') + 'thioate'
    if general_phosphoramidate:
        name = ' '.join([*(word if word.isalpha() else enclose_name_unit(word)
                           for _, word, _ in words), anion])
        rule = 'P-65.6.3.2/P-67.1.3.2'
        return MethodDecision(name, 'functional-class', frozenset({site.center}), heavy, rule,
                              component_roles=tuple(roles), preference_issues=(SelectionLimitation(
                                  'phosphoramidate-general', 'P-41/P-51/P-67.1.3.2',
                                  'Both organyl esters and the amido arm are mapped; '
                                  'preferred acid-class selection remains unresolved.'),))
    name = ' '.join([*(word for _, word, _ in words), *hydrogen, anion])
    rule = 'P-41/P-65.6.3.2/P-67.1.3.2'
    return MethodDecision(name, 'functional-class', frozenset({site.center}), heavy, rule,
                          component_roles=tuple(roles), preference_issues=(SelectionLimitation(
                              'oxoacid-ester', rule, 'The acid core and organyl components are mapped.'),))


def _ester_charge_balance(mol, site, words, charged):
    """A zwitterionic ester needs the anion and its mapped cationic organyl word."""
    if len(site.anions) != 1:
        return False
    cations = charged - set(site.anions)
    if len(cations) != 1:
        return False
    idx = next(iter(cations))
    atom = mol.GetAtomWithIdx(idx)
    return (atom.GetAtomicNum() == 7 and atom.GetFormalCharge() == 1
            and any(idx in row['owned'] and ('ium' in row['text'] or 'ammonio' in row['text'])
                    for _, _, rows in words for row in rows))
