"""Run seven generated/control acceptance gates against one package.

Each gate runs in a separate process. Preferred-construction obligations remain
outside pytest; passing them does not certify whole names as PINs.
"""
from __future__ import annotations

# ---- frozen_import_scope.py ----
"""Prevent editable-install fallback from filling gaps in a selected source."""
import importlib.abc
from importlib.machinery import PathFinder
from pathlib import Path
import sys

class _NispoScope(importlib.abc.MetaPathFinder):

    def __init__(self, root):
        self.root = root
        self.package = root / 'nispo'

    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'nispo' and (not fullname.startswith('nispo.')):
            return None
        spec = PathFinder.find_spec(fullname, path)
        if spec is None:
            raise ModuleNotFoundError('Module is absent from the selected NISPO source: ' + fullname, name=fullname)
        if spec.origin is not None:
            if not Path(spec.origin).resolve().is_relative_to(self.package):
                raise ImportError('NISPO import escaped the selected package source: ' + fullname)
        elif not spec.submodule_search_locations or any((not Path(folder).resolve().is_relative_to(self.package) for folder in spec.submodule_search_locations)):
            raise ImportError('NISPO namespace escaped the selected package source: ' + fullname)
        return spec

def check_nispo_scope(root):
    package = Path(root).resolve() / 'nispo'
    count = 0
    for name, module in tuple(sys.modules.items()):
        if name != 'nispo' and (not name.startswith('nispo.')):
            continue
        source = getattr(module, '__file__', None)
        if source is not None:
            if not Path(source).resolve().is_relative_to(package):
                raise RuntimeError('Loaded NISPO module escaped selected source: ' + name)
            count += 1
    return count

def isolate_nispo_imports(root):
    root = Path(root).resolve()
    check_nispo_scope(root)
    sys.path.insert(0, str(root))
    if not any((isinstance(finder, _NispoScope) and finder.root == root for finder in sys.meta_path)):
        sys.meta_path.insert(0, _NispoScope(root))


# ---- audit_public_writers422.py ----
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from rdkit import Chem

def canonical(mol):
    return Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True, isomericSmiles=True)

def opsin(names):
    with tempfile.TemporaryDirectory(prefix='nispo-public422-') as directory:
        source = Path(directory) / 'names.txt'
        target = Path(directory) / 'structures.tsv'
        source.write_text('\n'.join(names) + '\n')
        subprocess.run([os.environ.get('JAVA', 'java'), '-jar', os.environ['NISPO_OPSIN_JAR'], '-o', 'smi', '-n', str(source), str(target)], capture_output=True, check=True, timeout=120)
        rows = target.read_text().splitlines()
        if len(rows) != len(names):
            raise AssertionError('OPSIN row count changed')
        return [Chem.MolFromSmiles(row.partition('\t')[0]) if row.partition('\t')[0] else None for row in rows]


# ---- audit_gap_sm1_principal_composition.py ----
"""SM1 independently constructed repeated-functional-group matrix, not a panel."""
import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import random
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

def controls():
    rows = []
    functions = (('acid', 'OC(=O)', 'C(=O)O', 'pentanedioic acid'), ('ester', 'COC(=O)', 'C(=O)OC', 'dimethyl pentanedioate'), ('amide', 'NC(=O)', 'C(=O)N', 'pentanediamide'), ('nitrile', 'N#C', 'C#N', 'pentanedinitrile'))
    spacers = (('bare', 'CCC', ''), ('hydroxy', 'C(O)CC', '2-hydroxy'), ('oxo', 'CC(=O)C', '3-oxo'), ('amino', 'C(N)CC', '2-amino'), ('fluoro', 'C(F)CC', '2-fluoro'), ('methyl', 'C(C)CC', '2-methyl'), ('three-lower', 'C(O)C(=O)C(N)', '2-amino-4-hydroxy-3-oxo'), ('two-lower', 'C(O)CC(F)', '2-fluoro-4-hydroxy'), ('E', '/C=C/C', None), ('Z', '/C=C\\C', None), ('hydroxy-R', '[C@H](O)CC', None), ('hydroxy-S', '[C@@H](O)CC', None), ('RS-E', '[C@H](O)/C=C/CC', None), ('RS-Z', '[C@@H](O)/C=C\\CC', None), ('carbon13', 'C[13CH2]C', None), ('off-parent-R', 'CC([C@H](O)C)C', None), ('off-parent-S', 'CC([C@@H](O)C)C', None))
    for kind, left, right, parent in functions:
        for label, middle, prefix in spacers:
            expected = None
            if prefix is not None:
                expected = 'dimethyl ' + prefix + parent.removeprefix('dimethyl ') if kind == 'ester' else prefix + parent
            rows.append(dict(id=kind + ':' + label, smiles=left + middle + right, expected=expected))
        for label, center in (('bare', 'C'), ('carbon13', '[13CH]')):
            rows.append(dict(id=kind + ':three-sites-' + label, smiles=center + '(' + right + ')(' + right + ')' + right, expected=None))
    classes = (('acid', 'C(=O)O'), ('ester', 'C(=O)OC'), ('carbamate', 'OC(=O)N'), ('amide', 'C(=O)N'), ('sulfonamide', 'S(=O)(=O)N'), ('urea', 'NC(=O)N'), ('nitrile', 'C#N'), ('aldehyde', 'C=O'), ('ketone', 'C(=O)C'), ('alcohol', 'O'), ('thiol', 'S'), ('amine', 'N'))
    for (senior, left), (junior, right) in combinations(classes, 2):
        for scaffold, smiles in (('chain', 'CCC(' + left + ')(' + right + ')CCC'), ('benzene', left + 'c1ccc(' + right + ')cc1')):
            if scaffold == 'benzene':
                smiles = 'c1(' + left + ')ccc(' + right + ')cc1'
            rows.append(dict(id=f'classes:{senior}>{junior}:{scaffold}', smiles=smiles, expected=None, principal=senior))
    return rows

def _default_result(result, completed_name):
    """Check the bounded API contract without requiring a fallback synonym."""
    if result is None:
        return dict(name='', generation_status='missing', preferred_status='not-assessed', route='', unresolved=[], runtime_limited=False, matches_completed_full=False, contract=False)
    issues = [dict(stage=issue.stage, rule=issue.rule, reason=issue.reason) for issue in result.unresolved]
    limited = any((issue['stage'] == 'runtime' and issue['rule'] in {'timeout', 'full-timeout'} for issue in issues))
    if limited:
        valid = result.generation_status == 'available' and result.preferred_status == 'unresolved' if result.name else result.generation_status == 'timeout' and result.preferred_status == 'not-assessed'
    else:
        valid = bool(completed_name and result.name == completed_name and (result.generation_status == 'available'))
    return dict(name=result.name, generation_status=result.generation_status, preferred_status=result.preferred_status, route=result.route, unresolved=issues, runtime_limited=limited, matches_completed_full=result.name == completed_name, contract=valid)

def run(root, output, diagnose=False):
    isolate_nispo_imports(root)
    import nispo
    from nispo.extended_inventory import selection_inventory
    cases = controls()
    rows = []
    mols = []
    names = []
    for case in cases:
        mol = Chem.MolFromSmiles(case['smiles'])
        assert mol is not None, case['id']
        snapshot = Chem.MolToMolBlock(mol)
        result = nispo.mol_to_iupac_result(mol, timeout_seconds=None)
        order = list(range(mol.GetNumAtoms()))
        random.Random(case['id']).shuffle(order)
        row = dict(id=case['id'], nonempty=bool(result.name), source_name=None if case['expected'] is None else result.name == case['expected'], scalar_parity=result.name == nispo.mol_to_iupac(mol, timeout_seconds=None), atom_order=result.name == nispo.mol_to_iupac(Chem.RenumberAtoms(mol, order), timeout_seconds=None), nonmutation=snapshot == Chem.MolToMolBlock(mol), preferred_status=result.preferred_status)
        inventory = selection_inventory(mol)
        row['source_class'] = None if 'principal' not in case else inventory.supported and bool(inventory.principal) and all((group.kind == case['principal'] for group in inventory.principal))
        row['source_parent'] = None if 'parent_ending' not in case else result.name.endswith(case['parent_ending'])
        row['source_prefix'] = None if 'preferred_unit' not in case else case['preferred_unit'] in result.name
        if diagnose:
            row.update(name=result.name, expected=case['expected'], unresolved=[(r.rule, r.reason) for r in result.unresolved])
        rows.append(row)
        mols.append(mol)
        names.append(result.name)
    for row, mol, parsed in zip(rows, mols, opsin(names), strict=True):
        row['raw_OPSIN_graph'] = parsed is not None and canonical(mol) == canonical(parsed)
        row['formula'] = parsed is not None and rdMolDescriptors.CalcMolFormula(mol, separateIsotopes=True) == rdMolDescriptors.CalcMolFormula(parsed, separateIsotopes=True)
    inputs = [case['smiles'] for case in cases]
    batched = nispo.smiles_to_iupac_batch(inputs, timeout_seconds=None)
    differences = []
    for index in range(max(len(cases), len(batched))):
        case = cases[index] if index < len(cases) else None
        expected = names[index] if index < len(names) else None
        actual = batched[index] if index < len(batched) else None
        if index >= len(cases) or index >= len(batched) or expected != actual:
            differences.append(dict(index=index, id=case['id'] if case else None, smiles=case['smiles'] if case else None, scalar_name=expected, batch_name=actual))
    default = nispo.smiles_to_iupac_batch_results(inputs)
    runtime_events = []
    for index, (case, row, name) in enumerate(zip(cases, rows, names, strict=True)):
        receipt = _default_result(default[index] if index < len(default) else None, name)
        row['default_result'] = receipt
        if receipt['runtime_limited'] or not receipt['contract']:
            runtime_events.append(dict(index=index, id=case['id'], smiles=case['smiles'], completed_full_name=name, default_result=receipt))
    default_contract = len(default) == len(cases) and all((row['default_result']['contract'] for row in rows))
    flags = ('nonempty', 'scalar_parity', 'atom_order', 'nonmutation', 'raw_OPSIN_graph', 'formula')
    failures = [r['id'] for r in rows if any((not r[k] for k in flags)) or any((r[k] is False for k in ('source_name', 'source_class', 'source_parent', 'source_prefix'))) or (not r['default_result']['contract'])]
    failures.extend((difference['id'] or 'batch-output-count' for difference in differences))
    if len(default) != len(cases):
        failures.append('default-batch-output-count')
    report = dict(source_root=str(root.resolve()), total=len(rows), counts={k: sum((r[k] for r in rows)) for k in flags}, source_total=sum((r['source_name'] is not None for r in rows)), source_matches=sum((r['source_name'] is True for r in rows)), class_total=sum((r['source_class'] is not None for r in rows)), class_matches=sum((r['source_class'] is True for r in rows)), parent_total=sum((r['source_parent'] is not None for r in rows)), parent_matches=sum((r['source_parent'] is True for r in rows)), prefix_total=sum((r['source_prefix'] is not None for r in rows)), prefix_matches=sum((r['source_prefix'] is True for r in rows)), preferred_status_counts=dict(Counter((r['preferred_status'] for r in rows))), batch_parity=not differences, batch_parity_scope='completed full naming; timeout_seconds=None', batch_differences=differences, default_batch_contract=default_contract, default_batch_result_count=len(default), default_batch_extra_results=[_default_result(result, '') for result in default[len(cases):]], default_batch_runtime_limited=sum((row['default_result']['runtime_limited'] for row in rows)), default_runtime_events=runtime_events, failures=list(dict.fromkeys(failures)), rows=rows)
    check_nispo_scope(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    summary = {k: v for k, v in report.items() if k != 'rows'}
    summary['failure_total'] = len(report['failures'])
    if len(report['failures']) > 20:
        summary['failures'] = report['failures'][:20]
        summary['failure_list_truncated'] = True
    print(json.dumps(summary, indent=2))
    return int(bool(report['failures']) or not report['batch_parity'] or (not default_contract))

def _gate_sm1_principal_composition():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--diagnose', action='store_true')
    a = p.parse_args()
    raise SystemExit(run(a.source_root, a.output, a.diagnose))


# ---- audit_gap_sm1_carbamoyl.py ----
"""Source-qualified mapped carbamoyl vocabulary, independent of external panels."""
import argparse
from pathlib import Path

def _gap_sm1_carbamoyl_controls():
    rows = []
    for kind, head, parent, owner in (('acid', 'C(=O)O', 'benzoic acid', ''), ('ester', 'C(=O)OC', 'benzoate', 'methyl '), ('sulfonic', 'S(=O)(=O)O', 'benzenesulfonic acid', '')):
        for unit, fragment in (('carbamoyl', 'C(=O)N'), ('methylcarbamoyl', 'C(=O)NC'), ('dimethylcarbamoyl', 'C(=O)N(C)C')):
            for count in (1, 2):
                ring = 'c1ccc(' + fragment + ')cc1' if count == 1 else 'c1cc(' + fragment + ')cc(' + fragment + ')c1'
                prefix = '4-' + unit if unit == 'carbamoyl' else '4-(' + unit + ')'
                expected = owner + prefix + parent if count == 1 and kind in {'acid', 'ester'} else None
                target = unit if count == 1 else 'dicarbamoyl' if unit == 'carbamoyl' else 'bis(' + unit + ')'
                rows.append(dict(id=f'{kind}:{unit}:{count}', smiles=ring[:2] + '(' + head + ')' + ring[2:], expected=expected, preferred_unit=target))
    for smiles in ('NC(=O)CCC(=O)N', 'NC(=O)NC', '[13CH3]C(=O)N', 'OC(=O)[C@H](c1ccc(C(=O)N)cc1)CC(=O)O'):
        rows.append(dict(id='graph:' + str(len(rows)), smiles=smiles, expected=None))
    return rows

def _gate_sm1_carbamoyl():
    global controls
    controls = _gap_sm1_carbamoyl_controls
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--diagnose', action='store_true')
    a = p.parse_args()
    raise SystemExit(run(a.source_root, a.output, a.diagnose))


# ---- audit_gap_sm2_parent_composition.py ----
"""Independent ordinary ring/chain site-count and numbering obligations."""
import argparse
from pathlib import Path

def _gap_sm2_parent_composition_controls():
    rows = []
    functions = (('acid', 'OC(=O)', 'C(=O)O', 'dioic acid'), ('ester', 'COC(=O)', 'C(=O)OC', 'dioate'), ('amide', 'NC(=O)', 'C(=O)N', 'diamide'), ('nitrile', 'N#C', 'C#N', 'dinitrile'))
    rings = (('benzene', 'c1ccc({g})cc1', 'c1({g})cc({s})cc({g})c1'), ('cyclohexane', 'C1CCC({g})CC1', 'C1({g})CC({s})CC({g})C1'), ('pyridine', 'c1cc({g})ccn1', 'c1({g})cc({s})cc({g})n1'), ('furan', 'c1cc({g})co1', 'c1({g})c({s})cc({g})o1'), ('thiophene', 'c1cc({g})cs1', 'c1({g})c({s})cc({g})s1'), ('pyrrole', 'c1cc({g})c[nH]1', 'c1({g})c({s})cc({g})[nH]1'))
    for kind, left, right, ending in functions:
        for ring, one, two in rings:
            for label, center in (('unspecified', 'C'), ('assigned', '[C@H]')):
                core = one.format(g=right)
                rows.append(dict(id=f'{kind}:{ring}:chain-two:{label}', smiles=left + center + '(' + core + ')C' + right, expected=None, principal=kind, parent_ending=ending))
            rows.append(dict(id=f'{kind}:{ring}:one-each', smiles=left + 'CC' + one.format(g=right), expected=None, principal=kind, parent_ending={'acid': ('carboxylic acid', 'benzoic acid', 'nicotinic acid', 'isonicotinic acid', 'picolinic acid'), 'amide': ('carboxamide', 'benzamide'), 'nitrile': ('carbonitrile', 'benzonitrile'), 'ester': ('carboxylate', 'benzoate')}[kind]))
            rows.append(dict(id=f'{kind}:{ring}:ring-two', smiles=two.format(g=right, s='CC' + right), expected=None, principal=kind, parent_ending={'acid': 'dicarboxylic acid', 'amide': 'dicarboxamide', 'nitrile': 'dicarbonitrile', 'ester': 'dicarboxylate'}[kind]))
    for label, middle, prefix in (('longest', 'C(CC)(CCC)C', '2-ethyl-2-methyl'), ('suffix-first', 'CC(O)C', '3-hydroxy'), ('unsaturation-first', '/C=C/C(C)C', '4-methyl')):
        rows.append(dict(id='numbering:' + label, smiles='OC(=O)' + middle, expected={'longest': '2-ethyl-2-methylpentanoic acid', 'suffix-first': '3-hydroxybutanoic acid', 'unsaturation-first': '(2E)-4-methylpent-2-enoic acid'}.get(label), principal='acid'))
    for label, smiles in (('ethyl-dipyridine', 'CCOC(=O)c1cncc(C(=O)OCC)c1'), ('N-methyl-pyrrole', 'COC(=O)c1cc(C(=O)OC)n(C)c1'), ('ring-carbon13', 'COC(=O)[13CH]1CC(C(=O)OC)CCN1'), ('ring-RS-pair', 'COC(=O)[C@H]1CC[C@@H](C(=O)OC)CN1'), ('ring-RS-carbon13', 'COC(=O)[13C@H]1CC[C@@H](C(=O)OC)CN1'), ('phenyl-dihydropyridine', 'COC(=O)C1=C(C)NC(C)=C(C(=O)OC)C1c1ccccc1'), ('lactam', 'CCOC(=O)C1CC(=O)NCC1'), ('N-methyl-piperidine', 'CCOC(=O)C1CCN(C)CC1'), ('fluoro-pyridine', 'CCOC(=O)c1cc(F)ncc1C(=O)OCC')):
        rows.append(dict(id='ester-extension:' + label, smiles=smiles, expected=None, principal='ester'))
    rows.append(dict(id='ester-extension:printed-P65-dihydropyridine', smiles='COC(=O)C1=C(C)NC(C)=C(C(=O)OC)C1c1ccccc1[N+](=O)[O-]', expected='dimethyl 2,6-dimethyl-4-(2-nitrophenyl)-1,4-dihydropyridine-3,5-dicarboxylate', principal='ester'))
    return rows

def _gate_sm2_parent_composition():
    global controls
    controls = _gap_sm2_parent_composition_controls
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--diagnose', action='store_true')
    a = p.parse_args()
    raise SystemExit(run(a.source_root, a.output, a.diagnose))


# ---- audit_gap_sm3_heteromonocycles.py ----
"""Finite source-motivated N/O/S monocycle, H-state and carbonyl matrix."""
import argparse
from itertools import combinations
from pathlib import Path
from rdkit import Chem

def _gap_sm3_heteromonocycles_patterns():
    for size in (5, 6, 7):
        for symbol, number in (('N', 7), ('O', 8), ('S', 16)):
            yield (f'{size}-{symbol}', (number,) + (6,) * (size - 1))
        placements = {'NN12': {0: 7, 1: 7}, 'NN13': {0: 7, 2: 7}, 'ON12': {0: 8, 1: 7}, 'ON13': {0: 8, 2: 7}, 'SN12': {0: 16, 1: 7}, 'SN13': {0: 16, 2: 7}, 'NNN123': {0: 7, 1: 7, 2: 7}, 'NNN124': {0: 7, 1: 7, 3: 7}, 'ONN124': {0: 8, 1: 7, 3: 7}, 'SNN124': {0: 16, 1: 7, 3: 7}} if size == 5 else {'NN12': {0: 7, 1: 7}, 'NN13': {0: 7, 2: 7}, 'NN14': {0: 7, 3: 7}, 'ON14': {0: 8, 3: 7}, 'SN14': {0: 16, 3: 7}, 'NNN135': {0: 7, 2: 7, 4: 7}, 'ONN134': {0: 8, 2: 7, 3: 7}, 'SNN134': {0: 16, 2: 7, 3: 7}}
        for label, placement in placements.items():
            yield (f'{size}-{label}', tuple((placement.get(i, 6) for i in range(size))))

def _gap_sm3_heteromonocycles_ring_graph(numbers, matching):
    graph = Chem.RWMol()
    for number in numbers:
        graph.AddAtom(Chem.Atom(number))
    for i in range(len(numbers)):
        graph.AddBond(i, (i + 1) % len(numbers), Chem.BondType.DOUBLE if i in matching else Chem.BondType.SINGLE)
    mol = graph.GetMol()
    Chem.SanitizeMol(mol)
    return mol

def _gap_sm3_heteromonocycles_attach(mol, site, fragment, order=Chem.BondType.SINGLE):
    component = Chem.MolFromSmiles(fragment)
    assert component is not None
    graph = Chem.RWMol(Chem.CombineMols(mol, component))
    graph.AddBond(site, mol.GetNumAtoms(), order)
    result = graph.GetMol()
    Chem.SanitizeMol(result)
    return result

def _gap_sm3_heteromonocycles_controls():
    rows = []
    seen = set()
    simple = {5: {7: 'pyrrolidine', 8: 'oxolane', 16: 'thiolane'}, 6: {7: 'piperidine', 8: 'oxane', 16: 'thiane'}, 7: {7: 'azepane', 8: 'oxepane', 16: 'thiepane'}}

    def add(cid, mol, expected=None, principal=None):
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        if smiles in seen:
            return
        seen.add(smiles)
        row = dict(id=cid, smiles=smiles, expected=expected)
        if principal:
            row['principal'] = principal
        rows.append(row)
    for label, numbers in _gap_sm3_heteromonocycles_patterns():
        size = len(numbers)
        edges = range(size)
        for count in range(size // 2 + 1):
            for matching in combinations(edges, count):
                endpoints = [v for i in matching for v in (i, (i + 1) % size)]
                if len(set(endpoints)) != len(endpoints) or any((numbers[i] not in {6, 7} for i in endpoints)):
                    continue
                mol = _gap_sm3_heteromonocycles_ring_graph(numbers, matching)
                cid = label + ':pi-' + ','.join(map(str, matching))
                bare = simple[size][numbers[0]] if count == 0 and label in {f'{size}-N', f'{size}-O', f'{size}-S'} else None
                add(cid, mol, bare)
                carbon = [i for i in edges if numbers[i] == 6 and mol.GetAtomWithIdx(i).GetTotalNumHs() > 0]
                if carbon:
                    first = carbon[0]
                    for family, fragment in (('alcohol', 'O'), ('acid', 'C(=O)O'), ('amide', 'C(=O)N'), ('nitrile', 'C#N')):
                        add(cid + ':' + family, _gap_sm3_heteromonocycles_attach(mol, first, fragment), principal=family)
                    isotope = Chem.Mol(mol)
                    isotope.GetAtomWithIdx(first).SetIsotope(13)
                    add(cid + ':carbon13', isotope)
                oxo = [i for i in carbon if mol.GetAtomWithIdx(i).GetTotalNumHs() >= 2]
                for site in oxo:
                    expected = bare.removesuffix('e') + '-2-one' if bare and site == 1 else None
                    add(cid + f':oxo-{site}', _gap_sm3_heteromonocycles_attach(mol, site, 'O', Chem.BondType.DOUBLE), expected)
                if len(oxo) > 1:
                    a, b = oxo[:2]
                    one = _gap_sm3_heteromonocycles_attach(mol, a, 'O', Chem.BondType.DOUBLE)
                    add(cid + f':dioxo-{a},{b}', _gap_sm3_heteromonocycles_attach(one, b, 'O', Chem.BondType.DOUBLE))
    return rows

def _gate_sm3_heteromonocycles():
    global controls
    controls = _gap_sm3_heteromonocycles_controls
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--diagnose', action='store_true')
    a = p.parse_args()
    raise SystemExit(run(a.source_root, a.output, a.diagnose))


# ---- audit_gap_sm4_fused_composition.py ----
"""Independent two-face N/O/S fusion, hydrogenation and carbonyl controls."""
import argparse
from itertools import combinations
from pathlib import Path
from rdkit import Chem

def _gap_sm4_fused_composition_patterns():
    for left, right in ((5, 5), (5, 6), (6, 6)):
        placements = (('N', {2: 7}), ('O', {2: 8}), ('S', {2: 16}), ('NN', {2: 7, 3: 7}), ('ON', {2: 8, 3: 7}), ('SN', {2: 16, 3: 7}), ('N-N', {2: 7, left: 7}), ('O-N', {2: 8, left: 7}), ('S-N', {2: 16, left: 7}), ('ON-NN', {2: 8, 3: 7, left: 7, left + 1: 7}))
        for label, placement in placements:
            yield (f'{left}+{right}:{label}', left, right, placement)
        for label, placement in (('junction-N', {0: 7}), ('junction-NN', {0: 7, 2: 7}), ('junction-NNN', {0: 7, 2: 7, left: 7})):
            yield (f'{left}+{right}:{label}', left, right, placement)

def _gap_sm4_fused_composition_states(left, right, placement):
    size = left + right - 2
    edges = tuple([(i, i + 1) for i in range(left - 1)] + [(left - 1, 0)] + [(1, left)] + [(i, i + 1) for i in range(left, size - 1)] + [(size - 1, 0)])
    numbers = tuple((placement.get(i, 6) for i in range(size)))
    degrees = {i: sum((i in edge for edge in edges)) for i in range(size)}
    matchings = []
    for count in range(size // 2 + 1):
        for matching in combinations(range(len(edges)), count):
            endpoints = [i for edge in matching for i in edges[edge]]
            if len(set(endpoints)) == len(endpoints) and all((numbers[i] == 6 or (numbers[i] == 7 and degrees[i] == 2) for i in endpoints)):
                matchings.append(matching)
    maximum = max(map(len, matchings))
    for matching in matchings:
        if len(matching) not in {0, 1, maximum}:
            continue
        graph = Chem.RWMol()
        for number in numbers:
            graph.AddAtom(Chem.Atom(number))
        for j, (a, b) in enumerate(edges):
            graph.AddBond(a, b, Chem.BondType.DOUBLE if j in matching else Chem.BondType.SINGLE)
        mol = graph.GetMol()
        Chem.SanitizeMol(mol)
        yield (matching, mol)

def _gap_sm4_fused_composition_controls():
    rows = []
    seen = set()

    def add(cid, mol, principal=None):
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        if smiles in seen:
            return
        seen.add(smiles)
        row = dict(id=cid, smiles=smiles, expected=None)
        if principal:
            row['principal'] = principal
        rows.append(row)
    for label, left, right, placement in _gap_sm4_fused_composition_patterns():
        for matching, mol in _gap_sm4_fused_composition_states(left, right, placement):
            cid = label + ':pi-' + ','.join(map(str, matching))
            add(cid, mol)
            carbon = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6 and a.GetTotalNumHs() and (a.GetDegree() == 2)]
            if carbon:
                for family, fragment in (('acid', 'C(=O)O'), ('amide', 'C(=O)N'), ('nitrile', 'C#N')):
                    add(cid + ':' + family, _gap_sm3_heteromonocycles_attach(mol, carbon[0], fragment), family)
            oxo = [i for i in carbon if mol.GetAtomWithIdx(i).GetTotalNumHs() >= 2]
            if oxo:
                add(cid + ':oxo', _gap_sm3_heteromonocycles_attach(mol, oxo[0], 'O', Chem.BondType.DOUBLE))
            if len(oxo) > 1:
                one = _gap_sm3_heteromonocycles_attach(mol, oxo[0], 'O', Chem.BondType.DOUBLE)
                add(cid + ':dioxo', _gap_sm3_heteromonocycles_attach(one, oxo[1], 'O', Chem.BondType.DOUBLE))
    return rows

def _gate_sm4_fused_composition():
    global controls
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--diagnose', action='store_true')
    p.add_argument('--smoke', action='store_true', help='First declared state of each topology/heteroatom family.')
    a = p.parse_args()
    recipe = _gap_sm4_fused_composition_controls
    if a.smoke:

        def selected():
            rows = recipe()
            seen = set()
            out = []
            for row in rows:
                family = row['id'].split(':pi-')[0]
                if family not in seen:
                    seen.add(family)
                    out.append(row)
            return out
        controls = selected
    else:
        controls = _gap_sm4_fused_composition_controls
    raise SystemExit(run(a.source_root, a.output, a.diagnose))


# ---- audit_gap_sm5_native_stereo.py ----
"""Independent R/S and E/Z compositions on original ordinary ring graphs."""
import argparse
from pathlib import Path
from rdkit import Chem

def _gap_sm5_native_stereo_controls():
    seeds = (('annular-imine', 'C1=NCCC1'), ('ON-carbonyl', 'O=C1C=NCCO1'), ('SN-carbonyl', 'O=C1C=NCCS1'), ('seven-N-carbonyl', 'O=C1CCNCCC1'), ('O-carbonyl', 'O=C1OCCCC1'), ('S-carbonyl', 'O=C1SCCCC1'), ('fused-N-carbonyl', 'O=C1NC2=CC=CC=C2C1'), ('fused-NN-carbonyl', 'O=C1NC2=C(N=C1)CCCC2'), ('fused-NO', 'C1COC2CNCCC2C1'), ('fused-NS', 'C1CSC2CNCCC2C1'), ('fused-junction-N-carbonyl', 'O=C1CN2CCCC2C1'), ('fused-junction-NN-carbonyl', 'O=C1CCN2CCNC2C1'))
    rows = []
    seen = set()
    for label, smiles in seeds:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, label
        site = next((a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6 and a.IsInRing() and (a.GetDegree() == 2) and (a.GetTotalNumHs() >= 2)))
        for family, group in (('hydride', None), ('acid', 'C(=O)O'), ('amide', 'C(=O)N'), ('nitrile', 'C#N'), ('ester', 'C(=O)OC')):
            base = mol if group is None else _gap_sm3_heteromonocycles_attach(mol, site, group)
            for tag in ('@', '@@'):
                for geometry, tail in (('E', '/C=C/C'), ('Z', '/C=C\\C'), ('unassigned', 'C=CC')):
                    branch = Chem.MolFromSmiles(f'[*][C{tag}H](O)' + tail)
                    joined = Chem.RWMol(Chem.CombineMols(base, branch))
                    joined.AddBond(site, base.GetNumAtoms() + 1, Chem.BondType.SINGLE)
                    joined.RemoveAtom(base.GetNumAtoms())
                    result = joined.GetMol()
                    Chem.SanitizeMol(result)
                    result.GetAtomWithIdx(site).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW if tag == '@' else Chem.ChiralType.CHI_TETRAHEDRAL_CW)
                    Chem.AssignStereochemistry(result, cleanIt=True, force=True)
                    variants = (('native', result),)
                    isotope = Chem.Mol(result)
                    isotope.GetAtomWithIdx(site).SetIsotope(13)
                    Chem.AssignStereochemistry(isotope, cleanIt=True, force=True)
                    variants += (('carbon13', isotope),)
                    for annotation, graph in variants:
                        smi = Chem.MolToSmiles(graph, isomericSmiles=True)
                        if smi in seen:
                            continue
                        seen.add(smi)
                        row = dict(id=f'{label}:{family}:{tag}:{geometry}:{annotation}', smiles=smi, expected=None)
                        if group is not None:
                            row['principal'] = family
                        rows.append(row)
    return rows

def _gate_sm5_native_stereo():
    global controls
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--diagnose', action='store_true')
    p.add_argument('--smoke', action='store_true', help='One simultaneous native R/S,E/Z composition per seed.')
    a = p.parse_args()
    recipe = _gap_sm5_native_stereo_controls
    controls = (lambda: [row for row in recipe() if ':hydride:@:E:native' in row['id']]) if a.smoke else recipe
    raise SystemExit(run(a.source_root, a.output, a.diagnose))


# ---- audit_fast_parent_proofs.py ----
"""Development-only chemical counterexamples for bounded fast parent admission."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
_fast_parent_proofs_CONTROLS = (('ring-chain-tie', 'CC(=O)c1ccc2[nH]c(=O)[nH]c2c1', True, 'One principal site on the ring and one on a carbon chain; the ring wins the tie.'), ('two-chain-sites', 'CC(=O)CC(=O)CC1NC(=O)Nc2ccccc21', False, 'A chain can cite two principal sites, so a one-site ring cannot discharge it.'), ('heterochain-tie', 'CC(=O)COCC1NC(=O)Nc2ccccc21', False, 'A tied heteroatomic chain remains a comparison barrier.'), ('multiple-functional-rings', 'O=C1NC(=O)N(Cc2ccc3[nH]c(=O)[nH]c3c2)C1', False, 'Separate functional rings require assembly/multiplicative method evidence.'), ('senior-acid', 'O=C(O)CC1NC(=O)Nc2ccccc21', False, 'A senior acid cannot be hidden by a carbonyl subset construction.'), ('compound-sulfanyl-enclosure', 'CC(=O)c1c(SC)cc2[nH]c(=O)[nH]c2c1', False, 'P-16.5 compound-prefix enclosure evidence is required independently of parent dominance.'))

def _fast_parent_proofs_main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('--source-root', type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.source_root.resolve()))
    from rdkit import Chem
    from nispo.extended_inventory import selection_inventory
    from nispo.fast_parent_construction import _proved_carbonyl_subset_name
    rows = []
    for key, smiles, admit, reason in _fast_parent_proofs_CONTROLS:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        value = _proved_carbonyl_subset_name(mol, Chem, selection_inventory(mol))
        row = dict(case_id=key, smiles=smiles, name=value, admission_expected=admit, admission_matches=bool(value) == admit, reason=reason)
        if value:
            run = subprocess.run([os.environ.get('JAVA', 'java'), '-jar', os.environ['NISPO_OPSIN_JAR'], '-r', '-o', 'smi', '-n'], input=value + '\n', text=True, capture_output=True, check=True, timeout=30)
            parsed, separator, echo = run.stdout.strip().partition('\t')
            actual = Chem.MolFromSmiles(parsed) if parsed else None
            row['raw_graph_exact'] = bool(separator and echo == value and (actual is not None) and (Chem.MolToSmiles(mol) == Chem.MolToSmiles(actual)))
        rows.append(row)
    result = dict(scope='Independent structural controls chosen on 2026-11-17 from P-41/P-44/P-51 parent-selection principles; no benchmark panel or source names used. No complete expected names.', cases=len(rows), failures=[r['case_id'] for r in rows if not r['admission_matches'] or r.get('raw_graph_exact') is False], rows=rows)
    with args.output.open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')
    return int(bool(result['failures']))

def _gate_fast_parent_proofs():
    raise SystemExit(_fast_parent_proofs_main())


_GATE_FUNCTIONS = {'sm1_principal_composition': _gate_sm1_principal_composition, 'sm1_carbamoyl': _gate_sm1_carbamoyl, 'sm2_parent_composition': _gate_sm2_parent_composition, 'sm3_heteromonocycles': _gate_sm3_heteromonocycles, 'sm4_fused_composition': _gate_sm4_fused_composition, 'sm5_native_stereo': _gate_sm5_native_stereo, 'fast_parent_proofs': _gate_fast_parent_proofs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = ['--source-root', str(args.source_root.resolve())]
    commands = [
        ('fast_parent_proofs', [str(output/'fast-parent-proofs.json')]),
    ]
    for name in ('sm1_principal_composition', 'sm1_carbamoyl',
                 'sm2_parent_composition', 'sm3_heteromonocycles',
                 'sm4_fused_composition', 'sm5_native_stereo'):
        commands.append((name, ['--output', str(output/(name+'.json'))]))
    results = {}
    for name, options in commands:
        command = [sys.executable, str(Path(__file__).resolve()), '--gate', name, *options, *source]
        with (output/(name+'.log')).open('w') as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        results[name] = {'returncode': completed.returncode, 'command': command}
        print(name, completed.returncode, flush=True)
    passed = set(results) == set(_GATE_FUNCTIONS) and all(row['returncode'] == 0 for row in results.values())
    (output/'release-acceptance.json').write_text(json.dumps(
        {'passed': passed, 'checks': results, 'full_PIN_certification': False}, indent=2)+'\n')
    return 0 if passed else 1


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == '--gate':
        gate = sys.argv[2]
        if gate not in _GATE_FUNCTIONS:
            raise SystemExit('Unknown acceptance gate: '+gate)
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        raise SystemExit(_GATE_FUNCTIONS[gate]())
    raise SystemExit(main())
