"""Public graph and atom-order contracts for independent development graphs."""
import json
import os
from pathlib import Path
import random
import shutil
import subprocess

import pytest
from rdkit import Chem

from nispo import smiles_to_iupac_batch, smiles_to_iupac_result
from nispo.fast_selection import smiles_to_iupac_fast


def test_batch_and_fast_preserve_graph_and_atom_order():
    development = Path(__file__).parents[1] / 'development'
    ledgers = ('surechembl_batch_pin_20261030', 'surechembl_polycyclic_acid_20261030',
               'ring_polyacid_20261030', 'fused_nitrogen_prefix_20261030')
    cohorts = json.loads((development / 'batch_graph_controls.json').read_text())['cohorts']
    cases = [case for ledger in ledgers
             for case in cohorts[ledger]['cases']]
    originals, variants = [], []
    for case in cases:
        mol = Chem.MolFromSmiles(case['smiles'])
        assert mol is not None
        order = list(range(mol.GetNumAtoms()))
        random.Random(2026103012).shuffle(order)
        originals.append(case['smiles'])
        variants.append(Chem.MolToSmiles(Chem.RenumberAtoms(mol, order),
                                         canonical=False, isomericSmiles=True))
    inputs = originals + variants
    batch = smiles_to_iupac_batch(inputs)
    single = [smiles_to_iupac_result(value).name for value in inputs]
    fast = [smiles_to_iupac_fast(value) for value in inputs]
    assert len(batch) == len(single) == len(fast) == len(inputs)
    assert all(batch) and all(fast)
    assert batch == single
    assert batch[:len(cases)] == batch[len(cases):]
    assert fast[:len(cases)] == fast[len(cases):]
    jar = Path(os.environ.get('NISPO_OPSIN_JAR',
               Path(__file__).resolve().parents[2] / 'benchmark/opsin-cli-2.9.0-jar-with-dependencies.jar'))
    java = shutil.which('java')
    if not java or not jar.is_file():
        pytest.skip('OPSIN and Java are required for graph recovery')
    process = subprocess.run([java, '-jar', str(jar), '-r', '-o', 'smi', '-n'],
                             input='\n'.join(batch + fast) + '\n', text=True,
                             capture_output=True, check=True, timeout=120)
    lines = process.stdout.splitlines()
    assert len(lines) == len(batch) + len(fast)
    for source, name, line in zip(inputs + inputs, batch + fast, lines, strict=True):
        parsed, separator, echoed = line.partition('\t')
        actual = Chem.MolFromSmiles(parsed) if parsed else None
        expected = Chem.MolFromSmiles(source)
        assert separator and echoed == name and actual is not None and expected is not None
        assert Chem.MolToSmiles(actual, isomericSmiles=True) == Chem.MolToSmiles(expected, isomericSmiles=True)
