"""Public API isolation controls using independently constructed molecules."""
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import sys

from rdkit import Chem

import nispo
from nispo import mol_to_iupac, smiles_to_iupac, smiles_to_iupac_result
from nispo.fast_selection import smiles_to_iupac_fast, smiles_to_iupac_fast_result


def test_ordinary_fast_does_not_import_full_engine_in_clean_process():
    # The fast engine may import shared stereochemistry helpers.
    script = '''\nimport json, sys\nfrom nispo.fast_selection import smiles_to_iupac_fast, smiles_to_iupac_fast_result\nvalues = ("CCO", "C[C@H](O)F", "F/C=C/F", "C[C@H](O)F |&1:1|",\n          "N1CC2(CC2)c2cccnc21", "CC1=NNC(=O)OC1",\n          "CON(C)C(=O)[C@H]1C[C@@H](OC(F)(F)F)C1")\npairs = [(smiles_to_iupac_fast(v), smiles_to_iupac_fast_result(v).name) for v in values]\nprint(json.dumps([all(a == b for a, b in pairs), [name for name in ("nispo.api", "nispo.rule_engine",\n    "nispo.parent_rules") if name in sys.modules]]))\n'''
    output = subprocess.run([sys.executable, '-c', script], capture_output=True,
                            text=True, check=True, timeout=90)
    assert json.loads(output.stdout) == [True, []]


def test_lazy_public_exports_and_structured_result_contract():
    assert all(hasattr(nispo, name) for name in nispo.__all__)
    value = 'CCO'
    fast = smiles_to_iupac_fast_result(value)
    assert fast.name == smiles_to_iupac_fast(value)
    assert fast.generation_status == 'available'
    assert fast.preferred_status == 'unresolved' and fast.unresolved
    assert isinstance(fast, nispo.NamingResult)
    assert isinstance(smiles_to_iupac_result(value), nispo.NamingResult)


def test_fast_and_full_are_deterministic_when_interleaved_and_concurrent():
    values = ('CCO', 'c1ccccc1', 'C[C@H](O)F', 'F/C=C/F')
    expected = [(smiles_to_iupac_fast(s), smiles_to_iupac_result(s)) for s in values]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(lambda s: (smiles_to_iupac_fast(s), smiles_to_iupac_result(s)),
                               values * 2))
    assert actual == expected * 2
    for value in values:
        mol = Chem.MolFromSmiles(value)
        mol_to_iupac(mol)
        assert not hasattr(mol, '_nispo_python_cache')
        assert not hasattr(mol, '_nispo_fast017_python_cache')
        assert smiles_to_iupac(value) == smiles_to_iupac_result(value).name


def test_fast_uses_full_only_when_annotation_is_unsupported():
    for value in ('[13CH3]C', '[CH3]'):
        result = smiles_to_iupac_fast_result(value)
        assert result.name == smiles_to_iupac_fast(value) == smiles_to_iupac(value)
        assert result.generation_status == 'available'
        assert result.preferred_status == 'unresolved' and result.unresolved


def test_full_handoff_requires_fast_name_failure(monkeypatch):
    import nispo.api as full_api
    import nispo.fast_selection as fast_api

    class StubWriter:
        value = 'usable-general-name'

        def name(self, mol):
            assert mol.GetNumAtoms()
            return self.value

    writer = StubWriter()
    fallback_calls = []
    monkeypatch.setattr(fast_api._public, '_engine', lambda: writer)
    def full_name(value, *, fallback_to_fast):
        fallback_calls.append((value, fallback_to_fast))
        return 'full-name'

    monkeypatch.setattr(full_api, 'smiles_to_iupac', full_name)
    assert smiles_to_iupac_fast('CCO') == writer.value
    assert fallback_calls == []
    writer.value = ''
    assert smiles_to_iupac_fast('CCO') == 'full-name'
    assert fallback_calls == [('CCO', False)]
