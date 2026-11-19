"""Shared public-contract adapters; default runs use the bounded core suite."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


def pytest_addoption(parser):
    group = parser.getgroup('nispo')
    group.addoption('--nispo-mode', choices=('public', 'fast', 'full'), default='full',
                    help='API under test; public requires the separately installed 0.1.7 release.')
    group.addoption('--require-opsin', action='store_true', default=False,
                    help='Fail graph verification setup if OPSIN/Java is unavailable.')


def pytest_collection_modifyitems(config, items):
    mode = config.getoption('--nispo-mode')
    selected, deselected = [], []
    for item in items:
        tier = ('core' if 'core' in item.path.parts else
                'integration' if 'integration' in item.path.parts else 'corpus')
        item.add_marker(getattr(pytest.mark, tier))
        scopes = [('full_only', mode=='full'), ('fast_only', mode=='fast'),
                  ('public_only', mode=='public'), ('current_only', mode!='public')]
        applicable = all(not item.get_closest_marker(marker) or allowed
                         for marker, allowed in scopes)
        for modes in item.iter_markers('modes'):
            allowed = set(modes.args)
            if not allowed or not allowed <= {'full', 'fast', 'public'}:
                raise pytest.UsageError('Invalid modes marker: '+item.nodeid)
            applicable = applicable and mode in allowed
        (selected if applicable else deselected).append(item)
    items[:] = selected
    if deselected:
        config.hook.pytest_deselected(items=deselected)


def pytest_report_header(config):
    mode = config.getoption('--nispo-mode')
    return f'NISPO contracts: {mode} mode; shared and applicable mode/cohort checks'


class NamingAdapter:
    def __init__(self, mode):
        self.mode = mode
        self.supports_results = mode != 'public'
        self.supports_molecules = mode != 'fast'

    def name(self, smiles):
        if self.mode == 'fast':
            from nispo.fast_selection import smiles_to_iupac_fast
            return smiles_to_iupac_fast(smiles)
        from nispo import smiles_to_iupac
        return smiles_to_iupac(smiles, timeout_seconds=None) if self.mode=='full' else smiles_to_iupac(smiles)

    def names(self, values):
        if self.mode == 'fast':
            return [self.name(value) for value in values]
        from nispo import smiles_to_iupac_batch
        return (smiles_to_iupac_batch(values, timeout_seconds=None) if self.mode=='full'
                else smiles_to_iupac_batch(values))

    def result(self, smiles):
        if not self.supports_results:
            pytest.skip('Public 0.1.7 has no structured result API')
        if self.mode == 'fast':
            from nispo.fast_selection import smiles_to_iupac_fast_result
            return smiles_to_iupac_fast_result(smiles)
        from nispo import smiles_to_iupac_result
        return smiles_to_iupac_result(smiles, timeout_seconds=None)

    def results(self, values):
        if self.mode == 'fast':
            return [self.result(value) for value in values]
        if not self.supports_results:
            pytest.skip('Public 0.1.7 has no structured batch result API')
        from nispo import smiles_to_iupac_batch_results
        return smiles_to_iupac_batch_results(values, timeout_seconds=None)

    def molecule_name(self, mol):
        if not self.supports_molecules:
            pytest.skip('Fast mode exposes a SMILES entrypoint, not a molecule entrypoint')
        from nispo import mol_to_iupac
        return mol_to_iupac(mol, timeout_seconds=None) if self.mode=='full' else mol_to_iupac(mol)

    def molecule_result(self, mol):
        if self.mode != 'full':
            pytest.skip('Structured molecule naming is a comprehensive API contract')
        from nispo import mol_to_iupac_result
        return mol_to_iupac_result(mol, timeout_seconds=None)


@pytest.fixture(scope='session')
def naming(request):
    return NamingAdapter(request.config.getoption('--nispo-mode'))


def _parser_paths():
    workspace = Path(__file__).resolve().parents[2]
    supplied = os.environ.get('NISPO_OPSIN_JAR') or os.environ.get('OPSIN_JAR')
    candidates = [Path(supplied)] if supplied else [
        workspace/'benchmark/opsin-cli-2.9.0-jar-with-dependencies.jar',
        workspace/'evaluation/test_coverage_20261107/benchmark/opsin-cli-2.9.0-jar-with-dependencies.jar']
    jar = next((p for p in candidates if p.is_file()), None)
    bundled_java = Path('/Library/Internet Plug-Ins/JavaAppletPlugin.plugin/Contents/Home/bin/java')
    java = os.environ.get('JAVA') or (str(bundled_java) if bundled_java.is_file() else shutil.which('java'))
    return java, jar


@pytest.fixture(scope='session')
def assert_graphs(request):
    """Parse each distinct delivered name once, in batches, without name snapshots."""
    from rdkit import Chem
    java, jar = _parser_paths()
    if not java or not jar:
        message = 'OPSIN JAR and Java are required for graph-equivalence contracts'
        if request.config.getoption('--require-opsin'):
            pytest.fail(message)
        pytest.skip(message)
    parsed = {}
    raw = {}
    supplements = {}

    def canonical(value):
        mol = Chem.Mol(value) if isinstance(value, Chem.Mol) else Chem.MolFromSmiles(value)
        assert mol is not None
        # Atom-map numbers identify caller atoms; nomenclature does not encode them.
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        return Chem.MolToSmiles(mol, isomericSmiles=True)

    def parse_names(names, *, raw_only=False):
        names = list(names)
        missing = list(dict.fromkeys(name for name in names if name and name not in parsed))
        if missing:
            completed = subprocess.run([java, '-jar', str(jar), '-r', '-o', 'smi', '-n'],
                                       input='\n'.join(missing)+'\n', capture_output=True,
                                       text=True, check=True, timeout=120)
            lines = completed.stdout.splitlines()
            assert len(lines)==len(missing), 'OPSIN returned a different number of records'
            for name, line in zip(missing, lines, strict=True):
                value, separator, echoed = line.partition('\t')
                assert separator and echoed==name
                raw[name] = canonical(value) if value else None
                parsed[name] = raw[name]
            # Independent name-only proofs, never given the original molecule.
            # The CIP supplement preserves the existing regression oracle.
            from _name_oracles.numbered_stereo import verify_names
            from _name_oracles.ol_hydrogen import recover_ol_hydrogen_isotope
            from _name_oracles.acid_hydrogen import recover_acid_hydrogen_isotope
            from _name_oracles.mixed_esters import recover_mixed_diyl_ester
            from _name_oracles.peroxolate_salts import recover_alkali_peroxolate_salt
            from _name_oracles.functional_operations import (
                recover_amine_dianion, recover_symmetric_dianhydride)
            from _name_oracles.phospho_hydrogen import recover_neutral_diphosphooxy
            for name, proof in verify_names(missing, include_uppercase=True).items():
                if proof.get('unique'):
                    parsed[name] = canonical(proof['graph'])
                    supplements[name] = proof['mode']
            for name in missing:
                phospho = recover_neutral_diphosphooxy(
                    name, lambda part: parse_names([part], raw_only=True)[0])
                if phospho:
                    parsed[name] = canonical(phospho)
                    supplements[name] = 'unique-neutral-diphosphooxy-prefix'
                # OPSIN accepts this narrow grammar but can lose its explicit
                # anion meaning. A unique name-only proof may correct that
                # graph; the untouched raw result remains separately visible.
                graph = recover_alkali_peroxolate_salt(
                    name, lambda part: parse_names([part], raw_only=True)[0])
                if graph:
                    parsed[name] = canonical(graph)
                    supplements[name] = 'unique-binary-alkali-peroxolate'
                if parsed[name] is None:
                    graph = recover_ol_hydrogen_isotope(name, lambda base: parse_names([base])[0])
                    if graph:
                        parsed[name] = canonical(graph)
                        supplements[name] = 'unique-ol-suffix-hydrogen-isotope'
                if parsed[name] is None:
                    graph = recover_acid_hydrogen_isotope(
                        name, lambda base: parse_names([base], raw_only=True)[0])
                    if graph:
                        parsed[name] = canonical(graph)
                        supplements[name] = 'unique-acid-suffix-hydrogen-isotope'
                if parsed[name] is None:
                    graph = recover_mixed_diyl_ester(
                        name, lambda component: parse_names([component], raw_only=True)[0])
                    if graph:
                        parsed[name] = canonical(graph)
                        supplements[name] = 'unique-mixed-diyl-ester'
                for decoder, label in (
                        (recover_amine_dianion, 'unique-primary-amine-dianion'),
                        (recover_symmetric_dianhydride, 'unique-symmetric-dianhydride')):
                    if parsed[name] is None:
                        graph = decoder(name, lambda part: parse_names([part], raw_only=True)[0])
                        if graph:
                            parsed[name] = canonical(graph)
                            supplements[name] = label
        recovered = raw if raw_only else parsed
        return [recovered.get(name) or '' for name in names]

    def check(inputs, names, *, raw_only=False):
        inputs, names = list(inputs), list(names)
        assert len(inputs)==len(names)
        assert all(isinstance(name, str) and name for name in names), 'A required graph was not named'
        recovered = parse_names(names, raw_only=raw_only)
        unparsed = [(index, name) for index, (name, graph) in enumerate(zip(names, recovered)) if not graph]
        assert not unparsed, f'Name oracle could not recover structures (index, name): {unparsed}'
        failures = [(index, canonical(source), graph)
                    for index, (source, graph) in enumerate(zip(inputs, recovered, strict=True))
                    if canonical(source)!=graph]
        assert not failures, f'Graph mismatches (index, expected, recovered): {failures}'
    check.parse_names = parse_names
    with pytest.MonkeyPatch.context() as validator_environment:
        validator_environment.setenv('JAVA', str(java))
        validator_environment.setenv('NISPO_OPSIN_JAR', str(jar))
        try:
            yield check
        finally:
            output = os.environ.get('NISPO_MEASUREMENT_OUTPUT')
            if output:
                report = {'raw_names_parsed': len(raw),
                          'raw_rejected_names': [name for name, graph in raw.items() if graph is None],
                          'name_only_supplements': supplements,
                          'raw_graph_overrides': {
                              name: {'raw_graph': raw[name], 'name_proof_graph': parsed[name],
                                     'method': method}
                              for name, method in supplements.items()
                              if raw.get(name) and raw[name] != parsed.get(name)},
                          'unrecovered_names': [name for name, graph in parsed.items() if graph is None],
                          'policy': 'No original input molecule is supplied to a supplement.'}
                (Path(output)/f'graph_oracle_{os.getpid()}.json').write_text(json.dumps(report, indent=2)+'\n')
