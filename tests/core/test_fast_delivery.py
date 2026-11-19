"""Small public fast-delivery boundaries, independent of naming preferences.

Provenance: ethanol, its carbon-13 variant, a methyl radical and a single
fluorochloro stereocentre are independently authored elementary transport and
annotation controls. Byte/CX encodings probe information preservation rather
than a benchmark molecule or generated-name disagreement. No reference names
or private selector/producer identities are asserted.
"""
from __future__ import annotations

import os
import subprocess

import pytest


pytestmark = pytest.mark.fast_only


def _assert_fast_delivery_is_explicitly_provisional(result):
    assert result.name and result.generation_status == 'available'
    assert result.preferred_status == 'unresolved'
    assert any(issue.stage == 'explicit-fast-mode' for issue in result.unresolved)


@pytest.mark.parametrize('encoded', (b'CCO', b'OCC'), ids=('canonical-bytes', 'reordered-bytes'))
def test_ordinary_bytes_preserve_graph_and_traversal_independence(naming, assert_graphs, encoded):
    result = naming.result(encoded)
    _assert_fast_delivery_is_explicitly_provisional(result)
    assert result.name == naming.name('CCO')
    assert_graphs(['CCO'], [result.name])


def test_non_utf8_input_is_rejected_without_launching_a_worker(naming, monkeypatch):
    attempts = []

    def forbidden(*_args, **_kwargs):
        attempts.append(True)
        raise AssertionError('Invalid byte input must be rejected before external work')

    with monkeypatch.context() as patch:
        patch.setattr(subprocess, 'Popen', forbidden)
        result = naming.result(b'\xff')
    assert not attempts
    assert not result.name
    assert result.generation_status == 'invalid-input'
    assert result.preferred_status == 'not-assessed'


@pytest.mark.parametrize(
    'encoded',
    (b'C[C@H](F)Cl |r|', b'C[C@H](F)Cl |&1:1|'),
    ids=('relative-cx-bytes', 'and-group-cx-bytes'),
)
def test_nonabsolute_cx_bytes_are_rejected_by_both_delivery_interfaces(naming, monkeypatch, encoded):
    attempts = []

    def forbidden(*_args, **_kwargs):
        attempts.append(True)
        raise AssertionError('Nonabsolute input must not be promoted through a naming worker')

    with monkeypatch.context() as patch:
        patch.setattr(subprocess, 'Popen', forbidden)
        result = naming.result(encoded)
        assert naming.name(encoded) == ''
    assert not attempts
    assert not result.name and result.generation_status == 'unavailable'
    assert result.preferred_status == 'unresolved'
    assert any(issue.stage == 'input-stereochemistry' for issue in result.unresolved)


@pytest.mark.parametrize(
    'encoded, expected',
    ((b'[13CH3]CO', '[13CH3]CO'), (b'[CH3]', '[CH3]')),
    ids=('carbon-isotope-bytes', 'radical-bytes'),
)
def test_annotation_handoff_preserves_graph_and_fast_uncertainty(naming, assert_graphs, encoded, expected):
    result = naming.result(encoded)
    # The public handoff is bounded. A genuine timeout is a runtime outcome,
    # not permission to emit an unlabelled or electron-count-losing name.
    assert result.generation_status in {'available', 'timeout'}
    if result.generation_status == 'timeout':
        assert not result.name and result.preferred_status == 'not-assessed'
        assert any(issue.stage == 'runtime' for issue in result.unresolved)
    else:
        _assert_fast_delivery_is_explicitly_provisional(result)
        assert_graphs([expected], [result.name])


def test_supported_fast_delivery_does_not_launch_external_programs(naming, assert_graphs, monkeypatch):
    attempts = []

    def forbidden(*_args, **_kwargs):
        attempts.append(True)
        raise AssertionError('The ordinary fast route must not start an external program')

    with monkeypatch.context() as patch:
        patch.setattr(subprocess, 'Popen', forbidden)
        patch.setattr(os, 'system', forbidden)
        result = naming.result('CCO')
    # Check attempts separately: a caught exception must not conceal a launch.
    assert not attempts
    _assert_fast_delivery_is_explicitly_provisional(result)
    assert_graphs(['CCO'], [result.name])
