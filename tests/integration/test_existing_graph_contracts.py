"""Run the unchanged scalar graph contracts without their legacy harness.

Selection is structural, never based on a molecule, name, naming outcome or
runtime. Every function in the original ``test_api_contract.py`` with exactly
one public SMILES naming call, one parser call and a graph-equivalence
assertion is re-exported with its original parametrization. Its statements
and case records remain verbatim. Contracts about private selectors, ranking
or generated-name spelling do not match the syntax accepted here.

This opt-in integration tier preserves historical regression provenance; it
does not turn existing corpus records into new development fixtures. The
module records the original source hash and line numbers. No original file is
imported, so its private-engine imports and older parser startup machinery do
not run during collection. Naming is lazy and batched in groups of at most
four; duplicate inputs and delivered-name parsing are reused within this
module. Original test functions still assert each individual graph result.
"""
from __future__ import annotations

import __future__
import ast
import hashlib
from pathlib import Path

import pytest


pytestmark = pytest.mark.integration

_SOURCE = Path(__file__).resolve().parents[1]/'test_api_contract.py'
_SOURCE_BYTES = _SOURCE.read_bytes()
_TREE = ast.parse(_SOURCE_BYTES.decode('utf-8'), filename=str(_SOURCE))
_BATCH_SIZE = 4

_GRAPH_BODY = ast.parse(
    'generated_name = smiles_to_iupac(input_smiles)\n'
    'assert generated_name\n'
    'parsed_smiles = _opsin_parse_name(generated_name)\n'
    'assert _canonical_smiles(parsed_smiles) == _canonical_smiles(input_smiles)\n'
).body


def _literal_expression(node):
    """Evaluate only literal case expressions already present in the source."""
    permitted = (ast.Constant, ast.List, ast.Tuple, ast.BinOp, ast.Add, ast.Mult,
                 ast.UnaryOp, ast.UAdd, ast.USub, ast.Load)
    if not all(isinstance(part, permitted) for part in ast.walk(node)):
        raise ValueError('Graph-contract parametrization is not a literal case expression')
    return eval(compile(ast.Expression(node), str(_SOURCE), 'eval'), {'__builtins__': {}})


def _same_statements(left, right):
    return ([ast.dump(node, include_attributes=False) for node in left]
            == [ast.dump(node, include_attributes=False) for node in right])


def _original_cases(node):
    """Accept just the single-call graph contract and literal parametrization."""
    if not isinstance(node, ast.FunctionDef) or not node.name.startswith('test_'):
        return None
    arguments = node.args
    if (arguments.posonlyargs or arguments.kwonlyargs or arguments.defaults
            or arguments.kw_defaults or arguments.vararg or arguments.kwarg):
        return None
    body = list(node.body)
    if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    names = [argument.arg for argument in arguments.args]
    if not names:
        if node.decorator_list or len(body)!=5:
            return None
        first = body.pop(0)
        if not (isinstance(first, ast.Assign) and len(first.targets)==1
                and isinstance(first.targets[0], ast.Name)
                and first.targets[0].id=='input_smiles'
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            return None
        cases = (first.value.value,)
    elif names==['input_smiles'] and len(node.decorator_list)==1:
        decorator = node.decorator_list[0]
        if not (isinstance(decorator, ast.Call) and len(decorator.args)==2
                and not decorator.keywords
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr=='parametrize'
                and isinstance(decorator.func.value, ast.Attribute)
                and decorator.func.value.attr=='mark'
                and isinstance(decorator.func.value.value, ast.Name)
                and decorator.func.value.value.id=='pytest'):
            return None
        if _literal_expression(decorator.args[0])!='input_smiles':
            return None
        cases = _literal_expression(decorator.args[1])
        if not isinstance(cases, (list, tuple)) or not all(isinstance(value, str) for value in cases):
            return None
        cases = tuple(cases)
    else:
        return None
    return cases if _same_statements(body, _GRAPH_BODY) else None


def _unbound(*_args, **_kwargs):
    raise RuntimeError('The integration callbacks require the autouse module fixture')


_SELECTED = []
_CASES = {}
for _node in _TREE.body:
    _records = _original_cases(_node)
    if _records is not None:
        _SELECTED.append(_node)
        _CASES[_node.name] = _records

if not _SELECTED:
    raise RuntimeError('No original public graph contracts matched the syntax selector')

# Preserve the original canonicalizer rather than changing the graph oracle.
_CANONICALIZER = next(node for node in _TREE.body
                      if isinstance(node, ast.FunctionDef) and node.name=='_canonical_smiles')
_NAMESPACE = {
    '__name__': __name__, '__builtins__': {}, 'pytest': pytest,
    'smiles_to_iupac': _unbound, '_opsin_parse_name': _unbound,
}
exec(compile(ast.Module(body=[_CANONICALIZER, *_SELECTED], type_ignores=[]),
             str(_SOURCE), 'exec', flags=__future__.annotations.compiler_flag,
             dont_inherit=True), _NAMESPACE)
globals().update({node.name: _NAMESPACE[node.name] for node in _SELECTED})

_CONTRACT_PROVENANCE = {
    'source': str(_SOURCE), 'sha256': hashlib.sha256(_SOURCE_BYTES).hexdigest(),
    'selection': 'exact public single-name/parser/canonical-graph-equality AST pattern',
    'functions': {node.name: {'line': node.lineno, 'records': len(_CASES[node.name])}
                  for node in _SELECTED},
    'input_selection': 'All original matching records, verbatim; no molecule or outcome filtering.',
}


@pytest.fixture(scope='module', autouse=True)
def _batched_original_callbacks(request, naming, assert_graphs):
    """Bind lightweight callbacks; name no inputs until a contract requests one."""
    from rdkit import Chem

    selected_inputs = []
    for item in request.session.items:
        if item.path.resolve()!=Path(__file__).resolve():
            continue
        name = item.originalname or item.name.split('[', 1)[0]
        if name not in _CASES:
            continue
        callspec = getattr(item, 'callspec', None)
        selected_inputs.append(callspec.params['input_smiles'] if callspec else _CASES[name][0])
    order = list(dict.fromkeys(selected_inputs))
    positions = {value: index for index, value in enumerate(order)}
    names_by_input = {}
    graphs_by_name = {}

    def name_one(value):
        if value not in names_by_input:
            index = positions.get(value)
            nearby = order[index:index+_BATCH_SIZE] if index is not None else [value]
            batch = [source for source in nearby if source not in names_by_input]
            names = list(naming.names(batch))
            assert len(names)==len(batch), 'The batch API did not preserve input count'
            assert all(isinstance(name, str) for name in names)
            names_by_input.update(zip(batch, names, strict=True))
            missing = list(dict.fromkeys(name for name in names if name and name not in graphs_by_name))
            if missing:
                parsed = list(assert_graphs.parse_names(missing))
                assert len(parsed)==len(missing), 'The parser did not preserve delivered-name count'
                graphs_by_name.update(zip(missing, parsed, strict=True))
        return names_by_input[value]

    def parse_one(name):
        if name not in graphs_by_name:
            graphs_by_name[name] = list(assert_graphs.parse_names([name]))[0]
        return graphs_by_name[name]

    _NAMESPACE.update(Chem=Chem, smiles_to_iupac=name_one, _opsin_parse_name=parse_one)
    yield
    _NAMESPACE.update(smiles_to_iupac=_unbound, _opsin_parse_name=_unbound)
