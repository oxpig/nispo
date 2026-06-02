from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_BENCHMARK = PROJECT_ROOT / "benchmark" / "run_benchmark.py"


def _load_run_benchmark_module():
    spec = importlib.util.spec_from_file_location("nispo_run_benchmark_for_test", RUN_BENCHMARK)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_nispo_chunk_returns_benchmark_visible_telemetry():
    run_benchmark = _load_run_benchmark_module()
    records = [
        ("CC", "CC", "ok"),
        ("c1ccccc1", "c1ccccc1", "ok"),
        ("not-a-smiles", "", "invalid"),
    ]

    result = run_benchmark.run_nispo_chunk((1, records))

    assert len(result["rows"]) == len(records)
    assert "nispo_telemetry" in result
    telemetry = result["nispo_telemetry"]
    assert "rule_seconds" in telemetry
    assert "candidate_count" in telemetry
    assert "template_candidate_count" in telemetry
    assert "substructure_match_calls" in telemetry
    assert "substructure_match_calls_by_rule" in telemetry


def test_run_nispo_chunk_telemetry_import_fallback_does_not_mark_rows_failed(monkeypatch):
    run_benchmark = _load_run_benchmark_module()
    real_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "nispo.api" and "_smiles_to_iupac_batch_with_telemetry" in fromlist:
            raise RuntimeError("blocked telemetry helper")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    result = run_benchmark.run_nispo_chunk((1, [("CC", "CC", "ok")]))

    assert len(result["rows"]) == 1
    assert result["rows"][0][3]
    assert result["rows"][0][4] == ""
    assert "could not import NISPO telemetry helper" in result["error"]
