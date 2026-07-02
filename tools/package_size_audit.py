#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "nispo"
RESOURCE_ROOT = PACKAGE_ROOT / "resources"
DEV_ARTIFACT_MARKERS = (
    "/__pycache__/",
    "/.pytest_cache/",
    "/graphify-out/",
    "/benchmark/",
    "/runs/",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit installable NISPO package size and dev artifacts.")
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--build-wheel", action="store_true", help="Build a wheel with pip and inspect its contents.")
    parser.add_argument(
        "--build-isolation",
        action="store_true",
        help="Let pip create an isolated build env. Off by default so local audits do not need network.",
    )
    parser.add_argument("--wheel", type=Path, help="Inspect an existing wheel instead of building one.")
    parser.add_argument("--json", action="store_true", help="Emit only JSON.")
    args = parser.parse_args()

    repo = args.repo.resolve()
    wheel = args.wheel
    temp_dir = None
    if args.build_wheel:
        temp_dir = tempfile.TemporaryDirectory(prefix="nispo_wheel_audit_")
        temp_root = Path(temp_dir.name)
        build_source = temp_root / "src"
        wheel_dir = temp_root / "wheel"
        wheel_dir.mkdir()
        shutil.copytree(repo, build_source, ignore=_ignore_staging_artifacts)
        command = [sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "--no-cache-dir", "-w", str(wheel_dir)]
        if not args.build_isolation:
            command.append("--no-build-isolation")
        subprocess.run(
            command,
            cwd=build_source,
            check=True,
        )
        wheels = sorted(wheel_dir.glob("nispo-*.whl"))
        if not wheels:
            raise SystemExit("pip wheel did not produce a nispo wheel")
        wheel = wheels[-1]

    report = {
        "source": _source_report(repo),
        "wheel": _wheel_report(wheel) if wheel else None,
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_report(report)
    if temp_dir is not None:
        temp_dir.cleanup()
    return 0


def _source_report(repo: Path) -> dict[str, Any]:
    package_root = repo / "nispo"
    resource_root = package_root / "resources"
    all_files = [path for path in package_root.rglob("*") if path.is_file()]
    runtime_files = [
        path
        for path in all_files
        if "__pycache__" not in path.parts
        and ".pytest_cache" not in path.parts
        and "graphify-out" not in path.parts
    ]
    resources = [path for path in resource_root.glob("*") if path.is_file()]
    return {
        "package_root": str(package_root),
        "package_bytes_in_tree": _sum_size(all_files),
        "package_bytes_excluding_dev_artifacts": _sum_size(runtime_files),
        "resource_bytes": _sum_size(resources),
        "top_resources": _top_files(resources),
        "dev_artifact_bytes_in_tree": _sum_size(all_files) - _sum_size(runtime_files),
    }


def _ignore_staging_artifacts(_directory: str, names: list[str]) -> set[str]:
    ignored = {
        name
        for name in names
        if name in {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "build", "dist"}
        or name.endswith(".egg-info")
        or name == "graphify-out"
        or name == ".DS_Store"
        or name.endswith((".pyc", ".pyo"))
    }
    return ignored


def _wheel_report(wheel: Path | None) -> dict[str, Any] | None:
    if wheel is None:
        return None
    with zipfile.ZipFile(wheel) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
    package_infos = [info for info in infos if info.filename.startswith("nispo/")]
    resource_infos = [info for info in package_infos if info.filename.startswith("nispo/resources/")]
    dev_artifacts = [
        info.filename
        for info in package_infos
        if any(marker in f"/{info.filename}" for marker in DEV_ARTIFACT_MARKERS)
    ]
    return {
        "path": str(wheel),
        "wheel_file_bytes": wheel.stat().st_size,
        "installed_package_bytes": sum(info.file_size for info in package_infos),
        "installed_resource_bytes": sum(info.file_size for info in resource_infos),
        "file_count": len(package_infos),
        "resource_file_count": len(resource_infos),
        "dev_artifact_count": len(dev_artifacts),
        "dev_artifacts": dev_artifacts[:50],
        "top_resources": [
            {"path": info.filename, "bytes": info.file_size}
            for info in sorted(resource_infos, key=lambda item: item.file_size, reverse=True)[:15]
        ],
    }


def _sum_size(paths) -> int:
    return sum(path.stat().st_size for path in paths)


def _top_files(paths: list[Path], *, limit: int = 15) -> list[dict[str, Any]]:
    return [
        {"path": str(path), "bytes": path.stat().st_size}
        for path in sorted(paths, key=lambda item: item.stat().st_size, reverse=True)[:limit]
    ]


def _print_report(report: dict[str, Any]) -> None:
    source = report["source"]
    print(json.dumps(source, indent=2, sort_keys=True))
    if report["wheel"] is not None:
        print(json.dumps(report["wheel"], indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
