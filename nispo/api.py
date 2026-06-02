from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache

try:
    from rdkit import Chem, RDLogger
except Exception:  # pragma: no cover - benchmark environment provides RDKit.
    Chem = None
else:
    RDLogger.DisableLog("rdApp.*")

from .parent_rules import parent_rules
from .rule_engine import NamingEngine
from .telemetry import CandidateTelemetry, telemetry_to_dict


def smiles_to_iupac(smiles: str) -> str:
    """Convert one SMILES string to an OPSIN-roundtripping IUPAC-style name."""
    if Chem is None:
        return ""
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return ""
    if mol is None or mol.GetNumAtoms() == 0:
        return ""
    try:
        return _name_molecule(mol)
    except Exception:
        return ""


def smiles_to_iupac_batch(smiles_list: Iterable[str]) -> list[str]:
    """Convert a batch of SMILES strings to IUPAC-style names.

    One output is returned for each input. Molecules outside the current rule
    coverage return an empty string so the benchmark can continue.
    """
    if Chem is None:
        return ["" for _ in smiles_list]
    engine = _engine()
    names: list[str] = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or mol.GetNumAtoms() == 0:
                names.append("")
                continue
            names.append(engine.name(mol))
        except Exception:
            names.append("")
    return names


def _smiles_to_iupac_batch_with_telemetry(smiles_list: Iterable[str]) -> tuple[list[str], dict]:
    """Convert a batch and return serialized internal naming telemetry.

    This helper is intentionally private: benchmark tooling can use it for
    measurement while the public batch API remains unchanged.
    """

    smiles_values = list(smiles_list)
    telemetry = CandidateTelemetry()
    if Chem is None:
        return ["" for _ in smiles_values], telemetry_to_dict(telemetry)
    engine = _engine()
    names: list[str] = []
    for smiles in smiles_values:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or mol.GetNumAtoms() == 0:
                names.append("")
                continue
            names.append(engine.name(mol, telemetry=telemetry))
        except Exception:
            names.append("")
    return names, telemetry_to_dict(telemetry)


def _name_molecule(mol) -> str:
    return _engine().name(mol)


@lru_cache(maxsize=1)
def _engine() -> NamingEngine:
    return NamingEngine(Chem, parent_rules(Chem))
