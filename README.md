# NISPO: Nomenclature Inference for Structures, Powered by OPSIN

NISPO is a Python tool for generating IUPAC-style names from molecular SMILES strings.

The goal of NISPO is to produce names that successfully round-trip through OPSIN:

```text
SMILES -> NISPO name -> OPSIN -> SMILES
```

In other words, NISPO generates a name, OPSIN parses that name back into a structure, and RDKit is used to check whether the original and parsed structures are equivalent.

## Status

This repository snapshot is **NISPO version 0.1**.

Version 0.1 was developed using the MolJSON benchmark set of molecules and
achieved approximately **99% OPSIN round-trip accuracy** on that subset.

## Usage

```python
from nispo import smiles_to_iupac, smiles_to_iupac_batch

print(smiles_to_iupac("CCO"))
```

## AI Disclosure

This repository was developed using OpenAI Codex. The codebase is AI-generated and was created through an autonomous self-iteration process: Codex repeatedly ran the benchmark, inspected failures, updated the implementation, and reran the benchmark until high OPSIN round-trip accuracy was achieved.

OPSIN was used as the validation oracle during development. NISPO does not depend on OPSIN at runtime.
