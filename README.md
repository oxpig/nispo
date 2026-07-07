# NISPO: Nomenclature Inference for Structures, Powered by OPSIN

NISPO is a Python tool for generating IUPAC-style names from molecular SMILES strings.

The goal of NISPO is to produce names that successfully round-trip through OPSIN:

```text
SMILES -> NISPO name -> OPSIN -> SMILES
```

In other words, NISPO generates a name, OPSIN parses that name back into a structure, and RDKit is used to check whether the original and parsed structures are equivalent.

## Status

This repository snapshot is **NISPO version 0.1.5**.

Version 0.1 was developed using the MolJSON benchmark set of molecules and
achieved approximately **99% OPSIN round-trip accuracy** on that subset.

Version 0.1.3 was assessed on a SureChEMBL subset filtered to molecules with
50 or fewer heavy atoms, a single fragment, and no unusual elements. On
2,682,641 molecules, NISPO achieved **95.26% OPSIN round-trip accuracy**
(2,555,499 / 2,682,641), with 88,413 mismatches, 21,277 empty names, and
17,452 OPSIN parse failures. The run processed 616.6 molecules/s with 8
workers.

Against the best `!`-separated SureChEMBL name variant for each molecule,
the generated NISPO names had mean raw edit distance 79.5 (median 76), mean
simplified alphanumeric casefold edit distance 55.9 (median 53), simplified
exact string match rate 0.39%, and mean simplified similarity 0.351.

Round-trip accuracy by broad structural group was 96.93% for acyclic
molecules, 95.17% for ring-containing molecules, 93.98% for fused systems,
88.30% for spiro systems, 77.39% for bridged systems, and 88.52% for
stereo-labelled molecules.

## Installation

Install the 0.1.5 release directly from GitHub:

```bash
pip install "nispo @ git+https://github.com/oxpig/nispo.git@0.1.5"
```

Or install from a local clone:

```bash
git clone https://github.com/oxpig/nispo.git
cd nispo
git checkout 0.1.5
pip install .
```

NISPO requires Python 3.10 or newer. RDKit is declared as a package dependency
and will be installed automatically by pip where a compatible wheel is
available.

After the 0.1.5 artifacts are uploaded to PyPI, the release can also be
installed with:

```bash
pip install nispo==0.1.5
```

## Usage

```python
from nispo import smiles_to_iupac, smiles_to_iupac_batch

print(smiles_to_iupac("CCO"))
```

## AI Disclosure

This repository was developed using OpenAI Codex. The codebase is AI-generated and was created through an autonomous self-iteration process: Codex repeatedly ran the benchmark, inspected failures, updated the implementation, and reran the benchmark until high OPSIN round-trip accuracy was achieved.

OPSIN was used as the validation oracle during development. NISPO does not depend on OPSIN at runtime.
