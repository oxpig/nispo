# NISPO: Nomenclature Inference for Structures, Powered by OPSIN

NISPO is a Python package for generating OPSIN-roundtripping names from
molecular SMILES strings.

NISPO is designed around this round trip:

```text
SMILES -> NISPO name -> OPSIN -> SMILES
```

NISPO-generated names are not IUPAC systematic names as described in the
IUPAC Blue Book. They are IUPAC-style names that may include retained terms
supported by OPSIN. Do not use NISPO where reliable systematic IUPAC
nomenclature is required. This open-source project is provided without
warranty or liability.

## Quick Start

Install NISPO from PyPI:

```bash
pip install nispo==0.1.5
```

Convert a SMILES string:

```python
from nispo import smiles_to_iupac, smiles_to_iupac_batch

print(smiles_to_iupac("CCO"))
# ethanol

print(smiles_to_iupac_batch(["CCO", "not-a-smiles"]))
# ["ethanol", ""]
```

Unsupported molecules, invalid inputs, or internal naming failures return an
empty string (`""`). Batch calls always return one output for each input.

## Installation From GitHub

Install the 0.1.5 release directly from GitHub:

```bash
pip install "nispo @ git+https://github.com/oxpig/nispo.git@0.1.5"
```

For private-repository or SSH-authenticated installs, use:

```bash
pip install "nispo @ git+ssh://git@github.com/oxpig/nispo.git@0.1.5"
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
available. OPSIN was used to develop and evaluate NISPO, but OPSIN is not
required at runtime.

## Benchmark Status

On our PubChem benchmark, NISPO reached 98.1% OPSIN round-trip accuracy. Full
benchmark details will be provided with the paper.

## Citation

If you use MolJSON in your work, please cite:

```bibtex
@article{runcie2026moljson,
  title = {Molecular Representations for Large Language Models},
  author = {Runcie, Nicholas T. and Imrie, Fergus and Deane, Charlotte M.},
  year = {2026},
  journal = {arXiv preprint arXiv:2605.01822},
  doi = {10.48550/arXiv.2605.01822},
  url = {http://arxiv.org/abs/2605.01822},
}
```

## AI Disclosure

This repository was developed using OpenAI Codex. The codebase is AI-generated and was created through an autonomous self-iteration process: Codex repeatedly ran the benchmark, inspected failures, updated the implementation, and reran the benchmark until high OPSIN round-trip accuracy was achieved.

OPSIN was used as the validation oracle during development. NISPO does not depend on OPSIN at runtime.
