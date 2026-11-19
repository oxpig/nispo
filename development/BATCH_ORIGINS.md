The four frozen batch-control cohorts retain their original records in
`batch_graph_controls.json`, keyed by the original directory names. Original
file hashes are stored with the cohorts. The origin notes below are preserved
verbatim; their historical paths refer to the separate pre-cleanup archive.

# Original development/surechembl_batch_pin_20261030/README.md

# Batch PIN and fast-mode development

This development starts from `generalparentsfinal`, package-content manifest
SHA-256 `73c707b91ea6dd4d006dd644333fb6084ee8c8ccdeba68423bb127e66e79514b`.
The source is the 2025-08-06 release of
[SureChEMBL Zenodo 16755947](https://zenodo.org/records/16755947), archive
SHA-256 `1ded2042ca927ee3e2eb38c3fd4e2d78333b8a409811f7036eade776c7db5b50`.
The source graphs in [cases.json](cases.json) retain their independent sample
origin, selection seed and original SMILES. Some use the earlier 900-graph
sample (seed `2026103003`); others use the new source sample (seed
`2026103011`). The source names and all resulting measurements are development
evidence, not certified PIN labels or a blind evaluation.

The source-ranked acid group must be expressed by the suffix when a valid
source-mapped parent is available. A subordinate carbamate can be a mapped
prefix without blocking a ring acid, and a fused acid with an annular oxo
group must retain the fused parent when its H state and atom ownership are
proved. For a one-position retained acetic-acid parent, an independently
proved mapped ring prefix can use the locant-omission construction. These
local obligations follow [P-44.1.1](https://iupac.qmul.ac.uk/BlueBook/P4.html#P-44.1.1),
[P-14.3.4.6](https://iupac.qmul.ac.uk/BlueBook/P1.html#P-14.3.4.6),
[P-25.3](https://iupac.qmul.ac.uk/BlueBook/P2.html#P-25.3) and
[P-65.1](https://iupac.qmul.ac.uk/BlueBook/P6.html#P-65.1).
Complete whole-name preference remains unresolved where other parent or
naming-method alternatives have not been proved.

The [audit](audit.py) checks these source-local obligations only in standard
mode, with graph recovery and atom-order controls for both APIs. The public
pytest contract checks graph and determinism rather than nomenclature wording.
Fast names are allowed to differ from standard names and need not be PINs.


# Original development/surechembl_polycyclic_acid_20261030/README.md

# SureChEMBL bridged and spiro acid controls

These ten graphs are from the authorized SureChEMBL development source,
Zenodo record 16755947, release 2025-08-06. They were selected from the
independently SHA-sampled development set with seed 2026103011; the source
archive hash, source row identifiers, original and canonical structures,
source names, selection hashes and frozen baseline are recorded in
`cases.json`. Their selection used the SureChEMBL source and development
outputs, not Blue Book panel records.

The local obligation is to express the mapped principal carboxylic acid as a
suffix on a structurally proved bridged or spiro ring parent. Each example
has a source-selected acid group and a mapped, numbered parent; resolving
that obligation does not certify the whole name as a PIN. `audit.py` checks
the local standard-mode obligation and graph recovery and atom-order
stability in both modes. No preferred-name requirement is placed on fast
mode.


# Original development/ring_polyacid_20261030/README.md

# Ring dicarboxylic acid controls

The first graph is SureChEMBL development row 2325208, independently selected
from the authorized source release (Zenodo 16755947, 2025-08-06) using the
seed and graph provenance recorded in `cases.json`. The other eight graphs
were independently constructed from simple cyclic, bridged and spiro cores
with two carboxyl groups; their input graphs and origins are recorded in
`cases.json`. No panel rows were used to select these examples.

The standard-mode local obligation is to express both selected ring-attached
acid groups using a dicarboxylic acid suffix. This is consistent with the
carboxylic acid suffix conventions in [Blue Book P-65](https://iupac.qmul.ac.uk/BlueBook/P6.html#P-65).
It is not a certification of the complete name. The audit also checks both
public modes for graph recovery and atom-order stability; fast mode has no
preferred-wording obligation.


# Original development/fused_nitrogen_prefix_20261030/README.md

# Fused aromatic nitrogen prefixes on acid chains

The first graph is an independently constructed acetic acid derivative of
the capped fused nitrogen ring core found in SureChEMBL review row 3717683.
Four other graphs were constructed directly from common fused heterocycle
connectivities. Four more are actual rows from the independently sampled,
authorized SureChEMBL development set. They were not selected from the Blue
Book panel. The source release, review and sample hashes, selection seeds,
frozen baseline, and original and canonical graph strings are recorded in
`cases.json`. The original five-case ledger is preserved as `cases_initial.json`.

The local standard-mode obligations are to preserve the selected fused
parent and indicated hydrogen at the nitrogen attachment, and, where an
acetic-acid parent is present, to omit its unnecessary locant. These follow the fused-parent
rules in [Blue Book P-25](https://iupac.qmul.ac.uk/BlueBook/P2.html#P-25.3)
and the locant-omission rules in [P-14](https://iupac.qmul.ac.uk/BlueBook/P1.html#P-14.3).
They do not certify the complete name as a PIN. The audit checks the local
obligations and graph recovery and atom-order stability for both public modes;
fast mode has no preferred-wording obligation.
