"""Guard input stereo semantics that ordinary molecular names cannot retain.

RDKit does not translate ChemAxon's molecule-level ``r`` marker into a
StereoGroup, but preserves the source extension in ``_CXSMILES_Data`` after
parsing. A plain isomeric-SMILES serialization discards that property; once
discarded the originating relative-stereo intent cannot be inferred here.

The ordinary molecular naming API also must not turn native RDKit OR/AND
stereogroups into a name that asserts just the drawn absolute member. Explicit
ensemble/subgroup APIs handle declared relationships within their own scopes.
"""
from __future__ import annotations

import re


_CX_ATOM_LABELS = re.compile(r"\$[^$]*\$")
_CX_RELATIVE_FLAG = re.compile(r"(?:^|[|,])r(?=[:,|]|$)")


def _source_has_relative_flag(mol) -> bool:
    if not mol.HasProp("_CXSMILES_Data"):
        return False
    source = mol.GetProp("_CXSMILES_Data")
    # Commas in $...$ atom labels are data, not top-level CX field separators.
    source = _CX_ATOM_LABELS.sub("", source)
    return bool(_CX_RELATIVE_FLAG.search(source))


def unsupported_relative_stereochemistry(mol, Chem) -> bool:
    """Whether an ordinary name would assert unsupported source stereo.

    ``r`` without an explicitly encoded chiral unit does not itself assert a
    configuration. An ordinary E/Z alkene also is not an absolute chiral
    center and may still be named. Atom tags and atropisomeric bond stereo are
    guarded conservatively whenever the ``r`` marker is present.
    """
    absolute = Chem.StereoGroupType.STEREO_ABSOLUTE
    if any(group.GetGroupType() != absolute for group in mol.GetStereoGroups()):
        return True
    if not _source_has_relative_flag(mol):
        return False
    if any(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for atom in mol.GetAtoms()):
        return True
    return any(str(bond.GetStereo()).startswith("STEREOATROP") for bond in mol.GetBonds())


def unsupported_relative_stereochemistry_result():
    from .naming_result import NamingResult, SelectionLimitation

    return NamingResult(
        "", "unavailable", "unresolved",
        (SelectionLimitation(
            "input-stereochemistry", "P-91/P-93",
            "Input carries relative, mixture, or non-absolute CX stereochemistry; "
            "the ordinary molecular writer cannot safely assert one absolute configuration. "
            "Use an applicable explicitly declared stereo ensemble or subgroup operation."
        ),),
        "unsupported-relative-stereochemistry",
    )
