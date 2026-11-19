"""Typed oxygen-role analysis for acid components in polyol esters.

This module classifies isotope placement on the two oxygen atoms of each
carboxylate ester unit. Its per-unit profile classifier admits natural units
and units uniformly labelled at both roles with 17O or 18O. Partial-role and
mixed-role profiles remain outside the bounded multivalent naming path.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EsterOxygenProfile:
    """The original-graph identities and isotope roles for one ester unit."""

    carbonyl_oxygen: int
    bridge_oxygen: int
    carbonyl_isotope: int
    bridge_isotope: int

    @property
    def role_isotopes(self) -> tuple[int, int]:
        """Mass numbers in carbonyl-O, bridge-O role order (zero is natural)."""
        return self.carbonyl_isotope, self.bridge_isotope

    @property
    def suffix_oxygen_isotopes(self) -> tuple[int, ...]:
        """Modified O atoms represented in the acid-derived suffix count."""
        return tuple(mass for mass in self.role_isotopes if mass)

    @property
    def suffix_oxygen_counts(self) -> tuple[tuple[int, int], ...]:
        counts = Counter(self.suffix_oxygen_isotopes)
        return tuple(sorted(counts.items()))

    @property
    def component_class(self) -> str:
        carbonyl, bridge = self.role_isotopes
        if not carbonyl and not bridge:
            return "unmodified"
        if carbonyl and carbonyl == bridge:
            return "uniform_same_nuclide"
        if carbonyl and not bridge:
            return "carbonyl_only"
        if bridge and not carbonyl:
            return "bridge_only"
        if carbonyl == bridge:
            return "uniform_same_nuclide"
        return "mixed_role_masses"

    @property
    def uniform_mass(self) -> int | None:
        """Return the mass when both ester O sites are modified identically."""
        carbonyl, bridge = self.role_isotopes
        return carbonyl if carbonyl in {17, 18} and carbonyl == bridge else None

    @property
    def monoester_organyl_marker(self) -> str | None:
        """Return P-82.6.4's unambiguous single-role monoester marker.

        `None` means that a single role marker cannot describe this profile.
        A uniform two-O profile needs no role distinction at component scope;
        this property does not imply that the marker can be transferred to a
        multivalent polyol name.
        """
        carbonyl, bridge = self.role_isotopes
        if bridge and not carbonyl:
            return f"{bridge}O-"
        if carbonyl and not bridge:
            return "O-"
        if not carbonyl and not bridge:
            return ""
        return None


def ester_oxygen_profile(mol, carbonyl_oxygen: int, bridge_oxygen: int) -> EsterOxygenProfile | None:
    """Validate and classify one ester C=O / C-O-C pair by original indices.

    The two O atoms must share a carbonyl carbon, with the designated
    carbonyl O double-bonded and the bridge O single-bonded. The bridge O
    must also connect to one carbon atom on the hydroxylic side. Only the
    natural-abundance default, 17O, and 18O are admitted by this bounded role
    classifier.
    """
    if carbonyl_oxygen == bridge_oxygen:
        return None
    try:
        carbonyl = mol.GetAtomWithIdx(int(carbonyl_oxygen))
        bridge = mol.GetAtomWithIdx(int(bridge_oxygen))
    except (IndexError, TypeError, ValueError):
        return None
    if carbonyl.GetAtomicNum() != 8 or bridge.GetAtomicNum() != 8:
        return None
    if (carbonyl.GetFormalCharge() or bridge.GetFormalCharge()
            or carbonyl.GetNumRadicalElectrons() or bridge.GetNumRadicalElectrons()
            or int(carbonyl.GetChiralTag()) or int(bridge.GetChiralTag())):
        return None
    carbonyl_bonds = [bond for bond in carbonyl.GetBonds()
                      if bond.GetBondTypeAsDouble() == 2.0]
    if len(carbonyl_bonds) != 1:
        return None
    center = carbonyl_bonds[0].GetOtherAtomIdx(carbonyl_oxygen)
    if mol.GetAtomWithIdx(center).GetAtomicNum() != 6:
        return None
    carbonyl_link = mol.GetBondBetweenAtoms(center, carbonyl_oxygen)
    bridge_link = mol.GetBondBetweenAtoms(center, bridge_oxygen)
    if (carbonyl_link is None or carbonyl_link.GetBondTypeAsDouble() != 2.0
            or bridge_link is None or bridge_link.GetBondTypeAsDouble() != 1.0):
        return None
    carbonyl_oxygen_neighbors = [atom.GetIdx() for atom in mol.GetAtomWithIdx(center).GetNeighbors()
                                 if atom.GetAtomicNum() == 8]
    if set(carbonyl_oxygen_neighbors) != {carbonyl_oxygen, bridge_oxygen}:
        return None
    if bridge.GetDegree() != 2:
        return None
    other = [atom.GetIdx() for atom in bridge.GetNeighbors() if atom.GetIdx() != center]
    if len(other) != 1 or mol.GetAtomWithIdx(other[0]).GetAtomicNum() != 6:
        return None
    if any(neighbor.GetAtomicNum() == 8
           and mol.GetBondBetweenAtoms(other[0], neighbor.GetIdx()).GetBondTypeAsDouble() == 2.0
           for neighbor in mol.GetAtomWithIdx(other[0]).GetNeighbors()):
        return None
    masses = (int(carbonyl.GetIsotope()), int(bridge.GetIsotope()))
    if any(mass not in {0, 17, 18} for mass in masses):
        return None
    return EsterOxygenProfile(
        int(carbonyl_oxygen), int(bridge_oxygen), masses[0], masses[1],
    )


def supported_uniform_polyol_oxygen_profiles(profiles) -> bool:
    """Whether each unit is natural or uniformly labelled with 17O/18O."""
    profiles = tuple(profiles)
    return bool(profiles) and all(
        profile.role_isotopes in {(0, 0), (17, 17), (18, 18)}
        for profile in profiles
    )


def uniform_polyol_acid_oxygen_mass(profiles) -> int | None:
    """Return one mass only when every supplied acid ester unit has both O
    roles uniformly modified with the same 17O or 18O.

    This is a structural scope classifier grounded in P-82.6.1.3. The caller
    uses the returned mass to place an O2 descriptor on each acid-derived
    component without transferring monoester organyl role qualifiers to the
    shared multivalent organyl.
    """
    profiles = tuple(profiles)
    if not profiles:
        return None
    masses = {profile.uniform_mass for profile in profiles}
    return next(iter(masses)) if len(masses) == 1 and None not in masses else None
