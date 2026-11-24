"""NISPO's public API, loaded on first use.

Importing a fast-mode entry point must not eagerly import the full
naming engine. Public exports retain their defining module aliases.
"""
from importlib import import_module as _import_module

__version__ = "0.2.1"

__all__ = [
    "__version__", "mol_to_iupac", "smiles_to_iupac", "smiles_to_iupac_batch",
    "mol_to_iupac_result", "smiles_to_iupac_result", "smiles_to_iupac_batch_results",
    "NamingResult", "SelectionLimitation", "PreferredNameJustification", "SelectionStep", "IUPACBatchPool",
    "AdductComponent", "AdductNamingResult", "NamedAdductComponent",
    "adduct_to_iupac", "adduct_to_iupac_result",
    "StereoEnsemble", "StereoEnsembleNamingResult",
    "stereo_ensemble_to_iupac", "stereo_ensemble_to_iupac_result",
    "ElectronTransfer", "ElectronTransferNamingResult",
    "electron_transfer_to_iupac", "electron_transfer_to_iupac_result",
    "Nuclide", "IsotopeLabelling", "LabelledSite", "IsotopeLabellingNamingResult",
    "isotope_labelling_to_iupac", "isotope_labelling_to_iupac_result",
    "TorsionReference", "TorsionDescription", "RingConformation",
    "ConformerDescription", "describe_conformer",
    "StereoSubgroup", "StereoSubgroupNamingResult",
    "stereo_subgroup_to_iupac", "stereo_subgroup_to_iupac_result",
    "AxialUnit", "AxialNamingResult", "atropisomer_to_iupac", "atropisomer_to_iupac_result",
    "CumuleneNamingResult", "cumulene_to_iupac", "cumulene_to_iupac_result",
    "PeptideResidue", "PeptideSequence", "PeptideResidueBinding", "PeptideNamingResult",
    "peptide_to_iupac", "peptide_to_iupac_result",
    "SugarModification", "BaseModification", "NucleosideSpec", "NucleosideNamingResult",
    "nucleoside_to_iupac", "nucleoside_to_iupac_result",
    "NucleotideSpec", "NucleotideNamingResult", "nucleotide_to_iupac", "nucleotide_to_iupac_result",
    "OligonucleotideSequence", "NucleosideResidueBinding", "PhosphodiesterBinding", "OligonucleotideNamingResult",
    "oligonucleotide_to_iupac", "oligonucleotide_to_iupac_result",
    "GlycerolAcyl", "GlycerolEther", "GlycerolSpec", "LipidAcylBinding", "LipidEtherBinding", "GlycerolNamingResult", "glycerol_to_iupac", "glycerol_to_iupac_result",
    "PhosphoglycerolSpec", "LipidPhosphateBinding", "PhosphoglycerolNamingResult", "phosphoglycerol_to_iupac", "phosphoglycerol_to_iupac_result",
    "SphingoidDoubleBond", "SphingoidOAcyl", "SphingoidSpec", "SphingoidAcylBinding", "SphingoidNamingResult",
    "sphingoid_to_iupac", "sphingoid_to_iupac_result",
    "FischerSugar", "CarbohydrateSpec", "CarbohydrateNamingResult", "carbohydrate_to_iupac", "carbohydrate_to_iupac_result",
    "CarbohydrateModification",
    "SugarPhosphateBinding", "ModifiedCarbohydrateNamingResult",
    "GlycosideSpec", "GlycosidicBinding", "GlycosideNamingResult", "glycoside_to_iupac", "glycoside_to_iupac_result",
    "SugarLink", "OligosaccharideSpec", "SugarLinkBinding", "OligosaccharidePhosphateBinding", "OligosaccharideNamingResult", "oligosaccharide_to_iupac", "oligosaccharide_to_iupac_result",
    "CyclicOligosaccharideSpec", "cyclic_oligosaccharide_to_iupac", "cyclic_oligosaccharide_to_iupac_result",
    "GlycoglycerolSpec", "GlycosphingoidSpec", "GlycolipidNamingResult", "glycolipid_to_iupac", "glycolipid_to_iupac_result",
    "RepeatedParentCore", "RepeatedParentNamingResult", "repeated_parent_to_iupac", "repeated_parent_to_iupac_result",
    "FullereneHomo", "FullereneReplacement", "FullereneSpec", "FullereneNamingResult", "fullerene_to_iupac", "fullerene_to_iupac_result",
    "CoordinationLigand", "CoordinationGeometry", "CoordinationSpec", "CoordinationBinding", "CoordinationNamingResult", "coordination_to_iupac", "coordination_to_iupac_result",
    "PolyhedralGeometry", "PolyhedralDescription", "PolyhedralNamingResult", "describe_polyhedral_configuration", "polyhedral_to_iupac", "polyhedral_to_iupac_result",
    "NaturalProductHomo", "NaturalProductReplacement", "NaturalProductSpec", "NaturalProductNamingResult", "natural_product_to_iupac", "natural_product_to_iupac_result",
    "CyclitolSpec", "InositolSpec", "CyclitolNamingResult", "cyclitol_to_iupac", "cyclitol_to_iupac_result", "inositol_to_iupac", "inositol_to_iupac_result",
    "CyclitolSubstituent", "CyclitolSubstituentBinding",
    "HelicalUnit", "HelicalNamingResult", "helicene_to_iupac", "helicene_to_iupac_result",
]

_EXPORT_GROUPS = {
    'api': 'mol_to_iupac mol_to_iupac_result',
    '_smiles_dispatch': 'smiles_to_iupac smiles_to_iupac_batch smiles_to_iupac_result smiles_to_iupac_batch_results',
    'naming_result': 'NamingResult SelectionLimitation',
    'selection_evidence': 'PreferredNameJustification SelectionStep',
    'conformations': 'TorsionReference TorsionDescription RingConformation ConformerDescription describe_conformer',
    'isotope_labelling': 'Nuclide IsotopeLabelling LabelledSite IsotopeLabellingNamingResult isotope_labelling_to_iupac isotope_labelling_to_iupac_result',
    'radical_ions': 'ElectronTransfer ElectronTransferNamingResult electron_transfer_to_iupac electron_transfer_to_iupac_result',
    'parallel': 'IUPACBatchPool',
    'adducts': 'AdductComponent AdductNamingResult NamedAdductComponent adduct_to_iupac adduct_to_iupac_result',
    'stereo_ensembles': 'StereoEnsemble StereoEnsembleNamingResult stereo_ensemble_to_iupac stereo_ensemble_to_iupac_result',
    'stereo_subgroups': 'StereoSubgroup StereoSubgroupNamingResult stereo_subgroup_to_iupac stereo_subgroup_to_iupac_result',
    'axial_stereochemistry': 'AxialUnit AxialNamingResult atropisomer_to_iupac atropisomer_to_iupac_result',
    'cumulene_configuration': 'CumuleneNamingResult cumulene_to_iupac cumulene_to_iupac_result',
    'peptides': 'PeptideResidue PeptideSequence PeptideResidueBinding PeptideNamingResult peptide_to_iupac peptide_to_iupac_result',
    'nucleosides': 'SugarModification BaseModification NucleosideSpec NucleosideNamingResult nucleoside_to_iupac nucleoside_to_iupac_result',
    'nucleotides': 'NucleotideSpec NucleotideNamingResult nucleotide_to_iupac nucleotide_to_iupac_result',
    'oligonucleotides': 'OligonucleotideSequence NucleosideResidueBinding PhosphodiesterBinding OligonucleotideNamingResult oligonucleotide_to_iupac oligonucleotide_to_iupac_result',
    'lipids': 'GlycerolAcyl GlycerolEther GlycerolSpec LipidAcylBinding LipidEtherBinding GlycerolNamingResult glycerol_to_iupac glycerol_to_iupac_result',
    'phosphoglycerols': 'PhosphoglycerolSpec LipidPhosphateBinding PhosphoglycerolNamingResult phosphoglycerol_to_iupac phosphoglycerol_to_iupac_result',
    'sphingoids': 'SphingoidDoubleBond SphingoidOAcyl SphingoidSpec SphingoidAcylBinding SphingoidNamingResult sphingoid_to_iupac sphingoid_to_iupac_result',
    'carbohydrates': 'FischerSugar CarbohydrateModification CarbohydrateSpec CarbohydrateNamingResult carbohydrate_to_iupac carbohydrate_to_iupac_result',
    'carbohydrate_modifications': 'SugarPhosphateBinding ModifiedCarbohydrateNamingResult',
    'glycosides': 'GlycosideSpec GlycosidicBinding GlycosideNamingResult glycoside_to_iupac glycoside_to_iupac_result',
    'oligosaccharides': 'SugarLink OligosaccharideSpec SugarLinkBinding OligosaccharidePhosphateBinding OligosaccharideNamingResult oligosaccharide_to_iupac oligosaccharide_to_iupac_result CyclicOligosaccharideSpec cyclic_oligosaccharide_to_iupac cyclic_oligosaccharide_to_iupac_result',
    'glycolipids': 'GlycoglycerolSpec GlycosphingoidSpec GlycolipidNamingResult glycolipid_to_iupac glycolipid_to_iupac_result',
    'repeated_parent_paths': 'RepeatedParentCore RepeatedParentNamingResult repeated_parent_to_iupac repeated_parent_to_iupac_result',
    'fullerenes': 'FullereneHomo FullereneReplacement FullereneSpec FullereneNamingResult fullerene_to_iupac fullerene_to_iupac_result',
    'coordination': 'CoordinationLigand CoordinationGeometry CoordinationSpec CoordinationBinding CoordinationNamingResult coordination_to_iupac coordination_to_iupac_result',
    'polyhedral_configuration': 'PolyhedralGeometry PolyhedralDescription PolyhedralNamingResult describe_polyhedral_configuration polyhedral_to_iupac polyhedral_to_iupac_result',
    'natural_products': 'NaturalProductHomo NaturalProductReplacement NaturalProductSpec NaturalProductNamingResult natural_product_to_iupac natural_product_to_iupac_result',
    'cyclitols': 'CyclitolSpec InositolSpec CyclitolNamingResult cyclitol_to_iupac cyclitol_to_iupac_result inositol_to_iupac inositol_to_iupac_result',
    'cyclitol_substitution': 'CyclitolSubstituent CyclitolSubstituentBinding',
    'helical_configuration': 'HelicalUnit HelicalNamingResult helicene_to_iupac helicene_to_iupac_result',
}
_EXPORTS = {name: module for module, names in _EXPORT_GROUPS.items()
            for name in names.split()}


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_import_module('.' + module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
