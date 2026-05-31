from __future__ import annotations

import csv
import json
from importlib import resources

from .opsin_resource_utils import (
    OpsinTokenRecord,
    iter_opsin_token_records,
    preferred_token_name,
    primary_label,
    unsupported_token_attributes,
)
from .resource_manifest import FUSION_COMPONENT_FAMILY, RETAINED_PARENT_FAMILY
from .resource_operations import apply_token_operations


# Template resources often contain many acceptable names and locant schemes for
# the same scaffold SMILES. Keep those aliases on one immutable RDKit query so
# MoleculeContext's identity-based substructure cache can reuse the match work.
_RAW_TEMPLATE_MOL_CACHE: dict[tuple[int, str], object | None] = {}


# Static parent templates taken from OPSIN's ring-token resource families and
# verified with OPSIN as standalone names. The runtime package does not read or
# depend on OPSIN; these are NISPO inverse rules.
OPSIN_PARENT_TEMPLATES = (
    (
        "1,5-naphthyridine",
        "N1=CC=CC2=NC=CC=C12",
        ("1", "2", "3", "4", "4a", "5", "6", "7", "8", "8a"),
    ),
    (
        "1,6-naphthyridine",
        "N1=CC=CC2=CN=CC=C12",
        ("1", "2", "3", "4", "4a", "5", "6", "7", "8", "8a"),
    ),
    (
        "1,7-naphthyridine",
        "N1=CC=CC2=CC=NC=C12",
        ("1", "2", "3", "4", "4a", "5", "6", "7", "8", "8a"),
    ),
    (
        "1,8-naphthyridine",
        "N1=CC=CC2=CC=CN=C12",
        ("1", "2", "3", "4", "4a", "5", "6", "7", "8", "8a"),
    ),
    (
        "2,6-naphthyridine",
        "C1=NC=CC2=CN=CC=C12",
        ("1", "2", "3", "4", "4a", "5", "6", "7", "8", "8a"),
    ),
    (
        "2,7-naphthyridine",
        "C1=NC=CC2=CC=NC=C12",
        ("1", "2", "3", "4", "4a", "5", "6", "7", "8", "8a"),
    ),
    (
        "1,2-dihydronaphthalene",
        "C1CC=CC2=CC=CC=C12",
        ("1", "2", "3", "4", "4a", "5", "6", "7", "8", "8a"),
    ),
    (
        "tetralin",
        "C1CCCC2=CC=CC=C12",
        ("1", "2", "3", "4", "4a", "5", "6", "7", "8", "8a"),
    ),
    (
        "1,3-dihydroisobenzofuran",
        "C1OCC2=CC=CC=C12",
        ("1", "2", "3", "3a", "4", "5", "6", "7", "7a"),
    ),
    (
        "adamantane",
        "C12CC3CC(CC(C1)C3)C2",
        ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10"),
    ),
    (
        "spiro[2.3]hexane",
        "C1CC12CCC2",
        ("1", "2", "3", "4", "5", "6"),
    ),
    (
        "spiro[2.4]heptane",
        "C1CC12CCCC2",
        ("1", "2", "3", "4", "5", "6", "7"),
    ),
    (
        "spiro[2.5]octane",
        "C1CC12CCCCC2",
        ("1", "2", "3", "4", "5", "6", "7", "8"),
    ),
    (
        "5-azabicyclo[2.1.1]hexane",
        "C12CCC(N1)C2",
        ("1", "2", "3", "4", "5", "6"),
    ),
    (
        "6-azabicyclo[2.1.1]hexane",
        "C12CCC(C1)N2",
        ("1", "2", "3", "4", "5", "6"),
    ),
    (
        "2-azabicyclo[2.2.1]heptane",
        "C12NCC(CC1)C2",
        ("1", "2", "3", "4", "5", "6", "7"),
    ),
    (
        "3-azabicyclo[2.2.1]heptane",
        "C12CNC(CC1)C2",
        ("1", "2", "3", "4", "5", "6", "7"),
    ),
    (
        "6-azabicyclo[3.1.0]hexane",
        "C12CCCC2N1",
        ("1", "2", "3", "4", "5", "6"),
    ),
    (
        "5-azabicyclo[2.1.1]hex-2-ene",
        "C12C=CC(N1)C2",
        ("1", "2", "3", "4", "5", "6"),
    ),
    (
        "1-azabicyclo[3.2.0]heptane",
        "N12CCCC2CC1",
        ("1", "2", "3", "4", "5", "6", "7"),
    ),
    (
        "2,3-diazabicyclo[3.1.1]heptane",
        "C12NNCC(C1)C2",
        ("1", "2", "3", "4", "5", "6", "7"),
    ),
    (
        "3,6-diazabicyclo[3.2.1]octane",
        "C12CNCC(NC1)C2",
        ("1", "2", "3", "4", "5", "6", "7", "8"),
    ),
    (
        "3-oxa-6-azabicyclo[3.1.1]heptane",
        "C12COCC(N1)C2",
        ("1", "2", "3", "4", "5", "6", "7"),
    ),
    (
        "3-aza-6-oxabicyclo[3.2.1]octane",
        "C12CNCC(OC1)C2",
        ("1", "2", "3", "4", "5", "6", "7", "8"),
    ),
    (
        "2-oxa-6-azabicyclo[2.2.1]heptane",
        "C12OCC(CN1)C2",
        ("1", "2", "3", "4", "5", "6", "7"),
    ),
    (
        "5,6,7,8-tetrahydro-4,6-naphthyridin-5-one",
        "C1=CC=NC=2C(NCCC12)=O",
        ("1", "2", "3", "4", "4a", "5", "6", "7", "8", "8a", ""),
    ),
    (
        "5,6,7,8-tetrahydro-3,6-naphthyridin-5-one",
        "C1=CN=CC=2C(NCCC12)=O",
        ("1", "2", "3", "4", "4a", "5", "6", "7", "8", "8a", ""),
    ),
    (
        "2H-1,4-benzoxazin-3-one",
        "O1CC(NC2=C1C=CC=C2)=O",
        ("1", "2", "3", "4", "4a", "8a", "8", "7", "6", "5", ""),
    ),
    (
        "chromone",
        "O1C=CC(C2=CC=CC=C12)=O",
        ("1", "2", "3", "4", "4a", "5", "6", "7", "8", "8a", ""),
    ),
    (
        "chroman-4-one",
        "O1CCC(C2=CC=CC=C12)=O",
        ("1", "2", "3", "4", "4a", "5", "6", "7", "8", "8a", ""),
    ),
    (
        "benzocyclohexane",
        "C1=CC=CC2=C1CCCC2",
        ("1", "2", "3", "4", "4a", "8a", "5", "6", "7", "8"),
    ),
    (
        "benzocycloheptane",
        "C1=CC=CC2=C1CCCCC2",
        ("1", "2", "3", "4", "4a", "9a", "5", "6", "7", "8", "9"),
    ),
    (
        "benzocyclooctane",
        "C1=CC=CC2=C1CCCCCC2",
        ("1", "2", "3", "4", "4a", "10a", "5", "6", "7", "8", "9", "10"),
    ),
    (
        "benzocyclononane",
        "C1=CC=CC2=C1CCCCCCC2",
        ("1", "2", "3", "4", "4a", "11a", "5", "6", "7", "8", "9", "10", "11"),
    ),
    (
        "benzazepane",
        "N1CCCCC2=C1C=CC=C2",
        ("1", "2", "3", "4", "5", "", "", "9", "8", "7", "6"),
    ),
    (
        "benzazocane",
        "N1CCCCCC2=C1C=CC=C2",
        ("1", "2", "3", "4", "5", "6", "", "", "10", "9", "8", "7"),
    ),
    (
        "benzazonane",
        "N1CCCCCCC2=C1C=CC=C2",
        ("1", "2", "3", "4", "5", "6", "7", "", "", "11", "10", "9", "8"),
    ),
    (
        "benzoxepane",
        "O1CCCCC2=C1C=CC=C2",
        ("1", "2", "3", "4", "5", "", "", "9", "8", "7", "6"),
    ),
    (
        "benzoxocane",
        "O1CCCCCC2=C1C=CC=C2",
        ("1", "2", "3", "4", "5", "6", "", "", "10", "9", "8", "7"),
    ),
    (
        "benzoxonane",
        "O1CCCCCCC2=C1C=CC=C2",
        ("1", "2", "3", "4", "5", "6", "7", "", "", "11", "10", "9", "8"),
    ),
    (
        "benzthiepane",
        "S1CCCCC2=C1C=CC=C2",
        ("1", "2", "3", "4", "5", "", "", "9", "8", "7", "6"),
    ),
    (
        "benzthiocane",
        "S1CCCCCC2=C1C=CC=C2",
        ("1", "2", "3", "4", "5", "6", "", "", "10", "9", "8", "7"),
    ),
    (
        "benzthionane",
        "S1CCCCCCC2=C1C=CC=C2",
        ("1", "2", "3", "4", "5", "6", "7", "", "", "11", "10", "9", "8"),
    ),
    (
        "benzdiazepane",
        "N1NCCCC2=C1C=CC=C2",
        ("1", "2", "3", "4", "5", "", "", "9", "8", "7", "6"),
    ),
    (
        "benzdiazocane",
        "N1NCCCCC2=C1C=CC=C2",
        ("1", "2", "3", "4", "5", "6", "", "", "10", "9", "8", "7"),
    ),
    (
        "benzdiazonane",
        "N1NCCCCCC2=C1C=CC=C2",
        ("1", "2", "3", "4", "5", "6", "7", "", "", "11", "10", "9", "8"),
    ),
    (
        "benzoxazepane",
        "O1NCCCC2=C1C=CC=C2",
        ("1", "2", "3", "4", "5", "", "", "9", "8", "7", "6"),
    ),
    (
        "benzoxazocane",
        "O1NCCCCC2=C1C=CC=C2",
        ("1", "2", "3", "4", "5", "6", "", "", "10", "9", "8", "7"),
    ),
    (
        "benzoxazonane",
        "O1NCCCCCC2=C1C=CC=C2",
        ("1", "2", "3", "4", "5", "6", "7", "", "", "11", "10", "9", "8"),
    ),
    (
        "benzthiazepane",
        "S1NCCCC2=C1C=CC=C2",
        ("1", "2", "3", "4", "5", "", "", "9", "8", "7", "6"),
    ),
    (
        "benzthiazocane",
        "S1NCCCCC2=C1C=CC=C2",
        ("1", "2", "3", "4", "5", "6", "", "", "10", "9", "8", "7"),
    ),
    (
        "benzthiazonane",
        "S1NCCCCCC2=C1C=CC=C2",
        ("1", "2", "3", "4", "5", "6", "7", "", "", "11", "10", "9", "8"),
    ),
    (
        "1,3-benzodioxane",
        "O1COCc2c1cccc2",
        ("1", "2", "3", "4", "4a", "5", "6", "7", "8", "8a"),
    ),
    (
        "1,3-benzoxazine",
        "O1CN=Cc2c1cccc2",
        ("1", "2", "3", "4", "4a", "5", "6", "7", "8", "8a"),
    ),
    (
        "1,4-benzoxazine",
        "O1CC=Nc2c1cccc2",
        ("1", "2", "3", "4", "4a", "5", "6", "7", "8", "8a"),
    ),
    (
        "3,4-dihydro-2H-1,4-benzoxazine",
        "O1CCNc2c1cccc2",
        ("1", "2", "3", "4", "4a", "5", "6", "7", "8", "8a"),
    ),
    (
        "1,3-benzothiazine",
        "S1CN=Cc2c1cccc2",
        ("1", "2", "3", "4", "4a", "5", "6", "7", "8", "8a"),
    ),
    (
        "1,4-benzothiazine",
        "S1CC=Nc2c1cccc2",
        ("1", "2", "3", "4", "4a", "5", "6", "7", "8", "8a"),
    ),
    (
        "1,5-dihydro-4,1-benzothiazepin-2-one",
        "O=C1CSCc2ccccc2N1",
        ("", "2", "3", "4", "5", "5a", "6", "7", "8", "9", "9a", "1"),
    ),
    (
        "2,3-dihydro-1,5-benzothiazepin-4(5H)-one",
        "S1CCC(NC2=C1C=CC=C2)=O",
        ("1", "2", "3", "4", "5", "5a", "9a", "9", "8", "7", "6", ""),
    ),
    (
        "3,5-dihydro-2H-1,4-benzothiazepine",
        "S1CCNCC2=C1C=CC=C2",
        ("1", "2", "3", "4", "5", "5a", "9a", "9", "8", "7", "6"),
    ),
    (
        "1,1-dioxido-2,3,4,5-tetrahydro-1,5-benzothiazepine",
        "O=S1(CCCNC2=C1C=CC=C2)=O",
        ("", "1", "2", "3", "4", "5", "5a", "9a", "9", "8", "7", "6", ""),
    ),
    (
        "1,1-dioxido-2,3,4,5-tetrahydro-1,2,5-benzothiadiazepine",
        "O=S1(NCCNC2=C1C=CC=C2)=O",
        ("", "1", "2", "3", "4", "5", "5a", "9a", "9", "8", "7", "6", ""),
    ),
    (
        "2,7-diazaspiro[3.5]nonane",
        "C1NCC12CCNCC2",
        ("1", "2", "3", "4", "5", "6", "7", "8", "9"),
    ),
    (
        "5-azaspiro[2.5]octane",
        "C1CC12CNCCC2",
        ("1", "2", "3", "4", "5", "6", "7", "8"),
    ),
    (
        "2-azaspiro[3.3]heptane",
        "C1NCC12CCC2",
        ("1", "2", "3", "4", "5", "6", "7"),
    ),
    (
        "5-azaspiro[3.4]octane",
        "C1CCC12NCCC2",
        ("1", "2", "3", "4", "5", "6", "7", "8"),
    ),
    (
        "6-azaspiro[3.4]octane",
        "C1CCC12CNCC2",
        ("1", "2", "3", "4", "5", "6", "7", "8"),
    ),
    (
        "2-azaspiro[3.5]nonane",
        "C1NCC12CCCCC2",
        ("1", "2", "3", "4", "5", "6", "7", "8", "9"),
    ),
    (
        "beta-carboline",
        "c1nccc2c3ccccc3[nH]c12",
        ("1", "2", "3", "4", "4a", "4b", "5", "6", "7", "8", "8a", "9", "9a"),
    ),
    (
        "bibenzyl",
        "c1(ccccc1)CCc2ccccc2",
        ("1", "2", "3", "4", "5", "6", "alpha", "alpha'", "1'", "2'", "3'", "4'", "5'", "6'"),
    ),
    (
        "indolizine",
        "c1ccn2ccccc12",
        ("1", "2", "3", "4", "5", "6", "7", "8", "8a"),
    ),
    (
        "indoxazene",
        "o1ncc2ccccc12",
        ("1", "2", "3", "3a", "4", "5", "6", "7", "7a"),
    ),
    (
        "isobenzothiophene",
        "c1scc2ccccc12",
        ("1", "2", "3", "3a", "4", "5", "6", "7", "7a"),
    ),
    (
        "isochroman",
        "C1OCCc2ccccc12",
        ("1", "2", "3", "4", "4a", "5", "6", "7", "8", "8a"),
    ),
    (
        "isochromene",
        "[cH2]1occc2ccccc12",
        ("1", "2", "3", "4", "4a", "5", "6", "7", "8", "8a"),
    ),
    (
        "norbornene",
        "C12C=CC(CC2)C1",
        ("1", "2", "3", "4", "5", "6", "7"),
    ),
    (
        "phenanthrene",
        "c1cccc2c3ccccc3ccc12",
        ("1", "2", "3", "4", "4a", "4b", "5", "6", "7", "8", "8a", "9", "10", "10a"),
    ),
    (
        "phenazine",
        "c1cccc2nc3ccccc3nc12",
        ("1", "2", "3", "4", "4a", "5", "5a", "6", "7", "8", "9", "9a", "10", "10a"),
    ),
    (
        "phenothiazine",
        "c1cccc2sc3ccccc3[nH]c12",
        ("1", "2", "3", "4", "4a", "5", "5a", "6", "7", "8", "9", "9a", "10", "10a"),
    ),
    (
        "phenoxathiin",
        "c1cccc2oc3ccccc3sc12",
        ("1", "2", "3", "4", "4a", "5", "5a", "6", "7", "8", "9", "9a", "10", "10a"),
    ),
    (
        "quinuclidine",
        "N1(CCC2CC1)CC2",
        ("1", "2", "3", "4", "5", "6", "7", "8"),
    ),
    (
        "stilbene",
        "c1(ccccc1)C=Cc2ccccc2",
        ("1", "2", "3", "4", "5", "6", "alpha", "beta", "1'", "2'", "3'", "4'", "5'", "6'"),
    ),
    (
        "xanthene",
        "c1cccc2oc3ccccc3[cH2]c12",
        ("1", "2", "3", "4", "4a", "10", "10a", "5", "6", "7", "8", "8a", "9", "9a"),
    ),
    (
        "isoflavone",
        "o1cc(c(=O)c2ccccc12)c3ccccc3",
        ("1", "2", "3", "4", "", "4a", "5", "6", "7", "8", "8a", "1'", "2'", "3'", "4'", "5'", "6'"),
    ),
    (
        "anthra-9,10-quinone",
        "c1cccc2c(=O)c3ccccc3c(=O)c12",
        ("1", "2", "3", "4", "4a", "10", "", "10a", "5", "6", "7", "8", "8a", "9", "", "9a"),
    ),
    (
        "benzil",
        "c1(ccccc1)C(=O)C(=O)c2ccccc2",
        ("1", "2", "3", "4", "5", "6", "", "", "", "", "1'", "2'", "3'", "4'", "5'", "6'"),
    ),
    (
        "naphtho-1,4-quinone",
        "c1(=O)ccc(=O)c2ccccc12",
        ("1", "", "2", "3", "4", "", "4a", "5", "6", "7", "8", "8a"),
    ),
    (
        "quinolizine",
        "c1cc[cH2]n2ccccc12",
        ("1", "2", "3", "4", "5", "6", "7", "8", "9", "9a"),
    ),
    (
        "pterin",
        "n1c(N)[nH]c(=O)c2nccnc12",
        ("1", "2", "", "3", "4", "", "4a", "5", "6", "7", "8", "8a"),
    ),
    (
        "purine",
        "n1cnc2nc[nH]c2c1",
        ("1", "2", "3", "4", "9", "8", "7", "5", "6"),
    ),
    (
        "acenaphthene",
        "C1Cc2cccc3cccc1c23",
        ("1", "2", "2a", "3", "4", "5", "5a", "6", "7", "8", "8a", "8b"),
    ),
    (
        "tritan",
        "C(c1ccccc1)(c1ccccc1)c1ccccc1",
        ("", "1", "2", "3", "4", "5", "6", "1'", "2'", "3'", "4'", "5'", "6'", "1''", "2''", "3''", "4''", "5''", "6''"),
    ),
)

OPSIN_RESOURCE_FILES = (
    *RETAINED_PARENT_FAMILY.files,
)

SUPPORTED_TOKEN_ATTRIBUTES = {
    "value",
    "labels",
    "addGroup",
    "addBond",
    "addHeteroAtom",
    "frontLocantsExpected",
    "fusedRingNumbering",
    "functionalIDs",
    "defaultInLocant",
    "defaultInID",
    "outIDs",
    "usableAsAJoiner",
    "suffixAppliesToByDefault",
    "suffixAppliesTo",
    "alphaBetaClockWiseAtomOrdering",
    "naturalEntIsOpposite",
    "additionalValue",
}

PARENT_NAME_NORMALIZATIONS = {
    "acridin": "acridine",
    "anthrazin": "anthrazine",
    "anthyridin": "anthyridine",
    "arsanthridin": "arsanthridine",
    "arsindolizin": "arsindolizine",
    "arsinolin": "arsinoline",
    "arsinolizin": "arsinolizine",
    "as-triazin": "as-triazine",
    "benzen": "benzene",
    "benzol": "benzene",
    "benzimidazol": "benzimidazole",
    "betacarbolin": "beta-carboline",
    "carbazol": "carbazole",
    "chinolin": "quinoline",
    "coumaron": "benzofuran",
    "cinnolin": "cinnoline",
    "imidazol": "imidazole",
    "indazol": "indazole",
    "indol": "indole",
    "indolizin": "indolizine",
    "isoarsinolin": "isoarsinoline",
    "isochinolin": "isoquinoline",
    "isoindol": "isoindole",
    "isoquinolin": "isoquinoline",
    "isoxazol": "isoxazole",
    "isothiazol": "isothiazole",
    "naphthyridin": "naphthyridine",
    "perimidin": "perimidine",
    "phenanthrazin": "phenanthrazine",
    "phenanthridin": "phenanthridine",
    "phenazin": "phenazine",
    "phenothiazin": "phenothiazine",
    "phenoxazin": "phenoxazine",
    "phthalazin": "phthalazine",
    "pteridin": "pteridine",
    "pyrazin": "pyrazine",
    "pyrazol": "pyrazole",
    "pyridazin": "pyridazine",
    "pyridin": "pyridine",
    "pyrimidin": "pyrimidine",
    "pyrrol": "pyrrole",
    "quinolin": "quinoline",
    "quinazolin": "quinazoline",
    "quinuclidin": "quinuclidine",
    "quinoxalin": "quinoxaline",
    "s-triazol": "s-triazole",
    "s-triazin": "s-triazine",
    "tetrazol": "tetrazole",
    "thiazol": "thiazole",
    "thiofuran": "thiophene",
    "triazol": "triazole",
    "v-triazin": "v-triazine",
}

FUSION_COMPONENT_PARENT_NORMALIZATIONS = {
    "benzo": "benzene",
    "benz": "benzene",
    "betacarbolino": "beta-carboline",
    "beta-carbolino": "beta-carboline",
    "b-carbolino": "beta-carboline",
    "chinolino": "quinoline",
    "cinnolino": "cinnoline",
    "furo": "furan",
    "furano": "furan",
    "imidazo": "imidazole",
    "imidazolo": "imidazole",
    "isobenzofurano": "isobenzofuran",
    "isobenzofuro": "isobenzofuran",
    "isooxazolo": "isoxazole",
    "isothiazolo": "isothiazole",
    "isoquino": "isoquinoline",
    "isoquinolino": "isoquinoline",
    "oxazolo": "oxazole",
    "pyridino": "pyridine",
    "pyrido": "pyridine",
    "pyrimidino": "pyrimidine",
    "pyrimido": "pyrimidine",
    "pyridazino": "pyridazine",
    "pyrazino": "pyrazine",
    "pyrrolo": "pyrrole",
    "pyrazolo": "pyrazole",
    "quino": "quinoline",
    "quinolino": "quinoline",
    "selenopheno": "selenophene",
    "s-triazino": "s-triazine",
    "sym-triazino": "sym-triazine",
    "s-triazolo": "s-triazole",
    "sym-triazolo": "sym-triazole",
    "telluropheno": "tellurophene",
    "thieno": "thiophene",
    "thiopheno": "thiophene",
}

FUSION_COMPONENT_SUFFIX_REWRITES = (
    ("azolino", "azoline"),
    ("oxalino", "oxaline"),
    ("azinino", "azinine"),
    ("azino", "azine"),
    ("olino", "oline"),
    ("ino", "ine"),
    ("eno", "ene"),
    ("ano", "ane"),
    ("olo", "ole"),
)

FUSION_COMPONENT_PARENT_BLOCKLIST = {
    "acenaphth",
    "anthra",
    "naphth",
    "phenanthr",
    "peryl",
}


def build_opsin_parent_templates(Chem) -> tuple[dict, ...]:
    return (
        build_static_parent_templates(Chem)
        + build_opsin_test_scaffold_templates(Chem)
        + build_generated_spiro_scaffold_templates(Chem)
        + build_opsin_retained_ring_csv_templates(Chem)
        + build_opsin_resource_templates(Chem)
        + build_fusion_component_parent_templates(Chem)
        + build_generated_fused_parent_templates(Chem)
    )


def build_static_parent_templates(Chem) -> tuple[dict, ...]:
    templates = []
    for name, smiles, locants in OPSIN_PARENT_TEMPLATES:
        mol = _raw_template_mol(Chem, smiles)
        if mol is None:
            continue
        templates.append(
            {
                "name": name,
                "mol": mol,
                "locants": {idx: locant for idx, locant in enumerate(locants)},
                "source_family": "static-retained",
                "source_file": "",
                "source_subtype": "",
                "stereochemical": _has_stereochemical_smiles(smiles),
            }
        )
    return tuple(templates)


def build_opsin_test_scaffold_templates(Chem) -> tuple[dict, ...]:
    return _build_json_scaffold_templates(
        Chem,
        _read_json_records("opsin_test_scaffold_templates.json"),
        source_family="opsin-test-scaffold",
        default_source_file="opsin_test_scaffold_templates.json",
    )


def build_generated_spiro_scaffold_templates(Chem) -> tuple[dict, ...]:
    return _build_json_scaffold_templates(
        Chem,
        _read_json_records("generated_spiro_scaffold_templates.json"),
        source_family="generated-spiro-scaffold",
        default_source_file="generated_spiro_scaffold_templates.json",
    )


def build_opsin_retained_ring_csv_templates(Chem, max_atoms: int = 54) -> tuple[dict, ...]:
    locanted_records = _read_json_records("opsin_retained_mono_fused_ring_templates.json")
    if locanted_records:
        return tuple(
            template
            for template in _build_json_scaffold_templates(
                Chem,
                locanted_records,
                source_family="opsin-retained-ring-cml",
                default_source_file="opsin_retained_mono_fused_ring_templates.json",
            )
            if template["mol"].GetNumAtoms() <= max_atoms
        )

    templates = []
    seen: set[tuple[str, str]] = set()
    for record in _read_retained_ring_csv_records("opsin_retained_mono_fused_ring_smiles.csv"):
        name = record.get("iupac_name", "").strip()
        smiles = record.get("smiles", "").strip()
        if not name or not smiles:
            continue
        mol = _raw_template_mol(Chem, smiles)
        if mol is None or mol.GetNumAtoms() > max_atoms or mol.GetRingInfo().NumRings() == 0:
            continue
        locants = {idx: str(idx + 1) for idx in range(mol.GetNumAtoms())}
        template = {
            "name": name,
            "mol": mol,
            "locants": locants,
            "source_family": "opsin-retained-ring-csv",
            "source_file": "opsin_retained_mono_fused_ring_smiles.csv",
            "source_subtype": "exact-retained-ring",
            "stereochemical": _has_stereochemical_smiles(smiles),
        }
        dedupe_key = (template["name"], Chem.MolToSmiles(mol, canonical=True))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        templates.append(template)
    return tuple(templates)


def build_opsin_resource_templates(Chem) -> tuple[dict, ...]:
    return build_retained_parent_templates(Chem, min_rings=2)


def build_fusion_component_parent_templates(Chem, min_rings: int = 2, max_atoms: int = 48) -> tuple[dict, ...]:
    templates = []
    seen: set[tuple[str, str]] = set()
    for record in iter_opsin_token_records(FUSION_COMPONENT_FAMILY.files):
        for name in _fusion_component_parent_names(record.token.text or ""):
            template = _template_from_token(
                Chem,
                record,
                min_rings=min_rings,
                max_atoms=max_atoms,
                name_override=name,
                allow_dot_ring_closures=True,
            )
            if not template:
                continue
            key = (template["name"], Chem.MolToSmiles(template["mol"], canonical=True))
            if key in seen:
                continue
            seen.add(key)
            templates.append(template)
    return tuple(templates)


def build_generated_fused_parent_templates(Chem) -> tuple[dict, ...]:
    return build_generated_fused_parent_templates_from_records(
        Chem,
        _read_generated_fused_records(),
        source_file="generated_fused_templates.json",
    )


def build_generated_fused_parent_templates_from_records(
    Chem, records: tuple[dict, ...] | list[dict], source_file: str
) -> tuple[dict, ...]:
    return _build_json_scaffold_templates(
        Chem,
        records,
        source_family="generated-fused",
        default_source_file=source_file,
    )


def _build_json_scaffold_templates(
    Chem,
    records: tuple[dict, ...] | list[dict],
    *,
    source_family: str,
    default_source_file: str,
) -> tuple[dict, ...]:
    templates = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        mol = _raw_template_mol(Chem, record.get("smiles", ""))
        if mol is None:
            continue
        locants = {}
        for key, value in record.get("locants", {}).items():
            try:
                locants[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
        if not locants:
            continue
        template = {"name": record.get("name", ""), "mol": mol, "locants": locants}
        if not template["name"]:
            continue
        template.update(
            {
                "source_family": source_family,
                "source_file": record.get("source_file") or default_source_file,
                "source_subtype": record.get("source_subtype", ""),
                "stereochemical": _has_stereochemical_smiles(record.get("smiles", "")),
            }
        )
        dedupe_key = (template["name"], Chem.MolToSmiles(mol, canonical=True))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        templates.append(template)
    return tuple(templates)


def _raw_template_mol(Chem, smiles: str):
    if not smiles:
        return None
    key = (id(Chem), smiles)
    if key not in _RAW_TEMPLATE_MOL_CACHE:
        try:
            _RAW_TEMPLATE_MOL_CACHE[key] = Chem.MolFromSmiles(smiles)
        except Exception:
            _RAW_TEMPLATE_MOL_CACHE[key] = None
    return _RAW_TEMPLATE_MOL_CACHE[key]


def _read_generated_fused_records() -> tuple[dict, ...]:
    try:
        path = resources.files(__package__).joinpath("resources", "generated_fused_template_index_records.json")
        table = json.loads(path.read_text(encoding="utf-8"))
        index_path = resources.files(__package__).joinpath("resources", "generated_fused_templates.json")
        return _resolve_generated_record_refs(json.loads(index_path.read_text(encoding="utf-8")), table)
    except Exception:
        return ()


def _read_json_records(filename: str) -> tuple[dict, ...]:
    try:
        path = resources.files(__package__).joinpath("resources", filename)
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ()
    if not isinstance(raw, list):
        return ()
    return tuple(record for record in raw if isinstance(record, dict))


def _read_retained_ring_csv_records(filename: str) -> tuple[dict, ...]:
    try:
        path = resources.files(__package__).joinpath("resources", filename)
        with path.open("r", encoding="utf-8", newline="") as handle:
            return tuple(dict(row) for row in csv.DictReader(handle))
    except Exception:
        return ()


def _resolve_generated_record_refs(value, table: list[dict]) -> tuple[dict, ...]:
    if not isinstance(value, list):
        return ()
    records = []
    for item in value:
        if isinstance(item, int) and 0 <= item < len(table):
            record = table[item]
            if isinstance(record, dict):
                records.append(record)
        elif isinstance(item, dict):
            records.append(item)
    return tuple(records)


def build_retained_parent_templates(
    Chem,
    min_rings: int = 0,
    max_rings: int | None = None,
    max_atoms: int = 32,
) -> tuple[dict, ...]:
    templates = []
    seen: set[tuple[str, str]] = set()
    for record in iter_opsin_token_records(OPSIN_RESOURCE_FILES):
        template = _template_from_token(Chem, record, min_rings=min_rings, max_rings=max_rings, max_atoms=max_atoms)
        if not template:
            continue
        key = (template["name"], Chem.MolToSmiles(template["mol"], canonical=True))
        if key in seen:
            continue
        seen.add(key)
        templates.append(template)
    return tuple(templates)


def _template_from_token(
    Chem,
    record_or_token,
    min_rings: int = 2,
    max_rings: int | None = None,
    max_atoms: int = 32,
    name_override: str | None = None,
    allow_dot_ring_closures: bool = False,
) -> dict | None:
    record = record_or_token if isinstance(record_or_token, OpsinTokenRecord) else None
    token = record.token if record is not None else record_or_token
    if unsupported_token_attributes(token, SUPPORTED_TOKEN_ATTRIBUTES):
        return None
    smiles = token.attrib.get("value", "")
    labels_text = token.attrib.get("labels", "")
    if not smiles or not labels_text:
        return None
    if "." in smiles and not allow_dot_ring_closures:
        return None
    name = name_override or preferred_token_name(token.text or "", PARENT_NAME_NORMALIZATIONS)
    if not name:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    labels = tuple(primary_label(label) for label in labels_text.split("/"))
    if len(labels) != mol.GetNumAtoms():
        return None
    mol = apply_token_operations(Chem, mol, labels, token)
    if mol is None:
        return None
    if mol.GetNumAtoms() > max_atoms:
        return None
    ring_count = mol.GetRingInfo().NumRings()
    if ring_count < min_rings:
        return None
    if max_rings is not None and ring_count > max_rings:
        return None
    return {
        "name": name,
        "mol": mol,
        "locants": {idx: locant for idx, locant in enumerate(labels)},
        "source_family": _record_family(record),
        "source_file": record.file_name if record is not None else "",
        "source_subtype": record.context.sub_type if record is not None else "",
        "stereochemical": _has_stereochemical_smiles(smiles),
    }


def _record_family(record: OpsinTokenRecord | None) -> str:
    if record is None:
        return "opsin-token"
    context = record.context
    parts = [record.file_name.removesuffix(".xml")]
    if context.type:
        parts.append(context.type)
    if context.sub_type:
        parts.append(context.sub_type)
    return ":".join(parts)


def _has_stereochemical_smiles(smiles: str) -> bool:
    return any(marker in smiles for marker in ("@", "/", "\\"))


def _fusion_component_parent_names(text: str) -> tuple[str, ...]:
    names = []
    for alias in (part.strip() for part in text.split("|")):
        if not alias or not alias.isascii() or alias.startswith("("):
            continue
        name = _fusion_component_parent_name(alias)
        if name and name not in FUSION_COMPONENT_PARENT_BLOCKLIST and name not in names:
            names.append(name)
    return tuple(names)


def _fusion_component_parent_name(alias: str) -> str:
    normalized = FUSION_COMPONENT_PARENT_NORMALIZATIONS.get(alias)
    if normalized:
        return normalized
    for suffix, replacement in FUSION_COMPONENT_SUFFIX_REWRITES:
        if alias.endswith(suffix):
            return alias[: -len(suffix)] + replacement
    return alias[:-1] if alias.endswith("o") else alias
