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
        "tetracene",
        "C1=CC=CC2=CC3=CC4=CC=CC=C4C=C3C=C12",
        ("1", "2", "3", "4", "4a", "5", "5a", "6", "6a", "7", "8", "9", "10", "10a", "11", "11a", "12", "12a"),
    ),
    (
        "pentacene",
        "C1=CC=CC2=CC3=CC4=CC5=CC=CC=C5C=C4C=C3C=C12",
        (
            "1",
            "2",
            "3",
            "4",
            "4a",
            "5",
            "5a",
            "6",
            "6a",
            "7",
            "7a",
            "8",
            "9",
            "10",
            "11",
            "11a",
            "12",
            "12a",
            "13",
            "13a",
            "14",
            "14a",
        ),
    ),
    (
        "hexacene",
        "C1=CC=CC2=CC3=CC4=CC5=CC6=CC=CC=C6C=C5C=C4C=C3C=C12",
        (
            "1",
            "2",
            "3",
            "4",
            "4a",
            "5",
            "5a",
            "6",
            "6a",
            "7",
            "7a",
            "8",
            "8a",
            "9",
            "10",
            "11",
            "12",
            "12a",
            "13",
            "13a",
            "14",
            "14a",
            "15",
            "15a",
            "16",
            "16a",
        ),
    ),
    (
        "spirobifluorene",
        "C12(C=CC=C3C4=CC=CC=C4C=C13)C=CC=C1C3=CC=CC=C3C=C12",
        (
            "1",
            "2",
            "3",
            "4",
            "4a",
            "4b",
            "5",
            "6",
            "7",
            "8",
            "8a",
            "9",
            "9a",
            "2'",
            "3'",
            "4'",
            "4'a",
            "4'b",
            "5'",
            "6'",
            "7'",
            "8'",
            "8'a",
            "9'",
            "9'a",
        ),
    ),
    (
        "benzopentacene",
        "C1=CC=CC=2C=CC=3C=C4C=C5C=C6C=CC=CC6=CC5=CC4=CC3C21",
        (
            "1",
            "2",
            "3",
            "4",
            "4a",
            "5",
            "6",
            "6a",
            "7",
            "7a",
            "8",
            "8a",
            "9",
            "9a",
            "10",
            "11",
            "12",
            "13",
            "13a",
            "14",
            "14a",
            "15",
            "15a",
            "16",
            "16a",
            "16b",
        ),
    ),
    (
        "pentahelicene",
        "C1=CC=CC2=CC=C3C=CC4=CC=C5C=CC=CC5=C4C3=C12",
        (
            "1",
            "2",
            "3",
            "4",
            "4a",
            "5",
            "6",
            "6a",
            "7",
            "8",
            "8a",
            "9",
            "10",
            "10a",
            "11",
            "12",
            "13",
            "14",
            "14a",
            "14b",
            "14c",
            "14d",
        ),
    ),
    (
        "hexahelicene",
        "C1=CC=CC2=CC=C3C=CC4=CC=C5C=CC6=CC=CC=C6C5=C4C3=C12",
        (
            "1",
            "2",
            "3",
            "4",
            "4a",
            "5",
            "6",
            "6a",
            "7",
            "8",
            "8a",
            "9",
            "10",
            "10a",
            "11",
            "12",
            "12a",
            "13",
            "14",
            "15",
            "16",
            "16a",
            "16b",
            "16c",
            "16d",
            "16e",
        ),
    ),
    (
        "heptahelicene",
        "C1=CC=CC2=CC=C3C=CC4=CC=C5C=CC6=CC=C7C=CC=CC7=C6C5=C4C3=C12",
        (
            "1",
            "2",
            "3",
            "4",
            "4a",
            "5",
            "6",
            "6a",
            "7",
            "8",
            "8a",
            "9",
            "10",
            "10a",
            "11",
            "12",
            "12a",
            "13",
            "14",
            "14a",
            "15",
            "16",
            "17",
            "18",
            "18a",
            "18b",
            "18c",
            "18d",
            "18e",
            "18f",
        ),
    ),
    (
        "indolocarbazole",
        "C1=C2C(=CC=C1)N=C1C=CC3=C4C=CC=CC4=NC3=C12",
        (
            "1",
            "12c",
            "4a",
            "4",
            "3",
            "2",
            "5",
            "5a",
            "6",
            "7",
            "7a",
            "7b",
            "8",
            "9",
            "10",
            "11",
            "11a",
            "12",
            "12a",
            "12b",
        ),
    ),
    (
        "indolocarbazoloquinoline",
        "C1=CC=NC=2C3=C(C=CC12)C=1N=C2C=4C(=CC=C2C1C=C3)N=C3C=CC=CC34",
        (
            "1",
            "2",
            "3",
            "N",
            "4a",
            "4b",
            "5",
            "6",
            "6a",
            "6b",
            "7",
            "8",
            "8a",
            "N'",
            "9a",
            "10",
            "11",
            "12",
            "13",
            "13a",
            "13b",
            "13c",
            "N''",
            "14a",
            "14b",
            "15",
            "16",
            "16a",
        ),
    ),
    (
        "tetrabenzoazonine",
        "N1C2=C(C3=C(C4=C(C5=C1C=CC=C5)C=CC=C4)C=CC=C3)C=CC=C2",
        (
            "N",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        ),
    ),
    (
        "4,5,9,16-tetraaza-dibenzo[a,c]naphthacene",
        "C1=CC=NC2=C1C1=NC3=CC4=CC=CC=C4C=C3N=C1C1=C2N=CC=C1",
        (
            "1",
            "2",
            "3",
            "N'''",
            "4a",
            "4b",
            "N''",
            "6",
            "7",
            "8",
            "8a",
            "8b",
            "N'",
            "9a",
            "10",
            "10a",
            "11",
            "12",
            "13",
            "14",
            "14a",
            "15",
            "15a",
            "N",
            "16a",
            "16b",
        ),
    ),
    (
        "benzochrysene",
        "C1=CC=CC2=C1C1=C3C=CC=CC3=CC=C1C=1C=CC=CC21",
        (
            "1",
            "2",
            "3",
            "4",
            "4a",
            "4b",
            "5",
            "6",
            "7",
            "8",
            "8a",
            "8b",
            "9",
            "10",
            "10a",
            "11",
            "12",
            "13",
            "14",
            "14a",
            "14b",
            "14c",
        ),
    ),
    (
        "benzoanthracene",
        "C1=CC=CC=2C=CC=3C=C4C=CC=CC4=CC3C21",
        (
            "1",
            "2",
            "3",
            "4",
            "4a",
            "5",
            "6",
            "6a",
            "7",
            "7a",
            "8",
            "9",
            "10",
            "11",
            "11a",
            "12",
            "12a",
            "12b",
        ),
    ),
    (
        "benzo[a]pyrene",
        "C1=CC=C2C=CC=3C=C4C(=C5C=CC1=C2C53)C=CC=C4",
        (
            "1",
            "2",
            "3",
            "3a",
            "4",
            "5",
            "5a",
            "6",
            "6a",
            "7",
            "8",
            "9",
            "10",
            "10a",
            "10b",
            "11",
            "12",
            "12a",
            "12b",
            "12c",
        ),
    ),
    (
        "benz[a]anthracene",
        "C1=CC=CC=2C1=C1C=C3C=CC=CC3=CC1=CC2",
        (
            "1",
            "2",
            "3",
            "4",
            "4a",
            "5",
            "6",
            "6a",
            "7",
            "7a",
            "8",
            "9",
            "10",
            "11",
            "11a",
            "12",
            "12a",
            "12b",
        ),
    ),
    (
        "11H-indeno[1,2-b]quinolin-11-one",
        "C1=C2C(C=3C(=NC=4C=CC=CC4C3)C2=CC=C1)=O",
        (
            "1",
            "2",
            "3",
            "4",
            "4a",
            "4b",
            "N",
            "5a",
            "6",
            "7",
            "8",
            "9",
            "9a",
            "10",
            "10a",
            "11",
            "11a",
            "",
        ),
    ),
    (
        "spiro[fluoreno[3,2-b]benzofuran-11,5'-indeno[1,2-c]pyridine]",
        "C1=NC=CC2=C1C1=CC=CC=C1C21C2=CC=CC=C2C2=CC=3OC4=C(C3C=C21)C=CC=C4",
        (
            "1'",
            "N",
            "3'",
            "4'",
            "4'a",
            "5'",
            "5'a",
            "6'",
            "7'",
            "8'",
            "9'",
            "9'a",
            "9'b",
            "1",
            "2",
            "3",
            "4",
            "4a",
            "5",
            "5a",
            "6",
            "6a",
            "6b",
            "7",
            "8",
            "9",
            "10",
            "10a",
            "11a",
            "12",
            "12a",
            "12b",
        ),
    ),
    (
        "spiro[fluorene-9,8'-indolo[3,2,1-de]acridine]",
        "C1=CC=CC2=C1N1C3=C2C=CC=C3C3(C=2C=CC=CC12)C1=CC=CC=C1C=1C=CC=CC13",
        (
            "1'",
            "2'",
            "3'",
            "4'",
            "4'a",
            "4'b",
            "5'",
            "6'",
            "7'",
            "7'a",
            "9",
            "8'a",
            "9'",
            "10'",
            "11'",
            "12'",
            "12'a",
            "13'",
            "13'a",
            "13'b",
            "1",
            "2",
            "3",
            "4",
            "4a",
            "4b",
            "5",
            "6",
            "7",
            "8",
            "8a",
            "9a",
        ),
    ),
    (
        "fluoreno[3,2-b]benzofuran",
        "C1=CC=CC2=C1C1=C(O2)CC2=C3C=CC=CC3=CC2=C1",
        (
            "1",
            "2",
            "3",
            "4",
            "4a",
            "O",
            "5a",
            "6",
            "6a",
            "6b",
            "7",
            "8",
            "9",
            "10",
            "10a",
            "11",
            "11a",
            "12",
            "12a",
            "12b",
        ),
    ),
    (
        "indeno[1,2-c]pyridine",
        "C1=NC=CC2=C1C1=CC=CC=C1C2",
        (
            "1",
            "N",
            "3",
            "4",
            "4a",
            "5",
            "5a",
            "6",
            "7",
            "8",
            "9",
            "9a",
            "9b",
        ),
    ),
    (
        "dibenzo[c,h]acridine",
        "C1=CC=CC=2C=CC=3C=C4C=CC5=C(C4=NC3C21)C=CC=C5",
        (
            "1",
            "2",
            "3",
            "4",
            "4a",
            "5",
            "6",
            "6a",
            "7",
            "7a",
            "8",
            "9",
            "9a",
            "13a",
            "13b",
            "14",
            "14a",
            "14b",
            "13",
            "12",
            "11",
            "10",
        ),
    ),
    (
        "dibenzoxanthene",
        "C1=CC=CC2=C3C(=C4OC=5C=CC=CC5CC4=C21)C=CC=C3",
        (
            "1",
            "2",
            "3",
            "4",
            "4a",
            "4b",
            "8a",
            "8b",
            "9",
            "9a",
            "10",
            "11",
            "12",
            "13",
            "13a",
            "14",
            "14a",
            "14b",
            "8",
            "7",
            "6",
            "5",
        ),
    ),
    (
        "phenanthroindole",
        "N1C=CC2=CC=C3C(=C12)C=CC=1C=2C=CC=CC2C=CC13",
        (
            "1",
            "2",
            "3",
            "3a",
            "4",
            "5",
            "5a",
            "13a",
            "13b",
            "13",
            "12",
            "11b",
            "11a",
            "11",
            "10",
            "9",
            "8",
            "7a",
            "7",
            "6",
            "5b",
        ),
    ),
    (
        "naphtho[2,1-b]pentahelicene",
        "C1=CC=2C=CC=3C=CC=4C=C5C(=CC4C3C2C=2C=CC=CC12)C=CC=1C=CC=CC15",
        (
            "1",
            "2",
            "2a",
            "3",
            "4",
            "4a",
            "5",
            "6",
            "6a",
            "7",
            "7a",
            "13a",
            "14",
            "14a",
            "14b",
            "14c",
            "14d",
            "15",
            "16",
            "17",
            "18",
            "18a",
            "13",
            "12",
            "11a",
            "11",
            "10",
            "9",
            "8",
            "7b",
        ),
    ),
    (
        "dibenzo[a,c]phenazine",
        "C1=CC=CC2=C1C1=NC3=CC=CC=C3N=C1C1=C2C=CC=C1",
        (
            "1",
            "2",
            "3",
            "4",
            "4a",
            "14b",
            "14a",
            "14",
            "13a",
            "13",
            "12",
            "11",
            "10",
            "9a",
            "9",
            "8b",
            "8a",
            "4b",
            "5",
            "6",
            "7",
            "8",
        ),
    ),
    (
        "7H-dibenzocarbazole",
        "C1=CC=CC=2C1=C1C=3C=CC=CC3N=C1C=1C2C=CCC1",
        (
            "1",
            "2",
            "3",
            "4",
            "4a",
            "13c",
            "13b",
            "13a",
            "13",
            "12",
            "11",
            "10",
            "9a",
            "9",
            "8b",
            "8a",
            "4b",
            "5",
            "6",
            "7",
            "8",
        ),
    ),
    (
        "naphtho[2,1-b]benzofuran",
        "C1=CC=CC=2C=CC=3OC4=C(C3C12)C=CC=C4",
        (
            "1",
            "2",
            "3",
            "4",
            "4a",
            "5",
            "6",
            "6a",
            "7",
            "7a",
            "11a",
            "11b",
            "11c",
            "11",
            "10",
            "9",
            "8",
        ),
    ),
    (
        "dibenzo[c,g]carbazole",
        "C1=CC=CC=2C=CC=3NC=4C=CC5=C(C4C3C21)C=CC=C5",
        (
            "1",
            "2",
            "3",
            "4",
            "4a",
            "5",
            "6",
            "6a",
            "7",
            "7a",
            "8",
            "9",
            "9a",
            "13a",
            "13b",
            "13c",
            "13d",
            "13",
            "12",
            "11",
            "10",
        ),
    ),
    (
        "benzo[c][1]benzothiepine",
        "C1=CC=CC2=C1C=C1C(=CS2)C=CC=C1",
        (
            "1",
            "2",
            "3",
            "4",
            "4a",
            "11a",
            "11",
            "10a",
            "6a",
            "6",
            "5",
            "7",
            "8",
            "9",
            "10",
        ),
    ),
    (
        "phenanthro[4,5-bcd]thiophene",
        "C1=CC=C2C=3C4=C(S2)C=CC=C4C=CC13",
        (
            "1",
            "2",
            "3",
            "3a",
            "9c",
            "9b",
            "4a",
            "4",
            "5",
            "6",
            "7",
            "7a",
            "8",
            "9",
            "9a",
        ),
    ),
    (
        "furat",
        "O1C(=CC=C1)C(=O)[O-]",
        ("1", "2", "3", "4", "5", "", "", ""),
    ),
    (
        "furoic acid",
        "O1C(=CC=C1)C(=O)O",
        ("1", "2", "3", "4", "5", "", "", ""),
    ),
    (
        "furan-3-carboxylate",
        "O1C=C(C=C1)C(=O)[O-]",
        ("1", "2", "3", "4", "5", "", "", ""),
    ),
    (
        "thenoate",
        "C1(=CC=CS1)C(=O)[O-]",
        ("2", "3", "4", "5", "1", "", "", ""),
    ),
    (
        "thiophene-2-carboxylic acid",
        "S1C(=CC=C1)C(=O)O",
        ("1", "2", "3", "4", "5", "", "", ""),
    ),
    (
        "thiophene-3-carboxylate",
        "S1C=C(C=C1)C(=O)[O-]",
        ("1", "2", "3", "4", "5", "", "", ""),
    ),
    (
        "picolinate",
        "N1=C(C=CC=C1)C(=O)[O-]",
        ("1", "2", "3", "4", "5", "6", "", "", ""),
    ),
    (
        "picolinic acid",
        "N1=C(C=CC=C1)C(=O)O",
        ("1", "2", "3", "4", "5", "6", "", "", ""),
    ),
    (
        "nicotinate",
        "C(C1=CN=CC=C1)(=O)[O-]",
        ("", "3", "2", "1", "6", "5", "4", "", ""),
    ),
    (
        "nicotinic acid",
        "C(C1=CN=CC=C1)(=O)O",
        ("", "3", "2", "1", "6", "5", "4", "", ""),
    ),
    (
        "isonicotinate",
        "C(C1=CC=NC=C1)(=O)[O-]",
        ("", "4", "3", "2", "1", "6", "5", "", ""),
    ),
    (
        "isonicotinic acid",
        "C(C1=CC=NC=C1)(=O)O",
        ("", "4", "3", "2", "1", "6", "5", "", ""),
    ),
    (
        "picolinamide",
        "N1=C(C=CC=C1)C(=O)N",
        ("1", "2", "3", "4", "5", "6", "", "", ""),
    ),
    (
        "nicotinamide",
        "C(C1=CN=CC=C1)(=O)N",
        ("", "3", "2", "1", "6", "5", "4", "", ""),
    ),
    (
        "isonicotinamide",
        "C(C1=CC=NC=C1)(=O)N",
        ("", "4", "3", "2", "1", "6", "5", "", ""),
    ),
    (
        "thiazole-4-carboxamide",
        "S1C=NC(=C1)C(=O)N",
        ("1", "2", "3", "4", "5", "", "", ""),
    ),
    (
        "thiazole-5-carboxamide",
        "S1C=NC=C1C(=O)N",
        ("1", "2", "3", "4", "5", "", "", ""),
    ),
    (
        "benzo[c]phenanthrene",
        "C1=CC=CC=2C=CC=3C=CC=4C=CC=CC4C3C21",
        ("1", "2", "3", "4", "4a", "5", "6", "6a", "7", "8", "8a", "9", "10", "11", "12", "12a", "12b", "12c"),
    ),
    (
        "benzo[c]fluorene",
        "C1=CC=CC=2C=CC=3CC=4C=CC=CC4C3C21",
        ("1", "2", "3", "4", "4a", "5", "6", "6a", "7", "7a", "8", "9", "10", "11", "11a", "11b", "11c"),
    ),
    (
        "benzo[a]carbazole",
        "C1=CC=CC=2C1=C1NC3=CC=CC=C3C1=CC2",
        ("1", "2", "3", "4", "4a", "11b", "11a", "11", "10a", "10", "9", "8", "7", "6b", "6a", "6", "5"),
    ),
    (
        "pyrido[4,3-b]carbazole",
        "C1=NC=CC2=CC=3NC=4C=CC=CC4C3C=C21",
        ("1", "2", "3", "4", "4a", "5", "5a", "6", "6a", "7", "8", "9", "10", "10a", "10b", "11", "11a"),
    ),
    (
        "phenothiazine",
        "C1=CC=CC=2SC3=CC=CC=C3NC12",
        ("1", "2", "3", "4", "4a", "5", "5a", "6", "7", "8", "9", "9a", "10", "10a"),
    ),
    (
        "phenanthro[9,10-d]imidazole",
        "N1=CNC2=C1C1=CC=CC=C1C=1C=CC=CC12",
        ("1", "2", "3", "3a", "11b", "11a", "11", "10", "9", "8", "7b", "7a", "7", "6", "5", "4", "3b"),
    ),
    (
        "benzo[c]carbazole",
        "C1=CC=CC=2C=CC=3NC=4C=CC=CC4C3C21",
        ("1", "2", "3", "4", "4a", "5", "6", "6a", "7", "7a", "8", "9", "10", "11", "11a", "11b", "11c"),
    ),
    (
        "benzo[b]benzothieno[3,2-e]benzofuran",
        "C1=CC=CC=2OC3=C(C21)C2=C(C=C3)SC3=C2C=CC=C3",
        ("1", "2", "3", "4", "4a", "5", "5a", "12c", "12d", "12b", "7a", "7", "6", "8", "8a", "12a", "12", "11", "10", "9"),
    ),
    (
        "benzofuro[3,2-e]indeno[1,2-b]indole",
        "C1=CC=CC2=C1C1=C3C=4C(=NC3=CC=C1O2)C=2C=CC=CC2C4",
        ("1", "2", "3", "4", "4a", "5", "5a", "6", "7", "7a", "8", "8a", "8b", "9", "10", "11", "12", "12a", "13", "13a", "13b", "13c", "13d"),
    ),
    (
        "benzo[a]chrysene",
        "C1=CC=CC=2C1=C1C=CC3=C4C=CC=CC4=CC=C3C1=CC2",
        ("1", "2", "3", "4", "4a", "14b", "14a", "14", "13", "12b", "12a", "12", "11", "10", "9", "8a", "8", "7", "6b", "6a", "6", "5"),
    ),
    (
        "tribenzo[a,f,k]trindene",
        "C1=CC=CC2C1=CC1=C3C4=C(C=C3C3=C5C(=CC3=C21)C=CC=C5)C=CC=C4",
        ("1", "2", "3", "4", "4a", "15a", "15", "14c", "14b", "14a", "10a", "10", "9c", "9b", "9a", "5a", "5", "4c", "4b", "6", "7", "8", "9", "11", "12", "13", "14"),
    ),
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
        "1,4-diazepin-5-one",
        "O=c1ccnccn1",
        ("", "5", "6", "7", "1", "2", "3", "4"),
    ),
    (
        "1,4-diazepinone",
        "O=c1cncccn1",
        ("", "2", "3", "4", "5", "6", "7", "1"),
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


# Exact retained biochemical parents generated from OPSIN-validated standard
# nucleoside/nucleotide names. These are intentionally exact-only templates:
# they should not act as substructure scaffolds for arbitrary molecules.
EXACT_BIOCHEMICAL_PARENT_TEMPLATES = (
    ("deoxycytidine", "[C@@H]1(C[C@H](O)[C@@H](CO)O1)N1C(=O)N=C(N)C=C1"),
    (
        "cytidin diphosphate",
        "P(O)(=O)(OP(=O)(O)O)OC[C@@H]1[C@H]([C@H]([C@@H](O1)N1C(=O)N=C(N)C=C1)O)O",
    ),
    (
        "2'-azido-2'-deoxy-uridine",
        "N(=[N+]=[N-])[C@H]1[C@@H](O[C@@H]([C@H]1O)CO)N1C(=O)NC(=O)C=C1",
    ),
    (
        "2,6-dichloro-adenosine",
        "ClC1=NC(C2=NCN([C@H]3[C@H](O)[C@H](O)[C@@H](CO)O3)C2=N1)(N)Cl",
    ),
    ("N6-methyl-8-azaadenosine", "CNC=1C=2N=NN([C@H]3[C@H](O)[C@H](O)[C@@H](CO)O3)C2N=CN1"),
    (
        "xanthosine 3'-diphosphate",
        "P(O)(=O)(OP(=O)(O)O)O[C@H]1[C@H]([C@@H](O[C@@H]1CO)N1C=NC=2C(=O)NC(=O)NC12)O",
    ),
    (
        "adenosine 5'-pentaphosphate",
        "P(O)(=O)(OP(=O)(O)OP(=O)(O)OP(=O)(O)OP(=O)(O)O)OC[C@@H]1[C@H]([C@H]([C@@H](O1)N1C=NC=2C(N)=NC=NC12)O)O",
    ),
    (
        "5-methylcytidine diphosphate",
        "P(O)(=O)(OP(=O)(O)O)OC[C@@H]1[C@H]([C@H]([C@@H](O1)N1C(=O)N=C(N)C(=C1)C)O)O",
    ),
    (
        "6-fluoro-uridine diphosphate",
        "P(O)(=O)(OP(=O)(O)O)OC[C@@H]1[C@H]([C@H]([C@@H](O1)N1C(=O)NC(=O)C=C1F)O)O",
    ),
    (
        "3'-thioguanosine triphosphate",
        "P(O)(=O)(OP(=O)(O)OP(=O)(O)O)OC[C@@H]1[C@H]([C@H]([C@@H](O1)N1C=NC=2C(=O)NC(N)=NC12)O)S",
    ),
    (
        "5-Fluorocytidine triphosphate",
        "P(O)(=O)(OP(=O)(O)OP(=O)(O)O)OC[C@@H]1[C@H]([C@H]([C@@H](O1)N1C(=O)N=C(N)C(=C1)F)O)O",
    ),
    (
        "2-thio-cytidine triphosphate",
        "P(O)(=O)(OP(=O)(O)OP(=O)(O)O)OC[C@@H]1[C@H]([C@H]([C@@H](O1)N1C(=S)N=C(N)C=C1)O)O",
    ),
    (
        "7-methylguanosine triphosphate",
        "P(O)(=O)(OP(=O)(O)OP(=O)(O)O)OC[C@@H]1[C@H]([C@H]([C@@H](O1)N1C=[N+](C=2C(=O)NC(N)=NC12)C)O)O",
    ),
    (
        "8-aza-guanosine triphosphate",
        "P(O)(=O)(OP(=O)(O)OP(=O)(O)O)OC[C@@H]1[C@H]([C@H]([C@@H](O1)N1N=NC=2C(=O)NC(N)=NC12)O)O",
    ),
    (
        "2-thio-pseudouridine triphosphate",
        "P(O)(=O)(OP(=O)(O)OP(=O)(O)O)OC[C@@H]1[C@H]([C@H]([C@@H](O1)C1=CNC(=S)NC1=O)O)O",
    ),
    (
        "4-methyl-cytidine triphosphate",
        "P(O)(=O)(OP(=O)(O)OP(=O)(O)O)OC[C@@H]1[C@H]([C@H]([C@@H](O1)N1C(=O)NC(N)(C=C1)C)O)O",
    ),
    (
        "5-carboxy-uridine triphosphate",
        "P(O)(=O)(OP(=O)(O)OP(=O)(O)O)OC[C@@H]1[C@H]([C@H]([C@@H](O1)N1C(=O)NC(=O)C(=C1)C(=O)O)O)O",
    ),
    (
        "8-azido-adenosine triphosphate",
        "P(O)(=O)(OP(=O)(O)OP(=O)(O)O)OC[C@@H]1[C@H]([C@H]([C@@H](O1)N1C(=NC=2C(N)=NC=NC12)N=[N+]=[N-])O)O",
    ),
    ("O-propargyl-uridine 3'-phosphate", "P(=O)(O)(O)O[C@H]1[C@H]([C@@H](O[C@@H]1CO)N1C(=O)NC(=O)C=C1)OCC#C"),
    (
        "2'-azido-deoxyuridine triphosphate",
        "P(O)(=O)(OP(=O)(O)OP(=O)(O)O)OC[C@@H]1[C@H]([C@H]([C@@H](O1)N1C(=O)NC(=O)C=C1)N=[N+]=[N-])O",
    ),
    (
        "2'-Fluorothymidine-5'-Triphosphate",
        "P(O)(=O)(OP(=O)(O)OP(=O)(O)O)OC[C@@H]1[C@H](C([C@@H](O1)N1C(=O)NC(=O)C(C)=C1)F)O",
    ),
    ("2-mercapto-inosine phosphate", "P(=O)(O)(O)OC[C@@H]1[C@H]([C@H]([C@@H](O1)N1C=NC=2C(O)=NC(=NC12)S)O)O"),
    (
        "dimethyl-guanosine-5'-triphosphate",
        "P(O)(=O)(OP(=O)(O)OP(=O)(O)O)OC[C@@H]1[C@H]([C@H]([C@@H](O1)N1C=NC=2C(=O)NC(N(C)C)=NC12)O)O",
    ),
    (
        "N2-isobutyl-guanosine triphosphate",
        "P(O)(=O)(OP(=O)(O)OP(=O)(O)O)OC[C@@H]1[C@H]([C@H]([C@@H](O1)N1C=NC=2C(=O)NC(NCC(C)C)=NC12)O)O",
    ),
    (
        "2-chloro-deoxyadenosine triphosphate",
        "P(O)(=O)(OP(=O)(O)OP(=O)(O)O)OC[C@@H]1[C@H](C[C@@H](O1)N1C=NC=2C(N)=NC(=NC12)Cl)O",
    ),
    (
        "5-ethynyl-deoxyuridine triphosphate",
        "P(O)(=O)(OP(=O)(O)OP(=O)(O)O)OC[C@@H]1[C@H](C[C@@H](O1)N1C(=O)NC(=O)C(=C1)C#C)O",
    ),
    (
        "5-formyl-deoxycytidine triphosphate",
        "P(O)(=O)(OP(=O)(O)OP(=O)(O)O)OC[C@@H]1[C@H](C[C@@H](O1)N1C(=O)N=C(N)C(=C1)C=O)O",
    ),
    (
        "2'-azido-2'-deoxy-cytidine triphosphate",
        "P(O)(=O)(OP(=O)(O)OP(=O)(O)O)OC[C@@H]1[C@H]([C@H]([C@@H](O1)N1C(=O)N=C(N)C=C1)N=[N+]=[N-])O",
    ),
    (
        "N4-methyl-deoxycytidine triphosphate",
        "P(O)(=O)(OP(=O)(O)OP(=O)(O)O)OC[C@@H]1[C@H](C[C@@H](O1)N1C(=O)N=C(NC)C=C1)O",
    ),
    (
        "propargyl-deoxycytidine triphosphate",
        "P(O)(=O)(OP(=O)(O)OP(=O)(O)O)OC[C@@H]1[C@H](C[C@@](O1)(N1C(=O)N=C(N)C=C1)CC#C)O",
    ),
    (
        "N6-methyl-deoxyadenosine 3',5'-diphosphate",
        "P(=O)(O)(O)O[C@H]1C[C@@H](O[C@@H]1COP(=O)(O)O)N1C=NC=2C(NC)=NC=NC12",
    ),
    ("ribosyl pyrophosphate", "O(P([O-])(=O)OP(=O)([O-])[O-])C1[C@H](O)[C@H](O)[C@H](O1)CO"),
    (
        "O-ribosyl-guanosine phosphate",
        "P(=O)(O)(O)OC[C@@H]1[C@H]([C@H]([C@@H](O1)N1C=NC=2C(=O)NC(N)=NC12)OC1[C@H](O)[C@H](O)[C@H](O1)CO)O",
    ),
    (
        "mannosyl-guanosine",
        "C1([C@@H](O)[C@@H](O)[C@H](O)[C@H](O1)CO)[C@@]1([C@H](O)[C@H](O)[C@@H](CO)O1)N1C=NC=2C(=O)NC(N)=NC12",
    ),
    (
        "beta-D-galactosyl-guanosine",
        "[C@@H]1([C@H](O)[C@@H](O)[C@@H](O)[C@H](O1)CO)[C@@]1([C@H](O)[C@H](O)[C@@H](CO)O1)N1C=NC=2C(=O)NC(N)=NC12",
    ),
    ("alpha-D-galactosamine 1-phosphate", "P(=O)(O)(O)O[C@@H]1[C@H](N)[C@@H](O)[C@@H](O)[C@H](O1)CO"),
    ("6-phospho-fructose", "P(=O)(O)(O)OC[C@H]([C@H]([C@@H](C(CO)=O)O)O)O"),
    ("fructosyl-glutamine", "OCC1([C@@H](O)[C@H](O)[C@H](O1)CO)N[C@@H](CCC(N)=O)C(=O)O"),
    ("fructosyl-galactose", "OCC1([C@@H](O)[C@H](O)[C@H](O1)CO)C(=O)[C@H](O)[C@@H](O)[C@@H](O)[C@H](O)CO"),
    (
        "fructosyl-curcumin",
        "OCC1([C@@H](O)[C@H](O)[C@H](O1)CO)COC1=CC(=CC=C1O)\\C=C\\C(=O)CC(=O)\\C=C\\C1=CC=C(O)C(OC)=C1",
    ),
    (
        "pyranyl-glucosyl-L-ascorbic acid",
        "O1C(C=CC=C1)[C@]([C@@]1(C(=C(C(=O)O1)O)O)C1[C@H](O)[C@@H](O)[C@H](O)[C@H](O1)CO)(O)CO",
    ),
    (
        "tetraisobutyl-ascorbate",
        "C(C(C)C)C([C@@]([C@@]1(C(=C(C(=O)O1)O)[O-])CC(C)C)(O)CC(C)C)(O)CC(C)C",
    ),
    (
        "tetrabenzyl-ascorbate",
        "C(C1=CC=CC=C1)C([C@@]([C@@]1(C(=C(C(=O)O1)O)[O-])CC1=CC=CC=C1)(O)CC1=CC=CC=C1)(O)CC1=CC=CC=C1",
    ),
    (
        "tetra-cyclopentyl-ascorbate",
        "C1(CCCC1)C([C@@]([C@@]1(C(=C(C(=O)O1)O)[O-])C1CCCC1)(O)C1CCCC1)(O)C1CCCC1",
    ),
    ("diisobutyl-ascorbate", "C(C(C)C)C([C@@H]([C@@H]1C(=C(C(=O)O1)O)[O-])O)(O)CC(C)C"),
    (
        "tetraoctyl-ascorbate",
        "C(CCCCCCC)C([C@@]([C@@]1(C(=C(C(=O)O1)O)[O-])CCCCCCCC)(O)CCCCCCCC)(O)CCCCCCCC",
    ),
    ("diethyl-ascorbate", "C(C)C([C@@H]([C@@H]1C(=C(C(=O)O1)O)[O-])O)(O)CC"),
    ("tetra-n-butyl-ascorbate", "C(CCC)C([C@@]([C@@]1(C(=C(C(=O)O1)O)[O-])CCCC)(O)CCCC)(O)CCCC"),
    (
        "3'-O-Fucosyllactose",
        "C1([C@@H](O)[C@H](O)[C@H](O)[C@@H](O1)C)O[C@@H]1[C@H]([C@H](O[C@H]2[C@@H]([C@H](C(O)O[C@@H]2CO)O)O)O[C@@H]([C@@H]1O)CO)O",
    ),
    (
        "5'-O-alpha-L-fucosyl-2'-deoxyuridine",
        "[C@@H]1([C@@H](O)[C@H](O)[C@H](O)[C@@H](O1)C)OC[C@@H]1[C@H](C[C@@H](O1)N1C(=O)NC(=O)C=C1)O",
    ),
    (
        "3-O-alpha-L-Rhamnopyranosyl-D-galactose",
        "[C@@H]1([C@H](O)[C@H](O)[C@@H](O)[C@@H](O1)C)O[C@H]([C@H](C=O)O)[C@@H](O)[C@H](O)CO",
    ),
    (
        "alpha-rhamnosyl-(1->6)glucose",
        "[C@@H]1([C@H](O)[C@H](O)[C@@H](O)[C@@H](O1)C)OC[C@H]([C@H]([C@@H]([C@H](C=O)O)O)O)O",
    ),
    (
        "4-O-beta-D-glucopyranosyl-D-fructose",
        "[C@@H]1([C@H](O)[C@@H](O)[C@H](O)[C@H](O1)CO)O[C@@H]([C@@H](C(CO)=O)O)[C@H](O)CO",
    ),
    (
        "2-alpha-D-glucopyranosyl-alpha-D-glucopyranose",
        "[C@H]1([C@H](O)[C@@H](O)[C@H](O)[C@H](O1)CO)[C@@]1([C@@H](O)O[C@@H]([C@H]([C@@H]1O)O)CO)O",
    ),
    (
        "3-O-(beta-D-galactopyranosyl)-D-galactopyranose",
        "[C@@H]1([C@H](O)[C@@H](O)[C@@H](O)[C@H](O1)CO)O[C@@H]1[C@H](C(O)O[C@@H]([C@@H]1O)CO)O",
    ),
    (
        "beta-D-glucopyranosyl-(1-4)-alpha-L-rhamnopyranose",
        "[C@@H]1([C@H](O)[C@@H](O)[C@H](O)[C@H](O1)CO)O[C@@H]1[C@H]([C@H]([C@H](O)O[C@H]1C)O)O",
    ),
    (
        "D-mannosyl-methyluridine",
        "C1([C@@H](O)[C@@H](O)[C@H](O)[C@H](O1)CO)[C@@]1([C@@](O[C@@H]([C@H]1O)CO)(N1C(=O)NC(=O)C=C1)C)O",
    ),
    (
        "L-threofuranosyl-cytidine",
        "C1([C@H](O)[C@@H](O)CO1)[C@@]1([C@H](O)[C@H](O)[C@@H](CO)O1)N1C(=O)N=C(N)C=C1",
    ),
    (
        "4-O-beta-D-glucosyl-D-gluconic acid",
        "[C@@H]1([C@H](O)[C@@H](O)[C@H](O)[C@H](O1)CO)O[C@@H]([C@@H]([C@H](C(=O)O)O)O)[C@H](O)CO",
    ),
    (
        "2-acetamido-2-deoxy-alpha-D-mannopyranosyl phosphate",
        "P(=O)(O[C@@H]1[C@H]([C@@H](O)[C@H](O)[C@H](O1)CO)NC(C)=O)([O-])[O-]",
    ),
    ("azidoacetyl-galactosamine", "N(=[N+]=[N-])CC(=O)C1(O)[C@H](N)[C@@H](O)[C@@H](O)[C@H](O1)CO"),
    ("Mannosyl diethyldithiocarbamate", "C(C)N(C(SC1[C@@H](O)[C@@H](O)[C@H](O)[C@H](O1)CO)=S)CC"),
    (
        "Tetrabenzyl-glucose",
        "C(C1=CC=CC=C1)[C@@]([C@@]([C@](C(=O)CC1=CC=CC=C1)(O)CC1=CC=CC=C1)(O)CC1=CC=CC=C1)(O)[C@H](O)CO",
    ),
    ("4-Tert-butoxyphenyl-glucose", "C(C)(C)(C)OC1=CC=C(C=C1)C(=O)[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO"),
    ("2-acetamido-L-glucose", "C(C)(=O)N[C@](C=O)(O)[C@H](O)[C@@H](O)[C@@H](O)CO"),
    ("Allofuranose", "OC1[C@H](O)[C@H](O)[C@H](O1)[C@H](O)CO"),
    ("beta-L-gulofuranose", "O[C@@H]1[C@@H](O)[C@@H](O)[C@H](O1)[C@@H](O)CO"),
    ("beta-D-gulofuranose", "O[C@H]1[C@H](O)[C@H](O)[C@@H](O1)[C@H](O)CO"),
    ("Mannofuranose", "OC1[C@@H](O)[C@@H](O)[C@H](O1)[C@H](O)CO"),
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
    return tuple(templates) + build_high_impact_fused_parent_templates(Chem)


def build_high_impact_fused_parent_templates(Chem) -> tuple[dict, ...]:
    return _build_json_scaffold_templates(
        Chem,
        _read_json_records("high_impact_fused_parent_templates.json"),
        source_family="static-retained",
        default_source_file="high_impact_fused_parent_templates.json",
    )


def build_exact_biochemical_parent_templates(Chem) -> tuple[dict, ...]:
    templates = []
    for name, smiles in EXACT_BIOCHEMICAL_PARENT_TEMPLATES:
        mol = _raw_template_mol(Chem, smiles)
        if mol is None:
            continue
        templates.append(
            {
                "name": name,
                "mol": mol,
                "locants": {idx: "" for idx in range(mol.GetNumAtoms())},
                "source_family": "static-exact-retained",
                "source_file": "",
                "source_subtype": "biochemical",
                "stereochemical": _has_stereochemical_smiles(smiles),
            }
        )
    return tuple(templates)


def build_exact_unlocanted_resource_templates(Chem) -> tuple[dict, ...]:
    """Exact-only retained parents for OPSIN tokens without locant labels.

    These resources are useful as whole-molecule retained names, but their
    missing locant maps make them unsafe as generalized scaffold parents.
    """

    templates = []
    seen: set[tuple[str, str]] = set()
    operation_attrs = {"addGroup", "addBond", "addHeteroAtom", "additionalValue"}
    for record in iter_opsin_token_records(OPSIN_RESOURCE_FILES):
        token = record.token
        if unsupported_token_attributes(token, SUPPORTED_TOKEN_ATTRIBUTES):
            continue
        if token.attrib.get("labels"):
            continue
        if any(attr in token.attrib for attr in operation_attrs):
            continue
        smiles = token.attrib.get("value", "")
        if not smiles or "." in smiles:
            continue
        name = preferred_token_name(token.text or "", PARENT_NAME_NORMALIZATIONS)
        if not name:
            continue
        mol = _raw_template_mol(Chem, smiles)
        if mol is None or mol.GetNumAtoms() > 64:
            continue
        key = (name, Chem.MolToSmiles(mol, canonical=True))
        if key in seen:
            continue
        seen.add(key)
        templates.append(
            {
                "name": name,
                "mol": mol,
                "locants": {idx: str(idx + 1) for idx in range(mol.GetNumAtoms())},
                "source_family": "opsin-exact-unlocanted",
                "source_file": record.file_name,
                "source_subtype": record.context.sub_type,
                "stereochemical": _has_stereochemical_smiles(smiles),
                "exact_only": True,
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


_PHENE_RETAINED_RING_RECORDS = (
    ('pentaphene', 'C1=CC=CC2=CC3=CC=C4C=C5C=CC=CC5=CC4=C3C=C12', ('1', '2', '3', '4', '4a', '5', '5a', '6', '7', '7a', '8', '8a', '9', '10', '11', '12', '12a', '13', '13a', '13b', '14', '14a')),
    ('hexaphene', 'C1=CC=CC2=CC3=CC=C4C=C5C=C6C=CC=CC6=CC5=CC4=C3C=C12', ('1', '2', '3', '4', '4a', '5', '5a', '6', '7', '7a', '8', '8a', '9', '9a', '10', '11', '12', '13', '13a', '14', '14a', '15', '15a', '15b', '16', '16a')),
    ('heptaphene', 'C1=CC=CC2=CC3=CC4=CC=C5C=C6C=C7C=CC=CC7=CC6=CC5=C4C=C3C=C12', ('1', '2', '3', '4', '4a', '5', '5a', '6', '6a', '7', '8', '8a', '9', '9a', '10', '10a', '11', '12', '13', '14', '14a', '15', '15a', '16', '16a', '16b', '17', '17a', '18', '18a')),
    ('octaphene', 'C1=CC=CC2=CC3=CC4=CC=C5C=C6C=C7C=C8C=CC=CC8=CC7=CC6=CC5=C4C=C3C=C12', ('1', '2', '3', '4', '4a', '5', '5a', '6', '6a', '7', '8', '8a', '9', '9a', '10', '10a', '11', '11a', '12', '13', '14', '15', '15a', '16', '16a', '17', '17a', '18', '18a', '18b', '19', '19a', '20', '20a')),
    ('nonaphene', 'C1=CC=CC2=CC3=CC4=CC5=CC=C6C=C7C=C8C=C9C=CC=CC9=CC8=CC7=CC6=C5C=C4C=C3C=C12', ('1', '2', '3', '4', '4a', '5', '5a', '6', '6a', '7', '7a', '8', '9', '9a', '10', '10a', '11', '11a', '12', '12a', '13', '14', '15', '16', '16a', '17', '17a', '18', '18a', '19', '19a', '19b', '20', '20a', '21', '21a', '22', '22a')),
    ('decaphene', 'C1=CC=CC2=CC3=CC4=CC5=CC=C6C=C7C=C8C=C9C=C%10C=CC=CC%10=CC9=CC8=CC7=CC6=C5C=C4C=C3C=C12', ('1', '2', '3', '4', '4a', '5', '5a', '6', '6a', '7', '7a', '8', '9', '9a', '10', '10a', '11', '11a', '12', '12a', '13', '13a', '14', '15', '16', '17', '17a', '18', '18a', '19', '19a', '20', '20a', '21', '21a', '21b', '22', '22a', '23', '23a', '24', '24a')),
    ('undecaphene', 'C1=CC=CC2=CC3=CC4=CC5=CC6=CC=C7C=C8C=C9C=C%10C=C%11C=CC=CC%11=CC%10=CC9=CC8=CC7=C6C=C5C=C4C=C3C=C12', ('1', '2', '3', '4', '4a', '5', '5a', '6', '6a', '7', '7a', '8', '8a', '9', '10', '10a', '11', '11a', '12', '12a', '13', '13a', '14', '14a', '15', '16', '17', '18', '18a', '19', '19a', '20', '20a', '21', '21a', '22', '22a', '22b', '23', '23a', '24', '24a', '25', '25a', '26', '26a')),
    ('dodecaphene', 'C1=CC=CC2=CC3=CC4=CC5=CC6=CC=C7C=C8C=C9C=C%10C=C%11C=C%12C=CC=CC%12=CC%11=CC%10=CC9=CC8=CC7=C6C=C5C=C4C=C3C=C12', ('1', '2', '3', '4', '4a', '5', '5a', '6', '6a', '7', '7a', '8', '8a', '9', '10', '10a', '11', '11a', '12', '12a', '13', '13a', '14', '14a', '15', '15a', '16', '17', '18', '19', '19a', '20', '20a', '21', '21a', '22', '22a', '23', '23a', '24', '24a', '24b', '25', '25a', '26', '26a', '27', '27a', '28', '28a')),
)


def _opsin_derived_phene_records() -> tuple[dict, ...]:
    return tuple(
        {
            "name": name,
            "smiles": smiles,
            "locants": {str(idx): locant for idx, locant in enumerate(locants)},
            "source_file": "opsin-derived-phene-templates",
            "source_subtype": "opsin-cml-located-retained-ring",
        }
        for name, smiles, locants in _PHENE_RETAINED_RING_RECORDS
    )


def build_opsin_retained_ring_csv_templates(Chem, max_atoms: int = 54) -> tuple[dict, ...]:
    locanted_records = (
        _read_json_records("opsin_retained_mono_fused_ring_templates.json")
        + _opsin_derived_phene_records()
    )
    if locanted_records:
        smiles_by_name = {
            str(record.get("name", "")): str(record.get("smiles", "")) for record in locanted_records
        }
        return tuple(
            _annotate_opsin_retained_ring_template(Chem, template, smiles_by_name.get(template["name"], ""))
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
        template = _annotate_opsin_retained_ring_template(Chem, template, smiles)
        dedupe_key = (template["name"], Chem.MolToSmiles(mol, canonical=True))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        templates.append(template)
    return tuple(templates)


_OPSIN_RETAINED_RING_GENERAL_PARENT_EXCLUSIONS = {
    "(2S)-2-[(5R,6R,7R,14S)-N-cyclopropylmethyl-4,5-epoxy-6,14-ethano-3-hydroxy-6-methoxymorphinan-7-yl]-3,3-dimethylpentan-2-ol",
    "benzosemiquinon",
    "benzosemiquinone",
    "uranocen",
    "uranocene",
}


def _annotate_opsin_retained_ring_template(Chem, template: dict, smiles: str) -> dict:
    mol = template["mol"]
    ring_info = mol.GetRingInfo()
    ring_atoms = {idx for ring in ring_info.AtomRings() for idx in ring}
    flags = set(template.get("flags", ()))
    if ring_info.NumRings() == 1:
        scaffold_class = "mono-ring"
        flags.add("mono-ring")
    else:
        scaffold_class = "fused-or-polycyclic-ring"
        flags.add("multi-ring")
    if ring_atoms and len(ring_atoms) < mol.GetNumAtoms():
        flags.add("exocyclic-containing")
    if any(atom.GetFormalCharge() for atom in mol.GetAtoms()):
        flags.add("charged")
    if template.get("stereochemical") or _has_stereochemical_smiles(smiles):
        flags.add("stereochemical")
    if any(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()):
        flags.add("radical")
    if any(atom.GetSymbol() in {"U", "Th", "Pu", "Np"} for atom in mol.GetAtoms()):
        flags.add("organometallic")
    general_parent_safe = template["name"] not in _OPSIN_RETAINED_RING_GENERAL_PARENT_EXCLUSIONS
    if not general_parent_safe:
        flags.add("general-parent-unsupported")
    annotated = dict(template)
    annotated.update(
        {
            "flags": tuple(sorted(flags)),
            "scaffold_class": scaffold_class,
            "general_parent_safe": general_parent_safe,
        }
    )
    return annotated


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
