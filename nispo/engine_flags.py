from __future__ import annotations

FAST_POLY_BRIDGE_ROUTE_PROP = "_nispo_fast_poly_bridge_route"
FAST_POLY_RETAINED_DEFER_CACHE_KEY = "fast_poly_deferred_to_retained"
FAST_POLY_RETAINED_DEFER_BYPASS_PROP = "_nispo_fast_poly_retained_defer_bypass"

FAST_POLY_RETAINED_DEFER_FRAGILE_TEMPLATE_KEYS = frozenset(
    {
        ("", "1,2,3,4-tetrahydroisoquinoline"),
        ("", "2-azabicyclo[2.2.2]octane"),
        ("", "2-oxabicyclo[2.1.1]hexane"),
        ("", "2H-benzotriazole"),
        ("", "3-azabicyclo[3.1.0]hexane"),
        ("arylGroups:ring:ring", "carbostyril"),
        ("arylGroups:ring:ring", "fluoren"),
        ("arylGroups:ring:ring", "indolin"),
        ("arylGroups:ring:ring", "purin"),
        ("generated-fused", "benzo[1,2-d]pyrazole"),
        ("generated-fused", "chromeno[4,3-d]pyrimidine"),
        ("generated-fused", "imidazo[4,5-b]pyridine"),
        ("generated-fused", "imidazo[4,5-d]pyrimidine"),
        ("generated-fused", "pyrido[1,6-a]imidazole"),
        ("generated-fused", "pyrido[3,4-a]benzene"),
        ("generated-fused", "spiro[cyclopropane-1,2'-pyrrolin]"),
        ("generated-fused", "thieno[2,3-b]pyridine"),
        ("opsin-retained-ring-cml", "quinoxalin"),
        ("simpleCyclicGroups:ring:simpleGroup", "tetrahydropterin"),
        ("simpleGroups:simpleGroup:simpleGroup", "benzocyclobutene"),
        ("static-retained", "2,3-dihydro-1,5-benzothiazepin-4(5H)-one"),
        ("static-retained", "3,4-dihydro-2H-1,4-benzoxazine"),
        ("static-retained", "4H-benzo[e][1,2,4]thiadiazine 1,1-dioxide"),
        ("static-retained", "imidazo[1,2-a]pyridine"),
        ("static-retained", "pterin"),
        ("static-retained", "pyrido[1,2-a]pyrimidin-4-one"),
        ("static-retained", "pyrimido[1,2-b]pyridazin-4-one"),
    }
)
