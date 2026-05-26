from __future__ import annotations

from nispo.parent.rule_helpers import _fused_parent_carbonyl_rank


def test_fused_carbonyl_rank_avoids_carboxyl_like_side_chains():
    assert _fused_parent_carbonyl_rank("3-hydroxy-20-oxo-pregnane") == 0
    assert _fused_parent_carbonyl_rank("23-ethyl-7,24-dihydroxy-24-oxo-cholane") == 1
    assert _fused_parent_carbonyl_rank("24-amino-24-oxo-23-(4-azabutyl)-cholane") == 1
