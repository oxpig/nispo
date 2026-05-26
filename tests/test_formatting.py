from __future__ import annotations

from nispo.formatting import unsaturated_parent


def test_mixed_en_yne_parent_does_not_insert_double_hyphen():
    assert unsaturated_parent("oct", [5], [1, 3]) == "oct-5-en-1,3-diyne"
