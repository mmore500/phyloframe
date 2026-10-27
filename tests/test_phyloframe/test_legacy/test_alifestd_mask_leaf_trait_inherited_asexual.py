import numpy as np
import pandas as pd
import pytest

from phyloframe.legacy import (
    alifestd_make_empty,
)
from phyloframe.legacy import (
    alifestd_mask_leaf_trait_inherited_asexual as alifestd_mask_leaf_trait_inherited_asexual_,
)

from ._impl import enforce_dtype_stability_pandas

alifestd_mask_leaf_trait_inherited_asexual = enforce_dtype_stability_pandas(
    alifestd_mask_leaf_trait_inherited_asexual_
)


def _make_tree() -> pd.DataFrame:
    #        0
    #     1     2
    #    3 4   5 6
    #   7 8
    return pd.DataFrame(
        {
            "id": range(9),
            "ancestor_id": [0, 0, 0, 1, 1, 2, 2, 3, 3],
            "origin_time": range(9),
        },
    )


def test_empty():
    res = alifestd_mask_leaf_trait_inherited_asexual(
        alifestd_make_empty(), trait_mask=np.array([], dtype=bool)
    )
    assert "alifestd_mask_leaf_trait_inherited_asexual" in res.columns
    assert len(res) == 0


@pytest.mark.parametrize("mutate", [False, True])
def test_simple(mutate: bool):
    phylogeny_df = _make_tree()
    original_df = phylogeny_df.copy()
    trait_mask = np.array([0, 0, 0, 0, 1, 1, 0, 1, 1], dtype=bool)
    res = alifestd_mask_leaf_trait_inherited_asexual(
        phylogeny_df, mutate=mutate, trait_mask=trait_mask
    )
    res.index = res["id"]
    # 4, 7, 8 share the trait with every leaf of their parent's clade;
    # 5's sibling 6 lacks it
    expected = [0, 0, 0, 0, 1, 0, 0, 1, 1]
    assert res.loc[
        range(9), "alifestd_mask_leaf_trait_inherited_asexual"
    ].tolist() == list(map(bool, expected))
    if not mutate:
        assert original_df.equals(phylogeny_df)


def test_inner_values_ignored():
    phylogeny_df = _make_tree()
    trait_mask = np.array([1, 1, 1, 1, 0, 1, 0, 1, 1], dtype=bool)
    res = alifestd_mask_leaf_trait_inherited_asexual(
        phylogeny_df, trait_mask=trait_mask, mark_as="m"
    )
    res.index = res["id"]
    assert res.loc[range(9), "m"].tolist() == list(
        map(bool, [0, 0, 0, 0, 0, 0, 0, 1, 1])
    )


def test_root_leaf():
    phylogeny_df = pd.DataFrame(
        {"id": [0, 1, 2], "ancestor_id": [0, 1, 1], "origin_time": [0, 0, 1]}
    )
    res = alifestd_mask_leaf_trait_inherited_asexual(
        phylogeny_df, trait_mask=np.array([True, False, True])
    )
    res.index = res["id"]
    assert not res.loc[0, "alifestd_mask_leaf_trait_inherited_asexual"]
    # 2 is the only leaf of its parent's clade
    assert res.loc[2, "alifestd_mask_leaf_trait_inherited_asexual"]


def test_unsorted_noncontiguous():
    phylogeny_df = _make_tree()
    trait_mask = np.array([0, 0, 0, 0, 1, 1, 0, 1, 1], dtype=bool)
    expected = alifestd_mask_leaf_trait_inherited_asexual(
        phylogeny_df, trait_mask=trait_mask
    ).set_index("id")["alifestd_mask_leaf_trait_inherited_asexual"]

    phylogeny_df["id"] = phylogeny_df["id"] * 10 + 3
    phylogeny_df["ancestor_id"] = phylogeny_df["ancestor_id"] * 10 + 3
    order = np.random.default_rng(1).permutation(9)
    res = alifestd_mask_leaf_trait_inherited_asexual(
        phylogeny_df.iloc[order].reset_index(drop=True),
        trait_mask=trait_mask[order],
    ).set_index("id")["alifestd_mask_leaf_trait_inherited_asexual"]
    assert [res[i * 10 + 3] for i in range(9)] == expected.tolist()
