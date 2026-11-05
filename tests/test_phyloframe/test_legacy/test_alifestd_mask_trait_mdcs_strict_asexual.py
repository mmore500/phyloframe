import numpy as np
import pandas as pd
import pytest

from phyloframe.legacy import (
    alifestd_make_empty,
)
from phyloframe.legacy import (
    alifestd_mask_trait_mdcs_strict_asexual as alifestd_mask_trait_mdcs_strict_asexual_,
)
from phyloframe.legacy import (
    alifestd_mask_trait_singletons_asexual,
)

from ._impl import enforce_dtype_stability_pandas

alifestd_mask_trait_mdcs_strict_asexual = enforce_dtype_stability_pandas(
    alifestd_mask_trait_mdcs_strict_asexual_
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
    res = alifestd_mask_trait_mdcs_strict_asexual(
        alifestd_make_empty(), trait_mask=np.array([], dtype=bool)
    )
    assert "alifestd_mask_trait_mdcs_strict_asexual" in res.columns
    assert len(res) == 0


@pytest.mark.parametrize("mutate", [False, True])
def test_simple(mutate: bool):
    phylogeny_df = _make_tree()
    original_df = phylogeny_df.copy()
    trait_mask = np.array([0, 0, 0, 0, 1, 1, 0, 1, 1], dtype=bool)
    res = alifestd_mask_trait_mdcs_strict_asexual(
        phylogeny_df, mutate=mutate, trait_mask=trait_mask
    )
    res.index = res["id"]
    # leaves 4, 7, 8 make up clade 1, whose parent's clade includes leaf 6,
    # which lacks the trait; leaf 5's sibling 6 lacks it too
    expected = [0, 1, 0, 0, 0, 1, 0, 0, 0]
    assert res.loc[
        range(9), "alifestd_mask_trait_mdcs_strict_asexual"
    ].tolist() == list(map(bool, expected))
    if not mutate:
        assert original_df.equals(phylogeny_df)


def test_root_and_inner_values_ignored():
    phylogeny_df = _make_tree()
    # every leaf has the trait, so it predates the root
    res = alifestd_mask_trait_mdcs_strict_asexual(
        phylogeny_df,
        trait_mask=np.array([0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=bool),
        mark_as="m",
    )
    assert not res["m"].any()


def test_unifurcation_earliest_marked():
    #     0
    #   1   2
    #   3
    phylogeny_df = pd.DataFrame(
        {"id": [0, 1, 2, 3], "ancestor_id": [0, 0, 0, 1]}
    )
    res = alifestd_mask_trait_mdcs_strict_asexual(
        phylogeny_df, trait_mask=np.array([False, False, False, True])
    )
    assert res["alifestd_mask_trait_mdcs_strict_asexual"].tolist() == [
        False,
        True,
        False,
        False,
    ]


def test_matches_singletons_at_leaves():
    # random bifurcating tree, splitting a random leaf each step; trait
    # singletons bypass unifurcations, so compare without any
    rng = np.random.default_rng(1)
    ancestor_ids, leaves = [0], [0]
    for __ in range(300):
        parent = leaves.pop(rng.integers(len(leaves)))
        for __ in range(2):
            leaves.append(len(ancestor_ids))
            ancestor_ids.append(parent)
    phylogeny_df = pd.DataFrame(
        {"id": range(len(ancestor_ids)), "ancestor_id": ancestor_ids}
    )
    is_leaf = ~phylogeny_df["id"].isin(phylogeny_df["ancestor_id"])
    trait_mask = rng.random(len(phylogeny_df)) < 0.3
    mdcs = alifestd_mask_trait_mdcs_strict_asexual(
        phylogeny_df, trait_mask=trait_mask
    )["alifestd_mask_trait_mdcs_strict_asexual"]
    singletons = alifestd_mask_trait_singletons_asexual(
        phylogeny_df, trait_mask=trait_mask
    )["alifestd_mask_trait_singletons_asexual"]
    assert mdcs[is_leaf].tolist() == singletons[is_leaf].tolist()
    assert mdcs[is_leaf].any()
    # inner MDCs are clades sharing one gain
    assert mdcs[~is_leaf].any()


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pytest.param(
            pd.DataFrame({"id": [3, 13, 23], "ancestor_id": [3, 3, 3]}),
            id="noncontiguous",
        ),
        pytest.param(
            pd.DataFrame({"id": [0, 1, 2], "ancestor_id": [1, 1, 1]}),
            id="unsorted",
        ),
    ],
)
def test_not_working_format_raises(phylogeny_df: pd.DataFrame):
    with pytest.raises(NotImplementedError):
        alifestd_mask_trait_mdcs_strict_asexual(
            phylogeny_df, trait_mask=np.zeros(3, dtype=bool)
        )


def test_non_bool_trait_mask_raises():
    phylogeny_df = pd.DataFrame({"id": [0, 1, 2], "ancestor_id": [0, 0, 0]})
    with pytest.raises(NotImplementedError):
        alifestd_mask_trait_mdcs_strict_asexual(
            phylogeny_df, trait_mask=[0, 1, 0]
        )


def test_is_leaf_column_used_if_present():
    phylogeny_df = _make_tree()
    trait_mask = np.zeros(len(phylogeny_df), dtype=bool)
    trait_mask[[4, 7]] = True
    res = alifestd_mask_trait_mdcs_strict_asexual(
        phylogeny_df, trait_mask=trait_mask
    )
    assert "is_leaf" not in res.columns
    assert res["alifestd_mask_trait_mdcs_strict_asexual"].any()
    # a (deliberately wrong) is_leaf column is used rather than recomputed
    res = alifestd_mask_trait_mdcs_strict_asexual(
        phylogeny_df.assign(is_leaf=False), trait_mask=trait_mask
    )
    assert not res["alifestd_mask_trait_mdcs_strict_asexual"].any()
