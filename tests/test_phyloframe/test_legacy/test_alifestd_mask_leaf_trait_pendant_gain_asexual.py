import numpy as np
import pandas as pd
import pytest

from phyloframe.legacy import (
    alifestd_make_empty,
)
from phyloframe.legacy import (
    alifestd_mask_leaf_trait_pendant_gain_asexual as alifestd_mask_leaf_trait_pendant_gain_asexual_,
)

from ._impl import enforce_dtype_stability_pandas

alifestd_mask_leaf_trait_pendant_gain_asexual = enforce_dtype_stability_pandas(
    alifestd_mask_leaf_trait_pendant_gain_asexual_
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
    res = alifestd_mask_leaf_trait_pendant_gain_asexual(
        alifestd_make_empty(), trait_mask=np.array([], dtype=bool)
    )
    assert "alifestd_mask_leaf_trait_pendant_gain_asexual" in res.columns
    assert len(res) == 0


@pytest.mark.parametrize("mutate", [False, True])
def test_simple(mutate: bool):
    phylogeny_df = _make_tree()
    original_df = phylogeny_df.copy()
    trait_mask = np.array([0, 0, 0, 0, 1, 1, 0, 1, 1], dtype=bool)
    res = alifestd_mask_leaf_trait_pendant_gain_asexual(
        phylogeny_df, mutate=mutate, trait_mask=trait_mask
    )
    res.index = res["id"]
    # 5's sibling 6 lacks the trait, so 5 gained it on its pendant branch;
    # 4, 7, 8 share it with every leaf of their parent's clade
    expected = [0, 0, 0, 0, 0, 1, 0, 0, 0]
    assert res.loc[
        range(9), "alifestd_mask_leaf_trait_pendant_gain_asexual"
    ].tolist() == list(map(bool, expected))
    if not mutate:
        assert original_df.equals(phylogeny_df)


def test_inner_values_ignored():
    phylogeny_df = _make_tree()
    trait_mask = np.array([1, 1, 1, 1, 0, 1, 0, 1, 1], dtype=bool)
    res = alifestd_mask_leaf_trait_pendant_gain_asexual(
        phylogeny_df, trait_mask=trait_mask, mark_as="m"
    )
    res.index = res["id"]
    assert res.loc[range(9), "m"].tolist() == list(
        map(bool, [0, 0, 0, 0, 0, 1, 0, 0, 0])
    )


def test_root_leaf():
    phylogeny_df = pd.DataFrame(
        {"id": [0, 1, 2], "ancestor_id": [0, 1, 1], "origin_time": [0, 0, 1]}
    )
    res = alifestd_mask_leaf_trait_pendant_gain_asexual(
        phylogeny_df, trait_mask=np.array([True, False, True])
    )
    res.index = res["id"]
    assert not res.loc[0, "alifestd_mask_leaf_trait_pendant_gain_asexual"]
    # 2 is the only leaf of its parent's clade, so shares its trait with it
    assert not res.loc[2, "alifestd_mask_leaf_trait_pendant_gain_asexual"]


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pytest.param(  # noncontiguous ids
            pd.DataFrame({"id": [3, 13, 23], "ancestor_id": [3, 3, 3]}),
            id="noncontiguous",
        ),
        pytest.param(  # child 0 listed before its parent 1
            pd.DataFrame({"id": [0, 1, 2], "ancestor_id": [1, 1, 1]}),
            id="unsorted",
        ),
    ],
)
def test_not_working_format_raises(phylogeny_df: pd.DataFrame):
    with pytest.raises(NotImplementedError):
        alifestd_mask_leaf_trait_pendant_gain_asexual(
            phylogeny_df, trait_mask=np.zeros(3, dtype=bool)
        )


@pytest.mark.parametrize(
    "trait_mask",
    [
        [np.nan, np.nan, 1.0, np.nan],
        pd.array([None, None, True, None], dtype="boolean"),
        [0, 0, 1, 0],
    ],
)
def test_non_bool_trait_mask_raises(trait_mask):
    #     0
    #   1   2
    #       3
    phylogeny_df = pd.DataFrame(
        {"id": [0, 1, 2, 3], "ancestor_id": [0, 0, 0, 2]}
    )
    with pytest.raises(NotImplementedError):
        alifestd_mask_leaf_trait_pendant_gain_asexual(
            phylogeny_df, trait_mask=trait_mask
        )


def test_unifurcations_bypassed():
    #     0
    #   1   2
    #   3
    # node 1 is a unifurcation, so leaf 3's parent is effectively root 0,
    # whose leaves (2, 3) differ, so leaf 3's trait arose on its own branch
    phylogeny_df = pd.DataFrame(
        {"id": [0, 1, 2, 3], "ancestor_id": [0, 0, 0, 1]}
    )
    res = alifestd_mask_leaf_trait_pendant_gain_asexual(
        phylogeny_df, trait_mask=np.array([False, False, False, True])
    )
    assert res["alifestd_mask_leaf_trait_pendant_gain_asexual"].tolist() == [
        False,
        False,
        False,
        True,
    ]

    # with the trait on both leaves, root's clade is monomorphic
    res = alifestd_mask_leaf_trait_pendant_gain_asexual(
        phylogeny_df, trait_mask=np.array([False, False, True, True])
    )
    assert not res["alifestd_mask_leaf_trait_pendant_gain_asexual"].any()


def test_unifurcations_match_collapsed():
    from phyloframe.legacy import (
        alifestd_collapse_unifurcations,
        alifestd_to_working_format,
    )

    rng = np.random.default_rng(1)
    num_nodes = 300
    ancestor_ids = np.concatenate(
        [[0], rng.integers(0, np.arange(1, num_nodes))]
    )
    phylogeny_df = pd.DataFrame(
        {"id": np.arange(num_nodes), "ancestor_id": ancestor_ids}
    )
    trait_mask = rng.random(num_nodes) < 0.5
    res = alifestd_mask_leaf_trait_pendant_gain_asexual(
        phylogeny_df, trait_mask=trait_mask
    ).set_index("id")["alifestd_mask_leaf_trait_pendant_gain_asexual"]

    collapsed = phylogeny_df.assign(trait=trait_mask)
    collapsed = alifestd_to_working_format(
        alifestd_collapse_unifurcations(
            collapsed, ignore_topological_sensitivity=True
        ).assign(orig_id=lambda df: df["id"]),
    )
    expected = alifestd_mask_leaf_trait_pendant_gain_asexual(
        collapsed, trait_mask=collapsed["trait"].to_numpy()
    )
    for orig_id, mark in zip(
        expected["orig_id"],
        expected["alifestd_mask_leaf_trait_pendant_gain_asexual"],
    ):
        assert res[orig_id] == mark


#   0       5
#   1      6 7
#  2 3
#  4
# root 0 is unifurcating, root 5 is branching, and 2 is unifurcating
_FOREST_ANCESTOR_IDS = np.array([0, 0, 1, 1, 2, 5, 5, 5])
_FOREST_IS_BRANCHING = np.array([0, 1, 0, 0, 0, 1, 0, 0], dtype=bool)


def test_stem_node_ids():
    from phyloframe.legacy._alifestd_mask_leaf_trait_pendant_gain_asexual import (
        _alifestd_mask_leaf_trait_pendant_gain_asexual_stem_node_ids as stem_node_ids,
    )

    res = stem_node_ids(_FOREST_ANCESTOR_IDS, _FOREST_IS_BRANCHING)
    # 0, 1, and 5 have no stem node above them, so are their own
    assert res.tolist() == [0, 1, 1, 1, 1, 5, 5, 5]


def test_no_stem_node():
    # a single unbranched lineage, so leaf 2 has no clade-mate to compare
    phylogeny_df = pd.DataFrame(
        {"id": [0, 1, 2], "ancestor_id": [0, 0, 1], "origin_time": [0, 1, 2]}
    )
    res = alifestd_mask_leaf_trait_pendant_gain_asexual(
        phylogeny_df, trait_mask=np.array([False, False, True])
    )
    assert not res["alifestd_mask_leaf_trait_pendant_gain_asexual"].any()
