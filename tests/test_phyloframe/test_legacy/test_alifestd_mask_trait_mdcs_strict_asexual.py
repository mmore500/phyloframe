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


def _leaves(phylogeny_df: pd.DataFrame) -> np.ndarray:
    return (~phylogeny_df["id"].isin(phylogeny_df["ancestor_id"])).to_numpy(
        copy=True
    )


def test_empty():
    res = alifestd_mask_trait_mdcs_strict_asexual(
        alifestd_make_empty(),
        observed_mask=np.array([], dtype=bool),
        trait_mask=np.array([], dtype=bool),
    )
    assert "alifestd_mask_trait_mdcs_strict_asexual" in res.columns
    assert len(res) == 0


@pytest.mark.parametrize("mutate", [False, True])
def test_simple(mutate: bool):
    phylogeny_df = _make_tree()
    original_df = phylogeny_df.copy()
    trait_mask = np.array([0, 0, 0, 0, 1, 1, 0, 1, 1], dtype=bool)
    res = alifestd_mask_trait_mdcs_strict_asexual(
        phylogeny_df,
        mutate=mutate,
        observed_mask=_leaves(phylogeny_df),
        trait_mask=trait_mask,
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
        observed_mask=_leaves(phylogeny_df),
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
        phylogeny_df,
        observed_mask=_leaves(phylogeny_df),
        trait_mask=np.array([False, False, False, True]),
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
        phylogeny_df, observed_mask=is_leaf.to_numpy(), trait_mask=trait_mask
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
            phylogeny_df,
            observed_mask=np.ones(3, dtype=bool),
            trait_mask=np.zeros(3, dtype=bool),
        )


@pytest.mark.parametrize(
    "masks",
    [
        dict(observed_mask=[False, True, True], trait_mask=[0, 1, 0]),
        dict(observed_mask=[0, 1, 1], trait_mask=[False, True, False]),
    ],
)
def test_non_bool_mask_raises(masks: dict):
    phylogeny_df = pd.DataFrame({"id": [0, 1, 2], "ancestor_id": [0, 0, 0]})
    with pytest.raises(NotImplementedError):
        alifestd_mask_trait_mdcs_strict_asexual(phylogeny_df, **masks)


def test_observed_inner_node():
    phylogeny_df = _make_tree()
    trait_mask = np.zeros(9, dtype=bool)
    trait_mask[[7, 8]] = True
    observed_mask = _leaves(phylogeny_df)
    col = "alifestd_mask_trait_mdcs_strict_asexual"
    # with only leaves observed, leaves 7 and 8 share one gain, above node 3
    res = alifestd_mask_trait_mdcs_strict_asexual(
        phylogeny_df, observed_mask=observed_mask, trait_mask=trait_mask
    )
    assert np.flatnonzero(res[col]).tolist() == [3]
    # observing node 3 without the trait splits that MDC into two gains
    observed_mask[3] = True
    res = alifestd_mask_trait_mdcs_strict_asexual(
        phylogeny_df, observed_mask=observed_mask, trait_mask=trait_mask
    )
    assert np.flatnonzero(res[col]).tolist() == [7, 8]


def test_clade_without_observed_nodes_not_marked():
    phylogeny_df = _make_tree()
    # leaf 5 is unobserved, so its clade is never marked, despite its trait
    observed_mask = np.zeros(9, dtype=bool)
    observed_mask[[4, 6, 7, 8]] = True
    trait_mask = np.ones(9, dtype=bool)
    trait_mask[6] = False
    res = alifestd_mask_trait_mdcs_strict_asexual(
        phylogeny_df, observed_mask=observed_mask, trait_mask=trait_mask
    )
    assert np.flatnonzero(
        res["alifestd_mask_trait_mdcs_strict_asexual"]
    ).tolist() == [1]
