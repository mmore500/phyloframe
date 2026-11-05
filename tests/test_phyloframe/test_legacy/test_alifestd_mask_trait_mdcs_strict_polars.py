import typing

import numpy as np
import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_mask_trait_mdcs_strict_asexual,
    alifestd_mask_trait_mdcs_strict_polars,
)

apply_params = pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)


def _make_tree() -> pl.DataFrame:
    #        0
    #     1     2
    #    3 4   5 6
    #   7 8
    return pl.DataFrame(
        {
            "id": range(9),
            "ancestor_id": [0, 0, 0, 1, 1, 2, 2, 3, 3],
            "trait": [
                False,
                False,
                False,
                False,
                True,
                True,
                False,
                True,
                True,
            ],
            "is_leaf": [False] * 4 + [True] * 5,
        },
    )


@apply_params
def test_empty(apply: typing.Callable):
    df_pl = apply(
        pl.DataFrame(
            {
                "id": pl.Series([], dtype=pl.Int64),
                "ancestor_id": pl.Series([], dtype=pl.Int64),
                "trait": pl.Series([], dtype=pl.Boolean),
                "is_leaf": pl.Series([], dtype=pl.Boolean),
            }
        ),
    )
    res = (
        alifestd_mask_trait_mdcs_strict_polars(
            df_pl, observed_mask="is_leaf", trait_mask="trait"
        )
        .lazy()
        .collect()
    )
    assert res.is_empty()
    assert "alifestd_mask_trait_mdcs_strict_polars" in res.columns


@apply_params
def test_matches_pandas(apply: typing.Callable):
    df_pl = _make_tree()
    res = (
        alifestd_mask_trait_mdcs_strict_polars(
            apply(df_pl),
            observed_mask=pl.col("is_leaf"),
            trait_mask=pl.col("trait"),
            mark_as="m",
        )
        .lazy()
        .collect()
    )
    expected = alifestd_mask_trait_mdcs_strict_asexual(
        df_pl.to_pandas(),
        observed_mask=df_pl["is_leaf"].to_numpy(),
        trait_mask=df_pl["trait"].to_numpy(),
        mark_as="m",
    )
    assert res["m"].to_list() == expected["m"].tolist()
    assert res["m"].to_list() == list(map(bool, [0, 1, 0, 0, 0, 1, 0, 0, 0]))


def test_null_trait_raises():
    df_pl = _make_tree().with_columns(
        trait=pl.when(pl.col("id") == 4).then(None).otherwise(pl.col("trait"))
    )
    with pytest.raises(NotImplementedError):
        alifestd_mask_trait_mdcs_strict_polars(
            df_pl, observed_mask="is_leaf", trait_mask="trait"
        )


def test_null_observed_raises():
    df_pl = _make_tree().with_columns(
        is_leaf=pl.when(pl.col("id") == 4)
        .then(None)
        .otherwise(pl.col("is_leaf"))
    )
    with pytest.raises(NotImplementedError):
        alifestd_mask_trait_mdcs_strict_polars(
            df_pl, observed_mask="is_leaf", trait_mask="trait"
        )


def test_observed_inner_node():
    df_pl = _make_tree().with_columns(
        trait=pl.col("id").is_in([7, 8]),
        observed=pl.col("is_leaf") | (pl.col("id") == 3),
    )
    res = alifestd_mask_trait_mdcs_strict_polars(
        df_pl, observed_mask="observed", trait_mask="trait", mark_as="m"
    )
    # observing node 3 without the trait splits clade 3 into two gains
    assert res["m"].arg_true().to_list() == [7, 8]


def test_unsorted_raises():
    df_pl = pl.DataFrame(
        {"id": [0, 1, 2], "ancestor_id": [1, 1, 1], "trait": [1, 0, 1]},
    ).with_columns(pl.col("trait").cast(pl.Boolean))
    with pytest.raises(NotImplementedError):
        alifestd_mask_trait_mdcs_strict_polars(
            df_pl, observed_mask=pl.lit(True), trait_mask="trait"
        )


def test_random_tree_matches_pandas():
    rng = np.random.default_rng(1)
    n = 500
    ancestor_id = np.concatenate([[0], rng.integers(0, np.arange(1, n))])
    df_pd = pd.DataFrame(
        {
            "id": range(n),
            "ancestor_id": ancestor_id,
            "trait": rng.random(n) < 0.3,
            "observed": rng.random(n) < 0.5,
        },
    )
    expected = alifestd_mask_trait_mdcs_strict_asexual(
        df_pd,
        observed_mask=df_pd["observed"].to_numpy(),
        trait_mask=df_pd["trait"].to_numpy(),
    )
    res = alifestd_mask_trait_mdcs_strict_polars(
        pl.from_pandas(df_pd), observed_mask="observed", trait_mask="trait"
    )
    assert (
        res["alifestd_mask_trait_mdcs_strict_polars"].to_list()
        == expected["alifestd_mask_trait_mdcs_strict_asexual"].tolist()
    )
