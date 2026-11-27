import typing

import numpy as np
import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_mask_persistence_asexual,
    alifestd_mask_persistence_polars,
    alifestd_prune_extinct_lineages_polars,
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
            "origin_time": [0, 1, 1, 2, 3, 3, 9, 9, 4],
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
                "origin_time": pl.Series([], dtype=pl.Float64),
                "is_leaf": pl.Series([], dtype=pl.Boolean),
            }
        ),
    )
    res = (
        alifestd_mask_persistence_polars(
            df_pl, candidate_mask="is_leaf", extant_mask="is_leaf"
        )
        .lazy()
        .collect()
    )
    assert res.is_empty()
    assert "alifestd_mask_persistence_polars" in res.columns


@apply_params
def test_matches_pandas(apply: typing.Callable):
    df_pl = _make_tree()
    candidate_mask = pl.col("is_leaf") & pl.col("origin_time").is_between(3, 4)
    extant_mask = pl.col("is_leaf") & (pl.col("origin_time") > 8)
    res = (
        alifestd_mask_persistence_polars(
            apply(df_pl),
            candidate_mask=candidate_mask,
            extant_mask=extant_mask,
            mark_as="m",
        )
        .lazy()
        .collect()
    )
    expected = alifestd_mask_persistence_asexual(
        df_pl.to_pandas(),
        candidate_mask=df_pl.select(candidate_mask).to_series().to_numpy(),
        extant_mask=df_pl.select(extant_mask).to_series().to_numpy(),
        mark_as="m",
    )
    assert res["m"].to_list() == expected["m"].tolist()
    # leaf 7 is closest to its sibling 8, and leaf 6 to its sibling 5; leaf
    # 4 is further from leaf 7 than leaf 8 is, so is not marked
    assert res["m"].arg_true().to_list() == [5, 8]
    assert res.drop("m").equals(df_pl)


def test_mask_column_names():
    df_pl = _make_tree().with_columns(
        is_candidate=pl.col("id").is_in([4, 5, 8]),
        is_extant=pl.col("id").is_in([6, 7]),
    )
    res = alifestd_mask_persistence_polars(
        df_pl, candidate_mask="is_candidate", extant_mask="is_extant"
    )
    assert res["alifestd_mask_persistence_polars"].arg_true().to_list() == [
        5,
        8,
    ]


def test_criterion():
    df_pl = _make_tree().with_columns(
        depth=pl.Series([0, 1, 1, 2, 2, 2, 2, 3, 3]),
        origin_time=pl.Series([0, 1, 1, 2, 3, 3, 9, 9, 20]),
    )
    kwargs = dict(
        candidate_mask=pl.col("id").is_in([4, 8]),
        extant_mask=pl.col("id") == 7,
        mark_as="m",
    )
    # leaf 8's long branch puts it further from leaf 7 than leaf 4
    res = alifestd_mask_persistence_polars(df_pl, **kwargs)
    assert res["m"].arg_true().to_list() == [4]
    res = alifestd_mask_persistence_polars(df_pl, criterion="depth", **kwargs)
    assert res["m"].arg_true().to_list() == [8]


def test_prune_extinct_lineages():
    res = alifestd_mask_persistence_polars(
        _make_tree(),
        candidate_mask=pl.col("id").is_in([4, 5, 8]),
        extant_mask=pl.col("id").is_in([6, 7]),
        mark_as="is_persistent",
    ).pipe(alifestd_prune_extinct_lineages_polars, criterion="is_persistent")
    assert res["id"].to_list() == [0, 1, 2, 3, 5, 8]


@pytest.mark.parametrize("name", ["is_leaf", "origin_time"])
def test_null_raises(name: str):
    df_pl = _make_tree().with_columns(
        pl.when(pl.col("id") == 4)
        .then(None)
        .otherwise(pl.col(name))
        .alias(name)
    )
    for kwargs in (
        dict(candidate_mask="is_leaf", extant_mask=pl.lit(True)),
        dict(candidate_mask=pl.lit(True), extant_mask="is_leaf"),
    ):
        with pytest.raises(NotImplementedError):
            alifestd_mask_persistence_polars(df_pl, **kwargs)


@pytest.mark.parametrize(
    "origin_time",
    [
        pytest.param([1.0, 0.0, 2.0], id="decreasing"),
        pytest.param([0.0, float("nan"), 1.0], id="nan"),
        pytest.param([0.0, float("inf"), 1.0], id="inf"),
    ],
)
def test_unsupported_criterion_raises(origin_time: list):
    df_pl = pl.DataFrame(
        {"id": [0, 1, 2], "ancestor_id": [0, 0, 0], "origin_time": origin_time}
    )
    with pytest.raises(NotImplementedError):
        alifestd_mask_persistence_polars(
            df_pl, candidate_mask=pl.lit(True), extant_mask=pl.lit(True)
        )


def test_missing_criterion_raises():
    df_pl = pl.DataFrame({"id": [0, 1, 2], "ancestor_id": [0, 0, 0]})
    with pytest.raises(ValueError):
        alifestd_mask_persistence_polars(
            df_pl, candidate_mask=pl.lit(True), extant_mask=pl.lit(True)
        )


@pytest.mark.parametrize(
    "df_pl",
    [
        pytest.param(
            pl.DataFrame(
                {
                    "id": [3, 13, 23],
                    "ancestor_id": [3, 3, 3],
                    "origin_time": [0, 1, 1],
                }
            ),
            id="noncontiguous",
        ),
        pytest.param(
            pl.DataFrame(
                {
                    "id": [0, 1, 2],
                    "ancestor_id": [1, 1, 1],
                    "origin_time": [1, 0, 1],
                }
            ),
            id="unsorted",
        ),
    ],
)
def test_not_working_format_raises(df_pl: pl.DataFrame):
    with pytest.raises(NotImplementedError):
        alifestd_mask_persistence_polars(
            df_pl, candidate_mask=pl.lit(True), extant_mask=pl.lit(True)
        )


def test_random_tree_matches_pandas():
    rng = np.random.default_rng(1)
    n = 500
    ancestor_id = np.concatenate([[0], rng.integers(0, np.arange(1, n))])
    origin_time = np.zeros(n, dtype=int)
    for id_, ancestor_id_ in enumerate(ancestor_id):  # integers, so ties
        origin_time[id_] = origin_time[ancestor_id_] + rng.integers(3)
    df_pd = pd.DataFrame(
        {
            "id": range(n),
            "ancestor_id": ancestor_id,
            "origin_time": origin_time,
            "is_candidate": rng.random(n) < 0.3,
            "is_extant": rng.random(n) < 0.1,
        },
    )
    expected = alifestd_mask_persistence_asexual(
        df_pd,
        candidate_mask=df_pd["is_candidate"].to_numpy(),
        extant_mask=df_pd["is_extant"].to_numpy(),
    )
    res = alifestd_mask_persistence_polars(
        pl.from_pandas(df_pd),
        candidate_mask="is_candidate",
        extant_mask="is_extant",
    )
    assert (
        res["alifestd_mask_persistence_polars"].to_list()
        == expected["alifestd_mask_persistence_asexual"].tolist()
    )
    assert res["alifestd_mask_persistence_polars"].any()
    assert not res["alifestd_mask_persistence_polars"].equals(
        res["is_candidate"]
    )
