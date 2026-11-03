import os
import typing

import numpy as np
import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_mark_lineage_ffill_asexual,
    alifestd_mark_lineage_ffill_polars,
    alifestd_mark_to_array_polars,
    alifestd_to_working_format,
    alifestd_try_add_ancestor_id_col,
)

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

_apply = pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)


def _make_df(values: typing.Optional[pl.Series] = None) -> pl.DataFrame:
    """Tree:
    0 (root, null)
    +-- 1 (5)
    |   +-- 3 (null)
    |   |   +-- 5 (null)
    |   +-- 4 (7)
    +-- 2 (null)
    6 (root, 2)
    +-- 7 (null)
    """
    return pl.DataFrame(
        {
            "id": [*range(8)],
            "ancestor_id": [0, 0, 0, 1, 1, 3, 6, 6],
            "v": (
                pl.Series([None, 5.0, None, None, 7.0, None, 2.0, None])
                if values is None
                else values
            ),
        },
    )


@_apply
def test_hand_built_tree(apply: typing.Callable):
    res = (
        alifestd_mark_lineage_ffill_polars(apply(_make_df()), "v")
        .lazy()
        .collect()
    )
    assert res["lineage_ffill"].to_list() == [
        None,
        5.0,
        None,
        5.0,
        7.0,
        5.0,
        2.0,
        2.0,
    ]
    assert res["v"].equals(_make_df()["v"])


@_apply
@pytest.mark.parametrize(
    "values, expected",
    [
        (
            pl.Series([None, 5, None, None, 7, None, 2, None], dtype=pl.Int32),
            [None, 5, None, 5, 7, 5, 2, 2],
        ),
        (
            pl.Series([None, "a", None, None, "b", None, "c", None]),
            [None, "a", None, "a", "b", "a", "c", "c"],
        ),
    ],
)
def test_dtypes(apply: typing.Callable, values: pl.Series, expected: list):
    df = _make_df(values)
    res = alifestd_mark_lineage_ffill_polars(apply(df), "v").lazy().collect()
    assert res["lineage_ffill"].dtype == values.dtype
    assert res["lineage_ffill"].to_list() == expected


def test_nan_not_filled():
    nan = float("nan")
    df = _make_df(pl.Series([None, nan, None, None, 7.0, None, 2.0, None]))
    res = alifestd_mark_lineage_ffill_polars(df, "v")
    np.testing.assert_array_equal(
        res["lineage_ffill"].to_numpy(),
        [np.nan, nan, np.nan, nan, 7.0, nan, 2.0, 2.0],
    )
    assert res["lineage_ffill"].null_count() == 2  # 0 and 2 remain null


def test_expression():
    res = alifestd_mark_lineage_ffill_polars(_make_df(), pl.col("v") * 2)
    assert res["lineage_ffill"].to_list() == [
        None,
        10.0,
        None,
        10.0,
        14.0,
        10.0,
        4.0,
        4.0,
    ]


def test_mark_to_array():
    res = alifestd_mark_to_array_polars(
        _make_df(), alifestd_mark_lineage_ffill_polars, values="v"
    )
    np.testing.assert_array_equal(
        res, [np.nan, 5.0, np.nan, 5.0, 7.0, 5.0, 2.0, 2.0]
    )


@_apply
def test_mark_as(apply: typing.Callable):
    res = alifestd_mark_lineage_ffill_polars(
        apply(_make_df()), "v", mark_as="blueberry"
    )
    names = res.lazy().collect_schema().names()
    assert "blueberry" in names
    assert "lineage_ffill" not in names


@_apply
def test_empty(apply: typing.Callable):
    df = pl.DataFrame(
        {"id": [], "ancestor_id": [], "v": []},
        schema={"id": pl.Int64, "ancestor_id": pl.Int64, "v": pl.Float64},
    )
    res = alifestd_mark_lineage_ffill_polars(apply(df), "v").lazy().collect()
    assert "lineage_ffill" in res.columns
    assert res.is_empty()


@_apply
@pytest.mark.parametrize(
    "df",
    [
        pl.DataFrame(  # non-contiguous
            {"id": [0, 2], "ancestor_id": [0, 0], "v": [1.0, None]},
        ),
        pl.DataFrame(  # unsorted
            {"id": [0, 1, 2], "ancestor_id": [0, 2, 0], "v": [1.0] * 3},
        ),
        pl.DataFrame(  # sexual
            {
                "id": [0, 1, 2],
                "ancestor_list": ["[none]", "[0]", "[0,1]"],
                "v": [1.0] * 3,
            },
        ),
    ],
)
def test_unsupported(apply: typing.Callable, df: pl.DataFrame):
    with pytest.raises(NotImplementedError):
        alifestd_mark_lineage_ffill_polars(apply(df), "v")


@_apply
@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pd.read_csv(f"{assets}/nk_ecoeaselection-workingformat.csv"),
        pd.read_csv(f"{assets}/nk_lexicaseselection.csv"),
        pd.read_csv(f"{assets}/nk_tournamentselection.csv"),
    ],
)
@pytest.mark.parametrize("seed", range(3))
def test_matches_pandas(
    apply: typing.Callable, phylogeny_df: pd.DataFrame, seed: int
):
    phylogeny_df = alifestd_try_add_ancestor_id_col(
        alifestd_to_working_format(phylogeny_df)
    )[["id", "ancestor_id", "origin_time"]]
    rng = np.random.default_rng(seed)
    phylogeny_df["v"] = np.where(
        rng.random(len(phylogeny_df)) < 0.5,
        np.nan,
        phylogeny_df["origin_time"].astype(float),
    )
    expected = alifestd_mark_lineage_ffill_asexual(phylogeny_df, "v")
    res = (
        alifestd_mark_lineage_ffill_polars(
            apply(pl.from_pandas(phylogeny_df)), "v"  # NaN -> null
        )
        .lazy()
        .collect()
    )
    np.testing.assert_array_equal(
        res["lineage_ffill"].to_numpy(), expected["lineage_ffill"]
    )
