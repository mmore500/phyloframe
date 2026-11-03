import os
import typing

import numpy as np
import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_find_chronological_inconsistency_polars,
    alifestd_pool_chronological_consistency_asexual,
    alifestd_to_working_format,
    alifestd_try_add_ancestor_id_col,
)
from phyloframe.legacy._alifestd_pool_chronological_consistency_polars import (
    alifestd_pool_chronological_consistency_polars as alifestd_pool_chronological_consistency_polars_,
)

from ._impl import enforce_dtype_stability_polars

alifestd_pool_chronological_consistency_polars = (
    enforce_dtype_stability_polars(
        alifestd_pool_chronological_consistency_polars_,
    )
)

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

_apply = pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)


@_apply
def test_pool_chronological_consistency_polars_empty(
    apply: typing.Callable,
):
    df = pl.DataFrame(
        {
            "id": pl.Series([], dtype=pl.Int64),
            "ancestor_id": pl.Series([], dtype=pl.Int64),
            "origin_time": pl.Series([], dtype=pl.Float64),
        }
    )
    result = alifestd_pool_chronological_consistency_polars(apply(df))
    assert result.lazy().collect().is_empty()


@_apply
def test_pool_chronological_consistency_polars_consistent(
    apply: typing.Callable,
):
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6],
            "ancestor_id": [0, 0, 0, 1, 1, 2, 2],
            "origin_time": [0.1, 0.1, 0.7, 0.3, 1.0, 0.7, 0.9],
        }
    )
    result = alifestd_pool_chronological_consistency_polars(apply(df))
    assert result.lazy().collect().equals(df)


@_apply
def test_pool_chronological_consistency_polars_inconsistent(
    apply: typing.Callable,
):
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6],
            "ancestor_id": [0, 0, 0, 1, 1, 2, 2],
            "origin_time": [5, 1, 6, 10, 10, 10, 10],
            "trait": [1, 2, 3, 4, 5, 6, 7],
        }
    )
    result = (
        alifestd_pool_chronological_consistency_polars(apply(df))
        .lazy()
        .collect()
    )
    assert result["origin_time"].to_list() == [4.2, 4.2, 6, 10, 10, 10, 10]
    assert result["origin_time"].dtype == pl.Float64
    assert result.drop("origin_time").equals(df.drop("origin_time"))

    result = (
        alifestd_pool_chronological_consistency_polars(
            apply(df), weight_by="uniform"
        )
        .lazy()
        .collect()
    )
    assert result["origin_time"].to_list() == [3, 3, 6, 10, 10, 10, 10]


@_apply
def test_pool_chronological_consistency_polars_unifurcation(
    apply: typing.Callable,
):
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5],
            "ancestor_id": [0, 0, 1, 2, 2, 0],
            "origin_time": [5.0, 7.0, 2.0, 10.0, 10.0, 10.0],
        }
    )
    result = (
        alifestd_pool_chronological_consistency_polars(apply(df))
        .lazy()
        .collect()
    )
    assert result["origin_time"].to_list() == [4, 4, 4, 10, 10, 10]


@_apply
@pytest.mark.parametrize("pin_leaves", [True, False])
def test_pool_chronological_consistency_polars_pin_leaves(
    apply: typing.Callable, pin_leaves: bool
):
    df = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 0, 0],
            "origin_time": [5.0, 3.0, 10.0],
        }
    )
    result = (
        alifestd_pool_chronological_consistency_polars(
            apply(df), weight_by="uniform", pin_leaves=pin_leaves
        )
        .lazy()
        .collect()
    )
    expected = [3.0, 3.0, 10.0] if pin_leaves else [4.0, 4.0, 10.0]
    assert result["origin_time"].to_list() == expected


@_apply
@pytest.mark.parametrize("as_type", ["column", "ndarray", "series"])
def test_pool_chronological_consistency_polars_custom_weights(
    apply: typing.Callable, as_type: str
):
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 1, 1],
            "origin_time": [6.0, 0.0, 9.0, 9.0],
            "w": [1.0, 2.0, 0.0, 0.0],
        }
    )
    weight_by = {
        "column": "w",
        "ndarray": df["w"].to_numpy(),
        "series": df["w"],
    }[as_type]
    result = (
        alifestd_pool_chronological_consistency_polars(
            apply(df), weight_by=weight_by
        )
        .lazy()
        .collect()
    )
    assert result["origin_time"].to_list() == [2.0, 2.0, 9.0, 9.0]


@_apply
def test_pool_chronological_consistency_polars_invalid_weights(
    apply: typing.Callable,
):
    df = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 0, 1],
            "origin_time": [1.0, 0.0, 2.0],
        }
    )
    with pytest.raises(ValueError):
        alifestd_pool_chronological_consistency_polars(
            apply(df), weight_by=np.array([1.0, -1.0, 0.0])
        )


@_apply
@pytest.mark.parametrize(
    "dtype, expected_dtype",
    [
        (pl.Int64, pl.Float64),
        (pl.UInt32, pl.Float64),
        (pl.Float32, pl.Float32),
        (pl.Float64, pl.Float64),
    ],
)
@pytest.mark.parametrize("id_dtype", [pl.UInt32, pl.Int32, pl.Int64])
def test_pool_chronological_consistency_polars_dtype(
    apply: typing.Callable,
    dtype: pl.DataType,
    expected_dtype: pl.DataType,
    id_dtype: pl.DataType,
):
    df = pl.DataFrame(
        {
            "id": pl.Series([0, 1, 2, 3], dtype=id_dtype),
            "ancestor_id": pl.Series([0, 0, 1, 1], dtype=id_dtype),
            "origin_time": pl.Series([2, 1, 3, 3], dtype=dtype),
        }
    )
    result = (
        alifestd_pool_chronological_consistency_polars(
            apply(df), weight_by="uniform"
        )
        .lazy()
        .collect()
    )
    assert result["id"].dtype == id_dtype
    assert result["ancestor_id"].dtype == id_dtype
    assert result["origin_time"].dtype == expected_dtype
    assert result["origin_time"].to_list() == [1.5, 1.5, 3, 3]


@_apply
@pytest.mark.parametrize(
    "df",
    [
        pl.DataFrame(  # non-contiguous
            {
                "id": [0, 1, 3],
                "ancestor_id": [0, 0, 1],
                "origin_time": [0, 1, 2],
            },
        ),
        pl.DataFrame(  # unsorted
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 2, 0],
                "origin_time": [0, 1, 2],
            },
        ),
        pl.DataFrame(  # sexual
            {
                "id": [0, 1, 2],
                "ancestor_list": ["[none]", "[0]", "[0,1]"],
                "origin_time": [0, 1, 2],
            },
        ),
        pl.DataFrame(  # non-numeric
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 0, 1],
                "origin_time": ["a", "c", "b"],
            },
        ),
    ],
)
def test_pool_chronological_consistency_polars_unsupported(
    apply: typing.Callable, df: pl.DataFrame
):
    with pytest.raises(NotImplementedError):
        alifestd_pool_chronological_consistency_polars(apply(df))


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
@pytest.mark.parametrize("weight_by", ["mrca_leaf_pairs", "uniform"])
@pytest.mark.parametrize("pin_leaves", [True, False])
def test_pool_chronological_consistency_polars_matches_pandas(
    apply: typing.Callable,
    phylogeny_df: pd.DataFrame,
    seed: int,
    weight_by: str,
    pin_leaves: bool,
):
    phylogeny_df = alifestd_try_add_ancestor_id_col(
        alifestd_to_working_format(phylogeny_df), mutate=True
    )
    rng = np.random.default_rng(seed)
    phylogeny_df["origin_time"] = phylogeny_df["origin_time"] + rng.normal(
        scale=phylogeny_df["origin_time"].std() / 4,
        size=len(phylogeny_df),
    )
    phylogeny_df = phylogeny_df[
        ["id", "ancestor_id", "ancestor_list", "origin_time"]
    ]

    expected = alifestd_pool_chronological_consistency_asexual(
        phylogeny_df, weight_by=weight_by, pin_leaves=pin_leaves
    )
    result = (
        alifestd_pool_chronological_consistency_polars(
            apply(pl.from_pandas(phylogeny_df)),
            weight_by=weight_by,
            pin_leaves=pin_leaves,
        )
        .lazy()
        .collect()
    )
    assert alifestd_find_chronological_inconsistency_polars(result) is None
    for col in expected.columns:
        assert result[col].to_list() == expected[col].tolist(), col
