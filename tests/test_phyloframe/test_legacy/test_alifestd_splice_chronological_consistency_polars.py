import os
import typing

import numpy as np
import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_splice_chronological_consistency_asexual,
    alifestd_to_working_format,
    alifestd_try_add_ancestor_id_col,
)
from phyloframe.legacy._alifestd_splice_chronological_consistency_polars import (
    alifestd_splice_chronological_consistency_polars as alifestd_splice_chronological_consistency_polars_,
)

from ._impl import enforce_dtype_stability_polars

alifestd_splice_chronological_consistency_polars = (
    enforce_dtype_stability_polars(
        alifestd_splice_chronological_consistency_polars_,
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


def _collect(df: typing.Union[pl.DataFrame, pl.LazyFrame]) -> pl.DataFrame:
    return df.lazy().collect()


@_apply
def test_splice_chronological_consistency_polars_empty(
    apply: typing.Callable,
):
    df = pl.DataFrame(
        {
            "id": pl.Series([], dtype=pl.Int64),
            "ancestor_id": pl.Series([], dtype=pl.Int64),
            "origin_time": pl.Series([], dtype=pl.Float64),
        }
    )
    result = alifestd_splice_chronological_consistency_polars(apply(df))
    assert _collect(result).is_empty()


@_apply
def test_splice_chronological_consistency_polars_consistent(
    apply: typing.Callable,
):
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 1, 1],
            "origin_time": [0.0, 1.0, 2.0, 1.0],
        }
    )
    result = alifestd_splice_chronological_consistency_polars(apply(df))
    assert _collect(result).equals(df)


@_apply
def test_splice_chronological_consistency_polars_inconsistent(
    apply: typing.Callable,
):
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5],
            "ancestor_id": [0, 0, 1, 2, 3, 4],
            "origin_time": [0, 5, 3, 6, 4, -1],
            "trait": [1, 2, 3, 4, 5, 6],
        }
    )
    result = _collect(
        alifestd_splice_chronological_consistency_polars(
            apply(df), ignore_topological_sensitivity=True
        ),
    )
    assert result["id"].to_list() == [*range(9)]
    assert result["ancestor_id"].to_list() == [8, 6, 6, 7, 7, 8, 0, 2, 8]
    assert result["origin_time"].to_list() == [0, 5, 3, 6, 4, -1, 3, 4, -1]
    assert result["trait"].to_list() == [1, 2, 3, 4, 5, 6, None, None, None]
    assert result["trait"].dtype == df["trait"].dtype


@_apply
def test_splice_chronological_consistency_polars_ancestor_list(
    apply: typing.Callable,
):
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 1, 1],
            "ancestor_list": ["[None]", "[0]", "[1]", "[1]"],
            "origin_time": [1, 2, 0, 3],
        }
    )
    result = _collect(
        alifestd_splice_chronological_consistency_polars(apply(df)),
    )
    assert result["ancestor_id"].to_list() == [4, 0, 4, 1, 4]
    assert result["ancestor_list"].to_list() == [
        "[4]",
        "[0]",
        "[4]",
        "[1]",
        "[none]",
    ]


@_apply
@pytest.mark.parametrize("dtype", [pl.UInt32, pl.Int32, pl.Int64])
def test_splice_chronological_consistency_polars_dtype(
    apply: typing.Callable, dtype: pl.DataType
):
    df = pl.DataFrame(
        {
            "id": pl.Series([0, 1, 2], dtype=dtype),
            "ancestor_id": pl.Series([0, 0, 1], dtype=dtype),
            "origin_time": pl.Series([0.0, 2.0, 1.0], dtype=pl.Float32),
        }
    )
    result = _collect(
        alifestd_splice_chronological_consistency_polars(apply(df)),
    )
    assert result["id"].dtype == dtype
    assert result["ancestor_id"].dtype == dtype
    assert result["origin_time"].dtype == pl.Float32
    assert result["ancestor_id"].to_list() == [0, 3, 3, 0]


@_apply
def test_splice_chronological_consistency_polars_str(
    apply: typing.Callable,
):
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 1, 2],
            "origin_time": ["a", "c", "b", "d"],
        }
    )
    result = _collect(
        alifestd_splice_chronological_consistency_polars(apply(df)),
    )
    assert result["ancestor_id"].to_list() == [0, 4, 4, 2, 0]
    assert result["origin_time"].to_list() == ["a", "c", "b", "d", "b"]


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
    ],
)
def test_splice_chronological_consistency_polars_unsupported(
    apply: typing.Callable, df: pl.DataFrame
):
    with pytest.raises(NotImplementedError):
        alifestd_splice_chronological_consistency_polars(apply(df))


@_apply
def test_splice_chronological_consistency_polars_drop_sensitivity(
    apply: typing.Callable,
):
    df = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 0, 1],
            "origin_time": [0, 2, 1],
            "num_children": [1, 1, 0],
        }
    )
    with pytest.warns(UserWarning):
        alifestd_splice_chronological_consistency_polars(apply(df))
    result = alifestd_splice_chronological_consistency_polars(
        apply(df), drop_topological_sensitivity=True
    )
    assert "num_children" not in result.lazy().collect_schema().names()


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
def test_splice_chronological_consistency_polars_matches_pandas(
    apply: typing.Callable, phylogeny_df: pd.DataFrame, seed: int
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

    expected = alifestd_splice_chronological_consistency_asexual(
        phylogeny_df, ignore_topological_sensitivity=True
    )
    result = _collect(
        alifestd_splice_chronological_consistency_polars(
            apply(pl.from_pandas(phylogeny_df)),
            ignore_topological_sensitivity=True,
        ),
    )
    assert len(result) > len(phylogeny_df)
    for col in expected.columns:
        assert result[col].to_list() == expected[col].tolist(), col
