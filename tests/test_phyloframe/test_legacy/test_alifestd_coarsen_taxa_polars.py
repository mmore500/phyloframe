import os
import typing

import numpy as np
import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_coarsen_taxa_asexual,
    alifestd_to_working_format,
    alifestd_try_add_ancestor_id_col,
)
from phyloframe.legacy._alifestd_coarsen_taxa_polars import (
    alifestd_coarsen_taxa_polars as alifestd_coarsen_taxa_polars_,
)

from ._impl import enforce_dtype_stability_polars

alifestd_coarsen_taxa_polars = enforce_dtype_stability_polars(
    alifestd_coarsen_taxa_polars_,
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
@pytest.mark.parametrize("col", ["id", "ancestor_id", "ancestor_list"])
def test_coarsen_taxa_polars_agg_override_error(
    apply: typing.Callable, col: str
):
    df = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 0, 1],
            "ancestor_list": ["[none]", "[0]", "[1]"],
            "dummy": [1, 1, 1],
        }
    )
    with pytest.raises(ValueError):
        alifestd_coarsen_taxa_polars(apply(df), agg={col: "sum"}, by="dummy")


@_apply
def test_coarsen_taxa_polars_empty(apply: typing.Callable):
    df = pl.DataFrame(
        {
            "id": pl.Series([], dtype=pl.Int64),
            "ancestor_id": pl.Series([], dtype=pl.Int64),
            "dummy": pl.Series([], dtype=pl.Int64),
        }
    )
    result = alifestd_coarsen_taxa_polars(apply(df), by="dummy")
    assert result.lazy().collect().is_empty()
    assert "dummy" in result.lazy().collect_schema().names()


@_apply
def test_coarsen_taxa_polars_singleton(apply: typing.Callable):
    df = pl.DataFrame(
        {
            "id": [0],
            "ancestor_id": [0],
            "branch_length": [0],
            "edge_length": [0],
            "origin_time": [100],
            "destruction_time": [200],
            "group": [7],
        }
    )
    result = (
        alifestd_coarsen_taxa_polars(apply(df), by="group").lazy().collect()
    )
    assert "branch_length" not in result.columns
    assert "edge_length" not in result.columns
    assert result["id"].to_list() == [0]
    assert result["origin_time"].to_list() == [100]
    assert result["destruction_time"].to_list() == [200]
    assert result["is_root"].to_list() == [True]


@_apply
def test_coarsen_taxa_polars_chain(apply: typing.Callable):
    df = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 0, 1],
            "origin_time": [0, 10, 20],
            "destruction_time": [10, 20, 30],
            "dummy": [0, 0, 0],
        }
    )
    result = (
        alifestd_coarsen_taxa_polars(apply(df), by="dummy").lazy().collect()
    )
    assert result["id"].to_list() == [0]
    assert result["ancestor_id"].to_list() == [0]
    assert result["origin_time"].to_list() == [0]
    assert result["destruction_time"].to_list() == [30]


@_apply
def test_coarsen_taxa_polars_two_groups(apply: typing.Callable):
    # 0 -> 1 -> 2 -> 3 -> 4, groups a a b b a
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_id": [0, 0, 1, 2, 3],
            "ancestor_list": ["[None]", "[0]", "[1]", "[2]", "[3]"],
            "trait": ["a", "a", "b", "b", "a"],
            "value": [1, 2, 3, 4, 5],
        }
    )
    result = (
        alifestd_coarsen_taxa_polars(
            apply(df),
            by="trait",
            agg={"trait": "first", "value": "sum"},
        )
        .lazy()
        .collect()
    )
    assert result["id"].to_list() == [0, 2, 4]
    assert result["ancestor_id"].to_list() == [0, 0, 2]
    assert result["ancestor_list"].to_list() == ["[none]", "[0]", "[2]"]
    assert result["trait"].to_list() == ["a", "b", "a"]
    assert result["value"].to_list() == [3, 7, 5]
    assert "is_root" not in result.columns  # not in agg


@_apply
def test_coarsen_taxa_polars_by_sequence(apply: typing.Callable):
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 1, 2],
            "x": [0, 0, 0, 1],
            "y": [0, 0, 1, 1],
        }
    )
    result = (
        alifestd_coarsen_taxa_polars(apply(df), by=["x", "y"]).lazy().collect()
    )
    assert result["id"].to_list() == [0, 2, 3]
    assert result["ancestor_id"].to_list() == [0, 0, 2]


@_apply
def test_coarsen_taxa_polars_expr_agg(apply: typing.Callable):
    df = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 0, 1],
            "dummy": [0, 0, 0],
            "value": [1, 2, 3],
        }
    )
    result = (
        alifestd_coarsen_taxa_polars(
            apply(df),
            by="dummy",
            agg={"value": pl.col("value").max(), "count": pl.len()},
        )
        .lazy()
        .collect()
    )
    assert result["value"].to_list() == [3]
    assert result["count"].to_list() == [3]


@_apply
def test_coarsen_taxa_polars_first_skips_nulls(apply: typing.Callable):
    # zero-length branches, as from alifestd_splice_chronological_consistency
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 1, 1],
            "origin_time": [0.0, 1.0, 1.0, 2.0],
            "taxon": ["a", None, "b", "c"],
        }
    )
    result = (
        alifestd_coarsen_taxa_polars(apply(df), by="origin_time")
        .lazy()
        .collect()
    )
    assert result["id"].to_list() == [0, 1, 3]
    assert result["ancestor_id"].to_list() == [0, 0, 1]
    assert result["taxon"].to_list() == ["a", "b", "c"]


@_apply
def test_coarsen_taxa_polars_nulls_compare_equal(apply: typing.Callable):
    df = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 0, 1],
            "trait": [None, None, 1],
        }
    )
    result = (
        alifestd_coarsen_taxa_polars(apply(df), by="trait").lazy().collect()
    )
    assert result["id"].to_list() == [0, 2]


@_apply
@pytest.mark.parametrize("dtype", [pl.UInt32, pl.Int32, pl.Int64])
def test_coarsen_taxa_polars_dtype(apply: typing.Callable, dtype: pl.DataType):
    df = pl.DataFrame(
        {
            "id": pl.Series([0, 1, 2], dtype=dtype),
            "ancestor_id": pl.Series([0, 0, 1], dtype=dtype),
            "dummy": [0, 1, 1],
        }
    )
    result = (
        alifestd_coarsen_taxa_polars(apply(df), by="dummy").lazy().collect()
    )
    assert result["id"].dtype == dtype
    assert result["ancestor_id"].dtype == dtype
    assert result["id"].to_list() == [0, 1]


@_apply
@pytest.mark.parametrize(
    "df",
    [
        pl.DataFrame(  # non-contiguous
            {"id": [0, 1, 3], "ancestor_id": [0, 0, 1], "dummy": [0, 0, 0]},
        ),
        pl.DataFrame(  # unsorted
            {"id": [0, 1, 2], "ancestor_id": [0, 2, 0], "dummy": [0, 0, 0]},
        ),
        pl.DataFrame(  # sexual
            {
                "id": [0, 1, 2],
                "ancestor_list": ["[none]", "[0]", "[0,1]"],
                "dummy": [0, 0, 0],
            },
        ),
    ],
)
def test_coarsen_taxa_polars_unsupported(
    apply: typing.Callable, df: pl.DataFrame
):
    with pytest.raises(NotImplementedError):
        alifestd_coarsen_taxa_polars(apply(df), by="dummy")


@_apply
def test_coarsen_taxa_polars_drop_sensitivity(apply: typing.Callable):
    df = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 0, 1],
            "dummy": [0, 0, 1],
            "num_children": [1, 1, 0],
        }
    )
    with pytest.warns(UserWarning):
        alifestd_coarsen_taxa_polars(apply(df), by="dummy")
    result = alifestd_coarsen_taxa_polars(
        apply(df), by="dummy", drop_topological_sensitivity=True
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
@pytest.mark.parametrize("by", ["trait", ["trait", "trait2"], "origin_time"])
@pytest.mark.parametrize("seed", range(3))
def test_coarsen_taxa_polars_matches_pandas(
    apply: typing.Callable,
    phylogeny_df: pd.DataFrame,
    by: typing.Union[str, typing.List[str]],
    seed: int,
):
    phylogeny_df = alifestd_try_add_ancestor_id_col(
        alifestd_to_working_format(phylogeny_df), mutate=True
    )
    rng = np.random.default_rng(seed)
    num_taxa = len(phylogeny_df)
    phylogeny_df["trait"] = rng.integers(0, 2, size=num_taxa)
    phylogeny_df["trait2"] = rng.integers(0, 3, size=num_taxa)
    phylogeny_df["origin_time"] = (phylogeny_df["origin_time"] // 10) * 10
    phylogeny_df["sparse"] = np.where(
        rng.random(num_taxa) < 0.5, np.nan, rng.random(num_taxa)
    )
    phylogeny_df = phylogeny_df[
        [
            "id",
            "ancestor_id",
            "ancestor_list",
            "origin_time",
            "destruction_time",
            "trait",
            "trait2",
            "sparse",
        ]
    ]

    expected = alifestd_coarsen_taxa_asexual(phylogeny_df, by=by)
    result = (
        alifestd_coarsen_taxa_polars(
            apply(pl.from_pandas(phylogeny_df)), by=by
        )
        .lazy()
        .collect()
        .to_pandas()
    )
    assert len(result) < len(phylogeny_df)
    assert sorted(result.columns) == sorted(expected.columns)
    pd.testing.assert_frame_equal(
        result[expected.columns], expected, check_dtype=False
    )
