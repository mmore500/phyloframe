import os
import typing

import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_mark_num_mrca_leaf_pairs_asexual,
    alifestd_to_working_format,
)
from phyloframe.legacy._alifestd_mark_num_mrca_leaf_pairs_polars import (
    alifestd_mark_num_mrca_leaf_pairs_polars as alifestd_mark_num_mrca_leaf_pairs_polars_,
)

from ._impl import enforce_dtype_stability_polars

alifestd_mark_num_mrca_leaf_pairs_polars = enforce_dtype_stability_polars(
    alifestd_mark_num_mrca_leaf_pairs_polars_,
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")

_apply = pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)


@_apply
def test_empty(apply: typing.Callable):
    df = pl.DataFrame(
        {
            "id": pl.Series([], dtype=pl.Int64),
            "ancestor_id": pl.Series([], dtype=pl.Int64),
        }
    )
    result = alifestd_mark_num_mrca_leaf_pairs_polars(apply(df))
    result = result.lazy().collect()
    assert "num_mrca_leaf_pairs" in result.columns
    assert result.is_empty()


@_apply
def test_simple(apply: typing.Callable):
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6],
            "ancestor_id": [0, 0, 0, 0, 3, 4, 4],
        }
    )
    result = (
        alifestd_mark_num_mrca_leaf_pairs_polars(apply(df)).lazy().collect()
    )
    assert result["num_mrca_leaf_pairs"].to_list() == [5, 0, 0, 0, 1, 0, 0]
    assert result["num_mrca_leaf_pairs"].dtype == pl.Int64


@_apply
def test_existing_num_leaves_used(apply: typing.Callable):
    # deliberately inconsistent column, to check it is used as-is
    df = pl.DataFrame(
        {"id": [0, 1, 2], "ancestor_id": [0, 0, 0], "num_leaves": [3, 1, 1]},
    )
    result = (
        alifestd_mark_num_mrca_leaf_pairs_polars(apply(df)).lazy().collect()
    )
    assert result["num_mrca_leaf_pairs"].to_list() == [3, 0, 0]


@_apply
def test_num_leaves_added(apply: typing.Callable):
    df = pl.DataFrame({"id": [0, 1, 2], "ancestor_id": [0, 0, 0]})
    result = (
        alifestd_mark_num_mrca_leaf_pairs_polars(apply(df)).lazy().collect()
    )
    assert result["num_leaves"].to_list() == [2, 1, 1]


@_apply
def test_multiple_roots_ancestor_list(apply: typing.Callable):
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6],
            "ancestor_list": [
                "[none]",
                "[0]",
                "[0]",
                "[none]",
                "[3]",
                "[3]",
                "[3]",
            ],
        }
    )
    result = (
        alifestd_mark_num_mrca_leaf_pairs_polars(apply(df), mark_as="custom")
        .lazy()
        .collect()
    )
    assert result["custom"].to_list() == [1, 0, 0, 3, 0, 0, 0]


@_apply
@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pd.read_csv(
            f"{assets_path}/example-standard-toy-asexual-phylogeny.csv"
        ),
        pd.read_csv(
            f"{assets_path}/example-standard-toy-asexual-phylogeny-tworoots.csv"
        ),
        pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv"),
        pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv"),
        pd.read_csv(f"{assets_path}/nk_tournamentselection.csv"),
        pd.read_csv(f"{assets_path}/trunktestphylo.csv"),
    ],
)
def test_matches_pandas(apply: typing.Callable, phylogeny_df: pd.DataFrame):
    phylogeny_df = alifestd_to_working_format(phylogeny_df)
    expected = alifestd_mark_num_mrca_leaf_pairs_asexual(phylogeny_df)
    result = (
        alifestd_mark_num_mrca_leaf_pairs_polars(
            apply(pl.from_pandas(phylogeny_df)),
        )
        .lazy()
        .collect()
    )
    assert (
        result["num_mrca_leaf_pairs"].to_list()
        == expected["num_mrca_leaf_pairs"].tolist()
    )


@_apply
@pytest.mark.parametrize(
    "df",
    [
        pl.DataFrame(  # non-contiguous
            {"id": [0, 1, 3], "ancestor_id": [0, 0, 1]},
        ),
        pl.DataFrame(  # unsorted
            {"id": [0, 1, 2], "ancestor_id": [0, 2, 0]},
        ),
        pl.DataFrame(  # sexual
            {"id": [0, 1, 2], "ancestor_list": ["[none]", "[0]", "[0,1]"]},
        ),
    ],
)
def test_unsupported(apply: typing.Callable, df: pl.DataFrame):
    with pytest.raises(NotImplementedError):
        alifestd_mark_num_mrca_leaf_pairs_polars(apply(df))
