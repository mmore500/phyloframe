import os
import typing

import numpy as np
import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_is_working_format_polars,
    alifestd_test_leaves_isomorphic_asexual,
    alifestd_to_working_format,
    alifestd_try_add_ancestor_id_col,
)
from phyloframe.legacy._alifestd_shuffle_working_format_polars import (
    alifestd_shuffle_working_format_polars as alifestd_shuffle_working_format_polars_,
)

from ._impl import enforce_dtype_stability_polars

alifestd_shuffle_working_format_polars = enforce_dtype_stability_polars(
    alifestd_shuffle_working_format_polars_,
)

assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

_apply = pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)


def _label_parents(phylogeny_df: pl.DataFrame) -> dict:
    """Map each taxon's label to its parent's label."""
    label_lookup = dict(
        zip(phylogeny_df["id"].to_list(), phylogeny_df["label"].to_list())
    )
    return {
        label: label_lookup[ancestor_id]
        for label, ancestor_id in zip(
            phylogeny_df["label"].to_list(),
            phylogeny_df["ancestor_id"].to_list(),
        )
    }


def _prep(phylogeny_df: pd.DataFrame) -> pl.DataFrame:
    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df)
    phylogeny_df["label"] = phylogeny_df["id"]
    return pl.from_pandas(phylogeny_df.drop(columns=["ancestor_list"]))


@_apply
def test_shuffle_working_format_polars_empty(apply: typing.Callable):
    df = pl.DataFrame(
        {
            "id": pl.Series([], dtype=pl.Int64),
            "ancestor_id": pl.Series([], dtype=pl.Int64),
        }
    )
    result = alifestd_shuffle_working_format_polars(apply(df))
    assert result.is_empty()


@_apply
@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pd.read_csv(f"{assets}/nk_ecoeaselection-workingformat.csv"),
        pd.read_csv(f"{assets}/nk_lexicaseselection.csv"),
        pd.read_csv(f"{assets}/nk_tournamentselection.csv"),
        pd.read_csv(f"{assets}/example-standard-toy-asexual-phylogeny.csv"),
        pd.read_csv(
            f"{assets}/example-standard-toy-asexual-phylogeny-tworoots.csv",
        ),
    ],
)
@pytest.mark.parametrize("seed", range(3))
def test_shuffle_working_format_polars(
    apply: typing.Callable, phylogeny_df: pd.DataFrame, seed: int
):
    df = _prep(phylogeny_df)
    result = alifestd_shuffle_working_format_polars(apply(df), seed=seed)

    assert alifestd_is_working_format_polars(result)
    assert len(result) == len(df)
    assert set(result.columns) == set(df.columns)
    assert _label_parents(result) == _label_parents(df)
    assert alifestd_test_leaves_isomorphic_asexual(
        alifestd_to_working_format(df.to_pandas()),
        result.to_pandas(),
        taxon_label="label",
    )
    # non-topological columns travel with their taxa
    merged = df.join(result, on="label", suffix="_")
    for col in df.columns:
        if col not in ("id", "ancestor_id", "label"):
            assert merged[col].equals(merged[f"{col}_"])


@_apply
def test_shuffle_working_format_polars_seed(apply: typing.Callable):
    df = _prep(pd.read_csv(f"{assets}/nk_lexicaseselection.csv"))
    result1 = alifestd_shuffle_working_format_polars(apply(df), seed=1)
    result2 = alifestd_shuffle_working_format_polars(apply(df), seed=1)
    result3 = alifestd_shuffle_working_format_polars(apply(df), seed=2)
    assert result1.equals(result2)
    assert not result1["label"].equals(result3["label"])


@_apply
@pytest.mark.parametrize("dtype", [pl.UInt32, pl.Int32, pl.Int64])
def test_shuffle_working_format_polars_dtype(
    apply: typing.Callable, dtype: pl.DataType
):
    df = pl.DataFrame(
        {
            "id": pl.Series([0, 1, 2, 3], dtype=dtype),
            "ancestor_id": pl.Series([0, 0, 1, 1], dtype=dtype),
        }
    )
    result = alifestd_shuffle_working_format_polars(apply(df), seed=1)
    assert result["id"].dtype == dtype
    assert result["ancestor_id"].dtype == dtype
    assert alifestd_is_working_format_polars(result)


@_apply
def test_shuffle_working_format_polars_ancestor_list(apply: typing.Callable):
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 1, 1],
            "ancestor_list": ["[none]", "[0]", "[1]", "[1]"],
        }
    )
    result = alifestd_shuffle_working_format_polars(apply(df), seed=2)
    expected = [
        "[none]" if id_ == ancestor_id else f"[{ancestor_id}]"
        for id_, ancestor_id in zip(result["id"], result["ancestor_id"])
    ]
    assert result["ancestor_list"].to_list() == expected


@_apply
def test_shuffle_working_format_polars_sexual(apply: typing.Callable):
    df = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_list": ["[none]", "[0]", "[0,1]"],
        }
    )
    with pytest.raises(NotImplementedError):
        alifestd_shuffle_working_format_polars(apply(df))


def test_shuffle_working_format_polars_matches_input_values():
    df = pl.DataFrame(
        {
            "id": np.arange(5),
            "ancestor_id": [0, 0, 1, 1, 2],
            "label": list("abcde"),
        }
    )
    result = alifestd_shuffle_working_format_polars(df, seed=3)
    assert sorted(result["label"].to_list()) == list("abcde")
