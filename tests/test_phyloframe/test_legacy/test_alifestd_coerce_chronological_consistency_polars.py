import os
import typing

import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_coerce_chronological_consistency,
    alifestd_find_chronological_inconsistency_polars,
    alifestd_to_working_format,
    alifestd_try_add_ancestor_id_col,
)
from phyloframe.legacy._alifestd_coerce_chronological_consistency_polars import (
    alifestd_coerce_chronological_consistency_polars as alifestd_coerce_chronological_consistency_polars_,
)

from ._impl import enforce_dtype_stability_polars

alifestd_coerce_chronological_consistency_polars = (
    enforce_dtype_stability_polars(
        alifestd_coerce_chronological_consistency_polars_,
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
def test_coerce_chronological_consistency_polars_consistent(
    apply: typing.Callable,
):
    df = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 0, 1],
            "origin_time": [0.0, 1.0, 2.0],
        }
    )
    result = alifestd_coerce_chronological_consistency_polars(apply(df))
    assert result.lazy().collect().equals(df)


@_apply
def test_coerce_chronological_consistency_polars_inconsistent(
    apply: typing.Callable,
):
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 1, 0],
            "origin_time": [5.0, 1.0, 0.0, 7.0],
        }
    )
    result = (
        alifestd_coerce_chronological_consistency_polars(apply(df))
        .lazy()
        .collect()
    )
    assert result["origin_time"].to_list() == [5.0, 5.0, 5.0, 7.0]
    assert alifestd_find_chronological_inconsistency_polars(result) is None


@_apply
@pytest.mark.parametrize("epsilon", [0.0, 0.5, 1, 2.0])
def test_coerce_chronological_consistency_polars_epsilon(
    apply: typing.Callable, epsilon: float
):
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_id": [0, 0, 1, 1, 0],
            "origin_time": [0.0, 0.5, 3.0, 1.0, 2.0],
        }
    )
    result = (
        alifestd_coerce_chronological_consistency_polars(
            apply(df), epsilon=epsilon
        )
        .lazy()
        .collect()
    )
    t1 = max(0.5, epsilon)
    expected = [0.0, t1, max(3.0, t1 + epsilon)]
    expected += [max(1.0, t1 + epsilon), max(2.0, epsilon)]
    assert result["origin_time"].to_list() == expected


def test_coerce_chronological_consistency_polars_int_dtype():
    df = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 0, 1],
            "origin_time": [3, 1, 2],
        }
    )
    result = alifestd_coerce_chronological_consistency_polars(df)
    assert result["origin_time"].to_list() == [3, 3, 3]
    assert result["origin_time"].dtype == pl.Int64

    result = alifestd_coerce_chronological_consistency_polars_(df, epsilon=0.5)
    assert result["origin_time"].to_list() == [3.0, 3.5, 4.0]
    assert result["origin_time"].dtype == pl.Float64


@_apply
def test_coerce_chronological_consistency_polars_empty(
    apply: typing.Callable,
):
    df = pl.DataFrame(
        {
            "id": pl.Series([], dtype=pl.Int64),
            "ancestor_id": pl.Series([], dtype=pl.Int64),
            "origin_time": pl.Series([], dtype=pl.Float64),
        }
    )
    result = alifestd_coerce_chronological_consistency_polars(apply(df))
    assert result.lazy().collect().is_empty()


def test_coerce_chronological_consistency_polars_noncontiguous():
    df = pl.DataFrame(
        {
            "id": [0, 2, 4],
            "ancestor_id": [0, 0, 2],
            "origin_time": [5.0, 1.0, 0.0],
        }
    )
    with pytest.raises(NotImplementedError):
        alifestd_coerce_chronological_consistency_polars(df)


def test_coerce_chronological_consistency_polars_unsorted():
    df = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [1, 1, 0],
            "origin_time": [0.0, 5.0, 0.0],
        }
    )
    with pytest.raises(NotImplementedError):
        alifestd_coerce_chronological_consistency_polars(df)


@pytest.mark.parametrize(
    "phylogeny_file",
    ["nk_ecoeaselection.csv", "nk_lexicaseselection.csv"],
)
@pytest.mark.parametrize("epsilon", [0.0, 1.0, 25.0])
def test_coerce_chronological_consistency_polars_matches_pandas(
    phylogeny_file: str, epsilon: float
):
    df_pd = alifestd_to_working_format(
        pd.read_csv(f"{assets}/{phylogeny_file}"),
    )
    df_pd["origin_time"] = df_pd["origin_time"].astype(float)
    df_pd = alifestd_try_add_ancestor_id_col(df_pd)
    # perturb origin times to introduce inconsistencies
    df_pd["origin_time"] = df_pd["origin_time"] * (
        (df_pd["id"] % 3 == 0) * 0.5 + 0.75
    )

    expected = alifestd_coerce_chronological_consistency(
        df_pd, epsilon=epsilon
    ).reset_index(drop=True)
    result = alifestd_coerce_chronological_consistency_polars(
        pl.from_pandas(df_pd), epsilon=epsilon
    ).to_pandas()

    pd.testing.assert_frame_equal(
        result, expected, check_dtype=False, check_index_type=False
    )


def test_coerce_chronological_consistency_polars_non_numeric():
    df = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 0, 1],
            "origin_time": ["b", "a", "c"],
        }
    )
    with pytest.raises(NotImplementedError):
        alifestd_coerce_chronological_consistency_polars(df)
