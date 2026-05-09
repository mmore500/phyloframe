import os
import typing

import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_to_working_format,
    alifestd_ultrametricize,
)
from phyloframe.legacy._alifestd_ultrametricize_polars import (
    alifestd_ultrametricize_polars as alifestd_ultrametricize_polars_,
)

from ._impl import enforce_dtype_stability_polars

alifestd_ultrametricize_polars = enforce_dtype_stability_polars(
    alifestd_ultrametricize_polars_,
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_missing_origin_time(apply: typing.Callable):
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 0, 1],
            }
        ),
    )
    with pytest.raises(ValueError):
        alifestd_ultrametricize_polars(df_pl)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_unknown_method(apply: typing.Callable):
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 0, 1],
                "origin_time": [0.0, 1.0, 2.0],
            }
        ),
    )
    with pytest.raises(ValueError):
        alifestd_ultrametricize_polars(df_pl, method="bogus")


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_empty(apply: typing.Callable):
    df_pl = apply(
        pl.DataFrame(
            {"id": [], "ancestor_id": [], "origin_time": []},
            schema={
                "id": pl.Int64,
                "ancestor_id": pl.Int64,
                "origin_time": pl.Float64,
            },
        ),
    )
    res = alifestd_ultrametricize_polars(df_pl).lazy().collect()
    assert "origin_time" in res.columns
    assert res.is_empty()


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_simple_chain(apply: typing.Callable):
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 0, 1],
                "origin_time": [0.0, 1.0, 2.0],
            }
        ),
    )
    result = alifestd_ultrametricize_polars(df_pl).lazy().collect()
    assert result["origin_time"].to_list() == [0.0, 1.0, 2.0]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_simple_tree_extends(apply: typing.Callable):
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "ancestor_id": [0, 0, 0, 1],
                "origin_time": [0.0, 1.0, 2.0, 5.0],
            }
        ),
    )
    result = alifestd_ultrametricize_polars(df_pl).lazy().collect()
    # leaves: id 2 and id 3 -> 5.0; inner unchanged
    assert result["origin_time"].to_list() == [0.0, 1.0, 5.0, 5.0]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_forest_multiple_roots(apply: typing.Callable):
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3, 4, 5],
                "ancestor_id": [0, 0, 0, 3, 3, 3],
                "origin_time": [0.0, 2.0, 3.0, 0.0, 4.0, 7.0],
            }
        ),
    )
    result = alifestd_ultrametricize_polars(df_pl).lazy().collect()
    # leaves: 1,2,4,5 -> 7.0
    assert result["origin_time"].to_list() == [0.0, 7.0, 7.0, 0.0, 7.0, 7.0]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_already_ultrametric(apply: typing.Callable):
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 0, 0],
                "origin_time": [0.0, 5.0, 5.0],
            }
        ),
    )
    result = alifestd_ultrametricize_polars(df_pl).lazy().collect()
    assert result["origin_time"].to_list() == [0.0, 5.0, 5.0]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_preexisting_is_leaf_column(apply: typing.Callable):
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 0, 1],
                "origin_time": [0.0, 1.0, 2.0],
                "is_leaf": [False, True, False],
            }
        ),
    )
    result = alifestd_ultrametricize_polars(df_pl).lazy().collect()
    assert result["origin_time"].to_list() == [0.0, 1.0, 2.0]


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv")
        ),
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv")
        ),
        alifestd_to_working_format(
            pd.read_csv(f"{assets_path}/nk_tournamentselection.csv")
        ),
    ],
)
@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_polars_matches_pandas(
    phylogeny_df: pd.DataFrame, apply: typing.Callable
):
    expected = alifestd_ultrametricize(phylogeny_df, mutate=False)

    df_pl = apply(pl.from_pandas(phylogeny_df))
    result = alifestd_ultrametricize_polars(df_pl).lazy().collect()

    expected_ot = expected["origin_time"].to_numpy()
    actual_ot = result["origin_time"].to_numpy()
    assert (expected_ot == actual_ot).all()


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_noncontiguous_ids_with_preexisting_is_leaf(apply: typing.Callable):
    # The polars implementation falls back on alifestd_mark_leaves_polars,
    # which requires contiguous ids. To exercise non-contiguous ids we
    # provide is_leaf upfront so leaf detection is skipped.
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [10, 20, 30, 40],
                "ancestor_id": [10, 10, 20, 20],
                "origin_time": [0.0, 1.0, 2.0, 5.0],
                "is_leaf": [False, False, True, True],
            }
        ),
    )
    result = alifestd_ultrametricize_polars(df_pl).lazy().collect()
    assert result["origin_time"].to_list() == [0.0, 1.0, 5.0, 5.0]
