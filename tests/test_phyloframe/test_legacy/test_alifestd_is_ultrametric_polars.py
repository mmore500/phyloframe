import os
import typing

import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_is_ultrametric_polars,
    alifestd_to_working_format,
    alifestd_ultrametricize_polars,
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
        alifestd_is_ultrametric_polars(df_pl)


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
    assert alifestd_is_ultrametric_polars(df_pl) is True


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_simple_chain_is_ultrametric(apply: typing.Callable):
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 0, 1],
                "origin_time": [0.0, 1.0, 2.0],
            }
        ),
    )
    assert alifestd_is_ultrametric_polars(df_pl)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_non_ultrametric(apply: typing.Callable):
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "ancestor_id": [0, 0, 0, 1],
                "origin_time": [0.0, 1.0, 2.0, 5.0],
            }
        ),
    )
    assert not alifestd_is_ultrametric_polars(df_pl)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_ultrametric_after_extend(apply: typing.Callable):
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "ancestor_id": [0, 0, 0, 1],
                "origin_time": [0.0, 1.0, 2.0, 5.0],
            }
        ),
    )
    res = alifestd_ultrametricize_polars(df_pl)
    assert alifestd_is_ultrametric_polars(res)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_atol(apply: typing.Callable):
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 0, 0],
                "origin_time": [0.0, 4.9, 5.0],
            }
        ),
    )
    assert not alifestd_is_ultrametric_polars(df_pl)
    assert not alifestd_is_ultrametric_polars(df_pl, atol=0.05)
    assert alifestd_is_ultrametric_polars(df_pl, atol=0.11)
    assert alifestd_is_ultrametric_polars(df_pl, atol=1.0)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_forest(apply: typing.Callable):
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3, 4, 5],
                "ancestor_id": [0, 0, 0, 3, 3, 3],
                "origin_time": [0.0, 2.0, 3.0, 0.0, 4.0, 7.0],
            }
        ),
    )
    assert not alifestd_is_ultrametric_polars(df_pl)
    df_pl_uniform = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3, 4, 5],
                "ancestor_id": [0, 0, 0, 3, 3, 3],
                "origin_time": [0.0, 5.0, 5.0, 0.0, 5.0, 5.0],
            }
        ),
    )
    assert alifestd_is_ultrametric_polars(df_pl_uniform)


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
                "id": [10, 20, 30, 40, 50],
                "ancestor_id": [10, 10, 20, 30, 30],
                "origin_time": [0.0, 1.0, 2.0, 5.0, 3.0],
                "is_leaf": [False, False, True, True, True],
            }
        ),
    )
    assert not alifestd_is_ultrametric_polars(df_pl)
    res = alifestd_ultrametricize_polars(df_pl)
    assert alifestd_is_ultrametric_polars(res)


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
                "is_leaf": [True, False, False],
            }
        ),
    )
    # only "leaf" is id 0; trivially ultrametric
    assert alifestd_is_ultrametric_polars(df_pl)


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
def test_fuzz_after_extend_is_ultrametric(
    phylogeny_df: pd.DataFrame, apply: typing.Callable
):
    df_pl = apply(pl.from_pandas(phylogeny_df))
    res = alifestd_ultrametricize_polars(df_pl)
    assert alifestd_is_ultrametric_polars(res)
