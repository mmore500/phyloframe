import os
import typing

import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_mark_oldest_root,
    alifestd_to_working_format,
)
from phyloframe.legacy._alifestd_mark_oldest_root_polars import (
    alifestd_mark_oldest_root_polars as alifestd_mark_oldest_root_polars_,
)

from ._impl import enforce_dtype_stability_polars

alifestd_mark_oldest_root_polars = enforce_dtype_stability_polars(
    alifestd_mark_oldest_root_polars_
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        alifestd_to_working_format(
            pd.read_csv(
                f"{assets_path}/example-standard-toy-asexual-phylogeny.csv"
            )
        ),
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
def test_alifestd_mark_oldest_root_polars_matches_pandas(
    phylogeny_df: pd.DataFrame, apply: typing.Callable
):
    """Verify polars result matches pandas result."""
    result_pd = alifestd_mark_oldest_root(phylogeny_df, mutate=False)

    df_pl = apply(pl.from_pandas(phylogeny_df))
    result_pl = alifestd_mark_oldest_root_polars(df_pl).lazy().collect()

    assert result_pd["is_oldest_root"].tolist() == (
        result_pl["is_oldest_root"].to_list()
    )


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_oldest_root_polars_single_root(
    apply: typing.Callable,
):
    """Single root is the oldest root."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 0, 1],
            }
        ),
    )

    result = alifestd_mark_oldest_root_polars(df_pl).lazy().collect()

    assert result["is_oldest_root"].to_list() == [True, False, False]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_oldest_root_polars_two_roots_with_origin_time(
    apply: typing.Callable,
):
    """Two roots: pick the one with lowest origin_time."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "ancestor_id": [0, 1, 0, 1],
                "origin_time": [10.0, 5.0, 15.0, 8.0],
            }
        ),
    )

    result = alifestd_mark_oldest_root_polars(df_pl).lazy().collect()

    assert result["is_oldest_root"].to_list() == [
        False,
        True,
        False,
        False,
    ]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_oldest_root_polars_two_roots_without_origin_time(
    apply: typing.Callable,
):
    """Two roots without origin_time: pick the one with lowest id."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "ancestor_id": [0, 1, 0, 1],
            }
        ),
    )

    result = alifestd_mark_oldest_root_polars(df_pl).lazy().collect()

    assert result["is_oldest_root"].to_list() == [
        True,
        False,
        False,
        False,
    ]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_oldest_root_polars_single_node(
    apply: typing.Callable,
):
    """A single node is the oldest root."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0],
                "ancestor_id": [0],
            }
        ),
    )

    result = alifestd_mark_oldest_root_polars(df_pl).lazy().collect()

    assert result["is_oldest_root"].to_list() == [True]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_oldest_root_polars_empty(apply: typing.Callable):
    """Empty dataframe gets is_oldest_root column."""
    df_pl = apply(
        pl.DataFrame(
            {"id": [], "ancestor_id": []},
            schema={"id": pl.Int64, "ancestor_id": pl.Int64},
        ),
    )

    result = alifestd_mark_oldest_root_polars(df_pl).lazy().collect()

    assert "is_oldest_root" in result.columns
    assert result.is_empty()
