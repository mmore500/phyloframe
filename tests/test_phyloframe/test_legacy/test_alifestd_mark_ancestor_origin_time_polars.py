import os
import typing

import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_mark_ancestor_origin_time_asexual,
    alifestd_to_working_format,
)
from phyloframe.legacy._alifestd_mark_ancestor_origin_time_polars import (
    alifestd_mark_ancestor_origin_time_polars as alifestd_mark_ancestor_origin_time_polars_,
)

from ._impl import enforce_dtype_stability_polars

alifestd_mark_ancestor_origin_time_polars = enforce_dtype_stability_polars(
    alifestd_mark_ancestor_origin_time_polars_
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


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
def test_alifestd_mark_ancestor_origin_time_polars_matches_pandas(
    phylogeny_df: pd.DataFrame, apply: typing.Callable
):
    """Verify polars result matches pandas result."""
    result_pd = alifestd_mark_ancestor_origin_time_asexual(
        phylogeny_df, mutate=False
    )

    df_pl = apply(pl.from_pandas(phylogeny_df))
    result_pl = (
        alifestd_mark_ancestor_origin_time_polars(df_pl).lazy().collect()
    )

    pd_vals = result_pd["ancestor_origin_time"].tolist()
    pl_vals = result_pl["ancestor_origin_time"].to_list()
    assert pd_vals == pytest.approx(pl_vals)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_ancestor_origin_time_polars_simple(
    apply: typing.Callable,
):
    """Test simple tree with origin times."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 0, 1],
                "origin_time": [0.0, 1.0, 2.0],
            }
        ),
    )

    result = alifestd_mark_ancestor_origin_time_polars(df_pl).lazy().collect()

    assert "ancestor_origin_time" in result.columns
    # root's ancestor is itself, so ancestor_origin_time == origin_time
    assert result["ancestor_origin_time"][0] == 0.0
    # node 1's ancestor is 0, origin_time 0.0
    assert result["ancestor_origin_time"][1] == 0.0
    # node 2's ancestor is 1, origin_time 1.0
    assert result["ancestor_origin_time"][2] == 1.0


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_ancestor_origin_time_polars_empty(
    apply: typing.Callable,
):
    """Empty dataframe gets ancestor_origin_time column."""
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

    result = alifestd_mark_ancestor_origin_time_polars(df_pl).lazy().collect()

    assert "ancestor_origin_time" in result.columns
    assert result.is_empty()


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_ancestor_origin_time_polars_non_contiguous_ids(
    apply: typing.Callable,
):
    """Verify NotImplementedError for non-contiguous ids."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 2, 5],
                "ancestor_id": [0, 0, 2],
                "origin_time": [0.0, 1.0, 2.0],
            }
        ),
    )
    with pytest.raises(NotImplementedError):
        alifestd_mark_ancestor_origin_time_polars(df_pl).lazy().collect()


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_ancestor_origin_time_polars_unsorted(
    apply: typing.Callable,
):
    """Verify NotImplementedError for unsorted data."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 2, 0],
                "origin_time": [0.0, 2.0, 1.0],
            }
        ),
    )
    with pytest.raises(NotImplementedError):
        alifestd_mark_ancestor_origin_time_polars(df_pl).lazy().collect()
