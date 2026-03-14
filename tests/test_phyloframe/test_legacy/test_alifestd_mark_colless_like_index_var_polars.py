import os
import typing

import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_mark_colless_like_index_var_asexual,
    alifestd_to_working_format,
)
from phyloframe.legacy._alifestd_mark_colless_like_index_var_polars import (
    alifestd_mark_colless_like_index_var_polars as alifestd_mark_colless_like_index_var_polars_,
)

from ._impl import enforce_dtype_stability_polars

alifestd_mark_colless_like_index_var_polars = enforce_dtype_stability_polars(
    alifestd_mark_colless_like_index_var_polars_
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
def test_alifestd_mark_colless_like_index_var_polars_matches_pandas(
    phylogeny_df: pd.DataFrame, apply: typing.Callable
):
    """Verify polars result matches pandas result."""
    result_pd = alifestd_mark_colless_like_index_var_asexual(
        phylogeny_df, mutate=False
    )

    df_pl = apply(pl.from_pandas(phylogeny_df))
    result_pl = (
        alifestd_mark_colless_like_index_var_polars(df_pl).lazy().collect()
    )

    assert result_pd["colless_like_index_var"].tolist() == pytest.approx(
        result_pl["colless_like_index_var"].to_list()
    )


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_colless_like_index_var_polars_empty(
    apply: typing.Callable,
):
    """Empty dataframe gets colless_like_index_var column."""
    df_pl = apply(
        pl.DataFrame(
            {"id": [], "ancestor_id": []},
            schema={"id": pl.Int64, "ancestor_id": pl.Int64},
        ),
    )

    result = (
        alifestd_mark_colless_like_index_var_polars(df_pl).lazy().collect()
    )

    assert "colless_like_index_var" in result.columns
    assert result.is_empty()


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_colless_like_index_var_polars_non_contiguous_ids(
    apply: typing.Callable,
):
    """Verify NotImplementedError for non-contiguous ids."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 2, 5],
                "ancestor_id": [0, 0, 2],
            }
        ),
    )
    with pytest.raises(NotImplementedError):
        alifestd_mark_colless_like_index_var_polars(df_pl).lazy().collect()
