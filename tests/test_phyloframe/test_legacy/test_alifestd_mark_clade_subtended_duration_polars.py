import os
import typing

import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_mark_clade_subtended_duration_asexual,
    alifestd_to_working_format,
)
from phyloframe.legacy._alifestd_mark_clade_subtended_duration_polars import (
    alifestd_mark_clade_subtended_duration_polars as alifestd_mark_clade_subtended_duration_polars_,
)

from ._impl import enforce_dtype_stability_polars

alifestd_mark_clade_subtended_duration_polars = enforce_dtype_stability_polars(
    alifestd_mark_clade_subtended_duration_polars_,
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_clade_subtended_duration_polars_simple_tree(
    apply: typing.Callable,
):
    """Test a simple tree.

    Tree structure:
        0 (root, ot=0)
        +-- 1 (ot=1)
        |   +-- 3 (ot=3)
        |   +-- 4 (ot=4)
        +-- 2 (ot=2)

    max_desc_ot: [4, 4, 2, 3, 4]
    ancestor_ot for non-roots: [-, 0, 0, 1, 1]
    For roots, ancestor_ot = 0

    clade_subtended_duration = max_desc_ot - effective_ancestor_ot
    node 0: 4 - 0 = 4 (root, ancestor_ot = 0)
    node 1: 4 - 0 = 4
    node 2: 2 - 0 = 2
    node 3: 3 - 1 = 2
    node 4: 4 - 1 = 3
    """
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3, 4],
                "ancestor_id": [0, 0, 0, 1, 1],
                "origin_time": [0.0, 1.0, 2.0, 3.0, 4.0],
            }
        ),
    )

    result = (
        alifestd_mark_clade_subtended_duration_polars(df_pl).lazy().collect()
    )

    assert result["clade_subtended_duration"].to_list() == [
        4.0,
        4.0,
        2.0,
        2.0,
        3.0,
    ]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_clade_subtended_duration_polars_single_node(
    apply: typing.Callable,
):
    """A single root node."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0],
                "ancestor_id": [0],
                "origin_time": [5.0],
            }
        ),
    )

    result = (
        alifestd_mark_clade_subtended_duration_polars(df_pl).lazy().collect()
    )

    # root: max_desc_ot = 5.0, ancestor_ot = 0 (root rule)
    assert result["clade_subtended_duration"].to_list() == [5.0]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_clade_subtended_duration_polars_empty(
    apply: typing.Callable,
):
    """Empty dataframe gets clade_subtended_duration column."""
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

    result = (
        alifestd_mark_clade_subtended_duration_polars(df_pl).lazy().collect()
    )

    assert "clade_subtended_duration" in result.columns
    assert result.is_empty()


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_clade_subtended_duration_polars_non_contiguous_ids(
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
        alifestd_mark_clade_subtended_duration_polars(df_pl).lazy().collect()


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_clade_subtended_duration_polars_unsorted(
    apply: typing.Callable,
):
    """Verify NotImplementedError for topologically unsorted data."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 2, 0],
                "origin_time": [0.0, 1.0, 2.0],
            }
        ),
    )
    with pytest.raises(NotImplementedError):
        alifestd_mark_clade_subtended_duration_polars(df_pl).lazy().collect()


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
def test_alifestd_mark_clade_subtended_duration_polars_matches_pandas(
    phylogeny_df: pd.DataFrame, apply: typing.Callable
):
    """Verify polars result matches pandas result."""
    result_pd = alifestd_mark_clade_subtended_duration_asexual(
        phylogeny_df,
        mutate=False,
    )

    df_pl = apply(pl.from_pandas(phylogeny_df))
    result_pl = (
        alifestd_mark_clade_subtended_duration_polars(df_pl).lazy().collect()
    )

    pd_vals = result_pd["clade_subtended_duration"].tolist()
    pl_vals = result_pl["clade_subtended_duration"].to_list()

    for pd_val, pl_val in zip(pd_vals, pl_vals):
        assert abs(pd_val - pl_val) < 1e-10, f"{pd_val} != {pl_val}"
